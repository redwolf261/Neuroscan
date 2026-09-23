"""
Phase E35-B (prompt-labeled "E34-B"): construct three target families --
EXACT, MORPHOLOGICAL DILATION, GAUSSIAN CONTEXT -- at each decoder scale,
and characterize their statistics. NO TRAINING. NO HYPERPARAMETER TUNING.

Scale grid (fixed by the architecture, not chosen for this experiment):
    D4 = 16^3   (dec3, coarsest auxiliary head)
    D2 = 32^3   (dec2)
    D1 = 64^3   (full resolution; eta_D1 = 0 per the hypothesis spec, so D1
                 targets are computed here for completeness/reference only)

Pre-declared sigma/radius grid (declared BEFORE looking at any result in this
script). Values are in VOXELS AT THE TARGET SCALE:
    sigma in {0.5, 1.0, 1.5, 2.0}, matched index-for-index with
    dilation radius in {1, 2, 3, 4} voxels (a rough support-matched pairing
    only; a MASS-matched pairing is done properly in E35-E, not here).

CONSTRUCTION -- REVISED AFTER A REAL BUG WAS CAUGHT BY DIRECT INSPECTION,
NOT ASSUMED CORRECT:

  A first version built the exact/dilation/Gaussian targets by resampling
  the native GT mask DOWN to the target scale first, and only then blurring
  or dilating at that (already-degraded) resolution. This was checked
  directly on a small subset before trusting it, and the check exposed a
  structural bug: for any component whose exact footprint has ALREADY
  vanished at a given scale (size_64 == 0, common for small lesions), the
  resampled mask is all-zero BEFORE the blur/dilation is applied, so
  blurring or dilating an all-zero field is still all-zero -- the "context"
  can never contain more information than the exact target it was derived
  from. This directly contradicts the proposed formula in the prompt,
  C_sigma(x) = (K_sigma * Y)(x), where Y is explicitly the NATIVE ground
  truth (convolved BEFORE any scale degradation), and it would have silently
  biased every downstream section (C, E, G, H) toward KILL for a
  construction-order reason, not a real empirical one.

  Fixed by constructing both the Gaussian context field and the dilation
  field AT NATIVE RESOLUTION FIRST (native Y, native sigma/radius converted
  from the target-scale grid via each axis's zoom factor), and only THEN
  resampling the resulting continuous field DOWN to the target scale
  (order=1 / linear interpolation, which is the correct interpolation for a
  continuous field -- nearest-neighbor, used elsewhere in this project for
  discrete component-ID volumes, would be wrong here). This lets contextual
  support meaningfully outlive the exact target's collapse, which is the
  entire point of the hypothesis being tested.

  The EXACT target itself is still the scale-appropriate resampled binary
  mask (matches size_64 / E30's own convention at D1, verified below), since
  the exact target is not supposed to survive degradation -- only the
  context field is.

For each component, at each scale, for each family member:
    total target mass, target max, target mean (over the WHOLE volume),
    nonzero support, effective support >0.01, >0.1, >0.5.

Output: E35_target_family_stats.json (list of per-component-per-scale-per-family records)
"""
import sys
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from scipy.ndimage import zoom, gaussian_filter, binary_dilation

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
ROSTER = OUT_DIR / "E35_component_roster.json"

SCALES = {"D4": 16, "D2": 32, "D1": 64}
SIGMA_GRID = [0.5, 1.0, 1.5, 2.0]     # in TARGET-scale voxels
RADIUS_GRID = [1, 2, 3, 4]            # in TARGET-scale voxels, matched index-for-index with SIGMA_GRID


def resize_nn(volume, target_shape):
    zf = tuple(t / c for t, c in zip(target_shape, volume.shape))
    return zoom(volume, zf, order=0)


def resize_lin(volume, target_shape):
    zf = tuple(t / c for t, c in zip(target_shape, volume.shape))
    return zoom(volume, zf, order=1)


def stats_for_target(vol_full_scale):
    n_total = vol_full_scale.size
    mass = float(vol_full_scale.sum())
    vmax = float(vol_full_scale.max()) if vol_full_scale.size else 0.0
    vmean = mass / n_total
    return {
        "mass": mass,
        "max": vmax,
        "mean": vmean,
        "support_nonzero": int((vol_full_scale > 0).sum()),
        "support_gt_0p01": int((vol_full_scale > 0.01).sum()),
        "support_gt_0p1": int((vol_full_scale > 0.1).sum()),
        "support_gt_0p5": int((vol_full_scale > 0.5).sum()),
    }


def main():
    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    subject_dirs = {Path(d).name: d for d in val_dataset.subject_dirs}

    roster = json.load(open(ROSTER))
    print(f"Roster loaded: {len(roster)} components", flush=True)

    by_subject = {}
    for r in roster:
        by_subject.setdefault(r["subject_id"], []).append(r)
    n_subjects_total = len(by_subject)

    records = []
    size64_crosscheck_mismatches = 0
    n_processed = 0
    n_subjects_done = 0
    CHECKPOINT_EVERY_N_SUBJECTS = 10

    import time
    t_start = time.time()

    for subject_id, comps in by_subject.items():
        subject_dir = subject_dirs[subject_id]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_labeled, n_native = ndi.label(seg_binary_native > 0.5)
        native_shape = np.array(seg_binary_native.shape)

        # EXACT target volumes at each scale: resample the integer-ID volume
        # (nearest-neighbor, identical convention to E30/E35-A) once per subject.
        labeled_at_scale = {}
        for scale_name, res in SCALES.items():
            resampled = resize_nn(native_labeled.astype(np.float32), (res, res, res))
            labeled_at_scale[scale_name] = np.round(resampled).astype(np.int32)

        # Per-component NATIVE binary masks AND bounding boxes, built once,
        # reused for every scale/sigma/radius combination's native-space
        # blur/dilation. A first implementation ran gaussian_filter/dilation
        # on the FULL native volume (240x240x155) per component -- this was
        # measured directly and found far too slow to scale to 749
        # components (killed after >10 min on a 24-component subset).
        # Fixed by cropping each component to its own bounding box + a
        # PER-SCALE-PER-SIGMA margin (computed just-in-time below, sized
        # correctly per axis via the true zoom factor -- not a single
        # worst-case constant), which is mathematically equivalent for a
        # compact Gaussian truncated at 4 sigma / a finite dilation radius,
        # and verified below via the same size_64 crosscheck plus an
        # explicit edge-leak check on the crop.
        comp_ids_here = [r["native_component_id"] for r in comps]
        comp_bboxes = {}
        for cid in comp_ids_here:
            coords = np.argwhere(native_labeled == cid)
            comp_bboxes[cid] = (coords.min(axis=0), coords.max(axis=0) + 1)

        for r in comps:
            comp_id = r["native_component_id"]
            bbox_lo, bbox_hi = comp_bboxes[comp_id]

            for scale_name, res in SCALES.items():
                target_shape = (res, res, res)
                zf = np.array([res / n for n in native_shape])  # native -> target zoom factor per axis
                # Convert target-scale sigma/radius to native-voxel-equivalent
                # units per axis (isotropic target-scale distance maps to
                # anisotropic native distance since native voxels aren't cubic
                # here -- 240x240x155). Use the geometric mean of per-axis
                # native-equivalent sigma for an isotropic Gaussian in native
                # space (gaussian_filter takes one sigma per axis natively,
                # so pass a 3-tuple directly rather than forcing isotropy).

                comp_mask_at_scale = (labeled_at_scale[scale_name] == comp_id).astype(np.float32)

                if scale_name == "D1":
                    size_here = int(comp_mask_at_scale.sum())
                    if size_here != r["size_64"]:
                        size64_crosscheck_mismatches += 1

                # --- EXACT TARGET: binary footprint at this scale (unchanged from before) ---
                exact_stats = stats_for_target(comp_mask_at_scale)

                sigma_records = []
                for sigma_t, radius_t in zip(SIGMA_GRID, RADIUS_GRID):
                    # Native-equivalent sigma per axis: a target-scale distance of
                    # sigma_t voxels corresponds to sigma_t / zf[axis] native voxels.
                    sigma_native = np.array([sigma_t / z for z in zf])
                    radius_native_per_axis = np.array([radius_t / z for z in zf])
                    radius_native_iters = int(np.ceil(radius_native_per_axis.max()))

                    # Crop margin (native voxels, per axis): must cover the
                    # larger of (a) 4*sigma_native (Gaussian truncation, matching
                    # scipy's own default truncate=4.0) and (b) radius_native_iters
                    # (dilation reach), plus a small safety pad.
                    margin_per_axis = np.ceil(np.maximum(4.0 * sigma_native, radius_native_iters)).astype(int) + 2
                    # Cap the margin ONLY to bound crop size for very large
                    # components' own bounding box inflation (not to truncate the
                    # kernel itself). A first version capped at 40 native voxels
                    # for speed; direct inspection of the run's own diagnostics
                    # (margin_capped rate) showed this capped EVERY component
                    # uniformly at D4 sigma>=1.0 and D2 sigma>=1.5 (up to 3x
                    # short of the true 4-sigma reach, e.g. D4 sigma=2.0 needs
                    # margin=122 native voxels in x/y), not just large
                    # components as assumed -- a real, scale-driven truncation
                    # bias affecting most of the pre-declared sigma grid, not a
                    # negligible edge effect. Fixed by raising the cap to the
                    # true worst-case requirement across the whole grid (D4,
                    # sigma=2.0, x/y axis: 122, rounded up with margin) so the
                    # kernel is never truncated by this cap; only a genuinely
                    # oversized component bbox (native_size very large, e.g.
                    # S5 lesions already spanning 100+ native voxels) can still
                    # trigger it, which is the population this cap was always
                    # meant to bound.
                    MARGIN_CAP = 130
                    margin_capped = bool(np.any(margin_per_axis > MARGIN_CAP))
                    margin_per_axis = np.minimum(margin_per_axis, MARGIN_CAP)
                    crop_lo = np.clip(bbox_lo - margin_per_axis, 0, None)
                    crop_hi = np.clip(bbox_hi + margin_per_axis, None, native_shape)
                    crop_slices = tuple(slice(l, h) for l, h in zip(crop_lo, crop_hi))
                    crop_native_mask = (native_labeled[crop_slices] == comp_id).astype(np.float32)

                    # Edge-leak check: did the crop clip against the real native
                    # volume boundary in a way that removes >1 voxel of the
                    # requested margin? If so this scale/sigma/component is
                    # flagged rather than silently trusted (mirrors the earlier
                    # verified crop-margin bug fix).
                    clip_lo = np.clip(-(bbox_lo - margin_per_axis), 0, None)
                    clip_hi = np.clip((bbox_hi + margin_per_axis) - native_shape, 0, None)
                    crop_edge_leak = bool(np.any(clip_lo > 1) or np.any(clip_hi > 1))

                    # GAUSSIAN CONTEXT: blur the CROPPED native mask (native-
                    # equivalent sigma per axis), THEN resample the resulting
                    # continuous field down to target scale (linear interpolation),
                    # embedding it back into a full-target-shape zero volume at
                    # the crop's proportional location.
                    gauss_crop = gaussian_filter(crop_native_mask, sigma=tuple(sigma_native), mode="constant", cval=0.0)
                    crop_lo_scaled = np.floor(crop_lo * zf).astype(int)
                    crop_hi_scaled = np.ceil(crop_hi * zf).astype(int)
                    crop_lo_scaled = np.clip(crop_lo_scaled, 0, res)
                    crop_hi_scaled = np.clip(crop_hi_scaled, crop_lo_scaled + 1, res)
                    crop_shape_at_scale = tuple(h - l for l, h in zip(crop_lo_scaled, crop_hi_scaled))
                    gauss_crop_at_scale = resize_lin(gauss_crop, crop_shape_at_scale)
                    gauss_crop_at_scale = np.clip(gauss_crop_at_scale, 0.0, None)
                    gauss_at_scale = np.zeros(target_shape, dtype=np.float32)
                    gauss_at_scale[tuple(slice(l, h) for l, h in zip(crop_lo_scaled, crop_hi_scaled))] = gauss_crop_at_scale
                    gauss_stats = stats_for_target(gauss_at_scale)

                    # MORPHOLOGICAL DILATION: dilate the cropped native binary
                    # mask by radius_native_iters voxels, THEN resample down and
                    # embed the same way.
                    dil_crop = crop_native_mask > 0.5
                    for _ in range(radius_native_iters):
                        dil_crop = binary_dilation(dil_crop)
                    dil_crop_at_scale = resize_lin(dil_crop.astype(np.float32), crop_shape_at_scale)
                    dil_at_scale = np.zeros(target_shape, dtype=np.float32)
                    dil_at_scale[tuple(slice(l, h) for l, h in zip(crop_lo_scaled, crop_hi_scaled))] = dil_crop_at_scale
                    dil_stats = stats_for_target(dil_at_scale)

                    sigma_records.append({
                        "sigma_target_vox": sigma_t, "radius_target_vox": radius_t,
                        "sigma_native_vox": sigma_native.tolist(), "radius_native_iters": radius_native_iters,
                        "gaussian": gauss_stats, "dilation": dil_stats,
                        "crop_edge_leak": crop_edge_leak,
                        "margin_capped": margin_capped,
                    })

                records.append({
                    "subject_id": subject_id,
                    "native_component_id": comp_id,
                    "scale": scale_name,
                    "resolution": res,
                    "exact": exact_stats,
                    "sigma_grid": sigma_records,
                })

            n_processed += 1
            if n_processed % 50 == 0:
                elapsed = time.time() - t_start
                print(f"  processed {n_processed}/{len(roster)} components ({elapsed:.0f}s elapsed)", flush=True)

        n_subjects_done += 1
        elapsed = time.time() - t_start
        rate = elapsed / n_subjects_done
        eta = rate * (n_subjects_total - n_subjects_done)
        print(f"subject {n_subjects_done}/{n_subjects_total} done ({subject_id}, {len(comps)} components) "
              f"-- {elapsed:.0f}s elapsed, ETA {eta:.0f}s more", flush=True)

        if n_subjects_done % CHECKPOINT_EVERY_N_SUBJECTS == 0:
            with open(OUT_DIR / "E35_target_family_stats.partial.json", "w") as f:
                json.dump(records, f)
            print(f"  [checkpoint] wrote {len(records)} records to E35_target_family_stats.partial.json", flush=True)

    print(f"\nTotal component-scale records: {len(records)} (expected {len(roster)*3})", flush=True)
    print(f"D1 exact-mask size vs E35-A's size_64 crosscheck mismatches: {size64_crosscheck_mismatches} / {len(roster)}", flush=True)
    if size64_crosscheck_mismatches > 0:
        print("  WARNING: mismatch -- resize convention differs from E35-A/E30. Investigate before trusting.", flush=True)

    n_leak = sum(1 for rec in records for sg in rec["sigma_grid"] if sg["crop_edge_leak"])
    n_capped = sum(1 for rec in records for sg in rec["sigma_grid"] if sg["margin_capped"])
    n_sigma_total = sum(len(rec["sigma_grid"]) for rec in records)
    print(f"Crop edge leaks (component near real native volume boundary): {n_leak} / {n_sigma_total} sigma-records", flush=True)
    print(f"Margin-capped (large component, reduced blur truncation accuracy): {n_capped} / {n_sigma_total} sigma-records", flush=True)

    with open(OUT_DIR / "E35_target_family_stats.json", "w") as f:
        json.dump(records, f)
    print(f"Saved to {OUT_DIR / 'E35_target_family_stats.json'}", flush=True)

    partial_path = OUT_DIR / "E35_target_family_stats.partial.json"
    if partial_path.exists():
        partial_path.unlink()
        print("Removed intermediate checkpoint file (final output saved successfully).", flush=True)


if __name__ == "__main__":
    main()
