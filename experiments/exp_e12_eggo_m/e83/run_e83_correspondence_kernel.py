"""
Phase E83: Correspondence Kernel Characterization (NO TRAINING, pure
inference-time intervention) -- per the user's design, following E82's
corrected finding that local donor substitution damage is real,
distance-graded, AND anisotropic (z-axis shifts hurt less than x/y-axis
shifts at the same VOXEL magnitude).

CRITICAL CONFOUND CHECKED FIRST (before any further interpretation, per
the user's explicit instruction): BraTS native volumes are 240x240x155
at 1mm ISOTROPIC spacing (verified directly from a real subject's NIfTI
header: zooms = (1.0, 1.0, 1.0) mm). This project's own preprocessing
resizes to a 64^3 cube via scipy.ndimage.zoom with PER-AXIS zoom factors
(target_shape / native_shape), since native shape is NOT cubic:
    x: 64/240 = 0.2667  ->  1 resized-voxel = 3.750 mm
    y: 64/240 = 0.2667  ->  1 resized-voxel = 3.750 mm
    z: 64/155 = 0.4129  ->  1 resized-voxel = 2.422 mm
This means a fixed VOXEL offset (e.g. the project's own standard 3-voxel
shift) corresponds to a DIFFERENT PHYSICAL distance depending on axis:
3 voxels = 11.25mm in x/y, but only 7.27mm in z. E82's finding that
z-shifts hurt LESS than x/y-shifts at the same voxel magnitude is
DIRECTLY CONSISTENT with this being a trivial physical-distance artifact
(smaller physical shift naturally hurts less), not a special property of
the learned representation. This phase resolves whether the anisotropy
SURVIVES normalizing offsets to equal PHYSICAL distance, per the user's
explicit two-kernel design: K_voxel(delta) vs K_physical(delta_mm).

METHOD: reuses E82(b)'s local donor swap construction (magnitude held
fixed via E78's decompose/recompose, LOCAL probability-change metric at
the swap sites -- confirmed the only sensitive metric for this sparse
intervention in E82). Three characterizations:

  1. RADIAL DEPENDENCE: K(delta) as a function of ||delta||_2 in voxel
     space AND in physical mm space (using the actual per-axis mm/voxel
     conversion above) -- does a smooth f(||delta||) fit better in voxel
     space or physical space?
  2. AXIS DEPENDENCE, PHYSICAL-DISTANCE-MATCHED: instead of comparing
     K(3,0,0) vs K(0,0,3) (unequal physical distance: 11.25mm vs 7.27mm),
     compare offsets chosen to be approximately EQUAL physical distance
     across axes (x/y voxel-offset ~2 [7.5mm] vs z voxel-offset ~3
     [7.27mm], the closest achievable integer-voxel match). If the axis
     difference DISAPPEARS once physical distance is matched, E82's
     anisotropy was the confound. If it PERSISTS, the anisotropy is a
     genuine representational property surviving the physical-spacing
     control.
  3. REFLECTION SYMMETRY: K(+delta) vs K(-delta) for each axis -- tests
     handedness/directional bias independent of the isotropy question.

PRE-DECLARED READING:
  - Axis difference vanishes under physical-distance matching -> E82's
    anisotropy was substantially the voxel-spacing confound. RETRACT any
    "the learned representation has anisotropic structure" claim; the
    finding reduces to ordinary distance-decay in physical space.
  - Axis difference PERSISTS under physical-distance matching -> genuine
    anisotropy in the learned representation, independent of resampling
    geometry -- THEN worth treating as a real object to characterize
    further, per the user's own "only then" framing.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e74"))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e78"))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402
from run_e78_magnitude_direction_decomposition import decompose, recompose, forward_with_enc1  # noqa: E402

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 0
N_PERM = 1000
N_BOOT = 2000
N_SWAP_LOCATIONS_PER_SUBJECT = 12

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")

# Verified from a real subject's NIfTI header: native (240,240,155) at 1mm
# isotropic; this project's own resize target is (64,64,64).
NATIVE_SHAPE = (240, 240, 155)
TARGET_SHAPE = (64, 64, 64)
MM_PER_VOXEL = {axis: native / target for axis, native, target in
                zip(["x", "y", "z"], NATIVE_SHAPE, TARGET_SHAPE)}
print(f"[Confound check] mm per resized-voxel: {MM_PER_VOXEL}")

# Offsets to test: (dz,dy,dx) tuples, name, axis label -- note tensor dim
# order is (D,H,W) matching (z,y,x) in this codebase's own convention
# (verified: BraTSMultimodalDataset loads volumes in (D,H,W) = native
# (x,y,z) order via the same resize call for all axes, so tensor dim 0 <->
# native axis with NATIVE_SHAPE[0]=240=x... note: NEEDS VERIFICATION,
# flagged explicitly below rather than silently assumed).
OFFSET_TESTS = {
    # axis-aligned, magnitude 3 in TENSOR space (matches E65/E82's own reference)
    "tensor_dim0_+3": (3, 0, 0), "tensor_dim0_-3": (-3, 0, 0),
    "tensor_dim1_+3": (0, 3, 0), "tensor_dim1_-3": (0, -3, 0),
    "tensor_dim2_+3": (0, 0, 3), "tensor_dim2_-3": (0, 0, -3),
    # physical-distance-matched pair (dim0/dim1 at voxel=2, dim2 at voxel=3
    # -- chosen so the induced physical mm is as close as achievable with
    # integer voxel offsets, using MM_PER_VOXEL computed above)
    "dim0_matched": (2, 0, 0), "dim1_matched": (0, 2, 0), "dim2_matched": (0, 0, 3),
}


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def local_swap_direction(u, locations, offset, D, H, W):
    u_out = u.clone()
    d0, d1, d2 = offset
    for (a, b, c) in locations:
        a2, b2, c2 = (a + d0) % D, (b + d1) % H, (c + d2) % W
        u_out[:, a, b, c] = u[:, a2, b2, c2]
    return u_out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(MM_CKPT), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded MM seed0 checkpoint: best_per_subject_dice={ckpt.get('best_per_subject_dice')}\n")

    val_ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                    split="val", val_split=0.1, target_shape=TARGET_SHAPE)
    rng = np.random.default_rng(SEED)

    from scipy.ndimage import distance_transform_edt

    records = []
    for idx in range(len(val_ds)):
        img, msk, sid = val_ds[idx]
        img_b = img.unsqueeze(0).to(device)
        target_bin = (msk.squeeze(0).numpy() > 0.5).astype(np.float32)
        D, H, W = target_bin.shape

        with torch.no_grad():
            enc1_intact = model.enc1(img_b).squeeze(0)
        r_intact, u_intact = decompose(enc1_intact)
        probs_intact, _ = forward_with_enc1(model, img_b, enc1_intact, device)

        if target_bin.sum() > 0 and target_bin.sum() < target_bin.size:
            dist_out = distance_transform_edt(1 - target_bin)
            dist_in = distance_transform_edt(target_bin)
            is_boundary = ((target_bin > 0) & (dist_in <= 2)) | ((target_bin == 0) & (dist_out <= 2))
            is_interior = (target_bin > 0) & (dist_in > 2)
            candidate_mask = is_boundary | is_interior
            candidate_voxels = np.argwhere(candidate_mask)
            margin = 6
            valid = ((candidate_voxels[:, 0] >= margin) & (candidate_voxels[:, 0] < D - margin) &
                    (candidate_voxels[:, 1] >= margin) & (candidate_voxels[:, 1] < H - margin) &
                    (candidate_voxels[:, 2] >= margin) & (candidate_voxels[:, 2] < W - margin))
            candidate_voxels = candidate_voxels[valid]
        else:
            candidate_voxels = np.empty((0, 3), dtype=int)

        if len(candidate_voxels) < N_SWAP_LOCATIONS_PER_SUBJECT:
            continue

        chosen = []
        perm = rng.permutation(len(candidate_voxels))
        for i in perm:
            cand = candidate_voxels[i]
            if all(np.abs(cand - c).max() >= 8 for c in chosen):
                chosen.append(cand)
            if len(chosen) >= N_SWAP_LOCATIONS_PER_SUBJECT:
                break
        locations = [tuple(int(v) for v in c) for c in chosen]

        row = {"subject_id": sid, "n_locations": len(locations)}
        for name, offset in OFFSET_TESTS.items():
            u_swapped = local_swap_direction(u_intact, locations, offset, D, H, W)
            z_swapped = recompose(r_intact, u_swapped)
            probs_swapped, _ = forward_with_enc1(model, img_b, z_swapped, device)
            local_changes = [abs(float(probs_swapped[loc] - probs_intact[loc])) for loc in locations]
            row[f"K_{name}"] = float(np.mean(local_changes))

        records.append(row)
        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1}/{len(val_ds)}", flush=True)

    with open(OUT_DIR / "E83_correspondence_kernel_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.\n")

    K_means = {}
    for name in OFFSET_TESTS.keys():
        vals = np.array([r[f"K_{name}"] for r in records])
        K_means[name] = float(vals.mean())
        print(f"  K[{name}] = {vals.mean():.4f} (std {vals.std():.4f})")

    print(f"\n=== (1) Original E82-style axis comparison, EQUAL VOXEL magnitude (3), UNEQUAL physical mm ===")
    d0 = np.array([r["K_tensor_dim0_+3"] for r in records])
    d1 = np.array([r["K_tensor_dim1_+3"] for r in records])
    d2 = np.array([r["K_tensor_dim2_+3"] for r in records])
    print(f"  dim0 (voxel=3): {d0.mean():.4f}   dim1 (voxel=3): {d1.mean():.4f}   dim2 (voxel=3): {d2.mean():.4f}")
    t_02, p_02 = stats.ttest_rel(d0, d2)
    print(f"  dim0 vs dim2 (voxel-matched, mm-mismatched): paired t p={p_02:.4e}")

    print(f"\n=== (2) PHYSICAL-DISTANCE-MATCHED comparison (dim0/dim1 voxel=2 [7.50mm] vs dim2 voxel=3 [7.27mm]) ===")
    dm0 = np.array([r["K_dim0_matched"] for r in records])
    dm1 = np.array([r["K_dim1_matched"] for r in records])
    dm2 = np.array([r["K_dim2_matched"] for r in records])
    print(f"  dim0 (voxel=2, 7.50mm): {dm0.mean():.4f}   dim1 (voxel=2, 7.50mm): {dm1.mean():.4f}   "
          f"dim2 (voxel=3, 7.27mm): {dm2.mean():.4f}")
    t_m02, p_m02 = stats.ttest_rel(dm0, dm2)
    w_m02, pw_m02 = stats.wilcoxon(dm0, dm2)
    print(f"  dim0 vs dim2 (physical-distance-matched): paired t p={p_m02:.4e}, Wilcoxon p={pw_m02:.4e}")
    print(f"  Effect size, voxel-matched comparison: {abs(d0.mean()-d2.mean()):.4f}")
    print(f"  Effect size, physical-matched comparison: {abs(dm0.mean()-dm2.mean()):.4f}")

    survives_control = (pw_m02 < 0.05) and (abs(dm0.mean() - dm2.mean()) > 0.3 * abs(d0.mean() - d2.mean()))
    reduces_substantially = abs(dm0.mean() - dm2.mean()) < 0.3 * abs(d0.mean() - d2.mean())

    if reduces_substantially:
        confound_reading = ("CONFOUND CONFIRMED: the axis difference shrinks substantially once offsets are "
                           "matched for physical distance -- E82's anisotropy was substantially the voxel-"
                           "spacing/resampling confound (z resized-voxels are physically smaller than x/y "
                           "resized-voxels: 2.42mm vs 3.75mm). RETRACT any claim of intrinsic representational "
                           "anisotropy from E82; the underlying phenomenon reduces to ordinary physical-"
                           "distance decay.")
    elif survives_control:
        confound_reading = ("ANISOTROPY SURVIVES the physical-distance control: the axis difference persists "
                           "even when offsets are matched for approximately equal physical mm distance. This "
                           "points to a genuine property of the learned representation, not merely the "
                           "resampling geometry -- worth further characterization.")
    else:
        confound_reading = "AMBIGUOUS: the physical-matched comparison is not clearly significant either way -- report as inconclusive, not as a confirmed finding in either direction."

    print(f"\n=== READING ===\n{confound_reading}")

    print(f"\n=== (3) Reflection symmetry K(+delta) vs K(-delta) ===")
    for dim in ["dim0", "dim1", "dim2"]:
        pos = np.array([r[f"K_tensor_{dim}_+3"] for r in records])
        neg = np.array([r[f"K_tensor_{dim}_-3"] for r in records])
        t_pn, p_pn = stats.ttest_rel(pos, neg)
        print(f"  {dim}: K(+3)={pos.mean():.4f}, K(-3)={neg.mean():.4f}, paired t p={p_pn:.4e} "
              f"({'asymmetric' if p_pn < 0.05 else 'symmetric'})")

    summary = {
        "n_subjects": len(records),
        "mm_per_voxel": MM_PER_VOXEL,
        "K_means": K_means,
        "voxel_matched_effect_size": float(abs(d0.mean() - d2.mean())),
        "physical_matched_effect_size": float(abs(dm0.mean() - dm2.mean())),
        "physical_matched_ttest_p": float(p_m02), "physical_matched_wilcoxon_p": float(pw_m02),
        "confound_reading": confound_reading,
    }
    with open(OUT_DIR / "E83_correspondence_kernel_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E83_correspondence_kernel_summary.json")


if __name__ == "__main__":
    main()
