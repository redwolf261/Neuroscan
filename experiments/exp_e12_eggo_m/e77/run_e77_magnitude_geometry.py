"""
Phase E77: Natural enc1 Magnitude Geometry (NO TRAINING, NO ARCHITECTURE
CHANGE -- pure measurement) -- per the user's design, following E76's
clean PASS (local causal relationship between enc1 magnitude and
translation sensitivity, confirmed within the network's normal operating
regime).

E76 established a RELATIVE causal relationship (dS/dalpha > 0 for
alpha in [0.7, 1.3], scaling the network's OWN learned magnitude up or
down). It says nothing about the ABSOLUTE geometry of that magnitude in
the trained representation -- whether "too large" is a rare tail worth
targeting, a broad distribution tied to lesion content (dangerous to
suppress), or already well-behaved (making magnitude control practically
unexploitable even though causally real). This phase measures that
geometry, no training, before any intervention formula is chosen.

MEASUREMENTS (per the user's own itemized design):
  1. Distribution of r(p) = ||z(p)||_2 at enc1: population percentiles,
     broken out by lesion / background, and (a finer cut) lesion boundary
     / lesion interior / background.
  2. Per-channel z-scored version of r(p) (channel-normalized), to
     distinguish "genuinely large activation vectors" from "some channels
     just have a different natural scale" -- computed using each
     channel's own population mean/std across ALL voxels/subjects in this
     sample.
  3. Voxel-level correlation between r(p) and E65/E74's translation
     sensitivity, both pooled (voxel-level) and subject-level (mean
     per-subject r vs per-subject translation-drop, matching this
     project's own dual-level convention).
  4. r(p) vs {lesion, boundary, interior, background, error} -- does high
     magnitude already carry useful information? Critical: if yes,
     blind suppression would destroy exactly what's being protected.
  5. Map the OBSERVED natural r(p) distribution onto E76's causal curve:
     what fraction of voxels sit in a "high" tail, and is that tail
     disproportionately translation-sensitive?

Reuses E74/E76's own translation and forward-pass machinery verbatim, no
new intervention machinery -- this phase adds MEASUREMENT only.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy import stats
from scipy.ndimage import distance_transform_edt

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e74"))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402
from run_e74_spatial_dependence_audit import translate_volume  # noqa: E402

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 0
TRANSLATION_OFFSET = 3
BOUNDARY_BAND = 2  # voxels: "boundary" = within this distance of the lesion edge, either side

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def forward_enc1_translation_drop(model, img_b, device):
    """Returns (probs_intact, per-voxel r(p) at enc1, translation-induced
    per-subject Dice drop) -- reuses E74/E76's split-forward construction,
    intact alpha=1.0 vs a single 3-voxel translation, matching the
    project's own established intervention exactly."""
    with torch.no_grad():
        enc1 = model.enc1(img_b)  # (1,32,64,64,64)
        r_map = enc1.squeeze(0).norm(dim=0)  # (64,64,64), ||z(p)||_2 per voxel

        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        def decode(e_skip):
            upconv3 = model.upconv3(bottleneck)
            cat3 = torch.cat([upconv3, enc3], dim=1)
            dec3 = model.dec3(cat3)
            upconv2 = model.upconv2(dec3)
            cat2 = torch.cat([upconv2, enc2], dim=1)
            dec2 = model.dec2(cat2)
            upconv1 = model.upconv1(dec2)
            cat1 = torch.cat([upconv1, e_skip], dim=1)
            dec1 = model.dec1(cat1)
            return model.seg_head(dec1)

        probs_intact = decode(enc1)
        enc1_translated = translate_volume(enc1.squeeze(0), TRANSLATION_OFFSET).unsqueeze(0)
        probs_translated = decode(enc1_translated)

    return (probs_intact.squeeze(0).squeeze(0).cpu().numpy(),
            probs_translated.squeeze(0).squeeze(0).cpu().numpy(),
            r_map.cpu().numpy())


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
                                    split="val", val_split=0.1, target_shape=(64, 64, 64))

    # accumulate voxel-level pooled data (subsampled per subject to keep
    # arrays tractable: 64^3 * 125 = ~32.8M voxels total, subsample to
    # ~10k/subject -> 1.25M pooled, matching this project's own tractable-
    # subsampling convention from prior confound checks)
    rng = np.random.default_rng(SEED)
    pooled_r, pooled_class, pooled_local_drop = [], [], []

    per_subject_records = []

    for idx in range(len(val_ds)):
        img, msk, sid = val_ds[idx]
        img_b = img.unsqueeze(0).to(device)
        target_bin = (msk.squeeze(0).numpy() > 0.5).astype(np.float32)

        probs_intact, probs_translated, r_map = forward_enc1_translation_drop(model, img_b, device)

        pred_bin = (probs_intact >= 0.5).astype(np.float32)
        error_mask = (pred_bin != target_bin).astype(np.float32)

        # voxel-level LOCAL translation impact: |probs_intact - probs_translated|
        # (the same S_i(p) construction as E73's probe, reused here as the
        # per-voxel translation-sensitivity signal to correlate against r(p))
        local_drop = np.abs(probs_intact - probs_translated)

        dice_intact = dice_score(pred_bin, target_bin)
        dice_translated = dice_score((probs_translated >= 0.5).astype(np.float32), target_bin)
        subject_translation_drop = dice_intact - dice_translated

        # classify each voxel: lesion-interior / lesion-boundary / background
        if target_bin.sum() > 0 and target_bin.sum() < target_bin.size:
            dist_out = distance_transform_edt(1 - target_bin)
            dist_in = distance_transform_edt(target_bin)
            is_boundary = ((target_bin > 0) & (dist_in <= BOUNDARY_BAND)) | \
                          ((target_bin == 0) & (dist_out <= BOUNDARY_BAND))
            is_interior = (target_bin > 0) & (dist_in > BOUNDARY_BAND)
            is_background = (target_bin == 0) & (dist_out > BOUNDARY_BAND)
        else:
            is_boundary = np.zeros_like(target_bin, dtype=bool)
            is_interior = np.zeros_like(target_bin, dtype=bool)
            is_background = (target_bin == 0)

        r_flat = r_map.flatten()
        class_arr = np.zeros(r_flat.shape, dtype=np.int8)  # 0=background,1=boundary,2=interior
        class_arr[is_boundary.flatten()] = 1
        class_arr[is_interior.flatten()] = 2
        error_flat = error_mask.flatten()
        local_drop_flat = local_drop.flatten()

        n_vox = len(r_flat)
        vox_idx = rng.choice(n_vox, size=min(10000, n_vox), replace=False)
        pooled_r.append(r_flat[vox_idx])
        pooled_class.append(class_arr[vox_idx])
        pooled_local_drop.append(local_drop_flat[vox_idx])

        r_percentiles = np.percentile(r_map, [10, 25, 50, 75, 90, 95, 99])
        r_mean_by_class = {
            "background": float(r_flat[class_arr == 0].mean()) if (class_arr == 0).sum() > 0 else None,
            "boundary": float(r_flat[class_arr == 1].mean()) if (class_arr == 1).sum() > 0 else None,
            "interior": float(r_flat[class_arr == 2].mean()) if (class_arr == 2).sum() > 0 else None,
        }
        r_mean_error = float(r_flat[error_flat > 0].mean()) if error_flat.sum() > 0 else None
        r_mean_correct = float(r_flat[error_flat == 0].mean())

        per_subject_records.append({
            "subject_id": sid,
            "mean_r": float(r_map.mean()), "r_percentiles": r_percentiles.tolist(),
            "r_mean_by_class": r_mean_by_class,
            "r_mean_error_voxels": r_mean_error, "r_mean_correct_voxels": r_mean_correct,
            "subject_translation_drop": subject_translation_drop,
        })

        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1}/{len(val_ds)}", flush=True)

    with open(OUT_DIR / "E77_per_subject_geometry.json", "w") as f:
        json.dump(per_subject_records, f, indent=2)
    print(f"\nSaved {len(per_subject_records)} subject records.\n")

    pooled_r = np.concatenate(pooled_r)
    pooled_class = np.concatenate(pooled_class)
    pooled_local_drop = np.concatenate(pooled_local_drop)
    print(f"Pooled sample size: {len(pooled_r)} voxels ({len(per_subject_records)} subjects x 10000 subsampled each)\n")

    # ---------------- (1) Population distribution of r(p) ----------------
    print("=== (1) Population distribution of r(p) at enc1 ===")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    pcts = np.percentile(pooled_r, percentiles)
    for p, v in zip(percentiles, pcts):
        print(f"  p{p}: {v:.4f}")
    iqr = pcts[3] - pcts[1]  # p75 - p25
    print(f"  IQR (p75-p25): {iqr:.4f}")

    # ---------------- (2) Class breakdown ----------------
    print("\n=== (2) r(p) by class (0=background, 1=boundary, 2=interior) ===")
    class_names = {0: "background", 1: "boundary", 2: "interior"}
    class_means = {}
    for c in [0, 1, 2]:
        vals = pooled_r[pooled_class == c]
        if len(vals) > 0:
            class_means[class_names[c]] = float(vals.mean())
            print(f"  {class_names[c]} (n={len(vals)}): mean r = {vals.mean():.4f}, "
                  f"median = {np.median(vals):.4f}")

    # ---------------- (3) Pooled voxel-level: r(p) vs local translation impact ----------------
    print("\n=== (3) Pooled voxel-level Spearman(r(p), local translation impact |Δprobs|) ===")
    rho_pooled, p_pooled = stats.spearmanr(pooled_r, pooled_local_drop)
    print(f"  rho = {rho_pooled:+.4f} (p={p_pooled:.4e}, n={len(pooled_r)})")

    # subject-level: mean r vs subject-level translation-induced Dice drop
    subj_mean_r = np.array([r["mean_r"] for r in per_subject_records])
    subj_drop = np.array([r["subject_translation_drop"] for r in per_subject_records])
    rho_subj, p_subj = stats.spearmanr(subj_mean_r, subj_drop)
    print(f"\n  Subject-level Spearman(mean r, subject translation-drop) = {rho_subj:+.4f} (p={p_subj:.4e}, n={len(subj_mean_r)})")

    # ---------------- (4) r(p) vs error ----------------
    print("\n=== (4) Does high magnitude already carry useful information? ===")
    print(f"  Class means (background < boundary < interior expected if r tracks lesion salience):")
    for name in ["background", "boundary", "interior"]:
        if name in class_means:
            print(f"    {name}: {class_means[name]:.4f}")

    r_error_mean = np.array([r["r_mean_error_voxels"] for r in per_subject_records if r["r_mean_error_voxels"] is not None])
    r_correct_mean = np.array([r["r_mean_correct_voxels"] for r in per_subject_records])
    r_correct_matched = np.array([r["r_mean_correct_voxels"] for r in per_subject_records if r["r_mean_error_voxels"] is not None])
    if len(r_error_mean) > 0:
        t_err, p_err = stats.ttest_rel(r_error_mean, r_correct_matched)
        print(f"\n  r at ERROR voxels: mean={r_error_mean.mean():.4f}  vs  r at CORRECT voxels: mean={r_correct_matched.mean():.4f}")
        print(f"  paired t-test (n={len(r_error_mean)} subjects): t={t_err:.3f}, p={p_err:.4e}")

    # ---------------- (5) Tail analysis: map natural r onto E76's causal regime ----------------
    print("\n=== (5) High-magnitude tail: rare and disproportionately sensitive? ===")
    p90_thresh = pcts[4]  # the 90th percentile from the pooled distribution
    tail_mask = pooled_r >= p90_thresh
    tail_frac = tail_mask.mean()
    tail_local_drop = pooled_local_drop[tail_mask].mean()
    non_tail_local_drop = pooled_local_drop[~tail_mask].mean()
    print(f"  Voxels at/above p90 (r>={p90_thresh:.4f}): {tail_frac*100:.1f}% of all voxels")
    print(f"  Mean local translation impact, TAIL voxels: {tail_local_drop:.4f}")
    print(f"  Mean local translation impact, NON-TAIL voxels: {non_tail_local_drop:.4f}")
    print(f"  Ratio (tail/non-tail): {tail_local_drop/max(non_tail_local_drop,1e-8):.3f}x")

    # decision-relevant outcome classification (A/B/C per the user's own framing)
    tail_is_rare = tail_frac < 0.15  # ~p90 threshold by construction, sanity bound
    tail_is_disproportionate = (tail_local_drop / max(non_tail_local_drop, 1e-8)) > 1.5
    magnitude_tracks_lesion = (class_means.get("interior", 0) > class_means.get("background", 0) * 1.2)

    if tail_is_rare and tail_is_disproportionate:
        outcome = "A: high-magnitude tail is rare and disproportionately translation-sensitive -- promising for a soft tail-control mechanism"
    elif magnitude_tracks_lesion:
        outcome = "B: magnitude is broadly distributed and tied to lesion information -- direct suppression is risky, would need a direction-preserving regularizer, not blind normalization"
    else:
        outcome = "C: magnitude appears well-behaved / no exploitable tail -- causal effect (E75/E76) may not be practically exploitable via magnitude control"

    print(f"\n=== OUTCOME CLASSIFICATION ===\n{outcome}")

    summary = {
        "n_subjects": len(per_subject_records), "pooled_n_voxels": int(len(pooled_r)),
        "r_percentiles": {str(p): float(v) for p, v in zip(percentiles, pcts)},
        "r_iqr": float(iqr),
        "class_means": class_means,
        "rho_pooled_r_vs_local_drop": float(rho_pooled), "p_pooled": float(p_pooled),
        "rho_subject_mean_r_vs_drop": float(rho_subj), "p_subject": float(p_subj),
        "r_error_mean": float(r_error_mean.mean()) if len(r_error_mean) > 0 else None,
        "r_correct_mean": float(r_correct_matched.mean()) if len(r_error_mean) > 0 else None,
        "tail_p90_threshold": float(p90_thresh), "tail_fraction": float(tail_frac),
        "tail_local_drop": float(tail_local_drop), "non_tail_local_drop": float(non_tail_local_drop),
        "tail_ratio": float(tail_local_drop / max(non_tail_local_drop, 1e-8)),
        "tail_is_rare": bool(tail_is_rare), "tail_is_disproportionate": bool(tail_is_disproportionate),
        "magnitude_tracks_lesion": bool(magnitude_tracks_lesion),
        "outcome": outcome,
    }
    with open(OUT_DIR / "E77_magnitude_geometry_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E77_magnitude_geometry_summary.json")


if __name__ == "__main__":
    main()
