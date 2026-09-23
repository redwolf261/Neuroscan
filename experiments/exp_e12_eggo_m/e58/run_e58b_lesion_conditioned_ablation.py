"""
Phase E58 Stage 1b: lesion-conditioned bottleneck octant ablation.

CONTEXT: Stage 1 (run_e58_octant_ablation.py) found that neither of two
FIXED, arbitrary octants (diagonally opposite corners of the 8^3
bottleneck) reproduces E48's full-ablation size-dependence signature
(octant A: rho=-0.186, CI does not overlap E48's -0.454; octant B:
rho=-0.081, not significant). This established "not concentrated in an
arbitrary fixed corner" -- it did NOT establish whether the bottleneck's
relevant content is organized RELATIVE TO LESION LOCATION rather than
in absolute/global coordinates.

THIS PHASE asks the sharper, different question: for EACH subject,
zero the SPECIFIC octant of the bottleneck grid that spatially
corresponds to that subject's own lesion location (computed post-hoc
from the GT mask -- a legitimate causal-analysis use of GT, not a model
input/leak), versus a FIXED reference octant chosen to be maximally far
from the lesion for that same subject. If the lesion-corresponding
octant produces a much larger, more size-dependent effect than a
lesion-far octant (for the SAME subject, isolating the lesion-location
factor from any subject-level confound), that is evidence the
bottleneck's relevant content is spatially organized RELATIVE TO the
lesion, even though Stage 1 found no single ABSOLUTE octant dominates
globally -- i.e. "distributed in absolute coordinates, but lesion-
relative/localized in representation space" (exactly the alternative
the user's own decision tree flagged).

METHOD: for each subject, the lesion's centroid (in the 64^3 grid, from
the resized mask) is mapped to bottleneck coordinates (8^3 grid, a
straightforward 1/8 scalar ratio, matching this project's own
established coordinate-mapping convention from E55) to identify which
one of the 8 octants contains it. Two conditions per subject:
  1. lesion_octant zeroed -- the octant containing the lesion centroid.
  2. far_octant zeroed -- the octant whose CENTER is geometrically
     farthest (Euclidean, in octant-index space) from the lesion
     octant, computed per-subject (not the same fixed octant for
     everyone -- this is the key difference from Stage 1's own design).
Manual trunk reimplementation, verified bit-for-bit before any ablated
result is trusted, exactly as Stage 1's own discipline.

STATISTICS: (a) PAIRED comparison (lesion_octant drop vs far_octant
drop, same subject, same octant SIZE, only WHICH octant differs) --
paired t-test + Wilcoxon, isolating the lesion-location factor cleanly.
(b) Spearman(native_size, drop) for each condition separately, same
1000-trial permutation + bootstrap-CI discipline as Stage 1, for direct
comparison to both Stage 1's own octant rhos and E48's full-ablation
reference (-0.454).

PRE-DECLARED DECISION RULE: lesion-relative organization is supported
if (a) lesion_octant drop is significantly larger than far_octant drop
(paired test, p<0.05) AND (b) lesion_octant's own rho(size, drop) is
closer in magnitude to E48's reference (-0.454) than far_octant's rho
is. Otherwise: no evidence of lesion-relative spatial organization
beyond what Stage 1 already showed -- report as a further data point
supporting "distributed," not evidence for a new "lesion-relative"
alternative.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000
N_BOOT = 1000

CKPT_PATH = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs"
             / "DeepSup_D4only_seed0" / "checkpoints" / "best.pth")

E48_REFERENCE_RHO = -0.454

# All 8 octant index-triples (which octant of the 2x2x2 grid, matching
# Stage 1's own 4x4x4-per-octant partition of the 8^3 bottleneck).
OCTANT_INDICES = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]


def octant_slice(idx):
    i, j, k = idx
    return (slice(i * 4, i * 4 + 4), slice(j * 4, j * 4 + 4), slice(k * 4, k * 4 + 4))


def fractional_occupancy_64(seg_binary_native):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=(64, 64, 64), mode="area").squeeze().numpy()
    return frac


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def lesion_octant_index(target_bin_64):
    """target_bin_64: (64,64,64) binary mask. Returns the (i,j,k) octant
    index (each in {0,1}) containing the mask's own centroid, mapping
    64^3 grid coords -> 8^3 bottleneck grid coords via the straightforward
    1/8 scalar ratio (64/8=8 voxels per bottleneck cell), then to a
    2x2x2 octant index via //4 on the resulting 8^3 coordinate."""
    coords = np.argwhere(target_bin_64 > 0.5)
    centroid_64 = coords.mean(axis=0)  # (3,) in 64^3 grid
    centroid_8 = centroid_64 / 8.0  # -> 8^3 bottleneck grid
    octant = tuple(int(min(7, max(0, round(c)))) // 4 for c in centroid_8)
    return octant


def farthest_octant(lesion_oct):
    """Returns the octant index whose Euclidean distance (in 3-bit
    octant-index space) from lesion_oct is maximal -- the geometric
    opposite corner. For a 2x2x2 space this is always well-defined:
    flipping every bit gives the unique farthest octant."""
    return tuple(1 - b for b in lesion_oct)


def forward_with_octant_ablation(model, image, octant_slice_, device):
    with torch.no_grad():
        enc1 = model.enc1(image)
        pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1)
        pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2)
        pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)

        if octant_slice_ is not None:
            bottleneck = bottleneck.clone()
            d_sl, h_sl, w_sl = octant_slice_
            bottleneck[:, :, d_sl, h_sl, w_sl] = 0.0

        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)

        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)

        upconv1 = model.upconv1(dec2)
        cat1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = model.dec1(cat1)

        probs = model.seg_head(dec1)
        return probs.squeeze(0).squeeze(0).cpu().numpy()


def spearman_with_permutation(x, y, n_perm, seed):
    rho, p_parametric = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    perm_rhos = np.empty(n_perm)
    for i in range(n_perm):
        perm_y = rng.permutation(y)
        perm_rhos[i], _ = stats.spearmanr(x, perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    return float(rho), float(p_parametric), p_perm


def bootstrap_rho_ci(x, y, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rhos[i], _ = stats.spearmanr(x[idx], y[idx])
    return float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded D4-only checkpoint: best_val_dice={ckpt.get('best_val_dice')}", flush=True)

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    image0, _, _ = val_dataset[0]
    image0_b = image0.unsqueeze(0).to(device)
    with torch.no_grad():
        real = model(image0_b)["probs"].squeeze(0).squeeze(0).cpu().numpy()
    manual = forward_with_octant_ablation(model, image0_b, None, device)
    max_diff = float(np.abs(real - manual).max())
    print(f"\n[Sanity check] octant_slice=None vs real forward: max abs diff = {max_diff:.6e}")
    assert max_diff == 0.0, "Manual trunk reimplementation does not match real forward() -- STOP, bug present."
    print("[Sanity check] PASS.\n")

    records = []
    octant_distribution = {}
    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_size = int(seg_binary_native.sum())

        mask_frac_64 = fractional_occupancy_64(seg_binary_native)
        target_bin = (mask_frac_64 > 0.5).astype(np.float32)

        if target_bin.sum() == 0:
            print(f"  [{subject_id}] SKIP -- empty resized mask (lesion vanished at 64^3), "
                  f"no centroid to condition on")
            continue

        lesion_oct = lesion_octant_index(target_bin)
        far_oct = farthest_octant(lesion_oct)
        octant_distribution[str(lesion_oct)] = octant_distribution.get(str(lesion_oct), 0) + 1

        probs_intact = forward_with_octant_ablation(model, image_b, None, device)
        probs_lesion_oct = forward_with_octant_ablation(model, image_b, octant_slice(lesion_oct), device)
        probs_far_oct = forward_with_octant_ablation(model, image_b, octant_slice(far_oct), device)

        dice_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)
        dice_lesion_oct = dice_score((probs_lesion_oct >= 0.5).astype(np.float32), target_bin)
        dice_far_oct = dice_score((probs_far_oct >= 0.5).astype(np.float32), target_bin)

        records.append({
            "subject_id": subject_id, "native_size": native_size,
            "lesion_octant": lesion_oct, "far_octant": far_oct,
            "dice_intact": dice_intact,
            "dice_lesion_oct": dice_lesion_oct, "drop_lesion_oct": dice_intact - dice_lesion_oct,
            "dice_far_oct": dice_far_oct, "drop_far_oct": dice_intact - dice_far_oct,
        })

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{len(val_dataset)} subjects", flush=True)

    with open(OUT_DIR / "E58b_lesion_conditioned_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records ({len(val_dataset) - len(records)} skipped, empty resized mask).")
    print(f"Lesion octant distribution across subjects: {octant_distribution}")

    native_size = np.array([r["native_size"] for r in records], dtype=np.float64)
    dice_intact = np.array([r["dice_intact"] for r in records])
    drop_lesion = np.array([r["drop_lesion_oct"] for r in records], dtype=np.float64)
    drop_far = np.array([r["drop_far_oct"] for r in records], dtype=np.float64)

    print(f"\n=== E58b Lesion-Conditioned Octant Ablation: n={len(records)} subjects ===")
    print(f"Mean dice_intact = {dice_intact.mean():.4f}")
    print(f"Mean drop (lesion octant) = {drop_lesion.mean():.4f}")
    print(f"Mean drop (far octant)    = {drop_far.mean():.4f}")

    # ---- Paired comparison: lesion octant vs far octant, same subjects ----
    t_stat, t_p = stats.ttest_rel(drop_lesion, drop_far)
    w_stat, w_p = stats.wilcoxon(drop_lesion, drop_far)
    print(f"\nPaired comparison (lesion_oct drop vs far_oct drop, n={len(records)}):")
    print(f"  paired t-test: t={t_stat:.4f}, p={t_p:.4f}")
    print(f"  Wilcoxon signed-rank: p={w_p:.4f}")

    # ---- Spearman(size, drop) for each condition ----
    rho_lesion, p_lesion_param, p_lesion_perm = spearman_with_permutation(native_size, drop_lesion, N_PERM, SEED)
    rho_far, p_far_param, p_far_perm = spearman_with_permutation(native_size, drop_far, N_PERM, SEED + 1)
    ci_lesion = bootstrap_rho_ci(native_size, drop_lesion, N_BOOT, SEED)
    ci_far = bootstrap_rho_ci(native_size, drop_far, N_BOOT, SEED + 1)

    print(f"\nLesion octant: Spearman(native_size, drop) = {rho_lesion:+.4f} "
          f"(perm p={p_lesion_perm:.4f}) 95% CI=[{ci_lesion[0]:+.4f}, {ci_lesion[1]:+.4f}]")
    print(f"Far octant:    Spearman(native_size, drop) = {rho_far:+.4f} "
          f"(perm p={p_far_perm:.4f}) 95% CI=[{ci_far[0]:+.4f}, {ci_far[1]:+.4f}]")
    print(f"Reference (E48 full ablation): rho = {E48_REFERENCE_RHO}")
    print(f"Reference (Stage 1, fixed octant A): rho = -0.1856")
    print(f"Reference (Stage 1, fixed octant B): rho = -0.0814")

    # ---- Pre-declared decision rule ----
    paired_sig = (t_p < 0.05) and (w_p < 0.05)
    lesion_closer_to_ref = abs(rho_lesion - E48_REFERENCE_RHO) < abs(rho_far - E48_REFERENCE_RHO)
    lesion_relative_supported = paired_sig and lesion_closer_to_ref

    print(f"\nPaired drop significantly larger for lesion octant: {paired_sig}")
    print(f"Lesion-octant rho closer to E48 reference than far-octant rho: {lesion_closer_to_ref}")
    print(f"\n=== DECISION: {'LESION-RELATIVE spatial organization supported' if lesion_relative_supported else 'NOT supported -- consistent with distributed, not lesion-relative, organization'} ===")

    summary = {
        "n_subjects": len(records), "n_skipped_empty_mask": len(val_dataset) - len(records),
        "mean_dice_intact": float(dice_intact.mean()),
        "mean_drop_lesion_octant": float(drop_lesion.mean()),
        "mean_drop_far_octant": float(drop_far.mean()),
        "paired_ttest": {"t": float(t_stat), "p": float(t_p)},
        "wilcoxon": {"p": float(w_p)},
        "lesion_octant_stats": {"rho": rho_lesion, "perm_p": p_lesion_perm, "ci_95": list(ci_lesion)},
        "far_octant_stats": {"rho": rho_far, "perm_p": p_far_perm, "ci_95": list(ci_far)},
        "e48_reference_rho": E48_REFERENCE_RHO,
        "stage1_fixed_octant_A_rho": -0.1856, "stage1_fixed_octant_B_rho": -0.0814,
        "decision": "lesion_relative_supported" if lesion_relative_supported else "not_supported",
    }
    with open(OUT_DIR / "E58b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E58b_summary.json")


if __name__ == "__main__":
    main()
