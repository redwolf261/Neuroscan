"""
Phase E79: Directional Positional-Dependence Audit (NO TRAINING, pure
inference-time intervention) -- per the user's design, following E78's
finding that direction u(p) = z(p)/||z(p)|| at enc1 carries the large
majority of translation vulnerability (S_u ~9.6x S_r), with magnitude
acting as an amplifier (super-additive interaction, I=0.082).

OPEN QUESTION E78 DID NOT ANSWER: WHY is direction vulnerable to
translation? Two distinct hypotheses, with different intervention
implications:

  HYPOTHESIS A -- direction varies too sharply spatially (locally noisy/
    oversharp). If true, LOCAL SMOOTHING of direction (magnitude held
    fixed) should recover most of the lost robustness, since smoothing
    directly removes local angular noise.
  HYPOTHESIS B -- direction is tied to ABSOLUTE coordinates, not local
    noise. The representation could be locally smooth (neighboring
    voxels agree) while still being a lookup keyed by exact spatial
    address -- translation would destroy this even with zero local
    noise. If true, LOCAL PERMUTATION of direction (scrambling within
    small cells, same construction as E62/E65's local-permutation arm)
    should hurt MUCH LESS than translation, because local permutation
    preserves each voxel's rough coordinate while translation does not.

This mirrors E65's own translation-vs-local-permutation logic exactly,
narrowed here to the direction channel only (magnitude r(p) held fixed
throughout every arm in this phase, per E77's finding that magnitude
must be preserved).

MEASUREMENTS:
  1. Local angular coherence: cos(theta(p, p+delta)) = u(p)^T u(p+delta)
     for 6-connected neighbors, broken out by class (background/boundary/
     interior) and by the network's own error/correct voxels -- does
     direction already look locally noisy in lesion-relevant regions
     BEFORE any intervention? (prerequisite descriptive check for A)
  2. LOCAL DIRECTION SMOOTHING: z' = r * normalize(local_avg(u)),
     magnitude r held EXACTLY fixed. Measures S_smooth = D_intact -
     D_smoothed_direction (with NO translation) -- a pure smoothness
     probe -- AND, more importantly, whether translating the ALREADY-
     SMOOTHED direction still hurts as much as translating the raw
     direction (S_u_after_smooth vs E78's S_u).
  3. LOCAL DIRECTION PERMUTATION: same 2x2x2 non-overlapping-cell
     derangement as E62/E65/E74, applied to u(p) ONLY (magnitude r(p)
     stays at its OWN original spatial location throughout -- only
     direction identity is shuffled within each cell). Measures
     S_direction_local_perm = D_intact - D_direction_permuted.

DECISIVE COMPARISON (mirrors E65's translation vs local-permutation
logic): S_u (E78, full direction translation) vs
S_direction_local_perm (this phase, local direction scrambling only).
  - S_direction_local_perm >> comparable to S_u -> local noise IS the
    problem (hypothesis A): scrambling nearby directions is nearly as
    bad as moving them away entirely.
  - S_direction_local_perm << S_u -> absolute coordinate dependence
    (hypothesis B): preserving rough position (permutation) is much
    safer than losing it (translation), even though both disturb local
    arrangement.

Reuses E65/E74/E78's own translate_volume, permute_cells_derangement,
and decompose/recompose machinery verbatim.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e74"))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e78"))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402
from run_e74_spatial_dependence_audit import translate_volume  # noqa: E402
from run_e78_magnitude_direction_decomposition import decompose, recompose, forward_with_enc1  # noqa: E402

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 0
N_PERM = 1000
N_BOOT = 2000
TRANSLATION_OFFSET = 3
EPS = 1e-8

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def local_avg_direction(u):
    """u: (C,D,H,W) unit vectors. 3x3x3 box average per channel
    (reflection padding, matching E65's own smoothing arm construction),
    then RE-NORMALIZE to unit length -- the averaged vector is not
    generally unit length, so this step is required to stay a pure
    direction (no magnitude leakage back into u)."""
    C = u.shape[0]
    kernel = torch.ones(C, 1, 3, 3, 3, dtype=u.dtype, device=u.device) / 27.0
    u_b = u.unsqueeze(0)
    u_padded = F.pad(u_b, (1, 1, 1, 1, 1, 1), mode="reflect")
    avg = F.conv3d(u_padded, kernel, groups=C).squeeze(0)
    norm = avg.norm(dim=0, keepdim=True).clamp_min(EPS)
    return avg / norm


def random_derangement_batch(n_items, k, rng):
    """Verbatim from E62/E65/E74 (verified correct there)."""
    arange = np.arange(n_items)
    keys = rng.random((k, n_items))
    perms = np.argsort(keys, axis=1)
    for _ in range(n_items):
        fixed_mask = perms == arange[None, :]
        if not fixed_mask.any():
            break
        rows_with_fixed = fixed_mask.any(axis=1)
        perms[rows_with_fixed] = np.roll(perms[rows_with_fixed], shift=1, axis=1)
    assert not (perms == arange[None, :]).any(), "Derangement construction failed to converge -- STOP."
    return perms


def permute_direction_cells(u_np, rng):
    """Same 2x2x2 non-overlapping-cell derangement as E62/E65/E74, applied
    to the DIRECTION tensor u only. Magnitude is recombined afterward at
    each voxel's OWN original spatial location (this function only
    returns the permuted u; recompose() with the UNPERMUTED r is done by
    the caller, so each voxel keeps its own magnitude, only its
    direction identity is shuffled within its 2x2x2 cell)."""
    C, D, H, W = u_np.shape
    assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0
    nD, nH, nW = D // 2, H // 2, W // 2
    x = u_np.reshape(C, nD, 2, nH, 2, nW, 2)
    x = x.transpose(0, 1, 3, 5, 2, 4, 6)
    rows = x.reshape(-1, 8)
    n_rows = rows.shape[0]
    perms = random_derangement_batch(8, n_rows, rng)
    permuted_rows = np.take_along_axis(rows, perms, axis=1)
    out = permuted_rows.reshape(C, nD, nH, nW, 2, 2, 2)
    out = out.transpose(0, 1, 4, 2, 5, 3, 6)
    out = out.reshape(C, D, H, W)
    return out


def angular_coherence_map(u):
    """cos(theta(p, p+delta)) for 6-connected neighbors, averaged over
    the 6 directions -- one scalar map (D,H,W), same shape as the volume
    (boundary voxels use whatever neighbors exist via replicate padding)."""
    C, D, H, W = u.shape
    shifts = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    total = torch.zeros(D, H, W, device=u.device, dtype=u.dtype)
    for dz, dy, dx in shifts:
        u_shifted = torch.roll(u, shifts=(dz, dy, dx), dims=(1, 2, 3))
        dot = (u * u_shifted).sum(dim=0)  # cos(theta) per voxel
        total += dot
    return total / len(shifts)


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

    rng = np.random.default_rng(SEED)
    records = []
    pooled_coherence, pooled_class = [], []

    for idx in range(len(val_ds)):
        img, msk, sid = val_ds[idx]
        img_b = img.unsqueeze(0).to(device)
        target_bin = (msk.squeeze(0).numpy() > 0.5).astype(np.float32)

        with torch.no_grad():
            enc1_intact = model.enc1(img_b).squeeze(0)
        r_intact, u_intact = decompose(enc1_intact)

        # ---- (1) angular coherence descriptive measurement ----
        coherence_map = angular_coherence_map(u_intact)
        coherence_flat = coherence_map.cpu().numpy().flatten()

        from scipy.ndimage import distance_transform_edt
        if target_bin.sum() > 0 and target_bin.sum() < target_bin.size:
            dist_out = distance_transform_edt(1 - target_bin)
            dist_in = distance_transform_edt(target_bin)
            is_boundary = ((target_bin > 0) & (dist_in <= 2)) | ((target_bin == 0) & (dist_out <= 2))
            is_interior = (target_bin > 0) & (dist_in > 2)
        else:
            is_boundary = np.zeros_like(target_bin, dtype=bool)
            is_interior = np.zeros_like(target_bin, dtype=bool)
        class_arr = np.zeros(coherence_flat.shape, dtype=np.int8)
        class_arr[is_boundary.flatten()] = 1
        class_arr[is_interior.flatten()] = 2

        vox_idx = rng.choice(len(coherence_flat), size=min(10000, len(coherence_flat)), replace=False)
        pooled_coherence.append(coherence_flat[vox_idx])
        pooled_class.append(class_arr[vox_idx])

        # ---- baseline (intact) prediction ----
        probs_intact, _ = forward_with_enc1(model, img_b, enc1_intact, device)
        d_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)

        # ---- (2) local direction smoothing ----
        u_smoothed = local_avg_direction(u_intact)
        z_smoothed = recompose(r_intact, u_smoothed)
        probs_smoothed, _ = forward_with_enc1(model, img_b, z_smoothed, device)
        d_smoothed = dice_score((probs_smoothed >= 0.5).astype(np.float32), target_bin)

        # translate the ALREADY-SMOOTHED direction (magnitude still r_intact, untranslated)
        u_smoothed_translated = translate_volume(u_smoothed, TRANSLATION_OFFSET)
        z_smoothed_translated = recompose(r_intact, u_smoothed_translated)
        probs_smoothed_translated, _ = forward_with_enc1(model, img_b, z_smoothed_translated, device)
        d_smoothed_translated = dice_score((probs_smoothed_translated >= 0.5).astype(np.float32), target_bin)

        # ---- (3) local direction permutation (magnitude untouched at each voxel's own location) ----
        u_perm_np = permute_direction_cells(u_intact.cpu().numpy(), rng)
        u_perm = torch.from_numpy(u_perm_np).to(device)
        z_dir_perm = recompose(r_intact, u_perm)
        probs_dir_perm, _ = forward_with_enc1(model, img_b, z_dir_perm, device)
        d_dir_perm = dice_score((probs_dir_perm >= 0.5).astype(np.float32), target_bin)

        records.append({
            "subject_id": sid,
            "D_intact": d_intact,
            "D_smoothed": d_smoothed, "S_smooth": d_intact - d_smoothed,
            "D_smoothed_translated": d_smoothed_translated,
            "S_u_after_smooth": d_intact - d_smoothed_translated,
            "D_dir_local_perm": d_dir_perm, "S_dir_local_perm": d_intact - d_dir_perm,
            "mean_coherence": float(coherence_map.mean().item()),
        })

        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1}/{len(val_ds)}", flush=True)

    with open(OUT_DIR / "E79_directional_audit_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.\n")

    pooled_coherence = np.concatenate(pooled_coherence)
    pooled_class = np.concatenate(pooled_class)

    # ---------------- (1) Angular coherence by class ----------------
    print("=== (1) Local angular coherence cos(theta) by class (higher = smoother/more locally aligned) ===")
    class_names = {0: "background", 1: "boundary", 2: "interior"}
    coherence_by_class = {}
    for c in [0, 1, 2]:
        vals = pooled_coherence[pooled_class == c]
        if len(vals) > 0:
            coherence_by_class[class_names[c]] = float(vals.mean())
            print(f"  {class_names[c]} (n={len(vals)}): mean cos(theta) = {vals.mean():.4f}")

    # ---------------- (2) Smoothing results ----------------
    S_smooth = np.array([r["S_smooth"] for r in records])
    S_u_after_smooth = np.array([r["S_u_after_smooth"] for r in records])

    def paired_stats(drop, label):
        t_stat, p_t = stats.ttest_1samp(drop, 0.0)
        w_stat, p_w = stats.wilcoxon(drop)
        rng2 = np.random.default_rng(SEED)
        signs = rng2.choice([-1, 1], size=(N_PERM, len(drop)))
        perm_means = (signs * drop[None, :]).mean(axis=1)
        p_perm = float((np.abs(perm_means) >= np.abs(drop.mean())).mean())
        boot_idx = rng2.integers(0, len(drop), size=(N_BOOT, len(drop)))
        boot_means = drop[boot_idx].mean(axis=1)
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        print(f"=== {label}: mean = {drop.mean():.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}]) ===")
        print(f"  t-test p={p_t:.4e}, Wilcoxon p={p_w:.4e}, sign-flip permutation p={p_perm:.4f}\n")
        return {"mean": float(drop.mean()), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                "ttest_p": float(p_t), "wilcoxon_p": float(p_w), "perm_p": p_perm}

    print("\n=== (2) LOCAL DIRECTION SMOOTHING ===")
    stat_smooth = paired_stats(S_smooth, "S_smooth (pure smoothing effect, NO translation)")
    stat_u_after_smooth = paired_stats(S_u_after_smooth,
        "S_u_after_smooth (translate the ALREADY-SMOOTHED direction)")
    print(f"  Reference: E78's S_u (translate RAW direction) = 0.2134")
    print(f"  If smoothing recovers robustness: S_u_after_smooth should be << 0.2134")
    recovery_frac = 1 - (S_u_after_smooth.mean() / 0.2134)
    print(f"  Recovery fraction: {recovery_frac*100:.1f}% of S_u's damage removed by pre-smoothing direction\n")

    # ---------------- (3) Local direction permutation ----------------
    S_dir_perm = np.array([r["S_dir_local_perm"] for r in records])
    print("=== (3) LOCAL DIRECTION PERMUTATION (magnitude untouched, direction scrambled within 2x2x2 cells) ===")
    stat_dir_perm = paired_stats(S_dir_perm, "S_dir_local_perm")
    print(f"  Reference: E78's S_u (translate RAW direction) = 0.2134")
    ratio_perm_to_translate = S_dir_perm.mean() / 0.2134
    print(f"  Ratio (local-perm / translate): {ratio_perm_to_translate:.3f}")

    # ---------------- Decisive comparison: A vs B ----------------
    print("\n=== DECISIVE COMPARISON: Hypothesis A (local noise) vs B (absolute coordinate) ===")
    if ratio_perm_to_translate > 0.6:
        hypothesis_reading = "A (local noise): local permutation is nearly as damaging as translation -- direction's vulnerability is mostly about LOCAL arrangement/smoothness, not absolute coordinate binding."
    elif ratio_perm_to_translate < 0.3:
        hypothesis_reading = "B (absolute coordinate): local permutation is much LESS damaging than translation -- direction is bound to ABSOLUTE spatial address, not just locally noisy. Preserving rough position (permutation) is far safer than losing it (translation)."
    else:
        hypothesis_reading = "MIXED: local permutation is meaningfully damaging but clearly less than translation -- both local noise and absolute coordinate binding likely contribute."
    print(hypothesis_reading)

    smoothing_reading = ("Smoothing meaningfully recovers translation-robustness (supports hypothesis A / a "
                         "smoothing-based intervention)" if recovery_frac > 0.4 else
                         "Smoothing does NOT meaningfully recover translation-robustness (argues against a "
                         "simple local-smoothing intervention, consistent with hypothesis B)")
    print(smoothing_reading)

    summary = {
        "n_subjects": len(records),
        "coherence_by_class": coherence_by_class,
        "S_smooth": stat_smooth, "S_u_after_smooth": stat_u_after_smooth,
        "recovery_fraction_from_smoothing": float(recovery_frac),
        "S_dir_local_perm": stat_dir_perm,
        "e78_S_u_reference": 0.2134,
        "ratio_local_perm_to_translate": float(ratio_perm_to_translate),
        "hypothesis_reading": hypothesis_reading,
        "smoothing_reading": smoothing_reading,
    }
    with open(OUT_DIR / "E79_directional_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E79_directional_audit_summary.json")


if __name__ == "__main__":
    main()
