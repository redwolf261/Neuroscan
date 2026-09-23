"""
Phase E81: Translation-and-Realignment Control (NO TRAINING, pure
inference-time intervention) -- per the user's design, testing the
strongest alternative explanation for E80's coherent-shift result before
accepting "independent absolute-coordinate binding" as the interpretation.

THE CONFOUND E80 DID NOT RULE OUT: a conventional convolutional decoder
is spatially structured (translation-equivariant-ish by construction,
modulo boundary/padding effects). If BOTH internal feature maps are
coherently shifted by t voxels (E80's Arm 3), the network may simply
produce a segmentation that is ALSO displaced by approximately t voxels
-- P_t ~ T_t(P_0). Comparing that displaced prediction against the
UNSHIFTED ground truth would then produce a large Dice drop for a
trivial reason (coordinate-frame mismatch against a fixed label), not
because the internal representations are "corrupted" or because the
network lost genuine information. This would fully explain E80's
counter-intuitive Arm3 > Arm1 result WITHOUT any absolute-coordinate-key
story at all.

THIS PHASE is the clean discriminator, per the user's design:
  1. Reproduce E80's Arm 3 exactly (coherent joint shift of enc1's
     direction AND upconv1 by the same offset t=3).
  2. Compute D_unrealigned = Dice(P_t, Y) -- E80's own Arm 3 quantity,
     reproduced here for a matched comparison.
  3. Compute D_realigned = Dice(T_{-t}(P_t), Y) -- undo the SAME
     translation on the final prediction before scoring against the
     (always fixed) ground truth.
  4. Feature-level equivariance check: compare P_t against T_t(P_0)
     directly (normalized voxel-wise error), not just via Dice -- does
     the coherently-shifted output behave like a simple translated
     version of the clean output, or does it show a genuine
     representational deformation beyond a pure shift?

INTERPRETATION (pre-declared, per the user's own framing):
  OUTCOME A -- D_realigned approx D_clean (D_intact): E80's damage was
    substantially a coordinate-frame displacement artifact. RETRACT the
    "absolute-coordinate binding" interpretation of E80. E65's original
    finding (translation of enc1 ALONE, decoder side untouched --
    genuinely asymmetric, no coordinate-frame confound since only one
    side moves) remains the standing, uncomplicated result.
  OUTCOME B -- D_realigned << D_clean: undoing the coordinate shift on
    the output does NOT recover performance -- something beyond a simple
    global translation is happening; the internal intervention produces
    a genuine representational deformation. E80's phenomenon is real and
    interesting, not an artifact.
Feature-level check refines whichever outcome holds: P_t ~= T_t(P_0)
(near-equivariant) vs P_t far from T_t(P_0) (non-equivariant, genuine
deformation) is diagnostic independent of the Dice-based test.
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
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e80"))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402
from run_e74_spatial_dependence_audit import translate_volume  # noqa: E402
from run_e80_decoder_readout_causality import forward_dual_shift  # noqa: E402

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 0
N_PERM = 1000
N_BOOT = 2000
BASE_OFFSET = 3  # matches E65/E74/E75/E78/E79/E80's own reference offset

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")


def dice_score(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def translate_probs_back(probs_np, offset):
    """probs_np: (D,H,W) numpy array. Undo a +offset roll by rolling
    -offset (torch.roll's inverse), matching translate_volume's own
    wrap-around convention exactly for consistency."""
    t = torch.from_numpy(probs_np)
    t_back = torch.roll(t, shifts=(-offset, -offset, -offset), dims=(0, 1, 2))
    return t_back.numpy()


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

    records = []
    for idx in range(len(val_ds)):
        img, msk, sid = val_ds[idx]
        img_b = img.unsqueeze(0).to(device)
        target_bin = (msk.squeeze(0).numpy() > 0.5).astype(np.float32)

        # clean/intact prediction P_0
        probs_intact = forward_dual_shift(model, img_b, 0, 0, device)
        d_intact = dice_score((probs_intact >= 0.5).astype(np.float32), target_bin)

        # E80's Arm 3: coherent joint shift -> P_t
        probs_t = forward_dual_shift(model, img_b, BASE_OFFSET, BASE_OFFSET, device)
        d_unrealigned = dice_score((probs_t >= 0.5).astype(np.float32), target_bin)

        # realign: undo the SAME shift on the prediction before scoring
        probs_t_realigned = translate_probs_back(probs_t, BASE_OFFSET)
        d_realigned = dice_score((probs_t_realigned >= 0.5).astype(np.float32), target_bin)

        # feature-level equivariance check: P_t vs T_t(P_0)
        probs_intact_translated = translate_volume(
            torch.from_numpy(probs_intact).unsqueeze(0), BASE_OFFSET
        ).squeeze(0).numpy()
        # normalized voxel-wise error between the ACTUAL shifted-condition
        # output and what a PURE translation of the clean output would be
        diff = np.abs(probs_t - probs_intact_translated)
        equivariance_error = float(diff.mean())
        # reference: how different is probs_intact_translated from probs_intact itself
        # (sanity scale -- how much the ground truth's own translated version differs)
        baseline_scale = float(np.abs(probs_intact_translated - probs_intact).mean())

        records.append({
            "subject_id": sid,
            "D_intact": d_intact,
            "D_unrealigned": d_unrealigned, "S_unrealigned": d_intact - d_unrealigned,
            "D_realigned": d_realigned, "S_realigned": d_intact - d_realigned,
            "equivariance_error": equivariance_error, "baseline_scale": baseline_scale,
        })

        if (idx + 1) % 25 == 0:
            print(f"  processed {idx+1}/{len(val_ds)}", flush=True)

    with open(OUT_DIR / "E81_realignment_control_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} subject records.\n")

    d_intact_arr = np.array([r["D_intact"] for r in records])
    d_unrealigned_arr = np.array([r["D_unrealigned"] for r in records])
    d_realigned_arr = np.array([r["D_realigned"] for r in records])
    S_unrealigned = np.array([r["S_unrealigned"] for r in records])
    S_realigned = np.array([r["S_realigned"] for r in records])

    def paired_stats(drop, label):
        t_stat, p_t = stats.ttest_1samp(drop, 0.0)
        w_stat, p_w = stats.wilcoxon(drop)
        rng = np.random.default_rng(SEED)
        signs = rng.choice([-1, 1], size=(N_PERM, len(drop)))
        perm_means = (signs * drop[None, :]).mean(axis=1)
        p_perm = float((np.abs(perm_means) >= np.abs(drop.mean())).mean())
        boot_idx = rng.integers(0, len(drop), size=(N_BOOT, len(drop)))
        boot_means = drop[boot_idx].mean(axis=1)
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        print(f"=== {label}: mean = {drop.mean():.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}]) ===")
        print(f"  t-test p={p_t:.4e}, Wilcoxon p={p_w:.4e}, sign-flip permutation p={p_perm:.4f}\n")
        return {"mean": float(drop.mean()), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                "ttest_p": float(p_t), "wilcoxon_p": float(p_w), "perm_p": p_perm}

    print(f"D_intact (clean) mean: {d_intact_arr.mean():.4f}")
    print(f"D_unrealigned (E80's Arm 3) mean: {d_unrealigned_arr.mean():.4f}")
    print(f"D_realigned (undo shift on output) mean: {d_realigned_arr.mean():.4f}\n")

    stat_unrealigned = paired_stats(S_unrealigned, "S_unrealigned (= E80's Arm 3 damage, reproduced)")
    stat_realigned = paired_stats(S_realigned, "S_realigned (damage AFTER undoing the shift on the output)")

    recovery_frac = 1 - (S_realigned.mean() / S_unrealigned.mean()) if S_unrealigned.mean() > 0 else float("nan")
    print(f"=== Realignment recovery ===")
    print(f"  S_unrealigned = {S_unrealigned.mean():.4f}")
    print(f"  S_realigned   = {S_realigned.mean():.4f}")
    print(f"  Recovery fraction: {recovery_frac*100:.1f}%\n")

    t_comp, p_comp = stats.ttest_rel(S_unrealigned, S_realigned)
    w_comp, pw_comp = stats.wilcoxon(S_unrealigned, S_realigned)
    print(f"  S_unrealigned vs S_realigned paired comparison: t={t_comp:.3f}, p={p_comp:.4e}, Wilcoxon p={pw_comp:.4e}")

    equiv_errors = np.array([r["equivariance_error"] for r in records])
    baseline_scales = np.array([r["baseline_scale"] for r in records])
    print(f"\n=== Feature-level equivariance check ===")
    print(f"  Mean |P_t - T_t(P_0)| (equivariance error): {equiv_errors.mean():.4f}")
    print(f"  Mean |T_t(P_0) - P_0| (reference scale, how much translation alone moves the clean output): {baseline_scales.mean():.4f}")
    relative_equiv_error = equiv_errors.mean() / max(baseline_scales.mean(), 1e-8)
    print(f"  Relative equivariance error: {relative_equiv_error:.3f} "
          f"(near 0 = near-equivariant, near/above 1 = P_t is as different from T_t(P_0) as T_t(P_0) is from P_0 itself)")

    # decision per the user's pre-declared outcome A/B
    if recovery_frac > 0.6:
        outcome = ("OUTCOME A: D_realigned ~ D_intact -- E80's damage was substantially a coordinate-frame "
                  "displacement artifact. RETRACT the 'independent absolute-coordinate binding' interpretation "
                  "of E80's Arm 3 result specifically. E65's original asymmetric finding (translate enc1 ALONE, "
                  "decoder untouched -- no coordinate-frame confound, since only one side moves and the network "
                  "cannot simply produce a globally-shifted output to compensate) remains the standing, "
                  "uncomplicated result and should anchor the causal chain going forward.")
    elif recovery_frac < 0.2:
        outcome = ("OUTCOME B: undoing the coordinate shift on the output does NOT meaningfully recover "
                  "performance -- something beyond a simple global translation is happening under E80's "
                  "coherent shift. The internal intervention produces a genuine representational deformation, "
                  "not just an output-coordinate-frame mismatch. E80's phenomenon stands as real and "
                  "interesting, not an artifact.")
    else:
        outcome = "PARTIAL: some recovery from realignment, but not clean -- both a coordinate-frame effect AND a genuine deformation likely contribute."

    print(f"\n=== VERDICT ===\n{outcome}")

    summary = {
        "n_subjects": len(records), "base_offset": BASE_OFFSET,
        "d_intact_mean": float(d_intact_arr.mean()),
        "d_unrealigned_mean": float(d_unrealigned_arr.mean()),
        "d_realigned_mean": float(d_realigned_arr.mean()),
        "S_unrealigned": stat_unrealigned, "S_realigned": stat_realigned,
        "recovery_fraction": float(recovery_frac),
        "comparison_ttest_p": float(p_comp), "comparison_wilcoxon_p": float(pw_comp),
        "mean_equivariance_error": float(equiv_errors.mean()),
        "mean_baseline_scale": float(baseline_scales.mean()),
        "relative_equivariance_error": float(relative_equiv_error),
        "outcome": outcome,
    }
    with open(OUT_DIR / "E81_realignment_control_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E81_realignment_control_summary.json")


if __name__ == "__main__":
    main()
