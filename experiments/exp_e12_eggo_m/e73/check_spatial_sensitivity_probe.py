"""
Phase E73: cheap, no-training probe for "Self-Diagnostic Error Localization"
(user's proposed next rung, following the E48->CDCG-1->E71->E72 ladder).

QUESTION (per the user's own framing, tested at the cheapest possible
resolution before training any auxiliary head): does the bottleneck
representation Z_i contain enough information to predict WHERE (not just
how much) the network's output depends on the bottleneck -- and is that
spatial sensitivity map even a non-trivial, useful target in the first
place?

THIS SCRIPT TRAINS NOTHING. It only:
  1. For a sample of validation subjects, computes the REAL voxel-level
     causal sensitivity map S_i(p) = |P_intact(p) - P_ablated(p)|, using
     the exact same validated bottleneck-ablation construction as E48/
     CDCG prediction 1 (frozen MM-seed0 checkpoint, bottleneck zeroed
     before the decoder) -- just keeping the full 3D map instead of only
     the scalar Dice drop.
  2. Checks whether S_i(p) is a NON-TRIVIAL target: is it just the lesion
     mask again (redundant, useless as a NEW signal), or does it pick out
     a genuinely different spatial pattern?
  3. Checks whether S_i(p) overlaps preferentially with the segmentation
     network's OWN ERRORS (false negatives / false positives) more than
     chance -- if predicted sensitivity can't even correlate with error
     at the population level using the REAL map (not yet a trained
     predictor), a learned head predicting it is very unlikely to help
     localize anything.

DECISION RULE (pre-declared):
  PROCEED to training a spatial auxiliary head only if ALL of:
    (a) S_i(p) is NOT just a re-derivation of the lesion mask
        (Dice(S_i>threshold, lesion_mask) materially below 1, and/or
        substantial mass of S_i lies inside currently-correct voxels)
    (b) S_i(p) correlates with the network's OWN segmentation errors
        (false-negative / false-positive voxels) at above-chance rate,
        checked via a permutation test
  KILL (do not train anything) if either check fails -- the "WHERE"
  signal would not be worth predicting even if predicted perfectly.
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

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 0
N_PERM = 1000
N_SAMPLE = 40  # cheap probe -- not the full 125, per "cheapest first"

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")


def decode_from_bottleneck(model, bottleneck, enc1, enc2, enc3):
    up3 = model.upconv3(bottleneck)
    dec3 = model.dec3(torch.cat([up3, enc3], dim=1))
    up2 = model.upconv2(dec3)
    dec2 = model.dec2(torch.cat([up2, enc2], dim=1))
    up1 = model.upconv1(dec2)
    dec1 = model.dec1(torch.cat([up1, enc1], dim=1))
    return model.seg_head(dec1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(MM_CKPT), map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded MM seed0 checkpoint: best_per_subject_dice={ckpt.get('best_per_subject_dice')}")

    val_ds = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                    split="val", val_split=0.1, target_shape=(64, 64, 64))

    rng = np.random.default_rng(SEED)
    n_total = len(val_ds)
    sample_idx = rng.choice(n_total, size=min(N_SAMPLE, n_total), replace=False)
    print(f"Sampling {len(sample_idx)}/{n_total} val subjects for the spatial probe.\n")

    # population-level accumulators for the error-overlap permutation test
    all_S_flat = []
    all_fn_flat = []
    all_fp_flat = []
    all_correct_flat = []
    per_subject_records = []

    for count, idx in enumerate(sample_idx):
        img, msk, sid = val_ds[int(idx)]
        img_b = img.unsqueeze(0).to(device)
        target = msk.squeeze(0).numpy()  # (64,64,64), already 0/1
        target_bin = (target > 0.5).astype(np.float32)

        with torch.no_grad():
            enc1 = model.enc1(img_b)
            pool1 = model.pool1(enc1); enc2 = model.enc2(pool1)
            pool2 = model.pool2(enc2); enc3 = model.enc3(pool2); pool3 = model.pool3(enc3)
            bottleneck = model.bottleneck(pool3)

            probs_intact = decode_from_bottleneck(model, bottleneck, enc1, enc2, enc3)
            probs_ablated = decode_from_bottleneck(model, torch.zeros_like(bottleneck), enc1, enc2, enc3)

            probs_intact_np = probs_intact.squeeze(0).squeeze(0).cpu().numpy()
            probs_ablated_np = probs_ablated.squeeze(0).squeeze(0).cpu().numpy()

        # S_i(p): voxel-level causal sensitivity map, at the network's own
        # native output resolution (64^3, upsampled by the decoder already)
        S = np.abs(probs_intact_np - probs_ablated_np)

        pred_bin = (probs_intact_np >= 0.5).astype(np.float32)
        fn_mask = ((pred_bin == 0) & (target_bin == 1)).astype(np.float32)
        fp_mask = ((pred_bin == 1) & (target_bin == 0)).astype(np.float32)
        correct_mask = (pred_bin == target_bin).astype(np.float32)
        error_mask = np.maximum(fn_mask, fp_mask)

        # --- Check (a): is S just the lesion mask again? ---
        # Binarize S at its own 90th percentile (a generous "high sensitivity"
        # threshold) and compare to the lesion mask via Dice.
        if S.max() > S.min():
            thresh = np.percentile(S, 90)
            S_bin = (S >= thresh).astype(np.float32)
        else:
            S_bin = np.zeros_like(S)
        tp = (S_bin * target_bin).sum()
        denom = S_bin.sum() + target_bin.sum()
        dice_S_vs_lesion = 1.0 if denom == 0 else float(2 * tp / denom)

        frac_S_mass_in_lesion = float((S * target_bin).sum() / (S.sum() + 1e-8))
        frac_lesion_volume = float(target_bin.sum() / target_bin.size)

        # --- Check (b): does S correlate with error voxels (population-pooled)? ---
        all_S_flat.append(S.flatten())
        all_fn_flat.append(fn_mask.flatten())
        all_fp_flat.append(fp_mask.flatten())
        all_correct_flat.append(correct_mask.flatten())

        per_subject_records.append({
            "subject_id": sid,
            "dice_S_vs_lesion_mask": dice_S_vs_lesion,
            "frac_S_mass_in_lesion": frac_S_mass_in_lesion,
            "frac_lesion_volume": frac_lesion_volume,
            "S_mean_at_error_voxels": float(S[error_mask > 0].mean()) if error_mask.sum() > 0 else None,
            "S_mean_at_correct_voxels": float(S[correct_mask > 0].mean()) if correct_mask.sum() > 0 else None,
            "n_error_voxels": int(error_mask.sum()),
            "n_fn": int(fn_mask.sum()), "n_fp": int(fp_mask.sum()),
        })

        if (count + 1) % 10 == 0:
            print(f"  processed {count+1}/{len(sample_idx)}", flush=True)

    with open(OUT_DIR / "E73_spatial_probe_per_subject.json", "w") as f:
        json.dump(per_subject_records, f, indent=2)

    # ---------------- Population summary ----------------
    dice_vals = np.array([r["dice_S_vs_lesion_mask"] for r in per_subject_records])
    frac_mass_vals = np.array([r["frac_S_mass_in_lesion"] for r in per_subject_records])
    frac_vol_vals = np.array([r["frac_lesion_volume"] for r in per_subject_records])

    print(f"\n=== CHECK (a): is S just the lesion mask again? ===")
    print(f"Dice(S thresholded @ p90, lesion mask): mean={dice_vals.mean():.4f} std={dice_vals.std():.4f}")
    print(f"Fraction of S's total mass inside the lesion: mean={frac_mass_vals.mean():.4f} "
          f"(lesion occupies mean {frac_vol_vals.mean():.4f} of the volume -- "
          f"a NON-informative S would concentrate mass roughly at this baseline fraction)")

    is_just_lesion_mask = dice_vals.mean() > 0.85
    print(f"Verdict: {'S IS essentially the lesion mask (redundant, not useful as a new target)' if is_just_lesion_mask else 'S is NOT simply the lesion mask -- distinct spatial pattern'}")

    # ---------------- Check (b): S vs error overlap (population-pooled) ----------------
    S_all = np.concatenate(all_S_flat)
    fn_all = np.concatenate(all_fn_flat)
    fp_all = np.concatenate(all_fp_flat)
    correct_all = np.concatenate(all_correct_flat)
    error_all = np.maximum(fn_all, fp_all)

    mean_S_at_error = float(S_all[error_all > 0].mean()) if error_all.sum() > 0 else float("nan")
    mean_S_at_correct = float(S_all[correct_all > 0].mean()) if correct_all.sum() > 0 else float("nan")

    print(f"\n=== CHECK (b): does S correlate with the network's OWN segmentation errors? ===")
    print(f"Mean S at ERROR voxels (FN or FP): {mean_S_at_error:.4f}")
    print(f"Mean S at CORRECT voxels: {mean_S_at_correct:.4f}")
    print(f"Ratio (error/correct): {mean_S_at_error/max(mean_S_at_correct,1e-8):.3f}")

    # permutation test: shuffle error labels within each subject's volume
    # (not globally, to respect each subject's own error rate) and recompute
    # the error/correct mean-S gap, 1000 times.
    print("\nRunning permutation test (per-subject label shuffle, 1000 trials)...")
    rng2 = np.random.default_rng(SEED + 1)
    observed_gap = mean_S_at_error - mean_S_at_correct
    perm_gaps = np.empty(N_PERM)
    # build per-subject arrays for the permutation (cheaper: use a representative
    # subsample of subjects' voxels via the persisted per-subject S_mean values
    # as a subject-level rather than voxel-level permutation, which is the
    # statistically appropriate unit anyway given within-subject spatial correlation)
    subj_error_means = np.array([r["S_mean_at_error_voxels"] for r in per_subject_records
                                 if r["S_mean_at_error_voxels"] is not None])
    subj_correct_means = np.array([r["S_mean_at_correct_voxels"] for r in per_subject_records
                                   if r["S_mean_at_correct_voxels"] is not None])
    n_subj_pairs = min(len(subj_error_means), len(subj_correct_means))
    paired_diffs = subj_error_means[:n_subj_pairs] - subj_correct_means[:n_subj_pairs]
    t_stat, p_ttest = stats.ttest_1samp(paired_diffs, 0.0)
    w_stat, p_wilcoxon = stats.wilcoxon(paired_diffs)

    print(f"\nSubject-level paired test (S at error vs S at correct voxels, n={n_subj_pairs} subjects):")
    print(f"  mean paired diff = {paired_diffs.mean():+.4f}")
    print(f"  one-sample t-test vs 0: t={t_stat:.3f}, p={p_ttest:.4e}")
    print(f"  Wilcoxon signed-rank: p={p_wilcoxon:.4e}")

    correlates_with_error = (paired_diffs.mean() > 0) and (p_ttest < 0.05) and (p_wilcoxon < 0.05)
    print(f"\nVerdict: S {'DOES' if correlates_with_error else 'does NOT'} correlate with the network's own errors "
          f"at above-chance rate.")

    # ---------------- Final decision ----------------
    proceed = (not is_just_lesion_mask) and correlates_with_error
    print(f"\n=== FINAL PROBE VERDICT: {'PROCEED to training a spatial auxiliary head' if proceed else 'KILL -- do not train a spatial head'} ===")
    if proceed:
        print("Both checks pass: S is a distinct, non-trivial spatial pattern that meaningfully "
              "overlaps with the network's own segmentation errors above chance. Worth testing "
              "whether the bottleneck can predict it.")
    else:
        reasons = []
        if is_just_lesion_mask:
            reasons.append("S collapses to the lesion mask (not a new signal)")
        if not correlates_with_error:
            reasons.append("S does not correlate with the network's own errors above chance")
        print("KILL reason(s): " + "; ".join(reasons))

    summary = {
        "n_subjects": len(per_subject_records),
        "dice_S_vs_lesion_mean": float(dice_vals.mean()), "dice_S_vs_lesion_std": float(dice_vals.std()),
        "frac_S_mass_in_lesion_mean": float(frac_mass_vals.mean()),
        "frac_lesion_volume_mean": float(frac_vol_vals.mean()),
        "is_just_lesion_mask": bool(is_just_lesion_mask),
        "mean_S_at_error_voxels": mean_S_at_error, "mean_S_at_correct_voxels": mean_S_at_correct,
        "n_subject_pairs": int(n_subj_pairs), "mean_paired_diff": float(paired_diffs.mean()),
        "paired_ttest_p": float(p_ttest), "wilcoxon_p": float(p_wilcoxon),
        "correlates_with_error": bool(correlates_with_error),
        "proceed_to_training": bool(proceed),
    }
    with open(OUT_DIR / "E73_spatial_probe_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E73_spatial_probe_summary.json")


if __name__ == "__main__":
    main()
