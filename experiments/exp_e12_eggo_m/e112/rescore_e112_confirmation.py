"""
Phase E112: 3-seed per-subject confirmation re-scoring, following the EXACT
same methodology as e56/rescore_e45_e54_confirmation.py (per-subject Dice,
paired t-test + Wilcoxon per seed AND pooled across all 3 seeds, 3-seed
mean/std/95% CI), so this result is directly comparable to E45's and E54's
own confirmed numbers -- not a new, ad-hoc metric.

Reuses the SAME stored baseline_A per-subject Dice values from
E56_per_subject_rescoring.json (never recomputed), matching E45/E54's own
precedent of treating that file as the single source of truth for the
baseline comparison population.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v4 import UNet3D_v4  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

BASE = Path(__file__).parent
E56_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e56"

CHECKPOINTS = [
    ("e112_s0", "runs/D4D8_A96_seed0/checkpoints/best.pth"),
    ("e112_s1", "runs/D4D8_A96_seed1/checkpoints/best.pth"),
    ("e112_s2", "runs/D4D8_A96_seed2/checkpoints/best.pth"),
]
TARGET_SHAPE = (96, 96, 96)

CORRECTED_TARGET = 0.8942  # baseline per-subject mean (0.8842) + 1pp, per E56/PHASE_EVIDENCE_MAP
POOLED_1PP_TARGET = 0.9163
E45_3SEED_MEAN = 0.8939  # for direct secondary comparison, per this phase's own pre-declared criterion
E54_3SEED_MEAN = 0.8944


def dice_per_subject(pred_bin, gt_bin, eps=1e-6):
    inter = (pred_bin * gt_bin).sum()
    return float((2 * inter + eps) / (pred_bin.sum() + gt_bin.sum() + eps))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}
    for name, ckpt_rel in CHECKPOINTS:
        ckpt_path = BASE / ckpt_rel
        if not ckpt_path.exists():
            print(f"[{name}] SKIP -- checkpoint not found: {ckpt_path}")
            continue

        print(f"\n[{name}] Loading {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = UNet3D_v4(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        val_ds = BraTSDataset(
            root_dir=str(project_root / "Dataset" / "Training"),
            split="val", val_split=0.1, target_shape=TARGET_SHAPE,
        )

        per_subject_dice = {}
        with torch.no_grad():
            for idx in tqdm(range(len(val_ds)), desc=f"[{name}] scoring"):
                image, mask, subject_id = val_ds[idx]
                image = image.unsqueeze(0).to(device)
                mask_np = mask[0].numpy()

                out = model(image)
                probs = out["probs"] if isinstance(out, dict) else out
                pred_bin = (probs[0, 0].cpu().numpy() >= 0.5).astype(np.float32)
                gt_bin = (mask_np >= 0.5).astype(np.float32)

                per_subject_dice[subject_id] = dice_per_subject(pred_bin, gt_bin)

        dice_values = np.array(list(per_subject_dice.values()))
        mean_dice = float(dice_values.mean())
        std_dice = float(dice_values.std(ddof=1))
        pooled_dice_reported = float(ckpt.get("best_val_dice", float("nan")))

        print(f"[{name}] n_subjects={len(dice_values)} mean_per_subject_dice={mean_dice:.4f} "
              f"std={std_dice:.4f} (reported pooled_dice={pooled_dice_reported:.4f})")

        results[name] = {
            "checkpoint": ckpt_rel, "reported_pooled_dice": pooled_dice_reported,
            "mean_per_subject_dice": mean_dice, "std_per_subject_dice": std_dice,
            "n_subjects": len(dice_values), "per_subject_dice": per_subject_dice,
        }

        del model
        torch.cuda.empty_cache()

    with open(BASE / "E112_seeds_rescoring.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved per-seed results to E112_seeds_rescoring.json")

    if len(results) < 3:
        print(f"\n[STOP] Only {len(results)}/3 seeds have checkpoints -- "
              f"cannot run the 3-seed confirmation statistics yet.")
        return

    # ================= Combine with baseline_A (reused, never recomputed) =================
    with open(E56_DIR / "E56_per_subject_rescoring.json") as f:
        e56_results = json.load(f)
    baseline_subjects = e56_results["baseline_A"]["per_subject_dice"]
    baseline_ids = sorted(baseline_subjects.keys())

    seed_data = {
        "seed0": results["e112_s0"]["per_subject_dice"],
        "seed1": results["e112_s1"]["per_subject_dice"],
        "seed2": results["e112_s2"]["per_subject_dice"],
    }
    pooled_by_seed = {
        0: results["e112_s0"]["reported_pooled_dice"],
        1: results["e112_s1"]["reported_pooled_dice"],
        2: results["e112_s2"]["reported_pooled_dice"],
    }

    print(f"\n{'='*80}\n=== E112 (D4+D8 @ A96 combined): 3-seed confirmation ===\n{'='*80}")

    seed_means = []
    seed_stats = {}
    for seed_key in ["seed0", "seed1", "seed2"]:
        subj_dice = seed_data[seed_key]
        common_ids = [sid for sid in baseline_ids if sid in subj_dice]
        mech_arr = np.array([subj_dice[sid] for sid in common_ids])
        base_arr = np.array([baseline_subjects[sid] for sid in common_ids])
        delta = mech_arr - base_arr

        t_stat, t_p = stats.ttest_rel(mech_arr, base_arr)
        w_stat, w_p = stats.wilcoxon(mech_arr, base_arr)

        seed_num = int(seed_key[-1])
        pooled_val = pooled_by_seed[seed_num]
        seed_means.append(mech_arr.mean())
        seed_stats[seed_key] = {
            "pooled_dice": pooled_val,
            "per_subject_mean": float(mech_arr.mean()), "per_subject_std": float(mech_arr.std(ddof=1)),
            "delta_mean": float(delta.mean()), "delta_std": float(delta.std(ddof=1)),
            "delta_median": float(np.median(delta)),
            "paired_t_p": float(t_p), "wilcoxon_p": float(w_p),
            "n_subjects": len(common_ids),
        }
        print(f"  {seed_key}: pooled={pooled_val:.4f} per_subj_mean={mech_arr.mean():.4f} "
              f"delta_mean={delta.mean():+.4f} paired_t_p={t_p:.4f} wilcoxon_p={w_p:.4f}")

    seed_means_arr = np.array(seed_means)
    mean_3seed = float(seed_means_arr.mean())
    std_3seed = float(seed_means_arr.std(ddof=1))
    se_3seed = std_3seed / np.sqrt(3)
    ci_3seed = stats.t.interval(0.95, df=2, loc=mean_3seed, scale=se_3seed)

    print(f"\n  3-seed mean per-subject Dice: {mean_3seed:.4f}  std: {std_3seed:.4f}  "
          f"95% CI: [{ci_3seed[0]:.4f}, {ci_3seed[1]:.4f}]")

    all_mech_vals, all_base_vals = [], []
    for seed_key in ["seed0", "seed1", "seed2"]:
        subj_dice = seed_data[seed_key]
        common_ids = [sid for sid in baseline_ids if sid in subj_dice]
        all_mech_vals.extend([subj_dice[sid] for sid in common_ids])
        all_base_vals.extend([baseline_subjects[sid] for sid in common_ids])
    all_mech_arr = np.array(all_mech_vals)
    all_base_arr = np.array(all_base_vals)
    all_delta = all_mech_arr - all_base_arr
    t_pooled, p_t_pooled = stats.ttest_rel(all_mech_arr, all_base_arr)
    w_pooled, p_w_pooled = stats.wilcoxon(all_mech_arr, all_base_arr)
    print(f"  Pooled-across-seeds paired test (n={len(all_mech_arr)}): "
          f"paired-t p={p_t_pooled:.4e}  Wilcoxon p={p_w_pooled:.4e}")
    print(f"  Patient-level delta-Dice distribution: mean={all_delta.mean():+.4f} "
          f"median={np.median(all_delta):+.4f} std={all_delta.std():.4f} "
          f"[{np.percentile(all_delta,5):.4f}, {np.percentile(all_delta,95):.4f}] (5th-95th pctile)")

    pooled_dice_3seed_mean = float(np.mean([seed_stats[k]["pooled_dice"] for k in seed_stats]))

    confirmed_corrected = (mean_3seed >= CORRECTED_TARGET)
    confirmed_original = (pooled_dice_3seed_mean >= POOLED_1PP_TARGET)
    # PRE-DECLARED MEANINGFUL-MARGIN CHECK (per this phase's own docstring):
    # a margin smaller than 1/10 of the between-seed std does NOT count as
    # a real pass, matching E54's own precedent for treating a bare
    # threshold-crossing as inconclusive rather than confirmed.
    margin = mean_3seed - CORRECTED_TARGET
    meaningful_margin = abs(margin) >= 0.1 * std_3seed
    beats_e45 = mean_3seed > E45_3SEED_MEAN
    beats_e54 = mean_3seed > E54_3SEED_MEAN
    beats_better_parent = mean_3seed > max(E45_3SEED_MEAN, E54_3SEED_MEAN)

    print(f"\n  PRE-DECLARED CRITERIA:")
    print(f"  Corrected target (3-seed mean per-subject Dice >= {CORRECTED_TARGET}): "
          f"{mean_3seed:.4f} >= {CORRECTED_TARGET} = {confirmed_corrected}")
    print(f"  Margin = {margin:+.4f}, 1/10 between-seed std = {0.1*std_3seed:.4f} -- "
          f"MEANINGFUL MARGIN: {meaningful_margin}")
    print(f"  Original target (3-seed mean pooled Dice >= {POOLED_1PP_TARGET}): "
          f"{pooled_dice_3seed_mean:.4f} >= {POOLED_1PP_TARGET} = {confirmed_original}")
    print(f"  Beats E45 alone (0.8939)?  {mean_3seed:.4f} > 0.8939 = {beats_e45}")
    print(f"  Beats E54 alone (0.8944)?  {mean_3seed:.4f} > 0.8944 = {beats_e54}")
    print(f"  SECONDARY CRITERION -- beats the BETTER of its two parent mechanisms: {beats_better_parent}")

    if confirmed_corrected and meaningful_margin and beats_better_parent:
        final_verdict = "GO -- combination clears the bar with a meaningful margin and beats both parents"
    elif confirmed_corrected and not meaningful_margin:
        final_verdict = "INCONCLUSIVE -- crosses the threshold but margin is not meaningful (same failure mode as E45/E54 individually)"
    elif not confirmed_corrected:
        final_verdict = "NOT MET -- 3-seed mean per-subject Dice below corrected target"
    else:
        final_verdict = "MIXED -- crosses corrected target with meaningful margin but does not beat the better individual parent"

    print(f"\n  === FINAL VERDICT: {final_verdict} ===")

    final_summary = {
        "per_seed": seed_stats,
        "three_seed_mean_per_subject_dice": mean_3seed,
        "three_seed_std_per_subject_dice": std_3seed,
        "three_seed_95ci_per_subject_dice": list(ci_3seed),
        "three_seed_mean_pooled_dice": pooled_dice_3seed_mean,
        "pooled_across_seeds_paired_t_p": float(p_t_pooled),
        "pooled_across_seeds_wilcoxon_p": float(p_w_pooled),
        "patient_delta_dice_distribution": {
            "mean": float(all_delta.mean()), "median": float(np.median(all_delta)),
            "std": float(all_delta.std()),
            "p5": float(np.percentile(all_delta, 5)), "p95": float(np.percentile(all_delta, 95)),
        },
        "confirmed_corrected_target": bool(confirmed_corrected),
        "meaningful_margin": bool(meaningful_margin),
        "confirmed_original_target": bool(confirmed_original),
        "beats_e45_alone": bool(beats_e45),
        "beats_e54_alone": bool(beats_e54),
        "beats_better_parent": bool(beats_better_parent),
        "final_verdict_CONVENIENCE_LABEL_ONLY": final_verdict,
    }
    with open(BASE / "E112_3seed_confirmation_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)
    print(f"\n\nSaved final summary to E112_3seed_confirmation_summary.json")


if __name__ == "__main__":
    main()
