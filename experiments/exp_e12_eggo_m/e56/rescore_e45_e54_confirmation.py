"""
Phase E56 (continued): 3-seed confirmation re-scoring for E45 (D4+D8)
and E54 (A96) -- the two mechanisms flagged by PHASE_EVIDENCE_MAP.md as
the only doubly-significant (paired-t AND Wilcoxon) per-subject results
in the entire post-pivot arc, on a single seed each. This scores the
NEW seeds 1/2 (seed 0 already scored in E56_per_subject_rescoring.json,
reused unchanged, not recomputed) and runs the full statistical battery
specified by the user: per-seed mean/SD/95% CI, paired t-test +
Wilcoxon per seed AND pooled across all 3 seeds, and the distribution
of patient-level delta-Dice.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy import stats
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_v4 import UNet3D_v4  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

BASE = Path(__file__).parent

NEW_CHECKPOINTS = [
    ("e45_D4D8_s1", "e45/runs/D4_D8_seed1/checkpoints/best.pth", UNet3D_v4, (64, 64, 64)),
    ("e45_D4D8_s2", "e45/runs/D4_D8_seed2/checkpoints/best.pth", UNet3D_v4, (64, 64, 64)),
    ("e54_A96_s1",  "e54/runs/A96_seed1/checkpoints/best.pth",   UNet3D_v3, (96, 96, 96)),
    ("e54_A96_s2",  "e54/runs/A96_seed2/checkpoints/best.pth",   UNet3D_v3, (96, 96, 96)),
]


def fractional_occupancy_64(seg_binary_native):
    t = torch.from_numpy(seg_binary_native).unsqueeze(0).unsqueeze(0)
    frac = F.interpolate(t, size=(64, 64, 64), mode="area").squeeze().numpy()
    return frac


def dice_per_subject(pred_bin, gt_bin, eps=1e-6):
    inter = (pred_bin * gt_bin).sum()
    return float((2 * inter + eps) / (pred_bin.sum() + gt_bin.sum() + eps))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}
    for name, ckpt_rel, model_cls, target_shape in NEW_CHECKPOINTS:
        ckpt_path = project_root / "experiments" / "exp_e12_eggo_m" / ckpt_rel
        if not ckpt_path.exists():
            print(f"[{name}] SKIP -- checkpoint not found: {ckpt_path}")
            continue

        print(f"\n[{name}] Loading {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = model_cls(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        val_ds = BraTSDataset(
            root_dir=str(project_root / "Dataset" / "Training"),
            split="val", val_split=0.1, target_shape=target_shape,
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

    with open(BASE / "E45_E54_seed12_rescoring.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved seed1/2 results to E45_E54_seed12_rescoring.json")

    # ================= Combine with seed 0 (already scored in E56) =================
    with open(BASE / "E56_per_subject_rescoring.json") as f:
        e56_results = json.load(f)
    baseline_subjects = e56_results["baseline_A"]["per_subject_dice"]
    baseline_ids = sorted(baseline_subjects.keys())
    baseline_arr = np.array([baseline_subjects[sid] for sid in baseline_ids])

    mechanisms = {
        "E45_D4D8": {
            "seed0": e56_results["e45_D4D8"]["per_subject_dice"],
            "seed1": results.get("e45_D4D8_s1", {}).get("per_subject_dice"),
            "seed2": results.get("e45_D4D8_s2", {}).get("per_subject_dice"),
            "pooled": {
                0: e56_results["e45_D4D8"]["reported_pooled_dice"],
                1: results.get("e45_D4D8_s1", {}).get("reported_pooled_dice"),
                2: results.get("e45_D4D8_s2", {}).get("reported_pooled_dice"),
            },
        },
        "E54_A96": {
            "seed0": e56_results["e54_A96"]["per_subject_dice"],
            "seed1": results.get("e54_A96_s1", {}).get("per_subject_dice"),
            "seed2": results.get("e54_A96_s2", {}).get("per_subject_dice"),
            "pooled": {
                0: e56_results["e54_A96"]["reported_pooled_dice"],
                1: results.get("e54_A96_s1", {}).get("reported_pooled_dice"),
                2: results.get("e54_A96_s2", {}).get("reported_pooled_dice"),
            },
        },
    }

    CORRECTED_TARGET = 0.8942  # baseline per-subject mean (0.8842) + 1pp, per PHASE_E56/EVIDENCE_MAP
    POOLED_1PP_TARGET = 0.9163  # original stronger criterion, retained per explicit instruction

    final_summary = {}
    for mech_name, mech_data in mechanisms.items():
        if mech_data["seed1"] is None or mech_data["seed2"] is None:
            print(f"\n[{mech_name}] INCOMPLETE -- seed1/seed2 missing, skipping full analysis")
            continue

        print(f"\n{'='*80}\n=== {mech_name}: 3-seed confirmation ===\n{'='*80}")

        seed_means = []
        seed_stats = {}
        for seed_key in ["seed0", "seed1", "seed2"]:
            subj_dice = mech_data[seed_key]
            common_ids = [sid for sid in baseline_ids if sid in subj_dice]
            mech_arr = np.array([subj_dice[sid] for sid in common_ids])
            base_arr = np.array([baseline_subjects[sid] for sid in common_ids])
            delta = mech_arr - base_arr

            t_stat, t_p = stats.ttest_rel(mech_arr, base_arr)
            w_stat, w_p = stats.wilcoxon(mech_arr, base_arr)

            seed_num = int(seed_key[-1])
            pooled_val = mech_data["pooled"][seed_num]
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

        # Pooled-across-all-3-seeds paired test (all subjects x all 3 seeds, n=375, paired to matched baseline repeated 3x)
        all_mech_vals, all_base_vals = [], []
        for seed_key in ["seed0", "seed1", "seed2"]:
            subj_dice = mech_data[seed_key]
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

        print(f"\n  PRE-DECLARED CRITERIA:")
        print(f"  Corrected target (3-seed mean per-subject Dice >= {CORRECTED_TARGET}): "
              f"{mean_3seed:.4f} >= {CORRECTED_TARGET} = {confirmed_corrected}")
        print(f"  Original target (3-seed mean pooled Dice >= {POOLED_1PP_TARGET}): "
              f"{pooled_dice_3seed_mean:.4f} >= {POOLED_1PP_TARGET} = {confirmed_original}")

        final_summary[mech_name] = {
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
            "confirmed_original_target": bool(confirmed_original),
        }

    with open(BASE / "E45_E54_3seed_confirmation_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)
    print(f"\n\nSaved final summary to E45_E54_3seed_confirmation_summary.json")


if __name__ == "__main__":
    main()
