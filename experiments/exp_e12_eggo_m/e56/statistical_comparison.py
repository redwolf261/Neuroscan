"""
Phase E56, Step 2 (continued): paired statistical comparison of every
post-pivot mechanism against baseline A, on PER-SUBJECT Dice (not
pooled), matching E25's own established convention (paired t-test +
Wilcoxon signed-rank, n=125).
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).parent

with open(BASE / "E56_per_subject_rescoring.json") as f:
    results = json.load(f)

baseline_subjects = results["baseline_A"]["per_subject_dice"]
baseline_ids = sorted(baseline_subjects.keys())
baseline_arr = np.array([baseline_subjects[sid] for sid in baseline_ids])

print("=" * 100)
print(f"{'Mechanism':<16} {'pooled':>8} {'per-subj mean':>14} {'delta vs base':>14} {'paired-t p':>11} {'Wilcoxon p':>11} {'sig?':>6}")
print("=" * 100)

rows = []
for name, data in results.items():
    if name == "baseline_A":
        continue
    subj = data["per_subject_dice"]
    common_ids = [sid for sid in baseline_ids if sid in subj]
    if len(common_ids) != len(baseline_ids):
        print(f"WARNING: {name} has {len(common_ids)}/{len(baseline_ids)} subjects in common with baseline -- subject ID mismatch?")
    mech_arr = np.array([subj[sid] for sid in common_ids])
    base_arr_matched = np.array([baseline_subjects[sid] for sid in common_ids])

    delta = mech_arr - base_arr_matched
    t_stat, t_p = stats.ttest_rel(mech_arr, base_arr_matched)
    w_stat, w_p = stats.wilcoxon(mech_arr, base_arr_matched)

    sig = "YES" if (t_p < 0.05 and w_p < 0.05) else ("t-only" if t_p < 0.05 else ("W-only" if w_p < 0.05 else "no"))

    rows.append({
        "name": name,
        "pooled_dice": data["reported_pooled_dice"],
        "per_subject_mean": float(mech_arr.mean()),
        "delta_mean": float(delta.mean()),
        "t_stat": float(t_stat), "t_p": float(t_p),
        "w_stat": float(w_stat), "w_p": float(w_p),
        "significant": sig,
        "n_subjects": len(common_ids),
    })

    print(f"{name:<16} {data['reported_pooled_dice']:>8.4f} {mech_arr.mean():>14.4f} "
          f"{delta.mean():>+14.4f} {t_p:>11.4f} {w_p:>11.4f} {sig:>6}")

print("=" * 100)

# ---- 3-seed mean/CI for multi-seed mechanisms, per-subject basis ----
print("\n3-seed per-subject means (mechanisms with 3 seeds):")
for prefix in ["e49_CCABA", "e50_IECG", "e51_CCAG"]:
    seed_means = [r["per_subject_mean"] for r in rows if r["name"].startswith(prefix)]
    if len(seed_means) == 3:
        arr = np.array(seed_means)
        se = arr.std(ddof=1) / np.sqrt(3)
        ci = stats.t.interval(0.95, df=2, loc=arr.mean(), scale=se)
        print(f"  {prefix}: seeds={[f'{v:.4f}' for v in seed_means]} mean={arr.mean():.4f} "
              f"std={arr.std(ddof=1):.4f} 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

# ---- Baseline's own historical 4-seed variance (from PHASE_RESEARCH_ARC_MASTER_REPORT.md,
# pooled Dice only -- per-subject numbers for seeds 1-3 not yet computed in this script,
# but the POOLED 4-seed spread itself is directly relevant context) ----
baseline_4seed_pooled = np.array([0.9063020758330822, 0.9064021334052086, 0.9027954526245594, 0.902003463357687])
print(f"\nBaseline A's own historical 4-seed POOLED Dice (seed0-3, from e13_pilot_calibrated_seed{{1,2,3}} "
      f"+ e24 seed0, previously never aggregated):")
print(f"  values: {baseline_4seed_pooled}")
print(f"  mean: {baseline_4seed_pooled.mean():.4f}  std: {baseline_4seed_pooled.std(ddof=1):.4f}")
se = baseline_4seed_pooled.std(ddof=1) / np.sqrt(4)
ci = stats.t.interval(0.95, df=3, loc=baseline_4seed_pooled.mean(), scale=se)
print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"  CORRECTED +1pp target (mean + 0.01): {baseline_4seed_pooled.mean() + 0.01:.4f}  "
      f"(vs. the previously-used single-seed target of 0.9163)")

with open(BASE / "E56_statistical_comparison.json", "w") as f:
    json.dump({
        "per_mechanism_comparisons": rows,
        "baseline_4seed_pooled_historical": {
            "values": baseline_4seed_pooled.tolist(),
            "mean": float(baseline_4seed_pooled.mean()),
            "std": float(baseline_4seed_pooled.std(ddof=1)),
            "ci_95": [float(ci[0]), float(ci[1])],
            "corrected_target": float(baseline_4seed_pooled.mean() + 0.01),
        },
    }, f, indent=2)
print(f"\nSaved to {BASE / 'E56_statistical_comparison.json'}")
