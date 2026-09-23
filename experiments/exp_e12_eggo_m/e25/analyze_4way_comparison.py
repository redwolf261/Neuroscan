import json
from pathlib import Path
import numpy as np
from scipy import stats

RES_DIR = Path(__file__).parent / "deep_sup_4way_comparison_results"
data = json.load(open(RES_DIR / "deep_sup_4way_comparison.json"))

recs = data["subject_records"]
n = len(recs)
conds = ["A", "D4only", "D2only", "Both"]

dice = {c: np.array([r[f"dice_{c}"] for r in recs]) for c in conds}
missed = {c: np.array([r[f"n_missed_{c}"] for r in recs]) for c in conds}
mean_comp_size = np.array([r["mean_component_size"] for r in recs])

print("=== Mean Dice (full val set, whole-volume Dice, threshold 0.5) ===")
for c in conds:
    print(f"  {c}: {dice[c].mean():.4f}")

print("\n=== Q1/Q5: Where does D4-only gain its +0.33pp, and is it a small subset or broad? ===")
delta_d4 = dice["D4only"] - dice["A"]
print(f"Mean delta (D4only - A): {delta_d4.mean():+.4f}")
print(f"  n subjects improved: {(delta_d4 > 1e-6).sum()}, worsened: {(delta_d4 < -1e-6).sum()}, unchanged: {(np.abs(delta_d4) <= 1e-6).sum()}")
print(f"  median delta: {np.median(delta_d4):+.4f}, std: {delta_d4.std():.4f}")
# tercile by mean component size
order = np.argsort(mean_comp_size)
terc = np.array_split(order, 3)
labels = ["Small", "Medium", "Large"]
for lab, idx in zip(labels, terc):
    print(f"  {lab} tercile (n={len(idx)}): mean delta D4only-A = {delta_d4[idx].mean():+.4f}")
# is the gain concentrated in a few subjects?
sorted_delta = np.sort(delta_d4)[::-1]
top10_sum = sorted_delta[:10].sum()
total_sum = delta_d4[delta_d4 > 0].sum()
print(f"  top-10 positive-delta subjects account for {100*top10_sum/total_sum:.1f}% of total positive delta ({len(recs)} subjects total)")

print("\n=== Q2: small-lesion component-quality improvement, D4-only vs D2-only vs Both (vs A) ===")
SIZE_BINS = [0, 50, 150, 400, 1000, float("inf")]
SIZE_LABELS = ["1-50", "50-150", "150-400", "400-1000", ">1000"]

def component_quality_by_size(comps_a, comps_cand):
    """comps_a, comps_cand: lists of per-component dicts (gt_size, detected, component_dice),
    SAME ORDER (component list length differs only if n_gt differs across images -- it's the
    SAME ground truth per subject regardless of model, so gt component structure is identical
    across conditions; using A's component list to define bins works for both)."""
    assert len(comps_a) == len(comps_cand)
    results = {}
    for lo, hi, lab in zip(SIZE_BINS[:-1], SIZE_BINS[1:], SIZE_LABELS):
        idx = [i for i, c in enumerate(comps_a) if lo < c["gt_size"] <= hi] if lo > 0 else [i for i, c in enumerate(comps_a) if c["gt_size"] <= hi]
        detect_a = [comps_a[i]["detected"] for i in idx]
        detect_c = [comps_cand[i]["detected"] for i in idx]
        both_detected = [i for i in idx if comps_a[i]["detected"] and comps_cand[i]["detected"]]
        dice_a = [comps_a[i]["component_dice"] for i in both_detected]
        dice_c = [comps_cand[i]["component_dice"] for i in both_detected]
        results[lab] = {
            "n": len(idx),
            "detect_rate_a": np.mean(detect_a) if idx else None,
            "detect_rate_cand": np.mean(detect_c) if idx else None,
            "n_both_detected": len(both_detected),
            "mean_comp_dice_a": np.mean(dice_a) if dice_a else None,
            "mean_comp_dice_cand": np.mean(dice_c) if dice_c else None,
        }
    return results

comps_a = data["all_components"]["A"]
for cand in ["D4only", "D2only", "Both"]:
    comps_c = data["all_components"][cand]
    res = component_quality_by_size(comps_a, comps_c)
    print(f"\n  --- A vs {cand} ---")
    for lab in SIZE_LABELS:
        r = res[lab]
        if r["mean_comp_dice_a"] is not None:
            delta = r["mean_comp_dice_cand"] - r["mean_comp_dice_a"]
            print(f"    {lab:>10} (n={r['n']:3d}, both-detected={r['n_both_detected']:3d}): "
                  f"detect A={r['detect_rate_a']*100:.1f}% {cand}={r['detect_rate_cand']*100:.1f}% | "
                  f"comp-dice A={r['mean_comp_dice_a']:.3f} {cand}={r['mean_comp_dice_cand']:.3f} (delta {delta:+.3f})")

print("\n=== Q4: D4-only vs Both -- paired subject-level comparison (is D2 branch competing?) ===")
delta_d4_vs_both = dice["D4only"] - dice["Both"]
print(f"Mean delta (D4only - Both): {delta_d4_vs_both.mean():+.5f}")
print(f"  n improved: {(delta_d4_vs_both > 1e-6).sum()}, worsened: {(delta_d4_vs_both < -1e-6).sum()}, unchanged: {(np.abs(delta_d4_vs_both) <= 1e-6).sum()}")
w_stat, w_p = stats.wilcoxon(dice["D4only"], dice["Both"])
t_stat, t_p = stats.ttest_rel(dice["D4only"], dice["Both"])
print(f"  paired t-test: t={t_stat:.3f}, p={t_p:.4f}")
print(f"  wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.4f}")

print("\n=== Q6: are D4only, D2only, Both, effectively distinct or same-within-noise? Pairwise stats ===")
pairs = [("A", "D4only"), ("A", "D2only"), ("A", "Both"), ("D4only", "D2only"), ("D4only", "Both"), ("D2only", "Both")]
for a, b in pairs:
    d = dice[a] - dice[b]
    t_stat, t_p = stats.ttest_rel(dice[a], dice[b])
    print(f"  {a} vs {b}: mean_whole_vol_dice_delta={d.mean():+.5f}, paired-t p={t_p:.4f}")

print("\n=== Missed component counts (detection completeness) ===")
for c in conds:
    print(f"  {c}: total missed components = {missed[c].sum()}, mean per subject = {missed[c].mean():.3f}")

print("\n=== Boundary error rate (from main script) ===")
for c, v in data["boundary_error_rate"].items():
    print(f"  {c}: {v:.4f}")
