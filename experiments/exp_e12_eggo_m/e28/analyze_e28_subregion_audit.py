"""
Phase E28 analysis: does WT component subregion composition relate to A's
detection/quality outcomes? No training. Consumes
E28_subregion_component_table.json.

Strong gate (user's own framing): if subregion composition has no
meaningful relationship with delta-Dice/missed-status/component-quality,
kill the idea. Test this directly and report honestly either way.
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

BASE = Path(__file__).parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

with open(BASE / "E28_subregion_component_table.json") as f:
    comp = json.load(f)
n = len(comp)
print(f"Loaded {n} WT components")

gt_size = np.array([r["gt_size"] for r in comp])
pct_ncr = np.array([r["pct_ncr"] for r in comp])
pct_ed = np.array([r["pct_ed"] for r in comp])
pct_et = np.array([r["pct_et"] for r in comp])
pct_other = np.array([r["pct_other"] for r in comp])
dominant = np.array([r["dominant_subregion"] for r in comp])
detected = np.array([r["detected_A"] for r in comp])
dice = np.array([r["dice_A"] for r in comp])
coverage = np.array([r["coverage_A"] for r in comp])

# ============================================================
# 1. Overall subregion composition summary
# ============================================================
print("\n=== Overall composition (population mean %, across all components) ===")
print(f"  mean %NCR={pct_ncr.mean()*100:.1f}  %ED={pct_ed.mean()*100:.1f}  %ET={pct_et.mean()*100:.1f}  %other(resample-edge)={pct_other.mean()*100:.1f}")

print("\n=== Dominant-subregion breakdown ===")
for lab in ("NCR", "ED", "ET", "other"):
    mask = dominant == lab
    print(f"  {lab}: n={mask.sum()} ({100*mask.sum()/n:.1f}%)")

# ============================================================
# 2. Detection rate by dominant subregion
# ============================================================
print("\n=== Detection rate by dominant subregion ===")
detect_by_dominant = {}
for lab in ("NCR", "ED", "ET", "other"):
    mask = dominant == lab
    if mask.sum() == 0:
        continue
    rate = detected[mask].mean()
    print(f"  {lab}: n={mask.sum()}  detect_rate={rate*100:.1f}%")
    detect_by_dominant[lab] = {"n": int(mask.sum()), "detect_rate": float(rate)}

# chi-square test: is detection independent of dominant subregion?
labels_present = [lab for lab in ("NCR", "ED", "ET") if (dominant == lab).sum() >= 5]
if len(labels_present) >= 2:
    contingency = np.array([
        [int((detected[dominant == lab]).sum()), int((~detected[dominant == lab]).sum())]
        for lab in labels_present
    ])
    chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)
    print(f"\nChi-square test (detection x dominant subregion, {labels_present}): chi2={chi2:.3f}, p={p_chi2:.4f}, dof={dof}")
else:
    chi2, p_chi2 = None, None
    print("\nInsufficient per-category n for chi-square test")

# ============================================================
# 3. Detection rate by dominant subregion, WITHIN size bins (control for size)
# ============================================================
print("\n=== Detection rate by dominant subregion, stratified by size bin ===")
size_bins = [(0, 50, "1-50"), (50, 150, "50-150"), (150, 400, "150-400"), (400, 1000, "400-1000"), (1000, np.inf, ">1000")]
stratified_results = {}
for lo, hi, lab_bin in size_bins:
    mask_bin = (gt_size > lo) & (gt_size <= hi) if lo > 0 else (gt_size <= hi)
    print(f"  --- size {lab_bin} (n={mask_bin.sum()}) ---")
    bin_result = {}
    for lab in ("NCR", "ED", "ET"):
        mask = mask_bin & (dominant == lab)
        if mask.sum() == 0:
            continue
        rate = detected[mask].mean()
        print(f"    {lab}: n={mask.sum()}  detect_rate={rate*100:.1f}%  mean_dice={dice[mask].mean():.3f}")
        bin_result[lab] = {"n": int(mask.sum()), "detect_rate": float(rate), "mean_dice": float(dice[mask].mean())}
    stratified_results[lab_bin] = bin_result

# ============================================================
# 4. Continuous %ET / %ED / %NCR vs detection / dice / coverage
# ============================================================
print("\n=== Continuous subregion fraction vs outcome (point-biserial / Spearman) ===")
continuous_results = {}
for pct_name, pct_arr in (("pct_ncr", pct_ncr), ("pct_ed", pct_ed), ("pct_et", pct_et)):
    r_detect, p_detect = stats.pointbiserialr(detected.astype(int), pct_arr)
    r_dice, p_dice = stats.spearmanr(pct_arr, dice)
    r_cov, p_cov = stats.spearmanr(pct_arr, coverage)
    print(f"  {pct_name}: vs detected (point-biserial) r={r_detect:+.3f} p={p_detect:.4f} | vs dice (spearman) rho={r_dice:+.3f} p={p_dice:.4f} | vs coverage rho={r_cov:+.3f} p={p_cov:.4f}")
    continuous_results[pct_name] = {
        "vs_detected": {"r": r_detect, "p": p_detect},
        "vs_dice": {"rho": r_dice, "p": p_dice},
        "vs_coverage": {"rho": r_cov, "p": p_cov},
    }

# ============================================================
# 5. Does subregion composition predict outcome AFTER controlling for size?
#    (nonlinear size controls, per the E26/E27 lesson)
# ============================================================
print("\n=== Partial correlation: %ET vs dice_A, controlling for nonlinear size ===")
def partial_corr(x, y, controls):
    X = np.column_stack([np.ones(len(x))] + [c for c in controls])
    def resid(v):
        coefv, _, _, _ = np.linalg.lstsq(X, v, rcond=None)
        return v - X @ coefv
    return stats.pearsonr(resid(x), resid(y))

nonlinear_size_controls = [gt_size, gt_size ** (-1/3), np.log(gt_size + 1)]
partial_results = {}
for pct_name, pct_arr in (("pct_ncr", pct_ncr), ("pct_ed", pct_ed), ("pct_et", pct_et)):
    r_p, p_p = partial_corr(pct_arr, dice, nonlinear_size_controls)
    print(f"  {pct_name} vs dice_A, controlling nonlinear size: partial r={r_p:+.3f} p={p_p:.4f}")
    partial_results[pct_name] = {"partial_r": r_p, "p": p_p}

# joint model: does composition add anything beyond size alone?
X_size_only = np.column_stack([np.ones(n)] + nonlinear_size_controls)
coef, _, _, _ = np.linalg.lstsq(X_size_only, dice, rcond=None)
pred = X_size_only @ coef
r2_size_only = 1 - np.sum((dice - pred) ** 2) / np.sum((dice - dice.mean()) ** 2)

X_size_plus_comp = np.column_stack([np.ones(n)] + nonlinear_size_controls + [pct_ncr, pct_ed, pct_et])
coef2, _, _, _ = np.linalg.lstsq(X_size_plus_comp, dice, rcond=None)
pred2 = X_size_plus_comp @ coef2
r2_size_plus_comp = 1 - np.sum((dice - pred2) ** 2) / np.sum((dice - dice.mean()) ** 2)

print(f"\nR^2(dice_A ~ nonlinear size only) = {r2_size_only:.4f}")
print(f"R^2(dice_A ~ nonlinear size + %NCR + %ED + %ET) = {r2_size_plus_comp:.4f}")
print(f"Incremental R^2 from adding subregion composition = {r2_size_plus_comp - r2_size_only:.4f}")

# ============================================================
# 6. Subject-clustered bootstrap on the key detect-rate gap (ED vs ET dominant)
# ============================================================
print("\n=== Subject-clustered bootstrap: detect-rate gap (ED-dominant vs ET-dominant) ===")
subj_to_indices = defaultdict(list)
for i, r in enumerate(comp):
    subj_to_indices[r["subject_idx"]].append(i)
subj_list = list(subj_to_indices.keys())

rng = np.random.RandomState(0)
n_boot = 2000
boot_gaps = []
for _ in range(n_boot):
    sampled_subjs = rng.choice(subj_list, size=len(subj_list), replace=True)
    idxs = np.concatenate([subj_to_indices[s] for s in sampled_subjs]) if sampled_subjs.size else np.array([], dtype=int)
    if len(idxs) == 0:
        continue
    dom_b = dominant[idxs]
    det_b = detected[idxs]
    ed_mask = dom_b == "ED"
    et_mask = dom_b == "ET"
    if ed_mask.sum() == 0 or et_mask.sum() == 0:
        continue
    gap = det_b[ed_mask].mean() - det_b[et_mask].mean()
    boot_gaps.append(gap)
boot_gaps = np.array(boot_gaps)
if len(boot_gaps) > 100:
    ci_lo, ci_hi = np.percentile(boot_gaps, [2.5, 97.5])
    print(f"  ED-detect-rate minus ET-detect-rate: bootstrap mean={boot_gaps.mean():+.3f}, 95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}] (n_boot={len(boot_gaps)})")
    bootstrap_result = {"mean_gap": float(boot_gaps.mean()), "ci_low": float(ci_lo), "ci_high": float(ci_hi), "n_boot": int(len(boot_gaps))}
else:
    print("  insufficient bootstrap samples with both ED and ET dominant components present")
    bootstrap_result = {"status": "insufficient_samples"}

# ============================================================
# Save results
# ============================================================
results = {
    "n_components": n,
    "composition_summary": {
        "mean_pct_ncr": float(pct_ncr.mean()), "mean_pct_ed": float(pct_ed.mean()),
        "mean_pct_et": float(pct_et.mean()), "mean_pct_other": float(pct_other.mean()),
    },
    "dominant_subregion_counts": {lab: int((dominant == lab).sum()) for lab in ("NCR", "ED", "ET", "other")},
    "detect_rate_by_dominant": detect_by_dominant,
    "chi_square_detection_vs_dominant": {"chi2": chi2, "p": p_chi2} if chi2 is not None else None,
    "detect_rate_by_dominant_stratified_by_size": stratified_results,
    "continuous_correlations": continuous_results,
    "partial_correlations_controlling_nonlinear_size": partial_results,
    "r2_size_only": float(r2_size_only),
    "r2_size_plus_composition": float(r2_size_plus_comp),
    "incremental_r2_from_composition": float(r2_size_plus_comp - r2_size_only),
    "subject_clustered_bootstrap_ED_minus_ET_detect_gap": bootstrap_result,
}
with open(BASE / "E28_subregion_audit_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved results to {BASE / 'E28_subregion_audit_results.json'}")

# ============================================================
# Figures
# ============================================================
if HAVE_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labs = ["NCR", "ED", "ET"]
    rates = [detect_by_dominant.get(l, {}).get("detect_rate", 0) * 100 for l in labs]
    ns = [detect_by_dominant.get(l, {}).get("n", 0) for l in labs]
    axes[0].bar(labs, rates, color=["#c44e52", "#55a868", "#4c72b0"])
    for i, (r, nn) in enumerate(zip(rates, ns)):
        axes[0].text(i, r + 1, f"n={nn}", ha="center")
    axes[0].set_ylabel("A detection rate (%)")
    axes[0].set_title("Figure 1: Detection rate by dominant subregion")
    axes[0].set_ylim(0, 105)

    axes[1].scatter(pct_et, dice, s=8, alpha=0.4, color="#4c72b0")
    axes[1].set_xlabel("%ET within component")
    axes[1].set_ylabel("Component Dice (A)")
    axes[1].set_title("Figure 2: %ET vs component Dice")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "e28_figures.png", dpi=120)
    plt.close()
    print("Saved figures")
