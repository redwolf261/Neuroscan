"""
Phase E27 analysis: component-specific decoder-scale preference.
Consumes E27_component_table.json (best-checkpoint, 4 conditions) and
E27_trajectory_table.json (periodic checkpoints, for early predictability).
No training. Produces E27_scale_preference_results.json and the numbers
needed for E27_scale_preference_report.md, plus figures/.
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

with open(BASE / "E27_component_table.json") as f:
    comp = json.load(f)
print(f"Loaded {len(comp)} GT components (best-checkpoint table)")

EARLY_CUTOFF_EPOCH = 10  # ~33% of 30 epochs; documented BEFORE looking at final-target correlations (Section 10/11)

# ============================================================
# Section 4-5: per-component metrics + gains + S_c
# ============================================================
records = []
for r in comp:
    size = r["gt_size"]
    row = {
        "subject_id": r["subject_id"], "subject_idx": r["subject_idx"], "component_id": r["component_id"],
        "gt_size": size,
    }
    for name in ("A", "D4", "D2", "Both"):
        row[f"detected_{name}"] = r[f"detected_{name}"]
        row[f"dice_{name}"] = r[f"dice_{name}"]
        row[f"iou_{name}"] = r[f"iou_{name}"]
        row[f"coverage_{name}"] = r[f"coverage_{name}"]
        row[f"precision_{name}"] = r[f"precision_{name}"]  # may be None

    for metric in ("dice", "iou", "coverage"):
        row[f"gain_D4_{metric}"] = row[f"{metric}_D4"] - row[f"{metric}_A"]
        row[f"gain_D2_{metric}"] = row[f"{metric}_D2"] - row[f"{metric}_A"]
        row[f"gain_Both_{metric}"] = row[f"{metric}_Both"] - row[f"{metric}_A"]
        row[f"S_{metric}"] = row[f"gain_D4_{metric}"] - row[f"gain_D2_{metric}"]
    # precision: only defined where both A and candidate have a prediction region
    for name in ("D4", "D2", "Both"):
        pa, pc = row["precision_A"], row[f"precision_{name}"]
        row[f"gain_{name}_precision"] = (pc - pa) if (pa is not None and pc is not None) else None
    if row["gain_D4_precision"] is not None and row["gain_D2_precision"] is not None:
        row["S_precision"] = row["gain_D4_precision"] - row["gain_D2_precision"]
    else:
        row["S_precision"] = None

    records.append(row)

n = len(records)
print(f"n components with S_dice defined: {n} (all components have S_dice/S_iou/S_coverage defined; missed components contribute 0-valued metrics per the documented convention)")

# ============================================================
# Section 6: raw distribution of S_c (dice, primary endpoint)
# ============================================================
S_dice = np.array([r["S_dice"] for r in records])
S_iou = np.array([r["S_iou"] for r in records])
S_cov = np.array([r["S_coverage"] for r in records])
gt_size = np.array([r["gt_size"] for r in records])

def describe(x, label):
    pct = np.percentile(x, [5, 25, 50, 75, 95])
    print(f"\n--- {label} (n={len(x)}) ---")
    print(f"  mean={x.mean():.4f} median={np.median(x):.4f} std={x.std():.4f} IQR={pct[3]-pct[1]:.4f}")
    print(f"  min={x.min():.4f} max={x.max():.4f}")
    print(f"  p5={pct[0]:.4f} p25={pct[1]:.4f} p50={pct[2]:.4f} p75={pct[3]:.4f} p95={pct[4]:.4f}")
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std()),
            "iqr": float(pct[3]-pct[1]), "min": float(x.min()), "max": float(x.max()),
            "p5": float(pct[0]), "p25": float(pct[1]), "p50": float(pct[2]), "p75": float(pct[3]), "p95": float(pct[4])}

print("\n\n========== SECTION 6: RAW S_c DISTRIBUTION ==========")
S_dice_desc = describe(S_dice, "S_c (dice-based)")
S_iou_desc = describe(S_iou, "S_c (IoU-based)")
S_cov_desc = describe(S_cov, "S_c (coverage-based)")

# ============================================================
# Section 6 cont: principled epsilon via bootstrap noise floor
# Estimate measurement noise: bootstrap S_c on a "null" comparison,
# same-model-vs-itself is not available (no repeated inference noise --
# forward pass is deterministic at eval()). Instead, use subject-clustered
# bootstrap SE of the MEAN S_c as the basis for a per-component noise floor:
# epsilon = 1 bootstrap-SE of a single component's S_c, approximated via the
# within-subject std of S_c for subjects with >=2 components (a real,
# measured dispersion, not an arbitrary constant).
# ============================================================
subj_ids = sorted(set(r["subject_idx"] for r in records))
by_subject = defaultdict(list)
for r in records:
    by_subject[r["subject_idx"]].append(r["S_dice"])
within_subj_stds = [np.std(v) for v in by_subject.values() if len(v) >= 2]
if within_subj_stds:
    noise_floor = float(np.median(within_subj_stds))
else:
    noise_floor = 0.05
print(f"\nPrincipled epsilon candidate (median within-subject S_c std, subjects with >=2 components, n={len(within_subj_stds)}): {noise_floor:.4f}")
EPSILON = round(noise_floor, 2) if noise_floor > 0.01 else 0.05
print(f"Using EPSILON = {EPSILON} for descriptive categorization (EXPLORATORY, per Section 6 instruction)")

# ============================================================
# Section 7: preference categories, various subsets
# ============================================================
def categorize(S, eps):
    pref = np.full(S.shape, "Neutral", dtype=object)
    pref[S > eps] = "D4-preferred"
    pref[S < -eps] = "D2-preferred"
    return pref

pref_all = categorize(S_dice, EPSILON)
for r, p in zip(records, pref_all):
    r["scale_preference"] = p

def report_categories(mask, label):
    sub = pref_all[mask]
    tot = len(sub)
    if tot == 0:
        print(f"  {label}: n=0")
        return {"n": 0}
    d4 = int((sub == "D4-preferred").sum())
    d2 = int((sub == "D2-preferred").sum())
    neu = int((sub == "Neutral").sum())
    print(f"  {label}: n={tot}  D4-preferred={d4} ({100*d4/tot:.1f}%)  D2-preferred={d2} ({100*d2/tot:.1f}%)  Neutral={neu} ({100*neu/tot:.1f}%)")
    return {"n": tot, "D4_preferred": d4, "D2_preferred": d2, "Neutral": neu,
            "D4_pct": 100*d4/tot, "D2_pct": 100*d2/tot, "Neutral_pct": 100*neu/tot}

print("\n\n========== SECTION 7: PREFERENCE CATEGORIES ==========")
category_results = {}
category_results["all_components"] = report_categories(np.ones(n, dtype=bool), "All components")

detected_all = np.array([r["detected_A"] and r["detected_D4"] and r["detected_D2"] and r["detected_Both"] for r in records])
category_results["detected_by_all_four"] = report_categories(detected_all, "Detected by all four")

size_bins = [(0, 50, "1-50"), (50, 150, "50-150"), (150, 400, "150-400"), (400, 1000, "400-1000"), (1000, np.inf, ">1000")]
for lo, hi, lab in size_bins:
    mask = (gt_size > lo) & (gt_size <= hi) if lo > 0 else (gt_size <= hi)
    category_results[f"size_{lab}"] = report_categories(mask, f"Size {lab}")

# ============================================================
# Section 8: heterogeneity test -- is D4 always winning?
# ============================================================
print("\n\n========== SECTION 8: HETEROGENEITY TEST ==========")
non_neutral_mask = pref_all != "Neutral"
n_non_neutral = non_neutral_mask.sum()
n_d4 = int((pref_all == "D4-preferred").sum())
n_d2 = int((pref_all == "D2-preferred").sum())
print(f"Non-neutral components: {n_non_neutral} / {n} ({100*n_non_neutral/n:.1f}%)")
print(f"  D4-preferred: {n_d4} ({100*n_d4/max(n_non_neutral,1):.1f}% of non-neutral)")
print(f"  D2-preferred: {n_d2} ({100*n_d2/max(n_non_neutral,1):.1f}% of non-neutral)")

if n_non_neutral > 0:
    binom_result = stats.binomtest(n_d4, n_non_neutral, p=0.5)
    ci = binom_result.proportion_ci(confidence_level=0.95)
    print(f"Binomial test vs p=0.5 (degenerate 'D4 always wins' would predict ~100% D4): p-value={binom_result.pvalue:.2e}")
    print(f"D4-preferred fraction among non-neutral: {n_d4/n_non_neutral:.3f}, 95% CI [{ci.low:.3f}, {ci.high:.3f}]")
    heterogeneity_result = {
        "n_non_neutral": int(n_non_neutral), "n_d4_preferred": n_d4, "n_d2_preferred": n_d2,
        "d4_fraction": n_d4/n_non_neutral, "d4_fraction_ci_low": ci.low, "d4_fraction_ci_high": ci.high,
        "binom_pvalue_vs_0.5": binom_result.pvalue,
    }
else:
    heterogeneity_result = {"n_non_neutral": 0}

# ============================================================
# Section 9: confounders -- baseline difficulty & size
# ============================================================
print("\n\n========== SECTION 9: CONFOUNDER ANALYSIS ==========")
baseline_dice = np.array([r["dice_A"] for r in records])
baseline_cov = np.array([r["coverage_A"] for r in records])

print("Raw Spearman correlations with S_dice:")
r1, p1 = stats.spearmanr(baseline_dice, S_dice)
r2, p2 = stats.spearmanr(baseline_cov, S_dice)
r3, p3 = stats.spearmanr(gt_size, S_dice)
print(f"  baseline dice_A  vs S_dice: rho={r1:+.3f} p={p1:.4f}")
print(f"  baseline cov_A   vs S_dice: rho={r2:+.3f} p={p2:.4f}")
print(f"  gt_size          vs S_dice: rho={r3:+.3f} p={p3:.4f}")

def partial_corr(x, y, controls):
    X = np.column_stack([np.ones(len(x))] + [c for c in controls])
    def resid(v):
        coefv, _, _, _ = np.linalg.lstsq(X, v, rcond=None)
        return v - X @ coefv
    return stats.pearsonr(resid(x), resid(y))

nonlinear_size_controls = [gt_size, gt_size ** (-1/3), np.log(gt_size)]
r_bd_partial, p_bd_partial = partial_corr(baseline_dice, S_dice, nonlinear_size_controls)
print(f"\nPartial correlation baseline_dice vs S_dice, controlling for nonlinear size (size, size^-1/3, log size):")
print(f"  partial r={r_bd_partial:+.3f} p={p_bd_partial:.4f}")

size_controls_alone = [gt_size, gt_size ** (-1/3), np.log(gt_size)]
# does a size/baseline-dice-only model explain S_c? (R^2)
X_conf = np.column_stack([np.ones(n), gt_size, gt_size ** (-1/3), np.log(gt_size), baseline_dice])
coef, _, _, _ = np.linalg.lstsq(X_conf, S_dice, rcond=None)
pred = X_conf @ coef
r2_conf = 1 - np.sum((S_dice - pred) ** 2) / np.sum((S_dice - S_dice.mean()) ** 2)
print(f"\nR^2 of S_dice ~ [size, size^-1/3, log(size), baseline_dice_A] (nonlinear size + baseline difficulty): {r2_conf:.4f}")

confounder_result = {
    "spearman_baseline_dice_vs_S": {"rho": r1, "p": p1},
    "spearman_baseline_coverage_vs_S": {"rho": r2, "p": p2},
    "spearman_size_vs_S": {"rho": r3, "p": p3},
    "partial_r_baseline_dice_controlling_nonlinear_size": {"r": r_bd_partial, "p": p_bd_partial},
    "r2_nonlinear_size_and_baseline_dice_model": r2_conf,
}

# ============================================================
# Section 15: comparison against Both
# ============================================================
print("\n\n========== SECTION 15: COMPARISON AGAINST BOTH ==========")
gain_both = np.array([r["gain_Both_dice"] for r in records])
gain_d4 = np.array([r["gain_D4_dice"] for r in records])
gain_d2 = np.array([r["gain_D2_dice"] for r in records])
max_single = np.maximum(gain_d4, gain_d2)
both_minus_max = gain_both - max_single
print(f"Q_Both - max(Q_D4,Q_D2) [component dice gain basis]: mean={both_minus_max.mean():+.4f} median={np.median(both_minus_max):+.4f}")
n_both_dominates = int((both_minus_max > EPSILON).sum())
n_max_single_dominates = int((both_minus_max < -EPSILON).sum())
n_tied = n - n_both_dominates - n_max_single_dominates
print(f"  Both clearly dominates max(D4,D2): {n_both_dominates} ({100*n_both_dominates/n:.1f}%)")
print(f"  A single scale clearly beats Both: {n_max_single_dominates} ({100*n_max_single_dominates/n:.1f}%)")
print(f"  Tied within epsilon: {n_tied} ({100*n_tied/n:.1f}%)")

both_comparison = {
    "mean_both_minus_max_single": float(both_minus_max.mean()),
    "median_both_minus_max_single": float(np.median(both_minus_max)),
    "n_both_dominates": n_both_dominates, "n_single_dominates": n_max_single_dominates, "n_tied": n_tied,
    "pct_both_dominates": 100*n_both_dominates/n, "pct_single_dominates": 100*n_max_single_dominates/n, "pct_tied": 100*n_tied/n,
}

# ============================================================
# Section 14: stability across endpoints (dice vs iou vs coverage)
# ============================================================
print("\n\n========== SECTION 14: CROSS-ENDPOINT STABILITY ==========")
pref_iou = categorize(S_iou, EPSILON)
pref_cov = categorize(S_cov, EPSILON)
agree_dice_iou = float((pref_all == pref_iou).mean())
agree_dice_cov = float((pref_all == pref_cov).mean())
print(f"Agreement dice-pref vs iou-pref: {100*agree_dice_iou:.1f}%")
print(f"Agreement dice-pref vs coverage-pref: {100*agree_dice_cov:.1f}%")
stability_result = {"agreement_dice_iou": agree_dice_iou, "agreement_dice_coverage": agree_dice_cov}

# ============================================================
# Section 13: subject-clustered bootstrap CI for D4-preferred fraction
# ============================================================
print("\n\n========== SECTION 13: SUBJECT-CLUSTERED BOOTSTRAP ==========")
subj_to_indices = defaultdict(list)
for i, r in enumerate(records):
    subj_to_indices[r["subject_idx"]].append(i)
subj_list = list(subj_to_indices.keys())

rng = np.random.RandomState(0)
n_boot = 2000
boot_d4_fracs = []
for _ in range(n_boot):
    sampled_subjs = rng.choice(subj_list, size=len(subj_list), replace=True)
    idxs = np.concatenate([subj_to_indices[s] for s in sampled_subjs])
    boot_pref = pref_all[idxs]
    nn = (boot_pref != "Neutral").sum()
    if nn == 0:
        continue
    boot_d4_fracs.append((boot_pref == "D4-preferred").sum() / nn)
boot_d4_fracs = np.array(boot_d4_fracs)
ci_lo, ci_hi = np.percentile(boot_d4_fracs, [2.5, 97.5])
print(f"Subject-clustered bootstrap (n={n_boot}) of D4-preferred fraction among non-neutral: mean={boot_d4_fracs.mean():.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
cluster_bootstrap_result = {"mean": float(boot_d4_fracs.mean()), "ci_low": float(ci_lo), "ci_high": float(ci_hi), "n_boot": n_boot}

# ============================================================
# Save master results
# ============================================================
results = {
    "n_components": n,
    "epsilon_used": EPSILON,
    "noise_floor_estimate": noise_floor,
    "early_cutoff_epoch": EARLY_CUTOFF_EPOCH,
    "S_dice_distribution": S_dice_desc,
    "S_iou_distribution": S_iou_desc,
    "S_coverage_distribution": S_cov_desc,
    "category_results": category_results,
    "heterogeneity_test": heterogeneity_result,
    "confounder_analysis": confounder_result,
    "both_comparison": both_comparison,
    "cross_endpoint_stability": stability_result,
    "subject_clustered_bootstrap": cluster_bootstrap_result,
}
with open(BASE / "E27_scale_preference_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n\nSaved results to {BASE / 'E27_scale_preference_results.json'}")

with open(BASE / "E27_component_full_table.json", "w") as f:
    json.dump(records, f, default=str)
print(f"Saved full per-component table to {BASE / 'E27_component_full_table.json'}")

# ============================================================
# Figures
# ============================================================
if HAVE_MPL:
    plt.figure(figsize=(7, 5))
    plt.hist(S_dice, bins=60, color="steelblue", edgecolor="none")
    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("S_c = G_c,D4 - G_c,D2 (component Dice basis)")
    plt.ylabel("count")
    plt.title("Figure 1: Distribution of scale preference S_c")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure1_S_c_histogram.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(gain_d4, gain_d2, s=8, alpha=0.4, color="darkorange")
    lims = [min(gain_d4.min(), gain_d2.min()), max(gain_d4.max(), gain_d2.max())]
    plt.plot(lims, lims, "k--", linewidth=1, label="G_D4 = G_D2")
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.xlabel("G_c,D4 (component Dice gain, D4 vs A)")
    plt.ylabel("G_c,D2 (component Dice gain, D2 vs A)")
    plt.title("Figure 2: D4 gain vs D2 gain per component")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_gain_scatter.png", dpi=120)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(baseline_dice, S_dice, s=8, alpha=0.4, color="seagreen")
    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Baseline (A) component Dice")
    plt.ylabel("S_c")
    plt.title("Figure 3: S_c vs baseline component difficulty")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure3_S_vs_baseline_dice.png", dpi=120)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(np.log(gt_size), S_dice, s=8, alpha=0.4, color="indianred")
    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("log(component size)")
    plt.ylabel("S_c")
    plt.title("Figure 4: S_c vs log(component size)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure4_S_vs_log_size.png", dpi=120)
    plt.close()

    print("Saved figures 1-4 to figures/ (figure 5 requires the trajectory table, produced separately)")
else:
    print("matplotlib not available -- skipped figure generation")
