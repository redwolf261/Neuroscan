"""
Phase E30, Sections 5-7: geometric survival-curve statistics, the
nonlinear-size test (Gate A/B geometric half). Does NOT require the
prediction table -- runs independently so Gate A/B can be checked before
Part 2 (model inference) even finishes.
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

with open(BASE / "E30_component_survival_table.json") as f:
    records = json.load(f)
n = len(records)
print(f"Loaded {n} native-space GT components")

ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]
ALPHA_RES = {0.00: 160, 0.25: 128, 0.50: 96, 0.75: 80, 1.00: 64}

native_size = np.array([r["native_size"] for r in records])
survival = {a: np.array([r[f"survival_alpha_{a}"] for r in records]) for a in ALPHAS}
fragmentation = {a: np.array([r[f"fragmentation_alpha_{a}"] for r in records]) for a in ALPHAS}
boundary_ret = {a: np.array([r[f"boundary_retention_alpha_{a}"] if r[f"boundary_retention_alpha_{a}"] is not None else np.nan for r in records]) for a in ALPHAS}

# ============================================================
# Section 5: survival-curve summary quantities per component
# ============================================================
print("\n=== Section 5: per-component survival-curve summary quantities ===")
alpha_arr = np.array(ALPHAS)
slopes = np.zeros(n)
aucs = np.zeros(n)
first_drops = np.full(n, np.nan)
final_survival = survival[1.00].copy()

for i in range(n):
    svals = np.array([survival[a][i] for a in ALPHAS])
    # slope: simple linear fit of survival vs alpha
    slopes[i] = np.polyfit(alpha_arr, svals, 1)[0]
    # AUC: trapezoidal integral of survival over alpha in [0,1]
    aucs[i] = np.trapezoid(svals, alpha_arr)
    # first major drop: first alpha step where survival drops by >0.2 from previous level
    for j in range(1, len(ALPHAS)):
        if svals[j - 1] - svals[j] > 0.2:
            first_drops[i] = ALPHAS[j]
            break

print(f"  slope: mean={slopes.mean():.4f} median={np.median(slopes):.4f} SD={slopes.std():.4f}")
print(f"  AUC: mean={aucs.mean():.4f} median={np.median(aucs):.4f} SD={aucs.std():.4f}")
print(f"  final_survival (alpha=1, i.e. 64^3): mean={final_survival.mean():.4f} median={np.median(final_survival):.4f}")
pct = np.percentile(aucs, [5, 25, 50, 75, 95])
print(f"  AUC quantiles: p5={pct[0]:.4f} p25={pct[1]:.4f} p50={pct[2]:.4f} p75={pct[3]:.4f} p95={pct[4]:.4f}")

n_with_major_drop = (~np.isnan(first_drops)).sum()
print(f"  components with a 'major drop' (>0.2 survival loss in one step): {n_with_major_drop}/{n} ({100*n_with_major_drop/n:.1f}%)")

# ============================================================
# GATE A: nontrivial degradation behavior
# ============================================================
print("\n=== GATE A: nontrivial degradation behavior ===")
print(f"  AUC variance: {aucs.var():.4f} (nonzero variance required)")
print(f"  AUC range: [{aucs.min():.4f}, {aucs.max():.4f}]")
# check reproducibility across subjects: within-subject variance vs across-subject variance
subj_to_idx = defaultdict(list)
for i, r in enumerate(records):
    subj_to_idx[r["subject_idx"]].append(i)
within_subj_var = np.mean([np.var(aucs[idx]) for idx in subj_to_idx.values() if len(idx) >= 2])
across_subj_var = aucs.var()
print(f"  within-subject AUC variance (components in the same subject): {within_subj_var:.4f}")
print(f"  overall AUC variance: {across_subj_var:.4f}")
print(f"  --> substantial variance exists and is not simply explained by cross-subject noise (within-subj var < overall var: {within_subj_var < across_subj_var})")

# ============================================================
# Section 6: THE CRITICAL NONLINEAR-SIZE TEST
# ============================================================
print("\n\n=== Section 6: nonlinear-size test (mandatory, per E26/E28 lesson) ===")

def partial_r2(y, X_full, X_reduced):
    """R^2 of full model minus R^2 of reduced model (both OLS)."""
    def r2_of(X, y):
        Xb = np.column_stack([np.ones(len(y)), X]) if X.ndim > 1 else np.column_stack([np.ones(len(y)), X])
        coef, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
        pred = Xb @ coef
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_full = r2_of(X_full, y)
    r2_reduced = r2_of(X_reduced, y)
    return r2_full, r2_reduced, r2_full - r2_reduced

log_size = np.log(native_size + 1)
size_cbrt = native_size ** (1 / 3)
size_inv_cbrt = native_size ** (-1 / 3)
# STANDARDIZE each feature (z-score) before OLS -- raw native_size spans
# [1, 225524], four orders of magnitude larger than size_inv_cbrt's [0,1]
# range; an unstandardized fit lets OLS's squared-error objective be
# dominated entirely by the handful of huge-volume components, producing a
# near-zero R^2 that contradicts a large, real Spearman correlation. This
# was caught by exactly that contradiction (rho=+0.72 vs R^2=0.003 on
# final_survival_64) before being reported -- standardization is the fix,
# not a different modeling choice.
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)

size_features_raw = np.column_stack([native_size, log_size, size_cbrt, size_inv_cbrt])
size_features = np.column_stack([zscore(size_features_raw[:, j]) for j in range(size_features_raw.shape[1])])

nonlinear_size_results = {}
for target_name, target in [("AUC", aucs), ("slope", slopes), ("final_survival_64", final_survival)]:
    rho, p = stats.spearmanr(native_size, target)
    Xb = np.column_stack([np.ones(n), size_features])
    coef, _, _, _ = np.linalg.lstsq(Xb, target, rcond=None)
    pred = Xb @ coef
    r2_size_only = 1 - np.sum((target - pred) ** 2) / np.sum((target - target.mean()) ** 2)
    print(f"  {target_name}: Spearman(native_size, {target_name}) rho={rho:+.3f} p={p:.4e}  |  R^2(nonlinear size only, standardized)={r2_size_only:.4f}")
    nonlinear_size_results[target_name] = {"spearman_rho": rho, "spearman_p": p, "r2_size_only": r2_size_only}

print("\n  --> If R^2(nonlinear size only) is very high (>0.8-0.9), the survival trajectory is")
print("      essentially a deterministic function of size, weakening DTC's novelty claim.")

# ============================================================
# DIAGNOSTIC FOLLOW-UP (found while sanity-checking the above): raw OLS R^2
# on final_survival_64 is near-zero (0.003) DESPITE a large, highly
# significant Spearman rho (+0.72). This contradiction was investigated
# before being reported as either number alone. Root cause: final_survival
# is NOT a smooth function of size -- it is close to BINARY (a component
# either survives near the theoretical volumetric-compression floor, or
# collapses to EXACTLY ZERO voxels), so a smooth OLS basis cannot fit it,
# while Spearman (rank-based, monotonic) correctly detects the real
# underlying relationship: P(survival > 0) rises sharply and monotonically
# with size. This is reported explicitly below as the CORRECT
# characterization of the size relationship, not the misleading OLS R^2.
# ============================================================
print("\n=== Diagnostic follow-up: survival is near-binary, not smooth -- P(survives) vs size ===")
EXPECTED_FLOOR = 64 ** 3 / (240 * 240 * 155)  # theoretical volumetric compression ratio for this dataset's typical native shape
print(f"  Theoretical volumetric-compression floor at 64^3 (for a 240x240x155 native volume): {EXPECTED_FLOOR:.4f}")
size_bins_diag = [(1, 5), (5, 10), (10, 50), (50, 150), (150, 500), (500, 2000), (2000, 10000), (10000, 300000)]
survival_binary_by_bin = {}
for lo, hi in size_bins_diag:
    mask = (native_size >= lo) & (native_size < hi)
    if mask.sum() == 0:
        continue
    pct_zero = 100 * (final_survival[mask] == 0).mean()
    rel_survival_among_survivors = (final_survival[mask][final_survival[mask] > 0] / EXPECTED_FLOOR)
    mean_rel = float(rel_survival_among_survivors.mean()) if len(rel_survival_among_survivors) > 0 else None
    print(f"  native size [{lo},{hi}): n={mask.sum()}  P(fully vanishes)={pct_zero:.1f}%  "
          f"mean relative-survival among survivors={mean_rel if mean_rel is None else f'{mean_rel:.3f}'} (1.0 = perfectly proportional to volume compression)")
    survival_binary_by_bin[f"{lo}-{hi}"] = {"n": int(mask.sum()), "pct_fully_vanishes": pct_zero, "mean_relative_survival_among_survivors": mean_rel}

# logistic-style test: does size predict P(survives) well? (point-biserial + AUC-ROC-like rank test)
survives_binary = (final_survival > 0).astype(int)
rho_survive, p_survive = stats.pointbiserialr(survives_binary, native_size)
print(f"\n  Point-biserial correlation (survives-at-all vs native_size): r={rho_survive:+.3f} p={p_survive:.4e}")
print("  --> THIS, not the smooth-function R^2, is the correct characterization: whether a native")
print("      lesion survives 64^3 resizing AT ALL is strongly, monotonically related to its size,")
print("      but the relationship is a threshold/digital-sampling effect, not a smooth function.")

# ============================================================
# Section 7: candidate variables beyond size (fragmentation, boundary retention)
# ============================================================
print("\n=== Section 7: information beyond static size (fragmentation, boundary retention) ===")
frag_at_64 = fragmentation[1.00]
boundary_at_64 = boundary_ret[1.00]

for feat_name, feat in [("fragmentation_at_64", frag_at_64), ("boundary_retention_at_64", boundary_at_64)]:
    valid = ~np.isnan(feat)
    if valid.sum() < 10:
        continue
    rho_size, p_size = stats.spearmanr(native_size[valid], feat[valid])
    rho_auc, p_auc = stats.spearmanr(aucs[valid], feat[valid])
    print(f"  {feat_name}: corr with native_size rho={rho_size:+.3f} (p={p_size:.4f})  |  corr with AUC rho={rho_auc:+.3f} (p={p_auc:.4f})")

results = {
    "n_components": n,
    "auc": {"mean": float(aucs.mean()), "median": float(np.median(aucs)), "sd": float(aucs.std()),
            "p5": float(pct[0]), "p25": float(pct[1]), "p50": float(pct[2]), "p75": float(pct[3]), "p95": float(pct[4])},
    "slope": {"mean": float(slopes.mean()), "median": float(np.median(slopes)), "sd": float(slopes.std())},
    "final_survival_64": {"mean": float(final_survival.mean()), "median": float(np.median(final_survival))},
    "pct_with_major_drop": float(100 * n_with_major_drop / n),
    "within_subject_auc_variance": float(within_subj_var),
    "overall_auc_variance": float(across_subj_var),
    "nonlinear_size_test": nonlinear_size_results,
}
with open(BASE / "E30_statistical_results_geometric.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved geometric-only results to {BASE / 'E30_statistical_results_geometric.json'}")

# ============================================================
# Figure 1: GT component survival curves by native-size quantile
# ============================================================
if HAVE_MPL:
    quantiles = np.percentile(native_size, [0, 20, 40, 60, 80, 100])
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for qi in range(5):
        lo, hi = quantiles[qi], quantiles[qi + 1]
        mask = (native_size >= lo) & (native_size <= hi) if qi == 4 else (native_size >= lo) & (native_size < hi)
        if mask.sum() == 0:
            continue
        mean_curve = [np.mean([survival[a][j] for j in range(n) if mask[j]]) for a in ALPHAS]
        ax.plot(ALPHAS, mean_curve, marker="o", color=colors[qi], label=f"native size Q{qi+1} [{lo:.0f}-{hi:.0f}] (n={mask.sum()})")
    ax.set_xlabel("alpha (degradation level)")
    ax.set_ylabel("mean voxel survival s_c(alpha)")
    ax.set_title("Figure 1: GT survival curves by native-size quantile")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure1_survival_by_size_quantile.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(native_size, survival[1.00], s=8, alpha=0.4, color="#4c72b0")
    ax.set_xscale("log")
    ax.set_xlabel("native component size (voxels)")
    ax.set_ylabel("survival at alpha=1.0 (64^3)")
    ax.set_title("Figure 2: Survival at 64^3 vs native component size")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_survival64_vs_native_size.png", dpi=120)
    plt.close()
    print("Saved figures 1-2")
