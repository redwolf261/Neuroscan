"""
Phase E1.2: Is uncertainty simply controlled by latent feature magnitude?

H0: Evidence ~ f(||z||) almost entirely (evidential head just converts
    feature norm into uncertainty; no need for representation-learning
    algorithms).
H1: Feature norm explains only a fraction; evidence depends on direction/
    semantic structure, not magnitude alone -- motivates representation
    geometry as the next investigation target.

Reads experiments/exp_e_latent_analysis/extracted/per_voxel_stats.csv
(60,000 voxels, 30 validation volumes, frozen baseline seed 0). Reuses
E1.1's boundary_distance column for the boundary-controlled and multiple-
regression analyses. No retraining, no new inference.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

base = Path(__file__).parent
data_path = base / "extracted" / "per_voxel_stats.csv"
out_dir = base / "e1_2_results"
out_dir.mkdir(exist_ok=True)

print("Loading per-voxel data...")
with open(data_path, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

feature_norm = np.array([float(r["feature_norm"]) for r in rows])
boundary_distance = np.array([float(r["boundary_distance"]) for r in rows])
evidence = np.array([float(r["evidence"]) for r in rows])
entropy = np.array([float(r["entropy"]) for r in rows])
confidence = np.array([float(r["confidence"]) for r in rows])
correct = np.array([float(r["correct"]) for r in rows]).astype(bool)
ground_truth = np.array([float(r["ground_truth"]) for r in rows]).astype(bool)

n = len(rows)
print(f"n = {n} voxels, {sum(ground_truth)} tumor ({100*sum(ground_truth)/n:.2f}%), "
      f"{sum(~ground_truth)} background")

results = {}
plot_idx = np.random.RandomState(0).choice(n, size=min(15000, n), replace=False)

# ---------------------------------------------------------------------
# Analysis 1: Scatter + Pearson/Spearman/polynomial R^2
# ---------------------------------------------------------------------
print("\n[Analysis 1] Feature norm vs evidence: scatter + correlations")
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(feature_norm[plot_idx], evidence[plot_idx], s=2, alpha=0.15, c="teal")
ax.set_xlabel("Feature norm ||dec1||")
ax.set_ylabel("Total evidence")
ax.set_title("Evidence vs. Feature Norm (all 30 volumes, 60k voxels)")
fig.tight_layout()
fig.savefig(out_dir / "01_scatter_evidence_vs_featnorm.png", dpi=150)
plt.close(fig)

r_pearson, p_pearson = stats.pearsonr(feature_norm, evidence)
rho_spearman, p_spearman = stats.spearmanr(feature_norm, evidence)
print(f"  Pearson r = {r_pearson:+.4f}, p = {p_pearson:.2e}")
print(f"  Spearman rho = {rho_spearman:+.4f}, p = {p_spearman:.2e}")
results["pearson_r"] = float(r_pearson)
results["pearson_p"] = float(p_pearson)
results["spearman_rho"] = float(rho_spearman)
results["spearman_p"] = float(p_spearman)

poly_r2 = {}
for degree in (1, 2, 3, 4):
    coeffs = np.polyfit(feature_norm, evidence, degree)
    y_pred = np.polyval(coeffs, feature_norm)
    r2 = r2_score(evidence, y_pred)
    poly_r2[degree] = float(r2)
    print(f"  polynomial degree {degree}: R^2 = {r2:.4f}")
results["polynomial_r2"] = poly_r2

# ---------------------------------------------------------------------
# Analysis 2: LOWESS
# ---------------------------------------------------------------------
print("\n[Analysis 2] LOWESS smoothing")
lowess_idx = np.random.RandomState(1).choice(n, size=min(8000, n), replace=False)
lowess_result = lowess(evidence[lowess_idx], feature_norm[lowess_idx], frac=0.15, it=1)
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(feature_norm[plot_idx], evidence[plot_idx], s=2, alpha=0.1, c="lightgray", label="voxels")
ax.plot(lowess_result[:, 0], lowess_result[:, 1], color="darkorange", linewidth=2.5, label="LOWESS")
ax.set_xlabel("Feature norm ||dec1||")
ax.set_ylabel("Total evidence")
ax.set_title("LOWESS: Evidence vs. Feature Norm")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "02_lowess_evidence_vs_featnorm.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# Analysis 3: Correct vs Incorrect (feature norm distributions)
# ---------------------------------------------------------------------
print("\n[Analysis 3] Feature norm: correct vs incorrect voxels")
fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(feature_norm[correct], bins=60, alpha=0.5, density=True, label=f"Correct (n={correct.sum()})", color="seagreen")
ax.hist(feature_norm[~correct], bins=60, alpha=0.5, density=True, label=f"Incorrect (n={(~correct).sum()})", color="crimson")
ax.set_xlabel("Feature norm ||dec1||")
ax.set_ylabel("Density")
ax.set_title("Feature Norm Distribution: Correct vs. Incorrect")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "03_featnorm_correct_vs_incorrect.png", dpi=150)
plt.close(fig)

t_fn, p_fn = stats.ttest_ind(feature_norm[correct], feature_norm[~correct], equal_var=False)
print(f"  mean feature_norm correct={feature_norm[correct].mean():.3f}, "
      f"incorrect={feature_norm[~correct].mean():.3f}")
print(f"  Welch t-test: t={t_fn:.3f}, p={p_fn:.2e}")
results["featnorm_correct_vs_incorrect"] = {
    "mean_correct": float(feature_norm[correct].mean()),
    "mean_incorrect": float(feature_norm[~correct].mean()),
    "t": float(t_fn), "p": float(p_fn),
}
# Direct comparison: which separates errors better, evidence or feature norm?
# Use a simple separation metric: |mean difference| / pooled std (Cohen's d)
def cohens_d(a, b):
    pooled_std = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0

d_evidence = cohens_d(evidence[correct], evidence[~correct])
d_featnorm = cohens_d(feature_norm[correct], feature_norm[~correct])
print(f"  Cohen's d (correct vs incorrect): evidence={d_evidence:.3f}, feature_norm={d_featnorm:.3f}")
results["cohens_d_evidence"] = float(d_evidence)
results["cohens_d_featnorm"] = float(d_featnorm)

# ---------------------------------------------------------------------
# Analysis 4: Boundary-controlled analysis (|boundary_distance| <= 2)
# ---------------------------------------------------------------------
print("\n[Analysis 4] Boundary-controlled: voxels within +/-2 of boundary")
near_boundary = np.abs(boundary_distance) <= 2
n_near = near_boundary.sum()
print(f"  n voxels within +/-2 of boundary: {n_near}")
if n_near >= 30:
    r_nb, p_nb = stats.pearsonr(feature_norm[near_boundary], evidence[near_boundary])
    rho_nb, ps_nb = stats.spearmanr(feature_norm[near_boundary], evidence[near_boundary])
    print(f"  Within this band: Pearson r={r_nb:+.4f} (p={p_nb:.2e}), Spearman rho={rho_nb:+.4f} (p={ps_nb:.2e})")
    results["boundary_controlled"] = {
        "n": int(n_near), "pearson_r": float(r_nb), "pearson_p": float(p_nb),
        "spearman_rho": float(rho_nb), "spearman_p": float(ps_nb),
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(feature_norm[near_boundary], evidence[near_boundary], s=4, alpha=0.3, c="darkviolet")
    ax.set_xlabel("Feature norm ||dec1||")
    ax.set_ylabel("Total evidence")
    ax.set_title(f"Evidence vs. Feature Norm, boundary-controlled (|dist|<=2, n={n_near})")
    fig.tight_layout()
    fig.savefig(out_dir / "04_boundary_controlled.png", dpi=150)
    plt.close(fig)
else:
    print("  WARNING: too few voxels within +/-2 of boundary for a stable estimate")
    results["boundary_controlled"] = {"n": int(n_near), "note": "too few for stable estimate"}

# ---------------------------------------------------------------------
# Analysis 5: Tumor / background split
# ---------------------------------------------------------------------
print("\n[Analysis 5] Feature norm vs evidence: tumor vs background")
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
class_stats = {}
for ax, mask, label, color in (
    (axes[0], ground_truth, "Tumor (GT positive)", "purple"),
    (axes[1], ~ground_truth, "Background (GT negative)", "gray"),
):
    idx = np.where(mask)[0]
    plot_sub = np.random.RandomState(3).choice(idx, size=min(8000, len(idx)), replace=False) if len(idx) > 0 else idx
    ax.scatter(feature_norm[plot_sub], evidence[plot_sub], s=3, alpha=0.2, c=color)
    ax.set_title(f"{label} (n={mask.sum()})")
    ax.set_xlabel("Feature norm ||dec1||")
    if mask.sum() >= 2:
        r, p = stats.pearsonr(feature_norm[mask], evidence[mask])
        class_stats[label] = {"n": int(mask.sum()), "pearson_r": float(r), "pearson_p": float(p)}
        print(f"  {label}: n={mask.sum()}, r={r:+.4f} (p={p:.2e})")
axes[0].set_ylabel("Total evidence")
fig.suptitle("Evidence vs. Feature Norm, split by ground-truth class")
fig.tight_layout()
fig.savefig(out_dir / "05_tumor_vs_background.png", dpi=150)
plt.close(fig)
results["class_split"] = class_stats

# ---------------------------------------------------------------------
# Analysis 6: Multiple regression, incremental R^2
# ---------------------------------------------------------------------
print("\n[Analysis 6] Multiple regression: Evidence ~ boundary_distance + feature_norm")
X_boundary = boundary_distance.reshape(-1, 1)
X_both = np.column_stack([boundary_distance, feature_norm])
X_featnorm = feature_norm.reshape(-1, 1)

lr_boundary = LinearRegression().fit(X_boundary, evidence)
r2_boundary_only = r2_score(evidence, lr_boundary.predict(X_boundary))

lr_featnorm = LinearRegression().fit(X_featnorm, evidence)
r2_featnorm_only = r2_score(evidence, lr_featnorm.predict(X_featnorm))

lr_both = LinearRegression().fit(X_both, evidence)
r2_both = r2_score(evidence, lr_both.predict(X_both))

incremental_r2 = r2_both - r2_boundary_only
print(f"  R^2 (boundary only)              = {r2_boundary_only:.4f}")
print(f"  R^2 (feature_norm only)          = {r2_featnorm_only:.4f}")
print(f"  R^2 (boundary + feature_norm)    = {r2_both:.4f}")
print(f"  Incremental R^2 from adding feature_norm to boundary = {incremental_r2:.4f}")
results["multiple_regression"] = {
    "r2_boundary_only": float(r2_boundary_only),
    "r2_featnorm_only": float(r2_featnorm_only),
    "r2_both": float(r2_both),
    "incremental_r2_featnorm_given_boundary": float(incremental_r2),
}

# ---------------------------------------------------------------------
# Analysis 7: Partial correlation (evidence, feature_norm | boundary_distance)
# ---------------------------------------------------------------------
print("\n[Analysis 7] Partial correlation: corr(evidence, feature_norm | boundary_distance)")
def partial_corr(x, y, z):
    """Linear residualization partial correlation."""
    def residualize(a, b):
        b2 = b.reshape(-1, 1)
        beta = LinearRegression().fit(b2, a)
        return a - beta.predict(b2)
    rx = residualize(x, z)
    ry = residualize(y, z)
    r, p = stats.pearsonr(rx, ry)
    return r, p

r_partial, p_partial = partial_corr(evidence, feature_norm, boundary_distance)
print(f"  partial r = {r_partial:+.4f}, p = {p_partial:.2e}")
print(f"  (raw, uncontrolled r = {r_pearson:+.4f} for comparison)")
results["partial_correlation_featnorm_given_boundary"] = {"r": float(r_partial), "p": float(p_partial)}

# ---------------------------------------------------------------------
# Analysis 8: Permutation importance (simple RF regressor on both features)
# ---------------------------------------------------------------------
print("\n[Analysis 8] Permutation importance (RandomForest on boundary_distance + feature_norm)")
rng = np.random.RandomState(42)
sample_idx = rng.choice(n, size=min(20000, n), replace=False)  # RF on 20k for speed
X_imp = np.column_stack([boundary_distance, feature_norm])[sample_idx]
y_imp = evidence[sample_idx]

rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_imp, y_imp)
rf_r2 = r2_score(y_imp, rf.predict(X_imp))
print(f"  RandomForest train R^2 = {rf_r2:.4f} (in-sample, for reference only)")

perm_result = permutation_importance(rf, X_imp, y_imp, n_repeats=10, random_state=42, n_jobs=-1)
feature_names = ["boundary_distance", "feature_norm"]
print("  Permutation importance (mean decrease in R^2 when shuffled):")
for i, name in enumerate(feature_names):
    print(f"    {name:20s}: {perm_result.importances_mean[i]:.4f} +/- {perm_result.importances_std[i]:.4f}")
results["permutation_importance"] = {
    name: {"mean": float(perm_result.importances_mean[i]), "std": float(perm_result.importances_std[i])}
    for i, name in enumerate(feature_names)
}
results["rf_r2_in_sample"] = float(rf_r2)

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(feature_names, perm_result.importances_mean, yerr=perm_result.importances_std, capsize=5,
       color=["steelblue", "teal"])
ax.set_ylabel("Permutation importance (mean R^2 decrease)")
ax.set_title("Feature importance for predicting evidence")
fig.tight_layout()
fig.savefig(out_dir / "06_permutation_importance.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# Save results + verdict
# ---------------------------------------------------------------------
with open(out_dir / "results_summary.json", "w") as f:
    json.dump(results, f, indent=2)

best_featnorm_r2 = max(poly_r2.values())
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Feature norm alone: Pearson r={r_pearson:+.4f}, best polynomial R^2={best_featnorm_r2:.4f}")
print(f"Boundary alone (from E1.1 comparison): R^2=0.150")
print(f"Both together: R^2={r2_both:.4f} (incremental from feature_norm: {incremental_r2:+.4f})")
print(f"Partial corr(evidence, feature_norm | boundary) = {r_partial:+.4f}")

if best_featnorm_r2 > 0.7:
    verdict = "H0 supported: feature magnitude alone explains most evidence variance -- no need for representation-level algorithms."
elif best_featnorm_r2 > 0.3:
    verdict = "Partial: feature magnitude explains SOME variance, but substantial variance remains unexplained."
else:
    verdict = ("H1 supported: neither boundary distance NOR feature magnitude explain uncertainty well. "
               "Latent representation GEOMETRY (not just scale) is the remaining candidate explanation.")
print(f"\nVerdict: {verdict}")
print(f"\nAll outputs saved to {out_dir}")
