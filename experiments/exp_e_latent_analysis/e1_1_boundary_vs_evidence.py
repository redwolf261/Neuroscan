"""
Phase E1.1: Is the evidential branch's uncertainty just boundary ambiguity?

H0: Evidence ~ f(boundary_distance) almost entirely (evidence head behaves
    like a boundary detector; representation learning isn't the bottleneck).
H1: Boundary distance explains only part of the variance; something else
    in the latent representation is being encoded.

Reads experiments/exp_e_latent_analysis/extracted/per_voxel_stats.csv
(60,000 voxels, 30 validation volumes, frozen baseline seed 0). Runs the
7 analyses from the design brief and saves plots + a results summary.
No retraining, no new inference -- pure analysis of already-extracted data.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import r2_score

base = Path(__file__).parent
data_path = base / "extracted" / "per_voxel_stats.csv"
out_dir = base / "e1_1_results"
out_dir.mkdir(exist_ok=True)

print("Loading per-voxel data...")
with open(data_path, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

boundary_distance = np.array([float(r["boundary_distance"]) for r in rows])
evidence = np.array([float(r["evidence"]) for r in rows])
entropy = np.array([float(r["entropy"]) for r in rows])
confidence = np.array([float(r["confidence"]) for r in rows])
correct = np.array([float(r["correct"]) for r in rows]).astype(bool)
ground_truth = np.array([float(r["ground_truth"]) for r in rows]).astype(bool)
prediction = np.array([float(r["prediction"]) for r in rows]).astype(bool)

n = len(rows)
print(f"n = {n} voxels, {sum(ground_truth)} tumor ({100*sum(ground_truth)/n:.2f}%), "
      f"{sum(~ground_truth)} background")

results = {}

# ---------------------------------------------------------------------
# Analysis 1: Scatter (boundary_distance -> evidence)
# ---------------------------------------------------------------------
print("\n[Analysis 1] Scatter plot: boundary_distance vs evidence")
fig, ax = plt.subplots(figsize=(9, 6))
# Subsample for plotting only (all 60k points used in every numeric analysis below)
plot_idx = np.random.RandomState(0).choice(n, size=min(15000, n), replace=False)
ax.scatter(boundary_distance[plot_idx], evidence[plot_idx], s=2, alpha=0.15, c="steelblue")
ax.axvline(0, color="red", linestyle="--", linewidth=1, label="boundary (dist=0)")
ax.set_xlabel("Signed boundary distance (voxels; + = inside tumor, - = outside)")
ax.set_ylabel("Total evidence (alpha + beta - 2)")
ax.set_title("Evidence vs. Boundary Distance (all 30 volumes, 60k voxels)")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "01_scatter_evidence_vs_boundary.png", dpi=150)
plt.close(fig)
print(f"  saved 01_scatter_evidence_vs_boundary.png")

# ---------------------------------------------------------------------
# Analysis 2: Pearson correlation
# ---------------------------------------------------------------------
print("\n[Analysis 2] Pearson correlation")
r_pearson, p_pearson = stats.pearsonr(boundary_distance, evidence)
print(f"  r = {r_pearson:+.4f}, p = {p_pearson:.2e}")
results["pearson_r"] = float(r_pearson)
results["pearson_p"] = float(p_pearson)

# ---------------------------------------------------------------------
# Analysis 3: Spearman correlation
# ---------------------------------------------------------------------
print("\n[Analysis 3] Spearman correlation")
rho_spearman, p_spearman = stats.spearmanr(boundary_distance, evidence)
print(f"  rho = {rho_spearman:+.4f}, p = {p_spearman:.2e}")
results["spearman_rho"] = float(rho_spearman)
results["spearman_p"] = float(p_spearman)

# ---------------------------------------------------------------------
# Analysis 4: LOWESS curve
# ---------------------------------------------------------------------
print("\n[Analysis 4] LOWESS smoothing")
# LOWESS on the full 60k points is expensive; use a deterministic random
# subsample of 8000 for the smoother itself (standard practice), but note
# Pearson/Spearman/R^2 above already used the FULL dataset.
lowess_idx = np.random.RandomState(1).choice(n, size=min(8000, n), replace=False)
lowess_result = lowess(evidence[lowess_idx], boundary_distance[lowess_idx], frac=0.15, it=1)
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(boundary_distance[plot_idx], evidence[plot_idx], s=2, alpha=0.1, c="lightgray", label="voxels")
ax.plot(lowess_result[:, 0], lowess_result[:, 1], color="darkorange", linewidth=2.5, label="LOWESS")
ax.axvline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("Signed boundary distance")
ax.set_ylabel("Total evidence")
ax.set_title("LOWESS: Evidence vs. Boundary Distance")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "02_lowess_evidence_vs_boundary.png", dpi=150)
plt.close(fig)
print(f"  saved 02_lowess_evidence_vs_boundary.png")

# ---------------------------------------------------------------------
# Analysis 5: Polynomial regression R^2
# ---------------------------------------------------------------------
print("\n[Analysis 5] Polynomial regression R^2 (degrees 1-4)")
poly_r2 = {}
x = boundary_distance.reshape(-1, 1)
y = evidence
for degree in (1, 2, 3, 4):
    coeffs = np.polyfit(boundary_distance, evidence, degree)
    y_pred = np.polyval(coeffs, boundary_distance)
    r2 = r2_score(y, y_pred)
    poly_r2[degree] = float(r2)
    print(f"  degree {degree}: R^2 = {r2:.4f}")
results["polynomial_r2"] = poly_r2

# ---------------------------------------------------------------------
# Analysis 6: Split by correctness
# ---------------------------------------------------------------------
print("\n[Analysis 6] Correct vs. Incorrect voxels")
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
for ax, mask, label, color in (
    (axes[0], correct, "Correct", "seagreen"),
    (axes[1], ~correct, "Incorrect", "crimson"),
):
    idx = np.where(mask)[0]
    plot_sub = np.random.RandomState(2).choice(idx, size=min(8000, len(idx)), replace=False) if len(idx) > 0 else idx
    ax.scatter(boundary_distance[plot_sub], evidence[plot_sub], s=3, alpha=0.2, c=color)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{label} voxels (n={mask.sum()})")
    ax.set_xlabel("Signed boundary distance")
axes[0].set_ylabel("Total evidence")
fig.suptitle("Evidence vs. Boundary Distance, split by prediction correctness")
fig.tight_layout()
fig.savefig(out_dir / "03_correct_vs_incorrect.png", dpi=150)
plt.close(fig)

correctness_stats = {}
for label, mask in (("correct", correct), ("incorrect", ~correct)):
    if mask.sum() >= 2:
        r, p = stats.pearsonr(boundary_distance[mask], evidence[mask])
        correctness_stats[label] = {
            "n": int(mask.sum()),
            "mean_evidence": float(evidence[mask].mean()),
            "mean_confidence": float(confidence[mask].mean()),
            "pearson_r": float(r),
            "pearson_p": float(p),
        }
        print(f"  {label}: n={mask.sum()}, mean_evidence={evidence[mask].mean():.3f}, "
              f"mean_confidence={confidence[mask].mean():.3f}, r={r:+.4f}")
results["correctness_split"] = correctness_stats

# Direct test: does the network have HIGHER uncertainty (lower confidence,
# lower evidence) on voxels it gets wrong? This is the calibration question,
# independent of boundary distance.
if (~correct).sum() >= 2:
    t_conf, p_conf = stats.ttest_ind(confidence[correct], confidence[~correct], equal_var=False)
    print(f"  Welch t-test confidence (correct vs incorrect): t={t_conf:.3f}, p={p_conf:.2e}")
    results["confidence_correct_vs_incorrect_ttest"] = {"t": float(t_conf), "p": float(p_conf)}

# ---------------------------------------------------------------------
# Analysis 7: Split by class (tumor vs background)
# ---------------------------------------------------------------------
print("\n[Analysis 7] Tumor vs. Background voxels")
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
for ax, mask, label, color in (
    (axes[0], ground_truth, "Tumor (GT positive)", "purple"),
    (axes[1], ~ground_truth, "Background (GT negative)", "gray"),
):
    idx = np.where(mask)[0]
    plot_sub = np.random.RandomState(3).choice(idx, size=min(8000, len(idx)), replace=False) if len(idx) > 0 else idx
    ax.scatter(boundary_distance[plot_sub], evidence[plot_sub], s=3, alpha=0.2, c=color)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{label} (n={mask.sum()})")
    ax.set_xlabel("Signed boundary distance")
axes[0].set_ylabel("Total evidence")
fig.suptitle("Evidence vs. Boundary Distance, split by ground-truth class")
fig.tight_layout()
fig.savefig(out_dir / "04_tumor_vs_background.png", dpi=150)
plt.close(fig)

class_stats = {}
for label, mask in (("tumor", ground_truth), ("background", ~ground_truth)):
    if mask.sum() >= 2:
        r, p = stats.pearsonr(boundary_distance[mask], evidence[mask])
        class_stats[label] = {
            "n": int(mask.sum()),
            "mean_evidence": float(evidence[mask].mean()),
            "pearson_r": float(r),
            "pearson_p": float(p),
        }
        print(f"  {label}: n={mask.sum()}, mean_evidence={evidence[mask].mean():.3f}, r={r:+.4f}")
results["class_split"] = class_stats

# ---------------------------------------------------------------------
# Same 7 analyses repeated for entropy and confidence (not just evidence)
# per the design brief's column list -- evidence alone could be
# misleading if it's driven by only one of alpha/beta's scale.
# ---------------------------------------------------------------------
print("\n[Additional] Same correlations for entropy and confidence")
for target_name, target in (("entropy", entropy), ("confidence", confidence)):
    r, p = stats.pearsonr(boundary_distance, target)
    rho, ps = stats.spearmanr(boundary_distance, target)
    print(f"  {target_name}: pearson r={r:+.4f} (p={p:.2e}), spearman rho={rho:+.4f} (p={ps:.2e})")
    results[f"{target_name}_pearson_r"] = float(r)
    results[f"{target_name}_spearman_rho"] = float(rho)

# ---------------------------------------------------------------------
# Save results summary
# ---------------------------------------------------------------------
with open(out_dir / "results_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Pearson r (boundary_distance, evidence)  = {results['pearson_r']:+.4f}")
print(f"Spearman rho (boundary_distance, evidence) = {results['spearman_rho']:+.4f}")
print(f"Best polynomial R^2 (degree 1-4): {max(poly_r2.values()):.4f} "
      f"(degree {max(poly_r2, key=poly_r2.get)})")
r2_best = max(poly_r2.values())
if r2_best > 0.7:
    verdict = "H0 supported: boundary distance explains most of the variance in evidence."
elif r2_best > 0.3:
    verdict = "Partial: boundary distance explains SOME variance, but substantial variance remains unexplained."
else:
    verdict = "H1 supported: boundary distance explains little; evidence encodes something else."
print(f"\nVerdict: {verdict}")
print(f"\nAll outputs saved to {out_dir}")
