"""
Phase E1.4: Is uncertainty related to local manifold density in latent space?

E1.1-E1.3 established: image boundary distance R^2=0.150, latent boundary
distance R^2=0.219 -- roughly 78% of evidence variance remains
unexplained. This asks whether local density (how many nearby neighbors
a voxel's dec1 feature vector has in the pooled 32-dim representation
space) is part of the remaining explanation, independent of latent
boundary distance.

Reads the same per_voxel_stats.csv (60,000 voxels, 30 volumes). kNN
density computed over the FULL pooled 32-dim feature space (not
per-volume, not spatial adjacency) -- this measures representation-manifold
density, which is what the hypothesis is about. No new inference.
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
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

base = Path(__file__).parent
data_path = base / "extracted" / "per_voxel_stats.csv"
out_dir = base / "e1_4_results"
out_dir.mkdir(exist_ok=True)

print("Loading per-voxel data...")
with open(data_path, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

feat_cols = [f"dec1_f{i}" for i in range(32)]
Z = np.array([[float(r[c]) for c in feat_cols] for r in rows])
evidence = np.array([float(r["evidence"]) for r in rows])
boundary_distance = np.array([float(r["boundary_distance"]) for r in rows])
correct = np.array([float(r["correct"]) for r in rows]).astype(bool)
ground_truth = np.array([float(r["ground_truth"]) for r in rows]).astype(bool)
feature_norm = np.array([float(r["feature_norm"]) for r in rows])

n = Z.shape[0]
print(f"n = {n} voxels, feature dim = {Z.shape[1]}")

results = {}

# ---------------------------------------------------------------------
# Recompute latent_boundary_distance exactly as in E1.3 (needed for
# partial correlation / multiple regression against density)
# ---------------------------------------------------------------------
print("\nRecomputing latent decision-boundary distance (as in E1.3)...")
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(Z, ground_truth.astype(int))
w = clf.coef_[0]
b = clf.intercept_[0]
latent_boundary_distance = (Z @ w + b) / np.linalg.norm(w)
abs_latent_bd = np.abs(latent_boundary_distance)

# ---------------------------------------------------------------------
# Step 1-2: k-NN density (k=20), rho = mean distance to k nearest
# neighbors (large rho = low density, small rho = high density)
# ---------------------------------------------------------------------
K = 20
print(f"\n[Step 1-2] Computing k-NN density (k={K}) over pooled {n}-voxel feature space...")
nn = NearestNeighbors(n_neighbors=K + 1, algorithm="auto", n_jobs=-1)  # +1 to exclude self
nn.fit(Z)
distances, _ = nn.kneighbors(Z)
rho = distances[:, 1:].mean(axis=1)  # exclude self (distance 0 at index 0)
print(f"  rho (mean dist to {K}-NN) range: [{rho.min():.4f}, {rho.max():.4f}], mean={rho.mean():.4f}")
# density = 1/rho for intuition (large=dense), but all stats below use rho
# directly since it's the more numerically stable quantity (no division-by-
# near-zero risk) and monotonic transforms don't change correlation signs
# for Spearman / change R^2 negligibly for Pearson on well-behaved data --
# reported as "rho" (low=dense) throughout, sign-flipped in interpretation.
density = 1.0 / (rho + 1e-8)

results["density_summary"] = {
    "k": K, "rho_min": float(rho.min()), "rho_max": float(rho.max()), "rho_mean": float(rho.mean()),
}

# ---------------------------------------------------------------------
# Step 3: Density vs evidence -- Pearson, Spearman, polynomial R^2, LOWESS
# ---------------------------------------------------------------------
print("\n[Step 3] rho (inverse density) vs evidence")
r_pearson, p_pearson = stats.pearsonr(rho, evidence)
rho_spearman, p_spearman = stats.spearmanr(rho, evidence)
print(f"  Pearson r = {r_pearson:+.4f} (p={p_pearson:.2e}), Spearman rho = {rho_spearman:+.4f} (p={p_spearman:.2e})")
results["rho_vs_evidence"] = {
    "pearson_r": float(r_pearson), "pearson_p": float(p_pearson),
    "spearman_rho": float(rho_spearman), "spearman_p": float(p_spearman),
}

poly_r2 = {}
for degree in (1, 2, 3, 4):
    coeffs = np.polyfit(rho, evidence, degree)
    y_pred = np.polyval(coeffs, rho)
    r2 = r2_score(evidence, y_pred)
    poly_r2[degree] = float(r2)
    print(f"  polynomial degree {degree}: R^2 = {r2:.4f}")
results["rho_polynomial_r2"] = poly_r2

fig, ax = plt.subplots(figsize=(9, 6))
plot_idx = np.random.RandomState(0).choice(n, size=min(15000, n), replace=False)
ax.scatter(rho[plot_idx], evidence[plot_idx], s=2, alpha=0.15, c="darkred")
ax.set_xlabel(f"rho: mean distance to {K} nearest neighbors (low = dense region)")
ax.set_ylabel("Total evidence")
ax.set_title("Evidence vs. Local Manifold Density (inverse)")
fig.tight_layout()
fig.savefig(out_dir / "01_scatter_evidence_vs_density.png", dpi=150)
plt.close(fig)

lowess_idx = np.random.RandomState(1).choice(n, size=min(8000, n), replace=False)
lowess_result = lowess(evidence[lowess_idx], rho[lowess_idx], frac=0.15, it=1)
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(rho[plot_idx], evidence[plot_idx], s=2, alpha=0.1, c="lightgray", label="voxels")
ax.plot(lowess_result[:, 0], lowess_result[:, 1], color="darkorange", linewidth=2.5, label="LOWESS")
ax.set_xlabel(f"rho: mean distance to {K}-NN (low = dense)")
ax.set_ylabel("Total evidence")
ax.set_title("LOWESS: Evidence vs. Local Density")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "02_lowess_evidence_vs_density.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# Step 4: Correct vs incorrect -- density distributions
# ---------------------------------------------------------------------
print("\n[Step 4] Density: correct vs incorrect voxels")
fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(rho[correct], bins=60, alpha=0.5, density=True, label=f"Correct (n={correct.sum()})", color="seagreen")
ax.hist(rho[~correct], bins=60, alpha=0.5, density=True, label=f"Incorrect (n={(~correct).sum()})", color="crimson")
ax.set_xlabel(f"rho: mean distance to {K}-NN (low = dense)")
ax.set_ylabel("Density")
ax.set_title("Local Manifold Density: Correct vs. Incorrect Voxels")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "03_density_correct_vs_incorrect.png", dpi=150)
plt.close(fig)

t_rho, p_rho = stats.ttest_ind(rho[correct], rho[~correct], equal_var=False)
def cohens_d(a, b):
    pooled_std = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0
d_rho = cohens_d(rho[correct], rho[~correct])
print(f"  mean rho correct={rho[correct].mean():.4f}, incorrect={rho[~correct].mean():.4f}")
print(f"  Welch t-test: t={t_rho:.3f}, p={p_rho:.2e}, Cohen's d={d_rho:.3f}")
print(f"  (compare: E1.1 evidence d=3.09, E1.2 feature_norm d=-1.29, E1.3 latent_boundary d=1.41)")
results["density_correct_vs_incorrect"] = {
    "mean_rho_correct": float(rho[correct].mean()), "mean_rho_incorrect": float(rho[~correct].mean()),
    "t": float(t_rho), "p": float(p_rho), "cohens_d": float(d_rho),
}

# ---------------------------------------------------------------------
# Step 5: Density vs latent boundary distance -- independence check
# ---------------------------------------------------------------------
print("\n[Step 5] rho vs latent_boundary_distance (are they measuring the same thing?)")
r_rho_lbd, p_rho_lbd = stats.pearsonr(rho, abs_latent_bd)
print(f"  Pearson r(rho, |latent_boundary_distance|) = {r_rho_lbd:+.4f} (p={p_rho_lbd:.2e})")
results["rho_vs_latent_boundary"] = {"pearson_r": float(r_rho_lbd), "pearson_p": float(p_rho_lbd)}

def partial_corr(x, y, z):
    def residualize(a, b):
        b2 = b.reshape(-1, 1)
        beta = LinearRegression().fit(b2, a)
        return a - beta.predict(b2)
    rx = residualize(x, z)
    ry = residualize(y, z)
    r, p = stats.pearsonr(rx, ry)
    return r, p

r_partial, p_partial = partial_corr(evidence, rho, abs_latent_bd)
print(f"  Partial corr(evidence, rho | latent_boundary_distance) = {r_partial:+.4f} (p={p_partial:.2e})")
print(f"  (raw, uncontrolled r = {r_pearson:+.4f} for comparison)")
results["partial_correlation_density_given_latent_boundary"] = {"r": float(r_partial), "p": float(p_partial)}

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(abs_latent_bd[plot_idx], rho[plot_idx], s=2, alpha=0.15, c="teal")
ax.set_xlabel("|Latent decision-boundary distance|")
ax.set_ylabel(f"rho: mean distance to {K}-NN (low = dense)")
ax.set_title("Density vs. Latent Boundary Distance")
fig.tight_layout()
fig.savefig(out_dir / "04_density_vs_latent_boundary.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# Step 6: Multiple regression -- evidence ~ boundary + latent_boundary + density
# ---------------------------------------------------------------------
print("\n[Step 6] Multiple regression: Evidence ~ image_boundary + latent_boundary + density")
X1 = boundary_distance.reshape(-1, 1)
X2 = np.column_stack([boundary_distance, abs_latent_bd])
X3 = np.column_stack([boundary_distance, abs_latent_bd, rho])

r2_1 = r2_score(evidence, LinearRegression().fit(X1, evidence).predict(X1))
r2_2 = r2_score(evidence, LinearRegression().fit(X2, evidence).predict(X2))
r2_3 = r2_score(evidence, LinearRegression().fit(X3, evidence).predict(X3))

print(f"  R^2 (image boundary only)                          = {r2_1:.4f}")
print(f"  R^2 (image boundary + latent boundary)             = {r2_2:.4f}")
print(f"  R^2 (image boundary + latent boundary + density)   = {r2_3:.4f}")
print(f"  Incremental R^2 from adding density = {r2_3 - r2_2:+.4f}")
results["multiple_regression"] = {
    "r2_image_boundary_only": float(r2_1),
    "r2_image_plus_latent_boundary": float(r2_2),
    "r2_all_three": float(r2_3),
    "incremental_r2_from_density": float(r2_3 - r2_2),
}

# ---------------------------------------------------------------------
# PCA visualization colored by density
# ---------------------------------------------------------------------
print("\n[Visualization] PCA colored by density")
pca = PCA(n_components=2, random_state=0)
Z_2d = pca.fit_transform(Z[plot_idx])
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(Z_2d[:, 0], Z_2d[:, 1], c=rho[plot_idx], cmap="inferno_r", s=3, alpha=0.5)
plt.colorbar(sc, ax=ax, label=f"rho ({K}-NN mean distance, low=dense)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"dec1 feature space, PCA colored by local density (explained var: {pca.explained_variance_ratio_.sum():.1%})")
fig.tight_layout()
fig.savefig(out_dir / "05_pca_colored_by_density.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# Save + verdict
# ---------------------------------------------------------------------
with open(out_dir / "results_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"rho vs evidence: Pearson r={r_pearson:+.4f}, best polynomial R^2={max(poly_r2.values()):.4f}")
print(f"Correct vs incorrect density gap: Cohen's d={d_rho:.3f}")
print(f"Partial corr(evidence, density | latent_boundary) = {r_partial:+.4f}")
print(f"Incremental R^2 from adding density to (image+latent boundary) model = {r2_3-r2_2:+.4f}")
print(f"Final combined R^2 (image boundary + latent boundary + density) = {r2_3:.4f}")

best_density_r2 = max(poly_r2.values())
incremental = r2_3 - r2_2
if incremental < 0.02:
    verdict = "Case A: density explains almost nothing beyond latent boundary distance. Algorithm should focus on latent boundary geometry alone."
elif incremental < 0.15:
    verdict = "Case B: density contributes a real, moderate independent signal. Algorithm should consider both latent boundary AND density."
else:
    verdict = ("Case C: density is a major independent contributor. Uncertainty isn't just boundary proximity -- "
               "it's boundary proximity AND local sparsity. This changes the E1.3 interpretation substantially.")
print(f"\nVerdict: {verdict}")
print(f"\nAll outputs saved to {out_dir}")
