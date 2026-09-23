"""
Phase E1.3: How are tumor and background organized geometrically in the
latent (dec1) space, and does uncertainty track proximity to the latent
decision boundary rather than (or in addition to) the image boundary?

Follows directly from E1.2's finding: tumor and background voxels have
OPPOSITE-SIGNED relationships between feature norm and evidence (tumor
r=+0.91, background r=-0.68), suggesting two differently-organized
representations, not one shared uncertainty manifold.

Questions (per the redirected E1.3 scope):
  1. Are tumor features clustered? Are background features clustered?
  2. Where do incorrect voxels lie relative to those clusters?
  3. Do low-evidence voxels lie between clusters (near a latent decision
     boundary) rather than deep inside one?
  4. Does uncertainty increase near the LATENT decision boundary more
     than it does near the IMAGE boundary (E1.1's boundary_distance)?

Reads the same per_voxel_stats.csv (60,000 voxels, 32-dim dec1 features)
-- no new inference. Uses the full 32-dim feature space for the actual
distance/clustering computations; PCA/t-SNE are for visualization only,
never for the numeric claims.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score

base = Path(__file__).parent
data_path = base / "extracted" / "per_voxel_stats.csv"
out_dir = base / "e1_3_results"
out_dir.mkdir(exist_ok=True)

print("Loading per-voxel data...")
with open(data_path, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

feat_cols = [f"dec1_f{i}" for i in range(32)]
Z = np.array([[float(r[c]) for c in feat_cols] for r in rows])  # (n, 32)
evidence = np.array([float(r["evidence"]) for r in rows])
boundary_distance = np.array([float(r["boundary_distance"]) for r in rows])
correct = np.array([float(r["correct"]) for r in rows]).astype(bool)
ground_truth = np.array([float(r["ground_truth"]) for r in rows]).astype(bool)
feature_norm = np.array([float(r["feature_norm"]) for r in rows])

n = Z.shape[0]
print(f"n = {n} voxels, feature dim = {Z.shape[1]}, "
      f"{ground_truth.sum()} tumor ({100*ground_truth.sum()/n:.2f}%), "
      f"{(~ground_truth).sum()} background")

results = {}

# ---------------------------------------------------------------------
# Q1: Cluster structure within each class -- are tumor / background
# features themselves tightly clustered, or diffuse?
# ---------------------------------------------------------------------
print("\n[Q1] Within-class cluster tightness (silhouette-free proxy: "
      "mean distance to own-class centroid vs. mean distance to other-class centroid)")

tumor_Z = Z[ground_truth]
bg_Z = Z[~ground_truth]
tumor_centroid = tumor_Z.mean(axis=0)
bg_centroid = bg_Z.mean(axis=0)

dist_tumor_to_own = np.linalg.norm(tumor_Z - tumor_centroid, axis=1)
dist_bg_to_own = np.linalg.norm(bg_Z - bg_centroid, axis=1)
dist_tumor_to_other = np.linalg.norm(tumor_Z - bg_centroid, axis=1)
dist_bg_to_other = np.linalg.norm(bg_Z - tumor_centroid, axis=1)

print(f"  Tumor voxels: mean dist to OWN centroid = {dist_tumor_to_own.mean():.3f} "
      f"(std {dist_tumor_to_own.std():.3f}), to OTHER (bg) centroid = {dist_tumor_to_other.mean():.3f}")
print(f"  Background voxels: mean dist to OWN centroid = {dist_bg_to_own.mean():.3f} "
      f"(std {dist_bg_to_own.std():.3f}), to OTHER (tumor) centroid = {dist_bg_to_other.mean():.3f}")
centroid_separation = np.linalg.norm(tumor_centroid - bg_centroid)
print(f"  Distance between the two class centroids: {centroid_separation:.3f}")

results["cluster_tightness"] = {
    "tumor_mean_dist_to_own_centroid": float(dist_tumor_to_own.mean()),
    "tumor_std_dist_to_own_centroid": float(dist_tumor_to_own.std()),
    "tumor_mean_dist_to_other_centroid": float(dist_tumor_to_other.mean()),
    "background_mean_dist_to_own_centroid": float(dist_bg_to_own.mean()),
    "background_std_dist_to_own_centroid": float(dist_bg_to_own.std()),
    "background_mean_dist_to_other_centroid": float(dist_bg_to_other.mean()),
    "centroid_separation": float(centroid_separation),
}

# ---------------------------------------------------------------------
# Q2/Q3: The KEY metric -- "latent decision boundary distance".
# Fit a linear classifier (logistic regression) tumor-vs-background in
# the 32-dim feature space. Its signed decision function IS a genuine
# latent decision boundary distance (positive = confidently tumor-side,
# negative = confidently background-side, near-zero = near the boundary
# the classifier itself draws). This is the direct latent analogue of
# E1.1's IMAGE boundary_distance.
# ---------------------------------------------------------------------
print("\n[Q2/Q3] Fitting linear classifier for latent decision boundary distance")
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(Z, ground_truth.astype(int))
train_auc = roc_auc_score(ground_truth.astype(int), clf.predict_proba(Z)[:, 1])
print(f"  Linear separability of tumor vs background in dec1 space: AUC = {train_auc:.4f}")
results["latent_classifier_auc"] = float(train_auc)

# Signed distance to the decision hyperplane, in units of ||w|| (standard
# formula: (w.x + b) / ||w||). This is 0 exactly ON the latent boundary.
w = clf.coef_[0]
b = clf.intercept_[0]
latent_boundary_distance = (Z @ w + b) / np.linalg.norm(w)
print(f"  latent_boundary_distance range: [{latent_boundary_distance.min():.3f}, {latent_boundary_distance.max():.3f}]")

# ---------------------------------------------------------------------
# Q3 continued: does evidence track LATENT boundary distance better than
# it tracks IMAGE boundary distance (E1.1's result was R^2=0.150)?
# ---------------------------------------------------------------------
print("\n[Q3] Evidence vs. latent decision boundary distance")
r_latent, p_latent = stats.pearsonr(latent_boundary_distance, evidence)
rho_latent, ps_latent = stats.spearmanr(latent_boundary_distance, evidence)
print(f"  Pearson r = {r_latent:+.4f} (p={p_latent:.2e}), Spearman rho = {rho_latent:+.4f} (p={ps_latent:.2e})")

# Use |latent_boundary_distance| -- proximity to the boundary regardless
# of side -- since the H1 claim is "near the latent boundary = uncertain",
# which is symmetric, not "which side" (that's what the sign captures
# separately, tested above).
abs_latent_bd = np.abs(latent_boundary_distance)
r_abs_latent, p_abs_latent = stats.pearsonr(abs_latent_bd, evidence)
rho_abs_latent, ps_abs_latent = stats.spearmanr(abs_latent_bd, evidence)
print(f"  |latent boundary distance| vs evidence: Pearson r = {r_abs_latent:+.4f} (p={p_abs_latent:.2e}), "
      f"Spearman rho = {rho_abs_latent:+.4f} (p={ps_abs_latent:.2e})")

poly_r2_latent = {}
for degree in (1, 2, 3, 4):
    coeffs = np.polyfit(abs_latent_bd, evidence, degree)
    y_pred = np.polyval(coeffs, abs_latent_bd)
    r2 = r2_score(evidence, y_pred)
    poly_r2_latent[degree] = float(r2)
    print(f"  |latent boundary dist| polynomial degree {degree}: R^2 = {r2:.4f}")

results["latent_boundary_distance"] = {
    "signed_pearson_r": float(r_latent), "signed_pearson_p": float(p_latent),
    "signed_spearman_rho": float(rho_latent), "signed_spearman_p": float(ps_latent),
    "abs_pearson_r": float(r_abs_latent), "abs_pearson_p": float(p_abs_latent),
    "abs_spearman_rho": float(rho_abs_latent), "abs_spearman_p": float(ps_abs_latent),
    "abs_polynomial_r2": poly_r2_latent,
}

print(f"\n  COMPARISON: IMAGE boundary distance explained R^2=0.150 (E1.1)")
print(f"              LATENT boundary distance explains R^2={max(poly_r2_latent.values()):.4f} (this analysis)")

fig, ax = plt.subplots(figsize=(9, 6))
plot_idx = np.random.RandomState(0).choice(n, size=min(15000, n), replace=False)
sc = ax.scatter(latent_boundary_distance[plot_idx], evidence[plot_idx], s=2, alpha=0.15,
                 c=ground_truth[plot_idx].astype(int), cmap="coolwarm")
ax.axvline(0, color="black", linestyle="--", linewidth=1, label="latent decision boundary")
ax.set_xlabel("Signed latent decision-boundary distance (in ||w|| units)")
ax.set_ylabel("Total evidence")
ax.set_title("Evidence vs. Latent Decision-Boundary Distance\n(blue=background, red=tumor)")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "01_evidence_vs_latent_boundary.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# Q2: Where do INCORRECT voxels lie? Near the latent boundary (as H1
# from the redirected E1.3 predicts) or scattered / deep inside a class?
# ---------------------------------------------------------------------
print("\n[Q2] Incorrect voxels: distance to latent decision boundary")
print(f"  Correct voxels:   mean |latent_boundary_distance| = {abs_latent_bd[correct].mean():.4f} "
      f"(n={correct.sum()})")
print(f"  Incorrect voxels: mean |latent_boundary_distance| = {abs_latent_bd[~correct].mean():.4f} "
      f"(n={(~correct).sum()})")
t_lbd, p_lbd = stats.ttest_ind(abs_latent_bd[correct], abs_latent_bd[~correct], equal_var=False)
print(f"  Welch t-test: t={t_lbd:.3f}, p={p_lbd:.2e}")

def cohens_d(a, b):
    pooled_std = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0

d_lbd = cohens_d(abs_latent_bd[correct], abs_latent_bd[~correct])
print(f"  Cohen's d = {d_lbd:.3f}  (compare: E1.1 evidence d=3.09, E1.2 feature_norm d=-1.29)")
results["incorrect_voxels_latent_boundary"] = {
    "mean_abs_dist_correct": float(abs_latent_bd[correct].mean()),
    "mean_abs_dist_incorrect": float(abs_latent_bd[~correct].mean()),
    "t": float(t_lbd), "p": float(p_lbd), "cohens_d": float(d_lbd),
}

fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(abs_latent_bd[correct], bins=60, alpha=0.5, density=True, label=f"Correct (n={correct.sum()})", color="seagreen")
ax.hist(abs_latent_bd[~correct], bins=60, alpha=0.5, density=True, label=f"Incorrect (n={(~correct).sum()})", color="crimson")
ax.set_xlabel("|Latent decision-boundary distance|")
ax.set_ylabel("Density")
ax.set_title("Distance to Latent Decision Boundary: Correct vs. Incorrect")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "02_latent_boundary_correct_vs_incorrect.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# Direct horse race: latent boundary distance vs image boundary distance
# vs feature norm, as predictors of (a) evidence, (b) correctness.
# ---------------------------------------------------------------------
print("\n[Horse race] Which predictor best separates correct/incorrect: "
      "image boundary, feature norm, or latent boundary distance?")
from sklearn.metrics import roc_auc_score as auc_fn
predictors = {
    "abs_image_boundary_distance": np.abs(boundary_distance),
    "feature_norm": feature_norm,
    "abs_latent_boundary_distance": abs_latent_bd,
}
auc_results = {}
incorrect_int = (~correct).astype(int)
for name, pred in predictors.items():
    # incorrect voxels should have LOWER |boundary| or appropriate direction;
    # test both directions and report the better one (AUC is direction-free
    # if we allow 1-AUC, so just report max(auc, 1-auc) as "separability")
    auc = auc_fn(incorrect_int, -pred)  # predict "incorrect" via low value of predictor
    auc = max(auc, 1 - auc)
    auc_results[name] = float(auc)
    print(f"  {name:32s}: AUC (separating correct/incorrect) = {auc:.4f}")
# evidence itself, for reference
auc_evidence = auc_fn(incorrect_int, -evidence)
auc_evidence = max(auc_evidence, 1 - auc_evidence)
print(f"  {'evidence (reference, E1.1)':32s}: AUC (separating correct/incorrect) = {auc_evidence:.4f}")
auc_results["evidence_reference"] = float(auc_evidence)
results["horse_race_auc_vs_correctness"] = auc_results

# ---------------------------------------------------------------------
# PCA visualization (2D, for the figure only -- not used for any numeric claim)
# ---------------------------------------------------------------------
print("\n[Visualization] PCA projection of dec1 features (2D, class + evidence + correctness)")
pca = PCA(n_components=2, random_state=0)
Z_2d = pca.fit_transform(Z[plot_idx])
print(f"  PCA explained variance ratio (2 components): {pca.explained_variance_ratio_}")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sc0 = axes[0].scatter(Z_2d[:, 0], Z_2d[:, 1], c=ground_truth[plot_idx].astype(int), cmap="coolwarm", s=3, alpha=0.4)
axes[0].set_title("PCA colored by class (blue=bg, red=tumor)")
sc1 = axes[1].scatter(Z_2d[:, 0], Z_2d[:, 1], c=evidence[plot_idx], cmap="viridis", s=3, alpha=0.4)
plt.colorbar(sc1, ax=axes[1], label="evidence")
axes[1].set_title("PCA colored by evidence")
incorrect_plot = ~correct[plot_idx]
axes[2].scatter(Z_2d[~incorrect_plot, 0], Z_2d[~incorrect_plot, 1], c="lightgray", s=3, alpha=0.3, label="correct")
axes[2].scatter(Z_2d[incorrect_plot, 0], Z_2d[incorrect_plot, 1], c="red", s=15, alpha=0.8, label="incorrect")
axes[2].set_title("PCA: incorrect voxels highlighted")
axes[2].legend()
for ax in axes:
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
fig.suptitle(f"dec1 feature space, 2D PCA (explained var: {pca.explained_variance_ratio_.sum():.1%})")
fig.tight_layout()
fig.savefig(out_dir / "03_pca_visualization.png", dpi=150)
plt.close(fig)
results["pca_explained_variance_ratio_2d"] = pca.explained_variance_ratio_.tolist()

# ---------------------------------------------------------------------
# Save + verdict
# ---------------------------------------------------------------------
with open(out_dir / "results_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Class separability in dec1 space (linear classifier AUC): {train_auc:.4f}")
print(f"Class centroid separation: {centroid_separation:.3f} "
      f"(tumor own-centroid spread: {dist_tumor_to_own.mean():.3f}, "
      f"bg own-centroid spread: {dist_bg_to_own.mean():.3f})")
print(f"Evidence vs |latent boundary distance|: best poly R^2 = {max(poly_r2_latent.values()):.4f} "
      f"(vs. image boundary R^2=0.150 from E1.1)")
print(f"Incorrect voxels closer to latent boundary: Cohen's d = {d_lbd:.3f}")
print(f"\nAUC separability of correct/incorrect voxels:")
for name, auc in auc_results.items():
    print(f"  {name:32s}: {auc:.4f}")
print(f"\nAll outputs saved to {out_dir}")
