"""
Phase E6: Stress-test the EGGO hypothesis before implementing anything.

Problem 1: multivariate regression -- does density survive controlling
  for boundary distance, evidence, and feature norm SIMULTANEOUSLY (not
  pairwise partial correlation, a full joint model)?
Problem 3: four-group breakdown -- correct/sparse, correct/dense,
  incorrect/sparse, incorrect/dense. Averages can hide exceptions.
Counterfactual analysis: explicitly pull out the exception cases that
  would distinguish hypotheses A/B/C/D:
  - sparse but correct (contradicts "sparse=bad")
  - dense but incorrect (contradicts "dense=safe")
  - near latent boundary but high evidence (contradicts "boundary=uncertain")
  - far from latent boundary but low evidence (contradicts "far=confident")

Reuses per_voxel_stats.csv. Recomputes latent_boundary_distance (E1.3
method) and rho (E1.4 method, k=20) since neither was persisted per-voxel
in prior runs.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

base = Path(__file__).parent
data_path = base / "extracted" / "per_voxel_stats.csv"
out_dir = base / "e6_results"
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
subject_id = np.array([r["subject_id"] for r in rows])

n = Z.shape[0]
print(f"n = {n} voxels")

results = {}

# Recompute latent boundary distance (E1.3 method)
print("\nRecomputing latent boundary distance...")
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(Z, ground_truth.astype(int))
w = clf.coef_[0]
b = clf.intercept_[0]
latent_boundary_distance = (Z @ w + b) / np.linalg.norm(w)
abs_latent_bd = np.abs(latent_boundary_distance)

# Recompute density (E1.4 method, k=20)
print("Recomputing density (k=20)...")
nn = NearestNeighbors(n_neighbors=21, n_jobs=-1)
nn.fit(Z)
distances, _ = nn.kneighbors(Z)
rho = distances[:, 1:].mean(axis=1)

# ======================================================================
# PROBLEM 1: Full multivariate regression -- does density survive?
# ======================================================================
print("\n" + "=" * 70)
print("PROBLEM 1: Multivariate regression, error/evidence ~ all 4 predictors jointly")
print("=" * 70)

# Standardize predictors so coefficients are comparable (beta weights)
X_raw = np.column_stack([np.abs(boundary_distance), abs_latent_bd, rho, feature_norm])
predictor_names = ["image_boundary_dist", "latent_boundary_dist", "density_rho", "feature_norm"]
scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)

# Target 1: evidence (continuous) -- standardized joint regression
y_evidence_std = (evidence - evidence.mean()) / evidence.std()
lr_evidence = LinearRegression().fit(X_std, y_evidence_std)
print("\n-- Target: evidence (standardized coefficients / 'beta weights') --")
for name, coef in zip(predictor_names, lr_evidence.coef_):
    print(f"  beta_{name:24s} = {coef:+.4f}")
r2_joint_evidence = lr_evidence.score(X_std, y_evidence_std)
print(f"  Joint R^2 = {r2_joint_evidence:.4f}")

results["multivariate_evidence"] = {
    "coefficients": {name: float(c) for name, c in zip(predictor_names, lr_evidence.coef_)},
    "r2": float(r2_joint_evidence),
}

# Target 2: correctness (binary) -- logistic regression, standardized
incorrect_int = (~correct).astype(int)
lr_error = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_std, incorrect_int)
print("\n-- Target: P(incorrect) (standardized logistic coefficients) --")
for name, coef in zip(predictor_names, lr_error.coef_[0]):
    print(f"  beta_{name:24s} = {coef:+.4f}")

# Also report which predictor's beta is largest in magnitude (most
# independently predictive after accounting for the others)
abs_betas_evidence = {name: abs(c) for name, c in zip(predictor_names, lr_evidence.coef_)}
abs_betas_error = {name: abs(c) for name, c in zip(predictor_names, lr_error.coef_[0])}
print(f"\n  Ranked by |beta| for evidence: {sorted(abs_betas_evidence.items(), key=lambda x: -x[1])}")
print(f"  Ranked by |beta| for error:    {sorted(abs_betas_error.items(), key=lambda x: -x[1])}")

results["multivariate_error"] = {
    "coefficients": {name: float(c) for name, c in zip(predictor_names, lr_error.coef_[0])},
}

# Does density's beta vanish (Problem 1's core question)?
density_beta_evidence = abs_betas_evidence["density_rho"]
density_rank_evidence = sorted(abs_betas_evidence.items(), key=lambda x: -x[1]).index(("density_rho", density_beta_evidence)) + 1
print(f"\n  VERDICT: density's |beta| for evidence = {density_beta_evidence:.4f}, "
      f"rank {density_rank_evidence}/4 among predictors")
if density_beta_evidence < 0.05:
    print("  -> Density's independent contribution is near-zero once other predictors are controlled. "
          "SUPPORTS Problem 1's confound concern (density may be a proxy, not causal).")
else:
    print("  -> Density retains meaningful independent signal even in a full joint model.")

# ======================================================================
# PROBLEM 3: Four-group breakdown
# ======================================================================
print("\n" + "=" * 70)
print("PROBLEM 3: Four-group breakdown (correct/incorrect x sparse/dense)")
print("=" * 70)

rho_median = np.median(rho)
sparse = rho > rho_median
dense = ~sparse

groups = {
    "correct_dense": correct & dense,
    "correct_sparse": correct & sparse,
    "incorrect_dense": (~correct) & dense,
    "incorrect_sparse": (~correct) & sparse,
}
group_stats = {}
for name, mask in groups.items():
    cnt = mask.sum()
    mean_ev = evidence[mask].mean() if cnt > 0 else float("nan")
    mean_rho = rho[mask].mean() if cnt > 0 else float("nan")
    print(f"  {name:20s}: n={cnt:6d} ({100*cnt/n:.2f}%)  mean_evidence={mean_ev:.3f}  mean_rho={mean_rho:.3f}")
    group_stats[name] = {"n": int(cnt), "mean_evidence": float(mean_ev), "mean_rho": float(mean_rho)}
results["four_group_breakdown"] = group_stats

# Key question: how many "incorrect but dense" voxels exist? (the case
# that would contradict "sparse=bad, dense=safe")
n_incorrect_dense = groups["incorrect_dense"].sum()
n_incorrect_total = (~correct).sum()
print(f"\n  'Incorrect but dense' voxels: {n_incorrect_dense}/{n_incorrect_total} "
      f"({100*n_incorrect_dense/max(1,n_incorrect_total):.1f}% of all incorrect voxels)")

# ======================================================================
# COUNTEREXAMPLE / EXCEPTION ANALYSIS
# ======================================================================
print("\n" + "=" * 70)
print("COUNTEREXAMPLE ANALYSIS: the exception cases")
print("=" * 70)

rho_p90 = np.percentile(rho, 90)
rho_p10 = np.percentile(rho, 10)
evidence_p90 = np.percentile(evidence, 90)
evidence_p10 = np.percentile(evidence, 10)
bd_p90 = np.percentile(abs_latent_bd, 90)
bd_p10 = np.percentile(abs_latent_bd, 10)

exceptions = {}

# 1. Sparse but correct (rho > p90, correct) -- contradicts "sparse=bad"
mask = (rho > rho_p90) & correct
exceptions["sparse_but_correct"] = mask
print(f"\n1. Sparse (top 10% rho) but CORRECT: n={mask.sum()} "
      f"({100*mask.sum()/(rho>rho_p90).sum():.1f}% of all sparse voxels)")

# 2. Dense but incorrect (rho < p10, incorrect) -- contradicts "dense=safe"
mask = (rho < rho_p10) & (~correct)
exceptions["dense_but_incorrect"] = mask
print(f"2. Dense (bottom 10% rho) but INCORRECT: n={mask.sum()} "
      f"({100*mask.sum()/max(1,(~correct).sum()):.1f}% of all incorrect voxels)")

# 3. Near latent boundary but HIGH evidence -- contradicts "boundary=uncertain"
mask = (abs_latent_bd < bd_p10) & (evidence > evidence_p90)
exceptions["near_boundary_high_evidence"] = mask
print(f"3. Near latent boundary (bottom 10%) but HIGH evidence (top 10%): n={mask.sum()} "
      f"({100*mask.sum()/(abs_latent_bd<bd_p10).sum():.1f}% of near-boundary voxels)")

# 4. Far from latent boundary but LOW evidence -- contradicts "far=confident"
mask = (abs_latent_bd > bd_p90) & (evidence < evidence_p10)
exceptions["far_boundary_low_evidence"] = mask
print(f"4. Far from latent boundary (top 10%) but LOW evidence (bottom 10%): n={mask.sum()} "
      f"({100*mask.sum()/(abs_latent_bd>bd_p90).sum():.1f}% of far-boundary voxels)")

results["exceptions"] = {k: int(v.sum()) for k, v in exceptions.items()}

# Save a sample of exception voxels (subject_id + key stats) for manual inspection
exception_sample = {}
for name, mask in exceptions.items():
    idx = np.where(mask)[0]
    sample_idx = idx[:20]  # first 20 for inspection
    exception_sample[name] = [
        {
            "subject_id": str(subject_id[i]),
            "evidence": float(evidence[i]),
            "rho": float(rho[i]),
            "abs_latent_bd": float(abs_latent_bd[i]),
            "correct": bool(correct[i]),
            "ground_truth_tumor": bool(ground_truth[i]),
        }
        for i in sample_idx
    ]
with open(out_dir / "exception_samples.json", "w") as f:
    json.dump(exception_sample, f, indent=2)

# ======================================================================
# Save + verdict
# ======================================================================
with open(out_dir / "results_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("OVERALL SUMMARY")
print("=" * 70)
print(f"Density's standardized beta (evidence, joint model): {density_beta_evidence:.4f} (rank {density_rank_evidence}/4)")
print(f"Incorrect-but-dense voxels: {n_incorrect_dense}/{n_incorrect_total} "
      f"({100*n_incorrect_dense/max(1,n_incorrect_total):.1f}%)")
print(f"Exception counts: {results['exceptions']}")
print(f"\nAll outputs saved to {out_dir}")
