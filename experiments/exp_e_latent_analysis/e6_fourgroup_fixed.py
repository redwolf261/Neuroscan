"""
Fix for E6 Problem 3: the median-split four-group breakdown was
uninformative (median rho splits the data 50/50 among a population where
99.86% of voxels are correct, so it just reproduces the overall
correct/incorrect base rate in both buckets, and puts zero incorrect
voxels in the 'dense' bucket by construction, not by finding).

This uses a threshold matched to E1.4's own measured crossover
(geometric mean of correct-mean and incorrect-mean rho, same tau_rho
construction as PHASE_E5_ALGORITHM_DESIGN.md section 2.3) so the
sparse/dense split is actually diagnostic of the phenomenon being tested,
not an arbitrary population split.
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

base = Path(__file__).parent
data_path = base / "extracted" / "per_voxel_stats.csv"
out_dir = base / "e6_results"

with open(data_path, newline="") as f:
    rows = list(csv.DictReader(f))

feat_cols = [f"dec1_f{i}" for i in range(32)]
Z = np.array([[float(r[c]) for c in feat_cols] for r in rows])
evidence = np.array([float(r["evidence"]) for r in rows])
correct = np.array([float(r["correct"]) for r in rows]).astype(bool)
ground_truth = np.array([float(r["ground_truth"]) for r in rows]).astype(bool)

n = Z.shape[0]

nn = NearestNeighbors(n_neighbors=21, n_jobs=-1)
nn.fit(Z)
distances, _ = nn.kneighbors(Z)
rho = distances[:, 1:].mean(axis=1)

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(Z, ground_truth.astype(int))
w = clf.coef_[0]
b = clf.intercept_[0]
latent_boundary_distance = (Z @ w + b) / np.linalg.norm(w)
abs_latent_bd = np.abs(latent_boundary_distance)

# tau_rho from PHASE_E5 (geometric mean of measured correct/incorrect means)
mean_rho_correct = rho[correct].mean()
mean_rho_incorrect = rho[~correct].mean()
tau_rho = np.sqrt(mean_rho_correct * mean_rho_incorrect)
print(f"mean_rho_correct={mean_rho_correct:.4f}, mean_rho_incorrect={mean_rho_incorrect:.4f}, "
      f"tau_rho (threshold)={tau_rho:.4f}")

sparse = rho > tau_rho
dense = ~sparse
print(f"\nUsing tau_rho={tau_rho:.4f} as sparse/dense threshold "
      f"(vs. median which was uninformative): sparse n={sparse.sum()} ({100*sparse.sum()/n:.2f}%), "
      f"dense n={dense.sum()} ({100*dense.sum()/n:.2f}%)")

groups = {
    "correct_dense": correct & dense,
    "correct_sparse": correct & sparse,
    "incorrect_dense": (~correct) & dense,
    "incorrect_sparse": (~correct) & sparse,
}
print("\n--- Four-group breakdown with DIAGNOSTIC threshold ---")
group_stats = {}
for name, mask in groups.items():
    cnt = mask.sum()
    mean_ev = evidence[mask].mean() if cnt > 0 else float("nan")
    mean_rho_g = rho[mask].mean() if cnt > 0 else float("nan")
    print(f"  {name:20s}: n={cnt:6d} ({100*cnt/n:.3f}%)  mean_evidence={mean_ev:.3f}  mean_rho={mean_rho_g:.3f}")
    group_stats[name] = {"n": int(cnt), "mean_evidence": float(mean_ev), "mean_rho": float(mean_rho_g)}

n_incorrect_dense = groups["incorrect_dense"].sum()
n_incorrect_sparse = groups["incorrect_sparse"].sum()
n_incorrect_total = (~correct).sum()
print(f"\nOf {n_incorrect_total} incorrect voxels: {n_incorrect_sparse} ({100*n_incorrect_sparse/n_incorrect_total:.1f}%) "
      f"are sparse, {n_incorrect_dense} ({100*n_incorrect_dense/n_incorrect_total:.1f}%) are dense")

n_sparse_total = sparse.sum()
n_correct_sparse = groups["correct_sparse"].sum()
print(f"Of {n_sparse_total} sparse voxels: {n_correct_sparse} ({100*n_correct_sparse/n_sparse_total:.1f}%) are correct "
      f"(i.e. sparsity's false-positive rate for predicting error, at this threshold)")

with open(out_dir / "fourgroup_fixed_results.json", "w") as f:
    json.dump({
        "tau_rho": float(tau_rho),
        "groups": group_stats,
        "pct_incorrect_that_are_sparse": float(100*n_incorrect_sparse/n_incorrect_total),
        "pct_sparse_that_are_correct": float(100*n_correct_sparse/n_sparse_total),
    }, f, indent=2)
