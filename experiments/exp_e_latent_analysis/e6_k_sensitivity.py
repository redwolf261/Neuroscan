"""
Problem 2 / stability check: does the density effect (correct vs
incorrect separation, Cohen's d) remain stable across different choices
of k in the k-NN density estimate? If the effect only shows up at one
specific k, that's a red flag for an unstable, overfit-to-noise measurement
rather than a robust geometric property.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors

base = Path(__file__).parent
data_path = base / "extracted" / "per_voxel_stats.csv"
out_dir = base / "e6_results"

with open(data_path, newline="") as f:
    rows = list(csv.DictReader(f))

feat_cols = [f"dec1_f{i}" for i in range(32)]
Z = np.array([[float(r[c]) for c in feat_cols] for r in rows])
evidence = np.array([float(r["evidence"]) for r in rows])
correct = np.array([float(r["correct"]) for r in rows]).astype(bool)

n = Z.shape[0]


def cohens_d(a, b):
    pooled_std = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


print(f"Fitting single NearestNeighbors index (max k=80) once, reusing for all k values...")
nn = NearestNeighbors(n_neighbors=81, n_jobs=-1)
nn.fit(Z)
all_distances, _ = nn.kneighbors(Z)  # (n, 81), sorted ascending, col 0 is self (dist 0)

results = {}
print(f"\n{'k':<6}{'mean_rho_correct':<20}{'mean_rho_incorrect':<22}{'cohens_d':<12}{'pearson_r(rho,evidence)':<26}")
for k in (5, 10, 20, 40, 80):
    rho_k = all_distances[:, 1:k+1].mean(axis=1)
    d_k = cohens_d(rho_k[correct], rho_k[~correct])
    r_k, p_k = stats.pearsonr(rho_k, evidence)
    print(f"{k:<6}{rho_k[correct].mean():<20.4f}{rho_k[~correct].mean():<22.4f}{d_k:<12.4f}{r_k:<26.4f}")
    results[f"k={k}"] = {
        "mean_rho_correct": float(rho_k[correct].mean()),
        "mean_rho_incorrect": float(rho_k[~correct].mean()),
        "cohens_d": float(d_k),
        "pearson_r_evidence": float(r_k),
        "pearson_p_evidence": float(p_k),
    }

with open(out_dir / "k_sensitivity_results.json", "w") as f:
    json.dump(results, f, indent=2)

ds = [results[f"k={k}"]["cohens_d"] for k in (5, 10, 20, 40, 80)]
print(f"\nCohen's d range across k=5..80: [{min(ds):.3f}, {max(ds):.3f}]")
print(f"Relative variation: {(max(ds)-min(ds))/np.mean(np.abs(ds))*100:.1f}% of mean magnitude")
if max(ds) - min(ds) < 0.5 * np.mean(np.abs(ds)):
    print("STABLE: effect size does not vary much across k choices.")
else:
    print("UNSTABLE: effect size is sensitive to the choice of k -- flag as a real concern.")
