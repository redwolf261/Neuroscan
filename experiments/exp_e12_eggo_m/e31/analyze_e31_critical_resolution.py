"""
Phase E31, Section 10: GT-only resolution survival curves and the critical-
resolution object alpha_c = inf{alpha : G_c(alpha) = 0}. Purely geometric,
model-independent. Reuses E30_component_survival_table.json (no new data
collection). No training, no model inference.
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
E30_DIR = BASE.parent / "e30"

with open(E30_DIR / "E30_component_survival_table.json") as f:
    records = json.load(f)
n = len(records)
print(f"Loaded {n} native-space GT components (from E30's own survival table)")

ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]
native_size = np.array([r["native_size"] for r in records])
G_c = {a: np.array([r[f"survival_alpha_{a}"] for r in records]) for a in ALPHAS}  # G_c(alpha) = geometric survival ratio, same quantity as E30's survival_alpha

# ============================================================
# Critical resolution alpha_c = inf{alpha : G_c(alpha) = 0}
# Since alpha is only measured at 5 discrete points, alpha_c is estimated as
# the SMALLEST alpha at which survival first hits exactly 0, with linear
# interpolation between the last-nonzero and first-zero points for a
# continuous estimate. Components that never hit 0 (survive at alpha=1.0)
# get alpha_c = None (right-censored -- critical resolution is beyond 64^3,
# i.e. beyond what we tested).
# ============================================================
alpha_c = np.full(n, np.nan)
alpha_c_censored = np.zeros(n, dtype=bool)  # True = never vanished within tested range

for i in range(n):
    svals = np.array([G_c[a][i] for a in ALPHAS])
    zero_idx = np.where(svals == 0)[0]
    if len(zero_idx) == 0:
        alpha_c_censored[i] = True
        alpha_c[i] = np.nan  # censored -- never vanished at any tested alpha
        continue
    first_zero = zero_idx[0]
    if first_zero == 0:
        # already zero at alpha=0 (160^3) -- shouldn't happen for any real
        # lesion (alpha=0 is the finest grid) but guard anyway
        alpha_c[i] = 0.0
        continue
    # linear interpolation between last nonzero point and first zero point
    a_lo, a_hi = ALPHAS[first_zero - 1], ALPHAS[first_zero]
    s_lo, s_hi = svals[first_zero - 1], svals[first_zero]
    if s_lo == s_hi:
        alpha_c[i] = a_lo
    else:
        frac = s_lo / (s_lo - s_hi)  # fraction of the way from a_lo to a_hi where survival crosses zero (linear interp)
        alpha_c[i] = a_lo + frac * (a_hi - a_lo)

n_censored = alpha_c_censored.sum()
n_estimated = n - n_censored
print(f"\nComponents that vanish within the tested range (alpha_c estimable): {n_estimated}/{n} ({100*n_estimated/n:.1f}%)")
print(f"Components that NEVER vanish (right-censored, alpha_c > 1.0, i.e. survive even at 64^3): {n_censored}/{n} ({100*n_censored/n:.1f}%)")

valid_ac = ~np.isnan(alpha_c)
print(f"\nalpha_c distribution (n={valid_ac.sum()}): mean={alpha_c[valid_ac].mean():.3f} median={np.median(alpha_c[valid_ac]):.3f} SD={alpha_c[valid_ac].std():.3f}")
pct = np.percentile(alpha_c[valid_ac], [5, 25, 50, 75, 95])
print(f"  p5={pct[0]:.3f} p25={pct[1]:.3f} p50={pct[2]:.3f} p75={pct[3]:.3f} p95={pct[4]:.3f}")

# ============================================================
# Is survival a smooth decline, an abrupt phase transition, or non-monotonic?
# Check: for components that vanish, how sharp is the transition (survival
# in the step immediately before vanishing, vs the step before that)?
# ============================================================
print("\n=== Sharpness of the vanishing transition ===")
sharp_drops = []
for i in np.where(valid_ac & ~alpha_c_censored)[0]:
    svals = np.array([G_c[a][i] for a in ALPHAS])
    zero_idx = np.where(svals == 0)[0][0]
    if zero_idx >= 1:
        pre_vanish_survival = svals[zero_idx - 1]
        sharp_drops.append(pre_vanish_survival)
sharp_drops = np.array(sharp_drops)
pct_full_to_zero = float((sharp_drops == 1.0).mean())
pct_near_full = float((sharp_drops >= 0.8).mean())
print(f"  Survival value in the step immediately BEFORE vanishing (n={len(sharp_drops)}): mean={sharp_drops.mean():.4f} median={np.median(sharp_drops):.4f}")
print(f"  Fraction going from FULL (>=1.0x volumetric floor) survival directly to 0 in ONE step: {100*pct_full_to_zero:.1f}%")
print(f"  Fraction with survival >=0.8x floor immediately before vanishing: {100*pct_near_full:.1f}%")
print(f"  --> {'ABRUPT: the majority go from near-full/proportional survival straight to complete vanishing in a single alpha step (consistent with E30 Section 3''s near-binary finding), NOT a gradual multi-step decay' if pct_full_to_zero > 0.4 else 'GRADUAL multi-step decline before vanishing'}")

# ============================================================
# Does alpha_c contain information beyond native size?
# ============================================================
print("\n=== Does alpha_c (critical resolution) contain information beyond native size? ===")
size_valid = native_size[valid_ac & ~alpha_c_censored]
ac_valid = alpha_c[valid_ac & ~alpha_c_censored]
rho, p = stats.spearmanr(size_valid, ac_valid)
print(f"  Spearman(native_size, alpha_c): rho={rho:+.3f} p={p:.4e}  (n={len(size_valid)})")

# alpha_c should be STRONGLY monotonic with size almost by definition (larger
# lesions need MORE degradation to vanish) -- test whether it's PURELY
# determined by size (i.e. does knowing size alone predict alpha_c almost
# perfectly?) using a monotonic/rank-based check since alpha_c is coarsely
# discretized to 4 possible interpolated bands
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)

log_size = np.log(size_valid + 1)
size_cbrt = size_valid ** (1 / 3)
size_features = np.column_stack([zscore(size_valid), zscore(log_size), zscore(size_cbrt)])
Xb = np.column_stack([np.ones(len(ac_valid)), size_features])
coef, _, _, _ = np.linalg.lstsq(Xb, ac_valid, rcond=None)
pred = Xb @ coef
r2 = 1 - np.sum((ac_valid - pred) ** 2) / np.sum((ac_valid - ac_valid.mean()) ** 2)
print(f"  R^2(alpha_c ~ nonlinear size, OLS): {r2:.4f}")
print(f"  --> alpha_c is {'ALMOST ENTIRELY' if r2 > 0.7 else 'NOT purely'} determined by native size alone")

results = {
    "n_components": n,
    "n_censored_never_vanish": int(n_censored),
    "pct_censored": float(100 * n_censored / n),
    "alpha_c_distribution": {
        "n": int(valid_ac.sum()), "mean": float(alpha_c[valid_ac].mean()), "median": float(np.median(alpha_c[valid_ac])),
        "sd": float(alpha_c[valid_ac].std()), "p5": float(pct[0]), "p25": float(pct[1]), "p50": float(pct[2]), "p75": float(pct[3]), "p95": float(pct[4]),
    },
    "transition_sharpness": {
        "n": len(sharp_drops), "mean_pre_vanish_survival": float(sharp_drops.mean()) if len(sharp_drops) else None,
        "median_pre_vanish_survival": float(np.median(sharp_drops)) if len(sharp_drops) else None,
        "pct_full_to_zero_in_one_step": pct_full_to_zero, "pct_near_full_before_vanish": pct_near_full,
    },
    "alpha_c_vs_size": {"spearman_rho": rho, "spearman_p": p, "r2_nonlinear_size": r2},
}
with open(BASE / "E31_critical_resolution_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {BASE / 'E31_critical_resolution_results.json'}")

# Save alpha_c per component for downstream use (E31's other analyses)
alpha_c_table = []
for i, r in enumerate(records):
    alpha_c_table.append({
        "subject_idx": r["subject_idx"], "native_component_id": r["native_component_id"],
        "native_size": r["native_size"], "alpha_c": (None if np.isnan(alpha_c[i]) else float(alpha_c[i])),
        "censored": bool(alpha_c_censored[i]),
    })
with open(BASE / "E31_alpha_c_table.json", "w") as f:
    json.dump(alpha_c_table, f)

if HAVE_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(alpha_c[valid_ac], bins=20, color="#4c72b0", edgecolor="none")
    axes[0].set_xlabel("alpha_c (critical resolution, interpolated)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Distribution of alpha_c (n={valid_ac.sum()}, {n_censored} censored/never-vanish)")

    axes[1].scatter(np.log(size_valid + 1), ac_valid, s=8, alpha=0.4, color="#c44e52")
    axes[1].set_xlabel("log(native size)")
    axes[1].set_ylabel("alpha_c")
    axes[1].set_title("alpha_c vs native size")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "e31_critical_resolution.png", dpi=120)
    plt.close()
    print("Saved figure")
