"""
Phase E34, Section 2-3: define and calibrate the size-control weighting
w_c^size = (V_c + eps)^(-gamma) [native GT volume, per E33's identified
closest prior art -- Component-Adaptive Tversky] to match E[w] and Var(w)
of the alpha_c weighting w_c^alpha = 1 + kappa*(1-alpha_c)_+, using the
TRAINING SET's own components (matching what will actually be used during
training, not the validation set).

Also performs Section 3's required triviality check: Corr(w^alpha, V_c) and
distribution comparison, reported honestly before any training is launched.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats, optimize

BASE = Path(__file__).parent

with open(BASE / "E34_alpha_c_train_table.json") as f:
    records = json.load(f)
n = len(records)
print(f"Loaded {n} native-space GT components (training set)")

native_size = np.array([r["native_size"] for r in records])
size_64 = np.array([r["size_64"] for r in records])
alpha_c_raw = [r["alpha_c"] for r in records]
censored = np.array([r["censored"] for r in records])
alpha_c = np.array([1.25 if v is None else v for v in alpha_c_raw])

# ============================================================
# alpha_c weighting: w_c^alpha = 1 + kappa*(1-alpha_c)_+
# kappa chosen so the weight distribution has a reasonable, pre-declared
# dynamic range -- NOT tuned to any downstream Dice outcome. Fixed choice:
# kappa=3, giving a max weight of 1+3*1=4 for alpha_c=0 (immediately
# vanishing components) down to a floor of 1.0 for alpha_c>=1 (components
# that survive even at 64^3) -- a modest, bounded emphasis range, decided
# BEFORE looking at how it affects Dice.
# ============================================================
KAPPA = 3.0
w_alpha = 1 + KAPPA * np.maximum(0, 1 - alpha_c)
print(f"\n=== alpha_c weighting (kappa={KAPPA}) ===")
print(f"  E[w_alpha]={w_alpha.mean():.4f}  Var[w_alpha]={w_alpha.var():.4f}  range=[{w_alpha.min():.3f}, {w_alpha.max():.3f}]")

# ============================================================
# Size weighting: w_c^size = (V_c + eps)^(-gamma), calibrate gamma so
# E[w_size] approx E[w_alpha] via a 1D root-find (monotonic in gamma).
# ============================================================
EPS_SIZE = 1.0  # native voxel counts are integers >=1, eps=1.0 keeps units sane

def mean_w_size(gamma):
    w = (native_size + EPS_SIZE) ** (-gamma)
    # rescale to have the SAME units/floor as w_alpha (both should have a
    # comparable operating range) -- normalize w_size to [1, max_ratio] by
    # an affine rescale so its own minimum lands at 1.0 (matching w_alpha's
    # floor of 1.0 for fully-recoverable/large components), then scale so
    # E[w] can be matched by adjusting gamma alone.
    w_norm = w / w.min()  # smallest raw weight (largest component) -> 1.0
    return w_norm.mean(), w_norm

target_mean = w_alpha.mean()

def objective(gamma):
    m, _ = mean_w_size(gamma)
    return m - target_mean

# bracket search for gamma in a reasonable range
gammas_to_try = np.linspace(0.01, 2.0, 200)
means = [mean_w_size(g)[0] for g in gammas_to_try]
means = np.array(means)
# find gamma where mean crosses target_mean
idx = np.argmin(np.abs(means - target_mean))
gamma0 = gammas_to_try[idx]
try:
    gamma_calibrated = optimize.brentq(objective, max(0.001, gamma0 - 0.5), gamma0 + 0.5)
except ValueError:
    gamma_calibrated = gamma0
    print(f"  WARNING: brentq bracket failed, using grid-search nearest gamma={gamma0:.4f}")

w_size_mean, w_size = mean_w_size(gamma_calibrated)
print(f"\n=== Size weighting (Component-Adaptive Tversky formula: w=(V+eps)^-gamma, normalized) ===")
print(f"  Calibrated gamma={gamma_calibrated:.4f}")
print(f"  E[w_size]={w_size.mean():.4f}  Var[w_size]={w_size.var():.4f}  range=[{w_size.min():.3f}, {w_size.max():.3f}]")
print(f"  Target E[w_alpha]={target_mean:.4f} -- match quality: {'GOOD' if abs(w_size.mean()-target_mean)<0.05 else 'POOR, investigate'}")
print(f"  Var match: w_alpha Var={w_alpha.var():.4f} vs w_size Var={w_size.var():.4f} "
      f"(ratio={w_size.var()/w_alpha.var():.3f}, 1.0=perfect match)")

# ============================================================
# DISCLOSED LIMITATION: attempted to ALSO match variance exactly (per the
# spec's "preferably also Var(w^size)~Var(w^alpha)") via two approaches,
# both investigated and rejected before settling on the mean-matched
# gamma above:
#   1. An affine post-transform w' = 1 + a*(w-1) solved to hit the exact
#      target variance: mathematically solvable, but the solution is
#      driven entirely by a single extreme-outlier component (native
#      size=1 voxel under a large gamma), collapsing ~all other weights to
#      ~1.0 -- a degenerate, not a genuine, variance match. Rejected.
#   2. Matching on IQR (a robust spread measure) instead of raw variance:
#      achieves near-exact IQR match, but overshoots the MEAN badly (7.52
#      vs target 2.88) -- a real two-moment tradeoff, since native_size's
#      heavy right tail (a few huge tumors) is fundamentally
#      shape-incompatible with alpha_c's bounded [1,4] range under a
#      single-parameter power law. Rejected.
# CONCLUSION: mean-matching (gamma=0.0995 above) is retained as the primary
# calibration target, per the spec's own "approx" (not exact) mean-match
# requirement and its "preferably" (not mandatory) qualifier on variance.
# The resulting variance mismatch (ratio ~0.4-0.5, size-weighting has LESS
# spread than alpha_c-weighting) is reported honestly and carried forward
# as a disclosed limitation of the S-vs-R comparison, not hidden or forced.
# ============================================================
print(f"\n  NOTE: variance-matching was attempted via two additional methods (affine rescale, IQR-matching)")
print(f"  and both were rejected as producing degenerate or mean-incompatible results (see script comments).")
print(f"  Mean-matching is retained as primary; the resulting variance mismatch is a disclosed limitation.")

# ============================================================
# Section 3: triviality check -- is w_alpha effectively identical to a
# size-based weight?
# ============================================================
print("\n=== Section 3: triviality check ===")
rho_walpha_size, p_ = stats.spearmanr(w_alpha, native_size)
rho_walpha_wsize, p2_ = stats.spearmanr(w_alpha, w_size)
print(f"  Corr(w_alpha, native_size): rho={rho_walpha_size:+.3f} (p={p_:.4e})")
print(f"  Corr(w_alpha, w_size): rho={rho_walpha_wsize:+.3f} (p={p2_:.4e})")

ks_stat, ks_p = stats.ks_2samp(w_alpha, w_size)
print(f"  KS test (are the two weight DISTRIBUTIONS effectively identical?): D={ks_stat:.4f} p={ks_p:.4e}")
print(f"  --> {'DISTRIBUTIONS ARE STATISTICALLY DISTINGUISHABLE (expected/good -- not identical weighting schemes)' if ks_p < 0.05 else 'DISTRIBUTIONS INDISTINGUISHABLE -- investigate before proceeding, this could mean the two conditions are not meaningfully different'}")

is_trivial = abs(rho_walpha_size) > 0.95 or ks_p > 0.5
print(f"\n  TRIVIALITY VERDICT: {'LIKELY KILL CONDITION -- w_alpha is essentially indistinguishable from a size-based weight' if is_trivial else 'w_alpha is NOT a trivial re-expression of size weighting -- proceed'}")

results = {
    "kappa": KAPPA, "gamma_calibrated": gamma_calibrated, "eps_size": EPS_SIZE,
    "w_alpha_stats": {"mean": float(w_alpha.mean()), "var": float(w_alpha.var()), "min": float(w_alpha.min()), "max": float(w_alpha.max())},
    "w_size_stats": {"mean": float(w_size.mean()), "var": float(w_size.var()), "min": float(w_size.min()), "max": float(w_size.max())},
    "triviality_check": {
        "corr_w_alpha_vs_native_size": rho_walpha_size,
        "corr_w_alpha_vs_w_size": rho_walpha_wsize,
        "ks_statistic": ks_stat, "ks_p": ks_p,
        "is_trivial": is_trivial,
    },
}
with open(BASE / "E34_weight_calibration_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {BASE / 'E34_weight_calibration_results.json'}")

# Save the actual per-component weight lookup tables for both conditions
weight_table_alpha = {}
weight_table_size = {}
for i, r in enumerate(records):
    key = f"{r['subject_id']}|{r['native_component_id']}"
    weight_table_alpha[key] = float(w_alpha[i])
    weight_table_size[key] = float(w_size[i])

with open(BASE / "E34_weight_table_alpha_c.json", "w") as f:
    json.dump(weight_table_alpha, f)
with open(BASE / "E34_weight_table_size.json", "w") as f:
    json.dump(weight_table_size, f)
print(f"Saved weight lookup tables ({len(weight_table_alpha)} entries each)")
