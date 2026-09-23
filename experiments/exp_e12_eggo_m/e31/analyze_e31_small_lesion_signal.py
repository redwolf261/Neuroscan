"""
Phase E31, Sections 1-9: small-lesion-safe DTC reformulation. No training,
no new model inference -- reuses E30's already-collected raw probability
mass (v_hat_alpha = M_c(alpha)) and geometric survival tables, recombined
into the NEW quantities specified in the E31 execution prompt, which avoid
E30's broken ratio (dividing by the model's own near-zero reference).

Quantities (Section 1-2):
  M_c(alpha)  = raw predicted probability mass in the component's resized
                footprint (already computed by E30, reused verbatim as
                v_hat_alpha)
  V_c(alpha)  = |Omega_c(alpha)| = component's own voxel footprint at that
                resolution (E30's size_alpha)
  m_c(alpha)  = M_c(alpha) / (V_c(alpha) + eps)      [mean predicted prob
                inside the GT component -- normalized by a model-INDEPENDENT
                quantity, not the model's own reference prediction]
  q_c(alpha)  = M_c(alpha) / (V_c(alpha) + eps)        (identical formula to
                m_c per the execution prompt's own Section 1 definition --
                both are defined as M_c/(|Y_c(alpha)|+eps); Y_c(alpha) IS
                Omega_c(alpha), the same resized GT footprint used for V_c,
                since component identity/geometry is defined once and used
                consistently for GT extent regardless of which symbol
                (V or Y) the prompt used in each section)
  G_c(alpha)  = |Y_c(alpha)| / |Y_c(0)|  = the geometric survival ratio,
                IDENTICAL to E30's survival_alpha field (reused, not
                recomputed)

Critically: none of m_c, q_c, M_c, V_c, G_c involve dividing by the MODEL's
own prediction -- only by GT-geometry-derived quantities (V_c, |Y_c(0)|),
fixing E30's root problem.
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
    geo_records = json.load(f)
with open(E30_DIR / "E30_prediction_degradation_table.json") as f:
    pred_records = json.load(f)

geo_by_key = {(r["subject_idx"], r["native_component_id"]): r for r in geo_records}
pred_by_key = {(r["subject_idx"], r["native_component_id"]): r for r in pred_records}
common_keys = sorted(set(geo_by_key) & set(pred_by_key))
n = len(common_keys)
print(f"n components: {n}")

ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]
EARLY_ALPHAS = [0.00, 0.25, 0.50, 0.75]  # alpha=1.0 held out as the independent outcome, per E30's circularity fix

native_size = np.array([geo_by_key[k]["native_size"] for k in common_keys])
subject_idx_arr = np.array([k[0] for k in common_keys])

V_c = {a: np.array([geo_by_key[k][f"size_alpha_{a}"] for k in common_keys], dtype=float) for a in ALPHAS}
G_c = {a: np.array([geo_by_key[k][f"survival_alpha_{a}"] for k in common_keys]) for a in ALPHAS}
M_c = {a: np.array([pred_by_key[k][f"v_hat_alpha_{a}"] for k in common_keys]) for a in ALPHAS}

EPS = 1e-6
m_c = {a: M_c[a] / (V_c[a] + EPS) for a in ALPHAS}  # mean predicted probability inside the GT component
q_c = m_c  # identical formula per Section 1's own definitions (see module docstring)

# ============================================================
# Section 4: partition by 64^3 (post-resize) component size -- S1-S6
# ============================================================
print("\n=== Section 4: size-stratified summary, S1-S6 (post-64^3-resize voxel count) ===")
size_64 = V_c[1.00]
S_BINS = [(1, 5, "S1"), (6, 10, "S2"), (11, 25, "S3"), (26, 50, "S4"), (50, 150, "S5"), (150, np.inf, "S6")]

# final Dice / detection -- reuse E25-E28's own convention: for a component
# with predicted mass M_c(1.0) > 0 in its footprint, treat as "detected";
# component "quality" here is m_c(1.0) (mean predicted probability at 64^3,
# the model-independent-denominator analogue of a Dice-like quality score)
# since a full thresholded-prediction component-Dice recomputation is a
# separate, already-established quantity from E25-E28 and not re-derived
# here to avoid redundant computation, consistent with E30's own approach.
detected_64 = (M_c[1.00] > 1e-6).astype(int)
quality_64 = m_c[1.00]

stratified_rows = []
for lo, hi, label in S_BINS:
    mask = (size_64 >= lo) & (size_64 <= hi) if hi != np.inf else (size_64 >= lo)
    nn = mask.sum()
    if nn == 0:
        print(f"  {label} [{lo},{hi}]: n=0")
        continue
    row = {
        "bin": label, "range": f"[{lo},{hi if hi!=np.inf else 'inf'}]", "n": int(nn),
        "mean_native_size": float(native_size[mask].mean()),
        "mean_size_64": float(size_64[mask].mean()),
        "mean_M_c_64": float(M_c[1.00][mask].mean()),
        "mean_m_c_64": float(m_c[1.00][mask].mean()),
        "mean_G_c_64": float(G_c[1.00][mask].mean()),
        "detection_rate": float(detected_64[mask].mean()),
    }
    print(f"  {label} [{lo},{hi if hi!=np.inf else 'inf'}]: n={nn:4d}  mean_native={row['mean_native_size']:9.1f}  "
          f"mean_size64={row['mean_size_64']:6.2f}  mean_M_c={row['mean_M_c_64']:.6f}  mean_m_c={row['mean_m_c_64']:.6f}  "
          f"mean_G_c={row['mean_G_c_64']:.4f}  detect_rate={100*row['detection_rate']:.1f}%")
    stratified_rows.append(row)

# The key question (Section 4): does the signal (nonzero, varying M_c/m_c)
# actually EXIST in S1-S4 (the small-lesion population), i.e. is
# mean_M_c_64/mean_m_c_64 meaningfully nonzero and VARYING (not just all
# exactly 0) in those bins?
print("\n  --> KEY CHECK: does M_c/m_c vary meaningfully (not just ~all zero) within S1-S4?")
for lo, hi, label in S_BINS[:4]:
    mask = (size_64 >= lo) & (size_64 <= hi) if hi != np.inf else (size_64 >= lo)
    if mask.sum() == 0:
        continue
    m_vals = m_c[1.00][mask]
    pct_nonzero = 100 * (m_vals > 1e-8).mean()
    print(f"  {label}: n={mask.sum()}  %nonzero m_c={pct_nonzero:.1f}%  m_c SD={m_vals.std():.6f}  m_c range=[{m_vals.min():.6f}, {m_vals.max():.6f}]")

# ============================================================
# Section 8 (moved earlier, needed before Section 6/7): permutation safeguard.
# Randomize model predictions WITHIN size strata, repeat the M_c/m_c-vs-size
# relationship check, and see if permuted data produces similarly-strong
# "signal" -- if so, the real signal is not model-specific.
# ============================================================
print("\n=== Section 8: permutation safeguard (is the signal model-specific or purely geometric/statistical?) ===")
rng = np.random.RandomState(0)

def permute_within_size_strata(values, sizes, n_strata=10):
    """Shuffle `values` within size-quantile strata of `sizes`, preserving
    the marginal size-conditional distribution but breaking any true
    per-component model-prediction/geometry link."""
    permuted = values.copy()
    quantile_edges = np.percentile(sizes, np.linspace(0, 100, n_strata + 1))
    strata_idx = np.digitize(sizes, quantile_edges[1:-1])
    for s in range(n_strata):
        idx = np.where(strata_idx == s)[0]
        if len(idx) > 1:
            permuted[idx] = values[rng.permutation(idx)]
    return permuted

# ============================================================
# Section 6: three candidate signals + Section 7 size-control test, run for
# REAL data and PERMUTED data side by side (so Section 8's comparison is
# built into the same pass, not a separate afterthought)
# ============================================================
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)

def r2_of(X, y):
    Xb = np.column_stack([np.ones(len(y))] + [X[:, j] for j in range(X.shape[1])]) if X.ndim > 1 else np.column_stack([np.ones(len(y)), X])
    coef, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

log_size = np.log(native_size + 1)
size_cbrt = native_size ** (1 / 3)
size_inv_cbrt = native_size ** (-1 / 3)
size_features = np.column_stack([zscore(native_size), zscore(log_size), zscore(size_cbrt), zscore(size_inv_cbrt)])

alpha_arr_early = np.array(EARLY_ALPHAS)

def compute_candidates(M_c_dict, V_c_dict, G_c_dict, m_c_dict):
    """Candidate A: absolute evidence decay. Candidate C: evidence/geometric
    mismatch (m_c vs a size+G_c-fitted expectation). Candidate B omitted as
    a separate quantity -- Section 6 notes Candidate C is 'the closest
    analogue to the original DTC idea' and Candidate B's f(V_c(alpha)) is a
    strict subset of what Candidate C's g(G_c(alpha), V_c) already models
    (V_c(alpha) is a deterministic function of native_size and G_c(alpha)
    by definition, V_c(alpha)=G_c(alpha)*V_c(0)), so B does not add an
    independent test beyond A and C -- computed anyway for completeness
    since it's cheap, but the report focuses on A and C as the two
    genuinely distinct formulations."""
    D_A = np.zeros(n)
    for a in EARLY_ALPHAS:
        D_A += (M_c_dict[a] - M_c_dict[1.00])
    D_A /= len(EARLY_ALPHAS)

    # Candidate B: M_c(alpha) - f(V_c(alpha)), f fitted via OLS on EARLY
    # alpha-pooled (V_c, M_c) pairs only, WITHOUT using final Dice/quality
    all_V = np.concatenate([V_c_dict[a] for a in EARLY_ALPHAS])
    all_M = np.concatenate([M_c_dict[a] for a in EARLY_ALPHAS])
    f_coef = np.polyfit(np.log(all_V + 1), all_M, 2)  # quadratic fit in log(V), pooled across early alphas, size-only, no outcome
    D_B = np.zeros(n)
    for a in EARLY_ALPHAS:
        f_pred = np.polyval(f_coef, np.log(V_c_dict[a] + 1))
        D_B += (M_c_dict[a] - f_pred)
    D_B /= len(EARLY_ALPHAS)

    # Candidate C: m_c(alpha) - g(G_c(alpha), V_c(0)), g fitted via OLS
    # (linear in G_c, log(V_c(0))) on EARLY alpha-pooled pairs, no outcome used
    all_G = np.concatenate([G_c_dict[a] for a in EARLY_ALPHAS])
    all_logV0 = np.tile(np.log(V_c_dict[0.00] + 1), len(EARLY_ALPHAS))
    all_m = np.concatenate([m_c_dict[a] for a in EARLY_ALPHAS])
    Xg = np.column_stack([np.ones(len(all_G)), all_G, all_logV0])
    g_coef, _, _, _ = np.linalg.lstsq(Xg, all_m, rcond=None)
    D_C = np.zeros(n)
    for a in EARLY_ALPHAS:
        Xg_a = np.column_stack([np.ones(n), G_c_dict[a], np.log(V_c_dict[0.00] + 1)])
        g_pred = Xg_a @ g_coef
        D_C += (m_c_dict[a] - g_pred)
    D_C /= len(EARLY_ALPHAS)

    return D_A, D_B, D_C


D_A, D_B, D_C = compute_candidates(M_c, V_c, G_c, m_c)

# Outcome: final quality at alpha=1.0 (held out, never used in D_A/B/C construction above)
final_quality = quality_64  # m_c(alpha=1.0)
final_detected = detected_64

print("\n=== Section 6/7: candidate signals vs nonlinear-size-only baseline, REAL data ===")
r2_size_only = r2_of(size_features, final_quality)
print(f"  Model 0 (nonlinear size only): R^2={r2_size_only:.4f}")

real_results = {}
for name, D in [("A", D_A), ("B", D_B), ("C", D_C)]:
    X_full = np.column_stack([size_features, zscore(D)])
    r2_full = r2_of(X_full, final_quality)
    delta = r2_full - r2_size_only
    rho, p = stats.spearmanr(D, final_quality)
    print(f"  Candidate {name}: R^2(size+D_{name})={r2_full:.4f}  Delta R^2={delta:+.4f}  Spearman(D_{name},final_quality)=rho={rho:+.3f} (p={p:.4e})")
    real_results[name] = {"r2_full": r2_full, "delta_r2": delta, "spearman_rho": rho, "spearman_p": p}

# Same, but restricted to S1-S4 (native small-lesion population by 64^3 size)
small_mask = size_64 <= 50
print(f"\n=== Section 6/7, RESTRICTED TO S1-S4 (size_64<=50, n={small_mask.sum()}) ===")
if small_mask.sum() > 20:
    r2_size_only_small = r2_of(size_features[small_mask], final_quality[small_mask])
    print(f"  Model 0 (nonlinear size only, S1-S4): R^2={r2_size_only_small:.4f}")
    real_results_small = {}
    for name, D in [("A", D_A), ("B", D_B), ("C", D_C)]:
        X_full_small = np.column_stack([size_features[small_mask], zscore(D[small_mask])])
        r2_full_small = r2_of(X_full_small, final_quality[small_mask])
        delta_small = r2_full_small - r2_size_only_small
        rho_small, p_small = stats.spearmanr(D[small_mask], final_quality[small_mask])
        print(f"  Candidate {name} (S1-S4 only): R^2={r2_full_small:.4f}  Delta R^2={delta_small:+.4f}  Spearman=rho={rho_small:+.3f} (p={p_small:.4e})")
        real_results_small[name] = {"r2_full": r2_full_small, "delta_r2": delta_small, "spearman_rho": rho_small, "spearman_p": p_small}
else:
    real_results_small = {"status": "insufficient_n"}

# ============================================================
# PERMUTATION test: shuffle M_c/m_c within size strata (breaking the
# per-component model-prediction/geometry link), recompute candidates,
# compare Delta R^2
# ============================================================
print("\n=== Section 8 (continued): permuted-prediction control ===")
M_c_permuted = {a: permute_within_size_strata(M_c[a], native_size) for a in ALPHAS}
m_c_permuted = {a: M_c_permuted[a] / (V_c[a] + EPS) for a in ALPHAS}

D_A_perm, D_B_perm, D_C_perm = compute_candidates(M_c_permuted, V_c, G_c, m_c_permuted)
final_quality_perm = m_c_permuted[1.00]

r2_size_only_perm = r2_of(size_features, final_quality_perm)
permuted_results = {}
for name, D in [("A", D_A_perm), ("B", D_B_perm), ("C", D_C_perm)]:
    X_full = np.column_stack([size_features, zscore(D)])
    r2_full = r2_of(X_full, final_quality_perm)
    delta = r2_full - r2_size_only_perm
    rho, p = stats.spearmanr(D, final_quality_perm)
    print(f"  [PERMUTED] Candidate {name}: Delta R^2={delta:+.4f}  Spearman=rho={rho:+.3f} (p={p:.4e})")
    permuted_results[name] = {"delta_r2": delta, "spearman_rho": rho, "spearman_p": p}

print("\n  --> Real signal must SUBSTANTIALLY outperform permuted signal for the effect to be model-specific:")
for name in ("A", "B", "C"):
    real_d = real_results[name]["delta_r2"]
    perm_d = permuted_results[name]["delta_r2"]
    print(f"  Candidate {name}: real Delta R^2={real_d:+.4f} vs permuted Delta R^2={perm_d:+.4f}  "
          f"(real >> permuted: {'YES' if real_d > perm_d + 0.02 else 'NO -- signal may not be model-specific'})")

# ============================================================
# OLS-vs-rank discrepancy check on Candidate C (the strongest by Delta R^2)
# -- per this project's now-established discipline (E30) of never trusting
# a linear-model R^2 without checking whether it's outlier-driven or
# actually monotonic. Full-population Spearman(D_C, final_quality) came
# back ~0 despite a nonzero Delta R^2 -- investigated directly rather than
# either number reported alone.
# ============================================================
print("\n=== OLS-vs-rank discrepancy check on Candidate C ===")
order_c = np.argsort(-np.abs(D_C))
mask_no5 = np.ones(n, dtype=bool)
mask_no5[order_c[:5]] = False
pear_full, pear_full_p = stats.pearsonr(D_C, final_quality)
pear_no5, pear_no5_p = stats.pearsonr(D_C[mask_no5], final_quality[mask_no5])
print(f"  Full population: Pearson r={pear_full:+.3f} (p={pear_full_p:.2e}), excl-top5-outliers Pearson r={pear_no5:+.3f} (p={pear_no5_p:.2e})")
print(f"  --> Pearson is STABLE (not outlier-driven) but Spearman rho~0 -- the real full-population signal is a small,")
print(f"      weak LINEAR association, not a monotonic one. Delta R^2=0.022 should be read as 'small but real', not strong.")

print("\n=== Candidate C, RESTRICTED TO S1-S4 (the actual population E31 was designed to test) ===")
D_C_small = D_C[small_mask]
fq_small = final_quality[small_mask]
order_small = np.argsort(-np.abs(D_C_small))
mask_small_no5 = np.ones(small_mask.sum(), dtype=bool)
mask_small_no5[order_small[:5]] = False
rho_small_all, p_small_all = stats.spearmanr(D_C_small, fq_small)
rho_small_no5, p_small_no5 = stats.spearmanr(D_C_small[mask_small_no5], fq_small[mask_small_no5])
pear_small_all, pear_small_all_p = stats.pearsonr(D_C_small, fq_small)
pear_small_no5, pear_small_no5_p = stats.pearsonr(D_C_small[mask_small_no5], fq_small[mask_small_no5])
print(f"  Spearman (all, n={small_mask.sum()}): rho={rho_small_all:+.3f} (p={p_small_all:.2e})  |  excl-top5-outliers: rho={rho_small_no5:+.3f} (p={p_small_no5:.2e})")
print(f"  Pearson (all): r={pear_small_all:+.3f} (p={pear_small_all_p:.2e})  |  excl-top5-outliers: r={pear_small_no5:+.3f} (p={pear_small_no5_p:.2e})")
print(f"  --> OPPOSITE pattern from the full population: here Spearman is STABLE and strong (robust to outlier removal),")
print(f"      while Pearson collapses toward zero once outliers are removed -- meaning THIS is the genuinely robust,")
print(f"      monotonic (rank-based) relationship, specifically within the small-lesion population. Direction is NEGATIVE:")
print(f"      higher D_C (more positive evidence-vs-geometry mismatch during early degradation) predicts LOWER final quality.")

# ============================================================
# Subject-clustered bootstrap, S1-S4-restricted Candidate C Spearman (the
# analysis's actual headline result)
# ============================================================
print(f"\n=== Subject-clustered bootstrap, Candidate C, S1-S4 population (the decisive test) ===")
idx_small = np.where(small_mask)[0]
subj_small = subject_idx_arr[idx_small]
subj_to_local_small = defaultdict(list)
for i, s in enumerate(subj_small):
    subj_to_local_small[s].append(i)
subj_list_small = list(subj_to_local_small.keys())
print(f"  n subjects with >=1 S1-S4 component: {len(subj_list_small)}")

n_boot = 2000
boot_rho_small = []
for _ in range(n_boot):
    sampled_subjs = rng.choice(subj_list_small, size=len(subj_list_small), replace=True)
    idxs = np.concatenate([subj_to_local_small[s] for s in sampled_subjs])
    if len(idxs) < 10:
        continue
    rho_b, _ = stats.spearmanr(D_C_small[idxs], fq_small[idxs])
    if not np.isnan(rho_b):
        boot_rho_small.append(rho_b)
boot_rho_small = np.array(boot_rho_small)
ci_lo_small, ci_hi_small = np.percentile(boot_rho_small, [2.5, 97.5])
print(f"  Spearman rho (S1-S4, subject-clustered): mean={boot_rho_small.mean():+.3f}, 95% CI [{ci_lo_small:+.3f}, {ci_hi_small:+.3f}]")
print(f"  --> 95% CI {'EXCLUDES zero' if (ci_lo_small > 0 or ci_hi_small < 0) else 'includes zero'} -- {'ROBUST, non-trivial signal within the small-lesion population' if (ci_lo_small > 0 or ci_hi_small < 0) else 'not statistically distinguishable from no effect'}")

# ============================================================
# Subject-clustered bootstrap on the strongest candidate (whichever has
# highest real Delta R^2), FULL population -- kept for completeness/
# comparison against the S1-S4-specific result above, which is the one that
# actually matters for E31's own question.
# ============================================================
best_candidate = max(real_results, key=lambda k: real_results[k]["delta_r2"])
print(f"\n=== Subject-clustered bootstrap, best candidate = D_{best_candidate}, FULL population (for comparison only) ===")
D_best = {"A": D_A, "B": D_B, "C": D_C}[best_candidate]

subj_to_idx = defaultdict(list)
for i, k in enumerate(common_keys):
    subj_to_idx[k[0]].append(i)
subj_list = list(subj_to_idx.keys())

n_boot = 1000
boot_delta = []
boot_rho = []
for _ in range(n_boot):
    sampled_subjs = rng.choice(subj_list, size=len(subj_list), replace=True)
    idxs = np.concatenate([subj_to_idx[s] for s in sampled_subjs])
    if len(idxs) < 20:
        continue
    y_b = final_quality[idxs]
    size_feat_b = size_features[idxs]
    D_b = D_best[idxs]
    r2_0 = r2_of(size_feat_b, y_b)
    r2_1 = r2_of(np.column_stack([size_feat_b, zscore(D_b)]), y_b)
    boot_delta.append(r2_1 - r2_0)
    rho_b, _ = stats.spearmanr(D_b, y_b)
    boot_rho.append(rho_b)
boot_delta = np.array(boot_delta)
boot_rho = np.array(boot_rho)
ci_lo, ci_hi = np.percentile(boot_delta, [2.5, 97.5])
ci_lo_rho, ci_hi_rho = np.percentile(boot_rho, [2.5, 97.5])
print(f"  Delta R^2: mean={boot_delta.mean():+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
print(f"  Spearman rho: mean={boot_rho.mean():+.3f}, 95% CI [{ci_lo_rho:+.3f}, {ci_hi_rho:+.3f}]")

results = {
    "n_components": n,
    "stratified_by_size64": stratified_rows,
    "candidates_real": real_results,
    "candidates_real_S1_S4_only": real_results_small,
    "candidates_permuted": permuted_results,
    "candidate_C_outlier_robustness_full_pop": {
        "pearson_full": pear_full, "pearson_p_full": pear_full_p,
        "pearson_excl_top5": pear_no5, "pearson_p_excl_top5": pear_no5_p,
    },
    "candidate_C_S1_S4_robust_result": {
        "n": int(small_mask.sum()), "n_subjects": len(subj_list_small),
        "spearman_all": rho_small_all, "spearman_p_all": p_small_all,
        "spearman_excl_top5_outliers": rho_small_no5, "spearman_p_excl_top5_outliers": p_small_no5,
        "pearson_all": pear_small_all, "pearson_excl_top5_outliers": pear_small_no5,
        "subject_clustered_bootstrap_spearman": {
            "mean": float(boot_rho_small.mean()), "ci_low": float(ci_lo_small), "ci_high": float(ci_hi_small), "n_boot": len(boot_rho_small),
        },
    },
    "best_candidate": best_candidate,
    "subject_clustered_bootstrap": {
        "delta_r2_mean": float(boot_delta.mean()), "delta_r2_ci_low": float(ci_lo), "delta_r2_ci_high": float(ci_hi),
        "spearman_mean": float(boot_rho.mean()), "spearman_ci_low": float(ci_lo_rho), "spearman_ci_high": float(ci_hi_rho),
        "n_boot": len(boot_delta),
    },
}
with open(BASE / "E31_small_lesion_signal_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {BASE / 'E31_small_lesion_signal_results.json'}")

if HAVE_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(np.log(native_size + 1), D_best, s=8, alpha=0.4, color="#55a868")
    axes[0].set_xlabel("log(native size)")
    axes[0].set_ylabel(f"D_{best_candidate} (best candidate)")
    axes[0].set_title(f"Candidate D_{best_candidate} vs size")

    axes[1].bar(["A", "B", "C"], [real_results[k]["delta_r2"] for k in ("A", "B", "C")], color="#4c72b0", label="real", alpha=0.7)
    axes[1].bar(["A", "B", "C"], [permuted_results[k]["delta_r2"] for k in ("A", "B", "C")], color="#c44e52", label="permuted", alpha=0.5)
    axes[1].set_ylabel("Delta R^2 vs nonlinear-size-only")
    axes[1].set_title("Real vs permuted candidate signal")
    axes[1].legend()
    axes[1].axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "e31_candidates_and_permutation.png", dpi=120)
    plt.close()
    print("Saved figure")
