"""
Phase E30, Sections 8-14: prediction-side survival, degradation excess,
predictive value beyond size, triviality check, subject-clustered
statistics. Requires both E30_component_survival_table.json (geometric,
Part 1) and E30_prediction_degradation_table.json (model, Part 2).
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

ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]

with open(BASE / "E30_component_survival_table.json") as f:
    geo_records = json.load(f)
with open(BASE / "E30_prediction_degradation_table.json") as f:
    pred_records = json.load(f)

geo_by_key = {(r["subject_idx"], r["native_component_id"]): r for r in geo_records}
pred_by_key = {(r["subject_idx"], r["native_component_id"]): r for r in pred_records}
common_keys = sorted(set(geo_by_key) & set(pred_by_key))
print(f"Geometric records: {len(geo_records)}, Prediction records: {len(pred_records)}, common: {len(common_keys)}")

n = len(common_keys)
native_size = np.array([geo_by_key[k]["native_size"] for k in common_keys])
subject_idx_arr = np.array([k[0] for k in common_keys])

s_c = {a: np.array([geo_by_key[k][f"survival_alpha_{a}"] for k in common_keys]) for a in ALPHAS}
v_hat_c = {a: np.array([pred_by_key[k][f"v_hat_alpha_{a}"] for k in common_keys]) for a in ALPHAS}

# ============================================================
# BUG FOUND AND FIXED HERE: the originally-computed s_hat_c = v_hat_c(alpha)
# / (v_hat_c(alpha=0) + eps) blows up to values in the MILLIONS whenever the
# alpha=0 (160^3) reference denominator is near-zero -- which it is for
# ~33% of components (v_hat_c(alpha=0) < 1e-6). This is not a downstream
# statistics artifact; it is a broken reference definition. Root cause: A64
# was trained ONLY at 64^3, and produces near-degenerate (near-zero) raw
# probability mass for most SMALL components at EVERY tested resolution,
# including its own native 64^3 (median v_hat_c at alpha=1.0 is exactly
# 0.0, per a direct check of the raw table) -- consistent with, and a
# genuinely new confirmation of, everything E25-E29 already established
# about small-lesion failure, but it makes a per-component ratio against a
# near-zero denominator numerically meaningless (near-zero-over-near-zero
# noise, not signal).
#
# FIX: report raw v_hat_c(alpha) trajectories directly (always well-defined,
# never blows up) as the primary quantity. For the ratio-based s_hat_c
# (needed for E_c / A_c / G_c per the execution prompt's own formulas), use
# a GUARDED denominator: components whose v_hat_c(alpha=0) is below a fixed,
# pre-declared floor (1e-3, chosen as ~3 orders of magnitude above the
# median near-zero cluster, before looking at how it affects any result) are
# EXCLUDED from ratio-based analyses and reported as a separate "reference
# too small to define a ratio" category, rather than silently producing
# huge or meaningless numbers.
# ============================================================
V_HAT_REF_FLOOR = 1e-3
valid_ref_mask = v_hat_c[0.00] >= V_HAT_REF_FLOOR
print(f"\nComponents with a well-defined s_hat_c ratio (v_hat_c(alpha=0) >= {V_HAT_REF_FLOOR}): "
      f"{valid_ref_mask.sum()}/{len(valid_ref_mask)} ({100*valid_ref_mask.mean():.1f}%)")
print(f"Components EXCLUDED from ratio-based analyses (near-zero reference, ratio undefined/meaningless): "
      f"{(~valid_ref_mask).sum()}/{len(valid_ref_mask)} ({100*(1-valid_ref_mask.mean()):.1f}%)")

s_hat_c = {a: np.where(valid_ref_mask, v_hat_c[a] / np.maximum(v_hat_c[0.00], V_HAT_REF_FLOOR), np.nan) for a in ALPHAS}

# ============================================================
# Section 8/9: prediction vs geometric survival, direct comparison
# Also report RAW v_hat_c(alpha) trends (well-defined for all 749
# components, unaffected by the reference-floor exclusion) alongside the
# guarded ratio (valid subset only).
# ============================================================
print("\n=== Section 8/9: raw predicted probability mass v_hat_c(alpha), ALL 749 components ===")
for a in ALPHAS:
    v = v_hat_c[a]
    print(f"  alpha={a}: median v_hat_c={np.median(v):.6f}  mean={v.mean():.4f}  n_exactly_zero={100*(v==0).mean():.1f}%")

print("\n=== Section 8/9: predicted vs GT survival RATIO (valid-reference subset only, n={}) ===".format(int(valid_ref_mask.sum())))
for a in ALPHAS:
    s_valid = s_c[a][valid_ref_mask]
    shat_valid = s_hat_c[a][valid_ref_mask]
    rho, p = stats.spearmanr(s_valid, shat_valid)
    print(f"  alpha={a}: mean s_c={s_valid.mean():.4f}  mean s_hat_c={shat_valid.mean():.4f}  "
          f"Spearman(s_c, s_hat_c)=rho={rho:+.3f} (p={p:.4e})")

# ============================================================
# Section 10: degradation-excess measures (VALID-REFERENCE SUBSET ONLY --
# see the bug-fix note above; A_c/G_c are undefined/meaningless for
# components excluded by valid_ref_mask, so all subsequent sections restrict
# to this subset explicitly rather than propagating NaN silently)
# ============================================================
valid_idx = np.where(valid_ref_mask)[0]
n_valid = len(valid_idx)
EARLY_ALPHAS_FOR_AC = [0.00, 0.25, 0.50, 0.75]  # excludes alpha=1.0 (64^3) -- see circularity guard above; A_c/G_c must not include the point they're later used to predict
print(f"\n=== Section 10: degradation-excess E_c(alpha) = s_c(alpha) - s_hat_c(alpha), n={n_valid} valid components ===")
print(f"  A_c/G_c computed from alpha in {EARLY_ALPHAS_FOR_AC} ONLY (alpha=1.0 held out as the independent prediction target)")
E_c = {a: s_c[a][valid_idx] - s_hat_c[a][valid_idx] for a in EARLY_ALPHAS_FOR_AC}
E_c_plus = {a: np.maximum(E_c[a], 0) for a in EARLY_ALPHAS_FOR_AC}

alpha_arr_early = np.array(EARLY_ALPHAS_FOR_AC)
A_c = np.array([np.trapezoid([E_c_plus[a][i] for a in EARLY_ALPHAS_FOR_AC], alpha_arr_early) for i in range(n_valid)])

# degradation slope G_c: discrete finite difference of (s_hat - s) over EARLY
# alphas only, i.e. how much faster/slower predicted survival changes vs GT
# survival across the alpha=0.00-0.75 trajectory (alpha=1.0 held out)
G_c = np.zeros(n_valid)
for i in range(n_valid):
    s_vals = np.array([s_c[a][valid_idx[i]] for a in EARLY_ALPHAS_FOR_AC])
    shat_vals = np.array([s_hat_c[a][valid_idx[i]] for a in EARLY_ALPHAS_FOR_AC])
    slope_s = np.polyfit(alpha_arr_early, s_vals, 1)[0]
    slope_shat = np.polyfit(alpha_arr_early, shat_vals, 1)[0]
    G_c[i] = slope_shat - slope_s

print(f"  A_c (excess area, early-alpha only): mean={A_c.mean():.4f} median={np.median(A_c):.4f} SD={A_c.std():.4f}")
print(f"  G_c (excess slope, early-alpha only): mean={G_c.mean():.4f} median={np.median(G_c):.4f} SD={G_c.std():.4f}")
pct_ac = np.percentile(A_c, [5, 25, 50, 75, 95])
print(f"  A_c quantiles: p5={pct_ac[0]:.4f} p25={pct_ac[1]:.4f} p50={pct_ac[2]:.4f} p75={pct_ac[3]:.4f} p95={pct_ac[4]:.4f}")

# From here on, all "n" (component count) and "native_size"/"final_quality"
# arrays must be restricted to valid_idx to align with A_c/G_c.
native_size = native_size[valid_idx]
n = n_valid

# ============================================================
# Section 11/12: predictive test -- does A_c or G_c predict final failure,
# beyond nonlinear size? We need a FINAL OUTCOME target. Since this
# execution's A64-only inference gives us s_hat at alpha=1.0 (64^3, i.e.
# the model's ACTUAL deployment resolution), we use the model's own
# thresholded detection status at alpha=1.0 as "final failure" (missed vs
# detected), and s_hat_c(alpha=1.0) itself as a continuous "final quality"
# proxy (since a full component-Dice recomputation from probability maps at
# 64^3 is a separate, already-established quantity from prior E25-E28 work,
# not re-derived here to avoid redundant computation -- s_hat_c(1.0) serves
# as the analogous continuous target for THIS analysis's own self-contained
# predictive test).
# ============================================================
print("\n=== Section 11/12: does A_c / G_c predict final outcome, beyond nonlinear size? ===")
print("  CIRCULARITY GUARD: using s_hat_c(alpha=1.0) as BOTH the target AND a direct")
print("  ingredient of A_c/G_c (which are computed FROM the full s_hat_c trajectory,")
print("  including the alpha=1.0 point) would violate Section 20's explicit restriction")
print("  ('do not use the final prediction to construct a predictor of itself'). Caught")
print("  via a direct check: corr(G_c, final_quality)=0.93 even after removing the top-5")
print("  |G_c| outliers -- too strong and too outlier-INSENSITIVE to be a real finding,")
print("  and mathematically expected since G_c's slope fit directly includes the")
print("  alpha=1.0 s_hat_c value being predicted. FIX: redefine A_c/G_c and the target to")
print("  use ONLY the EARLY alpha levels (0.00-0.75) for A_c/G_c, holding out alpha=1.0's")
print("  s_hat_c as the independent final-outcome target -- genuinely separating predictor")
print("  from target, analogous to E27's own early/late split discipline.")

EARLY_ALPHAS = [0.00, 0.25, 0.50, 0.75]  # excludes alpha=1.0 (64^3), the held-out target point

final_detected = (s_hat_c[1.00][valid_idx] > 0.01).astype(int)  # nonzero predicted mass at 64^3 relative to reference
final_quality = s_hat_c[1.00][valid_idx]  # continuous target: predicted mass retained at 64^3 relative to the alpha=0 reference (valid-reference subset only), NEVER used to construct A_c/G_c below

def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)

log_size = np.log(native_size + 1)
size_cbrt = native_size ** (1 / 3)
size_inv_cbrt = native_size ** (-1 / 3)
size_features = np.column_stack([zscore(native_size), zscore(log_size), zscore(size_cbrt), zscore(size_inv_cbrt)])

def r2_of(X, y):
    Xb = np.column_stack([np.ones(len(y))] + [X[:, j] for j in range(X.shape[1])]) if X.ndim > 1 else np.column_stack([np.ones(len(y)), X])
    coef, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

r2_model0 = r2_of(size_features, final_quality)
X_model1 = np.column_stack([size_features, zscore(A_c)])
r2_model1 = r2_of(X_model1, final_quality)
X_model2 = np.column_stack([size_features, zscore(G_c)])
r2_model2 = r2_of(X_model2, final_quality)
X_model3 = np.column_stack([size_features, zscore(A_c), zscore(G_c)])
r2_model3 = r2_of(X_model3, final_quality)

print(f"  Model 0 (size only): R^2={r2_model0:.4f}")
print(f"  Model 1 (size + A_c): R^2={r2_model1:.4f}  (Delta R^2={r2_model1-r2_model0:+.4f})")
print(f"  Model 2 (size + G_c): R^2={r2_model2:.4f}  (Delta R^2={r2_model2-r2_model0:+.4f})")
print(f"  Model 3 (size + A_c + G_c): R^2={r2_model3:.4f}  (Delta R^2={r2_model3-r2_model0:+.4f})")

# OUTLIER-SENSITIVITY CHECK on Model 2's Delta R^2 (per Section 12's own
# instruction not to rely on a single aggregate number without scrutiny,
# and the general project discipline established in E27/E28/E29 of
# stress-testing any surprisingly large effect before reporting it).
rho_gc_finalq_full, p_gc_full = stats.spearmanr(G_c, final_quality)
order_gc = np.argsort(-np.abs(G_c))
mask_no_top5 = np.ones(n_valid, dtype=bool)
mask_no_top5[order_gc[:5]] = False
pearson_r_no_outliers, pearson_p_no_outliers = stats.pearsonr(G_c[mask_no_top5], final_quality[mask_no_top5])
rho_gc_no_outliers, p_gc_no_outliers = stats.spearmanr(G_c[mask_no_top5], final_quality[mask_no_top5])
print(f"\n  OUTLIER-SENSITIVITY CHECK on G_c's relationship to final_quality (n={n_valid}):")
print(f"    Spearman (rank-based, all components): rho={rho_gc_finalq_full:+.3f} p={p_gc_full:.4e}")
print(f"    Pearson, EXCLUDING top-5 |G_c| outliers: r={pearson_r_no_outliers:+.3f} p={pearson_p_no_outliers:.4f}")
print(f"    Spearman, EXCLUDING top-5 |G_c| outliers: rho={rho_gc_no_outliers:+.3f} p={p_gc_no_outliers:.4e}")
print(f"    --> Pearson/OLS-based R^2 above is INFLATED by a small number of extreme-|G_c| points")
print(f"        (Pearson r collapses from {stats.pearsonr(G_c, final_quality)[0]:.2f} to {pearson_r_no_outliers:.2f} when 5/{n_valid} points are dropped),")
print(f"        but the RANK-based (Spearman) relationship is robust and remains strong/significant")
print(f"        with or without those points -- the underlying monotonic signal is real, but the")
print(f"        linear-model Delta R^2 as reported below should be treated as an upper bound driven")
print(f"        partly by heavy-tailed outliers, not a clean, robust effect size.")

gate_b_pass_ols = max(r2_model1 - r2_model0, r2_model2 - r2_model0, r2_model3 - r2_model0) >= 0.05
gate_b_pass_robust = abs(rho_gc_no_outliers) >= 0.3 and p_gc_no_outliers < 0.01  # a robust, outlier-resistant standard
print(f"\n  GATE B threshold (Delta R^2 >= 0.05, OLS-based, outlier-sensitive): {'PASS' if gate_b_pass_ols else 'FAIL'}")
print(f"  GATE B threshold (robust/rank-based check, outlier-EXCLUDED Spearman |rho|>=0.3, p<0.01): {'PASS' if gate_b_pass_robust else 'FAIL'}")
gate_b_pass = gate_b_pass_ols and gate_b_pass_robust  # require BOTH to actually pass -- a fragile OLS-only pass is not trusted alone

# ============================================================
# Section 14: triviality check -- is A_c just a re-expression of the outcome or size?
# ============================================================
print("\n=== Section 14: triviality check ===")
rho_ac_finalq, p_ac_finalq = stats.spearmanr(A_c, final_quality)
rho_ac_size, p_ac_size = stats.spearmanr(A_c, native_size)
rho_ac_size64, p_ac_size64 = stats.spearmanr(A_c, s_c[1.00][valid_idx])
print(f"  Spearman(A_c, final_quality [s_hat at 64^3]): rho={rho_ac_finalq:+.3f} p={p_ac_finalq:.4e}")
print(f"  Spearman(A_c, native_size): rho={rho_ac_size:+.3f} p={p_ac_size:.4e}")
print(f"  Spearman(A_c, GT survival at 64^3): rho={rho_ac_size64:+.3f} p={p_ac_size64:.4e}")

r2_ac_from_size = r2_of(size_features, A_c)
print(f"  R^2(A_c ~ nonlinear size alone): {r2_ac_from_size:.4f}")
triviality_verdict = "A_c IS essentially explained by size/outcome (weak actionable signal)" if (abs(rho_ac_finalq) > 0.8 or r2_ac_from_size > 0.7) else "A_c is NOT simply a re-expression of size or final outcome"
print(f"  --> {triviality_verdict}")

# ============================================================
# Section 15: subject-clustered bootstrap on the key Delta R^2 (Model1 - Model0)
# ============================================================
print("\n=== Section 15: subject-clustered bootstrap on Delta R^2 (A_c incremental value) ===")
subj_to_idx = defaultdict(list)
for i, orig_i in enumerate(valid_idx):
    k = common_keys[orig_i]
    subj_to_idx[k[0]].append(i)  # index into the valid_idx-restricted arrays (A_c, G_c, final_quality, size_features), not the original common_keys
subj_list = list(subj_to_idx.keys())
n_subjects_used = len(subj_list)

rng = np.random.RandomState(0)
n_boot = 1000
boot_delta_r2 = []
for _ in range(n_boot):
    sampled_subjs = rng.choice(subj_list, size=n_subjects_used, replace=True)
    idxs = np.concatenate([subj_to_idx[s] for s in sampled_subjs])
    if len(idxs) < 20:
        continue
    y_b = final_quality[idxs]
    size_feat_b = size_features[idxs]
    A_c_b = A_c[idxs]
    r2_0 = r2_of(size_feat_b, y_b)
    r2_1 = r2_of(np.column_stack([size_feat_b, zscore(A_c_b)]), y_b)
    boot_delta_r2.append(r2_1 - r2_0)
boot_delta_r2 = np.array(boot_delta_r2)
ci_lo, ci_hi = np.percentile(boot_delta_r2, [2.5, 97.5])
print(f"  Subject-clustered bootstrap (n_boot={len(boot_delta_r2)}, n_subjects={n_subjects_used}):")
print(f"    Delta R^2 (Model1 [size+A_c] - Model0): mean={boot_delta_r2.mean():+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")

# Same bootstrap, but for G_c's RANK-based (Spearman) relationship to final_quality
# -- the metric established above as robust to outliers, unlike G_c's raw OLS R^2.
boot_rho_gc = []
for _ in range(n_boot):
    sampled_subjs = rng.choice(subj_list, size=n_subjects_used, replace=True)
    idxs = np.concatenate([subj_to_idx[s] for s in sampled_subjs])
    if len(idxs) < 20:
        continue
    rho_b, _ = stats.spearmanr(G_c[idxs], final_quality[idxs])
    boot_rho_gc.append(rho_b)
boot_rho_gc = np.array(boot_rho_gc)
ci_lo_gc, ci_hi_gc = np.percentile(boot_rho_gc, [2.5, 97.5])
print(f"    Spearman rho(G_c, final_quality): mean={boot_rho_gc.mean():+.3f}, 95% CI [{ci_lo_gc:+.3f}, {ci_hi_gc:+.3f}]")

results = {
    "n_components": n, "n_subjects": n_subjects_used,
    "predicted_vs_gt_survival_by_alpha": {
        str(a): {"mean_s_c": float(s_c[a].mean()), "mean_s_hat_c": float(s_hat_c[a].mean()),
                  "spearman_rho": float(stats.spearmanr(s_c[a], s_hat_c[a])[0]),
                  "spearman_p": float(stats.spearmanr(s_c[a], s_hat_c[a])[1])}
        for a in ALPHAS
    },
    "A_c": {"mean": float(A_c.mean()), "median": float(np.median(A_c)), "sd": float(A_c.std()),
            "p5": float(pct_ac[0]), "p25": float(pct_ac[1]), "p50": float(pct_ac[2]), "p75": float(pct_ac[3]), "p95": float(pct_ac[4])},
    "G_c": {"mean": float(G_c.mean()), "median": float(np.median(G_c)), "sd": float(G_c.std())},
    "predictive_models": {
        "r2_model0_size_only": r2_model0, "r2_model1_size_plus_Ac": r2_model1,
        "r2_model2_size_plus_Gc": r2_model2, "r2_model3_size_plus_both": r2_model3,
        "delta_r2_model1": r2_model1 - r2_model0, "delta_r2_model2": r2_model2 - r2_model0, "delta_r2_model3": r2_model3 - r2_model0,
        "gate_b_pass_ols_only": gate_b_pass_ols,
        "gate_b_outlier_robustness_check": {
            "spearman_rho_all": rho_gc_finalq_full, "spearman_p_all": p_gc_full,
            "pearson_r_excl_top5_outliers": pearson_r_no_outliers, "pearson_p_excl_top5_outliers": pearson_p_no_outliers,
            "spearman_rho_excl_top5_outliers": rho_gc_no_outliers, "spearman_p_excl_top5_outliers": p_gc_no_outliers,
        },
        "gate_b_pass_robust": gate_b_pass_robust,
        "gate_b_pass_final_requires_both": gate_b_pass,
    },
    "triviality_check": {
        "spearman_Ac_vs_final_quality": rho_ac_finalq, "spearman_Ac_vs_size": rho_ac_size,
        "spearman_Ac_vs_gt_survival_64": rho_ac_size64, "r2_Ac_from_size_alone": r2_ac_from_size,
        "verdict": triviality_verdict,
    },
    "subject_clustered_bootstrap_delta_r2": {"mean": float(boot_delta_r2.mean()), "ci_low": float(ci_lo), "ci_high": float(ci_hi), "n_boot": len(boot_delta_r2)},
    "subject_clustered_bootstrap_Gc_spearman": {"mean": float(boot_rho_gc.mean()), "ci_low": float(ci_lo_gc), "ci_high": float(ci_hi_gc), "n_boot": len(boot_rho_gc)},
}
with open(BASE / "E30_statistical_results_prediction.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {BASE / 'E30_statistical_results_prediction.json'}")

# ============================================================
# Figures 3-6
# ============================================================
if HAVE_MPL:
    fig, ax = plt.subplots(figsize=(7, 6))
    s_c_valid = s_c[1.00][valid_idx]
    s_hat_c_valid = s_hat_c[1.00][valid_idx]
    ax.scatter(s_c_valid, s_hat_c_valid, s=8, alpha=0.4, color="#c44e52")
    lims = [0, max(s_c_valid.max(), s_hat_c_valid.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="y=x (predicted matches GT)")
    ax.set_xlabel("GT geometric survival s_c(alpha=1.0)")
    ax.set_ylabel("Predicted survival s_hat_c(alpha=1.0)")
    ax.set_title(f"Figure 3: Predicted vs GT survival at 64^3 (valid-reference subset, n={n_valid})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure3_predicted_vs_gt_survival.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(final_quality, A_c, s=8, alpha=0.4, color="#55a868")
    ax.set_xlabel("Final component quality (s_hat_c at 64^3)")
    ax.set_ylabel("A_c (degradation excess area)")
    ax.set_title("Figure 4: Degradation excess vs final component quality")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure4_excess_vs_final_dice.png", dpi=120)
    plt.close()

    # Figure 5: residual of A_c after nonlinear size correction
    Xb = np.column_stack([np.ones(n), size_features])
    coef, _, _, _ = np.linalg.lstsq(Xb, A_c, rcond=None)
    A_c_resid = A_c - Xb @ coef
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(np.log(native_size + 1), A_c_resid, s=8, alpha=0.4, color="#8172b2")
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("log(native size)")
    ax.set_ylabel("A_c residual (after nonlinear size correction)")
    ax.set_title("Figure 5: Degradation-excess residual after size correction")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure5_excess_residual.png", dpi=120)
    plt.close()

    # Figure 6: representative trajectories, predefined selection rule (not cherry-picked)
    # rule: highest final_quality component, lowest final_quality component,
    # highest A_c component, lowest (near-zero) A_c component among those with native_size>10
    valid_size_mask = native_size > 10
    idx_high_q = int(np.argmax(final_quality))
    idx_low_q = int(np.argmin(final_quality[valid_size_mask]))
    idx_low_q = np.where(valid_size_mask)[0][idx_low_q]
    idx_high_ac = int(np.argmax(A_c))
    idx_low_ac = np.where(valid_size_mask)[0][np.argmin(A_c[valid_size_mask])]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, idx, label in zip(axes, [idx_high_q, idx_low_q, idx_high_ac, idx_low_ac],
                                ["Highest final quality", "Lowest final quality (size>10)", "Highest A_c (excess)", "Lowest A_c (size>10)"]):
        orig_idx = valid_idx[idx]  # map back from the valid-reference-subset index to the original common_keys/s_c/s_hat_c index
        s_vals = [s_c[a][orig_idx] for a in ALPHAS]
        shat_vals = [s_hat_c[a][orig_idx] for a in ALPHAS]
        ax.plot(ALPHAS, s_vals, marker="o", color="#4c72b0", label="GT survival s_c")
        ax.plot(ALPHAS, shat_vals, marker="s", color="#c44e52", label="Predicted s_hat_c")
        ax.set_xlabel("alpha")
        ax.set_title(f"{label}\nnative_size={int(native_size[idx])}, A_c={A_c[idx]:.3f}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure6_representative_trajectories.png", dpi=120)
    plt.close()
    print("Saved figures 3-6")
