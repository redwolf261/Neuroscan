"""
Phase E41-C/D: does the model's actual residual error against the true
occupancy target predict where D4 supervision helps -- beyond size?

NO TRAINING. Reads E41_occupancy_and_error_table.json (749 components, 227
with nonzero D4/D2 footprint -- same population E32/E35/E37 have
consistently found survives the mandatory 64^3 resize).

Reuses E32's own established nonlinear-size-control and permutation-
safeguard methodology VERBATIM (zscore/r2_of/permute_within_2d_strata,
copied unchanged from
experiments/exp_e12_eggo_m/e32/analyze_e32_alpha_c_incremental_value.py),
exactly as done in E37's own reuse of the same functions.

E41C: Delta_E = E_16 - E_2 (mean_bce at D4 minus mean_bce at D2), tested
against nonlinear size, per the exact same protocol E32/E35/E37 already
established for this project.

E41D (the decisive test): does E_s (residual BCE error against the TRUE
soft target) predict where D4 supervision actually helps, beyond size? Two
outcome definitions tested, both real and already-verified in E35's roster:
    1. detected_by_D4 (binary: was this component detected at all by the
       D4-supervised model) vs detected_by_A (baseline) -- a "rescue" outcome.
    2. component_dice_D4 - component_dice_A -- a continuous Dice-improvement
       outcome, for components detected by BOTH (matching this project's own
       established "quality of detection, not just presence" distinction
       from the original Deep Supervision mechanism audit).
Tested separately for small (native_size<=150, S1-S4) vs large (S5) components,
per the explicit instruction not to let a size-driven effect masquerade as a
"does D4 help" effect.
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

OUT_DIR = Path(__file__).parent
TABLE_PATH = OUT_DIR / "E41_occupancy_and_error_table.json"


# ---- reused verbatim from e32/analyze_e32_alpha_c_incremental_value.py (and E37's own reuse) ----
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def r2_of(X, y):
    Xb = np.column_stack([np.ones(len(y))] + [X[:, j] for j in range(X.shape[1])]) if X.ndim > 1 else np.column_stack([np.ones(len(y)), X])
    coef, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def permute_within_2d_strata(values, size_a, size_b, rng, n_strata=6):
    permuted = values.copy()
    edges_a = np.percentile(size_a, np.linspace(0, 100, n_strata + 1))
    edges_b = np.percentile(size_b, np.linspace(0, 100, n_strata + 1))
    strata_a = np.digitize(size_a, edges_a[1:-1])
    strata_b = np.digitize(size_b, edges_b[1:-1])
    strata = strata_a * n_strata + strata_b
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        if len(idx) > 1:
            permuted[idx] = values[rng.permutation(idx)]
    return permuted


def residualize(y, X):
    Xb = np.column_stack([np.ones(len(y)), X])
    coef, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
    return y - Xb @ coef
# ---------------------------------------------------------------------------


def main():
    records = json.load(open(TABLE_PATH))
    valid = [r for r in records if r["D4"]["mean_bce"] is not None and r["D2"]["mean_bce"] is not None]
    print(f"Total components: {len(records)}, valid (nonzero D4+D2 footprint): {len(valid)}", flush=True)

    native_size = np.array([r["native_size"] for r in valid], dtype=np.float64)
    size_64 = np.array([r["size_64"] for r in valid], dtype=np.float64)
    E_16 = np.array([r["D4"]["mean_bce"] for r in valid])
    E_2 = np.array([r["D2"]["mean_bce"] for r in valid])
    detected_A = np.array([r["component_dice_A"] > 0 for r in valid])
    detected_D4 = np.array([r["detected_by_D4"] for r in valid])
    dice_A = np.array([r["component_dice_A"] for r in valid])
    dice_D4 = np.array([r["component_dice_D4"] for r in valid])

    log_native = np.log(native_size + 1)
    native_cbrt = native_size ** (1 / 3)
    log_size64 = np.log(size_64 + 1)
    base_features = np.column_stack([
        zscore(native_size), zscore(log_native), zscore(native_cbrt),
        zscore(size_64), zscore(log_size64),
    ])

    # ================= E41C: Delta_E = E_16 - E_2 =================
    print("\n=== E41C: Delta_E = E_16 - E_2 (D4 BCE minus D2 BCE) ===", flush=True)
    delta_E = E_16 - E_2
    frac_positive = float((delta_E > 0).mean())
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(E_16, E_2)
    print(f"  Mean E_16={E_16.mean():.3f}  Mean E_2={E_2.mean():.3f}  Mean Delta_E={delta_E.mean():+.3f}", flush=True)
    print(f"  Fraction with Delta_E > 0 (D4 harder than D2): {frac_positive:.3f}", flush=True)
    print(f"  Wilcoxon signed-rank (paired, E_16 vs E_2): stat={wilcoxon_stat:.1f} p={wilcoxon_p:.4e}", flush=True)

    r2_base_deltaE = r2_of(base_features, delta_E)
    print(f"  R^2(Delta_E ~ nonlinear size alone): {r2_base_deltaE:.4f}  "
          f"({'Delta_E is LARGELY explained by size' if r2_base_deltaE > 0.5 else 'Delta_E retains variation beyond size'})", flush=True)

    results = {"n_valid": len(valid)}
    results["e41c"] = {
        "mean_E16": float(E_16.mean()), "mean_E2": float(E_2.mean()), "mean_delta_E": float(delta_E.mean()),
        "frac_delta_E_positive": frac_positive,
        "wilcoxon": {"stat": float(wilcoxon_stat), "p": float(wilcoxon_p)},
        "r2_deltaE_from_size": r2_base_deltaE,
    }

    # ================= E41D: does E_s predict where D4 helps, beyond size? =================
    print("\n=== E41D: does residual BCE error predict D4's rescue/improvement, beyond size? ===", flush=True)

    # Outcome 1: rescue (detected_by_D4 AND NOT already detected_by_A)
    # -- among components NOT detected by baseline A, does E_16 (or E_2)
    # predict whether D4 supervision rescues them?
    not_detected_by_A = ~detected_A
    n_not_detected = int(not_detected_by_A.sum())
    print(f"  Components not detected by baseline A: {n_not_detected}/{len(valid)}", flush=True)

    results["e41d"] = {}

    if n_not_detected >= 10:
        rescued = detected_D4[not_detected_by_A].astype(float)
        E16_sub = E_16[not_detected_by_A]
        size_sub = native_size[not_detected_by_A]
        rho_rescue, p_rescue = stats.spearmanr(E16_sub, rescued)
        print(f"  Spearman(E_16, rescued-by-D4) among undetected-by-A components (n={n_not_detected}): "
              f"rho={rho_rescue:+.3f} (p={p_rescue:.3f})", flush=True)
        results["e41d"]["rescue_outcome"] = {"n": n_not_detected, "spearman_rho": float(rho_rescue), "spearman_p": float(p_rescue)}
    else:
        print(f"  Too few undetected-by-A components (n={n_not_detected}) for a meaningful rescue test -- skipped.", flush=True)
        results["e41d"]["rescue_outcome"] = None

    # Outcome 2: Dice improvement, among components detected by BOTH A and D4
    # (matching this project's own established "quality of detection, not
    # just presence" distinction from the original Deep Supervision audit)
    both_detected = detected_A & detected_D4
    n_both = int(both_detected.sum())
    print(f"\n  Components detected by BOTH A and D4: {n_both}/{len(valid)}", flush=True)

    dice_improvement = (dice_D4 - dice_A)[both_detected]
    E16_both = E_16[both_detected]
    E2_both = E_2[both_detected]
    size_both = native_size[both_detected]
    size64_both = size_64[both_detected]

    rho_raw, p_raw = stats.spearmanr(E16_both, dice_improvement)
    print(f"  Spearman(E_16, Dice_improvement) among both-detected (n={n_both}), RAW: rho={rho_raw:+.3f} (p={p_raw:.3f})", flush=True)

    base_features_both = np.column_stack([
        zscore(size_both), zscore(np.log(size_both + 1)), zscore(size_both ** (1 / 3)),
        zscore(size64_both), zscore(np.log(size64_both + 1)),
    ])
    r2_base_both = r2_of(base_features_both, dice_improvement)
    X_with_E16 = np.column_stack([base_features_both, zscore(E16_both)])
    r2_with_E16 = r2_of(X_with_E16, dice_improvement)
    delta_r2 = r2_with_E16 - r2_base_both

    E16_resid = residualize(E16_both, base_features_both)
    dice_imp_resid = residualize(dice_improvement, base_features_both)
    rho_partial, p_partial = stats.spearmanr(E16_resid, dice_imp_resid)
    print(f"  R^2(size-only)={r2_base_both:.4f}  R^2(size+E_16)={r2_with_E16:.4f}  Delta R^2={delta_r2:+.4f}", flush=True)
    print(f"  Partial rho(E_16, Dice_improvement | nonlinear size) = {rho_partial:+.3f} (p={p_partial:.3f})", flush=True)

    results["e41d"]["dice_improvement_outcome"] = {
        "n": n_both, "spearman_raw": {"rho": float(rho_raw), "p": float(p_raw)},
        "r2_size_only": r2_base_both, "r2_size_plus_E16": r2_with_E16, "delta_r2": delta_r2,
        "spearman_partial": {"rho": float(rho_partial), "p": float(p_partial)},
    }

    # ================= Small vs large split =================
    print("\n=== E41D continued: small (S1-S4, native_size<=150) vs large (S5) split ===", flush=True)
    small_mask_both = size_both <= 150
    large_mask_both = size_both > 150
    n_small, n_large = int(small_mask_both.sum()), int(large_mask_both.sum())
    print(f"  Both-detected AND small (n={n_small}), Both-detected AND large (n={n_large})", flush=True)

    split_results = {}
    for label, mask in [("small", small_mask_both), ("large", large_mask_both)]:
        n_here = int(mask.sum())
        if n_here < 8:
            print(f"  [{label}] n={n_here} -- too few for a meaningful test, skipped.", flush=True)
            split_results[label] = {"n": n_here, "skipped": True}
            continue
        rho_s, p_s = stats.spearmanr(E16_both[mask], dice_improvement[mask])
        print(f"  [{label}] n={n_here}: Spearman(E_16, Dice_improvement) RAW = {rho_s:+.3f} (p={p_s:.3f})", flush=True)
        split_results[label] = {"n": n_here, "spearman_raw": {"rho": float(rho_s), "p": float(p_s)}}
    results["e41d"]["small_vs_large_split"] = split_results

    # ================= Permutation safeguard (500+ trials, size-stratified) =================
    print("\n=== Permutation safeguard on the partial correlation (500 trials, size-stratified) ===", flush=True)
    n_perm = 500
    perm_rhos = []
    for trial in range(n_perm):
        rng_t = np.random.RandomState(trial)
        dice_imp_perm = permute_within_2d_strata(dice_improvement, size_both, size64_both, rng_t, n_strata=6)
        dice_imp_perm_resid = residualize(dice_imp_perm, base_features_both)
        rho_p, _ = stats.spearmanr(E16_resid, dice_imp_perm_resid)
        perm_rhos.append(rho_p)
    perm_rhos = np.array(perm_rhos)
    empirical_p = float((np.abs(perm_rhos) >= np.abs(rho_partial)).mean())
    percentile = float(stats.percentileofscore(np.abs(perm_rhos), np.abs(rho_partial)))
    print(f"  Real partial rho={rho_partial:+.3f}  Permuted mean|rho|={np.abs(perm_rhos).mean():.3f} SD={perm_rhos.std():.3f}", flush=True)
    print(f"  Empirical p-value: {empirical_p:.4f}  (percentile of real: {percentile:.1f})", flush=True)
    results["e41d"]["permutation_safeguard"] = {
        "real_partial_rho": float(rho_partial), "perm_mean_abs_rho": float(np.abs(perm_rhos).mean()),
        "perm_sd": float(perm_rhos.std()), "empirical_p": empirical_p, "percentile_of_real": percentile, "n_perm": n_perm,
    }

    # ================= Outlier sensitivity =================
    print("\n=== Outlier sensitivity (excl top-5 |E_16 z-score| outliers) ===", flush=True)
    z_E16 = zscore(E16_both)
    order = np.argsort(-np.abs(z_E16 - z_E16.mean()))
    mask_no5 = np.ones(n_both, dtype=bool)
    mask_no5[order[:5]] = False
    rho_no5, p_no5 = stats.spearmanr(E16_both[mask_no5], dice_improvement[mask_no5])
    print(f"  Spearman(E_16, Dice_improvement) excl top-5 outliers: rho={rho_no5:+.3f} (p={p_no5:.3f}) vs full raw rho={rho_raw:+.3f}", flush=True)
    results["e41d"]["outlier_sensitivity"] = {"rho_excl_top5": float(rho_no5), "p_excl_top5": float(p_no5)}

    with open(OUT_DIR / "E41_transformation_error_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved E41_transformation_error_results.json", flush=True)


if __name__ == "__main__":
    main()
