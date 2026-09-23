"""
Phase E37: statistical analysis of grid-induced entropy vs actual auxiliary
error. NO TRAINING. Reads E37_component_entropy_table.json (749 components,
227 with valid D4/D2/D1 values -- i.e. survive the mandatory 64^3 resize).

Reuses E32's OWN established nonlinear-size-control and permutation-safeguard
methodology VERBATIM (not reinvented): nonlinear size features (raw, log,
cube-root, applied to both native_size and size_64), an OLS incremental-R^2
model, and a 2D-stratified permutation test (permute WITHIN joint native_size
x size_64 strata, not a naive full shuffle) -- see
experiments/exp_e12_eggo_m/e32/analyze_e32_alpha_c_incremental_value.py's own
zscore/r2_of/permute_within_2d_strata functions, copied here unchanged.

Tests A-E from the E37 prompt, plus the mandatory Delta-H vs Delta-E
differential mechanistic test, plus outlier sensitivity and subject-clustered
bootstrap CI.
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

OUT_DIR = Path(__file__).parent
TABLE_PATH = OUT_DIR / "E37_component_entropy_table.json"


# ---- reused verbatim from e32/analyze_e32_alpha_c_incremental_value.py ----
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
# ---------------------------------------------------------------------------


def main():
    records = json.load(open(TABLE_PATH))
    valid = [r for r in records if r["H_D4"] is not None and r["H_D2"] is not None]
    print(f"Total components: {len(records)}, valid (survive 64^3 resize): {len(valid)}", flush=True)

    subject_idx_map = {}
    for i, r in enumerate(records):
        subject_idx_map.setdefault(r["subject_id"], len(subject_idx_map))

    native_size = np.array([r["native_size"] for r in valid], dtype=np.float64)
    size_64 = np.array([r["size_64"] for r in valid], dtype=np.float64)
    H_D4 = np.array([r["H_D4"] for r in valid])
    H_D2 = np.array([r["H_D2"] for r in valid])
    E_D4 = np.array([r["E_D4"] for r in valid])
    E_D2 = np.array([r["E_D2"] for r in valid])
    B_D4 = np.array([r["B_D4"] for r in valid])
    B_D2 = np.array([r["B_D2"] for r in valid])
    subj_idx = np.array([subject_idx_map[r["subject_id"]] for r in valid])

    n = len(valid)
    results = {}

    # ================= TEST A: resolution effect =================
    print("\n=== TEST A: does H_D4 > H_D2 systematically? ===", flush=True)
    diff = H_D4 - H_D2
    frac_greater = float((diff > 0).mean())
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(H_D4, H_D2)
    print(f"  Fraction of components with H_D4 > H_D2: {frac_greater:.3f} ({(diff>0).sum()}/{n})", flush=True)
    print(f"  Mean H_D4={H_D4.mean():.4f}  Mean H_D2={H_D2.mean():.4f}  Mean diff={diff.mean():+.4f}", flush=True)
    print(f"  Wilcoxon signed-rank test (paired, non-parametric): stat={wilcoxon_stat:.1f} p={wilcoxon_p:.4e}", flush=True)
    # Report the non-monotonicity found during data construction, not hidden
    small_mask = native_size <= 20
    frac_greater_small = float((diff[small_mask] > 0).mean()) if small_mask.sum() > 0 else None
    print(f"  Among SMALL components (native_size<=20, n={small_mask.sum()}): "
          f"fraction with H_D4>H_D2 = {frac_greater_small}", flush=True)
    results["test_A_resolution_effect"] = {
        "frac_H_D4_greater": frac_greater, "mean_H_D4": float(H_D4.mean()), "mean_H_D2": float(H_D2.mean()),
        "mean_diff": float(diff.mean()), "wilcoxon_stat": float(wilcoxon_stat), "wilcoxon_p": float(wilcoxon_p),
        "frac_H_D4_greater_among_small_le20": frac_greater_small,
        "note": "H_D4>H_D2 does NOT hold universally -- binary entropy is non-monotonic in grid "
                "coarseness (a component's coverage fraction pi can be pushed TOWARD 0 rather than "
                "toward 0.5 when diluted into a much larger coarse cell, which LOWERS entropy). "
                "Verified directly during data construction on a synthetic single-voxel case "
                "(pi_D4=1/64 vs pi_D2=1/8, H(1/64)=0.080 < H(1/8)=0.377) before trusting the "
                "aggregate statistic reported here.",
    }

    # ================= TEST B: does entropy predict actual error? =================
    print("\n=== TEST B: entropy vs actual auxiliary error (Spearman) ===", flush=True)
    rho_D4, p_D4 = stats.spearmanr(H_D4, E_D4)
    rho_D2, p_D2 = stats.spearmanr(H_D2, E_D2)
    print(f"  Spearman(H_D4, E_D4): rho={rho_D4:+.3f} p={p_D4:.4e}", flush=True)
    print(f"  Spearman(H_D2, E_D2): rho={rho_D2:+.3f} p={p_D2:.4e}", flush=True)
    results["test_B_entropy_predicts_error"] = {
        "D4": {"spearman_rho": rho_D4, "spearman_p": p_D4},
        "D2": {"spearman_rho": rho_D2, "spearman_p": p_D2},
    }

    # ================= TEST C: nonlinear size control =================
    print("\n=== TEST C: does entropy contain info beyond nonlinear size? (E32 methodology, reused) ===", flush=True)
    log_native = np.log(native_size + 1)
    native_cbrt = native_size ** (1 / 3)
    log_size64 = np.log(size_64 + 1)
    base_features = np.column_stack([
        zscore(native_size), zscore(log_native), zscore(native_cbrt),
        zscore(size_64), zscore(log_size64),
    ])

    test_c_results = {}
    for scale, H, E in [("D4", H_D4, E_D4), ("D2", H_D2, E_D2)]:
        r2_base = r2_of(base_features, E)
        X_with_H = np.column_stack([base_features, zscore(H)])
        r2_with_H = r2_of(X_with_H, E)
        delta_r2 = r2_with_H - r2_base

        # partial Spearman: residualize H and E against the SAME nonlinear
        # size features via OLS, correlate residuals (standard partial-rank approach)
        def residualize(y, X):
            Xb = np.column_stack([np.ones(len(y)), X])
            coef, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
            return y - Xb @ coef

        H_resid = residualize(H, base_features)
        E_resid = residualize(E, base_features)
        rho_partial, p_partial = stats.spearmanr(H_resid, E_resid)

        rho_raw, p_raw = stats.spearmanr(H, E)

        print(f"  [{scale}] R^2(size-only)={r2_base:.4f}  R^2(size+H)={r2_with_H:.4f}  "
              f"Delta R^2={delta_r2:+.4f}", flush=True)
        print(f"  [{scale}] rho(H,E) raw={rho_raw:+.3f} (p={p_raw:.4e})  "
              f"partial rho(H,E | nonlinear size)={rho_partial:+.3f} (p={p_partial:.4e})", flush=True)

        test_c_results[scale] = {
            "r2_size_only": r2_base, "r2_size_plus_H": r2_with_H, "delta_r2": delta_r2,
            "spearman_raw": {"rho": rho_raw, "p": p_raw},
            "spearman_partial_size_controlled": {"rho": rho_partial, "p": p_partial},
        }
    results["test_C_nonlinear_size_control"] = test_c_results

    # ================= TEST D: entropy vs boundary fraction =================
    print("\n=== TEST D: does entropy contain info beyond the crude boundary indicator B? ===", flush=True)
    test_d_results = {}
    for scale, H, B in [("D4", H_D4, B_D4), ("D2", H_D2, B_D2)]:
        rho_HB, p_HB = stats.spearmanr(H, B)
        # R^2 of H ~ f(B) nonlinearly (B, B^2, sqrt(B)) -- does B alone explain H?
        B_features = np.column_stack([zscore(B), zscore(B ** 2), zscore(np.sqrt(np.clip(B, 0, None)))])
        r2_H_from_B = r2_of(B_features, H)
        print(f"  [{scale}] Spearman(H,B)={rho_HB:+.3f} (p={p_HB:.4e})  R^2(H ~ nonlinear f(B))={r2_H_from_B:.4f}", flush=True)
        test_d_results[scale] = {"spearman_H_vs_B": {"rho": rho_HB, "p": p_HB}, "r2_H_explained_by_B": r2_H_from_B}
    results["test_D_entropy_vs_boundary_fraction"] = test_d_results
    warn_D = any(test_d_results[s]["r2_H_explained_by_B"] > 0.95 for s in ("D4", "D2"))
    if warn_D:
        print("  WARNING: H approx f(B) with R^2>0.95 -- entropy may add little beyond the crude boundary indicator.", flush=True)

    # ================= TEST E: permutation safeguard (500+ trials) =================
    print("\n=== TEST E: permutation safeguard (mandatory, size-stratified, 500 trials) ===", flush=True)
    n_perm = 500
    test_e_results = {}
    for scale, H, E in [("D4", H_D4, E_D4), ("D2", H_D2, E_D2)]:
        real_rho_partial = test_c_results[scale]["spearman_partial_size_controlled"]["rho"]
        perm_rhos = []
        for trial in range(n_perm):
            rng_t = np.random.RandomState(trial)
            E_perm = permute_within_2d_strata(E, native_size, size_64, rng_t, n_strata=6)

            def residualize(y, X):
                Xb = np.column_stack([np.ones(len(y)), X])
                coef, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
                return y - Xb @ coef

            H_resid = residualize(H, base_features)
            E_perm_resid = residualize(E_perm, base_features)
            rho_p, _ = stats.spearmanr(H_resid, E_perm_resid)
            perm_rhos.append(rho_p)
        perm_rhos = np.array(perm_rhos)
        frac_exceeds = float((np.abs(perm_rhos) >= np.abs(real_rho_partial)).mean())
        percentile = float(stats.percentileofscore(np.abs(perm_rhos), np.abs(real_rho_partial)))
        print(f"  [{scale}] real |rho_partial|={abs(real_rho_partial):.3f}  "
              f"perm mean|rho|={np.abs(perm_rhos).mean():.3f} SD={perm_rhos.std():.3f}  "
              f"empirical p (frac perm >= real) = {frac_exceeds:.4f}  percentile={percentile:.1f}", flush=True)
        test_e_results[scale] = {
            "real_partial_rho": real_rho_partial, "perm_mean_abs_rho": float(np.abs(perm_rhos).mean()),
            "perm_sd": float(perm_rhos.std()), "empirical_p": frac_exceeds, "percentile_of_real": percentile,
            "n_perm": n_perm,
        }
    results["test_E_permutation_safeguard"] = test_e_results

    # ================= Delta-H vs Delta-E differential test =================
    print("\n=== Delta-H vs Delta-E: mechanistic differential test ===", flush=True)
    delta_H = H_D4 - H_D2
    delta_E = E_D4 - E_D2
    rho_delta, p_delta = stats.spearmanr(delta_H, delta_E)
    print(f"  Spearman(Delta_H, Delta_E): rho={rho_delta:+.3f} p={p_delta:.4e}  (n={n})", flush=True)
    results["differential_test_delta_H_vs_delta_E"] = {"spearman_rho": rho_delta, "spearman_p": p_delta, "n": n}

    # ================= Outlier sensitivity =================
    print("\n=== Outlier sensitivity (excl top-5 |H_D4 z-score| outliers) ===", flush=True)
    z_H_D4 = zscore(H_D4)
    order = np.argsort(-np.abs(z_H_D4 - z_H_D4.mean()))
    mask_no5 = np.ones(n, dtype=bool)
    mask_no5[order[:5]] = False
    rho_no5, p_no5 = stats.spearmanr(H_D4[mask_no5], E_D4[mask_no5])
    print(f"  Spearman(H_D4,E_D4) excl top-5 outliers: rho={rho_no5:+.3f} (p={p_no5:.4e}) "
          f"vs full rho={rho_D4:+.3f}", flush=True)
    results["outlier_sensitivity"] = {"rho_excl_top5": rho_no5, "p_excl_top5": p_no5, "rho_full": rho_D4}

    # ================= Subject-clustered bootstrap =================
    print("\n=== Subject-clustered bootstrap (2000 trials, E32 methodology reused) ===", flush=True)
    rng = np.random.RandomState(0)
    subj_to_idx = defaultdict(list)
    for i, si in enumerate(subj_idx):
        subj_to_idx[si].append(i)
    subj_list = list(subj_to_idx.keys())

    # IMPORTANT: bootstrap BOTH the raw Spearman AND the size-controlled
    # partial Spearman -- a first version of this bootstrap only computed
    # the RAW correlation, which produced a bootstrap CI with the OPPOSITE
    # SIGN from Test C's partial correlation at D4 (raw rho=-0.37 vs partial
    # rho=+0.34). This discrepancy was investigated directly rather than
    # reported as-is: native_size is strongly correlated with BOTH H_D4
    # (rho=+0.49) and E_D4 (rho=-0.89) -- i.e. larger, better-resolved
    # components have both higher entropy AND lower error, a classic
    # confound that dominates the raw correlation's sign. The bootstrap CI
    # must be computed on the SAME size-controlled quantity Test C/E report,
    # not the raw one, to be a meaningful uncertainty estimate for the claim
    # actually being made (that entropy predicts error BEYOND size).
    def residualize(y, X):
        Xb = np.column_stack([np.ones(len(y)), X])
        coef, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
        return y - Xb @ coef

    boot_results = {}
    for scale, H, E in [("D4", H_D4, E_D4), ("D2", H_D2, E_D2)]:
        n_boot = 2000
        boot_rho_raw = []
        boot_rho_partial = []
        for _ in range(n_boot):
            sampled_subjs = rng.choice(subj_list, size=len(subj_list), replace=True)
            idxs = np.concatenate([subj_to_idx[s] for s in sampled_subjs]) if len(sampled_subjs) else np.array([], dtype=int)
            if len(idxs) < 20:
                continue
            rho_b, _ = stats.spearmanr(H[idxs], E[idxs])
            if not np.isnan(rho_b):
                boot_rho_raw.append(rho_b)

            H_b, E_b, feat_b = H[idxs], E[idxs], base_features[idxs]
            H_resid_b = residualize(H_b, feat_b)
            E_resid_b = residualize(E_b, feat_b)
            rho_pb, _ = stats.spearmanr(H_resid_b, E_resid_b)
            if not np.isnan(rho_pb):
                boot_rho_partial.append(rho_pb)

        boot_rho_raw = np.array(boot_rho_raw)
        boot_rho_partial = np.array(boot_rho_partial)
        ci_lo_raw, ci_hi_raw = np.percentile(boot_rho_raw, [2.5, 97.5])
        ci_lo_p, ci_hi_p = np.percentile(boot_rho_partial, [2.5, 97.5])
        print(f"  [{scale}] Bootstrap Spearman RAW: mean={boot_rho_raw.mean():+.3f}, "
              f"95% CI [{ci_lo_raw:+.3f}, {ci_hi_raw:+.3f}]", flush=True)
        print(f"  [{scale}] Bootstrap Spearman PARTIAL (size-controlled, the claim that matters): "
              f"mean={boot_rho_partial.mean():+.3f}, 95% CI [{ci_lo_p:+.3f}, {ci_hi_p:+.3f}]", flush=True)
        boot_results[scale] = {
            "raw": {"mean": float(boot_rho_raw.mean()), "ci_95": [float(ci_lo_raw), float(ci_hi_raw)]},
            "partial_size_controlled": {"mean": float(boot_rho_partial.mean()), "ci_95": [float(ci_lo_p), float(ci_hi_p)]},
        }
    results["subject_clustered_bootstrap"] = boot_results

    with open(OUT_DIR / "E37_statistical_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved E37_statistical_results.json", flush=True)


if __name__ == "__main__":
    main()
