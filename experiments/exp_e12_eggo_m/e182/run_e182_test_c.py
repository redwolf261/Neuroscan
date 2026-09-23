"""
E182 Test C -- the decisive magnitude-conditional comparison, now with a
genuinely magnitude-varying T7 (4 severities, calibration-verified
non-degenerate spread).

Pre-registered in docs/phases/PHASE_E181_MECHANISM_DISSECTION_PREREG.md and
PHASE_E182_ASYMMETRIC_ENERGY_CONTROL_PREREG.md. Read both. T6 budgets and T7
levels are NOT compared by raw label -- conditions explicitly on the
measured uniform_magnitude covariate.

Model: delta_i_global ~ gamma_dice + uniform_magnitude + transform_family
(subject fixed effects, cluster-robust SE), pooling all severities within
each family (T6: 2 severities; T7: 4 severities, unchanged model form from
E181 -- pooling logic does not depend on severity count). transform_family
is a 0/1 indicator (T6=0, T7=1). Also reports Gamma's own coefficient
SEPARATELY within each family after magnitude is in the model.

Per explicit instruction: this script reports RAW results only. It does NOT
collapse them into a single classification label (no H2_SUPPORTED /
BOTH_SURVIVE / etc auto-verdict) -- that interpretation is deliberately
deferred to a separate step after inspecting the actual coefficients,
magnitude overlap, and subject-level behavior.

Associational, not causal proof (stated in the prereg and repeated in the
output) -- Gamma is measured, not independently randomized.

CPU-only, no GPU. Operates on FROZEN E182-D data.
"""
import sys
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent
FROZEN_DIR = OUT_DIR / "FROZEN"
e180_dir = OUT_DIR.parent / "e180"
sys.path.insert(0, str(e180_dir))

from e180_stats import subject_fixed_effects_ols, within_subject_permutation_null  # noqa: E402

N_PERM = 1000


def load_joined():
    gamma = json.load(open(FROZEN_DIR / "E182_D_gamma.json"))
    delta = json.load(open(FROZEN_DIR / "E182_D_delta.json"))
    g_by_key = {(r["sid"], r["window_id"], r["transform"], r["severity"]): r for r in gamma}
    d_by_key = {(r["sid"], r["window_id"], r["transform"], r["severity"]): r for r in delta}
    keys = sorted(set(g_by_key) & set(d_by_key))
    return g_by_key, d_by_key, keys


def main():
    g_by_key, d_by_key, keys = load_joined()
    print(f"[Data] {len(keys)} joined records")

    y, gam, mag, fam_indicator, sids = [], [], [], [], []
    for k in keys:
        sid, wid, transform, severity = k
        y.append(d_by_key[k]["delta_i_global"])
        gam.append(g_by_key[k]["gamma_dice"])
        mag.append(g_by_key[k]["uniform_magnitude"])
        fam_indicator.append(1.0 if transform == "T7" else 0.0)
        sids.append(sid)
    y = np.array(y)
    gam = np.array(gam)
    mag = np.array(mag)
    fam_indicator = np.array(fam_indicator)

    print(f"[Magnitude context] T6 uniform_magnitude range: "
          f"[{mag[fam_indicator==0].min():.4f}, {mag[fam_indicator==0].max():.4f}] "
          f"mean={mag[fam_indicator==0].mean():.4f}")
    print(f"[Magnitude context] T7 uniform_magnitude range: "
          f"[{mag[fam_indicator==1].min():.4f}, {mag[fam_indicator==1].max():.4f}] "
          f"mean={mag[fam_indicator==1].mean():.4f}")

    # ---- pooled model: does transform family predict Delta after controlling
    # for Gamma and magnitude? ----
    X_pooled = np.column_stack([gam, mag, fam_indicator])
    fe_pooled = subject_fixed_effects_ols(y, X_pooled, sids)
    null_family = within_subject_permutation_null(y, X_pooled, sids, gamma_col_idx=2,
                                                   n_perm=N_PERM, seed=0)

    print("\n[Test C -- pooled] Delta ~ Gamma + magnitude + transform_family (subject FE)")
    print(f"  beta_gamma={fe_pooled['beta'][0]:+.4f} (p={fe_pooled['p'][0]:.4f})")
    print(f"  beta_magnitude={fe_pooled['beta'][1]:+.4f} (p={fe_pooled['p'][1]:.4f})")
    perm_p_display = f"< {1/N_PERM:.3f}" if null_family["permutation_p"] == 0 else f"{null_family['permutation_p']:.4f}"
    print(f"  beta_transform_family(T7=1)={fe_pooled['beta'][2]:+.4f} (naive_p={fe_pooled['p'][2]:.4f}, "
          f"subject-permutation_p={perm_p_display})")
    print(f"  R2_within={fe_pooled['r2']:.4f}  n_tiles={fe_pooled['n']}  n_subjects={fe_pooled['n_subjects']}")

    # ---- within-family: does Gamma survive magnitude control, separately per family? ----
    print("\n[Test C -- within-family Gamma-over-magnitude]")
    within_family = {}
    for fam_name, fam_val in [("T6", 0.0), ("T7", 1.0)]:
        idx = fam_indicator == fam_val
        y_f, gam_f, mag_f = y[idx], gam[idx], mag[idx]
        sids_f = [s for s, keep in zip(sids, idx) if keep]

        X_f = np.column_stack([mag_f, gam_f])
        null_f = within_subject_permutation_null(y_f, X_f, sids_f, gamma_col_idx=1,
                                                  n_perm=N_PERM, seed=0)
        fe_gamma_alone = subject_fixed_effects_ols(y_f, gam_f, sids_f)
        fe_mag_alone = subject_fixed_effects_ols(y_f, mag_f, sids_f)

        p_display = f"< {1/N_PERM:.3f}" if null_f["permutation_p"] == 0 else f"{null_f['permutation_p']:.4f}"
        survives = null_f["permutation_p"] < 0.05
        within_family[fam_name] = {
            "gamma_alone_beta": fe_gamma_alone["beta"][0], "gamma_alone_p": fe_gamma_alone["p"][0],
            "magnitude_alone_beta": fe_mag_alone["beta"][0], "magnitude_alone_p": fe_mag_alone["p"][0],
            "gamma_over_magnitude_control": null_f,
            "SURVIVES_MAGNITUDE_CONTROL": bool(survives),
            "n_tiles": len(y_f), "n_subjects": len(set(sids_f)),
        }
        print(f"  {fam_name}: gamma_alone_beta={fe_gamma_alone['beta'][0]:+.4f} (p={fe_gamma_alone['p'][0]:.4f})  "
              f"gamma|magnitude perm_p={p_display}  SURVIVES={survives}")

    # ---- raw per-severity breakdown within T7 (informational context only --
    # not a substitute for the pooled within-family test above, but useful
    # given T7 now spans a genuine magnitude range rather than 2 near-
    # identical points) ----
    print("\n[Context] T7 raw Gamma-alone effect per individual severity")
    t7_per_severity = {}
    for sev in [0.6, 0.8, 1.1, 1.45]:
        sev_keys = [k for k in keys if k[2] == "T7" and k[3] == sev]
        sev_sids = [k[0] for k in sev_keys]
        sev_gam = np.array([g_by_key[k]["gamma_dice"] for k in sev_keys])
        sev_y = np.array([d_by_key[k]["delta_i_global"] for k in sev_keys])
        sev_mag = np.array([g_by_key[k]["uniform_magnitude"] for k in sev_keys])
        fe_sev = subject_fixed_effects_ols(sev_y, sev_gam, sev_sids)
        t7_per_severity[str(sev)] = {
            "gamma_alone_beta": fe_sev["beta"][0], "gamma_alone_p": fe_sev["p"][0],
            "magnitude_mean": float(sev_mag.mean()),
            "n_tiles": fe_sev["n"], "n_subjects": fe_sev["n_subjects"],
        }
        print(f"  T7 sev={sev} (magnitude_mean={sev_mag.mean():.4f}): "
              f"beta={fe_sev['beta'][0]:+.4f} p={fe_sev['p'][0]:.4f}")

    print("\n[NOTE] Raw results only -- no automatic classification label. "
          "Interpretation deferred to a separate step.")

    results = {
        "pooled_model": {
            "specification": "Delta ~ Gamma + magnitude + transform_family (subject FE, cluster-robust)",
            "beta_gamma": fe_pooled["beta"][0], "p_gamma": fe_pooled["p"][0],
            "beta_magnitude": fe_pooled["beta"][1], "p_magnitude": fe_pooled["p"][1],
            "beta_transform_family": fe_pooled["beta"][2], "naive_p_transform_family": fe_pooled["p"][2],
            "transform_family_permutation_p": null_family["permutation_p"],
            "r2_within": fe_pooled["r2"], "n_tiles": fe_pooled["n"], "n_subjects": fe_pooled["n_subjects"],
        },
        "within_family_gamma_over_magnitude": within_family,
        "t7_per_severity_context": t7_per_severity,
        "magnitude_context": {
            "T6_range": [float(mag[fam_indicator==0].min()), float(mag[fam_indicator==0].max())],
            "T7_range": [float(mag[fam_indicator==1].min()), float(mag[fam_indicator==1].max())],
        },
        "epistemic_note": ("Associational, not causal proof of Gamma causing Delta. The "
                           "counterfactual restoration gives causal evidence about the tested "
                           "transformation/restoration mechanism; the Gamma->Delta statistical "
                           "relationship itself is associational (Gamma is measured, not "
                           "independently randomized)."),
        "interpretation_note": ("Raw results only. No automatic classification applied -- "
                                "per explicit instruction, interpretation of these coefficients "
                                "is a separate, deliberate step, not collapsed here."),
    }
    with open(OUT_DIR / "E182_test_C.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nSaved E182_test_C.json")


if __name__ == "__main__":
    main()
