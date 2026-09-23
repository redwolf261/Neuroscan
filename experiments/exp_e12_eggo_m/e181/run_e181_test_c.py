"""
E181 Test C -- the decisive magnitude-conditional comparison.

Pre-registered in docs/phases/PHASE_E181_MECHANISM_DISSECTION_PREREG.md.
Read that first. T6 budgets and T7 levels are NOT compared by raw label --
this conditions explicitly on the measured uniform_magnitude covariate.

Model: delta_i_global ~ gamma_dice + uniform_magnitude + transform_family
(subject fixed effects, cluster-robust SE), pooling both severities within
each family. transform_family is a 0/1 indicator (T6=0, T7=1). Also reports
Gamma's own coefficient SEPARATELY within each family after magnitude is in
the model, to see whether T6's relationship survives magnitude control while
T7's does not (or vice versa, or neither).

Associational, not causal proof (stated in the prereg and repeated in the
output) -- Gamma is measured, not independently randomized.

CPU-only, no GPU. Operates on FROZEN E181-D data.
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
    gamma = json.load(open(FROZEN_DIR / "E181_D_gamma.json"))
    delta = json.load(open(FROZEN_DIR / "E181_D_delta.json"))
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

    # ---- classification per the pre-registered reading rule ----
    t6_survives = within_family["T6"]["SURVIVES_MAGNITUDE_CONTROL"]
    t7_survives = within_family["T7"]["SURVIVES_MAGNITUDE_CONTROL"]
    if t6_survives and not t7_survives:
        classification = "H2_SUPPORTED -- T6 survives magnitude control, T7 does not"
    elif t6_survives and t7_survives:
        classification = "BOTH_SURVIVE -- phenomenon broader than spectral shape, do not force H2"
    elif not t6_survives and not t7_survives:
        classification = "NEITHER_SURVIVES -- raw associations may be explained by generic perturbation magnitude"
    else:
        classification = "T7_SUPPORTED_NOT_T6 -- unexpected pattern, report honestly"

    print(f"\n[CLASSIFICATION] {classification}")

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
        "magnitude_context": {
            "T6_range": [float(mag[fam_indicator==0].min()), float(mag[fam_indicator==0].max())],
            "T7_range": [float(mag[fam_indicator==1].min()), float(mag[fam_indicator==1].max())],
        },
        "CLASSIFICATION": classification,
        "epistemic_note": ("Associational, not causal proof of Gamma causing Delta. The "
                           "counterfactual restoration gives causal evidence about the tested "
                           "transformation/restoration mechanism; the Gamma->Delta statistical "
                           "relationship itself is associational (Gamma is measured, not "
                           "independently randomized)."),
    }
    with open(OUT_DIR / "E181_test_C.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nSaved E181_test_C.json")


if __name__ == "__main__":
    main()
