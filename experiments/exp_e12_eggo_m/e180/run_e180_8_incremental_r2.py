"""
E180 Stage 8 -- The decisive gate: incremental information.

Pre-registered in docs/phases/PHASE_E180_6_9_ANALYSIS_PREREG.md. Read that
first. Nested subject fixed-effects models:
  M0 = subject FE only (basic controls)
  M1 = M0 + Uncertainty
  M2 = M1 + {Magnitude, Perturbation magnitude, VJP sensitivity}
  M3 = M2 + Gamma

Decisive quantities:
  delta_R2_Gamma        = R2(M3) - R2(M2)
  delta_R2_Gamma_given_U = R2(Gamma added to M1) - R2(M1)   -- Gamma over
                            uncertainty ALONE, the exact comparison the
                            novelty audit's narrow claim depends on.

Permutation null respects subject structure: Gamma is shuffled WITHIN each
subject's own tiles, never across all 2604 rows as IID.

CPU-only, no GPU. Operates on already-computed Gamma/Delta/competitor files.
"""
import json
from pathlib import Path

import numpy as np

from e180_stats import subject_fixed_effects_ols, within_subject_permutation_null
from e180_strata import e167_good_subject_ids

OUT_DIR = Path(__file__).parent
FAMILIES = ["T1_rank", "T4_spectral", "T5_smooth"]
N_PERM = 1000


def load_all():
    gamma = json.load(open(OUT_DIR / "E180_gamma_per_subject_full.json"))["per_family_metric"]
    delta = json.load(open(OUT_DIR / "E180_delta_per_subject_full.json"))
    comp = json.load(open(OUT_DIR / "E180_7_competitor_features.json"))

    g_by_key = {(r["sid"], r["window_id"], r["family"]): r for r in gamma}
    d_by_key = {(r["sid"], r["window_id"], r["family"]): r for r in delta}
    c_by_tile = {(r["sid"], r["window_id"]): r for r in comp}
    return g_by_key, d_by_key, c_by_tile


def build_design_matrix(fam, subject_filter, g_by_key, d_by_key, c_by_tile):
    """Returns (y, X, col_names, sids) for one family, one stratum. X columns:
    [magnitude, pert_mag_<fam>, uncertainty(entropy), sensitivity, gamma_dice]."""
    keys = sorted(k for k in g_by_key if k[2] == fam and k[0] in subject_filter)
    keys = [k for k in keys if k in d_by_key and (k[0], k[1]) in c_by_tile]

    y, rows, sids = [], [], []
    for k in keys:
        sid, wid, _ = k
        comp = c_by_tile[(sid, wid)]
        y.append(d_by_key[k]["delta_i_global"])
        rows.append([
            comp["magnitude"],
            comp[f"pert_mag_{fam}"],
            comp["entropy"],
            comp["sensitivity"],
            g_by_key[k]["gamma_dice"],
        ])
        sids.append(sid)
    return (np.array(y), np.array(rows),
            ["magnitude", "pert_mag", "uncertainty", "sensitivity", "gamma"], sids)


def run_nested(y, X, col_names, sids):
    idx = {name: i for i, name in enumerate(col_names)}

    def cols(*names):
        return X[:, [idx[n] for n in names]]

    m0 = subject_fixed_effects_ols(y, np.zeros((len(y), 1)), sids)  # subject FE only
    m1 = subject_fixed_effects_ols(y, cols("uncertainty"), sids)
    m2 = subject_fixed_effects_ols(y, cols("uncertainty", "magnitude", "pert_mag",
                                            "sensitivity"), sids)
    m3 = subject_fixed_effects_ols(y, cols("uncertainty", "magnitude", "pert_mag",
                                            "sensitivity", "gamma"), sids)

    delta_r2_gamma = m3["r2"] - m2["r2"]

    # Gamma over uncertainty ALONE (M1 -> M1+Gamma)
    X_u_gamma = cols("uncertainty", "gamma")
    null_u = within_subject_permutation_null(y, X_u_gamma, sids, gamma_col_idx=1,
                                             n_perm=N_PERM, seed=0)

    # Gamma over the full competitor set M2 -> M3
    X_full = cols("uncertainty", "magnitude", "pert_mag", "sensitivity", "gamma")
    null_full = within_subject_permutation_null(y, X_full, sids, gamma_col_idx=4,
                                                 n_perm=N_PERM, seed=0)

    # decision rule
    kill = null_u["permutation_p"] > 0.05
    weak = (not kill) and null_full["permutation_p"] > 0.05
    verdict = "KILL" if kill else ("WEAK" if weak else "SURVIVES")

    return {
        "r2_M0_subject_FE_only": m0["r2"],
        "r2_M1_plus_uncertainty": m1["r2"],
        "r2_M2_plus_competitors": m2["r2"],
        "r2_M3_plus_gamma": m3["r2"],
        "delta_R2_gamma_over_full_M2": float(delta_r2_gamma),
        "gamma_over_uncertainty_alone": null_u,
        "gamma_over_full_competitors": null_full,
        "n_tiles": len(y), "n_subjects": len(set(sids)),
        "PREREGISTERED_VERDICT": verdict,
        "decision_rule": "KILL if gamma-over-uncertainty-alone perm p>0.05; "
                         "WEAK if that survives but gamma-over-full-M2 does not; "
                         "else SURVIVES",
    }


def main():
    g_by_key, d_by_key, c_by_tile = load_all()
    ledger = json.load(open(OUT_DIR / "E180_tile_ledger_full.json"))
    all_subjects = set(ledger.keys())
    good_110 = e167_good_subject_ids() & all_subjects

    results = {"A_full_125": {}, "B_E167_110": {}}
    for stratum_name, subj_filter in [("A_full_125", all_subjects), ("B_E167_110", good_110)]:
        for fam in FAMILIES:
            y, X, col_names, sids = build_design_matrix(fam, subj_filter, g_by_key,
                                                         d_by_key, c_by_tile)
            if len(y) < 20:
                results[stratum_name][fam] = {"note": "too few tiles"}
                continue
            res = run_nested(y, X, col_names, sids)
            results[stratum_name][fam] = res
            print(f"[{stratum_name}] {fam}: R2(M2)={res['r2_M2_plus_competitors']:.4f} "
                  f"R2(M3)={res['r2_M3_plus_gamma']:.4f} "
                  f"dR2(gamma|M2)={res['delta_R2_gamma_over_full_M2']:+.4f}  "
                  f"gamma|uncertainty perm_p={res['gamma_over_uncertainty_alone']['permutation_p']:.4f}  "
                  f"gamma|M2 perm_p={res['gamma_over_full_competitors']['permutation_p']:.4f}  "
                  f"VERDICT={res['PREREGISTERED_VERDICT']}")

    with open(OUT_DIR / "E180_8_incremental_r2.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nSaved E180_8_incremental_r2.json")


if __name__ == "__main__":
    main()
