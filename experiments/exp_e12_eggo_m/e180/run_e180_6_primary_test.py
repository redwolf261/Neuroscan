"""
E180 Stage 6 -- Primary test: does Gamma predict Delta?

Pre-registered in docs/phases/PHASE_E180_6_9_ANALYSIS_PREREG.md. Read that
first. Headline result is the SUBJECT FIXED-EFFECTS regression, not pooled
tile-level Spearman -- 868 tiles across 125 subjects are not independent
observations (adjacent tiles share 83-94% enc3 extent). Pooled correlation is
reported only as explicitly-labelled descriptive context.

Two strata, identical specification: A = full 125, B = E167's predefined 110
good subjects (reused verbatim, never redefined from E180 data).

CPU-only, no GPU, no new inference -- operates on the already-computed
E180_gamma_per_subject_full.json / E180_delta_per_subject_full.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from e180_stats import (subject_fixed_effects_ols, reduced_granularity_pairs,
                        sign_concordance_test)
from e180_strata import e167_good_subject_ids

OUT_DIR = Path(__file__).parent
FAMILIES = ["T1_rank", "T4_spectral", "T5_smooth"]
METRICS = ["gamma_dice", "gamma_voxel", "gamma_kl", "gamma_boundary"]


def load_joined():
    gamma = json.load(open(OUT_DIR / "E180_gamma_per_subject_full.json"))["per_family_metric"]
    delta = json.load(open(OUT_DIR / "E180_delta_per_subject_full.json"))
    g_by_key = {(r["sid"], r["window_id"], r["family"]): r for r in gamma}
    d_by_key = {(r["sid"], r["window_id"], r["family"]): r for r in delta}
    keys = sorted(set(g_by_key) & set(d_by_key))
    return g_by_key, d_by_key, keys


def run_stratum(stratum_name, subject_filter, g_by_key, d_by_key, keys, ledger):
    result = {"stratum": stratum_name}
    for fam in FAMILIES:
        fam_keys = [k for k in keys if k[2] == fam and k[0] in subject_filter]
        if len(fam_keys) < 10:
            result[fam] = {"note": "too few tiles in this stratum"}
            continue
        sids = [k[0] for k in fam_keys]
        g = np.array([g_by_key[k]["gamma_dice"] for k in fam_keys])
        d = np.array([d_by_key[k]["delta_i_global"] for k in fam_keys])

        # ---- PRIMARY: subject fixed-effects ----
        fe = subject_fixed_effects_ols(d, g, sids)

        # ---- SECONDARY, explicitly labelled descriptive ----
        rho_pooled, p_pooled = stats.spearmanr(g, d)

        reduced_pairs = {sid: p for sid, p in reduced_granularity_pairs(ledger).items()
                         if sid in subject_filter}
        sign_test = sign_concordance_test(g_by_key, d_by_key, reduced_pairs, fam,
                                          metric="gamma_dice")

        result[fam] = {
            "primary_subject_fixed_effects": {
                "beta_gamma": fe["beta"][0], "se_cluster_robust": fe["se_cluster_robust"][0],
                "t": fe["t"][0], "p": fe["p"][0], "r2_within_model": fe["r2"],
                "n_tiles": fe["n"], "n_subjects": fe["n_subjects"],
            },
            "secondary_descriptive_NOT_EVIDENCE": {
                "pooled_spearman_rho": float(rho_pooled), "pooled_spearman_p": float(p_pooled),
                "note": "NON-INDEPENDENT observations (tile overlap 83-94%), descriptive "
                        "context only, never used as evidence for or against H180",
            },
            "reduced_granularity_robustness": sign_test,
        }
        print(f"[{stratum_name}] {fam}: FE beta={fe['beta'][0]:+.4f} p={fe['p'][0]:.4f} "
              f"(n_tiles={fe['n']}, n_subj={fe['n_subjects']})  "
              f"pooled_rho={rho_pooled:+.3f} (descriptive)  "
              f"sign_concordance={sign_test['n_concordant']}/{sign_test['n_total']} "
              f"(p={sign_test['binomial_p']:.4f})")
    return result


def main():
    g_by_key, d_by_key, keys = load_joined()
    ledger = json.load(open(OUT_DIR / "E180_tile_ledger_full.json"))
    all_subjects = set(ledger.keys())
    good_110 = e167_good_subject_ids() & all_subjects
    print(f"[Data] {len(keys)} joined records, {len(all_subjects)} subjects total, "
          f"{len(good_110)} in E167-good stratum")

    results = {
        "A_full_125_primary": run_stratum("A_full_125", all_subjects, g_by_key, d_by_key,
                                          keys, ledger),
        "B_E167_110_secondary": run_stratum("B_E167_110", good_110, g_by_key, d_by_key,
                                            keys, ledger),
    }

    with open(OUT_DIR / "E180_6_primary_test.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nSaved E180_6_primary_test.json")


if __name__ == "__main__":
    main()
