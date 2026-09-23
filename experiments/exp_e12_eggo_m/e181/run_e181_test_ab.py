"""
E181 Test A/B -- does T6 (shape-only) / T7 (energy-only) reproduce E180's
Gamma->Delta relationship?

Pre-registered in docs/phases/PHASE_E181_MECHANISM_DISSECTION_PREREG.md
("Test A/B/C -- pre-specified analysis" section). Read that first. Fixed
BEFORE running: subject-aware framework (subject fixed effects, cluster-
robust by subject), pooled Spearman reported only as labelled-descriptive
context, never as evidence. Operates on the FROZEN E181-D data
(experiments/exp_e12_eggo_m/e181/FROZEN/), never regenerated.

CPU-only, no GPU, no new inference.
"""
import sys
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT_DIR = Path(__file__).parent
FROZEN_DIR = OUT_DIR / "FROZEN"
e180_dir = OUT_DIR.parent / "e180"
sys.path.insert(0, str(e180_dir))

from e180_stats import subject_fixed_effects_ols  # noqa: E402


def load_joined():
    gamma = json.load(open(FROZEN_DIR / "E181_D_gamma.json"))
    delta = json.load(open(FROZEN_DIR / "E181_D_delta.json"))
    g_by_key = {(r["sid"], r["window_id"], r["transform"], r["severity"]): r for r in gamma}
    d_by_key = {(r["sid"], r["window_id"], r["transform"], r["severity"]): r for r in delta}
    keys = sorted(set(g_by_key) & set(d_by_key))
    return g_by_key, d_by_key, keys


def run_cell(transform, severity, g_by_key, d_by_key, keys):
    fam_keys = [k for k in keys if k[2] == transform and k[3] == severity]
    sids = [k[0] for k in fam_keys]
    gam = np.array([g_by_key[k]["gamma_dice"] for k in fam_keys])
    delt = np.array([d_by_key[k]["delta_i_global"] for k in fam_keys])

    fe = subject_fixed_effects_ols(delt, gam, sids)
    rho, p_rho = stats.spearmanr(gam, delt)

    return {
        "transform": transform, "severity": severity,
        "n_tiles": fe["n"], "n_subjects": fe["n_subjects"],
        "primary_subject_fixed_effects": {
            "beta_gamma": fe["beta"][0], "se_cluster_robust": fe["se_cluster_robust"][0],
            "t": fe["t"][0], "p": fe["p"][0], "r2_within_model": fe["r2"],
        },
        "secondary_descriptive_NOT_EVIDENCE": {
            "pooled_spearman_rho": float(rho), "pooled_spearman_p": float(p_rho),
            "note": "NON-INDEPENDENT observations (tile overlap), descriptive context only",
        },
    }


def main():
    g_by_key, d_by_key, keys = load_joined()
    print(f"[Data] {len(keys)} joined records from FROZEN E181-D")

    results = {"Test_A_T6": {}, "Test_B_T7": {}}
    for sev in [0.35, 0.5]:
        res = run_cell("T6", sev, g_by_key, d_by_key, keys)
        results["Test_A_T6"][str(sev)] = res
        fe = res["primary_subject_fixed_effects"]
        print(f"[Test A] T6 sev={sev}: beta={fe['beta_gamma']:+.4f} p={fe['p']:.4f} "
              f"(n_tiles={res['n_tiles']}, n_subj={res['n_subjects']})  "
              f"pooled_rho={res['secondary_descriptive_NOT_EVIDENCE']['pooled_spearman_rho']:+.3f} (descriptive)")

    for sev in [0.7, 1.3]:
        res = run_cell("T7", sev, g_by_key, d_by_key, keys)
        results["Test_B_T7"][str(sev)] = res
        fe = res["primary_subject_fixed_effects"]
        print(f"[Test B] T7 sev={sev}: beta={fe['beta_gamma']:+.4f} p={fe['p']:.4f} "
              f"(n_tiles={res['n_tiles']}, n_subj={res['n_subjects']})  "
              f"pooled_rho={res['secondary_descriptive_NOT_EVIDENCE']['pooled_spearman_rho']:+.3f} (descriptive)")

    with open(OUT_DIR / "E181_test_A_B.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nSaved E181_test_A_B.json")


if __name__ == "__main__":
    main()
