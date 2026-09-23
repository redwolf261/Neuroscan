"""
E180 Stage 11 -- T4_spectral coupling test.

Pre-registered in docs/phases/PHASE_E180_10_11_T4_DOSE_RESPONSE_PREREG.md.
Read that first. Three pre-registered tests, all subject-aware (cluster by
subject, never pooled-tile statistics as evidence):

  Test 1: monotonicity of Delta as alpha increases
  Test 2: monotonicity of Gamma (decreasing) as alpha increases
  Test 3: coupling -- does d(Gamma) predict d(Dice) across dose STEPS,
          not just Gamma predicting Delta at one point

CPU-only, no GPU. Operates on E180_10_dose_response.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from e180_stats import subject_fixed_effects_ols, within_subject_permutation_null
from e180_strata import e167_good_subject_ids

OUT_DIR = Path(__file__).parent
ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]
N_PERM = 1000


def load_by_tile():
    records = json.load(open(OUT_DIR / "E180_10_dose_response.json"))
    by_tile = {}
    for r in records:
        by_tile.setdefault((r["sid"], r["window_id"]), {})[r["alpha"]] = r
    return by_tile


def test1_monotonicity_delta(by_tile, subject_filter):
    """Fraction of tiles with non-decreasing Dice as alpha increases (allowing
    ONE non-monotonic step for noise), plus a subject-aware trend regression
    of dice_at_dose on alpha."""
    frac_monotonic = []
    y_all, x_all, sids_all = [], [], []
    for (sid, wid), doses in by_tile.items():
        if sid not in subject_filter:
            continue
        vals = [doses[a]["dice_at_dose"] for a in ALPHAS if a in doses]
        if len(vals) != len(ALPHAS):
            continue
        diffs = np.diff(vals)
        n_violations = int((diffs < -1e-6).sum())
        frac_monotonic.append(n_violations <= 1)
        for a, v in zip(ALPHAS, vals):
            y_all.append(v)
            x_all.append(a)
            sids_all.append(sid)

    trend = subject_fixed_effects_ols(np.array(y_all), np.array(x_all), sids_all)
    return {
        "frac_tiles_monotonic_allow1violation": float(np.mean(frac_monotonic)),
        "n_tiles": len(frac_monotonic),
        "subject_aware_trend_beta": trend["beta"][0], "trend_p": trend["p"][0],
    }


def test2_monotonicity_gamma(by_tile, subject_filter):
    """Fraction of tiles with non-increasing Gamma as alpha increases, plus
    a subject-aware trend regression of gamma_at_dose on alpha (expect
    NEGATIVE beta -- instability should fall as restoration increases)."""
    frac_monotonic = []
    y_all, x_all, sids_all = [], [], []
    for (sid, wid), doses in by_tile.items():
        if sid not in subject_filter:
            continue
        vals = [doses[a]["gamma_at_dose"] for a in ALPHAS if a in doses]
        if len(vals) != len(ALPHAS):
            continue
        diffs = np.diff(vals)
        n_violations = int((diffs > 1e-6).sum())
        frac_monotonic.append(n_violations <= 1)
        for a, v in zip(ALPHAS, vals):
            y_all.append(v)
            x_all.append(a)
            sids_all.append(sid)

    trend = subject_fixed_effects_ols(np.array(y_all), np.array(x_all), sids_all)
    return {
        "frac_tiles_monotonic_decreasing_allow1violation": float(np.mean(frac_monotonic)),
        "n_tiles": len(frac_monotonic),
        "subject_aware_trend_beta": trend["beta"][0], "trend_p": trend["p"][0],
        "expected_sign": "negative (Gamma should fall as alpha/restoration increases)",
    }


def test3_coupling(by_tile, subject_filter):
    """d(Gamma) predicts d(Dice) across dose STEPS. Pools the 4 consecutive
    dose-step differences (alpha 0->.25, .25->.5, .5->.75, .75->1.0) per
    tile, each step treated as one observation, clustered by subject (not
    by tile*step, since steps within a tile/subject are still correlated
    with the subject's overall difficulty -- subject clustering is the
    conservative choice here, consistent with every other E180 stage)."""
    y, x, sids = [], [], []
    for (sid, wid), doses in by_tile.items():
        if sid not in subject_filter:
            continue
        vals = {a: doses[a] for a in ALPHAS if a in doses}
        if len(vals) != len(ALPHAS):
            continue
        for a0, a1 in zip(ALPHAS[:-1], ALPHAS[1:]):
            d_gamma = vals[a1]["gamma_at_dose"] - vals[a0]["gamma_at_dose"]
            d_dice = vals[a1]["dice_at_dose"] - vals[a0]["dice_at_dose"]
            x.append(d_gamma)
            y.append(d_dice)
            sids.append(sid)

    y = np.array(y)
    x = np.array(x)
    fe = subject_fixed_effects_ols(y, x, sids)
    X2 = x.reshape(-1, 1)
    null = within_subject_permutation_null(y, X2, sids, gamma_col_idx=0,
                                           n_perm=N_PERM, seed=0)
    # expected sign: d_gamma DECREASING (negative) should predict d_dice
    # INCREASING (positive) -- i.e. beta should be NEGATIVE
    expected_sign_correct = fe["beta"][0] < 0

    return {
        "beta_d_dice_on_d_gamma": fe["beta"][0], "se": fe["se_cluster_robust"][0],
        "p_from_FE": fe["p"][0],
        "permutation_p": null["permutation_p"], "n_perm": N_PERM,
        "n_observations": len(y), "n_subjects": len(set(sids)),
        "expected_sign": "negative (d_Gamma down should predict d_Dice up)",
        "expected_sign_correct": expected_sign_correct,
    }


def classify(t1, t2, t3):
    perm_p = t3["permutation_p"]
    p_display = f"< {1/t3['n_perm']:.3f}" if perm_p == 0 else f"{perm_p:.4f}"
    t1_ok = t1["frac_tiles_monotonic_allow1violation"] >= 0.7 and t1["trend_p"] < 0.05
    t2_ok = t2["frac_tiles_monotonic_decreasing_allow1violation"] >= 0.7 and t2["trend_p"] < 0.05
    t3_ok = perm_p < 0.05 and t3["expected_sign_correct"]

    if t1_ok and t2_ok and t3_ok:
        verdict = "STRONGEST_graded_dose_dependent_coupling"
    elif t3_ok:
        verdict = "CORRELATE_ONLY_coupling_significant_but_irregular_monotonicity"
    else:
        verdict = "NULL_dose_response_does_not_hold"
    return verdict, p_display


def main():
    by_tile = load_by_tile()
    ledger = json.load(open(OUT_DIR / "E180_tile_ledger_full.json"))
    all_subjects = set(ledger.keys())
    good_110 = e167_good_subject_ids() & all_subjects

    results = {}
    for stratum_name, subj_filter in [("A_full_125", all_subjects), ("B_E167_110", good_110)]:
        t1 = test1_monotonicity_delta(by_tile, subj_filter)
        t2 = test2_monotonicity_gamma(by_tile, subj_filter)
        t3 = test3_coupling(by_tile, subj_filter)
        verdict, p_display = classify(t1, t2, t3)

        results[stratum_name] = {
            "test1_delta_monotonicity": t1,
            "test2_gamma_monotonicity": t2,
            "test3_coupling": t3,
            "PREREGISTERED_VERDICT": verdict,
        }
        print(f"[{stratum_name}]")
        print(f"  Test 1 (Delta monotonic up): frac={t1['frac_tiles_monotonic_allow1violation']:.3f} "
              f"trend_beta={t1['subject_aware_trend_beta']:+.4f} p={t1['trend_p']:.4f}")
        print(f"  Test 2 (Gamma monotonic down): frac={t2['frac_tiles_monotonic_decreasing_allow1violation']:.3f} "
              f"trend_beta={t2['subject_aware_trend_beta']:+.4f} p={t2['trend_p']:.4f}")
        print(f"  Test 3 (coupling d_Gamma -> d_Dice): beta={t3['beta_d_dice_on_d_gamma']:+.4f} "
              f"perm_p={p_display}  sign_correct={t3['expected_sign_correct']}")
        print(f"  VERDICT: {verdict}\n")

    with open(OUT_DIR / "E180_11_coupling_test.json", "w") as f:
        json.dump(results, f, indent=1)
    print("Saved E180_11_coupling_test.json")


if __name__ == "__main__":
    main()
