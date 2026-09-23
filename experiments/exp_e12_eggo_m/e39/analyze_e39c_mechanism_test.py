"""
Phase E39C: mechanism test. Does D4's orthogonal-residual information
(P_4, M_4), summed across the E39A/B controlled lambda sweep's own training
trajectories, actually predict the controlled Dice response (Delta Dice
relative to lambda=0)? n=6 conditions only -- treated throughout as
DESCRIPTIVE evidence, not a powered statistical test. No new training.

Reuses P_4/M_4's exact definitions from E38 (already verified in closed form
against direct vector projection): |g_r^perp| = |g_r|*sqrt(1-cos^2).
P_4 = sum_t |g_4^perp(t)| / (|g_0(t)|+eps)      [E38's own "Q_r", summed not integrated]
M_4 = sum_t |g_4^perp(t)|^2 / (|g_0(t)|^2+eps)  [squared update-energy version, per this session's own derivation]

IMPORTANT: E38 used a TRAPEZOIDAL INTEGRAL (np.trapezoid) over epoch for its
own "P_r" quantity; this phase's prompt asks for a plain SUM over t instead.
These are not identical (trapezoidal integration halves the endpoint
weights), so this script uses a plain sum, exactly as specified here, and
does NOT claim numerical equivalence with E38's own P_r values -- a
different, explicitly-requested aggregation, not a silent redefinition.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT_DIR = Path(__file__).parent
RUNS_DIR = OUT_DIR / "runs"
EPS = 1e-8

CONDITIONS = [
    ("A_lambda0", 0.0), ("lambda_0.125", 0.125), ("lambda_0.25", 0.25),
    ("lambda_0.5", 0.5), ("lambda_1.0", 1.0), ("lambda_2.0", 2.0),
]

# Pre-declared 3-way phase split of the 30-epoch trajectory (0-indexed
# epochs 0-29), NOT chosen after looking at any result -- a natural equal
# three-way split of the fixed 30-epoch schedule every condition shares.
PHASE_EPOCHS = {
    "early": list(range(0, 10)),
    "middle": list(range(10, 20)),
    "late": list(range(20, 30)),
}


def load_trajectory(cond_name):
    path = RUNS_DIR / cond_name / f"E39_gradient_trajectory_{cond_name}.json"
    return json.load(open(path))


def load_best_dice(cond_name):
    csv_path = RUNS_DIR / cond_name / "epoch_metrics.csv"
    import csv
    best = 0.0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = float(row["val_dice"])
            if v > best:
                best = v
    return best


def compute_quantities(traj_records):
    """For every epoch record, compute Q_r(t)=|g4_perp|/(|g0|+eps) and
    M_4(t)=|g4_perp|^2/(|g0|^2+eps), using the record's ALREADY-COMPUTED
    perp_norm_g4 and norm_g0 (both saved directly by the training script,
    not re-derived here) -- avoids any risk of silently recomputing these
    differently from how E39A/B itself measured them."""
    out = []
    for r in traj_records:
        norm_g0 = r["norm_g0"]
        perp = r["perp_norm_g4"]
        Q_t = perp / (norm_g0 + EPS)
        M_t = (perp ** 2) / (norm_g0 ** 2 + EPS)
        out.append({"epoch": r["epoch"], "norm_g0": norm_g0, "norm_g4": r["norm_g4"],
                     "perp_norm_g4": perp, "cosine": r["cosine_g0_g4"], "Q_t": Q_t, "M_t": M_t})
    return out


def main():
    data = {}
    for cond, lam in CONDITIONS:
        traj = load_trajectory(cond)
        q = compute_quantities(traj)
        best_dice = load_best_dice(cond)
        data[cond] = {"lambda": lam, "best_dice": best_dice, "per_epoch": q}

    dice_A = data["A_lambda0"]["best_dice"]
    for cond in data:
        data[cond]["delta_dice"] = data[cond]["best_dice"] - dice_A

    # ================= Full-trajectory P_4, M_4 (plain sum, per spec) =================
    print("=== Full-trajectory P_4 (sum of Q_t) and M_4 (sum of M_t), n=6 conditions ===\n", flush=True)
    for cond, lam in CONDITIONS:
        per_epoch = data[cond]["per_epoch"]
        P4 = sum(r["Q_t"] for r in per_epoch)
        M4 = sum(r["M_t"] for r in per_epoch)
        data[cond]["P4_full"] = P4
        data[cond]["M4_full"] = M4
        print(f"  {cond:14s} lambda={lam:.3f}  best_dice={data[cond]['best_dice']:.4f}  "
              f"Delta_dice={data[cond]['delta_dice']:+.4f}  P4={P4:.3f}  M4={M4:.3f}", flush=True)

    labels = [c for c, _ in CONDITIONS]
    delta_dice = np.array([data[c]["delta_dice"] for c in labels])
    P4_full = np.array([data[c]["P4_full"] for c in labels])
    M4_full = np.array([data[c]["M4_full"] for c in labels])
    lambdas = np.array([lam for _, lam in CONDITIONS])

    results = {"n_conditions": 6, "labels": labels, "lambdas": lambdas.tolist(),
               "delta_dice": delta_dice.tolist(), "P4_full": P4_full.tolist(), "M4_full": M4_full.tolist()}

    # ================= Tests 1-3: Spearman + Pearson, full trajectory =================
    print("\n=== Tests 1-3: correlation vs Delta Dice (n=6, DESCRIPTIVE ONLY) ===", flush=True)
    rho_P4, p_P4 = stats.spearmanr(P4_full, delta_dice)
    rho_M4, p_M4 = stats.spearmanr(M4_full, delta_dice)
    r_P4, pr_P4 = stats.pearsonr(P4_full, delta_dice)
    r_M4, pr_M4 = stats.pearsonr(M4_full, delta_dice)
    print(f"  Spearman(P4, Delta_dice): rho={rho_P4:+.3f} (p={p_P4:.3f}, n=6 -- NOT a powered test)", flush=True)
    print(f"  Spearman(M4, Delta_dice): rho={rho_M4:+.3f} (p={p_M4:.3f}, n=6 -- NOT a powered test)", flush=True)
    print(f"  Pearson(P4, Delta_dice):  r={r_P4:+.3f} (p={pr_P4:.3f})", flush=True)
    print(f"  Pearson(M4, Delta_dice):  r={r_M4:+.3f} (p={pr_M4:.3f})", flush=True)
    results["correlations_full_trajectory"] = {
        "spearman_P4": {"rho": rho_P4, "p": p_P4}, "spearman_M4": {"rho": rho_M4, "p": p_M4},
        "pearson_P4": {"r": r_P4, "p": pr_P4}, "pearson_M4": {"r": r_M4, "p": pr_M4},
    }

    # ================= Test 5: leave-one-out sensitivity =================
    print("\n=== Test 5: leave-one-condition-out sensitivity ===", flush=True)
    loo_results = {}
    for i, held_out in enumerate(labels):
        mask = np.arange(6) != i
        rho_P4_loo, _ = stats.spearmanr(P4_full[mask], delta_dice[mask])
        rho_M4_loo, _ = stats.spearmanr(M4_full[mask], delta_dice[mask])
        loo_results[held_out] = {"rho_P4_loo": float(rho_P4_loo), "rho_M4_loo": float(rho_M4_loo)}
        print(f"  excl {held_out:14s}: rho(P4,dDice)={rho_P4_loo:+.3f}  rho(M4,dDice)={rho_M4_loo:+.3f}", flush=True)
    results["leave_one_out"] = loo_results

    # ================= Test 6: is lambda=0.25 mechanistically distinguished? =================
    print("\n=== Test 6: is lambda=0.25 (best Dice) mechanistically distinguished from 0.5/1.0? ===", flush=True)
    for cond in ["lambda_0.25", "lambda_0.5", "lambda_1.0"]:
        print(f"  {cond}: P4={data[cond]['P4_full']:.3f}  M4={data[cond]['M4_full']:.3f}  "
              f"Delta_dice={data[cond]['delta_dice']:+.4f}", flush=True)
    results["test6_lambda025_vs_neighbors"] = {
        c: {"P4": data[c]["P4_full"], "M4": data[c]["M4_full"], "delta_dice": data[c]["delta_dice"]}
        for c in ["lambda_0.25", "lambda_0.5", "lambda_1.0"]
    }

    # ================= Test 7: inverted-U structure comparison =================
    print("\n=== Test 7: does Dice's inverted-U shape match P4/M4's shape across lambda? ===", flush=True)
    dice_order_by_lambda = [data[c]["best_dice"] for c, _ in CONDITIONS]
    print(f"  Dice by lambda:  {[round(v,4) for v in dice_order_by_lambda]}", flush=True)
    print(f"  P4 by lambda:    {[round(v,2) for v in P4_full]}", flush=True)
    print(f"  M4 by lambda:    {[round(v,2) for v in M4_full]}", flush=True)
    dice_argmax = int(np.argmax(dice_order_by_lambda))
    P4_argmax = int(np.argmax(P4_full))
    M4_argmax = int(np.argmax(M4_full))
    print(f"  Dice peaks at: {labels[dice_argmax]}  |  P4 peaks at: {labels[P4_argmax]}  |  M4 peaks at: {labels[M4_argmax]}", flush=True)
    results["test7_peak_comparison"] = {
        "dice_peak": labels[dice_argmax], "P4_peak": labels[P4_argmax], "M4_peak": labels[M4_argmax],
    }

    # ================= Phase-decomposed P4/M4 (early/middle/late) =================
    print("\n=== Phase-decomposed P4/M4 (pre-declared 3-way split: epochs 0-9/10-19/20-29) ===", flush=True)
    phase_results = {}
    for cond, lam in CONDITIONS:
        per_epoch = data[cond]["per_epoch"]
        phases = {}
        for phase_name, ep_list in PHASE_EPOCHS.items():
            recs = [r for r in per_epoch if r["epoch"] in ep_list]
            P4_phase = sum(r["Q_t"] for r in recs)
            M4_phase = sum(r["M_t"] for r in recs)
            phases[phase_name] = {"P4": P4_phase, "M4": M4_phase}
        phase_results[cond] = phases
        print(f"  {cond:14s} P4: early={phases['early']['P4']:.3f} mid={phases['middle']['P4']:.3f} "
              f"late={phases['late']['P4']:.3f}   M4: early={phases['early']['M4']:.3f} "
              f"mid={phases['middle']['M4']:.3f} late={phases['late']['M4']:.3f}", flush=True)
    results["phase_decomposed"] = phase_results

    print("\n=== Correlation vs Delta Dice, BY PHASE (n=6, descriptive) ===", flush=True)
    phase_corr = {}
    for phase_name in ("early", "middle", "late"):
        P4_phase_vals = np.array([phase_results[c]["early" if phase_name == "early" else phase_name]["P4"] for c in labels])
        M4_phase_vals = np.array([phase_results[c][phase_name]["M4"] for c in labels])
        rho_P4_p, p_P4_p = stats.spearmanr(P4_phase_vals, delta_dice)
        rho_M4_p, p_M4_p = stats.spearmanr(M4_phase_vals, delta_dice)
        phase_corr[phase_name] = {
            "spearman_P4": {"rho": float(rho_P4_p), "p": float(p_P4_p)},
            "spearman_M4": {"rho": float(rho_M4_p), "p": float(p_M4_p)},
        }
        print(f"  [{phase_name}] rho(P4,dDice)={rho_P4_p:+.3f} (p={p_P4_p:.3f})   "
              f"rho(M4,dDice)={rho_M4_p:+.3f} (p={p_M4_p:.3f})", flush=True)
    results["phase_correlations"] = phase_corr

    # ================= Confound check: does M4 track |g0| collapsing, not real signal? =================
    print("\n=== Confound check: |g0|, |g4_perp|, M4 reported SEPARATELY, across late-training epochs ===", flush=True)
    confound_check = {}
    for cond, lam in CONDITIONS:
        late_recs = [r for r in data[cond]["per_epoch"] if r["epoch"] in PHASE_EPOCHS["late"]]
        mean_g0 = float(np.mean([r["norm_g0"] for r in late_recs]))
        mean_g4perp = float(np.mean([r["perp_norm_g4"] for r in late_recs]))
        mean_M4 = float(np.mean([r["M_t"] for r in late_recs]))
        confound_check[cond] = {"mean_g0_late": mean_g0, "mean_g4perp_late": mean_g4perp, "mean_M4_late": mean_M4}
        print(f"  {cond:14s} late-phase: mean|g0|={mean_g0:.4f}  mean|g4_perp|={mean_g4perp:.4f}  mean_M4={mean_M4:.4f}", flush=True)

    # Does M4's cross-condition variation correlate with |g0| SHRINKING (a
    # negative correlation between M4 and |g0| across conditions would be
    # the confound signature -- M4 high BECAUSE g0 is small, not because
    # g4_perp is genuinely large)
    g0_lates = np.array([confound_check[c]["mean_g0_late"] for c in labels])
    g4perp_lates = np.array([confound_check[c]["mean_g4perp_late"] for c in labels])
    M4_lates = np.array([confound_check[c]["mean_M4_late"] for c in labels])
    rho_M4_vs_g0, p_M4_vs_g0 = stats.spearmanr(M4_lates, g0_lates)
    rho_M4_vs_g4perp, p_M4_vs_g4perp = stats.spearmanr(M4_lates, g4perp_lates)
    print(f"\n  Spearman(M4_late, |g0|_late) across conditions: rho={rho_M4_vs_g0:+.3f} (p={p_M4_vs_g0:.3f}) "
          f"-- a STRONG NEGATIVE value here would indicate M4 is driven by g0 collapsing, not real residual signal", flush=True)
    print(f"  Spearman(M4_late, |g4_perp|_late) across conditions: rho={rho_M4_vs_g4perp:+.3f} (p={p_M4_vs_g4perp:.3f}) "
          f"-- should be STRONGLY POSITIVE if M4 tracks genuine residual magnitude", flush=True)
    results["confound_check"] = {
        "per_condition_late_phase": confound_check,
        "spearman_M4_vs_g0": {"rho": float(rho_M4_vs_g0), "p": float(p_M4_vs_g0)},
        "spearman_M4_vs_g4perp": {"rho": float(rho_M4_vs_g4perp), "p": float(p_M4_vs_g4perp)},
    }

    with open(OUT_DIR / "E39C_mechanism_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved E39C_mechanism_test_results.json", flush=True)


if __name__ == "__main__":
    main()
