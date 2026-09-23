"""
Phase E69: Offline mining of E54's existing 3 seeds -- no new training.

CONTEXT: PHASE_E45_E54_3SEED_CONFIRMATION_RESULT.md found E54's 3-seed
mean per-subject Dice (0.8944) exceeds the corrected target (0.8942) by
only 0.0002 -- roughly a tenth of E54's own between-seed std (0.0032).
Per the user's own staged-compute-gate proposal, before spending ~24
GPU-hours on 7 more seeds, mine the ALREADY-TRAINED 3 seeds (seed0,
seed1, seed2, all on disk, zero new compute) for whether there is a
COHERENT phenomenon (heterogeneous-but-real signal) or the delta looks
like pure noise scattered around zero.

Uses EXISTING artifacts only:
  - experiments/exp_e12_eggo_m/e56/E56_per_subject_rescoring.json
    (baseline_A seed0, e54_A96 seed0 per-subject Dice)
  - experiments/exp_e12_eggo_m/e56/E45_E54_seed12_rescoring.json
    (e54_A96 seed1, seed2 per-subject Dice)
  - experiments/exp_e12_eggo_m/e48/E48_encoding_audit_table.json
    (native_size per subject, for size-stratification -- reused
    verbatim, not re-derived)
  - experiments/exp_e12_eggo_m/e54/runs/A96_seed{0,1,2}/epoch_metrics.csv
    (training trajectories)
  - experiments/exp_e12_eggo_m/e24/gate6_runs/A_baseline_seed0/epoch_metrics.csv
    (matched baseline trajectory)

ANALYSES (all pre-declared before viewing results, matching this
project's own discipline):
  1. Per-seed paired subject-level deltas (E54 - baseline_A), same
     baseline reused for all 3 seeds (matches this project's own
     established single-baseline-reference convention, E56).
  2. Per-seed summary: mean, median, std, fraction improved/degraded,
     fraction with |delta|>0.05 (a "meaningfully changed" threshold,
     declared here not tuned post-hoc).
  3. Consistency check: for each subject, is the SIGN of delta the same
     across all 3 seeds? (coherent-subject-level-effect test -- if a
     substantial fraction of subjects consistently improve/degrade
     across all 3 independently-trained seeds, that is a real signal
     a pure-noise story cannot explain).
  4. Size-stratified delta: does delta correlate with native lesion
     size (Spearman + permutation, matching E48's own convention),
     computed per seed and pooled across all 3 seeds x 125 subjects.
  5. Training-trajectory comparison: val_dice curves, convergence
     speed (epoch of first val_dice > 0.85), and epoch-to-epoch
     variance for E54's 3 seeds vs. the single matched baseline
     trajectory.

DECISION RULE (three-way, pre-declared):
  A. No coherent per-subject pattern (subject-level sign not consistent
     across seeds, no significant size correlation, trajectories not
     visibly different from baseline) -> the +0.0002 mean looks like
     noise around zero. KILL, do not spend the 24 GPU-hours.
  B. A coherent subject-level or size-stratified pattern exists but is
     modest -> WORTH a cheap follow-up (not yet defined here), short of
     the full 7-seed replication.
  C. A strong, coherent, size-structured pattern with large consistent
     per-subject effects -> the 7-additional-seed replication is
     justified; report exactly what pattern justifies it.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
OUT_DIR = Path(__file__).parent
SEED = 0
N_PERM = 1000

E56_TABLE = project_root / "experiments" / "exp_e12_eggo_m" / "e56" / "E56_per_subject_rescoring.json"
SEED12_TABLE = project_root / "experiments" / "exp_e12_eggo_m" / "e56" / "E45_E54_seed12_rescoring.json"
E48_TABLE = project_root / "experiments" / "exp_e12_eggo_m" / "e48" / "E48_encoding_audit_table.json"

E54_TRAJ = {
    0: project_root / "experiments" / "exp_e12_eggo_m" / "e54" / "runs" / "A96_seed0" / "epoch_metrics.csv",
    1: project_root / "experiments" / "exp_e12_eggo_m" / "e54" / "runs" / "A96_seed1" / "epoch_metrics.csv",
    2: project_root / "experiments" / "exp_e12_eggo_m" / "e54" / "runs" / "A96_seed2" / "epoch_metrics.csv",
}
BASELINE_TRAJ = (project_root / "experiments" / "exp_e12_eggo_m" / "e24" / "gate6_runs"
                  / "A_baseline_seed0" / "epoch_metrics.csv")


def load_csv_column(path, col_name):
    import csv
    vals = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals.append(float(row[col_name]))
    return np.array(vals)


def main():
    with open(E56_TABLE) as f:
        e56 = json.load(f)
    with open(SEED12_TABLE) as f:
        seed12 = json.load(f)
    with open(E48_TABLE) as f:
        e48_records = json.load(f)
    native_size_by_id = {r["subject_id"]: r["native_size"] for r in e48_records}

    baseline_per_subject = e56["baseline_A"]["per_subject_dice"]
    e54_per_seed = {
        0: e56["e54_A96"]["per_subject_dice"],
        1: seed12["e54_A96_s1"]["per_subject_dice"],
        2: seed12["e54_A96_s2"]["per_subject_dice"],
    }

    subject_ids = sorted(baseline_per_subject.keys())
    print(f"n subjects (baseline) = {len(subject_ids)}")
    for s in (0, 1, 2):
        missing = set(subject_ids) - set(e54_per_seed[s].keys())
        assert not missing, f"seed{s}: missing subjects {missing}"
    print("All 3 seeds cover the same 125 subjects as baseline -- verified.\n")

    # ================= 1-2. Per-seed paired deltas =================
    deltas_by_seed = {}
    print("=== Per-seed summary (E54 - baseline_A, paired per-subject) ===")
    for s in (0, 1, 2):
        deltas = np.array([e54_per_seed[s][sid] - baseline_per_subject[sid] for sid in subject_ids])
        deltas_by_seed[s] = deltas
        frac_improved = float((deltas > 0).mean())
        frac_degraded = float((deltas < 0).mean())
        frac_meaningful = float((np.abs(deltas) > 0.05).mean())
        print(f"  seed{s}: mean={deltas.mean():+.5f} median={np.median(deltas):+.5f} std={deltas.std():.5f} "
              f"frac_improved={frac_improved:.3f} frac_degraded={frac_degraded:.3f} "
              f"frac_|delta|>0.05={frac_meaningful:.3f}")

    # ================= 3. Cross-seed sign-consistency (coherent subject-level effect?) =================
    print("\n=== Cross-seed sign consistency (does the same subject improve/degrade across all 3 seeds?) ===")
    all_same_sign_positive = 0
    all_same_sign_negative = 0
    mixed_sign = 0
    per_subject_deltas = {}
    for sid in subject_ids:
        d = [e54_per_seed[s][sid] - baseline_per_subject[sid] for s in (0, 1, 2)]
        per_subject_deltas[sid] = d
        signs = set(np.sign(d))
        if signs == {1.0} or (1.0 in signs and 0.0 in signs and -1.0 not in signs):
            all_same_sign_positive += 1
        elif signs == {-1.0} or (-1.0 in signs and 0.0 in signs and 1.0 not in signs):
            all_same_sign_negative += 1
        else:
            mixed_sign += 1
    n = len(subject_ids)
    print(f"  Consistently IMPROVED across all 3 seeds: {all_same_sign_positive}/{n} ({all_same_sign_positive/n:.3f})")
    print(f"  Consistently DEGRADED across all 3 seeds: {all_same_sign_negative}/{n} ({all_same_sign_negative/n:.3f})")
    print(f"  Mixed sign across seeds: {mixed_sign}/{n} ({mixed_sign/n:.3f})")
    # Null expectation under independent coin-flips (p=0.5 per seed, if truly noise around 0):
    # P(all 3 same sign) = 2 * 0.5^3 = 0.25 expected by chance alone.
    expected_consistent_frac = 0.25
    print(f"  (Null/chance expectation under pure noise, independent per-seed coin flips: "
          f"~{expected_consistent_frac:.3f} of subjects would show all-same-sign by chance alone)")

    # Binomial test: is the observed consistent fraction higher than chance?
    n_consistent = all_same_sign_positive + all_same_sign_negative
    binom_p = stats.binomtest(n_consistent, n, expected_consistent_frac, alternative="greater").pvalue
    print(f"  Binomial test (observed consistent count vs. chance rate): p={binom_p:.4f}")

    # ================= 4. Size-stratified delta =================
    print("\n=== Size-dependence of delta (pooled across 3 seeds x 125 subjects, n=375) ===")
    pooled_deltas = []
    pooled_sizes = []
    for s in (0, 1, 2):
        for sid in subject_ids:
            pooled_deltas.append(e54_per_seed[s][sid] - baseline_per_subject[sid])
            pooled_sizes.append(native_size_by_id.get(sid, np.nan))
    pooled_deltas = np.array(pooled_deltas)
    pooled_sizes = np.array(pooled_sizes)
    valid = ~np.isnan(pooled_sizes)
    rho, p_param = stats.spearmanr(pooled_sizes[valid], pooled_deltas[valid])
    rng = np.random.default_rng(SEED)
    perm_rhos = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_y = rng.permutation(pooled_deltas[valid])
        perm_rhos[i], _ = stats.spearmanr(pooled_sizes[valid], perm_y)
    p_perm = float((np.abs(perm_rhos) >= np.abs(rho)).mean())
    print(f"  Spearman(native_size, delta) = {rho:+.4f} (parametric p={p_param:.4e}, permutation p={p_perm:.4f})")
    print(f"  n valid = {valid.sum()}/{len(pooled_deltas)}")

    # Also per-seed, for transparency (not the primary test, pooled is)
    for s in (0, 1, 2):
        sizes_s = np.array([native_size_by_id.get(sid, np.nan) for sid in subject_ids])
        valid_s = ~np.isnan(sizes_s)
        rho_s, p_s = stats.spearmanr(sizes_s[valid_s], deltas_by_seed[s][valid_s])
        print(f"    seed{s} alone: rho={rho_s:+.4f}, p={p_s:.4e}")

    # ================= 5. Training trajectory comparison =================
    print("\n=== Training trajectory comparison (val_dice by epoch) ===")
    baseline_val_dice = load_csv_column(BASELINE_TRAJ, "val_dice")
    print(f"  Baseline (A_baseline_seed0): final val_dice={baseline_val_dice[-1]:.4f}, "
          f"epoch of first val_dice>0.85={int(np.argmax(baseline_val_dice > 0.85)) if (baseline_val_dice > 0.85).any() else 'never'}")

    e54_trajectories = {}
    for s in (0, 1, 2):
        vd = load_csv_column(E54_TRAJ[s], "val_dice")
        e54_trajectories[s] = vd
        first_85 = int(np.argmax(vd > 0.85)) if (vd > 0.85).any() else None
        print(f"  E54 seed{s}: final val_dice={vd[-1]:.4f}, epoch of first val_dice>0.85={first_85}, "
              f"n_epochs={len(vd)}, mean epoch-to-epoch |diff| (last 10 epochs)="
              f"{np.mean(np.abs(np.diff(vd[-10:]))):.4f}")

    # ================= Decision rule =================
    print("\n\n=== Pre-declared decision (A/B/C) ===")
    coherent_subject_effect = binom_p < 0.05 and n_consistent / n > expected_consistent_frac
    coherent_size_effect = p_perm < 0.05
    print(f"  Coherent subject-level effect (binom p<0.05, consistent frac > chance): {coherent_subject_effect}")
    print(f"  Coherent size-stratified effect (permutation p<0.05): {coherent_size_effect}")

    if not coherent_subject_effect and not coherent_size_effect:
        outcome = "A_NOISE_KILL"
        print("\n=== OUTCOME A: no coherent pattern found. The +0.0002 mean looks like noise around zero. "
              "Recommend NOT spending the ~24 GPU-hours on 7 more seeds. ===")
    elif coherent_subject_effect or coherent_size_effect:
        # distinguish B vs C by effect size
        strong = (n_consistent / n > 0.40) or (abs(rho) > 0.2 and coherent_size_effect)
        if strong:
            outcome = "C_STRONG_JUSTIFIES_REPLICATION"
            print("\n=== OUTCOME C: a strong, coherent pattern was found. The 7-seed replication is "
                  "justified -- see the specific pattern reported above for what to investigate further. ===")
        else:
            outcome = "B_MODEST_WORTH_CHEAP_FOLLOWUP"
            print("\n=== OUTCOME B: a modest but real coherent pattern was found. Worth a cheap follow-up "
                  "(not yet defined), short of the full 7-seed replication. ===")

    summary = {
        "per_seed_stats": {
            str(s): {
                "mean": float(deltas_by_seed[s].mean()), "median": float(np.median(deltas_by_seed[s])),
                "std": float(deltas_by_seed[s].std()),
                "frac_improved": float((deltas_by_seed[s] > 0).mean()),
                "frac_degraded": float((deltas_by_seed[s] < 0).mean()),
            } for s in (0, 1, 2)
        },
        "cross_seed_consistency": {
            "consistently_improved": all_same_sign_positive, "consistently_degraded": all_same_sign_negative,
            "mixed_sign": mixed_sign, "n_subjects": n, "binomial_p": float(binom_p),
        },
        "size_dependence_pooled": {"rho": float(rho), "parametric_p": float(p_param), "permutation_p": p_perm},
        "trajectory_summary": {
            "baseline_final_val_dice": float(baseline_val_dice[-1]),
            **{f"e54_seed{s}_final_val_dice": float(e54_trajectories[s][-1]) for s in (0, 1, 2)},
        },
        "outcome": outcome,
    }
    with open(OUT_DIR / "E69_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved E69_summary.json")


if __name__ == "__main__":
    main()
