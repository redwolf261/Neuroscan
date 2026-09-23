"""
Phase E24, Gate 6, Section 6 step 2: HARD regression gate.

Runs the refactored run_counterfactual(mode="baseline") against the
ORIGINAL E12f checkpoints (the exact same checkpoints E22's own published
run used) and compares the output ROW BY ROW against the original,
already-published experiments/exp_e12_eggo_m/e22/results.json -- not just
the headline correlation, per the user's explicit instruction that a
refactor could accidentally preserve the headline statistic while
changing individual rows.

Row identity: (epoch, subject_idx, direction, epsilon) -- if this key
doesn't uniquely identify a row in both the original and refactored
outputs, that is itself reported as a finding, not silently handled.

STOP CONDITION, per explicit instruction: if this regression fails, DO
NOT proceed to train A/B/E. Diagnose the discrepancy against the original
E22 implementation first.
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e24"))

from run_counterfactual import run_counterfactual, ObjectiveConfig  # noqa: E402

ORIGINAL_E22_RESULTS = project_root / "experiments" / "exp_e12_eggo_m" / "e22" / "results.json"
ORIGINAL_E22_CHECKPOINT_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e12f_pilot_calibrated_seed0" / "checkpoints"
ORIGINAL_CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]
ORIGINAL_N_SUBJECTS = 8

# Numeric fields to compare per row. Deterministic floating-point
# quantities (same execution path, same seeds, same operations) --
# expected to match near machine precision. NOT platform/kernel-level
# nondeterministic (no cuDNN autotune, no multi-threaded reduction
# nondeterminism triggered anywhere in this specific script), so 1e-6
# relative tolerance is appropriate HERE specifically (stated explicitly,
# per the audit's caution not to apply this blindly) -- distinguished
# from the alternative case (different kernels/platforms) which would
# warrant a looser bound; this run uses the SAME machine/environment/
# GPU as the original E22 run, so that concern does not apply.
NUMERIC_FIELDS = [
    "delta_norm", "dice_before", "dice_after", "delta_dice",
    "seg_loss_before", "seg_loss_after", "delta_seg_loss",
    "margin_loss_before", "margin_loss_after", "delta_margin_loss",
    "geometry_before", "geometry_after", "delta_geometry",
    "cos_with_g_seg", "cos_with_g_margin", "cos_with_useful",
]
ABS_TOL = 1e-6
REL_TOL = 1e-6


def row_key(row):
    return (row["epoch"], row["subject_idx"], row["direction"], row["epsilon"])


def compare_value(a, b):
    """Returns (abs_diff, rel_diff, comparable) for a pair of values that
    may be None (both, one, or neither)."""
    if a is None and b is None:
        return 0.0, 0.0, True
    if a is None or b is None:
        return None, None, False  # one is None, other isn't -- a real mismatch, not a numeric diff
    abs_diff = abs(a - b)
    denom = max(abs(a), abs(b), 1e-12)
    rel_diff = abs_diff / denom
    return abs_diff, rel_diff, True


def main():
    print("=" * 70)
    print("PHASE E24 GATE 6, SECTION 6 STEP 2: HARD REGRESSION GATE")
    print("=" * 70)
    print(f"Comparing refactored run_counterfactual(mode='baseline') against")
    print(f"original {ORIGINAL_E22_RESULTS}")
    print(f"on the ORIGINAL E12f checkpoints: {ORIGINAL_E22_CHECKPOINT_DIR}")

    if not ORIGINAL_E22_RESULTS.exists():
        print(f"\nFATAL: original E22 results not found at {ORIGINAL_E22_RESULTS} -- cannot run regression gate")
        sys.exit(2)

    with open(ORIGINAL_E22_RESULTS) as f:
        original_rows = json.load(f)
    print(f"\nOriginal E22 results: {len(original_rows)} rows")

    print("\nRunning refactored run_counterfactual(mode='baseline')...")
    baseline_objective = ObjectiveConfig(mode="baseline")
    refactored_rows = run_counterfactual(
        checkpoint_dir=ORIGINAL_E22_CHECKPOINT_DIR,
        objective_config=baseline_objective,
        checkpoint_epochs=ORIGINAL_CHECKPOINT_EPOCHS,
        n_subjects_to_use=ORIGINAL_N_SUBJECTS,
        verbose=True,
    )
    print(f"\nRefactored results: {len(refactored_rows)} rows")

    # --- Build lookup dicts by row key, check for uniqueness ---
    original_by_key = defaultdict(list)
    for r in original_rows:
        original_by_key[row_key(r)].append(r)
    refactored_by_key = defaultdict(list)
    for r in refactored_rows:
        refactored_by_key[row_key(r)].append(r)

    duplicate_keys_original = {k: len(v) for k, v in original_by_key.items() if len(v) > 1}
    duplicate_keys_refactored = {k: len(v) for k, v in refactored_by_key.items() if len(v) > 1}
    if duplicate_keys_original:
        print(f"\nWARNING: {len(duplicate_keys_original)} row keys are NOT unique in the ORIGINAL results "
              f"(e.g. {list(duplicate_keys_original.items())[:3]}) -- row identity (epoch,subject_idx,direction,epsilon) "
              f"does not uniquely identify a row in the original data; reporting this as a finding.")
    if duplicate_keys_refactored:
        print(f"\nWARNING: {len(duplicate_keys_refactored)} row keys are NOT unique in the REFACTORED results "
              f"(e.g. {list(duplicate_keys_refactored.items())[:3]})")

    original_keys = set(original_by_key.keys())
    refactored_keys = set(refactored_by_key.keys())
    missing_in_refactored = original_keys - refactored_keys
    extra_in_refactored = refactored_keys - original_keys
    common_keys = original_keys & refactored_keys

    print(f"\nRow-key coverage: {len(common_keys)} common, {len(missing_in_refactored)} missing in refactored, "
          f"{len(extra_in_refactored)} extra in refactored")
    if missing_in_refactored:
        print(f"  Missing keys (first 10): {list(missing_in_refactored)[:10]}")
    if extra_in_refactored:
        print(f"  Extra keys (first 10): {list(extra_in_refactored)[:10]}")

    # --- Per-field comparison across all common rows ---
    field_stats = {f: {"max_abs_diff": 0.0, "max_rel_diff": 0.0, "sum_abs_diff": 0.0,
                        "n_compared": 0, "n_sign_changed": 0, "n_none_mismatch": 0}
                   for f in NUMERIC_FIELDS}

    for key in sorted(common_keys):
        orig_row = original_by_key[key][0]
        refac_row = refactored_by_key[key][0]
        for field in NUMERIC_FIELDS:
            a, b = orig_row.get(field), refac_row.get(field)
            abs_diff, rel_diff, comparable = compare_value(a, b)
            if not comparable:
                field_stats[field]["n_none_mismatch"] += 1
                continue
            if a is None and b is None:
                continue  # both legitimately None (e.g. cos_with_useful for most directions) -- not a discrepancy
            field_stats[field]["max_abs_diff"] = max(field_stats[field]["max_abs_diff"], abs_diff)
            field_stats[field]["max_rel_diff"] = max(field_stats[field]["max_rel_diff"], rel_diff)
            field_stats[field]["sum_abs_diff"] += abs_diff
            field_stats[field]["n_compared"] += 1
            if a * b < 0:  # sign changed (both nonzero, opposite signs)
                field_stats[field]["n_sign_changed"] += 1

    print("\n" + "=" * 70)
    print("PER-FIELD REGRESSION REPORT")
    print("=" * 70)
    print(f"{'field':<22} {'n_cmp':>7} {'max_abs':>12} {'max_rel':>12} {'mean_abs':>12} {'n_sign_chg':>11} {'n_none_mismatch':>16}")
    any_field_failed = False
    for field, stats in field_stats.items():
        mean_abs = stats["sum_abs_diff"] / max(1, stats["n_compared"])
        failed = stats["max_abs_diff"] > ABS_TOL and stats["max_rel_diff"] > REL_TOL
        if stats["n_sign_changed"] > 0 or stats["n_none_mismatch"] > 0:
            failed = True
        marker = " <-- FAIL" if failed else ""
        if failed:
            any_field_failed = True
        print(f"{field:<22} {stats['n_compared']:>7} {stats['max_abs_diff']:>12.2e} {stats['max_rel_diff']:>12.2e} "
              f"{mean_abs:>12.2e} {stats['n_sign_changed']:>11} {stats['n_none_mismatch']:>16}{marker}")

    # --- Also check whether the CLASSIFICATION outcome changes (the
    # headline correlation and its decision-relevant sign, per the user's
    # explicit "any changed classification/decision outcome" requirement) ---
    print("\n" + "=" * 70)
    print("HEADLINE STATISTIC CHECK (secondary -- row-level check above is primary)")
    print("=" * 70)
    from scipy import stats as scipy_stats

    def pooled_corr(rows, direction):
        subset = [r for r in rows if r["direction"] == direction]
        dm = np.array([r["delta_margin_loss"] for r in subset])
        dd = np.array([r["delta_dice"] for r in subset])
        corr, p = scipy_stats.pearsonr(dm, dd)
        return corr, p, len(dm)

    orig_corr, orig_p, orig_n = pooled_corr(original_rows, "B_margin")
    refac_corr, refac_p, refac_n = pooled_corr(refactored_rows, "B_margin")
    print(f"Original:    corr(delta_L_margin, delta_Dice) for B_margin = {orig_corr:+.4f}, p={orig_p:.2e}, n={orig_n}")
    print(f"Refactored:  corr(delta_L_margin, delta_Dice) for B_margin = {refac_corr:+.4f}, p={refac_p:.2e}, n={refac_n}")
    sign_matches = (orig_corr > 0) == (refac_corr > 0)
    print(f"Sign matches: {sign_matches}")
    corr_diff = abs(orig_corr - refac_corr)
    print(f"Correlation absolute difference: {corr_diff:.6f}")

    print("\n" + "=" * 70)
    print("GATE VERDICT")
    print("=" * 70)
    gate_pass = (
        len(missing_in_refactored) == 0
        and len(extra_in_refactored) == 0
        and not duplicate_keys_original
        and not duplicate_keys_refactored
        and not any_field_failed
        and sign_matches
        and corr_diff < 0.01
    )
    if gate_pass:
        print("REGRESSION GATE: PASS")
        print("The refactored run_counterfactual reproduces the original E22 implementation's")
        print("row-level output (not just the headline correlation) within the defined tolerance.")
        print("Section 6 steps 1-2 are complete. Training (step 3 onward) may be authorized.")
    else:
        print("REGRESSION GATE: FAIL")
        print("DO NOT proceed to train A/B/E. Diagnose the discrepancy against the original")
        print("e22_counterfactual_geometry.py implementation before any further action.")
        sys.exit(1)


if __name__ == "__main__":
    main()
