"""
Phase E24, Gate 6, H1: exact counterfactual rerun on A/B/E's own checkpoints,
each evaluated under its OWN objective_config -- per the user's explicit
instruction not to cross-evaluate (e.g. never score B's checkpoints under
A's Euclidean metric).

Uses run_counterfactual() UNCHANGED (the same function already
regression-verified to reproduce E22's own published row-level output
exactly, 1536/1536 rows, 0.00e+00 max diff) -- no new logic, no
reimplementation. Only the checkpoint_dir and objective_config vary per
condition, exactly the interface run_counterfactual() was built for.

Frozen protocol, held IDENTICAL to the original E22 run and to Gate 6's
own locked specification:
  - CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]
  - N_SUBJECTS_TO_USE = 8
  - Same epsilon grids (EPSILON_SWEEP_PARAM, EPSILON_SWEEP_ACTIVATION),
    same directions (A_seg/B_margin/C_total/D_useful/E_random_matched_*),
    same anchor-fixing/torch.manual_seed discipline, same row-key
    structure (epoch, subject_idx, direction, epsilon) -- all inherited
    unchanged from run_counterfactual.py.

Output: three separate results files (h1_results_A.json/csv,
h1_results_B.json/csv, h1_results_E.json/csv), each condition's rows kept
completely separate -- no merging, no cross-condition row keys, per the
user's explicit "do not compare raw magnitudes across A/B/E" instruction
(mixing them into one file would invite exactly that mistake later).
"""
import sys
import json
import csv
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e24"))

from run_counterfactual import run_counterfactual, ObjectiveConfig  # noqa: E402

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]  # frozen, matches original E22 exactly
N_SUBJECTS_TO_USE = 8  # frozen, matches original E22 exactly

GATE6_RUNS_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e24" / "gate6_runs"
OUT_DIR = Path(__file__).parent / "h1_results"

CONDITIONS = [
    ("A", "baseline", GATE6_RUNS_DIR / "A_baseline_seed0" / "checkpoints"),
    ("B", "task_aligned", GATE6_RUNS_DIR / "B_task_aligned_seed0" / "checkpoints"),
    ("E", "random_projection", GATE6_RUNS_DIR / "E_random_projection_seed0" / "checkpoints"),
]


def main():
    OUT_DIR.mkdir(exist_ok=True)

    for condition_label, mode, checkpoint_dir in CONDITIONS:
        print("\n" + "#" * 70)
        print(f"# H1 counterfactual rerun: condition {condition_label} (objective={mode})")
        print(f"# checkpoints: {checkpoint_dir}")
        print("#" * 70)

        assert checkpoint_dir.exists(), f"FATAL: checkpoint dir not found: {checkpoint_dir}"

        objective = ObjectiveConfig(mode=mode)
        rows = run_counterfactual(
            checkpoint_dir=checkpoint_dir,
            objective_config=objective,
            checkpoint_epochs=CHECKPOINT_EPOCHS,
            n_subjects_to_use=N_SUBJECTS_TO_USE,
            verbose=True,
        )

        print(f"\nCondition {condition_label}: {len(rows)} rows")

        json_path = OUT_DIR / f"h1_results_{condition_label}.json"
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)

        csv_path = OUT_DIR / f"h1_results_{condition_label}.csv"
        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        print(f"Saved to {json_path} and {csv_path}")

    print("\n" + "=" * 70)
    print("H1 COUNTERFACTUAL RERUNS: ALL THREE CONDITIONS COMPLETE")
    print("=" * 70)
    print("Results kept in SEPARATE files per condition -- no cross-condition merging.")
    print("Next step (per locked analysis order): compute rho_A, rho_B, rho_E independently,")
    print("each condition's own objective against its own Dice change, before any Dice comparison.")


if __name__ == "__main__":
    main()
