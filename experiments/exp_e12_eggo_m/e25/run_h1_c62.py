"""
Phase E25, C6-2, H1: exact counterfactual rerun on C6-2's own checkpoints,
evaluated under C6-2's OWN objective (margin_mode="sc_tam", m_ij=0.3089) --
per the project's standing rule not to cross-evaluate (never score C6-2's
checkpoints under A's Euclidean metric or B's unsigned task-aligned
metric).

Uses run_counterfactual() with ObjectiveConfig(mode="sc_tam") -- the SAME
function E24 already regression-verified against E22's own published
row-level output (1536/1536 rows, 0.00e+00 max diff), now extended with a
4th mode whose addition was ITSELF re-verified via the same regression
gate (0.00e+00 on all fields, unchanged) before this script was run,
confirming the sc_tam mode addition did not alter baseline/task_aligned/
random_projection's existing behavior.

Frozen protocol, IDENTICAL to E24's H1 runs for A/B/E:
  - CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]
  - N_SUBJECTS_TO_USE = 8
  - Same epsilon grids, directions, anchor-fixing/torch.manual_seed
    discipline, row-key structure -- all inherited unchanged.

Output: h1_results/h1_results_C62.json/csv, kept completely separate from
A/B/E's own H1 files -- no merging, no cross-condition row keys.
"""
import sys
import json
import csv
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e24"))

from run_counterfactual import run_counterfactual, ObjectiveConfig  # noqa: E402

CHECKPOINT_EPOCHS = [5, 10, 15, 20, 25, 30]  # frozen, matches E24's A/B/E H1 exactly
N_SUBJECTS_TO_USE = 8  # frozen, matches E24's A/B/E H1 exactly

C62_CHECKPOINT_DIR = (project_root / "experiments" / "exp_e12_eggo_m" / "e25" /
                       "c62_runs" / "C62_sc_tam_seed0" / "checkpoints")
OUT_DIR = Path(__file__).parent / "h1_results"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("\n" + "#" * 70)
    print("# H1 counterfactual rerun: condition C6-2 (objective=sc_tam)")
    print(f"# checkpoints: {C62_CHECKPOINT_DIR}")
    print("#" * 70)

    assert C62_CHECKPOINT_DIR.exists(), f"FATAL: checkpoint dir not found: {C62_CHECKPOINT_DIR}"

    objective = ObjectiveConfig(mode="sc_tam")
    rows = run_counterfactual(
        checkpoint_dir=C62_CHECKPOINT_DIR,
        objective_config=objective,
        checkpoint_epochs=CHECKPOINT_EPOCHS,
        n_subjects_to_use=N_SUBJECTS_TO_USE,
        verbose=True,
    )

    print(f"\nCondition C6-2: {len(rows)} rows")

    json_path = OUT_DIR / "h1_results_C62.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    csv_path = OUT_DIR / "h1_results_C62.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved to {json_path} and {csv_path}")

    print("\n" + "=" * 70)
    print("H1 COUNTERFACTUAL RERUN: C6-2 COMPLETE")
    print("=" * 70)
    print("Next step (per locked analysis order): compute rho_C62 (corr(delta_L_margin_sc_tam,")
    print("delta_Dice) for the B_margin direction), compare against rho_A/rho_B/rho_E as")
    print("HISTORICAL REFERENCE ONLY (different objectives -- not a threshold), before any")
    print("Dice comparison (H2 next, then H3, then H4).")


if __name__ == "__main__":
    main()
