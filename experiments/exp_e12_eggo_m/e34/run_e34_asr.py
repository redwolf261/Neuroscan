"""
Phase E34: orchestrator for the A/S/R controlled comparison. Seed 0 ONLY,
per the explicit "do not immediately multi-seed" instruction -- seed 0 is a
screening experiment; seeds 1/2 only run if R clears the performance and
size-control gates.

Trains, in order: A (baseline, byte-identical loss to A64/A29's own A
condition), S (size-control), R (alpha_c). Same protocol as every prior
orchestrator: same config, same optimizer/scheduler, same 30 epochs, same
checkpoint schedule, initial-parameter-equality gate before any real
training epoch runs.
"""
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e34"))

from train_e34_component_weighted import ComponentWeightedExperiment, state_dict_hash  # noqa: E402

SEED = 0
MU = 0.1
EPOCHS = 30
CHECKPOINT_EVERY = 5
CONFIG_PATH = project_root / "configs" / "brats.yaml"
EXP_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e34" / "asr_runs"
E34_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e34"

CONDITIONS = [
    ("A", None, None),
    ("S", "size", E34_DIR / "E34_weight_table_size.json"),
    ("R", "alpha_c", E34_DIR / "E34_weight_table_alpha_c.json"),
]


def run_initial_parameter_equality_gate(condition_name, mode, weight_table_path):
    print("=" * 70)
    print(f"SEED-REPRODUCIBILITY GATE: {condition_name} (hard assertion, before real training)")
    print("=" * 70)

    exp1 = ComponentWeightedExperiment(condition_name, mode, weight_table_path, str(CONFIG_PATH), str(EXP_DIR), SEED, MU,
                                        run_name=f"_init_check_{condition_name}_1", num_workers=0)
    exp2 = ComponentWeightedExperiment(condition_name, mode, weight_table_path, str(CONFIG_PATH), str(EXP_DIR), SEED, MU,
                                        run_name=f"_init_check_{condition_name}_2", num_workers=0)

    hash1 = state_dict_hash(exp1.model.state_dict())
    hash2 = state_dict_hash(exp2.model.state_dict())
    print(f"  init 1: {hash1}")
    print(f"  init 2: {hash2}")

    exp1.close_logs()
    exp2.close_logs()

    assert hash1 == hash2, f"FATAL: same-seed re-construction gives different initial parameters for {condition_name} -- DO NOT PROCEED"
    print(f"GATE PASSED: {condition_name} seed={SEED} is reproducible across construction.")
    print("=" * 70 + "\n")

    del exp1, exp2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_condition(condition_name, mode, weight_table_path):
    print("\n" + "#" * 70)
    print(f"# LAUNCHING {condition_name} (mode={mode})")
    print("#" * 70)

    run_initial_parameter_equality_gate(condition_name, mode, weight_table_path)

    exp = ComponentWeightedExperiment(
        condition_name, mode, weight_table_path, str(CONFIG_PATH), str(EXP_DIR), SEED, MU,
        run_name=f"{condition_name}_seed{SEED}",
        num_workers=None,  # falls back to configs/brats.yaml's num_workers=4
    )
    exp.train(epochs=EPOCHS, checkpoint_every=CHECKPOINT_EVERY)
    del exp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("\n" + "=" * 70)
    print("PHASE E34: A / S / R CONTROLLED COMPARISON, SEED 0 (screening)")
    print("=" * 70)

    for condition_name, mode, weight_table_path in CONDITIONS:
        run_condition(condition_name, mode, weight_table_path)

    print("\n" + "=" * 70)
    print("E34 A/S/R SEED-0 SCREENING: ALL CONDITIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
