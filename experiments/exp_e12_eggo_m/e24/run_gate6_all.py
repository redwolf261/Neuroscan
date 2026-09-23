"""
Phase E24, Gate 6: orchestrator for conditions A/B/E.
Per PHASE_E24_GATE6_EXPERIMENTAL_SPECIFICATION.md Section 6, steps 3-6:

  3. Launch condition A (w_hat=None, delta_d=3.6659)
  4. BEFORE any real training epoch: construct A/B/E fresh, hash-compare
     initial parameters + optimizer metadata -- HARD assertion, not a log.
  5. Launch condition B (w_hat=live seg_head, delta_d_w=0.2553)
  6. Launch condition E (w_hat=frozen r, delta_d_r=0.2022)

All three: seed=0, mu=0.1, lambda_margin=0.1, 30 epochs, same
configs/brats.yaml, same checkpoint schedule. Per the user's explicit
instruction: do not change anything based on intermediate results --
this script runs all three conditions to completion under the fixed
protocol; no adaptive logic anywhere in this file reacts to a metric
value.
"""
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e24"))
from train_gate6 import Gate6Experiment  # noqa: E402
from run_counterfactual import ObjectiveConfig  # noqa: E402

SEED = 0
MU = 0.1
LAMBDA_MARGIN = 0.1
EPOCHS = 30
CHECKPOINT_EVERY = 5
CONFIG_PATH = project_root / "configs" / "brats.yaml"
EXP_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e24" / "gate6_runs"


def run_initial_parameter_equality_gate():
    """Constructs A/B/E fresh (WITHOUT running any training), hashes their
    initial model state_dicts and compares optimizer metadata, and HARD-
    ASSERTS all three match -- per Section 3's gate, upgraded from "log"
    to "assert" per the user's explicit instruction. Returns nothing;
    raises AssertionError and halts if the gate fails. The three
    experiment objects constructed here are DISCARDED after the check
    (real training constructs its own fresh instances below) -- this
    keeps the gate a pure pre-check, not entangled with the real training
    objects' own subsequent state changes."""
    print("=" * 70)
    print("INITIAL-PARAMETER-EQUALITY GATE (hard assertion, before any training)")
    print("=" * 70)

    objective_a = ObjectiveConfig(mode="baseline")
    objective_b = ObjectiveConfig(mode="task_aligned")
    objective_e = ObjectiveConfig(mode="random_projection")

    exp_a = Gate6Experiment("A_check", objective_a, str(CONFIG_PATH), str(EXP_DIR), SEED, MU, LAMBDA_MARGIN,
                             run_name="_init_check_A", num_workers=0)
    exp_b = Gate6Experiment("B_check", objective_b, str(CONFIG_PATH), str(EXP_DIR), SEED, MU, LAMBDA_MARGIN,
                             run_name="_init_check_B", num_workers=0)
    exp_e = Gate6Experiment("E_check", objective_e, str(CONFIG_PATH), str(EXP_DIR), SEED, MU, LAMBDA_MARGIN,
                             run_name="_init_check_E", num_workers=0)

    hash_a = exp_a.get_initial_state_hash()
    hash_b = exp_b.get_initial_state_hash()
    hash_e = exp_e.get_initial_state_hash()

    print(f"A model_state hash: {hash_a['model_state_hash']}")
    print(f"B model_state hash: {hash_b['model_state_hash']}")
    print(f"E model_state hash: {hash_e['model_state_hash']}")
    print(f"A optimizer metadata: {hash_a['optimizer_metadata']}")
    print(f"B optimizer metadata: {hash_b['optimizer_metadata']}")
    print(f"E optimizer metadata: {hash_e['optimizer_metadata']}")

    exp_a.close_logs()
    exp_b.close_logs()
    exp_e.close_logs()

    assert hash_a["model_state_hash"] == hash_b["model_state_hash"] == hash_e["model_state_hash"], (
        f"FATAL: initial model parameters DIFFER across A/B/E -- "
        f"A={hash_a['model_state_hash'][:16]}... B={hash_b['model_state_hash'][:16]}... "
        f"E={hash_e['model_state_hash'][:16]}... -- DO NOT PROCEED TO TRAINING, diagnose the divergence first"
    )
    assert hash_a["optimizer_metadata"] == hash_b["optimizer_metadata"] == hash_e["optimizer_metadata"], (
        f"FATAL: optimizer metadata DIFFERS across A/B/E -- "
        f"A={hash_a['optimizer_metadata']} B={hash_b['optimizer_metadata']} E={hash_e['optimizer_metadata']} -- "
        f"DO NOT PROCEED TO TRAINING, diagnose the divergence first"
    )

    print("\nGATE PASSED: theta_A^(0) == theta_B^(0) == theta_E^(0), optimizer metadata identical.")
    print("=" * 70 + "\n")

    del exp_a, exp_b, exp_e
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    # --- Step 4: hard initial-parameter-equality gate, BEFORE any real training ---
    run_initial_parameter_equality_gate()

    # --- Step 3: condition A ---
    print("\n" + "#" * 70)
    print("# LAUNCHING CONDITION A (baseline Euclidean L_margin)")
    print("#" * 70)
    objective_a = ObjectiveConfig(mode="baseline")
    exp_a = Gate6Experiment("A", objective_a, str(CONFIG_PATH), str(EXP_DIR), SEED, MU, LAMBDA_MARGIN,
                             run_name="A_baseline_seed0", num_workers=None)  # falls back to configs/brats.yaml's num_workers=4, matching E12f's own convention -- FIXED from an erroneous num_workers=0 carried over from the Gate 5 smoke test, which caused ~3.5x slower epochs (~535s vs the expected ~150-160s)
    exp_a.train(epochs=EPOCHS, checkpoint_every=CHECKPOINT_EVERY)
    del exp_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Step 5: condition B ---
    print("\n" + "#" * 70)
    print("# LAUNCHING CONDITION B (task-aligned L_margin^w)")
    print("#" * 70)
    objective_b = ObjectiveConfig(mode="task_aligned")
    exp_b = Gate6Experiment("B", objective_b, str(CONFIG_PATH), str(EXP_DIR), SEED, MU, LAMBDA_MARGIN,
                             run_name="B_task_aligned_seed0", num_workers=None)
    exp_b.train(epochs=EPOCHS, checkpoint_every=CHECKPOINT_EVERY)
    del exp_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Step 6: condition E ---
    print("\n" + "#" * 70)
    print("# LAUNCHING CONDITION E (random-projection L_margin^r)")
    print("#" * 70)
    objective_e = ObjectiveConfig(mode="random_projection")
    exp_e = Gate6Experiment("E", objective_e, str(CONFIG_PATH), str(EXP_DIR), SEED, MU, LAMBDA_MARGIN,
                             run_name="E_random_projection_seed0", num_workers=None)
    exp_e.train(epochs=EPOCHS, checkpoint_every=CHECKPOINT_EVERY)
    del exp_e
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("GATE 6 CONDITIONS A/B/E: ALL COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
