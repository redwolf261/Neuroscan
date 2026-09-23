"""
Phase E25: orchestrator for C6-2 (SC-TAM) training.
Per PHASE_E25_CANDIDATE6_SC_TAM_DESIGN.md Section 11, decision 6:
"Same lambda, same training protocol as Gate 6 (seed 0, mu=0.1,
lambda=0.1, 30 epochs, same configs/brats.yaml, same checkpoint
schedule) -- no protocol changes beyond the loss term itself."

Before any real training epoch: construct C6-2 and Gate 6's condition A
fresh, hash-compare initial parameters + optimizer metadata -- HARD
assertion, mirroring run_gate6_all.py's own gate exactly (same
set_seed(0) call, same position before model construction, same model
architecture/optimizer construction code path), so C6-2's own training
starts from the IDENTICAL theta^(0) as A/B/E already did. This directly
extends Gate 6's own 3-way gate to a 4th condition without needing to
rerun A/B/E (they are frozen, completed results); only a fresh,
disposable A instance needs to be reconstructed here for the comparison,
exactly as run_gate6_all.py's own _init_check instances were disposable.
"""
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e24"))
from train_gate6 import Gate6Experiment  # noqa: E402
from run_counterfactual import ObjectiveConfig  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e25"))
from train_c62 import C62Experiment  # noqa: E402

SEED = 0
MU = 0.1
LAMBDA_MARGIN = 0.1
EPOCHS = 30
CHECKPOINT_EVERY = 5
M_IJ_CALIBRATED = 0.3089  # PHASE_E25 Section 12 (e25_calibrate_m_ij.py)
CONFIG_PATH = project_root / "configs" / "brats.yaml"
EXP_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "c62_runs"


def run_initial_parameter_equality_gate():
    """Constructs a fresh, disposable condition-A instance (Gate6Experiment,
    objective_config mode='baseline') and a fresh, disposable C6-2 instance,
    hashes their initial model state_dicts and compares optimizer metadata,
    HARD-ASSERTING they match -- exactly mirroring run_gate6_all.py's own
    gate, extended to cover C6-2 against the same theta^(0) A/B/E already
    share."""
    print("=" * 70)
    print("INITIAL-PARAMETER-EQUALITY GATE: C6-2 vs condition A (hard assertion, before any training)")
    print("=" * 70)

    objective_a = ObjectiveConfig(mode="baseline")
    exp_a = Gate6Experiment("A_check", objective_a, str(CONFIG_PATH), str(EXP_DIR), SEED, MU, LAMBDA_MARGIN,
                             run_name="_init_check_A", num_workers=0)
    exp_c62 = C62Experiment(M_IJ_CALIBRATED, str(CONFIG_PATH), str(EXP_DIR), SEED, MU, LAMBDA_MARGIN,
                             run_name="_init_check_C62", num_workers=0)

    hash_a = exp_a.get_initial_state_hash()
    hash_c62 = exp_c62.get_initial_state_hash()

    print(f"A model_state hash:   {hash_a['model_state_hash']}")
    print(f"C6-2 model_state hash: {hash_c62['model_state_hash']}")
    print(f"A optimizer metadata:   {hash_a['optimizer_metadata']}")
    print(f"C6-2 optimizer metadata: {hash_c62['optimizer_metadata']}")

    exp_a.close_logs()
    exp_c62.close_logs()

    assert hash_a["model_state_hash"] == hash_c62["model_state_hash"], (
        f"FATAL: initial model parameters DIFFER between A and C6-2 -- "
        f"A={hash_a['model_state_hash'][:16]}... C6-2={hash_c62['model_state_hash'][:16]}... "
        f"DO NOT PROCEED TO TRAINING, diagnose the divergence first"
    )
    assert hash_a["optimizer_metadata"] == hash_c62["optimizer_metadata"], (
        f"FATAL: optimizer metadata DIFFERS between A and C6-2 -- "
        f"A={hash_a['optimizer_metadata']} C6-2={hash_c62['optimizer_metadata']} -- "
        f"DO NOT PROCEED TO TRAINING, diagnose the divergence first"
    )

    print("\nGATE PASSED: theta_A^(0) == theta_C6-2^(0), optimizer metadata identical.")
    print("=" * 70 + "\n")

    del exp_a, exp_c62
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    run_initial_parameter_equality_gate()

    print("\n" + "#" * 70)
    print("# LAUNCHING C6-2 (SC-TAM: signed, class-conditional, task-aligned margin)")
    print("#" * 70)
    exp_c62 = C62Experiment(
        M_IJ_CALIBRATED, str(CONFIG_PATH), str(EXP_DIR), SEED, MU, LAMBDA_MARGIN,
        run_name="C62_sc_tam_seed0",
        num_workers=None,  # falls back to configs/brats.yaml's num_workers=4 -- applying the Gate 6 lesson directly, not repeating the num_workers=0 performance bug
    )
    exp_c62.train(epochs=EPOCHS, checkpoint_every=CHECKPOINT_EVERY)
    del exp_c62
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("C6-2 TRAINING: COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
