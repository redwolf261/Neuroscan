"""
Phase E25: orchestrator for C6-3 (Confidence-Gated SC-TAM) training.
Per PHASE_E25_C63_GATE_RETROSPECTIVE_TEST.md's GO verdict: the
ground-truth-conditioned gate w(x)=|p(x)-target(x)| passed the user's
own strict two-part criterion decisively (92.3% corrective movement
retained, 99.2% of TP/TN damaging movement removed, 122.55x ratio
improvement), retrospectively validated on C6-2's own real data before
this training run was ever launched. Same protocol as C6-2 (Section 11's
"same lambda, same training protocol" carried forward): seed 0, mu=0.1,
lambda=0.1, 30 epochs, same configs/brats.yaml, same checkpoint
schedule, same calibrated m_ij=0.3089 (verified not to need recalibration
for the gated mode -- see train_c63.py's own docstring).

Before any real training epoch: construct C6-3 and condition A fresh,
hash-compare initial parameters + optimizer metadata -- HARD assertion,
identical mechanism to run_c62.py's own gate, extended to C6-3.
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
from train_c63 import C63Experiment  # noqa: E402

SEED = 0
MU = 0.1
LAMBDA_MARGIN = 0.1
EPOCHS = 30
CHECKPOINT_EVERY = 5
M_IJ_CALIBRATED = 0.3089  # SAME as C6-2 -- gate doesn't change dist/margin_target, verified by test_sc_tam_gated.py
CONFIG_PATH = project_root / "configs" / "brats.yaml"
EXP_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "c63_runs"


def run_initial_parameter_equality_gate():
    """Identical mechanism to run_c62.py's own gate, extended to C6-3."""
    print("=" * 70)
    print("INITIAL-PARAMETER-EQUALITY GATE: C6-3 vs condition A (hard assertion, before any training)")
    print("=" * 70)

    objective_a = ObjectiveConfig(mode="baseline")
    exp_a = Gate6Experiment("A_check", objective_a, str(CONFIG_PATH), str(EXP_DIR), SEED, MU, LAMBDA_MARGIN,
                             run_name="_init_check_A", num_workers=0)
    exp_c63 = C63Experiment(M_IJ_CALIBRATED, str(CONFIG_PATH), str(EXP_DIR), SEED, MU, LAMBDA_MARGIN,
                             run_name="_init_check_C63", num_workers=0)

    hash_a = exp_a.get_initial_state_hash()
    hash_c63 = exp_c63.get_initial_state_hash()

    print(f"A model_state hash:   {hash_a['model_state_hash']}")
    print(f"C6-3 model_state hash: {hash_c63['model_state_hash']}")
    print(f"A optimizer metadata:   {hash_a['optimizer_metadata']}")
    print(f"C6-3 optimizer metadata: {hash_c63['optimizer_metadata']}")

    exp_a.close_logs()
    exp_c63.close_logs()

    assert hash_a["model_state_hash"] == hash_c63["model_state_hash"], (
        f"FATAL: initial model parameters DIFFER between A and C6-3 -- "
        f"A={hash_a['model_state_hash'][:16]}... C6-3={hash_c63['model_state_hash'][:16]}... "
        f"DO NOT PROCEED TO TRAINING, diagnose the divergence first"
    )
    assert hash_a["optimizer_metadata"] == hash_c63["optimizer_metadata"], (
        f"FATAL: optimizer metadata DIFFERS between A and C6-3 -- "
        f"A={hash_a['optimizer_metadata']} C6-3={hash_c63['optimizer_metadata']} -- "
        f"DO NOT PROCEED TO TRAINING, diagnose the divergence first"
    )

    print("\nGATE PASSED: theta_A^(0) == theta_C6-3^(0), optimizer metadata identical.")
    print("=" * 70 + "\n")

    del exp_a, exp_c63
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    run_initial_parameter_equality_gate()

    print("\n" + "#" * 70)
    print("# LAUNCHING C6-3 (Confidence-Gated SC-TAM: w(x)=|p(x)-target(x)|)")
    print("#" * 70)
    exp_c63 = C63Experiment(
        M_IJ_CALIBRATED, str(CONFIG_PATH), str(EXP_DIR), SEED, MU, LAMBDA_MARGIN,
        run_name="C63_sc_tam_gated_seed0",
        num_workers=None,  # falls back to configs/brats.yaml's num_workers=4 -- applying the Gate 6/C6-2 lesson directly
    )
    exp_c63.train(epochs=EPOCHS, checkpoint_every=CHECKPOINT_EVERY)
    del exp_c63
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("C6-3 TRAINING: COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
