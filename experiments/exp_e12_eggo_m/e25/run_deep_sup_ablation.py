"""
Phase E25, Structural Pivot 1B: scale ablation. Trains D/4-only and
D/2-only conditions, using the IDENTICAL protocol, calibration constants
(lambda_ds3=0.9927, lambda_ds2=1.0014, same as the successful "both"
run), seed (0), architecture (UNet3D_v3, same as "both" -- the aux heads
exist in BOTH ablation conditions' architecture, only which branch's
LOSS is applied differs, per DeepSupExperiment's own enable_aux3/
enable_aux2 toggles), optimizer, LR schedule, epochs (30), dataset
split, and checkpoint-selection procedure as condition A and the
already-completed "both" (DeepSup_seed0) run.

The ONLY independent variable across the three deep-supervision
conditions (D/4-only, D/2-only, both) is which auxiliary loss term(s)
are active -- verified directly by inspecting DeepSupExperiment.__init__
and train_epoch's own total_loss construction before writing this
orchestrator (Section 3 of PHASE_E25_STRUCTURAL_PIVOT_1B... 's own
Implementation Verification requirement).
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
from train_deep_sup import DeepSupExperiment, state_dict_hash  # noqa: E402

SEED = 0
MU = 0.1
EPOCHS = 30
CHECKPOINT_EVERY = 5
LAMBDA_DS3 = 0.9927  # SAME calibration as the "both" run -- not re-derived, per the requirement to keep everything except the branch-selection identical
LAMBDA_DS2 = 1.0014
CONFIG_PATH = project_root / "configs" / "brats.yaml"
EXP_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs"


def run_initial_parameter_equality_gate(exp_candidate, label):
    """Same mechanism as run_deep_sup.py's own gate, parameterized to
    reuse for each of the two new ablation conditions."""
    print("=" * 70)
    print(f"INITIAL-PARAMETER-EQUALITY GATE: {label} vs condition A (hard assertion, before any training)")
    print("=" * 70)

    objective_a = ObjectiveConfig(mode="baseline")
    exp_a = Gate6Experiment("A_check", objective_a, str(CONFIG_PATH), str(EXP_DIR), SEED, MU, 0.0,
                             run_name=f"_init_check_A_{label}", num_workers=0)

    shared_keys = set(exp_a.model.state_dict().keys())
    assert shared_keys.issubset(set(exp_candidate.model.state_dict().keys())), "v3 does not contain all of v2's own parameter keys"

    hash_a_shared = state_dict_hash(exp_a.model.state_dict(), keys_only=shared_keys)
    hash_candidate = state_dict_hash(exp_candidate.model.state_dict(), keys_only=shared_keys)

    print(f"A model_state hash (shared keys):          {hash_a_shared}")
    print(f"{label} model_state hash (shared keys):    {hash_candidate}")

    exp_a.close_logs()

    assert hash_a_shared == hash_candidate, (
        f"FATAL: initial model parameters DIFFER between A and {label} on shared keys -- "
        f"A={hash_a_shared[:16]}... {label}={hash_candidate[:16]}... DO NOT PROCEED TO TRAINING"
    )
    print(f"\nGATE PASSED: theta_A^(0) == theta_{label}^(0) on all shared parameters.")
    print("=" * 70 + "\n")

    del exp_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_condition(enable_aux3, enable_aux2, label, run_name):
    print("\n" + "#" * 70)
    print(f"# LAUNCHING {label}")
    print("#" * 70)

    exp = DeepSupExperiment(
        LAMBDA_DS3, LAMBDA_DS2, str(CONFIG_PATH), str(EXP_DIR), SEED, MU,
        run_name=f"_init_check_{run_name}",
        num_workers=0,
        enable_aux3=enable_aux3, enable_aux2=enable_aux2,
        condition_label=label,
    )
    run_initial_parameter_equality_gate(exp, label)
    exp.close_logs()
    del exp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    exp = DeepSupExperiment(
        LAMBDA_DS3, LAMBDA_DS2, str(CONFIG_PATH), str(EXP_DIR), SEED, MU,
        run_name=run_name,
        num_workers=None,  # falls back to configs/brats.yaml's num_workers=4
        enable_aux3=enable_aux3, enable_aux2=enable_aux2,
        condition_label=label,
    )
    exp.train(epochs=EPOCHS, checkpoint_every=CHECKPOINT_EVERY)
    del exp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("\n" + "=" * 70)
    print("STRUCTURAL PIVOT 1B: SCALE ABLATION -- D/4-only and D/2-only")
    print("=" * 70)

    run_condition(enable_aux3=True, enable_aux2=False, label="DeepSup_D4only", run_name="DeepSup_D4only_seed0")
    run_condition(enable_aux3=False, enable_aux2=True, label="DeepSup_D2only", run_name="DeepSup_D2only_seed0")

    print("\n" + "=" * 70)
    print("SCALE ABLATION: BOTH CONDITIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
