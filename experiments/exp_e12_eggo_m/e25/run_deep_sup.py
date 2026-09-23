"""
Phase E25, Structural Pivot 1: orchestrator for Condition B (Deep
Supervision, no SC-TAM). Per PHASE_E25_NEXT_STRUCTURAL_PIVOT.md Section
7-9: single locked seed (0), same protocol as A (mu=0.1, 30 epochs, same
configs/brats.yaml, same checkpoint schedule), preserving dataset split/
preprocessing/optimizer/LR-schedule/checkpoint-selection identically.

Before any real training epoch: construct condition B and condition A
fresh, hash-compare initial parameters ON THE SHARED PARAMETER KEYS ONLY
(v3 has 2 extra modules -- aux_head3/aux_head2 -- that v2/A's model does
not; comparing the full state_dict would trivially mismatch on key
presence alone, which is not the meaningful question here) + optimizer
metadata -- HARD assertion, same mechanism as every prior C6-family
orchestrator's own gate.
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
from train_deep_sup import DeepSupExperiment  # noqa: E402

SEED = 0
MU = 0.1
EPOCHS = 30
CHECKPOINT_EVERY = 5
LAMBDA_DS3 = 0.9927  # PHASE_E25 Structural Pivot 1 calibration (e25b_calibrate_deep_supervision.py)
LAMBDA_DS2 = 1.0014
CONFIG_PATH = project_root / "configs" / "brats.yaml"
EXP_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs"


def run_initial_parameter_equality_gate():
    print("=" * 70)
    print("INITIAL-PARAMETER-EQUALITY GATE: DeepSup vs condition A (hard assertion, before any training)")
    print("=" * 70)

    objective_a = ObjectiveConfig(mode="baseline")
    exp_a = Gate6Experiment("A_check", objective_a, str(CONFIG_PATH), str(EXP_DIR), SEED, MU, 0.0,  # lambda_margin=0.0 -- A itself never used the margin term either way (mu-only path), matching this experiment's own SC-TAM-free scope
                             run_name="_init_check_A", num_workers=0)
    exp_ds = DeepSupExperiment(LAMBDA_DS3, LAMBDA_DS2, str(CONFIG_PATH), str(EXP_DIR), SEED, MU,
                                run_name="_init_check_DeepSup", num_workers=0)

    shared_keys = set(exp_a.model.state_dict().keys())  # v2's own keys -- the subset v3 also has, by construction (v3 extends v2)
    assert shared_keys.issubset(set(exp_ds.model.state_dict().keys())), "v3 does not contain all of v2's own parameter keys -- extension is broken"

    hash_a = exp_a.get_initial_state_hash()
    hash_ds = exp_ds.get_initial_state_hash(shared_keys=shared_keys)

    # exp_a's own get_initial_state_hash (Gate6Experiment's, unchanged) hashes its FULL state_dict --
    # since A's model (v2) has EXACTLY the shared keys and nothing more, this is equivalent to
    # hashing only the shared keys for A specifically. Verified directly, not assumed:
    hash_a_shared = None
    from train_deep_sup import state_dict_hash
    hash_a_shared = state_dict_hash(exp_a.model.state_dict(), keys_only=shared_keys)
    assert hash_a_shared == hash_a["model_state_hash"], "A's own full-state-dict hash differs from its shared-keys-only hash -- A has unexpected extra keys, investigate"

    print(f"A model_state hash (shared keys):        {hash_a_shared}")
    print(f"DeepSup model_state hash (shared keys):  {hash_ds['model_state_hash']}")
    print(f"A optimizer metadata:       {hash_a['optimizer_metadata']}")
    print(f"DeepSup optimizer metadata: {hash_ds['optimizer_metadata']}")

    exp_a.close_logs()
    exp_ds.close_logs()

    assert hash_a_shared == hash_ds["model_state_hash"], (
        f"FATAL: initial model parameters DIFFER between A and DeepSup on their SHARED keys -- "
        f"A={hash_a_shared[:16]}... DeepSup={hash_ds['model_state_hash'][:16]}... "
        f"DO NOT PROCEED TO TRAINING, diagnose the divergence first"
    )
    assert hash_a["optimizer_metadata"] == hash_ds["optimizer_metadata"], (
        f"FATAL: optimizer metadata DIFFERS between A and DeepSup -- "
        f"A={hash_a['optimizer_metadata']} DeepSup={hash_ds['optimizer_metadata']} -- "
        f"DO NOT PROCEED TO TRAINING, diagnose the divergence first"
    )

    print("\nGATE PASSED: theta_A^(0) == theta_DeepSup^(0) on all SHARED parameters, optimizer metadata identical.")
    print("(DeepSup's aux_head3/aux_head2 are NEW parameters with no A-side counterpart to compare against --")
    print(" their own initialization is standard PyTorch default init, not compared here, not a gate concern.)")
    print("=" * 70 + "\n")

    del exp_a, exp_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    run_initial_parameter_equality_gate()

    print("\n" + "#" * 70)
    print("# LAUNCHING CONDITION B (Deep Supervision at dec3/dec2, no SC-TAM)")
    print("#" * 70)
    exp_ds = DeepSupExperiment(
        LAMBDA_DS3, LAMBDA_DS2, str(CONFIG_PATH), str(EXP_DIR), SEED, MU,
        run_name="DeepSup_seed0",
        num_workers=None,  # falls back to configs/brats.yaml's num_workers=4 -- applying the Gate 6/C6-2/C6-3 lesson directly
        enable_aux3=True, enable_aux2=True,  # the "both" condition -- ALREADY RUN, this file preserved for reproducibility, not rerun here
    )
    exp_ds.train(epochs=EPOCHS, checkpoint_every=CHECKPOINT_EVERY)
    del exp_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("DEEP SUPERVISION TRAINING (CONDITION B): COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
