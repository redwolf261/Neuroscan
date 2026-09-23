"""
Phase E29, Part 2: orchestrator for the resolution-ceiling experiment
(A64 / A96 / A128). Trains the SAME architecture (UNet3D_v2, condition-A
equivalent loss: FocalTversky+Evidential+boundary BCE, mu=0.1, no margin
term) at three input resolutions, same seed (0), same 30-epoch schedule,
same optimizer/scheduler configuration, same checkpoint-selection rule.

A64 is trained fresh here (not reusing the archived A_baseline_seed0
checkpoint) so that all three conditions are produced by the exact same
orchestration code path, batch-size bookkeeping, and logging schema --
avoiding any risk of comparing checkpoints produced under subtly different
harness versions. A64's result is expected to closely reproduce (not
necessarily bit-identically, since DataLoader worker nondeterminism and
non-deterministic CUDA ops are both present per the E27 project audit's
B6 finding) the archived A's 0.9063 best_val_dice -- this is itself a
useful internal consistency check, reported in the final E29 report.

DISCLOSED, documented deviation: batch size varies by resolution (64->8,
96->2, 128->1) due to GPU memory limits (RTX 5050, 8GB) -- verified via a
direct memory probe before writing this script. See train_e29_resolution.py's
own docstring for the exact peak-memory numbers.
"""
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e29"))
from train_e29_resolution import ResolutionExperiment, state_dict_hash  # noqa: E402

SEED = 0
MU = 0.1
EPOCHS = 30
CHECKPOINT_EVERY = 5
CONFIG_PATH = project_root / "configs" / "brats.yaml"
EXP_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e29" / "resolution_runs"

# (resolution, batch_size) -- batch sizes chosen from a direct GPU memory
# probe (8GB RTX 5050): 64^3/bs=8 matches A's own original batch size
# exactly; 96^3/bs=2 and 128^3/bs=1 are the largest batch sizes that fit
# with a safe margin (peak 4.25GB and 5.02GB respectively, vs 8.15GB total).
CONDITIONS = [(64, 8), (96, 2), (128, 1)]


def run_initial_parameter_equality_gate(resolution, batch_size):
    """Trivial-but-consistent-with-project-pattern gate: construct the
    experiment twice with the same seed, confirm identical initial
    parameters. This does NOT compare across resolutions (there is no
    meaningful "same initial parameters" claim to make across different
    training-loader configurations beyond the model itself, which is
    resolution-independent by construction -- a fully convolutional net)."""
    print("=" * 70)
    print(f"SEED-REPRODUCIBILITY GATE: A{resolution} (hard assertion, before real training)")
    print("=" * 70)

    exp1 = ResolutionExperiment(resolution, batch_size, str(CONFIG_PATH), str(EXP_DIR), SEED, MU,
                                 run_name=f"_init_check_A{resolution}_1", num_workers=0)
    exp2 = ResolutionExperiment(resolution, batch_size, str(CONFIG_PATH), str(EXP_DIR), SEED, MU,
                                 run_name=f"_init_check_A{resolution}_2", num_workers=0)

    hash1 = state_dict_hash(exp1.model.state_dict())
    hash2 = state_dict_hash(exp2.model.state_dict())
    print(f"  init 1: {hash1}")
    print(f"  init 2: {hash2}")

    exp1.close_logs()
    exp2.close_logs()

    assert hash1 == hash2, f"FATAL: same-seed re-construction gives different initial parameters for A{resolution} -- DO NOT PROCEED"
    print(f"GATE PASSED: A{resolution} seed={SEED} is reproducible across construction.")
    print("=" * 70 + "\n")

    del exp1, exp2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_condition(resolution, batch_size):
    print("\n" + "#" * 70)
    print(f"# LAUNCHING A{resolution} (resolution={resolution}^3, batch_size={batch_size})")
    print("#" * 70)

    run_initial_parameter_equality_gate(resolution, batch_size)

    exp = ResolutionExperiment(
        resolution, batch_size, str(CONFIG_PATH), str(EXP_DIR), SEED, MU,
        run_name=f"A{resolution}_seed{SEED}",
        num_workers=None,  # falls back to configs/brats.yaml's num_workers=4
    )
    exp.train(epochs=EPOCHS, checkpoint_every=CHECKPOINT_EVERY)
    del exp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("\n" + "=" * 70)
    print("PHASE E29: RESOLUTION-CEILING EXPERIMENT (A64 / A96 / A128)")
    print("=" * 70)

    for resolution, batch_size in CONDITIONS:
        run_condition(resolution, batch_size)

    print("\n" + "=" * 70)
    print("E29 RESOLUTION-CEILING EXPERIMENT: ALL CONDITIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
