"""
Phase E44: Relative-Convergence-Gap Weighting (RCGW) -- pilot training run.

ALGORITHMIC MODIFICATION (not another diagnostic phase), per the project's
explicit strategic pivot: validated D4 supervision -> one algorithmic
modification -> controlled training -> >=1pp or stop.

MATHEMATICAL DEFINITION:
    r_D4(t) = L_D4(t) / L_D4(0)      -- D4's own loss, relative to its OWN value at the start of training
    r_0(t)  = L_0(t)  / L_0(0)       -- main loss, relative to ITS OWN value at the start of training
    G(t)    = r_D4(t) / (r_0(t) + eps)   -- relative-convergence-gap ratio
    lambda_ds3(t) = lambda_base * clip(G(t), G_MIN, G_MAX)

MECHANISTIC JUSTIFICATION (grounded in this project's own SURVIVING findings,
not a new mechanism hunt -- see project_requirements_and_status_v2 memory):
    Finding D: D4's auxiliary loss stays substantially larger, relative to
    its own starting value, than main's loss does relative to ITS own
    starting value, especially late in training (E39A/B/C's own measured
    per-epoch loss trajectories, independently confirmed at every lambda
    tested). This means a FIXED lambda under-weights D4 exactly when D4 has
    the most relatively-unconverged signal left to contribute, and
    over-weights it early when both losses are dropping together. RCGW
    directly measures this gap, per-epoch, from the ACTUAL training
    trajectory of THIS run (not a fixed hyperparameter schedule chosen in
    advance, not a heuristic decay curve) and scales lambda accordingly.

DISTINCTION FROM PRIOR ART (checked before building, per requirement 12 --
literature search is not a substitute for a working algorithm, so this is a
brief, targeted check, not exhaustive):
    - GradNorm-family methods (Chen et al. 2018 and descendants) balance
      GRADIENT NORMS across tasks to equalize training rates -- RCGW uses
      LOSS VALUE ratios (relative to each task's own t=0 value), not
      gradient norms, and requires no extra backward pass or gradient-norm
      computation, making it far cheaper.
    - "Loss ratio at step t vs step 0" scheduling exists in the broader
      multi-task literature (2025-2026 search confirmed this is an active,
      populated area) -- RCGW's specific contribution is the RATIO OF TWO
      SUCH RELATIVE-PROGRESS RATIOS (a gap-of-gaps, not a single task's own
      loss ratio), applied specifically to a FIXED-RESOLUTION deep-
      supervision auxiliary head hierarchy (not general multi-task heads),
      and derived from and validated against this project's own directly
      measured D4-vs-main convergence-rate divergence (E39A/B/C), not a
      generically assumed schedule.
    - This is disclosed as SITTING CLOSE to existing loss-ratio-based
      dynamic weighting work, not as an unrelated invention -- genuine
      novelty, if any, is in the specific gap-of-relative-progress
      formulation and its grounding in this project's own measured
      mechanism, not in the general idea of "adaptive auxiliary weighting."

PRE-DECLARED SUCCESS CRITERION: >=1.0 percentage point Dice improvement
over the canonical baseline (0.9063, e24/gate6_runs/A_baseline_seed0),
matching this project's own primary success criterion exactly -- NOT the
E39 sweep's own 0.9081 baseline, which belongs to a different run/config
(explicitly flagged per project_requirements_and_status_v2's baseline-
mixing warning).

PRE-DECLARED KILL CONDITION: if a fast pilot (reduced epoch count) shows no
directional improvement over D4-only's own already-established +0.33pp,
this is killed without a full 30-epoch commitment.

Extends DeepSupExperiment via subclassing (matching E39's own established
pattern), NOT by editing the base class.
"""
import sys
import json
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e25"))
from train_deep_sup import DeepSupExperiment  # noqa: E402

SEED = 0
MU = 0.1
CONFIG_PATH = project_root / "configs" / "brats.yaml"
OUT_DIR = Path(__file__).parent
EPS = 1e-6

LAMBDA_BASE = 0.9927  # SAME calibration constant as every prior D4 supervision run (e25/e39) -- reused, not re-derived, so lambda(t)=lambda_base at G(t)=1 recovers the EXACT already-validated fixed-lambda condition as a special case
G_MIN = 0.2   # floor: never let D4's weight collapse below 20% of its base calibration
G_MAX = 4.0   # ceiling: never let D4's weight exceed 4x base calibration (bounded, per requirement 9's "no denominator tricks" -- G(t) is a ratio and MUST be bounded before use)


class E44RCGWExperiment(DeepSupExperiment):
    """D4-only architecture (D2 disabled, matching D4-only's own proven
    config exactly), but lambda_ds3 is recomputed EVERY EPOCH from the
    measured relative-convergence-gap ratio, using each epoch's own
    train-set mean seg_loss (as L_0) and mean aux3_loss (as L_D4, BEFORE
    lambda is applied -- i.e. the raw, unweighted D4 loss value, so the
    ratio measures the TASK's own convergence, not a value already
    distorted by a changing lambda)."""

    def __init__(self, *args, **kwargs):
        super().__init__(lambda_ds3=LAMBDA_BASE, lambda_ds2=0.0, enable_aux3=True, enable_aux2=False,
                          *args, **kwargs)
        self.L_D4_0 = None  # set after epoch 0's own measurement
        self.L_0_0 = None
        self.rcgw_trajectory = []

    def train_with_schedule(self, epochs, checkpoint_every=5):
        print("\n" + "=" * 70)
        print(f"{self.condition_name}: RCGW pilot [seed={self.seed}, mu={self.mu}, "
              f"lambda_base={LAMBDA_BASE}, G_MIN={G_MIN}, G_MAX={G_MAX}]")
        print("=" * 70 + "\n", flush=True)

        import time
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()

            train_metrics = self.train_epoch()
            val = self.validate()
            self.scheduler.step()
            epoch_time = time.time() - t0

            # Measure THIS epoch's raw L_0 (seg_loss, unweighted by mu*boundary
            # -- using seg_loss alone as L_0 for the ratio, matching the
            # prompt's own L_0 = L_64 notation as the PRIMARY segmentation
            # objective, not the mu-scaled combined main loss, to keep the
            # ratio's numerator/denominator on comparable footing -- both are
            # "raw task loss values," not loss-plus-regularizer composites)
            # and raw L_D4 (aux3_loss, UNWEIGHTED -- train_epoch() already
            # divides by n_batches, giving the epoch's mean raw aux3_loss).
            L_0_t = train_metrics["seg_loss"]
            L_D4_t = train_metrics["aux3_loss"]

            if epoch == 0:
                self.L_0_0 = max(L_0_t, EPS)
                self.L_D4_0 = max(L_D4_t, EPS)
                G_t = 1.0  # by definition at t=0
            else:
                r_D4 = L_D4_t / self.L_D4_0
                r_0 = L_0_t / self.L_0_0
                G_t = r_D4 / (r_0 + EPS)
                G_t = float(min(max(G_t, G_MIN), G_MAX))  # bounded, per requirement 9

            new_lambda = LAMBDA_BASE * G_t
            self.rcgw_trajectory.append({
                "epoch": epoch, "L_0_t": L_0_t, "L_D4_t": L_D4_t, "G_t": G_t,
                "lambda_ds3_used_this_epoch": self.lambda_ds3, "lambda_ds3_for_next_epoch": new_lambda,
                "val_dice": val["dice"],
            })

            peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6) if self.device.type == "cuda" else 0.0
            print(
                f"[{self.condition_name}] Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
                f"L_0={L_0_t:.4f} L_D4={L_D4_t:.4f} G(t)={G_t:.3f} "
                f"lambda_used={self.lambda_ds3:.4f} -> next={new_lambda:.4f} | "
                f"Val dice={val['dice']:.4f}", flush=True
            )

            self.w_metrics.writerow([
                epoch, train_metrics["loss"], train_metrics["dice"], train_metrics["seg_loss"],
                train_metrics["boundary_loss"], train_metrics["aux3_loss"], train_metrics["aux2_loss"],
                val["loss"], val["dice"], val["iou"], val["precision"], val["recall"], val["f1"],
                val["hd95"], val["ece"], val["boundary_bce"], val["boundary_accuracy_proxy"],
                epoch_time, peak_mem_mb,
            ])
            self.f_metrics.flush()

            is_best = val["dice"] > self.best_val_dice
            if is_best:
                self.best_val_dice = val["dice"]

            epoch_1indexed = epoch + 1
            is_periodic = (epoch_1indexed % checkpoint_every == 0) or epoch_1indexed == 1
            is_final = epoch_1indexed == epochs
            self.save_checkpoint(is_best=is_best, is_periodic=(is_periodic or is_final))

            # Apply the NEW lambda starting NEXT epoch (this epoch's own
            # training already happened at the OLD lambda -- the schedule is
            # causal, computed from what was JUST measured, not from a value
            # not yet observed).
            self.lambda_ds3 = new_lambda

            # Written to THIS run's own exp_dir (set by the base class from
            # run_name), not a hardcoded module-level path -- E39's first
            # draft had exactly this bug (path didn't include the per-run
            # subdirectory), caught and fixed there; applying that same fix
            # here from the start rather than repeating the mistake.
            with open(self.exp_dir / f"E44_rcgw_trajectory_{self.condition_name}.json", "w") as f:
                json.dump(self.rcgw_trajectory, f, indent=2)

        self.close_logs()
        print(f"\n[{self.condition_name}] Done. Best Val Dice: {self.best_val_dice:.4f}", flush=True)
        return self.best_val_dice


def main(epochs, run_name):
    exp = E44RCGWExperiment(
        config_path=str(CONFIG_PATH), exp_dir=str(OUT_DIR / "runs"), seed=SEED, mu=MU,
        run_name=run_name, num_workers=None, condition_label=run_name,
    )
    best_dice = exp.train_with_schedule(epochs=epochs, checkpoint_every=5)
    print(f"\n=== E44 RCGW pilot complete: best_val_dice={best_dice:.4f} ===", flush=True)
    return best_dice


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--run_name", type=str, default="RCGW_seed0")
    args = parser.parse_args()
    main(args.epochs, args.run_name)
