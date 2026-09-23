"""
Phase E44-B: Relative-Convergence-Gap Weighting, SATURATING formulation.

WHY THIS EXISTS: E44's first formulation (train_e44_rcgw.py),
G(t) = r_D4(t) / (r_0(t)+eps), lambda(t) = lambda_base * clip(G(t), G_MIN, G_MAX),
was piloted for 15 epochs and KILLED. Diagnosed precisely before attempting
any fix: G(t) grew ROUGHLY LINEARLY across the entire 15-epoch window (1.0 ->
2.04, still climbing at epoch 14 with no sign of leveling off), because once
L_D4 plateaus while L_0 keeps shrinking (E39C's own independently-documented
pattern, true for MOST of training, not a transient), the raw ratio G(t) has
NO restoring force -- it just keeps growing for as long as that asymmetry
holds. The external G_MAX=4.0 clip was being approached BY TREND, not
avoided as a safety margin -- a real design failure in the ratio itself, not
merely a case of needing a tighter bound. This also coincided with one clear
training instability (epoch 6 dice dropped to 0.7135 vs D4-only's own
reference 0.8463 at the same epoch, exactly as G(t) first crossed 1.0).

FIX: replace the raw, externally-clipped ratio with a STRUCTURALLY bounded
transform of the SAME underlying signal (same L_D4/L_0 relative-progress
measurement, same mechanistic justification), using tanh to saturate
SMOOTHLY as the gap grows, rather than growing linearly and needing an
external clip:

    g(t)      = ln( r_D4(t) / (r_0(t) + eps) )        -- log of the SAME ratio E44's first version used
    lambda(t) = lambda_base * exp( K * tanh( g(t) / S ) )

Verified directly (small numeric check, not assumed) before use: this maps
g(t)=0 (i.e. D4 and main progressing at the same relative rate) to
lambda(t)=lambda_base exactly (recovering the calibrated baseline as a fixed
point, same property E44's first version had), and asymptotes SMOOTHLY
toward lambda_base*exp(K) and lambda_base*exp(-K) as g(t)->+-infinity,
NEVER exceeding those bounds by construction (no external clip needed).
K = ln(4) chosen so the bounds match E44's own G_MIN=0.2/G_MAX=4.0 exactly
(a like-for-like comparison against the killed formulation, not a new
arbitrary choice), S=1.0 sets how quickly the saturation kicks in (S=1.0
means the ratio has to move by a factor of e before the schedule is
roughly halfway to its asymptote -- a soft, gradual saturation, not a hard
knee).

PRE-DECLARED SUCCESS CRITERION: >=1.0pp Dice improvement over the CANONICAL
baseline (0.9063, e24/gate6_runs/A_baseline_seed0) on the full 30-epoch run,
matching project_requirements_and_status_v2's own primary criterion exactly.

PRE-DECLARED KILL CONDITION: if this pilot (same 15-epoch window as E44's
first attempt, for a direct apples-to-apples comparison) still shows the
epoch-6-style instability, or still shows no directional improvement over
D4-only's own +0.33pp reference trajectory, this formulation is killed too,
and the RCGW family is abandoned in favor of a genuinely different
algorithmic direction -- not a third patched ratio.
"""
import sys
import json
import math
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

LAMBDA_BASE = 0.9927
K_BOUND = math.log(4.0)  # matches E44's own G_MAX=4.0/G_MIN=0.25(~=1/4) bounds exactly, for a like-for-like comparison
S_SCALE = 1.0


class E44BSaturatingExperiment(DeepSupExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(lambda_ds3=LAMBDA_BASE, lambda_ds2=0.0, enable_aux3=True, enable_aux2=False,
                          *args, **kwargs)
        self.L_D4_0 = None
        self.L_0_0 = None
        self.rcgw_trajectory = []

    def train_with_schedule(self, epochs, checkpoint_every=5):
        print("\n" + "=" * 70)
        print(f"{self.condition_name}: RCGW-saturating pilot [seed={self.seed}, mu={self.mu}, "
              f"lambda_base={LAMBDA_BASE}, K={K_BOUND:.4f}, S={S_SCALE}]")
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

            L_0_t = train_metrics["seg_loss"]
            L_D4_t = train_metrics["aux3_loss"]

            if epoch == 0:
                self.L_0_0 = max(L_0_t, EPS)
                self.L_D4_0 = max(L_D4_t, EPS)
                g_t = 0.0
                factor = 1.0
            else:
                r_D4 = L_D4_t / self.L_D4_0
                r_0 = L_0_t / self.L_0_0
                raw_ratio = r_D4 / (r_0 + EPS)
                g_t = math.log(max(raw_ratio, EPS))
                factor = math.exp(K_BOUND * math.tanh(g_t / S_SCALE))

            new_lambda = LAMBDA_BASE * factor
            self.rcgw_trajectory.append({
                "epoch": epoch, "L_0_t": L_0_t, "L_D4_t": L_D4_t, "g_t": g_t, "factor": factor,
                "lambda_ds3_used_this_epoch": self.lambda_ds3, "lambda_ds3_for_next_epoch": new_lambda,
                "val_dice": val["dice"],
            })

            print(
                f"[{self.condition_name}] Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
                f"L_0={L_0_t:.4f} L_D4={L_D4_t:.4f} g(t)={g_t:+.3f} factor={factor:.3f} "
                f"lambda_used={self.lambda_ds3:.4f} -> next={new_lambda:.4f} | "
                f"Val dice={val['dice']:.4f}", flush=True
            )

            peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6) if self.device.type == "cuda" else 0.0
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

            self.lambda_ds3 = new_lambda

            with open(self.exp_dir / f"E44B_rcgw_trajectory_{self.condition_name}.json", "w") as f:
                json.dump(self.rcgw_trajectory, f, indent=2)

        self.close_logs()
        print(f"\n[{self.condition_name}] Done. Best Val Dice: {self.best_val_dice:.4f}", flush=True)
        return self.best_val_dice


def main(epochs, run_name):
    exp = E44BSaturatingExperiment(
        config_path=str(CONFIG_PATH), exp_dir=str(OUT_DIR / "runs"), seed=SEED, mu=MU,
        run_name=run_name, num_workers=None, condition_label=run_name,
    )
    best_dice = exp.train_with_schedule(epochs=epochs, checkpoint_every=5)
    print(f"\n=== E44B RCGW-saturating pilot complete: best_val_dice={best_dice:.4f} ===", flush=True)
    return best_dice


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--run_name", type=str, default="RCGW_B_pilot_seed0")
    args = parser.parse_args()
    main(args.epochs, args.run_name)
