"""
Phase E39A/B: controlled lambda sweep for D4-only auxiliary supervision,
WITH per-epoch gradient-trajectory logging (|g_0|, |g_4|, cos(g_0,g_4),
|g_4^perp|, M_4). This is the first phase in the E36->E37->E38->E39 chain
that actually trains models -- justified because E38 found a real,
statistically robust signal (the pre-declared Q_r/P_Q quantity) that
survived a demanding, multi-part check, and this phase exists to test
whether that signal has genuine PREDICTIVE value across a controlled
lambda sweep, not just descriptive value across the three pre-existing
conditions E36-E38 examined.

Extends DeepSupExperiment (e25/train_deep_sup.py) via SUBCLASSING, not by
editing the base class -- matching this project's own established
extend-not-edit convention for validated training code. Everything except
lambda_ds3 and the gradient-logging addition is IDENTICAL to the proven
D4-only condition: architecture (UNet3D_v3), dataset split, preprocessing,
optimizer (AdamW), LR schedule (CosineAnnealingLR), seed (0), epochs (30),
mu (0.1), evaluation protocol (validate(), UNCHANGED). D2's own auxiliary
head is DISABLED (enable_aux2=False), matching D4-only's own proven
configuration exactly -- this sweep varies lambda_ds3 ONLY.

Conditions (pre-declared, per the user's exact specification):
    A            (lambda_ds3=0.0, i.e. D4 head architecturally present but
                  contributing zero loss/gradient -- the established
                  ablation-control pattern from e25's own scale ablation)
    lambda=0.125, 0.25, 0.5, 1.0, 2.0

GRADIENT TRAJECTORY LOGGING: once per epoch, on a FIXED, held-out batch of
8 validation subjects (identical across every epoch and every condition --
matching E36/E38's own established convention exactly, for direct
comparability to those phases' own numbers), BEFORE that epoch's training
begins:
    1. zero_grad, forward, backward on MAIN loss alone (seg_loss + mu*boundary_loss) -> g_0
    2. zero_grad, forward, backward on D4 loss alone (lambda_ds3 * aux3_loss) -> g_4
       (note: g_4 is measured WITH the current lambda_ds3 already applied,
       matching the prompt's own M_4 formula, which is defined on lambda*g_4,
       not on a lambda-independent unit gradient)
    3. Compute cos(g_0,g_4), |g_4^perp| = |g_4|*sqrt(1-cos^2) (closed-form,
       verified identical to direct vector projection in E38), and
       M_4 = |g_4^perp|^2 / (|g_0|^2 + eps).
    4. zero_grad again (these measurement backward passes must NOT affect
       the optimizer's real update for this epoch's training batches).

This measurement is a genuine EXTRA cost (2 extra backward passes per
epoch, on top of the real training epoch) but is cheap relative to a full
epoch of training (30+ real training batches), and matches the frequency
already decided with the user (once per epoch, not per-batch) to keep the
6-condition sweep affordable.

Output per condition: standard DeepSupExperiment checkpoints/logs, PLUS
E39_gradient_trajectory_{condition_name}.json (per-epoch g_0/g_4/cos/M_4).
"""
import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e25"))
from train_deep_sup import DeepSupExperiment  # noqa: E402

SEED = 0
MU = 0.1
EPOCHS = 30
CHECKPOINT_EVERY = 5
CONFIG_PATH = project_root / "configs" / "brats.yaml"
OUT_DIR = Path(__file__).parent
EPS = 1e-8

LAMBDA_GRID = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0]  # 0.0 = condition A (D4 head present, architecturally, but contributes zero loss)
FIXED_BATCH_SUBJECT_INDICES = list(range(8))  # SAME first-8 convention as E36/E38, for direct comparability


class E39Experiment(DeepSupExperiment):
    """Adds per-epoch gradient-trajectory logging on a fixed diagnostic
    batch, without touching the base class's own train_epoch()/train() --
    logging happens as an EXTRA step this subclass calls once per epoch,
    using its own separate forward/backward passes that are immediately
    discarded (zero_grad) before the base class's real training proceeds."""

    def __init__(self, lambda_ds3, config_path, exp_dir, seed, mu, run_name=None,
                 num_workers=None, condition_label=None):
        # D2 is unconditionally disabled and D4 unconditionally enabled here --
        # this sweep only ever varies lambda_ds3, matching D4-only's own
        # proven configuration exactly. All arguments passed explicitly by
        # keyword (no *args/**kwargs passthrough) so a future call-site
        # change can't silently mismatch positional argument order against
        # the base class's own __init__ signature.
        super().__init__(
            lambda_ds3=lambda_ds3, lambda_ds2=0.0, config_path=config_path, exp_dir=exp_dir,
            seed=seed, mu=mu, run_name=run_name, num_workers=num_workers,
            enable_aux3=True, enable_aux2=False, condition_label=condition_label,
        )
        self.trajectory = []

        # Build the SAME fixed diagnostic batch every epoch/condition uses
        # (matching E36/E38's own convention), taken from val_loader.dataset
        # directly (not the train loader) so it never overlaps with training data.
        imgs, masks = [], []
        for idx in FIXED_BATCH_SUBJECT_INDICES:
            img, mask, _ = self.val_loader.dataset[idx]
            imgs.append(img)
            masks.append(mask)
        self.diag_images = torch.stack(imgs).to(self.device)
        self.diag_masks = torch.stack(masks).to(self.device)

    def measure_gradient_trajectory(self):
        """Called ONCE PER EPOCH, before that epoch's real training. Model
        is put in eval() mode for this measurement (matching E36/E38's own
        decision: use each checkpoint's/epoch's own learned BN running
        stats, not this small diagnostic batch's own noisy batch stats),
        then restored to train() mode before real training proceeds."""
        was_training = self.model.training
        self.model.eval()

        # g_0: main loss alone
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(self.diag_images)
        probs, alpha, beta = outputs["probs"], outputs["alpha"], outputs["beta"]
        boundary_logit = outputs["boundary_logit"]
        focal_loss = self.focal_fn(probs, self.diag_masks)
        evidential_loss = self.evidential_fn(alpha, beta, self.diag_masks)
        seg_loss = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss
        boundary_loss = self.boundary_criterion(boundary_logit, self.diag_masks)
        loss_main = seg_loss + self.mu * boundary_loss
        loss_main.backward()
        g0_flat = torch.cat([p.grad.detach().reshape(-1) if p.grad is not None
                              else torch.zeros(p.numel(), device=p.device)
                              for p in self.model.parameters()])
        norm_g0 = float(g0_flat.norm().item())

        # g_4: D4 loss alone, WITH the current lambda_ds3 applied (matching
        # the prompt's own M_4 definition on lambda*g_4, not a unit gradient)
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(self.diag_images)
        aux_probs3 = outputs["aux_probs3"]
        mask_d4 = F.avg_pool3d(self.diag_masks, kernel_size=4, stride=4)
        aux3_loss = self.focal_fn(aux_probs3, mask_d4)
        loss_d4 = self.lambda_ds3 * aux3_loss
        if self.lambda_ds3 > 0:
            loss_d4.backward()
            g4_flat = torch.cat([p.grad.detach().reshape(-1) if p.grad is not None
                                  else torch.zeros(p.numel(), device=p.device)
                                  for p in self.model.parameters()])
        else:
            # lambda=0 (condition A): the D4 head still exists architecturally
            # and receives a forward pass, but contributes exactly zero loss.
            # Its "gradient" is trivially the zero vector -- computed this way
            # (rather than skipped) so g4_flat has the correct shape/device
            # for the rest of this function, and so norm_g4=0 is measured
            # directly rather than assumed.
            g4_flat = torch.zeros_like(g0_flat)
        norm_g4 = float(g4_flat.norm().item())

        if norm_g0 > 0 and norm_g4 > 0:
            cosine = float(torch.dot(g0_flat, g4_flat).item() / (norm_g0 * norm_g4))
        else:
            cosine = None

        if cosine is not None:
            sin_sq = max(0.0, 1.0 - cosine ** 2)
            perp_norm = norm_g4 * (sin_sq ** 0.5)
        else:
            perp_norm = norm_g4  # if g0 or g4 is exactly zero, the "orthogonal part" is just g4's own full magnitude (nothing to project away)

        M4 = (perp_norm ** 2) / (norm_g0 ** 2 + EPS)

        self.model.zero_grad(set_to_none=True)  # CRITICAL: discard measurement gradients before real training
        if was_training:
            self.model.train()

        record = {
            "epoch": self.epoch, "lambda_ds3": self.lambda_ds3,
            "norm_g0": norm_g0, "norm_g4": norm_g4, "cosine_g0_g4": cosine,
            "perp_norm_g4": perp_norm, "M4": M4,
            "loss_main_diag": float(loss_main.item()), "loss_d4_diag": float(loss_d4.item()),
        }
        self.trajectory.append(record)
        return record

    def train_with_trajectory(self, epochs, checkpoint_every=5):
        """The REAL training entry point for this subclass: identical to
        DeepSupExperiment.train()'s own loop structure (epoch loop, periodic
        checkpointing, CSV logging via self.w_metrics -- all reused verbatim
        via the parent class's own attributes), with ONE addition: a
        gradient-trajectory measurement immediately before each epoch's
        train_epoch() call."""
        print("\n" + "=" * 70)
        print(f"{self.condition_name}: lambda_ds3={self.lambda_ds3} [seed={self.seed}, mu={self.mu}]")
        print("=" * 70 + "\n")

        import time
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for epoch in range(epochs):
            self.epoch = epoch
            traj_rec = self.measure_gradient_trajectory()
            print(f"[{self.condition_name}] epoch {epoch+1} gradient trajectory: "
                  f"|g0|={traj_rec['norm_g0']:.4f} |g4|={traj_rec['norm_g4']:.4f} "
                  f"cos={traj_rec['cosine_g0_g4']} M4={traj_rec['M4']:.6f}", flush=True)

            t0 = time.time()
            train_metrics = self.train_epoch()
            val = self.validate()
            self.scheduler.step()
            epoch_time = time.time() - t0

            peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6) if self.device.type == "cuda" else 0.0

            print(
                f"[{self.condition_name}] Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s, peak_mem={peak_mem_mb:.0f}MB) | "
                f"Train: loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} "
                f"seg={train_metrics['seg_loss']:.4f} bnd={train_metrics['boundary_loss']:.4f} "
                f"aux3={train_metrics['aux3_loss']:.4f} aux2={train_metrics['aux2_loss']:.4f} | "
                f"Val: dice={val['dice']:.4f} precision={val['precision']:.4f} recall={val['recall']:.4f} "
                f"hd95={val['hd95']:.2f} ece={val['ece']:.4f}", flush=True
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

            # Written next to this condition's OWN checkpoints/logs
            # (self.exp_dir, set by the base class from the run_name/exp_dir
            # passed at construction), NOT a hardcoded module-level path --
            # a first version hardcoded this to OUT_DIR/"runs", which meant
            # a smoke test that redirected exp_dir to an isolated test
            # directory still silently wrote real trajectory files into the
            # production runs/ directory. Caught directly by inspecting the
            # smoke test's actual output location before trusting it matched
            # what was intended.
            with open(self.exp_dir / f"E39_gradient_trajectory_{self.condition_name}.json", "w") as f:
                json.dump(self.trajectory, f, indent=2)

        self.close_logs()
        print(f"\n[{self.condition_name}] Done. Best Val Dice: {self.best_val_dice:.4f}", flush=True)
        return self.best_val_dice


def run_one_condition(lam):
    label = "A_lambda0" if lam == 0.0 else f"lambda_{lam}"
    print(f"\n\n{'#'*70}\n# Starting condition: {label} (lambda_ds3={lam})\n{'#'*70}\n", flush=True)
    exp = E39Experiment(
        lambda_ds3=lam,
        config_path=str(CONFIG_PATH), exp_dir=str(OUT_DIR / "runs"), seed=SEED, mu=MU,
        run_name=label, num_workers=None, condition_label=label,
    )
    best_dice = exp.train_with_trajectory(epochs=EPOCHS, checkpoint_every=CHECKPOINT_EVERY)
    return label, best_dice


def main():
    results = {}
    for lam in LAMBDA_GRID:
        label, best_dice = run_one_condition(lam)
        results[label] = {"lambda_ds3": lam, "best_val_dice": best_dice}
        with open(OUT_DIR / "E39_lambda_sweep_summary.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n\n=== E39A lambda sweep complete ===", flush=True)
    for label, r in results.items():
        print(f"  {label}: lambda={r['lambda_ds3']}  best_val_dice={r['best_val_dice']:.4f}", flush=True)


if __name__ == "__main__":
    main()
