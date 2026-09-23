"""
Experiment C1: Adaptive Branch Optimizer (ABO) -- first implementation.

Everything except the gradient-update rule immediately before
optimizer.step() is frozen and identical to Phase A.5/B/C0: same
architecture, same losses (unweighted forms), same static loss weights
(focal_weight=0.5, evidential_weight=0.5), same AdamW/CosineAnnealingLR,
same lr=4e-4, same batch_size=8, same data split, same seed protocol.

Two modes:
  --mode monitor : ABOController runs and logs every step, but the
                   multiplier applied to gradients is forced to 1.0 --
                   i.e. the update rule is mathematically IDENTICAL to
                   the baseline's combined-loss backward (see
                   abo_grad_utils.apply_abo_update docstring: multiplier=1
                   reduces exactly to focal_weight*g_focal +
                   evidential_weight*g_evid for every parameter). This
                   mode exists purely to verify the new instrumentation
                   reproduces Phase B/C0 numbers before trusting it to
                   drive real gradient modification (Design Step 3).
  --mode active  : ABOController's computed multiplier is actually applied
                   to the trunk's evidential gradient contribution.

Update rule (trunk only; heads always receive their frozen-baseline
gradient unchanged -- see abo_grad_utils.apply_abo_update):
  trunk.grad = focal_weight * g_focal[trunk]
             + (alpha * delta) * evidential_weight * g_evid[trunk]
"""

import os
import sys
import csv
import math
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_fixed import UNet3D, FocalTverskyLoss, EvidentialBetaLoss

sys.path.insert(0, str(project_root / "experiments" / "exp01_diagnostics"))
from gradient_utils import get_param_groups, cosine_similarity, grad_norm, EMATracker  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp00b_baseline_convergence"))
from metrics import MetricAccumulator  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_c0_weight_ablation"))
from calibration import ECEAccumulator  # noqa: E402

from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from abo_controller import ABOController  # noqa: E402
from abo_grad_utils import capture_named_grads, trunk_norms, group_flat_vector, apply_abo_update, TRUNK_MODULE_NAMES  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ABOExperiment:
    def __init__(self, config_path, exp_dir, seed, mode, r_target=None, gamma=None, run_name=None):
        self.seed = seed
        self.mode = mode
        assert mode in ("monitor", "active")
        set_seed(seed)

        dir_name = run_name if run_name else f"{mode}_seed{seed}"
        self.exp_dir = Path(exp_dir) / dir_name
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.log_dir = self.exp_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        dataset_root = self.config["dataset"]["root_dir"]
        if not Path(dataset_root).is_absolute():
            dataset_root = project_root / dataset_root
        self.config["dataset"]["root_dir"] = str(dataset_root)

        self.device = torch.device(self.config.get("hardware", {}).get("device", "cpu"))
        self.best_val_dice = 0.0
        self.epoch = 0

        self.model = UNet3D(
            in_channels=self.config["model"]["in_channels"],
            out_channels=self.config["model"]["out_channels"]
        ).to(self.device)

        self.focal_fn = FocalTverskyLoss()
        self.evid_fn = EvidentialBetaLoss(weight=0.5)
        # Frozen static weights -- IDENTICAL to Phase A.5/B/C0 baseline.
        # ABO does not change these; it adds a dynamic multiplier ON TOP
        # of evidential_weight, applied only to the trunk contribution.
        self.focal_weight = 0.5
        self.evidential_weight = 0.5

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.config["training"]["epochs"], eta_min=1e-6
        )

        self.train_loader, self.val_loader = create_brats_loaders(
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["training"].get("num_workers", 0),
            root_dir=self.config["dataset"]["root_dir"],
            val_split=self.config["dataset"]["val_split"]
        )

        self.diag_ema = EMATracker(decay=0.9)  # visualization-only EMA, matches Phase B/C0 logging convention
        controller_kwargs = {}
        if r_target is not None:
            controller_kwargs["r_target"] = r_target
        if gamma is not None:
            controller_kwargs["gamma"] = gamma
        self.controller = ABOController(**controller_kwargs)  # the real controller driving gradient modification

        self._open_logs()

        print(f"[{mode}_seed{seed}] Training subjects: {len(self.train_loader.dataset)}")
        print(f"[{mode}_seed{seed}] Validation subjects: {len(self.val_loader.dataset)}")
        print(f"[{mode}_seed{seed}] Device: {self.device}")
        groups = get_param_groups(self.model)
        print(f"[{mode}_seed{seed}] Param groups: "
              f"trunk={sum(p.numel() for _, p in groups['trunk'])} "
              f"seg_head={sum(p.numel() for _, p in groups['seg_head'])} "
              f"evid_head={sum(p.numel() for _, p in groups['evid_head'])}")

    def _open_logs(self):
        self.f_norms = open(self.log_dir / "branch_gradient_norms.csv", "w", newline="")
        self.w_norms = csv.writer(self.f_norms)
        self.w_norms.writerow(["epoch", "batch", "focal_trunk", "evidential_trunk"])

        self.f_cos = open(self.log_dir / "cosine_similarity.csv", "w", newline="")
        self.w_cos = csv.writer(self.f_cos)
        self.w_cos.writerow(["epoch", "batch", "cosine_trunk"])

        self.f_ctrl = open(self.log_dir / "abo_controller.csv", "w", newline="")
        self.w_ctrl = csv.writer(self.f_ctrl)
        self.w_ctrl.writerow([
            "epoch", "batch", "raw_ratio", "ratio_ema",
            "alpha", "delta", "multiplier", "applied_multiplier",
            "evid_norm_mean_ema", "evid_norm_std_ema",
        ])

        self.f_loss = open(self.log_dir / "loss_history.csv", "w", newline="")
        self.w_loss = csv.writer(self.f_loss)
        self.w_loss.writerow(["epoch", "batch", "focal_loss", "evidential_loss", "total_loss"])

        self.f_metrics = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w_metrics = csv.writer(self.f_metrics)
        self.w_metrics.writerow([
            "epoch", "train_loss", "train_dice",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall", "val_f1", "val_hd95", "val_ece",
            "pct_batches_damped", "mean_effective_ratio", "epoch_time_sec",
        ])

    def close_logs(self):
        for f in (self.f_norms, self.f_cos, self.f_ctrl, self.f_loss, self.f_metrics):
            f.close()

    def train_epoch(self):
        self.model.train()
        acc = MetricAccumulator()
        damped_count = 0
        n_batches = 0
        effective_ratio_sum = 0.0

        pbar = tqdm(self.train_loader, desc=f"[{self.mode}] Epoch {self.epoch+1} [Train]")
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)

            focal_loss = self.focal_fn(outputs["probs"], masks)
            evidential_loss = self.evid_fn(outputs["alpha"], outputs["beta"], masks)

            if not torch.isfinite(focal_loss) or not torch.isfinite(evidential_loss):
                raise RuntimeError(
                    f"[{self.mode}] NaN/Inf loss at epoch {self.epoch} batch {batch_idx}: "
                    f"focal={focal_loss.item()} evidential={evidential_loss.item()}"
                )

            # --- Independent gradient extraction (Design Step 2) ---
            # torch.autograd.grad -- see abo_grad_utils.capture_named_grads
            # docstring for why this replaced backward()+.grad.clone()+
            # zero_grad() (lower peak memory, no redundant buffer copies,
            # no need to zero .grad between or after these two calls since
            # neither one ever writes to it).
            focal_grads = capture_named_grads(self.model, focal_loss, retain_graph=True)
            evid_grads = capture_named_grads(self.model, evidential_loss, retain_graph=False)

            # --- Controller (Design Steps 3-5) ---
            focal_trunk_norm, evid_trunk_norm = trunk_norms(focal_grads, evid_grads)
            alpha, delta, multiplier, diag = self.controller.step(focal_trunk_norm, evid_trunk_norm)

            applied_multiplier = 1.0 if self.mode == "monitor" else multiplier
            # "Requiring damping" means delta (the anomaly damper) actually
            # fired -- NOT that the combined multiplier differs from 1.0.
            # alpha alone is expected to differ from 1.0 on nearly every
            # batch by design (it continuously tracks the ratio EMA); that
            # is routine magnitude control, not an anomaly event. Only delta
            # dropping meaningfully below 1 indicates the anomaly damper
            # actually suppressed something. 0.9 is a REPORTING threshold
            # for this statistic only -- it does not gate any control
            # decision (delta itself is the smooth, threshold-free function;
            # this is purely for the epoch-level summary requested in the
            # design brief: "record the percentage of batches requiring damping").
            if delta < 0.9:
                damped_count += 1

            # Effective ratio: what the trunk actually received this batch,
            # |g_evidential_effective| / |g_focal| = raw_ratio * applied_multiplier.
            # raw_ratio alone (diag["raw_ratio"]) is PRE-multiplier -- see
            # phase_c1_abo_refactor memory note; this is the reconstruction
            # validated during the C2a/C2b sweep analysis, now logged directly
            # per-epoch instead of requiring post-hoc CSV joins.
            effective_ratio_sum += diag["raw_ratio"] * applied_multiplier

            # --- Apply update (Design Step 6) ---
            apply_abo_update(
                self.model, focal_grads, evid_grads,
                self.focal_weight, self.evidential_weight, applied_multiplier,
            )

            grad_total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_total_norm):
                raise RuntimeError(
                    f"[{self.mode}] Non-finite gradient norm at epoch {self.epoch} batch {batch_idx}"
                )

            self.optimizer.step()

            self._log_batch(batch_idx, focal_grads, evid_grads, focal_trunk_norm, evid_trunk_norm,
                             alpha, delta, multiplier, applied_multiplier, diag, focal_loss, evidential_loss)

            with torch.no_grad():
                total_val = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss
                acc.update(total_val.item(), outputs["probs"], masks, compute_hd95=False)

            n_batches += 1
            pbar.set_postfix({"loss": total_val.item(), "mult": applied_multiplier})

        summary = acc.summary()
        pct_damped = 100.0 * damped_count / max(1, n_batches)
        mean_effective_ratio = effective_ratio_sum / max(1, n_batches)
        return summary["loss"], summary["dice"], pct_damped, mean_effective_ratio

    def _log_batch(self, batch_idx, focal_grads, evid_grads, focal_trunk_norm, evid_trunk_norm,
                   alpha, delta, multiplier, applied_multiplier, diag, focal_loss, evidential_loss):
        self.w_norms.writerow([self.epoch, batch_idx, focal_trunk_norm, evid_trunk_norm])

        trunk_set = set(TRUNK_MODULE_NAMES)
        f_trunk_vec = group_flat_vector(focal_grads, trunk_set)
        e_trunk_vec = group_flat_vector(evid_grads, trunk_set)
        cos_trunk = cosine_similarity(f_trunk_vec, e_trunk_vec)
        self.w_cos.writerow([self.epoch, batch_idx, cos_trunk])

        self.w_ctrl.writerow([
            self.epoch, batch_idx, diag["raw_ratio"], diag["ratio_ema"],
            alpha, delta, multiplier, applied_multiplier,
            diag["evid_norm_mean_ema"], diag["evid_norm_std_ema"],
        ])

        total_val = (self.focal_weight * focal_loss + self.evidential_weight * evidential_loss).item()
        self.w_loss.writerow([self.epoch, batch_idx, focal_loss.item(), evidential_loss.item(), total_val])

        if batch_idx % 20 == 0:
            for f_ in (self.f_norms, self.f_cos, self.f_ctrl, self.f_loss):
                f_.flush()

    def validate(self):
        self.model.eval()
        acc = MetricAccumulator()
        ece_acc = ECEAccumulator(n_bins=15)

        pbar = tqdm(self.val_loader, desc=f"[{self.mode}] Epoch {self.epoch+1} [Val]")
        with torch.no_grad():
            for images, masks, _ in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                focal_loss = self.focal_fn(outputs["probs"], masks)
                evidential_loss = self.evid_fn(outputs["alpha"], outputs["beta"], masks)
                total = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss

                acc.update(total.item(), outputs["probs"], masks, compute_hd95=True)
                ece_acc.update(outputs["alpha"], outputs["beta"], masks)
                pbar.set_postfix({"loss": total.item()})

        summary = acc.summary()
        ece, _ = ece_acc.compute()
        summary["ece"] = ece
        return summary

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            "epoch": self.epoch, "seed": self.seed, "mode": self.mode,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice, "config": self.config,
        }
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
            print(f"[{self.mode}_seed{self.seed}] Best model saved (Dice: {self.best_val_dice:.4f})")

    def train(self, epochs):
        print("\n" + "=" * 70)
        print(f"EXPERIMENT C1: ABO [{self.mode} mode, seed={self.seed}]")
        print("=" * 70 + "\n")

        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()

            train_loss, train_dice, pct_damped, mean_effective_ratio = self.train_epoch()
            val = self.validate()
            self.scheduler.step()

            epoch_time = time.time() - t0
            print(
                f"[{self.mode}] Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
                f"Train: loss={train_loss:.4f} dice={train_dice:.4f} | "
                f"Val: loss={val['loss']:.4f} dice={val['dice']:.4f} "
                f"iou={val['iou']:.4f} precision={val['precision']:.4f} recall={val['recall']:.4f} "
                f"f1={val['f1']:.4f} hd95={val['hd95']:.2f} ece={val['ece']:.4f} | "
                f"damped={pct_damped:.1f}% eff_ratio={mean_effective_ratio:.4f}"
            )

            self.w_metrics.writerow([
                epoch, train_loss, train_dice,
                val["loss"], val["dice"], val["iou"], val["precision"], val["recall"], val["f1"],
                val["hd95"], val["ece"],
                pct_damped, mean_effective_ratio, epoch_time,
            ])
            self.f_metrics.flush()

            if val["dice"] > self.best_val_dice:
                self.best_val_dice = val["dice"]
                self.save_checkpoint(is_best=True)

        self.close_logs()
        print(f"\n[{self.mode}_seed{self.seed}] Done. Best Val Dice: {self.best_val_dice:.4f}")
        return self.best_val_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment C1: Adaptive Branch Optimizer")
    parser.add_argument("--config", default="../../configs/brats.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["monitor", "active"], required=True)
    parser.add_argument("--r_target", type=float, default=None,
                         help="Override ABOController.r_target (default: controller's own default, 0.294)")
    parser.add_argument("--gamma", type=float, default=None,
                         help="Override ABOController.gamma (default: controller's own default, 0.1)")
    parser.add_argument("--run_name", default=None,
                         help="Output subdir name (default: {mode}_seed{seed}); use for sweep runs to avoid collisions")
    args = parser.parse_args()

    exp_dir = Path(__file__).parent
    trainer = ABOExperiment(
        args.config, exp_dir, seed=args.seed, mode=args.mode,
        r_target=args.r_target, gamma=args.gamma, run_name=args.run_name,
    )
    trainer.train(args.epochs)
