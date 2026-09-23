"""
Experiment 01: Gradient Diagnostics (Phase B)

Research Question:
  Does genuine gradient conflict exist between the Segmentation branch
  (FocalTverskyLoss, via seg_head) and the Evidential branch
  (EvidentialBetaLoss, via evidential_head) at the shared trunk, and is
  that conflict concentrated in the trunk vs. the independent heads?

Method:
  - Same model, same data, same optimizer/scheduler/config as the frozen
    Phase A.5 baseline (batch_size=8, lr=4e-4, 50 epochs) -- training
    dynamics are UNCHANGED. This experiment only adds instrumentation:
    two extra backward passes per batch (focal-only, evidential-only)
    to capture each loss's gradient independently, per parameter group
    (trunk / seg_head / evid_head), before the real combined-loss
    backward + optimizer.step() runs.
  - Logs four CSVs: branch_gradient_norms, cosine_similarity,
    gradient_variance (EMA), loss_history.

This does NOT modify ABO or the optimizer. It only measures.
"""

import os
import sys
import csv
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
from Dataset.brats_dataset import create_brats_loaders
from gradient_utils import (
    get_param_groups, compute_branch_gradients, cosine_similarity, grad_norm, EMATracker
)

sys.path.insert(0, str(project_root / "experiments" / "exp00b_baseline_convergence"))
from metrics import MetricAccumulator  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DiagnosticExperiment:
    def __init__(self, config_path, exp_dir, seed):
        self.seed = seed
        set_seed(seed)

        self.exp_dir = Path(exp_dir) / f"seed_{seed}"
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

        # Individual loss modules (not HybridLoss) so we can compute each
        # loss's scalar separately -- this is required for independent backward.
        self.focal_fn = FocalTverskyLoss()
        self.evid_fn = EvidentialBetaLoss(weight=0.5)
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

        self.ema = EMATracker(decay=0.9)

        # CSV writers
        self.f_norms = open(self.log_dir / "branch_gradient_norms.csv", "w", newline="")
        self.w_norms = csv.writer(self.f_norms)
        self.w_norms.writerow([
            "epoch", "batch",
            "focal_trunk", "focal_seg_head", "focal_evid_head",
            "evidential_trunk", "evidential_seg_head", "evidential_evid_head",
        ])

        # NOTE: cosine_seg_head and cosine_evid_head will be blank (None) every
        # row. focal_loss has no gradient path to evidential_head's params and
        # vice versa (disjoint branches), so one side of each of those cosines
        # is always the zero vector -- the value is architecturally undefined,
        # not a measured zero. Only cosine_trunk is a real conflict signal:
        # both losses flow through the shared trunk, so it compares two
        # genuinely independent, nonzero gradient vectors.
        self.f_cos = open(self.log_dir / "cosine_similarity.csv", "w", newline="")
        self.w_cos = csv.writer(self.f_cos)
        self.w_cos.writerow(["epoch", "batch", "cosine_trunk", "cosine_seg_head", "cosine_evid_head"])

        self.f_var = open(self.log_dir / "gradient_variance.csv", "w", newline="")
        self.w_var = csv.writer(self.f_var)
        self.w_var.writerow([
            "epoch", "batch",
            "focal_trunk_ema_mean", "focal_trunk_ema_var",
            "evidential_trunk_ema_mean", "evidential_trunk_ema_var",
            "grad_ratio_evid_over_focal_trunk",
        ])

        self.f_loss = open(self.log_dir / "loss_history.csv", "w", newline="")
        self.w_loss = csv.writer(self.f_loss)
        self.w_loss.writerow(["epoch", "batch", "focal_loss", "evidential_loss", "total_loss"])

        print(f"[seed {seed}] Training subjects: {len(self.train_loader.dataset)}")
        print(f"[seed {seed}] Validation subjects: {len(self.val_loader.dataset)}")
        print(f"[seed {seed}] Device: {self.device}")

        groups = get_param_groups(self.model)
        print(f"[seed {seed}] Param groups: "
              f"trunk={sum(p.numel() for _, p in groups['trunk'])} "
              f"seg_head={sum(p.numel() for _, p in groups['seg_head'])} "
              f"evid_head={sum(p.numel() for _, p in groups['evid_head'])}")

    def close_logs(self):
        for f in (self.f_norms, self.f_cos, self.f_var, self.f_loss):
            f.close()

    def train_epoch(self, batch_offset=0):
        self.model.train()
        total_loss, total_dice, n_batches = 0.0, 0.0, 0
        acc = MetricAccumulator()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]")
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)

            focal_loss = self.focal_fn(outputs["probs"], masks)
            evidential_loss = self.evid_fn(outputs["alpha"], outputs["beta"], masks)

            # --- Diagnostics: independent gradients, does not affect training ---
            branch_grads = compute_branch_gradients(self.model, focal_loss, evidential_loss)
            self._log_diagnostics(batch_idx, branch_grads, focal_loss, evidential_loss)

            # --- Real training step: combined loss, single backward, as in Phase A.5 ---
            total = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            with torch.no_grad():
                acc.update(total.item(), outputs["probs"], masks, compute_hd95=False)

            total_loss += total.item()
            n_batches += 1
            pbar.set_postfix({"loss": total.item()})

        summary = acc.summary()
        return summary["loss"], summary["dice"]

    def _log_diagnostics(self, batch_idx, branch_grads, focal_loss, evidential_loss):
        f = branch_grads["focal"]
        e = branch_grads["evidential"]

        norms = {
            "focal_trunk": grad_norm(f["trunk"]),
            "focal_seg_head": grad_norm(f["seg_head"]),
            "focal_evid_head": grad_norm(f["evid_head"]),
            "evidential_trunk": grad_norm(e["trunk"]),
            "evidential_seg_head": grad_norm(e["seg_head"]),
            "evidential_evid_head": grad_norm(e["evid_head"]),
        }
        self.w_norms.writerow([self.epoch, batch_idx, *[norms[k] for k in [
            "focal_trunk", "focal_seg_head", "focal_evid_head",
            "evidential_trunk", "evidential_seg_head", "evidential_evid_head",
        ]]])

        cos_trunk = cosine_similarity(f["trunk"], e["trunk"])
        cos_seg = cosine_similarity(f["seg_head"], e["seg_head"])
        cos_evid = cosine_similarity(f["evid_head"], e["evid_head"])
        self.w_cos.writerow([self.epoch, batch_idx, cos_trunk, cos_seg, cos_evid])

        ft_mean, ft_var = self.ema.update("focal_trunk", norms["focal_trunk"])
        et_mean, et_var = self.ema.update("evidential_trunk", norms["evidential_trunk"])
        ratio = norms["evidential_trunk"] / (norms["focal_trunk"] + 1e-12)
        self.w_var.writerow([self.epoch, batch_idx, ft_mean, ft_var, et_mean, et_var, ratio])

        total_val = (self.focal_weight * focal_loss + self.evidential_weight * evidential_loss).item()
        self.w_loss.writerow([self.epoch, batch_idx, focal_loss.item(), evidential_loss.item(), total_val])

        # Flush periodically so partial progress survives an interrupted run
        if batch_idx % 20 == 0:
            for f_ in (self.f_norms, self.f_cos, self.f_var, self.f_loss):
                f_.flush()

    def validate(self):
        self.model.eval()
        acc = MetricAccumulator()
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch+1} [Val]")
        with torch.no_grad():
            for images, masks, _ in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                focal_loss = self.focal_fn(outputs["probs"], masks)
                evidential_loss = self.evid_fn(outputs["alpha"], outputs["beta"], masks)
                total = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss
                acc.update(total.item(), outputs["probs"], masks, compute_hd95=True)
                pbar.set_postfix({"loss": total.item()})
        summary = acc.summary()
        return summary["loss"], summary["dice"]

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            "epoch": self.epoch, "seed": self.seed,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice, "config": self.config,
        }
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
            print(f"[seed {self.seed}] Best model saved (Dice: {self.best_val_dice:.4f})")

    def train(self, epochs):
        print("\n" + "=" * 70)
        print(f"EXPERIMENT 01: Gradient Diagnostics (seed={self.seed})")
        print("=" * 70 + "\n")

        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()

            train_loss, train_dice = self.train_epoch()
            val_loss, val_dice = self.validate()
            self.scheduler.step()

            epoch_time = time.time() - t0
            print(
                f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
                f"Train: loss={train_loss:.4f} dice={train_dice:.4f} | "
                f"Val: loss={val_loss:.4f} dice={val_dice:.4f}"
            )

            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.save_checkpoint(is_best=True)

        self.close_logs()
        print(f"\n[seed {self.seed}] Done. Best Val Dice: {self.best_val_dice:.4f}")
        print(f"[seed {self.seed}] Diagnostic logs: {self.log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 01: Gradient Diagnostics")
    parser.add_argument("--config", default="../../configs/brats.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    exp_dir = Path(__file__).parent
    trainer = DiagnosticExperiment(args.config, exp_dir, seed=args.seed)
    trainer.train(args.epochs)
