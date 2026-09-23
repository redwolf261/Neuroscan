"""
Experiment C0: Fixed-Weight Ablation (Focal vs Evidential)

Research Question:
  Phase B showed trunk gradient conflict is transient (resolves by epoch 3)
  but the gradient-magnitude ratio (||g_evidential|| / ||g_focal||) collapses
  to ~0.1 and stays there. Two hypotheses explain this:

    H1: Conflict genuinely resolved -- both objectives agree, evidential's
        smaller trunk-gradient share reflects settled optimization.
        Predicts: changing the loss weighting should NOT meaningfully help
        the evidential branch or change the ratio's steady-state behavior.

    H2: Evidential became too weak to matter -- the fixed 0.5/0.5 weighting
        lets focal dominate, evidential's gradient shrinks not because
        it's satisfied but because it's drowned out.
        Predicts: raising evidential_weight should restore its trunk
        influence, improve calibration (possibly Dice), without destabilizing
        training (conflict should not return with modest weight changes).

Method:
  Three otherwise-identical training runs (seed=0, same architecture, same
  batch_size=8/lr=4e-4, same data split as Phase A.5/B), varying only
  (focal_weight, evidential_weight):
    - 0.8 / 0.2  (focal-dominant)
    - 0.5 / 0.5  (baseline, matches Phase A.5/B)
    - 0.2 / 0.8  (evidential-dominant)

  Reuses the exact gradient-diagnostic instrumentation from Phase B
  (gradient_utils.py, independent backward passes, non-invasive to the
  real training step) plus adds ECE calibration measurement on validation.

Metrics per run: Dice, IoU, F1, HD95 (segmentation quality),
ECE (calibration), gradient ratio and cosine_trunk trajectories
(same diagnostic signal as Phase B, to see whether weighting shifts them).
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

sys.path.insert(0, str(project_root / "experiments" / "exp01_diagnostics"))
from gradient_utils import (  # noqa: E402
    get_param_groups, compute_branch_gradients, cosine_similarity, grad_norm, EMATracker
)

sys.path.insert(0, str(project_root / "experiments" / "exp00b_baseline_convergence"))
from metrics import MetricAccumulator  # noqa: E402

from calibration import ECEAccumulator  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WeightAblationExperiment:
    def __init__(self, config_path, exp_dir, seed, focal_weight, evidential_weight):
        self.seed = seed
        self.focal_weight = focal_weight
        self.evidential_weight = evidential_weight
        set_seed(seed)

        tag = f"focal{focal_weight:.1f}_evid{evidential_weight:.1f}"
        self.exp_dir = Path(exp_dir) / tag
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

        self.f_norms = open(self.log_dir / "branch_gradient_norms.csv", "w", newline="")
        self.w_norms = csv.writer(self.f_norms)
        self.w_norms.writerow([
            "epoch", "batch",
            "focal_trunk", "focal_seg_head", "focal_evid_head",
            "evidential_trunk", "evidential_seg_head", "evidential_evid_head",
        ])

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

        self.f_metrics = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w_metrics = csv.writer(self.f_metrics)
        self.w_metrics.writerow([
            "epoch", "train_loss", "train_dice",
            "val_loss", "val_dice", "val_iou", "val_f1", "val_hd95", "val_ece",
            "epoch_time_sec",
        ])

        print(f"[{tag}] seed={seed} Training subjects: {len(self.train_loader.dataset)}")
        print(f"[{tag}] Validation subjects: {len(self.val_loader.dataset)}")
        print(f"[{tag}] Device: {self.device}")
        self.tag = tag

    def close_logs(self):
        for f in (self.f_norms, self.f_cos, self.f_var, self.f_loss, self.f_metrics):
            f.close()

    def train_epoch(self):
        self.model.train()
        acc = MetricAccumulator()

        pbar = tqdm(self.train_loader, desc=f"[{self.tag}] Epoch {self.epoch+1} [Train]")
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)

            focal_loss = self.focal_fn(outputs["probs"], masks)
            evidential_loss = self.evid_fn(outputs["alpha"], outputs["beta"], masks)

            branch_grads = compute_branch_gradients(self.model, focal_loss, evidential_loss)
            self._log_diagnostics(batch_idx, branch_grads, focal_loss, evidential_loss)

            total = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            with torch.no_grad():
                acc.update(total.item(), outputs["probs"], masks, compute_hd95=False)

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

        if batch_idx % 20 == 0:
            for f_ in (self.f_norms, self.f_cos, self.f_var, self.f_loss):
                f_.flush()

    def validate(self):
        self.model.eval()
        acc = MetricAccumulator()
        ece_acc = ECEAccumulator(n_bins=15)

        pbar = tqdm(self.val_loader, desc=f"[{self.tag}] Epoch {self.epoch+1} [Val]")
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
            "epoch": self.epoch, "seed": self.seed,
            "focal_weight": self.focal_weight, "evidential_weight": self.evidential_weight,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice, "config": self.config,
        }
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
            print(f"[{self.tag}] Best model saved (Dice: {self.best_val_dice:.4f})")

    def train(self, epochs):
        print("\n" + "=" * 70)
        print(f"EXPERIMENT C0: Weight Ablation [{self.tag}]")
        print("=" * 70 + "\n")

        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()

            train_loss, train_dice = self.train_epoch()
            val = self.validate()
            self.scheduler.step()

            epoch_time = time.time() - t0
            print(
                f"[{self.tag}] Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
                f"Train: loss={train_loss:.4f} dice={train_dice:.4f} | "
                f"Val: loss={val['loss']:.4f} dice={val['dice']:.4f} "
                f"iou={val['iou']:.4f} f1={val['f1']:.4f} hd95={val['hd95']:.2f} ece={val['ece']:.4f}"
            )

            self.w_metrics.writerow([
                epoch, train_loss, train_dice,
                val["loss"], val["dice"], val["iou"], val["f1"], val["hd95"], val["ece"],
                epoch_time,
            ])
            self.f_metrics.flush()

            if val["dice"] > self.best_val_dice:
                self.best_val_dice = val["dice"]
                self.save_checkpoint(is_best=True)

        self.close_logs()
        print(f"\n[{self.tag}] Done. Best Val Dice: {self.best_val_dice:.4f}")
        return self.best_val_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment C0: Fixed-Weight Ablation")
    parser.add_argument("--config", default="../../configs/brats.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--focal_weight", type=float, required=True)
    parser.add_argument("--evidential_weight", type=float, required=True)
    args = parser.parse_args()

    exp_dir = Path(__file__).parent
    trainer = WeightAblationExperiment(
        args.config, exp_dir, seed=args.seed,
        focal_weight=args.focal_weight, evidential_weight=args.evidential_weight,
    )
    trainer.train(args.epochs)
