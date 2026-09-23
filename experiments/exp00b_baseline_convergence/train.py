"""
Experiment 00b: Baseline Convergence Study (Phase A.5)

Research Question:
  What is the true converged performance of the 3D U-Net baseline on BraTS,
  and how stable is that performance across random seeds?

Method:
  - Train UNet3D to convergence (up to --epochs, early stopping on val Dice)
  - Track full metric suite: Dice, IoU, Precision, Recall, F1, HD95
  - Track training time per epoch and peak GPU memory
  - Repeat across --seeds (default: 3) for variance estimate

This is the frozen reference point Phase B/C/D/E measure against.
No optimizer or loss changes happen here — this run exists to answer
"how good is plain AdamW + HybridLoss, and how much does it vary by seed?"
"""

import os
import sys
import json
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

from neuroscan_3d_fixed import UNet3D, HybridLoss
from Dataset.brats_dataset import create_brats_loaders
from metrics import MetricAccumulator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ConvergenceExperiment:
    def __init__(self, config_path, exp_dir, seed, patience):
        self.seed = seed
        self.patience = patience
        set_seed(seed)

        self.exp_dir = Path(exp_dir) / f"seed_{seed}"
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        dataset_root = self.config["dataset"]["root_dir"]
        if not Path(dataset_root).is_absolute():
            dataset_root = project_root / dataset_root
        self.config["dataset"]["root_dir"] = str(dataset_root)

        self.device = torch.device(self.config.get("hardware", {}).get("device", "cpu"))
        self.best_val_dice = 0.0
        self.epochs_since_best = 0
        self.epoch = 0

        self.model = UNet3D(
            in_channels=self.config["model"]["in_channels"],
            out_channels=self.config["model"]["out_channels"]
        ).to(self.device)

        self.criterion = HybridLoss(device=self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["training"]["epochs"],
            eta_min=1e-6
        )

        self.train_loader, self.val_loader = create_brats_loaders(
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["training"].get("num_workers", 0),
            root_dir=self.config["dataset"]["root_dir"],
            val_split=self.config["dataset"]["val_split"]
        )

        print(f"[seed {seed}] Training subjects: {len(self.train_loader.dataset)}")
        print(f"[seed {seed}] Validation subjects: {len(self.val_loader.dataset)}")
        print(f"[seed {seed}] Device: {self.device}")

    def run_epoch(self, loader, train, compute_hd95):
        self.model.train(mode=train)
        acc = MetricAccumulator()

        desc = f"Epoch {self.epoch+1} [{'Train' if train else 'Val'}]"
        pbar = tqdm(loader, desc=desc)

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for images, masks, _ in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                acc.update(loss.item(), outputs["probs"], masks, compute_hd95=compute_hd95)
                pbar.set_postfix({"loss": loss.item()})

        return acc.summary()

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            "epoch": self.epoch,
            "seed": self.seed,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice,
            "config": self.config
        }
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
            print(f"[seed {self.seed}] Best model saved (Dice: {self.best_val_dice:.4f})")

    def train(self, epochs):
        history = {"train": [], "val": []}
        epoch_times = []
        peak_mem_mb = 0.0

        print("\n" + "=" * 70)
        print(f"EXPERIMENT 00b: Baseline Convergence Study (seed={self.seed})")
        print("=" * 70 + "\n")

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()

            train_metrics = self.run_epoch(self.train_loader, train=True, compute_hd95=False)
            # HD95 is expensive; only compute it on validation, every epoch is still fine at this dataset size
            val_metrics = self.run_epoch(self.val_loader, train=False, compute_hd95=True)

            self.scheduler.step()
            epoch_time = time.time() - t0
            epoch_times.append(epoch_time)

            if self.device.type == "cuda":
                peak_mem_mb = max(peak_mem_mb, torch.cuda.max_memory_allocated() / 1e6)

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            print(
                f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
                f"Train: loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} | "
                f"Val: loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f} "
                f"iou={val_metrics['iou']:.4f} f1={val_metrics['f1']:.4f} hd95={val_metrics['hd95']:.2f}"
            )

            if val_metrics["dice"] > self.best_val_dice:
                self.best_val_dice = val_metrics["dice"]
                self.epochs_since_best = 0
                self.save_checkpoint(is_best=True)
            else:
                self.epochs_since_best += 1

            if self.epochs_since_best >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {self.patience} epochs)")
                break

        result = {
            "seed": self.seed,
            "best_val_dice": self.best_val_dice,
            "final_epoch": self.epoch + 1,
            "avg_epoch_time_sec": float(np.mean(epoch_times)),
            "peak_gpu_memory_mb": peak_mem_mb,
            "history": history,
        }

        with open(self.exp_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n[seed {self.seed}] Done. Best Val Dice: {self.best_val_dice:.4f}")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 00b: Baseline Convergence Study")
    parser.add_argument("--config", default="../../configs/brats.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    exp_dir = Path(__file__).parent
    all_results = []

    for seed in args.seeds:
        trainer = ConvergenceExperiment(args.config, exp_dir, seed=seed, patience=args.patience)
        result = trainer.train(args.epochs)
        all_results.append(result)

    dice_values = [r["best_val_dice"] for r in all_results]
    summary = {
        "seeds": args.seeds,
        "best_val_dice_per_seed": dice_values,
        "mean_dice": float(np.mean(dice_values)),
        "std_dice": float(np.std(dice_values)),
    }

    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE A.5 SUMMARY")
    print("=" * 70)
    print(f"Seeds: {args.seeds}")
    print(f"Dice per seed: {dice_values}")
    print(f"Mean ± Std: {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
