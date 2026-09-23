"""
Experiment 00: NeuroScan Baseline on BraTS

Research Question:
  Can the frozen NeuroScan architecture (trained on PediMS MS lesions)
  achieve reasonable performance on BraTS tumor segmentation with NO modifications?

Hypothesis:
  Yes, the 2.5D architecture should transfer reasonably well.
  We expect validation Dice ~40-60% (lower than PediMS due to domain/task shift).

Method:
  - Train HybridMiniSwin2D5_CBAM on BraTS 2023 GLI
  - Use FLAIR modality only (single channel)
  - Binary segmentation (tumor vs background)
  - Original loss function (HybridLoss with evidential uncertainty)
  - Original optimizer (AdamW with cosine annealing)
  - NO architectural modifications

Metrics to track:
  - Training loss, validation loss
  - Training Dice, validation Dice
  - Gradient norms (for Phase B diagnostics)
  - Per-component losses (focal vs evidential)
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_fixed import UNet3D, HybridLoss
from Dataset.brats_dataset import create_brats_loaders


def dice_score(pred, target, smooth=1e-6):
    """Dice coefficient."""
    pred_binary = (pred > 0.5).float()
    intersection = torch.sum(pred_binary * target)
    union = torch.sum(pred_binary) + torch.sum(target)
    return (2.0 * intersection + smooth) / (union + smooth + 1e-8)


class BaselineExperiment:
    def __init__(self, config_path, exp_dir):
        self.exp_dir = Path(exp_dir)
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.log_dir = self.exp_dir / "logs"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Make dataset path absolute
        dataset_root = self.config["dataset"]["root_dir"]
        if not Path(dataset_root).is_absolute():
            dataset_root = project_root / dataset_root
        self.config["dataset"]["root_dir"] = str(dataset_root)

        self.device = torch.device(self.config.get("hardware", {}).get("device", "cpu"))
        self.best_val_dice = 0.0
        self.epoch = 0

        # Initialize model
        self.model = UNet3D(
            in_channels=self.config["model"]["in_channels"],
            out_channels=self.config["model"]["out_channels"]
        ).to(self.device)

        # Initialize loss
        self.criterion = HybridLoss(device=self.device)

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )

        # Initialize scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["training"]["epochs"],
            eta_min=1e-6
        )

        # Load data
        self.train_loader, self.val_loader = create_brats_loaders(
            batch_size=self.config["training"]["batch_size"],
            root_dir=self.config["dataset"]["root_dir"],
            val_split=self.config["dataset"]["val_split"]
        )

        print(f"Model: HybridMiniSwin2D5_CBAM (frozen)")
        print(f"Training subjects: {len(self.train_loader.dataset)}")
        print(f"Validation subjects: {len(self.val_loader.dataset)}")
        print(f"Device: {self.device}")

    def train_epoch(self):
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]")
        for images, masks, _ in pbar:
            images = images.to(self.device)  # (B, 1, D, H, W)
            masks = masks.to(self.device)    # (B, 1, D, H, W)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Model outputs 3D predictions matching input shape
            loss = self.criterion(outputs, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            with torch.no_grad():
                dice = dice_score(outputs["probs"], masks)

            total_loss += loss.item()
            total_dice += dice.item()
            n_batches += 1

            pbar.set_postfix({"loss": loss.item(), "dice": dice.item()})

        return total_loss / n_batches, total_dice / n_batches

    def validate(self):
        """Validate."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        n_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch+1} [Val]")
        with torch.no_grad():
            for images, masks, _ in pbar:
                images = images.to(self.device)  # (B, 1, D, H, W)
                masks = masks.to(self.device)    # (B, 1, D, H, W)

                outputs = self.model(images)

                # Model outputs 3D predictions
                loss = self.criterion(outputs, masks)
                dice = dice_score(outputs["probs"], masks)

                total_loss += loss.item()
                total_dice += dice.item()
                n_batches += 1

                pbar.set_postfix({"loss": loss.item(), "dice": dice.item()})

        return total_loss / n_batches, total_dice / n_batches

    def save_checkpoint(self, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice,
            "config": self.config
        }

        ckpt_path = self.checkpoint_dir / f"epoch_{self.epoch:03d}.pth"
        torch.save(checkpoint, ckpt_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved (Dice: {self.best_val_dice:.4f})")

    def train(self, epochs):
        """Train for multiple epochs."""
        history = {
            "train_loss": [],
            "train_dice": [],
            "val_loss": [],
            "val_dice": []
        }

        print("\n" + "="*70)
        print("EXPERIMENT 00: NeuroScan Baseline on BraTS")
        print("="*70 + "\n")

        for epoch in range(epochs):
            self.epoch = epoch

            train_loss, train_dice = self.train_epoch()
            history["train_loss"].append(train_loss)
            history["train_dice"].append(train_dice)

            val_loss, val_dice = self.validate()
            history["val_loss"].append(val_loss)
            history["val_dice"].append(val_dice)

            self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train: loss={train_loss:.4f}, dice={train_dice:.4f} | "
                f"Val: loss={val_loss:.4f}, dice={val_dice:.4f}"
            )

            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.save_checkpoint(is_best=True)
            elif (epoch + 1) % max(1, epochs // 10) == 0:
                self.save_checkpoint(is_best=False)

        # Save history
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Save config
        config_path = self.exp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f)

        print(f"\n✓ Training complete")
        print(f"  Best Val Dice: {self.best_val_dice:.4f}")
        print(f"  Results saved to: {self.checkpoint_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 00: NeuroScan Baseline")
    parser.add_argument("--config", default="../../configs/brats.yaml", help="Config file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    exp_dir = Path(__file__).parent
    trainer = BaselineExperiment(args.config, exp_dir)
    trainer.train(args.epochs)
