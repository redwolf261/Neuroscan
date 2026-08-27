"""
Baseline training on BraTS 2023 GLI dataset.

Simple version: Uses standard PyTorch UNet to verify BraTS data pipeline works.
Once this runs successfully, we'll integrate the frozen NeuroScan architecture.

Usage:
    python train_baseline_brats_simple.py --epochs 2 --batch_size 4
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import dataset module directly
import importlib.util
dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "brats_dataset.py")
spec = importlib.util.spec_from_file_location("brats_dataset", dataset_path)
brats_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brats_dataset_module)

BraTSDataset = brats_dataset_module.BraTSDataset
create_brats_loaders = brats_dataset_module.create_brats_loaders


class SimpleUNet3D(nn.Module):
    """Minimal 3D UNet for testing."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.upconv2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upconv1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv3d(32, out_channels, 1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)

        enc2 = self.enc2(x)
        x = self.pool2(enc2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)

        x = self.final(x)
        return torch.sigmoid(x)


def dice_score(pred, target, smooth=1.0):
    """Calculate Dice coefficient."""
    pred_binary = (pred > 0.5).float()
    intersection = torch.sum(pred_binary * target)
    union = torch.sum(pred_binary) + torch.sum(target)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


class SimpleBaselineTrainer:
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device
        self.epoch = 0
        self.best_val_dice = 0.0

        # Create checkpoint directory
        self.checkpoint_dir = Path(config["training"]["save_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = SimpleUNet3D(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
        ).to(device)

        # Loss
        self.criterion = nn.BCELoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["epochs"],
            eta_min=1e-6,
        )

        # Dataloaders
        self.train_loader, self.val_loader = create_brats_loaders(
            batch_size=config["training"]["batch_size"],
            root_dir=config["dataset"]["root_dir"],
            val_split=config["dataset"]["val_split"],
        )

        print(f"Training subjects: {len(self.train_loader.dataset)}")
        print(f"Validation subjects: {len(self.val_loader.dataset)}")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1} [Train]")
        for images, masks, subject_ids in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Loss
            loss = self.criterion(outputs, masks)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Metrics
            with torch.no_grad():
                dice = dice_score(outputs, masks)

            total_loss += loss.item()
            total_dice += dice
            n_batches += 1

            pbar.set_postfix({"loss": loss.item(), "dice": dice})

        return total_loss / n_batches, total_dice / n_batches

    def validate(self):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        n_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch + 1} [Val]")
        with torch.no_grad():
            for images, masks, subject_ids in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward
                outputs = self.model(images)

                # Loss
                loss = self.criterion(outputs, masks)

                # Metrics
                dice = dice_score(outputs, masks)

                total_loss += loss.item()
                total_dice += dice
                n_batches += 1

                pbar.set_postfix({"loss": loss.item(), "dice": dice})

        return total_loss / n_batches, total_dice / n_batches

    def save_checkpoint(self, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_dice": self.best_val_dice,
        }

        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch:03d}.pth"
        torch.save(checkpoint, ckpt_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (Dice: {self.best_val_dice:.4f})")

    def train(self, epochs):
        """Train for multiple epochs."""
        history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}

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
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train: loss={train_loss:.4f}, dice={train_dice:.4f} | "
                f"Val: loss={val_loss:.4f}, dice={val_dice:.4f}"
            )

            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.save_checkpoint(is_best=True)
            elif (epoch + 1) % self.config["training"]["save_frequency"] == 0:
                self.save_checkpoint(is_best=False)

        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"\n✓ Training complete. Best Val Dice: {self.best_val_dice:.4f}")
        return history


def main():
    parser = argparse.ArgumentParser(description="Simple UNet baseline on BraTS")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/brats.yaml",
        help="Path to config file",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    print("=" * 70)
    print("Simple UNet Baseline Training on BraTS 2023 GLI")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Device: {args.device}")
    print("=" * 70)

    trainer = SimpleBaselineTrainer(config, device=args.device)
    history = trainer.train(config["training"]["epochs"])


if __name__ == "__main__":
    main()
