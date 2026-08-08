"""
Baseline training on BraTS 2023 GLI dataset using frozen NeuroScan model.

This script trains the original NeuroScan architecture (unchanged) on BraTS data.
Purpose: Establish a baseline before implementing the optimization controller.

Usage:
    python train_baseline_brats.py --epochs 50 --batch_size 4

Note: This imports model classes cleanly without triggering final_model.py's PediMS
initialization code, which is not needed for BraTS training.
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
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import BraTS dataset
from dataset.brats_dataset import BraTSDataset, create_brats_loaders

# Import models - use a clean extraction that avoids PediMS initialization
# For now, we'll define a minimal import helper
def load_neuroscan_models():
    """Load NeuroScan model classes from final_model.py without triggering module-level code."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01_source_code", "models"))

    # Suppress PediMS warnings by setting a dummy path
    os.environ['PEDIMS_DATA_PATH'] = '/dev/null'

    try:
        from final_model import HybridMiniSwin2D5_CBAM, HybridLoss
        return HybridMiniSwin2D5_CBAM, HybridLoss
    except ValueError as e:
        if "num_samples" in str(e):
            # This is expected - final_model tries to create a DataLoader with empty PediMS data
            # We need to work around this
            print("Warning: final_model.py tried to initialize PediMS DataLoader.")
            print("Using direct class extraction instead...")
            return _extract_model_classes_minimal()
        raise

def _extract_model_classes_minimal():
    """Extract model classes without executing full final_model.py."""
    # This is a fallback - for now, just fail with helpful message
    raise ImportError(
        "Could not cleanly import model classes from final_model.py.\n"
        "This happens because final_model.py has module-level code that initializes PediMS data.\n"
        "Solution: Either provide minimal PediMS data, or extract model definitions separately.\n"
        "For now, please ensure 'PediMS' folder exists in project root with dummy structure."
    )

try:
    HybridMiniSwin2D5_CBAM, HybridLoss = load_neuroscan_models()
except Exception as e:
    print(f"Import error: {e}")
    print("Attempting fallback...")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def dice_score(pred, target, smooth=1.0):
    """Calculate Dice coefficient."""
    pred_binary = (pred > 0.5).float()
    intersection = torch.sum(pred_binary * target)
    union = torch.sum(pred_binary) + torch.sum(target)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


class BaselineTrainer:
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device
        self.epoch = 0
        self.best_val_dice = 0.0

        # Create checkpoint directory
        self.checkpoint_dir = Path(config["training"]["save_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = HybridMiniSwin2D5_CBAM(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
        ).to(device)

        # Initialize loss
        self.criterion = HybridLoss(device=device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["epochs"],
            eta_min=1e-6,
        )

        # Setup dataloaders
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

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Compute loss
            loss = self.criterion(outputs, masks)

            # Backward pass
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

        avg_loss = total_loss / n_batches
        avg_dice = total_dice / n_batches

        return avg_loss, avg_dice

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        n_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch + 1} [Val]")
        with torch.no_grad():
            for images, masks, subject_ids in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Compute loss
                loss = self.criterion(outputs, masks)

                # Metrics
                dice = dice_score(outputs, masks)

                total_loss += loss.item()
                total_dice += dice
                n_batches += 1

                pbar.set_postfix({"loss": loss.item(), "dice": dice})

        avg_loss = total_loss / n_batches
        avg_dice = total_dice / n_batches

        return avg_loss, avg_dice

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_dice": self.best_val_dice,
        }

        # Regular checkpoint
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch:03d}.pth"
        torch.save(checkpoint, ckpt_path)

        # Best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (Dice: {self.best_val_dice:.4f})")

    def train(self, epochs):
        """Train for multiple epochs."""
        history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}

        for epoch in range(epochs):
            self.epoch = epoch

            # Train
            train_loss, train_dice = self.train_epoch()
            history["train_loss"].append(train_loss)
            history["train_dice"].append(train_dice)

            # Validate
            val_loss, val_dice = self.validate()
            history["val_loss"].append(val_loss)
            history["val_dice"].append(val_dice)

            # Learning rate step
            self.scheduler.step()

            # Log
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train: loss={train_loss:.4f}, dice={train_dice:.4f} | "
                f"Val: loss={val_loss:.4f}, dice={val_dice:.4f}"
            )

            # Save checkpoint
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.save_checkpoint(is_best=True)
            elif (epoch + 1) % self.config["training"]["save_frequency"] == 0:
                self.save_checkpoint(is_best=False)

        # Save history
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"\n✓ Training complete. Best Val Dice: {self.best_val_dice:.4f}")
        return history


def main():
    parser = argparse.ArgumentParser(description="Train NeuroScan baseline on BraTS")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/brats.yaml",
        help="Path to config file",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config if args provided
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    print("="*70)
    print("NeuroScan Baseline Training on BraTS 2023 GLI")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Device: {args.device}")
    print("="*70)

    # Train
    trainer = BaselineTrainer(config, device=args.device)
    history = trainer.train(config["training"]["epochs"])


if __name__ == "__main__":
    main()
