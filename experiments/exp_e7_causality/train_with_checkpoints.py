"""
Experiment E7: does latent geometry (boundary separability, density
structure) emerge BEFORE evidential-head calibration during training, or
only AFTER -- i.e. is geometry a plausible causal driver of uncertainty,
or merely a descriptive downstream consequence?

Identical training loop to experiments/exp00b_baseline_convergence/train.py
(same model, same HybridLoss, same optimizer/scheduler config, same seed
protocol) -- NOT a modification of the frozen baseline script itself
(never touch it again, per baseline_frozen_milestone), a separate script
that reuses the same frozen components. The only difference: saves a full
model checkpoint at a fixed set of epochs {1,3,5,10,20,35,50} regardless
of whether that epoch was the best val_dice epoch, so a later analysis
pass can measure geometry/calibration/Dice trajectories together across
training. The frozen baseline's `best.pth`-only checkpointing pattern is
insufficient for this since it only saves the single best epoch, never
early ones.

No EGGO changes here -- this is a plain baseline run, purely
observational, to test Hypothesis D (circularity) from PHASE_E6_STRESS_TEST.md.
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

sys.path.insert(0, str(project_root / "experiments" / "exp00b_baseline_convergence"))
from metrics import MetricAccumulator  # noqa: E402

CHECKPOINT_EPOCHS = {1, 3, 5, 10, 20, 35, 50}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CausalityExperiment:
    def __init__(self, config_path, exp_dir, seed):
        self.seed = seed
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
            self.optimizer, T_max=self.config["training"]["epochs"], eta_min=1e-6
        )

        self.train_loader, self.val_loader = create_brats_loaders(
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["training"].get("num_workers", 0),
            root_dir=self.config["dataset"]["root_dir"],
            val_split=self.config["dataset"]["val_split"]
        )

        print(f"[E7 seed {seed}] Training subjects: {len(self.train_loader.dataset)}")
        print(f"[E7 seed {seed}] Validation subjects: {len(self.val_loader.dataset)}")
        print(f"[E7 seed {seed}] Device: {self.device}")
        print(f"[E7 seed {seed}] Checkpoint epochs: {sorted(CHECKPOINT_EPOCHS)}")

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

    def save_fixed_checkpoint(self, epoch_1indexed):
        checkpoint = {
            "epoch": self.epoch,
            "seed": self.seed,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
        }
        path = self.checkpoint_dir / f"epoch_{epoch_1indexed}.pth"
        torch.save(checkpoint, path)
        print(f"[E7 seed {self.seed}] Saved fixed checkpoint at epoch {epoch_1indexed}: {path}")

    def train(self, epochs):
        history = {"train": [], "val": []}
        epoch_times = []

        print("\n" + "=" * 70)
        print(f"EXPERIMENT E7: Causality Test -- Checkpointed Training (seed={self.seed})")
        print("=" * 70 + "\n")

        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()

            train_metrics = self.run_epoch(self.train_loader, train=True, compute_hd95=False)
            val_metrics = self.run_epoch(self.val_loader, train=False, compute_hd95=True)

            self.scheduler.step()
            epoch_time = time.time() - t0
            epoch_times.append(epoch_time)

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

            epoch_1indexed = epoch + 1
            if epoch_1indexed in CHECKPOINT_EPOCHS:
                self.save_fixed_checkpoint(epoch_1indexed)

        result = {
            "seed": self.seed,
            "best_val_dice": self.best_val_dice,
            "final_epoch": self.epoch + 1,
            "avg_epoch_time_sec": float(np.mean(epoch_times)),
            "checkpoint_epochs": sorted(CHECKPOINT_EPOCHS),
            "history": history,
        }

        with open(self.exp_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n[E7 seed {self.seed}] Done. Best Val Dice: {self.best_val_dice:.4f}")
        return self.best_val_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment E7: Causality test via checkpointed training")
    parser.add_argument("--config", default="../../configs/brats.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    exp_dir = Path(__file__).parent
    trainer = CausalityExperiment(args.config, exp_dir, seed=args.seed)
    trainer.train(args.epochs)
