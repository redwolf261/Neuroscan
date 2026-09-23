"""
Phase E52: Adaptive Size-Reweighting (ASR), [R]/alpha_c condition,
gradient-magnitude-calibrated re-attempt.

CONTEXT: E34's original [R] condition collapsed training (best Dice
0.7108 vs 0.9063 baseline), using a lambda_cw calibrated by VALUE match
only. calibrate_lambda_cw_v2.py measured this directly: E34's original
lambda_cw (4.75-5.2) would produce a 43x gradient blowup relative to the
main segmentation loss -- almost certainly the actual cause of the
collapse, not a flaw in the component-weighted-Dice mechanism itself.
This script reuses E34's own architecture (v3, unmodified -- this is a
LOSS-level change, not an architecture change) and E34's own
component_weighted_loss.py (already tested, unmodified) with ONLY the
lambda recalibrated by gradient magnitude (0.1101, target ratio 1.0,
matching E45's lambda_d8 comparable-weight treatment).

PRE-DECLARED SUCCESS CRITERION: mean >=1.0pp over canonical baseline
(0.9063) across 3 seeds, 95% CI excluding D4-only (0.9096).

PRE-DECLARED SMOKE-TEST GATE: a 2-epoch run must show val Dice reaching
at least the 0.5-0.6 range other healthy runs reach by epoch 2 (NOT
E34's own [R] pattern: 0.057-0.069 at epoch 1-2). If the smoke test
reproduces E34's collapse pattern even under gradient calibration, this
phase stops immediately -- no full run chasing an already-failed result.

PRE-DECLARED KILL CONDITION: dice < 0.5 after epoch 5 -> immediate stop,
matching every prior condition's own established practice.

NOTE: this uses UNet3D_v3 (NOT v8/v6/v5) -- deliberately isolated from
every post-pivot ARCHITECTURE mechanism (CCABA, attention gate, IECG),
since this phase specifically tests an OBJECTIVE-level lever, kept
separate to cleanly attribute any effect to the loss change alone.
"""
import sys
import csv
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndi
from scipy.ndimage import zoom
import nibabel as nib
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e34"))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402
from component_weighted_loss import WeightLookup, compute_component_weighted_loss, precompute_labeled_64_cache  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp00b_baseline_convergence"))
from metrics import MetricAccumulator  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_c0_weight_ablation"))
from calibration import ECEAccumulator  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import set_seed  # noqa: E402

SEED = 0
MU = 0.1
LAMBDA_CW = 0.1101  # gradient-matched, see calibrate_lambda_cw_v2.py / E52_lambda_cw_calibration.json
CONFIG_PATH = project_root / "configs" / "brats.yaml"
E34_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e34"
OUT_DIR = Path(__file__).parent


def resize_nn(volume, target_shape):
    current_shape = volume.shape
    zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
    return zoom(volume, zoom_factors, order=0)


def load_weight_table(path):
    with open(path) as f:
        raw = json.load(f)
    parsed = {}
    for key, w in raw.items():
        subj, comp_id = key.rsplit("|", 1)
        parsed[(subj, int(comp_id))] = w
    return WeightLookup(parsed)


class E52Experiment:
    def __init__(self, config_path, exp_dir, seed, mu, run_name, num_workers=None):
        self.condition_name = run_name
        self.seed = seed
        self.mu = mu
        set_seed(seed)

        self.exp_dir = Path(exp_dir) / run_name
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

        self.model = UNet3D_v3(
            in_channels=self.config["model"]["in_channels"],
            out_channels=self.config["model"]["out_channels"],
        ).to(self.device)

        self.focal_fn = FocalTverskyLoss()
        self.evidential_fn = EvidentialBetaLoss(weight=0.5)
        self.focal_weight = 0.5
        self.evidential_weight = 0.5
        self.boundary_criterion = nn.BCEWithLogitsLoss()

        self.weight_lookup = load_weight_table(E34_DIR / "E34_weight_table_alpha_c.json")
        self.native_cache = {}

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.config["training"]["epochs"], eta_min=1e-6
        )

        self.train_loader, self.val_loader = create_brats_loaders(
            batch_size=self.config["training"]["batch_size"],
            num_workers=(num_workers if num_workers is not None else self.config["training"].get("num_workers", 0)),
            root_dir=self.config["dataset"]["root_dir"],
            val_split=self.config["dataset"]["val_split"],
        )
        self.subject_dir_by_id = {Path(d).name: d for d in self.train_loader.dataset.subject_dirs}

        print(f"[{self.condition_name} seed{seed}] lambda_cw={LAMBDA_CW} (gradient-matched) mu={mu} (ASR [R]/alpha_c, v3 base)")
        print(f"[{self.condition_name}] Training subjects: {len(self.train_loader.dataset)}")
        print(f"[{self.condition_name}] Validation subjects: {len(self.val_loader.dataset)}")
        print(f"[{self.condition_name}] Device: {self.device}")

        # PERFORMANCE: precompute labeled_64_cache for ALL training subjects
        # ONCE, upfront -- matching E34's own established fix (its own
        # docstring documents a ~5s/batch, ~6-7x slowdown bug from
        # recomputing this per-batch, fixed by precomputing once). Doing
        # this per-batch (an earlier version of this script) reproduced
        # that exact bug: >600s without completing epoch 1.
        print(f"[{self.condition_name}] Precomputing labeled_64_cache for all {len(self.train_loader.dataset)} training subjects...")
        t_precompute = time.time()
        all_subject_ids = [Path(d).name for d in self.train_loader.dataset.subject_dirs]
        native_labeled_cache = {}
        for i, sid in enumerate(all_subject_ids):
            native_labeled_cache[sid] = self._get_native(sid)
            if (i + 1) % 200 == 0:
                print(f"  ...{i+1}/{len(all_subject_ids)}")
        self.labeled_64_cache = precompute_labeled_64_cache(native_labeled_cache, resize_nn)
        print(f"[{self.condition_name}] Precompute done in {time.time()-t_precompute:.1f}s")

        self._open_logs()

    def _open_logs(self):
        self.f_metrics = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w_metrics = csv.writer(self.f_metrics)
        self.w_metrics.writerow([
            "epoch", "train_loss", "train_dice", "train_seg_loss", "train_boundary_loss",
            "train_cw_loss", "val_loss", "val_dice",
            "val_iou", "val_precision", "val_recall", "val_f1", "val_hd95", "val_ece",
            "boundary_bce", "boundary_auc_proxy",
            "epoch_time_sec", "peak_gpu_memory_mb",
        ])

    def close_logs(self):
        self.f_metrics.close()

    def _get_native(self, subject_id):
        if subject_id not in self.native_cache:
            seg_path = Path(self.subject_dir_by_id[subject_id]) / f"{subject_id}-seg.nii.gz"
            seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
            seg_binary = (seg_data > 0).astype(np.float32)
            native_labeled, _ = ndi.label(seg_binary > 0.5)
            self.native_cache[subject_id] = (native_labeled, seg_data.shape)
        return self.native_cache[subject_id]

    def train_epoch(self):
        self.model.train()
        acc = MetricAccumulator()
        total_seg_loss = 0.0
        total_boundary_loss = 0.0
        total_cw_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"[{self.condition_name}] Epoch {self.epoch+1} [Train]")
        for batch_idx, (images, masks, subject_ids) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            probs = outputs["probs"]
            boundary_logit = outputs["boundary_logit"]

            focal_loss = self.focal_fn(probs, masks)
            evidential_loss = self.evidential_fn(outputs["alpha"], outputs["beta"], masks)
            seg_loss = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss

            if not torch.isfinite(seg_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf seg_loss at epoch {self.epoch} batch {batch_idx}")

            boundary_loss = self.boundary_criterion(boundary_logit, masks)
            if not torch.isfinite(boundary_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf boundary_loss at epoch {self.epoch} batch {batch_idx}")

            cw_loss, _ = compute_component_weighted_loss(
                probs, masks, subject_ids, self.weight_lookup, self.labeled_64_cache, self.device
            )
            if not torch.isfinite(cw_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf cw_loss at epoch {self.epoch} batch {batch_idx}")

            total_loss = seg_loss + self.mu * boundary_loss + LAMBDA_CW * cw_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            acc.update(seg_loss.item(), probs, masks, compute_hd95=False)
            total_seg_loss += seg_loss.item()
            total_boundary_loss += boundary_loss.item()
            total_cw_loss += cw_loss.item()
            n_batches += 1

            pbar.set_postfix({"seg": seg_loss.item(), "bnd": boundary_loss.item(), "cw": cw_loss.item()})

        summary = acc.summary()
        return {
            "loss": summary["loss"], "dice": summary["dice"],
            "seg_loss": total_seg_loss / max(1, n_batches),
            "boundary_loss": total_boundary_loss / max(1, n_batches),
            "cw_loss": total_cw_loss / max(1, n_batches),
        }

    def validate(self):
        """Verbatim copy of every prior condition's own validate() logic --
        no component-weighted loss at validation time (matching E34's own
        convention: the mechanism is a TRAINING-time supervision weighting,
        not part of the evaluated model's own forward pass or metric)."""
        self.model.eval()
        acc = MetricAccumulator()
        ece_acc = ECEAccumulator(n_bins=15)
        boundary_bce_total = 0.0
        boundary_correct = 0
        boundary_total = 0
        n_batches = 0

        pbar = tqdm(self.val_loader, desc=f"[{self.condition_name}] Epoch {self.epoch+1} [Val]")
        with torch.no_grad():
            for images, masks, _ in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)

                probs, alpha, beta = outputs["probs"], outputs["alpha"], outputs["beta"]
                boundary_logit = outputs["boundary_logit"]

                focal_loss = self.focal_fn(probs, masks)
                evidential_loss = self.evidential_fn(alpha, beta, masks)
                total = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss

                acc.update(total.item(), probs, masks, compute_hd95=True)
                ece_acc.update(alpha, beta, masks)

                b_bce = self.boundary_criterion(boundary_logit, masks)
                boundary_bce_total += b_bce.item()
                boundary_pred = (torch.sigmoid(boundary_logit) >= 0.5).float()
                boundary_correct += (boundary_pred == masks).sum().item()
                boundary_total += masks.numel()
                n_batches += 1

                pbar.set_postfix({"loss": total.item()})

        summary = acc.summary()
        ece, _ = ece_acc.compute()
        summary["ece"] = ece
        summary["boundary_bce"] = boundary_bce_total / max(1, n_batches)
        summary["boundary_accuracy_proxy"] = boundary_correct / max(1, boundary_total)
        return summary

    def save_checkpoint(self, is_best=False, is_periodic=True):
        checkpoint = {
            "epoch": self.epoch, "seed": self.seed, "mu": self.mu,
            "lambda_cw": LAMBDA_CW,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice, "config": self.config,
            "condition": self.condition_name,
        }
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
        if is_periodic:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{self.epoch+1}.pth")

    def train(self, epochs, checkpoint_every=5):
        print("\n" + "=" * 70)
        print(f"{self.condition_name}: lambda_cw={LAMBDA_CW} [seed={self.seed}, mu={self.mu}] ASR [R]/alpha_c, gradient-matched")
        print("=" * 70 + "\n")

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()

            train_metrics = self.train_epoch()
            val = self.validate()
            self.scheduler.step()
            epoch_time = time.time() - t0

            if epoch >= 5 and val["dice"] < 0.5:
                print(f"\n[{self.condition_name}] INSTABILITY DETECTED: val_dice={val['dice']:.4f} "
                      f"< 0.5 after epoch 5. Stopping immediately per pre-declared kill condition.", flush=True)
                self.close_logs()
                raise RuntimeError(f"Training instability at epoch {epoch}: val_dice={val['dice']:.4f}")

            peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6) if self.device.type == "cuda" else 0.0
            print(
                f"[{self.condition_name}] Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s, peak_mem={peak_mem_mb:.0f}MB) | "
                f"Train: loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} "
                f"seg={train_metrics['seg_loss']:.4f} bnd={train_metrics['boundary_loss']:.4f} "
                f"cw={train_metrics['cw_loss']:.4f} | "
                f"Val: dice={val['dice']:.4f} precision={val['precision']:.4f} recall={val['recall']:.4f} "
                f"hd95={val['hd95']:.2f} ece={val['ece']:.4f}"
            )

            self.w_metrics.writerow([
                epoch, train_metrics["loss"], train_metrics["dice"], train_metrics["seg_loss"],
                train_metrics["boundary_loss"], train_metrics["cw_loss"],
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

        self.close_logs()
        print(f"\n[{self.condition_name}] Done. Best Val Dice: {self.best_val_dice:.4f}")
        return self.best_val_dice


def main(epochs, run_name, seed):
    exp = E52Experiment(
        config_path=str(CONFIG_PATH), exp_dir=str(OUT_DIR / "runs"), seed=seed, mu=MU,
        run_name=run_name, num_workers=None,
    )
    best_dice = exp.train(epochs=epochs, checkpoint_every=5)
    print(f"\n=== E52 ASR-gradient-calibrated run complete (seed={seed}): best_val_dice={best_dice:.4f} ===")
    return best_dice


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--run_name", type=str, default="ASRv2_seed0")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(args.epochs, args.run_name, args.seed)
