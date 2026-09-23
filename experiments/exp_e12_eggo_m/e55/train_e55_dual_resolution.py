"""
Phase E55: Dual-Resolution Local Refinement training script. UNet3D_v9
-- v3's global pathway (unchanged) + a differentiable native-resolution
local refinement pathway (soft centroid -> grid_sample crop -> local
CNN -> differentiable scatter -> gated fusion).

CONTEXT: E54 showed whole-volume 96^3 training is flat (0.9066, +0.03pp)
vs baseline. E55 targets resolution recovery specifically at small/
uncertain lesions (where E48's causal evidence says it matters) rather
than paying whole-volume cost uniformly. All verification checks
(coordinate mapping, real-subject label-crop alignment, ablation-safety,
gradient-flow, memory profiling) passed before this script was written --
see test_v9_coordinate_mapping.py, test_v9_label_crop_alignment.py,
test_v9_ablation_safety.py, profile_memory_v9.py.

MEMORY: batch=4 measured at 4.85GB (profile_memory_v9.py) -- fits
comfortably. Using gradient accumulation (physical batch=4, accumulate
over 2 steps) for an effective batch of 8, matching every prior
condition's own optimizer update dynamics (same pattern as E54's own
train_e54_a96_resolution.py).

CALIBRATION: lambda_local=0.2340, gradient-matched (target ratio 1.0),
see calibrate_lambda_local.py / E55_lambda_local_calibration.json.
Value-matched calibration (1.1823) would have caused a 5.05x gradient
blowup -- rejected per E34/E52's own safeguard.

PRE-DECLARED SUCCESS CRITERION: >=1.0pp Dice over canonical baseline
(0.9063) -> target >=0.9163, full 125-subject validation set, 3-seed
mean/variance if seed 0 shows a real signal.

PRE-DECLARED KILL CONDITION: val dice < 0.5 after epoch 5 -> immediate
stop, matching every prior condition's own established practice.
"""
import sys
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v9 import UNet3D_v9, build_sampling_grid  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp00b_baseline_convergence"))
from metrics import MetricAccumulator  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_c0_weight_ablation"))
from calibration import ECEAccumulator  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import set_seed  # noqa: E402

SEED = 0
MU = 0.1
LAMBDA_DS3 = 0.9927  # REUSED unchanged from every prior D4 condition
LAMBDA_DS2 = 1.0014  # REUSED unchanged from every prior D2 condition
LAMBDA_LOCAL = 0.2340  # gradient-matched, see calibrate_lambda_local.py / E55_lambda_local_calibration.json
PHYSICAL_BATCH_SIZE = 4  # measured to fit 8.15GB budget, see profile_memory_v9.py
ACCUM_STEPS = 2  # effective batch = 4*2 = 8, matching every prior condition's optimizer dynamics
CONFIG_PATH = project_root / "configs" / "brats.yaml"
OUT_DIR = Path(__file__).parent


def crop_label(mask_native, centroid_native, crop_size, native_shape):
    """Extract the label crop via the IDENTICAL mapping used for the
    image crop -- mode='nearest' for a binary mask."""
    grid = build_sampling_grid(centroid_native, crop_size, native_shape, mask_native.device, mask_native.dtype)
    return F.grid_sample(mask_native, grid, mode="nearest", padding_mode="zeros", align_corners=True)


class E55Experiment:
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

        self.model = UNet3D_v9(
            in_channels=self.config["model"]["in_channels"],
            out_channels=self.config["model"]["out_channels"],
        ).to(self.device)

        self.focal_fn = FocalTverskyLoss()
        self.evidential_fn = EvidentialBetaLoss(weight=0.5)
        self.focal_weight = 0.5
        self.evidential_weight = 0.5
        self.boundary_criterion = nn.BCEWithLogitsLoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.config["training"]["epochs"], eta_min=1e-6
        )

        self.train_loader, self.val_loader = create_brats_loaders(
            batch_size=PHYSICAL_BATCH_SIZE,
            num_workers=(num_workers if num_workers is not None else self.config["training"].get("num_workers", 0)),
            root_dir=self.config["dataset"]["root_dir"],
            val_split=self.config["dataset"]["val_split"],
            target_shape=(64, 64, 64),
            return_native=True,
        )
        self.subject_dir_by_id = {Path(d).name: d for d in self.train_loader.dataset.subject_dirs}
        self.native_seg_cache = {}  # subject_id -> native binary seg ndarray, cached across epochs

        print(f"[{self.condition_name} seed{seed}] lambda_local={LAMBDA_LOCAL} physical_batch={PHYSICAL_BATCH_SIZE} "
              f"accum_steps={ACCUM_STEPS} (effective_batch={PHYSICAL_BATCH_SIZE*ACCUM_STEPS}) mu={mu}")
        print(f"[{self.condition_name}] Training subjects: {len(self.train_loader.dataset)}")
        print(f"[{self.condition_name}] Validation subjects: {len(self.val_loader.dataset)}")
        print(f"[{self.condition_name}] Device: {self.device}")

        self._open_logs()

    def _get_native_seg(self, subject_id, subject_dir_by_id):
        if subject_id not in self.native_seg_cache:
            seg_path = Path(subject_dir_by_id[subject_id]) / f"{subject_id}-seg.nii.gz"
            seg_native = (nib.load(str(seg_path)).get_fdata() > 0).astype(np.float32)
            self.native_seg_cache[subject_id] = seg_native
        return self.native_seg_cache[subject_id]

    def _native_masks_for_batch(self, subject_ids, subject_dir_by_id, device):
        masks = [torch.from_numpy(self._get_native_seg(sid, subject_dir_by_id)) for sid in subject_ids]
        return torch.stack(masks).unsqueeze(1).to(device)  # (B,1,240,240,155)

    def _open_logs(self):
        self.f_metrics = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w_metrics = csv.writer(self.f_metrics)
        self.w_metrics.writerow([
            "epoch", "train_loss", "train_dice", "train_seg_loss", "train_boundary_loss",
            "train_aux3_loss", "train_aux2_loss", "train_local_loss", "val_loss", "val_dice",
            "val_iou", "val_precision", "val_recall", "val_f1", "val_hd95", "val_ece",
            "boundary_bce", "boundary_auc_proxy", "fusion_gate",
            "epoch_time_sec", "peak_gpu_memory_mb",
        ])

    def close_logs(self):
        self.f_metrics.close()

    def train_epoch(self):
        self.model.train()
        acc = MetricAccumulator()
        total_seg_loss = 0.0
        total_boundary_loss = 0.0
        total_aux3_loss = 0.0
        total_aux2_loss = 0.0
        total_local_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc=f"[{self.condition_name}] Epoch {self.epoch+1} [Train]")
        for batch_idx, (images, masks, subject_ids, native_images) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            native_images = native_images.to(self.device)
            native_masks = self._native_masks_for_batch(subject_ids, self.subject_dir_by_id, self.device)

            outputs = self.model(images, native_x=native_images)
            probs = outputs["probs"]
            alpha, beta = outputs["alpha"], outputs["beta"]
            boundary_logit = outputs["boundary_logit"]
            aux_probs3, aux_probs2 = outputs["aux_probs3"], outputs["aux_probs2"]
            centroid_native = outputs["centroid_native"]
            logit_local_crop = outputs["logit_local_crop"]

            focal_loss = self.focal_fn(probs, masks)
            evidential_loss = self.evidential_fn(alpha, beta, masks)
            seg_loss = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss

            if not torch.isfinite(seg_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf seg_loss at epoch {self.epoch} batch {batch_idx}")

            boundary_loss = self.boundary_criterion(boundary_logit, masks)
            if not torch.isfinite(boundary_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf boundary_loss at epoch {self.epoch} batch {batch_idx}")

            mask_d4 = F.avg_pool3d(masks, kernel_size=4, stride=4)
            mask_d2 = F.avg_pool3d(masks, kernel_size=2, stride=2)
            aux3_loss = self.focal_fn(aux_probs3, mask_d4)
            aux2_loss = self.focal_fn(aux_probs2, mask_d2)
            if not torch.isfinite(aux3_loss) or not torch.isfinite(aux2_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf aux loss at epoch {self.epoch} batch {batch_idx}")

            label_crop = crop_label(native_masks, centroid_native, self.model.CROP_SIZE, self.model.NATIVE_SHAPE)
            probs_local = torch.sigmoid(logit_local_crop)
            local_loss = self.focal_fn(probs_local, label_crop)
            if not torch.isfinite(local_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf local_loss at epoch {self.epoch} batch {batch_idx}")

            total_loss = (seg_loss + self.mu * boundary_loss + LAMBDA_DS3 * aux3_loss
                          + LAMBDA_DS2 * aux2_loss + LAMBDA_LOCAL * local_loss)
            (total_loss / ACCUM_STEPS).backward()

            is_accum_step = (batch_idx + 1) % ACCUM_STEPS == 0
            is_last_batch = (batch_idx + 1) == len(self.train_loader)
            if is_accum_step or is_last_batch:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            acc.update(seg_loss.item(), probs, masks, compute_hd95=False)
            total_seg_loss += seg_loss.item()
            total_boundary_loss += boundary_loss.item()
            total_aux3_loss += aux3_loss.item()
            total_aux2_loss += aux2_loss.item()
            total_local_loss += local_loss.item()
            n_batches += 1

            pbar.set_postfix({
                "seg": seg_loss.item(), "bnd": boundary_loss.item(),
                "aux3": aux3_loss.item(), "aux2": aux2_loss.item(),
                "local": local_loss.item(), "gate": self.model.fusion_gate.item(),
            })

        summary = acc.summary()
        return {
            "loss": summary["loss"], "dice": summary["dice"],
            "seg_loss": total_seg_loss / max(1, n_batches),
            "boundary_loss": total_boundary_loss / max(1, n_batches),
            "aux3_loss": total_aux3_loss / max(1, n_batches),
            "aux2_loss": total_aux2_loss / max(1, n_batches),
            "local_loss": total_local_loss / max(1, n_batches),
        }

    def validate(self):
        """Uses the full dual-pathway forward pass (native_x provided) --
        the fused prediction IS the model's real output; evaluating with
        native_x=None would silently score only the coarse pathway,
        which is NOT what this phase is testing."""
        self.model.eval()
        acc = MetricAccumulator()
        ece_acc = ECEAccumulator(n_bins=15)
        boundary_bce_total = 0.0
        boundary_correct = 0
        boundary_total = 0
        n_batches = 0

        pbar = tqdm(self.val_loader, desc=f"[{self.condition_name}] Epoch {self.epoch+1} [Val]")
        with torch.no_grad():
            for images, masks, subject_ids, native_images in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                native_images = native_images.to(self.device)
                outputs = self.model(images, native_x=native_images)

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
            "lambda_ds3": LAMBDA_DS3, "lambda_ds2": LAMBDA_DS2, "lambda_local": LAMBDA_LOCAL,
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
        print(f"{self.condition_name}: lambda_local={LAMBDA_LOCAL} [seed={self.seed}, mu={self.mu}] Dual-Resolution Local Refinement")
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

            fusion_gate_val = float(self.model.fusion_gate.detach().cpu())
            peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6) if self.device.type == "cuda" else 0.0
            print(
                f"[{self.condition_name}] Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s, peak_mem={peak_mem_mb:.0f}MB) | "
                f"Train: loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} "
                f"seg={train_metrics['seg_loss']:.4f} bnd={train_metrics['boundary_loss']:.4f} "
                f"aux3={train_metrics['aux3_loss']:.4f} aux2={train_metrics['aux2_loss']:.4f} "
                f"local={train_metrics['local_loss']:.4f} | "
                f"Val: dice={val['dice']:.4f} precision={val['precision']:.4f} recall={val['recall']:.4f} "
                f"hd95={val['hd95']:.2f} ece={val['ece']:.4f} | gate={fusion_gate_val:.4f}"
            )

            self.w_metrics.writerow([
                epoch, train_metrics["loss"], train_metrics["dice"], train_metrics["seg_loss"],
                train_metrics["boundary_loss"], train_metrics["aux3_loss"], train_metrics["aux2_loss"],
                train_metrics["local_loss"],
                val["loss"], val["dice"], val["iou"], val["precision"], val["recall"], val["f1"],
                val["hd95"], val["ece"], val["boundary_bce"], val["boundary_accuracy_proxy"],
                fusion_gate_val, epoch_time, peak_mem_mb,
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
    exp = E55Experiment(
        config_path=str(CONFIG_PATH), exp_dir=str(OUT_DIR / "runs"), seed=seed, mu=MU,
        run_name=run_name, num_workers=None,
    )
    best_dice = exp.train(epochs=epochs, checkpoint_every=5)
    print(f"\n=== E55 Dual-Resolution run complete (seed={seed}): best_val_dice={best_dice:.4f} ===")
    return best_dice


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--run_name", type=str, default="DualRes_seed0")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(args.epochs, args.run_name, args.seed)
