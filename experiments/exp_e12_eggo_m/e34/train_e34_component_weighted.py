"""
Phase E34: A/S/R controlled training experiment. Adapted copy of
train_e29_resolution.py's ResolutionExperiment (matching this project's
established extend-via-copy pattern), fixed at resolution=64 (matching A
exactly), extended with the OPTIONAL component-weighted loss term from
component_weighted_loss.py.

Condition A (baseline): component_weight_mode=None -> loss is BYTE-IDENTICAL
to train_e29_resolution.py's A64 (FocalTversky+Evidential seg_loss + mu*
boundary_loss, nothing else) -- this IS the frozen baseline, re-run here
(not reusing the archived A64 checkpoint) so all three conditions share
one harness/logging path, exactly as E29 did for A64/A96/A128.

Condition S (size-control) / R (alpha_c): component_weight_mode="size" or
"alpha_c" -> loss = seg_loss + mu*boundary_loss + lambda_cw*L_component_weighted,
using the calibrated weight lookup tables from calibrate_weights.py.
"""
import sys
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml
import scipy.ndimage as ndi
from scipy.ndimage import zoom
import nibabel as nib

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders, BraTSDataset  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp00b_baseline_convergence"))
from metrics import MetricAccumulator  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_c0_weight_ablation"))
from calibration import ECEAccumulator  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import set_seed  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e34"))
from component_weighted_loss import WeightLookup, compute_component_weighted_loss, precompute_labeled_64_cache  # noqa: E402

LAMBDA_CW = 5.2  # weight of the component-weighted term, calibrated in calibrate_lambda_cw.py BEFORE any A/S/R training run (median seg_loss/cw_loss ratio ~5.2-5.3 across 6 real batches at fresh-init, both S and R conditions -- consistent, matched lambda used for both per the fairness requirement)


def resize_nn(volume, target_shape):
    current_shape = volume.shape
    zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
    return zoom(volume, zoom_factors, order=0)


def state_dict_hash(state_dict, keys_only=None):
    import hashlib
    h = hashlib.sha256()
    keys = sorted(state_dict.keys()) if keys_only is None else sorted(keys_only)
    for key in keys:
        t = state_dict[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class ComponentWeightedExperiment:
    def __init__(self, condition_name, component_weight_mode, weight_table_path, config_path, exp_dir, seed, mu,
                 run_name=None, num_workers=None, lambda_cw=LAMBDA_CW):
        """component_weight_mode: None (baseline A), "size" (S), or "alpha_c" (R)"""
        self.condition_name = condition_name
        self.component_weight_mode = component_weight_mode
        self.lambda_cw = lambda_cw if component_weight_mode is not None else 0.0
        self.seed = seed
        self.mu = mu
        self.num_workers_override = num_workers
        set_seed(seed)  # IDENTICAL call/position to A64/every prior experiment -- keeps theta^(0) reproducible

        dir_name = run_name if run_name else f"e34_{condition_name}_seed{seed}"
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

        self.model = UNet3D_v2(
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
            batch_size=self.config["training"]["batch_size"],
            num_workers=(self.num_workers_override if self.num_workers_override is not None
                         else self.config["training"].get("num_workers", 0)),
            root_dir=self.config["dataset"]["root_dir"],
            val_split=self.config["dataset"]["val_split"],
            target_shape=(64, 64, 64),
        )

        # ---- component-weighted-loss setup (only if enabled) ----
        self.weight_lookup = None
        self.labeled_64_cache = None
        if component_weight_mode is not None:
            import json
            with open(weight_table_path) as f:
                raw_table = json.load(f)
            parsed = {}
            for key, w in raw_table.items():
                subj, comp_id = key.rsplit("|", 1)
                parsed[(subj, int(comp_id))] = w
            self.weight_lookup = WeightLookup(parsed)

            # Precompute native-space component labeling AND its 64^3 resize
            # ONCE per subject (not per-batch, not per-epoch) -- see
            # component_weighted_loss.py's own docstring for the performance
            # bug this fixes (originally ~5s/batch, now amortized to a single
            # one-time cost at experiment construction).
            native_labeled_cache = {}
            train_ds_for_native = self.train_loader.dataset
            print(f"[{self.condition_name}] Precomputing native component labels for {len(train_ds_for_native.subject_dirs)} training subjects...")
            for subject_dir in train_ds_for_native.subject_dirs:
                subject_id = Path(subject_dir).name
                seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
                seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
                seg_binary = (seg_data > 0).astype(np.float32)
                native_labeled, _ = ndi.label(seg_binary > 0.5)
                native_labeled_cache[subject_id] = (native_labeled, seg_data.shape)
            print(f"[{self.condition_name}] Native component cache built. Precomputing 64^3 resize cache...")
            self.labeled_64_cache = precompute_labeled_64_cache(native_labeled_cache, resize_nn)
            print(f"[{self.condition_name}] 64^3 resize cache built.")

        print(f"[{self.condition_name} seed{seed}] mode={component_weight_mode} lambda_cw={self.lambda_cw} mu={mu}")
        print(f"[{self.condition_name}] Training subjects: {len(self.train_loader.dataset)}")
        print(f"[{self.condition_name}] Validation subjects: {len(self.val_loader.dataset)}")
        print(f"[{self.condition_name}] Device: {self.device}")

        self._open_logs()

    def _open_logs(self):
        self.f_metrics = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w_metrics = csv.writer(self.f_metrics)
        self.w_metrics.writerow([
            "epoch", "train_loss", "train_dice", "train_seg_loss", "train_boundary_loss", "train_cw_loss",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall", "val_f1",
            "val_hd95", "val_ece", "boundary_bce", "boundary_auc_proxy",
            "epoch_time_sec", "peak_gpu_memory_mb",
        ])

    def close_logs(self):
        self.f_metrics.close()

    def get_initial_state_hash(self):
        return {"model_state_hash": state_dict_hash(self.model.state_dict())}

    def train_epoch(self):
        self.model.train()
        acc = MetricAccumulator()
        total_seg_loss = 0.0
        total_boundary_loss = 0.0
        total_cw_loss = 0.0
        n_batches = 0
        n_comp_seen_total = 0
        n_comp_matched_total = 0

        pbar = tqdm(self.train_loader, desc=f"[{self.condition_name}] Epoch {self.epoch+1} [Train]")
        for batch_idx, (images, masks, subject_ids) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)

            probs = outputs["probs"]
            alpha, beta = outputs["alpha"], outputs["beta"]
            boundary_logit = outputs["boundary_logit"]

            focal_loss = self.focal_fn(probs, masks)
            evidential_loss = self.evidential_fn(alpha, beta, masks)
            seg_loss = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss

            if not torch.isfinite(seg_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf seg_loss at epoch {self.epoch} batch {batch_idx}")

            boundary_loss = self.boundary_criterion(boundary_logit, masks)
            if not torch.isfinite(boundary_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf boundary_loss at epoch {self.epoch} batch {batch_idx}")

            cw_loss = torch.tensor(0.0, device=self.device)
            if self.component_weight_mode is not None:
                cw_loss, diag = compute_component_weighted_loss(
                    probs, masks, list(subject_ids), self.weight_lookup, self.labeled_64_cache,
                    self.device,
                )
                if not torch.isfinite(cw_loss):
                    raise RuntimeError(f"[{self.condition_name}] NaN/Inf cw_loss at epoch {self.epoch} batch {batch_idx}")
                n_comp_seen_total += diag["n_components_seen"]
                n_comp_matched_total += diag["n_components_matched"]

            total_loss = seg_loss + self.mu * boundary_loss + self.lambda_cw * cw_loss
            if not torch.isfinite(total_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf total_loss at epoch {self.epoch} batch {batch_idx}")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            acc.update(seg_loss.item(), probs, masks, compute_hd95=False)
            total_seg_loss += seg_loss.item()
            total_boundary_loss += boundary_loss.item()
            total_cw_loss += cw_loss.item() if isinstance(cw_loss, torch.Tensor) else cw_loss
            n_batches += 1

            pbar.set_postfix({"seg": seg_loss.item(), "bnd": boundary_loss.item(), "cw": cw_loss.item() if isinstance(cw_loss, torch.Tensor) else cw_loss})

        if self.component_weight_mode is not None and n_batches > 0:
            print(f"[{self.condition_name}] Epoch {self.epoch+1}: components seen={n_comp_seen_total}, matched={n_comp_matched_total} "
                  f"({100*n_comp_matched_total/max(n_comp_seen_total,1):.1f}% match rate)")

        summary = acc.summary()
        return {
            "loss": summary["loss"], "dice": summary["dice"],
            "seg_loss": total_seg_loss / max(1, n_batches),
            "boundary_loss": total_boundary_loss / max(1, n_batches),
            "cw_loss": total_cw_loss / max(1, n_batches),
        }

    def validate(self):
        """Verbatim copy of every prior experiment's validate() -- UNCHANGED,
        no component weighting at validation time (this is a training-only
        supervision mechanism, per the design)."""
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
            "component_weight_mode": self.component_weight_mode, "lambda_cw": self.lambda_cw,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice, "config": self.config,
            "condition_name": self.condition_name,
        }
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
        if is_periodic:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{self.epoch+1}.pth")

    def train(self, epochs, checkpoint_every=5):
        print("\n" + "=" * 70)
        print(f"E34 CONDITION {self.condition_name}: mode={self.component_weight_mode} lambda_cw={self.lambda_cw} [seed={self.seed}, mu={self.mu}]")
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

            peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6) if self.device.type == "cuda" else 0.0

            print(
                f"[{self.condition_name}] Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s, peak_mem={peak_mem_mb:.0f}MB) | "
                f"Train: loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} "
                f"seg={train_metrics['seg_loss']:.4f} bnd={train_metrics['boundary_loss']:.4f} cw={train_metrics['cw_loss']:.6f} | "
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
