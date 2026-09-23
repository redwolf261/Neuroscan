"""
Phase E25, Structural Pivot 1: training launcher for Condition B (A +
deep supervision at dec3/dec2, NO SC-TAM, NO margin loss at all).

Directly adapted from train_c62.py/train_c63.py's own established
pattern (extend via an adapted copy, not by editing the shared/frozen
architecture or training scripts). Uses UNet3D_v3 (neuroscan_3d_v3.py)
instead of UNet3D_v2 -- the ONLY architectural difference from condition
A. Uses calibrated lambda_ds3=0.9927, lambda_ds2=1.0014 (Structural
Pivot 1's own calibration, e25b_calibrate_deep_supervision.py) for the
two new auxiliary loss terms, ADDED to A's own existing loss composition
(L_seg + mu*L_boundary) -- SC-TAM's margin term is NOT present at all
(lambda_margin implicitly 0, matching the isolation requirement: this
experiment tests deep supervision ALONE, not in combination with
anything from the C6 family).

Preserves, per the user's explicit requirement, EVERYTHING else
identical to condition A: dataset split, preprocessing, optimizer
(AdamW), LR schedule (CosineAnnealingLR), seed (0), training epochs
(30), evaluation protocol (validate(), UNCHANGED), final-output head
(seg_head/dec1, UNCHANGED), checkpoint-selection procedure (best.pth by
val_dice, periodic saves every 5 epochs, UNCHANGED).
"""
import sys
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp00b_baseline_convergence"))
from metrics import MetricAccumulator  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_c0_weight_ablation"))
from calibration import ECEAccumulator  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import set_seed  # noqa: E402


def state_dict_hash(state_dict, keys_only=None):
    """Byte-identical mechanism to train_c62.py/train_c63.py's own
    function, EXTENDED with an optional keys_only filter -- v3 has
    EXTRA parameters (aux_head3/aux_head2) that v2/condition A's model
    does not, so a literal full-state_dict hash comparison against A
    would always mismatch trivially. The init-parity gate below compares
    only the SHARED keys (everything both v2 and v3 have), which is the
    actually meaningful comparison: does theta^(0) match on every
    parameter the two architectures have in common."""
    import hashlib
    h = hashlib.sha256()
    keys = sorted(state_dict.keys()) if keys_only is None else sorted(keys_only)
    for key in keys:
        t = state_dict[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def optimizer_metadata(optimizer):
    pg = optimizer.param_groups[0]
    return {k: pg[k] for k in ("lr", "betas", "eps", "weight_decay") if k in pg}


class DeepSupExperiment:
    def __init__(self, lambda_ds3, lambda_ds2, config_path, exp_dir, seed,
                 mu, run_name=None, num_workers=None, enable_aux3=True, enable_aux2=True,
                 condition_label=None):
        """Per PHASE_E25_STRUCTURAL_PIVOT_1B's own scale-ablation
        requirement: enable_aux3/enable_aux2 are now INDEPENDENT toggles
        (the original use_aux_loss flag controlled both together, only
        sufficient for the earlier both-vs-neither ablation). This is
        the SAME mechanism as before -- the auxiliary heads always exist
        in the architecture (same forward-pass compute, same parameter
        slots, UNet3D_v3 is unchanged), only which branch's loss/gradient
        is ACTUALLY applied changes, isolating whether D/4, D/2, or their
        combination causes the observed effect. condition_label overrides
        the auto-generated name for clearer logging/checkpoint naming."""
        self.enable_aux3 = enable_aux3
        self.enable_aux2 = enable_aux2
        if condition_label is not None:
            self.condition_name = condition_label
        elif enable_aux3 and enable_aux2:
            self.condition_name = "DeepSup_both"
        elif enable_aux3:
            self.condition_name = "DeepSup_D4only"
        elif enable_aux2:
            self.condition_name = "DeepSup_D2only"
        else:
            self.condition_name = "DeepSup_ablation_neither"
        self.lambda_ds3 = lambda_ds3 if enable_aux3 else 0.0
        self.lambda_ds2 = lambda_ds2 if enable_aux2 else 0.0
        self.use_aux_loss = enable_aux3 or enable_aux2  # kept for any external code still reading this attribute
        self.seed = seed
        self.mu = mu
        self.num_workers_override = num_workers
        set_seed(seed)  # IDENTICAL call/position to A/C6-2/C6-3 -- keeps theta^(0) identical on all SHARED parameters

        dir_name = run_name if run_name else f"deep_sup_seed{seed}"
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

        self.model = UNet3D_v3(
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
        )

        self.rng = np.random.RandomState(seed)

        print(f"[{self.condition_name} seed{seed}] lambda_ds3={self.lambda_ds3} lambda_ds2={self.lambda_ds2} mu={mu}")
        print(f"[{self.condition_name}] Training subjects: {len(self.train_loader.dataset)}")
        print(f"[{self.condition_name}] Validation subjects: {len(self.val_loader.dataset)}")
        print(f"[{self.condition_name}] Device: {self.device}")

        self._open_logs()

    def _open_logs(self):
        self.f_metrics = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w_metrics = csv.writer(self.f_metrics)
        self.w_metrics.writerow([
            "epoch", "train_loss", "train_dice", "train_seg_loss", "train_boundary_loss",
            "train_aux3_loss", "train_aux2_loss", "val_loss", "val_dice",
            "val_iou", "val_precision", "val_recall", "val_f1", "val_hd95", "val_ece",
            "boundary_bce", "boundary_auc_proxy", "epoch_time_sec", "peak_gpu_memory_mb",
        ])

    def close_logs(self):
        self.f_metrics.close()

    def get_initial_state_hash(self, shared_keys):
        return {
            "model_state_hash": state_dict_hash(self.model.state_dict(), keys_only=shared_keys),
            "optimizer_metadata": optimizer_metadata(self.optimizer),
        }

    def train_epoch(self):
        self.model.train()
        acc = MetricAccumulator()
        total_seg_loss = 0.0
        total_boundary_loss = 0.0
        total_aux3_loss = 0.0
        total_aux2_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"[{self.condition_name}] Epoch {self.epoch+1} [Train]")
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)

            probs = outputs["probs"]
            alpha, beta = outputs["alpha"], outputs["beta"]
            boundary_logit = outputs["boundary_logit"]
            aux_probs3 = outputs["aux_probs3"]
            aux_probs2 = outputs["aux_probs2"]

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

            total_loss = seg_loss + self.mu * boundary_loss + self.lambda_ds3 * aux3_loss + self.lambda_ds2 * aux2_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            acc.update(seg_loss.item(), probs, masks, compute_hd95=False)
            total_seg_loss += seg_loss.item()
            total_boundary_loss += boundary_loss.item()
            total_aux3_loss += aux3_loss.item()
            total_aux2_loss += aux2_loss.item()
            n_batches += 1

            pbar.set_postfix({
                "seg": seg_loss.item(), "bnd": boundary_loss.item(),
                "aux3": aux3_loss.item(), "aux2": aux2_loss.item(),
            })

        summary = acc.summary()
        return {
            "loss": summary["loss"], "dice": summary["dice"],
            "seg_loss": total_seg_loss / max(1, n_batches),
            "boundary_loss": total_boundary_loss / max(1, n_batches),
            "aux3_loss": total_aux3_loss / max(1, n_batches),
            "aux2_loss": total_aux2_loss / max(1, n_batches),
        }

    def validate(self):
        """Verbatim copy of A/C6-2/C6-3's own validate() -- UNCHANGED.
        Uses ONLY model(images) -> probs, no reference to aux outputs at
        all, matching the requirement that evaluation protocol/final-
        output head are unaffected by this mechanism."""
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
            "lambda_ds3": self.lambda_ds3, "lambda_ds2": self.lambda_ds2,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice, "config": self.config,
            "condition": self.condition_name,
            "use_aux_loss": self.use_aux_loss,
        }
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
        if is_periodic:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{self.epoch+1}.pth")

    def train(self, epochs, checkpoint_every=5):
        print("\n" + "=" * 70)
        print(f"{self.condition_name}: lambda_ds3={self.lambda_ds3} lambda_ds2={self.lambda_ds2} "
              f"[seed={self.seed}, mu={self.mu}]")
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
                f"seg={train_metrics['seg_loss']:.4f} bnd={train_metrics['boundary_loss']:.4f} "
                f"aux3={train_metrics['aux3_loss']:.4f} aux2={train_metrics['aux2_loss']:.4f} | "
                f"Val: dice={val['dice']:.4f} precision={val['precision']:.4f} recall={val['recall']:.4f} "
                f"hd95={val['hd95']:.2f} ece={val['ece']:.4f}"
            )

            self.w_metrics.writerow([
                epoch, train_metrics["loss"], train_metrics["dice"], train_metrics["seg_loss"],
                train_metrics["boundary_loss"], train_metrics["aux3_loss"], train_metrics["aux2_loss"],
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
