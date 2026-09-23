"""
Phase E24, Gate 6: training launcher for conditions A/B/E, per
PHASE_E24_GATE6_EXPERIMENTAL_SPECIFICATION.md.

Adapted copy of train_eggo_m.py's EGGOMExperiment (matching this
project's established pattern -- e18_ablation_train.py -- of extending
via an adapted copy rather than editing the shared script), extended
with EXACTLY the capability this gate needs:

  1. objective_config-aware compute_margin_loss calls (w_hat/delta_d
     switch per condition, using run_counterfactual.py's own
     ObjectiveConfig class -- the SAME class already regression-verified
     against the original E22 implementation, not a reimplementation).
  2. Checkpoint provenance fields (condition, objective mode, calibration
     constant, r's seed/hash if applicable) -- per Section 3's "don't
     rely on filenames to reconstruct experimental provenance" requirement.
  3. Initial-parameter-equality hash gate -- computed and asserted by
     run_gate6_all.py (the orchestrator) BEFORE any of A/B/E's real
     training epochs run, per Section 6 step 4's hard gate.

Everything else -- train_epoch's core loop structure, validate(),
save_checkpoint's base fields, the optimizer/scheduler construction, the
data loader construction -- is copied verbatim from train_eggo_m.py's
EGGOMExperiment, not redesigned.
"""
import sys
import csv
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp00b_baseline_convergence"))
from metrics import MetricAccumulator  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_c0_weight_ablation"))
from calibration import ECEAccumulator  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import (  # noqa: E402
    sample_stratified_anchors, compute_margin_loss,
    ANCHORS_PER_VOLUME, MAX_NEGATIVES_PER_ANCHOR, EVIDENCE_P99_DEFAULT,
    EMATauB, set_seed,
)

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e24"))
from run_counterfactual import ObjectiveConfig  # noqa: E402 -- SAME class, regression-verified against original E22


def state_dict_hash(state_dict):
    """SHA-256 of the concatenated, flattened state_dict tensors -- per
    PHASE_E24_GATE6_EXPERIMENTAL_SPECIFICATION.md Section 3's
    initial-parameter-equality gate."""
    import hashlib
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        t = state_dict[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def optimizer_metadata(optimizer):
    """Extracts the param_groups metadata (lr, betas, eps, weight_decay)
    for the initial-parameter-equality gate's optimizer-state comparison
    -- NOT the (empty, at construction time) moment buffers, which have
    no values yet to compare."""
    pg = optimizer.param_groups[0]
    return {k: pg[k] for k in ("lr", "betas", "eps", "weight_decay") if k in pg}


class Gate6Experiment:
    def __init__(self, condition_name, objective_config, config_path, exp_dir, seed,
                 mu, lambda_margin, run_name=None, num_workers=None):
        self.condition_name = condition_name  # "A", "B", or "E"
        self.objective_config = objective_config
        self.seed = seed
        self.mu = mu
        self.lambda_margin = lambda_margin
        self.num_workers_override = num_workers
        set_seed(seed)  # identical call, identical position (before model construction), across A/B/E -- per Section 3

        dir_name = run_name if run_name else f"gate6_{condition_name}_seed{seed}"
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
        )

        self.rng = np.random.RandomState(seed)
        self.tau_b_tracker = EMATauB()

        print(f"[Gate6-{condition_name} seed{seed}] objective={objective_config.mode} delta_d={objective_config.delta_d} "
              f"mu={mu} lambda={lambda_margin}")
        print(f"[Gate6-{condition_name}] Training subjects: {len(self.train_loader.dataset)}")
        print(f"[Gate6-{condition_name}] Validation subjects: {len(self.val_loader.dataset)}")
        print(f"[Gate6-{condition_name}] Device: {self.device}, initial tau_b={self.tau_b_tracker.tau_b:.4f}")

        self._open_logs()

    def _open_logs(self):
        self.f_metrics = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w_metrics = csv.writer(self.f_metrics)
        self.w_metrics.writerow([
            "epoch", "train_loss", "train_dice", "train_seg_loss", "train_boundary_loss",
            "train_margin_loss", "active_hinge_pct", "tau_b_end_of_epoch", "val_loss", "val_dice",
            "val_iou", "val_precision", "val_recall", "val_f1", "val_hd95", "val_ece",
            "boundary_bce", "boundary_auc_proxy", "epoch_time_sec", "peak_gpu_memory_mb",
        ])

    def close_logs(self):
        self.f_metrics.close()

    def get_initial_state_hash(self):
        """For the pre-training initial-parameter-equality gate -- called
        by the orchestrator immediately after __init__, before any
        training epoch runs."""
        return {
            "model_state_hash": state_dict_hash(self.model.state_dict()),
            "optimizer_metadata": optimizer_metadata(self.optimizer),
        }

    def train_epoch(self):
        self.model.train()
        acc = MetricAccumulator()
        total_seg_loss = 0.0
        total_boundary_loss = 0.0
        total_margin_loss = 0.0
        total_active_hinge_pct = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"[Gate6-{self.condition_name}] Epoch {self.epoch+1} [Train]")
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)

            probs = outputs["probs"]
            alpha, beta = outputs["alpha"], outputs["beta"]
            boundary_logit = outputs["boundary_logit"]
            dec1 = outputs["dec1"]

            focal_loss = self.focal_fn(probs, masks)
            evidential_loss = self.evidential_fn(alpha, beta, masks)
            seg_loss = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss

            if not torch.isfinite(seg_loss):
                raise RuntimeError(f"[Gate6-{self.condition_name}] NaN/Inf seg_loss at epoch {self.epoch} batch {batch_idx}")

            boundary_loss = self.boundary_criterion(boundary_logit, masks)
            if not torch.isfinite(boundary_loss):
                raise RuntimeError(f"[Gate6-{self.condition_name}] NaN/Inf boundary_loss at epoch {self.epoch} batch {batch_idx}")

            margin_loss = torch.tensor(0.0, device=self.device)
            active_hinge_pct = 0.0
            current_tau_b = self.tau_b_tracker.tau_b
            if self.lambda_margin > 0:
                with torch.no_grad():
                    evidence_full = (alpha + beta - 2.0)

                B, C, D, H, W = dec1.shape
                dec1_perm = dec1.permute(0, 2, 3, 4, 1).reshape(-1, C)
                evidence_flat = evidence_full.reshape(-1)
                boundary_flat = boundary_logit.reshape(-1)
                gt_flat = masks.reshape(-1)

                voxels_per_vol = D * H * W
                anchor_idx_list = []
                for b in range(B):
                    vol_evidence = evidence_flat[b * voxels_per_vol:(b + 1) * voxels_per_vol]
                    local_idx = sample_stratified_anchors(vol_evidence, ANCHORS_PER_VOLUME, self.rng)
                    anchor_idx_list.append(local_idx + b * voxels_per_vol)
                anchor_idx = torch.cat(anchor_idx_list)

                # ONLY CHANGE FROM train_eggo_m.py's own train_epoch(): use
                # objective_config's w_hat/delta_d instead of the hardcoded
                # w_hat=None/self.delta_d -- everything else in this block
                # (anchor sampling, tau_b tracking, the compute_margin_loss
                # call's other arguments) is byte-identical.
                w_hat = self.objective_config.get_w_hat(self.model, self.device)
                margin_loss, mw, margin_diag = compute_margin_loss(
                    dec1_perm, evidence_flat, boundary_flat, gt_flat,
                    anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT, self.objective_config.delta_d,
                    MAX_NEGATIVES_PER_ANCHOR, self.rng, self.device,
                    w_hat=w_hat,
                )
                if not torch.isfinite(margin_loss):
                    raise RuntimeError(f"[Gate6-{self.condition_name}] NaN/Inf margin_loss at epoch {self.epoch} batch {batch_idx}")

                active_hinge_pct = margin_diag["active_hinge_pct"]
                self.tau_b_tracker.update(margin_diag["abs_boundary_logit"])

            total_loss = seg_loss + self.mu * boundary_loss + self.lambda_margin * margin_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            acc.update(seg_loss.item(), probs, masks, compute_hd95=False)
            total_seg_loss += seg_loss.item()
            total_boundary_loss += boundary_loss.item()
            total_margin_loss += margin_loss.item() if isinstance(margin_loss, torch.Tensor) else margin_loss
            total_active_hinge_pct += active_hinge_pct
            n_batches += 1

            pbar.set_postfix({
                "seg": seg_loss.item(), "bnd": boundary_loss.item(),
                "margin": margin_loss.item() if isinstance(margin_loss, torch.Tensor) else margin_loss,
                "active%": f"{active_hinge_pct*100:.1f}", "tau_b": f"{current_tau_b:.3f}",
            })

        summary = acc.summary()
        return {
            "loss": summary["loss"], "dice": summary["dice"],
            "seg_loss": total_seg_loss / max(1, n_batches),
            "boundary_loss": total_boundary_loss / max(1, n_batches),
            "margin_loss": total_margin_loss / max(1, n_batches),
            "active_hinge_pct": total_active_hinge_pct / max(1, n_batches),
            "tau_b_end_of_epoch": self.tau_b_tracker.tau_b,
        }

    def validate(self):
        """Verbatim copy from train_eggo_m.py's EGGOMExperiment.validate() -- UNCHANGED."""
        self.model.eval()
        acc = MetricAccumulator()
        ece_acc = ECEAccumulator(n_bins=15)
        boundary_bce_total = 0.0
        boundary_correct = 0
        boundary_total = 0
        n_batches = 0

        pbar = tqdm(self.val_loader, desc=f"[Gate6-{self.condition_name}] Epoch {self.epoch+1} [Val]")
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
            "lambda_margin": self.lambda_margin,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice, "config": self.config,
            # NEW fields, per Section 3/6's "don't rely on filenames to
            # reconstruct experimental provenance" requirement:
            "gate6_condition": self.condition_name,
            "objective_mode": self.objective_config.mode,
            "delta_d_used": self.objective_config.delta_d,
            "random_seed_for_r": self.objective_config.random_seed if self.objective_config.mode == "random_projection" else None,
            "r_vector": self.objective_config._frozen_random_w_hat.tolist() if self.objective_config.mode == "random_projection" else None,
        }
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
        if is_periodic:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{self.epoch+1}.pth")

    def train(self, epochs, checkpoint_every=5):
        print("\n" + "=" * 70)
        print(f"GATE 6 CONDITION {self.condition_name}: objective={self.objective_config.mode} "
              f"[seed={self.seed}, mu={self.mu}, lambda={self.lambda_margin}, delta_d={self.objective_config.delta_d}]")
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
                f"[Gate6-{self.condition_name}] Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s, peak_mem={peak_mem_mb:.0f}MB) | "
                f"Train: loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} "
                f"seg={train_metrics['seg_loss']:.4f} bnd={train_metrics['boundary_loss']:.4f} "
                f"margin={train_metrics['margin_loss']:.6f} "
                f"active%={train_metrics['active_hinge_pct']*100:.2f} tau_b={train_metrics['tau_b_end_of_epoch']:.3f} | "
                f"Val: dice={val['dice']:.4f} precision={val['precision']:.4f} recall={val['recall']:.4f} "
                f"hd95={val['hd95']:.2f} ece={val['ece']:.4f} "
                f"boundary_bce={val['boundary_bce']:.4f} boundary_acc={val['boundary_accuracy_proxy']:.4f}"
            )

            self.w_metrics.writerow([
                epoch, train_metrics["loss"], train_metrics["dice"], train_metrics["seg_loss"],
                train_metrics["boundary_loss"], train_metrics["margin_loss"],
                train_metrics["active_hinge_pct"], train_metrics["tau_b_end_of_epoch"],
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
        print(f"\n[Gate6-{self.condition_name}] Done. Best Val Dice: {self.best_val_dice:.4f}")
        return self.best_val_dice
