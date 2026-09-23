"""
Phase E18: Representation Rotation Source -- ablation training script.

Adapts train_eggo_m.py (unchanged copy of its core training loop, loss
formulas, and calibrated constants) to add exactly ONE new capability:
an --ablation flag selecting which single component of the network is
frozen, per PHASE_E18's controlled-ablation design. No new losses, no
new hyperparameters, no tuning -- every ablation modifies exactly one
component's trainability while leaving the rest of the training
procedure (data, seed, optimizer type, LR schedule, loss weights,
delta_d, checkpoint schedule) byte-for-byte identical to the E12f/E13
baseline protocol.

Ablations (mutually exclusive, selected via --ablation):
  none            : baseline, reproduces train_eggo_m.py exactly (sanity check)
  freeze_bn       : every BatchNorm3d's running_mean/running_var stop
                    updating (momentum set to 0 post-init), but the
                    forward pass still uses LIVE per-batch statistics
                    for normalization (model stays in .train() mode) --
                    affine (weight/bias) parameters still train normally
                    via backprop. Isolates "does the running-stat
                    bookkeeping drift" from forward-pass dynamics --
                    the less disruptive of two possible interpretations,
                    chosen explicitly (see PHASE_E18 doc's design note).
  freeze_encoder  : enc1/enc2/enc3/bottleneck parameters excluded from
                    the optimizer (requires_grad=False) -- decoder, both
                    heads, boundary head train exactly as usual.
  freeze_decoder  : upconv3/dec3/upconv2/dec2/upconv1/dec1 parameters
                    excluded from the optimizer -- encoder, both heads,
                    boundary head train exactly as usual. NOTE:
                    seg_head/evidential_head/boundary_head all read
                    dec1's OUTPUT, so freezing dec1 means their INPUT
                    distribution is frozen too, though their own
                    weights still adapt -- flagged explicitly in the
                    PHASE_E18 doc, not hidden.
  freeze_seg_head : seg_head parameters excluded from the optimizer --
                    encoder, decoder, evidential_head, boundary_head all
                    train exactly as usual.
  lambda_zero     : identical to train_eggo_m.py's own existing
                    --lambda_margin 0 mode (boundary head still trains
                    via its own BCE, contributes nothing to L_margin
                    since it's zeroed) -- this ablation requires no new
                    code, just the existing CLI flag, included here only
                    for a single consistent launcher across all 6 configs.

All ablations use E12f/E13's exact calibrated constants (delta_d=3.6659,
adaptive tau_b, mu=0.1, lambda=0.1 unless ablation=lambda_zero, 30
epochs, same seed=0, same checkpoint schedule
{1,5,10,15,20,25,30}) -- the ONLY thing that differs between configs is
which single component is frozen.
"""
import os
import sys
import csv
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

project_root = Path(__file__).parent.parent.parent
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
    DELTA_D_CALIBRATED, ANCHORS_PER_VOLUME, MAX_NEGATIVES_PER_ANCHOR,
    EVIDENCE_P99_DEFAULT, EMATauB, set_seed,
    sample_stratified_anchors, compute_margin_loss,
)

ABLATIONS = ["none", "freeze_bn", "freeze_encoder", "freeze_decoder", "freeze_seg_head", "lambda_zero"]

ENCODER_MODULES = ["enc1", "enc2", "enc3", "bottleneck"]
DECODER_MODULES = ["upconv3", "dec3", "upconv2", "dec2", "upconv1", "dec1"]
SEG_HEAD_MODULES = ["seg_head"]


def freeze_bn_running_stats(model):
    """Applies the freeze_bn ablation's momentum=0 change. Split out from
    apply_ablation() so it can be called separately AFTER a warmup period
    (see BUG notes below -- there were two, in sequence), not just once
    at construction time.

    BUG 1 (original): the original code called this at model
    construction, BEFORE any real data had ever updated
    running_mean/running_var -- so with momentum=0 set immediately, the
    running stats stayed frozen at PyTorch's BatchNorm3d default init
    (running_mean=0, running_var=1) FOREVER, never having tracked
    anything from the actual data distribution even once. Training
    looked fine (train() mode uses live batch stats, ignoring the
    running buffers entirely) but validation collapsed (eval() mode uses
    the running buffers directly, which were pure random-init garbage,
    never real). Produced HD95~40-52 and Dice~1e-11 despite train_dice
    climbing normally to 0.92 -- a dead giveaway of train/eval BatchNorm
    divergence.

    BUG 2 (the "fix" for Bug 1 was itself insufficient): tried a 1-epoch
    warmup before freezing -- val_dice STILL degraded progressively from
    epoch 2 onward, collapsing to ~0 again by epoch 20+ (same pattern,
    just delayed). Root cause: 1 epoch of momentum=0.1 updates only
    partially converges running_mean/running_var toward the true
    distribution; freezing that immature snapshot goes stale as the LIVE
    (train-mode) distribution keeps drifting away from it. This is
    directly consistent with PHASE_E17's own finding that the
    representation keeps rotating substantially through roughly epoch
    5-10 before stabilizing -- freezing at epoch 1 means freezing
    mid-rotation, so staleness was reintroduced regardless of warmup
    length, just delayed rather than prevented.

    FIX (current): warmup through epoch 10 (0-indexed epoch==9, i.e.
    after the 10th real training epoch), THEN freeze -- chosen because
    it matches E17's own measured stabilization point, so the running
    stats are locked in AFTER most of the early-training rotation has
    already happened, addressing the staleness problem structurally
    (freezing at a point the representation has mostly stopped moving)
    rather than just delaying it. This is also more scientifically apt
    for E18's actual question -- whether BN's ongoing adaptation drives
    the EARLY rotation specifically (epochs 1-10) -- than an arbitrary
    short warmup would have been."""
    n_bn = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.momentum = 0.0
            n_bn += 1
    print(f"[E18 ablation=freeze_bn] Set momentum=0.0 on {n_bn} BatchNorm3d layers "
          f"(running stats now frozen at their CURRENT values -- forward pass still uses live batch stats "
          f"in train() mode, affine params still train)")


def apply_ablation(model, ablation):
    """Mutates model in place per the selected ablation. Returns the list
    of parameters that should be passed to the optimizer (excludes frozen
    ones entirely, rather than relying on requires_grad=False alone, so
    there is no ambiguity about whether AdamW's weight_decay could still
    touch a "frozen" parameter -- excluding from the param group is the
    unambiguous, spec-literal interpretation of "frozen").

    NOTE: freeze_bn is deliberately NOT applied here anymore -- see
    freeze_bn_running_stats()'s docstring for why freezing at construction
    time (before any real running-stat updates) was a bug. The caller
    (E18AblationExperiment.train()) now applies it after a 1-epoch warmup
    instead."""
    frozen_module_names = []
    if ablation == "none" or ablation == "lambda_zero" or ablation == "freeze_bn":
        pass  # freeze_bn's freezing is applied post-warmup by the caller, not here; lambda_zero's effect is entirely in the loss weight
    elif ablation == "freeze_encoder":
        frozen_module_names = ENCODER_MODULES
    elif ablation == "freeze_decoder":
        frozen_module_names = DECODER_MODULES
    elif ablation == "freeze_seg_head":
        frozen_module_names = SEG_HEAD_MODULES
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    frozen_param_ids = set()
    for name in frozen_module_names:
        submodule = getattr(model, name)
        for p in submodule.parameters():
            p.requires_grad_(False)
            frozen_param_ids.add(id(p))
        print(f"[E18 ablation={ablation}] Froze module '{name}' "
              f"({sum(p.numel() for p in submodule.parameters())} params)")

    trainable_params = [p for p in model.parameters() if id(p) not in frozen_param_ids]
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[E18 ablation={ablation}] Trainable params: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.1f}%)")
    return trainable_params


class E18AblationExperiment:
    def __init__(self, config_path, exp_dir, seed, mu, lambda_margin, delta_d, ablation, run_name=None, num_workers=None):
        self.seed = seed
        self.mu = mu
        self.lambda_margin = lambda_margin
        self.delta_d = delta_d
        self.ablation = ablation
        self.num_workers_override = num_workers
        set_seed(seed)

        dir_name = run_name if run_name else f"e18_{ablation}_seed{seed}"
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

        trainable_params = apply_ablation(self.model, ablation)

        self.focal_fn = FocalTverskyLoss()
        self.evidential_fn = EvidentialBetaLoss(weight=0.5)
        self.focal_weight = 0.5
        self.evidential_weight = 0.5
        self.boundary_criterion = nn.BCEWithLogitsLoss()

        self.optimizer = AdamW(
            trainable_params,
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

        print(f"[E18 {ablation} seed{seed}] Training subjects: {len(self.train_loader.dataset)}")
        print(f"[E18 {ablation}] Validation subjects: {len(self.val_loader.dataset)}")
        print(f"[E18 {ablation}] Device: {self.device}, delta_d={delta_d}, mu={mu}, lambda={lambda_margin}")

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

    def train_epoch(self):
        self.model.train()
        acc = MetricAccumulator()
        total_seg_loss = 0.0
        total_boundary_loss = 0.0
        total_margin_loss = 0.0
        total_active_hinge_pct = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"[E18 {self.ablation}] Epoch {self.epoch+1} [Train]")
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
                raise RuntimeError(f"[E18 {self.ablation}] NaN/Inf seg_loss at epoch {self.epoch} batch {batch_idx}")

            boundary_loss = self.boundary_criterion(boundary_logit, masks)
            if not torch.isfinite(boundary_loss):
                raise RuntimeError(f"[E18 {self.ablation}] NaN/Inf boundary_loss at epoch {self.epoch} batch {batch_idx}")

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

                margin_loss, mw, margin_diag = compute_margin_loss(
                    dec1_perm, evidence_flat, boundary_flat, gt_flat,
                    anchor_idx, current_tau_b, EVIDENCE_P99_DEFAULT, self.delta_d,
                    MAX_NEGATIVES_PER_ANCHOR, self.rng, self.device,
                )
                if not torch.isfinite(margin_loss):
                    raise RuntimeError(f"[E18 {self.ablation}] NaN/Inf margin_loss at epoch {self.epoch} batch {batch_idx}")

                active_hinge_pct = margin_diag["active_hinge_pct"]
                self.tau_b_tracker.update(margin_diag["abs_boundary_logit"])

            total_loss = seg_loss + self.mu * boundary_loss + self.lambda_margin * margin_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], max_norm=1.0)
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
        self.model.eval()
        acc = MetricAccumulator()
        ece_acc = ECEAccumulator(n_bins=15)
        boundary_bce_total = 0.0
        boundary_correct = 0
        boundary_total = 0
        n_batches = 0

        pbar = tqdm(self.val_loader, desc=f"[E18 {self.ablation}] Epoch {self.epoch+1} [Val]")
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
            "lambda_margin": self.lambda_margin, "ablation": self.ablation,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice, "config": self.config,
        }
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
        if is_periodic:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{self.epoch+1}.pth")

    def train(self, epochs, checkpoint_every=5):
        print("\n" + "=" * 70)
        print(f"EXPERIMENT E18: ablation={self.ablation} [seed={self.seed}, mu={self.mu}, lambda={self.lambda_margin}]")
        print("=" * 70 + "\n")

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()

            train_metrics = self.train_epoch()

            # freeze_bn: freeze running-stat updates AFTER a 10-epoch
            # warmup (epoch==9, 0-indexed, i.e. after the 10th real
            # training epoch) -- see freeze_bn_running_stats()'s
            # docstring for the two prior bugs (freezing at construction;
            # freezing after only 1 warmup epoch) this warmup length
            # specifically fixes, chosen to match PHASE_E17's own
            # measured representation-stabilization point so the running
            # stats are locked in AFTER most of the early rotation has
            # already happened, not mid-rotation.
            if self.ablation == "freeze_bn" and epoch == 9:
                freeze_bn_running_stats(self.model)

            val = self.validate()
            self.scheduler.step()
            epoch_time = time.time() - t0

            peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6) if self.device.type == "cuda" else 0.0

            print(
                f"[E18 {self.ablation}] Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s, peak_mem={peak_mem_mb:.0f}MB) | "
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
        print(f"\n[E18 {self.ablation}] Done. Best Val Dice: {self.best_val_dice:.4f}")
        return self.best_val_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment E18: Representation Rotation Source ablations")
    parser.add_argument("--config", default="../../configs/brats.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--lambda_margin", type=float, default=0.1)
    parser.add_argument("--delta_d", type=float, default=DELTA_D_CALIBRATED)
    parser.add_argument("--ablation", choices=ABLATIONS, required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    lambda_margin = 0.0 if args.ablation == "lambda_zero" else args.lambda_margin

    exp_dir = Path(__file__).parent
    trainer = E18AblationExperiment(
        args.config, exp_dir, seed=args.seed, mu=args.mu,
        lambda_margin=lambda_margin, delta_d=args.delta_d, ablation=args.ablation,
        run_name=args.run_name, num_workers=args.num_workers,
    )
    trainer.train(args.epochs)
