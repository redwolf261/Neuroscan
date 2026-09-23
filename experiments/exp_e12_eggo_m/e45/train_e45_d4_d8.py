"""
Phase E45: D4 + D8 combined deep supervision -- training script.

ALGORITHMIC MODIFICATION per the project's explicit post-E44 strategic
pivot: extend the validated D4 mechanism with an additional, coarser
auxiliary head at the bottleneck (8^3), rather than another loss-weighting
schedule on the existing heads (E44's RCGW family, both formulations,
killed -- see project memory phase_e44_rcgw_killed).

Uses UNet3D_v4 (neuroscan_3d_v4.py) -- D4 (dec3, 16^3) + D8 (bottleneck,
8^3) auxiliary heads BOTH active; D2 (dec2, 32^3) NOT used, matching this
project's own established finding that D2 does not help (D2-only underperformed
D4-only; "Both"=D4+D2 did not beat D4-only either) -- extending toward D8 in
the direction that's already validated to work (coarser), not revisiting D2.

Calibration: lambda_d8 = 0.4581, GRADIENT-matched (not value-matched --
value-matching gave a 2.19x gradient-magnitude mismatch, checked and
rejected per E34's own mandatory safeguard; see calibrate_lambda_d8.py and
E45_lambda_d8_calibration.json). lambda_ds3 = 0.9927, REUSED UNCHANGED from
every prior D4 condition (e25, e39) -- not re-calibrated, since D4's own
task/head is architecturally identical to every prior D4-only run.

Everything else identical to the proven D4-only protocol: dataset split,
preprocessing, optimizer (AdamW), LR schedule (CosineAnnealingLR), seed (0),
epochs (30), mu (0.1), evaluation protocol (validate(), UNCHANGED).

PRE-DECLARED SUCCESS CRITERION: >=1.0pp Dice improvement over the canonical
baseline (0.9063, e24/gate6_runs/A_baseline_seed0/checkpoints/best.pth).

PRE-DECLARED SECONDARY CHECK: improvement over D4-only's own already-
validated +0.33pp (0.9096) -- if D4+D8 does not at least match D4-only, the
D8 addition is not helping and should be killed regardless of whether it
beats the canonical baseline.

PRE-DECLARED KILL CONDITION: training instability (NaN/Inf, dice collapse
below 0.5 after epoch 5) at ANY point kills this run immediately without
waiting for all 30 epochs -- matching this project's own established
practice (E34's own real collapse incident) of investigating instability
immediately rather than letting a doomed run finish.
"""
import sys
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v4 import UNet3D_v4  # noqa: E402
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
LAMBDA_D8 = 0.4581   # gradient-matched, see calibrate_lambda_d8.py
CONFIG_PATH = project_root / "configs" / "brats.yaml"
OUT_DIR = Path(__file__).parent


class E45Experiment:
    def __init__(self, config_path, exp_dir, seed, mu, run_name, num_workers=None):
        self.condition_name = run_name
        self.seed = seed
        self.mu = mu
        set_seed(seed)  # SAME call/position as every prior condition -- keeps theta^(0) identical on all SHARED parameters

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

        self.model = UNet3D_v4(
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
            num_workers=(num_workers if num_workers is not None else self.config["training"].get("num_workers", 0)),
            root_dir=self.config["dataset"]["root_dir"],
            val_split=self.config["dataset"]["val_split"],
        )

        print(f"[{self.condition_name} seed{seed}] lambda_ds3={LAMBDA_DS3} lambda_d8={LAMBDA_D8} mu={mu}")
        print(f"[{self.condition_name}] Training subjects: {len(self.train_loader.dataset)}")
        print(f"[{self.condition_name}] Validation subjects: {len(self.val_loader.dataset)}")
        print(f"[{self.condition_name}] Device: {self.device}")

        self._open_logs()

    def _open_logs(self):
        self.f_metrics = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w_metrics = csv.writer(self.f_metrics)
        self.w_metrics.writerow([
            "epoch", "train_loss", "train_dice", "train_seg_loss", "train_boundary_loss",
            "train_aux3_loss", "train_d8_loss", "val_loss", "val_dice",
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
        total_aux3_loss = 0.0
        total_d8_loss = 0.0
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
            aux_probs_d8 = outputs["aux_probs_d8"]

            focal_loss = self.focal_fn(probs, masks)
            evidential_loss = self.evidential_fn(alpha, beta, masks)
            seg_loss = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss

            if not torch.isfinite(seg_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf seg_loss at epoch {self.epoch} batch {batch_idx}")

            boundary_loss = self.boundary_criterion(boundary_logit, masks)
            if not torch.isfinite(boundary_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf boundary_loss at epoch {self.epoch} batch {batch_idx}")

            mask_d4 = F.avg_pool3d(masks, kernel_size=4, stride=4)
            mask_d8 = F.avg_pool3d(masks, kernel_size=8, stride=8)
            aux3_loss = self.focal_fn(aux_probs3, mask_d4)
            d8_loss = self.focal_fn(aux_probs_d8, mask_d8)
            if not torch.isfinite(aux3_loss) or not torch.isfinite(d8_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf aux loss at epoch {self.epoch} batch {batch_idx}")

            total_loss = seg_loss + self.mu * boundary_loss + LAMBDA_DS3 * aux3_loss + LAMBDA_D8 * d8_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            acc.update(seg_loss.item(), probs, masks, compute_hd95=False)
            total_seg_loss += seg_loss.item()
            total_boundary_loss += boundary_loss.item()
            total_aux3_loss += aux3_loss.item()
            total_d8_loss += d8_loss.item()
            n_batches += 1

            pbar.set_postfix({
                "seg": seg_loss.item(), "bnd": boundary_loss.item(),
                "aux3": aux3_loss.item(), "d8": d8_loss.item(),
            })

        summary = acc.summary()
        return {
            "loss": summary["loss"], "dice": summary["dice"],
            "seg_loss": total_seg_loss / max(1, n_batches),
            "boundary_loss": total_boundary_loss / max(1, n_batches),
            "aux3_loss": total_aux3_loss / max(1, n_batches),
            "d8_loss": total_d8_loss / max(1, n_batches),
        }

    def validate(self):
        """Verbatim copy of every prior condition's own validate() -- UNCHANGED."""
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
            "lambda_ds3": LAMBDA_DS3, "lambda_d8": LAMBDA_D8,
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
        print(f"{self.condition_name}: lambda_ds3={LAMBDA_DS3} lambda_d8={LAMBDA_D8} [seed={self.seed}, mu={self.mu}]")
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

            # KILL CONDITION: instability check, immediate, not deferred to
            # post-hoc analysis -- per E34's own established practice.
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
                f"aux3={train_metrics['aux3_loss']:.4f} d8={train_metrics['d8_loss']:.4f} | "
                f"Val: dice={val['dice']:.4f} precision={val['precision']:.4f} recall={val['recall']:.4f} "
                f"hd95={val['hd95']:.2f} ece={val['ece']:.4f}"
            )

            self.w_metrics.writerow([
                epoch, train_metrics["loss"], train_metrics["dice"], train_metrics["seg_loss"],
                train_metrics["boundary_loss"], train_metrics["aux3_loss"], train_metrics["d8_loss"],
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
    exp = E45Experiment(
        config_path=str(CONFIG_PATH), exp_dir=str(OUT_DIR / "runs"), seed=seed, mu=MU,
        run_name=run_name, num_workers=None,
    )
    best_dice = exp.train(epochs=epochs, checkpoint_every=5)
    print(f"\n=== E45 D4+D8 run complete (seed={seed}): best_val_dice={best_dice:.4f} ===")
    return best_dice


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--run_name", type=str, default="D4_D8_seed0")
    # NOTE: SEED (module-level, =0) remains the DEFAULT for consistency with
    # the original single-seed run; --seed is a NEW flag added for the E45/
    # E54 3-seed confirmation (same precedent as E49's own --seed addition),
    # not a silent behavior change to the original run.
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(args.epochs, args.run_name, args.seed)
