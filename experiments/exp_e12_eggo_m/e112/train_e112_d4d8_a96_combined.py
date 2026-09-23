"""
Phase E112: D4+D8 auxiliary supervision COMBINED with 96^3 resolution.

RATIONALE (existing evidence -> causal diagnosis -> intervention, not
blind search): E45 (D4+D8 auxiliary heads, UNet3D_v4) and E54 (96^3
whole-volume resolution, no architecture change) are the ONLY two
mechanisms in this project's entire post-pivot history whose single-seed
per-subject Dice was DOUBLY statistically significant (paired-t AND
Wilcoxon) against baseline -- E45: 0.8986 (+1.44pp), E54: 0.8981
(+1.39pp). On the mandatory 3-seed confirmation, BOTH landed within
0.0003 of the corrected +1pp target (E56/PHASE_EVIDENCE_MAP.md,
CORRECTED_TARGET=0.8942): E45's 3-seed mean = 0.8939 (0.0003 BELOW),
E54's 3-seed mean = 0.8944 (0.0002 ABOVE) -- both "coin-flip-level ties"
against the bar per the project's own explicit characterization
(PHASE_E45_E54_3SEED_CONFIRMATION_RESULT.md).

These two mechanisms fix CAUSALLY DIFFERENT, independently diagnosed
problems, never previously combined:
  - E29 (PHASE_E29_RESIZE_SURVIVAL.md): at 64^3, the MEDIAN native lesion
    component vanishes ENTIRELY (0 surviving voxels) before training even
    starts -- a preprocessing/information-loss problem. E54 (96^3)
    directly addresses this by reducing the resize-discretization loss.
  - E36 (PHASE_E36_DEEP_SUPERVISION_AUTOPSY.md): D4 supervision improves
    segmentation QUALITY for lesions that ARE detected (+40% relative
    Dice at 1-50 voxels) by supplying gradient signal at a resolution
    where the main loss has already converged -- a representation-level,
    not a detection-level or resolution-level, mechanism. E45 extends
    this one stage further (D8, the bottleneck) in the direction already
    shown to work (coarser beats finer: D4-only beat D2-only and "Both").

Twelve independent single-mechanism attempts to exploit this project's
OWN bottleneck-necessity causal signal (CCABA, IECG, CCAG, ASR, E93,
E97, and others) have all failed. This is the first attempt to combine
TWO ALREADY-INDIVIDUALLY-VALIDATED mechanisms addressing DIFFERENT root
causes, rather than search for a 13th untested single mechanism.

ARCHITECTURE: UNet3D_v4 (verified fully resolution-relative -- every
head is expressed in D/2, D/4, D/8 terms, no hardcoded dimension
anywhere in forward(); bottleneck at 96^3 input becomes 12^3 instead of
8^3, which is fine for aux_head_d8's 1x1x1 conv). D2 (aux_probs2,
inherited from v3) is produced by the model but DELIBERATELY UNUSED in
the loss, matching E45's own established precedent ("D2 does NOT help").

RESOLUTION/MEMORY: target_shape=(96,96,96), matching E54 exactly.
E54_memory_profile_96.json measured UNet3D_v3 at 96^3/batch=2 = 3.97GB,
far under the 8.15GB budget (4.18GB headroom) -- UNet3D_v4 adds one
extra 1x1x1 conv on a 12^3x256 tensor, a negligible addition relative to
that headroom, so E54's own batch=2/accum=4 config is reused directly
without re-profiling from scratch (verified via a smoke test before the
real campaign, not assumed blind).

LAMBDA CALIBRATION: LAMBDA_DS3=0.9927 and LAMBDA_D8=0.4581, REUSED
UNCHANGED from their original 64^3 gradient-matched calibrations,
following the SAME precedent E54 itself already established (E54 reused
LAMBDA_DS3=0.9927 and LAMBDA_DS2=1.0014 unchanged at 96^3 without
re-calibrating for the resolution change, and that run was stable/
uneventful). If the smoke test shows the D8 loss badly dominating or
negligible relative to seg_loss at 96^3, this must be revisited before
the real campaign -- not assumed away.

PRE-DECLARED SUCCESS CRITERION (identical structure to E45/E54's own
3-seed confirmation, PHASE_E45_E54_3SEED_CONFIRMATION_RESULT.md):
  - PRIMARY: 3-seed mean PER-SUBJECT Dice >= 0.8942 (corrected target),
    with a MEANINGFUL margin -- per this project's own explicit lesson
    from E45/E54 individually landing within 0.0002-0.0003 of this exact
    threshold, a margin smaller than 1/10 of the between-seed std will
    NOT be treated as a real pass, matching E54's own precedent.
  - SECONDARY: improvement over BOTH E45 alone (0.8939) AND E54 alone
    (0.8944) individually -- if the combination does not beat the better
    of its two parents, the combination is not adding anything beyond
    noise and should be reported as such, not spun as a partial win.

PRE-DECLARED KILL CONDITION: val dice < 0.5 after epoch 5 -> immediate
stop, matching every prior condition's own established practice.
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
LAMBDA_DS3 = 0.9927  # REUSED unchanged from every prior D4 condition (E25, E39, E45, E54)
LAMBDA_D8 = 0.4581   # REUSED unchanged from E45's gradient-matched 64^3 calibration
TARGET_SHAPE = (96, 96, 96)
PHYSICAL_BATCH_SIZE = 2  # measured (E54_memory_profile_96.json) to fit comfortably at 96^3
ACCUM_STEPS = 4  # effective batch = 2*4 = 8, matching every prior condition's optimizer dynamics
CONFIG_PATH = project_root / "configs" / "brats.yaml"
OUT_DIR = Path(__file__).parent


class E112Experiment:
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
            batch_size=PHYSICAL_BATCH_SIZE,
            num_workers=(num_workers if num_workers is not None else self.config["training"].get("num_workers", 0)),
            root_dir=self.config["dataset"]["root_dir"],
            val_split=self.config["dataset"]["val_split"],
            target_shape=TARGET_SHAPE,
        )

        print(f"[{self.condition_name} seed{seed}] target_shape={TARGET_SHAPE} physical_batch={PHYSICAL_BATCH_SIZE} "
              f"accum_steps={ACCUM_STEPS} (effective_batch={PHYSICAL_BATCH_SIZE*ACCUM_STEPS}) "
              f"lambda_ds3={LAMBDA_DS3} lambda_d8={LAMBDA_D8} mu={mu}")
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

        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc=f"[{self.condition_name}] Epoch {self.epoch+1} [Train]")
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)
            probs = outputs["probs"]
            alpha, beta = outputs["alpha"], outputs["beta"]
            boundary_logit = outputs["boundary_logit"]
            aux_probs3 = outputs["aux_probs3"]
            aux_probs_d8 = outputs["aux_probs_d8"]
            # aux_probs2 (D2) is produced by the model (inherited from v3)
            # but deliberately NOT used in the loss -- matching E45's own
            # established precedent that D2 does not help.

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
        """Verbatim copy of every prior condition's own validate() logic."""
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
            "lambda_ds3": LAMBDA_DS3, "lambda_d8": LAMBDA_D8, "target_shape": TARGET_SHAPE,
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
        print(f"{self.condition_name}: D4+D8 @ 96^3 [seed={self.seed}, mu={self.mu}] "
              f"lambda_ds3={LAMBDA_DS3} lambda_d8={LAMBDA_D8}")
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
    exp = E112Experiment(
        config_path=str(CONFIG_PATH), exp_dir=str(OUT_DIR / "runs"), seed=seed, mu=MU,
        run_name=run_name, num_workers=None,
    )
    best_dice = exp.train(epochs=epochs, checkpoint_every=5)
    print(f"\n=== E112 D4+D8@A96 run complete (seed={seed}): best_val_dice={best_dice:.4f} ===")
    return best_dice


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--run_name", type=str, default="D4D8_A96_seed0")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(args.epochs, args.run_name, args.seed)
