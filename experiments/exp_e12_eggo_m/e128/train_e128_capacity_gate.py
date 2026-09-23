"""
Phase E128: TRACK A -- Capacity Evolution Gate.

WHAT THIS IS: a deliberately UNORIGINAL capacity mutation, run as a
screening gate. It reproduces the BraTS 2021 winner's own capacity change
(Luu & Park, arXiv:2112.04653: double the encoder filters, keep the
decoder filters, raise the bottleneck), which they credit to SegResNet
(Myronenko, arXiv:1810.11654). NO NOVELTY IS CLAIMED. The question is not
"is this new" -- it is "does extra encoder capacity fix the causal
phenotype this project measured in E48/E121/E126 on THIS network".

TRACK A / TRACK B SPLIT (user's directive): Track A is allowed to evolve
the architecture aggressively using known-strong ideas, with the only bar
being Dice. Track B (the research contribution) is a separate question
answered later from what the mutations reveal. Conflating the two is what
produced the E118-E127 rabbit hole.

THE ONLY DIFFERENCE FROM E46's PROTOCOL: UNet3D_v5 -> UNet3D_v12.
Everything else is a verbatim clone of experiments/exp_e12_eggo_m/e46/
train_e46_attention_gate.py -- same dataset split, preprocessing,
optimizer (AdamW), LR schedule (CosineAnnealingLR), seed (0), mu (0.1),
LAMBDA_DS3 (0.9927), loss composition, gradient clipping, validate(),
instability kill condition, and checkpoint format. This is deliberate:
any Dice difference must be attributable to width alone.

v12 (neuroscan_3d_v12.py) verified BEFORE this run:
  - 17,001,671 params vs v5's 5,607,447 (3.03x)
  - all 10 forward() output tensors match v5's shapes EXACTLY
  - no stale v5-width parameters survive the module rebind
  - forward() itself is NOT overridden -- v5's control flow is reused
    verbatim, so v12 differs from v5 in width and nothing else.

PRE-DECLARED SCREENING GATE (1 seed):
  Baseline to beat: canonical v5 best_val_dice = 0.9101624600589275
  (experiments/exp_e12_eggo_m/e46/runs/AttnGate_seed0/checkpoints/best.pth)
  PASS  : delta_dice >= +0.5pp  (i.e. best_val_dice >= 0.9152)
          -> proceed to a 3-SEED confirmation before believing it, then
             characterize WHICH subjects benefit (per-subject error,
             lesion-size stratification, N_1/N_2/N_3, bottleneck rank).
  FAIL  : delta_dice < +0.5pp -> KILL capacity as a lineage, move to
          representation-complementarity (Mutation 2). No further compute.

WHY 1 SEED IS ONLY A SCREEN, STATED HONESTLY: this project's own E49
multiseed finding showed a +0.51pp single-seed result shrink to +0.31pp
across 3 seeds with a CI crossing zero. A 1-seed delta of ~0.5pp is
therefore INSIDE this project's own measured seed noise. A pass here is
NOT evidence the mutation works -- it is only permission to spend the
3-seed compute. A fail, however, is a reasonably safe kill, since the
mutation would have to be worth less than noise to fail.

PRE-DECLARED KILL CONDITION (inherited unchanged): NaN/Inf in any loss
term, or val_dice < 0.5 after epoch 5, stops the run immediately.
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

from neuroscan_3d_v12 import UNet3D_v12  # noqa: E402
from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402  (AMP-matched control arm)
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
CONFIG_PATH = project_root / "configs" / "brats.yaml"
OUT_DIR = Path(__file__).parent


class E128Experiment:
    def __init__(self, config_path, exp_dir, seed, mu, run_name, num_workers=None,
                 arch="v12", amp=True):
        self.condition_name = run_name
        self.arch = arch
        self.amp = amp
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

        arch_cls = {"v12": UNet3D_v12, "v5": UNet3D_v5}[arch]
        self.model = arch_cls(
            in_channels=self.config["model"]["in_channels"],
            out_channels=self.config["model"]["out_channels"],
        ).to(self.device)
        # AMP is REQUIRED for v12 on this 8.5GB card: fp32 v12 thrashes at
        # 7.64GB peak and runs 11.74 s/iter (16.5x v5) vs 0.49 s/iter with
        # AMP. Because AMP changes numerics, BOTH arms are trained with the
        # SAME amp setting so precision is held constant and width is the
        # only difference -- the canonical fp32 0.91016 is NOT a valid
        # control for an AMP run and is not used as one.
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(amp))

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

        print(f"[{self.condition_name} seed{seed}] lambda_ds3={LAMBDA_DS3} mu={mu} ({self.arch} arch, amp={self.amp})")
        print(f"[{self.condition_name}] Training subjects: {len(self.train_loader.dataset)}")
        print(f"[{self.condition_name}] Validation subjects: {len(self.val_loader.dataset)}")
        print(f"[{self.condition_name}] Device: {self.device}")

        self._open_logs()

    def _open_logs(self):
        self.f_metrics = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w_metrics = csv.writer(self.f_metrics)
        self.w_metrics.writerow([
            "epoch", "train_loss", "train_dice", "train_seg_loss", "train_boundary_loss",
            "train_aux3_loss", "val_loss", "val_dice",
            "val_iou", "val_precision", "val_recall", "val_f1", "val_hd95", "val_ece",
            "boundary_bce", "boundary_auc_proxy",
            "psi_mean", "psi_std", "psi_min", "psi_max",
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
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"[{self.condition_name}] Epoch {self.epoch+1} [Train]")
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.amp):
                outputs = self.model(images)

            probs = outputs["probs"]
            alpha, beta = outputs["alpha"], outputs["beta"]
            boundary_logit = outputs["boundary_logit"]
            aux_probs3 = outputs["aux_probs3"]

            focal_loss = self.focal_fn(probs, masks)
            evidential_loss = self.evidential_fn(alpha, beta, masks)
            seg_loss = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss

            if not torch.isfinite(seg_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf seg_loss at epoch {self.epoch} batch {batch_idx}")

            boundary_loss = self.boundary_criterion(boundary_logit, masks)
            if not torch.isfinite(boundary_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf boundary_loss at epoch {self.epoch} batch {batch_idx}")

            mask_d4 = F.avg_pool3d(masks, kernel_size=4, stride=4)
            aux3_loss = self.focal_fn(aux_probs3, mask_d4)
            if not torch.isfinite(aux3_loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf aux3_loss at epoch {self.epoch} batch {batch_idx}")

            total_loss = seg_loss + self.mu * boundary_loss + LAMBDA_DS3 * aux3_loss
            self.scaler.scale(total_loss).backward()
            # unscale before clipping so max_norm=1.0 means the same thing
            # it did in the fp32 protocol.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            acc.update(seg_loss.item(), probs, masks, compute_hd95=False)
            total_seg_loss += seg_loss.item()
            total_boundary_loss += boundary_loss.item()
            total_aux3_loss += aux3_loss.item()
            n_batches += 1

            pbar.set_postfix({
                "seg": seg_loss.item(), "bnd": boundary_loss.item(), "aux3": aux3_loss.item(),
            })

        summary = acc.summary()
        return {
            "loss": summary["loss"], "dice": summary["dice"],
            "seg_loss": total_seg_loss / max(1, n_batches),
            "boundary_loss": total_boundary_loss / max(1, n_batches),
            "aux3_loss": total_aux3_loss / max(1, n_batches),
        }

    def validate(self):
        """Verbatim copy of every prior condition's own validate() logic,
        PLUS attention_map (psi) diagnostic accumulation -- the only
        addition, and it does not affect any loss/metric computation."""
        self.model.eval()
        acc = MetricAccumulator()
        ece_acc = ECEAccumulator(n_bins=15)
        boundary_bce_total = 0.0
        boundary_correct = 0
        boundary_total = 0
        n_batches = 0

        psi_sum = 0.0
        psi_sq_sum = 0.0
        psi_min = float("inf")
        psi_max = float("-inf")
        psi_count = 0

        pbar = tqdm(self.val_loader, desc=f"[{self.condition_name}] Epoch {self.epoch+1} [Val]")
        with torch.no_grad():
            for images, masks, _ in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                with torch.amp.autocast("cuda", enabled=self.amp):
                    outputs = self.model(images)

                probs, alpha, beta = outputs["probs"], outputs["alpha"], outputs["beta"]
                boundary_logit = outputs["boundary_logit"]
                psi = outputs["attention_map"]

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

                # AMP FIX: W_psi is a Conv3d, so under autocast psi is fp16.
                # Summing ~2.1M fp16 values per batch overflows fp16's 65504
                # max INSIDE .sum(), yielding inf (observed as psi_mean=inf,
                # psi_std=0 in the Control_v5amp_seed0 arm). Accumulate in
                # fp32 explicitly. Diagnostic-only -- psi never enters any
                # loss or metric, so this cannot affect training or Dice.
                psi_sum += psi.sum(dtype=torch.float32).item()
                psi_sq_sum += (psi.float() ** 2).sum(dtype=torch.float32).item()
                psi_min = min(psi_min, psi.min().item())
                psi_max = max(psi_max, psi.max().item())
                psi_count += psi.numel()

                pbar.set_postfix({"loss": total.item()})

        summary = acc.summary()
        ece, _ = ece_acc.compute()
        summary["ece"] = ece
        summary["boundary_bce"] = boundary_bce_total / max(1, n_batches)
        summary["boundary_accuracy_proxy"] = boundary_correct / max(1, boundary_total)

        psi_mean = psi_sum / max(1, psi_count)
        psi_var = max(0.0, psi_sq_sum / max(1, psi_count) - psi_mean ** 2)
        summary["psi_mean"] = psi_mean
        summary["psi_std"] = psi_var ** 0.5
        summary["psi_min"] = psi_min
        summary["psi_max"] = psi_max
        return summary

    def save_checkpoint(self, is_best=False, is_periodic=True):
        checkpoint = {
            "epoch": self.epoch, "seed": self.seed, "mu": self.mu,
            "lambda_ds3": LAMBDA_DS3,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice, "config": self.config,
            "condition": self.condition_name,
            "arch": self.arch, "amp": self.amp,
        }
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
        if is_periodic:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{self.epoch+1}.pth")

    def train(self, epochs, checkpoint_every=5):
        print("\n" + "=" * 70)
        print(f"{self.condition_name}: lambda_ds3={LAMBDA_DS3} [seed={self.seed}, mu={self.mu}] arch={self.arch} amp={self.amp}")
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
                f"aux3={train_metrics['aux3_loss']:.4f} | "
                f"Val: dice={val['dice']:.4f} precision={val['precision']:.4f} recall={val['recall']:.4f} "
                f"hd95={val['hd95']:.2f} ece={val['ece']:.4f} | "
                f"psi: mean={val['psi_mean']:.4f} std={val['psi_std']:.4f} "
                f"min={val['psi_min']:.4f} max={val['psi_max']:.4f}"
            )

            self.w_metrics.writerow([
                epoch, train_metrics["loss"], train_metrics["dice"], train_metrics["seg_loss"],
                train_metrics["boundary_loss"], train_metrics["aux3_loss"],
                val["loss"], val["dice"], val["iou"], val["precision"], val["recall"], val["f1"],
                val["hd95"], val["ece"], val["boundary_bce"], val["boundary_accuracy_proxy"],
                val["psi_mean"], val["psi_std"], val["psi_min"], val["psi_max"],
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


def main(epochs, run_name, arch, amp, seed):
    exp = E128Experiment(
        config_path=str(CONFIG_PATH), exp_dir=str(OUT_DIR / "runs"), seed=seed, mu=MU,
        run_name=run_name, num_workers=None, arch=arch, amp=amp,
    )
    best_dice = exp.train(epochs=epochs, checkpoint_every=5)
    print(f"\n=== E128 capacity-gate run complete: best_val_dice={best_dice:.4f} ===")
    return best_dice


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--run_name", type=str, default="Capacity_v12_seed0")
    parser.add_argument("--arch", type=str, default="v12", choices=["v12", "v5"])
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(args.epochs, args.run_name, args.arch, bool(args.amp), args.seed)
