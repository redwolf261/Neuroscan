"""
Phase E100: Plain Lesion-Centric Augmentation (LCA) test on UNet3D_v5.

CONTEXT: Following the full-pipeline sweep after E48-E97's formal
closure (see phase_e97_preservation_killed_gate1 memory), Lesion-Centric
Augmentation (LCA, ISAIMS 2025) was verified as a real, published
technique giving +1.63pp Dice via lesion-region-targeted elastic/grid-
distortion augmentation. The differentiated candidate (Necessity-
Conditioned LCA, conditioning augmentation strength on E48's causal N_b
signal) was KILLED at its zero-training gate (E99, see phase_e99_nc_lca
_killed_gate memory) -- the model showed negligible, N_b-independent
fragility to lesion-region elastic perturbation at inference time.

E99's own memory record explicitly notes: "this diagnostic specifically
kills the NECESSITY-CONDITIONING angle, not the broader 'would generic
LCA-style augmentation help this project's model' question, which
remains untested." THIS PHASE tests that remaining open question
directly: does UNCONDITIONED (plain, uniform-strength) lesion-region
augmentation improve Dice on this project's own UNet3D_v5/E48-lineage
architecture at all?

IMPORTANT SCOPE NOTE: this is NOT proposed as this project's novel
contribution (LCA already exists, published). This is a cheap (1-seed
smoke test first, per this project's own established multi-seed
discipline before any claim) empirical check of whether the technique
transfers to this specific model/dataset, which is useful information
either way:
  - If it helps: confirms augmentation is a viable lever for THIS
    project specifically, and could inform a genuinely differentiated
    follow-up (a different differentiator than N_b-conditioning, since
    that angle is now closed).
  - If it doesn't help: this project's small-lesion Dice ceiling is
    NOT primarily an augmentation-diversity problem either, narrowing
    the remaining hypothesis space for the full-pipeline search.

CONDITIONS:
  BASELINE: UNet3D_v5 (E46/E48-E99's own architecture), D4-only style
    training (FocalTversky + Evidential + boundary + aux3 deep
    supervision, matching E46's own established config), standard
    dataset augmentation (whatever create_brats_loaders already applies
    by default -- none beyond resize/normalize per this project's own
    frozen strategy), seed=0.
  LCA: IDENTICAL to BASELINE in every respect (architecture, loss,
    optimizer, schedule, seed) EXCEPT: at each training step, with
    probability p_lca=0.5 (matching typical augmentation-probability
    conventions), apply a lesion-region-localized elastic deformation
    (same transform family as E99's diagnostic: smooth random
    displacement field, dilated-lesion-mask-restricted) to the training
    image before the forward pass. NOTHING else differs.

PRE-DECLARED SUCCESS CRITERION (1-seed smoke test only -- NOT a final
claim): if LCA's val_dice exceeds BASELINE's by >=0.5pp at ANY point
during a short training run (this is a SMOKE TEST threshold, lower than
the final 1.0pp bar, matching this project's own established practice
of a cheap first signal before committing to a full multi-seed run),
proceed to consider a 3-seed proper comparison. If LCA does not clear
even this lower smoke-test bar, do not proceed further with plain LCA
on this architecture.

PRE-DECLARED KILL CONDITION: training instability (NaN/Inf, dice
collapse below 0.5 after epoch 5) at ANY point kills the affected run
immediately, matching every prior condition's own established practice.
"""
import sys
import csv
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml
import scipy.ndimage as ndi

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp00b_baseline_convergence"))
from metrics import MetricAccumulator  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_c0_weight_ablation"))
from calibration import ECEAccumulator  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import set_seed  # noqa: E402

SEED = 0
MU = 0.1  # boundary loss weight, matching E46/E49's own established value
LAMBDA_DS3 = 0.9927  # REUSED unchanged from every prior D4 condition
P_LCA = 0.5  # per-step probability of applying lesion-centric augmentation, LCA condition only
ELASTIC_ALPHA = 8.0  # matches E99's diagnostic exactly, for direct comparability
ELASTIC_SIGMA = 3.0
DILATE_MARGIN = 4
CONFIG_PATH = project_root / "configs" / "brats.yaml"
OUT_DIR = Path(__file__).parent


def elastic_deform_lesion_region_batch(images, masks, rng, alpha, sigma, dilate_margin):
    """Batched version of E99's exact lesion-region elastic deformation,
    applied per-sample within a training batch (numpy/scipy on CPU, then
    moved back to the batch tensor -- deliberately simple/unoptimized,
    matching this project's own convention of correctness over
    throughput for a first smoke test)."""
    images_np = images.detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy()
    out = np.empty_like(images_np)
    for b in range(images_np.shape[0]):
        img = images_np[b, 0]
        msk = masks_np[b, 0]
        region = ndi.binary_dilation(msk > 0.5, iterations=dilate_margin).astype(np.float32)
        if region.sum() == 0:
            out[b, 0] = img
            continue
        shape = img.shape
        dx = ndi.gaussian_filter((rng.random(shape) * 2 - 1), sigma=sigma) * alpha * region
        dy = ndi.gaussian_filter((rng.random(shape) * 2 - 1), sigma=sigma) * alpha * region
        dz = ndi.gaussian_filter((rng.random(shape) * 2 - 1), sigma=sigma) * alpha * region
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
        indices = (x + dx, y + dy, z + dz)
        out[b, 0] = ndi.map_coordinates(img, indices, order=1, mode="reflect").astype(np.float32)
    return torch.from_numpy(out).to(images.device)


class E100Experiment:
    def __init__(self, config_path, exp_dir, seed, run_name, use_lca, num_workers=None):
        self.condition_name = run_name
        self.seed = seed
        self.use_lca = use_lca
        set_seed(seed)
        self.rng = np.random.default_rng(seed)

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

        self.model = UNet3D_v5(
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

        print(f"[{self.condition_name} seed{seed}] use_lca={use_lca} p_lca={P_LCA if use_lca else 0.0}")
        print(f"[{self.condition_name}] Training subjects: {len(self.train_loader.dataset)}")
        print(f"[{self.condition_name}] Validation subjects: {len(self.val_loader.dataset)}")
        print(f"[{self.condition_name}] Device: {self.device}")

        self._open_logs()

    def _open_logs(self):
        self.f_metrics = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w_metrics = csv.writer(self.f_metrics)
        self.w_metrics.writerow([
            "epoch", "train_loss", "train_dice", "val_loss", "val_dice",
            "val_iou", "val_precision", "val_recall", "val_f1", "val_hd95", "val_ece",
            "epoch_time_sec", "peak_gpu_memory_mb",
        ])

    def close_logs(self):
        self.f_metrics.close()

    def train_epoch(self):
        self.model.train()
        acc = MetricAccumulator()
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"[{self.condition_name}] Epoch {self.epoch+1} [Train]")
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            if self.use_lca and self.rng.random() < P_LCA:
                images = elastic_deform_lesion_region_batch(
                    images, masks, self.rng, ELASTIC_ALPHA, ELASTIC_SIGMA, DILATE_MARGIN
                )

            self.optimizer.zero_grad(set_to_none=True)
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
            mask_d4 = F.avg_pool3d(masks, kernel_size=4, stride=4)
            aux3_loss = self.focal_fn(aux_probs3, mask_d4)

            total_loss = seg_loss + MU * boundary_loss + LAMBDA_DS3 * aux3_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            acc.update(seg_loss.item(), probs, masks, compute_hd95=False)
            n_batches += 1
            pbar.set_postfix({"seg": seg_loss.item(), "bnd": boundary_loss.item(), "aux3": aux3_loss.item()})

        summary = acc.summary()
        return {"loss": summary["loss"], "dice": summary["dice"]}

    def validate(self):
        self.model.eval()
        acc = MetricAccumulator()
        ece_acc = ECEAccumulator(n_bins=15)
        n_batches = 0

        pbar = tqdm(self.val_loader, desc=f"[{self.condition_name}] Epoch {self.epoch+1} [Val]")
        with torch.no_grad():
            for images, masks, _ in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                probs, alpha, beta = outputs["probs"], outputs["alpha"], outputs["beta"]

                focal_loss = self.focal_fn(probs, masks)
                evidential_loss = self.evidential_fn(alpha, beta, masks)
                total = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss

                acc.update(total.item(), probs, masks, compute_hd95=True)
                ece_acc.update(alpha, beta, masks)
                n_batches += 1
                pbar.set_postfix({"loss": total.item()})

        summary = acc.summary()
        ece, _ = ece_acc.compute()
        summary["ece"] = ece
        return summary

    def save_checkpoint(self, is_best=False, is_periodic=True):
        checkpoint = {
            "epoch": self.epoch, "seed": self.seed, "use_lca": self.use_lca,
            "model_state": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice, "config": self.config, "condition": self.condition_name,
        }
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
        if is_periodic:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{self.epoch+1}.pth")

    def train(self, epochs, checkpoint_every=5):
        print("\n" + "=" * 70)
        print(f"{self.condition_name}: use_lca={self.use_lca} [seed={self.seed}]")
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
                      f"< 0.5 after epoch 5. Stopping immediately.", flush=True)
                self.close_logs()
                raise RuntimeError(f"Training instability at epoch {epoch}: val_dice={val['dice']:.4f}")

            peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6) if self.device.type == "cuda" else 0.0
            print(
                f"[{self.condition_name}] Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s, peak_mem={peak_mem_mb:.0f}MB) | "
                f"Train: loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} | "
                f"Val: dice={val['dice']:.4f} precision={val['precision']:.4f} recall={val['recall']:.4f} "
                f"hd95={val['hd95']:.2f} ece={val['ece']:.4f}"
            )

            self.w_metrics.writerow([
                epoch, train_metrics["loss"], train_metrics["dice"],
                val["loss"], val["dice"], val["iou"], val["precision"], val["recall"], val["f1"],
                val["hd95"], val["ece"], epoch_time, peak_mem_mb,
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


def main(epochs, run_name, seed, use_lca):
    exp = E100Experiment(
        config_path=str(CONFIG_PATH), exp_dir=str(OUT_DIR / "runs"), seed=seed,
        run_name=run_name, use_lca=use_lca, num_workers=None,
    )
    best_dice = exp.train(epochs=epochs, checkpoint_every=5)
    print(f"\n=== E100 {run_name} run complete (seed={seed}): best_val_dice={best_dice:.4f} ===")
    return best_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)  # smoke test: shorter than the full 30-epoch protocol
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--use_lca", action="store_true")
    args = parser.parse_args()
    main(args.epochs, args.run_name, args.seed, args.use_lca)
