"""
Phase E103: CC-DiceCE Mechanism-Validation Diagnostic (training-dependent,
short, BEFORE any full 3-seed performance campaign).

CONTEXT: E101 identified CC-DiceCE (instance-aware Voronoi-based loss,
arXiv 2511.17146) as a candidate whose OWN paper documents a specific
BraTS failure: precision drops because equal per-component weighting
over-penalizes missed small satellite components relative to false
positives, inducing over-detection. E102 ruled out an alternative
"insufficient evidence" story (graph-cut-style) for this project's
existing small-lesion failures, via a zero-training diagnostic.
CC-DiceCE's predicted failure mode is DIFFERENT in kind (excessive
positive predictions from loss geometry, not evidence absence), so it
survives as an independent candidate worth testing.

Per user's explicit staged plan: DO NOT run a full 3-seed campaign yet.
This phase runs a SHORT (5-epoch) side-by-side comparison of baseline
DiceCE (matching E100's own baseline convention exactly) vs CC-DiceCE
(frozen exact reimplementation, see cc_dicece_loss.py, verified against
the reference GitHub implementation) on the SAME UNet3D_v5 architecture,
SAME data, SAME seed -- tracking the specific quantities needed to
confirm or reject CC-DiceCE's predicted mechanism BEFORE spending
compute on a full run:

  - False positive / false negative voxel counts (does CC-DiceCE
    actually shift the FP/FN balance in the predicted direction --
    more FP, fewer FN -- relative to baseline?)
  - P(y_hat=1): mean predicted foreground probability (a coarse proxy
    for "is the model becoming more trigger-happy")
  - ||grad(L_Dice)||, ||grad(L_CE)||: gradient norms of the two loss
    components (measured on the GLOBAL DiceCE term only, for direct
    comparability to standard Dice/CE gradient-conflict literature)
  - cos(theta) = cosine similarity between grad(L_Dice) and grad(L_CE)
    w.r.t. the model's output logits -- do the two terms pull in
    increasingly different directions under CC-DiceCE vs baseline?

PRE-DECLARED DECISION RULE:
  MECHANISM CONFIRMED (proceed to a full 3-seed campaign) if, relative
  to baseline DiceCE, CC-DiceCE training shows:
    (a) FP count increases and/or FN count decreases in a directionally
        consistent way across epochs (matching the paper's own
        described precision-drop/recall-gain trade-off), AND
    (b) this is accompanied by a measurable shift in gradient geometry
        (either increased ||grad|| imbalance between Dice/CE terms, or
        a more negative/divergent cos(theta)) -- i.e. the predicted
        MECHANISM, not just a coincidental metric change, is present.
  MECHANISM NOT CONFIRMED (kill CC-DiceCE, do not proceed to a full
  campaign) if the FP/FN shift does not appear, or appears without any
  corresponding gradient-geometry signature (which would suggest an
  unrelated cause, not the paper's claimed mechanism).

NO correction/modification is applied to CC-DiceCE in this phase --
per the user's explicit guardrail, the paper's method is tested AS
PUBLISHED first; any correction is designed only after the predicted
failure is independently confirmed to occur in THIS project's own
architecture, and only then would the exact mathematical deficiency be
targeted specifically.

Architecture: UNet3D_v5, exactly matching E100's baseline scaffold
(FocalTversky+Evidential+boundary+aux3-D4 loss family REPLACED here by
plain DiceCE / CC-DiceCE specifically -- for a clean mechanism
comparison isolating the Dice/CE geometry, not entangled with this
project's other established auxiliary losses).
"""
import sys
import csv
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402
from cc_dicece_loss import CCDiceCELoss, compute_voronoi_cpu, cc_dice_ce_component_loss  # noqa: E402

sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m"))
from train_eggo_m import set_seed  # noqa: E402

SEED = 0
CONFIG_PATH = project_root / "configs" / "brats.yaml"
OUT_DIR = Path(__file__).parent
N_EPOCHS = 5  # short mechanism-audit run, NOT the full campaign


def plain_dice_ce_loss(probs, target, eps=1e-7):
    B = probs.shape[0]
    probs_flat = probs.reshape(B, -1)
    target_flat = target.reshape(B, -1)
    intersection = (probs_flat * target_flat).sum(dim=1)
    denom = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice_loss = (1.0 - (2.0 * intersection / denom.clamp_min(eps))).mean()
    ce = F.binary_cross_entropy(probs.clamp(eps, 1 - eps), target, reduction="mean")
    return dice_loss, ce


def gradient_geometry(model, probs, target, use_cc, eps=1e-7):
    """Computes ||grad(L_Dice)||, ||grad(L_CE)||, cos(theta) w.r.t. the
    model's OUTPUT LOGITS (not parameters -- cheaper, and directly
    comparable to standard gradient-conflict-in-output-space analyses).
    For CC-DiceCE, L_Dice/L_CE are still the GLOBAL Dice/CE terms (the
    instance-aware term is a separate, additional geometry not captured
    by this specific decomposition -- deliberately kept simple and
    directly comparable to the baseline condition)."""
    probs_detached = probs.detach().requires_grad_(True)
    dice_loss, ce_loss = plain_dice_ce_loss(probs_detached, target, eps=eps)

    grad_dice = torch.autograd.grad(dice_loss, probs_detached, retain_graph=True)[0]
    grad_ce = torch.autograd.grad(ce_loss, probs_detached, retain_graph=False)[0]

    norm_dice = float(grad_dice.norm().item())
    norm_ce = float(grad_ce.norm().item())
    cos_theta = float(F.cosine_similarity(
        grad_dice.reshape(1, -1), grad_ce.reshape(1, -1)
    ).item())
    return norm_dice, norm_ce, cos_theta


class E103Experiment:
    def __init__(self, config_path, exp_dir, seed, run_name, use_cc, num_workers=None):
        self.condition_name = run_name
        self.seed = seed
        self.use_cc = use_cc
        set_seed(seed)

        self.exp_dir = Path(exp_dir) / run_name
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        dataset_root = self.config["dataset"]["root_dir"]
        if not Path(dataset_root).is_absolute():
            dataset_root = project_root / dataset_root
        self.config["dataset"]["root_dir"] = str(dataset_root)

        self.device = torch.device(self.config.get("hardware", {}).get("device", "cpu"))
        self.epoch = 0

        self.model = UNet3D_v5(
            in_channels=self.config["model"]["in_channels"],
            out_channels=self.config["model"]["out_channels"],
        ).to(self.device)

        self.cc_loss_fn = CCDiceCELoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=N_EPOCHS, eta_min=1e-6)

        self.train_loader, self.val_loader = create_brats_loaders(
            batch_size=self.config["training"]["batch_size"],
            num_workers=(num_workers if num_workers is not None else self.config["training"].get("num_workers", 0)),
            root_dir=self.config["dataset"]["root_dir"],
            val_split=self.config["dataset"]["val_split"],
        )

        print(f"[{self.condition_name} seed{seed}] use_cc={use_cc}")
        print(f"[{self.condition_name}] Training subjects: {len(self.train_loader.dataset)}")
        print(f"[{self.condition_name}] Validation subjects: {len(self.val_loader.dataset)}")

        self.f_metrics = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w_metrics = csv.writer(self.f_metrics)
        self.w_metrics.writerow([
            "epoch", "train_loss", "val_dice", "val_fp_voxels", "val_fn_voxels",
            "val_mean_pred_prob", "grad_norm_dice", "grad_norm_ce", "grad_cos_theta",
            "epoch_time_sec",
        ])

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(self.train_loader, desc=f"[{self.condition_name}] Epoch {self.epoch+1} [Train]")
        for images, masks, _ in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(images)
            probs = out["probs"]

            if self.use_cc:
                loss, parts = self.cc_loss_fn(probs, masks)
            else:
                dice_loss, ce_loss = plain_dice_ce_loss(probs, masks)
                loss = dice_loss + ce_loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"[{self.condition_name}] NaN/Inf loss")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": loss.item()})

        return total_loss / max(1, n_batches)

    def validate(self):
        self.model.eval()
        dice_scores, fp_total, fn_total, prob_sum, n_voxels = [], 0, 0, 0.0, 0
        grad_norms_dice, grad_norms_ce, grad_coss = [], [], []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"[{self.condition_name}] Epoch {self.epoch+1} [Val]")
            for images, masks, _ in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                out = self.model(images)
                probs = out["probs"]

                pred_bin = (probs >= 0.5).float()
                tp = (pred_bin * masks).sum().item()
                fp = (pred_bin * (1 - masks)).sum().item()
                fn = ((1 - pred_bin) * masks).sum().item()
                denom = pred_bin.sum().item() + masks.sum().item()
                dice = (2 * tp / denom) if denom > 0 else 1.0

                dice_scores.append(dice)
                fp_total += fp
                fn_total += fn
                prob_sum += probs.sum().item()
                n_voxels += probs.numel()

        # Gradient geometry measured on a SMALL sample of training batches
        # (WITH grad enabled), separately from the no-grad validation loop above.
        self.model.eval()
        n_grad_samples = 0
        for images, masks, _ in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            with torch.no_grad():
                out = self.model(images)
                probs = out["probs"]
            norm_dice, norm_ce, cos_theta = gradient_geometry(self.model, probs, masks, self.use_cc)
            grad_norms_dice.append(norm_dice)
            grad_norms_ce.append(norm_ce)
            grad_coss.append(cos_theta)
            n_grad_samples += 1
            if n_grad_samples >= 10:  # sample only 10 batches for the geometry probe -- cheap, sufficient for a trend
                break

        return {
            "val_dice": float(np.mean(dice_scores)),
            "fp_voxels": fp_total, "fn_voxels": fn_total,
            "mean_pred_prob": prob_sum / n_voxels,
            "grad_norm_dice": float(np.mean(grad_norms_dice)),
            "grad_norm_ce": float(np.mean(grad_norms_ce)),
            "grad_cos_theta": float(np.mean(grad_coss)),
        }

    def train(self, epochs):
        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()
            train_loss = self.train_epoch()
            val = self.validate()
            self.scheduler.step()
            epoch_time = time.time() - t0

            print(
                f"[{self.condition_name}] Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
                f"loss={train_loss:.4f} val_dice={val['val_dice']:.4f} "
                f"FP={val['fp_voxels']:.0f} FN={val['fn_voxels']:.0f} "
                f"mean_pred={val['mean_pred_prob']:.4f} "
                f"||g_dice||={val['grad_norm_dice']:.4f} ||g_ce||={val['grad_norm_ce']:.4f} "
                f"cos(theta)={val['grad_cos_theta']:.4f}"
            )
            self.w_metrics.writerow([
                epoch, train_loss, val["val_dice"], val["fp_voxels"], val["fn_voxels"],
                val["mean_pred_prob"], val["grad_norm_dice"], val["grad_norm_ce"],
                val["grad_cos_theta"], epoch_time,
            ])
            self.f_metrics.flush()

        self.f_metrics.close()
        print(f"\n[{self.condition_name}] Mechanism audit complete.")


def main(run_name, seed, use_cc):
    exp = E103Experiment(
        config_path=str(CONFIG_PATH), exp_dir=str(OUT_DIR / "runs"), seed=seed,
        run_name=run_name, use_cc=use_cc, num_workers=None,
    )
    exp.train(epochs=N_EPOCHS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--use_cc", action="store_true")
    args = parser.parse_args()
    main(args.run_name, args.seed, args.use_cc)
