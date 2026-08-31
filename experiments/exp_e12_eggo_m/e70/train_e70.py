"""
Phase E70: Multimodal baseline vs. Multimodal + CAS.

Two conditions, IDENTICAL in every respect except the single variable
under test:

  MM      : UNet3D_v3, in_channels=4  (T1, T1ce, T2, FLAIR)
  MM_CAS  : UNet3D_v10, in_channels=4 (= MM + CorrespondenceAwareSkip at enc1)

Everything else is held fixed and inherited from this project's own
established recipe (e25/train_deep_sup.py, the script that produced the
DeepSup_D4only checkpoints used throughout E58-E69):
  AdamW, CosineAnnealingLR(T_max=EPOCHS, eta_min=1e-6), lr/weight_decay
  from configs/brats.yaml, grad-clip 1.0, 30 epochs, D4-only deep
  supervision (lambda_ds3=0.9927 from E25b's own calibration, D2 off),
  seg_loss = 0.5*FocalTversky + 0.5*EvidentialBeta, boundary term off.

CAS is the identity at initialization (verified bit-for-bit against v3),
so MM_CAS starts from exactly MM's function and must LEARN any deviation
-- this is what makes the ablation clean: a difference in outcome cannot
be attributed to a different initialization or to added capacity acting
as a better random feature extractor.

Evaluation follows E56's corrected protocol: BOTH pooled Dice and
per-subject-mean Dice are recorded every epoch, and best.pth is selected
on per-subject-mean (the metric E56 established as the appropriate
primary endpoint for this population).
"""
import sys
import csv
import json
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

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_v10 import UNet3D_v10  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset_multimodal import create_multimodal_brats_loaders  # noqa: E402

EPOCHS = 30
LAMBDA_DS3 = 0.9927          # E25b calibration, unchanged
CONFIG_PATH = project_root / "configs" / "brats.yaml"
OUT_ROOT = Path(__file__).parent / "runs"


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_from_counts(tp, fp, fn, eps=1e-6):
    return float((2 * tp) / (2 * tp + fp + fn + eps))


class Experiment:
    def __init__(self, condition, seed, batch_size, num_workers):
        assert condition in ("MM", "MM_CAS")
        self.condition = condition
        self.seed = seed
        set_seed(seed)

        with open(CONFIG_PATH) as f:
            self.config = yaml.safe_load(f)
        root = self.config["dataset"]["root_dir"]
        if not Path(root).is_absolute():
            root = project_root / root

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_dir = OUT_ROOT / f"{condition}_seed{seed}"
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        cls = UNet3D_v3 if condition == "MM" else UNet3D_v10
        self.model = cls(in_channels=4, out_channels=1).to(self.device)

        self.focal_fn = FocalTverskyLoss()
        self.evidential_fn = EvidentialBetaLoss(weight=0.5)

        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.config["training"]["learning_rate"],
                               weight_decay=self.config["training"]["weight_decay"])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=EPOCHS, eta_min=1e-6)

        self.train_loader, self.val_loader = create_multimodal_brats_loaders(
            batch_size=batch_size, num_workers=num_workers,
            root_dir=str(root), val_split=self.config["dataset"]["val_split"],
        )

        self.best_per_subject = 0.0
        self.epoch = 0
        n_par = sum(p.numel() for p in self.model.parameters())
        print(f"[{condition} seed{seed}] params={n_par:,} "
              f"train={len(self.train_loader.dataset)} val={len(self.val_loader.dataset)} "
              f"device={self.device}", flush=True)

        self.f = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["epoch", "train_loss", "val_pooled_dice", "val_per_subject_dice",
                         "val_per_subject_std", "epoch_time_sec", "peak_mem_mb"])

    def train_epoch(self):
        self.model.train()
        tot, nb = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"[{self.condition} s{self.seed}] ep{self.epoch+1} train")
        for images, masks, _ in pbar:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(images)

            seg = 0.5 * self.focal_fn(out["probs"], masks) + \
                  0.5 * self.evidential_fn(out["alpha"], out["beta"], masks)
            mask_d4 = F.avg_pool3d(masks, kernel_size=4, stride=4)
            aux3 = self.focal_fn(out["aux_probs3"], mask_d4)
            loss = seg + LAMBDA_DS3 * aux3

            if not torch.isfinite(loss):
                raise RuntimeError(f"NaN/Inf loss at epoch {self.epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            tot += loss.item(); nb += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        return tot / max(1, nb)

    @torch.no_grad()
    def validate(self):
        """Records BOTH endpoints per E56's corrected protocol."""
        self.model.eval()
        TP = FP = FN = 0.0
        per_subject = []
        for images, masks, _ in tqdm(self.val_loader, desc=f"[{self.condition} s{self.seed}] ep{self.epoch+1} val"):
            images, masks = images.to(self.device), masks.to(self.device)
            probs = self.model(images)["probs"]
            pred = (probs >= 0.5).float()
            for b in range(pred.shape[0]):
                tp = (pred[b] * masks[b]).sum().item()
                fp = (pred[b] * (1 - masks[b])).sum().item()
                fn = ((1 - pred[b]) * masks[b]).sum().item()
                TP += tp; FP += fp; FN += fn
                per_subject.append(dice_from_counts(tp, fp, fn))
        return dice_from_counts(TP, FP, FN), float(np.mean(per_subject)), float(np.std(per_subject))

    def save(self, is_best):
        ck = {"epoch": self.epoch, "seed": self.seed, "condition": self.condition,
              "model_state": self.model.state_dict(),
              "best_per_subject_dice": self.best_per_subject,
              "in_channels": 4, "modality_order": ["T1", "T1ce", "T2", "FLAIR"]}
        if is_best:
            torch.save(ck, self.ckpt_dir / "best.pth")

    def run(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for ep in range(EPOCHS):
            self.epoch = ep
            t0 = time.time()
            tr = self.train_epoch()
            pooled, per_sub, per_sub_std = self.validate()
            self.scheduler.step()
            dt = time.time() - t0
            mem = (torch.cuda.max_memory_allocated() / 1e6) if self.device.type == "cuda" else 0.0

            print(f"[{self.condition} s{self.seed}] ep{ep+1}/{EPOCHS} ({dt:.0f}s) "
                  f"train_loss={tr:.4f} pooled={pooled:.4f} per_subj={per_sub:.4f}", flush=True)
            self.w.writerow([ep, tr, pooled, per_sub, per_sub_std, dt, mem]); self.f.flush()

            is_best = per_sub > self.best_per_subject
            if is_best:
                self.best_per_subject = per_sub
            self.save(is_best)

        self.f.close()
        result = {"condition": self.condition, "seed": self.seed,
                  "best_per_subject_dice": self.best_per_subject}
        with open(self.exp_dir / "result.json", "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[{self.condition} s{self.seed}] DONE best_per_subject={self.best_per_subject:.4f}")
        return self.best_per_subject


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["MM", "MM_CAS"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    a = ap.parse_args()
    Experiment(a.condition, a.seed, a.batch_size, a.num_workers).run()
