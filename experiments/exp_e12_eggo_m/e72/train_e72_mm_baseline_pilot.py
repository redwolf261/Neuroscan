"""
Phase E72: matched-epoch, matched-everything MM baseline pilot (unweighted
loss), 12 epochs, seed 0 -- for an HONEST per-subject comparison against
FWL_seed0 (train_e72_fwl_pilot.py). E70's own MM_seed0 run's best.pth is
from epoch 26 (full 30-epoch run, checkpoint overwritten as training
improved), so it cannot serve as a matched-epoch comparison point -- only
its epoch_metrics.csv aggregate row for epoch index 11 exists, with no
saved per-subject breakdown. This script reproduces that exact epoch
count under E70's identical recipe, with per-subject Dice + IDs recorded
at the final epoch, enabling the mechanism-consistency check (per-subject
delta vs hat_d_i) that the FWL pilot was designed to test.

IDENTICAL to E70's MM condition (model, loss, optimizer, schedule/seed
math) except: (a) stopped at 12 epochs (not 30) to match the FWL pilot,
(b) records final-epoch per-subject Dice + subject IDs, matching the FWL
pilot's own logging so the two are directly comparable subject-for-subject.
"""
import sys
import csv
import json
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

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset_multimodal import create_multimodal_brats_loaders  # noqa: E402
from Dataset.brats_dataset_multimodal_cached import create_multimodal_brats_loaders_cached  # noqa: E402

PILOT_EPOCHS = 12       # matches train_e72_fwl_pilot.py exactly
LAMBDA_DS3 = 0.9927     # E25b calibration, unchanged, identical to E70
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


class MMBaselinePilot:
    def __init__(self, seed, batch_size, num_workers, use_cache=False):
        self.seed = seed
        set_seed(seed)

        with open(CONFIG_PATH) as f:
            self.config = yaml.safe_load(f)
        root = self.config["dataset"]["root_dir"]
        if not Path(root).is_absolute():
            root = project_root / root

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_dir = OUT_ROOT / f"MM_baseline_pilot_seed{seed}"
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model = UNet3D_v3(in_channels=4, out_channels=1).to(self.device)
        self.focal_fn = FocalTverskyLoss()
        self.evidential_fn = EvidentialBetaLoss(weight=0.5)

        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.config["training"]["learning_rate"],
                               weight_decay=self.config["training"]["weight_decay"])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=PILOT_EPOCHS, eta_min=1e-6)

        loader_fn = create_multimodal_brats_loaders_cached if use_cache else create_multimodal_brats_loaders
        self.train_loader, self.val_loader = loader_fn(
            batch_size=batch_size, num_workers=num_workers,
            root_dir=str(root), val_split=self.config["dataset"]["val_split"],
        )
        print(f"[MM_baseline_pilot seed{seed}] loader: {'CACHED' if use_cache else 'uncached'}", flush=True)

        self.best_per_subject = 0.0
        self.epoch = 0
        n_par = sum(p.numel() for p in self.model.parameters())
        print(f"[MM_baseline_pilot seed{seed}] params={n_par:,} "
              f"train={len(self.train_loader.dataset)} val={len(self.val_loader.dataset)} "
              f"device={self.device} pilot_epochs={PILOT_EPOCHS}", flush=True)

        self.f = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["epoch", "train_loss", "val_pooled_dice", "val_per_subject_dice",
                         "val_per_subject_std", "epoch_time_sec", "peak_mem_mb"])

    def train_epoch(self):
        self.model.train()
        tot, nb = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"[MM_base s{self.seed}] ep{self.epoch+1} train")
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
    def validate(self, record_per_subject_ids=False):
        self.model.eval()
        TP = FP = FN = 0.0
        per_subject = []
        per_subject_ids = []
        for images, masks, sids in tqdm(self.val_loader, desc=f"[MM_base s{self.seed}] ep{self.epoch+1} val"):
            images, masks = images.to(self.device), masks.to(self.device)
            probs = self.model(images)["probs"]
            pred = (probs >= 0.5).float()
            for b in range(pred.shape[0]):
                tp = (pred[b] * masks[b]).sum().item()
                fp = (pred[b] * (1 - masks[b])).sum().item()
                fn = ((1 - pred[b]) * masks[b]).sum().item()
                TP += tp; FP += fp; FN += fn
                per_subject.append(dice_from_counts(tp, fp, fn))
                per_subject_ids.append(sids[b])
        if record_per_subject_ids:
            return (dice_from_counts(TP, FP, FN), float(np.mean(per_subject)),
                    float(np.std(per_subject)), per_subject, per_subject_ids)
        return dice_from_counts(TP, FP, FN), float(np.mean(per_subject)), float(np.std(per_subject))

    def save(self, is_best):
        ck = {"epoch": self.epoch, "seed": self.seed, "condition": "MM_baseline_pilot",
              "model_state": self.model.state_dict(),
              "best_per_subject_dice": self.best_per_subject,
              "in_channels": 4, "modality_order": ["T1", "T1ce", "T2", "FLAIR"]}
        if is_best:
            torch.save(ck, self.ckpt_dir / "best.pth")

    def run(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        final_per_subject_dice = None
        final_per_subject_ids = None
        for ep in range(PILOT_EPOCHS):
            self.epoch = ep
            t0 = time.time()
            tr = self.train_epoch()
            is_last = (ep == PILOT_EPOCHS - 1)
            if is_last:
                pooled, per_sub, per_sub_std, per_subject_list, per_subject_ids = self.validate(record_per_subject_ids=True)
                final_per_subject_dice = per_subject_list
                final_per_subject_ids = per_subject_ids
            else:
                pooled, per_sub, per_sub_std = self.validate()
            self.scheduler.step()
            dt = time.time() - t0
            mem = (torch.cuda.max_memory_allocated() / 1e6) if self.device.type == "cuda" else 0.0

            print(f"[MM_base s{self.seed}] ep{ep+1}/{PILOT_EPOCHS} ({dt:.0f}s) "
                  f"train_loss={tr:.4f} pooled={pooled:.4f} per_subj={per_sub:.4f}", flush=True)
            self.w.writerow([ep, tr, pooled, per_sub, per_sub_std, dt, mem]); self.f.flush()

            is_best = per_sub > self.best_per_subject
            if is_best:
                self.best_per_subject = per_sub
            self.save(is_best)

        self.f.close()
        result = {"condition": "MM_baseline_pilot", "seed": self.seed,
                  "best_per_subject_dice": self.best_per_subject,
                  "final_epoch_per_subject_dice": dict(zip(final_per_subject_ids, final_per_subject_dice))}
        with open(self.exp_dir / "result.json", "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[MM_base s{self.seed}] DONE best_per_subject={self.best_per_subject:.4f}")
        return self.best_per_subject


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--use_cache", action="store_true")
    a = ap.parse_args()
    MMBaselinePilot(a.seed, a.batch_size, a.num_workers, use_cache=a.use_cache).run()
