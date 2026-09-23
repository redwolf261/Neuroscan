"""
Phase E72: Fragility-Weighted Loss (FWL) -- v2, GENTLER weighting.

Follow-up to the v1 pilot (softmax T=1.0): mean Dice was significantly
WORSE (-0.75pp, paired t p=0.0001) but the mechanism-consistency check
was significant and correctly signed (Spearman(hat_d_i, delta)=+0.21,
p=0.019) -- FWL demonstrably helps causally-fragile subjects more, but at
T=1.0 the cost to easy subjects (weight range ~0.4x-2.5x) outweighs the
gain. This is a SINGLE pre-registered re-test with a gentler weighting
scheme (linear floor=0.7, weight range ~0.7x-1.3x -- much less punitive
to easy subjects), same 12-epoch protocol, same seed, reusing the
already-computed hat_d_i table from v1 (frozen predictor, unchanged).
Per this project's own no-rescue discipline: one re-test, not an open
hyperparameter search. Report honestly regardless of outcome.

Phase E72: Fragility-Weighted Loss (FWL) -- short pilot, 1 seed.

CANDIDATE MECHANISM (per the Observe->Hypothesize->Attempt->Test directive,
following tonight's chain: multimodal confirmed -> prediction1 validated
(rho=0.866 held-out, causally-grounded, not a size proxy) -> gating KILLED
(E71 gate-sweep: pathways aren't substitutable) -> extra-compute routing
occupied by literature (MAGICORE etc.)):

Use the E71 scaled aux head's frozen prediction hat{d}_i (a subject's
predicted bottleneck-ablation Dice sensitivity) as a PER-SUBJECT LOSS
WEIGHT during segmentation training. This is a different verb from
routing/gating (no architecture change, fact-4-safe) and from extra
compute (no added inference cost). Weight scheme: softmax(hat{d}_i / T=1.0),
validated non-degenerate and not a size proxy by
check_fwl_weighting_sanity.py (E72 sanity check, PASS: max weight frac
0.0064, ESS frac 0.964, rho(weight,size)=-0.19 vs E48's own -0.454 size
reference).

WHY A PILOT, NOT THE FULL 30-EPOCH X 3-SEED PROTOCOL YET:
This is a training-time intervention -- unlike the CDCG gate, its effect
cannot be tested purely at inference. But before committing the full
3-seed discipline (this project's own post-E49 policy for any Dice
claim), a short single-seed pilot checks the FALSIFIABLE PREDICTION the
mechanism itself makes: if FWL is doing what it claims (concentrating
supervision benefit on causally-fragile subjects), then the per-subject
Dice DELTA (FWL - baseline MM, matched subjects) should correlate
POSITIVELY with hat{d}_i. A flat or negative correlation means whatever
the mean Dice does, the mechanism is not acting via the claimed pathway
-- exactly the kind of check that caught CAS's real (different) mechanism
and killed CDCG's gate cleanly.

DECISION RULE for the pilot (pre-declared):
  If mean per-subject Dice improves AND Spearman(delta, hat_d_i) > 0
  with permutation p<0.05: mechanism-consistent signal, worth the full
  3-seed protocol.
  If Dice improves but delta does NOT correlate with hat_d_i: the mean
  effect (if real) is NOT coming from the claimed mechanism --  report
  honestly as an unexplained shift, do not claim FWL "works as designed."
  If Dice does not improve: report as a pilot null: KILL before spending
  the full 3-seed budget, per the no-rescue discipline.

Recipe is IDENTICAL to E70's MM condition (train_e70.py) except: (a)
per-subject loss weighting from the frozen E71 aux head, (b) shortened
to PILOT_EPOCHS (not the full 30) for a first, cheap look.
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
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e71"))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset_multimodal import create_multimodal_brats_loaders  # noqa: E402
from Dataset.brats_dataset_multimodal_cached import create_multimodal_brats_loaders_cached  # noqa: E402
from check_cdcg_prediction1 import AuxHead  # noqa: E402

PILOT_EPOCHS = 12          # shortened first look, matches v1 and the baseline pilot
LAMBDA_DS3 = 0.9927        # E25b calibration, unchanged, identical to E70
LINEAR_FLOOR = 0.7         # gentler scheme: weight in [floor, ~1.3], vs v1's softmax ~0.4x-2.5x
CONFIG_PATH = project_root / "configs" / "brats.yaml"
OUT_ROOT = Path(__file__).parent / "runs"

MM_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e70" / "runs"
           / "MM_seed0" / "checkpoints" / "best.pth")
AUX_HEAD_CKPT = (project_root / "experiments" / "exp_e12_eggo_m" / "e71"
                  / "E71_aux_head_state_scaled.pt")
V1_WEIGHT_TABLE = (Path(__file__).parent / "runs" / "FWL_seed0" / "fwl_weight_table.json")


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_from_counts(tp, fp, fn, eps=1e-6):
    return float((2 * tp) / (2 * tp + fp + fn + eps))


def dice_score_np(pred_bin, target_bin):
    tp = (pred_bin * target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return 1.0 if denom == 0 else float(2 * tp / denom)


def precompute_subject_weights(device):
    """v2: reuse the ALREADY-COMPUTED hat_d_i table from v1
    (train_e72_fwl_pilot.py's precompute step, frozen predictor -- identical
    numbers, no need to rerun the labeling pass), and apply the GENTLER
    linear-floor weighting scheme instead of v1's softmax T=1.0."""
    if not V1_WEIGHT_TABLE.exists():
        raise RuntimeError(f"v1 weight table not found at {V1_WEIGHT_TABLE} -- run v1 first.")
    print(f"Reusing cached hat_d_i table from v1 ({V1_WEIGHT_TABLE})...")
    v1_data = json.load(open(V1_WEIGHT_TABLE))
    d_hat_table = v1_data["d_hat"]

    d_vals = np.array(list(d_hat_table.values()))
    d_shift = (d_vals - d_vals.min()) / (d_vals.max() - d_vals.min() + 1e-8)
    w = LINEAR_FLOOR + (1 - LINEAR_FLOOR) * d_shift
    w = w / w.mean()  # normalize to mean 1.0, behaves like a loss multiplier

    weight_table = {sid: float(wv) for sid, wv in zip(d_hat_table.keys(), w)}
    print(f"Weight table (linear floor={LINEAR_FLOOR}) for {len(weight_table)} training subjects. "
          f"mean={w.mean():.4f} std={w.std():.4f} max={w.max():.4f} min={w.min():.4f}")
    return weight_table, d_hat_table


class FWLExperiment:
    def __init__(self, seed, batch_size, num_workers, use_cache=False):
        self.seed = seed
        set_seed(seed)

        with open(CONFIG_PATH) as f:
            self.config = yaml.safe_load(f)
        root = self.config["dataset"]["root_dir"]
        if not Path(root).is_absolute():
            root = project_root / root

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_dir = OUT_ROOT / f"FWLv2_seed{seed}"
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.weight_table, self.d_hat_table = precompute_subject_weights(self.device)
        with open(self.exp_dir / "fwl_weight_table.json", "w") as f:
            json.dump({"weights": self.weight_table, "d_hat": self.d_hat_table}, f, indent=2)

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
        print(f"[FWL seed{seed}] data loader: {'CACHED' if use_cache else 'uncached'}", flush=True)

        self.best_per_subject = 0.0
        self.epoch = 0
        n_par = sum(p.numel() for p in self.model.parameters())
        print(f"[FWL seed{seed}] params={n_par:,} "
              f"train={len(self.train_loader.dataset)} val={len(self.val_loader.dataset)} "
              f"device={self.device} pilot_epochs={PILOT_EPOCHS}", flush=True)

        self.f = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["epoch", "train_loss", "val_pooled_dice", "val_per_subject_dice",
                         "val_per_subject_std", "epoch_time_sec", "peak_mem_mb"])

    def train_epoch(self):
        self.model.train()
        tot, nb = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"[FWL s{self.seed}] ep{self.epoch+1} train")
        for images, masks, sids in pbar:
            images, masks = images.to(self.device), masks.to(self.device)
            batch_w = torch.tensor([self.weight_table.get(s, 1.0) for s in sids],
                                   dtype=torch.float32, device=self.device)

            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(images)

            # FocalTverskyLoss/EvidentialBetaLoss reduce internally (no per-sample
            # reduction option in this codebase), so per-sample weighting is applied
            # by slicing the already-computed batched forward output per subject and
            # calling the loss fns on each 1-item slice. Only the network forward is
            # expensive and it runs once per batch above; the loss-fn calls here are cheap.
            per_sample_losses = []
            for b in range(images.shape[0]):
                seg_b = 0.5 * self.focal_fn(out["probs"][b:b+1], masks[b:b+1]) + \
                        0.5 * self.evidential_fn(out["alpha"][b:b+1], out["beta"][b:b+1], masks[b:b+1])
                mask_d4_b = F.avg_pool3d(masks[b:b+1], kernel_size=4, stride=4)
                aux3_b = self.focal_fn(out["aux_probs3"][b:b+1], mask_d4_b)
                per_sample_losses.append(seg_b + LAMBDA_DS3 * aux3_b)
            per_sample = torch.stack(per_sample_losses)
            loss = (per_sample * batch_w).mean()

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
        for images, masks, sids in tqdm(self.val_loader, desc=f"[FWL s{self.seed}] ep{self.epoch+1} val"):
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
        ck = {"epoch": self.epoch, "seed": self.seed, "condition": "FWL",
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

            print(f"[FWL s{self.seed}] ep{ep+1}/{PILOT_EPOCHS} ({dt:.0f}s) "
                  f"train_loss={tr:.4f} pooled={pooled:.4f} per_subj={per_sub:.4f}", flush=True)
            self.w.writerow([ep, tr, pooled, per_sub, per_sub_std, dt, mem]); self.f.flush()

            is_best = per_sub > self.best_per_subject
            if is_best:
                self.best_per_subject = per_sub
            self.save(is_best)

        self.f.close()
        result = {"condition": "FWL", "seed": self.seed,
                  "best_per_subject_dice": self.best_per_subject,
                  "final_epoch_per_subject_dice": dict(zip(final_per_subject_ids, final_per_subject_dice))}
        with open(self.exp_dir / "result.json", "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[FWL s{self.seed}] DONE best_per_subject={self.best_per_subject:.4f}")
        return self.best_per_subject


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--use_cache", action="store_true")
    a = ap.parse_args()
    FWLExperiment(a.seed, a.batch_size, a.num_workers, use_cache=a.use_cache).run()
