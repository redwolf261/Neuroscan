"""
Phase E131: RANK-ADAPTIVE POOLING (v13) vs the E130 baseline.

Same script as E130 with one flag: --arch {v5,v13}. v13 replaces pool3 with
RankAdaptivePool3d, which blends max-pool toward a learned depthwise
aggregation in windows whose participation ratio (effective rank) is high --
the statistic E124-E126 causally validated. v13 is bit-identical to v5 at
lam=0 and near-identical at the lam=0.05 default, so any Dice difference is
attributable to the mechanism, not to initialisation.

ORIGINAL E130 HEADER FOLLOWS.

Phase E130: UPGRADED BASELINE -- 4-modality, 3-region, 128^3 patch-based
BraTS 2023 GLI training.

WHY: the previous baseline was FLAIR-only, binary tumour-vs-background, and
resized to 64^3. That configuration cannot express the paper blueprint's
WT/TC/ET breakdown, cannot compute lesion-wise Dice, and is not comparable
to any published BraTS result. This phase upgrades the task itself, not the
architecture.

WHAT CHANGED (task), and what deliberately did NOT (model):
  CHANGED  input     1ch FLAIR            -> 4ch (t1c, t1n, t2f, t2w)
  CHANGED  target    binary tumour        -> 3 overlapping regions ET/TC/WT
  CHANGED  sampling  whole volume -> 64^3 -> 128^3 patches at native 1mm
  CHANGED  normalize min-max whole volume -> per-modality z-score, brain only
  CHANGED  val       resized 64^3 volume  -> full 240x240x155, sliding window
  UNCHANGED model    UNet3D_v5, same depth/width/attention gate/aux heads.
    Verified before writing this: UNet3D_v5(in_channels=4, out_channels=3)
    already produces correct shapes for every output including the aux
    heads -- out_channels propagates to seg_head, evidential_head,
    boundary_head, aux_head3 and aux_head2. No new architecture file is
    needed, which is deliberate: keeping the model fixed means the
    E48-E129 causal machinery transfers directly and the re-measurement is
    like-for-like.

SPLIT: the new loader reproduces BraTSDataset's split rule EXACTLY
(sorted glob, RandomState(42) shuffle, last n_val held out). Verified by
comparing subject-ID lists -- they match. A first draft used sorted order
and took the FIRST n_val, which overlapped only 11/125 and would have
silently broken comparability with every prior measurement.

LOSS: the old protocol's FocalTversky + Evidential + boundary + D4 aux is
retained in FORM, but every term is now computed per-region and averaged
over the 3 regions. Region weighting is uniform -- deliberately, since
this is a BASELINE and the point is to establish an honest reference, not
to tune. Any region weighting belongs to a later intervention, not here.

VALIDATION: sliding window over the full native volume with 50% overlap,
Gaussian-weighted blending. Metrics reported PER REGION (ET, TC, WT) and
as their mean, plus HD95 per region. This is the standard BraTS reporting
format and is what makes the resulting table comparable to published work.

AMP: enabled, matching E128's finding that AMP alone is worth +0.110pp --
every arm from here on is AMP so precision is held constant across all
future comparisons. Hardware verified before committing: 128^3 at batch 2
peaks at 4.62GB of 8.5GB, ~0.50 s/iter, ~284s/epoch at 563 iters.

BATCH SIZE / VRAM CLIFF (measured, not assumed): 128^3 at batch 2 peaks at
5.47GB on this 8.1GB card and runs at 2.016 s/iter drawing only 24.9W --
the allocator is thrashing, not computing. At batch 1 the SAME patch size
runs at 0.279 s/iter drawing 54.1W: a 7.2x speedup from HALVING the batch.
Every config under ~4.7GB draws 52-58W; the broken one draws 24.9W. So
E130 runs batch 1 with gradient accumulation.

CAVEAT, STATED PLAINLY: gradient accumulation restores the effective
OPTIMIZER batch size, but v5 uses BatchNorm3d and BatchNorm still sees only
ONE sample per forward pass. Accumulation stabilises the gradient estimate;
it does NOT fix batch-1 normalization statistics. The principled fix is
GroupNorm (what SegResNet/MedNeXt/the 2021 BraTS winner use for small
batches), but that changes the architecture and would force the E48-E129
causal chain to be re-derived rather than re-measured. Watch epoch-1 val
Dice against the batch-2 run's 0.808 for evidence of whether batch-1
BatchNorm is actually hurting.

NO NOVELTY CLAIM. This is an engineering baseline, Phase 1 of the paper
blueprint. Its purpose is to be a strong, honest reference that later
interventions are measured against.
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

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from neuroscan_3d_v13 import UNet3D_v13, UNet3D_v14  # noqa: E402
from neuroscan_3d_v15 import UNet3D_v15  # noqa: E402
from neuroscan_3d_v16 import UNet3D_v16  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders, REGIONS  # noqa: E402

OUT_DIR = Path(__file__).parent
SEED = 0
MU = 0.1                 # boundary loss weight, unchanged from E46/E128
LAMBDA_DS3 = 0.9927      # D4 deep-supervision weight, unchanged
PATCH = (128, 128, 128)
SW_OVERLAP = 0.5


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- losses
def focal_tversky(probs, target, alpha=0.5, beta=0.5, gamma=4.0 / 3.0, eps=1e-6):
    """Per-region FocalTversky, averaged over regions. Same functional form
    as the old protocol's FocalTverskyLoss, generalized to multi-region."""
    dims = tuple(range(2, probs.dim()))
    tp = (probs * target).sum(dims)
    fp = (probs * (1 - target)).sum(dims)
    fn = ((1 - probs) * target).sum(dims)
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1 - ti) ** gamma).mean()


def evidential_beta(alpha, beta, target, weight=0.5, eps=1e-6):
    """Beta-evidential NLL, per-region, averaged. Mirrors the old
    EvidentialBetaLoss in form (expected value + variance penalty)."""
    s = alpha + beta
    p = alpha / (s + eps)
    nll = (target - p) ** 2 + p * (1 - p) / (s + 1.0)
    return weight * nll.mean()


# --------------------------------------------------------------- metrics
def dice_per_region(pred_bin, target_bin):
    """pred_bin/target_bin: (R,D,H,W) numpy. Returns list of R Dice values.
    Empty-target convention: Dice=1.0 if prediction is also empty, else 0.0
    -- this matters for ET, which is genuinely absent in some subjects, and
    silently scoring those 1.0 regardless of prediction would inflate the
    headline number."""
    out = []
    for r in range(pred_bin.shape[0]):
        p, t = pred_bin[r], target_bin[r]
        ps, ts = p.sum(), t.sum()
        if ts == 0:
            out.append(1.0 if ps == 0 else 0.0)
        else:
            out.append(float(2.0 * (p * t).sum() / (ps + ts)))
    return out


def hd95_per_region(pred_bin, target_bin, spacing=1.0):
    """95th-percentile Hausdorff via scipy EDT. Returns nan where either
    set is empty (undefined), so nan-aware aggregation is required."""
    from scipy.ndimage import distance_transform_edt
    out = []
    for r in range(pred_bin.shape[0]):
        p, t = pred_bin[r].astype(bool), target_bin[r].astype(bool)
        if not p.any() or not t.any():
            out.append(float("nan"))
            continue
        dt_t = distance_transform_edt(~t) * spacing
        dt_p = distance_transform_edt(~p) * spacing
        pb = p & ~_erode(p)
        tb = t & ~_erode(t)
        d1 = dt_t[pb] if pb.any() else np.array([0.0])
        d2 = dt_p[tb] if tb.any() else np.array([0.0])
        out.append(float(max(np.percentile(d1, 95), np.percentile(d2, 95))))
    return out


def _erode(m):
    from scipy.ndimage import binary_erosion
    return binary_erosion(m, iterations=1, border_value=0)


# ------------------------------------------------- sliding-window inference
def _gaussian_weight(shape, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in shape]
    g = np.ones(shape, dtype=np.float32)
    for i, c in enumerate(coords):
        sh = [1] * len(shape)
        sh[i] = -1
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(sh).astype(np.float32)
    return np.maximum(g, 1e-4)


def sliding_window_predict(model, image, patch, overlap, n_out, device, amp=True):
    """image: (1,C,D,H,W) tensor on device. Returns (n_out,D,H,W) probs."""
    _, _, D, H, W = image.shape
    pd, ph, pw = patch
    stride = [max(1, int(p * (1 - overlap))) for p in patch]

    def starts(full, p, st):
        if full <= p:
            return [0]
        s = list(range(0, full - p + 1, st))
        if s[-1] != full - p:
            s.append(full - p)
        return s

    zs, ys, xs = (starts(D, pd, stride[0]), starts(H, ph, stride[1]), starts(W, pw, stride[2]))
    acc = torch.zeros((n_out, D, H, W), device=device, dtype=torch.float32)
    wsum = torch.zeros((1, D, H, W), device=device, dtype=torch.float32)
    gw = torch.from_numpy(_gaussian_weight((min(pd, D), min(ph, H), min(pw, W)))).to(device)

    with torch.no_grad():
        for z in zs:
            for y in ys:
                for x in xs:
                    zc, yc, xc = min(pd, D), min(ph, H), min(pw, W)
                    tile = image[:, :, z:z + zc, y:y + yc, x:x + xc]
                    if tile.shape[2:] != (pd, ph, pw):
                        tile = F.pad(tile, (0, pw - tile.shape[4], 0, ph - tile.shape[3], 0, pd - tile.shape[2]))
                    with torch.amp.autocast("cuda", enabled=amp):
                        out = model(tile)
                    pr = out["probs"].float()[:, :, :zc, :yc, :xc].squeeze(0)
                    acc[:, z:z + zc, y:y + yc, x:x + xc] += pr * gw
                    wsum[:, z:z + zc, y:y + yc, x:x + xc] += gw
    return (acc / wsum.clamp(min=1e-6)).cpu().numpy()


# ------------------------------------------------------------------ train
class E130Experiment:
    def __init__(self, run_name, epochs, lr, wd, batch_size, num_workers, amp, seed, fast_val,
                 accum_steps=1, bn_momentum=0.01, arch='v5', lam_init=0.05):
        self.run_name = run_name
        self.epochs = epochs
        self.amp = amp
        self.seed = seed
        self.fast_val = fast_val
        self.accum_steps = max(1, int(accum_steps))
        self.bn_momentum = bn_momentum
        self.arch = arch
        set_seed(seed)

        self.exp_dir = OUT_DIR / "runs" / run_name
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if arch == "v16":
            self.model = UNet3D_v16(in_channels=4, out_channels=3).to(self.device)
        elif arch == "v15":
            self.model = UNet3D_v15(in_channels=4, out_channels=3).to(self.device)
        elif arch == "v14":
            self.model = UNet3D_v14(in_channels=4, out_channels=3).to(self.device)
        elif arch == "v13":
            self.model = UNet3D_v13(in_channels=4, out_channels=3,
                                    lam_init=lam_init).to(self.device)
        else:
            self.model = UNet3D_v5(in_channels=4, out_channels=3).to(self.device)

        # BATCHNORM MOMENTUM FIX -- the E130 seed-0 collapse at epoch 39 was
        # DIAGNOSED, not guessed: perturbing ONLY BatchNorm running_var on the
        # healthy epoch-32 checkpoint (no weight change at all) dropped Dice
        # from 0.93 to EXACTLY 0.00. That reproduces the observed signature --
        # train_loss barely moved (0.126->0.150, because training uses BATCH
        # statistics) while val TC/WT collapsed (0.91->0.47, 0.89->0.34,
        # because validation uses RUNNING statistics). At batch 1 a single
        # anomalous patch (e.g. near-pure background, tiny variance) poisons
        # the running estimate through the momentum update.
        # Note the AMP hypothesis was REJECTED by evidence: the LR at collapse
        # was 5.5e-05, the regime where training is most stable, not least.
        # running_var spans 0.154 to 7303 across the 14 BN layers, so at the
        # default momentum=0.1 one batch moves it ~10%. At 0.01 it moves ~1%,
        # so no single patch can poison it.
        n_bn = 0
        for mod in self.model.modules():
            if isinstance(mod, nn.BatchNorm3d):
                mod.momentum = bn_momentum
                n_bn += 1
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp)
        self.boundary_criterion = nn.BCEWithLogitsLoss()

        self.train_loader, self.val_loader = create_multimodal_loaders(
            root_dir=str(project_root / "Dataset" / "Training"),
            batch_size=batch_size, num_workers=num_workers,
            val_split=0.1, patch_size=PATCH, seed=seed)

        self.best_mean_dice = 0.0
        self.epoch = 0
        n_par = sum(p.numel() for p in self.model.parameters())
        print(f"[{run_name}] UNet3D_v5 4-in/3-out, {n_par:,} params, amp={amp}, seed={seed}, "
              f"bn_momentum={bn_momentum} on {n_bn} BatchNorm3d layers, arch={arch}")
        print(f"[{run_name}] train={len(self.train_loader.dataset)} val={len(self.val_loader.dataset)} "
              f"patch={PATCH} bs={batch_size} accum={self.accum_steps} "
              f"(effective batch {batch_size*self.accum_steps}) iters/epoch={len(self.train_loader)}")

        self.f = open(self.exp_dir / "epoch_metrics.csv", "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["epoch", "train_loss", "train_dice",
                         "val_dice_ET", "val_dice_TC", "val_dice_WT", "val_dice_mean",
                         "val_hd95_ET", "val_hd95_TC", "val_hd95_WT",
                         "epoch_time_sec", "peak_gpu_mb"])

    def train_epoch(self):
        self.model.train()
        tot, dsum, n = 0.0, 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"[{self.run_name}] ep{self.epoch+1} train")
        self.optimizer.zero_grad(set_to_none=True)
        for it_idx, (images, targets, _) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.amp):
                out = self.model(images)
                probs, alpha, beta = out["probs"], out["alpha"], out["beta"]
                seg = 0.5 * focal_tversky(probs, targets) + 0.5 * evidential_beta(alpha, beta, targets)
                bnd = self.boundary_criterion(out["boundary_logit"], targets)
                t_d4 = F.avg_pool3d(targets, kernel_size=4, stride=4)
                aux3 = focal_tversky(out["aux_probs3"], t_d4)
                loss = seg + MU * bnd + LAMBDA_DS3 * aux3

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {self.epoch}")

            # GRADIENT ACCUMULATION. Divide by accum_steps so the accumulated
            # gradient equals the mean over the effective batch, not the sum
            # -- otherwise the effective LR would scale with accum_steps.
            self.scaler.scale(loss / self.accum_steps).backward()
            step_now = ((it_idx + 1) % self.accum_steps == 0) or (it_idx + 1 == len(self.train_loader))
            if step_now:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                pb = (probs.float() >= 0.5).float()
                inter = (pb * targets).sum((2, 3, 4))
                den = pb.sum((2, 3, 4)) + targets.sum((2, 3, 4))
                d = torch.where(den > 0, 2 * inter / den.clamp(min=1e-6), torch.ones_like(den)).mean()
            tot += loss.item(); dsum += d.item(); n += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{d.item():.4f}"})
        return tot / max(1, n), dsum / max(1, n)

    def validate(self):
        self.model.eval()
        n_sub = len(self.val_loader.dataset)
        limit = 25 if self.fast_val else n_sub
        dices, hds = [], []
        pbar = tqdm(self.val_loader, total=limit, desc=f"[{self.run_name}] ep{self.epoch+1} val")
        for i, (image, target, _) in enumerate(pbar):
            if i >= limit:
                break
            image = image.to(self.device, non_blocking=True)
            probs = sliding_window_predict(self.model, image, PATCH, SW_OVERLAP, 3, self.device, self.amp)
            pb = (probs >= 0.5).astype(np.float32)
            tb = target.squeeze(0).numpy()
            dices.append(dice_per_region(pb, tb))
            # HD95 is expensive (3 EDTs over 240^3); compute only on the
            # final epoch unless fast_val, to keep per-epoch cost sane.
            if self.epoch + 1 == self.epochs:
                hds.append(hd95_per_region(pb, tb))
        dices = np.array(dices)
        hd = np.array(hds) if hds else np.full((1, 3), np.nan)
        return dices.mean(0), np.nanmean(hd, 0), len(dices)

    def run(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for ep in range(self.epochs):
            self.epoch = ep
            t0 = time.time()
            tl, td = self.train_epoch()
            vd, vh, nval = self.validate()
            self.scheduler.step()
            dt = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1e6 if self.device.type == "cuda" else 0.0
            mean_d = float(np.mean(vd))
            gate = ""
            if self.arch == "v16":
                p3 = None
                try:
                    fw = self.model.film[-1].weight
                    gate = (f" | film_w_std={float(fw.std()):.5f} "
                            f"film_b_std={float(self.model.film[-1].bias.std()):.5f}")
                except Exception:
                    gate = ""
            elif self.arch in ("v13", "v14"):
                p3 = self.model.pool3
                lam = f"lam={float(p3.lam):+.4f} " if hasattr(p3, "lam") else ""
                gate = (f" | {lam}tau={float(p3.tau):.3f} "
                        f"sharp={float(F.softplus(p3.sharp_raw)):.2f} "
                        f"aggstd={float(p3.agg.weight.std()):.4f}")
            print(f"[{self.run_name}] ep{ep+1}/{self.epochs} ({dt:.0f}s, {mem:.0f}MB) "
                  f"train loss={tl:.4f} dice={td:.4f} | val ET={vd[0]:.4f} TC={vd[1]:.4f} "
                  f"WT={vd[2]:.4f} mean={mean_d:.4f} (n={nval}){gate}", flush=True)
            self.w.writerow([ep, tl, td, vd[0], vd[1], vd[2], mean_d, vh[0], vh[1], vh[2], dt, mem])
            self.f.flush()

            # PERIODIC CHECKPOINTS: the seed-0 collapse could not be fully
            # diagnosed because ONLY best.pth existed -- the collapsed weights
            # were never saved. Keep one every 5 epochs (and the last) so any
            # future instability is inspectable rather than lost.
            if ((ep + 1) % 5 == 0) or (ep + 1 == self.epochs):
                torch.save({"epoch": ep, "seed": self.seed,
                            "model_state": self.model.state_dict(),
                            "val_dice_per_region": {r: float(v) for r, v in zip(REGIONS, vd)},
                            "mean_dice": mean_d, "arch": self.arch, "amp": self.amp,
                            "bn_momentum": self.bn_momentum, "patch": PATCH},
                           self.ckpt_dir / f"epoch_{ep+1:03d}.pth")

            if mean_d > self.best_mean_dice:
                self.best_mean_dice = mean_d
                torch.save({"epoch": ep, "seed": self.seed, "model_state": self.model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "best_mean_dice": self.best_mean_dice,
                            "val_dice_per_region": {r: float(v) for r, v in zip(REGIONS, vd)},
                            "arch": self.arch, "amp": self.amp, "patch": PATCH,
                            "bn_momentum": self.bn_momentum,
                            "task": "brats2023gli_4mod_3region"},
                           self.ckpt_dir / "best.pth")
        self.f.close()
        print(f"\n[{self.run_name}] DONE. best mean Dice = {self.best_mean_dice:.4f}")
        return self.best_mean_dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--run_name", type=str, default="E131_v13_seed0")
    ap.add_argument("--arch", type=str, default="v13", choices=["v5","v13","v14","v15","v16"])
    ap.add_argument("--lam_init", type=float, default=0.05)
    ap.add_argument("--bn_momentum", type=float, default=0.01,
                    help="BatchNorm momentum; 0.01 guards batch-1 running-stat poisoning")
    ap.add_argument("--accum_steps", type=int, default=1,
                    help="optimizer steps every N batches; restores effective batch size at bs=1")
    ap.add_argument("--fast_val", type=int, default=1,
                    help="validate on 25 subjects per epoch instead of 125 (final epoch still full)")
    a = ap.parse_args()
    exp = E130Experiment(a.run_name, a.epochs, a.lr, a.wd, a.batch_size,
                         a.num_workers, bool(a.amp), a.seed, bool(a.fast_val),
                         accum_steps=a.accum_steps, bn_momentum=a.bn_momentum,
                         arch=a.arch, lam_init=a.lam_init)
    best = exp.run()
    json.dump({"run_name": a.run_name, "best_mean_dice": best, "epochs": a.epochs,
               "patch": list(PATCH), "seed": a.seed},
              open(OUT_DIR / f"E130_{a.run_name}_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
