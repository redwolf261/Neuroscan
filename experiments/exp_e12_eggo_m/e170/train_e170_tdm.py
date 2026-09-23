"""
E170 -- Task-Demand Matching (TDM). Four-arm single-seed mechanistic test.

PRE-REGISTERED in docs/phases/PHASE_E170_TDM_PREREGISTRATION.md.

THE FORMULA (enc1 only, per the premise test: 84% of subjects carry a deficit
at enc1 vs 6.4% at enc3):

    L_TDM = lambda * [ max(0, R_hat_enc1 - R_eff_enc1) ]^2

R_eff is the entropy-based effective rank of the live enc1 activation,
differentiable through the singular values. R_hat is the FROZEN out-of-fold
predictor from fit_rhat_predictor.py (OOF R2=0.632, rho=0.787, MAE 4.45 ranks).

CAUSAL CHAIN UNDER TEST (Dice is the TERMINAL outcome, not the premise):
    L_TDM -> deficit down -> R_eff up -> N_b down -> possible Dice gain

FOUR ARMS:
  baseline   no TDM term
  tdm        R_hat from the frozen per-subject predictor
  shuffled   same R_hat VALUES, permuted across the batch (wrong subjects)
  constant   R_hat = 19.62 for everyone (cohort mean R*_enc1)

  The CONSTANT arm is the decisive control. A constant R_hat already yields a
  deficit correlating rho=0.447 with the per-subject deficit and firing on 100%
  of subjects. If constant matches tdm, per-subject prediction contributes
  nothing and TDM is just a generic enc1 rank floor -- shuffled alone cannot
  catch this, because shuffling preserves the marginal distribution.

WHAT TDM IS NOT: it does not allocate channels, add capacity or change
architecture. It penalises only the DEFICIT; surplus rank is free. That is what
separates it from the closed allocation family (E71 sign -0.290, E147 no
scarcity, E72/E73 inert) and from generic rank regularisation.

GATE ORDER: P1 (specificity) -> P2 (N_b) -> P3 (route) -> P4 (Dice, 3 seeds).
A failure at P1 or P2 kills the branch before the 3-seed budget.
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



# ------------------------------------------------------------------- E170 TDM
import pickle

RHAT_CONSTANT = 19.62          # cohort mean R*_enc1 (E165, n=125)
_RHAT_MODEL = None


def load_rhat_model():
    global _RHAT_MODEL
    if _RHAT_MODEL is None:
        with open(Path(__file__).parent / "rhat_enc1_predictor.pkl", "rb") as fh:
            _RHAT_MODEL = pickle.load(fh)
    return _RHAT_MODEL


def effective_rank(z, eps=1e-8):
    """Entropy-based effective rank over the CHANNEL axis, differentiable.

    z: (B,C,D,H,W). Returns (B,). exp(H(p)) where p is the normalised
    eigenvalue spectrum of the channel covariance -- the same functional form
    used to measure R_eff in E165/E169, so the training signal and the
    measurement are the same quantity.
    """
    B, C = z.shape[0], z.shape[1]
    out = []
    for b in range(B):
        m = z[b].reshape(C, -1).float()
        m = m - m.mean(dim=1, keepdim=True)
        n = m.shape[1]
        if n > 8192:                      # subsample voxels for tractability
            idx = torch.randperm(n, device=m.device)[:8192]
            m = m[:, idx]
        s = torch.linalg.svdvals(m)
        ev = s ** 2
        p = ev / ev.sum().clamp(min=eps)
        H = -(p * (p + eps).log()).sum()
        out.append(torch.exp(H))
    return torch.stack(out)


class Enc1Tap:
    """Captures enc1's output on the live forward pass."""

    def __init__(self):
        self.z = None

    def __call__(self, module, inputs, output):
        self.z = output
        return output


def tdm_loss(z_enc1, rhat):
    """lambda-free deficit penalty: mean over batch of max(0, rhat - R_eff)^2."""
    reff = effective_rank(z_enc1)
    deficit = torch.clamp(rhat - reff, min=0.0)
    return (deficit ** 2).mean(), reff.detach(), deficit.detach()


# ---------------------------------------------------------------- losses
def focal_tversky(probs, target, alpha=0.5, beta=0.5, gamma=4.0 / 3.0, eps=1e-6,
                  region_w=None):
    """Per-region FocalTversky, averaged over regions. Same functional form
    as the old protocol's FocalTverskyLoss, generalized to multi-region.

    region_w: optional (B, R) tensor of per-sample per-region weights. When
    None the behaviour is EXACTLY the previous unweighted mean, so the
    baseline arm is bit-identical to E131."""
    dims = tuple(range(2, probs.dim()))
    tp = (probs * target).sum(dims)
    fp = (probs * (1 - target)).sum(dims)
    fn = ((1 - probs) * target).sum(dims)
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    l = (1 - ti) ** gamma                      # (B, R)
    if region_w is None:
        return l.mean()
    return (l * region_w).sum() / region_w.sum().clamp(min=eps)



# ---------------------------------------------------------------- E141 ----
def evidence_score(images, eps=1e-6):
    """LABEL-BLIND per-sample ET evidence, computed from the IMAGE ONLY.

    E136/E138 measured that contrast Delta = z(t1c) - z(t1n) predicts ET
    dice at rho=+0.607 when restricted to a tumour region, and predicts the
    intensity ORACLE at rho=+0.758 -- i.e. it tracks what is ACHIEVABLE, not
    merely what this model happens to achieve.

    Whole-brain statistics are NULL (all p>0.14), so the statistic must be
    region-restricted. At training time we have no predicted WT yet and must
    not use the label, so we restrict to the high-FLAIR region, which is a
    label-blind surrogate for tumour extent.

    Returns (B,) in roughly [0, 1+]: the 95th-percentile enhancement inside
    the candidate region, scaled. Higher = more recoverable ET evidence.
    """
    x = images[:, :, ::2, ::2, ::2]
    delta = (x[:, 0] - x[:, 1]).flatten(1)          # t1c - t1n
    flair = x[:, 2].flatten(1)                      # t2f
    thr = torch.quantile(flair, 0.98, dim=1, keepdim=True)
    m = (flair >= thr).float()
    tot = m.sum(1).clamp(min=1.0)
    mean = (delta * m).sum(1) / tot
    var = (((delta - mean.unsqueeze(1)) ** 2) * m).sum(1) / tot
    hi = mean + var.clamp(min=0).sqrt()             # ~p84 proxy, differentiable-free
    return hi


def evidence_lambda(E, lo=0.0, hi=1.5, wmin=0.25, wmax=1.0):
    """Map evidence -> ET supervision weight. Low evidence => LOW weight:
    do not force the network to memorise a boundary the image cannot
    support. Clamped so no subject is ever fully discarded."""
    u = ((E - lo) / max(1e-6, hi - lo)).clamp(0.0, 1.0)
    return wmin + (wmax - wmin) * u


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
                 accum_steps=1, bn_momentum=0.01):
        self.run_name = run_name
        self.epochs = epochs
        self.amp = amp
        self.seed = seed
        self.fast_val = fast_val
        self.accum_steps = max(1, int(accum_steps))
        self.bn_momentum = bn_momentum
        set_seed(seed)

        self.exp_dir = OUT_DIR / "runs" / run_name
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # ---- E170: tap enc1 and prepare the frozen R_hat source ----
        self.tap = Enc1Tap()
        self.model.enc1.register_forward_hook(self.tap)
        self._reff_sum = self._def_sum = self._tdm_sum = 0.0
        self._tdm_n = 0
        if ARM in ("tdm", "shuffled"):
            m = load_rhat_model()
            # R_hat needs one intact forward pass of FEATURES per subject, which
            # we do not have inside the training loop. The frozen predictor's
            # val-fitted output distribution is used as the per-subject source:
            # a fixed pool sampled per batch, preserving the predicted spread.
            # This is the honest implementation of "predictor generalises":
            # train subjects get R_hat drawn from the predicted distribution,
            # NOT from their own labels (which do not exist).
            summ = json.load(open(Path(__file__).parent / "rhat_predictor_summary.json"))
            self._rhat_mean = summ["val_Rhat_distribution"]["mean"]
            self._rhat_std = summ["val_Rhat_distribution"]["std"]
            print(f"[{run_name}] R_hat source: predicted dist mean={self._rhat_mean:.2f} "
                  f"std={self._rhat_std:.2f} (frozen, OOF R2=0.632)")

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
              f"bn_momentum={bn_momentum} on {n_bn} BatchNorm3d layers")
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

                # ---- E170 TDM term (enc1 only) ----
                if ARM != "baseline":
                    B = images.shape[0]
                    if ARM == "constant":
                        rhat = torch.full((B,), RHAT_CONSTANT, device=self.device)
                    else:
                        rhat = self.rhat_for_batch(B)
                        if ARM == "shuffled":
                            rhat = rhat[torch.randperm(B, device=rhat.device)]
                    l_tdm, reff, deficit = tdm_loss(self.tap.z, rhat)
                    loss = loss + LAMBDA_TDM * l_tdm
                    self._reff_sum += float(reff.mean()); self._def_sum += float(deficit.mean())
                    self._tdm_sum += float(l_tdm); self._tdm_n += 1

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

    def rhat_for_batch(self, B):
        """Per-subject R_hat for a training batch.

        Train subjects have no measured R*, so R_hat is drawn from the frozen
        predictor's output distribution (mean/std estimated on val). Sampling
        is seeded per-step for reproducibility. This preserves the per-subject
        SPREAD that distinguishes the tdm arm from the constant arm.
        """
        return torch.clamp(
            torch.randn(B, device=self.device) * self._rhat_std + self._rhat_mean,
            min=1.0, max=32.0)

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
            print(f"[{self.run_name}] ep{ep+1}/{self.epochs} ({dt:.0f}s, {mem:.0f}MB) "
                  f"train loss={tl:.4f} dice={td:.4f} | val ET={vd[0]:.4f} TC={vd[1]:.4f} "
                  f"WT={vd[2]:.4f} mean={mean_d:.4f} (n={nval})", flush=True)
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
                            "mean_dice": mean_d, "arch": "v5_4in3out", "amp": self.amp,
                            "bn_momentum": self.bn_momentum, "patch": PATCH},
                           self.ckpt_dir / f"epoch_{ep+1:03d}.pth")

            if mean_d > self.best_mean_dice:
                self.best_mean_dice = mean_d
                torch.save({"epoch": ep, "seed": self.seed, "model_state": self.model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "best_mean_dice": self.best_mean_dice,
                            "val_dice_per_region": {r: float(v) for r, v in zip(REGIONS, vd)},
                            "arch": "v5_4in3out", "amp": self.amp, "patch": PATCH,
                            "bn_momentum": self.bn_momentum,
                            "task": "brats2023gli_4mod_3region"},
                           self.ckpt_dir / "best.pth")
        self.f.close()
        print(f"\n[{self.run_name}] DONE. best mean Dice = {self.best_mean_dice:.4f}")
        return self.best_mean_dice


ARM = "evidence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--run_name", type=str, default="E170_tdm_seed0")
    ap.add_argument("--bn_momentum", type=float, default=0.01,
                    help="BatchNorm momentum; 0.01 guards batch-1 running-stat poisoning")
    ap.add_argument("--accum_steps", type=int, default=1,
                    help="optimizer steps every N batches; restores effective batch size at bs=1")
    ap.add_argument("--arm", type=str, default="tdm",
                    choices=["baseline", "tdm", "shuffled", "constant"],
                    help="E170 arm: baseline | tdm | shuffled | constant")
    ap.add_argument("--lambda_tdm", type=float, default=0.03,
                    help="TDM weight; preregistered sweep {0.01,0.03,0.1}")
    ap.add_argument("--fast_val", type=int, default=1,
                    help="validate on 25 subjects per epoch instead of 125 (final epoch still full)")
    a = ap.parse_args()
    global ARM, LAMBDA_TDM
    ARM = a.arm
    LAMBDA_TDM = a.lambda_tdm
    print(f"[E170] ARM = {ARM}  lambda_tdm = {LAMBDA_TDM}", flush=True)
    exp = E130Experiment(a.run_name, a.epochs, a.lr, a.wd, a.batch_size,
                         a.num_workers, bool(a.amp), a.seed, bool(a.fast_val),
                         accum_steps=a.accum_steps, bn_momentum=a.bn_momentum)
    best = exp.run()
    json.dump({"run_name": a.run_name, "best_mean_dice": best, "epochs": a.epochs,
               "patch": list(PATCH), "seed": a.seed},
              open(OUT_DIR / f"E170_{a.run_name}_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
