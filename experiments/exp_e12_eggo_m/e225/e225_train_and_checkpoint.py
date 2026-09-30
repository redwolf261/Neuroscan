"""E225 step 1 -- retrain E224-R's exact protocol (identical seed/subjects/
epochs/loss geometry) ONLY to regenerate the epoch-5 checkpoints, which
E224-R itself did not save. Training logic is UNCHANGED from E224-R (copied
verbatim) so the retrained models are the same models E224-R measured --
this is not a new experiment condition, just checkpoint capture for the
E225 mechanism audit (distributions, threshold sweep, component stats).

Original E224-R docstring below, preserved for context:
---
E224-R: does E103's CC-DiceCE FP/FN direction REVERSAL replicate on the
CURRENT E130+ multimodal, 3-region (ET/TC/WT) training geometry?

E103 (old FLAIR-only, single binarized whole-tumor label, Dataset/
brats_dataset.py): CC-DiceCE gave FP -12.3%, FN +16.5% vs baseline DiceCE
-- the OPPOSITE of the paper's documented BraTS failure (more FP, fewer FN
from equal-per-instance-weighting over-penalizing missed small
components). This script repeats that SAME short mechanism-audit protocol
(5 epochs, same seed, isolated Dice/CE loss geometry, no deep-supervision/
boundary/evidential terms -- matching E103's own deliberate simplification)
on the CURRENT pipeline: 4-modality input (t1c,t1n,t2f,t2w), 3-region
target (ET,TC,WT), Dataset/brats_multimodal_dataset.py, 128^3 patches,
current sampler (fg_bias=0.66, WT-centered).

Loss geometry: same as E103 exactly --
  baseline : plain DiceCE, global, region-averaged (matches E103's
             plain_dice_ce_loss, generalized to 3 regions the SAME way
             the current production focal_tversky() generalizes: per-
             region then uniform average)
  cc       : CCDiceCELossMultiRegion (cc_dicece_multiregion.py) -- E103's
             OWN Voronoi/per-component functions, UNCHANGED, applied per
             region

PRE-DECLARED DECISION RULE (identical structure to E103's):
  REVERSAL REPLICATES if, relative to baseline: FP decreases AND/OR FN
    increases (the E103 direction), consistently across epochs, on ET
    specifically (ET is where small/missed components concentrate, per
    E207/E218's own component-size distributions this session).
  REVERSAL DOES NOT REPLICATE if FP increases / FN decreases (the PAPER's
    original predicted direction) or the shift is directionally
    inconsistent across epochs.
Gradient-geometry (grad norm ratio + cosine) tracked as a SECONDARY
diagnostic, exactly as in E103 -- not required for the primary verdict.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders, REGIONS  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'e224'))
from cc_dicece_multiregion import CCDiceCELossMultiRegion  # noqa: E402

SEED = 0
N_EPOCHS = 5             # matches E103's own short-audit budget
PATCH = (128, 128, 128)
BATCH_SIZE = 1
NUM_WORKERS = 4
LR = 1e-4
WD = 1e-5
OUT_DIR = Path(__file__).parent
ET = 0
SMOKE_SUBJECTS = None    # set by main() when --smoke is passed
AUDIT_SUBJECTS = 300     # mechanism-audit subject cap; None = full 1126


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def plain_dicece_multiregion(probs, target, eps=1e-7):
    """Matches E103's plain_dice_ce_loss, generalized per-region + averaged
    -- the baseline condition's segmentation loss, structurally identical
    to CCDiceCELossMultiRegion's own dicece_avg term for apples-to-apples."""
    B, R = probs.shape[0], probs.shape[1]
    dice_losses, ces = [], []
    for ri in range(R):
        p = probs[:, ri:ri+1].reshape(B, -1)
        t = target[:, ri:ri+1].reshape(B, -1)
        inter = (p * t).sum(dim=1)
        denom = p.sum(dim=1) + t.sum(dim=1)
        dice_losses.append((1.0 - (2.0 * inter / denom.clamp_min(eps))).mean())
        ces.append(F.binary_cross_entropy(
            probs[:, ri:ri+1].clamp(eps, 1 - eps), target[:, ri:ri+1], reduction='mean'))
    dice_avg = torch.stack(dice_losses).mean()
    ce_avg = torch.stack(ces).mean()
    return dice_avg, ce_avg


def gradient_geometry(probs, target, eps=1e-7):
    """SAME method as E103: grad w.r.t. output probs (not params), dice
    vs CE terms, global (region-averaged) -- directly comparable number."""
    p = probs.detach().requires_grad_(True)
    dice_avg, ce_avg = plain_dicece_multiregion(p, target, eps=eps)
    g_dice = torch.autograd.grad(dice_avg, p, retain_graph=True)[0]
    g_ce = torch.autograd.grad(ce_avg, p, retain_graph=False)[0]
    n_dice = float(g_dice.norm().item())
    n_ce = float(g_ce.norm().item())
    cos = float(F.cosine_similarity(g_dice.reshape(1, -1), g_ce.reshape(1, -1)).item())
    return n_dice, n_ce, cos


def run_condition(use_cc, run_name, dev):
    set_seed(SEED)
    exp_dir = OUT_DIR / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = UNet3D_v5(in_channels=4, out_channels=3).to(dev)
    cc_loss_fn = CCDiceCELossMultiRegion().to(dev) if use_cc else None
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

    train_loader, val_loader = create_multimodal_loaders(
        root_dir=str(ROOT / 'Dataset' / 'Training'), batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, val_split=0.1, patch_size=PATCH, seed=SEED)

    cap = SMOKE_SUBJECTS if SMOKE_SUBJECTS is not None else AUDIT_SUBJECTS
    if cap is not None:
        # DETERMINISM VERIFIED (Dataset/brats_multimodal_dataset.py): the
        # train/val split uses a FIXED RandomState(42) shuffle, independent
        # of the `seed` param -- subject_dirs is bit-identical across
        # separate constructions. Capping to the first `cap` entries here
        # therefore gives baseline and cc the SAME 300 subjects, in the
        # same order, per the user's explicit no-different-subset requirement.
        train_loader.dataset.subject_dirs = train_loader.dataset.subject_dirs[:cap]

    print(f'[{run_name}] use_cc={use_cc}  train={len(train_loader.dataset)} '
          f'val={len(val_loader.dataset)}', flush=True)

    f = open(exp_dir / 'epoch_metrics.csv', 'w', newline='')
    w = csv.writer(f)
    w.writerow(['epoch', 'train_loss', 'val_dice_ET', 'val_dice_TC', 'val_dice_WT',
               'val_fp_ET', 'val_fn_ET', 'val_fp_TC', 'val_fn_TC', 'val_fp_WT', 'val_fn_WT',
               'val_mean_pred_prob_ET', 'grad_norm_dice', 'grad_norm_ce', 'grad_cos_theta',
               'epoch_time_sec'])

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f'[{run_name}] ep{epoch+1} train')
        for images, targets, _ in pbar:
            images = images.to(dev, non_blocking=True)
            targets = targets.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            probs = out['probs']
            if use_cc:
                loss, diag = cc_loss_fn(probs, targets)
            else:
                dice_avg, ce_avg = plain_dicece_multiregion(probs, targets)
                loss = dice_avg + ce_avg
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot += loss.item(); n += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        scheduler.step()
        train_loss = tot / max(1, n)

        # ---- validation: sliding-window-free, PATCH-level for speed
        # (matches E103's own short-audit choice of a quick proxy metric,
        # NOT full native-volume sliding-window inference -- that is the
        # production eval, reserved for a full campaign if this replicates) ----
        model.eval()
        dices = {r: [] for r in range(3)}
        fps = {r: 0 for r in range(3)}; fns = {r: 0 for r in range(3)}
        mean_prob_et = []
        grad_norms_dice, grad_norms_ce, grad_coss = [], [], []
        val_ds = val_loader.dataset
        n_val_patches = min((4 if SMOKE_SUBJECTS else 20), len(val_ds))
        with torch.no_grad():
            for vi in range(n_val_patches):
                image, target, _ = val_ds[vi]
                # val split returns FULL native volumes; crop a fg-biased
                # 128^3 patch here for a cheap proxy (full sliding-window
                # eval deferred to a full campaign if E224-R replicates)
                D, H, W = image.shape[1:]
                pd, ph, pw = PATCH
                fg = np.argwhere(target[2].numpy() > 0)  # WT
                rng = np.random.default_rng(1000 + vi)
                if len(fg):
                    c = fg[rng.integers(len(fg))]
                else:
                    c = np.array([D // 2, H // 2, W // 2])
                # BUG CAUGHT BY SMOKE TEST: several val subjects have a
                # native dimension < 128 (e.g. 01161 D=126, 01168 D=117).
                # max(0, full-p) clips the start to 0 but s+p still exceeds
                # `full`, silently producing a <128 crop that breaks the
                # U-Net skip-connection alignment 3 pooling levels down
                # (cat3 size mismatch). Fixed to PAD, matching the
                # production dataset's own _sample_patch pad-on-shortfall
                # behaviour (Dataset/brats_multimodal_dataset.py).
                st = [int(np.clip(int(cc) - p // 2, 0, max(0, full - p)))
                     for cc, p, full in zip(c, (pd, ph, pw), (D, H, W))]
                sl = tuple(slice(s, min(s + p, full))
                          for s, p, full in zip(st, (pd, ph, pw), (D, H, W)))
                img_c = image[(slice(None),) + sl]
                tgt_c = target[(slice(None),) + sl]
                if img_c.shape[1:] != (pd, ph, pw):
                    img_p_np = torch.zeros((img_c.shape[0], pd, ph, pw), dtype=img_c.dtype)
                    tgt_p_np = torch.zeros((tgt_c.shape[0], pd, ph, pw), dtype=tgt_c.dtype)
                    img_p_np[:, :img_c.shape[1], :img_c.shape[2], :img_c.shape[3]] = img_c
                    tgt_p_np[:, :tgt_c.shape[1], :tgt_c.shape[2], :tgt_c.shape[3]] = tgt_c
                    img_c, tgt_c = img_p_np, tgt_p_np
                img_p = img_c.unsqueeze(0).to(dev)
                tgt_p = tgt_c.unsqueeze(0).to(dev)
                out = model(img_p)
                p_ = out['probs']
                pb = (p_ >= 0.5).float()
                for r in range(3):
                    inter = (pb[:, r] * tgt_p[:, r]).sum()
                    den = pb[:, r].sum() + tgt_p[:, r].sum()
                    d = (2 * inter / den.clamp_min(1e-6)) if den > 0 else torch.tensor(1.0)
                    dices[r].append(float(d.item()))
                    fps[r] += int(((pb[:, r] == 1) & (tgt_p[:, r] == 0)).sum().item())
                    fns[r] += int(((pb[:, r] == 0) & (tgt_p[:, r] == 1)).sum().item())
                mean_prob_et.append(float(p_[:, ET].mean().item()))
                # gradient_geometry needs autograd ENABLED (it builds its own
                # detached, requires_grad_(True) leaf and backprops through
                # plain_dicece_multiregion) -- the enclosing validation loop
                # is under torch.no_grad(), which globally disables grad
                # tracking regardless of requires_grad_ on the leaf. Caught
                # by the smoke test (RuntimeError: does not require grad).
                with torch.enable_grad():
                    nd, nc, cs = gradient_geometry(p_, tgt_p)
                grad_norms_dice.append(nd); grad_norms_ce.append(nc); grad_coss.append(cs)

        row = [epoch + 1, train_loss,
              float(np.mean(dices[0])), float(np.mean(dices[1])), float(np.mean(dices[2])),
              fps[0], fns[0], fps[1], fns[1], fps[2], fns[2],
              float(np.mean(mean_prob_et)), float(np.mean(grad_norms_dice)),
              float(np.mean(grad_norms_ce)), float(np.mean(grad_coss)),
              time.time() - t0]
        w.writerow(row); f.flush()
        print(f'[{run_name}] ep{epoch+1}: loss={train_loss:.4f} '
              f'diceET={row[2]:.3f} fpET={fps[0]} fnET={fns[0]} ({time.time()-t0:.0f}s)',
              flush=True)
    f.close()
    # E225 addition: save final-epoch checkpoint (E224-R did not save any --
    # this run's SOLE purpose beyond reproducing E224-R's own numbers is to
    # capture these weights for the E225 mechanism audit).
    ckpt_path = exp_dir / 'epoch5.pth'
    torch.save({'model_state': model.state_dict(), 'epoch': N_EPOCHS,
               'use_cc': use_cc, 'seed': SEED, 'audit_subjects': AUDIT_SUBJECTS},
               str(ckpt_path))
    print(f'[{run_name}] saved checkpoint to {ckpt_path}', flush=True)


def main():
    dev = torch.device('cuda')
    smoke = '--smoke' in sys.argv
    global N_EPOCHS, SMOKE_SUBJECTS
    if smoke:
        N_EPOCHS = 1
        SMOKE_SUBJECTS = 8
    run_condition(use_cc=False, run_name='E225_baseline', dev=dev)
    run_condition(use_cc=True, run_name='E225_cc', dev=dev)
    print('E225 checkpoint-capture training complete.', flush=True)


if __name__ == '__main__':
    main()
