"""E225 -- mechanism audit of the E224-R CC-DiceCE FP-down/FN-up inversion.

Loads the epoch-5 checkpoints from e225_train_and_checkpoint.py (baseline,
cc) and runs a SINGLE inference pass per condition over the FULL 125-subject
val set (patch-level proxy eval, same crop convention as E224-R/E103's own
short-audit choice: one fg-biased 128^3 crop per subject, not full sliding-
window -- kept identical to E224-R so results are directly comparable, not
a different evaluation regime sneaking in alongside the mechanism question).

Streams everything needed for four analyses without persisting raw
probability volumes to disk:

  1. PROBABILITY DISTRIBUTIONS (per region: GT-positive voxels, GT-negative
     voxels, FP voxels at tau=0.5, FN voxels at tau=0.5) -- summarized as
     histograms (50 bins, [0,1]) so exact distributions are comparable
     without storing raw arrays.

  2. THRESHOLD SWEEP: Dice(tau), FP(tau), FN(tau) for tau in
     {0.1,...,0.9} step 0.1, per region. Distinguishes "CC just shifts the
     calibration point" (curves overlap after re-thresholding) from "CC
     changes the decision boundary" (CC dominates baseline over a range).

  3. CONNECTED-COMPONENT behaviour (ET primary, TC/WT secondary): predicted
     component count, FP component count + volume distribution, GT
     component detected/missed count, GT component size distribution split
     by detected/missed -- tells us what disappeared when FP fell.

  4. ET vs TC vs WT summary table (already implied by 1-3, but reported
     together) to check whether the effect is ET-topology-specific.

Deliberately NOT connecting this to E148-E150 framing here (per user
instruction) -- this script only measures, it does not interpret.
"""
import sys, csv, json
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import create_multimodal_loaders  # noqa: E402

HERE = Path(__file__).parent
REGIONS = ('ET', 'TC', 'WT')
PATCH = (128, 128, 128)
MIN_VOX = 5          # matches E204-E223 convention
TAUS = [round(0.1 * i, 1) for i in range(1, 10)]
N_BINS = 50


def load_model(ckpt_path, dev):
    ck = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    model = UNet3D_v5(in_channels=4, out_channels=3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def crop_val_subject(image, target, vi, dev):
    """Same fg-biased 128^3 crop + pad-on-shortfall as E224-R's validation
    loop (Dataset/brats_multimodal_dataset.py-style, seeded 1000+vi for
    reproducibility across BOTH conditions -- identical crop per subject,
    baseline and cc, so any distributional difference is attributable to
    the model, not to a different crop)."""
    D, H, W = image.shape[1:]
    pd, ph, pw = PATCH
    fg = np.argwhere(target[2].numpy() > 0)  # WT-biased centre, matches E224-R
    rng = np.random.default_rng(1000 + vi)
    if len(fg):
        c = fg[rng.integers(len(fg))]
    else:
        c = np.array([D // 2, H // 2, W // 2])
    st = [int(np.clip(int(cc) - p // 2, 0, max(0, full - p)))
          for cc, p, full in zip(c, (pd, ph, pw), (D, H, W))]
    sl = tuple(slice(s, min(s + p, full))
               for s, p, full in zip(st, (pd, ph, pw), (D, H, W)))
    img_c = image[(slice(None),) + sl]
    tgt_c = target[(slice(None),) + sl]
    if img_c.shape[1:] != (pd, ph, pw):
        img_p = torch.zeros((img_c.shape[0], pd, ph, pw), dtype=img_c.dtype)
        tgt_p = torch.zeros((tgt_c.shape[0], pd, ph, pw), dtype=tgt_c.dtype)
        img_p[:, :img_c.shape[1], :img_c.shape[2], :img_c.shape[3]] = img_c
        tgt_p[:, :tgt_c.shape[1], :tgt_c.shape[2], :tgt_c.shape[3]] = tgt_c
        img_c, tgt_c = img_p, tgt_p
    return (img_c.unsqueeze(0).to(dev), tgt_c.unsqueeze(0).to(dev))


def run_condition(ckpt_path, run_name, dev, val_ds, n_val):
    model = load_model(ckpt_path, dev)
    print(f'[{run_name}] loaded {ckpt_path.name}', flush=True)

    hist_edges = np.linspace(0, 1, N_BINS + 1)
    # per-region accumulators
    hist_pos = {r: np.zeros(N_BINS, dtype=np.int64) for r in range(3)}   # GT-positive voxel probs
    hist_neg = {r: np.zeros(N_BINS, dtype=np.int64) for r in range(3)}   # GT-negative voxel probs
    hist_fp = {r: np.zeros(N_BINS, dtype=np.int64) for r in range(3)}    # FP voxel probs (tau=0.5)
    hist_fn_gtprob = {r: np.zeros(N_BINS, dtype=np.int64) for r in range(3)}  # FN voxels' predicted prob

    sweep = {r: {tau: {'tp': 0, 'fp': 0, 'fn': 0} for tau in TAUS} for r in range(3)}

    # component-level (ET primary, TC/WT tracked too)
    comp_rows = []  # subject, region, kind(gt|pred), comp_id, size, detected(gt only)

    with torch.no_grad():
        for vi in range(n_val):
            image, target, sid = val_ds[vi]
            img_p, tgt_p = crop_val_subject(image, target, vi, dev)
            out = model(img_p)
            probs = out['probs']  # (1,3,D,H,W)
            for r in range(3):
                p = probs[0, r].cpu().numpy()
                t = (tgt_p[0, r].cpu().numpy() > 0.5)
                pb = p >= 0.5

                # --- 1. probability distributions ---
                if t.any():
                    hist_pos[r] += np.histogram(p[t], bins=hist_edges)[0]
                if (~t).any():
                    hist_neg[r] += np.histogram(p[~t], bins=hist_edges)[0]
                fp_mask = pb & (~t)
                fn_mask = (~pb) & t
                if fp_mask.any():
                    hist_fp[r] += np.histogram(p[fp_mask], bins=hist_edges)[0]
                if fn_mask.any():
                    hist_fn_gtprob[r] += np.histogram(p[fn_mask], bins=hist_edges)[0]

                # --- 2. threshold sweep ---
                for tau in TAUS:
                    pbt = p >= tau
                    sweep[r][tau]['tp'] += int((pbt & t).sum())
                    sweep[r][tau]['fp'] += int((pbt & ~t).sum())
                    sweep[r][tau]['fn'] += int((~pbt & t).sum())

                # --- 3. connected components (tau=0.5) ---
                gt_lbl, gt_n = ndimage.label(t)
                for g in range(1, gt_n + 1):
                    cm = gt_lbl == g
                    sz = int(cm.sum())
                    if sz < MIN_VOX:
                        continue
                    detected = int((cm & pb).sum() / sz >= 0.5)
                    comp_rows.append({'subject_id': sid, 'region': REGIONS[r],
                                      'kind': 'gt', 'comp_id': g, 'size': sz,
                                      'detected': detected})
                pred_lbl, pred_n = ndimage.label(pb)
                for g in range(1, pred_n + 1):
                    cm = pred_lbl == g
                    sz = int(cm.sum())
                    if sz < MIN_VOX:
                        continue
                    is_fp = int(not (cm & t).any())
                    comp_rows.append({'subject_id': sid, 'region': REGIONS[r],
                                      'kind': 'pred', 'comp_id': g, 'size': sz,
                                      'detected': -1, 'is_fp': is_fp})
            if (vi + 1) % 25 == 0:
                print(f'  [{run_name}] {vi+1}/{n_val}', flush=True)

    out = {
        'hist_edges': hist_edges.tolist(),
        'hist_pos': {REGIONS[r]: hist_pos[r].tolist() for r in range(3)},
        'hist_neg': {REGIONS[r]: hist_neg[r].tolist() for r in range(3)},
        'hist_fp': {REGIONS[r]: hist_fp[r].tolist() for r in range(3)},
        'hist_fn_gtprob': {REGIONS[r]: hist_fn_gtprob[r].tolist() for r in range(3)},
        'sweep': {REGIONS[r]: {str(tau): sweep[r][tau] for tau in TAUS} for r in range(3)},
    }
    with open(HERE / f'{run_name}_distributions.json', 'w') as f:
        json.dump(out, f)

    with open(HERE / f'{run_name}_components.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['subject_id', 'region', 'kind', 'comp_id',
                                          'size', 'detected', 'is_fp'])
        w.writeheader()
        for row in comp_rows:
            row.setdefault('is_fp', -1)
            w.writerow(row)

    print(f'[{run_name}] wrote distributions + {len(comp_rows)} component rows', flush=True)


def main():
    dev = torch.device('cuda')
    smoke = '--smoke' in sys.argv

    _, val_loader = create_multimodal_loaders(
        root_dir=str(ROOT / 'Dataset' / 'Training'), batch_size=1,
        num_workers=0, val_split=0.1, patch_size=PATCH, seed=0)
    val_ds = val_loader.dataset
    n_val = 5 if smoke else len(val_ds)
    print(f'E225 mechanism audit: {n_val} val subjects', flush=True)

    e225_dir = HERE
    run_condition(e225_dir / 'E225_baseline' / 'epoch5.pth', 'E225_baseline', dev, val_ds, n_val)
    run_condition(e225_dir / 'E225_cc' / 'epoch5.pth', 'E225_cc', dev, val_ds, n_val)
    print('E225 mechanism audit complete.', flush=True)


if __name__ == '__main__':
    main()
