"""LOCAL ORDERING GEOMETRY -- spatial structure of the FINAL logit field.

A different object from the dec3 displacement branch (which was KILLED by causal
patching): here we ask whether the missed lesion has a reproducible RADIAL
ORDERING signature in z(x) that the hard negative lacks.

For each missed component we build three matched regions:
  L : the missed GT component
  A : adjacent 1-6 voxel background shell (confirmed non-GT)
  H : hard negative -- size AND SHAPE matched, placed in confirmed background

SHAPE MATCHING (important, and stricter than the earlier hard negative):
  the earlier HARDNEG was "top-k logit voxels", which is a scattered set, not a
  compact object. A scattered set has a trivially different radial profile from
  a compact lesion, which would manufacture separation. Here H is built by
  TRANSLATING the component's own binary mask to a background location chosen
  to maximise mean logit -- so H has identical size AND identical shape, and
  only its position differs. This removes the shape confound by construction.

For every region we compute concentric shells S_0 (the region itself),
S_1..S_k (dilation rings) and record the profile m_i = mean z over S_i, plus
shape descriptors of that profile.

GT is used only to locate regions and label rows. Frozen weights, no training.
"""
import sys, csv, json, time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'exp_e12_eggo_m' / 'e130'))
from neuroscan_3d_v5 import UNet3D_v5
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, REGIONS
import importlib.util

spec = importlib.util.spec_from_file_location(
    "t", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["e"]; spec.loader.exec_module(t); sys.argv = _a

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
OUT = Path(__file__).resolve().parent
THR, PATCH, MIN_VOX, MAX_COMP = 0.5, 128, 5, 6
NSHELL = 6          # S_1..S_6 rings, each 1 voxel thick beyond the region
LOCAL_R = 12        # radius defining "local background" for the rank statistic


def shells(mask, k=NSHELL):
    """S_0 = mask itself; S_i = i-th 1-voxel dilation ring."""
    out = [mask.copy()]
    prev = mask.copy()
    for _ in range(k):
        d = ndimage.binary_dilation(prev, iterations=1)
        out.append(d & ~prev)
        prev = d
    return out


def profile_feats(lg, mask, valid, prefix=''):
    """Radial profile of the logit field around `mask`, restricted to `valid`."""
    S = shells(mask)
    m = []
    for s in S:
        sv = s & valid
        m.append(float(lg[sv].mean()) if sv.sum() > 0 else np.nan)
    m = np.array(m, dtype=np.float64)
    if np.isnan(m[0]):
        return None
    out = {f'{prefix}m{i}': float(v) for i, v in enumerate(m)}
    mm = m[~np.isnan(m)]
    if len(mm) < 3:
        return None
    # center-to-shell contrasts
    out[f'{prefix}c_1'] = float(m[0] - m[1]) if not np.isnan(m[1]) else np.nan
    out[f'{prefix}c_last'] = float(m[0] - mm[-1])
    out[f'{prefix}c_mean'] = float(m[0] - np.nanmean(m[1:]))
    # normalised contrast (scale-free): divide by the spread of the profile
    sd = float(np.nanstd(m)) + 1e-9
    out[f'{prefix}c_last_norm'] = out[f'{prefix}c_last'] / sd
    out[f'{prefix}c_mean_norm'] = out[f'{prefix}c_mean'] / sd
    # radial slope (linear fit over shell index)
    xi = np.arange(len(m))[~np.isnan(m)]
    out[f'{prefix}slope'] = float(np.polyfit(xi, mm, 1)[0])
    # monotonicity: fraction of adjacent steps that decrease
    d = np.diff(mm)
    out[f'{prefix}monotonic_frac'] = float((d < 0).mean())
    out[f'{prefix}n_sign_changes'] = int((np.diff(np.sign(d)) != 0).sum())
    # peak shell index
    out[f'{prefix}peak_shell'] = int(np.nanargmax(m))
    # decay rate: exponential fit to (m - min) profile
    y = mm - mm.min() + 1e-9
    try:
        out[f'{prefix}decay'] = float(-np.polyfit(xi, np.log(y), 1)[0])
    except Exception:
        out[f'{prefix}decay'] = np.nan
    # spatial coherence: 1 - (within-region sd / local sd)
    loc = ndimage.binary_dilation(mask, iterations=LOCAL_R) & valid
    sd_in = float(lg[mask & valid].std()) if (mask & valid).sum() > 1 else np.nan
    sd_loc = float(lg[loc].std()) if loc.sum() > 1 else np.nan
    out[f'{prefix}coherence'] = float(1.0 - sd_in / (sd_loc + 1e-9)) if sd_loc == sd_loc else np.nan
    # local percentile/rank of the region against its local background
    bg = loc & ~ndimage.binary_dilation(mask, iterations=1)
    if bg.sum() > 10:
        out[f'{prefix}local_rank'] = float((lg[bg] < lg[mask & valid].mean()).mean())
    else:
        out[f'{prefix}local_rank'] = np.nan
    out[f'{prefix}mu'] = float(lg[mask & valid].mean())
    return out


def place_shape_matched(shape_mask, lg, forbidden, valid, rng, n_try=400):
    """Translate `shape_mask` to a background location maximising mean logit.
    Identical size AND shape to the source; only position differs."""
    idx = np.argwhere(shape_mask)
    if len(idx) == 0:
        return None
    lo = idx.min(0); ext = idx.max(0) - lo + 1
    dims = np.array(shape_mask.shape)
    rel = idx - lo
    best, best_v = None, -np.inf
    for _ in range(n_try):
        hi = dims - ext
        if (hi <= 0).any():
            return None
        off = np.array([rng.integers(0, h + 1) for h in hi])
        p = rel + off
        cand = np.zeros_like(shape_mask)
        cand[tuple(p.T)] = True
        if (cand & forbidden).any() or not (cand & ~valid).sum() == 0:
            continue
        v = float(lg[cand].mean())
        if v > best_v:
            best_v, best = v, cand
    return best


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    fr = list(csv.DictReader(open(OUT / 'FORENSIC_per_subject.csv')))
    tail = {r['subject_id'] for r in fr
            if float(r['ET_dice']) < 0.5 or float(r['TC_dice']) < 0.5}
    idx = [i for i in range(len(ds)) if Path(ds.subject_dirs[i]).name in tail]

    rows = []
    t0 = time.time()
    for ii, i in enumerate(idx):
        img, tgt, sid = ds[i]
        Y = tgt.numpy() > 0.5
        probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                         t.PATCH, t.SW_OVERLAP, 3, dev, True)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        LOG = np.log(probs / (1 - probs))
        P = probs >= THR
        brain = img[0].numpy() != 0
        for ri, rn in [(0, 'ET'), (1, 'TC')]:
            lg_full = LOG[ri]
            lbl, nl = ndimage.label(Y[ri])
            miss = sorted([(int((lbl == g).sum()), g) for g in range(1, nl + 1)
                           if (lbl == g).sum() >= MIN_VOX and not ((lbl == g) & P[ri]).any()],
                          reverse=True)[:MAX_COMP]
            for sz, g in miss:
                cm = lbl == g
                com = np.array(ndimage.center_of_mass(cm)).astype(int)
                st = [int(np.clip(c - PATCH // 2, 0, s - PATCH)) for c, s in zip(com, cm.shape)]
                sl = tuple(slice(s, s + PATCH) for s in st)
                cm_p = cm[sl]; lg = lg_full[sl]
                if cm_p.sum() < MIN_VOX:
                    continue
                anygt = (Y[0] | Y[1] | Y[2])[sl]
                brain_p = brain[sl]
                dist = ndimage.distance_transform_edt(~cm_p)
                adj = (dist > 1) & (dist <= 6) & brain_p & ~anygt
                if adj.sum() < MIN_VOX:
                    continue
                rng = np.random.default_rng(abs(hash((sid, rn, g))) % (2 ** 31))
                forbid = ndimage.binary_dilation(anygt, iterations=6)
                H = place_shape_matched(cm_p, lg, forbid, brain_p, rng)
                if H is None or H.sum() < MIN_VOX:
                    continue
                for pop, m in [('LESION', cm_p), ('ADJ', adj), ('HARDNEG', H)]:
                    f = profile_feats(lg, m, brain_p)
                    if f is None:
                        continue
                    f.update({'subject_id': sid, 'region': rn, 'comp_id': int(g),
                              'population': pop, 'n_vox': int(m.sum()),
                              'comp_vox': int(cm_p.sum())})
                    rows.append(f)
        if (ii + 1) % 4 == 0:
            print(f'  {ii+1}/{len(idx)} subj, {len(rows)} rows ({time.time()-t0:.0f}s)', flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(OUT / 'ORDERING_features.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print(f'\nwrote ORDERING_features.csv ({len(rows)} rows) in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
