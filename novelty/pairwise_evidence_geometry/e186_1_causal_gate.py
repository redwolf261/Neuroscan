"""E186.1 -- Causal E3 spatial-permutation gate.

HYPOTHESIS: for missed lesions, the downstream network needs the SPATIAL
ARRANGEMENT of E3 features, not merely their channel content.

Perturbations (spatial-destruction dose-response), applied to E3 inside an ROI:
  P0 original
  P1 within-2x2x2-neighbourhood permutation   (mild, respects pool3 windows)
  P2 ROI-wide permutation                     (destroys larger-scale structure)
  P3 global permutation over the whole E3 map  (positive control, extreme)

Applied identically to three populations so any difference is attributable to
the population and not the manipulation:
  L missed lesion / S successfully detected lesion / H hard negative

CRITICAL ARCHITECTURAL NOTE. In UNet3D_v5, enc3 feeds BOTH pool3 (-> bottleneck)
AND the decoder skip (cat3 = [upconv3, enc3]). A naive permutation of a single
shared tensor would hit both paths at once -- exactly the shared-tensor confound
that invalidated E62/E63 and was only caught in E64. This script therefore runs
THREE separate arms per perturbation:
    both  : permuted E3 goes to pool3 AND to the skip   (naive)
    pool  : permuted E3 goes to pool3 only, skip gets ORIGINAL E3
    skip  : permuted E3 goes to the skip only, pool3 gets ORIGINAL E3
Only the `pool` arm tests the stated hypothesis. The decomposition is the point.

Permutation preserves the multiset of values exactly, so per-channel mean and
std inside the ROI are invariant by construction -- verified at runtime.

Frozen model, no training, no threshold/TTA/ensemble changes.
"""
import sys, csv, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from neuroscan_3d_v5 import UNet3D_v5
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, REGIONS
import importlib.util

spec = importlib.util.spec_from_file_location(
    "t", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["e"]; spec.loader.exec_module(t); sys.argv = _a

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
PATCH, MIN_VOX, MAX_PER_SUBJ = 128, 5, 3
ROI_DIL = 8            # input-res dilation -> ROI
ET = 0


def forward_from_e3(model, e3_pool, e3_skip, e1, e2):
    """Continue the frozen forward pass, allowing DIFFERENT E3 tensors to feed
    the pooling path and the decoder skip."""
    with torch.no_grad():
        p3 = model.pool3(e3_pool)
        b = model.bottleneck(p3)
        u3 = model.upconv3(b)
        d3 = model.dec3(torch.cat([u3, e3_skip], 1))
        u2 = model.upconv2(d3)
        d2 = model.dec2(torch.cat([u2, e2], 1))
        u1 = model.upconv1(d2)
        eg, _ = model.attn_gate1(gate=b, skip=e1)
        d1 = model.dec1(torch.cat([u1, eg], 1))
        z = model.seg_head[0](d1)          # raw pre-sigmoid
    return z[0]


def perm_local(F, roi, gen):
    """P1: permute positions within each 2x2x2 block (pool3's own windows)."""
    out = F.clone()
    C, D, H, W = F.shape
    idx = torch.nonzero(torch.from_numpy(roi), as_tuple=False)
    if idx.numel() == 0:
        return out
    blocks = {}
    for z, y, x in idx.tolist():
        blocks.setdefault((z // 2, y // 2, x // 2), []).append((z, y, x))
    for _, pts in blocks.items():
        if len(pts) < 2:
            continue
        pi = torch.randperm(len(pts), generator=gen)
        src = [pts[i] for i in pi.tolist()]
        vals = torch.stack([F[:, a, b, c] for a, b, c in src], 1)
        for k, (a, b, c) in enumerate(pts):
            out[:, a, b, c] = vals[:, k]
    return out


def perm_mask(F, mask, gen):
    """P2/P3: permute all positions within `mask`."""
    out = F.clone()
    m = torch.from_numpy(mask)
    idx = torch.nonzero(m, as_tuple=False)
    n = idx.shape[0]
    if n < 2:
        return out
    pi = torch.randperm(n, generator=gen)
    src = idx[pi]
    out[:, idx[:, 0], idx[:, 1], idx[:, 2]] = F[:, src[:, 0], src[:, 1], src[:, 2]]
    return out


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    print(f'E186.1 causal gate on {len(ds)} subjects', flush=True)

    rows = []
    t0 = time.time()
    for ii in range(len(ds)):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                         t.PATCH, t.SW_OVERLAP, 3, dev, True)
        P = probs >= 0.5
        brain = img[0].numpy() != 0
        lbl, nl = ndimage.label(Y[ET])
        missed, succ = [], []
        for g in range(1, nl + 1):
            cm = lbl == g
            if cm.sum() < MIN_VOX:
                continue
            ov = float((cm & P[ET]).sum()) / cm.sum()
            if ov == 0:
                missed.append((int(cm.sum()), g))
            elif ov >= 0.5:
                succ.append((int(cm.sum()), g))
        missed.sort(reverse=True); succ.sort(reverse=True)
        picks = [('L', g) for _, g in missed[:MAX_PER_SUBJ]] + \
                [('S', g) for _, g in succ[:MAX_PER_SUBJ]]
        if not picks:
            continue
        for pop0, g in picks:
            cm = lbl == g
            com = np.array(ndimage.center_of_mass(cm)).astype(int)
            st = [int(np.clip(c - PATCH // 2, 0, s - PATCH)) for c, s in zip(com, cm.shape)]
            sl = tuple(slice(s, s + PATCH) for s in st)
            cm_p = cm[sl]
            if cm_p.sum() < MIN_VOX:
                continue
            anygt = (Y[0] | Y[1] | Y[2])[sl]; brain_p = brain[sl]
            x = img[(slice(None),) + sl].unsqueeze(0).to(dev)
            with torch.no_grad():
                e1 = model.enc1(x); p1 = model.pool1(e1)
                e2 = model.enc2(p1); p2 = model.pool2(e2)
                e3 = model.enc3(p2)
            F3 = e3[0]
            z0 = forward_from_e3(model, e3, e3, e1, e2)
            lg0 = z0[ET].cpu().numpy()
            # hard negative, same shape, translated to max-logit background
            rng = np.random.default_rng(abs(hash((sid, g))) % (2**31))
            forbid = ndimage.binary_dilation(anygt, iterations=6)
            ii_ = np.argwhere(cm_p); lo = ii_.min(0); ext = ii_.max(0) - lo + 1
            rel = ii_ - lo; dims = np.array(cm_p.shape)
            best, bestv = None, -np.inf
            for _ in range(200):
                hi = dims - ext
                if (hi <= 0).any():
                    break
                off = np.array([rng.integers(0, h + 1) for h in hi])
                pp = rel + off
                cand = np.zeros_like(cm_p); cand[tuple(pp.T)] = True
                if (cand & forbid).any() or not brain_p[cand].all():
                    continue
                v = float(lg0[cand].mean())
                if v > bestv:
                    bestv, best = v, cand
            if best is None:
                continue
            for pop, msk in [(pop0, cm_p), ('H', best)]:
                roi_in = ndimage.binary_dilation(msk, iterations=ROI_DIL) & brain_p
                # map masks to E3 resolution (stride 4)
                shp3 = tuple(F3.shape[1:])
                m3 = (Fn.adaptive_max_pool3d(
                    torch.from_numpy(msk.astype(np.float32))[None, None], shp3)[0, 0] > 0.5).numpy()
                r3 = (Fn.adaptive_max_pool3d(
                    torch.from_numpy(roi_in.astype(np.float32))[None, None], shp3)[0, 0] > 0.5).numpy()
                if m3.sum() < 1 or r3.sum() < 8:
                    continue
                shell = ndimage.binary_dilation(msk, iterations=6) & ~msk & brain_p & ~anygt
                if shell.sum() < MIN_VOX:
                    continue
                base = {'subject_id': sid, 'comp_id': int(g), 'population': pop,
                        'vox': int(msk.sum()), 'e3_vox': int(m3.sum()),
                        'roi_e3_vox': int(r3.sum())}
                allmask = np.ones(shp3, bool)
                for pk, perm_fn, pmask in [('P0', None, None),
                                           ('P1', perm_local, r3),
                                           ('P2', perm_mask, r3),
                                           ('P3', perm_mask, allmask)]:
                    gen = torch.Generator(device='cpu')
                    gen.manual_seed(abs(hash((sid, g, pop, pk))) % (2**31))
                    if perm_fn is None:
                        Fp = F3
                    else:
                        Fp = perm_fn(F3.cpu(), pmask, gen).to(dev)
                    e3p = Fp.unsqueeze(0)
                    # statistics-preservation check (permutation => invariant)
                    if perm_fn is not None:
                        a = F3[:, torch.from_numpy(pmask).to(dev)]
                        b = Fp[:, torch.from_numpy(pmask).to(dev)]
                        dmu = float((a.mean(1) - b.mean(1)).abs().max())
                        dsd = float((a.std(1) - b.std(1)).abs().max())
                    else:
                        dmu = dsd = 0.0
                    for arm, ep, es in [('both', e3p, e3p), ('pool', e3p, e3),
                                        ('skip', e3, e3p)]:
                        if pk == 'P0' and arm != 'both':
                            continue
                        z = forward_from_e3(model, ep, es, e1, e2)
                        lg = z[ET].cpu().numpy()
                        rec = dict(base)
                        rec.update({'perturb': pk, 'arm': arm,
                                    'dmu_max': dmu, 'dsd_max': dsd,
                                    'z_comp': float(np.median(lg[msk])),
                                    'z_shell': float(np.median(lg[shell])),
                                    'dz': float(np.median(lg[msk]) - np.median(lg[shell])),
                                    'z_comp_mean': float(lg[msk].mean()),
                                    'frac_pos': float((lg[msk] > 0).mean())})
                        rows.append(rec)
        if (ii + 1) % 15 == 0:
            print(f'  {ii+1}/{len(ds)} subj, {len(rows)} rows ({time.time()-t0:.0f}s)', flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(HERE / 'E186_1_causal.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    print(f'\nwrote E186_1_causal.csv ({len(rows)} rows)  '
          f'pops={dict(Counter(r["population"] for r in rows))}')
    print(f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
