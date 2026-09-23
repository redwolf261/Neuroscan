"""E189-revised -- RELIEF vs RECOVERY at the enc3 skip. FROZEN. NO TRAINING.

E188 found: translating the enc3 skip HELPS missed lesions (+10.5% normalised)
while costing detected lesions 48%. E79 already proved the decoder uses the
skip as an ABSOLUTE-COORDINATE LOOKUP (direction locally coherent cos 0.88-0.95,
smoothing recovers only 11%), and E74 showed translation-sensitivity generalises
across depth, "arguing against a skip-specific fix".

So the ONLY unresolved question is WHY breaking registration helps L:

  H1 RELIEF   : mis-registered skip content actively SUPPRESSES the weak
                lesion evidence; translation helps by removing it.
                PREDICTION: zeroing the ROI helps as much as / more than
                translating it.   ->  Delta_Z >= Delta_T

  H2 RECOVERY : translation brings USEFUL content into a favourable decoder
                interaction -- something zeroing cannot supply.
                PREDICTION: Delta_T >> Delta_Z

  Strongest H2 signature: Delta_T > 0 AND Delta_Z < 0 (skip is not merely
  inhibitory; the RIGHT displaced content helps, removing content hurts).

THREE CONDITIONS, identical ROI geometry, skip path only (pool3 always gets
the ORIGINAL enc3 -- the E64 lesson):
  I  intact          S(x)
  T  translation     S(x+delta), exactly E188's manipulation (SHIFT=4 at enc3)
  Z  zero            S(x) := 0 inside the SAME ROI only

Primary population: the 58 L components. S/H are carried as secondary controls
because they are nearly free on the same pass.

Primary endpoint: paired Delta_T vs Delta_Z on L.
Reported per the spec: medians, means, counts, and CROSSINGS (z<0 -> z>0),
because E186.1 established that ~1 logit of significant gain can produce
almost no actual recovery.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from neuroscan_3d_v5 import UNet3D_v5
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset
import importlib.util

spec = importlib.util.spec_from_file_location(
    "t", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["e"]; spec.loader.exec_module(t); sys.argv = _a

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
PATCH, MIN_VOX, MAX_PER_SUBJ = 128, 5, 3
ROI_DIL = 8
ET = 0
SHIFT = 4          # identical to E188


def forward_from_e3(model, e3_pool, e3_skip, e1, e2):
    with torch.no_grad():
        b = model.bottleneck(model.pool3(e3_pool))
        d3 = model.dec3(torch.cat([model.upconv3(b), e3_skip], 1))
        d2 = model.dec2(torch.cat([model.upconv2(d3), e2], 1))
        eg, _ = model.attn_gate1(gate=b, skip=e1)
        d1 = model.dec1(torch.cat([model.upconv1(d2), eg], 1))
        z = model.seg_head[0](d1)
    return z[0]


def cond_translate(F, roi):
    out = F.clone()
    rolled = torch.roll(F, shifts=(SHIFT, SHIFT, SHIFT), dims=(1, 2, 3))
    m = torch.from_numpy(roi).to(F.device)
    out[:, m] = rolled[:, m]
    return out


def cond_zero(F, roi):
    out = F.clone()
    m = torch.from_numpy(roi).to(F.device)
    out[:, m] = 0.0
    return out


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for q in model.parameters():
        q.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    print(f'E189 relief-vs-recovery, enc3 SKIP ONLY, {len(ds)} subjects',
          flush=True)

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
            st = [int(np.clip(c - PATCH // 2, 0, s - PATCH))
                  for c, s in zip(com, cm.shape)]
            sl = tuple(slice(s, s + PATCH) for s in st)
            cm_p = cm[sl]
            if cm_p.sum() < MIN_VOX:
                continue
            anygt = (Y[0] | Y[1] | Y[2])[sl]; brain_p = brain[sl]
            x = img[(slice(None),) + sl].unsqueeze(0).to(dev)
            with torch.no_grad():
                e1 = model.enc1(x); e2 = model.enc2(model.pool1(e1))
                e3 = model.enc3(model.pool2(e2))
            z0 = forward_from_e3(model, e3, e3, e1, e2)
            lg0 = z0[ET].cpu().numpy()

            rng = np.random.default_rng(abs(hash((sid, g))) % (2**31))
            forbid = ndimage.binary_dilation(anygt, iterations=6)
            idx = np.argwhere(cm_p); lo = idx.min(0); ext = idx.max(0) - lo + 1
            rel = idx - lo; dims = np.array(cm_p.shape)
            best, bestv = None, -np.inf
            for _ in range(200):
                hi = dims - ext
                if (hi <= 0).any():
                    break
                off = np.array([rng.integers(0, h + 1) for h in hi])
                cand = np.zeros_like(cm_p); cand[tuple((rel + off).T)] = True
                if (cand & forbid).any() or not brain_p[cand].all():
                    continue
                v = float(lg0[cand].mean())
                if v > bestv:
                    bestv, best = v, cand
            if best is None:
                continue
            shp3 = tuple(e3.shape[2:])
            for pop, msk in [(pop0, cm_p), ('H', best)]:
                roi_in = ndimage.binary_dilation(msk, iterations=ROI_DIL) & brain_p
                r3 = (Fn.adaptive_max_pool3d(
                    torch.from_numpy(roi_in.astype(np.float32))[None, None],
                    shp3)[0, 0] > 0.5).numpy()
                if r3.sum() < 8:
                    continue
                shell = (ndimage.binary_dilation(msk, iterations=6) & ~msk
                         & brain_p & ~anygt)
                if shell.sum() < MIN_VOX:
                    continue
                rec = {'subject_id': sid, 'comp_id': int(g), 'population': pop,
                       'vox': int(msk.sum()), 'roi_e3_vox': int(r3.sum())}
                for tag, fn in [('I', None), ('T', cond_translate),
                                ('Z', cond_zero)]:
                    e3s = e3 if fn is None else fn(e3[0], r3).unsqueeze(0)
                    z = z0 if fn is None else forward_from_e3(
                        model, e3, e3s, e1, e2)
                    lg = z[ET].cpu().numpy()
                    rec[f'z_mean_{tag}'] = float(lg[msk].mean())
                    rec[f'z_med_{tag}'] = float(np.median(lg[msk]))
                    rec[f'z_max_{tag}'] = float(lg[msk].max())
                    rec[f'z_shell_{tag}'] = float(np.median(lg[shell]))
                    rec[f'dz_{tag}'] = float(np.median(lg[msk])
                                             - np.median(lg[shell]))
                    rec[f'fracpos_{tag}'] = float((lg[msk] > 0).mean())
                rows.append(rec)
        if (ii + 1) % 20 == 0:
            print(f'  {ii+1}/{len(ds)} subj, {len(rows)} rows '
                  f'({time.time()-t0:.0f}s)', flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(HERE / 'E189_relief.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    print(f'\nwrote E189_relief.csv ({len(rows)} rows) '
          f'pops={dict(Counter(r["population"] for r in rows))}  '
          f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
