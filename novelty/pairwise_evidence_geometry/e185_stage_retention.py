"""E185 -- Stage-wise Spatial Information Retention Audit.

At which stage does a MISSED lesion stop being spatially recoverable, and what
is lost: identity, localization, or readout accessibility?

Three populations, all matched, per component:
  L : missed GT component      (zero predicted overlap)
  S : SUCCESSFULLY detected GT component (>=50% of its voxels predicted)
  H : hard negative -- component mask translated to max-logit background
      (size AND shape matched by construction)

Stages hooked on the frozen model: E1 E2 E3 B D3 D2 D1, plus raw pre-sigmoid Z.

Per stage, per region we record:
  A. identity      : channel-mean vector -> LOSO logistic probe (L vs H, L vs S)
  B. localization  : spatial concentration ratio SCR = P(topK in C) / (|C|/|R|)
  C. diffusion     : normalised spatial entropy of the softmax response over R
  D. retention     : ||v_L - v_H|| and its ratio to the E1 value
  E. accessibility : POINTWISE 1x1x1 linear probe (channels -> mask), LOSO,
                     reported as voxel AUC and Dice inside the ROI

Everything is measured on the frozen checkpoint. Nothing is trained except the
offline linear probes, which are fit on OTHER subjects only.
"""
import sys, csv, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import ndimage, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
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
PATCH = 128
MIN_VOX, MAX_PER_SUBJ = 5, 4
STAGES = ['E1', 'E2', 'E3', 'B', 'D3', 'D2', 'D1']
# spatial downsample factor of each stage relative to input
STRIDE = {'E1': 1, 'E2': 2, 'E3': 4, 'B': 8, 'D3': 4, 'D2': 2, 'D1': 1}
ROI_DILATE = 8          # input-resolution dilation defining the ROI around C
TOPK = (0.05, 0.10, 0.25)


def capture(model, x):
    with torch.no_grad():
        e1 = model.enc1(x);  p1 = model.pool1(e1)
        e2 = model.enc2(p1); p2 = model.pool2(e2)
        e3 = model.enc3(p2); p3 = model.pool3(e3)
        b = model.bottleneck(p3)
        u3 = model.upconv3(b)
        d3 = model.dec3(torch.cat([u3, e3], 1))
        u2 = model.upconv2(d3)
        d2 = model.dec2(torch.cat([u2, e2], 1))
        u1 = model.upconv1(d2)
        eg, _ = model.attn_gate1(gate=b, skip=e1)
        d1 = model.dec1(torch.cat([u1, eg], 1))
        z = model.seg_head[0](d1)          # RAW pre-sigmoid
    return {'E1': e1[0], 'E2': e2[0], 'E3': e3[0], 'B': b[0],
            'D3': d3[0], 'D2': d2[0], 'D1': d1[0]}, z[0]


def down(mask_np, shape):
    m = torch.from_numpy(mask_np.astype(np.float32))[None, None]
    return (Fn.adaptive_max_pool3d(m, shape)[0, 0] > 0.5).numpy()


def stage_stats(F, cm_s, roi_s):
    """F: (C,d,h,w) tensor on cpu. cm_s/roi_s: bool arrays at that stage."""
    out = {}
    Cm = torch.from_numpy(cm_s); Rm = torch.from_numpy(roi_s)
    if Cm.sum() < 1 or Rm.sum() < 2:
        return None
    v = F[:, Cm].mean(1).double().numpy()              # identity vector
    out['_vec'] = v
    # response map: L2 norm across channels (magnitude of representation)
    resp = F.pow(2).sum(0).sqrt().double().numpy()
    rin = resp[roi_s]
    frac = float(cm_s.sum()) / float(roi_s.sum())
    out['chance'] = frac
    # spatial concentration at several K
    for k in TOPK:
        n = max(1, int(round(k * roi_s.sum())))
        thr = np.partition(rin, -n)[-n]
        top = (resp >= thr) & roi_s
        p = float((top & cm_s).sum()) / max(1, int(top.sum()))
        out[f'SCR{int(k*100)}'] = p / max(frac, 1e-9)
    # normalised spatial entropy over the ROI
    r = rin - rin.max()
    p = np.exp(r); p /= p.sum()
    out['H_norm'] = float(-(p * np.log(p + 1e-30)).sum() / np.log(len(p)))
    # mean response inside vs outside C (within ROI)
    out['resp_in'] = float(resp[cm_s].mean())
    out['resp_out'] = float(resp[roi_s & ~cm_s].mean())
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
    fr = list(csv.DictReader(open(ROOT / 'experiments/exp_forensics/FORENSIC_per_subject.csv')))
    tail = {r['subject_id'] for r in fr
            if float(r['ET_dice']) < 0.5 or float(r['TC_dice']) < 0.5}
    # include NON-tail subjects too -- that is where successfully detected
    # components are plentiful. L comes from tail, S from everywhere.
    idx = list(range(len(ds)))
    print(f'E185 on {len(idx)} subjects ({len(tail)} tail)', flush=True)

    rows, probe_data = [], []
    t0 = time.time()
    for ii, i in enumerate(idx):
        img, tgt, sid = ds[i]
        Y = tgt.numpy() > 0.5
        probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                         t.PATCH, t.SW_OVERLAP, 3, dev, True)
        P = probs >= 0.5
        brain = img[0].numpy() != 0
        for ri, rn in [(0, 'ET')]:                   # primary analysis: ET
            lbl, nl = ndimage.label(Y[ri])
            missed, succ = [], []
            for g in range(1, nl + 1):
                cm = lbl == g
                if cm.sum() < MIN_VOX:
                    continue
                ov = float((cm & P[ri]).sum()) / cm.sum()
                if ov == 0:
                    missed.append((int(cm.sum()), g))
                elif ov >= 0.5:
                    succ.append((int(cm.sum()), g))
            missed.sort(reverse=True); succ.sort(reverse=True)
            picks = [('L', g) for _, g in missed[:MAX_PER_SUBJ]] + \
                    [('S', g) for _, g in succ[:MAX_PER_SUBJ]]
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
                feats, zraw = capture(model, x)
                feats = {k: v.cpu() for k, v in feats.items()}
                lgp = zraw[ri].cpu().numpy()
                # hard negative: same shape, translated to max-logit background
                rng = np.random.default_rng(abs(hash((sid, rn, g))) % (2**31))
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
                    val = float(lgp[cand].mean())
                    if val > bestv:
                        bestv, best = val, cand
                if best is None:
                    continue
                for pop, msk in [(pop0, cm_p), ('H', best)]:
                    roi = ndimage.binary_dilation(msk, iterations=ROI_DILATE) & brain_p
                    if roi.sum() < 20:
                        continue
                    rec = {'subject_id': sid, 'region': rn, 'comp_id': int(g),
                           'population': pop, 'vox': int(msk.sum()),
                           'roi_vox': int(roi.sum())}
                    vecs = {}
                    okay = True
                    for s in STAGES:
                        F = feats[s]
                        shp = tuple(F.shape[1:])
                        cs = down(msk, shp); rs = down(roi, shp)
                        st_ = stage_stats(F, cs, rs)
                        if st_ is None:
                            okay = False; break
                        vecs[s] = st_.pop('_vec')
                        for k2, v2 in st_.items():
                            rec[f'{s}_{k2}'] = v2
                        # accessibility probe data: per-voxel channels + label
                        if rs.sum() >= 8:
                            Xv = F[:, torch.from_numpy(rs)].T.double().numpy()
                            yv = cs[rs].astype(int)
                            probe_data.append({'sid': sid, 'pop': pop, 'stage': s,
                                               'comp': int(g), 'X': Xv, 'y': yv})
                    if not okay:
                        continue
                    rec['_vecs'] = vecs
                    rows.append(rec)
        if (ii + 1) % 20 == 0:
            print(f'  {ii+1}/{len(idx)} subj, {len(rows)} regions ({time.time()-t0:.0f}s)', flush=True)

    # persist
    import pickle
    with open(HERE / 'E185_vectors.pkl', 'wb') as fh:
        pickle.dump([{k: r[k] for k in ('subject_id', 'population', 'comp_id', '_vecs')}
                     for r in rows], fh)
    with open(HERE / 'E185_probe.pkl', 'wb') as fh:
        pickle.dump(probe_data, fh)
    for r in rows:
        r.pop('_vecs', None)
    keys = sorted({k for r in rows for k in r})
    with open(HERE / 'E185_stages.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    print(f'\nwrote E185_stages.csv ({len(rows)} rows)  populations: '
          f'{dict(Counter(r["population"] for r in rows))}')
    print(f'probe records: {len(probe_data)}   total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
