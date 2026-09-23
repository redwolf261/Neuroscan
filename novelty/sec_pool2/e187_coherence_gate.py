"""E187 -- SEC++ coherence-surplus gate at E2 -> Pool2. FROZEN MODEL, NO TRAINING.

THE ONLY QUESTION: does coherence surplus dC carry LESION-SPECIFIC information
at the E2->pool2 transition, where E185 says spatial specificity starts dying
(SCR 2.73 at E1 -> 1.39 at E2 -> 0.99 at E3)?

Per 2x2x2 window of E2, per channel:
    p     = max_i f_i                                   (winner)
    r_i   = ReLU(1 - (p - f_i)/delta)                   (winner-relative support)
    S     = (1/7) sum_{i != i*} r_i                     (support magnitude)
    C_obs = sum_{(i,j) in E} min(r_i,r_j) / sum_i r_i   (observed coherence)
    C_null= (3/7)[6r_(1)+5r_(2)+4r_(3)+3r_(4)+2r_(5)+r_(6)] / sum_i r_i
    dC    = C_obs - C_null,   dC+ = max(0, dC)

C_null is the EXACT expectation over all 7! permutations of the support values
across the 7 non-winner cube positions. Verified against brute force to 1.7e-13
for all 8 winner positions, including ties and zeros. No sampling.

The induced graph (cube minus winner) has 9 edges; under exchangeability every
unordered value pair occupies an edge with probability 9/21 = 3/7.

delta is NOT tuned: it is measured from the E2 within-window spread and
reported, so the choice is auditable.

Populations matched exactly as in E185/E186.1:
  L missed lesion / S detected lesion / H hard negative (shape+size matched,
  translated to the max-logit background location).
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
ET = 0
EPS = 1e-6
CORNERS = [(z, y, x) for z in (0, 1) for y in (0, 1) for x in (0, 1)]
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8)
         if sum(c != d for c, d in zip(CORNERS[a], CORNERS[b])) == 1]
NULL_COEF = torch.tensor([6., 5., 4., 3., 2., 1., 0.])   # on ASCENDING sorted r


def windows(F):
    """(C,D,H,W) -> (C, D/2, H/2, W/2, 8) corner-stacked 2x2x2 windows.

    Odd dimensions are cropped from the end, exactly as MaxPool3d(2) with
    ceil_mode=False discards the trailing slice."""
    C, D, H, W = F.shape
    D, H, W = D - D % 2, H - H % 2, W - W % 2
    v = F[:, :D, :H, :W].reshape(C, D // 2, 2, H // 2, 2, W // 2, 2)
    return torch.stack([v[:, :, z, :, y, :, x] for z, y, x in CORNERS], -1)


def sec_stats(F, delta):
    """p, S, C_obs, C_null, dC -- each (C, D/2, H/2, W/2)."""
    w = windows(F)
    p = w.max(-1).values
    r = torch.clamp(1.0 - (p.unsqueeze(-1) - w) / (delta + EPS), min=0.0)
    am = w.argmax(-1, keepdim=True)
    r_nw = r.scatter(-1, am, 0.0)          # winner's own support -> 0
    R = r_nw.sum(-1)
    S = R / 7.0
    # Edges incident to the winner contribute min(0, r_j) = 0 automatically,
    # so summing all 12 cube edges IS the sum over the 9 non-winner edges.
    Nobs = sum(torch.minimum(r_nw[..., a], r_nw[..., b]) for a, b in EDGES)
    # gather the 7 non-winner supports, sort ascending, apply exact coefficients
    ar = torch.arange(1, 8, device=F.device).view(*([1] * (r_nw.dim() - 1)), 7)
    r7 = r_nw.gather(-1, (am + ar) % 8)
    srt = torch.sort(r7, dim=-1).values
    Nnull = (3.0 / 7.0) * (srt * NULL_COEF.to(F.device)).sum(-1)
    C_obs = Nobs / (R + EPS)
    C_null = Nnull / (R + EPS)
    return p, S, C_obs, C_null, C_obs - C_null


def forward_all(model, x):
    """Frozen forward; returns (e1, e2, raw pre-sigmoid logits)."""
    with torch.no_grad():
        e1 = model.enc1(x)
        e2 = model.enc2(model.pool1(e1))
        e3 = model.enc3(model.pool2(e2))
        b = model.bottleneck(model.pool3(e3))
        d3 = model.dec3(torch.cat([model.upconv3(b), e3], 1))
        d2 = model.dec2(torch.cat([model.upconv2(d3), e2], 1))
        eg, _ = model.attn_gate1(gate=b, skip=e1)
        d1 = model.dec1(torch.cat([model.upconv1(d2), eg], 1))
        z = model.seg_head[0](d1)
    return e1, e2, z[0]


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for q in model.parameters():
        q.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)

    # ---- PASS 1: measure delta from the data (reported, NOT tuned) ----
    spreads = []
    for i in range(6):
        img, _, _ = ds[i]
        # same 128^3 patch geometry as the analysis, centred on the volume
        st = [max(0, (s - PATCH) // 2) for s in img.shape[1:]]
        sl = (slice(None),) + tuple(slice(a, a + PATCH) for a in st)
        with torch.no_grad():
            e2 = model.enc2(model.pool1(model.enc1(
                img[sl].unsqueeze(0).to(dev))))
        w = windows(e2[0])
        spreads.append(float((w.max(-1).values - w.min(-1).values).median()))
    DELTA = float(np.median(spreads))
    print(f'measured E2 within-window spread (median) -> delta = {DELTA:.4f}',
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
            e1, e2, zraw = forward_all(model, x)
            lg0 = zraw[ET].cpu().numpy()
            p_, S_, Co_, Cn_, dC_ = sec_stats(e2[0], DELTA)

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
            shp2 = tuple(p_.shape[1:])
            for pop, msk in [(pop0, cm_p), ('H', best)]:
                m2 = (Fn.adaptive_max_pool3d(
                    torch.from_numpy(msk.astype(np.float32))[None, None],
                    shp2)[0, 0] > 0.5).numpy()
                if m2.sum() < 1:
                    continue
                mt = torch.from_numpy(m2).to(dev)

                def agg(T):
                    v = T[:, mt].max(0).values      # strongest channel per window
                    return float(v.median()), float(v.mean())

                pm, _ = agg(p_); sm, sa = agg(S_)
                com_, _ = agg(Co_); cnm, _ = agg(Cn_); dm, da = agg(dC_)
                dpos = float((dC_[:, mt].max(0).values > 0).float().mean())
                rows.append({'subject_id': sid, 'comp_id': int(g),
                             'population': pop, 'vox': int(msk.sum()),
                             'nwin': int(m2.sum()), 'delta': DELTA,
                             'p_med': pm, 'S_med': sm, 'S_mean': sa,
                             'Cobs_med': com_, 'Cnull_med': cnm,
                             'dC_med': dm, 'dC_mean': da, 'frac_dC_pos': dpos})
        if (ii + 1) % 15 == 0:
            print(f'  {ii+1}/{len(ds)} subj, {len(rows)} rows '
                  f'({time.time()-t0:.0f}s)', flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(HERE / 'E187_coherence.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    print(f'\nwrote E187_coherence.csv ({len(rows)} rows) '
          f'pops={dict(Counter(r["population"] for r in rows))}  '
          f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
