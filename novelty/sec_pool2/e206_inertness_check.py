"""E206 -- Inertness check for the Hamiltonian phase-space correction. FROZEN. NO TRAINING.

E205 passed the existence gate AT THE BOTTLENECK ONLY: K = ||grad h||^2/(h^2+eps)
distinguishes L from S/H (partial r=0.404, p=4.8e-10, surviving both boundary-
distance and magnitude controls). E3 failed G3 (collapsed to a magnitude
confound) and is excluded from this check.

Existence != actionability (E150's lesson: "readout gap real but inert").
This script tests ONLY: does h_B' = h_B*(1+alpha*K_B) move the DECODER'S
RAW LOGIT for L components, in the predicted direction, more than for S/H?

CRITICAL DESIGN POINT: K depends on h (both the spatial-gradient term and the
h^2 denominator). If K were recomputed from the PERTURBED h' at each alpha,
the sweep would be circular -- K would itself be moving as h' moves. K_B is
therefore computed ONCE from the INTACT bottleneck activation and held FIXED
across the entire alpha sweep. Only h_B is perturbed; K_B is always the
undamaged existence-gate quantity from E205.

The correction is applied GLOBALLY to h_B (the bottleneck IS the tensor that
feeds every downstream path -- there is no separate "pool vs skip" split to
worry about here, unlike E64/E186.1's enc3 shared-tensor trap). This is
therefore the correct application point structurally, not a simplification.

alpha in {-0.5,-0.25,-0.1,+0.1,+0.25,+0.5}, symmetric sweep, per component.

Measured, per the spec:
  1. DIRECTION   : dz(alpha=+small) > 0 for L?
  2. DOSE-RESPONSE: is dz(alpha) approximately monotonic?
  3. SPECIFICITY : dz_L vs dz_S vs dz_H at matched alpha
  4. CROSSING    : how many L components move z<0 -> z>0 at any alpha?

Direct component logit (NOT comp-minus-shell -- E189's contamination finding).
Frozen network. Subject-preserving permutation control on the primary
alpha=+0.25 comparison.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage, stats

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
ALPHAS = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]


def spatial_grad_sq(h):
    """Same as E205: ||grad h||^2 per spatial position, summed over channels."""
    C, D, H, W = h.shape
    gx = torch.zeros_like(h); gy = torch.zeros_like(h); gz = torch.zeros_like(h)
    gx[:, 1:-1, :, :] = (h[:, 2:, :, :] - h[:, :-2, :, :]) / 2
    gx[:, 0, :, :] = h[:, 1, :, :] - h[:, 0, :, :]
    gx[:, -1, :, :] = h[:, -1, :, :] - h[:, -2, :, :]
    gy[:, :, 1:-1, :] = (h[:, :, 2:, :] - h[:, :, :-2, :]) / 2
    gy[:, :, 0, :] = h[:, :, 1, :] - h[:, :, 0, :]
    gy[:, :, -1, :] = h[:, :, -1, :] - h[:, :, -2, :]
    gz[:, :, :, 1:-1] = (h[:, :, :, 2:] - h[:, :, :, :-2]) / 2
    gz[:, :, :, 0] = h[:, :, :, 1] - h[:, :, :, 0]
    gz[:, :, :, -1] = h[:, :, :, -1] - h[:, :, :, -2]
    g2 = (gx ** 2 + gy ** 2 + gz ** 2).sum(0)
    return g2


def tail_from_bottleneck(model, bn, e1, e2, e3):
    """bn may carry batch dim n>1; e1/e2/e3 are expanded to match."""
    n = bn.shape[0]
    with torch.no_grad():
        skip3 = e3.expand(n, -1, -1, -1, -1)
        d3 = model.dec3(torch.cat([model.upconv3(bn), skip3], 1))
        d2 = model.dec2(torch.cat([model.upconv2(d3),
                                   e2.expand(n, -1, -1, -1, -1)], 1))
        eg, _ = model.attn_gate1(gate=bn, skip=e1.expand(n, -1, -1, -1, -1))
        d1 = model.dec1(torch.cat([model.upconv1(d2), eg], 1))
        z = model.seg_head[0](d1)
    return z


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for q in model.parameters():
        q.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    print(f'E206 inertness check (bottleneck only), {len(ds)} subjects', flush=True)

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
                e3 = model.enc3(model.pool2(e2)); bn = model.bottleneck(model.pool3(e3))
            z0 = tail_from_bottleneck(model, bn, e1, e2, e3)[0]
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

            bn0 = bn[0]                        # (C,d,h,w)
            g2 = spatial_grad_sq(bn0)           # (d,h,w) fixed, from INTACT h
            hmag2 = (bn0 ** 2).sum(0)
            K = g2 / (hmag2 + EPS)              # (d,h,w), held fixed all alphas
            shp3 = bn0.shape[1:]

            for pop, msk in [(pop0, cm_p), ('H', best)]:
                com_m = np.array(ndimage.center_of_mass(msk))
                scale = np.array(shp3) / np.array(cm_p.shape)
                pos3 = tuple(np.clip((com_m * scale).astype(int), 0,
                                     np.array(shp3) - 1))
                Kv = float(K[pos3])
                rec = {'subject_id': sid, 'comp_id': int(g), 'population': pop,
                       'z_intact': float(lg0[msk].mean()), 'K_bottleneck': Kv}
                Kb = K.unsqueeze(0).unsqueeze(0)
                BS = 2   # small batch: avoid near-OOM allocator thrashing at 8GB
                for i0 in range(0, len(ALPHAS), BS):
                    chunk = ALPHAS[i0:i0 + BS]
                    bn_stack = torch.cat(
                        [bn * (1.0 + a * Kb) for a in chunk], dim=0)
                    z_stack = tail_from_bottleneck(model, bn_stack, e1, e2, e3)
                    for j, a in enumerate(chunk):
                        lg_a = z_stack[j, ET].cpu().numpy()
                        rec[f'z_a{a:+.2f}'] = float(lg_a[msk].mean())
                    del bn_stack, z_stack
                rows.append(rec)
        if (ii + 1) % 25 == 0:
            print(f'  {ii+1}/{len(ds)} subj, {len(rows)} rows '
                  f'({time.time()-t0:.0f}s)', flush=True)

    with open(HERE / 'E206_inertness.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    print(f'\nwrote E206_inertness.csv ({len(rows)} rows) '
          f'pops={dict(Counter(r["population"] for r in rows))}  '
          f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
