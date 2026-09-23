"""E205 -- Hamiltonian phase-space correction, EXISTENCE GATE. FROZEN. NO TRAINING.

CANDIDATE (user, 2026-09-22): h' = h * (1 + alpha * K),  K = ||grad h||^2 / (h^2+eps)

This script tests ONLY whether K(x) carries L-vs-S-vs-H discriminating signal at
E3/bottleneck, on the SAME populations used throughout this session (58 L /
119 S / 219 H, matched shape+size, translated to max-logit background).

TWO PRIOR RESULTS THIS GATE MUST NOT IGNORE:

1. E185 killed DIFFERENTIATING THE DECODER (d f/d h through MaxPool ties).
   K here uses d h/d x -- the SPATIAL gradient of the activation map itself,
   computed by finite differences across voxels, NEVER through the decoder.
   Different object, not blocked by E185. Confirmed distinct before writing
   this script.

2. E179 killed a STRUCTURALLY IDENTICAL statistic -- R(x) = ||grad p|| /
   (|p-tau|+eps) -- applied to OUTPUT probabilities. It failed because R
   collapsed to a boundary-proximity confound: S3(=R) beat plain boundary
   distance in only 22% of matched-distance bins, decisively below the 80%
   bar. K(x) here is the same functional FORM one layer earlier (h instead
   of p). The gate below is therefore STRICTER than a naive L/S/H comparison:
   it explicitly tests whether K reduces to boundary-proximity, exactly the
   confound that killed E179.

GATE (all required):
  G1  K_L significantly different from K_S (Mann-Whitney, matched to E188's
      normalisation-by-baseline discipline)
  G2  the difference SURVIVES a boundary-proximity partial control (E179's
      killer) -- i.e. K adds information beyond "how close to the lesion
      boundary is this voxel"
  G3  the difference SURVIVES controlling for |h| (activation magnitude) --
      otherwise K is just re-deriving "small activations are noisy"
  G4  subject-preserving permutation null (200 iterations) confirms G1 is not
      a component-count artifact

KILL if K_L ~= K_S ~= K_H, or if G2/G3 fail (confound reproduces E179's
death, one layer earlier).
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


def spatial_grad_sq(h):
    """h: (C,D,H,W). Returns ||grad h||^2 per spatial position, SUMMED over
    channels (matches how K would gate the whole channel fibre at x), via
    central differences with replicate padding at boundaries."""
    C, D, H, W = h.shape
    hp = h.unsqueeze(0)
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
    g2 = (gx ** 2 + gy ** 2 + gz ** 2).sum(0)          # (D,H,W), summed over C
    hmag2 = (h ** 2).sum(0)                             # (D,H,W)
    return g2, hmag2


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for q in model.parameters():
        q.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    print(f'E205 kinetic-energy existence gate, {len(ds)} subjects', flush=True)

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

            rng = np.random.default_rng(abs(hash((sid, g))) % (2**31))
            forbid = ndimage.binary_dilation(anygt, iterations=6)
            idx = np.argwhere(cm_p); lo = idx.min(0); ext = idx.max(0) - lo + 1
            rel = idx - lo; dims = np.array(cm_p.shape)
            best, bestv = None, -np.inf
            with torch.no_grad():
                d3 = model.dec3(torch.cat([model.upconv3(bn), e3], 1))
                d2 = model.dec2(torch.cat([model.upconv2(d3), e2], 1))
                eg, _ = model.attn_gate1(gate=bn, skip=e1)
                d1 = model.dec1(torch.cat([model.upconv1(d2), eg], 1))
                lg0 = model.seg_head[0](d1)[0, ET].cpu().numpy()
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

            # boundary-distance map (input-res), for the confound control
            bd = ndimage.distance_transform_edt(~anygt)

            for pop, msk in [(pop0, cm_p), ('H', best)]:
                com_m = np.array(ndimage.center_of_mass(msk))
                for locus, H0 in [('E3', e3[0]), ('bottleneck', bn[0])]:
                    shp3 = H0.shape[1:]
                    scale = np.array(shp3) / np.array(cm_p.shape)
                    pos3 = np.clip((com_m * scale).astype(int), 0,
                                   np.array(shp3) - 1)
                    g2, hmag2 = spatial_grad_sq(H0)
                    g2v = float(g2[pos3[0], pos3[1], pos3[2]])
                    hv = float(hmag2[pos3[0], pos3[1], pos3[2]])
                    K = g2v / (hv + EPS)
                    bdist = float(bd[tuple(com_m.astype(int))])
                    rows.append({'subject_id': sid, 'comp_id': int(g),
                                 'population': pop, 'locus': locus,
                                 'K': K, 'grad_sq': g2v, 'h_mag_sq': hv,
                                 'boundary_dist': bdist,
                                 'vox': int(msk.sum())})
        if (ii + 1) % 25 == 0:
            print(f'  {ii+1}/{len(ds)} subj, {len(rows)} rows '
                  f'({time.time()-t0:.0f}s)', flush=True)

    with open(HERE / 'E205_kinetic.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    print(f'\nwrote E205_kinetic.csv ({len(rows)} rows) '
          f'pops={dict(Counter(r["population"] for r in rows))}  '
          f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
