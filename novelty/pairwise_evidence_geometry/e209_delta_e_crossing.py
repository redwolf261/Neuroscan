"""E209 -- Does Delta_E predict actual segmentation crossing? FROZEN. NO TRAINING.

E208 PASSED: splicing donor tumor texture into a missed lesion's location
shifts the SHELL's prediction toward lesion-consistent (Delta_E), and this
survives a reactivity control (partial r=0.297, p=5.0e-07) -- the first
candidate this session (after E205/E206/E207 all failed) to clear that bar.

E209 asks the harder question your own framing identified: is Delta_E a real
diagnostic that is nonetheless INERT for actually recovering a missed lesion?
That is exactly the trap E205's bottleneck-K existence signal (partial
r=0.404) fell into at E206 (0/74 crossings, wrong sign).

REUSES E208's exact procedure (same donor bank, same splicing, same Y0/Y1
construction) so nothing is re-derived under a different definition. The only
addition is: does the REGION's OWN logit (not the shell's) cross z<0 -> z>0
after the SAME insertion, using the IDENTICAL crossing definition as E206
(component mean logit, z_intact<0 -> z_post>0) so the two results are
directly comparable on the same scale as the rest of this session's ladder
(E186.1: 1/58, E189: 5/57, E206: 0/74).

MEASURED per component: Delta_E (shell), z_region_Y0, z_region_Y1 (crossing
source), reactivity, crossing (bool).

TESTS (user's three, run for L only and, separately, for H as the negative
control -- per the user's explicit requirement to check Delta_E predicts
crossing ONLY for L, not "large perturbations cause large changes" generically):
  1. Delta_E^cross vs Delta_E^noncross (Mann-Whitney + magnitudes)
  2. LOSO/subject-grouped AUC(Delta_E -> crossing)
  3. partial correlation(Delta_E, crossing | reactivity)

KILL if: no separation, AUC ~ chance, or the association vanishes under the
reactivity control -- OR if H shows the same Delta_E->crossing relationship
as L (would mean it's generic perturbation-magnitude, not hypothesis-value).
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage, stats
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

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


def raw_logits(model, x):
    with torch.no_grad():
        e1 = model.enc1(x); e2 = model.enc2(model.pool1(e1))
        e3 = model.enc3(model.pool2(e2)); bn = model.bottleneck(model.pool3(e3))
        d3 = model.dec3(torch.cat([model.upconv3(bn), e3], 1))
        d2 = model.dec2(torch.cat([model.upconv2(d3), e2], 1))
        eg, _ = model.attn_gate1(gate=bn, skip=e1)
        d1 = model.dec1(torch.cat([model.upconv1(d2), eg], 1))
        z = model.seg_head[0](d1)
    return z[0]


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for q in model.parameters():
        q.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    print(f'E209 Delta_E -> crossing, {len(ds)} subjects', flush=True)

    donors = []
    for i in range(15):
        img, tgt, sid = ds[i]
        Y = tgt.numpy() > 0.5
        lbl, nl = ndimage.label(Y[ET])
        if nl == 0:
            continue
        sizes = [(int((lbl == g).sum()), g) for g in range(1, nl + 1)]
        sizes.sort(reverse=True)
        g = sizes[0][1]
        cm = lbl == g
        com = np.array(ndimage.center_of_mass(cm)).astype(int)
        r = 6
        lo = np.clip(com - r, 0, np.array(img.shape[1:]) - 2 * r)
        patch = img[:, lo[0]:lo[0]+2*r, lo[1]:lo[1]+2*r, lo[2]:lo[2]+2*r].numpy()
        donors.append(patch)
    print(f'  built {len(donors)} donor texture patches', flush=True)

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
        picks = [('L', g) for _, g in missed[:MAX_PER_SUBJ]]
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
            img_p = img[(slice(None),) + sl].clone()
            z0 = raw_logits(model, img_p.unsqueeze(0).to(dev))
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

            for pop, msk in [(pop0, cm_p), ('H', best)]:
                shell = (ndimage.binary_dilation(msk, iterations=6) & ~msk
                         & brain_p & ~anygt)
                if shell.sum() < MIN_VOX:
                    continue
                z_shell_Y0 = float(lg0[shell].mean())
                z_region_Y0 = float(lg0[msk].mean())

                com_m = np.array(ndimage.center_of_mass(msk)).astype(int)
                donor = donors[rng.integers(0, len(donors))]
                r = donor.shape[1] // 2
                img_p1 = img_p.clone()
                lo3 = np.clip(com_m - r, 0, np.array(img_p.shape[1:]) - 2 * r)
                sl3 = tuple(slice(a, a + 2 * r) for a in lo3)
                donor_t = torch.from_numpy(donor)
                for c in range(img_p.shape[0]):
                    img_p1[c][sl3] = donor_t[c]

                z1 = raw_logits(model, img_p1.unsqueeze(0).to(dev))
                lg1 = z1[ET].cpu().numpy()
                z_shell_Y1 = float(lg1[shell].mean())
                z_region_Y1 = float(lg1[msk].mean())

                crossing = bool(z_region_Y0 < 0 and z_region_Y1 > 0)
                rows.append({
                    'subject_id': sid, 'comp_id': int(g), 'population': pop,
                    'delta_E': z_shell_Y1 - z_shell_Y0,
                    'z_region_Y0': z_region_Y0, 'z_region_Y1': z_region_Y1,
                    'reactivity': abs(z_region_Y1 - z_region_Y0),
                    'crossing': int(crossing),
                    'vox': int(msk.sum()),
                })
        if (ii + 1) % 25 == 0:
            print(f'  {ii+1}/{len(ds)} subj, {len(rows)} rows '
                  f'({time.time()-t0:.0f}s)', flush=True)

    with open(HERE / 'E209_crossing.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    ncross = sum(r['crossing'] for r in rows if r['population'] == 'L')
    nL = sum(1 for r in rows if r['population'] == 'L')
    print(f'\nwrote E209_crossing.csv ({len(rows)} rows) '
          f'pops={dict(Counter(r["population"] for r in rows))}  '
          f'L crossings={ncross}/{nL}  {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
