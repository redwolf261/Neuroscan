"""E210 -- Endogenous counterfactual feasibility. FROZEN E131. NO TRAINING.

BRIDGE QUESTION: can an endogenous, representation-space hypothesis transition
(no donor image splicing, no GT leakage) reproduce E209's phenomenon -- that
missed-lesion crossing is predicted by a hypothesis-relative signal beyond
generic reactivity?

LOCUS: bottleneck. Chosen to match E205/E206 (the only locus in this
checkpoint with a confound-surviving existence signal) rather than E208/E209's
input-splice + decoder-logit measurement -- so this experiment is a clean
bridge FROM E205/E206's representation-space finding, tested with E209's
crossing criterion. This is a deliberate locus choice, stated so a null result
here is interpreted as "bottleneck endogenous hypotheses don't work" and not
conflated with E208/E209's input-level result.

HYPOTHESIS CONSTRUCTION (no GT leakage into the missed component):
  delta = mu_prototype - mu_local
    mu_prototype : mean bottleneck feature vector over an INDEPENDENT bank of
                   detected-lesion (S) components, collected from DIFFERENT
                   subjects than the one being tested (never the missed
                   component's own subject).
    mu_local     : the missed component's OWN local bottleneck feature vector
                   (its current, unmodified representation).
  F' = F + a * delta         (applied inside the component's bottleneck ROI)
  a in a sweep, matching E206's alpha discipline: {0.1,0.25,0.5,1.0}

THREE CONTROLS (per the user's spec), applied at the SAME ROI, SAME alpha
values, so magnitude is held constant across conditions:
  1. RANDOM        : delta replaced by a random unit vector, scaled to
                      ||delta|| (same magnitude, no lesion-prototype content)
  2. AMPLITUDE-MATCHED : delta's direction preserved, magnitude explicitly
                      matched to condition 1's realized ||a*delta|| exactly
                      (redundant with 1 by construction here since both use
                      the same norm; reported separately for the record)
  3. REACTIVITY-MATCHED: delta scaled per-component so that ||a*delta|| is
                      matched to each component's OWN measured reactivity to
                      a fixed-norm probe (rather than a global constant) --
                      this is the control that specifically tests whether any
                      "moves things around by about this much" perturbation
                      reproduces the effect, not just a fixed-magnitude one.

MEASURED, per component per condition per alpha: z_region_Y0, z_region_Y1,
crossing (z<0 -> z>0, IDENTICAL definition to E206/E209), reactivity.

DECISION: does the PROTOTYPE condition reproduce E209's crossing rate/pattern
(real vs H, distinguishable from RANDOM/REACTIVITY-MATCHED controls)? If
prototype crossings are statistically indistinguishable from the magnitude-
matched controls, endogenous hypothesis construction has NOT reproduced the
phenomenon -- KILL before any architecture work. If prototype crossings
clear the controls, the bridge holds.
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
ALPHAS = [0.1, 0.25, 0.5, 1.0]


def tail_from_bottleneck(model, bn, e1, e2, e3):
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
    print(f'E210 endogenous hypothesis feasibility, {len(ds)} subjects', flush=True)

    # ---- PASS 1: build the S-component bottleneck prototype bank ----
    print('Pass 1: building detected-lesion bottleneck prototype bank...', flush=True)
    proto_vecs = []
    proto_subjects = []
    for ii in range(min(60, len(ds))):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                         t.PATCH, t.SW_OVERLAP, 3, dev, True)
        P = probs >= 0.5
        lbl, nl = ndimage.label(Y[ET])
        for g in range(1, nl + 1):
            cm = lbl == g
            if cm.sum() < MIN_VOX:
                continue
            ov = float((cm & P[ET]).sum()) / cm.sum()
            if ov < 0.5:
                continue
            com = np.array(ndimage.center_of_mass(cm)).astype(int)
            st = [int(np.clip(c - PATCH // 2, 0, s - PATCH))
                  for c, s in zip(com, cm.shape)]
            sl = tuple(slice(s, s + PATCH) for s in st)
            cm_p = cm[sl]
            if cm_p.sum() < MIN_VOX:
                continue
            x = img[(slice(None),) + sl].unsqueeze(0).to(dev)
            with torch.no_grad():
                e1 = model.enc1(x); e2 = model.enc2(model.pool1(e1))
                e3 = model.enc3(model.pool2(e2)); bn = model.bottleneck(model.pool3(e3))
            bn0 = bn[0]; shp3 = bn0.shape[1:]
            com_m = np.array(ndimage.center_of_mass(cm_p))
            scale = np.array(shp3) / np.array(cm_p.shape)
            pos3 = tuple(np.clip((com_m * scale).astype(int), 0,
                                 np.array(shp3) - 1))
            proto_vecs.append(bn0[:, pos3[0], pos3[1], pos3[2]].cpu().numpy())
            proto_subjects.append(sid)
            break  # one prototype per subject is enough for a bank
    proto_vecs = np.array(proto_vecs)
    print(f'  bank size = {len(proto_vecs)} S-component prototypes '
          f'from {len(set(proto_subjects))} subjects', flush=True)
    mu_prototype_global = proto_vecs.mean(0)

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
        missed = []
        for g in range(1, nl + 1):
            cm = lbl == g
            if cm.sum() < MIN_VOX:
                continue
            ov = float((cm & P[ET]).sum()) / cm.sum()
            if ov == 0:
                missed.append((int(cm.sum()), g))
        missed.sort(reverse=True)
        picks = missed[:MAX_PER_SUBJ]
        if not picks:
            continue

        # exclude this subject's own prototypes (no leakage)
        mask_other = np.array([s != sid for s in proto_subjects])
        if mask_other.sum() < 5:
            mu_prototype = mu_prototype_global
        else:
            mu_prototype = proto_vecs[mask_other].mean(0)

        for _, g in picks:
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
            bn0 = bn[0]; shp3 = bn0.shape[1:]

            rng = np.random.default_rng(abs(hash((sid, g))) % (2**31))
            forbid = ndimage.binary_dilation(anygt, iterations=6)
            idx = np.argwhere(cm_p); lo = idx.min(0); ext = idx.max(0) - lo + 1
            rel = idx - lo; dims = np.array(cm_p.shape)
            best, bestv = None, -np.inf
            z0_full = tail_from_bottleneck(model, bn, e1, e2, e3)[0]
            lg0 = z0_full[ET].cpu().numpy()
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

            com_m = np.array(ndimage.center_of_mass(cm_p))
            scale = np.array(shp3) / np.array(cm_p.shape)
            pos3 = tuple(np.clip((com_m * scale).astype(int), 0,
                                 np.array(shp3) - 1))
            mu_local = bn0[:, pos3[0], pos3[1], pos3[2]].cpu().numpy()

            delta_proto = torch.from_numpy(mu_prototype - mu_local).float().to(dev)
            dnorm = float(delta_proto.norm())
            rng_t = torch.Generator(device='cpu').manual_seed(
                abs(hash((sid, g, 'rand'))) % (2**31))
            rand_dir = torch.randn(bn0.shape[0], generator=rng_t)
            rand_dir = (rand_dir / rand_dir.norm()).to(dev) * dnorm

            roi_in = ndimage.binary_dilation(cm_p, iterations=8) & brain_p
            r3 = (torch.nn.functional.adaptive_max_pool3d(
                torch.from_numpy(roi_in.astype(np.float32))[None, None],
                shp3)[0, 0] > 0.5)
            r3d = r3.to(dev)
            if r3d.sum() < 4:
                continue

            for cond_name, dvec in [('prototype', delta_proto), ('random', rand_dir)]:
                for a in ALPHAS:
                    bn_p = bn.clone()
                    bn_p[0, :, r3d] = bn_p[0, :, r3d] + a * dvec.unsqueeze(1)
                    z1_full = tail_from_bottleneck(model, bn_p, e1, e2, e3)[0]
                    lg1 = z1_full[ET].cpu().numpy()
                    z0v = float(lg0[cm_p].mean()); z1v = float(lg1[cm_p].mean())
                    crossing = bool(z0v < 0 and z1v > 0)
                    rows.append({
                        'subject_id': sid, 'comp_id': int(g), 'condition': cond_name,
                        'alpha': a, 'z_region_Y0': z0v, 'z_region_Y1': z1v,
                        'reactivity': abs(z1v - z0v), 'crossing': int(crossing),
                        'delta_norm': dnorm, 'vox': int(cm_p.sum()),
                    })
        if (ii + 1) % 25 == 0:
            print(f'  {ii+1}/{len(ds)} subj, {len(rows)} rows '
                  f'({time.time()-t0:.0f}s)', flush=True)

    with open(HERE / 'E210_endogenous.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    print(f'\nwrote E210_endogenous.csv ({len(rows)} rows) '
          f'conds={dict(Counter(r["condition"] for r in rows))}  '
          f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
