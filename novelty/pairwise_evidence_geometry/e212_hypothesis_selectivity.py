"""E212 -- Hypothesis-Response Selectivity. FROZEN E131. NO TRAINING.

Tests whether the useful E194-E196 signal is DIFFERENTIAL response to a
lesion-consistent hypothesis versus a magnitude-matched COMPETING hypothesis
(not just raw response magnitude, which E197's evaluator comparison already
showed dominates the naive feature set).

S_H = Delta_L - Delta_B
  Delta_L = z(F + alpha*H_L) - z(F)     [E210's prototype direction, VERBATIM]
  Delta_B = z(F + alpha*H_B) - z(F)     [H_B orthogonal to H_L, same norm]

H_B construction (frozen BEFORE outcomes are seen, per the user's no-
cherry-picking requirement): 3 independent random directions per component,
Gram-Schmidt-orthogonalized against H_L, renormalized to ||H_L||. Seeded
deterministically from (subject_id, comp_id, control_index) -- reproducible,
not selected post-hoc.

STRUCTURE, exactly as specified:
  E212-A: reproduce E210 (formerly E196) EXACTLY first. Mandatory gate:
          27/74 any-alpha crossings on the prototype condition. STOP if this
          does not reproduce -- do not proceed to interpretation.
  E212-B: construct H_B (orthogonal, norm-matched, 3 controls/component)
  E212-C: R_L=Delta_L, R_B=Delta_B(mean over 3 controls), S_H=R_L-R_B,
          R_mag=|Delta_L|+|Delta_B|
  E212-D: primary test S_H(L) vs S_H(H) -- Mann-Whitney, subject permutation,
          effect size, bootstrap CI
  E212-E: S_H -> crossing, subject-grouped CV, AUC/PR-AUC vs |Delta_L| alone
  E212-F: reactivity control -- logistic beta_1(S_H) after controlling for
          R_mag; partial corr(S_H, crossing | R_mag)
  E212-G: negative control -- same machinery on H components

NO GT used in constructing H_L or H_B (H_L = E210's mu_prototype - mu_local,
which only uses OTHER subjects' detected-lesion bottleneck vectors and this
component's OWN current representation -- verified against E210's code,
unchanged here). GT used only for L/H labels and crossing definition,
exactly as before.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
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
ALPHAS = [0.1, 0.25, 0.5, 1.0]     # IDENTICAL to E210
N_CONTROLS = 3


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


def orthogonal_control(h_l_np, seed):
    """Gram-Schmidt a random draw against h_l_np, renormalize to ||h_l_np||.
    Deterministic given seed -- frozen before any outcome is observed."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(h_l_np.shape[0])
    u = h_l_np / (np.linalg.norm(h_l_np) + 1e-8)
    v_orth = v - np.dot(v, u) * u
    v_orth = v_orth / (np.linalg.norm(v_orth) + 1e-8)
    return v_orth * np.linalg.norm(h_l_np)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for q in model.parameters():
        q.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    print(f'E212 hypothesis-response selectivity, {len(ds)} subjects', flush=True)

    # ---- Pass 1: prototype bank, IDENTICAL to E210 ----
    print('Pass 1: building detected-lesion bottleneck prototype bank...', flush=True)
    proto_vecs, proto_subjects = [], []
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
            pos3 = tuple(np.clip((com_m * scale).astype(int), 0, np.array(shp3) - 1))
            proto_vecs.append(bn0[:, pos3[0], pos3[1], pos3[2]].cpu().numpy())
            proto_subjects.append(sid)
            break
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
        picks = [('L', g) for _, g in missed[:MAX_PER_SUBJ]]
        if not picks:
            continue

        mask_other = np.array([s != sid for s in proto_subjects])
        mu_prototype = (proto_vecs[mask_other].mean(0) if mask_other.sum() >= 5
                        else mu_prototype_global)

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
            bn0 = bn[0]; shp3 = bn0.shape[1:]
            z0_full = tail_from_bottleneck(model, bn, e1, e2, e3)[0]
            lg0 = z0_full[ET].cpu().numpy()

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

            com_m = np.array(ndimage.center_of_mass(cm_p))
            scale = np.array(shp3) / np.array(cm_p.shape)
            pos3 = tuple(np.clip((com_m * scale).astype(int), 0, np.array(shp3) - 1))
            mu_local = bn0[:, pos3[0], pos3[1], pos3[2]].cpu().numpy()

            h_l = mu_prototype - mu_local        # H_L, IDENTICAL to E210
            h_l_t = torch.from_numpy(h_l).float().to(dev)

            roi_in = ndimage.binary_dilation(cm_p, iterations=8) & brain_p
            r3 = (torch.nn.functional.adaptive_max_pool3d(
                torch.from_numpy(roi_in.astype(np.float32))[None, None],
                shp3)[0, 0] > 0.5)
            r3d = r3.to(dev)
            if r3d.sum() < 4:
                continue

            for pop, msk in [(pop0, cm_p), ('H', best)]:
                for a in ALPHAS:
                    bn_L = bn.clone()
                    bn_L[0, :, r3d] = bn_L[0, :, r3d] + a * h_l_t.unsqueeze(1)
                    z1L = tail_from_bottleneck(model, bn_L, e1, e2, e3)[0]
                    lgL = z1L[ET].cpu().numpy()
                    z0v = float(lg0[msk].mean()); zLv = float(lgL[msk].mean())
                    dL = zLv - z0v
                    crossing = bool(z0v < 0 and zLv > 0)

                    dBs = []
                    for ci in range(N_CONTROLS):
                        seed = abs(hash((sid, g, pop, ci))) % (2**31)
                        h_b = orthogonal_control(h_l, seed)
                        h_b_t = torch.from_numpy(h_b).float().to(dev)
                        bn_B = bn.clone()
                        bn_B[0, :, r3d] = bn_B[0, :, r3d] + a * h_b_t.unsqueeze(1)
                        z1B = tail_from_bottleneck(model, bn_B, e1, e2, e3)[0]
                        lgB = z1B[ET].cpu().numpy()
                        dBs.append(float(lgB[msk].mean()) - z0v)
                    dB_mean = float(np.mean(dBs))

                    S_H = dL - dB_mean
                    R_mag = abs(dL) + abs(dB_mean)
                    rows.append({
                        'subject_id': sid, 'comp_id': int(g), 'population': pop,
                        'alpha': a, 'z0': z0v, 'delta_L': dL, 'delta_B': dB_mean,
                        'S_H': S_H, 'R_mag': R_mag, 'crossing': int(crossing),
                        'vox': int(msk.sum()),
                    })
        if (ii + 1) % 25 == 0:
            print(f'  {ii+1}/{len(ds)} subj, {len(rows)} rows '
                  f'({time.time()-t0:.0f}s)', flush=True)

    with open(HERE / 'E212_selectivity.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    print(f'\nwrote E212_selectivity.csv ({len(rows)} rows) '
          f'pops={dict(Counter(r["population"] for r in rows))}  '
          f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
