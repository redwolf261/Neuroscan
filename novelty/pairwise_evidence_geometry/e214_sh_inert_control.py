"""E214 -- Inert control for s_h (E213's bottleneck-relative score). FROZEN.

E213 found s_h = <h(x)-mu_B, mu_L-mu_B>/||mu_L-mu_B|| gives the best result
in the session: Dice-proxy 0.9175 vs baseline z's 0.8905, no FP explosion.
But the E153-style inert control was never actually run for s_h -- and a
real weakness was caught on inspection: mu_B (E213's mu_bg) was built from a
SINGLE FIXED CORNER VOXEL bn0[:,0,0,0] per subject, not a proper background
sample. That is a much thinner, more arbitrary reference than mu_prototype's
(49 independently-sourced S-component vectors averaged), and it is exactly
the kind of denominator that could make s_h's apparent signal an artifact of
WHICH corner got picked, not real background structure -- the same failure
mode that killed E213's spatial-relative scores.

THIS SCRIPT: recomputes s_h using FOUR alternative mu_B constructions, holds
h(x) and mu_L fixed, and asks whether s_h's VALUE and its DISCRIMINATIVE
POWER survive changing mu_B:
  mu_B_corner   : E213's original, bn0[:,0,0,0]  (for direct comparison)
  mu_B_opposite : the OPPOSITE corner, bn0[:,-1,-1,-1]
  mu_B_edge     : a different fixed edge voxel, bn0[:,0,-1,0]
  mu_B_spatial  : proper background average -- mean over ALL bottleneck
                  positions falling in brain-background (excl. any GT),
                  per subject -- the reference E213 SHOULD have used

If s_h changes substantially (correlation across mu_B choices well below
1.0, or discrimination/Dice-proxy collapses under mu_B_spatial specifically)
-- KILL, s_h is denominator-driven exactly like the spatial-relative scores.
If s_h is stable across all four -- s_h survives the control E213 was
missing, and the Dice-proxy result stands.
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
EPS = 1e-6


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
    print(f'E214 s_h inert control, {len(ds)} subjects', flush=True)

    print('Pass 1: building prototype bank + FOUR mu_B constructions...', flush=True)
    proto_vecs, proto_subjects = [], []
    mu_b_corner_list, mu_b_opp_list, mu_b_edge_list, mu_b_spatial_list = [], [], [], []
    for ii in range(min(60, len(ds))):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                         t.PATCH, t.SW_OVERLAP, 3, dev, True)
        P = probs >= 0.5
        brain = img[0].numpy() != 0
        lbl, nl = ndimage.label(Y[ET])
        got_proto = False
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
            anygt = (Y[0] | Y[1] | Y[2])[sl]; brain_p = brain[sl]
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

            mu_b_corner_list.append(bn0[:, 0, 0, 0].cpu().numpy())
            mu_b_opp_list.append(bn0[:, -1, -1, -1].cpu().numpy())
            mu_b_edge_list.append(bn0[:, 0, -1, 0].cpu().numpy())
            # spatial background average: bottleneck positions whose
            # corresponding input-res region is brain & not GT
            bg_mask_full = brain_p & ~anygt
            bg_ds = (torch.nn.functional.adaptive_avg_pool3d(
                torch.from_numpy(bg_mask_full.astype(np.float32))[None, None],
                shp3)[0, 0] > 0.5)
            if bg_ds.sum() >= 4:
                mu_b_spatial_list.append(
                    bn0[:, bg_ds].mean(dim=1).cpu().numpy())
            else:
                mu_b_spatial_list.append(bn0[:, 0, 0, 0].cpu().numpy())
            got_proto = True
            break
        if not got_proto:
            continue
    proto_vecs = np.array(proto_vecs)
    mu_prototype_global = proto_vecs.mean(0)
    mu_b_corner = np.mean(mu_b_corner_list, axis=0)
    mu_b_opp = np.mean(mu_b_opp_list, axis=0)
    mu_b_edge = np.mean(mu_b_edge_list, axis=0)
    mu_b_spatial = np.mean(mu_b_spatial_list, axis=0)
    print(f'  bank size = {len(proto_vecs)}', flush=True)
    print(f'  ||mu_B_corner - mu_B_spatial|| = '
          f'{np.linalg.norm(mu_b_corner - mu_b_spatial):.4f}', flush=True)
    print(f'  ||mu_B_corner|| = {np.linalg.norm(mu_b_corner):.4f}  '
          f'||mu_B_spatial|| = {np.linalg.norm(mu_b_spatial):.4f}', flush=True)

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

            # per-subject spatial background mean (matches the bank
            # construction, uses THIS subject's own background)
            bg_mask_full = brain_p & ~anygt
            bg_ds = (torch.nn.functional.adaptive_avg_pool3d(
                torch.from_numpy(bg_mask_full.astype(np.float32))[None, None],
                shp3)[0, 0] > 0.5)
            mu_b_spatial_subj = (bn0[:, bg_ds].mean(dim=1).cpu().numpy()
                                 if bg_ds.sum() >= 4 else mu_b_spatial)

            for pop, msk in [(pop0, cm_p), ('H', best)]:
                com_m = np.array(ndimage.center_of_mass(msk))
                scale = np.array(shp3) / np.array(cm_p.shape)
                pos3 = tuple(np.clip((com_m * scale).astype(int), 0,
                                     np.array(shp3) - 1))
                h_x = bn0[:, pos3[0], pos3[1], pos3[2]].cpu().numpy()

                s_h_vals = {}
                for tag, mu_b in [('corner', mu_b_corner), ('opposite', mu_b_opp),
                                  ('edge', mu_b_edge), ('spatial_global', mu_b_spatial),
                                  ('spatial_subject', mu_b_spatial_subj)]:
                    dir_vec = mu_prototype - mu_b
                    dir_norm = np.linalg.norm(dir_vec) + EPS
                    s_h_vals[tag] = float(np.dot(h_x - mu_b, dir_vec) / dir_norm)

                rows.append({
                    'subject_id': sid, 'comp_id': int(g), 'population': pop,
                    **{f's_h_{k}': v for k, v in s_h_vals.items()},
                })
        if (ii + 1) % 25 == 0:
            print(f'  {ii+1}/{len(ds)} subj, {len(rows)} rows '
                  f'({time.time()-t0:.0f}s)', flush=True)

    with open(HERE / 'E214_sh_inert.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    print(f'\nwrote E214_sh_inert.csv ({len(rows)} rows) '
          f'pops={dict(Counter(r["population"] for r in rows))}  '
          f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
