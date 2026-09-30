"""E213 -- Relative Evidence Readout. FROZEN E131. NO TRAINING.

HYPOTHESIS: segmentation failures may arise not because lesion evidence is
absent (E193 showed individual bottleneck voxels already carry near-perfect
L-vs-H information), but because the ABSOLUTE decision coordinate z(x)>0 is
the wrong readout. Test whether a RELATIVE score -- how z(x) or h(x) ranks
against a local reference distribution -- recovers missed lesions that
absolute z misses, without an unacceptable FP cost.

E153 WARNING, built in as a mandatory control: a normalised/relative
statistic that "looks like a strong result" can be a pure artifact of the
normalisation, not the network (E153's fractional-damage responsibility
matrix collapsed exactly this way -- an inert-scale control that should have
shown zero effect didn't, proving the structure was denominator-driven).
The analogous risk here: a voxel's rank against its neighbourhood can shift
because the NEIGHBOURHOOD changed, not because the voxel's own evidence
changed. MANDATORY INERT CONTROL below tests exactly this.

TWO NEIGHBOURHOOD DEFINITIONS (the spec doesn't fix N(x); reporting both
since the result may depend heavily on the choice):
  N_spatial  : fixed-radius local window around x, same resolution
  N_subject  : the full-brain background distribution for that subject

THREE SCORES per voxel/component, both at the LOGIT level (A/B/C from spec)
and the bottleneck-h level (D from spec):
  z(x)                        raw logit (existing decision coordinate)
  q_rank(x)  = ECDF_{N(x)}(z(x))                    rank transport
  q_robust(x)= (z(x) - median(N(x))) / (IQR(N(x))+eps)   robust standardized
  s_h(x)     = <h(x)-mu_B, mu_L-mu_B> / ||mu_L-mu_B||     bottleneck-relative
               (mu_L/mu_B from the SAME leave-subject-out prototype bank
               used throughout E204-E212, no GT leakage)

MANDATORY E153-STYLE INERT CONTROL: hold the TARGET voxel's z(x) FIXED and
resample N(x) from a matched-size population that should be statistically
equivalent (background voxels at the same depth-from-boundary, same
subject). If q(x) changes substantially under this resampling with z(x)
unchanged, the relative score is denominator-driven, not signal-driven --
exactly E153's failure mode.

OUTCOME MEASURED AS COUNTERFACTUAL DECISION SWAP, not a new forward pass:
for each component/negative already evaluated (the fixed L/H set from
E204-E212), recompute TP/FP/FN under z(x)>0 vs q(x)>tau (tau LOSO-calibrated
on the SAME populations, matching this project's threshold-search
discipline) and report Dice-proxy = 2TP/(2TP+FP+FN), not AUC alone.
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
SPATIAL_RADIUS = 8      # input-res voxels, N_spatial window


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


def ecdf(ref, val):
    return float((ref <= val).mean())


def robust_z(ref, val):
    med = float(np.median(ref))
    iqr = float(np.percentile(ref, 75) - np.percentile(ref, 25))
    return (val - med) / (iqr + EPS)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for q in model.parameters():
        q.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    print(f'E213 relative-evidence-readout, {len(ds)} subjects', flush=True)

    print('Pass 1: building detected-lesion bottleneck prototype bank...', flush=True)
    proto_vecs, proto_subjects = [], []
    bg_vecs = []
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
            # background sample: a corner of the same bottleneck map
            bg_vecs.append(bn0[:, 0, 0, 0].cpu().numpy())
            break
    proto_vecs = np.array(proto_vecs); bg_vecs = np.array(bg_vecs)
    mu_prototype_global = proto_vecs.mean(0)
    mu_bg_global = bg_vecs.mean(0)
    print(f'  bank size = {len(proto_vecs)}', flush=True)

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
        mu_bg = mu_bg_global

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

            # N_subject reference: whole-brain background logits (excl. any GT)
            bg_mask = brain_p & ~anygt
            N_subject = lg0[bg_mask]
            if len(N_subject) > 20000:
                N_subject = rng.choice(N_subject, 20000, replace=False)

            bn0 = bn[0]; shp3 = bn0.shape[1:]
            com_m0 = np.array(ndimage.center_of_mass(cm_p))

            for pop, msk in [(pop0, cm_p), ('H', best)]:
                com_m = np.array(ndimage.center_of_mass(msk))
                zc = float(lg0[msk].mean())

                # N_spatial: local window around the component, excluding
                # the component/GT itself
                lo_s = np.clip((com_m - SPATIAL_RADIUS).astype(int), 0, None)
                hi_s = np.clip((com_m + SPATIAL_RADIUS).astype(int), None,
                               np.array(cm_p.shape))
                sl_s = tuple(slice(a, b) for a, b in zip(lo_s, hi_s))
                local_mask = np.zeros_like(msk)
                local_mask[sl_s] = True
                local_mask &= brain_p & ~anygt
                N_spatial = lg0[local_mask] if local_mask.sum() >= 10 else N_subject

                q_rank_sp = ecdf(N_spatial, zc)
                q_rob_sp = robust_z(N_spatial, zc)
                q_rank_su = ecdf(N_subject, zc)
                q_rob_su = robust_z(N_subject, zc)

                # bottleneck-relative score s_h
                scale = np.array(shp3) / np.array(cm_p.shape)
                pos3 = tuple(np.clip((com_m * scale).astype(int), 0,
                                     np.array(shp3) - 1))
                h_x = bn0[:, pos3[0], pos3[1], pos3[2]].cpu().numpy()
                dir_vec = mu_prototype - mu_bg
                dir_norm = np.linalg.norm(dir_vec) + EPS
                s_h = float(np.dot(h_x - mu_bg, dir_vec) / dir_norm)

                # E153-STYLE INERT CONTROL: z(x) fixed, resample N_spatial
                # from a DIFFERENT matched-size background region (same
                # subject, same depth-from-boundary band) -- q should NOT
                # change much if it's tracking real local structure, not the
                # denominator
                bd = ndimage.distance_transform_edt(~anygt)
                target_depth = float(bd[tuple(com_m.astype(int))])
                depth_band = (np.abs(bd - target_depth) < 3) & brain_p & ~anygt
                alt_idx = np.argwhere(depth_band)
                if len(alt_idx) >= 10:
                    n_take = min(len(N_spatial), len(alt_idx))
                    sel = rng.choice(len(alt_idx), n_take, replace=False)
                    alt_coords = alt_idx[sel]
                    N_alt = lg0[tuple(alt_coords.T)]
                    q_rank_alt = ecdf(N_alt, zc)
                    q_rob_alt = robust_z(N_alt, zc)
                else:
                    q_rank_alt = q_rank_sp
                    q_rob_alt = q_rob_sp

                rows.append({
                    'subject_id': sid, 'comp_id': int(g), 'population': pop,
                    'z': zc, 'q_rank_spatial': q_rank_sp, 'q_robust_spatial': q_rob_sp,
                    'q_rank_subject': q_rank_su, 'q_robust_subject': q_rob_su,
                    's_h': s_h,
                    'q_rank_inert_alt': q_rank_alt, 'q_robust_inert_alt': q_rob_alt,
                    'vox': int(msk.sum()),
                })
        if (ii + 1) % 25 == 0:
            print(f'  {ii+1}/{len(ds)} subj, {len(rows)} rows '
                  f'({time.time()-t0:.0f}s)', flush=True)

    with open(HERE / 'E213_relative.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    print(f'\nwrote E213_relative.csv ({len(rows)} rows) '
          f'pops={dict(Counter(r["population"] for r in rows))}  '
          f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
