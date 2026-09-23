"""Representation-trajectory / spatial-persistence forensic probe.

Steps 2-5 of the preregistered plan, on the frozen E131_v5control_seed0 model.

POPULATIONS (three, not two -- the third is the real adversary):
  LESION : missed GT components (the 40 from the Tail Trace)
  ADJ    : matched adjacent-background shell (1-6 vox), confirmed non-GT
  HARDNEG: background regions with the HIGHEST local logit response in the same
           patch, confirmed non-GT, size-matched to the component. These are
           "things that look lesion-like to the model but are not lesions".

FEATURES per region (all from a single frozen forward pass):
  trajectory : for each stage s in [enc1,enc2,enc3,bottleneck,dec3,dec2,dec1]
               dZ_s = mean(Z_s over region) - mean(Z_s over patch background)
               M_s  = ||dZ_s||
               A_s  = cos(dZ_s, w_s)      task alignment, see below
               T_s  = cos(dZ_s, dZ_{s+1}) cross-layer coherence (shared dims only)
               turning = 1 - mean(T_s)    total trajectory turning
  spatial    : L_sigma = gaussian_filter(logit, sigma) at several scales,
               evaluated at the region, minus the same at matched background
  contrast   : plain local contrast of the logit (region mean minus a 6-vox
               dilated ring mean) -- the "am I just renaming local contrast?" control

TASK ALIGNMENT w_s. For dec1 we have the true readout row w_seg (32-d). For
other stages there is no native task direction, so we use the stage's own
Fisher direction fitted on OTHER subjects (LOSO) -- reported separately and
never mixed with the dec1 number.

GT is used only to locate regions and to label rows for scoring. Frozen weights.
"""
import sys, csv, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
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
OUT = Path(__file__).resolve().parent
THR = 0.5
PATCH = 128
MIN_VOX = 5
MAX_COMP = 6
STAGES = ['enc1', 'enc2', 'enc3', 'bottleneck', 'dec3', 'dec2', 'dec1']
SIGMAS = [0.0, 1.0, 2.0, 4.0, 8.0]


def capture(model, x):
    with torch.no_grad():
        enc1 = model.enc1(x); p1 = model.pool1(enc1)
        enc2 = model.enc2(p1); p2 = model.pool2(enc2)
        enc3 = model.enc3(p2); p3 = model.pool3(enc3)
        bott = model.bottleneck(p3)
        u3 = model.upconv3(bott)
        dec3 = model.dec3(torch.cat([u3, enc3], 1))
        u2 = model.upconv2(dec3)
        dec2 = model.dec2(torch.cat([u2, enc2], 1))
        u1 = model.upconv1(dec2)
        eg, _ = model.attn_gate1(gate=bott, skip=enc1)
        dec1 = model.dec1(torch.cat([u1, eg], 1))
        logits = model.seg_head[0](dec1)
    return (dict(enc1=enc1[0], enc2=enc2[0], enc3=enc3[0], bottleneck=bott[0],
                 dec3=dec3[0], dec2=dec2[0], dec1=dec1[0]), logits[0])


def dmask(mask_np, shape, dev):
    m = torch.from_numpy(mask_np.astype(np.float32))[None, None]
    return (Fn.adaptive_max_pool3d(m, shape)[0, 0] > 0.5).to(dev)


def region_features(feats, logit_r, mask_np, bg_np, w_seg, dev):
    """Compute trajectory + spatial features for one region."""
    out = {}
    dz = {}
    for s in STAGES:
        f = feats[s]
        shp = tuple(f.shape[1:])
        mr = dmask(mask_np, shp, dev)
        mb = dmask(bg_np, shp, dev)
        if mr.sum() < 1 or mb.sum() < 1:
            return None
        v = (f[:, mr].mean(1) - f[:, mb].mean(1)).double()
        dz[s] = v
        out[f'M_{s}'] = float(v.norm())
    out['_dz'] = {s: dz[s].cpu().numpy().tolist() for s in STAGES}   # raw vectors
    # task alignment at dec1 (true readout row)
    v1 = dz['dec1']
    out['A_dec1_seg'] = float(torch.dot(v1, w_seg) / (v1.norm() * w_seg.norm() + 1e-12))
    # cross-layer coherence on SHARED dims only (channel counts differ)
    Ts = []
    for a, b in zip(STAGES[:-1], STAGES[1:]):
        va, vb = dz[a], dz[b]
        k = min(va.numel(), vb.numel())
        c = float(torch.dot(va[:k], vb[:k]) / (va[:k].norm() * vb[:k].norm() + 1e-12))
        out[f'T_{a}_{b}'] = c
        Ts.append(c)
    out['T_mean'] = float(np.mean(Ts))
    out['turning'] = float(1.0 - np.mean(Ts))
    # spatial persistence of the logit
    lg = logit_r.detach().cpu().numpy().astype(np.float64)
    for sg in SIGMAS:
        Ls = lg if sg == 0 else ndimage.gaussian_filter(lg, sg)
        out[f'Lsig_{sg}'] = float(Ls[mask_np].mean())
        out[f'Lsig_{sg}_bg'] = float(Ls[bg_np].mean())
        out[f'Lsig_{sg}_delta'] = out[f'Lsig_{sg}'] - out[f'Lsig_{sg}_bg']
    # plain local contrast control: region mean vs 6-vox ring mean
    ring = ndimage.binary_dilation(mask_np, iterations=6) & ~ndimage.binary_dilation(mask_np, iterations=1)
    out['local_contrast'] = float(lg[mask_np].mean() - lg[ring].mean()) if ring.any() else 0.0
    out['logit_mean'] = float(lg[mask_np].mean())
    out['logit_max'] = float(lg[mask_np].max())
    out['n_vox'] = int(mask_np.sum())
    return out


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)
    W = model.seg_head[0].weight.detach()[:, :, 0, 0, 0].double()

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    fr = list(csv.DictReader(open(OUT / 'FORENSIC_per_subject.csv')))
    tail_ids = {r['subject_id'] for r in fr
                if float(r['ET_dice']) < 0.5 or float(r['TC_dice']) < 0.5}
    idx = [i for i in range(len(ds)) if Path(ds.subject_dirs[i]).name in tail_ids]

    rows = []
    t0 = time.time()
    for ii, i in enumerate(idx):
        img, tgt, sid = ds[i]
        Y = tgt.numpy() > 0.5
        probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                         t.PATCH, t.SW_OVERLAP, 3, dev, True)
        P = probs >= THR
        brain = img[0].numpy() != 0
        for ri, rn in [(0, 'ET'), (1, 'TC')]:
            w_seg = W[ri]
            lbl, nl = ndimage.label(Y[ri])
            miss = sorted([(int((lbl == g).sum()), g) for g in range(1, nl + 1)
                           if (lbl == g).sum() >= MIN_VOX and not ((lbl == g) & P[ri]).any()],
                          reverse=True)[:MAX_COMP]
            for sz, g in miss:
                cm = lbl == g
                com = np.array(ndimage.center_of_mass(cm)).astype(int)
                st = [int(np.clip(c - PATCH // 2, 0, s - PATCH)) for c, s in zip(com, cm.shape)]
                sl = tuple(slice(s, s + PATCH) for s in st)
                cm_p = cm[sl]
                if cm_p.sum() < MIN_VOX:
                    continue
                anygt = (Y[0] | Y[1] | Y[2])[sl]
                brain_p = brain[sl]
                x = img[(slice(None),) + sl].unsqueeze(0).to(dev)
                feats, lg = capture(model, x)
                lg_r = lg[ri]
                lgn = lg_r.detach().cpu().numpy().astype(np.float64)

                dist = ndimage.distance_transform_edt(~cm_p)
                adj = (dist > 1) & (dist <= 6) & brain_p & ~anygt
                if adj.sum() < MIN_VOX:
                    continue
                # patch-level background reference for dZ (far from any GT)
                far = brain_p & ~ndimage.binary_dilation(anygt, iterations=8)
                if far.sum() < 50:
                    far = brain_p & ~anygt
                if far.sum() < 50:
                    continue

                # ---- HARD NEGATIVE: highest-logit confirmed background ----
                cand = brain_p & ~ndimage.binary_dilation(anygt, iterations=4)
                if cand.sum() < cm_p.sum() * 2:
                    continue
                lg_masked = np.where(cand, lgn, -np.inf)
                # take the top-k voxels by logit, then keep the largest connected blob
                k = int(cm_p.sum())
                flat = np.argpartition(-lg_masked.ravel(), k)[:k]
                hn = np.zeros_like(cm_p); hn.ravel()[flat] = True
                if hn.sum() < MIN_VOX:
                    continue

                for pop, m in [('LESION', cm_p), ('ADJ', adj), ('HARDNEG', hn)]:
                    f = region_features(feats, lg_r, m, far, w_seg, dev)
                    if f is None:
                        continue
                    f.update({'subject_id': sid, 'region': rn, 'comp_id': int(g),
                              'population': pop, 'comp_vox': int(cm_p.sum())})
                    rows.append(f)
        if (ii + 1) % 4 == 0:
            print(f'  {ii+1}/{len(idx)} subjects, {len(rows)} rows ({time.time()-t0:.0f}s)', flush=True)

    import pickle
    with open(OUT / 'TRAJECTORY_vectors.pkl', 'wb') as fh:
        pickle.dump([{k: r[k] for k in ('subject_id','region','comp_id','population','_dz')}
                     for r in rows], fh)
    for r in rows:
        r.pop('_dz', None)
    keys = sorted({k for r in rows for k in r})
    with open(OUT / 'TRAJECTORY_features.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print(f'\nwrote TRAJECTORY_features.csv ({len(rows)} rows) in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
