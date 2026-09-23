"""EXPERIMENT 2 -- raw FP32 pre-sigmoid logit forensics.

Establishes the PHYSICS of the failure before any mechanism is proposed.

Critical difference from every previous run: the logits here are read directly
from seg_head[0] (the Conv3d) in float32, BEFORE the Sigmoid module. Nothing is
clamped, nothing is cast to float16, no probability round-trip. The earlier
cached fields were produced by inverting sigmoid(p) with p clipped to 1e-7,
which imposed a hard floor at logit = ln(1e-7/(1-1e-7)) = -16.118 and put 80.8%
of region medians exactly on it. That floor is an artifact of the interface, not
of the model.

Populations (identical protocol to the forensics runs):
  L : missed GT components (zero predicted overlap)
  A : adjacent 1-6 voxel background shell, confirmed non-GT, inside brain
  H : the component's OWN mask translated to the max-logit background location
      (size AND shape matched by construction)

Measured per region and per matched background:
  mu, sigma, Q10 Q25 Q50 Q75 Q90
  plus the headline quantity: DYNAMIC RANGE of the raw field.

NO training. GT used only to locate regions.
"""
import sys, csv, json, time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

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
MIN_VOX, MAX_COMP = 5, 6
QS = (10, 25, 50, 75, 90)


def sliding_raw_logits(model, image, patch, overlap, dev):
    """Sliding-window inference returning RAW FP32 pre-sigmoid logits.

    Mirrors the baseline's sliding_window_predict exactly (same patch, same
    overlap, same Gaussian blending) but taps seg_head[0] instead of the
    post-sigmoid 'probs', and keeps float32 throughout. AMP is DISABLED so no
    float16 enters the accumulation.
    """
    _, _, D, H, W = image.shape
    pd, ph, pw = patch
    stride = [max(1, int(p * (1 - overlap))) for p in patch]

    def starts(full, p, st):
        if full <= p:
            return [0]
        s = list(range(0, full - p + 1, st))
        if s[-1] != full - p:
            s.append(full - p)
        return s

    zs, ys, xs = (starts(D, pd, stride[0]), starts(H, ph, stride[1]), starts(W, pw, stride[2]))
    acc = torch.zeros((3, D, H, W), device=dev, dtype=torch.float32)
    wsum = torch.zeros((1, D, H, W), device=dev, dtype=torch.float32)
    gw = torch.from_numpy(t._gaussian_weight((min(pd, D), min(ph, H), min(pw, W)))).to(dev)

    with torch.no_grad():
        for z in zs:
            for y in ys:
                for x in xs:
                    zc, yc, xc = min(pd, D), min(ph, H), min(pw, W)
                    tile = image[:, :, z:z + zc, y:y + yc, x:x + xc]
                    if tile.shape[2:] != (pd, ph, pw):
                        tile = torch.nn.functional.pad(
                            tile, (0, pw - tile.shape[4], 0, ph - tile.shape[3], 0, pd - tile.shape[2]))
                    # full-precision forward, tap BEFORE the sigmoid
                    enc1 = model.enc1(tile); p1 = model.pool1(enc1)
                    enc2 = model.enc2(p1);   p2 = model.pool2(enc2)
                    enc3 = model.enc3(p2);   p3 = model.pool3(enc3)
                    bott = model.bottleneck(p3)
                    u3 = model.upconv3(bott)
                    dec3 = model.dec3(torch.cat([u3, enc3], 1))
                    u2 = model.upconv2(dec3)
                    dec2 = model.dec2(torch.cat([u2, enc2], 1))
                    u1 = model.upconv1(dec2)
                    eg, _ = model.attn_gate1(gate=bott, skip=enc1)
                    dec1 = model.dec1(torch.cat([u1, eg], 1))
                    raw = model.seg_head[0](dec1)          # RAW logits, no sigmoid
                    pr = raw.float()[:, :, :zc, :yc, :xc].squeeze(0)
                    acc[:, z:z + zc, y:y + yc, x:x + xc] += pr * gw
                    wsum[:, z:z + zc, y:y + yc, x:x + xc] += gw
    return (acc / wsum.clamp(min=1e-6)).cpu().numpy()


def qstats(v, prefix):
    out = {f'{prefix}_mu': float(v.mean()), f'{prefix}_sd': float(v.std()),
           f'{prefix}_min': float(v.min()), f'{prefix}_max': float(v.max()),
           f'{prefix}_n': int(v.size)}
    for q in QS:
        out[f'{prefix}_Q{q}'] = float(np.percentile(v, q))
    out[f'{prefix}_range'] = out[f'{prefix}_max'] - out[f'{prefix}_min']
    out[f'{prefix}_iqr'] = out[f'{prefix}_Q75'] - out[f'{prefix}_Q25']
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
    idx = [i for i in range(len(ds)) if Path(ds.subject_dirs[i]).name in tail]
    print(f'raw-logit extraction on {len(idx)} tail subjects (AMP OFF, fp32, no clamp)', flush=True)

    rows, vol_rows = [], []
    t0 = time.time()
    for ii, i in enumerate(idx):
        img, tgt, sid = ds[i]
        Y = tgt.numpy() > 0.5
        RAW = sliding_raw_logits(model, img.unsqueeze(0).to(dev), t.PATCH, t.SW_OVERLAP, dev)
        brain = img[0].numpy() != 0
        P = RAW > 0.0                       # decision boundary in raw logit space

        # whole-volume physics, per region
        for ri, rn in enumerate(REGIONS):
            v = RAW[ri][brain]
            vol_rows.append({'subject_id': sid, 'region': rn, **qstats(v, 'vol'),
                             'frac_at_vol_min': float(np.isclose(v, v.min()).mean()),
                             'n_unique_1dp': int(len(np.unique(np.round(v, 1))))})

        for ri, rn in [(0, 'ET'), (1, 'TC')]:
            lg = RAW[ri]
            lbl, nl = ndimage.label(Y[ri])
            miss = sorted([(int((lbl == g).sum()), g) for g in range(1, nl + 1)
                           if (lbl == g).sum() >= MIN_VOX and not ((lbl == g) & P[ri]).any()],
                          reverse=True)[:MAX_COMP]
            for sz, g in miss:
                cm = lbl == g
                anygt = Y[0] | Y[1] | Y[2]
                dist = ndimage.distance_transform_edt(~cm)
                adj = (dist > 1) & (dist <= 6) & ~anygt & brain
                if adj.sum() < MIN_VOX:
                    continue
                rng = np.random.default_rng(abs(hash((sid, rn, g))) % (2**31))
                forbid = ndimage.binary_dilation(anygt, iterations=6)
                idxs = np.argwhere(cm); lo = idxs.min(0); ext = idxs.max(0) - lo + 1
                rel = idxs - lo; dims = np.array(cm.shape)
                best, bestv = None, -np.inf
                for _ in range(300):
                    hi = dims - ext
                    if (hi <= 0).any():
                        break
                    off = np.array([rng.integers(0, h + 1) for h in hi])
                    p = rel + off
                    cand = np.zeros_like(cm); cand[tuple(p.T)] = True
                    if (cand & forbid).any() or not brain[cand].all():
                        continue
                    val = float(lg[cand].mean())
                    if val > bestv:
                        bestv, best = val, cand
                if best is None or best.sum() < MIN_VOX:
                    continue
                dh = ndimage.distance_transform_edt(~best)
                adjh = (dh > 1) & (dh <= 6) & ~anygt & brain
                if adjh.sum() < MIN_VOX:
                    continue
                for pop, mask, bgm in [('LESION', cm, adj), ('ADJ', adj, adj),
                                       ('HARDNEG', best, adjh)]:
                    zc = lg[mask].astype(np.float64)
                    zb = lg[bgm].astype(np.float64)
                    if zc.size < MIN_VOX or zb.size < MIN_VOX:
                        continue
                    rec = {'subject_id': sid, 'region': rn, 'comp_id': int(g),
                           'population': pop, 'comp_vox': int(cm.sum())}
                    rec.update(qstats(zc, 'C'))
                    rec.update(qstats(zb, 'B'))
                    rec['delta_mu'] = rec['C_mu'] - rec['B_mu']
                    rec['delta_Q50'] = rec['C_Q50'] - rec['B_Q50']
                    rows.append(rec)
        if (ii + 1) % 4 == 0:
            print(f'  {ii+1}/{len(idx)} subj, {len(rows)} rows ({time.time()-t0:.0f}s)', flush=True)

    for name, data in [('EXP2_raw_regions.csv', rows), ('EXP2_raw_volume.csv', vol_rows)]:
        keys = sorted({k for r in data for k in r})
        with open(HERE / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for r in data:
                w.writerow(r)
        print(f'wrote {name} ({len(data)} rows)')
    print(f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
