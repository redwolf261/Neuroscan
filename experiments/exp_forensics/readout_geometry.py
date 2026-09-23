"""Offset-vs-direction decomposition of the terminal readout failure.

GATE 1  bias-only: can a logit-space offset on the EXISTING head recover the
        missed components, and at what cost to the bulk?
GATE 2  probe direction: fit w_probe on dec1 (leave-one-subject-out), measure
        cos(w_probe, w_seg) and the row-space fraction
            r = ||P_R(W) w_probe|| / ||w_probe||
        which connects directly to E188's Gate F.

Everything is measured on the frozen checkpoint. The probe is fitted with strict
leave-one-subject-out: the direction used to score subject s never saw s.
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


def dec1_and_logits(model, x):
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
    return dec1[0], logits[0]


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- the frozen readout ----
    W = model.seg_head[0].weight.detach()[:, :, 0, 0, 0].to(torch.float64).cpu()   # (3,32)
    b = model.seg_head[0].bias.detach().to(torch.float64).cpu()                    # (3,)
    U, S, Vh = torch.linalg.svd(W)
    print(f'seg_head W: {tuple(W.shape)}  singular values {[round(float(v),3) for v in S]}')
    # BUGFIX: torch.linalg.svd returns the FULL 32x32 Vh, so Vh.T@Vh is the
    # identity, not the row-space projector. Take only the first rank(W)=3
    # right-singular vectors. Verified: random directions then give
    # r ~ 0.285 ~ sqrt(3/32)=0.306 as theory requires (was 1.0000 before).
    k = int(torch.linalg.matrix_rank(W))
    Vr = Vh[:k]
    P_row = Vr.T @ Vr          # projector onto row space R(W), rank k
    print(f'row-space rank {int(torch.linalg.matrix_rank(W))}, ker(W) dim {W.shape[1]-int(torch.linalg.matrix_rank(W))}')
    print(f'projector idempotency check: {float((P_row@P_row - P_row).abs().max()):.2e}\n')

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    fr = list(csv.DictReader(open(OUT / 'FORENSIC_per_subject.csv')))
    tail_ids = [r['subject_id'] for r in fr
                if float(r['ET_dice']) < 0.5 or float(r['TC_dice']) < 0.5]
    idx = [i for i in range(len(ds)) if Path(ds.subject_dirs[i]).name in tail_ids]

    # ---- collect dec1 features for missed components + adjacent shells ----
    print('collecting dec1 features at missed components...', flush=True)
    data = []   # (sid, region_idx, feats_comp (n,32), feats_shell (m,32), logits_comp)
    t0 = time.time()
    for ii, i in enumerate(idx):
        img, tgt, sid = ds[i]
        Y = tgt.numpy() > 0.5
        probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                         t.PATCH, t.SW_OVERLAP, 3, dev, True)
        P = probs >= THR
        brain = img[0].numpy() != 0
        for ri, rn in enumerate(REGIONS):
            if rn == 'WT':
                continue
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
                dist = ndimage.distance_transform_edt(~cm_p)
                shell = (dist > 1) & (dist <= 6) & brain[sl] & ~anygt
                if shell.sum() < MIN_VOX:
                    continue
                x = img[(slice(None),) + sl].unsqueeze(0).to(dev)
                d1, lg = dec1_and_logits(model, x)
                d1 = d1.to(torch.float64)
                mc = torch.from_numpy(cm_p).to(dev)
                mk = torch.from_numpy(shell).to(dev)
                data.append(dict(sid=sid, ri=ri, vox=int(cm_p.sum()),
                                 fc=d1[:, mc].T.cpu(), fk=d1[:, mk].T.cpu(),
                                 lc=lg[ri][mc].to(torch.float64).cpu()))
        if (ii + 1) % 5 == 0:
            print(f'  {ii+1}/{len(idx)} ({time.time()-t0:.0f}s, {len(data)} comps)', flush=True)
    print(f'collected {len(data)} components\n')

    # =================== GATE 2: probe direction vs readout ===================
    print('=' * 78)
    print('GATE 2 -- probe direction vs the frozen readout row space')
    print('=' * 78)
    rows = []
    subs = sorted({d['sid'] for d in data})
    for ri, rn in [(0, 'ET'), (1, 'TC')]:
        w_seg = W[ri]                     # (32,)
        sel_all = [d for d in data if d['ri'] == ri]
        if len(sel_all) < 3:
            continue
        for held in subs:
            tr = [d for d in sel_all if d['sid'] != held]
            te = [d for d in sel_all if d['sid'] == held]
            if not te or len(tr) < 2:
                continue
            A = torch.cat([d['fc'] for d in tr]); B = torch.cat([d['fk'] for d in tr])
            ma, mb = A.mean(0), B.mean(0)
            va, vb = A.var(0, unbiased=False), B.var(0, unbiased=False)
            w_p = (ma - mb) / (0.5 * (va + vb) + 1e-8)      # Fisher direction
            w_p = w_p / (w_p.norm() + 1e-12)
            cos = float(torch.dot(w_p, w_seg) / (w_seg.norm() + 1e-12))
            r_row = float((P_row @ w_p).norm() / (w_p.norm() + 1e-12))
            # held-out AUC of the probe
            Ac = torch.cat([d['fc'] for d in te]); Bc = torch.cat([d['fk'] for d in te])
            sa = (Ac @ w_p).numpy(); sb = (Bc @ w_p).numpy()
            n1, n2 = len(sa), len(sb)
            rk = np.argsort(np.argsort(np.concatenate([sa, sb]))) + 1
            auc = (rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
            # AUC of the ACTUAL readout direction on the same held-out voxels
            sa2 = (Ac @ w_seg).numpy(); sb2 = (Bc @ w_seg).numpy()
            rk2 = np.argsort(np.argsort(np.concatenate([sa2, sb2]))) + 1
            auc_seg = (rk2[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
            rows.append(dict(region=rn, held_out=held, n_comp=len(te),
                             cos_probe_seg=cos, rowspace_frac=r_row,
                             probe_auc_heldout=float(max(auc, 1 - auc)),
                             segdir_auc_heldout=float(max(auc_seg, 1 - auc_seg))))
    for rn in ['ET', 'TC']:
        s = [r for r in rows if r['region'] == rn]
        if not s:
            continue
        print(f'\n  {rn}  (LOSO over {len(s)} held-out subjects)')
        print(f"    cos(w_probe, w_seg)      : mean {np.mean([r['cos_probe_seg'] for r in s]):+.4f}"
              f"   |cos| mean {np.mean([abs(r['cos_probe_seg']) for r in s]):.4f}")
        print(f"    rowspace fraction r      : mean {np.mean([r['rowspace_frac'] for r in s]):.4f}"
              f"   (r~1 => visible to W;  r<<1 => trapped in ker W)")
        print(f"    probe AUC (held-out)     : mean {np.mean([r['probe_auc_heldout'] for r in s]):.4f}")
        print(f"    seg-dir AUC (held-out)   : mean {np.mean([r['segdir_auc_heldout'] for r in s]):.4f}")
    with open(OUT / 'READOUT_geometry.csv', 'w', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows: w.writerow(r)

    # ---- random-direction baseline for r (what does r look like by chance?) ----
    g = torch.Generator().manual_seed(0)
    rr = []
    for _ in range(2000):
        v = torch.randn(32, generator=g, dtype=torch.float64); v = v / v.norm()
        rr.append(float((P_row @ v).norm()))
    print(f'\n  RANDOM-DIRECTION BASELINE for r: mean {np.mean(rr):.4f} '
          f'(95th pct {np.percentile(rr,95):.4f})  [3/32 dims => sqrt(3/32)={np.sqrt(3/32):.4f}]')

    json.dump({'singular_values': [float(v) for v in S],
               'random_r_mean': float(np.mean(rr)),
               'rows': rows}, open(OUT / 'READOUT_geometry.json', 'w'), indent=2)
    print('\nwrote READOUT_geometry.csv / .json')


if __name__ == '__main__':
    main()
