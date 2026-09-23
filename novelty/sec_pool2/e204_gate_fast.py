"""E204 GATE (fast) -- spatial influence pattern validity at E3 / bottleneck.

Same science as e190_gate_gradient_validity.py, ~50x faster:
  * finite-difference probes are INDEPENDENT, so they are BATCHED through the
    decoder instead of run one at a time (the previous version did ~8600
    sequential forward passes; this does a few dozen batched ones)
  * one epsilon chosen per locus from a short calibration, not a 3-way sweep
    at every position
  * float32 (the TRAINING precision) is the gate; float64 runs on a reduced
    probe set purely to separate "numerical cancellation" from "MaxPool
    subgradient arbitrariness"
  * flush=True everywhere so progress is visible

PASS (all four, both loci, float32): sign>=0.95, cos>=0.95, spearman>=0.95,
relerr<=0.05.  Stratified by MaxPool tie status -- E185's root cause.
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
PATCH = 64
ET = 0


def tail_from(model, locus, h, e1, e2, e3):
    """h may carry a batch dim > 1; e1/e2/e3 are expanded to match."""
    n = h.shape[0]
    if locus == 'E3':
        b = model.bottleneck(model.pool3(h)); skip3 = h
    else:
        b = h; skip3 = e3.expand(n, -1, -1, -1, -1)
    d3 = model.dec3(torch.cat([model.upconv3(b), skip3], 1))
    d2 = model.dec2(torch.cat([model.upconv2(d3),
                               e2.expand(n, -1, -1, -1, -1)], 1))
    eg, _ = model.attn_gate1(gate=b, skip=e1.expand(n, -1, -1, -1, -1))
    d1 = model.dec1(torch.cat([model.upconv1(d2), eg], 1))
    return model.seg_head[0](d1)


def tie_map(H):
    C, D, Hh, W = H.shape
    v = H.reshape(C, D // 2, 2, Hh // 2, 2, W // 2, 2)
    w = torch.stack([v[:, :, a, :, b, :, c]
                     for a in (0, 1) for b in (0, 1) for c in (0, 1)], -1)
    mx = w.max(-1, keepdim=True).values
    return ((w == mx).sum(-1) > 1).any(0)


def probe(model, locus, H0, e1, e2, e3, cmp_, pos, eps, bs):
    """Batched central differences at the given positions."""
    out = []
    for i in range(0, len(pos), bs):
        chunk = pos[i:i + bs]
        n = len(chunk)
        hp = H0.expand(n, -1, -1, -1, -1).clone()
        hm = H0.expand(n, -1, -1, -1, -1).clone()
        for j, (a_, b_, c_) in enumerate(chunk):
            hp[j, :, a_, b_, c_] += eps
            hm[j, :, a_, b_, c_] -= eps
        with torch.no_grad():
            zp = tail_from(model, locus, hp, e1, e2, e3)[:, ET]
            zm = tail_from(model, locus, hm, e1, e2, e3)[:, ET]
            fp = zp[:, cmp_].mean(1); fm = zm[:, cmp_].mean(1)
        out.append(((fp - fm) / (2 * eps)).double().cpu().numpy())
    return np.concatenate(out)


def run(dtype, dname, npos, nsub, ck, ds, dev, rows, bs):
    model = UNet3D_v5(4, 3).to(dev).to(dtype).eval()
    model.load_state_dict(ck['model_state'])
    for q in model.parameters():
        q.requires_grad_(False)
    print('=' * 92, flush=True)
    print(f'PRECISION: {dname}' +
          ('   <-- TRAINING PRECISION, THIS IS THE GATE' if dname == 'float32'
           else '   (diagnostic only)'), flush=True)
    print('=' * 92, flush=True)
    print(f"{'subject':<16}{'locus':<11}{'n':>5}{'eps':>9}{'sign':>8}{'cos':>8}"
          f"{'spear':>8}{'relerr':>9}{'tie%':>7}{'cos|tie':>9}{'cos|no':>9}",
          flush=True)
    for si in range(nsub):
        img, tgt, sid = ds[si]
        Y = tgt.numpy() > 0.5
        lbl, nl = ndimage.label(Y[ET])
        if nl == 0:
            continue
        g = 1 + int(np.argmax([(lbl == k).sum() for k in range(1, nl + 1)]))
        cm = lbl == g
        com = np.array(ndimage.center_of_mass(cm)).astype(int)
        st = [int(np.clip(c - PATCH // 2, 0, s - PATCH))
              for c, s in zip(com, cm.shape)]
        sl = tuple(slice(a, a + PATCH) for a in st)
        cmp_ = torch.from_numpy(cm[sl]).to(dev)
        if cmp_.sum() < 5:
            continue
        x = img[(slice(None),) + sl].unsqueeze(0).to(dev).to(dtype)
        with torch.no_grad():
            e1 = model.enc1(x); e2 = model.enc2(model.pool1(e1))
            e3 = model.enc3(model.pool2(e2)); bn = model.bottleneck(model.pool3(e3))
        tmap = tie_map(e3[0]).cpu().numpy()
        for locus, H0 in [('E3', e3), ('bottleneck', bn)]:
            h0 = H0.detach().clone().requires_grad_(True)
            fv = tail_from(model, locus, h0, e1, e2, e3)[0, ET][cmp_].mean()
            signed = torch.autograd.grad(fv, h0)[0][0].sum(0)
            shp = signed.shape
            k = min(npos, signed.numel())
            top = torch.topk(signed.abs().flatten(), k).indices.cpu().numpy()
            pos = [np.unravel_index(int(i), shp) for i in top]
            ad = np.array([float(signed[p]) for p in pos])
            # calibrate eps on the first 16 positions
            cal = pos[:16]; ac = ad[:16]
            best = (None, 1e18)
            for e_ in (1e-1, 1e-2, 1e-3, 1e-4):
                q = probe(model, locus, H0, e1, e2, e3, cmp_, cal, e_, bs)
                r = float(np.median(np.abs(q - ac) / (np.abs(ac) + 1e-12)))
                if r < best[1]:
                    best = (e_, r)
            eps = best[0]
            fd = probe(model, locus, H0, e1, e2, e3, cmp_, pos, eps, bs)
            tied = np.array([bool(tmap[p[0] // 2, p[1] // 2, p[2] // 2])
                             if locus == 'E3' else False for p in pos])
            sign = float((np.sign(ad) == np.sign(fd)).mean())
            cos = float(ad @ fd / (np.linalg.norm(ad) * np.linalg.norm(fd) + 1e-30))
            sp = float(stats.spearmanr(np.abs(ad), np.abs(fd))[0])
            rel = float(np.median(np.abs(fd - ad) / (np.abs(ad) + 1e-12)))

            def c_of(m):
                if m.sum() < 3:
                    return float('nan')
                A, B = ad[m], fd[m]
                return float(A @ B / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-30))
            ct, cn = c_of(tied), c_of(~tied)
            print(f'{sid[:15]:<16}{locus:<11}{len(ad):>5}{eps:>9.0e}{sign:>8.3f}'
                  f'{cos:>8.3f}{sp:>8.3f}{rel:>9.3f}{tied.mean()*100:>7.1f}'
                  f'{ct:>9.3f}{cn:>9.3f}', flush=True)
            rows.append({'precision': dname, 'subject': sid, 'locus': locus,
                         'n': len(ad), 'eps': eps, 'sign_agree': sign,
                         'cosine': cos, 'spearman': sp, 'rel_err': rel,
                         'tie_frac': float(tied.mean()),
                         'cos_tied': ct, 'cos_untied': cn})
    print(flush=True)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    print('E204 GATE (fast) -- does the SPA influence map survive validation?',
          flush=True)
    print('PASS: sign>=0.95 cos>=0.95 spearman>=0.95 relerr<=0.05 in float32\n',
          flush=True)
    rows = []
    t0 = time.time()
    run(torch.float32, 'float32', 96, 3, ck, ds, dev, rows, bs=8)
    print(f'[float32 done at {time.time()-t0:.0f}s]', flush=True)
    try:
        run(torch.float64, 'float64', 32, 2, ck, ds, dev, rows, bs=2)
    except RuntimeError as e:
        print(f'float64 arm skipped: {str(e)[:90]}', flush=True)

    with open(HERE / 'E204_gradient_gate.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('=' * 92, flush=True)
    print('VERDICT', flush=True)
    print('=' * 92, flush=True)
    ok32 = True
    for prec in ('float32', 'float64'):
        for locus in ('E3', 'bottleneck'):
            sel = [r for r in rows if r['precision'] == prec and r['locus'] == locus]
            if not sel:
                continue
            sg = np.median([r['sign_agree'] for r in sel])
            co = np.median([r['cosine'] for r in sel])
            sp = np.median([r['spearman'] for r in sel])
            re_ = np.median([r['rel_err'] for r in sel])
            good = (sg >= 0.95) and (co >= 0.95) and (sp >= 0.95) and (re_ <= 0.05)
            if prec == 'float32':
                ok32 &= good
            print(f'  {prec:<9}{locus:<11} sign={sg:.3f} cos={co:.3f} '
                  f'spear={sp:.3f} relerr={re_:.3f}  -> '
                  f'{"USABLE" if good else "NOT TRUSTWORTHY"}', flush=True)
    ct = [r['cos_tied'] for r in rows
          if r['precision'] == 'float32' and r['locus'] == 'E3'
          and np.isfinite(r['cos_tied'])]
    cn = [r['cos_untied'] for r in rows
          if r['precision'] == 'float32' and r['locus'] == 'E3'
          and np.isfinite(r['cos_untied'])]
    if ct and cn:
        print(f'\n  E3 tie stratification (float32): cos|tied={np.median(ct):.3f}'
              f'  cos|untied={np.median(cn):.3f}', flush=True)
        print('  (large gap => E185 MaxPool tie-breaking is the cause;'
              ' small gap => broken everywhere)', flush=True)
    print(f'\n  ==> {"PASS: proceed to E204 forensic" if ok32 else "FAIL: KILL SPA -- do not patch around it"}',
          flush=True)
    print(f'\ntotal {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
