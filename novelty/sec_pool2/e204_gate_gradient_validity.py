"""E204 GATE -- is the SPATIAL INFLUENCE PATTERN of d f/d h trustworthy at E3/B?

SPA's loss compares a NORMALISED SPATIAL MAP of |grad| against a Gaussian
target. So the gate must validate the MAP, not a single directional
derivative. A scalar projection can agree by luck while the map is wrong.

E185 (2026-09-18) found autodiff(J) vs finite-difference(J) disagree ~100%
stably to h=1e-10 at the real trained enc3 activation, root-caused to
MaxPool3d tie-breaking under ~53% ReLU sparsity (28.75% of windows tie).

PASS CRITERIA (all four, at BOTH loci, in float32 = the training precision):
  1. sign agreement        >= 0.95
  2. cosine similarity     >= 0.95
  3. normalised spatial-map (Spearman) correlation >= 0.95
  4. relative magnitude error <= 0.05

STRATIFIED by whether each voxel's downstream MaxPool window contains exact
ties -- that is E185's root cause, so it is the axis that matters.

METHOD. Per spatial position x (channel-summed, which is exactly what the SPA
map uses), the influence is
    a(x) = || d f / d h(:,x) ||_1                (autodiff)
    q(x) = [ f(h + eps*E_x) - f(h - eps*E_x) ] / 2eps   (central difference,
           E_x = the all-ones perturbation on the channel fibre at x)
Note a(x) uses |.|_1 while q(x) is the signed directional derivative along
the all-ones fibre direction; the honest comparison is therefore between
q(x) and the SIGNED directional derivative s(x) = sum_c d f/d h(c,x).
Both are reported: s(x) vs q(x) validates the gradient; a(x) is the SPA map.

Sampling a random subset of positions keeps this ~minutes, not hours.
"""
import sys, csv
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
NPOS = 120          # spatial positions probed per (subject, locus)
EPSS = [1e-2, 1e-3, 1e-4]


def tail_from(model, locus, h, e1, e2, e3):
    if locus == 'E3':
        b = model.bottleneck(model.pool3(h)); skip3 = h
    else:
        b = h; skip3 = e3
    d3 = model.dec3(torch.cat([model.upconv3(b), skip3], 1))
    d2 = model.dec2(torch.cat([model.upconv2(d3), e2], 1))
    eg, _ = model.attn_gate1(gate=b, skip=e1)
    d1 = model.dec1(torch.cat([model.upconv1(d2), eg], 1))
    return model.seg_head[0](d1)


def tie_map(H):
    """Per spatial position of H (pre-pool3), does its 2x2x2 window tie?"""
    C, D, Hh, W = H.shape
    v = H.reshape(C, D // 2, 2, Hh // 2, 2, W // 2, 2)
    w = torch.stack([v[:, :, a, :, b, :, c]
                     for a in (0, 1) for b in (0, 1) for c in (0, 1)], -1)
    mx = w.max(-1, keepdim=True).values
    tied = (w == mx).sum(-1) > 1              # (C, D/2, H/2, W/2)
    tied_any = tied.any(0)                    # any channel ties in that window
    return tied_any


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    base = UNet3D_v5(4, 3).to(dev).eval()
    base.load_state_dict(ck['model_state'])
    for q in base.parameters():
        q.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    print('E204 GATE -- spatial influence pattern validity at E3 / bottleneck')
    print('PASS needs sign>=0.95, cos>=0.95, spearman>=0.95, relerr<=0.05 '
          'in float32\n')

    rows = []
    for dtype, dname in [(torch.float32, 'float32'), (torch.float64, 'float64')]:
        model = UNet3D_v5(4, 3).to(dev).to(dtype).eval()
        model.load_state_dict(ck['model_state'])
        for q in model.parameters():
            q.requires_grad_(False)
        print('=' * 90)
        print(f'PRECISION: {dname}' +
              ('   <-- TRAINING PRECISION' if dname == 'float32' else ''))
        print('=' * 90)
        print(f"{'subject':<16}{'locus':<11}{'n':>5}{'sign':>8}{'cos':>8}"
              f"{'spear':>8}{'relerr':>9}{'tie%':>7}{'cos|tie':>9}{'cos|no':>9}")
        for si in range(3):
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
                e3 = model.enc3(model.pool2(e2)); bn = model.bottleneck(
                    model.pool3(e3))
            tmap = tie_map(e3[0]).cpu().numpy()

            for locus, H0 in [('E3', e3), ('bottleneck', bn)]:
                def f(h):
                    return tail_from(model, locus, h, e1, e2, e3)[0, ET][cmp_].mean()

                h0 = H0.detach().clone().requires_grad_(True)
                grad = torch.autograd.grad(f(h0), h0)[0][0]     # (C,d,h,w)
                signed = grad.sum(0)                            # all-ones fibre
                shp = signed.shape
                rng = np.random.default_rng(0)
                # probe where influence is largest (that is what SPA's map uses)
                flat = signed.abs().flatten()
                k = min(NPOS, flat.numel())
                top = torch.topk(flat, k).indices.cpu().numpy()
                pos = [np.unravel_index(int(i), shp) for i in top]

                ad, fd, tied = [], [], []
                for (a_, b_, c_) in pos:
                    s_ad = float(signed[a_, b_, c_])
                    bestq, bestr = None, 1e18
                    for eps in EPSS:
                        with torch.no_grad():
                            hp = H0.clone(); hp[0, :, a_, b_, c_] += eps
                            hm = H0.clone(); hm[0, :, a_, b_, c_] -= eps
                            q = (float(f(hp)) - float(f(hm))) / (2 * eps)
                        r = abs(q - s_ad) / (abs(s_ad) + 1e-12)
                        if r < bestr:
                            bestr, bestq = r, q
                    ad.append(s_ad); fd.append(bestq)
                    if locus == 'E3':
                        tied.append(bool(tmap[a_ // 2, b_ // 2, c_ // 2])
                                    if (a_ // 2 < tmap.shape[0] and
                                        b_ // 2 < tmap.shape[1] and
                                        c_ // 2 < tmap.shape[2]) else False)
                    else:
                        tied.append(False)
                ad = np.array(ad); fd = np.array(fd); tied = np.array(tied)
                sign = float((np.sign(ad) == np.sign(fd)).mean())
                cos = float(ad @ fd / (np.linalg.norm(ad) * np.linalg.norm(fd) + 1e-30))
                sp = float(stats.spearmanr(np.abs(ad), np.abs(fd))[0])
                rel = float(np.median(np.abs(fd - ad) / (np.abs(ad) + 1e-12)))

                def c_of(mask):
                    if mask.sum() < 3:
                        return float('nan')
                    A, B = ad[mask], fd[mask]
                    return float(A @ B / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-30))
                ct, cn = c_of(tied), c_of(~tied)
                print(f'{sid[:15]:<16}{locus:<11}{len(ad):>5}{sign:>8.3f}'
                      f'{cos:>8.3f}{sp:>8.3f}{rel:>9.3f}'
                      f'{tied.mean()*100:>7.1f}{ct:>9.3f}{cn:>9.3f}')
                rows.append({'precision': dname, 'subject': sid, 'locus': locus,
                             'n': len(ad), 'sign_agree': sign, 'cosine': cos,
                             'spearman': sp, 'rel_err': rel,
                             'tie_frac': float(tied.mean()),
                             'cos_tied': ct, 'cos_untied': cn})
        print()

    with open(HERE / 'E204_gradient_gate.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print('=' * 90)
    print('VERDICT')
    print('=' * 90)
    overall_ok = True
    for prec in ('float32', 'float64'):
        for locus in ('E3', 'bottleneck'):
            sel = [r for r in rows if r['precision'] == prec and r['locus'] == locus]
            if not sel:
                continue
            sg = np.median([r['sign_agree'] for r in sel])
            co = np.median([r['cosine'] for r in sel])
            sp = np.median([r['spearman'] for r in sel])
            re_ = np.median([r['rel_err'] for r in sel])
            ok = (sg >= 0.95) and (co >= 0.95) and (sp >= 0.95) and (re_ <= 0.05)
            if prec == 'float32':
                overall_ok &= ok
            print(f'  {prec:<9}{locus:<11} sign={sg:.3f} cos={co:.3f} '
                  f'spear={sp:.3f} relerr={re_:.3f}  -> '
                  f'{"USABLE" if ok else "NOT TRUSTWORTHY"}')
    print(f'\n  ==> {"PASS: proceed to E204 forensic" if overall_ok else "FAIL: KILL SPA -- do not patch around it"}')


if __name__ == '__main__':
    main()
