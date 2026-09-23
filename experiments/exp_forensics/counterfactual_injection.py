"""Counterfactual injection: does the frozen network RESPOND to a locally
injected lesion hypothesis at a missed component?

THE QUESTION (step 6 of the trace chain):
    evidence exists -> normal pathway says background -> does a counter-hypothesis
    change the internal computation?

DESIGN. For each missed component C we inject a hypothesis at an intermediate
representation and measure whether the downstream computation moves:

  A. ENCODER INJECTION. At stage l, add alpha * u_l localised to C's footprint,
     where u_l is the mean feature direction that the SAME network produces at
     stage l over CORRECTLY-DETECTED lesion tissue elsewhere in the cohort
     (a "what lesion looks like here" prototype). No GT of the target component
     is used to build u_l -- it is a population prototype from OTHER subjects.
     Measure: does the ET/TC logit at C rise, and does it cross 0?

  B. DOSE-RESPONSE. Sweep alpha. A network that has genuinely "lost" the
     component should be insensitive; a network that is merely suppressing it
     should show a graded, monotone response.

  C. SPECIFICITY CONTROL. Inject the same-norm prototype at the ADJACENT control
     shell (confirmed background). If the response is identical there, the
     network is not responding to the component specifically -- it is just
     responding to "add lesion-direction anywhere", which would be uninformative.

  D. RANDOM-DIRECTION CONTROL. Inject a norm-matched random direction at C.
     Isolates whether the prototype direction matters or any perturbation works.

Prototype u_l is built ONCE from correctly-detected components in NON-TAIL
subjects, so no target-subject GT informs it. Frozen weights throughout.
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
MAX_COMP_PER_SUBJ = 5
INJECT_STAGES = ['enc3', 'bottleneck', 'dec3']
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
N_PROTO_SUBJ = 12


def down_mask(mask, shape):
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    return Fn.adaptive_max_pool3d(m, shape)[0, 0] > 0.5


def forward_with_injection(model, x, stage=None, mask=None, vec=None, alpha=0.0):
    """Run the v5 forward pass, optionally adding alpha*vec at `stage` on `mask`."""
    def inj(feat, name):
        if stage == name and mask is not None and alpha != 0.0:
            m = down_mask(mask, tuple(feat.shape[2:])).to(feat.device)
            feat = feat + alpha * vec.view(1, -1, 1, 1, 1) * m.view(1, 1, *m.shape)
        return feat

    with torch.no_grad():
        enc1 = model.enc1(x);                       p1 = model.pool1(enc1)
        enc2 = model.enc2(p1);                      p2 = model.pool2(enc2)
        enc3 = inj(model.enc3(p2), 'enc3');         p3 = model.pool3(enc3)
        bott = inj(model.bottleneck(p3), 'bottleneck')
        u3 = model.upconv3(bott)
        dec3 = inj(model.dec3(torch.cat([u3, enc3], 1)), 'dec3')
        u2 = model.upconv2(dec3)
        dec2 = model.dec2(torch.cat([u2, enc2], 1))
        u1 = model.upconv1(dec2)
        eg, _ = model.attn_gate1(gate=bott, skip=enc1)
        dec1 = model.dec1(torch.cat([u1, eg], 1))
        logits = model.seg_head[0](dec1)
    return logits


def build_prototypes(model, ds, tail_ids, dev):
    """Mean feature direction over CORRECTLY-DETECTED ET/TC tissue in non-tail
    subjects, per stage. Target-subject GT never used."""
    acc = {s: [] for s in INJECT_STAGES}
    used = 0
    for i in range(len(ds)):
        sid = Path(ds.subject_dirs[i]).name
        if sid in tail_ids:
            continue
        img, tgt, _ = ds[i]
        Y = tgt.numpy() > 0.5
        probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                         t.PATCH, t.SW_OVERLAP, 3, dev, True)
        P = probs >= THR
        hit = (Y[0] & P[0]) | (Y[1] & P[1])      # correctly detected ET/TC
        if hit.sum() < 50:
            continue
        com = np.array(ndimage.center_of_mass(hit)).astype(int)
        st = [int(np.clip(c - PATCH // 2, 0, s - PATCH)) for c, s in zip(com, hit.shape)]
        sl = tuple(slice(s, s + PATCH) for s in st)
        hp = hit[sl]
        if hp.sum() < 20:
            continue
        x = img[(slice(None),) + sl].unsqueeze(0).to(dev)
        with torch.no_grad():
            enc1 = model.enc1(x); p1 = model.pool1(enc1)
            enc2 = model.enc2(p1); p2 = model.pool2(enc2)
            enc3 = model.enc3(p2); p3 = model.pool3(enc3)
            bott = model.bottleneck(p3)
            u3 = model.upconv3(bott)
            dec3 = model.dec3(torch.cat([u3, enc3], 1))
        feats = {'enc3': enc3, 'bottleneck': bott, 'dec3': dec3}
        bg = ~hp
        for s, f in feats.items():
            mh = down_mask(hp, tuple(f.shape[2:])).to(dev)
            mb = down_mask(bg, tuple(f.shape[2:])).to(dev)
            if mh.sum() < 2 or mb.sum() < 2:
                continue
            v = f[0][:, mh].mean(1) - f[0][:, mb].mean(1)   # lesion-minus-background
            acc[s].append(v)
        used += 1
        if used >= N_PROTO_SUBJ:
            break
    proto = {}
    for s, vs in acc.items():
        if vs:
            v = torch.stack(vs).mean(0)
            proto[s] = v / (v.norm() + 1e-8)     # unit direction
    print(f'  prototypes built from {used} non-tail subjects: '
          + ', '.join(f'{s}(dim {proto[s].numel()})' for s in proto), flush=True)
    return proto


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    fr = list(csv.DictReader(open(OUT / 'FORENSIC_per_subject.csv')))
    tail_ids = {r['subject_id'] for r in fr
                if float(r['ET_dice']) < 0.5 or float(r['TC_dice']) < 0.5}

    print('building lesion prototypes from non-tail subjects...', flush=True)
    proto = build_prototypes(model, ds, tail_ids, dev)

    idx = [i for i in range(len(ds)) if Path(ds.subject_dirs[i]).name in tail_ids]
    if limit:
        idx = idx[:limit]
    print(f'injecting into {len(idx)} tail subjects', flush=True)

    rows = []
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
                          reverse=True)[:MAX_COMP_PER_SUBJ]
            for sz, g in miss:
                cm = lbl == g
                com = np.array(ndimage.center_of_mass(cm)).astype(int)
                st = [int(np.clip(c - PATCH // 2, 0, s - PATCH)) for c, s in zip(com, cm.shape)]
                sl = tuple(slice(s, s + PATCH) for s in st)
                cm_p = cm[sl]
                if cm_p.sum() < MIN_VOX:
                    continue
                x = img[(slice(None),) + sl].unsqueeze(0).to(dev)
                anygt = (Y[0] | Y[1] | Y[2])[sl]
                dist = ndimage.distance_transform_edt(~cm_p)
                shell = (dist > 1) & (dist <= 6) & brain[sl] & ~anygt
                if shell.sum() < MIN_VOX:
                    continue
                ci = np.argwhere(shell)
                rng = np.random.default_rng(abs(hash((sid, rn, g))) % (2**31))
                pick = ci[rng.choice(len(ci), min(len(ci), int(cm_p.sum())), replace=False)]
                km = np.zeros_like(cm_p); km[tuple(pick.T)] = True
                mcf = torch.from_numpy(cm_p)

                for stg in INJECT_STAGES:
                    if stg not in proto:
                        continue
                    u = proto[stg]
                    gen = torch.Generator(device=dev); gen.manual_seed(7)
                    rv = torch.randn(u.numel(), generator=gen, device=dev)
                    rv = rv / rv.norm()
                    for alpha in ALPHAS:
                        rec = {'subject_id': sid, 'region': rn, 'comp_id': int(g),
                               'comp_vox': int(cm_p.sum()), 'stage': stg, 'alpha': alpha}
                        # A: prototype at component
                        lg = forward_with_injection(model, x, stg, cm_p, u, alpha)
                        l_c = lg[0, ri].cpu()[mcf]
                        rec['logit_mean'] = float(l_c.mean())
                        rec['logit_max'] = float(l_c.max())
                        rec['frac_above_0'] = float((l_c > 0).float().mean())
                        # C: specificity -- same injection on the control shell
                        lg_k = forward_with_injection(model, x, stg, km, u, alpha)
                        rec['ctrl_logit_mean'] = float(lg_k[0, ri].cpu()[torch.from_numpy(km)].mean())
                        # D: random direction at component
                        lg_r = forward_with_injection(model, x, stg, cm_p, rv, alpha)
                        rec['rand_logit_mean'] = float(lg_r[0, ri].cpu()[mcf].mean())
                        rows.append(rec)
        if (ii + 1) % 3 == 0:
            print(f'  {ii+1}/{len(idx)} subjects, {len(rows)} records '
                  f'({time.time()-t0:.0f}s)', flush=True)

    if not rows:
        print('NO RECORDS'); return
    keys = sorted({k for r in rows for k in r})
    with open(OUT / 'INJECTION_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print(f'\nwrote INJECTION_results.csv ({len(rows)} records) in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
