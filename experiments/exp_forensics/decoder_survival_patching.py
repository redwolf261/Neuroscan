"""Decoder Displacement Survival + Causal Patching.

PART A (observational): how does the region-vs-background displacement dZ
survive dec3 -> dec2 -> dec1?
  - magnitude retention        ||dZ_{s+1}|| / ||dZ_s||
  - parallel / perpendicular   decomposition of dZ_{s+1} w.r.t. the image of
                               dZ_s under the (linear) upsampling operator
  - skip vs upsample split     cat2 = [upconv2(dec3) | enc2]  (64 = 64 + 64)
                               cat1 = [upconv1(dec2) | enc1_gated] (64 = 32 + 32)
                               so the concatenated displacement splits exactly.

PART B (causal): patch dec3 and measure the downstream logit response.
  P1 donor-swap : replace dec3 over the region with the dec3 of a LESION donor
                  (for H) / of a HARDNEG donor (for L). Tests whether dec3
                  displacement CONTROLS the downstream outcome.
  P2 scaling    : dec3_region <- bg + k*(dec3_region - bg), k in {0,.5,1,2,4}.
                  Dose-response on the displacement itself.
  P3 transplant : give a HARDNEG region the LESION mean displacement vector
                  (population prototype, built LOSO from OTHER subjects).

Decision rule (preregistered here, before looking at results):
  CONFIRM if scaling dec3 displacement moves the dec1 logit monotonically AND
  the L->H / H->L donor swap moves the logit toward the donor population by a
  materially larger amount than a norm-matched random patch.
  KILL if dec3 patching leaves the logit essentially unchanged, i.e. dec3
  displacement is a correlate, not a controller.
"""
import sys, csv, json, pickle, time
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
THR, PATCH, MIN_VOX, MAX_COMP = 0.5, 128, 5, 6
KS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def dmask(m_np, shape, dev):
    m = torch.from_numpy(m_np.astype(np.float32))[None, None]
    return (Fn.adaptive_max_pool3d(m, shape)[0, 0] > 0.5).to(dev)


def run_from_dec3(model, dec3, enc2, enc1_gated, ri):
    """Continue the frozen forward pass from a (possibly patched) dec3."""
    with torch.no_grad():
        u2 = model.upconv2(dec3)
        cat2 = torch.cat([u2, enc2], 1)
        dec2 = model.dec2(cat2)
        u1 = model.upconv1(dec2)
        cat1 = torch.cat([u1, enc1_gated], 1)
        dec1 = model.dec1(cat1)
        logits = model.seg_head[0](dec1)
    return dict(u2=u2[0], dec2=dec2[0], u1=u1[0], dec1=dec1[0], logit=logits[0, ri])


def capture_pre(model, x):
    with torch.no_grad():
        enc1 = model.enc1(x); p1 = model.pool1(enc1)
        enc2 = model.enc2(p1); p2 = model.pool2(enc2)
        enc3 = model.enc3(p2); p3 = model.pool3(enc3)
        bott = model.bottleneck(p3)
        u3 = model.upconv3(bott)
        dec3 = model.dec3(torch.cat([u3, enc3], 1))
        eg, _ = model.attn_gate1(gate=bott, skip=enc1)
    return dec3, enc2, eg


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    fr = list(csv.DictReader(open(OUT / 'FORENSIC_per_subject.csv')))
    tail = {r['subject_id'] for r in fr
            if float(r['ET_dice']) < 0.5 or float(r['TC_dice']) < 0.5}
    idx = [i for i in range(len(ds)) if Path(ds.subject_dirs[i]).name in tail]

    surv, caus = [], []
    proto_store = []       # (sid, dZ_dec3 of LESION) for the transplant prototype
    t0 = time.time()
    for ii, i in enumerate(idx):
        img, tgt, sid = ds[i]
        Y = tgt.numpy() > 0.5
        probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                         t.PATCH, t.SW_OVERLAP, 3, dev, True)
        P = probs >= THR
        brain = img[0].numpy() != 0
        for ri, rn in [(0, 'ET'), (1, 'TC')]:
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
                anygt = (Y[0] | Y[1] | Y[2])[sl]; brain_p = brain[sl]
                x = img[(slice(None),) + sl].unsqueeze(0).to(dev)
                dec3, enc2, eg = capture_pre(model, x)
                base = run_from_dec3(model, dec3, enc2, eg, ri)
                lgn = base['logit'].cpu().numpy().astype(np.float64)

                far = brain_p & ~ndimage.binary_dilation(anygt, iterations=8)
                if far.sum() < 50:
                    continue
                dist = ndimage.distance_transform_edt(~cm_p)
                adj = (dist > 1) & (dist <= 6) & brain_p & ~anygt
                cand = brain_p & ~ndimage.binary_dilation(anygt, iterations=4)
                if adj.sum() < MIN_VOX or cand.sum() < cm_p.sum() * 2:
                    continue
                k = int(cm_p.sum())
                lm = np.where(cand, lgn, -np.inf)
                hn = np.zeros_like(cm_p); hn.ravel()[np.argpartition(-lm.ravel(), k)[:k]] = True
                if hn.sum() < MIN_VOX:
                    continue

                pops = {'LESION': cm_p, 'ADJ': adj, 'HARDNEG': hn}
                # ---------------- PART A: survival ----------------
                for pop, m in pops.items():
                    rec = {'subject_id': sid, 'region': rn, 'comp_id': int(g),
                           'population': pop, 'n_vox': int(m.sum())}
                    dz = {}
                    for nm, feat in [('dec3', dec3[0]), ('u2', base['u2']), ('dec2', base['dec2']),
                                     ('u1', base['u1']), ('dec1', base['dec1'])]:
                        shp = tuple(feat.shape[1:])
                        mr, mb = dmask(m, shp, dev), dmask(far, shp, dev)
                        if mr.sum() < 1 or mb.sum() < 1:
                            dz = None; break
                        v = (feat[:, mr].mean(1) - feat[:, mb].mean(1)).double()
                        dz[nm] = v
                        rec[f'M_{nm}'] = float(v.norm())
                    if dz is None:
                        continue
                    # magnitude retention through the decoder
                    rec['ret_dec3_dec2'] = rec['M_dec2'] / (rec['M_dec3'] + 1e-9)
                    rec['ret_dec2_dec1'] = rec['M_dec1'] / (rec['M_dec2'] + 1e-9)
                    rec['ret_dec3_dec1'] = rec['M_dec1'] / (rec['M_dec3'] + 1e-9)
                    # parallel/perp of dec2 displacement vs upsampled dec3 displacement
                    a, b = dz['u2'], dz['dec2']
                    kk = min(a.numel(), b.numel())
                    # NOTE: u2 and dec2 have the SAME width (64) so this is well-posed
                    if a.numel() == b.numel():
                        u = a / (a.norm() + 1e-12)
                        par = float(torch.dot(b, u)); perp = float((b - par * u).norm())
                        rec['dec2_par'] = par; rec['dec2_perp'] = perp
                        rec['dec2_par_frac'] = abs(par) / (b.norm() + 1e-12)
                    a, b = dz['u1'], dz['dec1']
                    if a.numel() == b.numel():
                        u = a / (a.norm() + 1e-12)
                        par = float(torch.dot(b, u)); perp = float((b - par * u).norm())
                        rec['dec1_par'] = par; rec['dec1_perp'] = perp
                        rec['dec1_par_frac'] = abs(par) / (b.norm() + 1e-12)
                    # skip vs upsample split of the CONCAT displacement
                    for nm, cat_feat, split in [('cat2', torch.cat([base['u2'], enc2[0]], 0), base['u2'].shape[0]),
                                                ('cat1', torch.cat([base['u1'], eg[0]], 0), base['u1'].shape[0])]:
                        shp = tuple(cat_feat.shape[1:])
                        mr, mb = dmask(m, shp, dev), dmask(far, shp, dev)
                        v = (cat_feat[:, mr].mean(1) - cat_feat[:, mb].mean(1)).double()
                        up_n = float(v[:split].norm()); sk_n = float(v[split:].norm())
                        rec[f'{nm}_up_norm'] = up_n; rec[f'{nm}_skip_norm'] = sk_n
                        rec[f'{nm}_skip_frac'] = sk_n / (up_n + sk_n + 1e-12)
                    rec['logit_mean'] = float(lgn[m].mean())
                    surv.append(rec)
                    if pop == 'LESION':
                        proto_store.append((sid, dz['dec3'].cpu().numpy()))

                # ---------------- PART B: causal patching ----------------
                shp3 = tuple(dec3[0].shape[1:])
                mb3 = dmask(far, shp3, dev)
                bgv = dec3[0][:, mb3].mean(1)                       # background dec3 vector
                for pop, m in [('LESION', cm_p), ('HARDNEG', hn)]:
                    mr3 = dmask(m, shp3, dev)
                    if mr3.sum() < 1:
                        continue
                    cur = dec3[0][:, mr3].mean(1)
                    d_cur = cur - bgv
                    mfull = torch.from_numpy(m).to(dev)
                    l0 = float(base['logit'].cpu().numpy()[m].mean())
                    # P2 scaling
                    for kk_ in KS:
                        d3 = dec3.clone()
                        target = bgv + kk_ * d_cur
                        d3[0][:, mr3] = target.unsqueeze(1)
                        r = run_from_dec3(model, d3, enc2, eg, ri)
                        caus.append({'subject_id': sid, 'region': rn, 'comp_id': int(g),
                                     'population': pop, 'test': 'scale', 'k': kk_,
                                     'logit': float(r['logit'].cpu().numpy()[m].mean()),
                                     'logit0': l0, 'n_vox': int(m.sum()),
                                     'M_dec3_orig': float(d_cur.norm())})
                    # P1/P3 cross-population transplant (norm-matched random control)
                    other = 'HARDNEG' if pop == 'LESION' else 'LESION'
                    om = hn if pop == 'LESION' else cm_p
                    mo3 = dmask(om, shp3, dev)
                    if mo3.sum() >= 1:
                        d_other = dec3[0][:, mo3].mean(1) - bgv
                        for nm, dv in [('swap', d_other),
                                       ('rand', torch.randn_like(d_cur) *
                                        (d_other.norm() / (torch.randn_like(d_cur).norm() + 1e-9)))]:
                            d3 = dec3.clone()
                            d3[0][:, mr3] = (bgv + dv).unsqueeze(1)
                            r = run_from_dec3(model, d3, enc2, eg, ri)
                            caus.append({'subject_id': sid, 'region': rn, 'comp_id': int(g),
                                         'population': pop, 'test': nm, 'k': np.nan,
                                         'logit': float(r['logit'].cpu().numpy()[m].mean()),
                                         'logit0': l0, 'n_vox': int(m.sum()),
                                         'M_dec3_orig': float(d_cur.norm())})
        if (ii + 1) % 4 == 0:
            print(f'  {ii+1}/{len(idx)} subj | surv {len(surv)} caus {len(caus)} '
                  f'({time.time()-t0:.0f}s)', flush=True)

    for name, data in [('SURVIVAL_decoder.csv', surv), ('CAUSAL_patching.csv', caus)]:
        keys = sorted({k for r in data for k in r})
        with open(OUT / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for r in data: w.writerow(r)
        print(f'wrote {name} ({len(data)} rows)')
    print(f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
