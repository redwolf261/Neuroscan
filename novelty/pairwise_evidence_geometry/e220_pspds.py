"""E220 -- Patient-Specific Prototype Deficit Steering (PS-PDS). FROZEN E131.
NO TRAINING, NO GT at inference. PREREGISTERED, SIGNED gates.

Per scan (mu_L, mu_B from the model's OWN confident ET core / background,
as E219), d = (mu_L - mu_B)/||.||, s_x = (h_x - mu_B).d, s_L = ||mu_L - mu_B||,
r_x = (h_x - mu_B) - s_x d.
  PS-PDS : h' = h + alpha * g_x * (s_L - s_x)_+ / (1 + lam*||r_x||/rmed) * d
           g_x = sigmoid((s_x - tau)/T), tau = s_L/2 (amended from p99: inert),
           T = s_L/10, rmed = median ||r|| over brain.
Controls on the SAME region R = {s > tau} (dilated 3, brain):
  interp : h' = h + alpha*(mu_L_self - h)          (E219-style, no exclusion)
  global : h' = h + alpha*(mu_L_cross - h)         (E196/E216 prototype)
  dilate : ET logit grey-dilated (3x3x3 max) inside R   (post-processing)
  boost  : ET logit + c inside R, c in {2,4}              (local threshold)
Subjects with < MIN_CORE confident core positions: all arms = baseline.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
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
PATCH = (128, 128, 128)
OVERLAP = 0.5
ET = 0
MIN_VOX = 5
EPS = 1e-6
ROI_DIL = 3
Z_CONF, Z_BG = 4.0, -10.0
MIN_CORE = 4
THRESH = [-6.0, -4.02, -2.0, 0.0, 2.0]
COMP_THRESH = [0.0, -4.02]
CONFIGS = ([('none', 0.0, 0.0)] +
           [('pspds', a, l) for a in (0.5, 1.0) for l in (0.0, 1.0)] +
           [('interp', a, 0.0) for a in (0.5, 1.0)] +
           [('global', a, 0.0) for a in (0.5, 1.0)] +
           [('dilate', 0.0, 0.0)] +
           [('boost', c, 0.0) for c in (2.0, 4.0)])
ROWS_PER_SUBJ = len(CONFIGS) * 3 * len(THRESH)


def starts(full, p, st):
    if full <= p:
        return [0]
    s = list(range(0, full - p + 1, st))
    if s[-1] != full - p:
        s.append(full - p)
    return s


def decode3(model, e1, e2, e3, bn):
    d3 = model.dec3(torch.cat([model.upconv3(bn), e3], 1))
    d2 = model.dec2(torch.cat([model.upconv2(d3), e2], 1))
    eg, _ = model.attn_gate1(gate=bn, skip=e1)
    d1 = model.dec1(torch.cat([model.upconv1(d2), eg], 1))
    return model.seg_head[0](d1).float()[0]


def sliding(model, image, brain_t, dev, mode, P=None):
    """mode 'base' -> (z, mu_self, mu_B, n_core); 'sim' -> (s, rnorm) fields;
    'pspds'/'interp' -> z. P: dict of operator parameters."""
    _, C, D, H, W = image.shape
    pd, ph, pw = PATCH
    stride = [max(1, int(p * (1 - OVERLAP))) for p in PATCH]
    zs, ys, xs = starts(D, pd, stride[0]), starts(H, ph, stride[1]), starts(W, pw, stride[2])
    nch = 2 if mode == 'sim' else 3
    acc = torch.zeros((nch, D, H, W), device=dev)
    wsum = torch.zeros((D, H, W), device=dev)
    gw3 = torch.from_numpy(t._gaussian_weight((min(pd, D), min(ph, H), min(pw, W)))).to(dev)
    cs = torch.zeros(256, device=dev); cn = 0
    bs = torch.zeros(256, device=dev); bnn = 0
    m_t = (torch.from_numpy(P['mask'].astype(np.float32)).to(dev)
           if P is not None and 'mask' in P else None)
    with torch.no_grad():
        for z0 in zs:
            for y0 in ys:
                for x0 in xs:
                    zc, yc, xc = min(pd, D), min(ph, H), min(pw, W)
                    sl = (slice(z0, z0 + zc), slice(y0, y0 + yc), slice(x0, x0 + xc))
                    tile = image[:, :, sl[0], sl[1], sl[2]]
                    padw = (0, pw - tile.shape[4], 0, ph - tile.shape[3], 0, pd - tile.shape[2])
                    if tile.shape[2:] != (pd, ph, pw):
                        tile = Fn.pad(tile, padw)
                    with torch.amp.autocast("cuda", enabled=True):
                        e1 = model.enc1(tile); e2 = model.enc2(model.pool1(e1))
                        e3 = model.enc3(model.pool2(e2))
                        bn = model.bottleneck(model.pool3(e3))
                    bn = bn.float()
                    g = gw3[:zc, :yc, :xc]
                    if mode in ('sim', 'pspds'):
                        h = bn[0]                                  # (C,b,b,b)
                        v = h - P['mu_B'][:, None, None, None]
                        s = torch.tensordot(P['d'], v, dims=([0], [0]))
                        r = v - s[None] * P['d'][:, None, None, None]
                        rn = r.norm(dim=0)
                    if mode == 'sim':
                        f = torch.stack([s, rn])[None]
                        f_up = Fn.interpolate(f, size=(pd, ph, pw), mode='nearest')[0][:, :zc, :yc, :xc]
                        acc[:, sl[0], sl[1], sl[2]] += f_up * g
                        wsum[sl] += g
                        continue
                    if mode == 'pspds':
                        gate = torch.sigmoid((s - P['tau']) / P['T'])
                        deficit = (P['sL'] - s).clamp(min=0)
                        amt = P['alpha'] * gate * deficit / (1 + P['lam'] * rn / P['rmed'])
                        bn[0] = h + amt[None] * P['d'][:, None, None, None]
                    elif mode == 'interp':
                        mt = m_t[sl]
                        if mt.shape != (pd, ph, pw):
                            mt = Fn.pad(mt, padw)
                        if mt.any():
                            m3 = Fn.adaptive_max_pool3d(mt[None, None], bn.shape[2:])[0, 0] > 0.5
                            if m3.any():
                                b = bn[0][:, m3]
                                bn[0][:, m3] = b + P['alpha'] * (P['proto'][:, None] - b)
                    with torch.amp.autocast("cuda", enabled=True):
                        zt = decode3(model, e1, e2, e3, bn.to(e3.dtype))
                    if mode == 'base':
                        zp = Fn.adaptive_avg_pool3d(zt[ET][None, None], bn.shape[2:])[0, 0]
                        bt = brain_t[sl]
                        if bt.shape != (pd, ph, pw):
                            bt = Fn.pad(bt, padw)
                        bp = Fn.adaptive_avg_pool3d(bt[None, None], bn.shape[2:])[0, 0] > 0.5
                        core = (zp > Z_CONF) & bp; bgm = (zp < Z_BG) & bp
                        if core.any():
                            cs += bn[0][:, core].sum(1); cn += int(core.sum())
                        if bgm.any():
                            bs += bn[0][:, bgm].sum(1); bnn += int(bgm.sum())
                    acc[:, sl[0], sl[1], sl[2]] += zt[:, :zc, :yc, :xc] * g
                    wsum[sl] += g
    out = (acc / wsum.clamp(min=1e-6)).cpu().numpy()
    if mode == 'base':
        return out, cs / max(cn, 1), bs / max(bnn, 1), cn
    return out


def single_patch_bn_z(model, x):
    with torch.no_grad():
        e1 = model.enc1(x); e2 = model.enc2(model.pool1(e1))
        e3 = model.enc3(model.pool2(e2)); bn = model.bottleneck(model.pool3(e3))
        z = decode3(model, e1, e2, e3, bn)[ET]
    return z.cpu().numpy(), bn[0].cpu().numpy()


def comps(mask):
    lbl, n = ndimage.label(mask)
    return [(lbl == g) for g in range(1, n + 1) if (lbl == g).sum() >= MIN_VOX]


def main():
    dev = torch.device('cuda')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for q in model.parameters():
        q.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    smoke = '--smoke' in sys.argv
    n_total = 2 if smoke else len(ds)
    n_bank = 4 if smoke else 60
    shard, nshard = 0, 1
    for a in sys.argv:
        if a.startswith('--shard='):
            shard, nshard = map(int, a.split('=')[1].split('/'))
    print(f'E220 PS-PDS, {n_total} subjects, shard {shard}/{nshard}', flush=True)

    proto, proto_sid = [], []
    for ii in range(min(n_bank, len(ds))):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        lbl, nl = ndimage.label(Y[ET])
        for gg in range(1, nl + 1):
            cm = lbl == gg
            if cm.sum() < MIN_VOX:
                continue
            com = np.array(ndimage.center_of_mass(cm)).astype(int)
            st = [int(np.clip(c - 64, 0, img.shape[k + 1] - 128)) for k, c in enumerate(com)]
            sl = tuple(slice(s, s + 128) for s in st)
            cm_p = cm[sl]
            if cm_p.sum() < MIN_VOX:
                continue
            zp, bnp = single_patch_bn_z(model, img[(slice(None),) + sl].unsqueeze(0).to(dev))
            if float((cm_p & (zp > 0)).sum()) / cm_p.sum() < 0.5:
                continue
            shp3 = bnp.shape[1:]
            pos = tuple(np.clip((np.array(ndimage.center_of_mass(cm_p)) *
                                 np.array(shp3) / 128).astype(int), 0, np.array(shp3) - 1))
            proto.append(bnp[:, pos[0], pos[1], pos[2]]); proto_sid.append(sid)
            break
    proto = np.array(proto)
    print(f'  cross bank {len(proto)}', flush=True)

    out = HERE / ('E220_smoke.csv' if smoke else
                  ('E220_counts.csv' if nshard == 1 else f'E220_counts_n{nshard}s{shard}.csv'))
    done = set()
    if not smoke:
        from collections import Counter
        cnt = Counter()
        for p in HERE.glob('E220_counts*.csv'):
            if p == out:
                continue
            for r in csv.DictReader(open(p)):
                cnt[r['subject_id']] += 1
        done = {s for s, n in cnt.items() if n >= ROWS_PER_SUBJ}
        print(f'  resume: {len(done)} complete', flush=True)
    fh = open(out, 'w', newline='')
    fields = ['subject_id', 'arm', 'p1', 'p2', 'region', 'thresh', 'tp', 'fp', 'fn',
              'et_tp_missed', 'et_n_missed', 'et_recovered', 'et_fp_comp', 'n_core']
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
    t0 = time.time(); n_nocore = 0
    for ii in range(n_total):
        if ii % nshard != shard:
            continue
        img, tgt, sid = ds[ii]
        if sid in done:
            continue
        Y = tgt.numpy() > 0.5
        brain = img[0].numpy() != 0
        brain_t = torch.from_numpy(brain.astype(np.float32)).to(dev)
        x = img.unsqueeze(0).to(dev)
        other = np.array([s != sid for s in proto_sid])
        mu_cross = torch.from_numpy(proto[other].mean(0) if other.sum() >= 2
                                    else proto.mean(0)).float().to(dev)
        z0, mu_L, mu_B, n_core = sliding(model, x, brain_t, dev, 'base')
        base_et = (z0[ET] > 0) & brain
        missed = [cm for cm in comps(Y[ET]) if not (cm & base_et).any()]
        mm = np.zeros_like(Y[ET])
        for cm in missed:
            mm |= cm

        def emit(cfg, z):
            arm, p1, p2 = cfg
            for r in range(3):
                gt = Y[r][brain]; zr = z[r][brain]
                for th in THRESH:
                    p = zr > th
                    row = {'subject_id': sid, 'arm': arm, 'p1': p1, 'p2': p2, 'region': r,
                           'thresh': th, 'tp': int((p & gt).sum()), 'fp': int((p & ~gt).sum()),
                           'fn': int((~p & gt).sum()), 'et_tp_missed': '', 'et_n_missed': '',
                           'et_recovered': '', 'et_fp_comp': '', 'n_core': n_core}
                    if r == ET and th in COMP_THRESH:
                        pf = (z[ET] > th) & brain
                        row['et_tp_missed'] = int((pf & mm).sum())
                        row['et_n_missed'] = len(missed)
                        row['et_recovered'] = sum(1 for cm in missed if (cm & pf).any())
                        row['et_fp_comp'] = sum(1 for cm in comps(pf) if not (cm & Y[ET]).any())
                    w.writerow(row)

        if n_core < MIN_CORE:
            n_nocore += 1
            for cfg in CONFIGS:
                emit(cfg, z0)
        else:
            dvec = mu_L - mu_B
            sL = float(dvec.norm())
            d = dvec / (sL + EPS)
            s_f, rn_f = sliding(model, x, brain_t, dev, 'sim', P={'d': d, 'mu_B': mu_B})
            # AMENDED before any full-run result: tau=p99 was structurally inert
            # (p99 > s_L in both smoke subjects -> gate and deficit never overlap,
            # max correction 0.011). Gate now opens HALFWAY along the patient's
            # own lesion/background axis.
            tau = 0.5 * sL; T = max(0.1 * sL, 1e-3)
            rmed = max(float(np.median(rn_f[brain])), 1e-6)
            R = ndimage.binary_dilation((s_f > tau) & brain, iterations=ROI_DIL) & brain
            for cfg in CONFIGS:
                arm, p1, p2 = cfg
                if arm == 'none':
                    z = z0
                elif arm == 'pspds':
                    z = sliding(model, x, brain_t, dev, 'pspds',
                                P={'d': d, 'mu_B': mu_B, 'sL': sL, 'tau': tau, 'T': T,
                                   'rmed': rmed, 'alpha': p1, 'lam': p2})
                elif arm in ('interp', 'global'):
                    z = (sliding(model, x, brain_t, dev, 'interp',
                                 P={'mask': R, 'alpha': p1,
                                    'proto': mu_L if arm == 'interp' else mu_cross})
                         if R.any() else z0)
                elif arm == 'dilate':
                    z = z0.copy()
                    zd = ndimage.grey_dilation(z0[ET], size=(3, 3, 3))
                    z[ET] = np.where(R, zd, z0[ET])
                elif arm == 'boost':
                    z = z0.copy()
                    z[ET] = np.where(R, z0[ET] + p1, z0[ET])
                emit(cfg, z)
        fh.flush()
        if (ii + 1) % 5 == 0 or smoke:
            print(f'  {ii+1}/{n_total} subj ({time.time()-t0:.0f}s) no-core {n_nocore}', flush=True)
    fh.close()
    print(f'wrote {out.name} {time.time()-t0:.0f}s no-core {n_nocore}', flush=True)


if __name__ == '__main__':
    main()
