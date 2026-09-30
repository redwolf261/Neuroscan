"""E219 -- Self-Prototype Steering (SPS): training-free, GT-free, per-subject.
FROZEN E131. GPU-resident. PREREGISTERED (gates are SIGNED -- E218 lesson).

Algorithm, per test scan:
  1. baseline pass: ET logits; collect bottleneck vectors at the model's own
     CONFIDENT ET core (tile ET logit pooled to bottleneck res > Z_CONF) ->
     mu_self, and confident background (< Z_BG) -> mu_Bself. No GT.
  2. similarity pass: s_self(x) = <h(x)-mu_Bself, mu_self-mu_Bself>/||.||
  3. steer region R = top-q% of s_self inside brain, EXCLUDING dilate(baseline
     ET prediction, 3) -- blocks the boundary-expansion route that produced
     E217's gain; any gain must come from NEW regions.
  4. steered pass: h <- h + alpha*(proto - h) inside R, frozen decoder.

Arms (same pipeline, only proto / region differ):
  self   : proto = mu_self,  R = self-located region
  cross  : proto = cross-subject mu_L (E216 bank), R = SAME self region
  random : proto = mu_self,  R = random brain blobs, volume matched to self
Subjects with too few confident ET-core positions (< MIN_CORE) fall back to
no steering (counted and reported).

Prior art: SIPL (arXiv 2507.07602, IJCAI 2024) instance-adaptive prototypes
= TRAINED pixel-to-prototype READOUT, not brain tumor. Memory-bank TTA
(2607.17693) uses cross-image anchors. Not found: training-free same-scan
self-seeded latent STEERING + frozen-decoder re-decode.
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
EXCL_DIL = 3
Z_CONF, Z_BG = 4.0, -10.0
MIN_CORE = 4
ALPHAS = [0.5, 1.0]
QS = [99.0, 98.0]            # top-1%, top-2% of s_self
THRESH = [-6.0, -4.02, -2.0, 0.0, 2.0]
COMP_THRESH = [0.0, -4.02]


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


def sliding(model, image, brain_t, dev, mode, proto=None, mu_B=None, mask=None, alpha=0.0):
    """mode: 'base' -> (z, mu_self, mu_Bself, n_core)
             'sim'  -> s field (proto=mu_self, mu_B=mu_Bself)
             'steer'-> z (steer toward proto inside mask)"""
    _, C, D, H, W = image.shape
    pd, ph, pw = PATCH
    stride = [max(1, int(p * (1 - OVERLAP))) for p in PATCH]
    zs, ys, xs = starts(D, pd, stride[0]), starts(H, ph, stride[1]), starts(W, pw, stride[2])
    acc = torch.zeros(((3 if mode != 'sim' else 1), D, H, W), device=dev)
    wsum = torch.zeros((D, H, W), device=dev)
    gw3 = torch.from_numpy(t._gaussian_weight((min(pd, D), min(ph, H), min(pw, W)))).to(dev)
    core_sum = torch.zeros(256, device=dev); core_n = 0
    bg_sum = torch.zeros(256, device=dev); bg_n = 0
    if mode == 'sim':
        dvec = proto - mu_B; dn = dvec.norm() + EPS
    m_t = torch.from_numpy(mask.astype(np.float32)).to(dev) if mask is not None else None
    with torch.no_grad():
        for z0 in zs:
            for y0 in ys:
                for x0 in xs:
                    zc, yc, xc = min(pd, D), min(ph, H), min(pw, W)
                    tile = image[:, :, z0:z0 + zc, y0:y0 + yc, x0:x0 + xc]
                    padw = (0, pw - tile.shape[4], 0, ph - tile.shape[3], 0, pd - tile.shape[2])
                    if tile.shape[2:] != (pd, ph, pw):
                        tile = Fn.pad(tile, padw)
                    with torch.amp.autocast("cuda", enabled=True):
                        e1 = model.enc1(tile); e2 = model.enc2(model.pool1(e1))
                        e3 = model.enc3(model.pool2(e2))
                        bn = model.bottleneck(model.pool3(e3))
                    bn = bn.float()
                    g = gw3[:zc, :yc, :xc]
                    sl = (slice(z0, z0 + zc), slice(y0, y0 + yc), slice(x0, x0 + xc))
                    if mode == 'sim':
                        s = torch.tensordot(dvec, bn[0] - mu_B[:, None, None, None],
                                            dims=([0], [0])) / dn
                        s_up = Fn.interpolate(s[None, None], size=(pd, ph, pw),
                                              mode='nearest')[0, 0][:zc, :yc, :xc]
                        acc[0][sl] += s_up * g
                        wsum[sl] += g
                        continue
                    if mode == 'steer' and m_t is not None and alpha > 0:
                        mt = m_t[sl]
                        if mt.shape != (pd, ph, pw):
                            mt = Fn.pad(mt, padw)
                        if mt.any():
                            m3 = Fn.adaptive_max_pool3d(mt[None, None], bn.shape[2:])[0, 0] > 0.5
                            if m3.any():
                                b = bn[0][:, m3]
                                bn[0][:, m3] = b + alpha * (proto[:, None] - b)
                    with torch.amp.autocast("cuda", enabled=True):
                        zt = decode3(model, e1, e2, e3, bn.to(e3.dtype))
                    if mode == 'base':
                        zp = Fn.adaptive_avg_pool3d(zt[ET][None, None], bn.shape[2:])[0, 0]
                        bt = brain_t[sl]
                        if bt.shape != (pd, ph, pw):
                            bt = Fn.pad(bt, padw)
                        bp = Fn.adaptive_avg_pool3d(bt[None, None], bn.shape[2:])[0, 0] > 0.5
                        core = (zp > Z_CONF) & bp
                        bgm = (zp < Z_BG) & bp
                        if core.any():
                            core_sum += bn[0][:, core].sum(1); core_n += int(core.sum())
                        if bgm.any():
                            bg_sum += bn[0][:, bgm].sum(1); bg_n += int(bgm.sum())
                    acc[:, sl[0], sl[1], sl[2]] += zt[:, :zc, :yc, :xc] * g
                    wsum[sl] += g
    out = (acc / wsum.clamp(min=1e-6)).cpu().numpy()
    if mode == 'sim':
        return out[0]
    if mode == 'base':
        mu_self = core_sum / max(core_n, 1)
        mu_bs = bg_sum / max(bg_n, 1)
        return out, mu_self, mu_bs, core_n
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
    print(f'E219 self-prototype steering, {n_total} subjects, shard {shard}/{nshard}',
          flush=True)

    # cross-subject mu_L bank (control arm), same construction as E216
    proto, proto_sid = [], []
    for ii in range(min(n_bank, len(ds))):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        lbl, nl = ndimage.label(Y[ET])
        for g in range(1, nl + 1):
            cm = lbl == g
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

    out = HERE / ('E219_smoke.csv' if smoke else
                  ('E219_counts.csv' if nshard == 1 else f'E219_counts_n{nshard}s{shard}.csv'))
    # RESUME: subjects already COMPLETE (195 rows) in any prior counts file
    done = set()
    if not smoke:
        from collections import Counter as _C
        cnt = _C()
        for p in HERE.glob('E219_counts*.csv'):
            if p == out:
                continue
            for r in csv.DictReader(open(p)):
                cnt[r['subject_id']] += 1
        done = {s for s, n in cnt.items() if n >= 195}
        print(f'  resume: {len(done)} subjects already complete', flush=True)
    fh = open(out, 'w', newline='')
    fields = ['subject_id', 'arm', 'alpha', 'q', 'region', 'thresh', 'tp', 'fp', 'fn',
              'et_tp_missed', 'et_n_missed', 'et_recovered', 'et_fp_comp', 'steer_vox',
              'n_core']
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
    rng = np.random.default_rng(0)
    t0 = time.time()
    n_nocore = 0
    for ii in range(n_total):
        if ii % nshard != shard:
            continue
        rng = np.random.default_rng(1000 + ii)   # per-subject: shard-invariant random arm
        img, tgt, sid = ds[ii]
        if sid in done:
            continue
        Y = tgt.numpy() > 0.5
        brain = img[0].numpy() != 0
        brain_t =torch.from_numpy(brain.astype(np.float32)).to(dev)
        x = img.unsqueeze(0).to(dev)
        other = np.array([s != sid for s in proto_sid])
        mu_L = torch.from_numpy(proto[other].mean(0) if other.sum() >= 2
                                else proto.mean(0)).float().to(dev)

        z0, mu_self, mu_bs, n_core = sliding(model, x, brain_t, dev, 'base')
        base_et = (z0[ET] > 0) & brain
        missed = [cm for cm in comps(Y[ET]) if not (cm & base_et).any()]
        mm = np.zeros_like(Y[ET])
        for cm in missed:
            mm |= cm

        def emit(arm, a, q, z, sv):
            for r in range(3):
                gt = Y[r][brain]; zr = z[r][brain]
                for th in THRESH:
                    p = zr > th
                    row = {'subject_id': sid, 'arm': arm, 'alpha': a, 'q': q, 'region': r,
                           'thresh': th, 'tp': int((p & gt).sum()), 'fp': int((p & ~gt).sum()),
                           'fn': int((~p & gt).sum()), 'et_tp_missed': '', 'et_n_missed': '',
                           'et_recovered': '', 'et_fp_comp': '', 'steer_vox': sv,
                           'n_core': n_core}
                    if r == ET and th in COMP_THRESH:
                        pf = (z[ET] > th) & brain
                        row['et_tp_missed'] = int((pf & mm).sum())
                        row['et_n_missed'] = len(missed)
                        row['et_recovered'] = sum(1 for cm in missed if (cm & pf).any())
                        row['et_fp_comp'] = sum(1 for cm in comps(pf) if not (cm & Y[ET]).any())
                    w.writerow(row)

        emit('none', 0.0, 0.0, z0, 0)
        if n_core < MIN_CORE:
            n_nocore += 1
            for arm in ['self', 'cross', 'random']:
                for a in ALPHAS:
                    for q in QS:
                        emit(arm, a, q, z0, 0)
        else:
            s_self = sliding(model, x, brain_t, dev, 'sim', proto=mu_self, mu_B=mu_bs)
            excl = ndimage.binary_dilation(base_et, iterations=EXCL_DIL)
            cand = brain & ~excl
            sv_vals = s_self[cand]
            for q in QS:
                thr = np.percentile(sv_vals, q) if sv_vals.size else np.inf
                R = ndimage.binary_dilation((s_self > thr) & cand, iterations=ROI_DIL) & cand
                vol = int(R.sum())
                rnd = np.zeros_like(R)
                cidx = np.argwhere(cand)
                tries = 0
                while rnd.sum() < vol and tries < 600 and len(cidx):
                    c = cidx[rng.integers(len(cidx))]
                    rnd[tuple(slice(max(0, c[k] - 3), c[k] + 4) for k in range(3))] = True
                    tries += 1
                rnd &= cand
                for a in ALPHAS:
                    for arm, P, M in [('self', mu_self, R), ('cross', mu_L, R),
                                      ('random', mu_self, rnd)]:
                        zs = (sliding(model, x, brain_t, dev, 'steer', proto=P, mask=M, alpha=a)
                              if M.any() else z0)
                        emit(arm, a, q, zs, int(M.sum()))
        fh.flush()
        if (ii + 1) % 5 == 0 or smoke:
            print(f'  {ii+1}/{n_total} subj ({time.time()-t0:.0f}s) no-core so far {n_nocore}',
                  flush=True)
    fh.close()
    print(f'wrote {out.name} {time.time()-t0:.0f}s  no-core subjects {n_nocore}', flush=True)


if __name__ == '__main__':
    main()
