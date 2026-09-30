"""E217 -- Validation of GT-free s_h-located latent steering (E216 winner).
FROZEN E131. NO TRAINING. GPU-resident. PREREGISTERED -- config grid,
threshold grid, selection rule and endpoints fixed before any result.

E216: steering h <- h + alpha*(mu_L - h) inside the top-1% s_h region gave
ET Dice +1.07pp (z>0, p=1.4e-4) but was picked post hoc, ET-only, and
threshold-sensitive. This run collects everything needed to test it fairly:

  configs  : none + alpha {0.25,0.5,0.75,1.0} x s_h top-pct {2,1,0.5}  (13)
  regions  : ET, TC, WT (all three logits -- steering hits a SHARED trunk)
  thresholds on raw logit: {-6,-4.02,-2,0,2}
  stored per subject/config/region/threshold: TP, FP, FN (in brain)
  ET extras at z>0 and z>-4.02: TP inside missed-GT-components mask,
    recovered missed components, FP components.

Selection/scoring is done in e217_analyze.py by repeated subject-level
split-half: steering chooses (config, per-region thresholds) on the train
half; baseline chooses its own per-region thresholds on the SAME train half;
both scored on the held-out half.
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
ALPHAS = [0.25, 0.5, 0.75, 1.0]
PCTS = [98.0, 99.0, 99.5]
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
    return model.seg_head[0](d1).float()[0]          # (3, pd, ph, pw)


def sliding(model, image, dev, mu_L_t, mu_B_t, dir_norm, steer_mask=None,
            alpha=0.0, want_sh=False):
    _, C, D, H, W = image.shape
    pd, ph, pw = PATCH
    stride = [max(1, int(p * (1 - OVERLAP))) for p in PATCH]
    zs, ys, xs = starts(D, pd, stride[0]), starts(H, ph, stride[1]), starts(W, pw, stride[2])
    z_acc = torch.zeros((3, D, H, W), device=dev)
    sh_acc = torch.zeros((D, H, W), device=dev) if want_sh else None
    wsum = torch.zeros((D, H, W), device=dev)
    gw3 = torch.from_numpy(t._gaussian_weight((min(pd, D), min(ph, H), min(pw, W)))).to(dev)
    dir_vec = mu_L_t - mu_B_t
    sm_t = (torch.from_numpy(steer_mask.astype(np.float32)).to(dev)
            if steer_mask is not None else None)
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
                    if sm_t is not None and alpha > 0:
                        mtile = sm_t[z0:z0 + zc, y0:y0 + yc, x0:x0 + xc]
                        if mtile.shape != (pd, ph, pw):
                            mtile = Fn.pad(mtile, padw)
                        if mtile.any():
                            m3 = Fn.adaptive_max_pool3d(mtile[None, None], bn.shape[2:])[0, 0] > 0.5
                            if m3.any():
                                bsel = bn[0][:, m3]
                                bn[0][:, m3] = bsel + alpha * (mu_L_t[:, None] - bsel)
                    with torch.amp.autocast("cuda", enabled=True):
                        zt = decode3(model, e1, e2, e3, bn.to(e3.dtype))[:, :zc, :yc, :xc]
                    g = gw3[:zc, :yc, :xc]
                    z_acc[:, z0:z0 + zc, y0:y0 + yc, x0:x0 + xc] += zt * g
                    if want_sh:
                        diff = bn[0] - mu_B_t[:, None, None, None]
                        sh = torch.tensordot(dir_vec, diff, dims=([0], [0])) / dir_norm
                        sh_up = Fn.interpolate(sh[None, None], size=(pd, ph, pw),
                                               mode='nearest')[0, 0][:zc, :yc, :xc]
                        sh_acc[z0:z0 + zc, y0:y0 + yc, x0:x0 + xc] += sh_up * g
                    wsum[z0:z0 + zc, y0:y0 + yc, x0:x0 + xc] += g
    w = wsum.clamp(min=1e-6)
    z = (z_acc / w).cpu().numpy()
    sh = (sh_acc / w).cpu().numpy() if want_sh else None
    return z, sh


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
    print(f'E217 steering validation, {n_total} subjects', flush=True)

    proto, proto_sid, bgv = [], [], {}
    for ii in range(min(n_bank, len(ds))):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        brain = img[0].numpy() != 0
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
            bgm = brain[sl] & ~(Y[0] | Y[1] | Y[2])[sl]
            bgd = Fn.adaptive_avg_pool3d(torch.from_numpy(bgm.astype(np.float32))[None, None],
                                         shp3)[0, 0].numpy() > 0.5
            if bgd.sum() >= 4:
                bgv[sid] = bnp[:, bgd].mean(1)
            break
    proto = np.array(proto)
    mu_B_glob = np.mean(list(bgv.values()), 0)
    print(f'  bank {len(proto)} prototypes', flush=True)

    out = HERE / ('E217_smoke.csv' if smoke else 'E217_counts.csv')
    fh = open(out, 'w', newline='')
    fields = ['subject_id', 'alpha', 'pct', 'region', 'thresh', 'tp', 'fp', 'fn',
              'et_tp_missed', 'et_n_missed', 'et_recovered', 'et_fp_comp', 'steer_vox']
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()

    t0 = time.time()
    for ii in range(n_total):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        brain = img[0].numpy() != 0
        other = np.array([s != sid for s in proto_sid])
        mu_L = proto[other].mean(0) if other.sum() >= 2 else proto.mean(0)
        mu_B = bgv.get(sid, mu_B_glob)
        mu_L_t = torch.from_numpy(mu_L).float().to(dev)
        mu_B_t = torch.from_numpy(mu_B).float().to(dev)
        dn = float(np.linalg.norm(mu_L - mu_B)) + EPS
        x = img.unsqueeze(0).to(dev)

        z0, sh0 = sliding(model, x, dev, mu_L_t, mu_B_t, dn, want_sh=True)
        base_pred_et = (z0[ET] > 0) & brain
        missed = [cm for cm in comps(Y[ET]) if not (cm & base_pred_et).any()]
        missed_mask = np.zeros_like(Y[ET])
        for cm in missed:
            missed_mask |= cm
        shb = sh0[brain]

        def emit(alpha, pct, z, steer_vox):
            for r in range(3):
                gt = Y[r][brain]
                zr = z[r][brain]
                for th in THRESH:
                    p = zr > th
                    row = {'subject_id': sid, 'alpha': alpha, 'pct': pct, 'region': r,
                           'thresh': th, 'tp': int((p & gt).sum()), 'fp': int((p & ~gt).sum()),
                           'fn': int((~p & gt).sum()), 'et_tp_missed': '', 'et_n_missed': '',
                           'et_recovered': '', 'et_fp_comp': '', 'steer_vox': steer_vox}
                    if r == ET and th in COMP_THRESH:
                        pf = (z[ET] > th) & brain
                        row['et_tp_missed'] = int((pf & missed_mask).sum())
                        row['et_n_missed'] = len(missed)
                        row['et_recovered'] = sum(1 for cm in missed if (cm & pf).any())
                        row['et_fp_comp'] = sum(1 for cm in comps(pf) if not (cm & Y[ET]).any())
                    w.writerow(row)

        emit(0.0, 0.0, z0, 0)
        for pct in PCTS:
            thr = np.percentile(shb, pct) if shb.size else np.inf
            mask = ndimage.binary_dilation((sh0 > thr) & brain, iterations=ROI_DIL) & brain
            for a in ALPHAS:
                zs, _ = sliding(model, x, dev, mu_L_t, mu_B_t, dn, steer_mask=mask, alpha=a)
                emit(a, pct, zs, int(mask.sum()))
        fh.flush()
        if (ii + 1) % 5 == 0 or smoke:
            print(f'  {ii+1}/{n_total} subj ({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'wrote {out.name} {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
