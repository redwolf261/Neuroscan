"""E216 -- Oracle vs deployable latent steering, whole-volume ET Dice.
FROZEN E131. NO TRAINING. GPU-resident.

Prior art: Med-SegLens (arXiv 2602.10508, ICML 2026) recovers glioma
segmentation failures by latent steering, but selects WHAT/WHERE to steer
per case using ground truth (oracle) and never tests a GT-free selection.
This experiment measures that gap directly on our model.

Steering operator (identical in every arm, no GT): inside a region R at the
bottleneck, h <- h + alpha * (mu_L - h), where mu_L is the mean
detected-lesion bottleneck vector from OTHER subjects (E210's direction,
leave-subject-out).

Arms differ ONLY in the region R:
  none        : no steering (baseline, recomputed here for exact parity)
  oracle      : GT ET components MISSED by the baseline (z>0), dilated
  deploy_unc  : near-miss band -NEAR<z<=0 components (GT-free)
  deploy_sh   : top-1% s_h voxels in brain (GT-free)
  random      : random brain blobs, total volume matched to deploy_unc

Metrics per subject x arm x alpha x threshold: ET Dice (dice_per_region
convention), missed-GT-components recovered, FP components created.
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
dice_per_region = t.dice_per_region

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
PATCH = (128, 128, 128)
OVERLAP = 0.5
ET = 0
MIN_VOX = 5
EPS = 1e-6
ROI_DIL = 3            # input-res dilation of steering regions
NEAR = 6.0             # near-miss logit band for deploy_unc
ALPHAS = [0.5, 1.0]
THRESH = [0.0, -4.02]  # z>0 and production operating point


def starts(full, p, st):
    if full <= p:
        return [0]
    s = list(range(0, full - p + 1, st))
    if s[-1] != full - p:
        s.append(full - p)
    return s


def decode(model, e1, e2, e3, bn):
    d3 = model.dec3(torch.cat([model.upconv3(bn), e3], 1))
    d2 = model.dec2(torch.cat([model.upconv2(d3), e2], 1))
    eg, _ = model.attn_gate1(gate=bn, skip=e1)
    d1 = model.dec1(torch.cat([model.upconv1(d2), eg], 1))
    return model.seg_head[0](d1).float()[0, ET]


def sliding(model, image, dev, mu_L_t, mu_B_t, dir_norm, steer_mask=None, alpha=0.0):
    """Returns (z, s_h) full-res numpy. If steer_mask (full-res bool numpy)
    is given, bottleneck is steered toward mu_L inside it before decoding."""
    _, C, D, H, W = image.shape
    pd, ph, pw = PATCH
    stride = [max(1, int(p * (1 - OVERLAP))) for p in PATCH]
    zs, ys, xs = starts(D, pd, stride[0]), starts(H, ph, stride[1]), starts(W, pw, stride[2])
    z_acc = torch.zeros((D, H, W), device=dev)
    sh_acc = torch.zeros((D, H, W), device=dev)
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
                        zt = decode(model, e1, e2, e3, bn.to(e3.dtype))[:zc, :yc, :xc]
                    diff = bn[0] - mu_B_t[:, None, None, None]
                    sh = torch.tensordot(dir_vec, diff, dims=([0], [0])) / dir_norm
                    sh_up = Fn.interpolate(sh[None, None], size=(pd, ph, pw), mode='nearest')[0, 0][:zc, :yc, :xc]
                    g = gw3[:zc, :yc, :xc]
                    z_acc[z0:z0 + zc, y0:y0 + yc, x0:x0 + xc] += zt * g
                    sh_acc[z0:z0 + zc, y0:y0 + yc, x0:x0 + xc] += sh_up * g
                    wsum[z0:z0 + zc, y0:y0 + yc, x0:x0 + xc] += g
    w = wsum.clamp(min=1e-6)
    return (z_acc / w).cpu().numpy(), (sh_acc / w).cpu().numpy()


def single_patch_bn_z(model, x):
    with torch.no_grad():
        e1 = model.enc1(x); e2 = model.enc2(model.pool1(e1))
        e3 = model.enc3(model.pool2(e2)); bn = model.bottleneck(model.pool3(e3))
        z = decode(model, e1, e2, e3, bn)
    return z.cpu().numpy(), bn[0].cpu().numpy()


def comps(mask):
    lbl, n = ndimage.label(mask)
    return [(lbl == g) for g in range(1, n + 1) if (lbl == g).sum() >= MIN_VOX]


def metrics(pred, gt, missed_list):
    d = dice_per_region(pred[None].astype(np.float32), gt[None].astype(np.float32))[0]
    rec = sum(1 for cm in missed_list if (cm & pred).any())
    fp = sum(1 for cm in comps(pred) if not (cm & gt).any())
    return d, rec, fp


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
    n_total = 3 if smoke else len(ds)
    n_bank = 4 if smoke else 60
    print(f'E216 oracle-vs-deployable steering, {n_total} subjects', flush=True)

    # ---- bank (same as E215 Pass 0) ----
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

    rows = []
    t0 = time.time()
    rng = np.random.default_rng(0)
    for ii in range(n_total):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        gt = Y[ET]
        brain = img[0].numpy() != 0
        other = np.array([s != sid for s in proto_sid])
        mu_L = proto[other].mean(0) if other.sum() >= 2 else proto.mean(0)
        mu_B = bgv.get(sid, mu_B_glob)
        mu_L_t = torch.from_numpy(mu_L).float().to(dev)
        mu_B_t = torch.from_numpy(mu_B).float().to(dev)
        dn = float(np.linalg.norm(mu_L - mu_B)) + EPS
        x = img.unsqueeze(0).to(dev)

        z0, sh0 = sliding(model, x, dev, mu_L_t, mu_B_t, dn)
        base_pred = (z0 > 0) & brain
        missed = [cm for cm in comps(gt) if not (cm & base_pred).any()]

        R = {}
        o = np.zeros_like(gt)
        for cm in missed:
            o |= cm
        R['oracle'] = ndimage.binary_dilation(o, iterations=ROI_DIL) & brain
        near = (z0 > -NEAR) & (z0 <= 0) & brain
        near = ndimage.binary_opening(near)
        R['deploy_unc'] = ndimage.binary_dilation(near, iterations=ROI_DIL) & brain
        shb = sh0[brain]
        thr_sh = np.percentile(shb, 99) if shb.size else np.inf
        R['deploy_sh'] = ndimage.binary_dilation((sh0 > thr_sh) & brain,
                                                 iterations=ROI_DIL) & brain
        target_vol = int(R['deploy_unc'].sum())
        rnd = np.zeros_like(gt)
        bidx = np.argwhere(brain)
        tries = 0
        while rnd.sum() < target_vol and tries < 400 and len(bidx):
            c = bidx[rng.integers(len(bidx))]
            sl = tuple(slice(max(0, c[k] - 3), c[k] + 4) for k in range(3))
            rnd[sl] = True
            tries += 1
        R['random'] = rnd & brain

        for th in THRESH:
            d, rec, fp = metrics((z0 > th) & brain, gt, missed)
            rows.append({'subject_id': sid, 'arm': 'none', 'alpha': 0.0, 'thresh': th,
                         'dice': d, 'n_missed': len(missed), 'recovered': rec,
                         'fp_comp': fp, 'steer_vox': 0})
        for arm, mask in R.items():
            for a in ALPHAS:
                if mask.sum() == 0:
                    zs = z0
                else:
                    zs, _ = sliding(model, x, dev, mu_L_t, mu_B_t, dn, steer_mask=mask, alpha=a)
                for th in THRESH:
                    d, rec, fp = metrics((zs > th) & brain, gt, missed)
                    rows.append({'subject_id': sid, 'arm': arm, 'alpha': a, 'thresh': th,
                                 'dice': d, 'n_missed': len(missed), 'recovered': rec,
                                 'fp_comp': fp, 'steer_vox': int(mask.sum())})
        if (ii + 1) % 5 == 0 or smoke:
            print(f'  {ii+1}/{n_total} subj ({time.time()-t0:.0f}s)', flush=True)

    out = HERE / ('E216_smoke.csv' if smoke else 'E216_steering.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'wrote {out.name} ({len(rows)} rows) {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
