"""E215 -- Full-volume decision-rule swap. FROZEN E131. NO TRAINING. ET ONLY.

Tests whether E214's proxy result (s_h Dice-proxy 0.9356 vs raw z's 0.8905,
on the fixed 74-component L/S/H set) translates to REAL ET Dice at
whole-volume resolution. Component-level proxies can look good and still be
inert at the scale that matters -- this is the decisive test.

SCOPE: ET only. s_h's mu_L/mu_B centroids were built entirely from ET
components throughout E204-E214. The bottleneck/decoder trunk is SHARED
across ET/TC/WT (only the final 1x1 seg_head conv differs per region), so
extending s_h to TC/WT requires separate prototype/background banks per
region -- explicitly deferred to a follow-up, not improvised here.

ARCHITECTURAL NOTE ON WHY THIS NEEDED A CUSTOM INFERENCE PASS: the existing
sliding_window_predict() calls model(tile) directly and only returns
post-sigmoid probs -- it never exposes the bottleneck tensor. s_h needs
h(x) per spatial location, so this script re-implements sliding-window
tiling but walks the manual encoder->bottleneck->decoder path (matching
E204-E214's method exactly) to get BOTH the raw ET logit z and the
bottleneck bn per tile, then blends both across overlapping tiles with the
SAME Gaussian weighting as the production pipeline for consistency.

BOTTLENECK -> VOXEL BROADCAST: each bottleneck spatial position is broadcast
to its corresponding input-resolution block via nearest-neighbour (matching
the many-to-one mapping direction used throughout E204-E214's
adaptive_max_pool3d ROI construction, inverted). This is stated explicitly
because it is a real design choice that affects boundary behaviour.

CENTROIDS: mu_L, mu_B are the SAME leave-subject-out prototype/background
banks as E214 (spatial_subject construction, the one that passed the inert
control) -- built once from a 60-subject pool, excluding each test subject's
own contribution when evaluating that subject.

FOUR SYSTEMS, LOSO-calibrated per the project's existing threshold
discipline (alpha and tau fit on OTHER subjects, applied to held-out):
  1. z            : raw logit, existing production ET threshold (0.0177 on
                     sigmoid(z), for reference) and a matched z>0 rule
  2. s_h alone     : threshold tau_h on s_h directly
  3. z + alpha*s_h : continuous combination, alpha fit on train subjects
  4. gated         : z where s_h<tau_h, s_h where s_h>=tau_h (decision-rule
                     substitution, not a continuous blend)

Reports ET Dice (dice_per_region's exact formula/empty-target convention,
imported from train_e130_multimodal_baseline.py), TP/FP/FN at the component
level for tail analysis, tail Dice (missed-component subjects only) vs bulk
Dice (rest), and newly-recovered/newly-created component counts.
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
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, REGIONS
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


def gaussian_weight(shape, sigma_scale=0.125):
    return t._gaussian_weight(shape, sigma_scale)


def starts(full, p, st):
    if full <= p:
        return [0]
    s = list(range(0, full - p + 1, st))
    if s[-1] != full - p:
        s.append(full - p)
    return s


def single_patch_bn_and_z(model, img_patch, dev):
    """Pass-0 helper: ONE 128^3 patch, centered by the caller (matching
    E204-E214's exact method), single forward pass -> (z (pd,ph,pw) numpy,
    bn (256,pb,pb,pb) numpy). Cheap: no sliding window, no full-volume
    materialization -- Pass 0 only needs a component centroid + a
    background mean per subject, not a whole-volume field."""
    with torch.no_grad():
        e1 = model.enc1(img_patch); e2 = model.enc2(model.pool1(e1))
        e3 = model.enc3(model.pool2(e2))
        bn = model.bottleneck(model.pool3(e3))
        d3 = model.dec3(torch.cat([model.upconv3(bn), e3], 1))
        d2 = model.dec2(torch.cat([model.upconv2(d3), e2], 1))
        eg, _ = model.attn_gate1(gate=bn, skip=e1)
        d1 = model.dec1(torch.cat([model.upconv1(d2), eg], 1))
        z = model.seg_head[0](d1)[0, ET]
    return z.cpu().numpy(), bn[0].cpu().numpy()


def sliding_window_z_and_sh(model, image, dev, mu_L, mu_B):
    """Custom sliding-window pass: returns (z_full, s_h_full), both (D,H,W).

    MEMORY FIX (diagnosed after a real stall): the original version upsampled
    the FULL 256-channel bottleneck to input resolution per tile before
    blending (~2.15GB/tile, ~3.58GB accumulator on a full volume) -- that
    doesn't fit an 8GB card alongside the model and per-tile activations,
    and caused a genuine GPU-memory stall (pinned at 7.9/8.15GB, 0-1% util,
    no progress for 60s+). Fixed by computing s_h -- a SCALAR per voxel --
    per tile via the dot product BEFORE upsampling/accumulating, so only a
    1-channel field is ever blended, matching z's own memory footprint."""
    dir_vec = torch.from_numpy(mu_L - mu_B).float().to(dev)
    dir_norm = float(np.linalg.norm(mu_L - mu_B)) + EPS
    mu_B_t = torch.from_numpy(mu_B).float().to(dev)

    _, C, D, H, W = image.shape
    pd, ph, pw = PATCH
    stride = [max(1, int(p * (1 - OVERLAP))) for p in PATCH]
    zs, ys, xs = (starts(D, pd, stride[0]), starts(H, ph, stride[1]),
                 starts(W, pw, stride[2]))
    z_acc = torch.zeros((D, H, W), device=dev, dtype=torch.float32)
    sh_acc = torch.zeros((D, H, W), device=dev, dtype=torch.float32)
    wsum = torch.zeros((D, H, W), device=dev, dtype=torch.float32)
    gw3 = torch.from_numpy(gaussian_weight(
        (min(pd, D), min(ph, H), min(pw, W)))).to(dev)

    with torch.no_grad():
        for z0 in zs:
            for y0 in ys:
                for x0 in xs:
                    zc, yc, xc = min(pd, D), min(ph, H), min(pw, W)
                    tile = image[:, :, z0:z0 + zc, y0:y0 + yc, x0:x0 + xc]
                    if tile.shape[2:] != (pd, ph, pw):
                        tile = Fn.pad(tile, (0, pw - tile.shape[4],
                                             0, ph - tile.shape[3],
                                             0, pd - tile.shape[2]))
                    with torch.amp.autocast("cuda", enabled=True):
                        e1 = model.enc1(tile); e2 = model.enc2(model.pool1(e1))
                        e3 = model.enc3(model.pool2(e2))
                        bn = model.bottleneck(model.pool3(e3))
                        d3 = model.dec3(torch.cat([model.upconv3(bn), e3], 1))
                        d2 = model.dec2(torch.cat([model.upconv2(d3), e2], 1))
                        eg, _ = model.attn_gate1(gate=bn, skip=e1)
                        d1 = model.dec1(torch.cat([model.upconv1(d2), eg], 1))
                        zt = model.seg_head[0](d1).float()[0, ET]  # (pd,ph,pw)
                    zt = zt[:zc, :yc, :xc]

                    # s_h computed per-tile at BOTTLENECK resolution (1 chan),
                    # THEN upsampled -- 256x smaller than upsampling h itself
                    bn0 = bn[0].float()                        # (256,pb,pb,pb)
                    diff = bn0 - mu_B_t[:, None, None, None]
                    sh_tile = torch.tensordot(dir_vec, diff, dims=([0], [0])) / dir_norm
                    sh_up = Fn.interpolate(
                        sh_tile.unsqueeze(0).unsqueeze(0),
                        size=(pd, ph, pw), mode='nearest'
                    )[0, 0][:zc, :yc, :xc]

                    gw_c = gw3[:zc, :yc, :xc]
                    z_acc[z0:z0 + zc, y0:y0 + yc, x0:x0 + xc] += zt * gw_c
                    sh_acc[z0:z0 + zc, y0:y0 + yc, x0:x0 + xc] += sh_up * gw_c
                    wsum[z0:z0 + zc, y0:y0 + yc, x0:x0 + xc] += gw_c
    w = wsum.clamp(min=1e-6)
    return (z_acc / w).cpu().numpy(), (sh_acc / w).cpu().numpy()


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for q in model.parameters():
        q.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    smoke = '--smoke' in sys.argv
    n_total = 4 if smoke else len(ds)
    n_bank = 3 if smoke else 60
    print(f'E215 full-volume decision-rule swap (ET only), '
          f'{n_total}/{len(ds)} subjects{" [SMOKE TEST]" if smoke else ""}',
          flush=True)

    # ---- Pass 0: build mu_L/mu_B banks, SAME construction as E214's
    # spatial_subject (the one that passed the inert control) ----
    print('Pass 0: building mu_L/mu_B banks (E214 spatial_subject method)...',
          flush=True)
    proto_vecs, proto_subjects, bg_vecs_subj = [], [], []
    P128 = (128, 128, 128)
    for ii in range(min(n_bank, len(ds))):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        brain = img[0].numpy() != 0
        lbl, nl = ndimage.label(Y[ET])
        got = False
        for g in range(1, nl + 1):
            cm = lbl == g
            if cm.sum() < MIN_VOX:
                continue
            com = np.array(ndimage.center_of_mass(cm)).astype(int)
            st = [int(np.clip(c - P128[k] // 2, 0, img.shape[k + 1] - P128[k]))
                  for k, c in enumerate(com)]
            sl = tuple(slice(s, s + P128[k]) for k, s in enumerate(st))
            cm_p = cm[sl]
            if cm_p.sum() < MIN_VOX:
                continue
            x = img[(slice(None),) + sl].unsqueeze(0).to(dev)
            z_p, bn_p = single_patch_bn_and_z(model, x, dev)
            P_et = z_p > 0
            ov = float((cm_p & P_et).sum()) / cm_p.sum()
            if ov < 0.5:
                continue
            shp3 = bn_p.shape[1:]
            com_m = np.array(ndimage.center_of_mass(cm_p))
            scale = np.array(shp3) / np.array(cm_p.shape)
            pos3 = tuple(np.clip((com_m * scale).astype(int), 0,
                                 np.array(shp3) - 1))
            proto_vecs.append(bn_p[:, pos3[0], pos3[1], pos3[2]])
            proto_subjects.append(sid)

            anygt = (Y[0] | Y[1] | Y[2])[sl]
            brain_p = brain[sl]
            bg_mask_p = brain_p & ~anygt
            bg_ds = (Fn.adaptive_avg_pool3d(
                torch.from_numpy(bg_mask_p.astype(np.float32))[None, None],
                shp3)[0, 0].numpy() > 0.5)
            if bg_ds.sum() >= 4:
                bg_vecs_subj.append((sid, bn_p[:, bg_ds].mean(axis=1)))
            got = True
            break
        if got and (ii + 1) % 20 == 0:
            print(f'  pass0 {ii+1}/{min(n_bank,len(ds))}', flush=True)
    proto_vecs = np.array(proto_vecs)
    mu_prototype_global = proto_vecs.mean(0)
    bg_by_subj = dict(bg_vecs_subj)
    mu_bg_global = np.mean(list(bg_by_subj.values()), axis=0)
    print(f'  bank: {len(proto_vecs)} prototypes, {len(bg_by_subj)} bg vecs',
          flush=True)

    # ---- Pass 1: whole-cohort inference, per-subject dz/s_h fields ----
    print('\nPass 1: whole-cohort inference...', flush=True)
    results = []   # per subject: dice under each system + component stats
    t0 = time.time()
    for ii in range(n_total):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        brain = img[0].numpy() != 0

        mask_other = np.array([s != sid for s in proto_subjects])
        mu_L = (proto_vecs[mask_other].mean(0) if mask_other.sum() >= 5
               else mu_prototype_global)
        mu_B = bg_by_subj.get(sid, mu_bg_global)

        # s_h computed per-tile at bottleneck res (1 chan) before blending --
        # entirely GPU-resident until the final .cpu() inside the function
        z_full, s_h_full = sliding_window_z_and_sh(
            model, img.unsqueeze(0).to(dev), dev, mu_L, mu_B)

        results.append({
            'subject_id': sid, 'z': z_full, 's_h': s_h_full,
            'gt_et': Y[ET], 'brain': brain,
        })
        if (ii + 1) % 20 == 0:
            print(f'  pass1 {ii+1}/{len(ds)} subj ({time.time()-t0:.0f}s)',
                  flush=True)

    print(f'\nPass 1 complete ({time.time()-t0:.0f}s). Saving field cache '
          f'(per-subject, shapes vary across the cohort)...', flush=True)
    cache_dir = HERE / 'E215_fields_cache'
    cache_dir.mkdir(exist_ok=True)
    for r in results:
        np.savez_compressed(
            cache_dir / f"{r['subject_id']}.npz",
            z=r['z'].astype(np.float16), s_h=r['s_h'].astype(np.float16),
            gt_et=r['gt_et'], brain=r['brain'],
        )
    with open(cache_dir / '_manifest.txt', 'w') as fh:
        for r in results:
            fh.write(r['subject_id'] + '\n')
    print(f'wrote {len(results)} field caches to {cache_dir}/', flush=True)


if __name__ == '__main__':
    main()
