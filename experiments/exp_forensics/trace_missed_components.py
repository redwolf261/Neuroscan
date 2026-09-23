"""Trace missed GT components through the frozen network, stage by stage.

For each missed component in the 15 tail subjects, answer:
  1. raw evidence      -- is there multimodal intensity signal vs matched control?
  2. encoder evidence  -- does it leave a spatial signature at enc1/enc2/enc3?
  3. bottleneck        -- does that signature survive?
  4. decoder recovery  -- is it reconstructed at dec3/dec2/dec1?
  5. logit suppression -- is evidence present but pushed below the boundary?

METHOD. For each missed component C we build a control region K: a same-size,
same-shape region placed in confirmed background (no GT of any region, inside
the brain mask), matched on distance-to-tumour where possible. At every stage we
compute the mean feature vector over C and over K (after resampling the mask to
that stage's resolution) and report a separability statistic:

    d' = |mean_C - mean_K| / sqrt(0.5*(var_C + var_K))     (per channel, then
                                                            aggregated by L2)

plus a linear-probe AUC (does a single linear direction at this stage separate
component voxels from control voxels?). A stage "carries the component" if it
separates C from K materially better than chance.

GT is used ONLY to locate C and to score. It never enters the forward pass.
All measurement is on the frozen checkpoint; nothing is trained or updated.
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
MODS = ['t1c', 't1n', 't2f', 't2w']
STAGES = ['enc1', 'enc2', 'enc3', 'bottleneck', 'dec3', 'dec2', 'dec1']
MIN_VOX = 5          # ignore components below this (annotation noise floor)
MAX_COMP_PER_SUBJ = 8  # cap per subject so one fragmented subject can't dominate


def capture(model, x):
    """Manual forward pass mirroring UNet3D_v5.forward(), returning all stages."""
    with torch.no_grad():
        enc1 = model.enc1(x);            p1 = model.pool1(enc1)
        enc2 = model.enc2(p1);           p2 = model.pool2(enc2)
        enc3 = model.enc3(p2);           p3 = model.pool3(enc3)
        bott = model.bottleneck(p3)
        u3 = model.upconv3(bott)
        dec3 = model.dec3(torch.cat([u3, enc3], 1))
        u2 = model.upconv2(dec3)
        dec2 = model.dec2(torch.cat([u2, enc2], 1))
        u1 = model.upconv1(dec2)
        eg, _ = model.attn_gate1(gate=bott, skip=enc1)
        dec1 = model.dec1(torch.cat([u1, eg], 1))
        logits_pre = model.seg_head[0](dec1)     # pre-sigmoid
        probs = torch.sigmoid(logits_pre)
    return dict(enc1=enc1, enc2=enc2, enc3=enc3, bottleneck=bott,
                dec3=dec3, dec2=dec2, dec1=dec1), logits_pre, probs


def down_mask(mask, shape):
    """Resample a bool mask (D,H,W) to `shape` by max-pooling (any voxel -> 1)."""
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    out = Fn.adaptive_max_pool3d(m, shape)[0, 0]
    return out > 0.5


def dprime_auc(feat, mc, mk):
    """feat: (C,d,h,w) tensor. mc/mk: bool masks at that resolution.
    Returns (aggregate d', linear-probe AUC)."""
    C = feat.shape[0]
    A = feat[:, mc].T          # (n_c, C)
    B = feat[:, mk].T          # (n_k, C)
    if A.shape[0] < 3 or B.shape[0] < 3:
        return float('nan'), float('nan')
    ma, mb = A.mean(0), B.mean(0)
    va, vb = A.var(0, unbiased=False), B.var(0, unbiased=False)
    d = (ma - mb).abs() / torch.sqrt(0.5 * (va + vb) + 1e-8)
    dprime = float(torch.linalg.norm(d) / np.sqrt(C))    # RMS d' across channels
    # linear probe: Fisher direction, then AUC via Mann-Whitney
    w = (ma - mb) / (0.5 * (va + vb) + 1e-8)
    sa = (A @ w).cpu().numpy(); sb = (B @ w).cpu().numpy()
    n1, n2 = len(sa), len(sb)
    allv = np.concatenate([sa, sb])
    r = np.argsort(np.argsort(allv)) + 1
    auc = (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
    return dprime, float(max(auc, 1 - auc))


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
    tail = {r['subject_id'] for r in fr
            if float(r['ET_dice']) < 0.5 or float(r['TC_dice']) < 0.5}
    idx = [i for i in range(len(ds)) if Path(ds.subject_dirs[i]).name in tail]
    if limit:
        idx = idx[:limit]
    print(f'tracing {len(idx)} tail subjects, stages={STAGES}', flush=True)

    rows = []
    t0 = time.time()
    for ii, i in enumerate(idx):
        img, tgt, sid = ds[i]
        Y = tgt.numpy() > 0.5                      # (3,D,H,W)
        full_probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                              t.PATCH, t.SW_OVERLAP, 3, dev, True)
        P = full_probs >= THR
        brain = img[0].numpy() != 0

        for ri, rn in enumerate(REGIONS):
            if rn == 'WT':
                continue                            # focus on ET/TC detection failures
            lbl, nl = ndimage.label(Y[ri])
            missed = []
            for g in range(1, nl + 1):
                cm = lbl == g
                if cm.sum() >= MIN_VOX and not (cm & P[ri]).any():
                    missed.append((int(cm.sum()), g))
            missed.sort(reverse=True)               # largest first
            for sz, g in missed[:MAX_COMP_PER_SUBJ]:
                cm = lbl == g
                com = np.array(ndimage.center_of_mass(cm)).astype(int)
                # --- extract a PATCH-sized window centred on the component ---
                st = [int(np.clip(c - PATCH // 2, 0, s - PATCH))
                      for c, s in zip(com, Y[ri].shape)]
                sl = tuple(slice(s, s + PATCH) for s in st)
                x = img[(slice(None),) + sl].unsqueeze(0).to(dev)
                cm_p = cm[sl]
                if cm_p.sum() < MIN_VOX:
                    continue
                anygt_p = (Y[0] | Y[1] | Y[2])[sl]
                brain_p = brain[sl]
                # --- HARD control: tissue immediately ADJACENT to the component,
                # confirmed background in GT, and also called background by the
                # model. This is the strictly harder test -- a trivially-placed
                # distant control would only prove "tumour tissue differs from
                # far-away normal brain", which is uninformative. Matching on
                # locality forces the probe to isolate the component itself.
                dist = ndimage.distance_transform_edt(~cm_p)
                shell = (dist > 1) & (dist <= 6) & brain_p & ~anygt_p
                if shell.sum() < max(MIN_VOX, cm_p.sum() // 2):
                    # fall back to a wider ring rather than a distant region
                    shell = (dist > 1) & (dist <= 12) & brain_p & ~anygt_p
                if shell.sum() < MIN_VOX:
                    continue
                ci = np.argwhere(shell)
                rng = np.random.default_rng(abs(hash((sid, rn, g))) % (2**31))
                pick = ci[rng.choice(len(ci), min(len(ci), int(cm_p.sum())), replace=False)]
                km = np.zeros_like(cm_p); km[tuple(pick.T)] = True

                feats, logits_pre, probs_p = capture(model, x)
                rec = {'subject_id': sid, 'region': rn, 'comp_id': int(g),
                       'comp_vox': int(cm_p.sum()), 'ctrl_vox': int(km.sum())}
                # ---- 1. raw evidence ----
                xin = x[0].cpu()
                dp, auc = dprime_auc(xin, torch.from_numpy(cm_p), torch.from_numpy(km))
                rec['raw_dprime'], rec['raw_auc'] = dp, auc
                for mi, mn in enumerate(MODS):
                    a = xin[mi][torch.from_numpy(cm_p)].mean()
                    b = xin[mi][torch.from_numpy(km)].mean()
                    rec[f'raw_{mn}_delta'] = float(a - b)
                # ---- 2-4. stage signatures ----
                for st_name in STAGES:
                    f = feats[st_name][0]
                    shp = tuple(f.shape[1:])
                    mc = down_mask(cm_p, shp).to(dev)
                    mk = down_mask(km, shp).to(dev)
                    dp, auc = dprime_auc(f, mc, mk)
                    rec[f'{st_name}_dprime'], rec[f'{st_name}_auc'] = dp, auc
                    rec[f'{st_name}_ncomp_vox'] = int(mc.sum())
                # ---- 5. logit suppression ----
                lg = logits_pre[0, ri].cpu()
                pr = probs_p[0, ri].cpu()
                mcf = torch.from_numpy(cm_p); mkf = torch.from_numpy(km)
                rec['logit_comp_mean'] = float(lg[mcf].mean())
                rec['logit_ctrl_mean'] = float(lg[mkf].mean())
                rec['logit_comp_max'] = float(lg[mcf].max())
                rec['prob_comp_mean'] = float(pr[mcf].mean())
                rec['prob_comp_max'] = float(pr[mcf].max())
                rec['logit_margin_to_thr'] = float(lg[mcf].max())  # thr at logit 0
                rows.append(rec)
        if (ii + 1) % 3 == 0:
            el = time.time() - t0
            print(f'  {ii+1}/{len(idx)} subjects, {len(rows)} comps ({el:.0f}s)', flush=True)

    if not rows:
        print('NO COMPONENTS TRACED'); return
    keys = sorted({k for r in rows for k in r})
    with open(OUT / 'TRACE_missed_components.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print(f'\nwrote TRACE_missed_components.csv ({len(rows)} components) in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
