"""E202 -- GATE 3A/3B: linear-probe readability of ET at every trunk stage.

Question: where does ET information become linearly recoverable inside the
network, and where is it lost/distorted? Trains a per-voxel LOGISTIC
REGRESSION (the simplest possible linear probe -- a 1x1x1 conv is exactly
this) on FROZEN activations from a trained checkpoint, at each of 7 stages:
enc1, enc2, enc3, bottleneck, dec3, dec2, dec1.

Design constraints, deliberately mirroring E143's own cross-fitting so the
result is comparable to O_i on the same footing:
  - 5-fold cross-fitted per stage: probe trained on 4 folds' subjects,
    evaluated on the held-out fold. No probe ever sees its own eval
    subject's activations for training.
  - Frozen backbone: the trained E131_v5control_seed0 checkpoint's forward
    pass generates activations; the probe is a SEPARATE linear layer fit
    post-hoc, backbone never updated. This directly answers "is the
    information already linearly present" -- if a nonlinear head (the
    actual seg_head, also just 1x1x1 conv = also linear) still fails to
    use it, that's a readout/decision-formation problem, not an encoding
    problem.
  - Compares D_probe(stage) against D_seghead (the network's own actual
    ET Dice at dec1, i.e. what seg_head already achieves) at every stage.

Note dec1's OWN probe is directly comparable to boundary_head, which is
ALREADY a linear probe on dec1.detach() trained end-to-end with weight
mu=0.1 -- but boundary_head is trained on ALL 3 regions jointly (not ET
specifically) and never cross-fitted (it sees every subject during
training, no held-out generalization check). E202's dec1 probe is the
cross-fitted, ET-specific, generalization-honest analogue -- a stricter
version of the same question.

Run from repo root: python experiments/exp_recoverability/10_e202_linear_probe_trajectory.py
Cost: ~1 forward pass per subject (activations cached), then cheap sklearn
fits per stage/fold. GPU for the forward passes, CPU for the probes.
"""
import sys, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, '.')
from neuroscan_3d_v5 import UNet3D_v5
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, REGIONS

ROOT = Path('.')
CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
OUT_JSON = ROOT / 'experiments/exp_e12_eggo_m/E202_linear_probe_trajectory.json'
PATCH = (128, 128, 128)  # single centered crop per subject -- NOT sliding window;
                         # this is an activation-geometry probe, not a Dice benchmark,
                         # matching E188's own precedent (single centered 128^3 crop)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SUBJECTS = int(__import__('os').environ.get('E202_N', '60'))  # subsample for tractable runtime
MAX_VOX_PER_SUBJECT = 4000  # per-stage per-subject voxel cap for probe fitting/eval

STAGES = ['enc1', 'enc2', 'enc3', 'bottleneck', 'dec3', 'dec2', 'dec1']


def centered_crop(image, target, patch):
    D, H, W = image.shape[1:]
    pd, ph, pw = patch
    z = max(0, (D - pd) // 2); y = max(0, (H - ph) // 2); x = max(0, (W - pw) // 2)
    img = image[:, z:z+pd, y:y+ph, x:x+pw]
    tgt = target[:, z:z+pd, y:y+ph, x:x+pw]
    return img, tgt


def get_activations(model, x):
    """Manual forward pass, mirroring UNet3D_v5.forward(), returning every
    trunk stage's own feature map (pre-head). Duplicated deliberately
    (not calling model.forward()) so we can grab intermediate tensors
    without modifying the frozen architecture file."""
    with torch.no_grad():
        enc1 = model.enc1(x); pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1); pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2); pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)
        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1); dec3 = model.dec3(cat3)
        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1); dec2 = model.dec2(cat2)
        upconv1 = model.upconv1(dec2)
        enc1_gated, _ = model.attn_gate1(gate=bottleneck, skip=enc1)
        cat1 = torch.cat([upconv1, enc1_gated], dim=1); dec1 = model.dec1(cat1)
        probs = model.seg_head(dec1)
    return dict(enc1=enc1, enc2=enc2, enc3=enc3, bottleneck=bottleneck,
                dec3=dec3, dec2=dec2, dec1=dec1, probs=probs)


def flatten_stage(feat, et_target_fullres, rng, max_vox):
    """feat: (1,C,d,h,w) at this stage's own resolution. et_target_fullres:
    (D,H,W) binary at FULL patch resolution. Downsamples the label to the
    stage's resolution via nearest-neighbor (matching what the network's
    own D4 aux head does via avg_pool, but nearest is more appropriate for
    a binary presence/absence probe target), then flattens to (N,C)/(N,)."""
    _, C, d, h, w = feat.shape
    D, H, W = et_target_fullres.shape
    if (d, h, w) != (D, H, W):
        t = torch.from_numpy(et_target_fullres[None, None].astype(np.float32))
        t_ds = F.interpolate(t, size=(d, h, w), mode='nearest').numpy()[0, 0]
    else:
        t_ds = et_target_fullres
    X = feat[0].permute(1, 2, 3, 0).reshape(-1, C).cpu().numpy()
    y = t_ds.reshape(-1).astype(np.int32)
    n = X.shape[0]
    if n > max_vox:
        idx = rng.choice(n, max_vox, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def dice_bin(pred, y):
    inter = (pred * y).sum()
    den = pred.sum() + y.sum()
    return float(2 * inter / den) if den > 0 else 1.0


def main():
    print(f'device: {device}')
    ck = torch.load(str(CKPT), map_location=device, weights_only=False)
    model = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(ck['model_state'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val', val_split=0.1, patch_size=PATCH)
    rng = np.random.default_rng(0)
    all_idx = rng.permutation(len(ds))[:N_SUBJECTS]
    print(f'subsampling {len(all_idx)} of {len(ds)} validation subjects for probe fitting', flush=True)

    # ---- pass 1: cache activations + full-res ET target + seg_head's own Dice ----
    cache = {}
    et_idx = REGIONS.index('ET')
    t0 = time.time()
    for k, i in enumerate(all_idx):
        image, target, sid = ds[int(i)]
        img_c, tgt_c = centered_crop(image, target, PATCH)
        x = img_c.unsqueeze(0).to(device)
        acts = get_activations(model, x)
        et_full = tgt_c[et_idx].numpy().astype(np.float32)  # (128,128,128)
        seghead_pred = (acts['probs'][0, et_idx].cpu().numpy() >= 0.5).astype(np.float32)
        d_seghead = dice_bin(seghead_pred, et_full)
        cache[sid] = dict(acts={s: acts[s].cpu() for s in STAGES}, et_full=et_full, d_seghead=d_seghead)
        if (k + 1) % 15 == 0:
            print(f'  cached {k+1}/{len(all_idx)}  ({time.time()-t0:.0f}s)', flush=True)
    ids = list(cache.keys())
    print(f'cached {len(ids)} subjects, {time.time()-t0:.0f}s\n', flush=True)

    # ---- pass 2: per-stage, 5-fold cross-fitted linear probe ----
    K = 5
    folds = np.array_split(np.array(ids), K)
    stage_results = {s: {} for s in STAGES}
    probe_rng = np.random.default_rng(1)

    for stage in STAGES:
        print(f'probing {stage} ...', flush=True)
        for fi, te in enumerate(folds):
            tr = [s for s in ids if s not in set(te)]
            Xtr, ytr = [], []
            for s in tr:
                X, y = flatten_stage(cache[s]['acts'][stage].to(device).unsqueeze(0) if False else cache[s]['acts'][stage].unsqueeze(0),
                                      cache[s]['et_full'], probe_rng, MAX_VOX_PER_SUBJECT)
                Xtr.append(X); ytr.append(y)
            Xtr = np.concatenate(Xtr, 0); ytr = np.concatenate(ytr, 0)
            if ytr.sum() < 10 or ytr.sum() > len(ytr) - 10:
                clf = None  # degenerate fold (near-empty or near-full), skip
            else:
                clf = LogisticRegression(max_iter=300, C=1.0)
                clf.fit(Xtr, ytr)
            for s in te:
                feat = cache[s]['acts'][stage].unsqueeze(0)
                _, C, d, h, w = feat.shape
                Xall = feat[0].permute(1, 2, 3, 0).reshape(-1, C).numpy()
                yall_full = cache[s]['et_full']
                D, H, W = yall_full.shape
                if (d, h, w) != (D, H, W):
                    import torch as _t
                    t_ds = F.interpolate(_t.from_numpy(yall_full[None, None]), size=(d, h, w), mode='nearest').numpy()[0, 0]
                else:
                    t_ds = yall_full
                yall = t_ds.reshape(-1).astype(np.int32)
                if clf is None or yall.sum() == 0:
                    d_probe = float('nan')
                else:
                    pred = clf.predict(Xall).astype(np.float32)
                    d_probe = dice_bin(pred, yall.astype(np.float32))
                stage_results[stage][s] = d_probe

    # ---- summary ----
    out = {'stages': STAGES, 'per_subject': {}, 'summary': {}}
    for s in ids:
        out['per_subject'][s] = {'d_seghead': cache[s]['d_seghead'],
                                  **{st: stage_results[st].get(s, float('nan')) for st in STAGES}}
    print(f'\n{"="*80}')
    print('E202 -- LINEAR PROBE TRAJECTORY (mean cross-fitted Dice, ET, downsampled label)')
    print(f'{"="*80}')
    print(f'{"stage":12s} {"mean D_probe":>14s} {"n_valid":>9s}   (seg_head D at dec1 full-res = {np.mean([cache[s]["d_seghead"] for s in ids]):.4f})')
    for st in STAGES:
        vals = np.array([stage_results[st][s] for s in ids])
        valid = vals[~np.isnan(vals)]
        out['summary'][st] = {'mean': float(valid.mean()) if len(valid) else float('nan'),
                               'n_valid': int(len(valid))}
        print(f'{st:12s} {(valid.mean() if len(valid) else float("nan")):14.4f} {len(valid):9d}')

    json.dump(out, open(OUT_JSON, 'w'), indent=1)
    print(f'\nsaved {OUT_JSON}')


if __name__ == '__main__':
    main()
