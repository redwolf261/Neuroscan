"""E188 -- Cross-depth correspondence asymmetry at enc3/skip3. FROZEN. NO TRAINING.

HYPOTHESIS (preregistered): the causal dependence of the decoder output on
  (1) skip channel identity, (2) local spatial organization, (3) amplitude,
  (4) absolute spatial correspondence
differs between MISSED (L) and DETECTED (S) lesions at enc3, and differs from
the enc1 ordering E65 established.

E65 at enc1 (v3, ungated):
    translation +0.2143 (dominant) > channel +0.1009 > smoothing +0.0284
    ~ local permutation +0.0270
E65 did NOT stratify by detected-vs-missed. That split is what is new here.

PRIMARY ENDPOINT   : Delta_L - Delta_S, per perturbation.
SECONDARY ENDPOINT : ordering of the four effects within each of L, S, H.

KILL: if enc3 reproduces E65's enc1 ordering with no L-vs-S asymmetry, the
branch dies. A specific L-vs-S reversal/asymmetry is what licenses operator
design.

CRITICAL: enc3 feeds BOTH pool3->bottleneck AND the decoder skip cat3. All
perturbations here are applied to the SKIP PATH ONLY -- pool3 always receives
the ORIGINAL enc3. This is the E64 lesson (E62/E63 were invalidated by exactly
this shared-tensor confound) and it is what makes these numbers attributable to
the skip.

Effect metric, matching E186.1 so the two are comparable:
    dz = median(z_component) - median(z_shell)   on the raw ET logit
    Delta = dz(intact) - dz(perturbed)           (POSITIVE = damage)
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
PATCH, MIN_VOX, MAX_PER_SUBJ = 128, 5, 3
ROI_DIL = 8
ET = 0
SHIFT = 4          # translation magnitude at enc3 (= 16 input voxels)
SMOOTH_IT = 1      # amplitude/smoothing iterations


def forward_from_e3(model, e3_pool, e3_skip, e1, e2):
    """Frozen forward with INDEPENDENT tensors for the pooling and skip paths."""
    with torch.no_grad():
        b = model.bottleneck(model.pool3(e3_pool))
        d3 = model.dec3(torch.cat([model.upconv3(b), e3_skip], 1))
        d2 = model.dec2(torch.cat([model.upconv2(d3), e2], 1))
        eg, _ = model.attn_gate1(gate=b, skip=e1)
        d1 = model.dec1(torch.cat([model.upconv1(d2), eg], 1))
        z = model.seg_head[0](d1)
    return z[0]


# ---------------- the four perturbations, all ROI-restricted ----------------

def p_channel(F, roi, gen):
    """C1: permute CHANNEL identity inside the ROI. Every voxel keeps its
    spatial position; the channel ordering is shuffled."""
    out = F.clone()
    m = torch.from_numpy(roi).to(F.device)
    perm = torch.randperm(F.shape[0], generator=gen).to(F.device)
    out[:, m] = F[perm][:, m]
    return out


def p_local(F, roi, gen):
    """C2: permute spatial POSITIONS within the ROI (channel vector moves
    intact). Same manipulation as E186.1's P2."""
    out = F.clone()
    idx = torch.nonzero(torch.from_numpy(roi), as_tuple=False)
    if idx.shape[0] < 2:
        return out
    src = idx[torch.randperm(idx.shape[0], generator=gen)]
    out[:, idx[:, 0], idx[:, 1], idx[:, 2]] = F[:, src[:, 0], src[:, 1], src[:, 2]]
    return out


def p_amplitude(F, roi, gen):
    """C3: remove absolute magnitude, preserve the spatial pattern. Each
    channel is rescaled inside the ROI to unit std about its ROI mean, so the
    PATTERN survives and the AMPLITUDE does not."""
    out = F.clone()
    m = torch.from_numpy(roi).to(F.device)
    v = F[:, m]
    mu = v.mean(1, keepdim=True); sd = v.std(1, keepdim=True)
    out[:, m] = mu + (v - mu) / (sd + 1e-6)
    return out


def p_translate(F, roi, gen):
    """C4: absolute spatial CORRESPONDENCE. The ROI content is replaced by the
    content SHIFTED by a fixed offset -- identical local texture and channel
    statistics, wrong absolute position relative to the decoder state."""
    out = F.clone()
    sh = (SHIFT, SHIFT, SHIFT)
    rolled = torch.roll(F, shifts=sh, dims=(1, 2, 3))
    m = torch.from_numpy(roi).to(F.device)
    out[:, m] = rolled[:, m]
    return out


PERTS = [('channel', p_channel), ('local_spatial', p_local),
         ('amplitude', p_amplitude), ('translation', p_translate)]


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for q in model.parameters():
        q.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    print(f'E188 cross-depth correspondence, enc3 SKIP ONLY, {len(ds)} subjects',
          flush=True)

    rows = []
    t0 = time.time()
    for ii in range(len(ds)):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                         t.PATCH, t.SW_OVERLAP, 3, dev, True)
        P = probs >= 0.5
        brain = img[0].numpy() != 0
        lbl, nl = ndimage.label(Y[ET])
        missed, succ = [], []
        for g in range(1, nl + 1):
            cm = lbl == g
            if cm.sum() < MIN_VOX:
                continue
            ov = float((cm & P[ET]).sum()) / cm.sum()
            if ov == 0:
                missed.append((int(cm.sum()), g))
            elif ov >= 0.5:
                succ.append((int(cm.sum()), g))
        missed.sort(reverse=True); succ.sort(reverse=True)
        picks = [('L', g) for _, g in missed[:MAX_PER_SUBJ]] + \
                [('S', g) for _, g in succ[:MAX_PER_SUBJ]]
        if not picks:
            continue
        for pop0, g in picks:
            cm = lbl == g
            com = np.array(ndimage.center_of_mass(cm)).astype(int)
            st = [int(np.clip(c - PATCH // 2, 0, s - PATCH))
                  for c, s in zip(com, cm.shape)]
            sl = tuple(slice(s, s + PATCH) for s in st)
            cm_p = cm[sl]
            if cm_p.sum() < MIN_VOX:
                continue
            anygt = (Y[0] | Y[1] | Y[2])[sl]; brain_p = brain[sl]
            x = img[(slice(None),) + sl].unsqueeze(0).to(dev)
            with torch.no_grad():
                e1 = model.enc1(x); e2 = model.enc2(model.pool1(e1))
                e3 = model.enc3(model.pool2(e2))
            z0 = forward_from_e3(model, e3, e3, e1, e2)
            lg0 = z0[ET].cpu().numpy()

            rng = np.random.default_rng(abs(hash((sid, g))) % (2**31))
            forbid = ndimage.binary_dilation(anygt, iterations=6)
            idx = np.argwhere(cm_p); lo = idx.min(0); ext = idx.max(0) - lo + 1
            rel = idx - lo; dims = np.array(cm_p.shape)
            best, bestv = None, -np.inf
            for _ in range(200):
                hi = dims - ext
                if (hi <= 0).any():
                    break
                off = np.array([rng.integers(0, h + 1) for h in hi])
                cand = np.zeros_like(cm_p); cand[tuple((rel + off).T)] = True
                if (cand & forbid).any() or not brain_p[cand].all():
                    continue
                v = float(lg0[cand].mean())
                if v > bestv:
                    bestv, best = v, cand
            if best is None:
                continue
            shp3 = tuple(e3.shape[2:])
            for pop, msk in [(pop0, cm_p), ('H', best)]:
                roi_in = ndimage.binary_dilation(msk, iterations=ROI_DIL) & brain_p
                r3 = (Fn.adaptive_max_pool3d(
                    torch.from_numpy(roi_in.astype(np.float32))[None, None],
                    shp3)[0, 0] > 0.5).numpy()
                if r3.sum() < 8:
                    continue
                shell = (ndimage.binary_dilation(msk, iterations=6) & ~msk
                         & brain_p & ~anygt)
                if shell.sum() < MIN_VOX:
                    continue
                dz0 = float(np.median(lg0[msk]) - np.median(lg0[shell]))
                base = {'subject_id': sid, 'comp_id': int(g), 'population': pop,
                        'vox': int(msk.sum()), 'roi_e3_vox': int(r3.sum()),
                        'dz_intact': dz0}
                for pname, pfn in PERTS:
                    gen = torch.Generator(device='cpu')
                    gen.manual_seed(abs(hash((sid, g, pop, pname))) % (2**31))
                    e3p = pfn(e3[0], r3, gen).unsqueeze(0)
                    # SKIP PATH ONLY: pool3 receives the ORIGINAL e3 (E64 lesson)
                    z = forward_from_e3(model, e3, e3p, e1, e2)
                    lg = z[ET].cpu().numpy()
                    dz = float(np.median(lg[msk]) - np.median(lg[shell]))
                    rec = dict(base)
                    rec.update({'perturb': pname, 'dz_pert': dz,
                                'Delta': dz0 - dz})   # POSITIVE = damage
                    rows.append(rec)
        if (ii + 1) % 15 == 0:
            print(f'  {ii+1}/{len(ds)} subj, {len(rows)} rows '
                  f'({time.time()-t0:.0f}s)', flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(HERE / 'E188_correspondence.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    print(f'\nwrote E188_correspondence.csv ({len(rows)} rows) '
          f'pops={dict(Counter(r["population"] for r in rows))}  '
          f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
