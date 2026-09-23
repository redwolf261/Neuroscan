"""E208 -- Hypothesis-Relative Segmentation (HRS) falsification test. FROZEN. NO TRAINING.

HRS proposal (user, 2026-09-23): instead of learning P(Y(x)=1|X) per voxel,
learn the VALUE of a segmentation-hypothesis transition Y0 -> Y1 = Y0 (+) region.
"Does inserting this region make the image more internally compatible?"

STRUCTURAL FACT CHECKED BEFORE BUILDING THIS: UNet3D_v5.forward(x) takes ONLY
the image. There is no Y-conditioned pathway anywhere in the architecture --
the network has never been trained to evaluate a segmentation hypothesis, only
to produce one. So Delta_E(X,Y1,Y0) cannot be read off any existing learned
quantity. The only available proxy from a FROZEN forward pass is: how does the
network's OWN prediction, in the neighbourhood of the inserted region, respond
to local evidence being perturbed as if that region were foreground -- i.e.
a causal do(X) probe, structurally the same FAMILY as E107 (IG-flagged region
disruption), NOT the same test (E107 disrupted evidence; HRS asks whether
inserting a hypothesis changes the network's read of surrounding compatibility).

CRITICAL PRIOR RESULT THIS TEST MUST CHECK, NOT IGNORE: E107 found FN
(missed-lesion) sites sit in a GENUINELY FLATTER, less input-reactive
representation regime than TP sites (Delta_IG logit-space: TP=20.53 vs
FN=8.75, p<0.0001, double-corrected/verified). If HRS's Delta_E is fundamentally
another input-response measurement, E107's flatness could sink it for a reason
that has NOTHING to do with whether "hypothesis value" is the right primitive.
This script explicitly measures reactivity/flatness alongside Delta_E so a
null result can be attributed correctly.

PROXY CONSTRUCTION (what "inserting a hypothesis" means for a frozen model):
For a candidate region R (a missed GT component for L, the detected component
for S, a matched hard-negative for H), define two conditions:
  Y0 : the network's OWN unmodified forward pass -- baseline compatibility
       context, measured as the mean raw ET logit in the SHELL immediately
       around R (does the surrounding tissue currently look "background-like"
       or "lesion-adjacent-like"?).
  Y1 : the SAME shell logit, but AFTER splicing tumor-consistent evidence
       into R by pasting a DIFFERENT tumor's t1c/co-registered modality
       texture into region R's location (a real hypothesis insertion at the
       INPUT level -- the only level at which this frozen model can actually
       be made to "see" a hypothesis, since it has no Y-conditioned pathway).
       This is the closest honest realization of "insert the region and see
       if the image becomes more internally compatible" available without
       training a new architecture.
  Delta_E = shell_logit(Y1) - shell_logit(Y0)   -- does the NEIGHBOURHOOD's
       own prediction shift toward "this looks like a lesion boundary" once
       hypothesis-consistent evidence is inserted?

Also measured (the E107 check): REACTIVITY = |z(Y1) - z(Y0)| inside R itself,
comparable in spirit to E107's Delta_IG -- if L's reactivity is much smaller
than S's, that is the E107 flatness effect reproducing, and it must be
reported as the explanation for any null on Delta_E, not folded silently in.

KILL: if Delta_E does not discriminate L from H (Mann-Whitney n.s., or
magnitude negligible), AND/OR if the null is explained by L's reactivity
being uniformly near-zero (E107 flatness) -- report both explicitly.
PASS: Delta_E discriminates L from H beyond what reactivity alone predicts
(partial correlation controlling for reactivity survives).
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage, stats

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
ET = 0
# MODALITIES = ("t1c","t1n","t2f","t2w") -- verified against
# Dataset/brats_multimodal_dataset.py before use. t1c is channel 0.
T1C_CH = 0


def raw_logits(model, x):
    """Manual forward to the PRE-SIGMOID seg_head output -- model(x) returns
    a dict with 'probs' (post-evidential, not raw logits) and no 'logits'
    key; every other script this session (E188-E207) reads the raw logit via
    seg_head[0](d1) on a hand-walked forward pass, matched here exactly."""
    with torch.no_grad():
        e1 = model.enc1(x); e2 = model.enc2(model.pool1(e1))
        e3 = model.enc3(model.pool2(e2)); bn = model.bottleneck(model.pool3(e3))
        d3 = model.dec3(torch.cat([model.upconv3(bn), e3], 1))
        d2 = model.dec2(torch.cat([model.upconv2(d3), e2], 1))
        eg, _ = model.attn_gate1(gate=bn, skip=e1)
        d1 = model.dec1(torch.cat([model.upconv1(d2), eg], 1))
        z = model.seg_head[0](d1)
    return z[0]


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for q in model.parameters():
        q.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    print(f'E208 HRS falsification, {len(ds)} subjects', flush=True)

    # ---- gather a small bank of "donor" tumor-core patches for splicing ----
    donors = []
    for i in range(15):
        img, tgt, sid = ds[i]
        Y = tgt.numpy() > 0.5
        lbl, nl = ndimage.label(Y[ET])
        if nl == 0:
            continue
        sizes = [(int((lbl == g).sum()), g) for g in range(1, nl + 1)]
        sizes.sort(reverse=True)
        g = sizes[0][1]
        cm = lbl == g
        com = np.array(ndimage.center_of_mass(cm)).astype(int)
        r = 6
        lo = np.clip(com - r, 0, np.array(img.shape[1:]) - 2 * r)
        patch = img[:, lo[0]:lo[0]+2*r, lo[1]:lo[1]+2*r, lo[2]:lo[2]+2*r].numpy()
        donors.append(patch)
    print(f'  built {len(donors)} donor texture patches', flush=True)

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
            img_p = img[(slice(None),) + sl].clone()
            z0 = raw_logits(model, img_p.unsqueeze(0).to(dev))
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

            for pop, msk in [(pop0, cm_p), ('H', best)]:
                shell = (ndimage.binary_dilation(msk, iterations=6) & ~msk
                         & brain_p & ~anygt)
                if shell.sum() < MIN_VOX:
                    continue
                z_shell_Y0 = float(lg0[shell].mean())
                z_region_Y0 = float(lg0[msk].mean())

                com_m = np.array(ndimage.center_of_mass(msk)).astype(int)
                donor = donors[rng.integers(0, len(donors))]
                r = donor.shape[1] // 2
                img_p1 = img_p.clone()
                lo3 = np.clip(com_m - r, 0, np.array(img_p.shape[1:]) - 2 * r)
                sl3 = tuple(slice(a, a + 2 * r) for a in lo3)
                donor_t = torch.from_numpy(donor)
                for c in range(img_p.shape[0]):
                    img_p1[c][sl3] = donor_t[c]

                z1 = raw_logits(model, img_p1.unsqueeze(0).to(dev))
                lg1 = z1[ET].cpu().numpy()
                z_shell_Y1 = float(lg1[shell].mean())
                z_region_Y1 = float(lg1[msk].mean())

                rows.append({
                    'subject_id': sid, 'comp_id': int(g), 'population': pop,
                    'z_shell_Y0': z_shell_Y0, 'z_shell_Y1': z_shell_Y1,
                    'delta_E': z_shell_Y1 - z_shell_Y0,
                    'z_region_Y0': z_region_Y0, 'z_region_Y1': z_region_Y1,
                    'reactivity': abs(z_region_Y1 - z_region_Y0),
                    'vox': int(msk.sum()),
                })
        if (ii + 1) % 25 == 0:
            print(f'  {ii+1}/{len(ds)} subj, {len(rows)} rows '
                  f'({time.time()-t0:.0f}s)', flush=True)

    with open(HERE / 'E208_hrs.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    from collections import Counter
    print(f'\nwrote E208_hrs.csv ({len(rows)} rows) '
          f'pops={dict(Counter(r["population"] for r in rows))}  '
          f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
