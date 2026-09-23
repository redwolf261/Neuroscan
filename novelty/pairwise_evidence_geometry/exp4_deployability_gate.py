"""EXPERIMENT 4 -- DEPLOYABILITY GATE.

Can the missed components be DISCOVERED from the raw FP32 ET logit field with
NO ground truth, and does the forensic separation survive discovery?

Pipeline, per subject, per region (ET, TC):
  1. raw fp32 pre-sigmoid logits (no clamp, no fp16, no sigmoid)
  2. candidate masks  M_k = {x : z(x) > k}  for FIXED negative k
  3. connected-component label each M_k, discard comps < MIN_VOX
  4. per candidate, compute ONLY inference-available features:
         A(C) = median(z_C)                      absolute evidence
         S(C) = IQR(z_C)                         internal structure
         R(C) = median(z_C) - median(z_B)        local contrast vs its own shell
     where B(C) = dilate(C, r) \\ C, restricted to brain
  5. label each candidate by its GT overlap -- USED ONLY FOR SCORING, never
     as a feature and never in generation.

Thresholds are fixed in advance and are NOT tuned after seeing results.

Reported: missed-lesion candidate recall, candidates/subject, false
candidates/subject, precision, size distribution, A/R/S distributions, and the
L-vs-hard-negative separation AMONG DISCOVERED CANDIDATES (the decisive test).
"""
import sys, csv, json, time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'exp_e12_eggo_m' / 'e130'))
from neuroscan_3d_v5 import UNet3D_v5
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, REGIONS
import importlib.util

spec = importlib.util.spec_from_file_location(
    "t", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["e"]; spec.loader.exec_module(t); sys.argv = _a
sys.path.insert(0, str(HERE))
from exp2_raw_logit_physics import sliding_raw_logits

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
KS = [-2.0, -4.0, -6.0, -8.0, -10.0, -12.0, -16.0]      # FIXED, not tuned
MIN_VOX = 5
SHELL_R = 6
MAX_CAND_PER_K = 20000     # safety cap; if hit, that k is reported as exploding


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    fr = list(csv.DictReader(open(ROOT / 'experiments/exp_forensics/FORENSIC_per_subject.csv')))
    tail = {r['subject_id'] for r in fr
            if float(r['ET_dice']) < 0.5 or float(r['TC_dice']) < 0.5}
    idx = [i for i in range(len(ds)) if Path(ds.subject_dirs[i]).name in tail]
    print(f'deployability gate on {len(idx)} tail subjects; k = {KS}', flush=True)

    cand_rows, tgt_rows = [], []
    t0 = time.time()
    for ii, i in enumerate(idx):
        img, tgt, sid = ds[i]
        Y = tgt.numpy() > 0.5
        RAW = sliding_raw_logits(model, img.unsqueeze(0).to(dev), t.PATCH, t.SW_OVERLAP, dev)
        brain = img[0].numpy() != 0
        P = RAW > 0.0

        for ri, rn in [(0, 'ET'), (1, 'TC')]:
            lg = RAW[ri]
            gt = Y[ri]
            # --- the targets we must rediscover: GT comps with ZERO predicted overlap ---
            glbl, gn = ndimage.label(gt)
            targets = {}
            for g in range(1, gn + 1):
                cm = glbl == g
                if cm.sum() >= MIN_VOX and not (cm & P[ri]).any():
                    targets[g] = cm
            for g, cm in targets.items():
                tgt_rows.append({'subject_id': sid, 'region': rn, 'gt_comp': int(g),
                                 'vox': int(cm.sum()),
                                 'median_z': float(np.median(lg[cm]))})
            if not targets:
                continue

            for k in KS:
                M = (lg > k) & brain
                lab, nl = ndimage.label(M)
                if nl == 0:
                    continue
                exploded = nl > MAX_CAND_PER_K
                sizes = ndimage.sum(np.ones_like(lab), lab, range(1, nl + 1))
                keep = [c for c in range(1, nl + 1) if sizes[c - 1] >= MIN_VOX]
                # per-subject/region/k summary even if exploding
                n_keep = len(keep)
                if exploded:
                    cand_rows.append({'subject_id': sid, 'region': rn, 'k': k,
                                      'cand_id': -1, 'exploded': 1,
                                      'n_cand_this_k': int(n_keep)})
                    continue
                objs = ndimage.find_objects(lab)
                for c in keep:
                    sl = objs[c - 1]
                    sub = lab[sl] == c
                    zc = lg[sl][sub].astype(np.float64)
                    # shell, computed locally then restricted to brain
                    pad = tuple(slice(max(0, s.start - SHELL_R - 1),
                                      min(d, s.stop + SHELL_R + 1))
                                for s, d in zip(sl, lab.shape))
                    subp = (lab[pad] == c)
                    dil = ndimage.binary_dilation(subp, iterations=SHELL_R)
                    shell = dil & ~subp & brain[pad]
                    if shell.sum() < MIN_VOX:
                        continue
                    zb = lg[pad][shell].astype(np.float64)
                    A = float(np.median(zc))
                    S = float(np.percentile(zc, 75) - np.percentile(zc, 25))
                    R = A - float(np.median(zb))
                    # GT overlap -- SCORING ONLY
                    full = np.zeros_like(lab, bool); full[sl] = sub
                    ov_target = 0; ov_gid = -1
                    for g, cm in targets.items():
                        if (full & cm).any():
                            ov_target = 1; ov_gid = g; break
                    ov_any_gt = int((full & gt).any())
                    cand_rows.append({
                        'subject_id': sid, 'region': rn, 'k': k, 'cand_id': int(c),
                        'exploded': 0, 'n_cand_this_k': int(n_keep),
                        'vox': int(sub.sum()), 'A': A, 'S': S, 'R': R,
                        'shell_vox': int(shell.sum()),
                        'hits_missed_target': ov_target, 'target_gid': ov_gid,
                        'hits_any_gt': ov_any_gt})
        if (ii + 1) % 3 == 0:
            print(f'  {ii+1}/{len(idx)} subj, {len(cand_rows)} cand rows '
                  f'({time.time()-t0:.0f}s)', flush=True)

    for name, data in [('EXP4_candidates.csv', cand_rows), ('EXP4_targets.csv', tgt_rows)]:
        keys = sorted({k for r in data for k in r})
        with open(HERE / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for r in data:
                w.writerow(r)
        print(f'wrote {name} ({len(data)} rows)')
    print(f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
