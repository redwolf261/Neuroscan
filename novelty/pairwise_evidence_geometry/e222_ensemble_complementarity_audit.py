"""E222 -- Ensemble complementarity audit. FROZEN checkpoints, NO TRAINING.
PREREGISTERED.

Uses EXISTING trained checkpoints spanning genuinely different architectures
(not seed variants of the same arch), already on disk from prior phases:
  v5   (E131 control, this session's baseline throughout)
  v13, v14, v15, v16  (E131/E132/E137 architecture variants)
  A96, DualRes  (E54/E55 -- different receptive field / resolution designs)
Verified via checkpoint 'arch' tag before inclusion; A96/DualRes tag-checked
at runtime (older checkpoints may lack the field, handled explicitly, not
silently skipped).

Per model, per subject: ET/TC/WT prediction at THAT MODEL'S OWN best.pth,
via the existing sliding_window_predict, threshold z>0 (fixed, matching
raw-logit convention used throughout E204-E221 -- NOT each model's own
tuned operating point, so no threshold-search advantage confounds the
audit).

PREREGISTERED QUESTIONS (answered here, decide whether fusion is worth
designing -- NOT which fusion rule to use):
  Q1 Per-lesion detection: does model B detect ET components model A missed,
     and vice versa? (component-level, MIN_VOX=5, overlap>=0.5 = detected)
  Q2 Complementary-error rate: fraction of A's missed lesions detected by
     >=1 other model (the number that matters for whether fusion COULD help)
  Q3 Is disagreement genuinely informative, or mostly false positives?
     For voxels where models disagree, what fraction are GT-positive vs
     GT-negative?
  Q4 Naive fixed-weight ensembles (mean, majority vote, best-single-model
     oracle-selected per subject) -- whole-volume Dice, LOSO-style fixed
     threshold z>0 for fair comparison across all arms.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
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

from neuroscan_3d_v13 import UNet3D_v13, UNet3D_v14
from neuroscan_3d_v15 import UNet3D_v15
from neuroscan_3d_v16 import UNet3D_v16
from neuroscan_3d_v3 import UNet3D_v3
from neuroscan_3d_v9 import UNet3D_v9

MIN_VOX = 5
ET = 0

# (checkpoint path, model class) -- classes verified against each variant's
# own training script (train_e54_a96_resolution.py -> v3, train_e55_dual_
# resolution.py -> v9, neuroscan_3d_v13.py defines BOTH v13 and v14).
CANDIDATES = {
    'v5':      (ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth', UNet3D_v5),
    'v13':     (ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v13_seed0_GATECOLLAPSE/checkpoints/best.pth', UNet3D_v13),
    'v14':     (ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v14_seed0/checkpoints/best.pth', UNet3D_v14),
    'v15':     (ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E132_v15_seed0/checkpoints/best.pth', UNet3D_v15),
    'v16':     (ROOT / 'experiments/exp_e12_eggo_m/e137/runs/E137_v16_seed0/checkpoints/best.pth', UNet3D_v16),
    'A96':     (ROOT / 'experiments/exp_e12_eggo_m/e54/runs/A96_seed0/checkpoints/best.pth', UNet3D_v3),
    'DualRes': (ROOT / 'experiments/exp_e12_eggo_m/e55/runs/DualRes_seed0/checkpoints/best.pth', UNet3D_v9),
}


def load_model(path, cls, dev):
    ck = torch.load(str(path), map_location=dev, weights_only=False)
    arch = ck.get('arch', cls.__name__)
    m = cls(4, 3).to(dev).eval()
    try:
        m.load_state_dict(ck['model_state'])
    except Exception as e:
        return None, arch, str(e)
    for q in m.parameters():
        q.requires_grad_(False)
    return m, arch, None


def comps(mask):
    lbl, n = ndimage.label(mask)
    return [(lbl == g) for g in range(1, n + 1) if (lbl == g).sum() >= MIN_VOX]


def main():
    dev = torch.device('cuda')
    models = {}
    print('Loading candidate checkpoints...', flush=True)
    for name, (path, cls) in CANDIDATES.items():
        if not path.exists():
            print(f'  {name:<8} MISSING: {path}', flush=True)
            continue
        m, arch, err = load_model(path, cls, dev)
        if err:
            print(f'  {name:<8} LOAD FAILED (arch mismatch or missing keys): {err[:80]}',
                  flush=True)
            continue
        models[name] = m
        print(f'  {name:<8} loaded, arch tag = {arch}', flush=True)
    print(f'\n{len(models)} usable models: {list(models.keys())}\n', flush=True)
    if len(models) < 2:
        print('FEWER THAN 2 USABLE MODELS -- audit cannot proceed, stopping.')
        return

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    smoke = '--smoke' in sys.argv
    n_total = 5 if smoke else len(ds)
    names = sorted(models.keys())

    out = HERE / ('E222_smoke.csv' if smoke else 'E222_predictions.csv')
    fh = open(out, 'w', newline='')
    fields = (['subject_id', 'region'] +
             [f'{n}_dice' for n in names] +
             [f'{n}_tp' for n in names] + [f'{n}_fp' for n in names] +
             [f'{n}_fn' for n in names])
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()

    comp_out = HERE / ('E222_components_smoke.csv' if smoke else 'E222_components.csv')
    cfh = open(comp_out, 'w', newline='')
    cw = csv.DictWriter(cfh, fieldnames=['subject_id', 'comp_id', 'size'] +
                        [f'{n}_detected' for n in names])
    cw.writeheader()

    t0 = time.time()
    for ii in range(n_total):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        brain = img[0].numpy() != 0
        x = img.unsqueeze(0).to(dev)

        preds = {}
        for n in names:
            probs = t.sliding_window_predict(models[n], x, t.PATCH, t.SW_OVERLAP,
                                             3, dev, True)
            preds[n] = probs > 0.5   # note: probs already post-sigmoid; > 0.5 = z>0

        row = {'subject_id': sid}
        for r in range(3):
            gt = Y[r] & brain
            for n in names:
                p = preds[n][r] & brain
                tp, fp, fn = int((p & gt).sum()), int((p & ~gt).sum()), int((~p & gt).sum())
                row[f'{n}_dice_r{r}'] = (1.0 if fp == 0 else 0.0) if tp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
                row[f'{n}_tp_r{r}'] = tp; row[f'{n}_fp_r{r}'] = fp; row[f'{n}_fn_r{r}'] = fn
        # flatten per-region into 3 rows for the csv schema above
        for r in range(3):
            rr = {'subject_id': sid, 'region': r}
            for n in names:
                rr[f'{n}_dice'] = row[f'{n}_dice_r{r}']
                rr[f'{n}_tp'] = row[f'{n}_tp_r{r}']; rr[f'{n}_fp'] = row[f'{n}_fp_r{r}']
                rr[f'{n}_fn'] = row[f'{n}_fn_r{r}']
            w.writerow(rr)

        lbl, nl = ndimage.label(Y[ET])
        for g in range(1, nl + 1):
            cm = lbl == g
            if cm.sum() < MIN_VOX:
                continue
            crow = {'subject_id': sid, 'comp_id': g, 'size': int(cm.sum())}
            for n in names:
                ov = float((cm & preds[n][ET]).sum()) / cm.sum()
                crow[f'{n}_detected'] = int(ov >= 0.5)
            cw.writerow(crow)

        fh.flush(); cfh.flush()
        if (ii + 1) % 10 == 0 or smoke:
            print(f'  {ii+1}/{n_total} subj ({time.time()-t0:.0f}s)', flush=True)
    fh.close(); cfh.close()
    print(f'wrote {out.name}, {comp_out.name} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
