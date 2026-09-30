"""E218 -- Does the optimizer repeatedly SACRIFICE specific lesions?
FROZEN checkpoints, NO TRAINING. PREREGISTERED.

Checkpoints: E131_v5control_seed0 epoch_005..epoch_035 (7). Epochs 040-050
EXCLUDED: training collapse (val TC 0.914 @ep34 -> 0.19 @ep39), which would
dominate any 'rest improved' signal.

Per subject x GT-ET-component x checkpoint, via the validated
sliding_window_predict (ET prob > 0.5):
  recall   d_i(t)       = fraction of component voxels predicted ET
  rest     D_rest,i(t)  = ET Dice over brain EXCLUDING dilate(comp_i, 5)
  size, final label at epoch_035: L (recall 0), S (recall >= 0.5), M (else)

Collision (Toneva et al. 2019 example forgetting; pixel-forgetting in
segmentation 2301.04221/2302.14644): a plain count of recall drops is
OCCUPIED. Only the CONDITIONAL event -- d_i drops WHILE D_rest rises -- is
candidate-new, and only if it carries information beyond forgetting.
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

CKDIR = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints'
EPOCHS = ['005', '010', '015', '020', '025', '030', '035']
ET = 0
MIN_VOX = 5
EXCL_DIL = 5


def dice(p, g):
    ps, gs = p.sum(), g.sum()
    if gs == 0:
        return 1.0 if ps == 0 else 0.0
    return float(2 * (p & g).sum() / (ps + gs))


def main():
    dev = torch.device('cuda')
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    smoke = '--smoke' in sys.argv
    n_total = 2 if smoke else len(ds)
    model = UNet3D_v5(4, 3).to(dev).eval()
    for q in model.parameters():
        q.requires_grad_(False)

    # GT components (fixed across checkpoints)
    subj = []
    for ii in range(n_total):
        img, tgt, sid = ds[ii]
        Y = tgt.numpy() > 0.5
        brain = img[0].numpy() != 0
        lbl, n = ndimage.label(Y[ET])
        cl = [(g, lbl == g) for g in range(1, n + 1) if (lbl == g).sum() >= MIN_VOX]
        excl = [ndimage.binary_dilation(cm, iterations=EXCL_DIL) for _, cm in cl]
        subj.append((sid, img, Y[ET] & brain, brain, cl, excl))
    print(f'E218: {n_total} subjects, {sum(len(s[4]) for s in subj)} GT ET components',
          flush=True)

    out = HERE / ('E218_smoke.csv' if smoke else 'E218_trajectory.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['subject_id', 'comp_id', 'size', 'epoch',
                                       'recall', 'rest_dice', 'subj_dice'])
    w.writeheader()
    t0 = time.time()
    for ep in EPOCHS:
        ck = torch.load(str(CKDIR / f'epoch_{ep}.pth'), map_location=dev, weights_only=False)
        model.load_state_dict(ck['model_state'])
        for sid, img, gt, brain, cl, excl in subj:
            probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                             t.PATCH, t.SW_OVERLAP, 3, dev, True)
            pred = (probs[ET] > 0.5) & brain
            sd = dice(pred, gt)
            for (g, cm), ex in zip(cl, excl):
                keep = brain & ~ex
                w.writerow({'subject_id': sid, 'comp_id': g, 'size': int(cm.sum()),
                            'epoch': int(ep), 'recall': float((pred & cm).sum() / cm.sum()),
                            'rest_dice': dice(pred & keep, gt & keep), 'subj_dice': sd})
        fh.flush()
        print(f'  epoch {ep} done ({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'wrote {out.name} {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
