"""Cache per-subject logit fields + GT for the frozen baseline.

Stores, per subject, the 3-region logit volume as float16 (plus GT as packed
bits). This makes every downstream decision-rule experiment (thresholds, TTA,
ensembling) essentially free.

Frozen E131_v5control_seed0, unchanged preprocessing/split/inference.
"""
import sys, time, argparse
from pathlib import Path
import numpy as np
import torch

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

CKPT_DIR = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints'
CKPT = CKPT_DIR / 'best.pth'
OUT = Path(__file__).resolve().parent / 'cache'
OUT.mkdir(parents=True, exist_ok=True)


def flip_variants(x, axes):
    """axes: tuple of spatial axes to flip (in the (C,D,H,W) tensor => +1)."""
    return torch.flip(x, dims=[a + 2 for a in axes]) if axes else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tta', type=int, default=0, help='also cache mirror-TTA logits')
    ap.add_argument('--overlap', type=float, default=None)
    ap.add_argument('--tag', type=str, default='base')
    ap.add_argument('--ckpt', type=str, default='best.pth')
    a = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT_DIR / a.ckpt), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    ov = t.SW_OVERLAP if a.overlap is None else a.overlap
    print(f'caching {len(ds)} subjects | tag={a.tag} ckpt={a.ckpt} overlap={ov} tta={a.tta}', flush=True)

    # mirror-TTA set: identity + 3 single-axis flips
    AXSETS = [(), (0,), (1,), (2,)] if a.tta else [()]

    t0 = time.time()
    for i in range(len(ds)):
        img, tgt, sid = ds[i]
        x = img.unsqueeze(0).to(dev)
        acc = None
        for axes in AXSETS:
            xv = flip_variants(x, axes)
            pr = t.sliding_window_predict(model, xv, t.PATCH, ov, 3, dev, True)
            pr = np.clip(pr, 1e-7, 1 - 1e-7)
            lg = np.log(pr / (1 - pr))
            if axes:
                lg = np.flip(lg, axis=[ax + 1 for ax in axes])
            acc = lg if acc is None else acc + lg
        lg = (acc / len(AXSETS)).astype(np.float16)
        gt = (tgt.numpy() > 0.5)
        np.savez_compressed(OUT / f'{sid}__{a.tag}.npz',
                            logit=lg, gt=np.packbits(gt), gt_shape=np.array(gt.shape))
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f'  {i+1}/{len(ds)} ({el:.0f}s, ~{el/(i+1)*(len(ds)-i-1):.0f}s left)', flush=True)
    print(f'done in {time.time()-t0:.0f}s -> {OUT}')


if __name__ == '__main__':
    main()
