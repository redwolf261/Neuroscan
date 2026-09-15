"""
Build a fast on-disk cache of BraTS 2023 GLI for E130+ training.

WHY: measured, not assumed. The .nii.gz pipeline costs 1.53 s PER SAMPLE
(5 gzip volumes decompressed + 4 z-score passes), against 0.50 s of GPU
compute per iteration at 128^3 batch 2. With 4 workers the observed rate
was 1.99 s/iter -- a 4x slowdown versus the compute bound, i.e. 50 epochs
would take ~16h instead of ~4h, and EVERY later run (3-seed campaign,
causal re-measurement, interventions) would pay the same tax.

WHAT THIS STORES, and why each choice is lossless for our use:
  image  : float16, 4 channels (t1c, t1n, t2f, t2w), ALREADY z-scored over
           brain voxels. fp16 is ample for z-scored data whose values sit
           in roughly [-5, 5]; training runs under AMP in fp16 anyway.
  target : 3 regions (ET, TC, WT) bit-PACKED via np.packbits. The targets
           are binary by definition, so one bit per voxel loses nothing --
           this is 16x smaller than the fp16 it replaces.
  crop   : each subject is cropped to its BRAIN BOUNDING BOX. Everything
           outside is exactly zero in all 4 modalities after z-scoring
           (znorm_brain leaves background at 0), and no tumour can exist
           outside the brain, so the crop is information-preserving. It
           keeps ~38% of the volume, measured.

Net effect, measured on a real subject: 156 GB naive fp16 -> ~35 GB.

The bounding box origin and the original full shape are stored per subject
so any analysis needing native coordinates can invert the crop exactly.

Usage:  python Dataset/build_brats_cache.py [--workers 6]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Dataset.brats_multimodal_dataset import (  # noqa: E402
    MODALITIES, znorm_brain, labels_to_regions, BraTSMultimodalDataset,
)

CACHE_DIR = project_root / "Dataset" / "cache_brats2023gli"


def build_one(subject_dir, out_dir):
    sid = os.path.basename(subject_dir)
    out = Path(out_dir) / f"{sid}.npz"
    if out.exists():
        return sid, "skip"

    chans = []
    for m in MODALITIES:
        p = os.path.join(subject_dir, f"{sid}-{m}.nii.gz")
        chans.append(znorm_brain(nib.load(p).get_fdata().astype(np.float32)))
    img = np.stack(chans, axis=0)
    seg = nib.load(os.path.join(subject_dir, f"{sid}-seg.nii.gz")).get_fdata().astype(np.uint8)
    tgt = labels_to_regions(seg)

    brain = np.zeros(img.shape[1:], dtype=bool)
    for c in range(img.shape[0]):
        brain |= (img[c] != 0)
    if not brain.any():
        lo = np.array([0, 0, 0])
        hi = np.array(img.shape[1:])
    else:
        idx = np.argwhere(brain)
        lo, hi = idx.min(0), idx.max(0) + 1

    ic = img[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float16)
    tc = tgt[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.uint8)

    np.savez(out,
             image=ic,
             target_packed=np.packbits(tc, axis=None),
             target_shape=np.array(tc.shape, dtype=np.int32),
             bbox_lo=lo.astype(np.int32),
             full_shape=np.array(img.shape[1:], dtype=np.int32))
    return sid, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--root", type=str, default=str(project_root / "Dataset" / "Training"))
    a = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dirs = BraTSMultimodalDataset._discover(a.root)
    print(f"discovered {len(dirs)} subjects -> {CACHE_DIR}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(build_one, d, str(CACHE_DIR)) for d in dirs]
        for f in as_completed(futs):
            sid, status = f.result()
            done += 1
            if done % 50 == 0 or done == len(dirs):
                el = time.time() - t0
                gb = sum(p.stat().st_size for p in CACHE_DIR.glob("*.npz")) / 1e9
                print(f"  {done}/{len(dirs)}  {el:.0f}s elapsed, "
                      f"{el/done*(len(dirs)-done):.0f}s left, cache={gb:.1f} GB", flush=True)

    gb = sum(p.stat().st_size for p in CACHE_DIR.glob("*.npz")) / 1e9
    print(f"\nDONE in {time.time()-t0:.0f}s. {len(list(CACHE_DIR.glob('*.npz')))} files, {gb:.1f} GB")


if __name__ == "__main__":
    main()
