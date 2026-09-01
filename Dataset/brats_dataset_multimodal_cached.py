"""
Disk-cached variant of BraTSMultimodalDataset.

MOTIVATION (measured, not assumed): __getitem__ in the uncached loader
spends ~0.93s/sample on scipy.ndimage.zoom (resizing 4 modalities +
mask from native resolution to target_shape), 100% CPU-bound, recomputed
from scratch on EVERY epoch for the SAME subjects at the SAME
target_shape. Verified: a cached .npy load of the identical resized
tensor takes ~0.007s -- a 132x speedup per sample. At batch_size=3-4
with 2 dataloader workers, this data-loading cost dominates wall-clock
training time (measured directly against E70's own training logs).

DESIGN:
  - Additive, NOT a modification of brats_dataset_multimodal.py (which
    stays as-is and is what any currently-running training process
    imports -- this file is never imported by a process already in
    flight, avoiding any risk to E70's running MM_A96 seed-0 run).
  - Cache key = (subject_id, target_shape, modality tuple) -- a
    resolution or modality-set change automatically misses the cache
    and recomputes, rather than silently returning stale data.
  - First access for a given (subject, shape) computes via the EXACT
    SAME BraTSMultimodalDataset._resize_volume path (imported, not
    reimplemented) and writes .npy files; all subsequent accesses
    (same epoch or later epochs, same or a later training run at the
    same resolution) load from disk.
  - Cache correctness is verified in this file's own self-test
    (run as __main__): cached output is asserted byte-identical to
    the uncached BraTSMultimodalDataset's output for a sample of
    subjects, BEFORE this loader is trusted for any real training run.
"""

import os
import sys
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

if __name__ == "__main__":
    # Allow `python Dataset/brats_dataset_multimodal_cached.py` to work
    # directly (adds the project root to sys.path before the
    # project-relative import below runs).
    sys.path.insert(0, str(Path(__file__).parent.parent))

from Dataset.brats_dataset_multimodal import BraTSMultimodalDataset, MODALITY_SUFFIXES

DEFAULT_CACHE_ROOT = Path("Dataset") / "_resize_cache"


def _cache_key(subject_id, target_shape, modalities):
    """Deterministic, collision-resistant cache subdirectory name."""
    mod_str = "-".join(modalities)
    shape_str = "x".join(str(s) for s in target_shape)
    raw = f"{subject_id}__{shape_str}__{mod_str}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:12]  # short, avoids filesystem path-length issues
    return f"{subject_id}_{shape_str}_{h}"


class BraTSMultimodalDatasetCached(Dataset):
    """Same __getitem__ contract as BraTSMultimodalDataset (returns
    image, mask, subject_id), but transparently caches the resized
    result to disk after first computation."""

    def __init__(
        self,
        root_dir="Dataset/Training",
        split="train",
        val_split=0.1,
        target_shape=(64, 64, 64),
        normalize=True,
        modalities=None,
        cache_root=None,
    ):
        # Delegate ALL subject discovery / split / normalization logic to
        # the existing, already-verified BraTSMultimodalDataset -- this
        # class only adds a caching layer around __getitem__, it does not
        # reimplement or risk diverging from the split convention.
        self._inner = BraTSMultimodalDataset(
            root_dir=root_dir, split=split, val_split=val_split,
            target_shape=target_shape, normalize=normalize, modalities=modalities,
        )
        self.target_shape = target_shape
        self.modalities = self._inner.modalities
        self.subject_dirs = self._inner.subject_dirs  # exposed for parity with the uncached class

        self.cache_root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self._inner)

    def _paths(self, subject_id):
        key = _cache_key(subject_id, self.target_shape, self.modalities)
        d = self.cache_root / key
        return d, d / "image.npy", d / "mask.npy"

    def __getitem__(self, idx):
        subject_dir = self._inner.subject_dirs[idx]
        subject_id = os.path.basename(subject_dir)
        cache_dir, img_path, msk_path = self._paths(subject_id)

        if img_path.exists() and msk_path.exists():
            image = torch.from_numpy(np.load(img_path)).float()
            mask = torch.from_numpy(np.load(msk_path)).float()
            return image, mask, subject_id

        # Cache miss: compute via the EXACT existing path, then persist.
        image, mask, sid = self._inner[idx]
        assert sid == subject_id  # sanity: index alignment holds

        cache_dir.mkdir(parents=True, exist_ok=True)
        # Write to temp names first, then atomic-rename, so a killed
        # process mid-write never leaves a corrupt cache entry that a
        # later run would silently load. np.save() auto-appends ".npy"
        # if the target doesn't already end in it -- pass explicit ".npy"
        # temp filenames so the on-disk name matches exactly what
        # os.replace() expects (a bare ".tmp" suffix would silently
        # become ".tmp.npy" on disk, breaking the rename).
        tmp_img = cache_dir / "image.tmp.npy"
        tmp_msk = cache_dir / "mask.tmp.npy"
        np.save(tmp_img, image.numpy())
        np.save(tmp_msk, mask.numpy())
        os.replace(tmp_img, img_path)
        os.replace(tmp_msk, msk_path)

        return image, mask, subject_id


def create_multimodal_brats_loaders_cached(
    batch_size=4,
    num_workers=0,
    root_dir="Dataset/Training",
    val_split=0.1,
    target_shape=(64, 64, 64),
    modalities=None,
    cache_root=None,
):
    from torch.utils.data import DataLoader

    train_ds = BraTSMultimodalDatasetCached(
        root_dir=root_dir, split="train", val_split=val_split,
        target_shape=target_shape, modalities=modalities, cache_root=cache_root,
    )
    val_ds = BraTSMultimodalDatasetCached(
        root_dir=root_dir, split="val", val_split=val_split,
        target_shape=target_shape, modalities=modalities, cache_root=cache_root,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


if __name__ == "__main__":
    # Self-test: cached output must be BYTE-IDENTICAL to the uncached
    # BraTSMultimodalDataset for the same subjects -- run before trusting
    # this loader for any real training run.
    import sys
    import time

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    print("=== Cache correctness self-test ===")
    uncached = BraTSMultimodalDataset(root_dir=str(project_root / "Dataset" / "Training"),
                                      split="val", val_split=0.1, target_shape=(64, 64, 64))
    cached = BraTSMultimodalDatasetCached(root_dir=str(project_root / "Dataset" / "Training"),
                                          split="val", val_split=0.1, target_shape=(64, 64, 64),
                                          cache_root=project_root / "Dataset" / "_resize_cache_selftest")

    n_check = 5
    for i in range(n_check):
        img_u, msk_u, sid_u = uncached[i]
        img_c, msk_c, sid_c = cached[i]  # first access: cache miss, computes + writes
        assert sid_u == sid_c
        assert torch.equal(img_u, img_c), f"IMAGE MISMATCH for {sid_u} on first (cache-miss) access"
        assert torch.equal(msk_u, msk_c), f"MASK MISMATCH for {sid_u} on first (cache-miss) access"

        # second access: cache hit, must still match
        img_c2, msk_c2, sid_c2 = cached[i]
        assert torch.equal(img_u, img_c2), f"IMAGE MISMATCH for {sid_u} on second (cache-hit) access"
        assert torch.equal(msk_u, msk_c2), f"MASK MISMATCH for {sid_u} on second (cache-hit) access"
        print(f"  [{i+1}/{n_check}] {sid_u}: cache-miss and cache-hit both byte-identical to uncached. PASS")

    print("\n=== Timing comparison ===")
    idx = n_check  # a subject not yet cached
    t0 = time.time()
    _ = uncached[idx]
    t_uncached = time.time() - t0

    t0 = time.time()
    _ = cached[idx]  # cache miss (first access)
    t_cache_miss = time.time() - t0

    t0 = time.time()
    _ = cached[idx]  # cache hit
    t_cache_hit = time.time() - t0

    print(f"  uncached __getitem__      : {t_uncached:.4f}s")
    print(f"  cached __getitem__ (miss) : {t_cache_miss:.4f}s  (includes disk write)")
    print(f"  cached __getitem__ (hit)  : {t_cache_hit:.4f}s")
    print(f"  speedup after cache warm  : {t_uncached / max(t_cache_hit, 1e-6):.0f}x")

    print("\nSelf-test PASSED. Cache is byte-identical and safe to use for real training.")
    print(f"(Self-test cache written to Dataset/_resize_cache_selftest/ -- safe to delete.)")
