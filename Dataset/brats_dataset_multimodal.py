"""
BraTS 2023 GLI multimodal dataset loader for NeuroScan.

Additive, NOT a modification of Dataset/brats_dataset.py (FLAIR-only,
frozen). Loads all four modalities (T1, T1ce, T2, FLAIR) stacked as 4
input channels.

Every convention is copied VERBATIM from BraTSDataset -- same subject
discovery, same seed=42 shuffle/split, same val_split, same target_shape,
same per-volume min-max normalization (applied per modality), same binary
segmentation conversion -- so the train/val split is IDENTICAL (same
subjects) to every existing FLAIR-only experiment. The ONLY difference a
model sees is the extra 3 input channels.
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

MODALITY_SUFFIXES = ["t1n", "t1c", "t2w", "t2f"]
MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]


class BraTSMultimodalDataset(Dataset):
    """BraTS 2023 GLI multimodal adapter. Returns (4,D,H,W) image tensors."""

    def __init__(
        self,
        root_dir="Dataset/Training",
        split="train",
        val_split=0.1,
        target_shape=(64, 64, 64),
        normalize=True,
        modalities=None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.target_shape = target_shape
        self.normalize = normalize
        self.modalities = modalities if modalities is not None else list(MODALITY_SUFFIXES)
        for m in self.modalities:
            assert m in MODALITY_SUFFIXES, f"Unknown modality suffix: {m}"

        wrapper_dir = os.path.join(root_dir, "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
        subject_dirs = sorted(glob.glob(os.path.join(wrapper_dir, "BraTS-GLI-*")))
        if not subject_dirs:
            raise FileNotFoundError(f"No BraTS subjects found in {wrapper_dir}")

        # IDENTICAL seed=42 shuffle to BraTSDataset -- same subjects in each split.
        rng = np.random.RandomState(42)
        subject_dirs = list(subject_dirs)
        rng.shuffle(subject_dirs)

        n_total = len(subject_dirs)
        n_val = int(n_total * val_split)

        if split == "train":
            self.subject_dirs = subject_dirs[:-n_val] if n_val > 0 else subject_dirs
        elif split == "val":
            self.subject_dirs = subject_dirs[-n_val:] if n_val > 0 else []
        else:
            raise ValueError(f"Unknown split: {split}")

        if not self.subject_dirs:
            raise ValueError(f"No subjects for split '{split}'")

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_dir = self.subject_dirs[idx]
        subject_id = os.path.basename(subject_dir)

        channels = []
        for suffix in self.modalities:
            mod_path = os.path.join(subject_dir, f"{subject_id}-{suffix}.nii.gz")
            if not os.path.exists(mod_path):
                raise FileNotFoundError(f"Modality file not found: {mod_path}")
            mod_data = nib.load(mod_path).get_fdata().astype(np.float32)
            mod_resized = self._resize_volume(mod_data, self.target_shape)

            if self.normalize:
                mod_min, mod_max = np.min(mod_data), np.max(mod_data)
                if mod_max > mod_min:
                    mod_resized = (mod_resized - mod_min) / (mod_max - mod_min)
                else:
                    mod_resized = np.zeros_like(mod_resized)
            channels.append(mod_resized)

        image_stack = np.stack(channels, axis=0)

        seg_path = os.path.join(subject_dir, f"{subject_id}-seg.nii.gz")
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
        seg_data = nib.load(seg_path).get_fdata().astype(np.float32)
        seg_binary = (seg_data > 0).astype(np.float32)
        seg_resized = self._resize_volume(seg_binary, self.target_shape, order=0)

        image = torch.from_numpy(image_stack).float()
        mask = torch.from_numpy(seg_resized[np.newaxis, ...]).float()
        return image, mask, subject_id

    @staticmethod
    def _resize_volume(volume, target_shape, order=1):
        current_shape = volume.shape
        zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
        return zoom(volume, zoom_factors, order=order)


def create_multimodal_brats_loaders(
    batch_size=4,
    num_workers=0,
    root_dir="Dataset/Training",
    val_split=0.1,
    target_shape=(64, 64, 64),
    modalities=None,
):
    from torch.utils.data import DataLoader

    train_ds = BraTSMultimodalDataset(
        root_dir=root_dir, split="train", val_split=val_split,
        target_shape=target_shape, modalities=modalities,
    )
    val_ds = BraTSMultimodalDataset(
        root_dir=root_dir, split="val", val_split=val_split,
        target_shape=target_shape, modalities=modalities,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
