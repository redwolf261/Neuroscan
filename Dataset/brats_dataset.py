"""
BraTS 2023 GLI dataset loader for NeuroScan.

Loads T2-FLAIR modality only to keep architecture unchanged.
Resizes to 64×64×64 and normalizes intensity.
Converts 4-class segmentation to binary (tumor vs background).
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom


class BraTSDataset(Dataset):
    """
    BraTS 2023 GLI dataset adapter for NeuroScan.

    Args:
        root_dir: Path to Dataset/Training folder
        split: 'train', 'val', or 'test'
        val_split: Fraction of data to use for validation (default 0.1)
        target_shape: Tuple (depth, height, width) to resize to (default 64×64×64)
        normalize: If True, normalize intensity to [0, 1]
    """

    def __init__(
        self,
        root_dir="Dataset/Training",
        split="train",
        val_split=0.1,
        target_shape=(64, 64, 64),
        normalize=True,
        return_native=False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.target_shape = target_shape
        self.normalize = normalize
        # Additive, backward-compatible per E55's own design convention
        # (same pattern as target_shape itself): default False keeps
        # every existing call site's __getitem__ returning the original
        # 3-tuple (image, mask, subject_id) unchanged. When True, a 4th
        # tuple element (native_image, native-resolution FLAIR, PRE-
        # resize) is returned. Verified directly (this session, all 1251
        # subjects): every BraTS-GLI native FLAIR volume in this dataset
        # has IDENTICAL shape (240, 240, 155) -- no padding/cropping
        # logic is needed to batch native volumes together.
        self.return_native = return_native

        # Find all subject folders
        wrapper_dir = os.path.join(root_dir, "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
        subject_dirs = sorted(glob.glob(os.path.join(wrapper_dir, "BraTS-GLI-*")))

        if not subject_dirs:
            raise FileNotFoundError(f"No BraTS subjects found in {wrapper_dir}")

        # Shuffle with fixed seed so val set isn't biased by subject-ID ordering,
        # while keeping the split reproducible and patient-disjoint across runs.
        rng = np.random.RandomState(42)
        subject_dirs = list(subject_dirs)
        rng.shuffle(subject_dirs)

        # Split into train/val
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
        """
        Returns (return_native=False, default -- every existing call site):
            image: Tensor of shape (1, D, H, W) - T2-FLAIR, resized to target_shape
            mask: Tensor of shape (1, D, H, W) - binary segmentation, resized to target_shape
            subject_id: Subject folder name for reference

        Returns (return_native=True, added for E55's dual-resolution mechanism):
            image, mask, subject_id, native_image -- native_image is
            Tensor of shape (1, D_native, H_native, W_native), the SAME
            FLAIR volume BEFORE resize, min-max normalized identically
            to `image` (so both pathways see intensity on the same
            scale). Native segmentation is NOT returned here -- E55's
            own training script crops the label from the same
            in-`__getitem__` seg_binary array directly, following this
            method's own established pattern, rather than adding a 5th
            tuple element; see train_e55_dual_resolution.py.
        """
        subject_dir = self.subject_dirs[idx]
        subject_id = os.path.basename(subject_dir)

        # Load T2-FLAIR
        flair_path = os.path.join(subject_dir, f"{subject_id}-t2f.nii.gz")
        if not os.path.exists(flair_path):
            raise FileNotFoundError(f"FLAIR file not found: {flair_path}")

        flair_nib = nib.load(flair_path)
        flair_data = flair_nib.get_fdata().astype(np.float32)

        # Load segmentation
        seg_path = os.path.join(subject_dir, f"{subject_id}-seg.nii.gz")
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")

        seg_nib = nib.load(seg_path)
        seg_data = seg_nib.get_fdata().astype(np.float32)

        # Convert 4-class to binary: any label > 0 becomes 1
        seg_binary = (seg_data > 0).astype(np.float32)

        # Resize both to target shape
        flair_resized = self._resize_volume(flair_data, self.target_shape)
        seg_resized = self._resize_volume(seg_binary, self.target_shape, order=0)  # Nearest for mask

        # Normalize FLAIR intensity to [0, 1]. Computed from flair_data
        # (native, PRE-resize) rather than flair_resized so return_native's
        # native_image and the resized `image` share the EXACT SAME
        # normalization constants -- both pathways then see intensity on
        # an identical scale, not two independently-normalized copies of
        # the same volume (which would introduce a spurious inconsistency
        # between the global and local pathway's own inputs).
        if self.normalize:
            flair_min = np.min(flair_data)
            flair_max = np.max(flair_data)
            if flair_max > flair_min:
                flair_resized = (flair_resized - flair_min) / (flair_max - flair_min)
                flair_data_norm = (flair_data - flair_min) / (flair_max - flair_min)
            else:
                flair_resized = np.zeros_like(flair_resized)
                flair_data_norm = np.zeros_like(flair_data)
        else:
            flair_data_norm = flair_data

        # Convert to torch tensors with channel dimension
        image = torch.from_numpy(flair_resized[np.newaxis, ...]).float()  # (1, D, H, W)
        mask = torch.from_numpy(seg_resized[np.newaxis, ...]).float()      # (1, D, H, W)

        if self.return_native:
            native_image = torch.from_numpy(flair_data_norm[np.newaxis, ...]).float()
            return image, mask, subject_id, native_image

        return image, mask, subject_id

    @staticmethod
    def _resize_volume(volume, target_shape, order=1):
        """
        Resize volume to target shape using scipy zoom.

        Args:
            volume: 3D array
            target_shape: Tuple (D, H, W)
            order: Interpolation order (1=linear, 0=nearest for masks)

        Returns:
            Resized volume
        """
        current_shape = volume.shape
        zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
        resized = zoom(volume, zoom_factors, order=order)
        return resized


def create_brats_loaders(
    batch_size=4,
    num_workers=0,
    root_dir="Dataset/Training",
    val_split=0.1,
    target_shape=(64, 64, 64),
    return_native=False,
):
    """
    Create train and validation DataLoaders for BraTS.

    Args:
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        root_dir: Path to Dataset/Training
        val_split: Fraction of data for validation
        target_shape: Tuple (D, H, W) to resize to. Previously this was NEVER
            forwarded to BraTSDataset (silently always 64^3 regardless of any
            config's own "dataset.target_shape" field) -- fixed here, for
            E29's resolution-ceiling experiment (A64/A96/A128), as an
            additive/backward-compatible parameter: the default (64,64,64)
            keeps every prior call site (A, C6-2, C6-3, D4/D2/Both) byte-
            identical in behavior.

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader

    train_ds = BraTSDataset(
        root_dir=root_dir,
        split="train",
        val_split=val_split,
        target_shape=target_shape,
        return_native=return_native,
    )

    val_ds = BraTSDataset(
        root_dir=root_dir,
        split="val",
        val_split=val_split,
        target_shape=target_shape,
        return_native=return_native,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test
    dataset = BraTSDataset(split="train")
    print(f"BraTS dataset: {len(dataset)} training subjects")

    # Load first sample
    image, mask, subject_id = dataset[0]
    print(f"Sample: {subject_id}")
    print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"  Image range: [{image.min():.4f}, {image.max():.4f}]")
    print(f"  Mask unique values: {torch.unique(mask).tolist()}")
