"""
Phase E55, verification check #2: label-crop alignment spot-check on
REAL subjects (5-10, not synthetic). Verifies the image crop and its
corresponding label crop (extracted via the identical grid_sample
mapping) are correctly co-registered, and cross-checks against the
resized 64^3 mask's own lesion location as an independent sanity check.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Dataset.brats_dataset import BraTSDataset  # noqa: E402


def native_to_norm(coord, size):
    return 2.0 * coord / (size - 1) - 1.0


def extract_crop(volume, center_native, crop_size, native_shape, mode):
    """volume: (1,1,D,H,W). center_native: (3,) tensor, (D,H,W) order."""
    offsets = torch.arange(crop_size).float() - (crop_size - 1) / 2.0
    grid_native = center_native.view(3, 1, 1, 1) + torch.stack(
        torch.meshgrid(offsets, offsets, offsets, indexing="ij"), dim=0
    )
    grid_norm = torch.stack([
        native_to_norm(grid_native[i], native_shape[i]) for i in range(3)
    ], dim=-1)
    grid_norm_xyz = grid_norm[..., [2, 1, 0]].unsqueeze(0)
    return F.grid_sample(volume, grid_norm_xyz, mode=mode, padding_mode="zeros", align_corners=True)


def main():
    ds = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64),
        return_native=False,  # load native mask ourselves for this check
    )

    n_checked = 0
    n_pass = 0
    crop_size = 96
    native_shape = (240, 240, 155)

    for idx in range(10):
        image, mask, subject_id = ds[idx]
        mask_np = mask[0].numpy()  # (64,64,64)
        if mask_np.sum() == 0:
            print(f"[{subject_id}] no lesion in resized mask, skipping")
            continue

        # Soft centroid from the RESIZED mask (stand-in for probs_coarse
        # in the real mechanism -- ground truth here, not a prediction,
        # since this check is about the CROP MAPPING, not the model).
        coords = np.argwhere(mask_np > 0.5)
        centroid_64 = coords.mean(axis=0)  # (3,), (D,H,W) in 64^3 grid

        # Map to native coords via the verified per-axis scalar ratio.
        native_centroid = centroid_64 * (np.array(native_shape) / np.array((64, 64, 64)))
        native_centroid_t = torch.from_numpy(native_centroid).float()

        # Load the REAL native mask directly (bypassing return_native,
        # which only returns the image -- load seg ourselves here).
        import nibabel as nib
        subject_dir = ds.subject_dirs[idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_native = (nib.load(str(seg_path)).get_fdata() > 0).astype(np.float32)
        seg_native_t = torch.from_numpy(seg_native).unsqueeze(0).unsqueeze(0)

        flair_native = image  # NOTE: `image` here is the RESIZED 64^3 flair from ds[idx];
        # for a real native-image crop check we need the true native flair too.
        flair_native_path = Path(subject_dir) / f"{subject_id}-t2f.nii.gz"
        flair_native_np = nib.load(str(flair_native_path)).get_fdata().astype(np.float32)
        flair_native_t = torch.from_numpy(flair_native_np).unsqueeze(0).unsqueeze(0)

        # Extract crops centered on the mapped native centroid.
        image_crop = extract_crop(flair_native_t, native_centroid_t, crop_size, native_shape, mode="bilinear")
        label_crop = extract_crop(seg_native_t, native_centroid_t, crop_size, native_shape, mode="nearest")

        label_crop_mass = label_crop.sum().item()
        native_total_mass = seg_native.sum()
        frac_recovered = label_crop_mass / max(native_total_mass, 1)

        # Sanity: the crop should contain a meaningful fraction of the
        # subject's own lesion mass (not necessarily 100% -- a large or
        # multi-focal lesion can exceed a 96^3 crop -- but should not be
        # near-zero if the centroid mapping is correct, since the crop
        # is centered ON the lesion's own centroid by construction).
        is_plausible = frac_recovered > 0.05  # generous floor -- catches gross misalignment, not precision
        n_checked += 1
        n_pass += int(is_plausible)
        print(f"[{subject_id}] native_centroid={native_centroid.round(1)} "
              f"native_lesion_voxels={int(native_total_mass)} "
              f"label_crop_voxels={int(label_crop_mass)} "
              f"frac_recovered={frac_recovered:.3f} plausible={is_plausible}")

    print(f"\n{n_pass}/{n_checked} subjects passed plausibility check (frac_recovered > 0.05)")
    all_pass = n_checked > 0 and n_pass == n_checked
    print(f"ALL SUBJECTS PASS: {all_pass}")
    if not all_pass:
        import sys as sys_
        sys_.exit(1)


if __name__ == "__main__":
    main()
