"""
Phase E29, pre-training diagnostic: resize-survival analysis.

NO TRAINING, NO MODEL INFERENCE. Pure data-geometry question: for every
native-resolution GT lesion connected component, how many voxels does it
occupy after nearest-neighbor resize to 64^3, 96^3, and 128^3?

This directly tests the causal premise behind E29's training experiment
BEFORE spending any GPU-hours on it: if lesions that become 1-3 voxels at
64^3 become meaningfully larger (5-10+ voxels) at 128^3, that's direct,
model-independent evidence the current bottleneck is preprocessing-induced,
not a modeling deficiency. If native-size components already shrink to
near-nothing regardless of target resolution (e.g. sub-voxel-scale lesions
even at 128^3), that would falsify the resolution hypothesis outright and
this should be reported honestly before any training is launched.

Full 125-subject VALIDATION set only (matching every other analysis's
population; NOT the training set -- though the two are drawn from the
same distribution, so this result should generalize to training-set
lesion structure as well).
Uses the ORIGINAL native-resolution seg.nii.gz (same files E28 first
touched), never resized until this script does it explicitly and
independently at each target resolution.
"""
import sys
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from scipy.ndimage import zoom

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Dataset.brats_dataset import BraTSDataset  # noqa: E402

TARGET_RESOLUTIONS = [64, 96, 128]
OUT_DIR = Path(__file__).parent


def resize_binary_mask(seg_binary, target_shape):
    """IDENTICAL convention to BraTSDataset._resize_volume: scipy.ndimage.zoom,
    order=0 (nearest-neighbor) for masks."""
    current_shape = seg_binary.shape
    zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
    return zoom(seg_binary, zoom_factors, order=0)


def main():
    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_subjects = len(val_dataset)
    print(f"Full validation set: {n_subjects} subjects")

    records = []

    for subject_idx in range(n_subjects):
        subject_dir = val_dataset.subject_dirs[subject_idx]
        subject_id = Path(subject_dir).name

        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_nib = nib.load(str(seg_path))
        seg_data = seg_nib.get_fdata().astype(np.float32)
        native_shape = seg_data.shape
        seg_binary_native = (seg_data > 0).astype(np.float32)

        # Label connected components in NATIVE space -- this defines the
        # canonical set of "real" lesions, independent of any resize.
        native_labeled, n_native = ndi.label(seg_binary_native > 0.5)
        if n_native == 0:
            continue

        native_sizes = ndi.sum(seg_binary_native > 0.5, native_labeled, range(1, n_native + 1))

        # For each target resolution, resize the FULL binary mask once, then
        # measure how many voxels of each ORIGINAL native-space component
        # survive at that resolution. This is done by resizing an integer
        # component-ID volume (nearest-neighbor preserves IDs exactly, unlike
        # resizing per-component binary masks separately, which would need
        # renormalization) and counting per-ID voxel counts post-resize.
        resized_counts = {}  # {resolution: {native_component_id: voxel_count_at_that_resolution}}
        for res in TARGET_RESOLUTIONS:
            target_shape = (res, res, res)
            labeled_resized = resize_binary_mask(native_labeled.astype(np.float32), target_shape)
            labeled_resized = np.round(labeled_resized).astype(np.int32)  # nearest-neighbor should already be exact integers; round for float safety
            counts_at_res = {}
            for comp_id in range(1, n_native + 1):
                counts_at_res[comp_id] = int((labeled_resized == comp_id).sum())
            resized_counts[res] = counts_at_res

        for comp_id in range(1, n_native + 1):
            native_size = int(native_sizes[comp_id - 1])
            row = {
                "subject_id": subject_id, "subject_idx": subject_idx, "native_component_id": comp_id,
                "native_size": native_size,
            }
            for res in TARGET_RESOLUTIONS:
                row[f"size_at_{res}"] = resized_counts[res][comp_id]
            row["survives_at_64"] = resized_counts[64][comp_id] > 0
            row["survives_at_96"] = resized_counts[96][comp_id] > 0
            row["survives_at_128"] = resized_counts[128][comp_id] > 0
            records.append(row)

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{n_subjects} subjects")

    print(f"\nTotal native-space GT components: {len(records)}")
    with open(OUT_DIR / "E29_resize_survival_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved to {OUT_DIR / 'E29_resize_survival_table.json'}")


if __name__ == "__main__":
    main()
