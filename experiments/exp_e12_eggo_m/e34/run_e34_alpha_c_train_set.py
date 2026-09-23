"""
Phase E34, Step 0: compute alpha_c for the TRAINING set (1126 subjects),
using the IDENTICAL method as E31's validation-set computation (which only
covers the 125-subject validation set and cannot be used for a training-time
weighting mechanism). No training, no model inference -- pure geometry,
reusing E30/E31's exact degradation family and native-component-identity
convention.

alpha_c = min{alpha : V_c(alpha) > 0}, alpha in {0.00, 0.25, 0.50, 0.75, 1.00}
(160^3, 128^3, 96^3, 80^3, 64^3), estimated via linear interpolation between
the last-nonzero and first-zero alpha points, IDENTICAL to E31's own
alpha_c estimation logic (reused verbatim, not reimplemented differently).
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

ALPHA_LEVELS = [(0.00, 160), (0.25, 128), (0.50, 96), (0.75, 80), (1.00, 64)]
OUT_DIR = Path(__file__).parent


def resize_nn(volume, target_shape):
    current_shape = volume.shape
    zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
    return zoom(volume, zoom_factors, order=0)


def estimate_alpha_c(svals, alphas):
    """IDENTICAL logic to E31's analyze_e31_critical_resolution.py."""
    zero_idx = np.where(svals == 0)[0]
    if len(zero_idx) == 0:
        return None, True  # censored -- never vanishes within tested range
    first_zero = zero_idx[0]
    if first_zero == 0:
        return 0.0, False
    a_lo, a_hi = alphas[first_zero - 1], alphas[first_zero]
    s_lo, s_hi = svals[first_zero - 1], svals[first_zero]
    if s_lo == s_hi:
        return a_lo, False
    frac = s_lo / (s_lo - s_hi)
    return a_lo + frac * (a_hi - a_lo), False


def main():
    train_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="train", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_subjects = len(train_dataset)
    print(f"Full training set: {n_subjects} subjects")

    records = []

    for subject_idx in range(n_subjects):
        subject_dir = train_dataset.subject_dirs[subject_idx]
        subject_id = Path(subject_dir).name

        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)

        native_labeled, n_native = ndi.label(seg_binary_native > 0.5)
        if n_native == 0:
            continue

        sizes_per_alpha = {}
        for alpha, res in ALPHA_LEVELS:
            target_shape = (res, res, res)
            labeled_resized = resize_nn(native_labeled.astype(np.float32), target_shape)
            labeled_resized = np.round(labeled_resized).astype(np.int32)
            counts = {}
            for comp_id in range(1, n_native + 1):
                counts[comp_id] = int((labeled_resized == comp_id).sum())
            sizes_per_alpha[alpha] = counts

        for comp_id in range(1, n_native + 1):
            svals = np.array([sizes_per_alpha[a][comp_id] for a, _ in ALPHA_LEVELS])
            ac, censored = estimate_alpha_c(svals, [a for a, _ in ALPHA_LEVELS])
            records.append({
                "subject_id": subject_id, "subject_idx": subject_idx, "native_component_id": comp_id,
                "native_size": int((native_labeled == comp_id).sum()),
                "size_64": int(sizes_per_alpha[1.00][comp_id]),
                "alpha_c": ac, "censored": censored,
            })

        if (subject_idx + 1) % 100 == 0:
            print(f"  processed {subject_idx+1}/{n_subjects} subjects")

    print(f"\nTotal native-space GT components (training set): {len(records)}")
    with open(OUT_DIR / "E34_alpha_c_train_table.json", "w") as f:
        json.dump(records, f)
    print(f"Saved to {OUT_DIR / 'E34_alpha_c_train_table.json'}")


if __name__ == "__main__":
    main()
