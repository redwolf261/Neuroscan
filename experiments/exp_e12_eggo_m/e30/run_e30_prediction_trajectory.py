"""
Phase E30, Part 2: predicted component survival under the SAME degradation
family, using the EXISTING A64 checkpoint (NOT retrained). No training.

IMPORTANT CAVEAT, stated here and carried into the report: A64 was trained
ONLY at 64^3. Running it at 80/96/128/160^3 is out-of-distribution inference
-- this measures how A64's ALREADY-LEARNED features respond to inputs at
resolutions it never saw during training, not what a model trained at that
resolution would predict. Confirmed technically feasible (fully
convolutional architecture, no fixed-size layers) via a direct probe before
writing this script -- forward passes succeed cleanly at all 5 alpha
resolutions. This script proceeds on that basis, per Section 8's
instruction, but the interpretation is scoped accordingly throughout.

For each native GT component c and alpha level, native-component coordinates
are mapped to the degraded grid via the SAME nearest-neighbor resampling of
the native integer-ID component volume used in Part 1 (run_e30_survival_
trajectory.py) -- guaranteeing coordinate consistency between the geometric
(GT) and predicted survival measurements. Predicted mass is the SUM OF RAW
PROBABILITIES within a component's resampled voxel footprint, NOT
thresholded (per Section 9's explicit instruction), giving:

  v_hat_c(alpha) = sum_{x in Omega_c(alpha)} p_alpha(x)
  s_hat_c(alpha) = v_hat_c(alpha) / (v_hat_c(alpha=0) + eps)

where Omega_c(alpha) is the component's voxel footprint AT THAT ALPHA'S
RESOLUTION (from Part 1's per-component resampled masks), and alpha=0's
reference is the model's probability map at the finest tested resolution
(160^3), per this project's own alpha=0 convention (Section 3 fallback:
"define a high-resolution reference grid such as 160^3 ... and document it"
-- native full-resolution inference was not attempted, since A64 was never
trained anywhere near native resolution and it would not add a meaningful
reference point beyond 160^3 for an out-of-distribution model).
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
import scipy.ndimage as ndi
from scipy.ndimage import zoom

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

ALPHA_LEVELS = [
    (0.00, 160), (0.25, 128), (0.50, 96), (0.75, 80), (1.00, 64),
]
A64_CKPT = project_root / "experiments" / "exp_e12_eggo_m" / "e29" / "resolution_runs" / "A64_seed0" / "checkpoints" / "best.pth"
OUT_DIR = Path(__file__).parent


def resize_nn(volume, target_shape):
    current_shape = volume.shape
    zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
    return zoom(volume, zoom_factors, order=0)


def resize_linear(volume, target_shape):
    """For the FLAIR image itself -- linear interpolation, matching
    BraTSDataset's own convention for the image channel (order=1)."""
    current_shape = volume.shape
    zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
    return zoom(volume, zoom_factors, order=1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_subjects = len(val_dataset)
    print(f"Full validation set: {n_subjects} subjects")

    print(f"Loading A64 checkpoint: {A64_CKPT}")
    ckpt = torch.load(A64_CKPT, map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  epoch={ckpt.get('epoch')} best_val_dice={ckpt.get('best_val_dice')}")

    records = []

    for subject_idx in range(n_subjects):
        subject_dir = val_dataset.subject_dirs[subject_idx]
        subject_id = Path(subject_dir).name

        flair_path = Path(subject_dir) / f"{subject_id}-t2f.nii.gz"
        flair_data = nib.load(str(flair_path)).get_fdata().astype(np.float32)

        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)

        native_labeled, n_native = ndi.label(seg_binary_native > 0.5)
        if n_native == 0:
            continue

        # Per-alpha: resize FLAIR (linear, matching BraTSDataset), normalize
        # identically (min-max to [0,1], per-volume, post-resize -- same as
        # BraTSDataset), run inference, get raw probability map, and resize
        # the native component-ID volume (nearest-neighbor) to the SAME grid
        # to define each component's voxel footprint at that resolution.
        probs_by_alpha = {}
        labeled_by_alpha = {}
        for alpha, res in ALPHA_LEVELS:
            target_shape = (res, res, res)
            flair_resized = resize_linear(flair_data, target_shape)
            fmin, fmax = flair_resized.min(), flair_resized.max()
            if fmax > fmin:
                flair_resized = (flair_resized - fmin) / (fmax - fmin)
            else:
                flair_resized = np.zeros_like(flair_resized)

            image_t = torch.from_numpy(flair_resized[np.newaxis, np.newaxis, ...]).float().to(device)
            with torch.no_grad():
                probs = model(image_t)["probs"].squeeze(0).squeeze(0).cpu().numpy()
            probs_by_alpha[alpha] = probs

            labeled_resized = resize_nn(native_labeled.astype(np.float32), target_shape)
            labeled_by_alpha[alpha] = np.round(labeled_resized).astype(np.int32)

        for comp_id in range(1, n_native + 1):
            v_hat_ref = float(probs_by_alpha[0.00][labeled_by_alpha[0.00] == comp_id].sum())
            row = {"subject_id": subject_id, "subject_idx": subject_idx, "native_component_id": comp_id}
            for alpha, res in ALPHA_LEVELS:
                mask_at_alpha = labeled_by_alpha[alpha] == comp_id
                v_hat = float(probs_by_alpha[alpha][mask_at_alpha].sum())
                row[f"v_hat_alpha_{alpha}"] = v_hat
                row[f"s_hat_alpha_{alpha}"] = v_hat / (v_hat_ref + 1e-9)
            records.append(row)

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{n_subjects} subjects")

    print(f"\nTotal native-space GT components with prediction trajectories: {len(records)}")
    with open(OUT_DIR / "E30_prediction_degradation_table.json", "w") as f:
        json.dump(records, f)
    print(f"Saved to {OUT_DIR / 'E30_prediction_degradation_table.json'}")


if __name__ == "__main__":
    main()
