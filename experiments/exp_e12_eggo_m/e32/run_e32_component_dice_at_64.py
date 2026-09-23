"""
Phase E32: alpha_c incremental-information feasibility test -- data
collection. NO TRAINING. Uses the EXISTING A64 checkpoint (best.pth,
in-distribution inference, its own native 64^3 training resolution -- NOT
the out-of-distribution multi-resolution inference E30/E31 used, since this
experiment needs a real, trustworthy component-Dice, not the artifact-prone
m_c ratio).

For each NATIVE GT component (identity fixed in native space, per E30/E31's
own convention), compute a REAL, thresholded component Dice at 64^3:
  1. Resample the native integer component-ID volume to 64^3
     (nearest-neighbor, IDENTICAL to E30's own convention) -- this defines
     each native component's voxel footprint at 64^3, exactly matching
     E30_component_survival_table.json's own size_alpha_1.0 field (reused
     as a consistency check, not recomputed independently).
  2. Run A64 on the SAME 64^3-resized FLAIR volume BraTSDataset itself would
     produce (i.e. real, in-distribution inference, not the 80-160^3
     out-of-distribution sweep from E30/E31).
  3. Threshold the prediction at 0.5 (standard convention, matching every
     prior E25-E29 component-Dice measurement).
  4. Component Dice = 2*|P_c ∩ Y_c(64)| / (|P_c| + |Y_c(64)|), using the
     UNION of overlapping predicted components (matching rule identical to
     E25/E27/E28's own "union of overlapping predicted components").
  5. Detected = at least one predicted voxel overlaps the native component's
     64^3 footprint.

This gives one real outcome variable per native component: component Dice
at 64^3 (0 if the component vanished entirely under resize -- appropriately
scored as total failure, not as an undefined ratio).
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

A64_CKPT = project_root / "experiments" / "exp_e12_eggo_m" / "e29" / "resolution_runs" / "A64_seed0" / "checkpoints" / "best.pth"
OUT_DIR = Path(__file__).parent


def resize_nn(volume, target_shape):
    current_shape = volume.shape
    zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
    return zoom(volume, zoom_factors, order=0)


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
        image, mask, subject_id = val_dataset[subject_idx]  # image is ALREADY the standard 64^3, in-distribution resize
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)

        native_labeled, n_native = ndi.label(seg_binary_native > 0.5)
        if n_native == 0:
            continue

        labeled_64 = resize_nn(native_labeled.astype(np.float32), (64, 64, 64))
        labeled_64 = np.round(labeled_64).astype(np.int32)

        with torch.no_grad():
            pred_bin = (model(image_b)["probs"] >= 0.5).float().squeeze(0).squeeze(0).cpu().numpy()
        pred_labeled, _ = ndi.label(pred_bin > 0.5)

        for comp_id in range(1, n_native + 1):
            comp_mask_64 = labeled_64 == comp_id
            gt_size_64 = int(comp_mask_64.sum())

            overlap_ids = set(pred_labeled[comp_mask_64].flatten().tolist()) - {0} if gt_size_64 > 0 else set()
            if not overlap_ids:
                detected = False
                comp_dice = 0.0
            else:
                pred_region = np.isin(pred_labeled, list(overlap_ids))
                inter = int((comp_mask_64 & pred_region).sum())
                pred_size = int(pred_region.sum())
                detected = True
                comp_dice = 2 * inter / (gt_size_64 + pred_size) if (gt_size_64 + pred_size) > 0 else 1.0

            records.append({
                "subject_id": subject_id, "subject_idx": subject_idx, "native_component_id": comp_id,
                "size_64_check": gt_size_64,  # for consistency check against E30's own size_alpha_1.0
                "detected_64": detected, "component_dice_64": comp_dice,
            })

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{n_subjects} subjects")

    print(f"\nTotal native-space GT components with real 64^3 component Dice: {len(records)}")
    with open(OUT_DIR / "E32_component_dice_64_table.json", "w") as f:
        json.dump(records, f)
    print(f"Saved to {OUT_DIR / 'E32_component_dice_64_table.json'}")


if __name__ == "__main__":
    main()
