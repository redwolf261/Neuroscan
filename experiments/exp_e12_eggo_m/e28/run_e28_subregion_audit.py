"""
Phase E28: Subregion Information Audit.

NO TRAINING. Uses only A's existing best.pth checkpoint and the ORIGINAL
4-class BraTS segmentation labels (never loaded anywhere else in this
project -- Dataset/brats_dataset.py immediately binarizes them). Full
125-subject validation set, same deterministic split as every prior
analysis (BraTSDataset with val_split=0.1, seed=42 internal to the loader).

Question: for every GT whole-tumor (WT) connected component, what is its
subregion composition (%NCR/NET, %ED, %ET, using the original BraTS
2023 label convention: 1=NCR/NET, 2=ED, 3=ET), and does that composition
relate to whether A detects/segments it well?

Design choice, stated explicitly: subregion composition is measured in the
SAME 64^3 resized voxel space that both the WT mask and the model operate
in (nearest-neighbor zoom, order=0, identical to how BraTSDataset resizes
the binary mask) -- NOT at native resolution -- because the question is
about what the model's actual input-space decisions correlate with, and
every WT component in this whole project (E25-E27) has been defined in
64^3 space. This is a real, disclosed methodological choice: a 5-voxel
component in 64^3 space came from a much larger native-resolution lesion,
so "5 voxels" here is a post-resize quantity throughout, consistent with
everything else in this project's diagnostic history.

WT/TC/ET definitions (standard BraTS 2023 convention):
  WT (whole tumor)  = labels {1,2,3} (this IS the binary mask A is trained on)
  TC (tumor core)   = labels {1,3}
  ET (enhancing)    = label {3}
  NCR/NET           = label {1}
  ED (edema)        = label {2}
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

A_CKPT = project_root / "experiments" / "exp_e12_eggo_m" / "e24" / "gate6_runs" / "A_baseline_seed0" / "checkpoints" / "best.pth"
OUT_DIR = Path(__file__).parent
TARGET_SHAPE = (64, 64, 64)


def load_multiclass_seg_resized(subject_dir, subject_id, target_shape):
    seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
    seg_nib = nib.load(str(seg_path))
    seg_data = seg_nib.get_fdata().astype(np.float32)  # values in {0,1,2,3}
    current_shape = seg_data.shape
    zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
    # nearest-neighbor (order=0) -- IDENTICAL to how BraTSDataset resizes the binary mask,
    # required to keep integer class labels valid (no interpolation blending across classes)
    resized = zoom(seg_data, zoom_factors, order=0)
    return resized  # (64,64,64), values in {0,1,2,3}, may introduce resampling artifacts at boundaries -- documented, not hidden


def dice_score(pred, gt):
    tp = (pred * gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2 * tp / denom).item()) if denom > 0 else 1.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=TARGET_SHAPE, normalize=True,
    )
    n_subjects = len(val_dataset)
    print(f"Full validation set: {n_subjects} subjects")

    print(f"Loading A checkpoint: {A_CKPT}")
    ckpt = torch.load(A_CKPT, map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  epoch={ckpt.get('epoch')} best_val_dice={ckpt.get('best_val_dice')}")

    records = []
    n_composition_mismatch = 0

    for subject_idx in range(n_subjects):
        image, mask, subject_id = val_dataset[subject_idx]
        image_b = image.unsqueeze(0).to(device)
        gt_vol = mask.squeeze(0).numpy().astype(np.float32)
        gt_bool = gt_vol > 0.5

        subject_dir = val_dataset.subject_dirs[subject_idx]
        multiclass = load_multiclass_seg_resized(subject_dir, subject_id, TARGET_SHAPE)
        multiclass_wt = multiclass > 0  # labels {1,2,3} -> WT

        # Sanity check: multiclass-derived WT should closely match the binarized WT mask
        # (both come from the same seg.nii.gz, but via independently-computed resize calls,
        # so small resampling-order edge differences are possible -- checked, not assumed)
        agreement = float((multiclass_wt == gt_bool).mean())
        if agreement < 0.98:
            n_composition_mismatch += 1

        with torch.no_grad():
            pred_bin = (model(image_b)["probs"] >= 0.5).float().squeeze(0).squeeze(0).cpu().numpy()
        pred_bool = pred_bin > 0.5

        gt_labeled, n_gt = ndi.label(gt_bool)
        pred_labeled, _ = ndi.label(pred_bool)
        if n_gt == 0:
            continue

        for gt_id in range(1, n_gt + 1):
            comp_mask = gt_labeled == gt_id
            gt_size = int(comp_mask.sum())

            # subregion composition WITHIN this WT component, from the multiclass label
            # restricted to voxels the WT component mask actually covers (so composition
            # sums to gt_size regardless of any small multiclass/binary resize mismatch)
            labels_in_comp = multiclass[comp_mask]
            n_ncr = int((labels_in_comp == 1).sum())
            n_ed = int((labels_in_comp == 2).sum())
            n_et = int((labels_in_comp == 3).sum())
            n_other = gt_size - (n_ncr + n_ed + n_et)  # resampling-edge voxels with no matching multiclass label

            pct_ncr = n_ncr / gt_size if gt_size > 0 else 0.0
            pct_ed = n_ed / gt_size if gt_size > 0 else 0.0
            pct_et = n_et / gt_size if gt_size > 0 else 0.0
            pct_other = n_other / gt_size if gt_size > 0 else 0.0

            dominant = max([("NCR", pct_ncr), ("ED", pct_ed), ("ET", pct_et), ("other", pct_other)], key=lambda x: x[1])[0]

            overlap_ids = set(pred_labeled[comp_mask].flatten().tolist()) - {0}
            if not overlap_ids:
                detected = False
                comp_dice = 0.0
                coverage = 0.0
            else:
                pred_region = np.isin(pred_labeled, list(overlap_ids))
                inter = int((comp_mask & pred_region).sum())
                pred_size = int(pred_region.sum())
                detected = True
                comp_dice = 2 * inter / (gt_size + pred_size) if (gt_size + pred_size) > 0 else 1.0
                coverage = inter / gt_size if gt_size > 0 else 1.0

            records.append({
                "subject_id": subject_id, "subject_idx": subject_idx, "component_id": gt_id,
                "gt_size": gt_size,
                "n_ncr": n_ncr, "n_ed": n_ed, "n_et": n_et, "n_other": n_other,
                "pct_ncr": pct_ncr, "pct_ed": pct_ed, "pct_et": pct_et, "pct_other": pct_other,
                "dominant_subregion": dominant,
                "detected_A": detected, "dice_A": comp_dice, "coverage_A": coverage,
            })

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{n_subjects} subjects")

    print(f"\nTotal WT components: {len(records)}")
    print(f"Subjects with >2% multiclass/binary WT mask disagreement (resampling-order artifact check): {n_composition_mismatch}/{n_subjects}")

    with open(OUT_DIR / "E28_subregion_component_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved to {OUT_DIR / 'E28_subregion_component_table.json'}")


if __name__ == "__main__":
    main()
