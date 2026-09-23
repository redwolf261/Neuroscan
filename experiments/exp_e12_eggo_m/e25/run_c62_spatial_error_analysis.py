"""
Phase E25, C6-2.9: spatial error concentration & subject-level failure
analysis. Per the user's explicit redirect after C6-2.8 closed the
mechanistic branch (gradient direction, parameter/Jacobian mapping,
representation->logit, FN/FP correction, and real cumulative training
dynamics all verified functioning correctly, with C6-2's confusion-flux
profile even somewhat BETTER than A's -- yet Dice still falls short):
WHERE does the remaining ~1.5pp Dice deficit actually live, spatially
and at the lesion/subject level?

Uses A's and C6-2's REAL best.pth checkpoints directly (loaded by path,
not by assumed epoch number -- each script run prints the actual epoch/
best_val_dice read from the checkpoint's own saved metadata, verified at
runtime rather than hardcoded: A's best.pth is epoch 23, best_val_dice=
0.9063020758330822, matching PHASE_E25_C62_RESULTS.md's A=0.9063 exactly;
C6-2's best.pth is epoch 28, best_val_dice=0.9030254185199738, matching
C6-2's own reported 0.9030 exactly) on the SAME full 125-subject
validation set, identical
preprocessing (BraTSDataset, target_shape=(64,64,64)), identical
thresholding (p>=0.5) -- NO new training, NO perturbation, just two real
trained models' real predictions compared directly.

One prediction-generation pass per subject per condition; every
downstream analysis (slice-position, boundary-distance, 4-way spatial
classification, connected components, per-subject Dice ranking) is
computed from that single set of predictions, not re-run per analysis.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import scipy.ndimage as ndi

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

GATE6_A_BEST = project_root / "experiments" / "exp_e12_eggo_m" / "e24" / "gate6_runs" / "A_baseline_seed0" / "checkpoints" / "best.pth"
C62_BEST = project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "c62_runs" / "C62_sc_tam_seed0" / "checkpoints" / "best.pth"
OUT_DIR = Path(__file__).parent / "spatial_error_analysis_results"

SLICE_BINS = np.linspace(0, 1, 11)  # 10 bins, 0-10%...90-100%
BOUNDARY_BIN_EDGES = [0, 1, 2, 4, 8, np.inf]
BOUNDARY_BIN_LABELS = ["0-1", "1-2", "2-4", "4-8", ">8"]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  loaded {ckpt_path.name}: epoch={ckpt.get('epoch')} best_val_dice={ckpt.get('best_val_dice')}")
    return model


def dice_score(pred, gt):
    tp = (pred * gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2 * tp / denom).item()) if denom > 0 else 1.0  # both-empty = perfect agreement


def signed_distance_from_boundary(gt_vol):
    """Distance (in voxels) from each voxel to the nearest lesion
    boundary, computed via EDT on both the lesion mask and its
    complement -- returns UNSIGNED distance-to-boundary (magnitude only,
    matching the user's bin scheme, which doesn't distinguish inside vs
    outside direction)."""
    gt_bool = gt_vol > 0.5
    if gt_bool.sum() == 0 or gt_bool.sum() == gt_bool.size:
        return np.full(gt_vol.shape, np.inf)  # no boundary exists (empty or full volume)
    dist_outside = ndi.distance_transform_edt(~gt_bool)  # distance from background voxels to nearest lesion voxel
    dist_inside = ndi.distance_transform_edt(gt_bool)    # distance from lesion voxels to nearest background voxel
    # combine: for background voxels, distance-to-boundary = dist_outside; for lesion voxels, = dist_inside
    combined = np.where(gt_bool, dist_inside, dist_outside)
    return combined


def connected_component_analysis(gt_vol, pred_vol):
    """For each GT lesion component: is it detected (any overlap with
    pred)? component-wise Dice (best-matching overlapping pred
    component). For each PRED component with no GT overlap: false-
    positive component. Returns per-GT-component and per-FP-component
    records."""
    gt_labeled, n_gt = ndi.label(gt_vol > 0.5)
    pred_labeled, n_pred = ndi.label(pred_vol > 0.5)

    gt_records = []
    for gt_id in range(1, n_gt + 1):
        gt_mask = gt_labeled == gt_id
        gt_size = int(gt_mask.sum())
        overlapping_pred_ids = set(pred_labeled[gt_mask].flatten().tolist()) - {0}
        if not overlapping_pred_ids:
            gt_records.append({"gt_size": gt_size, "detected": False, "component_dice": 0.0, "n_overlapping_pred": 0})
            continue
        # best-matching overlapping pred component union (per-lesion dice against ALL overlapping pred voxels combined)
        combined_pred_mask = np.isin(pred_labeled, list(overlapping_pred_ids))
        inter = float((gt_mask & combined_pred_mask).sum())
        comp_dice = 2 * inter / (gt_size + combined_pred_mask.sum()) if (gt_size + combined_pred_mask.sum()) > 0 else 1.0
        gt_records.append({"gt_size": gt_size, "detected": True, "component_dice": float(comp_dice), "n_overlapping_pred": len(overlapping_pred_ids)})

    fp_component_count = 0
    for pred_id in range(1, n_pred + 1):
        pred_mask = pred_labeled == pred_id
        if not (gt_labeled[pred_mask] > 0).any():
            fp_component_count += 1

    return {
        "n_gt_components": n_gt, "n_pred_components": n_pred,
        "n_fp_components": fp_component_count,
        "gt_components": gt_records,
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_subjects = len(val_dataset)
    print(f"Full validation set: {n_subjects} subjects")

    print("\nLoading models...")
    model_a = load_model(GATE6_A_BEST, device)
    model_c62 = load_model(C62_BEST, device)

    subject_records = []
    slice_error_a = np.zeros(10)
    slice_error_c62 = np.zeros(10)
    slice_total = np.zeros(10)
    boundary_error_a = np.zeros(len(BOUNDARY_BIN_LABELS))
    boundary_error_c62 = np.zeros(len(BOUNDARY_BIN_LABELS))
    boundary_total = np.zeros(len(BOUNDARY_BIN_LABELS))

    # 4-way spatial classification totals (voxel counts, pooled across all subjects)
    both_correct = 0
    a_wrong_c62_correct = 0   # C6-2 improvement
    a_correct_c62_wrong = 0   # C6-2 regression
    both_wrong = 0

    for subject_idx in range(n_subjects):
        image, mask, subject_id = val_dataset[subject_idx]
        image_b = image.unsqueeze(0).to(device)
        gt_vol = mask.squeeze(0).numpy().astype(np.float32)  # (D,H,W)

        with torch.no_grad():
            pred_a = (model_a(image_b)["probs"] >= 0.5).float().squeeze(0).squeeze(0).cpu().numpy()
            pred_c62 = (model_c62(image_b)["probs"] >= 0.5).float().squeeze(0).squeeze(0).cpu().numpy()

        dice_a = dice_score(torch.from_numpy(pred_a), torch.from_numpy(gt_vol))
        dice_c62 = dice_score(torch.from_numpy(pred_c62), torch.from_numpy(gt_vol))

        err_a = (pred_a != gt_vol)
        err_c62 = (pred_c62 != gt_vol)

        # --- 4-way spatial classification ---
        both_correct += int((~err_a & ~err_c62).sum())
        a_wrong_c62_correct += int((err_a & ~err_c62).sum())
        a_correct_c62_wrong += int((~err_a & err_c62).sum())
        both_wrong += int((err_a & err_c62).sum())

        # --- slice-position decomposition ---
        D = gt_vol.shape[0]
        for d in range(D):
            z_norm = d / max(1, D - 1)
            bin_idx = min(9, int(z_norm * 10))
            slice_total[bin_idx] += gt_vol.shape[1] * gt_vol.shape[2]
            slice_error_a[bin_idx] += err_a[d].sum()
            slice_error_c62[bin_idx] += err_c62[d].sum()

        # --- boundary-distance decomposition ---
        dist = signed_distance_from_boundary(gt_vol)
        if np.isfinite(dist).any():
            for i, (lo, hi) in enumerate(zip(BOUNDARY_BIN_EDGES[:-1], BOUNDARY_BIN_EDGES[1:])):
                bin_mask = (dist >= lo) & (dist < hi) & np.isfinite(dist)
                n_bin = int(bin_mask.sum())
                boundary_total[i] += n_bin
                boundary_error_a[i] += int((err_a & bin_mask).sum())
                boundary_error_c62[i] += int((err_c62 & bin_mask).sum())

        # --- connected components ---
        cc_a = connected_component_analysis(gt_vol, pred_a)
        cc_c62 = connected_component_analysis(gt_vol, pred_c62)

        subject_records.append({
            "subject_idx": subject_idx, "subject_id": subject_id,
            "dice_a": dice_a, "dice_c62": dice_c62, "delta_dice": dice_c62 - dice_a,
            "gt_lesion_volume": int(gt_vol.sum()),
            "cc_a": {"n_gt": cc_a["n_gt_components"], "n_pred": cc_a["n_pred_components"], "n_fp_components": cc_a["n_fp_components"],
                     "n_missed_components": sum(1 for c in cc_a["gt_components"] if not c["detected"]),
                     "mean_component_dice": float(np.mean([c["component_dice"] for c in cc_a["gt_components"]])) if cc_a["gt_components"] else None},
            "cc_c62": {"n_gt": cc_c62["n_gt_components"], "n_pred": cc_c62["n_pred_components"], "n_fp_components": cc_c62["n_fp_components"],
                       "n_missed_components": sum(1 for c in cc_c62["gt_components"] if not c["detected"]),
                       "mean_component_dice": float(np.mean([c["component_dice"] for c in cc_c62["gt_components"]])) if cc_c62["gt_components"] else None},
        })

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{n_subjects} subjects")

    del model_a, model_c62
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = {
        "n_subjects": n_subjects,
        "subject_records": subject_records,
        "slice_position": {
            "bin_edges": SLICE_BINS.tolist(),
            "error_rate_a": (slice_error_a / np.maximum(slice_total, 1)).tolist(),
            "error_rate_c62": (slice_error_c62 / np.maximum(slice_total, 1)).tolist(),
        },
        "boundary_distance": {
            "bin_labels": BOUNDARY_BIN_LABELS,
            "error_rate_a": (boundary_error_a / np.maximum(boundary_total, 1)).tolist(),
            "error_rate_c62": (boundary_error_c62 / np.maximum(boundary_total, 1)).tolist(),
            "n_voxels_per_bin": boundary_total.tolist(),
        },
        "four_way_classification": {
            "both_correct": both_correct,
            "a_wrong_c62_correct_IMPROVEMENT": a_wrong_c62_correct,
            "a_correct_c62_wrong_REGRESSION": a_correct_c62_wrong,
            "both_wrong": both_wrong,
        },
    }

    json_path = OUT_DIR / "spatial_error_analysis_C62_vs_A.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {json_path}")

    print("\n=== 4-way spatial classification (pooled, all subjects) ===")
    print(f"Both correct: {both_correct}")
    print(f"A wrong, C6-2 correct (IMPROVEMENT): {a_wrong_c62_correct}")
    print(f"A correct, C6-2 wrong (REGRESSION): {a_correct_c62_wrong}")
    print(f"Both wrong: {both_wrong}")


if __name__ == "__main__":
    main()
