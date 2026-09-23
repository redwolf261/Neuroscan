"""
Phase E25, Structural Pivot 1A: small-lesion mechanism audit. Per the
user's explicit direction after Deep Supervision's first result (A=0.9063
-> DS=0.9091, +0.28pp -- the first candidate in the whole arc to beat A):
before running any ablation or further training, verify whether the
specific failure population that MOTIVATED deep supervision (small
ground-truth lesion components, per PHASE_E25_NEXT_STRUCTURAL_PIVOT.md's
own failure model: corr(component_size, missed_fraction)=-0.60, p<0.0001,
43.1% missed-component rate for small-lesion subjects vs 8.2% for large)
actually improved under Deep Supervision, or whether the +0.28pp Dice
gain is coming from somewhere else entirely.

Reuses run_c62_spatial_error_analysis.py's own validated machinery
(connected_component_analysis, dice_score, signed_distance_from_boundary)
UNCHANGED -- extended with PER-COMPONENT size tracking (the original
script only aggregated missed-component COUNTS per subject, not each
individual component's own size, which this audit needs for the
size-stratified recall breakdown the user specifically asked for) and
explicit confusion-transition totals (FN->TP, TP->FN, FP->TN, TN->FP,
matching C6-2.9's own 4-way classification convention).

Compares A's real best.pth (epoch 24, val_dice=0.9063) against Deep
Supervision's real best.pth (epoch 29, val_dice=0.9091) -- the SAME
comparison pattern as run_c62_spatial_error_analysis.py used for A vs
C6-2, just with different checkpoints. No new training, no perturbation.
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
from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

GATE6_A_BEST = project_root / "experiments" / "exp_e12_eggo_m" / "e24" / "gate6_runs" / "A_baseline_seed0" / "checkpoints" / "best.pth"
DS_BEST = project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_seed0" / "checkpoints" / "best.pth"
OUT_DIR = Path(__file__).parent / "deep_sup_mechanism_audit_results"

# Size bins for component-level recall stratification (voxel count) --
# chosen to give roughly comparable bin populations given this cohort's
# own component-size distribution (median component size ~200-400
# voxels per the failure model's own subject-level mean_component_size
# calculation), not arbitrary round numbers.
SIZE_BINS = [0, 50, 150, 400, 1000, float("inf")]
SIZE_BIN_LABELS = ["1-50", "50-150", "150-400", "400-1000", ">1000"]


def load_model(ckpt_path, device, model_class):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = model_class(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  loaded {ckpt_path.name} ({model_class.__name__}): epoch={ckpt.get('epoch')} best_val_dice={ckpt.get('best_val_dice')}")
    return model


def dice_score(pred, gt):
    tp = (pred * gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2 * tp / denom).item()) if denom > 0 else 1.0


def classify_state(gt, pred):
    gt_pos = gt > 0.5
    pred_pos = pred > 0.5
    tp = gt_pos & pred_pos
    tn = (~gt_pos) & (~pred_pos)
    fp = (~gt_pos) & pred_pos
    fn = gt_pos & (~pred_pos)
    cat = torch.zeros_like(gt, dtype=torch.int8)
    cat[tp] = 1
    cat[fp] = 2
    cat[fn] = 3
    return cat


def connected_component_analysis_with_sizes(gt_vol, pred_vol):
    """EXTENDS run_c62_spatial_error_analysis.py's own function: same
    detection/component_dice logic, but ALSO returns each individual
    GT component's own size (not just aggregate counts), needed for the
    size-stratified recall breakdown."""
    gt_labeled, n_gt = ndi.label(gt_vol > 0.5)
    pred_labeled, n_pred = ndi.label(pred_vol > 0.5)

    gt_records = []
    for gt_id in range(1, n_gt + 1):
        gt_mask = gt_labeled == gt_id
        gt_size = int(gt_mask.sum())
        overlapping_pred_ids = set(pred_labeled[gt_mask].flatten().tolist()) - {0}
        if not overlapping_pred_ids:
            gt_records.append({"gt_size": gt_size, "detected": False, "component_dice": 0.0})
            continue
        combined_pred_mask = np.isin(pred_labeled, list(overlapping_pred_ids))
        inter = float((gt_mask & combined_pred_mask).sum())
        comp_dice = 2 * inter / (gt_size + combined_pred_mask.sum()) if (gt_size + combined_pred_mask.sum()) > 0 else 1.0
        gt_records.append({"gt_size": gt_size, "detected": True, "component_dice": float(comp_dice)})

    fp_component_count = 0
    for pred_id in range(1, n_pred + 1):
        pred_mask = pred_labeled == pred_id
        if not (gt_labeled[pred_mask] > 0).any():
            fp_component_count += 1

    return {"n_gt_components": n_gt, "n_pred_components": n_pred, "n_fp_components": fp_component_count, "gt_components": gt_records}


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
    model_a = load_model(GATE6_A_BEST, device, UNet3D_v2)
    model_ds = load_model(DS_BEST, device, UNet3D_v3)

    subject_records = []
    # pooled confusion transition matrix (4x4, TN/TP/FP/FN order 0-3)
    matrix = torch.zeros(4, 4, dtype=torch.int64)
    all_components_a = []  # (size, detected, component_dice) across ALL subjects
    all_components_ds = []
    boundary_err_a_total = np.zeros(0)
    boundary_err_ds_total = np.zeros(0)

    for subject_idx in range(n_subjects):
        image, mask, subject_id = val_dataset[subject_idx]
        image_b = image.unsqueeze(0).to(device)
        gt_vol = mask.squeeze(0).numpy().astype(np.float32)
        gt_flat = mask.squeeze(0).reshape(-1).to(device)

        with torch.no_grad():
            pred_a = (model_a(image_b)["probs"] >= 0.5).float().squeeze(0).squeeze(0).cpu().numpy()
            pred_ds = (model_ds(image_b)["probs"] >= 0.5).float().squeeze(0).squeeze(0).cpu().numpy()

        dice_a = dice_score(torch.from_numpy(pred_a), torch.from_numpy(gt_vol))
        dice_ds = dice_score(torch.from_numpy(pred_ds), torch.from_numpy(gt_vol))

        # --- confusion transition matrix (state under A -> state under DS) ---
        pred_a_flat = torch.from_numpy(pred_a).to(device).reshape(-1)
        pred_ds_flat = torch.from_numpy(pred_ds).to(device).reshape(-1)
        state_a = classify_state(gt_flat, pred_a_flat)
        state_ds = classify_state(gt_flat, pred_ds_flat)
        idx = state_a.long() * 4 + state_ds.long()
        counts = torch.bincount(idx, minlength=16).reshape(4, 4)
        matrix += counts.cpu()

        # --- connected components, with per-component sizes ---
        cc_a = connected_component_analysis_with_sizes(gt_vol, pred_a)
        cc_ds = connected_component_analysis_with_sizes(gt_vol, pred_ds)
        all_components_a.extend(cc_a["gt_components"])
        all_components_ds.extend(cc_ds["gt_components"])

        # --- boundary check (only to confirm the +0.28pp isn't an unrelated boundary tradeoff) ---
        gt_bool = gt_vol > 0.5
        if gt_bool.sum() > 0 and gt_bool.sum() < gt_bool.size:
            dist_outside = ndi.distance_transform_edt(~gt_bool)
            dist_inside = ndi.distance_transform_edt(gt_bool)
            dist = np.where(gt_bool, dist_inside, dist_outside)
            near_boundary = (dist >= 1) & (dist < 2)  # matches C6-2.9's own 1-2 voxel "true boundary layer" bin
            err_a = (pred_a != gt_vol)
            err_ds = (pred_ds != gt_vol)
            if near_boundary.sum() > 0:
                boundary_err_a_total = np.append(boundary_err_a_total, err_a[near_boundary].mean())
                boundary_err_ds_total = np.append(boundary_err_ds_total, err_ds[near_boundary].mean())

        gt_lesion_volume = int(gt_vol.sum())
        n_components = cc_a["n_gt_components"]
        mean_component_size = gt_lesion_volume / max(n_components, 1)

        subject_records.append({
            "subject_idx": subject_idx, "subject_id": subject_id,
            "dice_a": dice_a, "dice_ds": dice_ds, "delta_dice": dice_ds - dice_a,
            "gt_lesion_volume": gt_lesion_volume, "n_gt_components": n_components,
            "mean_component_size": mean_component_size,
            "n_missed_a": sum(1 for c in cc_a["gt_components"] if not c["detected"]),
            "n_missed_ds": sum(1 for c in cc_ds["gt_components"] if not c["detected"]),
            "n_fp_components_a": cc_a["n_fp_components"], "n_fp_components_ds": cc_ds["n_fp_components"],
        })

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{n_subjects} subjects")

    del model_a, model_ds
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = {
        "n_subjects": n_subjects,
        "subject_records": subject_records,
        "confusion_transition_matrix": matrix.tolist(),  # rows=state under A, cols=state under DS (0=TN,1=TP,2=FP,3=FN)
        "all_components_a": all_components_a,
        "all_components_ds": all_components_ds,
        "boundary_error_rate_a": float(np.mean(boundary_err_a_total)) if len(boundary_err_a_total) else None,
        "boundary_error_rate_ds": float(np.mean(boundary_err_ds_total)) if len(boundary_err_ds_total) else None,
    }

    json_path = OUT_DIR / "deep_sup_mechanism_audit.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {json_path}")

    # --- print headline numbers directly ---
    tp_to_fn = int(matrix[1, 3].item())
    tn_to_fp = int(matrix[0, 2].item())
    fn_to_tp = int(matrix[3, 1].item())
    fp_to_tn = int(matrix[2, 0].item())
    print(f"\nConfusion transitions (A -> DS): FN->TP={fn_to_tp} FP->TN={fp_to_tn} TP->FN={tp_to_fn} TN->FP={tn_to_fp}")
    print(f"Boundary error rate (1-2 voxel layer): A={results['boundary_error_rate_a']:.4f} DS={results['boundary_error_rate_ds']:.4f}")


if __name__ == "__main__":
    main()
