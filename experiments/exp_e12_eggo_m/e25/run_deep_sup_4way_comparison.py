"""
Phase E25, Structural Pivot 1B: 4-way mechanistic comparison
(A vs D4-only vs D2-only vs Both), full 125-subject validation set.
Per the user's explicit direction after the scale ablation's headline
result (D4-only=0.9096 >= Both=0.9091 > D2-only=0.9080 > A=0.9063): NO
new training. Uses the already-trained real best.pth checkpoints for
all four conditions.

Reuses run_deep_sup_mechanism_audit.py's own validated machinery
(connected_component_analysis_with_sizes, dice_score, classify_state)
UNCHANGED, extended to FOUR conditions instead of two, plus:
  - per-subject Dice for all 4 conditions (not just A vs one candidate)
  - a same-basis D4-only vs Both comparison (paired, same subjects) to
    directly test whether the D4-only > Both headline gap (0.9096 vs
    0.9091, i.e. +0.05pp) is a real, consistent, subject-level pattern
    or noise-level (mixed direction across subjects).
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

CHECKPOINTS = {
    "A": (project_root / "experiments" / "exp_e12_eggo_m" / "e24" / "gate6_runs" / "A_baseline_seed0" / "checkpoints" / "best.pth", UNet3D_v2),
    "D4only": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_D4only_seed0" / "checkpoints" / "best.pth", UNet3D_v3),
    "D2only": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_D2only_seed0" / "checkpoints" / "best.pth", UNet3D_v3),
    "Both": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_seed0" / "checkpoints" / "best.pth", UNet3D_v3),
}
OUT_DIR = Path(__file__).parent / "deep_sup_4way_comparison_results"

SIZE_BINS = [0, 50, 150, 400, 1000, float("inf")]
SIZE_BIN_LABELS = ["1-50", "50-150", "150-400", "400-1000", ">1000"]


def load_model(ckpt_path, device, model_class):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = model_class(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  loaded {ckpt_path.parent.parent.name}/{ckpt_path.name} ({model_class.__name__}): epoch={ckpt.get('epoch')} best_val_dice={ckpt.get('best_val_dice')}")
    return model


def dice_score(pred, gt):
    tp = (pred * gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2 * tp / denom).item()) if denom > 0 else 1.0


def connected_component_analysis_with_sizes(gt_vol, pred_vol):
    """Verbatim from run_deep_sup_mechanism_audit.py -- UNCHANGED."""
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

    print("\nLoading all 4 models...")
    models = {name: load_model(path, device, cls) for name, (path, cls) in CHECKPOINTS.items()}

    subject_records = []
    all_components = {name: [] for name in CHECKPOINTS}
    boundary_err = {name: [] for name in CHECKPOINTS}

    for subject_idx in range(n_subjects):
        image, mask, subject_id = val_dataset[subject_idx]
        image_b = image.unsqueeze(0).to(device)
        gt_vol = mask.squeeze(0).numpy().astype(np.float32)

        preds = {}
        with torch.no_grad():
            for name, model in models.items():
                preds[name] = (model(image_b)["probs"] >= 0.5).float().squeeze(0).squeeze(0).cpu().numpy()

        dices = {name: dice_score(torch.from_numpy(preds[name]), torch.from_numpy(gt_vol)) for name in CHECKPOINTS}

        gt_bool = gt_vol > 0.5
        near_boundary = None
        if gt_bool.sum() > 0 and gt_bool.sum() < gt_bool.size:
            dist_outside = ndi.distance_transform_edt(~gt_bool)
            dist_inside = ndi.distance_transform_edt(gt_bool)
            dist = np.where(gt_bool, dist_inside, dist_outside)
            near_boundary = (dist >= 1) & (dist < 2)

        missed_counts = {}
        for name in CHECKPOINTS:
            cc = connected_component_analysis_with_sizes(gt_vol, preds[name])
            all_components[name].extend(cc["gt_components"])
            missed_counts[name] = sum(1 for c in cc["gt_components"] if not c["detected"])
            if near_boundary is not None and near_boundary.sum() > 0:
                err = (preds[name] != gt_vol)
                boundary_err[name].append(float(err[near_boundary].mean()))

        gt_lesion_volume = int(gt_vol.sum())
        n_components = connected_component_analysis_with_sizes(gt_vol, preds["A"])["n_gt_components"]
        mean_component_size = gt_lesion_volume / max(n_components, 1)

        record = {
            "subject_idx": subject_idx, "subject_id": subject_id,
            "gt_lesion_volume": gt_lesion_volume, "n_gt_components": n_components,
            "mean_component_size": mean_component_size,
        }
        for name in CHECKPOINTS:
            record[f"dice_{name}"] = dices[name]
            record[f"n_missed_{name}"] = missed_counts[name]
        subject_records.append(record)

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{n_subjects} subjects")

    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = {
        "n_subjects": n_subjects,
        "subject_records": subject_records,
        "all_components": all_components,
        "boundary_error_rate": {name: (float(np.mean(v)) if v else None) for name, v in boundary_err.items()},
    }

    json_path = OUT_DIR / "deep_sup_4way_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {json_path}")

    print("\n=== Boundary error rates (1-2 voxel layer) ===")
    for name, v in results["boundary_error_rate"].items():
        print(f"  {name}: {v:.4f}")


if __name__ == "__main__":
    main()
