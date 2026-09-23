"""
Phase E27: component-specific decoder-scale preference audit -- data collection.

NO TRAINING. Inference-only, on the already-trained best.pth checkpoints for
A / D4-only / D2-only / Both, full 125-subject validation set, plus periodic
epoch checkpoints (1,5,10,15,20,25,30) for the early-predictability analysis
(Section 10-12 of the execution prompt).

Matching rule (Section 3, documented explicitly, IDENTICAL across all four
models): for each GT connected component, find all predicted connected
components whose voxels overlap it, and union them into a single "matched
prediction region" for that GT component. This is a MANY-predicted-to-ONE-GT
matching (not GT-to-pred), chosen because it is well-defined even when a
model fragments one lesion into several predicted pieces or fails to
separate two nearby GT lesions -- both of which occur in this data. It does
NOT use a hard IoU threshold to decide "detected" for the union region
(union region is always compared to the GT component once formed); the
detected/missed decision is: matched-prediction-region is empty <=> missed.
This is the SAME rule used in every prior E25/E26 component analysis this
session, applied unchanged here for all four models and for every checkpoint
epoch, so results are directly comparable and no model gets a favorable
matching convention.
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

RUN_DIRS = {
    "A": (project_root / "experiments" / "exp_e12_eggo_m" / "e24" / "gate6_runs" / "A_baseline_seed0", UNet3D_v2),
    "D4": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_D4only_seed0", UNet3D_v3),
    "D2": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_D2only_seed0", UNet3D_v3),
    "Both": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_seed0", UNet3D_v3),
}
EARLY_EPOCHS = [1, 5, 10, 15, 20, 25, 30]  # all periodic checkpoints that exist on disk (verified before writing this script)
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(exist_ok=True)


def load_model(ckpt_path, device, model_class):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = model_class(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("epoch")


def component_metrics(gt_mask, pred_labeled, gt_id, gt_labeled):
    """Returns dict with detected, size, intersection, pred_size, gt_size for
    ONE gt component, using the union-of-overlapping-predicted-components
    matching rule documented at module level. Missed-component convention:
    dice=0.0, iou=0.0, coverage=0.0, precision=None (undefined -- no
    predicted region exists to compute precision over; documented, not
    silently defaulted to 0 or 1)."""
    comp_mask = gt_labeled == gt_id
    gt_size = int(comp_mask.sum())
    overlap_ids = set(pred_labeled[comp_mask].flatten().tolist()) - {0}
    if not overlap_ids:
        return {
            "gt_size": gt_size, "detected": False,
            "dice": 0.0, "iou": 0.0, "coverage": 0.0, "precision": None,
            "pred_size": 0, "intersection": 0,
        }
    pred_region = np.isin(pred_labeled, list(overlap_ids))
    pred_size = int(pred_region.sum())
    inter = int((comp_mask & pred_region).sum())
    union = gt_size + pred_size - inter
    dice = 2 * inter / (gt_size + pred_size) if (gt_size + pred_size) > 0 else 1.0
    iou = inter / union if union > 0 else 1.0
    coverage = inter / gt_size if gt_size > 0 else 1.0
    precision = inter / pred_size if pred_size > 0 else None
    return {
        "gt_size": gt_size, "detected": True,
        "dice": float(dice), "iou": float(iou), "coverage": float(coverage),
        "precision": (float(precision) if precision is not None else None),
        "pred_size": pred_size, "intersection": inter,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_subjects = len(val_dataset)
    print(f"Full validation set: {n_subjects} subjects")

    # ---------- Pass 1: best-checkpoint component table (main analysis) ----------
    print("\n=== Loading best.pth for A/D4/D2/Both ===")
    best_models = {}
    for name, (run_dir, cls) in RUN_DIRS.items():
        m, ep = load_model(run_dir / "checkpoints" / "best.pth", device, cls)
        best_models[name] = m
        print(f"  {name}: best.pth epoch={ep}")

    component_table = []
    for subject_idx in range(n_subjects):
        image, mask, subject_id = val_dataset[subject_idx]
        image_b = image.unsqueeze(0).to(device)
        gt_vol = mask.squeeze(0).numpy().astype(np.float32)
        gt_bool = gt_vol > 0.5
        gt_labeled, n_gt = ndi.label(gt_bool)
        if n_gt == 0:
            continue

        preds_labeled = {}
        with torch.no_grad():
            for name, model in best_models.items():
                pred_bin = (model(image_b)["probs"] >= 0.5).float().squeeze(0).squeeze(0).cpu().numpy()
                preds_labeled[name], _ = ndi.label(pred_bin > 0.5)

        for gt_id in range(1, n_gt + 1):
            row = {"subject_id": subject_id, "subject_idx": subject_idx, "component_id": gt_id}
            for name in ("A", "D4", "D2", "Both"):
                m = component_metrics(gt_bool, preds_labeled[name], gt_id, gt_labeled)
                row[f"gt_size"] = m["gt_size"]
                row[f"detected_{name}"] = m["detected"]
                row[f"dice_{name}"] = m["dice"]
                row[f"iou_{name}"] = m["iou"]
                row[f"coverage_{name}"] = m["coverage"]
                row[f"precision_{name}"] = m["precision"]
            component_table.append(row)

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{n_subjects} subjects")

    del best_models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\nTotal GT components collected: {len(component_table)}")
    with open(OUT_DIR / "E27_component_table.json", "w") as f:
        json.dump(component_table, f)
    print(f"Saved component table to {OUT_DIR / 'E27_component_table.json'}")

    # ---------- Pass 2: early-trajectory component metrics (periodic checkpoints) ----------
    print("\n=== Loading periodic checkpoints for early-trajectory features ===")
    trajectory_table = []
    for name, (run_dir, cls) in RUN_DIRS.items():
        for epoch in EARLY_EPOCHS:
            ckpt_path = run_dir / "checkpoints" / f"epoch_{epoch}.pth"
            if not ckpt_path.exists():
                print(f"  MISSING: {ckpt_path} -- skipping")
                continue
            model, actual_epoch = load_model(ckpt_path, device, cls)
            print(f"  {name} epoch_{epoch}.pth loaded (recorded epoch field={actual_epoch})")

            for subject_idx in range(n_subjects):
                image, mask, subject_id = val_dataset[subject_idx]
                image_b = image.unsqueeze(0).to(device)
                gt_vol = mask.squeeze(0).numpy().astype(np.float32)
                gt_bool = gt_vol > 0.5
                gt_labeled, n_gt = ndi.label(gt_bool)
                if n_gt == 0:
                    continue

                with torch.no_grad():
                    pred_bin = (model(image_b)["probs"] >= 0.5).float().squeeze(0).squeeze(0).cpu().numpy()
                pred_labeled, _ = ndi.label(pred_bin > 0.5)

                for gt_id in range(1, n_gt + 1):
                    m = component_metrics(gt_bool, pred_labeled, gt_id, gt_labeled)
                    trajectory_table.append({
                        "condition": name, "checkpoint_epoch": epoch,
                        "subject_id": subject_id, "subject_idx": subject_idx, "component_id": gt_id,
                        "gt_size": m["gt_size"], "detected": m["detected"],
                        "dice": m["dice"], "iou": m["iou"], "coverage": m["coverage"], "precision": m["precision"],
                    })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nTotal trajectory records: {len(trajectory_table)}")
    with open(OUT_DIR / "E27_trajectory_table.json", "w") as f:
        json.dump(trajectory_table, f)
    print(f"Saved trajectory table to {OUT_DIR / 'E27_trajectory_table.json'}")


if __name__ == "__main__":
    main()
