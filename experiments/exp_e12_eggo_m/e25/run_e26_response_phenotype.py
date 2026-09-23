"""
Phase E26: D4 response phenotype. No training -- pure re-analysis of the
same 4 already-trained checkpoints (A, D4only, D2only, Both) on the full
125-subject validation set, extended with per-subject AND per-component
candidate predictor features, to answer: what distinguishes components/
subjects that benefit strongly from D4 supervision, beyond lesion size /
baseline Dice / obvious confounders?

Reuses run_deep_sup_4way_comparison.py's model-loading and component
analysis machinery. Extends it with:
  - subject-level candidate features: baseline (A) Dice, n_components,
    component-size CV (coefficient of variation = std/mean, a fragmentation/
    heterogeneity measure), total lesion volume, mean/std FLAIR intensity
    inside GT lesion, lesion surface-to-volume ratio (boundary complexity),
    distance-to-brain-center (spatial location proxy), A's own component
    Dice variance (a "how uneven is baseline performance" measure).
  - component-level candidate features: component size, A's own component
    Dice (baseline difficulty), component surface-to-volume ratio,
    component mean FLAIR intensity/contrast, component distance from
    volume center, isolation (distance to nearest other GT component).
  - subject-level target: delta_dice = dice_D4only - dice_A (per-subject
    whole-volume Dice, matching the 4-way comparison's own definition).
  - component-level target: delta_component_dice = D4only_comp_dice -
    A_comp_dice, restricted to components detected by BOTH (matching the
    prior paired analysis).
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
}
OUT_DIR = Path(__file__).parent / "e26_response_phenotype_results"


def load_model(ckpt_path, device, model_class):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = model_class(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def dice_score(pred, gt):
    tp = (pred * gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2 * tp / denom).item()) if denom > 0 else 1.0


def surface_to_volume(mask_bool):
    """Boundary voxel count / total voxel count -- a shape-complexity proxy.
    Boundary = voxels in the mask adjacent (6-connectivity) to a voxel outside it."""
    if mask_bool.sum() == 0:
        return None
    eroded = ndi.binary_erosion(mask_bool)
    surface = mask_bool & ~eroded
    return float(surface.sum()) / float(mask_bool.sum())


def component_features(gt_labeled, comp_id, image_vol, all_centroids, this_centroid):
    mask = gt_labeled == comp_id
    size = int(mask.sum())
    intensity_vals = image_vol[mask]
    mean_intensity = float(intensity_vals.mean())
    std_intensity = float(intensity_vals.std())
    svr = surface_to_volume(mask)
    center = np.array(image_vol.shape) / 2.0
    dist_from_center = float(np.linalg.norm(this_centroid - center))
    other_centroids = [c for c in all_centroids if not np.allclose(c, this_centroid)]
    if other_centroids:
        dists = [np.linalg.norm(this_centroid - c) for c in other_centroids]
        isolation = float(min(dists))
    else:
        isolation = None
    return {
        "size": size, "mean_intensity": mean_intensity, "std_intensity": std_intensity,
        "surface_to_volume": svr, "dist_from_center": dist_from_center, "isolation": isolation,
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

    print("Loading A and D4only models...")
    models = {name: load_model(path, device, cls) for name, (path, cls) in CHECKPOINTS.items()}

    subject_records = []
    component_records = []

    for subject_idx in range(n_subjects):
        image, mask, subject_id = val_dataset[subject_idx]
        image_b = image.unsqueeze(0).to(device)
        image_vol = image.squeeze(0).numpy().astype(np.float32)
        gt_vol = mask.squeeze(0).numpy().astype(np.float32)

        preds = {}
        with torch.no_grad():
            for name, model in models.items():
                preds[name] = (model(image_b)["probs"] >= 0.5).float().squeeze(0).squeeze(0).cpu().numpy()

        dice_a = dice_score(torch.from_numpy(preds["A"]), torch.from_numpy(gt_vol))
        dice_d4 = dice_score(torch.from_numpy(preds["D4only"]), torch.from_numpy(gt_vol))

        gt_bool = gt_vol > 0.5
        gt_labeled, n_comp = ndi.label(gt_bool)
        gt_lesion_volume = int(gt_bool.sum())

        if n_comp == 0:
            continue

        sizes = []
        centroids = ndi.center_of_mass(gt_bool, gt_labeled, range(1, n_comp + 1))
        centroids = [np.array(c) for c in (centroids if isinstance(centroids, list) else [centroids])]

        comp_a_dices = []
        comp_d4_dices = []
        comp_feats_list = []

        pred_a_labeled, _ = ndi.label(preds["A"] > 0.5)
        pred_d4_labeled, _ = ndi.label(preds["D4only"] > 0.5)

        for comp_id in range(1, n_comp + 1):
            comp_mask = gt_labeled == comp_id
            size = int(comp_mask.sum())
            sizes.append(size)
            feats = component_features(gt_labeled, comp_id, image_vol, centroids, centroids[comp_id - 1])

            def comp_dice_for(pred_labeled, pred_vol):
                overlap_ids = set(pred_labeled[comp_mask].flatten().tolist()) - {0}
                if not overlap_ids:
                    return None  # not detected
                combined = np.isin(pred_labeled, list(overlap_ids))
                inter = float((comp_mask & combined).sum())
                denom = size + combined.sum()
                return 2 * inter / denom if denom > 0 else 1.0

            da = comp_dice_for(pred_a_labeled, preds["A"])
            dd4 = comp_dice_for(pred_d4_labeled, preds["D4only"])

            comp_record = {
                "subject_idx": subject_idx, "component_id": comp_id,
                **feats,
                "a_detected": da is not None, "d4_detected": dd4 is not None,
                "a_comp_dice": da, "d4_comp_dice": dd4,
            }
            if da is not None and dd4 is not None:
                comp_record["delta_comp_dice"] = dd4 - da
            component_records.append(comp_record)
            if da is not None:
                comp_a_dices.append(da)

        sizes = np.array(sizes, dtype=float)
        size_cv = float(sizes.std() / sizes.mean()) if sizes.mean() > 0 and n_comp > 1 else 0.0
        a_comp_dice_var = float(np.var(comp_a_dices)) if len(comp_a_dices) > 1 else None

        lesion_svr = surface_to_volume(gt_bool)
        intensity_vals = image_vol[gt_bool]

        subject_records.append({
            "subject_idx": subject_idx, "subject_id": subject_id,
            "dice_A": dice_a, "dice_D4only": dice_d4, "delta_dice": dice_d4 - dice_a,
            "gt_lesion_volume": gt_lesion_volume, "n_gt_components": n_comp,
            "mean_component_size": float(sizes.mean()), "size_cv": size_cv,
            "lesion_surface_to_volume": lesion_svr,
            "mean_intensity": float(intensity_vals.mean()), "std_intensity": float(intensity_vals.std()),
            "a_comp_dice_variance": a_comp_dice_var,
        })

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{n_subjects} subjects")

    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = {"n_subjects": len(subject_records), "subject_records": subject_records, "component_records": component_records}
    out_path = OUT_DIR / "e26_response_phenotype.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(subject_records)} subjects, {len(component_records)} components to {out_path}")


if __name__ == "__main__":
    main()
