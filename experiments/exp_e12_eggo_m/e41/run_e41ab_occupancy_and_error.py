"""
Phase E41-A/B: target-transformation occupancy field and residual-error
audit. NO TRAINING. NO NEW LOSS.

PREMISE CORRECTION, made before writing any code, per direct inspection of
the real training script: the original E41 prompt frames this around
"information destroyed by binarizing the coarse target." Checked directly
against experiments/exp_e12_eggo_m/e25/train_deep_sup.py (and e39's own
copy) -- there is NO thresholding/binarization step anywhere in this
project's real training pipeline. mask_d4 = F.avg_pool3d(masks, 4) and
mask_d2 = F.avg_pool3d(masks, 2) are used DIRECTLY as the auxiliary loss
targets, unthresholded. The model already trains against the raw occupancy
fraction p_j, which the accompanying mathematical derivation (optimal
q_j* = p_j under voxelwise cross-entropy) independently confirms is already
the RIGHT target -- so this project has no "binary approximation" gap to
close. Reframed, per this correction (agreed with the user before running):
instead of computing a hypothetical binary-conversion error against a
thresholding step that doesn't exist, this measures the model's ACTUAL
prediction error against the REAL (already-soft) target it was trained on,
and asks whether THAT residual error -- not size, not a synthetic entropy
measure, not a hypothetical binarization artifact -- predicts where D4
supervision actually helps.

E41A: occupancy field P_16 = D_16(Y), P_32 = D_32(Y), UNTHRESHOLDED, for all
749 native components. Records occupancy distribution, fraction of cells at
p=0/partial/p=1, total mass, boundary/partial-cell mass -- all at the CELL
level within each component's own touched footprint (not the whole-image
level), reusing the verified per-component contribution-decomposition method
from e36/run_e36b2_component_localization_avgpool.py (mass-conservation
already verified there to float precision; the SAME construction is reused
here unchanged, not reimplemented from scratch).

E41B (reframed): E_s = per-component mean BCE between the model's ACTUAL
prediction q_j and the TRUE occupancy p_j, mass-weighted by this
component's own contribution to each touched coarse cell (same linear
decomposition-of-an-average logic already verified in E36-B2). Uses the
"Both" checkpoint (D4 and D2 both genuinely active in one model), matching
E39C/E40's own established convention for a fair D4-vs-D2 comparison.

Output: E41_occupancy_and_error_table.json
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import scipy.ndimage as ndi
from scipy.ndimage import zoom

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
ROSTER_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e35" / "E35_component_roster.json"
CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_seed0" / "checkpoints" / "epoch_30.pth"
BCE_EPS = 1e-7  # clamp for log(0) safety, standard convention


def resize_nn(volume, target_shape):
    zf = tuple(t / c for t, c in zip(target_shape, volume.shape))
    return zoom(volume, zf, order=0)


def per_component_scale_stats(comp_mask_64, pred_np, target_np, factor):
    """factor=4 for D4 (16^3), factor=2 for D2 (32^3). Returns a dict of
    occupancy-distribution stats AND model-error stats for this component at
    this scale, using the SAME contribution-decomposition method verified in
    E36-B2 (mass-conservation checked there to float precision)."""
    res = 64 // factor
    comp_blocks = comp_mask_64.reshape(res, factor, res, factor, res, factor).astype(np.float32)
    contribution = comp_blocks.sum(axis=(1, 3, 5)) / (factor ** 3)  # this component's own share of each coarse cell's occupancy

    touched = contribution > 0
    n_touched = int(touched.sum())
    total_mass = float(contribution.sum())

    if n_touched == 0:
        return {
            "n_cells_touched": 0, "total_mass": total_mass,
            "frac_cells_p0": None, "frac_cells_partial": None, "frac_cells_p1": None,
            "boundary_partial_mass": None, "mean_bce": None, "mean_abs_error": None,
        }

    # Occupancy distribution restricted to this component's OWN touched cells
    # (contribution values ARE this component's own p_j restricted to its
    # own mass -- a cell is "p=1 for this component" only if the ENTIRE cell
    # belongs to this component, i.e. contribution==1 exactly; "partial" if
    # 0<contribution<1; there is no "p=0 touched" case by construction, since
    # touched already means contribution>0).
    touched_contribs = contribution[touched]
    n_p1 = int((touched_contribs >= 1.0 - 1e-9).sum())
    n_partial = int(((touched_contribs > 0) & (touched_contribs < 1.0 - 1e-9)).sum())
    frac_p1 = n_p1 / n_touched
    frac_partial = n_partial / n_touched
    boundary_mass = float(touched_contribs[touched_contribs < 1.0 - 1e-9].sum())

    # Model error at this component's touched cells: the REAL target here is
    # the WHOLE-CELL occupancy p_j (target_np, computed over ALL components
    # sharing this cell, since that's what the model is actually trained
    # against and actually predicts for) -- contribution is used only as the
    # ATTRIBUTION WEIGHT for how much of that cell's error belongs to this
    # component, matching E36-B2's own verified mass-conservation logic
    # exactly (never substituting contribution itself as if it were the
    # target the model saw).
    p_whole_cell = target_np[touched]
    q_pred = pred_np[touched]
    q_clamped = np.clip(q_pred, BCE_EPS, 1 - BCE_EPS)
    bce_per_cell = -(p_whole_cell * np.log(q_clamped) + (1 - p_whole_cell) * np.log(1 - q_clamped))
    weights = contribution[touched]

    mean_bce = float(np.sum(bce_per_cell * weights) / total_mass) if total_mass > 0 else None
    mean_abs_err = float(np.sum(np.abs(q_pred - p_whole_cell) * weights) / total_mass) if total_mass > 0 else None

    return {
        "n_cells_touched": n_touched, "total_mass": total_mass,
        "frac_cells_p0": 0.0,  # by construction of "touched"
        "frac_cells_partial": frac_partial, "frac_cells_p1": frac_p1,
        "boundary_partial_mass": boundary_mass,
        "mean_bce": mean_bce, "mean_abs_error": mean_abs_err,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    roster = json.load(open(ROSTER_PATH))
    print(f"Roster loaded: {len(roster)} components", flush=True)

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    subject_dirs = {Path(d).name: d for d in val_dataset.subject_dirs}

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = UNet3D_v3(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded 'Both' checkpoint epoch_30 (best_val_dice={ckpt.get('best_val_dice')})", flush=True)

    by_subject = {}
    for r in roster:
        by_subject.setdefault(r["subject_id"], []).append(r)

    records = []
    for subject_idx in range(len(val_dataset)):
        image, mask, subject_id = val_dataset[subject_idx]
        if subject_id not in by_subject:
            continue
        image_b = image.unsqueeze(0).to(device)
        mask_b = mask.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(image_b)
            pred_d4 = out["aux_probs3"].squeeze().cpu().numpy().astype(np.float64)
            pred_d2 = out["aux_probs2"].squeeze().cpu().numpy().astype(np.float64)
        target_d4 = F.avg_pool3d(mask_b, kernel_size=4, stride=4).squeeze().cpu().numpy().astype(np.float64)
        target_d2 = F.avg_pool3d(mask_b, kernel_size=2, stride=2).squeeze().cpu().numpy().astype(np.float64)

        subject_dir = subject_dirs[subject_id]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_labeled, n_native = ndi.label(seg_binary_native > 0.5)

        labeled_64 = resize_nn(native_labeled.astype(np.float32), (64, 64, 64))
        labeled_64 = np.round(labeled_64).astype(np.int32)

        for r in by_subject[subject_id]:
            comp_id = r["native_component_id"]
            comp_mask_64 = (labeled_64 == comp_id)

            stats_d4 = per_component_scale_stats(comp_mask_64, pred_d4, target_d4, factor=4)
            stats_d2 = per_component_scale_stats(comp_mask_64, pred_d2, target_d2, factor=2)

            records.append({
                "subject_id": subject_id, "native_component_id": comp_id,
                "native_size": r["native_size"], "size_64": r["size_64"],
                "detected_by_D4": r["detected_by_D4"], "detected_by_D2": r["detected_by_D2"],
                "component_dice_D4": r["component_dice_D4"], "component_dice_D2": r["component_dice_D2"],
                "component_dice_A": r["component_dice_A"],
                "D4": stats_d4, "D2": stats_d2,
            })

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{len(val_dataset)} subjects", flush=True)

    with open(OUT_DIR / "E41_occupancy_and_error_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} component records.", flush=True)

    # Mass-conservation spot check, SAME verified method as E36-B2
    by_subj_mass_d4 = {}
    for r in records:
        by_subj_mass_d4.setdefault(r["subject_id"], 0.0)
        by_subj_mass_d4[r["subject_id"]] += r["D4"]["total_mass"]
    print("\nMass-conservation spot check, D4 (first 3 subjects):")
    for sid in list(by_subj_mass_d4.keys())[:3]:
        print(f"  {sid}: sum of component D4 contributions = {by_subj_mass_d4[sid]:.4f}")

    n_touched_d4 = sum(1 for r in records if r["D4"]["n_cells_touched"] > 0)
    n_touched_d2 = sum(1 for r in records if r["D2"]["n_cells_touched"] > 0)
    print(f"\nComponents with nonzero D4 footprint: {n_touched_d4}/{len(records)}")
    print(f"Components with nonzero D2 footprint: {n_touched_d2}/{len(records)}")


if __name__ == "__main__":
    main()
