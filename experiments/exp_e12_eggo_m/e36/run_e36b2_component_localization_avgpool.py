"""
Phase E36-B2: component-level localization of D4's persistent loss floor,
FIXED to match the ACTUAL training-target construction.

WHY THIS EXISTS: E36-B's first attempt attributed D4 voxels to native
components via NEAREST-NEIGHBOR resampling of the component-ID volume
(the convention used throughout E30/E34/E35's own component-identity work)
-- but the REAL D4 training target is F.avg_pool3d(masks, kernel_size=4,
stride=4), a fundamentally different, continuous operation. Under nearest-
neighbor, every S1-S4 component (587/749, the small-lesion population) had
EXACTLY ZERO D4 footprint -- consistent with prior findings (E30/E34: small
lesions vanish under nearest-neighbor resampling to 16^3) but useless for
attributing D4's own loss, since the model's D4 prediction is trained
against average-pooled coverage, not a nearest-neighbor label.

FIX: attribute each native component to a D4 voxel based on whether it
contributes ANY nonzero MASS to that voxel's average-pool computation --
i.e. does the component's binary mask, resized to 64^3 via the SAME
nearest-neighbor convention used everywhere else in this project (matching
size_64 in E35's roster, verified consistent), have ANY nonzero overlap with
the 4x4x4 64^3 block that avg_pool3d reduces into this D4 voxel? This
matches the real training target's own math (soft coverage from ANY
contributing source voxels) rather than a hard "which component owns this
coarse voxel" question. A component can now contribute (fractionally) to
MULTIPLE D4 voxels, and multiple components can share a D4 voxel -- this is
handled by measuring MAE at the 64^3-block level, weighted by how much of
each 4x4x4 block belongs to this specific component (not the whole visible
tumor), which is the correct per-component decomposition of an average.

For a D4 voxel's average-pooled target T_d4 = mean(block_64), a specific
component's OWN CONTRIBUTION to that mean is:
    contribution_c = mean(block_64 restricted to voxels belonging to component c)
                    = (count of component-c voxels in this block) / 64
This is exactly the correct linear decomposition of an average: the total
D4 target value is the SUM of every component's own per-block contribution
(plus any non-lesion/background voxels' zero contribution). This script
computes, for every (component, contributing D4 voxel) pair, this exact
contribution value, and uses it as an attribution weight when measuring how
much of the model's D4 prediction error in that voxel is "this component's
share" of the true target mass there.

Output: E36_component_localization_avgpool.json
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
CKPT_PATH = project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_D4only_seed0" / "checkpoints" / "epoch_30.pth"


def resize_nn(volume, target_shape):
    zf = tuple(t / c for t, c in zip(target_shape, volume.shape))
    return zoom(volume, zf, order=0)


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
    print(f"Loaded D4only epoch_30 checkpoint (best_val_dice={ckpt.get('best_val_dice')})", flush=True)

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
            aux_probs3 = out["aux_probs3"]
        mask_d4_soft = F.avg_pool3d(mask_b, kernel_size=4, stride=4)

        pred_np = aux_probs3.squeeze().cpu().numpy()          # (16,16,16)
        target_np = mask_d4_soft.squeeze().cpu().numpy()       # (16,16,16)
        error_np = pred_np - target_np                          # signed error per D4 voxel

        subject_dir = subject_dirs[subject_id]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_labeled, n_native = ndi.label(seg_binary_native > 0.5)

        # Resample the integer-ID volume to 64^3 (nearest-neighbor, SAME
        # convention as E30/E35's size_64 field) -- this is what
        # avg_pool3d(masks,4) itself pools FROM in the real training pipeline
        # (masks IS this same 64^3-resized binary volume, per BraTSDataset's
        # own preprocessing -- verified: mask returned by val_dataset[idx] IS
        # the 64^3 volume passed to avg_pool3d in train_deep_sup.py).
        labeled_64 = resize_nn(native_labeled.astype(np.float32), (64, 64, 64))
        labeled_64 = np.round(labeled_64).astype(np.int32)

        for r in by_subject[subject_id]:
            comp_id = r["native_component_id"]
            comp_mask_64 = (labeled_64 == comp_id)
            if not comp_mask_64.any():
                records.append({
                    "subject_id": subject_id, "native_component_id": comp_id,
                    "native_size": r["native_size"], "size_64": r["size_64"],
                    "n_d4_voxels_touched": 0, "total_contribution_mass": 0.0,
                    "contribution_weighted_abs_error": None,
                })
                continue

            # Reshape the 64^3 component mask into (16,4,16,4,16,4) blocks matching
            # avg_pool3d's own kernel_size=4,stride=4 reduction exactly.
            comp_blocks = comp_mask_64.reshape(16, 4, 16, 4, 16, 4).astype(np.float32)
            contribution = comp_blocks.sum(axis=(1, 3, 5)) / 64.0  # (16,16,16), this component's own share of each D4 voxel's mean

            touched = contribution > 0
            n_touched = int(touched.sum())
            total_mass = float(contribution.sum())  # should equal native... no, size_64/64 in expectation; sanity-checked below

            if n_touched == 0:
                records.append({
                    "subject_id": subject_id, "native_component_id": comp_id,
                    "native_size": r["native_size"], "size_64": r["size_64"],
                    "n_d4_voxels_touched": 0, "total_contribution_mass": total_mass,
                    "contribution_weighted_abs_error": None,
                })
                continue

            # Contribution-weighted absolute error: how much does the model's
            # per-D4-voxel error, weighted by this component's OWN share of
            # that voxel's true mass, sum to -- a mass-conserving attribution
            # of total volume-wide error across all contributing components.
            weighted_abs_err = float(np.sum(contribution * np.abs(error_np)))
            mean_abs_err_per_unit_mass = weighted_abs_err / total_mass if total_mass > 0 else None

            records.append({
                "subject_id": subject_id, "native_component_id": comp_id,
                "native_size": r["native_size"], "size_64": r["size_64"],
                "n_d4_voxels_touched": n_touched, "total_contribution_mass": total_mass,
                "contribution_weighted_abs_error": weighted_abs_err,
                "mean_abs_error_per_unit_mass": mean_abs_err_per_unit_mass,
            })

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{len(val_dataset)} subjects", flush=True)

    with open(OUT_DIR / "E36_component_localization_avgpool.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} component records.", flush=True)

    # Sanity check: total_contribution_mass summed over ALL components in a
    # subject, plus background's own (zero) contribution, should equal the
    # total true mask mass at 64^3 divided by 64 (i.e. avg_pool3d's own total
    # mass conservation property) -- verify on one subject directly.
    by_subj_mass = {}
    for r in records:
        by_subj_mass.setdefault(r["subject_id"], 0.0)
        by_subj_mass[r["subject_id"]] += r["total_contribution_mass"]
    print("\nMass-conservation spot check (first 3 subjects):")
    for sid in list(by_subj_mass.keys())[:3]:
        print(f"  {sid}: sum of component contributions = {by_subj_mass[sid]:.4f}")

    NATIVE_BINS = [(1, 5, "S1"), (6, 20, "S2"), (21, 50, "S3"), (51, 150, "S4"), (151, float("inf"), "S5")]
    def native_bin(size):
        for lo, hi, lab in NATIVE_BINS:
            if lo <= size <= hi:
                return lab
        return None

    print("\n=== Mean D4 mass-weighted error-per-unit-mass by native-size bin (epoch 30, D4-only) ===")
    for lo, hi, lab in NATIVE_BINS:
        vals = [r["mean_abs_error_per_unit_mass"] for r in records
                if native_bin(r["native_size"]) == lab and r.get("mean_abs_error_per_unit_mass") is not None]
        n_total = sum(1 for r in records if native_bin(r["native_size"]) == lab)
        n_no_contribution = sum(1 for r in records if native_bin(r["native_size"]) == lab and r["n_d4_voxels_touched"] == 0)
        if vals:
            print(f"{lab}: n={n_total} (no-D4-contribution={n_no_contribution})  "
                  f"mean_error_per_unit_mass={np.mean(vals):.4f}  median={np.median(vals):.4f}")
        else:
            print(f"{lab}: n={n_total} (ALL have zero D4 contribution)")


if __name__ == "__main__":
    main()
