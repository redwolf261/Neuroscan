"""
Phase E36-B: component-level localization of D4's persistent loss floor.

NO TRAINING. NO NEW LOSS. Read-only analysis of the existing D4-only
checkpoint (epoch_30.pth, matching the "late training" regime where E36-A2
found D4's loss plateaus ~2.7x higher than main's).

QUESTION: is D4's persistently high residual loss (found in E36-A2: D4/main
loss ratio grows from ~0.7 at epoch 1 to ~2.7 by epoch 30, while D2's own
ratio only reaches ~1.5) concentrated on the SAME small-lesion population
this project's other work (E25-E35) already identified as the dominant
failure mode -- or is it a generic property unrelated to lesion size?

METHOD: the actual D4 training target is F.avg_pool3d(masks, kernel_size=4,
stride=4) -- a SOFT, fractional coverage target (verified directly: a 4x4x4
block of exact-1s becomes a single D4 voxel of value 0.125, NOT binarized --
this is NOT the same "exact target" E34/E35's audit used, which was a
nearest-neighbor-resampled binary mask). This script reuses that exact
training-time construction (F.avg_pool3d, identical to e25/train_deep_sup.py)
rather than a different resize convention, to stay faithful to what the
model was actually trained against.

For each NATIVE component (reusing the already-verified 749-component
roster from experiments/exp_e12_eggo_m/e35/E35_component_roster.json --
same identity/size fields, not recomputed), this script:
  1. Resamples the native component-ID volume to D4 (16^3) via nearest-
     neighbor (identical convention to E30/E35's own component-footprint
     definition) to get each component's D4 VOXEL FOOTPRINT (which D4
     voxels "belong" to this component, for attribution purposes only --
     NOT the soft training target itself).
  2. Restricted to that footprint, computes the mean absolute error between
     the model's D4 prediction (aux_probs3) and the true soft D4 target
     (avg_pool3d(masks,4)) -- a per-component error measure directly
     analogous to what the whole-image Tversky loss aggregates over.
  3. Also computes a component-level Tversky-style ratio (tp/fp/fn restricted
     to the component's footprint) for a second, more loss-faithful measure.

Output: E36_component_localization.json
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
    roster_by_key = {(r["subject_id"], r["native_component_id"]): r for r in roster}
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
            aux_probs3 = out["aux_probs3"]  # (1,1,16,16,16)
        mask_d4_soft = F.avg_pool3d(mask_b, kernel_size=4, stride=4)  # exact training target, verified above

        pred_np = aux_probs3.squeeze().cpu().numpy()
        target_np = mask_d4_soft.squeeze().cpu().numpy()

        subject_dir = subject_dirs[subject_id]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_labeled, n_native = ndi.label(seg_binary_native > 0.5)

        labeled_d4 = resize_nn(native_labeled.astype(np.float32), (16, 16, 16))
        labeled_d4 = np.round(labeled_d4).astype(np.int32)

        for r in by_subject[subject_id]:
            comp_id = r["native_component_id"]
            footprint = labeled_d4 == comp_id
            n_vox = int(footprint.sum())
            if n_vox == 0:
                records.append({
                    "subject_id": subject_id, "native_component_id": comp_id,
                    "native_size": r["native_size"], "size_64": r["size_64"],
                    "d4_footprint_voxels": 0, "mae": None, "tversky_component": None,
                })
                continue

            pred_here = pred_np[footprint]
            target_here = target_np[footprint]
            mae = float(np.mean(np.abs(pred_here - target_here)))

            tp = float((pred_here * target_here).sum())
            fp = float((pred_here * (1 - target_here)).sum())
            fn = float(((1 - pred_here) * target_here).sum())
            smooth = 1.0
            tversky = (tp + smooth) / (tp + 0.5 * fp + 0.5 * fn + smooth)

            records.append({
                "subject_id": subject_id, "native_component_id": comp_id,
                "native_size": r["native_size"], "size_64": r["size_64"],
                "d4_footprint_voxels": n_vox, "mae": mae, "tversky_component": tversky,
            })

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{len(val_dataset)} subjects", flush=True)

    with open(OUT_DIR / "E36_component_localization.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} component records.", flush=True)

    # Quick size-bin summary printed immediately
    NATIVE_BINS = [(1, 5, "S1"), (6, 20, "S2"), (21, 50, "S3"), (51, 150, "S4"), (151, float("inf"), "S5")]
    def native_bin(size):
        for lo, hi, lab in NATIVE_BINS:
            if lo <= size <= hi:
                return lab
        return None

    print("\n=== Mean D4 component MAE and Tversky by native-size bin (epoch 30, D4-only) ===")
    for lo, hi, lab in NATIVE_BINS:
        maes = [r["mae"] for r in records if native_bin(r["native_size"]) == lab and r["mae"] is not None]
        tvs = [r["tversky_component"] for r in records if native_bin(r["native_size"]) == lab and r["tversky_component"] is not None]
        n_total = sum(1 for r in records if native_bin(r["native_size"]) == lab)
        n_zero_footprint = sum(1 for r in records if native_bin(r["native_size"]) == lab and r["d4_footprint_voxels"] == 0)
        if maes:
            print(f"{lab}: n={n_total} (zero-footprint={n_zero_footprint})  mean_MAE={np.mean(maes):.4f}  mean_Tversky={np.mean(tvs):.4f}")
        else:
            print(f"{lab}: n={n_total} (ALL zero-footprint, no measurable D4 voxels)")


if __name__ == "__main__":
    main()
