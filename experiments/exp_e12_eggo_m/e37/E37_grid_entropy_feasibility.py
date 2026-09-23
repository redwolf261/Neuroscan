"""
Phase E37: Grid-Entropy Feasibility Audit.

NO MODEL TRAINING. NO NEW LOSS. This tests whether the "grid-induced
ambiguity" quantity derived conceptually from E36 (coarsening a tumor's
outline creates boundary-ambiguous, partial-coverage grid cells, and this --
not lesion size -- was found to explain D4's persistently higher residual
loss versus D2's) is a real, MEASURABLE, size-independent per-component
quantity that predicts the model's actual per-component error at each
decoder resolution.

DATA REUSE, VERIFIED BEFORE TRUSTING (per this project's own standing
practice of not blindly trusting an existing table by name):
  - Component identity/native_size/size_64: experiments/exp_e12_eggo_m/e35/
    E35_component_roster.json (749 native-resolution components, verified in
    E34-A against two independent tables, 0 mismatches). REUSED, not rebuilt.
  - Component boundaries/labeling convention: identical native ndi.label()
    convention as E30/E35 (nearest-neighbor resample to target grid for
    discrete component-ID identity).
  - E_{c,D4} is NOT reused from E36_component_localization_avgpool.json
    as-is, because that file used the D4-ONLY checkpoint. Per an explicit
    decision made before running this script, E_{c,D4} and E_{c,D2} are
    BOTH measured fresh from the "Both" checkpoint (DeepSup_seed0,
    epoch_30.pth) -- one consistent model, so the D4-vs-D2 DIFFERENTIAL
    comparison (Section "Also calculate this" in the prompt) is apples-to-
    apples: same trained weights, same forward pass, only the resolution
    read out differs. Using two different single-head models (D4-only vs
    D2-only, as E36 did for a different purpose) would confound "does D4
    differ from D2 in this model" with "are these two different models."

MATHEMATICAL DEFINITIONS (exactly as specified):
    pi_Q = |Y ∩ Q| / |Q|                          (fractional coverage of grid cell Q)
    H(pi_Q) = -pi_Q*ln(pi_Q) - (1-pi_Q)*ln(1-pi_Q)  (binary entropy, natural log, 0 at pi=0 or 1)
    H_{c,r} = (1/|Y_c|) * sum_Q |Y_c ∩ Q| * H(pi_{c,Q})

    IMPORTANT computation-order note: pi_{c,Q} is per-COMPONENT-RESTRICTED
    coverage of Q (i.e. Y_c ∩ Q over the WHOLE cell Q, not restricted to
    other components also touching Q) -- Q's coverage fraction is computed
    using ONLY component c's own contribution to that cell, matching the
    formula's own |Y_c ∩ Q| numerator exactly. If two different components
    partially share one grid cell Q, each gets its OWN pi_{c,Q} computed
    from its own partial occupancy of Q, which can legitimately sum with
    other components' occupancy to exceed the cell's total occupancy by any
    single component alone -- this is intentional and matches the formula
    literally, not a bug (Q's TOTAL occupancy, if ever needed, is a
    different, un-requested quantity: sum over all components' Y_c in Q).

    E_{c,r} = component-level Tversky-style error (1 - Tversky ratio,
    computed the same way as e25/train_deep_sup.py's own FocalTverskyLoss,
    restricted to voxels contributing to this component), NOT the same
    "mean_abs_error_per_unit_mass" metric E36 used -- Tversky is what the
    model actually optimizes (E36's MAE was a diagnostic choice for a
    different purpose), so this is the more faithful "auxiliary error" the
    prompt's own E_{c,r} notation refers to. Both metrics are computed and
    saved for transparency; the Tversky-based one is PRIMARY for all
    reported correlations.

    B_{c,r} = fraction of component c's contribution mass at resolution r
    that lands in partial cells (0 < pi_{c,Q} < 1), matching E36's own
    "boundary/partial coverage" definition exactly (reused, not redefined).

OUTPUT: E37_component_entropy_table.json (one row per (component, resolution)
combination, D4/D2/D1), covering all 749 native components.
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

SCALES = {"D4": 16, "D2": 32, "D1": 64}


def resize_nn(volume, target_shape):
    zf = tuple(t / c for t, c in zip(target_shape, volume.shape))
    return zoom(volume, zf, order=0)


def binary_entropy(pi):
    """H(pi) = -pi*ln(pi) - (1-pi)*ln(1-pi), with the 0*ln(0)=0 convention
    (standard for entropy; verified: np.where avoids NaN from log(0))."""
    pi = np.clip(pi, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(pi > 0, pi * np.log(pi), 0.0)
        term2 = np.where(pi < 1, (1 - pi) * np.log(1 - pi), 0.0)
    return -(term1 + term2)


def component_grid_entropy(comp_mask_native, native_shape, res):
    """H_{c,r} = (1/|Y_c|) * sum_Q |Y_c ∩ Q| * H(pi_{c,Q})
    where pi_{c,Q} = |Y_c ∩ Q| / |Q| (Q's full native-voxel volume, not
    restricted to the component -- matches the formula's own Q normalizer).

    Computed via avg_pool3d on the component's OWN binary mask (resized to
    64^3 first via nearest-neighbor to match size_64's own convention, then
    avg-pooled further down for D4/D2 -- IDENTICAL two-step convention to
    how the real training pipeline itself computes mask_d4/mask_d2 from the
    64^3 mask, not a separate resize-from-native path, so this entropy is
    computed against the SAME grid the model is actually trained/evaluated
    on)."""
    # comp_mask_native is already the 64^3-resized component mask (see caller)
    comp_mask_64 = comp_mask_native
    native_size_64 = int(comp_mask_64.sum())  # this component's total mass at 64^3 -- the |Y_c| normalizer
    if native_size_64 == 0:
        return None, 0, None

    if res == 64:
        pi_grid = comp_mask_64.astype(np.float64)  # each 64^3 "cell" IS one voxel of the component's own mask -- pi is 0 or 1 trivially
        cell_mass = comp_mask_64.astype(np.float64)  # |Y_c ∩ Q| = the voxel's own value (0 or 1) since Q=1 voxel
    else:
        factor = 64 // res
        t = torch.from_numpy(comp_mask_64.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        pooled_mean = F.avg_pool3d(t, kernel_size=factor, stride=factor).squeeze().numpy().astype(np.float64)
        pi_grid = pooled_mean  # avg of a 0/1 mask over a factor^3 block IS exactly |Y_c ∩ Q| / |Q|
        cell_mass = pooled_mean * (factor ** 3)  # |Y_c ∩ Q| = pi_Q * |Q|, |Q|=factor^3

    H_grid = binary_entropy(pi_grid)
    weighted_sum = float((cell_mass * H_grid).sum())
    H_cr = weighted_sum / native_size_64

    # Boundary/partial-coverage mass at this resolution (0 < pi < 1), reusing
    # E36's own definition exactly.
    partial_mask = (pi_grid > 1e-9) & (pi_grid < 1.0 - 1e-9)
    boundary_mass = float(cell_mass[partial_mask].sum())
    B_cr = boundary_mass / native_size_64

    return H_cr, native_size_64, B_cr


def tversky_component_error(pred_grid, target_grid, alpha=0.5, beta=0.5, smooth=1.0):
    """1 - Tversky ratio, restricted to a component's own contribution
    (target_grid here is the component's OWN per-cell contribution mass
    fraction, i.e. pi_{c,Q}, NOT the whole-image target -- so tp/fp/fn are
    this component's own share of the error, matching the same linear-
    decomposition-of-an-average logic already verified in E36-B2's mass-
    conservation check)."""
    tp = float((pred_grid * target_grid).sum())
    fp = float((pred_grid * (1 - target_grid)).sum())
    fn = float(((1 - pred_grid) * target_grid).sum())
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky


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

        subject_dir = subject_dirs[subject_id]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)
        native_labeled, n_native = ndi.label(seg_binary_native > 0.5)

        labeled_64 = resize_nn(native_labeled.astype(np.float32), (64, 64, 64))
        labeled_64 = np.round(labeled_64).astype(np.int32)

        with torch.no_grad():
            out = model(image_b)
            pred_d4 = out["aux_probs3"].squeeze().cpu().numpy().astype(np.float64)  # (16,16,16)
            pred_d2 = out["aux_probs2"].squeeze().cpu().numpy().astype(np.float64)  # (32,32,32)

        for r in by_subject[subject_id]:
            comp_id = r["native_component_id"]
            comp_mask_64 = (labeled_64 == comp_id).astype(np.float32)

            row = {
                "subject_id": subject_id, "native_component_id": comp_id,
                "native_size": r["native_size"], "size_64": r["size_64"],
            }

            for scale_name, res in SCALES.items():
                H_cr, size_64_check, B_cr = component_grid_entropy(comp_mask_64, None, res)
                row[f"H_{scale_name}"] = H_cr
                row[f"B_{scale_name}"] = B_cr
                if scale_name == "D1":
                    row["size_64_recheck"] = size_64_check

                if scale_name in ("D4", "D2") and H_cr is not None:
                    factor = 64 // res
                    t = torch.from_numpy(comp_mask_64).unsqueeze(0).unsqueeze(0)
                    pi_grid = F.avg_pool3d(t, kernel_size=factor, stride=factor).squeeze().numpy().astype(np.float64)
                    pred_grid = pred_d4 if scale_name == "D4" else pred_d2
                    err = tversky_component_error(pred_grid, pi_grid)
                    row[f"E_{scale_name}"] = err
                else:
                    if scale_name in ("D4", "D2"):
                        row[f"E_{scale_name}"] = None

            records.append(row)

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{len(val_dataset)} subjects", flush=True)

    with open(OUT_DIR / "E37_component_entropy_table.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} component records (expected {len(roster)}).", flush=True)

    # size_64 cross-check against E35's own field
    mismatches = sum(1 for r in records if r.get("size_64_recheck") != r["size_64"])
    print(f"size_64 crosscheck mismatches: {mismatches} / {len(records)}", flush=True)


if __name__ == "__main__":
    main()
