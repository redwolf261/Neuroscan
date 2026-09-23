"""
Phase E35 (prompt-labeled "E34-A"): reconstruct the native-component roster
for the Scale-Conditioned Context Supervision feasibility audit.

NO TRAINING. Inference-only against EXISTING checkpoints.

Provenance decisions made before writing this script (verified, not assumed):

1. E27's component table (experiments/exp_e12_eggo_m/e27/E27_component_full_table.json,
   363 rows) labels GT components on the ALREADY-RESIZED 64^3 mask. E30/E31/E32
   instead label components in NATIVE resolution (749 rows for the same 125
   subjects) and track how each native component's footprint survives
   resampling. These two tables use INCOMPATIBLE component-identity spaces
   (native components can merge or vanish entirely under the 64^3 resize, so
   E27's component_id numbering does not correspond 1:1 with native_component_id).
   For this audit -- which explicitly needs native_size, size_64, and
   per-component survival across scales -- the native-labeled convention
   (E30/E31/E32) is the only usable base. E27's table is NOT reused here.

2. E32's existing "component_dice_64" / "detected_64" fields were computed
   against experiments/exp_e12_eggo_m/e29/resolution_runs/A64_seed0/checkpoints/best.pth
   (best_val_dice=0.9038), NOT experiments/exp_e12_eggo_m/e24/gate6_runs/A_baseline_seed0
   (best_val_dice=0.9063, the canonical "A" referenced throughout the project's
   master report and reused by E27 for its own condition-A numbers). The two
   checkpoints have identical architecture/param shapes but verified NON-identical
   weights. E32's field is therefore NOT directly comparable to what this script
   produces for "A". This script recomputes detection/Dice for A (and D4, D2)
   fresh, from the canonical e24/A_baseline_seed0 checkpoint, at native-component
   granularity, so all three conditions share one consistent, honest baseline.
   E32's original field is left untouched in its own file; it is not imported here.

3. D4 / D2 checkpoints: experiments/exp_e12_eggo_m/e25/deep_sup_runs/
   DeepSup_D4only_seed0 and DeepSup_D2only_seed0 (same ones E27 used).

Output: E35_component_roster.json -- one row per NATIVE GT component (749
expected, matching E30/E31/E32), with:
    subject_id, subject_idx, native_component_id, native_size
    size_64 (native component's footprint at 64^3, nearest-neighbor resample
             of the native integer-ID volume -- identical convention to
             E30/E32; cross-checked against E30's own size_alpha_1.0 field)
    alpha_c, censored                      (joined from E31, same key)
    baseline_component_dice_A, detected_by_A   (recomputed here, canonical ckpt)
    detected_by_D4, dice_D4
    detected_by_D2, dice_D2

Matching rule for detection/Dice: identical to E27/E32's own convention --
union of all predicted connected components overlapping the GT component's
footprint at 64^3; detected <=> that union is nonempty.
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
from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from Dataset.brats_dataset import BraTSDataset  # noqa: E402

OUT_DIR = Path(__file__).parent
E30_TABLE = OUT_DIR.parent / "e30" / "E30_component_survival_table.json"
E31_TABLE = OUT_DIR.parent / "e31" / "E31_alpha_c_table.json"

CKPTS = {
    "A": (project_root / "experiments" / "exp_e12_eggo_m" / "e24" / "gate6_runs" / "A_baseline_seed0" / "checkpoints" / "best.pth", UNet3D_v2),
    "D4": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_D4only_seed0" / "checkpoints" / "best.pth", UNet3D_v3),
    "D2": (project_root / "experiments" / "exp_e12_eggo_m" / "e25" / "deep_sup_runs" / "DeepSup_D2only_seed0" / "checkpoints" / "best.pth", UNet3D_v3),
}


def resize_nn(volume, target_shape):
    zoom_factors = tuple(t / c for t, c in zip(target_shape, volume.shape))
    return zoom(volume, zoom_factors, order=0)


def load_model(ckpt_path, cls, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = cls(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("epoch"), ckpt.get("best_val_dice")


def component_dice_at_64(comp_mask_64, pred_labeled):
    gt_size_64 = int(comp_mask_64.sum())
    if gt_size_64 == 0:
        return False, 0.0, gt_size_64
    overlap_ids = set(pred_labeled[comp_mask_64].flatten().tolist()) - {0}
    if not overlap_ids:
        return False, 0.0, gt_size_64
    pred_region = np.isin(pred_labeled, list(overlap_ids))
    inter = int((comp_mask_64 & pred_region).sum())
    pred_size = int(pred_region.sum())
    dice = 2 * inter / (gt_size_64 + pred_size) if (gt_size_64 + pred_size) > 0 else 1.0
    return True, dice, gt_size_64


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BraTSDataset(
        root_dir=str(project_root / "Dataset" / "Training"),
        split="val", val_split=0.1, target_shape=(64, 64, 64), normalize=True,
    )
    n_subjects = len(val_dataset)
    print(f"Full validation set: {n_subjects} subjects")

    print("Loading checkpoints (canonical A = e24/A_baseline_seed0, NOT e29/A64_seed0):")
    models = {}
    for name, (path, cls) in CKPTS.items():
        m, ep, bvd = load_model(path, cls, device)
        models[name] = m
        print(f"  {name}: {path}  epoch={ep} best_val_dice={bvd}")

    # Load E31's alpha_c table for joining (verify key uniqueness before trusting the join)
    e31_rows = json.load(open(E31_TABLE))
    e31_key = {(r["subject_idx"], r["native_component_id"]): r for r in e31_rows}
    assert len(e31_key) == len(e31_rows), "E31_alpha_c_table.json has duplicate (subject_idx, native_component_id) keys -- join would be unsafe"
    print(f"E31 alpha_c table loaded: {len(e31_rows)} rows, keys unique, verified.")

    # Load E30's survival table for a size_64 cross-check (size_alpha_1.0 == our size_64)
    e30_rows = json.load(open(E30_TABLE))
    e30_key = {(r["subject_idx"], r["native_component_id"]): r for r in e30_rows}
    assert len(e30_key) == len(e30_rows), "E30_component_survival_table.json has duplicate keys"
    print(f"E30 survival table loaded: {len(e30_rows)} rows, keys unique, verified.")

    records = []
    size64_mismatches = 0

    for subject_idx in range(n_subjects):
        image, mask, subject_id = val_dataset[subject_idx]
        image_b = image.unsqueeze(0).to(device)

        subject_dir = val_dataset.subject_dirs[subject_idx]
        seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_binary_native = (seg_data > 0).astype(np.float32)

        native_labeled, n_native = ndi.label(seg_binary_native > 0.5)
        if n_native == 0:
            continue

        labeled_64 = resize_nn(native_labeled.astype(np.float32), (64, 64, 64))
        labeled_64 = np.round(labeled_64).astype(np.int32)

        pred_labeled = {}
        with torch.no_grad():
            for name, model in models.items():
                out = model(image_b)
                probs = out["probs"] if isinstance(out, dict) else out
                pred_bin = (probs >= 0.5).float().squeeze(0).squeeze(0).cpu().numpy()
                pred_labeled[name], _ = ndi.label(pred_bin > 0.5)

        for comp_id in range(1, n_native + 1):
            key = (subject_idx, comp_id)
            comp_mask_64 = labeled_64 == comp_id

            detected_A, dice_A, size_64 = component_dice_at_64(comp_mask_64, pred_labeled["A"])
            detected_D4, dice_D4, _ = component_dice_at_64(comp_mask_64, pred_labeled["D4"])
            detected_D2, dice_D2, _ = component_dice_at_64(comp_mask_64, pred_labeled["D2"])

            e31r = e31_key.get(key)
            e30r = e30_key.get(key)
            native_size_e30 = e30r["native_size"] if e30r else None
            size64_e30 = e30r["size_alpha_1.0"] if e30r else None
            if size64_e30 is not None and size64_e30 != size_64:
                size64_mismatches += 1

            row = {
                "subject_id": subject_id,
                "subject_idx": subject_idx,
                "native_component_id": comp_id,
                "native_size": native_size_e30,
                "size_64": size_64,
                "size_64_e30_crosscheck": size64_e30,
                "alpha_c": e31r["alpha_c"] if e31r else None,
                "censored": e31r["censored"] if e31r else None,
                "detected_by_A": detected_A,
                "component_dice_A": dice_A,
                "detected_by_D4": detected_D4,
                "component_dice_D4": dice_D4,
                "detected_by_D2": detected_D2,
                "component_dice_D2": dice_D2,
            }
            records.append(row)

        if (subject_idx + 1) % 25 == 0:
            print(f"  processed {subject_idx+1}/{n_subjects} subjects")

    print(f"\nTotal native-space GT components: {len(records)} (expected 749, matching E30/E31/E32)")
    print(f"size_64 vs E30's size_alpha_1.0 mismatches: {size64_mismatches} / {len(records)}")
    if size64_mismatches > 0:
        print("  WARNING: mismatches found -- resize convention may differ from E30. Investigate before trusting size_64.")

    missing_alpha_c = sum(1 for r in records if r["alpha_c"] is None and r["censored"] is None)
    if missing_alpha_c > 0:
        print(f"  WARNING: {missing_alpha_c} components failed to join against E31 (no matching key) -- investigate join integrity.")

    with open(OUT_DIR / "E35_component_roster.json", "w") as f:
        json.dump(records, f)
    print(f"Saved to {OUT_DIR / 'E35_component_roster.json'}")


if __name__ == "__main__":
    main()
