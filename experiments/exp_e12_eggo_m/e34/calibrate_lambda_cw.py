"""
Phase E34: calibrate lambda_cw, matching E25b's own pre-training
calibration convention (e25b_calibrate_deep_supervision.py): measure loss
magnitude on a FRESH-INIT model in .train() mode (never .eval(), per E12e's
own established lesson) over several real training batches, targeting
lambda*L_cw ~ L_seg at initialization. Uses the REAL training dataloader
and REAL native component cache (not synthetic data), for both the size
and alpha_c conditions (checked separately in case their typical magnitude
differs).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import scipy.ndimage as ndi
from scipy.ndimage import zoom
import nibabel as nib
import json

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e34"))

from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402
from component_weighted_loss import WeightLookup, compute_component_weighted_loss, precompute_labeled_64_cache  # noqa: E402

BASE = Path(__file__).parent
CONFIG_PATH = project_root / "configs" / "brats.yaml"


def resize_nn(volume, target_shape):
    current_shape = volume.shape
    zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
    return zoom(volume, zoom_factors, order=0)


def main():
    import yaml
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    dataset_root = project_root / config["dataset"]["root_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = UNet3D_v2(in_channels=1, out_channels=1).to(device)
    model.train()  # NEVER .eval() for calibration -- E12e's own lesson

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)
    boundary_criterion = nn.BCEWithLogitsLoss()

    train_loader, _ = create_brats_loaders(
        batch_size=config["training"]["batch_size"], num_workers=0,
        root_dir=str(dataset_root), val_split=config["dataset"]["val_split"],
        target_shape=(64, 64, 64),
    )

    # Load weight tables and build native cache (same as ComponentWeightedExperiment)
    def load_weights(path):
        with open(path) as f:
            raw = json.load(f)
        parsed = {}
        for key, w in raw.items():
            subj, comp_id = key.rsplit("|", 1)
            parsed[(subj, int(comp_id))] = w
        return WeightLookup(parsed)

    weight_lookup_alpha = load_weights(BASE / "E34_weight_table_alpha_c.json")
    weight_lookup_size = load_weights(BASE / "E34_weight_table_size.json")

    # BUG CAUGHT: originally precomputed a native-component cache for only
    # the first 32 subject_dirs (in fixed dataloader-listing order), then
    # skipped any batch whose randomly-shuffled subject_ids weren't a
    # complete subset of those 32 -- with batch_size=8 drawn uniformly from
    # 1126 subjects, the chance all 8 land in a fixed 32-subject slice is
    # ~(32/1126)^8, effectively zero, so every batch was skipped and
    # n_batches_tested stayed 0, silently returning the fallback default
    # (1.0) rather than a real calibration. FIX: cache subjects ON DEMAND
    # as they're encountered in real sampled batches, not a fixed prefix.
    native_cache = {}

    def get_native(subject_id, subject_dir):
        if subject_id not in native_cache:
            seg_path = Path(subject_dir) / f"{subject_id}-seg.nii.gz"
            seg_data = nib.load(str(seg_path)).get_fdata().astype(np.float32)
            seg_binary = (seg_data > 0).astype(np.float32)
            native_labeled, _ = ndi.label(seg_binary > 0.5)
            native_cache[subject_id] = (native_labeled, seg_data.shape)
        return native_cache[subject_id]

    subject_dir_by_id = {Path(d).name: d for d in train_loader.dataset.subject_dirs}

    ratios_alpha, ratios_size = [], []
    n_batches_tested = 0
    for images, masks, subject_ids in train_loader:
        for sid in subject_ids:
            get_native(sid, subject_dir_by_id[sid])
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        probs, alpha_out, beta_out = outputs["probs"], outputs["alpha"], outputs["beta"]
        seg_loss = 0.5 * focal_fn(probs, masks) + 0.5 * evidential_fn(alpha_out, beta_out, masks)

        labeled_64_cache = precompute_labeled_64_cache(native_cache, resize_nn)
        cw_loss_alpha, _ = compute_component_weighted_loss(probs, masks, list(subject_ids), weight_lookup_alpha, labeled_64_cache, device)
        cw_loss_size, _ = compute_component_weighted_loss(probs, masks, list(subject_ids), weight_lookup_size, labeled_64_cache, device)

        if cw_loss_alpha.item() > 1e-8:
            ratios_alpha.append(seg_loss.item() / cw_loss_alpha.item())
        if cw_loss_size.item() > 1e-8:
            ratios_size.append(seg_loss.item() / cw_loss_size.item())

        print(f"  batch: seg_loss={seg_loss.item():.4f}  cw_loss_alpha={cw_loss_alpha.item():.4f}  cw_loss_size={cw_loss_size.item():.4f}")
        n_batches_tested += 1
        if n_batches_tested >= 6:
            break

    lambda_alpha = float(np.median(ratios_alpha)) if ratios_alpha else 1.0
    lambda_size = float(np.median(ratios_size)) if ratios_size else 1.0
    print(f"\nCalibrated lambda_cw (alpha_c condition, target seg_loss~cw_loss at init): {lambda_alpha:.4f}")
    print(f"Calibrated lambda_cw (size condition, target seg_loss~cw_loss at init): {lambda_size:.4f}")
    print(f"Individual ratios (alpha): {ratios_alpha}")
    print(f"Individual ratios (size): {ratios_size}")

    # Use a SINGLE lambda_cw for BOTH S and R conditions (the smaller of
    # the two medians, rounded to 1 decimal for a clean, pre-declared
    # value) -- using the SAME lambda for both conditions is required for
    # a fair comparison (the whole point of Section 7's decisive test is
    # comparing R against S under MATCHED conditions, which includes a
    # matched lambda_cw, not just matched weight moments).
    lambda_cw_final = round(min(lambda_alpha, lambda_size), 1)
    print(f"\nFINAL lambda_cw (used for BOTH S and R, matched): {lambda_cw_final}")

    with open(BASE / "E34_lambda_cw_calibration.json", "w") as f:
        json.dump({
            "lambda_alpha_raw": lambda_alpha, "lambda_size_raw": lambda_size,
            "lambda_cw_final": lambda_cw_final, "n_batches": n_batches_tested,
        }, f, indent=2)
    print(f"Saved to {BASE / 'E34_lambda_cw_calibration.json'}")


if __name__ == "__main__":
    main()
