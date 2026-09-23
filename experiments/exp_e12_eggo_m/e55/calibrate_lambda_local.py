"""
Phase E55: gradient-magnitude calibration of lambda_local, the weight
on the local pathway's own FocalTverskyLoss against the native-
resolution GT crop.

SAME discipline as every prior calibration in this project (E45's
calibrate_lambda_d8.py, E49's calibrate_lambda_frac.py, E52's
calibrate_lambda_cw_v2.py): measure loss VALUE at fresh initialization
(seed=0, .train() mode, real batch), then explicitly measure GRADIENT
magnitude too, per E34/E52's own hard-won lesson (value-only calibration
has TWICE caused a training collapse in this project's history).

This term is meant to genuinely reshape learning (the local pathway's
whole purpose is to recover small-lesion detail lost at 64^3), matching
E45's lambda_d8 comparable-weight treatment (target ratio ~1.0), NOT
E49's light-auxiliary lambda_frac treatment (target ratio 0.10).
"""
import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v9 import UNet3D_v9, native_to_norm, build_sampling_grid  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402

BASE = Path(__file__).parent
TARGET_RATIO = 1.0  # comparable-weight term, matching E45's lambda_d8 treatment


def crop_label(mask_native, centroid_native, crop_size, native_shape):
    """Extract the label crop via the IDENTICAL mapping used for the
    image crop -- mode='nearest' for a binary mask, not bilinear."""
    grid = build_sampling_grid(centroid_native, crop_size, native_shape, mask_native.device, mask_native.dtype)
    return F.grid_sample(mask_native, grid, mode="nearest", padding_mode="zeros", align_corners=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("E55: lambda_local calibration -- FRESH, randomly-initialized model")
    print("=" * 70)

    torch.manual_seed(0)
    model = UNet3D_v9(in_channels=1, out_channels=1).to(device)
    model.train()
    with torch.no_grad():
        model.fusion_gate.fill_(0.5)  # nonzero so the local path genuinely contributes to probs during calibration

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)

    train_loader, _ = create_brats_loaders(
        batch_size=4, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1, target_shape=(64, 64, 64), return_native=True,
    )
    images, masks, subject_ids, native_images = next(iter(train_loader))
    images, masks, native_images = images.to(device), masks.to(device), native_images.to(device)

    # We ALSO need the native-resolution GT mask (return_native only
    # returns the image, per Dataset/brats_dataset.py's own docstring --
    # load native seg ourselves here, matching train_e55's own convention).
    import nibabel as nib
    native_masks = []
    for sid in subject_ids:
        subject_dir = [d for d in train_loader.dataset.subject_dirs if Path(d).name == sid][0]
        seg_path = Path(subject_dir) / f"{sid}-seg.nii.gz"
        seg_native = (nib.load(str(seg_path)).get_fdata() > 0).astype("float32")
        native_masks.append(torch.from_numpy(seg_native))
    native_masks = torch.stack(native_masks).unsqueeze(1).to(device)  # (B,1,240,240,155)

    with torch.no_grad():
        outputs = model(images, native_x=native_images)
        probs = outputs["probs"]
        probs_coarse = outputs["probs_coarse"]
        centroid_native = outputs["centroid_native"]
        logit_local_crop = outputs["logit_local_crop"]

        focal_loss = focal_fn(probs, masks)
        evidential_loss = evidential_fn(outputs["alpha"], outputs["beta"], masks)
        L_seg = 0.5 * focal_loss + 0.5 * evidential_loss

        label_crop = crop_label(native_masks, centroid_native, model.CROP_SIZE, model.NATIVE_SHAPE)
        probs_local = torch.sigmoid(logit_local_crop)
        L_local = focal_fn(probs_local, label_crop)

    print(f"\nAt fresh init (seed=0, train() mode, real batch, batch_size={len(subject_ids)}):")
    print(f"  L_seg (full res, dec1)     = {L_seg.item():.6f}")
    print(f"  L_local (local pathway)    = {L_local.item():.6f}")

    lambda_local_value_matched = L_seg.item() / max(L_local.item(), 1e-8)
    print(f"\nValue-matched calibration: lambda_local = {lambda_local_value_matched:.4f}")
    print("(NOT what will be used for training -- reproduced for comparison only, per")
    print(" E34/E52's own established discipline of checking gradient magnitude before trusting this.)")

    print("\n=== Gradient-magnitude check ===")
    params = [p for p in model.parameters() if p.requires_grad]

    def grad_norm_of(loss):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        total = 0.0
        for p in params:
            if p.grad is not None:
                total += float((p.grad ** 2).sum().item())
        return total ** 0.5

    outputs = model(images, native_x=native_images)
    probs = outputs["probs"]
    focal_loss = focal_fn(probs, masks)
    evidential_loss = evidential_fn(outputs["alpha"], outputs["beta"], masks)
    L_seg_grad = 0.5 * focal_loss + 0.5 * evidential_loss
    g_seg = grad_norm_of(L_seg_grad)

    outputs2 = model(images, native_x=native_images)
    probs_coarse2 = outputs2["probs_coarse"]
    centroid_native2 = outputs2["centroid_native"]
    logit_local_crop2 = outputs2["logit_local_crop"]
    label_crop2 = crop_label(native_masks, centroid_native2, model.CROP_SIZE, model.NATIVE_SHAPE)
    probs_local2 = torch.sigmoid(logit_local_crop2)
    L_local_grad = focal_fn(probs_local2, label_crop2)
    g_local_unweighted = grad_norm_of(L_local_grad)

    print(f"  |grad L_seg| (unweighted)                = {g_seg:.6f}")
    print(f"  |grad L_local| (unweighted, BEFORE lambda) = {g_local_unweighted:.6f}")
    raw_ratio = g_local_unweighted / max(g_seg, 1e-12)
    print(f"  Raw gradient ratio (g_local/g_seg, unweighted) = {raw_ratio:.4f}")

    effective_g_local_at_value_matched = lambda_local_value_matched * g_local_unweighted
    blowup_factor = effective_g_local_at_value_matched / max(g_seg, 1e-12)
    print(f"\n  If value-matched lambda_local={lambda_local_value_matched:.4f} were applied:")
    print(f"    effective |grad| ratio to |grad L_seg| = {blowup_factor:.4f}x")
    if blowup_factor > 1.5 or blowup_factor < 0.67:
        print(f"    >>> Would be REJECTED per E34/E52's own safeguard (outside [0.67x, 1.5x]). <<<")

    lambda_local_final = TARGET_RATIO * g_seg / max(g_local_unweighted, 1e-12)
    print(f"\nGradient-matched calibration (target ratio={TARGET_RATIO}): lambda_local = {lambda_local_final:.4f}")

    out = {
        "L_seg_init": L_seg.item(), "L_local_init": L_local.item(),
        "lambda_local_value_matched": lambda_local_value_matched,
        "grad_norm_L_seg": g_seg, "grad_norm_L_local_unweighted": g_local_unweighted,
        "raw_gradient_ratio": raw_ratio,
        "blowup_factor_at_value_matched_lambda": blowup_factor,
        "target_ratio": TARGET_RATIO,
        "final_lambda_local": lambda_local_final,
        "seed": 0,
    }
    with open(BASE / "E55_lambda_local_calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {BASE / 'E55_lambda_local_calibration.json'}")
    print(f"\n=== FINAL lambda_local (gradient-matched) = {lambda_local_final:.4f} ===")


if __name__ == "__main__":
    main()
