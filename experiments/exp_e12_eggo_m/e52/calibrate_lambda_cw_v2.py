"""
Phase E52: gradient-magnitude calibration of lambda_cw for the [R]
(alpha_c) component-weighted loss condition -- the recalibration E34
itself never received.

SAME discipline as E45's calibrate_lambda_d8.py and E49's
calibrate_lambda_frac.py (reused method, not reinvented): measure loss
VALUE at fresh initialization (seed=0, .train() mode, real batch --
never .eval(), per E12e's own established lesson), then explicitly
measure GRADIENT magnitude too, per E34's own hard-won lesson (a loss
term calibrated by matching VALUE magnitude alone caused this project's
worst training collapse to date: E34's own [S]/[R] conditions, 0.71-0.74
Dice vs 0.9063 baseline).

Unlike E49's lambda_frac (a deliberately LIGHT auxiliary signal, target
gradient ratio 0.10), this term is a full additional segmentation
weighting mechanism (per E34's own original design: L_total = L_seg +
mu*L_boundary + lambda_cw*L_component_weighted, no indication it was
meant as a light side-signal) -- so this calibration targets a
COMPARABLE gradient ratio, matching E45's own lambda_d8 treatment
(target ~1.0), not E49's light-auxiliary target.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import scipy.ndimage as ndi
from scipy.ndimage import zoom
import nibabel as nib

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments" / "exp_e12_eggo_m" / "e34"))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss, EvidentialBetaLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402
from component_weighted_loss import WeightLookup, compute_component_weighted_loss, precompute_labeled_64_cache  # noqa: E402

BASE = Path(__file__).parent
E34_DIR = project_root / "experiments" / "exp_e12_eggo_m" / "e34"
CONFIG_PATH = project_root / "configs" / "brats.yaml"
TARGET_RATIO = 1.0  # comparable-weight term, matching E45's lambda_d8 treatment


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
    model = UNet3D_v3(in_channels=1, out_channels=1).to(device)
    model.train()  # NEVER .eval() for calibration -- E12e's own lesson

    focal_fn = FocalTverskyLoss()
    evidential_fn = EvidentialBetaLoss(weight=0.5)
    boundary_criterion = nn.BCEWithLogitsLoss()
    focal_weight, evidential_weight, mu = 0.5, 0.5, 0.1

    train_loader, _ = create_brats_loaders(
        batch_size=config["training"]["batch_size"], num_workers=0,
        root_dir=str(dataset_root), val_split=config["dataset"]["val_split"],
    )

    # Load the [R] (alpha_c) weight table, REUSED unchanged from E34.
    def load_weights(path):
        with open(path) as f:
            raw = json.load(f)
        parsed = {}
        for key, w in raw.items():
            subj, comp_id = key.rsplit("|", 1)
            parsed[(subj, int(comp_id))] = w
        return WeightLookup(parsed)

    weight_lookup_alpha = load_weights(E34_DIR / "E34_weight_table_alpha_c.json")

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

    images, masks, subject_ids = next(iter(train_loader))
    images = images.to(device)
    masks = masks.to(device)

    # Build labeled_64_cache ON DEMAND for exactly this batch's subjects
    # (avoids E34's original fixed-32-subject-prefix bug by construction --
    # this calibration only ever touches the subjects actually sampled).
    native_labeled_cache = {}
    for sid in subject_ids:
        native_labeled_cache[sid] = get_native(sid, subject_dir_by_id[sid])
    labeled_64_cache = precompute_labeled_64_cache(native_labeled_cache, resize_nn)

    with torch.no_grad():
        outputs = model(images)
        probs = outputs["probs"]

        focal_loss = focal_fn(probs, masks)
        evidential_loss = evidential_fn(outputs["alpha"], outputs["beta"], masks)
        L_seg = focal_weight * focal_loss + evidential_weight * evidential_loss

        L_cw, diag = compute_component_weighted_loss(
            probs, masks, subject_ids, weight_lookup_alpha, labeled_64_cache, device
        )

    print("=" * 70)
    print("E52: lambda_cw (alpha_c/[R] condition) calibration -- FRESH init")
    print("=" * 70)
    print(f"\nAt fresh init (seed=0, train() mode, real batch, batch_size={len(subject_ids)}):")
    print(f"  L_seg (FocalTversky+Evidential)  = {L_seg.item():.6f}")
    print(f"  L_cw (component-weighted, alpha_c) = {L_cw.item():.6f}")
    print(f"  n_components_seen={diag['n_components_seen']} n_components_matched={diag['n_components_matched']}")

    lambda_cw_value_matched = L_seg.item() / max(L_cw.item(), 1e-8)
    print(f"\nValue-matched calibration: lambda_cw = {lambda_cw_value_matched:.4f}")
    print("(This is the SAME calibration method E34 originally used -- reproduced here")
    print(" for direct comparison, NOT what will be used for training.)")

    print("\n=== Gradient-magnitude check (E34's own mandatory safeguard, applied retroactively) ===")
    params = [p for p in model.parameters() if p.requires_grad]

    def grad_norm_of(loss):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        total = 0.0
        for p in params:
            if p.grad is not None:
                total += float((p.grad ** 2).sum().item())
        return total ** 0.5

    outputs = model(images)
    probs = outputs["probs"]
    focal_loss = focal_fn(probs, masks)
    evidential_loss = evidential_fn(outputs["alpha"], outputs["beta"], masks)
    L_seg_grad = focal_weight * focal_loss + evidential_weight * evidential_loss
    g_seg = grad_norm_of(L_seg_grad)

    outputs2 = model(images)  # fresh forward, backward() consumed the graph
    probs2 = outputs2["probs"]
    L_cw_grad, _ = compute_component_weighted_loss(
        probs2, masks, subject_ids, weight_lookup_alpha, labeled_64_cache, device
    )
    g_cw_unweighted = grad_norm_of(L_cw_grad)

    print(f"  |grad L_seg| (unweighted)              = {g_seg:.6f}")
    print(f"  |grad L_cw|  (unweighted, BEFORE lambda) = {g_cw_unweighted:.6f}")
    raw_ratio = g_cw_unweighted / max(g_seg, 1e-12)
    print(f"  Raw gradient ratio (g_cw/g_seg, unweighted) = {raw_ratio:.4f}")

    effective_g_cw_at_value_matched = lambda_cw_value_matched * g_cw_unweighted
    blowup_factor = effective_g_cw_at_value_matched / max(g_seg, 1e-12)
    print(f"\n  If the ORIGINAL E34 value-matched lambda_cw={lambda_cw_value_matched:.4f} were applied:")
    print(f"    effective |grad (lambda_cw * L_cw)| = {effective_g_cw_at_value_matched:.6f}")
    print(f"    ratio to |grad L_seg| = {blowup_factor:.4f}x")
    if blowup_factor > 1.5 or blowup_factor < 0.67:
        print(f"    >>> This WOULD be rejected per E34's own safeguard (outside [0.67x, 1.5x]). <<<")
        print(f"    This is likely the direct explanation for E34's original training collapse.")

    lambda_cw_final = TARGET_RATIO * g_seg / max(g_cw_unweighted, 1e-12)
    print(f"\nGradient-matched calibration (target ratio={TARGET_RATIO}): lambda_cw = {lambda_cw_final:.4f}")

    out = {
        "condition": "R_alpha_c",
        "L_seg_init": L_seg.item(), "L_cw_init": L_cw.item(),
        "lambda_cw_value_matched": lambda_cw_value_matched,
        "grad_norm_L_seg": g_seg, "grad_norm_L_cw_unweighted": g_cw_unweighted,
        "raw_gradient_ratio": raw_ratio,
        "blowup_factor_at_value_matched_lambda": blowup_factor,
        "target_ratio": TARGET_RATIO,
        "final_lambda_cw": lambda_cw_final,
        "n_components_seen": diag["n_components_seen"],
        "n_components_matched": diag["n_components_matched"],
        "seed": 0,
    }
    with open(BASE / "E52_lambda_cw_calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {BASE / 'E52_lambda_cw_calibration.json'}")
    print(f"\n=== FINAL lambda_cw (gradient-matched) = {lambda_cw_final:.4f} ===")


if __name__ == "__main__":
    main()
