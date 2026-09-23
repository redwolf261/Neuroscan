"""
Phase E25, Structural Pivot 1: calibration of lambda_ds3/lambda_ds2, the
auxiliary deep-supervision loss weights for UNet3D_v3's aux_head3/
aux_head2. Same discipline as every other constant in this project
(delta_d, tau_b, m_ij, etc.): derived from direct measurement on a
fresh, randomly-initialized model in .train() mode (never .eval() --
per E12e's own hard-won lesson about BatchNorm's ~68x train/eval
discrepancy at fresh initialization), not guessed.

Method: measure FocalTverskyLoss's own value at each of the three
resolutions (dec1/full, dec2/D-2, dec3/D-4) on the SAME real validation
batch, at a fresh model. Set lambda_ds3/lambda_ds2 so that, at
initialization, mu*L_aux3 and lambda_ds2*L_aux2 are the SAME ORDER OF
MAGNITUDE as L_seg itself (not dominating it, not negligible) --
matching the same "comparable initial magnitude" principle used
throughout this project's own calibration history (e.g. E24's
delta_d_w calibration targeting a comparable active-hinge rate, not an
arbitrary constant).
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Structural Pivot 1: lambda_ds3/lambda_ds2 calibration -- FRESH, randomly-initialized model")
    print("=" * 70)

    torch.manual_seed(0)  # SAME seed as A/C6-2/C6-3's own theta^(0) construction
    model = UNet3D_v3(in_channels=1, out_channels=1).to(device)
    model.train()  # CRITICAL, per E12e's own lesson -- never eval() on a fresh model

    train_loader, _ = create_brats_loaders(
        batch_size=2, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1,
    )
    images, masks, _ = next(iter(train_loader))
    images = images.to(device)
    masks = masks.to(device)

    focal_fn = FocalTverskyLoss()

    with torch.no_grad():
        outputs = model(images)
        probs = outputs["probs"]
        aux_probs3 = outputs["aux_probs3"]
        aux_probs2 = outputs["aux_probs2"]

        mask_d4 = F.avg_pool3d(masks, kernel_size=4, stride=4)
        mask_d2 = F.avg_pool3d(masks, kernel_size=2, stride=2)

        L_seg = focal_fn(probs, masks)
        L_aux3 = focal_fn(aux_probs3, mask_d4)
        L_aux2 = focal_fn(aux_probs2, mask_d2)

    print(f"\nAt fresh init (seed=0, train() mode, real batch):")
    print(f"  L_seg (full res, dec1)  = {L_seg.item():.6f}")
    print(f"  L_aux3 (D/4 res, dec3)  = {L_aux3.item():.6f}")
    print(f"  L_aux2 (D/2 res, dec2)  = {L_aux2.item():.6f}")

    # Target: lambda_ds3 * L_aux3 ~= L_seg, lambda_ds2 * L_aux2 ~= L_seg
    # -- comparable initial CONTRIBUTION to the total loss, matching the
    # project's own "comparable magnitude" calibration principle.
    lambda_ds3 = L_seg.item() / max(L_aux3.item(), 1e-8)
    lambda_ds2 = L_seg.item() / max(L_aux2.item(), 1e-8)

    print(f"\nCalibrated (targeting lambda*L_aux ~= L_seg at init):")
    print(f"  lambda_ds3 = {lambda_ds3:.4f}")
    print(f"  lambda_ds2 = {lambda_ds2:.4f}")

    # Sanity cross-check across a few more batches, confirm stability
    print("\nCross-check across 5 more real batches:")
    ratios3, ratios2 = [], []
    it = iter(train_loader)
    for i in range(5):
        images, masks, _ = next(it)
        images = images.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            outputs = model(images)
            probs = outputs["probs"]
            aux_probs3 = outputs["aux_probs3"]
            aux_probs2 = outputs["aux_probs2"]
            mask_d4 = F.avg_pool3d(masks, kernel_size=4, stride=4)
            mask_d2 = F.avg_pool3d(masks, kernel_size=2, stride=2)
            L_seg_i = focal_fn(probs, masks).item()
            L_aux3_i = focal_fn(aux_probs3, mask_d4).item()
            L_aux2_i = focal_fn(aux_probs2, mask_d2).item()
            ratios3.append(L_seg_i / max(L_aux3_i, 1e-8))
            ratios2.append(L_seg_i / max(L_aux2_i, 1e-8))
        print(f"  batch {i}: L_seg={L_seg_i:.4f} L_aux3={L_aux3_i:.4f} L_aux2={L_aux2_i:.4f} "
              f"ratio3={ratios3[-1]:.3f} ratio2={ratios2[-1]:.3f}")

    import numpy as np
    print(f"\nmean ratio3 across 6 batches (incl. first): {np.mean([lambda_ds3]+ratios3):.4f}")
    print(f"mean ratio2 across 6 batches (incl. first): {np.mean([lambda_ds2]+ratios2):.4f}")

    import json
    out_dir = Path(__file__).parent / "e25b_results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "deep_supervision_calibration.json", "w") as f:
        json.dump({
            "lambda_ds3": lambda_ds3, "lambda_ds2": lambda_ds2,
            "L_seg_init": L_seg.item(), "L_aux3_init": L_aux3.item(), "L_aux2_init": L_aux2.item(),
            "cross_check_ratios3": ratios3, "cross_check_ratios2": ratios2,
            "seed": 0,
        }, f, indent=2)
    print(f"\nSaved to {out_dir / 'deep_supervision_calibration.json'}")


if __name__ == "__main__":
    main()
