"""
Phase E45: calibration of lambda_d8, the new bottleneck (8^3) auxiliary
deep-supervision loss weight for UNet3D_v4's aux_head_d8.

SAME discipline as E25b's own lambda_ds3/lambda_ds2 calibration (reused
method, not reinvented): measure FocalTverskyLoss's value at fresh
initialization (seed=0, .train() mode -- NEVER .eval() on a fresh model,
per E12e's own hard-won BatchNorm lesson), set lambda so the new term's
CONTRIBUTION matches L_seg's own magnitude at init.

CRITICAL ADDITION, per E34's own hard-won lesson (a loss term calibrated by
matching VALUE magnitude alone caused a ~75x gradient-magnitude blowup and a
full training collapse in E34's own history): this script explicitly checks
GRADIENT magnitude too, not just loss VALUE magnitude, before trusting the
calibrated lambda_d8. The bottleneck's own loss is computed over only 8^3=512
cells (vs dec1's 64^3=262144), so per E34's own diagnosed mechanism, its
gradient could plausibly be disproportionately steep relative to its
seemingly-reasonable loss VALUE -- checked directly here, not assumed safe.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v4 import UNet3D_v4  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("E45: lambda_d8 calibration -- FRESH, randomly-initialized model")
    print("=" * 70)

    torch.manual_seed(0)  # SAME seed as every prior condition's own theta^(0) construction
    model = UNet3D_v4(in_channels=1, out_channels=1).to(device)
    model.train()  # CRITICAL, per E12e's own lesson

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
        aux_probs_d8 = outputs["aux_probs_d8"]

        mask_d8 = F.avg_pool3d(masks, kernel_size=8, stride=8)

        L_seg = focal_fn(probs, masks)
        L_d8 = focal_fn(aux_probs_d8, mask_d8)

    print(f"\nAt fresh init (seed=0, train() mode, real batch):")
    print(f"  L_seg (full res, dec1) = {L_seg.item():.6f}")
    print(f"  L_d8 (D/8 res, bottleneck) = {L_d8.item():.6f}")

    lambda_d8_value_matched = L_seg.item() / max(L_d8.item(), 1e-8)
    print(f"\nValue-matched calibration: lambda_d8 = {lambda_d8_value_matched:.4f}")

    # ================= GRADIENT-MAGNITUDE CHECK, per E34's own lesson =================
    print("\n=== Gradient-magnitude check (E34's own mandatory safeguard) ===")
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
    aux_probs_d8 = outputs["aux_probs_d8"]
    mask_d8 = F.avg_pool3d(masks, kernel_size=8, stride=8)

    L_seg_grad = focal_fn(probs, masks)
    g_seg = grad_norm_of(L_seg_grad)

    outputs2 = model(images)  # fresh forward, since backward() consumed the graph
    aux_probs_d8_2 = outputs2["aux_probs_d8"]
    L_d8_grad = focal_fn(aux_probs_d8_2, mask_d8)
    g_d8_unweighted = grad_norm_of(L_d8_grad)

    print(f"  |grad L_seg| (unweighted) = {g_seg:.6f}")
    print(f"  |grad L_d8| (unweighted, BEFORE lambda applied) = {g_d8_unweighted:.6f}")
    print(f"  Raw gradient ratio (g_d8/g_seg, unweighted) = {g_d8_unweighted/max(g_seg,1e-12):.4f}")

    # If lambda_d8_value_matched were applied, the EFFECTIVE gradient contribution
    # would be lambda_d8_value_matched * g_d8_unweighted -- check this against g_seg directly.
    effective_g_d8 = lambda_d8_value_matched * g_d8_unweighted
    print(f"\n  If lambda_d8={lambda_d8_value_matched:.4f} is applied:")
    print(f"    effective |grad (lambda_d8 * L_d8)| = {effective_g_d8:.6f}")
    print(f"    ratio to |grad L_seg| = {effective_g_d8/max(g_seg,1e-12):.4f}")
    print(f"    (a ratio near 1.0 is safe/comparable; a ratio >>1 or <<1 means value-matching "
          f"produced a gradient-magnitude mismatch, per E34's own diagnosed failure mode)")

    ratio = effective_g_d8 / max(g_seg, 1e-12)
    if ratio > 1.5 or ratio < 0.67:
        print(f"\n  WARNING: gradient ratio {ratio:.2f} is far from 1.0 -- value-matched calibration "
              f"is NOT safe to use as-is. Recalibrating lambda_d8 by GRADIENT magnitude instead.")
        lambda_d8_gradient_matched = g_seg / max(g_d8_unweighted, 1e-12)
        print(f"  Gradient-matched lambda_d8 = {lambda_d8_gradient_matched:.4f}")
        final_lambda_d8 = lambda_d8_gradient_matched
    else:
        print(f"\n  Gradient ratio is within safe bounds -- value-matched calibration is trustworthy.")
        final_lambda_d8 = lambda_d8_value_matched

    import json
    out_dir = Path(__file__).parent
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "E45_lambda_d8_calibration.json", "w") as f:
        json.dump({
            "L_seg_init": L_seg.item(), "L_d8_init": L_d8.item(),
            "lambda_d8_value_matched": lambda_d8_value_matched,
            "grad_norm_L_seg": g_seg, "grad_norm_L_d8_unweighted": g_d8_unweighted,
            "gradient_ratio_at_value_matched_lambda": ratio,
            "final_lambda_d8": final_lambda_d8,
            "seed": 0,
        }, f, indent=2)
    print(f"\nSaved to {out_dir / 'E45_lambda_d8_calibration.json'}")
    print(f"\n=== FINAL lambda_d8 = {final_lambda_d8:.4f} ===")


if __name__ == "__main__":
    main()
