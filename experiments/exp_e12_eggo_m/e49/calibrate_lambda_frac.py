"""
Phase E49: calibration of lambda_frac, the weight on frac_hat's own MSE
supervision loss (SizeProxyHead's prediction vs. real fractional
occupancy).

SAME discipline as E45's own calibrate_lambda_d8.py (reused method, not
reinvented): measure the loss's value at fresh initialization (seed=0,
.train() mode), then explicitly check GRADIENT magnitude too, not just
loss VALUE magnitude, before trusting a calibrated lambda -- per E34's
own hard-won lesson (a loss term calibrated by matching VALUE magnitude
alone caused a ~75x gradient-magnitude blowup and a full training
collapse in E34's own history).
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v6 import UNet3D_v6  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("E49: lambda_frac calibration -- FRESH, randomly-initialized model")
    print("=" * 70)

    torch.manual_seed(0)
    model = UNet3D_v6(in_channels=1, out_channels=1).to(device)
    model.train()

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
        frac_hat = outputs["frac_hat"]
        frac_target = masks.mean(dim=(1, 2, 3, 4))

        L_seg = focal_fn(probs, masks)
        L_frac = F.mse_loss(frac_hat, frac_target)

    print(f"\nAt fresh init (seed=0, train() mode, real batch):")
    print(f"  L_seg (full res, dec1)      = {L_seg.item():.6f}")
    print(f"  L_frac (size-proxy MSE)     = {L_frac.item():.6f}")

    lambda_frac_value_matched = L_seg.item() / max(L_frac.item(), 1e-8)
    print(f"\nValue-matched calibration: lambda_frac = {lambda_frac_value_matched:.4f}")

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
    L_seg_grad = focal_fn(probs, masks)
    g_seg = grad_norm_of(L_seg_grad)

    outputs2 = model(images)  # fresh forward, since backward() consumed the graph
    frac_hat_2 = outputs2["frac_hat"]
    frac_target_2 = masks.mean(dim=(1, 2, 3, 4))
    L_frac_grad = F.mse_loss(frac_hat_2, frac_target_2)
    g_frac_unweighted = grad_norm_of(L_frac_grad)

    print(f"  |grad L_seg| (unweighted)  = {g_seg:.6f}")
    print(f"  |grad L_frac| (unweighted, BEFORE lambda applied) = {g_frac_unweighted:.6f}")
    print(f"  Raw gradient ratio (g_frac/g_seg, unweighted) = {g_frac_unweighted/max(g_seg,1e-12):.4f}")

    effective_g_frac = lambda_frac_value_matched * g_frac_unweighted
    print(f"\n  If lambda_frac={lambda_frac_value_matched:.4f} is applied:")
    print(f"    effective |grad (lambda_frac * L_frac)| = {effective_g_frac:.6f}")
    print(f"    ratio to |grad L_seg| = {effective_g_frac/max(g_seg,1e-12):.4f}")

    ratio = effective_g_frac / max(g_seg, 1e-12)
    # NOTE: frac_hat is intended as a LIGHT auxiliary signal only (it
    # exists to keep the proxy meaningful, not to drive segmentation
    # directly) -- so unlike lambda_d8 (a full auxiliary segmentation
    # head), this project deliberately targets a SMALL effective
    # gradient ratio (well below 1.0), not a ratio near 1.0. The
    # threshold below still follows E34's own safeguard logic (checking
    # gradient magnitude explicitly, not assuming value-matching is
    # safe) but the ACCEPTABLE range is asymmetric and disclosed here.
    TARGET_RATIO = 0.1  # frac_hat's gradient contribution should be ~10% of seg's, not comparable to it
    if ratio > 1.5 or ratio < 0.001:
        print(f"\n  WARNING: gradient ratio {ratio:.2f} is far outside the safe auxiliary-signal range.")
        print(f"  Recalibrating lambda_frac to target a {TARGET_RATIO:.2f} effective gradient ratio (light auxiliary signal, not full comparable-weight supervision).")
        lambda_frac_final = TARGET_RATIO * g_seg / max(g_frac_unweighted, 1e-12)
    else:
        print(f"\n  Gradient ratio is within the safe auxiliary-signal range -- but still rescaling to the")
        print(f"  disclosed target ratio ({TARGET_RATIO:.2f}) for a deliberately light auxiliary weighting,")
        print(f"  matching this loss term's intended role (keep frac_hat meaningful, not drive segmentation).")
        lambda_frac_final = TARGET_RATIO * g_seg / max(g_frac_unweighted, 1e-12)

    import json
    out_dir = Path(__file__).parent
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "E49_lambda_frac_calibration.json", "w") as f:
        json.dump({
            "L_seg_init": L_seg.item(), "L_frac_init": L_frac.item(),
            "lambda_frac_value_matched": lambda_frac_value_matched,
            "grad_norm_L_seg": g_seg, "grad_norm_L_frac_unweighted": g_frac_unweighted,
            "gradient_ratio_at_value_matched_lambda": ratio,
            "target_ratio": TARGET_RATIO,
            "final_lambda_frac": lambda_frac_final,
            "seed": 0,
        }, f, indent=2)
    print(f"\nSaved to {out_dir / 'E49_lambda_frac_calibration.json'}")
    print(f"\n=== FINAL lambda_frac = {lambda_frac_final:.4f} ===")


if __name__ == "__main__":
    main()
