"""
Phase E50: calibration of lambda_sens, the weight on IECG's sensitivity-
head MSE loss (s_hat vs. s_true, the ON-THE-FLY-COMPUTED real causal
sensitivity from this training step's own ablation pass).

SAME discipline as E45's calibrate_lambda_d8.py / E49's
calibrate_lambda_frac.py: measure value AND gradient magnitude at fresh
init, explicitly check for a gradient-magnitude mismatch before trusting
any calibration (E34's own mandatory safeguard). Same "light auxiliary
signal" target-ratio convention as E49's lambda_frac (this loss exists to
keep s_hat meaningful, not to drive segmentation directly -- CCABA's own
lesson: the auxiliary loss should be small relative to seg_loss).
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v7 import UNet3D_v7  # noqa: E402
from neuroscan_3d_fixed import FocalTverskyLoss  # noqa: E402
from Dataset.brats_dataset import create_brats_loaders  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("E50: lambda_sens calibration -- FRESH, randomly-initialized model")
    print("=" * 70)

    torch.manual_seed(0)
    model = UNet3D_v7(in_channels=1, out_channels=1).to(device)
    model.train()

    train_loader, _ = create_brats_loaders(
        batch_size=8, num_workers=0,
        root_dir=str(project_root / "Dataset" / "Training"),
        val_split=0.1,
    )
    images, masks, _ = next(iter(train_loader))
    images = images.to(device)
    masks = masks.to(device)

    focal_fn = FocalTverskyLoss()

    outputs = model(images, masks=masks, compute_sensitivity_target=True)
    probs = outputs["probs"]
    s_hat = outputs["s_hat"]
    s_true = outputs["s_true"]

    L_seg = focal_fn(probs, masks)
    L_sens = F.mse_loss(s_hat, s_true)

    print(f"\nAt fresh init (seed=0, train() mode, real batch=8):")
    print(f"  L_seg (full res, dec1)   = {L_seg.item():.6f}")
    print(f"  L_sens (sensitivity MSE) = {L_sens.item():.6f}")
    print(f"  s_true sample values: {s_true.tolist()}")

    lambda_sens_value_matched = L_seg.item() / max(L_sens.item(), 1e-8)
    print(f"\nValue-matched calibration: lambda_sens = {lambda_sens_value_matched:.4f}")

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

    g_seg = grad_norm_of(L_seg)

    outputs2 = model(images, masks=masks, compute_sensitivity_target=True)
    s_hat_2 = outputs2["s_hat"]
    s_true_2 = outputs2["s_true"]
    L_sens_2 = F.mse_loss(s_hat_2, s_true_2)
    g_sens_unweighted = grad_norm_of(L_sens_2)

    print(f"  |grad L_seg| (unweighted)  = {g_seg:.6f}")
    print(f"  |grad L_sens| (unweighted, BEFORE lambda applied) = {g_sens_unweighted:.6f}")
    print(f"  Raw gradient ratio (g_sens/g_seg, unweighted) = {g_sens_unweighted/max(g_seg,1e-12):.4f}")

    effective_g_sens = lambda_sens_value_matched * g_sens_unweighted
    ratio = effective_g_sens / max(g_seg, 1e-12)
    print(f"\n  If lambda_sens={lambda_sens_value_matched:.4f} is applied:")
    print(f"    effective |grad (lambda_sens * L_sens)| = {effective_g_sens:.6f}")
    print(f"    ratio to |grad L_seg| = {ratio:.4f}")

    TARGET_RATIO = 0.1  # light auxiliary signal, same convention as E49's lambda_frac
    print(f"\n  Rescaling to the disclosed target ratio ({TARGET_RATIO:.2f}) for a deliberately")
    print(f"  light auxiliary weighting (s_hat should shape itself around segmentation's own")
    print(f"  learning, not compete with it) -- same policy as E49's lambda_frac.")
    lambda_sens_final = TARGET_RATIO * g_seg / max(g_sens_unweighted, 1e-12)

    import json
    out_dir = Path(__file__).parent
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "E50_lambda_sens_calibration.json", "w") as f:
        json.dump({
            "L_seg_init": L_seg.item(), "L_sens_init": L_sens.item(),
            "lambda_sens_value_matched": lambda_sens_value_matched,
            "grad_norm_L_seg": g_seg, "grad_norm_L_sens_unweighted": g_sens_unweighted,
            "gradient_ratio_at_value_matched_lambda": ratio,
            "target_ratio": TARGET_RATIO,
            "final_lambda_sens": lambda_sens_final,
            "seed": 0,
        }, f, indent=2)
    print(f"\nSaved to {out_dir / 'E50_lambda_sens_calibration.json'}")
    print(f"\n=== FINAL lambda_sens = {lambda_sens_final:.4f} ===")


if __name__ == "__main__":
    main()
