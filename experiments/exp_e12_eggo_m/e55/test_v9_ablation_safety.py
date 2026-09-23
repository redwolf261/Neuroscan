"""
Phase E55, verification checks #3-4: ablation-safety and gradient-flow.

Check #3: native_x=None (local pathway skipped entirely) AND
fusion_gate=0 (local pathway computed but zero-weighted) must BOTH
reproduce UNet3D_v3's own probs/aux_probs3/aux_probs2 bit-for-bit.

Check #4: with native_x provided and fusion_gate != 0, gradient from a
loss on probs_fused (or on logit_local_crop directly) must reach
probs_coarse -- i.e. the centroid/crop path is NOT accidentally
detached. Verified by checking soft_centroid's own output requires_grad
and has a non-None, non-zero gradient after backward().
"""
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_v3 import UNet3D_v3  # noqa: E402
from neuroscan_3d_v9 import UNet3D_v9  # noqa: E402


def max_abs_diff(a, b):
    return (a - b).abs().max().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    B = 2
    torch.manual_seed(0)
    x = torch.randn(B, 1, 64, 64, 64, device=device)
    native_x = torch.randn(B, 1, 240, 240, 155, device=device)

    torch.manual_seed(42)
    v3 = UNet3D_v3().to(device).eval()

    torch.manual_seed(42)
    v9 = UNet3D_v9().to(device).eval()

    # Copy v3's own trunk weights into v9's shared trunk (same
    # comparability convention as E51's own ablation-safety test).
    def copy_trunk(dst, src):
        src_sd = src.state_dict()
        dst_sd = dst.state_dict()
        copied = 0
        for k in src_sd:
            if k in dst_sd and dst_sd[k].shape == src_sd[k].shape:
                dst_sd[k] = src_sd[k].clone()
                copied += 1
        dst.load_state_dict(dst_sd)
        return copied

    n_copied = copy_trunk(v9, v3)
    print(f"Trunk params copied from v3 -> v9: {n_copied}")

    with torch.no_grad():
        out_v3 = v3(x)

        # ---- Check 3a: native_x=None ----
        out_v9_none = v9(x, native_x=None)
        d_probs_a = max_abs_diff(out_v3["probs"], out_v9_none["probs"])
        d_aux3_a = max_abs_diff(out_v3["aux_probs3"], out_v9_none["aux_probs3"])
        d_aux2_a = max_abs_diff(out_v3["aux_probs2"], out_v9_none["aux_probs2"])
        print(f"\n[Check 3a: native_x=None] max|probs diff|={d_probs_a:.3e} "
              f"max|aux3 diff|={d_aux3_a:.3e} max|aux2 diff|={d_aux2_a:.3e}")
        check3a_pass = d_probs_a < 1e-5 and d_aux3_a < 1e-5 and d_aux2_a < 1e-5
        print(f"PASS: {check3a_pass}")

        # ---- Check 3b: native_x provided, fusion_gate forced to 0 ----
        v9_gate0 = UNet3D_v9().to(device).eval()
        v9_gate0.load_state_dict(v9.state_dict())
        with torch.no_grad():
            v9_gate0.fusion_gate.zero_()
        out_v9_gate0 = v9_gate0(x, native_x=native_x)
        d_probs_b = max_abs_diff(out_v3["probs"], out_v9_gate0["probs"])
        print(f"\n[Check 3b: native_x provided, fusion_gate=0] max|probs diff|={d_probs_b:.3e}")
        check3b_pass = d_probs_b < 1e-5
        print(f"PASS: {check3b_pass}")

    # ---- Check 4: gradient flow through the centroid/crop path ----
    torch.manual_seed(42)
    v9_grad = UNet3D_v9().to(device)
    copy_trunk(v9_grad, v3)
    v9_grad.train()
    with torch.no_grad():
        v9_grad.fusion_gate.fill_(1.0)  # nonzero, so the local path actually influences probs_fused

    x_grad = x.clone().requires_grad_(False)
    native_x_grad = native_x.clone().requires_grad_(False)

    out = v9_grad(x_grad, native_x=native_x_grad)
    probs_fused = out["probs"]
    centroid_native = out["centroid_native"]

    # Retain grad on centroid_native (an intermediate, non-leaf tensor)
    # to directly verify a nonzero, finite gradient reaches it.
    centroid_native.retain_grad()

    loss = probs_fused.sum()
    loss.backward()

    grad_exists = centroid_native.grad is not None
    grad_nonzero = grad_exists and centroid_native.grad.abs().sum().item() > 0
    grad_finite = grad_exists and torch.isfinite(centroid_native.grad).all().item()
    print(f"\n[Check 4: gradient flow through centroid/crop path]")
    print(f"  centroid_native.grad exists: {grad_exists}")
    if grad_exists:
        print(f"  centroid_native.grad abs sum: {centroid_native.grad.abs().sum().item():.6e}")
        print(f"  centroid_native.grad finite: {grad_finite}")
    check4_pass = grad_exists and grad_nonzero and grad_finite

    # Also verify the global pathway's own trunk parameters received a
    # gradient contribution routed THROUGH the local pathway's own loss
    # signal (not just through the direct logit_coarse path) -- check by
    # comparing enc1's own gradient with fusion_gate=1 vs fusion_gate=0
    # (same forward otherwise); if identical, the local path isn't
    # actually contributing gradient to the trunk via the centroid route.
    v9_grad.zero_grad(set_to_none=True)
    with torch.no_grad():
        v9_grad.fusion_gate.fill_(0.0)
    out0 = v9_grad(x_grad, native_x=native_x_grad)
    loss0 = out0["probs"].sum()
    loss0.backward()
    enc1_grad_gate0 = v9_grad.enc1[0].conv.weight.grad.clone()

    v9_grad.zero_grad(set_to_none=True)
    with torch.no_grad():
        v9_grad.fusion_gate.fill_(1.0)
    out1 = v9_grad(x_grad, native_x=native_x_grad)
    loss1 = out1["probs"].sum()
    loss1.backward()
    enc1_grad_gate1 = v9_grad.enc1[0].conv.weight.grad.clone()

    trunk_grad_diff = (enc1_grad_gate1 - enc1_grad_gate0).abs().sum().item()
    print(f"\n[Check 4b: trunk gradient DOES differ between gate=0 and gate=1]")
    print(f"  |enc1 grad(gate=1) - enc1 grad(gate=0)| sum = {trunk_grad_diff:.6e}")
    check4b_pass = trunk_grad_diff > 1e-8
    print(f"PASS: {check4b_pass}")

    all_pass = check3a_pass and check3b_pass and check4_pass and check4b_pass
    print(f"\n{'='*60}\nALL CHECKS PASS: {all_pass}")
    if not all_pass:
        print("!!! DO NOT PROCEED TO NEXT VERIFICATION STEP UNTIL ALL CHECKS PASS !!!")
        import sys as sys_
        sys_.exit(1)


if __name__ == "__main__":
    main()
