"""
E51 -- UNet3D_v8 ablation-safety verification.

Checks (per PHASE-doc discipline: verify before trusting, not assumed):
  1. Full ablation-safety: alpha=0 AND attn gate forced to identity
     reproduces v3's own probs/aux_probs3/aux_probs2 bit-for-bit.
  2. CCABA-only ablation-safety: gate forced to identity (alpha left at
     whatever v8 init gives it, but the KEY check is that with the gate
     inert, v8's probs match v6's probs under the SAME alpha/frac_hat/w
     -- i.e. composition didn't silently change CCABA's own math).
  3. Gate-only ablation-safety: alpha=0 (so bottleneck_amp == bottleneck
     exactly) reproduces v5's own probs bit-for-bit, since the gate then
     receives exactly what v5 itself would feed it.
  4. Param overhead vs v3, v5, v6 (sanity, not a correctness check).

Run with: ./venv_gpu/Scripts/python.exe experiments/exp_e12_eggo_m/e51/test_v8_ablation_safety.py
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch

from neuroscan_3d_v3 import UNet3D_v3
from neuroscan_3d_v5 import UNet3D_v5
from neuroscan_3d_v6 import UNet3D_v6
from neuroscan_3d_v8 import UNet3D_v8


def max_abs_diff(a, b):
    return (a - b).abs().max().item()


def force_gate_identity(gate_module):
    """Force an AttentionGate3D to identity: W_g=0, W_x=0, W_psi bias
    large positive -> sigmoid(relu(0))=sigmoid(0)=0.5, NOT ~1.

    NOTE: relu(g_up + x) with W_g=W_x=0 gives relu(0)=0 regardless of
    the bias's own sign, so psi = sigmoid(W_psi_bias). To get psi->1 we
    need W_psi_bias large POSITIVE (not just "large"), verified below.
    """
    with torch.no_grad():
        gate_module.W_g.weight.zero_()
        gate_module.W_x.weight.zero_()
        gate_module.W_psi.weight.zero_()
        gate_module.W_psi.bias.fill_(20.0)  # sigmoid(20) approx 1.0 (2e-9 from 1)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    B, D, H, W = 2, 64, 64, 64
    x = torch.randn(B, 1, D, H, W, device=device)

    # ---------- Build all four models, SAME seed/init for comparability ----------
    torch.manual_seed(42)
    v3 = UNet3D_v3().to(device).eval()

    torch.manual_seed(42)
    v5 = UNet3D_v5().to(device).eval()

    torch.manual_seed(42)
    v6 = UNet3D_v6().to(device).eval()

    torch.manual_seed(42)
    v8 = UNet3D_v8().to(device).eval()

    # Copy v3's own trunk weights into v5/v6/v8's shared trunk so any
    # difference in outputs is due ONLY to the new mechanism(s), not to
    # differing random init across separate model instantiations.
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

    n5 = copy_trunk(v5, v3)
    n6 = copy_trunk(v6, v3)
    n8 = copy_trunk(v8, v3)
    print(f"Trunk params copied from v3 -> v5: {n5}, v6: {n6}, v8: {n8}")

    # Also sync v8's CCABA-specific weights (size_proxy_head, ccaba_alpha)
    # from v6, and v8's gate weights (attn_gate1) from v5, so the
    # per-mechanism ablation-safety checks (#2, #3) compare truly
    # matched mechanism weights, not just matched trunks.
    v8.size_proxy_head.load_state_dict(v6.size_proxy_head.state_dict())
    with torch.no_grad():
        v8.ccaba_alpha.copy_(v6.ccaba_alpha)
    v8.attn_gate1.load_state_dict(v5.attn_gate1.state_dict())

    # ================= Check 1: FULL ablation-safety (v8 -> v3) =================
    with torch.no_grad():
        v8_full_off = UNet3D_v8().to(device).eval()
        v8_full_off.load_state_dict(v8.state_dict())
        with torch.no_grad():
            v8_full_off.ccaba_alpha.zero_()
        force_gate_identity(v8_full_off.attn_gate1)

        out_v3 = v3(x)
        out_v8_off = v8_full_off(x)

    # With alpha=0: bottleneck_amp = bottleneck * (1+0*w) = bottleneck exactly.
    # With gate forced to psi=1 everywhere: enc1_gated = enc1 * 1 = enc1 exactly.
    d_probs = max_abs_diff(out_v3["probs"], out_v8_off["probs"])
    d_aux3 = max_abs_diff(out_v3["aux_probs3"], out_v8_off["aux_probs3"])
    d_aux2 = max_abs_diff(out_v3["aux_probs2"], out_v8_off["aux_probs2"])
    print(f"\n[Check 1: FULL ablation-safety, v8(alpha=0,gate=identity) vs v3]")
    print(f"  max|probs diff|      = {d_probs:.3e}")
    print(f"  max|aux_probs3 diff| = {d_aux3:.3e}")
    print(f"  max|aux_probs2 diff| = {d_aux2:.3e}")
    check1_pass = d_probs < 1e-5 and d_aux3 < 1e-5 and d_aux2 < 1e-5
    print(f"  PASS: {check1_pass}")

    # ================= Check 2: gate-only-off -> CCABA alone (v8 vs v6) =================
    with torch.no_grad():
        v8_gate_off = UNet3D_v8().to(device).eval()
        v8_gate_off.load_state_dict(v8.state_dict())
        force_gate_identity(v8_gate_off.attn_gate1)

        out_v6 = v6(x)
        out_v8_gate_off = v8_gate_off(x)

    d_probs_26 = max_abs_diff(out_v6["probs"], out_v8_gate_off["probs"])
    d_frac_26 = max_abs_diff(out_v6["frac_hat"], out_v8_gate_off["frac_hat"])
    print(f"\n[Check 2: gate forced to identity, v8 vs v6 (CCABA alone)]")
    print(f"  max|probs diff|    = {d_probs_26:.3e}")
    print(f"  max|frac_hat diff| = {d_frac_26:.3e}")
    check2_pass = d_probs_26 < 1e-5
    print(f"  PASS: {check2_pass}")

    # ================= Check 3: alpha=0 -> attention gate alone (v8 vs v5) =================
    with torch.no_grad():
        v8_alpha_off = UNet3D_v8().to(device).eval()
        v8_alpha_off.load_state_dict(v8.state_dict())
        with torch.no_grad():
            v8_alpha_off.ccaba_alpha.zero_()

        out_v5 = v5(x)
        out_v8_alpha_off = v8_alpha_off(x)

    d_probs_28 = max_abs_diff(out_v5["probs"], out_v8_alpha_off["probs"])
    d_attn_28 = max_abs_diff(out_v5["attention_map"], out_v8_alpha_off["attention_map"])
    print(f"\n[Check 3: alpha=0, v8 vs v5 (attention gate alone)]")
    print(f"  max|probs diff|         = {d_probs_28:.3e}")
    print(f"  max|attention_map diff| = {d_attn_28:.3e}")
    check3_pass = d_probs_28 < 1e-5 and d_attn_28 < 1e-5
    print(f"  PASS: {check3_pass}")

    # ================= Check 4: psi not saturated at fresh init, param counts =================
    torch.manual_seed(0)
    v8_fresh = UNet3D_v8().to(device).eval()
    with torch.no_grad():
        out_fresh = v8_fresh(x)
    psi = out_fresh["attention_map"]
    print(f"\n[Check 4: fresh-init sanity]")
    print(f"  psi range: [{psi.min().item():.4f}, {psi.max().item():.4f}], "
          f"mean={psi.mean().item():.4f}, std={psi.std().item():.4f}")
    psi_live = 0.01 < psi.std().item()
    print(f"  psi non-degenerate (std > 0.01): {psi_live}")

    p_v3 = sum(p.numel() for p in v3.parameters())
    p_v5 = sum(p.numel() for p in v5.parameters())
    p_v6 = sum(p.numel() for p in v6.parameters())
    p_v8 = sum(p.numel() for p in v8.parameters())
    print(f"\n  Param counts: v3={p_v3:,}  v5={p_v5:,} (+{p_v5-p_v3})  "
          f"v6={p_v6:,} (+{p_v6-p_v3})  v8={p_v8:,} (+{p_v8-p_v3})")
    expected_v8_overhead = (p_v5 - p_v3) + (p_v6 - p_v3)
    actual_v8_overhead = p_v8 - p_v3
    print(f"  Expected v8 overhead (v5's + v6's, additive): {expected_v8_overhead}")
    print(f"  Actual v8 overhead: {actual_v8_overhead}")
    check4_pass = expected_v8_overhead == actual_v8_overhead

    print(f"\n{'='*60}")
    all_pass = check1_pass and check2_pass and check3_pass and check4_pass
    print(f"ALL CHECKS PASS: {all_pass}")
    if not all_pass:
        print("!!! DO NOT PROCEED TO TRAINING UNTIL ALL CHECKS PASS !!!")
        sys.exit(1)


if __name__ == "__main__":
    main()
