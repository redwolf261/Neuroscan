"""
Verify baseline_frozen_v2 (UNet3D_v2) reproduces v1 (UNet3D) EXACTLY for
the shared probs/alpha/beta outputs, given identical weights and input.
This is a required check per PHASE_E11_5 Sec 6/7c before any EGGO-M
training run is trusted -- v2 must be a strict architectural superset
that changes nothing about v1's existing behavior.

Two checks:
  1. Same random init (same seed) -> v1 and v2's shared-layer weights
     must be identical (since v2's __init__ calls super().__init__()
     first, then adds boundary_head -- the RNG state consumed by the
     shared layers should be identical before boundary_head's own
     parameters are drawn).
  2. Given the SAME weights (copy v1's state_dict into v2's matching
     keys) and the same input, v2's probs/alpha/beta must be
     bit-identical (up to floating point determinism) to v1's output.
     This is the real test -- it doesn't depend on RNG ordering at all.
"""
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscan_3d_fixed import UNet3D  # noqa: E402
from neuroscan_3d_v2 import UNet3D_v2  # noqa: E402


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Check 1: same-seed initialization")
    print("=" * 70)
    torch.manual_seed(42)
    model_v1 = UNet3D(in_channels=1, out_channels=1).to(device)
    torch.manual_seed(42)
    model_v2 = UNet3D_v2(in_channels=1, out_channels=1).to(device)

    v1_state = model_v1.state_dict()
    v2_state = model_v2.state_dict()
    mismatches = []
    for key in v1_state:
        if key not in v2_state:
            mismatches.append(f"MISSING in v2: {key}")
            continue
        if not torch.equal(v1_state[key], v2_state[key]):
            mismatches.append(f"VALUE MISMATCH: {key}")
    if mismatches:
        print(f"  {len(mismatches)} mismatches found (expected -- boundary_head's own")
        print("  params are drawn AFTER v1's layers in v2's __init__, consuming extra")
        print("  RNG state that could shift subsequent... but v1 has no 'subsequent'")
        print("  layers after evidential_head, so this should NOT happen for v1's own")
        print("  keys specifically. Investigating:")
        for m in mismatches[:10]:
            print(f"    {m}")
    else:
        print("  PASS: all v1 shared-layer weights match v2 exactly under same seed.")

    print("\n" + "=" * 70)
    print("Check 2: same weights, same input -> identical outputs (the real test)")
    print("=" * 70)
    model_v1.eval()
    model_v2.eval()

    # Force v2's shared layers to have EXACTLY v1's weights, regardless
    # of check 1's outcome -- this is the check that actually matters.
    v2_state_dict = model_v2.state_dict()
    for key in v1_state:
        if key in v2_state_dict:
            v2_state_dict[key] = v1_state[key].clone()
    model_v2.load_state_dict(v2_state_dict)

    x = torch.randn(1, 1, 64, 64, 64, device=device)
    with torch.no_grad():
        out_v1 = model_v1(x)
        out_v2 = model_v2(x)

    all_match = True
    for key in ("probs", "alpha", "beta"):
        v1_val = out_v1[key]
        v2_val = out_v2[key]
        match = torch.equal(v1_val, v2_val)
        max_diff = (v1_val - v2_val).abs().max().item()
        print(f"  {key}: exact_match={match}, max_abs_diff={max_diff:.2e}")
        if not match:
            all_match = False

    print(f"\n  v2 also produces boundary_logit (new): shape={out_v2['boundary_logit'].shape}, "
          f"dec1 (new): shape={out_v2['dec1'].shape}")

    print("\n" + "=" * 70)
    if all_match:
        print("OVERALL: PASS -- v2 reproduces v1's probs/alpha/beta EXACTLY given identical weights.")
    else:
        print("OVERALL: FAIL -- v2 diverges from v1. DO NOT proceed to EGGO-M training.")
    print("=" * 70)

    return all_match


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
