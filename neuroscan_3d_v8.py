"""
NeuroScan 3D v8 -- E51: v3 architecture + BOTH surviving post-pivot
mechanisms combined -- E46's attention gate (UNet3D_v5) and E49's CCABA
bottleneck amplification (UNet3D_v6) -- stacked in a single model.

MOTIVATION (see PHASE_E49_CCABA_RESULT.md's own recommendation and
PHASE_E50_IECG_RESULT.md's implication section): five single-mechanism
attempts since the strategic pivot (E44 killed, E45 +0.47pp single-seed,
E46 +0.39pp single-seed, E49/CCABA +0.31pp 3-seed mean, E50/IECG +0.02pp
3-seed mean) have all failed to clear D4-only with confidence once seed
variance is accounted for, and the more structurally ambitious mechanism
(IECG) performed WORSE than the simpler one (CCABA). Per the user's own
direction, this is the project's last single-architecture-family attempt
before conceding the +1.0pp bar is not reachable this way: combine the
two mechanisms that were EACH independently verified as real (non-
degenerate, ablation-safe, causally or empirically grounded) rather than
invent a sixth new one.

WHY THESE TWO SPECIFICALLY: attn_gate1 (v5) and CCABA (v6) are
mechanistically non-conflicting by construction -- they act on DIFFERENT
tensors (attn_gate1 gates the enc1 skip; CCABA amplifies the bottleneck
itself, upstream of upconv3) and were designed, in the original E46/E49
work, without any awareness of each other. Nothing about either mechanism
assumes the other is absent. This is a genuine combination, not a
resweep of either mechanism's own hyperparameters.

COMPOSITION ORDER (the one new design decision this file makes): CCABA
amplifies the bottleneck FIRST (bottleneck_amp = bottleneck * (1 +
alpha*w(frac_hat))), and the attention gate's gating signal g is then
computed FROM bottleneck_amp, not from the raw bottleneck. Rationale:
CCABA's own framing (E49) is "strengthen the coarse/global context
pathway"; if the gate is meant to route that context to the fine
decoder path (E46's own framing), it should route the ALREADY-
STRENGTHENED context, not a stale pre-amplification copy. The
alternative order (gate computed from raw bottleneck, independently of
CCABA's amplification) would make attn_gate1 blind to CCABA's own
correction and was rejected as the more confounded choice. This is a
disclosed, deliberate design decision, not an arbitrary one.

bottleneck_amp (not raw bottleneck) is what reaches upconv3, exactly as
in CCABA (v6). enc1_gated (not raw enc1) is what reaches cat1, exactly
as in the attention gate (v5). Every other tensor in the trunk (enc2,
enc3, dec2, dec3, aux heads, evidential/boundary heads) is computed
identically to v3, byte-for-byte, from the same inputs.

SAFETY PROPERTIES (verified in test_v8_ablation_safety.py, not assumed):
  1. Full ablation-safety: alpha=0 AND attn forced to identity (W_g=0,
     W_x=0, psi bias -> sigmoid=~1) reproduces v3's own probs/aux_probs3/
     aux_probs2 bit-for-bit.
  2. Partial ablation-safety (CCABA only, gate forced to identity):
     reproduces v6's (CCABA alone) own probs bit-for-bit, GIVEN THE SAME
     gate-input ordering choice above is accounted for -- i.e. this
     checks the composition itself didn't silently change CCABA's own
     behavior when the gate is inert.
  3. Partial ablation-safety (gate only, alpha=0): reproduces v5's
     (attention gate alone) own probs bit-for-bit, since w(frac_hat) is
     provably 0-contribution when alpha=0 regardless of frac_hat's value,
     making bottleneck_amp = bottleneck exactly, so the gate is fed
     exactly what v5 itself would feed it.

v1/v2/v3/v4/v5/v6/v7 all remain permanent, git-tracked, NEVER edited
further. This file extends v3 directly (not v5 or v6) to keep the
composition logic in one place, explicit and auditable, rather than via
multiple inheritance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroscan_3d_v3 import UNet3D_v3  # noqa: F401
from neuroscan_3d_v5 import AttentionGate3D  # noqa: F401
from neuroscan_3d_v6 import (  # noqa: F401
    SizeProxyHead,
    ccaba_w,
    CCABA_B0,
    CCABA_B1,
    CCABA_W_MIN,
    CCABA_W_MAX,
    CCABA_TOTAL_VOXELS,
)


class UNet3D_v8(UNet3D_v3):
    """v3's UNet3D_v3 + CCABA bottleneck amplification (E49) + attention
    gate on the enc1 skip (E46), composed with the gate reading the
    CCABA-amplified bottleneck (see module docstring for the composition-
    order rationale)."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        self.size_proxy_head = SizeProxyHead(in_channels=256)
        self.ccaba_alpha = nn.Parameter(torch.tensor(0.1))

        # Channel counts identical to v5's own: bottleneck=256ch @ 8^3,
        # enc1=32ch @ 64^3.
        self.attn_gate1 = AttentionGate3D(
            gate_channels=256, skip_channels=32, inter_channels=16
        )

    def forward(self, x):
        """Returns everything v3's forward() returns, plus v6's own
        diagnostic outputs (frac_hat, ccaba_w, ccaba_alpha) and v5's own
        diagnostic output (attention_map).
        The two behavioral changes vs v3:
          1. bottleneck_amp = bottleneck * (1 + alpha*w(frac_hat))
             replaces bottleneck going into upconv3 (CCABA, v6).
          2. enc1_gated = enc1 * psi (psi computed from bottleneck_amp,
             not raw bottleneck -- see module docstring) replaces enc1
             going into cat1 (attention gate, v5).
        Every other tensor in the trunk is computed identically to v3,
        byte-for-byte, from the same inputs.
        """
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        bottleneck = self.bottleneck(pool3)

        frac_hat, occ_map = self.size_proxy_head(bottleneck)
        w = ccaba_w(frac_hat)  # (B,)
        gain = (1.0 + self.ccaba_alpha * w).view(-1, 1, 1, 1, 1)
        bottleneck_amp = bottleneck * gain

        upconv3 = self.upconv3(bottleneck_amp)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = self.dec3(cat3)  # (B, 128, D/4, H/4, W/4)

        upconv2 = self.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = self.dec2(cat2)  # (B, 64, D/2, H/2, W/2)

        upconv1 = self.upconv1(dec2)
        # Gate reads bottleneck_amp (CCABA-amplified), not raw bottleneck
        # -- the one deliberate composition-order decision (see docstring).
        enc1_gated, attention_map = self.attn_gate1(gate=bottleneck_amp, skip=enc1)
        cat1 = torch.cat([upconv1, enc1_gated], dim=1)
        dec1 = self.dec1(cat1)  # (B, 32, D, H, W)

        probs = self.seg_head(dec1)

        evidential_raw = self.evidential_head(dec1)
        alpha_raw, beta_raw = torch.chunk(evidential_raw, 2, dim=1)
        alpha = F.softplus(alpha_raw) + 1.0
        beta = F.softplus(beta_raw) + 1.0

        boundary_logit = self.boundary_head(dec1.detach())  # UNCHANGED from v2/v3

        aux_probs3 = self.aux_head3(dec3)  # (B, 1, D/4, H/4, W/4)
        aux_probs2 = self.aux_head2(dec2)  # (B, 1, D/2, H/2, W/2)

        return {
            "probs": probs,
            "alpha": alpha,
            "beta": beta,
            "boundary_logit": boundary_logit,
            "dec1": dec1,
            "dec2": dec2,
            "dec3": dec3,
            "aux_probs3": aux_probs3,
            "aux_probs2": aux_probs2,
            "frac_hat": frac_hat,
            "ccaba_w": w,
            "ccaba_alpha": self.ccaba_alpha.detach(),
            "attention_map": attention_map,
        }


__all__ = ["UNet3D_v8"]
