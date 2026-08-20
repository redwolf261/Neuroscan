"""
NeuroScan 3D v5 -- v3 architecture (v2 + aux_head3/aux_head2 deep
supervision) + ONE new mechanism: an attention gate on the enc1 skip
connection, conditioned on the bottleneck's own coarse semantic context.

MOTIVATION (per PHASE_E44_E45_POST_PIVOT_ALGORITHMIC_ATTEMPTS.md and the
project's own E43 finding): E43 established that the bottleneck DOES carry
real, reproducible representation change associated with the coarse
boundary problem -- but that change was NOT shown to be spatially
localized to boundary regions specifically (the interior-vs-boundary
comparison came back null after controlling the interior-cell-count
confound). The unresolved gap E43 leaves open is not "does the bottleneck
know something useful" (yes) but "is that knowledge being ROUTED to where
the fine decoder needs it" -- nothing in v1/v2/v3's architecture lets the
bottleneck influence WHICH fine-resolution (enc1, 64^3) features the
decoder emphasizes; the enc1 skip is a plain, un-gated concatenation,
identical at every spatial location regardless of what the bottleneck
"knows" about that location's difficulty.

This is a genuine departure from every prior post-pivot attempt (E44
RCGW, E45 D4+D8): those both tinkered with deep-supervision LOSS WEIGHTS
or added another deep-supervision HEAD at a new resolution. Neither
changes what information reaches the decoder or how it's combined -- both
are supervision-side changes. This file is an ARCHITECTURE-side change:
it alters what happens to enc1's own features before they are used by
dec1, gated by a signal computed from the bottleneck. This is the
Attention U-Net mechanism (Oktay et al. 2018, "Attention U-Net: Learning
Where to Look for the Pancreas") adapted to 3D and to this codebase's own
skip topology -- NOT reinvented from scratch, but genuinely novel FOR
THIS codebase (no attention gate exists anywhere in v1/v2/v3), and it
targets the specific gap (routing, not encoding) that E43 left open.

WHY THE ENC1 SKIP SPECIFICALLY (not enc2 or enc3's skips): dec1 is the
PRIMARY prediction path (probs = seg_head(dec1)) -- the one path whose
output quality is what the project's Dice metric actually measures. The
enc1 skip is also the FINEST-resolution skip (64^3, matching the full
prediction resolution 1:1, no upsampling-induced blur), so gating it is
the most direct way to let coarse (bottleneck, 8^3) context reshape the
finest-grained features right before the final prediction is made --
exactly the "boundary is a coarse-vs-fine information problem" framing
that has run through the whole EGGO-M arc since E36.

GATE DESIGN (Attention U-Net additive gate, adapted):
    g = W_g(bottleneck)                          # (B, F_int, 8, 8, 8)
    g_up = upsample(g, size=enc1.shape[2:])        # (B, F_int, 64, 64, 64)
    x = W_x(enc1)                                  # (B, F_int, 64, 64, 64)
    psi = sigmoid(W_psi(relu(g_up + x)))           # (B, 1, 64, 64, 64)
    enc1_gated = enc1 * psi                        # (B, 32, 64, 64, 64)
enc1_gated REPLACES enc1 in cat1 (the dec1 skip concatenation) ONLY.
Every other use of enc1 (there are none elsewhere in the trunk) and every
other skip (enc2, enc3) are UNCHANGED from v3, byte-for-byte.

W_g, W_x are 1x1x1 Conv3d (no bias, per Oktay et al.'s own design,
followed here rather than reinvented) into a shared intermediate channel
width F_int; W_psi is a 1x1x1 Conv3d, F_int -> 1. F_int chosen as 16
(half of enc1's own 32 channels, a conventional bottleneck-attention
sizing choice, not tuned).

psi is exposed in forward()'s return dict (attention_map) for diagnostic
inspection (e.g. checking whether psi is spatially non-trivial, i.e.
actually gating differentially rather than collapsing to an uninformative
constant near 1.0 everywhere -- a real risk with attention gates that
must be checked empirically, not assumed away).

v1 (neuroscan_3d_fixed.py), v2 (neuroscan_3d_v2.py), v3
(neuroscan_3d_v3.py), and v4 (neuroscan_3d_v4.py, E45's D4+D8 variant,
below-target and not iterated further) all remain permanent, git-tracked,
and are NEVER edited further. This file extends v3 (not v4 -- v4's D8
head is a separate, already-closed line of investigation per
PHASE_E45_D4_D8_below_target; this file starts fresh from v3 to isolate
the attention-gate mechanism's own effect without confounding it with
v4's D8 auxiliary loss).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroscan_3d_v3 import UNet3D_v3  # noqa: F401


class AttentionGate3D(nn.Module):
    """3D additive attention gate (Oktay et al. 2018), adapted: gating
    signal g comes from a coarser decoder-independent stage (here, the
    bottleneck) rather than the immediately-preceding decoder stage as in
    the original paper -- because at v3's own enc1-skip position, the
    "immediately preceding decoder stage" (dec2, D/2 resolution) is
    itself already downstream of the bottleneck, so gating from the
    bottleneck directly is the more DIRECT test of "does bottleneck
    context help route the finest skip", matching this project's own
    E43-motivated framing rather than the generic Attention U-Net setup.
    """

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.W_x = nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.W_psi = nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True)

    def forward(self, gate, skip):
        g = self.W_g(gate)
        g_up = F.interpolate(g, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.W_x(skip)
        psi = torch.sigmoid(self.W_psi(F.relu(g_up + x)))
        return skip * psi, psi


class UNet3D_v5(UNet3D_v3):
    """v3's UNet3D_v3 (v2 + aux_head3/aux_head2 deep supervision) + one
    attention gate on the enc1 skip, conditioned on the bottleneck."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        # Channel counts verified against neuroscan_3d_fixed.py directly
        # (not assumed from memory): bottleneck=256ch @ 8^3, enc1=32ch @ 64^3.
        self.attn_gate1 = AttentionGate3D(
            gate_channels=256, skip_channels=32, inter_channels=16
        )

    def forward(self, x):
        """
        Returns everything v3's forward() returns, plus:
            'attention_map': (B, 1, D, H, W) the gate's own psi, at
                enc1's (full) resolution -- diagnostic only, not consumed
                by any loss.
        The ONLY behavioral change vs v3: cat1 (dec1's skip input) is
        built from enc1_gated = enc1 * psi instead of raw enc1. Every
        other tensor in the trunk (enc2, enc3, bottleneck, dec2, dec3,
        aux heads, evidential/boundary heads) is computed identically to
        v3, byte-for-byte, from the same inputs.
        """
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        bottleneck = self.bottleneck(pool3)

        upconv3 = self.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = self.dec3(cat3)  # (B, 128, D/4, H/4, W/4)

        upconv2 = self.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = self.dec2(cat2)  # (B, 64, D/2, H/2, W/2)

        upconv1 = self.upconv1(dec2)
        # THE ONLY NEW STEP vs v3: gate enc1 using bottleneck context
        # before it enters dec1's skip concatenation.
        enc1_gated, attention_map = self.attn_gate1(gate=bottleneck, skip=enc1)
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
            "attention_map": attention_map,
        }


__all__ = ["UNet3D_v5", "AttentionGate3D"]
