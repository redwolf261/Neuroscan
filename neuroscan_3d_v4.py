"""
NeuroScan 3D v4 -- v3 architecture (D4/D2 deep-supervision heads) + ONE
additional auxiliary head at the bottleneck (8^3 resolution, 256 channels),
per the project's explicit post-E44 strategic pivot: extend the ALREADY-
VALIDATED deep-supervision mechanism one stage further in the direction
that's already shown to work (D4, the coarsest existing auxiliary stage,
beat both D2 and "Both"), rather than another loss-weighting/scheduling
trick on the existing heads (E44's RCGW family, both formulations, killed).

WHY THE BOTTLENECK SPECIFICALLY: verified directly (not assumed) against
neuroscan_3d_fixed.py's own UNet3D.forward() -- the bottleneck is already
computed, at 8x8x8 resolution (64^3 input -> pool1(32) -> pool2(16) ->
pool3(8) -> bottleneck at 8^3), inside EVERY model built on this trunk
(UNet3D -> UNet3D_v2 -> UNet3D_v3), currently used only as an internal
feature map with no direct supervision. This is the next coarser resolution
step after D4 (16^3) in the SAME geometric progression D4/D2 already use
(64 -> 32 -> 16 -> 8, each step halving resolution) -- not a new,
unrelated architectural idea.

Channel count (256) verified directly against neuroscan_3d_fixed.py's own
UNet3D.__init__: self.bottleneck = Sequential(Conv3DBlock(128,256),
Conv3DBlock(256,256)).

v1 (neuroscan_3d_fixed.py), v2 (neuroscan_3d_v2.py), v3 (neuroscan_3d_v3.py)
remain permanent, git-tracked, and are NEVER edited further, per this
project's own established frozen-lineage convention. This file is a new
candidate architecture, extending v3 via subclassing only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroscan_3d_v3 import UNet3D_v3  # noqa: F401


class UNet3D_v4(UNet3D_v3):
    """v3's UNet3D_v3 (D4/D2 auxiliary heads) + ONE new auxiliary head
    reading the bottleneck (8^3, 256ch) directly, UN-detached -- same
    convention as aux_head3/aux_head2 (gradient reaches the bottleneck's
    own parameters directly, not filtered through anything downstream)."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        # Same 1x1x1 conv + sigmoid pattern as aux_head3/aux_head2/seg_head
        # -- architecturally identical in FORM to every existing head in
        # this project's lineage (matching PHASE_E9's own established
        # "architecturally identical to existing heads" convention),
        # differing only in which stage it reads and its channel count.
        self.aux_head_d8 = nn.Sequential(
            nn.Conv3d(256, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Returns everything v3's forward() returns, plus:
            'aux_probs_d8': (B, 1, D/8, H/8, W/8) auxiliary prediction from
                the bottleneck, UN-detached.
        """
        # Duplicated trunk, byte-for-byte identical to v3's own forward()
        # -- same reasoning as v2/v3's own docstrings: v1/v2/v3 are frozen,
        # this avoids touching any of them, at the cost of needing to stay
        # in lockstep if the (frozen, unchanging) trunk ever changed.
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        bottleneck = self.bottleneck(pool3)  # (B, 256, D/8, H/8, W/8)

        upconv3 = self.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = self.dec3(cat3)  # (B, 128, D/4, H/4, W/4)

        upconv2 = self.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = self.dec2(cat2)  # (B, 64, D/2, H/2, W/2)

        upconv1 = self.upconv1(dec2)
        cat1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = self.dec1(cat1)  # (B, 32, D, H, W)

        probs = self.seg_head(dec1)

        evidential_raw = self.evidential_head(dec1)
        alpha_raw, beta_raw = torch.chunk(evidential_raw, 2, dim=1)
        alpha = F.softplus(alpha_raw) + 1.0
        beta = F.softplus(beta_raw) + 1.0

        boundary_logit = self.boundary_head(dec1.detach())  # UNCHANGED, still detached, still dormant unless mu>0

        aux_probs3 = self.aux_head3(dec3)
        aux_probs2 = self.aux_head2(dec2)

        # THE ONLY NEW LINE vs v3: auxiliary head on UN-detached bottleneck.
        aux_probs_d8 = self.aux_head_d8(bottleneck)  # (B, 1, D/8, H/8, W/8)

        return {
            "probs": probs,
            "alpha": alpha,
            "beta": beta,
            "boundary_logit": boundary_logit,
            "dec1": dec1,
            "dec2": dec2,
            "dec3": dec3,
            "bottleneck": bottleneck,  # exposed for potential future diagnostics, matching dec1/dec2/dec3's own precedent
            "aux_probs3": aux_probs3,
            "aux_probs2": aux_probs2,
            "aux_probs_d8": aux_probs_d8,
        }


__all__ = ["UNet3D_v4"]
