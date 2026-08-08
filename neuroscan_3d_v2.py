"""
NeuroScan 3D v2 (baseline_frozen_v2) - v1 architecture + dormant boundary head

Extends UNet3D (neuroscan_3d_fixed.py, v1, NEVER edited further per
baseline_frozen_milestone) with one addition: a boundary_head that reads
dec1 and predicts a per-voxel latent decision-boundary score, per
PHASE_E9_TRAINABLE_BOUNDARY_HEAD.md / PHASE_E11_5_READINESS_REVIEW.md.

Critical design property (PHASE_E11_5 Sec 5, the corrected design):
  d = self.boundary_head(dec1.detach())
boundary_head reads a DETACHED copy of dec1, so its own BCE loss can
never reshape the shared trunk -- it can only learn to accurately READ
whatever separability structure dec1 already has (driven by L_seg and,
once active, L_margin), never independently push dec1 to become MORE
separable on its own account. This prevents boundary_head from becoming
an uncontrolled second segmentation head.

With mu=0 (boundary head loss weight) and lambda=0 (margin loss weight)
in the EGGO-M training script, this must reproduce v1's UNet3D exactly
-- the boundary head is present (dormant) but contributes nothing to
training when disabled, verified empirically in E12a, not assumed.

v1 (neuroscan_3d_fixed.py) remains the permanent, git-tracked reference
point (commit daad7fa, tag baseline-frozen) and is NEVER edited further.
This file is the actual base architecture for all EGGO-M (and later
EGGO-MD / EGGO) experiments.
"""
import torch
import torch.nn as nn

from neuroscan_3d_fixed import UNet3D, Conv3DBlock  # noqa: F401 (Conv3DBlock re-exported for parity with v1's __all__ pattern)


class UNet3D_v2(UNet3D):
    """v1's UNet3D + a dormant boundary_head reading dec1.detach()."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        # Boundary head: dec1 (32ch) -> 1 scalar logit per voxel.
        # Same 1x1x1 conv pattern as seg_head/evidential_head, per
        # PHASE_E9's "architecturally identical to the existing heads"
        # requirement. Mathematically exact per-voxel linear projection
        # d_i = w.z_i + b, the same functional form as the offline
        # E1.3 logistic-regression classifier it replaces.
        self.boundary_head = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Returns everything v1's forward() returns, plus:
            'boundary_logit': (B, 1, D, H, W) raw logit d_i = w.z_i + b,
                computed from dec1.detach() -- see module docstring for
                why this detach is load-bearing.
        """
        # Re-run the encoder/decoder path exactly as v1 does. Not calling
        # super().forward() because we need dec1 itself (v1's forward()
        # only returns probs/alpha/beta, not the intermediate dec1
        # tensor) -- duplicating the trunk here, byte-for-byte identical
        # to v1's forward(), rather than modifying v1's method to expose
        # dec1 (which would touch the frozen file). If v1's encoder/
        # decoder ever changes (it won't, it's frozen), this would need
        # to be updated in lockstep -- flagged here for future
        # maintainers, not currently a risk since v1 is permanently frozen.
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        bottleneck = self.bottleneck(pool3)

        upconv3 = self.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = self.dec3(cat3)

        upconv2 = self.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = self.dec2(cat2)

        upconv1 = self.upconv1(dec2)
        cat1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = self.dec1(cat1)  # (B, 32, D, H, W) <- shared trunk ends here

        probs = self.seg_head(dec1)

        evidential_raw = self.evidential_head(dec1)
        alpha_raw, beta_raw = torch.chunk(evidential_raw, 2, dim=1)
        alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
        beta = torch.nn.functional.softplus(beta_raw) + 1.0

        # Boundary head: detached input, per PHASE_E11_5 Sec 5's corrected
        # design. This is the ONLY line that differs from v1's data flow.
        boundary_logit = self.boundary_head(dec1.detach())

        return {
            "probs": probs,
            "alpha": alpha,
            "beta": beta,
            "boundary_logit": boundary_logit,
            "dec1": dec1,  # exposed for EGGO-M's margin loss (needs raw, non-detached dec1)
        }


__all__ = ["UNet3D_v2"]
