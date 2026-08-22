"""
NeuroScan 3D v3 (baseline_frozen_v3 candidate) - v2 architecture + two
auxiliary deep-supervision heads, per PHASE_E25_NEXT_STRUCTURAL_PIVOT.md.

Extends UNet3D_v2 (neuroscan_3d_v2.py, v2, NEVER edited further per its
own frozen-lineage convention) with TWO additions:
    aux_head3: dec3 (128ch, D/4 resolution) -> 1 scalar logit per voxel
    aux_head2: dec2 (64ch, D/2 resolution)  -> 1 scalar logit per voxel

Both heads read their INPUT UN-DETACHED, deliberately the OPPOSITE
convention from v2's boundary_head (which reads dec1.detach() so its own
loss can never reshape the trunk). Here the entire point of the
mechanism is the reverse: aux_head3/aux_head2's own loss MUST reach
dec3/dec2's own parameters directly, forcing genuine deep-supervision
gradient into the intermediate decoder stages that otherwise only
receive gradient indirectly, filtered through dec1's own downstream
loss -- this IS the causal mechanism PHASE_E25_NEXT_STRUCTURAL_PIVOT.md
hypothesizes addresses small-lesion detection failure. This asymmetry
(v2's detach vs v3's non-detach) is intentional and load-bearing,
not an inconsistency -- each head's detach status is chosen to match
what that head is FOR.

Both aux heads are 1x1x1 Conv3d + Sigmoid, structurally identical in
FORM to the existing seg_head (matching the project's own established
"architecturally identical to existing heads" convention from PHASE_E9),
differing only in which decoder stage they read and therefore which
number of input channels they take.

CRITICAL PROPERTY, to be verified empirically (not assumed) before any
real training: with lambda_ds3=0 and lambda_ds2=0 in the training
script, this must reproduce v2's own behavior exactly for the PRIMARY
prediction path (probs from dec1/seg_head) -- the auxiliary heads exist
alongside the network but contribute zero loss and therefore zero
gradient when their own weight is zero, so the primary path's own
learning dynamics are unaffected by their mere presence. This is the
ablation control specified in PHASE_E25_NEXT_STRUCTURAL_PIVOT.md
Section 7 (Controls, item 3).

INFERENCE-TIME PROPERTY: aux_head3/aux_head2's outputs are returned by
forward() for training-time loss computation ONLY. The primary
prediction (probs, from seg_head(dec1)) is computed through EXACTLY the
same path as v2 -- unaffected by whether aux head outputs are used or
discarded downstream. A deployed/inference-time caller that only reads
the 'probs' key sees IDENTICAL behavior to v2, modulo the different
LEARNED weights dec1/dec2/dec3 end up with after training with the
auxiliary loss active (the auxiliary loss changes what the trunk learns
during training; it does not add any inference-time computation to the
primary path itself, though the aux heads' own forward computation does
still run every forward() call -- a real, measured runtime cost, not
hidden, but NOT part of the primary prediction's own computational path
in the sense of altering its output).

v1 (neuroscan_3d_fixed.py) and v2 (neuroscan_3d_v2.py) remain permanent,
git-tracked, and are NEVER edited further. This file is a new candidate
base architecture for PHASE_E25's structural pivot only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroscan_3d_v2 import UNet3D_v2  # noqa: F401


class UNet3D_v3(UNet3D_v2):
    """v2's UNet3D_v2 (v1 + dormant boundary_head) + two auxiliary deep-
    supervision heads reading dec3/dec2 directly (NOT detached)."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        # Auxiliary deep-supervision heads. Channel counts match dec3's
        # (128) and dec2's (64) own output channels exactly, per direct
        # inspection of neuroscan_3d_fixed.py's UNet3D.__init__ (verified
        # against the real source immediately before writing this file,
        # not assumed from memory) -- self.dec3 = Sequential(Conv3DBlock
        # (256,128), Conv3DBlock(128,128)), self.dec2 = Sequential(
        # Conv3DBlock(128,64), Conv3DBlock(64,64)).
        self.aux_head3 = nn.Sequential(
            nn.Conv3d(128, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.aux_head2 = nn.Sequential(
            nn.Conv3d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Returns everything v2's forward() returns, plus:
            'aux_probs3': (B, 1, D/4, H/4, W/4) auxiliary prediction from
                dec3, UN-detached -- gradient from this head's own loss
                reaches dec3's (and upstream: dec1..bottleneck's) own
                parameters directly.
            'aux_probs2': (B, 1, D/2, H/2, W/2) auxiliary prediction from
                dec2, UN-detached -- same property, one stage later.
        """
        # Duplicated trunk, byte-for-byte identical to v2's own forward()
        # -- same reasoning as v2's own docstring: v1/v2 are frozen, this
        # avoids touching either file, at the cost of needing to stay in
        # lockstep if the (frozen, unchanging) trunk ever changed.
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
        cat1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = self.dec1(cat1)  # (B, 32, D, H, W)

        probs = self.seg_head(dec1)

        evidential_raw = self.evidential_head(dec1)
        alpha_raw, beta_raw = torch.chunk(evidential_raw, 2, dim=1)
        alpha = F.softplus(alpha_raw) + 1.0
        beta = F.softplus(beta_raw) + 1.0

        boundary_logit = self.boundary_head(dec1.detach())  # UNCHANGED from v2 -- still detached, still dormant unless mu>0

        # THE ONLY NEW LINES vs v2: auxiliary heads on UN-detached dec3/dec2.
        aux_probs3 = self.aux_head3(dec3)  # (B, 1, D/4, H/4, W/4)
        aux_probs2 = self.aux_head2(dec2)  # (B, 1, D/2, H/2, W/2)

        return {
            "probs": probs,
            "alpha": alpha,
            "beta": beta,
            "boundary_logit": boundary_logit,
            "dec1": dec1,
            "dec2": dec2,  # exposed for potential future diagnostics, matching dec1's own precedent
            "dec3": dec3,
            "aux_probs3": aux_probs3,
            "aux_probs2": aux_probs2,
        }


__all__ = ["UNet3D_v3"]
