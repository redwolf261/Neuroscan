"""
NeuroScan 3D v11 -- Self-Diagnostic Localized Refinement (SDLR).

Extends UNet3D_v3 (in_channels=4-ready, D4-only deep supervision) with a
single spatial gate at dec1, driven by a SELF-PREDICTED causal
sensitivity map -- the smallest possible structural test of Phase E73's
finding (see docs/phases/PHASE_E73_*.md):

  1. E48/CDCG-1: bottleneck contains real, causally-grounded self-
     knowledge of its OWN ablation sensitivity (scalar, rho=0.87 held-out).
  2. E73 probe: the SPATIAL version of that sensitivity (S_i(p) = |P_intact
     - P_ablated|, voxel-wise) is a non-trivial map that correlates with
     the network's own segmentation errors far above chance (22.6x ratio,
     survives controlling for boundary distance).
  3. E73 predictor: a small head H(bottleneck) -> Ŝ (8^3 grid) can be
     trained to predict that real map (pooled rho=0.41, per-subject
     rho=0.41 held-out, both highly significant, 125/125 subjects) --
     WEAKER than the real map's own error-correlation (ratio ~1.2x vs
     22.6x) but statistically real and significant on its own.
  4. E71/E72: routing/reweighting BY a subject-level scalar version of
     this signal both failed -- because bottleneck/skip dependence is a
     shared marker of subject difficulty, not a separable resource.

THIS ARCHITECTURE deliberately does NOT repeat E71/E72's mistake. It is
NOT pathway routing (nothing is reallocated between bottleneck and skip)
and NOT subject-level reweighting (nothing is scaled per-subject). It is
a SPATIAL, per-voxel gate operating entirely within the existing primary
path: predicted sensitivity Ŝ (upsampled to dec1's resolution) modulates
dec1's own features before seg_head, on the premise that regions the
network itself flags as vulnerable deserve a different (learned) feature
transform than confident regions -- a within-pathway refinement, not a
between-pathway trade-off.

INIT-IDENTITY PROPERTY (verified below, same discipline as CAS/v10):
the gate's blend parameter starts at 0, so at initialization this
architecture is EXACTLY v3's forward() -- any deviation must be learned,
not inherited from a different random init acting as a better feature
extractor.

The spatial sensitivity head's own weights are LOADED FROZEN from E73's
scaled predictor (E73_spatial_head_state_scaled.pt) -- not retrained
here. This isolates the ONE question this pilot asks: given the already-
validated (if weak) predicted map, does gating dec1 with it move Dice at
all? A join training of the sensitivity head together with the trunk is
a materially different, bigger claim, deferred until this smaller
question is answered.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroscan_3d_v3 import UNet3D_v3  # noqa: F401


class SpatialSensitivityHead(nn.Module):
    """Identical architecture to E73's own head (train_spatial_sensitivity_
    predictor.py) -- kept as a duplicate class here (not imported from the
    experiments/ tree, which is gitignored and not meant to be a runtime
    dependency of the frozen architecture files) so this file can load the
    E73 scaled checkpoint's state_dict directly."""
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, bottleneck):
        x = F.relu(self.bn1(self.conv1(bottleneck)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))
        return x.squeeze(1)  # (B, 8,8,8)


class UNet3D_v11(UNet3D_v3):
    """v3 + a single spatial refinement gate at dec1, driven by a FROZEN,
    pre-trained self-predicted causal sensitivity map."""

    def __init__(self, in_channels=1, out_channels=1, sensitivity_head_ckpt=None):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        self.sensitivity_head = SpatialSensitivityHead(in_channels=256)
        if sensitivity_head_ckpt is not None:
            state = torch.load(sensitivity_head_ckpt, map_location="cpu")
            self.sensitivity_head.load_state_dict(state)
        # Frozen: this pilot tests gating with the ALREADY-VALIDATED
        # predictor, not a jointly-learned one (a materially different,
        # bigger claim -- deferred, see module docstring).
        for p in self.sensitivity_head.parameters():
            p.requires_grad_(False)
        self.sensitivity_head.eval()

        # The refinement transform: a small conv on dec1 (32ch) producing
        # a residual, blended in proportional to Ŝ and a learned scalar
        # gate strength that starts at 0 -- so at init, blend=0 and this
        # architecture's forward() is IDENTICAL to v3's, verified below.
        self.refine_conv = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
        )
        # Zero-init the final conv's weight+bias -> refine_conv(dec1) = 0
        # at init, independent of the gate_strength parameter below (belt
        # and suspenders, matching CAS's own double-safety init pattern).
        nn.init.zeros_(self.refine_conv[-1].weight)
        nn.init.zeros_(self.refine_conv[-1].bias)
        self.gate_strength = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
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
        dec1 = self.dec1(cat1)  # (B, 32, D, H, W)

        # ---- Self-diagnostic localized refinement (the ONLY new step) ----
        with torch.no_grad():
            S_hat_8 = self.sensitivity_head(bottleneck)  # (B, 8,8,8), frozen
        S_hat = F.interpolate(S_hat_8.unsqueeze(1), size=dec1.shape[2:],
                              mode="trilinear", align_corners=False)  # (B,1,D,H,W)
        residual = self.refine_conv(dec1)  # (B, 32, D, H, W)
        gate = torch.sigmoid(self.gate_strength) * S_hat  # (B,1,D,H,W), starts at 0.5*S_hat but...
        # NOTE on init: sigmoid(0)=0.5, not 0 -- multiplied by a ZERO
        # residual (refine_conv's last layer is zero-initialized), so the
        # PRODUCT gate*residual is exactly 0 at init regardless of
        # sigmoid(gate_strength)'s own value. This is the actual
        # identity-at-init guarantee (verified empirically below), not the
        # gate_strength parameter alone.
        dec1_refined = dec1 + gate * residual

        probs = self.seg_head(dec1_refined)

        evidential_raw = self.evidential_head(dec1_refined)
        alpha_raw, beta_raw = torch.chunk(evidential_raw, 2, dim=1)
        alpha = F.softplus(alpha_raw) + 1.0
        beta = F.softplus(beta_raw) + 1.0

        boundary_logit = self.boundary_head(dec1_refined.detach())

        aux_probs3 = self.aux_head3(dec3)
        aux_probs2 = self.aux_head2(dec2)

        return {
            "probs": probs,
            "alpha": alpha,
            "beta": beta,
            "boundary_logit": boundary_logit,
            "dec1": dec1_refined,
            "dec2": dec2,
            "dec3": dec3,
            "aux_probs3": aux_probs3,
            "aux_probs2": aux_probs2,
            "S_hat": S_hat,  # exposed for diagnostics
        }
