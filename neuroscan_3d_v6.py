"""
NeuroScan 3D v6 -- v3 architecture + Causally-Calibrated Adaptive
Bottleneck Amplification (CCABA).

NOVELTY CLAIM (see PHASE_E49_CCABA_DESIGN.md for the full argument and
literature comparison): every size-aware segmentation mechanism found in
a 2025-2026 literature scan (S3-Mamba/AAAI2025, SvANet, M4Fuse,
DCSNet, SGDC) derives its size-sensitivity from an ASSUMED architectural
prior or heuristic (e.g. S3-Mamba's curriculum learning explicitly
"initializes sample weights based on lesion size" -- an assumption about
what matters, not a measurement). Every causal-interpretability method
found (CausalX-Net, causal head gating) targets attribution/explanation/
robustness, not architecture design. CCABA is the first mechanism (to
the best of this literature scan's knowledge) whose size-conditioning
FUNCTION is fit directly to a measured, causally-verified per-subject
dependency curve (this project's own E48 audit: bottleneck-ablation
Dice-drop vs. native lesion size, rho=-0.454, p<0.001, n=125, verified
via a bit-for-bit-checked causal intervention on an already-trained
model) rather than assumed a priori. The SHAPE of the size-conditioning
curve is a measured constant, not a learned or assumed function.

MECHANISM:
  1. A lightweight auxiliary head on the bottleneck (8^3, 256ch -> 1ch,
     sigmoid, spatially pooled to a scalar) predicts frac_hat, this
     input's own estimated fractional tumor occupancy -- a cheap,
     inference-time-available PROXY for native_size (unavailable at
     inference, since it requires ground truth).
  2. frac_hat is passed through w(), a FIXED (non-learned) calibration
     function whose coefficients (CCABA_B0, CCABA_B1) come directly from
     an offline log-linear fit to E48's real (native_size, ablation_drop)
     data -- see calibrate_ccaba.py for the exact fit and verification.
     w() is clipped to the empirically OBSERVED drop range from E48 (not
     extrapolated past what was actually measured).
  3. The bottleneck tensor is AMPLIFIED (not gated/suppressed, unlike
     v5's attn_gate1, which is a sigmoid-bounded [0,1] routing gate --
     CCABA is a boost mechanism, (1 + alpha*w), mechanistically distinct
     and applied at the bottleneck itself rather than the enc1 skip):
         bottleneck_amp = bottleneck * (1 + alpha * w(frac_hat))
     where alpha is a SINGLE LEARNABLE SCALAR (init 0.1) controlling how
     strongly the (fixed-shape, causally-derived) calibration signal is
     trusted -- training can scale the mechanism's overall strength, but
     cannot alter the measured SHAPE of the size-dependency, which is
     the actual novel, causally-grounded content of this design.
  4. bottleneck_amp replaces bottleneck going into upconv3 (and, since
     this extends v3 not v5, there is no attn_gate1 here -- CCABA is
     tested in isolation first, per this project's own established
     isolation discipline, before ever considering stacking it with
     E46's attention gate).

SAFETY PROPERTY (to be verified before training, not assumed): with
alpha=0 (or w=0), bottleneck_amp = bottleneck * 1 = bottleneck exactly,
so this file's forward() must reproduce v3's own probs/aux_probs3/
aux_probs2 bit-for-bit under that condition -- same ablation-safety
discipline as v5's attn_gate1.

v1/v2/v3/v4/v5 all remain permanent, git-tracked, NEVER edited further.
This file extends v3 (not v5) to isolate CCABA's own effect.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroscan_3d_v3 import UNet3D_v3  # noqa: F401

# ================= CCABA calibration constants =================
# Fit via calibrate_ccaba.py directly from E48's real
# (native_size, ablation_drop) data (n=125, log-linear OLS):
#     drop = CCABA_B0 + CCABA_B1 * log(native_size + 1)
# NOT re-derived here -- these are frozen, measured constants, matching
# this project's own established convention of treating a calibration
# step as a separate, auditable script (e.g. E45's calibrate_lambda_d8.py).
CCABA_B0 = 1.797287
CCABA_B1 = -0.130993
CCABA_W_MIN = -0.132141  # observed drop range from E48 (min), clip bound
CCABA_W_MAX = 0.792624   # observed drop range from E48 (max), clip bound
CCABA_TOTAL_VOXELS = 240 * 240 * 155  # native BraTS volume size, verified against Dataset/brats_dataset.py's own convention


class SizeProxyHead(nn.Module):
    """Predicts a scalar fractional-occupancy proxy from the bottleneck,
    8^3 -> 1ch sigmoid map -> spatial mean -> scalar per subject in the
    batch. Deliberately POOLED to a scalar (not a per-voxel map like
    aux_head_d8 in v4) since CCABA needs one amplification strength per
    subject, not a spatial map -- a disclosed design choice, not an
    accident."""

    def __init__(self, in_channels=256):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)

    def forward(self, bottleneck):
        occ_map = torch.sigmoid(self.conv(bottleneck))  # (B,1,8,8,8)
        frac_hat = occ_map.mean(dim=(1, 2, 3, 4))  # (B,) scalar per subject
        return frac_hat, occ_map


def ccaba_w(frac_hat):
    """The FIXED, measured calibration function (see module docstring).
    frac_hat: (B,) tensor in [0,1]. Returns (B,) tensor, the causally-
    calibrated amplification weight for each subject."""
    size_hat = frac_hat * CCABA_TOTAL_VOXELS
    raw = CCABA_B0 + CCABA_B1 * torch.log(size_hat + 1.0)
    return torch.clamp(raw, CCABA_W_MIN, CCABA_W_MAX)


class UNet3D_v6(UNet3D_v3):
    """v3's UNet3D_v3 + Causally-Calibrated Adaptive Bottleneck
    Amplification (CCABA)."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        self.size_proxy_head = SizeProxyHead(in_channels=256)
        # Single learnable scalar controlling CCABA's overall strength.
        # Init 0.1 -- small, so training starts close to (but not
        # exactly at) v3's own behavior, matching this project's own
        # convention (e.g. v5's psi starting non-saturated but modest).
        self.ccaba_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """Returns everything v3's forward() returns, plus:
            'frac_hat': (B,) predicted fractional-occupancy proxy.
            'ccaba_w': (B,) the calibration function's output for this
                batch (diagnostic only, not consumed by any loss).
            'ccaba_alpha': scalar, the learned amplification strength
                (diagnostic only).
        The ONLY behavioral change vs v3: bottleneck is replaced by
        bottleneck_amp = bottleneck * (1 + alpha*w(frac_hat)) before
        upconv3. Every other tensor in the trunk is computed identically
        to v3, byte-for-byte, from the same inputs.
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
        gain = (1.0 + self.ccaba_alpha * w).view(-1, 1, 1, 1, 1)  # (B,1,1,1,1) for broadcasting
        bottleneck_amp = bottleneck * gain

        upconv3 = self.upconv3(bottleneck_amp)
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
        }


__all__ = ["UNet3D_v6", "SizeProxyHead", "ccaba_w", "CCABA_B0", "CCABA_B1", "CCABA_W_MIN", "CCABA_W_MAX", "CCABA_TOTAL_VOXELS"]
