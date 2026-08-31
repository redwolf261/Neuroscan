"""
NeuroScan 3D v10 -- UNet3D_v3 + CAS (Correspondence-Aware Skip) at enc1.

MOTIVATION (this is the part that is genuinely ours):
Phase E62-E65's causal-intervention chain established, on this exact
architecture, that the decoder's use of the enc1 skip connection is
causally sensitive to ABSOLUTE SPATIAL CORRESPONDENCE far more than to
any other property of that tensor:

    intervention on enc1 skip      mean Dice drop (n=125, v3 ckpt)
    -------------------------      -------------------------------
    3-voxel translation                    0.214      <-- dominant
    channel permutation                    0.101
    3x3x3 smoothing                        0.028
    2x2x2 local derangement                0.027

(E65; all four significant and size-specific, but translation dominates
local rearrangement by ~8x.) E64 separately proved, via a split
intervention, that this effect lives entirely in the skip path and NOT
in the pooling path (Delta_pool = exactly 0.0 on two checkpoints).

CAS is designed directly from that measurement: if the decoder's
dominant requirement at this junction is positional correspondence
between upconv1(dec2) and enc1, give the network an explicit, learned
mechanism to establish that correspondence rather than assuming the
naive index-aligned concatenation is always correct.

PRIOR ART, stated honestly: learned deformable offset correction at
encoder-decoder skip junctions exists (Dynamic U-Net's DCU module,
arXiv:2403.07303, 2D abdominal CT). CAS differs in three concrete ways,
each traceable to the E62-E65 measurement rather than to intuition:
  (a) 3D volumetric formulation with trilinear resampling (DCU is 2D);
  (b) a BOUNDED offset (tanh-scaled to +/- MAX_OFFSET voxels) rather
      than an unconstrained one -- justified because E65 measured the
      damage curve over a specific small displacement range (2-4 voxels),
      so unbounded offsets are outside the regime the evidence covers
      and are a known instability source in deformable alignment;
  (c) a residual/gated formulation: CAS outputs a convex blend between
      the raw skip and the resampled skip, controlled by a learned
      per-voxel gate, so the module can express "no correction needed
      here" exactly (gate=0 recovers the original architecture bit-for-
      bit). This makes CAS a strict superset of the baseline skip and
      removes the risk that a mis-learned offset degrades a junction
      that was already correct -- a failure mode DCU's always-on
      resampling cannot express.

Property (c) also gives a free interpretability read-out: the learned
gate map shows WHERE the network judges correspondence to be broken,
which is directly comparable to E65's own per-voxel intervention map.

v1/v2/v3 remain frozen and untouched; this subclasses v3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroscan_3d_v3 import UNet3D_v3

MAX_OFFSET_VOXELS = 4.0  # bounds the learned displacement; see (b) above


def _identity_grid(shape, device, dtype):
    """Base sampling grid in normalized [-1,1] coords, shape (1,D,H,W,3)
    in grid_sample's (x,y,z) axis order."""
    D, H, W = shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(D, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    sizes = torch.tensor([W, H, D], device=device, dtype=dtype)
    base = torch.stack([xx, yy, zz], dim=-1)  # (D,H,W,3)
    base = base / (sizes - 1).clamp(min=1) * 2.0 - 1.0
    return base.unsqueeze(0)


class CorrespondenceAwareSkip(nn.Module):
    """Learns (i) a bounded per-voxel displacement field aligning the skip
    tensor to the decoder tensor, and (ii) a per-voxel gate deciding how
    much of the realigned skip to use versus the original skip.

        delta = MAX_OFFSET * tanh(W_off([dec, skip]))        (voxels)
        skip_aligned = grid_sample(skip, identity + delta)
        g = sigmoid(W_gate([dec, skip]))                     (in [0,1])
        skip_out = (1 - g) * skip + g * skip_aligned

    W_off and W_gate are zero-initialized, so at initialization
    delta = 0 and g = 0.5 ... note: we zero-init W_off (delta=0 ->
    skip_aligned == skip) which makes the blend exactly the identity
    regardless of g. The network therefore starts bit-for-bit identical
    to the baseline architecture and must learn any deviation.
    """

    def __init__(self, decoder_channels, skip_channels, hidden=16):
        super().__init__()
        in_ch = decoder_channels + skip_channels
        self.shared = nn.Sequential(
            nn.Conv3d(in_ch, hidden, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.offset_head = nn.Conv3d(hidden, 3, kernel_size=3, padding=1)
        self.gate_head = nn.Conv3d(hidden, 1, kernel_size=3, padding=1)

        # Zero-init the offset head => delta = 0 => skip_aligned == skip
        # => module is the exact identity at init (verified in tests).
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def forward(self, decoder_feat, skip_feat, return_aux=False):
        h = self.shared(torch.cat([decoder_feat, skip_feat], dim=1))

        delta_vox = MAX_OFFSET_VOXELS * torch.tanh(self.offset_head(h))  # (B,3,D,H,W), (dz,dy,dx)
        gate = torch.sigmoid(self.gate_head(h))                          # (B,1,D,H,W)

        B, C, D, H, W = skip_feat.shape
        base = _identity_grid((D, H, W), skip_feat.device, skip_feat.dtype)  # (1,D,H,W,3)

        # Convert voxel displacement to normalized-grid displacement.
        sizes = torch.tensor([W, H, D], device=skip_feat.device, dtype=skip_feat.dtype)
        # delta_vox channels are (dz,dy,dx); grid wants (x,y,z)
        delta_xyz = delta_vox[:, [2, 1, 0]].permute(0, 2, 3, 4, 1)  # (B,D,H,W,3) as (dx,dy,dz)
        delta_norm = delta_xyz * 2.0 / (sizes - 1).clamp(min=1)

        grid = base + delta_norm
        skip_aligned = F.grid_sample(skip_feat, grid, mode="bilinear",
                                     padding_mode="border", align_corners=True)

        skip_out = (1.0 - gate) * skip_feat + gate * skip_aligned
        if return_aux:
            return skip_out, delta_vox, gate
        return skip_out


class UNet3D_v10(UNet3D_v3):
    """v3 trunk with CAS replacing the raw enc1 skip concatenation."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        # upconv1 outputs 32 ch; enc1 outputs 32 ch (see neuroscan_3d_fixed.UNet3D)
        self.cas1 = CorrespondenceAwareSkip(decoder_channels=32, skip_channels=32)

    def forward(self, x, return_cas_aux=False):
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        bottleneck = self.bottleneck(pool3)

        upconv3 = self.upconv3(bottleneck)
        dec3 = self.dec3(torch.cat([upconv3, enc3], dim=1))

        upconv2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([upconv2, enc2], dim=1))

        upconv1 = self.upconv1(dec2)

        # ---- THE ONLY ARCHITECTURAL CHANGE vs v3 ----
        if return_cas_aux:
            skip1, cas_delta, cas_gate = self.cas1(upconv1, enc1, return_aux=True)
        else:
            skip1 = self.cas1(upconv1, enc1)
            cas_delta = cas_gate = None
        dec1 = self.dec1(torch.cat([upconv1, skip1], dim=1))
        # ---------------------------------------------

        probs = self.seg_head(dec1)

        evidential_raw = self.evidential_head(dec1)
        alpha_raw, beta_raw = torch.chunk(evidential_raw, 2, dim=1)
        alpha = F.softplus(alpha_raw) + 1.0
        beta = F.softplus(beta_raw) + 1.0

        boundary_logit = self.boundary_head(dec1.detach())

        aux_probs3 = self.aux_head3(dec3)
        aux_probs2 = self.aux_head2(dec2)

        out = {
            "probs": probs, "alpha": alpha, "beta": beta,
            "boundary_logit": boundary_logit,
            "dec1": dec1, "dec2": dec2, "dec3": dec3,
            "aux_probs3": aux_probs3, "aux_probs2": aux_probs2,
        }
        if return_cas_aux:
            out["cas_delta"] = cas_delta
            out["cas_gate"] = cas_gate
        return out


__all__ = ["UNet3D_v10", "CorrespondenceAwareSkip", "MAX_OFFSET_VOXELS"]
