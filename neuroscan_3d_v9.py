"""
NeuroScan 3D v9 -- E55: Dual-Resolution Global-Context-Preserving Local
Refinement.

MOTIVATION: E54 showed whole-volume high-resolution training (96^3 vs
64^3, no architecture change) is flat vs baseline (0.9066, +0.03pp) --
see PHASE_E54_A96_RESOLUTION_RESULT.md. E29 (PHASE_E29_RESIZE_SURVIVAL.md)
showed the median native GT lesion component vanishes entirely under the
64^3 resize, but the recovery benefit concentrates narrowly (5-150-
native-voxel lesions, ~13% of the population) -- a whole-volume metric
washes this out. E48 (PHASE_E48_BOTTLENECK_ENCODING_AUDIT_REVERSED_
FINDING.md) showed small lesions depend disproportionately on GLOBAL
context (rho=-0.454, p<0.001) -- so a fix must recover local detail
WITHOUT discarding global context.

MECHANISM: the existing 64^3 global pathway (UNet3D_v3) is kept
completely intact -- it is the pathway E48 showed matters most. A second,
small local pathway examines a native-resolution crop, extracted via a
DIFFERENTIABLE soft-centroid + grid_sample mechanism conditioned on the
global pathway's OWN coarse prediction, so the crop location can be
corrected by the local pathway's own gradient during training (not a
fixed/detached crop). The local pathway's prediction is fused back into
the 64^3 grid via a learned gate, ablation-safe at gate=0.

COORDINATE MAPPING (verified numerically, see
experiments/exp_e12_eggo_m/e55/test_v9_coordinate_mapping.py, all 3
tests pass, and test_v9_label_crop_alignment.py, 10/10 real subjects,
99-100% lesion mass recovered):
  1. native_coord[i] = resized_coord[i] * (native_shape[i] / target_shape[i])
     -- per-axis scalar ratio, NOT a general affine (verified: this
     codebase's dataset never uses nib affines; all 1251 BraTS-GLI
     native FLAIR volumes have IDENTICAL shape (240,240,155)).
  2. norm_coord = 2*native_coord/(native_size-1) - 1  -- align_corners=True
     convention, verified to differ from scipy.ndimage.zoom's own
     indexing convention (this was the specific half-voxel-bug risk
     identified during design; the conversion above is the fix).
  3. grid_sample's channel order for a 5D input is (x,y,z), i.e.
     (W,H,D) -- NOT (D,H,W) -- verified via test 3's off-center
     positioning check (a second, independent bug risk from #2 above).

v1 (neuroscan_3d_fixed.py) through v8 (neuroscan_3d_v8.py) all remain
permanent, git-tracked, and are NEVER edited further. This file extends
v3 (the production base for all post-pivot work) directly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroscan_3d_v3 import UNet3D_v3  # noqa: F401
from neuroscan_3d_fixed import Conv3DBlock  # noqa: F401


def native_to_norm(coord, size):
    """align_corners=True convention: index 0 -> -1, index (size-1) -> +1.
    `size` may be a python int/float OR a tensor (kept differentiable
    through `size` is unnecessary -- size is a fixed constant per axis,
    not a learned or predicted quantity)."""
    return 2.0 * coord / (size - 1) - 1.0


class LocalRefinementHead(nn.Module):
    """Small CNN processing a native-resolution crop, producing a local
    refinement logit map at crop resolution. Deliberately NOT a second
    full U-Net -- 3 Conv3DBlocks, capped channel width, per the design's
    own memory-budget reasoning (E54's own measured GPU headroom is
    consumed mostly by the global pathway + gradient accumulation
    machinery; the local pathway must stay lightweight)."""

    def __init__(self, in_channels=1, width=24):
        super().__init__()
        self.block1 = Conv3DBlock(in_channels, width)
        self.block2 = Conv3DBlock(width, width)
        self.block3 = Conv3DBlock(width, width)
        self.out_conv = nn.Conv3d(width, 1, kernel_size=1)  # LOGITS, no sigmoid (fused pre-sigmoid with global logits)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.out_conv(x)  # (B, 1, crop, crop, crop) logits


class SoftCentroid3D(nn.Module):
    """Differentiable probability-weighted mean voxel-index from a
    probability map. NOT argmax/threshold -- gradient from any
    downstream loss on the crop location must be able to flow back
    through this op and correct the global pathway's own coarse
    prediction (this is the actual novel-mechanism property; see
    test_v9_gradient_flow.py for the verification)."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, probs):
        """probs: (B, 1, D, H, W). Returns (B, 3) soft centroid in
        (D,H,W) voxel-index units of the INPUT grid (i.e. 0..D-1 etc,
        NOT normalized)."""
        B, _, D, H, W = probs.shape
        device = probs.device
        d_idx = torch.arange(D, device=device, dtype=probs.dtype).view(1, 1, D, 1, 1)
        h_idx = torch.arange(H, device=device, dtype=probs.dtype).view(1, 1, 1, H, 1)
        w_idx = torch.arange(W, device=device, dtype=probs.dtype).view(1, 1, 1, 1, W)

        mass = probs.sum(dim=(1, 2, 3, 4)) + self.eps  # (B,)
        d_c = (probs * d_idx).sum(dim=(1, 2, 3, 4)) / mass
        h_c = (probs * h_idx).sum(dim=(1, 2, 3, 4)) / mass
        w_c = (probs * w_idx).sum(dim=(1, 2, 3, 4)) / mass
        return torch.stack([d_c, h_c, w_c], dim=1)  # (B, 3)


def build_sampling_grid(center_native, crop_size, native_shape, device, dtype):
    """center_native: (B, 3) tensor, (D,H,W) native voxel-index coords
    (float, differentiable). Returns a (B, crop, crop, crop, 3) grid in
    grid_sample's own (x,y,z)=(W,H,D) normalized-coordinate convention,
    align_corners=True."""
    B = center_native.shape[0]
    offsets = torch.arange(crop_size, device=device, dtype=dtype) - (crop_size - 1) / 2.0
    # (crop, crop, crop) each, native-grid RELATIVE offsets
    off_d, off_h, off_w = torch.meshgrid(offsets, offsets, offsets, indexing="ij")

    # (B, crop, crop, crop): absolute native coords per axis, differentiable wrt center_native
    d_abs = center_native[:, 0].view(B, 1, 1, 1) + off_d.unsqueeze(0)
    h_abs = center_native[:, 1].view(B, 1, 1, 1) + off_h.unsqueeze(0)
    w_abs = center_native[:, 2].view(B, 1, 1, 1) + off_w.unsqueeze(0)

    d_norm = native_to_norm(d_abs, native_shape[0])
    h_norm = native_to_norm(h_abs, native_shape[1])
    w_norm = native_to_norm(w_abs, native_shape[2])

    # grid_sample 5D convention: last dim order is (x,y,z) = (W,H,D)
    grid = torch.stack([w_norm, h_norm, d_norm], dim=-1)  # (B, crop, crop, crop, 3)
    return grid


class UNet3D_v9(UNet3D_v3):
    """v3's UNet3D_v3 (global pathway, UNCHANGED) + a differentiable
    native-resolution local refinement pathway, fused via a learned gate."""

    NATIVE_SHAPE = (240, 240, 155)  # verified fixed across all 1251 BraTS-GLI subjects
    CROP_SIZE = 96

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        self.soft_centroid = SoftCentroid3D()
        self.local_head = LocalRefinementHead(in_channels=in_channels, width=24)
        # Learned fusion gate, single scalar, init near 0 -- matching
        # CCABA's own "start close to but not exactly at the parent
        # behavior" convention (neuroscan_3d_v6.py).
        self.fusion_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, native_x=None):
        """
        x: (B, 1, 64, 64, 64) resized FLAIR (global pathway input).
        native_x: (B, 1, 240, 240, 155) native-resolution FLAIR, OR
            None. If None, the local pathway is SKIPPED entirely (not
            just zero-weighted) -- fusion_gate's own value is irrelevant
            in that case since scatter(logit_local) is never computed;
            probs_fused falls back to probs_coarse exactly. This is the
            ablation-safety property, verified bit-for-bit against v3's
            own output in test_v9_ablation_safety.py.

        Returns everything v3's forward() returns, but with 'probs'
        REPLACED by probs_fused (or probs_coarse if native_x is None or
        fusion_gate is exactly 0 -- both cases reduce to the identical
        v3 output). Adds:
            'probs_coarse': v3's own original prediction (diagnostic +
                what the ablation-safety check compares against).
            'centroid_native': (B,3) the predicted crop location
                (diagnostic).
            'fusion_gate': scalar (diagnostic).
        """
        v3_out = super().forward(x)
        probs_coarse = v3_out["probs"]
        logit_coarse = torch.logit(probs_coarse.clamp(1e-6, 1 - 1e-6))

        if native_x is None:
            v3_out["probs_coarse"] = probs_coarse
            v3_out["centroid_native"] = None
            v3_out["fusion_gate"] = self.fusion_gate.detach()
            return v3_out  # probs unchanged from v3's own output

        B = x.shape[0]
        target_shape = x.shape[2:]  # (64,64,64) or whatever the global pathway's own input resolution is

        centroid_64 = self.soft_centroid(probs_coarse)  # (B,3), differentiable, in x's own grid units

        # Map: x's own grid -> native voxel coords (per-axis scalar ratio)
        scale = torch.tensor(
            [self.NATIVE_SHAPE[i] / target_shape[i] for i in range(3)],
            device=x.device, dtype=x.dtype,
        )
        centroid_native = centroid_64 * scale.view(1, 3)  # (B,3), differentiable

        grid = build_sampling_grid(centroid_native, self.CROP_SIZE, self.NATIVE_SHAPE, x.device, x.dtype)
        crop = F.grid_sample(native_x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        logit_local_crop = self.local_head(crop)  # (B, 1, crop, crop, crop)

        # Scatter the local logits back into the 64^3 grid: build the
        # INVERSE mapping (64^3 grid -> normalized native-crop-local
        # coords) and grid_sample the crop's own logit map at each of
        # the 64^3 grid's own voxel locations. This is differentiable
        # through centroid_native exactly like the forward crop was.
        d_idx = torch.arange(target_shape[0], device=x.device, dtype=x.dtype)
        h_idx = torch.arange(target_shape[1], device=x.device, dtype=x.dtype)
        w_idx = torch.arange(target_shape[2], device=x.device, dtype=x.dtype)
        d_g, h_g, w_g = torch.meshgrid(d_idx, h_idx, w_idx, indexing="ij")  # (64,64,64) each

        # Native coords of every 64^3 grid voxel (per-axis scalar ratio, same formula as centroid mapping)
        d_native = d_g.unsqueeze(0) * scale[0]  # (1,64,64,64) broadcast, but need per-batch centroid offset
        h_native = h_g.unsqueeze(0) * scale[1]
        w_native = w_g.unsqueeze(0) * scale[2]

        # Position of each 64^3 voxel RELATIVE to this sample's own crop
        # center, in native voxel units, then to crop-local normalized
        # coords (crop has its own (CROP_SIZE) extent, centered on 0).
        rel_d = d_native - centroid_native[:, 0].view(B, 1, 1, 1)
        rel_h = h_native - centroid_native[:, 1].view(B, 1, 1, 1)
        rel_w = w_native - centroid_native[:, 2].view(B, 1, 1, 1)

        crop_center_offset = (self.CROP_SIZE - 1) / 2.0
        rel_d_cropidx = rel_d + crop_center_offset
        rel_h_cropidx = rel_h + crop_center_offset
        rel_w_cropidx = rel_w + crop_center_offset

        rel_d_norm = native_to_norm(rel_d_cropidx, self.CROP_SIZE)
        rel_h_norm = native_to_norm(rel_h_cropidx, self.CROP_SIZE)
        rel_w_norm = native_to_norm(rel_w_cropidx, self.CROP_SIZE)

        scatter_grid = torch.stack([rel_w_norm, rel_h_norm, rel_d_norm], dim=-1)  # (B,64,64,64,3), (x,y,z) order
        logit_local_scattered = F.grid_sample(
            logit_local_crop, scatter_grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        )  # (B, 1, 64, 64, 64) -- voxels outside the crop's own coverage get 0 (padding_mode='zeros')

        logit_fused = logit_coarse + self.fusion_gate * logit_local_scattered
        probs_fused = torch.sigmoid(logit_fused)

        v3_out["probs"] = probs_fused
        v3_out["probs_coarse"] = probs_coarse
        v3_out["logit_local_crop"] = logit_local_crop
        v3_out["centroid_native"] = centroid_native
        v3_out["fusion_gate"] = self.fusion_gate.detach()
        return v3_out


__all__ = ["UNet3D_v9", "LocalRefinementHead", "SoftCentroid3D", "build_sampling_grid", "native_to_norm"]
