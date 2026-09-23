"""
Phase E103: Exact CC-DiceCE Loss (frozen formulation), reimplemented on
CPU/scipy since the reference implementation
(github.com/TIO-IKIM/Learning-to-Look-Closer) requires CuPy, which is
not available in this environment. This module preserves the EXACT
mathematical semantics of the reference implementation, verified
directly against the actual source (nnunetv2/training/loss/
instance_losses.py and nnunetv2/utilities/connected_components.py,
fetched 2026-09-05), not reconstructed from the paper's prose alone:

  1. Voronoi partitioning: 26-connectivity connected-component labeling
     of the ground-truth foreground mask, then EVERY voxel (foreground
     AND background) is assigned to its NEAREST ground-truth component
     by Euclidean distance transform (scipy.ndimage.distance_transform_edt
     with return_indices=True is the exact CPU equivalent of the
     reference's cupyx.scipy.ndimage distance_transform_edt call).
  2. Per-component Dice+CE: for each Voronoi region (including the
     background-only region when there are zero foreground components,
     handled via the reference's own "whole volume fallback" logic),
     compute Dice and CE restricted to that region's voxels.
  3. EQUAL WEIGHTING: the final CC-loss is the MEAN over all valid
     components -- regardless of component size. This is the exact
     mechanism the paper's own Discussion attributes the BraTS
     precision-drop failure to (missing a small component is penalized
     identically to missing a large one, but a false positive only
     mildly affects the single Voronoi region it falls in).
  4. CC-DiceCE combines this instance-aware term with global DiceCE in
     a 1:1 ratio: L = 0.5 * L_DiceCE_global + 0.5 * L_CC(dice_ce mode).

FROZEN, NOT MODIFIED: this is the paper's method AS PUBLISHED. Per this
project's own established discipline (do not modify a method before
observing its predicted failure), no correction is applied here.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi


def compute_voronoi_cpu(binary_mask_np: np.ndarray) -> np.ndarray:
    """binary_mask_np: (D,H,W) numpy bool/float array (single sample,
    single foreground channel). Returns integer Voronoi labels (D,H,W),
    0 = no foreground components exist in this sample (whole-volume
    fallback case), 1..K = nearest-component id for every voxel
    (including background voxels)."""
    structure = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity, matches reference exactly
    labeled_cc, num_features = ndi.label(binary_mask_np > 0, structure=structure)
    if num_features == 0:
        return np.zeros_like(labeled_cc, dtype=np.int64)
    inv_mask = labeled_cc == 0
    # scipy's return_indices=True is the exact CPU equivalent of the
    # reference's cupyx distance_transform_edt(..., return_indices=True) call.
    _, indices = ndi.distance_transform_edt(inv_mask, return_indices=True)
    voronoi = labeled_cc[tuple(indices)]
    return voronoi.astype(np.int64)


def cc_dice_ce_component_loss(probs_fg, target_fg, voronoi_labels, eps=1e-7):
    """probs_fg, target_fg: (D,H,W) tensors (foreground probability,
    binary ground truth). voronoi_labels: (D,H,W) numpy int64 array,
    0..K as returned by compute_voronoi_cpu. Returns a scalar loss:
    mean over valid (non-empty) components of (Dice_loss + CE) for
    that component's voxels, matching the reference's
    _vectorized_cc_reduction exactly (segment-sum based per-component
    reduction, EQUAL weight per component regardless of size)."""
    device = probs_fg.device
    voronoi_t = torch.from_numpy(voronoi_labels).to(device).reshape(-1)
    num_slots = int(voronoi_labels.max()) + 1  # includes slot 0 (unused/background-only fallback)

    probs_flat = probs_fg.reshape(-1)
    target_flat = target_fg.reshape(-1)
    ce_flat = -(target_flat * torch.log(probs_flat.clamp_min(eps))
                + (1 - target_flat) * torch.log((1 - probs_flat).clamp_min(eps)))

    def segment_sum(values):
        out = torch.zeros(num_slots, device=device, dtype=values.dtype)
        return out.scatter_add(0, voronoi_t, values)

    intersection = segment_sum(probs_flat * target_flat)[1:]
    pred_sum = segment_sum(probs_flat)[1:]
    true_sum = segment_sum(target_flat)[1:]
    ce_sum = segment_sum(ce_flat)[1:]
    count = segment_sum(torch.ones_like(probs_flat))[1:]

    valid = count > 0
    if valid.sum() == 0:
        # Whole-volume fallback, matching the reference's _whole_volume_fallback_loss
        # for mode="dice_ce": ce_map.mean() + 1.0 (Dice loss is exactly 1.0 with no foreground).
        return ce_flat.mean() + 1.0

    dice_loss_per_comp = 1.0 - (2.0 * intersection / (pred_sum + true_sum).clamp_min(eps))
    ce_per_comp = ce_sum / count.clamp_min(1.0)
    component_loss = dice_loss_per_comp + ce_per_comp

    valid_f = valid.to(component_loss.dtype)
    return (component_loss * valid_f).sum() / valid_f.sum().clamp_min(1.0)


class CCDiceCELoss(torch.nn.Module):
    """Frozen, exact reimplementation of the paper's CC-DiceCE:
    L = 0.5 * L_DiceCE_global + 0.5 * L_CC_dice_ce(instance-aware)

    Binary segmentation only (foreground probability in [0,1], matching
    this project's UNet3D_v5 sigmoid output convention -- NOT the
    reference's 2-channel one-hot convention, adapted here to match
    THIS project's existing architecture without altering the loss's
    actual mathematical mechanism)."""

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, probs, target):
        """probs, target: (B, 1, D, H, W) tensors."""
        B = probs.shape[0]
        eps = self.eps

        # Global DiceCE (matches this project's own dice-loss convention: whole-batch reduction)
        probs_flat = probs.reshape(B, -1)
        target_flat = target.reshape(B, -1)
        intersection = (probs_flat * target_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice_loss_global = (1.0 - (2.0 * intersection / denom.clamp_min(eps))).mean()
        ce_global = F.binary_cross_entropy(probs.clamp(eps, 1 - eps), target, reduction="mean")
        dice_ce_global = dice_loss_global + ce_global

        # Per-sample instance-aware CC-Dice-CE
        cc_losses = []
        for b in range(B):
            target_np = target[b, 0].detach().cpu().numpy()
            voronoi = compute_voronoi_cpu(target_np)
            cc_loss = cc_dice_ce_component_loss(probs[b, 0], target[b, 0], voronoi, eps=eps)
            cc_losses.append(cc_loss)
        cc_loss_mean = torch.stack(cc_losses).mean()

        total = 0.5 * dice_ce_global + 0.5 * cc_loss_mean
        return total, {
            "dice_ce_global": dice_ce_global.item(),
            "cc_loss": cc_loss_mean.item(),
        }
