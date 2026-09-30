"""E224-R -- per-region wrapper around E103's FROZEN, verified CC-DiceCE
implementation (experiments/exp_e12_eggo_m/e103/cc_dicece_loss.py).

E103 tested CC-DiceCE on the OLD single-label binary whole-tumor setup and
found the mechanism REVERSED vs the paper's documented BraTS failure mode
(FP -12.3%, FN +16.5% -- opposite of predicted). This project has since
moved to the current E130+ ET/TC/WT 3-region training geometry
(UNet3D_v5, 4-modality input, FocalTversky averaged per-region). E103's
result has never been checked on this geometry.

THIS FILE DOES NOT REIMPLEMENT the Voronoi/connected-component mechanism.
It imports E103's own `compute_voronoi_cpu` and `cc_dice_ce_component_loss`
functions UNCHANGED and applies them independently to each of the 3 regions
(ET, TC, WT), then averages -- exactly matching how the CURRENT trainer's
own `focal_tversky()` generalizes DiceCE from single-region to 3-region
(per-region loss, uniform average, no region reweighting). This keeps the
comparison to the current baseline as close to a MINIMAL DIFF as possible:
same architecture, same data, same optimizer, same sampler, same
auxiliary/boundary/evidential terms -- ONLY the segmentation-loss term's
per-instance-awareness is added.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

E103_DIR = Path(__file__).resolve().parents[1] / 'e103'
sys.path.insert(0, str(E103_DIR))
from cc_dicece_loss import compute_voronoi_cpu, cc_dice_ce_component_loss  # noqa: E402

REGIONS = ('ET', 'TC', 'WT')


class CCDiceCELossMultiRegion(torch.nn.Module):
    """L = 0.5 * L_DiceCE_global(per-region, averaged)
         + 0.5 * L_CC(per-region, averaged)
    Each term computed independently per region using E103's UNCHANGED
    per-channel functions, then averaged uniformly across ET/TC/WT --
    matching the current baseline's own region-averaging convention
    (train_e130_multimodal_baseline.py's focal_tversky), so any Dice/FP/FN
    delta is attributable to the loss MECHANISM, not to a different
    region-weighting scheme sneaking in alongside it."""

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, probs, target):
        """probs, target: (B, 3, D, H, W) -- current project's convention
        (ET, TC, WT channel order, REGIONS in Dataset/brats_multimodal_dataset.py)."""
        B, R = probs.shape[0], probs.shape[1]
        assert R == 3, f'expected 3 regions (ET,TC,WT), got {R}'
        eps = self.eps

        region_dicece = []
        region_cc = []
        diag = {}
        for ri, rname in enumerate(REGIONS):
            p = probs[:, ri:ri + 1]      # (B,1,D,H,W)
            t = target[:, ri:ri + 1]

            p_flat = p.reshape(B, -1); t_flat = t.reshape(B, -1)
            inter = (p_flat * t_flat).sum(dim=1)
            denom = p_flat.sum(dim=1) + t_flat.sum(dim=1)
            dice_loss = (1.0 - (2.0 * inter / denom.clamp_min(eps))).mean()
            ce = F.binary_cross_entropy(p.clamp(eps, 1 - eps), t, reduction='mean')
            dicece = dice_loss + ce
            region_dicece.append(dicece)

            cc_losses = []
            for b in range(B):
                t_np = t[b, 0].detach().cpu().numpy()
                voronoi = compute_voronoi_cpu(t_np)          # UNCHANGED E103 fn
                cc_loss = cc_dice_ce_component_loss(p[b, 0], t[b, 0], voronoi, eps=eps)  # UNCHANGED
                cc_losses.append(cc_loss)
            cc_mean = torch.stack(cc_losses).mean()
            region_cc.append(cc_mean)
            diag[f'dicece_{rname}'] = float(dicece.item())
            diag[f'cc_{rname}'] = float(cc_mean.item())

        dicece_avg = torch.stack(region_dicece).mean()
        cc_avg = torch.stack(region_cc).mean()
        total = 0.5 * dicece_avg + 0.5 * cc_avg
        diag['dicece_avg'] = float(dicece_avg.item())
        diag['cc_avg'] = float(cc_avg.item())
        return total, diag
