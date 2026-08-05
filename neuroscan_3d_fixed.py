"""
NeuroScan 3D Fixed - Full volume processing for BraTS

Changes from frozen model:
1. Process all slices, not just center
2. Output 3D predictions matching input volume shape
3. Use 3D convolutions in decoder
4. Proper 3D feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """3D convolutional block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UNet3D(nn.Module):
    """Simple 3D U-Net for full volume segmentation."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            Conv3DBlock(in_channels, 32),
            Conv3DBlock(32, 32)
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            Conv3DBlock(32, 64),
            Conv3DBlock(64, 64)
        )
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = nn.Sequential(
            Conv3DBlock(64, 128),
            Conv3DBlock(128, 128)
        )
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            Conv3DBlock(128, 256),
            Conv3DBlock(256, 256)
        )

        # Decoder
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            Conv3DBlock(256, 128),
            Conv3DBlock(128, 128)
        )

        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            Conv3DBlock(128, 64),
            Conv3DBlock(64, 64)
        )

        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            Conv3DBlock(64, 32),
            Conv3DBlock(32, 32)
        )

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv3d(32, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Evidential head (independent branch, own parameters, shares dec1 trunk only)
        self.evidential_head = nn.Conv3d(32, out_channels * 2, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, 1, D, H, W)
        Returns:
            dict with keys:
                'probs': (B, out_channels, D, H, W) segmentation probabilities
                'alpha', 'beta': (B, out_channels, D, H, W) evidential Beta params (>=1)
        """
        # Encoder
        enc1 = self.enc1(x)          # (B, 32, D, H, W)
        pool1 = self.pool1(enc1)     # (B, 32, D/2, H/2, W/2)

        enc2 = self.enc2(pool1)      # (B, 64, D/2, H/2, W/2)
        pool2 = self.pool2(enc2)     # (B, 64, D/4, H/4, W/4)

        enc3 = self.enc3(pool2)      # (B, 128, D/4, H/4, W/4)
        pool3 = self.pool3(enc3)     # (B, 128, D/8, H/8, W/8)

        # Bottleneck
        bottleneck = self.bottleneck(pool3)  # (B, 256, D/8, H/8, W/8)

        # Decoder
        upconv3 = self.upconv3(bottleneck)   # (B, 128, D/4, H/4, W/4)
        cat3 = torch.cat([upconv3, enc3], dim=1)  # (B, 256, D/4, H/4, W/4)
        dec3 = self.dec3(cat3)                # (B, 128, D/4, H/4, W/4)

        upconv2 = self.upconv2(dec3)         # (B, 64, D/2, H/2, W/2)
        cat2 = torch.cat([upconv2, enc2], dim=1)  # (B, 128, D/2, H/2, W/2)
        dec2 = self.dec2(cat2)                # (B, 64, D/2, H/2, W/2)

        upconv1 = self.upconv1(dec2)         # (B, 32, D, H, W)
        cat1 = torch.cat([upconv1, enc1], dim=1)  # (B, 64, D, H, W)
        dec1 = self.dec1(cat1)                # (B, 32, D, H, W)  <- shared trunk ends here

        # Two independent heads branch off dec1
        probs = self.seg_head(dec1)                          # (B, 1, D, H, W)

        evidential_raw = self.evidential_head(dec1)           # (B, 2, D, H, W)
        alpha_raw, beta_raw = torch.chunk(evidential_raw, 2, dim=1)
        alpha = F.softplus(alpha_raw) + 1.0                   # (B, 1, D, H, W), >= 1
        beta = F.softplus(beta_raw) + 1.0                     # (B, 1, D, H, W), >= 1

        return {"probs": probs, "alpha": alpha, "beta": beta}


# Loss functions (from frozen model)
class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss."""
    def __init__(self, alpha=0.5, beta=0.5, gamma=4/3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        smooth = 1.0
        pred = pred.clamp(1e-6, 1 - 1e-6)

        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()

        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        return (1 - tversky) ** self.gamma


class EvidentialBetaLoss(nn.Module):
    """
    Evidential uncertainty loss with Beta distribution, operating on
    genuinely learned (alpha, beta) evidence parameters from a dedicated
    model head (not reconstructed algebraically from the segmentation probs).

    Two terms, following Sensoy et al. (2018) evidential deep learning,
    adapted to the binary Beta case:
      1. Expected data-fit term: BCE evaluated at the Beta mean mu = alpha/(alpha+beta).
      2. Evidence-removal regularizer: KL(Beta(alpha_tilde, beta_tilde) || Beta(1,1))
         where alpha_tilde/beta_tilde zero out the evidence supporting the
         CORRECT class, so the penalty only fires on evidence for the WRONG
         class. This is what actually teaches the model to be uncertain when
         wrong and confident when right (a blanket 1/(alpha+beta) penalty
         cannot distinguish the two).
    """
    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight

    def forward(self, alpha, beta, target):
        smooth = 1e-6
        S = alpha + beta

        # 1. Expected BCE under the Beta distribution
        mu = alpha / S
        bce = -(target * torch.log(mu + smooth) + (1 - target) * torch.log(1 - mu + smooth))

        # 2. Evidence-removal KL regularizer (only penalize evidence for the wrong class)
        alpha_tilde = target * 1.0 + (1 - target) * alpha
        beta_tilde = (1 - target) * 1.0 + target * beta
        S_tilde = alpha_tilde + beta_tilde

        kl = (
            torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde) - torch.lgamma(beta_tilde)
            - torch.lgamma(torch.tensor(2.0, device=alpha.device))
            + torch.lgamma(torch.tensor(1.0, device=alpha.device)) * 2
            + (alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))
            + (beta_tilde - 1) * (torch.digamma(beta_tilde) - torch.digamma(S_tilde))
        )

        return bce.mean() + self.weight * kl.mean()


class HybridLoss(nn.Module):
    """Combined loss function."""
    def __init__(self, focal_weight=0.5, evidential_weight=0.5, device='cpu'):
        super().__init__()
        self.focal = FocalTverskyLoss()
        self.evidential = EvidentialBetaLoss(weight=evidential_weight)
        self.focal_weight = focal_weight
        self.evidential_weight = evidential_weight
        self.device = device

    def forward(self, output, target):
        """
        Args:
            output: dict with 'probs', 'alpha', 'beta' from model
            target: ground truth mask (B, 1, D, H, W)
        """
        probs = output["probs"]
        alpha = output["alpha"]
        beta = output["beta"]

        focal_loss = self.focal(probs, target)
        evidential_loss = self.evidential(alpha, beta, target)

        total = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss

        return total


__all__ = [
    'UNet3D',
    'HybridLoss',
    'FocalTverskyLoss',
    'EvidentialBetaLoss',
]
