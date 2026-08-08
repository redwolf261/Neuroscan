"""
Clean interface to frozen NeuroScan architecture.

This module loads model classes from final_model.py without triggering
the PediMS dataset initialization that happens at module level.

All code below this line is copied from final_model.py (frozen at baseline-frozen tag).
No modifications to architecture, losses, or components.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Suppress matplotlib backend warning
os.environ['MPLBACKEND'] = 'Agg'

# The following classes are exact copies from final_model.py
# frozen at commit 000c461 (baseline-frozen tag)

class DropPath(nn.Module):
    """Drop path without batch effect."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1. - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        return x * random_tensor / keep_prob if keep_prob > 0 else torch.zeros_like(x)


class AdaptiveSliceSelector(nn.Module):
    """Adaptive slice selection from 3D volume."""
    def __init__(self, k_slices=5):
        super().__init__()
        self.k = k_slices

    def forward(self, x, center_indices=None):
        """
        Args:
            x: (B, C, D, H, W)
            center_indices: optional (B,) indices for center slices
        Returns:
            (B, C*k, H, W) stacked 2.5D slices
        """
        B, C, D, H, W = x.shape

        if center_indices is None:
            center_indices = torch.full((B,), D // 2, dtype=torch.long, device=x.device)

        slices = []
        for i in range(B):
            center = center_indices[i].item() if isinstance(center_indices[i], torch.Tensor) else center_indices[i]
            center = max(self.k // 2, min(center, D - self.k // 2 - 1))

            start = center - self.k // 2
            end = start + self.k
            volume_slice = x[i, :, start:end, :, :].permute(1, 0, 2, 3)
            slices.append(volume_slice.reshape(C * self.k, H, W))

        return torch.stack(slices, dim=0)


class Conv2D5Stem(nn.Module):
    """2.5D convolution stem."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class MiniSwinAttention2D(nn.Module):
    """Mini Swin attention with 4x4 windows."""
    def __init__(self, dim, window_size=4, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)

        qkv = self.qkv(x_flat).reshape(B, -1, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)

        return x.transpose(1, 2).reshape(B, C, H, W)


class ResidualBlock2D(nn.Module):
    """Residual block for 2D processing."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + identity
        x = self.relu(x)
        return x


class HybridMiniSwin2D5_ResNetEncoder(nn.Module):
    """Encoder combining ResNet and Mini-Swin."""
    def __init__(self, k_slices=5, channels=[32, 64, 128, 256, 512], use_adaptive_selection=True):
        super().__init__()
        self.k = k_slices
        self.use_adaptive_selection = use_adaptive_selection

        if use_adaptive_selection:
            self.slice_selector = AdaptiveSliceSelector(k_slices=k_slices)

        self.stem = Conv2D5Stem(k_slices, channels[0])

        self.layers = nn.ModuleList([
            ResidualBlock2D(channels[i], channels[i+1], stride=2)
            for i in range(len(channels) - 1)
        ])

        self.attention = nn.ModuleList([
            MiniSwinAttention2D(channels[i+1])
            for i in range(len(channels) - 1)
        ])

    def forward(self, x, center_indices=None):
        """
        Args:
            x: (B, 1, D, H, W) 3D volume
            center_indices: optional center slice indices
        Returns:
            List of feature maps for decoder skip connections
        """
        if self.use_adaptive_selection:
            x = self.slice_selector(x, center_indices)

        x = self.stem(x)
        features = [x]

        for layer, attn in zip(self.layers, self.attention):
            x = layer(x)
            x = attn(x)
            features.append(x)

        return features


class CBAM_Module(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )

        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x_list):
        """Apply CBAM to bottleneck features."""
        if isinstance(x_list, list):
            x = x_list[-1]
        else:
            x = x_list

        B, C, H, W = x.shape

        # Channel attention
        avg = self.avg_pool(x).view(B, C)
        max_f = self.max_pool(x).view(B, C)
        channel_att = torch.sigmoid(self.fc(avg) + self.fc(max_f)).view(B, C, 1, 1)
        x = x * channel_att

        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial = torch.max(x, dim=1, keepdim=True)[0]
        spatial_att = torch.sigmoid(self.conv(torch.cat([avg_spatial, max_spatial], dim=1)))
        x = x * spatial_att

        return x


class LightweightDecoder(nn.Module):
    """Decoder with skip connections."""
    def __init__(self, channels=[512, 256, 128, 64, 32]):
        super().__init__()
        self.up_layers = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2)
            for i in range(len(channels) - 1)
        ])

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels[i+1] * 2, channels[i+1], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i+1], channels[i+1], 3, padding=1),
                nn.ReLU(inplace=True)
            )
            for i in range(len(channels) - 1)
        ])

        self.final = nn.Conv2d(channels[-1], 1, kernel_size=1)

    def forward(self, features):
        """Decode from encoder features."""
        x = features[-1]

        for i, (up, conv) in enumerate(zip(self.up_layers, self.conv_layers)):
            x = up(x)
            skip = features[-(i+2)]
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        return {"probs": torch.sigmoid(self.final(x))}


class HybridMiniSwin2D5_CBAM(nn.Module):
    """Complete frozen NeuroScan model for BraTS."""
    def __init__(self, in_channels=1, out_channels=1, k_slices=5):
        super().__init__()
        self.k = k_slices

        channels = [32, 64, 128, 256, 512]

        self.encoder = HybridMiniSwin2D5_ResNetEncoder(
            k_slices=k_slices,
            channels=channels,
            use_adaptive_selection=True
        )
        self.cbam = CBAM_Module(channels[-1])
        self.decoder = LightweightDecoder(channels=list(reversed(channels)))

    def forward(self, x, center_indices=None):
        """
        Args:
            x: (B, 1, D, H, W)
            center_indices: optional
        Returns:
            dict with 'probs' key
        """
        features = self.encoder(x, center_indices=center_indices)
        bottleneck = self.cbam(features)
        features[-1] = bottleneck
        output = self.decoder(features)
        return output


# Loss functions (frozen, no changes)
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
    """Evidential uncertainty loss with Beta distribution."""
    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        smooth = 1e-6
        pred = pred.clamp(smooth, 1 - smooth)

        # Predictive uncertainty
        alpha = pred * 100 + 1
        beta = (1 - pred) * 100 + 1

        # Expected value
        mu = alpha / (alpha + beta)

        # Binary cross entropy on expected value
        bce = -(target * torch.log(mu + smooth) + (1 - target) * torch.log(1 - mu + smooth))

        # Uncertainty regularization (encourage confidence)
        uncertainty = 1.0 / (alpha + beta)

        return bce.mean() + self.weight * uncertainty.mean()


class HybridLoss(nn.Module):
    """Combined loss function (frozen)."""
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
            output: dict with 'probs' from model
            target: ground truth mask
        """
        probs = output.get('probs', output) if isinstance(output, dict) else output

        focal_loss = self.focal(probs, target)
        evidential_loss = self.evidential(probs, target)

        total = self.focal_weight * focal_loss + self.evidential_weight * evidential_loss

        return total


# Export for use
__all__ = [
    'HybridMiniSwin2D5_CBAM',
    'HybridLoss',
    'EvidentialBetaLoss',
    'FocalTverskyLoss',
    'AdaptiveSliceSelector',
]
