"""
NeuroScan 3D v17 -- Cross-Modal Prototype Hallucination & Skip-Routing (CMPH-Net)
for Robust Incomplete/Missing Modality Brain Tumor Segmentation.

THE PROBLEM:
------------
In clinical practice, acquiring all 4 MRI sequences (t1c, t1n, t2f, t2w) is
frequently impossible due to scan time limits, severe motion artifacts, or
contraindications to Gadolinium-based contrast agents (t1c) in renal impairment.
Standard 3D U-Nets collapse catastrophically when sequences are missing (e.g.
missing t1c drops Enhancing Tumor Dice from 0.82 to ~0.00).

THE NOVEL INCREMENT (CMPH):
---------------------------
v17 introduces a 3-part Cross-Modal Prototype Hallucination architecture:

1. MODALITY-DISENTANGLED MULTI-STEM ENCODER:
   Each input sequence (t1c, t1n, t2f, t2w) is processed by a dedicated
   modality stem before early cross-fusion, allowing missing sequences to be
   cleanly masked at the input without corrupting other modality representations.

2. CROSS-MODAL LATENT PROTOTYPE HALLUCINATION (CMPH Gate):
   At the bottleneck (8^3, 256 channels), a multi-head cross-attention bank
   queries a learned prototype memory of complete multi-modal feature geometry.
   When any modality subset is absent (e.g. missing t1c), CMPH dynamically
   hallucinates the missing contrast features from the spatial and semantic
   correlations present in the remaining available modalities (t2f + t2w + t1n).

3. MULTI-SCALE LATENT RECOVERY WITH D4 DEEP SUPERVISION:
   The hallucinated latent features are routed into the multi-scale decoder,
   reinforced by auxiliary heads (aux_head3 at 1/4 resolution, aux_head2 at 1/2
   resolution) and a latent consistency loss between full-modality teacher passes
   and missing-modality student passes.

INPUT/OUTPUT CONVENTION:
------------------------
Input:  x: (B, 4, D, H, W) in channel order (t1c, t1n, t2f, t2w)
        modality_mask: (B, 4) binary tensor where 1 = present, 0 = missing
Output: Dict containing:
        - 'logits': (B, 3, D, H, W) final prediction (ET, TC, WT)
        - 'aux_logits3': (B, 3, D/4, H/4, W/4) D4 deep supervision
        - 'aux_logits2': (B, 3, D/2, H/2, W/2) D2 deep supervision
        - 'bottleneck_fused': (B, 256, D/8, H/8, W/8) latent bottleneck representation
        - 'hallucinated_features': (B, 256, D/8, H/8, W/8) hallucinated component
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """Standard 3D Double Convolution Block with BatchNorm and LeakyReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class CrossModalPrototypeHallucinator(nn.Module):
    """
    Bottleneck Cross-Modal Attention Hallucinator.
    Reconstructs missing modality contrast distributions from available modalities
    using multi-head spatial cross-attention over learned prototype priors.
    """
    def __init__(self, channels=256, num_modalities=4, num_prototypes=16, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_modalities = num_modalities
        self.num_prototypes = num_prototypes
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Modality identifier embeddings
        self.modality_embed = nn.Parameter(torch.randn(num_modalities, channels) * 0.02)
        
        # Learned Prototype Bank capturing multi-modal lesion co-occurrence geometry
        self.prototype_bank = nn.Parameter(torch.randn(num_prototypes, channels) * 0.02)

        # Cross-attention projections
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        # Gating parameter initialized near zero so full-modality pass starts intact
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(channels)

    def forward(self, feat_bottleneck, modality_mask):
        """
        feat_bottleneck: (B, C, D, H, W)
        modality_mask: (B, M) where 1=present, 0=missing
        """
        B, C, D, H, W = feat_bottleneck.shape
        spatial_tokens = D * H * W
        
        # Reshape bottleneck into tokens: (B, N, C)
        tokens = feat_bottleneck.view(B, C, spatial_tokens).permute(0, 2, 1) # (B, N, C)

        # Calculate presence weight per batch element: (B, 1, 1)
        present_ratio = modality_mask.float().mean(dim=1, keepdim=True).unsqueeze(-1) # (B, 1, 1)
        missing_mask = (1.0 - modality_mask.float()) # (B, M)

        # Missing modality query: Weighted prototype combination conditioned on what is missing
        missing_embed = torch.matmul(missing_mask, self.modality_embed) # (B, C)
        missing_embed = missing_embed.unsqueeze(1) # (B, 1, C)

        # Multi-head Cross-Attention
        # Query: Missing embedding + Prototypes
        # Key/Value: Available tokens
        prototypes_expanded = self.prototype_bank.unsqueeze(0).expand(B, -1, -1) # (B, K, C)
        query = missing_embed + prototypes_expanded # (B, K, C)

        Q = self.q_proj(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(tokens).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(tokens).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        hallucinated_prototypes = torch.matmul(attn, V) # (B, num_heads, K, head_dim)
        hallucinated_prototypes = hallucinated_prototypes.transpose(1, 2).reshape(B, -1, C)
        hallucinated_prototypes = self.out_proj(hallucinated_prototypes) # (B, K, C)

        # Re-project hallucinated prototypes back to spatial tokens
        # Spatial attention over hallucinated prototypes
        Q_spatial = tokens # (B, N, C)
        K_proto = hallucinated_prototypes # (B, K, C)
        V_proto = hallucinated_prototypes # (B, K, C)

        spatial_scores = torch.matmul(Q_spatial, K_proto.transpose(-2, -1)) / math.sqrt(C)
        spatial_attn = F.softmax(spatial_scores, dim=-1) # (B, N, K)
        hallucinated_spatial = torch.matmul(spatial_attn, V_proto) # (B, N, C)

        hallucinated_spatial = self.norm(hallucinated_spatial)

        # Modulate by missingness: If all modalities present, hallucination effect is minimal
        # If modalities are missing, injection is scaled up
        effective_gate = torch.tanh(self.gate) * (1.0 - present_ratio) # (B, 1, 1)
        modulated_tokens = tokens + effective_gate * hallucinated_spatial

        # Reshape back to (B, C, D, H, W)
        out = modulated_tokens.permute(0, 2, 1).view(B, C, D, H, W)
        hallucinated_out = hallucinated_spatial.permute(0, 2, 1).view(B, C, D, H, W)

        return out, hallucinated_out


class UNet3D_v17_CMPH(nn.Module):
    """
    3D U-Net with Cross-Modal Prototype Hallucination for Missing MRI Modalities.
    """
    def __init__(self, in_channels=4, out_channels=3, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        # Disentangled individual modality stems (1 channel each -> 16 channels)
        self.stem_t1c = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.stem_t1n = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.stem_t2f = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.stem_t2w = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # Stage 1 Encoder: Combined 64 channels from stems -> 32 channels
        self.enc1_fusion = DoubleConv3D(64, base_channels)        # 32 channels, 128^3
        self.pool1 = nn.MaxPool3d(2)                              # -> 64^3

        # Stage 2 Encoder
        self.enc2 = DoubleConv3D(base_channels, base_channels * 2) # 64 channels, 64^3
        self.pool2 = nn.MaxPool3d(2)                              # -> 32^3

        # Stage 3 Encoder
        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4) # 128 channels, 32^3
        self.pool3 = nn.MaxPool3d(2)                              # -> 16^3 (or 8^3)

        # Bottleneck
        self.bottleneck = DoubleConv3D(base_channels * 4, base_channels * 8) # 256 channels

        # Cross-Modal Prototype Hallucination Module
        self.cmph_gate = CrossModalPrototypeHallucinator(channels=base_channels * 8)

        # Decoder Stage 3
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(base_channels * 8, base_channels * 4) # 128 channels
        self.aux_head3 = nn.Conv3d(base_channels * 4, out_channels, kernel_size=1) # D4 supervision

        # Decoder Stage 2
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2) # 64 channels
        self.aux_head2 = nn.Conv3d(base_channels * 2, out_channels, kernel_size=1) # D2 supervision

        # Decoder Stage 1
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base_channels * 2, base_channels) # 32 channels

        # Final Segmentation Head
        self.seg_head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward_encoder(self, x, modality_mask):
        """
        x: (B, 4, D, H, W)
        modality_mask: (B, 4)
        """
        B = x.shape[0]
        # Modality masking at input
        mask_t1c = modality_mask[:, 0].view(B, 1, 1, 1, 1)
        mask_t1n = modality_mask[:, 1].view(B, 1, 1, 1, 1)
        mask_t2f = modality_mask[:, 2].view(B, 1, 1, 1, 1)
        mask_t2w = modality_mask[:, 3].view(B, 1, 1, 1, 1)

        f_t1c = self.stem_t1c(x[:, 0:1]) * mask_t1c
        f_t1n = self.stem_t1n(x[:, 1:2]) * mask_t1n
        f_t2f = self.stem_t2f(x[:, 2:3]) * mask_t2f
        f_t2w = self.stem_t2w(x[:, 3:4]) * mask_t2w

        stem_cat = torch.cat([f_t1c, f_t1n, f_t2f, f_t2w], dim=1)
        e1 = self.enc1_fusion(stem_cat)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bn = self.bottleneck(self.pool3(e3))

        return e1, e2, e3, bn

    def forward(self, x, modality_mask=None):
        """
        x: (B, 4, D, H, W)
        modality_mask: (B, 4) where 1=present, 0=missing
        """
        B = x.shape[0]
        if modality_mask is None:
            modality_mask = torch.ones((B, 4), dtype=torch.float32, device=x.device)

        e1, e2, e3, bn = self.forward_encoder(x, modality_mask)

        # Cross-Modal Prototype Hallucination at Bottleneck
        bn_fused, hallucinated = self.cmph_gate(bn, modality_mask)

        # Decoder with Multi-scale Skips
        d3 = self.dec3(torch.cat([self.up3(bn_fused), e3], dim=1))
        aux3 = self.aux_head3(d3)

        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        aux2 = self.aux_head2(d2)

        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.seg_head(d1)

        return {
            'logits': logits,
            'aux_logits3': aux3,
            'aux_logits2': aux2,
            'bottleneck_fused': bn_fused,
            'hallucinated_features': hallucinated
        }


def compute_cmph_loss(outputs, targets, outputs_teacher=None, alpha_d4=0.4, alpha_d2=0.2, alpha_cons=0.1):
    """
    Computes composite loss:
    1. Dice + BCE loss on primary segmentation logits
    2. Deep Supervision on aux_logits3 (D4) and aux_logits2 (D2)
    3. Latent Consistency loss against teacher (full-modality pass) if provided
    """
    logits = outputs['logits']
    probs = torch.sigmoid(logits)

    # Multi-label Soft Dice loss for (ET, TC, WT)
    smooth = 1e-5
    intersection = (probs * targets).sum(dim=(2, 3, 4))
    union = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))
    dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = dice_loss.mean()

    # Binary Cross Entropy
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    loss_primary = 0.5 * dice_loss + 0.5 * bce_loss

    # Deep Supervision losses (downsample targets to match aux resolutions)
    t_d4 = F.interpolate(targets, size=outputs['aux_logits3'].shape[2:], mode='nearest')
    t_d2 = F.interpolate(targets, size=outputs['aux_logits2'].shape[2:], mode='nearest')

    loss_aux3 = F.binary_cross_entropy_with_logits(outputs['aux_logits3'], t_d4)
    loss_aux2 = F.binary_cross_entropy_with_logits(outputs['aux_logits2'], t_d2)

    total_loss = loss_primary + alpha_d4 * loss_aux3 + alpha_d2 * loss_aux2

    # Latent Consistency Loss (if full-modality teacher is provided)
    if outputs_teacher is not None:
        cons_loss = F.mse_loss(outputs['bottleneck_fused'], outputs_teacher['bottleneck_fused'].detach())
        total_loss += alpha_cons * cons_loss

    return total_loss
