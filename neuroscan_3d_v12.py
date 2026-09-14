"""
NeuroScan 3D v12 -- Track A "capacity evolution" mutation of UNet3D_v5.

PURPOSE (Track A, explicitly NOT a novelty claim): establish the
evolutionary ceiling of pure capacity on this baseline. The mutation is
deliberately UNORIGINAL -- it reproduces the BraTS 2021 winner's own
capacity change (Luu & Park, arXiv:2112.04653: "doubling the number of
filters in the encoder while maintaining the same filters in the
decoder", with the bottleneck raised from nnU-Net's 320 to 512), an idea
they in turn credit to Myronenko's SegResNet (arXiv:1810.11654). We are
not claiming this is new. We are asking whether extra encoder capacity
fixes the causal phenotype this project measured (E48/E121/E126), which
is a question nobody has asked of THIS network.

LINEAGE / WHY A NEW FILE: v1 (neuroscan_3d_fixed.py) and v2
(neuroscan_3d_v2.py) are permanently frozen and never edited; v3 and v5
follow the same convention. Channel widths are hardcoded through the
whole v1->v2->v3->v5 chain, so a capacity change cannot be a parameter --
it requires a new file. v12 subclasses UNet3D_v5 so the attention gate
(attn_gate1) and the two auxiliary deep-supervision heads
(aux_head3/aux_head2) used throughout E46-E127 are preserved.

NOTE ON THE VERSION NUMBER: v6-v11 already exist and subclass UNet3D_v3
(NOT v5). v12 deliberately subclasses UNet3D_v5 instead, because the
canonical checkpoint and the entire E48-E127 causal chain are v5-based --
the attention gate must be present for any comparison against that
lineage to be meaningful.

THE MUTATION (asymmetric -- encoder and bottleneck only):

    stage        baseline (v5)      v12
    enc1         32                 64
    enc2         64                 128
    enc3         128                256
    bottleneck   256                512
    ---- decoder OUTPUT widths deliberately UNCHANGED ----
    dec3         128                128
    dec2         64                 64
    dec1         32                 32

Decoder INPUT channels necessarily change, because the skip tensors they
concatenate are now wider:

    upconv3: ConvTranspose3d(512 -> 128)   [was 256 -> 128]
    cat3   = [upconv3(128), enc3(256)]  = 384ch   [was 256]
    dec3   : Conv3DBlock(384 -> 128), Conv3DBlock(128 -> 128)

    upconv2: ConvTranspose3d(128 -> 64)    [unchanged]
    cat2   = [upconv2(64), enc2(128)]   = 192ch   [was 128]
    dec2   : Conv3DBlock(192 -> 64), Conv3DBlock(64 -> 64)

    upconv1: ConvTranspose3d(64 -> 32)     [unchanged]
    cat1   = [upconv1(32), enc1_gated(64)] = 96ch [was 64]
    dec1   : Conv3DBlock(96 -> 32), Conv3DBlock(32 -> 32)

Because dec3/dec2/dec1 keep their OUTPUT widths, every head downstream
(seg_head, evidential_head, boundary_head, aux_head3, aux_head2) is
inherited from v5 UNCHANGED and remains dimensionally valid. This is the
main reason the asymmetric variant was chosen over uniform doubling: it
isolates "more encoder capacity" without simultaneously changing the
decoder's representational width or touching any head.

The attention gate IS rebuilt, because its gate input is the bottleneck
(now 512 instead of 256) and its skip input is enc1 (now 64 instead of
32). inter_channels is doubled 16 -> 32 to keep the gate's internal
bottleneck at the same RATIO to its inputs as v5's, rather than leaving
it at v5's absolute value (which would make the gate a relatively much
tighter bottleneck in v12 and confound "more capacity" with "relatively
narrower gate").

forward() is NOT overridden. v5's forward() is reused verbatim -- it
refers only to module attributes (self.enc1, self.pool1, ... ,
self.attn_gate1), all of which v12 rebinds to correctly-shaped modules,
so the identical control flow produces the identical tensor topology at
different widths. This is deliberate: it guarantees v12 differs from v5
ONLY in width, with zero chance of an accidental control-flow divergence.
"""
import torch.nn as nn

from neuroscan_3d_fixed import Conv3DBlock
from neuroscan_3d_v5 import UNet3D_v5, AttentionGate3D


class UNet3D_v12(UNet3D_v5):
    """UNet3D_v5 with a doubled-width encoder and bottleneck, decoder
    output widths unchanged. See module docstring for the full table."""

    def __init__(self, in_channels=1, out_channels=1, width_mult=2):
        # Build the full v5 module tree first (heads, aux heads, gate),
        # then rebind the encoder/bottleneck/decoder modules below. The
        # discarded v5-width modules are replaced before any forward pass
        # and never contribute parameters to the optimizer, because
        # nn.Module attribute assignment de-registers the old submodule.
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        w = width_mult
        c1, c2, c3, cb = 32 * w, 64 * w, 128 * w, 256 * w

        # ---- Encoder (widened) ----
        self.enc1 = nn.Sequential(
            Conv3DBlock(in_channels, c1),
            Conv3DBlock(c1, c1),
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            Conv3DBlock(c1, c2),
            Conv3DBlock(c2, c2),
        )
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = nn.Sequential(
            Conv3DBlock(c2, c3),
            Conv3DBlock(c3, c3),
        )
        self.pool3 = nn.MaxPool3d(2)

        # ---- Bottleneck (widened) ----
        self.bottleneck = nn.Sequential(
            Conv3DBlock(c3, cb),
            Conv3DBlock(cb, cb),
        )

        # ---- Decoder: OUTPUT widths pinned to v5's (128/64/32) ----
        self.upconv3 = nn.ConvTranspose3d(cb, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            Conv3DBlock(128 + c3, 128),
            Conv3DBlock(128, 128),
        )

        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            Conv3DBlock(64 + c2, 64),
            Conv3DBlock(64, 64),
        )

        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            Conv3DBlock(32 + c1, 32),
            Conv3DBlock(32, 32),
        )

        # ---- Attention gate: rebuilt for the new gate/skip widths ----
        # inter_channels scaled with width to preserve v5's ratio
        # (16/256 gate, 16/32 skip) rather than holding the absolute
        # value fixed.
        self.attn_gate1 = AttentionGate3D(
            gate_channels=cb, skip_channels=c1, inter_channels=16 * w
        )

        # seg_head, evidential_head, boundary_head, aux_head3, aux_head2
        # are inherited from v5 UNCHANGED and remain valid because
        # dec1/dec2/dec3 output widths are unchanged (32/64/128).
