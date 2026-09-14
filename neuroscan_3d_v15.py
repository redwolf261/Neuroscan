"""
NeuroScan 3D v15 -- structurally different architecture for the E132
methodology-transfer test.

PURPOSE. Not a performance attempt. v15 exists to answer one question:

    Does the causal loss-localisation methodology developed on UNet3D_v5
    (E48-E131) transfer to a network built differently?

If the effective-rank <-> bottleneck-necessity coupling only ever appears at
one hard-coded layer of one network, it is an artifact of v5. If the same
do(X) procedure localises loss sites in a structurally different network,
and the coupling appears at whichever site the ablation identifies, then the
finding is about segmentation networks rather than about v5.

WHAT IS STRUCTURALLY DIFFERENT (two changes, both load-bearing):

  1. DEPTH: 5 encoder levels instead of 4.
     128 -> 64 -> 32 -> 16 -> 8   (four downsampling transitions)
     v5 has three (64 -> 32 -> 16 -> 8 at 128^3 input).
     So v15 has a downsampling site v5 does not have, and its bottleneck
     sits one level deeper.

  2. DOWNSAMPLING OPERATOR: LEARNED strided convolution, not fixed MaxPool.
     This is the decisive difference. Every prior finding in this project
     concerns what MaxPool -- a fixed, non-learnable, winner-take-all
     operation -- discards. A strided conv is learnable and is a weighted
     sum rather than a selection, so it CAN in principle preserve any
     linear combination of the 8 values.
     If the rank coupling still appears, the claim is no longer
     "MaxPool throws information away"; it is the broader
     "downsampling transitions destroy task-relevant information in a way
     that is measurable in situ and predicts downstream necessity".

WHAT IS DELIBERATELY HELD CONSTANT so the comparison is interpretable:
  - GroupNorm is NOT used. v15 keeps BatchNorm, matching v5, so any
    difference is attributable to depth and downsampling operator rather
    than to normalisation. (GroupNorm would be the natural choice at batch
    1 and would probably train better -- that is exactly why it is not
    used here. A confounded transfer test is worthless.)
  - Same loss, same 4-in/3-out task, same training protocol.
  - Same Conv3DBlock as the whole v1->v5 lineage.

NAMED STAGES. The diagnostic scripts need stable attribute names. v15
exposes exactly the same naming pattern as v5 so the E121/E129-style
ablation code needs only a stage-list change, not a rewrite:
    enc1..enc5, down1..down4 (the strided convs), bottleneck,
    up4..up1, dec4..dec1
Note v5 calls its downsamplers pool1..pool3; v15 calls them down1..down4
because they are convolutions, not pooling. That rename is deliberate --
it prevents a diagnostic script from silently treating them as MaxPool.

NO NOVELTY IS CLAIMED. A 5-level UNet with strided-conv downsampling is
entirely standard (nnU-Net, SegResNet and most modern 3D segmentation
networks use learned strided downsampling). Being standard is the point:
a transfer test should use an ordinary architecture, not an exotic one.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroscan_3d_fixed import Conv3DBlock


class StridedDown(nn.Module):
    """Learned downsampling: stride-2 conv + norm + ReLU.

    Contrast with MaxPool3d(2), which is fixed, non-learnable and selects
    one of 8 values. This computes a learned weighted sum over the 2x2x2
    neighbourhood and can represent any linear aggregation of it.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UNet3D_v15(nn.Module):
    """5-level encoder-decoder, learned strided downsampling, BatchNorm.

    Channel schedule 16/32/64/128/256 keeps the parameter count close to
    v5's 5.6M despite the extra level, so E132 is not confounded by
    capacity (E128 already established that 3x capacity changes nothing,
    but matching anyway removes the question).
    """

    def __init__(self, in_channels=1, out_channels=1, base=16):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16

        # ---- encoder ----
        self.enc1 = nn.Sequential(Conv3DBlock(in_channels, c1), Conv3DBlock(c1, c1))
        self.down1 = StridedDown(c1, c1)
        self.enc2 = nn.Sequential(Conv3DBlock(c1, c2), Conv3DBlock(c2, c2))
        self.down2 = StridedDown(c2, c2)
        self.enc3 = nn.Sequential(Conv3DBlock(c2, c3), Conv3DBlock(c3, c3))
        self.down3 = StridedDown(c3, c3)
        self.enc4 = nn.Sequential(Conv3DBlock(c3, c4), Conv3DBlock(c4, c4))
        self.down4 = StridedDown(c4, c4)

        # ---- bottleneck (one level deeper than v5's) ----
        self.bottleneck = nn.Sequential(Conv3DBlock(c4, c5), Conv3DBlock(c5, c5))

        # ---- decoder ----
        self.up4 = nn.ConvTranspose3d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(Conv3DBlock(c4 * 2, c4), Conv3DBlock(c4, c4))
        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(Conv3DBlock(c3 * 2, c3), Conv3DBlock(c3, c3))
        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(Conv3DBlock(c2 * 2, c2), Conv3DBlock(c2, c2))
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(Conv3DBlock(c1 * 2, c1), Conv3DBlock(c1, c1))

        # ---- heads ----
        self.seg_head = nn.Sequential(nn.Conv3d(c1, out_channels, 1), nn.Sigmoid())
        self.evidential_head = nn.Conv3d(c1, out_channels * 2, 1)
        self.boundary_head = nn.Conv3d(c1, out_channels, 1)
        # deep supervision at the same relative depth v5 uses (D/4)
        self.aux_head3 = nn.Sequential(nn.Conv3d(c3, out_channels, 1), nn.Sigmoid())
        self.aux_head2 = nn.Sequential(nn.Conv3d(c2, out_channels, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        b = self.bottleneck(self.down4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        probs = self.seg_head(d1)
        ev = self.evidential_head(d1)
        alpha_raw, beta_raw = torch.chunk(ev, 2, dim=1)
        alpha = F.softplus(alpha_raw) + 1.0
        beta = F.softplus(beta_raw) + 1.0

        return {
            "probs": probs,
            "alpha": alpha,
            "beta": beta,
            "boundary_logit": self.boundary_head(d1.detach()),
            "dec1": d1, "dec2": d2, "dec3": d3, "dec4": d4,
            "aux_probs3": self.aux_head3(d3),
            "aux_probs2": self.aux_head2(d2),
            # v5 returns an attention map; v15 has no attention gate, so it
            # returns a constant-ones placeholder of the right shape rather
            # than omitting the key, so shared training code does not break.
            "attention_map": torch.ones_like(probs[:, :1]),
        }


if __name__ == "__main__":
    from neuroscan_3d_v5 import UNet3D_v5
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v5 = UNet3D_v5(4, 3)
    v15 = UNet3D_v15(4, 3)
    n5 = sum(p.numel() for p in v5.parameters())
    n15 = sum(p.numel() for p in v15.parameters())
    print(f"params  v5={n5:,}   v15={n15:,}   ratio={n15/n5:.2f}x")

    v15 = v15.to(dev).eval()
    x = torch.randn(1, 4, 128, 128, 128, device=dev)
    with torch.no_grad():
        o = v15(x)
    print("\noutput shapes:")
    for k, v in o.items():
        print(f"  {k:16} {tuple(v.shape)}")

    # the four downsampling sites, and the spatial resolution each maps between
    print("\ndownsampling transitions (the sites E132 will ablate):")
    with torch.no_grad():
        e1 = v15.enc1(x);                     print(f"  enc1  {tuple(e1.shape)}")
        p1 = v15.down1(e1); e2 = v15.enc2(p1); print(f"  down1 -> enc2  {tuple(e2.shape)}")
        p2 = v15.down2(e2); e3 = v15.enc3(p2); print(f"  down2 -> enc3  {tuple(e3.shape)}")
        p3 = v15.down3(e3); e4 = v15.enc4(p3); print(f"  down3 -> enc4  {tuple(e4.shape)}")
        p4 = v15.down4(e4); b = v15.bottleneck(p4); print(f"  down4 -> bneck {tuple(b.shape)}")
