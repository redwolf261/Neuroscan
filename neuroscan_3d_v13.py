"""
NeuroScan 3D v13 -- Rank-Adaptive Pooling (RAP) at the enc3->bottleneck
transition.

THE ONE-LINE IDEA
-----------------
MaxPool discards 7 of every 8 values. That is free when those 8 values are
redundant, and destructive when they are genuinely distinct. So measure the
distinctness -- with the exact statistic this project causally validated --
and only pay for a learned aggregation where max-pooling is actually
destroying something.

WHY THIS STATISTIC AND NOT ANOTHER GATE
---------------------------------------
This is NOT a generic attention gate with a learned scalar. The gating
signal is the participation ratio of the window's 8 sub-voxel vectors in
channel space:

    PR(w) = (sum_i lambda_i)^2 / sum_i lambda_i^2,   lambda = eig(G_w)
    G_w   = V_w V_w^T,  V_w in R^{8 x C}   (the 8 sub-voxels of window w)

PR is in [1, 8]: 1 when all 8 sub-voxels are collinear in channel space
(maximally redundant -- max-pooling loses nothing), 8 when they are mutually
orthogonal (maximally distinct -- max-pooling throws away 7 independent
directions).

That exact quantity was established CAUSALLY over E124-E126:
  E124  PR at pool3 predicts bottleneck necessity N_b (exploratory, rho=0.83)
  E125  replicated on a HELD-OUT split, partial rho=0.877 controlling size
  E126  dose-response causal test: collapsing PR toward 1 monotonically
        REDUCES N_b (0.272 -> 0.263 -> 0.244 -> 0.214 -> 0.168 across
        alpha=1.0/0.95/0.85/0.70/0.50), with the gentlest settings verified
        genuinely in-distribution (negative-N_b rate matching baseline)
  E129  the coupling survives 3x encoder capacity (rho 0.771 -> 0.757), so
        it is not a capacity artifact

MEASURED ON THE TRAINED 4-MODALITY BASELINE, before writing this file:
real enc3 PR spans 1.00-2.81 (median 1.32, std 0.24) -- NOT saturated. And
tumour-containing windows carry higher rank than background:
    tumour windows     n=2010   PR mean 1.602
    background windows n=18470  PR mean 1.344   Mann-Whitney p < 1e-300
So max-pooling is discarding distinct information preferentially WHERE THE
TUMOUR IS, and coasting on redundancy elsewhere. The gate has real signal to
act on; it is not gating on noise.

THE OPERATOR
------------
    y_w = (1 - a_w) * max(x_w) + a_w * g(x_w)
    a_w = sigmoid( s * (PR(w) - tau) )

  g   : learned aggregation over the 8 sub-voxels, implemented as a
        depthwise 2x2x2 stride-2 conv (one weight vector per channel over
        the 8 positions). Depthwise, not full, so it costs C*8 parameters
        rather than C*C*8 and cannot smuggle in channel mixing that the
        baseline's subsequent bottleneck conv already performs.
  tau : rank threshold, LEARNABLE, initialised to 1.45 (between the measured
        background mean 1.344 and tumour mean 1.602).
  s   : sharpness, LEARNABLE (via softplus to stay positive), init 4.0.

IDENTITY-AT-INIT PROPERTY (deliberate, and verified in __main__):
g is initialised to reproduce max-pool's behaviour as closely as a linear
map can -- uniform weights 1/8, i.e. mean-pooling -- and a_w is additionally
scaled by a learnable gain `lam` initialised to ZERO. So at step 0:

    y_w = max(x_w)  EXACTLY, bit-for-bit.

v13 therefore starts as a bit-exact copy of v5 and can only depart from it
by learning to. This is the same ablation-safety convention v5 itself used
for its attention gate (forcing the gate to identity reproduces v3 exactly),
and it means any Dice difference is attributable to the mechanism rather
than to a different initialisation.

WHAT IS DELIBERATELY NOT CLAIMED
--------------------------------
Adaptive/learned pooling is a populated field (soft-pool, mixed pooling,
gated pooling, LIP, DPP, and this project's own killed E93/E116/E117/E119
attempts at recombining the same 8 values). The narrow claim here is the
GATING SIGNAL: pooling gated on a causally-validated, label-blind rank
diagnostic of the specific window being pooled. The honest novelty
assessment belongs in the writeup, not in this docstring.

COST
----
PR is computed in CLOSED FORM as tr(G)^2/||G||_F^2 -- no eigensolver.
Measured 0.069 s fwd+bwd for the 32768 windows produced by a 128^3 input.
An eigendecomposition-based version crashed cusolver at that batch size
(the same failure E124 hit) and cost 2.33 s/iter where it did not crash.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroscan_3d_v5 import UNet3D_v5


def participation_ratio_windows(x, eps=1e-8):
    """x: (B,C,D,H,W) with even D,H,W. Returns (B, nW) participation ratio
    per non-overlapping 2x2x2 window, nW = (D/2)(H/2)(W/2).

    Computed from the 8x8 Gram of the window's 8 sub-voxel channel-vectors,
    NOT the CxC covariance: the nonzero spectrum is identical, the Gram is
    8x8 regardless of C (so the statistic stays on the same [1,8] scale for
    any channel width -- this is what made E129's v5-vs-v12 comparison
    valid), and it is vastly cheaper.

    NO EIGENDECOMPOSITION IS USED. PR needs only sum(lambda) and
    sum(lambda^2), and for any symmetric G both are traces:
        sum_i lambda_i   = tr(G)
        sum_i lambda_i^2 = tr(G^2) = ||G||_F^2   (Frobenius norm squared)
    so  PR = tr(G)^2 / ||G||_F^2.
    Verified numerically against torch.linalg.eigvalsh: max abs difference
    2.4e-06, i.e. float precision.

    This is not merely an optimisation. torch.linalg.eigvalsh CRASHES with
    CUSOLVER_STATUS_INVALID_VALUE past a batch limit -- exactly the failure
    E124 hit at 32768 windows -- and where it does not crash it silently
    falls back to a path that cost 2.33 s/iter in the E131 smoke test
    (8.3x the baseline, i.e. 36h for 50 epochs). The trace form has no
    batch limit and measured 0.069 s for the same 32768 windows.
    """
    B, C, D, H, W = x.shape
    u = x.unfold(2, 2, 2).unfold(3, 2, 2).unfold(4, 2, 2)
    u = u.contiguous().view(B, C, -1, 8)          # (B,C,nW,8)
    v = u.permute(0, 2, 3, 1).float()             # (B,nW,8,C) in fp32:
    # under AMP the activations are fp16 and an 8x8 Gram of fp16 values
    # loses precision in the squared terms.
    g = torch.matmul(v, v.transpose(-1, -2))      # (B,nW,8,8)
    tr = torch.diagonal(g, dim1=-2, dim2=-1).sum(-1)      # sum lambda
    fro = (g ** 2).sum((-1, -2))                          # sum lambda^2
    return torch.where(fro > eps, tr ** 2 / (fro + eps), torch.ones_like(tr))


class RankAdaptivePool3d(nn.Module):
    """Drop-in replacement for nn.MaxPool3d(2) that blends toward a learned
    depthwise aggregation in windows whose effective rank is high."""

    def __init__(self, channels, tau_init=1.45, sharp_init=4.0, lam_init=0.05):
        super().__init__()
        self.channels = channels
        # Depthwise 2x2x2 stride-2 conv = one learned weight per (channel,
        # sub-voxel position). Initialised to uniform 1/8 == mean-pooling.
        self.agg = nn.Conv3d(channels, channels, kernel_size=2, stride=2,
                             groups=channels, bias=False)
        with torch.no_grad():
            self.agg.weight.fill_(1.0 / 8.0)
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        self.sharp_raw = nn.Parameter(torch.tensor(float(sharp_init)))
        # lam scales the whole blend. lam=0 gives y == max-pool EXACTLY,
        # but VERIFIED BEFORE TRAINING: at lam=0 the gradients of tau,
        # sharp_raw and agg.weight are all identically ZERO (they are
        # multiplied by lam), and lam's own gradient is only ~4e-06. The
        # mechanism could plausibly never bootstrap within 50 epochs, and
        # v13 would silently reduce to v5 plus dead parameters.
        # So lam starts SMALL BUT NON-ZERO: near-identity (the blend is
        # capped at 5% of the learned aggregation) while every parameter
        # has live gradient from step 0. The exact-identity property is
        # still available and still tested -- set lam_init=0.0.
        self.lam = nn.Parameter(torch.full((1,), float(lam_init)))

    def forward(self, x):
        mx = F.max_pool3d(x, 2)
        if float(self.lam.detach().abs()) == 0.0 and not self.training:
            return mx  # exact identity fast-path at eval before any training
        B, C, D, H, W = x.shape
        # PR IS COMPUTED WITHOUT GRADIENT, deliberately.
        # It is a DIAGNOSTIC of the window -- E124-E126 established it as a
        # measurement that predicts and causally drives N_b, not as a
        # differentiable feature. Backprop through the unfold/Gram at 32768
        # windows x 128 channels is a non-coalesced gather that MEASURED
        # 8.141 s/iter vs the v5 baseline's 0.634 (12.8x), at IDENTICAL
        # memory (2.78 vs 2.74 GB allocated, 4.40 vs 4.34 reserved) and
        # 46.5W -- real compute, not allocator thrashing.
        # Detaching keeps the mechanism fully live: tau, sharp and lam still
        # receive gradient through the sigmoid and the blend, and agg through
        # the aggregation path. Only the route back through PR's own
        # construction is cut, which is the route we never needed.
        with torch.no_grad():
            pr = participation_ratio_windows(x)                   # (B,nW)
        a = torch.sigmoid(F.softplus(self.sharp_raw) * (pr - self.tau))
        a = a.view(B, 1, D // 2, H // 2, W // 2)
        a = a * self.lam
        return (1.0 - a) * mx + a * self.agg(x)

    def extra_repr(self):
        return (f"channels={self.channels}, tau={float(self.tau):.3f}, "
                f"lam={float(self.lam):.4f}")


class RankGatedPool3d(nn.Module):
    """v14 variant: the SAME operator as RankAdaptivePool3d but with the
    global lambda escape hatch REMOVED.

    WHY. In E131, v13's lambda decayed monotonically 0.1075 -> 0.0073 over
    22 epochs, i.e. the optimizer switched the mechanism off and reverted to
    plain max-pool. Weight decay was RULED OUT as the cause by direct
    calculation: AdamW decoupled wd at lr=4e-4, wd=1e-5 over 12386 steps
    gives a decay factor of 0.99995, so it would have moved lambda from
    0.1075 to 0.10749. The observed factor was 0.068. The decay is therefore
    entirely LOSS-DRIVEN.

    That is informative but confounded: lambda is ONE scalar multiplying the
    whole blend, so suppressing it is a far cheaper descent direction than
    learning 1024 aggregation weights. The optimizer took the escape route
    before the mechanism could ever demonstrate value. v14 removes the
    route, so the blend is governed ONLY by the per-window PR sigmoid:

        y_w = (1 - a_w) * max(x_w) + a_w * g(x_w),  a_w = sigmoid(s*(PR_w - tau))

    tau and s remain learnable, so the network can still tune WHICH windows
    are aggregated, and can still push tau high enough to gate almost
    everything out -- but that now costs it a genuinely different solution
    rather than one scalar going to zero.

    agg is initialised to uniform 1/8 (mean-pooling), which is a sensible
    neutral aggregation, NOT an identity to max-pool. v14 therefore does not
    have v13's bit-exact-identity-at-init property. That is the deliberate
    trade: v13 tested "will the optimizer adopt this mechanism if given a
    free choice" (answer: no). v14 tests "does this mechanism help when the
    network must actually use it".
    """

    def __init__(self, channels, tau_init=1.45, sharp_init=4.0):
        super().__init__()
        self.channels = channels
        self.agg = nn.Conv3d(channels, channels, kernel_size=2, stride=2,
                             groups=channels, bias=False)
        with torch.no_grad():
            self.agg.weight.fill_(1.0 / 8.0)
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        self.sharp_raw = nn.Parameter(torch.tensor(float(sharp_init)))

    def forward(self, x):
        mx = F.max_pool3d(x, 2)
        B, C, D, H, W = x.shape
        with torch.no_grad():
            pr = participation_ratio_windows(x)
        a = torch.sigmoid(F.softplus(self.sharp_raw) * (pr - self.tau))
        a = a.view(B, 1, D // 2, H // 2, W // 2)
        return (1.0 - a) * mx + a * self.agg(x)

    def extra_repr(self):
        return (f"channels={self.channels}, tau={float(self.tau):.3f}, "
                f"NO_GLOBAL_LAMBDA")


class UNet3D_v14(UNet3D_v5):
    """v5 with pool3 -> RankGatedPool3d (no global lambda). See that class."""

    def __init__(self, in_channels=1, out_channels=1, tau_init=1.45, sharp_init=4.0):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        enc3_out = self.enc3[-1].conv.out_channels
        assert enc3_out == 128, f"expected enc3 to emit 128ch, got {enc3_out}"
        self.pool3 = RankGatedPool3d(enc3_out, tau_init, sharp_init)


class UNet3D_v13(UNet3D_v5):
    """v5 with pool3 replaced by RankAdaptivePool3d.

    Only pool3 is replaced. E121 measured N_k at all three transitions and
    found the per-stage MAGNITUDES are run-dependent across equally-good
    checkpoints, so "N1~=N3>>N2" is not a stable target -- but the pool3
    effective-rank coupling replicated across THREE independent checkpoints
    (fp32 canonical, AMP v5, AMP v12) and is the durable finding. So the
    intervention goes where the evidence actually is, and pool1/pool2 stay
    untouched as internal controls.
    """

    def __init__(self, in_channels=1, out_channels=1, tau_init=1.45, sharp_init=4.0,
                 lam_init=0.05):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        # enc3 outputs 128 channels in the v5 lineage; assert rather than
        # assume, so a future width change fails loudly instead of silently
        # building a mis-sized depthwise conv.
        enc3_out = self.enc3[-1].conv.out_channels
        assert enc3_out == 128, f"expected enc3 to emit 128ch, got {enc3_out}"
        self.pool3 = RankAdaptivePool3d(enc3_out, tau_init, sharp_init, lam_init)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. PR sanity: collinear window -> 1, orthogonal window -> 8
    C = 16
    x_col = torch.zeros(1, C, 2, 2, 2)
    base = torch.randn(C)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x_col[0, :, i, j, k] = base * (i + j + k + 1)
    print("collinear window PR :", float(participation_ratio_windows(x_col)[0, 0]), "(expect ~1)")

    x_orth = torch.zeros(1, 8, 2, 2, 2)
    p = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x_orth[0, p, i, j, k] = 1.0
                p += 1
    print("orthogonal window PR:", float(participation_ratio_windows(x_orth)[0, 0]), "(expect ~8)")

    # 2. THE CRITICAL PROPERTY: v13 must equal v5 bit-for-bit at init.
    v5 = UNet3D_v5(4, 3).to(dev).eval()
    v13 = UNet3D_v13(4, 3, lam_init=0.0).to(dev).eval()   # exact-identity variant
    sd = v5.state_dict()
    v13.load_state_dict(sd, strict=False)   # copy shared weights
    xb = torch.randn(1, 4, 64, 64, 64, device=dev)
    with torch.no_grad():
        a = v5(xb)["probs"]
        b = v13(xb)["probs"]
    print("v13 == v5 at init   :", torch.equal(a, b), "| max|diff| =",
          float((a - b).abs().max()))

    # 3. Once lam != 0 it must actually diverge (mechanism is live).
    with torch.no_grad():
        v13.pool3.lam.fill_(0.5)
        c = v13(xb)["probs"]
    print("lam=0.5 diverges    :", not torch.equal(a, c), "| max|diff| =",
          float((a - c).abs().max()))

    # 4. default (lam_init=0.05) must be NEAR identity but not exact
    v13d = UNet3D_v13(4, 3).to(dev).eval()
    v13d.load_state_dict(sd, strict=False)
    with torch.no_grad():
        d = v13d(xb)["probs"]
    print("default lam=0.05    : near-identity, max|diff| =", float((a - d).abs().max()))

    # 5. v14: no global lambda -> gate is active from step 0 and cannot be
    #    globally switched off.
    v14 = UNet3D_v14(4, 3).to(dev).eval()
    v14.load_state_dict(sd, strict=False)
    with torch.no_grad():
        e = v14(xb)["probs"]
    print("v14 active at init  : differs from v5, max|diff| =", float((a - e).abs().max()))
    names = {n for n, _ in v14.pool3.named_parameters()}
    print("v14 has no lam param:", "lam" not in names, "| params:", sorted(names))

    n5 = sum(p.numel() for p in v5.parameters())
    n13 = sum(p.numel() for p in v13.parameters())
    print(f"params v5={n5:,}  v13={n13:,}  (+{n13-n5:,})")
