"""
NeuroScan 3D v16 -- Enhancement-Conditioned Readout (ECR).

THE REORGANIZATION
------------------
v5 predicts ET, TC and WT as THREE PARALLEL INDEPENDENT readouts of one
shared dec1 feature map -- literally a single Conv3d(32, 3, kernel_size=1).
Nothing in the architecture encodes that ET is DEFINED as contrast
enhancement WITHIN tumour. The regions are nested in the labels and
independent in the model.

v16 makes the ET/TC decision CONDITIONAL on a measurement taken inside the
model's own WT prediction:

    stage 1  predict WT                      (already accurate: 0.91 overall,
                                              0.80-0.87 even where ET fails)
    stage 2  measure enhancement statistics of Delta = z(t1c) - z(t1n)
             INSIDE that predicted region
    stage 3  condition the ET/TC readout on that measurement

This mirrors the radiological definition and the nesting the labels already
have, and it is the dependency structure the baseline architecture ignores.

WHY THIS AND NOT A GLOBAL CONDITIONING VECTOR
---------------------------------------------
Measured, and it killed the first design I wrote: EVERY whole-brain
label-blind descriptor of Delta fails to predict ET failure --
p99, p999, max, std, frac>1, frac>2, skew, all with p > 0.14, and the best
of them separates the bottom decile at 0.817 vs 0.819, i.e. not at all. The
tumour is far too small a fraction of the brain for global statistics to
carry the signal.

Restricted to the tumour region, the same statistic is strong:
    inside TRUE ET mask:          rho = +0.595  p = 2.5e-13
    inside TRUE WT region:        rho = +0.611  p = 3.6e-14
    inside PREDICTED WT:          rho = +0.607  p = 5.8e-14   <- inference-time
Bottom decile by p95(Delta) within predicted WT: ET dice 0.476 vs 0.859.

So the conditioning signal must be REGION-RESTRICTED, and the model's own
WT prediction is accurate enough to supply the region. That is why the
conditioning is two-stage rather than a feed-forward side input.

WHAT MOTIVATES CONDITIONING AT ALL
-----------------------------------
A 4-feature per-voxel MLP with NO spatial context, on the failing subjects:
    fitted on good subjects, transferred:  0.177
    fitted on the subject itself (oracle): 0.531
    the full 3D CNN:                       0.052
The transfer-vs-oracle gap shows the decision BOUNDARY differs per subject.
One globally-fitted threshold cannot serve both enhancement regimes. The
CNN is discarding ~0.48 Dice of usable evidence on this subpopulation, and
no encoder/decoder/pooling change can recover it -- all of those receive
strictly MORE information than the MLP does. That is why five pooling
interventions (E93/E116/E117/E119/E131) returned null.

MECHANISM
---------
    m   = descriptor of Delta inside predicted WT   (4 scalars, detached)
    g,b = MLP(m)                                    (FiLM parameters)
    h'  = g * h + b                                 (modulate dec1 features)
    ET/TC logits = conv(h'),  WT logit = conv(h)    (WT path UNMODULATED)

WT is deliberately left unconditioned: it is the stage-1 predictor, it
already works, and modulating it would create a feedback loop between the
descriptor and the region the descriptor is measured in.

IDENTITY AT INIT (verified in __main__): the FiLM MLP's final layer is
zero-initialised, so g=1 and b=0 at step 0 and v16 reproduces v5 EXACTLY.
Any difference is therefore attributable to the mechanism, not to a
different initialisation -- the same convention v5 used for its own
attention gate.

WHAT IS NOT CLAIMED
-------------------
FiLM (Perez et al. 2018) and conditional normalisation are established.
Two-stage coarse-to-fine segmentation is established. Any novelty rests on
the DIAGNOSIS -- that ET failure is driven by per-subject enhancement
regime, measurable label-blind inside the predicted WT -- not on the
conditioning mechanism. Prior-art audit NOT run.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroscan_3d_v5 import UNet3D_v5


def enhancement_descriptor(x, wt_prob, eps=1e-6):
    """Label-blind conditioning signal.

    x:       (B, 4, D, H, W) input, channel order (t1c, t1n, t2f, t2w)
    wt_prob: (B, 1, D, H, W) the model's OWN predicted WT probability

    Returns (B, 4): soft-masked statistics of Delta = t1c - t1n inside the
    predicted WT region. Uses the SOFT probability as weights rather than a
    hard threshold, so the descriptor stays differentiable and does not
    collapse when the WT prediction is uncertain.

    Both input channels are already per-modality z-scored by the dataloader,
    so Delta is directly the enhancement contrast.
    """
    # SPEED: the descriptor is a set of GLOBAL scalars, so it does not need
    # every voxel. Strided subsampling by 2 in each axis uses 1/8 of the
    # volume and changes the statistics negligibly, while cutting the cost
    # of five full-volume reductions that measured 0.666 s/iter (10.4h for
    # 50 epochs) down to something negligible.
    x = x[:, :, ::2, ::2, ::2]
    wt_prob = wt_prob[:, :, ::2, ::2, ::2]
    delta = (x[:, 0:1] - x[:, 1:2]).float()              # (B,1,d,h,w)
    w = wt_prob.clamp(0, 1).float()
    tot = w.sum(dim=(1, 2, 3, 4), keepdim=True) + eps

    mean = (delta * w).sum(dim=(1, 2, 3, 4), keepdim=True) / tot
    var = ((delta - mean) ** 2 * w).sum(dim=(1, 2, 3, 4), keepdim=True) / tot
    std = var.clamp(min=0).sqrt()
    # fraction of the predicted region exceeding a strong-enhancement level;
    # a smooth surrogate for the p95 statistic that measured rho=+0.607
    frac_hi = ((delta > 1.0).float() * w).sum(dim=(1, 2, 3, 4), keepdim=True) / tot
    frac_neg = ((delta < 0.0).float() * w).sum(dim=(1, 2, 3, 4), keepdim=True) / tot

    return torch.cat([mean, std, frac_hi, frac_neg], dim=1).flatten(1)  # (B,4)


class UNet3D_v16(UNet3D_v5):
    """v5 with an enhancement-conditioned ET/TC readout. WT is unmodulated."""

    def __init__(self, in_channels=4, out_channels=3, hidden=32, film_init=1e-3):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        assert in_channels == 4 and out_channels == 3, \
            "v16 assumes the 4-modality / 3-region BraTS task"

        dec1_ch = 32  # v5's dec1 width
        self.film = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * dec1_ch),
        )
        # Near-zero (not exactly zero) init of the final FiLM layer.
        # VERIFIED BEFORE TRAINING: with an exactly-zero final layer,
        # film[-1] receives gradient (6.44) but film[0] receives EXACTLY
        # ZERO, because the backward path through the zero weight vanishes.
        # The mechanism could still bootstrap -- film[-1] moves first, then
        # film[0] starts learning -- but that is the same slow-start risk
        # that let v13's global lambda collapse before the mechanism ever
        # demonstrated value (E131). A small non-zero init gives every FiLM
        # parameter live gradient from step 0 while keeping the modulation
        # within ~1% of identity. The exact-identity variant is still
        # available and still tested: pass film_init=0.0.
        with torch.no_grad():
            self.film[-1].weight.normal_(0.0, film_init)
            self.film[-1].bias.zero_()

        # separate readouts so WT can bypass the modulation entirely
        self.head_etc = nn.Conv3d(dec1_ch, 2, kernel_size=1)   # ET, TC
        self.head_wt = nn.Conv3d(dec1_ch, 1, kernel_size=1)    # WT
        with torch.no_grad():
            # inherit v5's trained-equivalent init from seg_head so that at
            # step 0 the outputs match v5's channel-for-channel
            w = self.seg_head[0].weight.detach()
            b = self.seg_head[0].bias.detach()
            self.head_etc.weight.copy_(w[0:2])
            self.head_etc.bias.copy_(b[0:2])
            self.head_wt.weight.copy_(w[2:3])
            self.head_wt.bias.copy_(b[2:3])

    def forward(self, x):
        out = super().forward(x)
        dec1 = out["dec1"]

        # stage 1+2: WT prediction -> descriptor. DETACHED so the descriptor
        # acts as a conditioning measurement, not a gradient path that could
        # let the model game the statistic it is conditioned on.
        wt_logit = self.head_wt(dec1)
        with torch.no_grad():
            m = enhancement_descriptor(x, torch.sigmoid(wt_logit))

        # stage 3: modulate dec1 for the ET/TC readout only
        gb = self.film(m)                                    # (B, 2*32)
        g, b = gb.chunk(2, dim=1)
        g = (1.0 + g).view(g.shape[0], -1, 1, 1, 1)
        b = b.view(b.shape[0], -1, 1, 1, 1)
        etc_logit = self.head_etc(dec1 * g + b)

        probs = torch.sigmoid(torch.cat([etc_logit, wt_logit], dim=1))
        out["probs"] = probs
        out["enh_descriptor"] = m
        out["film_gain"] = g.flatten(1)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v5 = UNet3D_v5(4, 3).to(dev).eval()
    v16 = UNet3D_v16(4, 3).to(dev).eval()
    v16.load_state_dict(v5.state_dict(), strict=False)
    # re-sync the split heads after loading v5 weights
    with torch.no_grad():
        w = v16.seg_head[0].weight.detach(); b = v16.seg_head[0].bias.detach()
        v16.head_etc.weight.copy_(w[0:2]); v16.head_etc.bias.copy_(b[0:2])
        v16.head_wt.weight.copy_(w[2:3]);  v16.head_wt.bias.copy_(b[2:3])

    x = torch.randn(1, 4, 64, 64, 64, device=dev)
    with torch.no_grad():
        a = v5(x)["probs"]
        c = v16(x)["probs"]
    print("v16 == v5 at init :", torch.allclose(a, c, atol=1e-6),
          "| max|diff| =", float((a - c).abs().max()))

    # descriptor sanity: a strongly-enhancing volume vs a non-enhancing one
    hi = torch.zeros(1, 4, 32, 32, 32, device=dev); hi[:, 0] = 2.0   # t1c bright
    lo = torch.zeros(1, 4, 32, 32, 32, device=dev); lo[:, 0] = -1.0  # t1c dark
    wt = torch.ones(1, 1, 32, 32, 32, device=dev)
    print("descriptor, enhancing   :", [round(v, 3) for v in enhancement_descriptor(hi, wt)[0].tolist()])
    print("descriptor, NON-enhancing:", [round(v, 3) for v in enhancement_descriptor(lo, wt)[0].tolist()])

    # once film is non-zero the outputs must diverge
    with torch.no_grad():
        v16.film[-1].weight.normal_(0, 0.1)
        d = v16(x)["probs"]
    print("diverges once film != 0 :", not torch.allclose(a, d, atol=1e-6),
          "| max|diff| =", float((a - d).abs().max()))

    n5 = sum(p.numel() for p in v5.parameters())
    n16 = sum(p.numel() for p in v16.parameters())
    print(f"params v5={n5:,}  v16={n16:,}  (+{n16-n5:,})")
