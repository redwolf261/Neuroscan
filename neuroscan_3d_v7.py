"""
NeuroScan 3D v7 -- v3 architecture + Internally-Estimated Counterfactual
Gating (IECG).

NOVELTY CLAIM (see PHASE_E50_IECG_DESIGN.md for the full argument and
literature comparison): CCABA (v6, E49) used a FIXED, externally-fit
calibration curve (from E48's offline 125-subject causal audit) to
condition a bottleneck-amplification gate. This is real but static --
the causal-sensitivity signal is computed ONCE, before training, and
never updates. A 2025-2026 literature scan for "self-estimated
sensitivity gate" / "internal counterfactual module" / "live ablation
estimate" found NOTHING matching this specific combination: every
causal/counterfactual method found (CausalX-Net, counterfactual MoE
routing analysis, TRACE-Seg3D's context auditing) is a POST-HOC
analysis/diagnostic tool applied to an already-trained, FROZEN model --
none are trainable, differentiable modules that run live during the
forward pass and are optimized jointly with the task loss.

IECG is a genuinely new mechanism class (not a variant of CCABA, E46, or
anything found in the literature scan): during EVERY training forward
pass, the model:
  1. Computes its own REAL bottleneck (as always).
  2. Computes a CHEAP SIMULATED ABLATION of that exact bottleneck (full
     zeroing -- the SAME intervention E48 already causally validated,
     reused not reinvented) and runs the decoder a SECOND time on the
     ablated bottleneck, producing probs_ablated.
  3. Computes s_true = Dice(probs, mask) - Dice(probs_ablated, mask) --
     the REAL, freshly-measured, per-subject, per-batch causal
     sensitivity of this exact input to bottleneck ablation (literally
     E48's own metric, computed online during training, not from a
     static offline table).
  4. A lightweight SensitivityPredictorHead reads ONLY the real
     bottleneck (never the ablated one) and predicts s_hat, trained via
     MSE against s_true (stop-gradient on the target -- s_true is a
     LABEL for this head, not a differentiable path back through the
     ablated decoder pass).
  5. s_hat gates bottleneck's OWN contribution: bottleneck_amp =
     bottleneck * (1 + alpha * s_hat), alpha a learnable scalar (same
     amplification form as CCABA, but s_hat is now a LIVE, PER-INPUT,
     JOINTLY-LEARNED estimate, not a fixed external curve).

AT INFERENCE: only the REAL bottleneck + the trained
SensitivityPredictorHead run. The ablated decoder pass is
TRAINING-ONLY machinery (needed to generate s_true targets) -- it is
NEVER computed at inference, so IECG adds ZERO extra inference-time
cost beyond one small head, a genuine practical advantage over
re-running any real ablation at deployment time.

TRAINING-TIME COST: the ablated decoder pass roughly adds one extra
decoder-only forward pass per training step (encoder is shared, computed
once) -- a real, measured, disclosed training-time overhead, verified
directly (not assumed) in calibrate_iecg.py before any real training run.

SAFETY PROPERTY: with alpha=0, bottleneck_amp = bottleneck exactly, so
forward() (in INFERENCE mode, ablation branch skipped) must reproduce
v3's own probs/aux_probs3/aux_probs2 bit-for-bit -- same discipline as
v5's attn_gate1 and v6's CCABA.

v1/v2/v3/v4/v5/v6 all remain permanent, git-tracked, NEVER edited
further. This file extends v3 (not v5 or v6) to isolate IECG's own
effect from either E46's attention gate or E49's CCABA.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroscan_3d_v3 import UNet3D_v3  # noqa: F401


class SensitivityPredictorHead(nn.Module):
    """Predicts a scalar per-subject causal-sensitivity estimate (s_hat)
    from the REAL bottleneck ONLY -- never sees the ablated bottleneck,
    so at inference (where no ablation is ever computed) this head still
    has everything it needs to produce s_hat."""

    def __init__(self, in_channels=256):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=1)
        self.conv2 = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, bottleneck):
        h = F.relu(self.conv1(bottleneck))
        s_map = self.conv2(h)  # (B,1,8,8,8)
        s_hat = s_map.mean(dim=(1, 2, 3, 4))  # (B,) scalar per subject
        return s_hat


class UNet3D_v7(UNet3D_v3):
    """v3's UNet3D_v3 + Internally-Estimated Counterfactual Gating
    (IECG). Adds a decoder-replaying mechanism used ONLY during training
    (see forward()'s `compute_sensitivity_target` flag) to generate a
    live causal-sensitivity training target, and a small head that
    learns to predict that sensitivity from the real bottleneck alone."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        self.sensitivity_head = SensitivityPredictorHead(in_channels=256)
        self.iecg_alpha = nn.Parameter(torch.tensor(0.1))

    def _run_decoder_from_bottleneck(self, bottleneck, enc3, enc2, enc1):
        """Runs the SHARED decoder trunk (upconv3/dec3/upconv2/dec2/
        upconv1/dec1/seg_head) from an arbitrary bottleneck tensor --
        factored out so it can be called TWICE (real, ablated) from the
        same encoder features without duplicating the encoder's own
        (expensive, shared) computation. Returns probs only (the aux
        heads / evidential / boundary heads are NOT recomputed for the
        ablated branch -- only probs is needed to measure s_true via
        Dice, per the pre-declared design; recomputing every head for
        the ablated branch would be wasted compute with no use)."""
        upconv3 = self.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = self.dec3(cat3)

        upconv2 = self.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = self.dec2(cat2)

        upconv1 = self.upconv1(dec2)
        cat1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = self.dec1(cat1)

        probs = self.seg_head(dec1)
        return probs, dec1, dec2, dec3

    def forward(self, x, masks=None, compute_sensitivity_target=False):
        """
        masks: (B,1,D,H,W) ground truth, REQUIRED only when
            compute_sensitivity_target=True (training mode) -- used to
            compute s_true via a real Dice comparison between the
            intact and ablated decoder passes. NOT used at all when
            compute_sensitivity_target=False (inference/eval mode) --
            inference NEVER needs masks, matching every prior
            architecture's own convention.

        Returns everything v3's forward() returns, plus:
            's_hat': (B,) the sensitivity head's live prediction (always
                computed, cheap, used at both train and inference time).
            's_true': (B,) the REAL measured sensitivity from this
                training step's own on-the-fly ablation (ONLY present
                when compute_sensitivity_target=True; None otherwise).
            'iecg_alpha': scalar, learned gate strength (diagnostic).

        The ONLY behavioral change vs v3 in the PRIMARY (non-ablated)
        path: bottleneck is replaced by bottleneck_amp = bottleneck *
        (1 + alpha*s_hat) before upconv3, identical in FORM to CCABA's
        own amplification (v6) but with s_hat as the live, per-input,
        jointly-learned estimate instead of a fixed external curve.
        """
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        bottleneck = self.bottleneck(pool3)

        s_hat = self.sensitivity_head(bottleneck)  # (B,), ALWAYS computed (train + inference)

        gain = (1.0 + self.iecg_alpha * s_hat).view(-1, 1, 1, 1, 1)
        bottleneck_amp = bottleneck * gain

        probs, dec1, dec2, dec3 = self._run_decoder_from_bottleneck(bottleneck_amp, enc3, enc2, enc1)

        s_true = None
        if compute_sensitivity_target:
            assert masks is not None, "masks required when compute_sensitivity_target=True"
            with torch.no_grad():
                bottleneck_ablated = torch.zeros_like(bottleneck)  # SAME intervention as E48's real causal audit
                probs_ablated, _, _, _ = self._run_decoder_from_bottleneck(bottleneck_ablated, enc3, enc2, enc1)

                # Per-subject Dice, matching this project's own hard-threshold convention
                pred_bin = (probs.detach() >= 0.5).float()
                pred_ablated_bin = (probs_ablated >= 0.5).float()
                mask_bin = masks

                def batched_dice(p, t):
                    dims = (1, 2, 3, 4)
                    tp = (p * t).sum(dim=dims)
                    denom = p.sum(dim=dims) + t.sum(dim=dims)
                    return torch.where(denom > 0, 2 * tp / denom.clamp(min=1e-8), torch.ones_like(tp))

                dice_intact = batched_dice(pred_bin, mask_bin)
                dice_ablated = batched_dice(pred_ablated_bin, mask_bin)
                s_true = (dice_intact - dice_ablated).detach()  # (B,), a LABEL, not a differentiable path

        evidential_raw = self.evidential_head(dec1)
        alpha_raw, beta_raw = torch.chunk(evidential_raw, 2, dim=1)
        alpha = F.softplus(alpha_raw) + 1.0
        beta = F.softplus(beta_raw) + 1.0

        boundary_logit = self.boundary_head(dec1.detach())

        aux_probs3 = self.aux_head3(dec3)
        aux_probs2 = self.aux_head2(dec2)

        return {
            "probs": probs,
            "alpha": alpha,
            "beta": beta,
            "boundary_logit": boundary_logit,
            "dec1": dec1,
            "dec2": dec2,
            "dec3": dec3,
            "aux_probs3": aux_probs3,
            "aux_probs2": aux_probs2,
            "s_hat": s_hat,
            "s_true": s_true,
            "iecg_alpha": self.iecg_alpha.detach(),
        }


__all__ = ["UNet3D_v7", "SensitivityPredictorHead"]
