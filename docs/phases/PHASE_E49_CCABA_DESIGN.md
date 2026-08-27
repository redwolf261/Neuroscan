# Phase E49 — Causally-Calibrated Adaptive Bottleneck Amplification (CCABA): Design & Literature Positioning

## Motivating chain

- **E43** (correlational): bottleneck representation change is real under D4 supervision, but not shown to be boundary-localized after confound correction. NULL on localization.
- **E47** (causal): clamping the E46 attention gate at boundary voxels vs. matched-count interior voxels produces no significant Dice-drop difference (p=0.808). Causally confirms E43's null — boundary-routing is not the mechanism.
- **E48** (causal, reversed finding): fully ablating the bottleneck hurts **small**-lesion subjects significantly more than large-lesion subjects (Spearman rho(native_size, drop) = −0.454, p<0.001, n=125). The bottleneck/global-context pathway is causally, disproportionately important for small lesions.

E48 is the first genuinely new causal mechanism found in this diagnostic chain, and is the basis for this design.

## Literature scan (2025–2026), before committing to a mechanism

Searched for: size-adaptive/scale-aware attention for small lesions, size-conditioned deep supervision/gating, dynamic gating with global context, causal-ablation-guided architecture design, ablation-informed gating.

**Size-aware segmentation mechanisms found** — all use an assumed prior or heuristic, not a measurement:
- **S³-Mamba** (AAAI 2025) — curriculum learning with a "Difficulty Measurer" that *initializes* sample weights based on lesion size — an assumed proxy for difficulty, not a measured causal dependency. [arxiv.org/html/2412.14546v1](https://arxiv.org/html/2412.14546v1)
- **SvANet** (Scale-Variant Attention Network) — scale-variant attention + cross-scale guidance, architecturally motivated, not causally calibrated. [arxiv.org/pdf/2407.07720](https://arxiv.org/pdf/2407.07720)
- **M⁴Fuse** — cross-scale gating bridge (MoE-style), gating learned end-to-end without a causal measurement step. [arxiv.org/pdf/2605.02444](https://arxiv.org/pdf/2605.02444)
- **DCSNet**, **SGDC** — detection-guided cropping / structurally-guided dynamic convolution, both architectural-prior-driven.

**Causal-interpretability methods found** — target attribution/robustness, not architecture design:
- **CausalX-Net** (2025, BraTS, 92.5% Dice) — do-calculus interventions at the **modality** level (T1/T2/FLAIR/T1CE) and per-voxel regional clamping, producing causal-effect maps for **explanation and spurious-correlation removal**. Confirmed via direct fetch of the full method section: does *not* correlate causal effect with lesion size, and its SCM layer is for attribution, not for conditioning a pathway's capacity. [pmc.ncbi.nlm.nih.gov/articles/PMC12593452](https://pmc.ncbi.nlm.nih.gov/articles/PMC12593452/)
- **Causal Head Gating** — soft-ablates attention heads to interpret their role in transformers (NLP), not applied to size-conditioning or medical segmentation.

**Gap identified**: no mechanism found uses a *measured, causally-verified, per-subject dependency curve* (drop vs. a covariate) as the literal specification of a gating/amplification function. Every size-aware mechanism assumes size matters and designs around that assumption architecturally; every causal method explains/attributes but doesn't feed back into architecture design. Two additional broadened searches (ablation-calibration-informed gating, empirical-dependency-function gate design) found nothing closer — "ablation study" in the standard ML sense (component removed, aggregate metric compared) is a different thing from what's proposed here (a *per-subject, per-covariate* causal measurement used as a *design specification*, not a validation check).

## Mechanism: CCABA

1. **Size-proxy head**: lightweight auxiliary head on the bottleneck (8³→1ch sigmoid, spatially pooled to a scalar `frac_hat` per subject) — an inference-time-available substitute for `native_size` (unavailable without ground truth).
2. **Fixed calibration function** `w(frac_hat)`: a log-linear function `w = clip(b0 + b1·log(size_hat+1), w_min, w_max)`, with `b0, b1, w_min, w_max` **fit directly from E48's real 125-subject causal-ablation data** (`calibrate_ccaba.py`), not assumed or learned. This is the novel content — the function's *shape* is a measured constant.
3. **Amplification**: `bottleneck_amp = bottleneck * (1 + alpha · w(frac_hat))`, where `alpha` is a single learnable scalar (init 0.1) controlling overall trust in the calibration-derived signal. Training can scale the mechanism's strength; it cannot alter the measured shape of the size-dependency.
4. `bottleneck_amp` replaces `bottleneck` before `upconv3`. Extends **v3** (not v5) to isolate CCABA's own effect from E46's (already causally-nulled) attention gate.

This is mechanistically distinct from E46's `attn_gate1`: that was a `sigmoid ∈ [0,1]` **routing/suppression** gate on the `enc1` skip; CCABA is an **amplification/boost** (`1 + α·w`, unbounded above) on the `bottleneck` itself, and — critically — its conditioning signal comes from a measured causal curve, not learned end-to-end from scratch.

## Verified properties (before training)

- Correct output shapes at all resolutions, correct per-subject broadcasting (`gain` shape `(B,1,1,1,1)`).
- **Ablation-safety**: forcing `ccaba_alpha=0` reproduces v3's `probs`/`aux_probs3`/`aux_probs2` bit-for-bit (max abs diff 0.0).
- Calibration constants regenerated by `calibrate_ccaba.py` and cross-checked against the hardcoded values in `neuroscan_3d_v6.py` (log-linear OLS: `drop = 1.797287 − 0.130993·log(size+1)`, R²=0.260; isotonic regression compared for disclosure, R²=0.359, not used due to needing smooth/monotone-by-construction behavior for a differentiable inference-time gate).
- Param overhead: +258 vs. v3 (negligible).

## Next steps

1. Design the `frac_hat` supervision term for training (light auxiliary loss against real fractional occupancy, same family as D4/D2 heads).
2. Full 30-epoch training run, same protocol as every prior condition (seed 0, AdamW, CosineAnnealingLR, mu=0.1).
3. Evaluate against the +1.0pp bar (canonical baseline 0.9063) and the secondary D4-only check (0.9096).
4. If successful, close the causal loop: re-run E48's own ablation-audit methodology on the new model and check whether the size-drop correlation weakens — direct evidence the fix addresses the diagnosed mechanism, not just a lucky Dice bump.
