# E173 — E169b novelty audit (the N1 gate)

**Date**: 2026-09-16
**Status**: Audit complete. **Verdict: 🟡 EXTEND — the concept is occupied; one axis is not.**
**Run before** any algorithm work, per the corrected sequence
`E169b → exact prior-art audit → identify gap → define principle → PoC → Dice`.

---

## The claim under audit

> Has anyone predicted an **instance-specific minimum task-required representation rank** at
> **intermediate layers** of a dense medical segmentation network, from a **single intact forward
> pass**, shown the prediction carries information **beyond lesion extent**, and used it as a
> scientific characterisation of segmentation difficulty?

Our evidence: $X_i \rightarrow \widehat R^*_i(l)$, out-of-sample $R^2_{\text{enc3}} = 0.518$ after
controlling **both** GT and predicted lesion volume; $R^2_{\text{dec1}} = 0.004$.

---

## Novelty matrix

| Prior art | What it measures | Per-instance | Label-free | Intermediate layer | Output-preserving | Segmentation | **Predicts** the requirement | Our difference |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| **Intrinsic Dimension of representations** (Ansuini NeurIPS'19; Konz 2408.08381 medical) | manifold ID per layer | ✗ dataset-level | ✓ | ✓ | ✗ | ✓ (medical) | ✗ | ID is a *manifold neighbourhood* property; ours is *minimum rank preserving the model's own output* |
| **Local Intrinsic Dimensionality (LID)** | per-sample manifold expansion | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | LID never references the model's prediction |
| **Feature Compression for Machines, channel truncation** (2512.11134) | per-sample active channels, truncated to preserve task performance | ✓ | ✓ | ✓ | **✓** | ✗ (detection) | ✗ measures, doesn't predict | closest on the *quantity*; but selects by channel range at inference, does not predict a requirement |
| **Adaptive Rate Control w/ R-D Prediction** (2412.18834) | predicts R-λ/D-λ per frame **without pre-encoding** | ✓ | ✓ | n/a | ✓ (distortion) | ✗ (video) | **✓** | the closest *structural* analogue: predict needed compression from a cheap pass. Different domain, different quantity |
| **Effective-rank regularisation** (NeurIPS'24), WERank, rank collapse | rank as an optimisation target | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | already audited OCCUPIED in E170b |
| **ARENA / AdaLoRA / SeLoRA** | rank of **adapter weights** | ✗ | ✓ | n/a | ✗ | ✓ | ✗ | different object entirely (weights, not activations) |
| **nnU-Net Revisited**, BraTS saturation | benchmark performance | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | unrelated |

## What the matrix shows

**Occupied:** measuring per-sample representational complexity at intermediate layers
(LID, ID); truncating channels per-sample to preserve task output (2512.11134); predicting a
required compression level from a cheap pass (2412.18834); effective rank as a regulariser.

**Every ingredient of E169b has been published.** Nothing here is a surprise after E162's law.

**Not found, after targeted search under ~20 alternative names:**

1. A requirement defined by **truncation against the model's own undegraded prediction**
   (our confound guard) rather than against labels, a manifold estimate, or a downstream metric.
2. That requirement **predicted** from one forward pass rather than measured by sweeping.
3. **Stage specificity as a finding**: $R^2$ 0.518 at enc3 vs **0.004** at dec1 — the encoder
   carries size-independent rank demand, the decoder carries none.
4. The **beyond-size control**: surviving partialling of *both* GT and model-predicted lesion
   volume (proxies agreeing at ρ=0.94/0.88/0.96).

## Verdict: EXTEND, not DEVELOP, and not KILL

$$\boxed{\text{🟡 Concept occupied. Contribution, if any, is the CHARACTERISATION — not the quantity.}}$$

The defensible claim is narrow and must be stated as an **analysis** result:

> In a 3D segmentation network, the minimum representation rank needed to preserve the model's
> own output is instance-specific, predictable from one forward pass, **carries information
> beyond lesion extent**, and is an **encoder property** — it vanishes at the decoder.

This is not a new *quantity* (2512.11134 has the quantity; 2412.18834 has the prediction
structure). It is a new *measurement about where and what that quantity depends on* in dense
segmentation.

## Honest limits, stated before anyone builds on this

- **n=125, one checkpoint, one architecture, one dataset.** Stage specificity could be a property
  of this U-Net, not of segmentation networks. Untested elsewhere.
- **$R^*$ and $R_{\mathrm{eff}}$ use three incompatible definitions** across E165/E169/E170
  (SVD truncation full-volume; entropy@20k per tile; entropy@8k per patch). Any published claim
  must unify these first.
- **The threshold (0.90) and dyadic grid are conventions**, inherited from E147. Sensitivity was
  audited (±0.02 moves 7–8/125 subjects) but the grid is coarse: adjacent ranks differ 2×.
- **No Dice consequence.** E172 falsified the intervention route at both sites; E167 closed both
  routes to +1pp. This measurement does not become a method.

## Consequence for the project

Under the stated priority hierarchy (N1 novelty > N2 mechanism > N3 implementable > N4 Dice), the
outcome is **B/EXTEND**: same underlying concept, one non-trivial added capability
(stage-specific, beyond-size, prediction-preserving requirement).

That is a **methods/analysis contribution**, not an algorithmic one — and per the hierarchy, a
novel principle worth reporting even at ΔDice = 0. It is consistent with, not a reversal of,
E158/E162/E168: the architecture-novelty search stays closed.

**Recommended next step is writing, not compute.** Specifically: unify the three rank definitions
(cheap, CPU-side) so the measurement is internally consistent, then write E169b up as the
analysis result with the four limits above stated in the paper, not buried.

## Sources

Intrinsic dimension of data representations (Ansuini et al.) · Hidden representation refinement
via intrinsic dimension, medical imaging (arXiv 2408.08381) · CoLafier / LID (2401.05458) ·
Feature Compression for Machines with Range-Based Channel Truncation (2512.11134) · Adaptive Rate
Control for Deep Video Compression with R-D Prediction (arXiv 2412.18834) · Effective Rank
Regularization (NeurIPS 2024) · WERank (2402.09586)
