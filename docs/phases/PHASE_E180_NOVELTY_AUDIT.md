# E180 — Novelty prior-art audit

**Date**: 2026-09-17
**Status**: Targeted audit complete, ~30 min, no compute. Run WHILE the 125-subject Stage 3
measurement was in progress, per the project's standing discipline of auditing before
committing to Stage 12+ (any actual algorithm).
**Verdict**: 🟡 **PARTIALLY OCCUPIED — the outer mechanism and the outer application shape are
both published; the specific construction of Γ (validated against a counterfactual restoration
benefit Δ, not against downstream loss/entropy) has not been located as a direct match.**

---

## Context

This project's algorithmic-novelty search (E1-E176) closed with an explicit terminal verdict:
"No algorithmic-novelty candidate remains that this project can reach" (E176, eight consecutive
OCCUPIED audits across architecture/objective/representation/adaptive-compute/capacity/
compression/cross-domain/evaluation). E180 was proposed as a genuinely new angle — phenomenon-
first, testing whether representation-induced decision instability (Γ) predicts recoverable
restoration benefit (Δ) — explicitly not yet claiming novelty, with the stated objective being
Stages 0-11 only (establish whether the phenomenon is real), Stages 12+ (an actual algorithm)
deliberately out of scope pending this kind of audit.

This audit runs that check now, before Stage 6-9's statistical verdict lands, so the novelty
question is not conflated with "does the correlation hold" — E162's structural law (informative
observations on well-studied tasks predict their own prior-art density) applies regardless of
whether Γ turns out to predict Δ.

## What was audited

Two components, audited separately per this project's established practice (E170b: audit the
exact chain, not the closest-sounding keyword):

1. **The mechanism**: perturb an intermediate representation in a controlled way, measure output
   displacement, as an inference-time probe.
2. **The application shape**: use that displacement/instability signal to decide where to spend
   refinement/adaptation compute in a segmentation model.

## Component 1 — the mechanism: 🔴 OCCUPIED

**APEX (arXiv 2602.03586, Ren/Luo/Li, Aalborg University, Feb 2026)** — "Probing Neural Networks
via Activation Perturbation." Directly matches E180's Stage 1 machinery: an inference-time
probing paradigm that perturbs hidden activations while keeping inputs and parameters fixed,
explicitly contrasted against input perturbation (a "constrained special case" of the same
framework) and parameter perturbation. APEX's stated theoretical contribution — perturbing
activations induces "a principled transition from sample-dependent to model-dependent behavior"
— is a different *use* of the mechanism than E180's, but the mechanism itself (controlled
activation-space perturbation as a probe, at a fixed intermediate stage, without touching
input/parameters) is the same object E180's T1-T5 transformation lab constructs.

**Consequence**: E180's transformation-laboratory *method* (Stage 1) is not a new probing
technique. This was expected — the closed E176 audit already covers "representation
intervention" as one of the eight occupied axes — and does not by itself kill E180, since the
mechanism was never the claimed contribution; what it's used *for* is.

## Component 2 — the application shape: 🔴 OCCUPIED, and densely so

The search surfaced an active, established subfield doing E180's *outer* loop almost exactly:

| Work | What it does |
|---|---|
| **TRUST** (arXiv 2509.22813, Sept 2025) | Test-time refinement for segmentation: generates multiple causal perspectives, computes prediction entropy per perspective, **selects low-entropy (stable) ones as a reliability signal**, uses them to guide adaptation. Explicitly frames "uncertainty-guided node selection... prioritizing high predictive entropy" against "indiscriminate intervention among all nodes" causing "computational inefficiency" — i.e. exactly E180's eventual Stage 12+ framing (instability signal → budget-constrained selective refinement), for segmentation, in 2025. |
| **CertainTTA** (ScienceDirect, May 2025) | Test-time adaptation for medical segmentation using dual uncertainty (output entropy + model/mutual-information-based adaptability) to decide adaptation targeting. |
| Broader TTA-for-segmentation literature (SicTTA, boundary-aware TTA, progressive test-time energy adaptation — all 2025) | Confirms this is a dense, active subfield, not a gap. |

**The general claim "use an instability/uncertainty signal to prioritize where a segmentation
model spends refinement compute" is thoroughly occupied.** This is the load-bearing finding of
this audit: even if Stage 6-9 confirms Γ→Δ cleanly, the eventual Stage 12 algorithm as currently
sketched (predicted instability → select regions → apply expensive refinement) would need to be
argued against this entire literature, not treated as a green field.

## What is NOT found — the narrower, potentially differentiated claim

Every occupied result above uses a **generic downstream signal** (predictive entropy, output
uncertainty, mutual information) as the selection criterion. None of the located work constructs
the selection signal the way E180's Γ is constructed:

1. **A per-tile, family-specific, controlled representation-space transformation** (rank
   truncation / spectral reshape / local smoothing, at a fixed named intermediate stage) as the
   probe, rather than a task-native quantity (entropy, MC-dropout variance, model confidence).
2. **Validated directly against a measured counterfactual restoration benefit (Δ)** — the actual
   Dice gain from *restoring* that tile's representation from a controlled degraded state —
   rather than against a proxy (downstream loss, calibration error, or agreement across TTA
   augmentations).
3. **The same-operating-point discipline** ($Z \to Z^{deg}$, both Γ and Δ measured from the
   identical degraded state) — a design correction this project made mid-experiment (see
   `PHASE_E180_1_TRANSFORMATION_LAB_PREREG.md`'s "same-operating-point requirement") that is not
   an established convention in the surveyed TTA/uncertainty literature, which typically measures
   uncertainty on the intact prediction, not on a deliberately degraded one.

No paper located in this pass constructs a selection signal this way, or validates it against a
genuine restoration counterfactual rather than a proxy. **Absence of evidence after one targeted
search pass is weak** — per this project's own standing caveat (E170b, E173) — and does not
establish novelty; it only narrows where a defensible claim, if any, could sit.

## Honest classification

| Component | Status |
|---|---|
| Activation-space perturbation as an inference-time probe | 🔴 OCCUPIED (APEX) |
| Instability/uncertainty signal → prioritized test-time refinement, segmentation | 🔴 OCCUPIED, densely (TRUST, CertainTTA, broader 2025 TTA literature) |
| Generic sensitivity/entropy as the selection criterion | 🔴 OCCUPIED |
| **Family-specific controlled representation transform as the selection signal** | 🟢 not found |
| **Signal validated against measured restoration Δ, not a downstream proxy** | 🟢 not found |
| **Same-operating-point (degraded-state) construction of both signal and counterfactual** | 🟢 not found |

## Verdict

$$
\boxed{\text{🟡 PARTIALLY OCCUPIED — mechanism and application shape both published; the}}
$$
$$
\boxed{\text{specific signal-construction-and-validation chain is narrower and not yet located.}}
$$

This is the same shape of verdict this project has reached before (E170b: "the MECHANISM is
occupied, the TARGET may be differentiated"). The defensible claim, if Stage 6-9 supports one at
all, would have to be stated narrowly:

> A per-tile signal, constructed from controlled family-specific representation-space
> perturbation at a fixed encoder stage and validated against a measured counterfactual
> restoration benefit from the same degraded operating point, is not the same object as the
> generic-uncertainty signals (entropy, MC-dropout, mutual information) that the existing
> test-time-refinement literature selects with — and may carry complementary information.

**This is not itself a novelty claim.** It is the outer bound of what a future novelty claim
could argue, contingent on Stage 6-9's statistical result. If Γ turns out to be statistically
redundant with the competitor ladder already planned for Stage 7-8 (magnitude, uncertainty,
sensitivity), the narrow claim above collapses too — Stage 8's incremental-$R^2$ gate is
therefore doing double duty: it is both the scientific decision gate for $H_{180}$ and, in
effect, the test of whether Γ is anything more than a re-derivation of the entropy-based signals
this literature already uses.

## Consequence for the plan

- **No change to Stages 0-11.** These were always scoped as phenomenon-establishment, not a
  novelty claim, and this audit does not alter that scope.
- **Stage 12+ (the actual algorithm) is now explicitly gated on two things, not one**: (a) Γ→Δ
  surviving Stage 8's incremental-R² test beyond the competitor ladder, per the existing plan,
  and (b) a demonstration that Γ is not redundant with entropy/uncertainty-based TTA selection
  specifically — since that is the exact literature Stage 12 would otherwise be quietly
  re-deriving. Item (b) is a natural extension of Stage 7's competitor ladder: entropy/predictive-
  uncertainty should be added as an explicit competitor signal alongside magnitude, perturbation
  magnitude, and decoder sensitivity, if Stage 6-9 supports proceeding at all.
- **If a Stage 12 algorithm is ever built**, TRUST and CertainTTA must be cited and argued
  against directly, not discovered post-hoc during a paper-writing pass — this is exactly the
  failure mode E162's structural law predicts and this project has hit before (E138, E170b).

## Sources

APEX: Probing Neural Networks via Activation Perturbation (arXiv 2602.03586) · TRUST: Test-Time
Refinement using Uncertainty-Guided SSM Traverses (arXiv 2509.22813) · CertainTTA: Estimating
uncertainty for test-time adaptation on medical image segmentation (ScienceDirect, May 2025) ·
SicTTA, Boundary-Aware TTA, Progressive Test-Time Energy Adaptation (arXiv, 2025, context on
subfield density)
