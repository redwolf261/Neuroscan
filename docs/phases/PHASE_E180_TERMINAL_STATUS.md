# E180 — Terminal status: Stages 0-11 complete, frozen

**Date**: 2026-09-17
**Status**: **COMPLETE and FROZEN**, per explicit user directive. Do not regenerate; read from
`experiments/exp_e12_eggo_m/e180/FROZEN/`.

---

## What was tested

$H_{180}$: representation-induced decision instability (Γ) predicts recoverable segmentation
benefit from representation restoration (Δ), beyond simpler signals or boundary proximity —
tested across three controlled representation-space transformation families at enc3 (T1_rank,
T4_spectral, T5_smooth), on the full 125-subject validation cohort and the predefined E167
110-subject "good" stratum.

## Result, stated at the correct confidence level (adopted wording)

> **For the tested T4 spectral perturbation/restoration regime**, representation-induced
> decision instability is a graded signal associated with, and counterfactually coupled to,
> recoverable segmentation performance.

Not: "Γ measures recoverable task information" (too general — this is a phenomenon under a
particular representation, perturbation family, decoder, dataset, and restoration operator, not
an established general measure).

## Per-family outcome

| Family | Stage 6 (primary) | Stage 8 (vs uncertainty) | Stage 9 (boundary) | Stage 10-11 (dose-response) | Final |
|---|---|---|---|---|---|
| **T1_rank** | β=+0.58, p=0.112 (full) — fails | — | — | not run (killed upstream) | **KILLED at Stage 6** |
| **T4_spectral** | β=+0.75, p<0.0001 (both strata) | ΔR²=+0.0043, survives | survives | **STRONGEST** (both strata) | **Sole survivor** |
| **T5_smooth** | β=+0.45, p=0.124 (full); p<0.0001 (110) | ΔR²=+0.0095, survives | survives | not run (not advanced) | **HELD, stratum-dependent** |

## Why the family split matters (not a nuisance to explain away)

If every transform family had produced a positive result, the finding would collapse to "any
perturbation creates measurable instability, and instability trivially predicts something" — a
generic-sensitivity artifact, not a specific phenomenon. The fact that **T1_rank (rank
truncation) and T5_smooth (local smoothing) do not cleanly replicate while T4_spectral
(spectral reshaping, Σ→Σ^γ) does** is evidence against that generic account. It narrows the
live hypothesis to:

$$
\boxed{\text{the relevant instability may specifically concern the spectral organization of the
representation} — \text{a hypothesis, not yet a conclusion.}}
$$

## Stage 10-11 — the strongest evidence in the program

Not "Γ correlates with Δ" (cross-sectional, vulnerable to the "just another proxy for difficult
subjects" objection already raised and addressed via subject fixed-effects). Stage 10-11
manipulated the representation across 5 graded restoration doses and measured the **step-level**
relationship:

$$
\Delta\Gamma^{(k)} = \Gamma^{(k+1)} - \Gamma^{(k)} \quad\text{predicts}\quad
\Delta Dice^{(k)} = Dice^{(k+1)} - Dice^{(k)}
$$

Result: β=−0.283 (full 125), β=−0.420 (E167 110), both permutation p<0.001, both correct sign
(Γ decreasing predicts Dice increasing). This is intervention-based, dose-graded evidence — not
a formal causal proof, but considerably closer to one than the original cross-sectional
correlation, and it independently attacks the subject-difficulty-confound objection by examining
*changes induced by the intervention itself* rather than static levels.

## Novelty status (unaffected by this result)

`PHASE_E180_NOVELTY_AUDIT.md`'s verdict stands: 🟡 partially occupied. The mechanism (activation
perturbation as inference-time probe) and the application shape (instability signal → prioritized
refinement, TRUST/CertainTTA) are both published 2025-2026. What is not found is Γ's specific
construction (family-specific transform, validated against measured restoration counterfactual
rather than a downstream proxy). Stage 10-11's result does not establish novelty by itself — it
establishes that the phenomenon is real and structured, which is a precondition for a novelty
claim to even be worth making, not the claim itself.

## What this authorizes and what it does not

**Authorized**: mechanism dissection of T4_spectral specifically — why does spectral reshaping
produce this coupling while rank truncation and smoothing do not? What spectral property is Γ
actually detecting? This is the next phase, explicitly *not* Stage 12 (an algorithm).

**Not authorized by this result**:
- Building an adaptive-refinement algorithm (Stage 12+) — still gated behind explaining the
  mechanism first, and behind differentiating from TRUST/CertainTTA specifically.
- Generalizing "Γ measures task information" beyond the tested T4_spectral regime.
- Treating T1_rank or T5_smooth as viable without new evidence — they remain killed/held.

**Locked progression** (per explicit instruction):
$$
E180 \rightarrow \text{mechanism dissection} \rightarrow \text{prior-art attack} \rightarrow
\text{principle} \rightarrow \text{algorithm}
$$
not $E180 \rightarrow \text{adaptive refinement module}$.

## Artifacts

`experiments/exp_e12_eggo_m/e180/FROZEN/MANIFEST.json` — 16 files, sha256_16-checksummed,
headline result embedded. All pre-registration docs: `PHASE_E180_FREEZE_MANIFEST.md`,
`PHASE_E180_1_TRANSFORMATION_LAB_PREREG.md`, `PHASE_E180_2_SEVERITY_CALIBRATION_PREREG.md`,
`PHASE_E180_3_4_GAMMA_PREREG.md`, `PHASE_E180_5_DELTA_PREREG.md`,
`PHASE_E180_5_5_PIPELINE_GATE_PREREG.md`, `PHASE_E180_6_9_ANALYSIS_PREREG.md`,
`PHASE_E180_7_COMPETITOR_LADDER_NOTE.md`, `PHASE_E180_NOVELTY_AUDIT.md`,
`PHASE_E180_10_11_T4_DOSE_RESPONSE_PREREG.md`.
