# E176 — The metric axis: prior-art audit

**Date**: 2026-09-16
**Status**: Audit complete, ~20 min, no compute.
**Verdict**: 🔴 **OCCUPIED — the framing is published under a name.** Eighth consecutive.

---

## The candidate

Every candidate this session (E166, E169b, E170, E171, E175) operated on the **forward pass of a
fixed architecture** trained by a fixed objective against a fixed label set. The unexamined
component was the **evaluation**.

The argument, built entirely from our own measurements:

- E167: **92.6%** of remaining ET error on the 110 good subjects lies within **1 voxel** of the
  GT boundary; 96.9% within 2; interior error **1.4%**.
- Dice scores a 1-voxel boundary disagreement identically to a missed lesion core.
- Menze (TMI 2015): published inter-rater ET agreement **median 0.77**. Our model: **0.900**.

So the model is being optimised and scored against a boundary that annotators themselves do not
agree on, past the point where the labels are reliable — and the metric cannot separate *error*
from *annotator disagreement*.

## Verdict: this is Peak Ground Truth

**"Approaching Peak Ground Truth"** (arXiv 2301.00243) defines exactly this, verbatim:

> "PGT marks the point beyond which an increase in similarity with the reference annotation
> stops translating to better RWMP [real-world model performance]."

It further "proposes quantitative methods using rater reliability metrics to approximate PGT and
reviews four categories of strategies for PGT-aware evaluation." That is our argument, our
proposed remedy, and our motivation — named, formalised, and surveyed.

### The rest of the neighbourhood is equally dense

| Component | Status |
|---|---|
| Ceiling where label quality limits measurable gain | 🔴 **Peak Ground Truth** (2301.00243) |
| Boundary-tolerant metric | 🔴 Surface Dice / NSD (Nikolov et al.) |
| **Tolerance set per class by annotation difficulty** | 🔴 NSD explicitly: *"class-specific thresholds τ can be used... since the difficulty of annotating varies between organs"* — the per-class-calibration idea is built into the metric's original design |
| Metrics for uncertain / small / empty references | 🔴 USE-Evaluator (2209.13008) |
| Calibration under ambiguous ground truth | 🔴 arXiv 2603.22879 |
| Multi-rater calibration error (MR-ECE) | 🔴 MICCAI; CURVAS challenge (2505.08685) |
| Learning inter-rater variability | 🔴 PULASki (2312.15686) |
| Disentangling human error from ground truth | 🔴 NeurIPS 2020 |
| Quantifying annotation ambiguity | 🔴 arXiv 2510.04366 |

The one axis I thought might be differentiated — *calibrating the tolerance empirically to
measured annotator ambiguity on this cohort rather than by convention* — is **explicitly the
stated design intent of NSD's τ**, and the multi-rater calibration literature does the
data-driven version. Our variant would additionally require multiple annotations per subject,
which **BraTS does not provide** (and BraTS 2024 ran a dual-annotation experiment but published
the method, not the agreement numbers).

---

## Assessment

This was the strongest remaining candidate: it inverted the arithmetic that closed everything
else (E167's boundary-localised residual is fatal to *model* interventions but is the *premise*
for a metric intervention), and it targeted the one axis with real headroom (0.900 already
exceeds the 0.77 published ceiling).

It is nonetheless occupied, and not marginally — the central claim has a name, a paper, and a
survey of remedies.

**Eighth consecutive OCCUPIED verdict.** E162's structural law continues to hold: on a
well-studied task, a well-motivated observation predicts its own prior-art density. The metric
axis was unexamined *by us*, not by the field.

## Consequence

$$\boxed{\text{No algorithmic-novelty candidate remains that this project can reach.}}$$

The searched space now covers: architecture, objective, representation intervention, adaptive
computation, capacity allocation, compression, cross-domain principles, and evaluation. Eight
audits, zero survivors, with a derived structural explanation.

**What this does NOT change** — the results that stand on their own measurement:

1. **Both routes to +1pp are closed, with a measured cause** (E167 + E142 + the published
   ceiling + the field-wide plateau). This is a finding, not a failure.
2. **Stage-specific required-rank predictability** (E169b/E174): R²=0.518 at enc3 after *both*
   size controls, 0.004 at dec1, stable across τ ∈ [0.80, 0.99], reproduced to 5e-5 on an
   independent re-run.
3. **The methodology**: ~20 pre-registered kills, eight prior-art audits, and a documented record
   of catching its own errors before they calcified (E62 shared-tensor, E25 sign convention, E80
   coordinate frame, E109 gate conditioning, E153 normalisation artifact, and this session's
   E172 oracle-implementation bug).

Notably, (1) and this audit are *the same finding from two directions*: the benchmark is
saturated, and the field knows it — nnU-Net Revisited calls BraTS21 "saturated," winning ET has
been flat since 2018, and Peak Ground Truth explains why.

**Recommendation: stop searching, write the report.** The search is complete in a defensible
sense — not abandoned.

## Sources

Approaching Peak Ground Truth (arXiv 2301.00243) · Surface Dice / NSD (Nikolov et al.) ·
USE-Evaluator (2209.13008) · Confidence Calibration under Ambiguous Ground Truth (2603.22879) ·
CURVAS multi-rater challenge (2505.08685) · PULASki (2312.15686) · Disentangling Human Error
from the Ground Truth (NeurIPS 2020) · Quantifying Ambiguity in Categorical Annotations
(2510.04366)
