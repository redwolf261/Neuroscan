# Phase E8: EGGO-v1 — Simplified to the Smallest Testable Hypothesis

**Status**: ✅ Complete — design simplified, ready for E9/E10 to fill in the remaining unknowns

**Date**: 2026-08-04

## Objective

Reduce the full EGGO design (`PHASE_E5_ALGORITHM_DESIGN.md`) to the
smallest hypothesis worth testing first, per the converging evidence from
E6 (density's independent contribution to evidence is modest, rank 3/4
in a joint model) and E7 (boundary geometry precedes calibration
cleanly; density's temporal ordering is ambiguous). Both lines of
evidence point the same direction: separation/margin is the
better-supported mechanism; compactification/density is not yet
justified as a co-equal component.

## Question

> Does encouraging uncertain voxels near the latent decision boundary to
> increase their margin from the opposite class improve segmentation?

This is deliberately narrower than "does latent geometry optimization
help" — it isolates one specific, well-supported mechanism rather than
bundling two mechanisms of unequal evidential support together, which
would make a null result impossible to attribute (if the combined EGGO
failed, was it the margin term, the density term, or their interaction?).

## What's kept vs. removed

| Component | Status | Rationale |
|---|---|---|
| Segmentation loss ($\mathcal{L}_{seg}$) | **Kept, unchanged** | Frozen baseline's FocalTversky, per `PHASE_A5_BASELINE_FROZEN.md` |
| Evidential head | **Kept, unchanged** | Only its output is read (as the uncertainty gate input); its own loss/parameters are untouched |
| Uncertainty gate ($\hat U_i$) | **Kept** | E1.1–E1.4 established evidence-based uncertainty as more informative than image geometry or feature magnitude alone |
| Boundary-proximity weight ($B_i$) | **Kept** | E7's strongest, least ambiguous finding: this geometry precedes calibration |
| Margin/separation loss ($\mathcal{L}_{sep}$) | **Kept**, exact form deferred to E10 | The mechanism with the most converging support |
| Density weight ($D_i$) | **Removed** | E6: modest independent contribution (rank 3/4). E7: ambiguous temporal ordering (tracks calibration's own timescale, doesn't clearly precede it) |
| Compactification loss ($\mathcal{L}_{comp}$) | **Removed** | Depends on $D_i$; removed alongside it |
| k-NN computation | **Removed** | Was only needed for $\rho_i$/$D_i$ and the local centroid $\mu_{c(i)}^{local}$; both gone |
| Local same-class centroid | **Removed** | Was $\mathcal{L}_{comp}$'s pull target; not needed without compactification |

## EGGO-v1 formulation

$$\mathcal{L} = \mathcal{L}_{seg} + \lambda \cdot \hat U_i \cdot B_i \cdot \mathcal{L}_{margin}$$

Using the same variable definitions as `PHASE_E5_ALGORITHM_DESIGN.md`
§2.1–2.2 ($\hat U_i$ = normalized uncertainty from the evidential head,
$B_i = \exp(-|d_i|/\tau_b)$ = boundary-proximity weight, $\tau_b$
measured from the correct/incorrect crossover). $\mathcal{L}_{margin}$'s
exact mathematical form is deliberately **not** decided here — that is
E10's job, done as a explicit comparison against alternatives rather than
assumed.

**Single new hyperparameter to tune**: $\lambda$ (down from EGGO-full's
two, $\lambda_{sep}$ and $\lambda_{comp}$) — a meaningfully smaller search
space for the pilot/sweep phases (E12/E13).

## What this simplification buys, and what it costs

**Buys**: a clean, single-mechanism test. If EGGO-v1 improves Dice/ECE,
that result is unambiguously attributable to boundary-margin pressure —
no confound from an untested density term. If it does not improve
anything, that is also a clean, interpretable null result (distinct from
Experiment D's ABO null — this would test a geometry-based, not a
gradient-magnitude-based, intervention), rather than an ambiguous "which
of two mechanisms failed" result.

**Costs**: defers testing whether density/compactification would have
added value on top of a working margin mechanism. Per the roadmap, this
is intentional and revisited in a later ablation (E14) only if EGGO-v1
shows a real effect worth extending — consistent with "test the smallest
hypothesis first" and avoiding the earlier mistake pattern flagged in
`abo_frozen_lessons_learned` (don't keep adding mechanisms to a method
that hasn't yet demonstrated its core premise works).

## Files

| File | Purpose |
|---|---|
| `PHASE_E5_ALGORITHM_DESIGN.md` | Full EGGO design this simplifies |
| `PHASE_E6_STRESS_TEST.md`, `PHASE_E7_CAUSALITY_TEST.md` | The two convergent lines of evidence motivating this simplification |
| `PHASE_E9_TRAINABLE_BOUNDARY_HEAD.md`, `PHASE_E10_MARGIN_LOSS_SELECTION.md`, `PHASE_E11_IMPLEMENTATION_SPEC.md` | Next steps filling in EGGO-v1's remaining unknowns |

---

**Completed**: 2026-08-04
