# Phase E3: Candidate Optimizer/Diagnostic Concepts

**Status**: Draft — not yet decided which (if any) to build

**Date**: 2026-08-04

## Purpose

Synthesizes `PHASE_D_ANALYSIS_REPORT.md` (Experiment D's null result plus
the ECE finding) and `PHASE_E1_LITERATURE_MATRIX.md` (45-paper literature
sweep) into concrete, buildable next steps. Per the design criteria set
before this phase, any candidate must be: motivated by Phase D,
motivated by the literature, mathematically different from ABO (not a
tweak), experimentally testable, and ablation-friendly. Candidates that
fail any of these are not proposed.

## The one-sentence bottleneck hypothesis (Step E3 deliverable)

> **Gradient magnitude imbalance between the segmentation and evidential
> branches is not the constraint on Dice; the constraint more likely lies
> in how the two branches' features are represented in the shared trunk
> — moving the gradient statistic (the symptom) without addressing
> representation entanglement (the possible cause) does not move the
> outcome.**

This is a hypothesis, not yet a confirmed fact — Concept 0 below exists
specifically to test it before committing further resources.

## Concept 0 (recommended first step): Representation Entanglement Diagnostic

**Not an optimizer. A diagnostic, in the spirit of Phase B before C1.**

**Motivated by**: CORE-MTL, Rep-MTL, and arXiv 2602.16125's shared thesis
that gradient conflict/imbalance is often a *symptom* of representation
entanglement, not the cause. Before building any representation-level
optimizer (which is a real architecture change, higher-risk than ABO's
scope), first check whether the shared trunk's activations actually show
entanglement between what the segmentation head needs and what the
evidential head needs.

**What to measure**: at the same trunk layer where Phase B measured
gradient norms/cosine similarity (`dec1`, 32ch, per `PHASE_A5_BASELINE_FROZEN.md`),
compute a feature-level analogue:
- **CKA (Centered Kernel Alignment) similarity** between the
  segmentation-head-relevant and evidential-head-relevant activation
  subspaces at the trunk, across training epochs — does it stay high
  (entangled) or drop (naturally separating), and does this pattern
  correlate with anything already measured (e.g., does entanglement
  track the epoch-0-3 conflict-resolution window from Phase B, or persist
  well past it)?
- **Feature-space cosine similarity** between per-branch saliency maps
  (gradient of each head's loss w.r.t. trunk activations, not
  parameters — a different, complementary measurement to Phase B's
  parameter-gradient cosine) at the same batches already logged.

**Why this order matters**: if entanglement turns out to be low/absent,
that would falsify the representation-bottleneck hypothesis too, and save
building an architectural intervention that wouldn't have a mechanism to
act on. If entanglement is high and doesn't resolve like the gradient
conflict did, that's direct empirical support for Concept 1 below, with
a clear, falsifiable prediction stated up front (same discipline as
Phase B → C1).

**Cost**: low. Reuses existing checkpoints
(`experiments/exp_c1_abo/expD_active_seed{0,1,2}/checkpoints/best.pth`
or, more informatively, the frozen baseline's checkpoints since no
gradient-manipulation confound is needed) and the existing batch-level
data pipeline. No new training run required — a forward-pass-only
analysis script over saved checkpoints and a validation batch.

## Concept 1: Orthogonal Trunk Feature Regularizer

**Motivated by**: CORE-MTL's causal orthogonal factorization and the
general "orthogonal bottleneck" pattern (arXiv 2605.26012, applied here
to a different domain). Directly targets representation entanglement
rather than gradient magnitude — mathematically a completely different
mechanism from ABO (a loss-space regularizer during the forward pass, not
a gradient-assembly rule at backward time).

**Mechanism** (draft, not finalized): add an orthogonality-encouraging
penalty between the segmentation-relevant and evidential-relevant
sub-representations within the shared trunk's final layer
(`dec1`, 32ch) — e.g., penalize the off-diagonal Gram matrix structure
between the two heads' input-relevant activation subspaces, similar in
spirit to how CORE-MTL structurally separates causal task-relevant
representations from residual/nuisance variation.

**Testable, falsifiable prediction**: if this helps, Dice/IoU should move
(unlike ABO), AND Concept 0's entanglement measurement should show
reduced entanglement correlating with the Dice change. If Dice doesn't
move even when entanglement measurably drops, that's a second,
independently informative null result (representation entanglement,
though real, also isn't the bottleneck) — still valuable, not a failure.

**Ablation-friendly**: regularizer strength is a single new
hyperparameter, sweepable with the same C2a/C2b-style sensitivity-sweep
protocol already built and validated (`--r_target`/`--gamma`-style CLI
overrides pattern in `train_abo.py` is directly reusable as a template).

**Risk**: this is a genuine architecture-adjacent change (a new loss
term touching the trunk), higher-risk than ABO's backward-pass-only
scope, which stayed within `PHASE_A5_BASELINE_FROZEN.md`'s "architecture
frozen" constraint. Would need explicit discussion of whether this
still counts as "optimization only" or crosses into architecture
territory the project has otherwise kept frozen since Phase A.5.

## Concept 2: Uncertainty-Signal-Driven Segmentation Loss Weighting

**Motivated by**: Progressive Uncertainty-Guided Evidential U-KAN and the
PGU/REH pixel-weighting paper (both Category 5 in the literature matrix)
— using the evidential head's *uncertainty value*, not gradient
statistics, to modulate the segmentation branch. Also motivated directly
by this project's own Step E2 finding: ABO's effective gradient ratio has
a real, non-confounded relationship with ECE (calibration), not with
Dice — suggesting the evidential branch's output does carry a genuine,
usable signal, just not one gradient-magnitude control was the right way
to exploit for the *segmentation* objective.

**Mechanism** (draft): use the evidential head's per-voxel uncertainty
(already computed via the Beta distribution's variance) to reweight the
segmentation loss per-voxel — higher uncertainty voxels get modulated
loss contribution, direction (up- or down-weight) to be determined
empirically rather than assumed.

**Prerequisite, per the literature caution**: before building this,
directly check the NeurIPS 2024 "Mirage" paper's concern for this
specific evidential head — is its epistemic uncertainty actually
informative (e.g., does it correlate with genuine error regions,
boundary voxels, or out-of-distribution cases) or is it closer to a
non-vanishing artifact? This is a cheap, existing-checkpoint diagnostic
(correlate predicted uncertainty against actual per-voxel error on the
validation set) that should run before investing in Concept 2's training
mechanism — otherwise this risks repeating the ABO pattern of an
elaborate mechanism on top of a branch whose signal isn't load-bearing.

**Testable, falsifiable prediction**: if the uncertainty signal is
genuine, per-voxel reweighting should move Dice/HD95 specifically in
regions where uncertainty is currently high (boundary/ambiguous voxels),
not uniformly — a specific, checkable prediction distinguishing this from
a generic regularization effect.

## Concept 3 (not recommended as primary, but noted): Boundary-Aware Loss Addition

**Motivated by**: Kervadec boundary loss, cbDice, generalized surface
loss — all well-evidenced, reproducible Dice/HD95 gains in the
literature. Does NOT build on ABO's infrastructure or this project's
specific gradient/representation findings — it targets a different,
independently well-supported bottleneck (Dice-HD95 metric mismatch).

**Why not primary**: doesn't follow from what Experiment D or the
literature review specifically taught this project (the whole point of
this redesign phase, per the user's explicit framing, was to build on
what was *learned*, not to pick any well-evidenced idea off the shelf).
Worth keeping as a secondary/combination idea — e.g., could be added
alongside Concept 1 or 2 later — but shouldn't be the primary next
experiment given the "motivated by Phase D" design criterion.

## What is explicitly NOT proposed

- Any further ABO hyperparameter tuning (r_target, gamma, alpha_power,
  ema_decay) — per [[abo_frozen_lessons_learned]], this question is
  answered and frozen.
- "ABO v2 / Dynamic ABO" or similar reactive renaming without a distinct
  mathematical mechanism and literature-grounded motivation.
- Gradient-direction methods (PCGrad/CAGrad/Nash-MTL/GradVac-style) as a
  next step — weakest literature support given this project's own
  Finding 1 (conflict is transient, not persistent, so these methods'
  core premise doesn't hold here).

## Recommended sequencing

1. **Concept 0 (representation entanglement diagnostic)** first — cheap,
   no new training, directly tests the hypothesis before committing to
   an architectural change. This is the natural Phase B-style diagnostic
   step for this new hypothesis, matching the project's established
   "measure before intervening" discipline.
2. **Concept 2's prerequisite check (is the uncertainty signal genuine)**
   — also cheap, no new training, can run in parallel with Concept 0.
3. Based on what 1 and 2 find: build **either** Concept 1 (if
   entanglement is real and persistent) **or** Concept 2 (if the
   uncertainty signal is genuine and the ECE-not-Dice pattern from Step
   E2 looks exploitable) — or both, if warranted, but not before the
   diagnostics report back.
4. Concept 3 stays a secondary/combination idea, not primary.

None of these four concepts should be built until the user reviews this
document and decides which (if any) to pursue — this is a draft menu, not
an implementation plan.

## Files

| File | Purpose |
|---|---|
| `PHASE_D_ANALYSIS_REPORT.md` | The null result + ECE finding motivating this document |
| `PHASE_E1_LITERATURE_MATRIX.md` | The 45-paper literature sweep this builds on |

---

**Drafted**: 2026-08-04
