# Phase E14: Gradient Conflict Analysis

**Status**: ✅ Complete — purely diagnostic, no algorithm or hyperparameter changes made. **Regime: Case B, mildly leaning A (weakly cooperative-to-independent). Not Case C.** Gradient conflict is not a compelling explanation for EGGO-M's Dice null.

**Date**: 2026-08-07

## Purpose

E12f/E13 established that EGGO-M's margin loss genuinely participates in
optimization (active hinge stays non-zero, adaptive τ_b works correctly)
across 4 seeds, yet Dice does not improve and is reproducibly slightly
worse than baseline. One remaining, previously untested explanation:
`L_seg` and `L_margin` might be pushing the shared `dec1` representation
in systematically conflicting directions, so that gains from one are
partially or fully undone by the other. This phase measures that
directly, rather than continuing to speculate about it.

This is a diagnostic-only phase: no changes to the algorithm, no
hyperparameter tuning, no proposed fixes. The only output is a
determination of which conflict regime the project is actually in.

## Methodology

### What was computed

For each of E12f seed 0's saved checkpoints (epochs 1, 5, 10, 15, 20, 25,
30 — the same checkpoint set used throughout E12b.5/E12d/E12f/E13's
mechanism-verification suite), the model was loaded and run in
`model.train()` mode (matching live training-time BatchNorm behavior,
per the E12e lesson that `eval()` mode on a mid-training checkpoint gives
a meaningfully different `dec1` than the optimizer actually saw) on 8
real training-set batches per checkpoint.

For each batch, from a **single forward pass**, both losses were
computed using their exact, unmodified formulas from `train_eggo_m.py`:

```
L_seg    = 0.5 * FocalTverskyLoss(probs, masks) + 0.5 * EvidentialBetaLoss(alpha, beta, masks)
L_margin = compute_margin_loss(dec1, evidence, boundary_logit, masks, anchor_idx, ...)
```

(both **unweighted** by their training-time coefficients `μ`/`λ` — this
phase measures conflict between the raw loss terms themselves, not
between their scaled contributions, per the spec.)

Then, from that same forward pass:

```
g_seg    = torch.autograd.grad(L_seg,    dec1, retain_graph=True)
g_margin = torch.autograd.grad(L_margin, dec1, retain_graph=False)
```

`torch.autograd.grad` (unlike `.backward()`) does not write into `.grad`
buffers or share mutable state between calls — the two gradient
computations are naturally independent of each other; neither call's
result was allowed to influence the other's.

### A methodological correction made during development

The original plan was to build an exact per-voxel decomposition of
`L_seg` so that "the segmentation loss at voxel i" would be a
well-defined scalar, directly comparable at the anchor level. This
turned out to be impossible to do exactly: `FocalTverskyLoss` computes
`tp`/`fp`/`fn` as **full-batch sums** (a global Tversky index), not a
mean of per-voxel losses — there is no way to write it as
`(1/N) Σ per_voxel_loss(i)` without changing its mathematical meaning.
An initial reimplementation attempt used the wrong formula entirely (a
digamma-based NLL instead of the codebase's actual mean-based BCE for
the evidential term, and incorrect Tversky hyperparameters) — this was
caught by directly reading `neuroscan_3d_fixed.py`'s real loss classes
before trusting any results, not after.

**Resolution**: no per-voxel decomposition is needed. `torch.autograd.grad`
applied directly to the true, global `seg_loss` scalar (exactly as
computed in training) still yields a well-defined gradient **tensor**
`∂L_seg/∂dec1` with the same `(B,32,D,H,W)` shape as `dec1` — this is
the same attribution backprop always performs for any scalar loss,
regardless of whether that loss is separable per-voxel. This is both
simpler and more faithful than the abandoned reimplementation approach:
it measures gradients of the exact loss the model was actually trained
on, not an approximation of it.

### Anchor restriction

`g_margin` is exactly zero away from the ~2000 anchors/volume sampled by
EGGO-M's own stratified sampling (only anchors enter
`compute_margin_loss` at all). Comparing the full dense `(B,32,D,H,W)`
gradient grids would trivially bias cosine similarity toward zero from
the ~99.9% of voxels where `g_margin` is identically zero by
construction — not a meaningful test of whether the two losses conflict
**where the margin loss actually acts**. Both gradients were therefore
restricted to the same anchor voxel set (the exact anchors
`compute_margin_loss` sampled that batch) before computing cosine
similarity — an apples-to-apples comparison at the physical locations
that matter.

### Metrics computed

1. **Cosine similarity** per anchor: `cos(g_seg_i, g_margin_i)`, computed
   only where both gradient norms exceed `1e-12` (avoids
   divide-by-near-zero noise).
2. **Gradient norms** `‖g_seg‖`, `‖g_margin‖` per anchor.
3. **Norm ratio** `‖g_margin‖ / ‖g_seg‖` per anchor.
4. Full trajectory (epoch → mean/median cosine), pooled histogram, and
   class-conditional (tumor vs. background) breakdowns.

## Results

### 1. Trajectory: mean cosine similarity stays small, positive, and stable

| Epoch | Mean cos | Median cos | Std | Min | Max | n |
|---|---|---|---|---|---|---|
| 1 | +0.0824 | +0.0810 | 0.156 | −0.560 | +0.586 | 14,543 |
| 5 | +0.1050 | +0.1069 | 0.162 | −0.435 | +0.585 | 5,345 |
| 10 | +0.0957 | +0.0999 | 0.182 | −0.505 | +0.606 | 8,267 |
| 15 | +0.0872 | +0.0927 | 0.191 | −0.560 | +0.618 | 6,434 |
| 20 | +0.0776 | +0.0728 | 0.196 | −0.595 | +0.589 | 2,821 |
| 25 | +0.1161 | +0.1190 | 0.200 | −0.543 | +0.645 | 6,101 |
| 30 | +0.1037 | +0.1035 | 0.192 | −0.577 | +0.633 | 4,096 |

Mean cosine similarity never drops below +0.08 or rises above +0.12
across the entire 30-epoch trajectory — this is a small, stable,
**mildly positive** relationship, not a large one, and importantly not
zero or negative at any checkpoint sampled. There is no visible drift
toward conflict as training progresses (epoch 20's local dip to +0.078
recovers by epoch 25, well within the noise band the other 6 epochs
already show).

### 2. Pooled distribution: predominantly cooperative-to-independent, genuine conflict is a real but minority behavior

Pooling all 47,607 anchor-level samples across all 7 checkpoints:

- **Mean cosine: +0.0938**, median +0.0952, std 0.178
- One-sample t-test vs. 0: t=114.8 (statistically distinguishable from
  zero at essentially any n this large — not a meaningful statement on
  its own; the **effect size**, not the p-value, is what matters here)
- **Sign-only split**: 69.97% of anchors have cos > 0, 30.03% have cos < 0
- **Thresholded, mutually-exclusive regime split** (|cos|<0.1 = practically
  independent, the more meaningful cut than sign alone): at epoch 30,
  **14.5% conflict** (cos ≤ −0.1), **34.7% independent** (−0.1 < cos <
  0.1), **50.8% cooperative** (cos ≥ 0.1). This ratio is stable across
  checkpoints (conflict share ranges 10.9%–19.0%, cooperative share
  45.1%–53.9%, independent 31.7%–42.2% — see plot, bottom-right panel).

The histogram (see `e14_gradient_conflict_results/gradient_conflict_plots.png`,
top-middle panel) is a single, roughly symmetric, mildly right-shifted
unimodal distribution — **not bimodal**, and not showing a distinct
"conflicting subpopulation." This argues against a scenario where some
identifiable subset of voxels experiences severe, persistent conflict
while the rest cooperate; the whole population sits on a continuum
centered slightly positive.

### 3. Class-conditional: tumor voxels are slightly more cooperative than background, but both are in the same regime

Pooled across all checkpoints: tumor voxels mean cos = **+0.1069**
(n=21,621), background voxels mean cos = **+0.0828** (n=25,986).
Welch's t-test: t=14.7, p<0.0001 — statistically distinguishable given
the large n, but the practical gap (0.024) is small; both classes sit
solidly in the same "mildly cooperative, not conflicting" regime. No
sign of the margin loss being systematically antagonistic to
segmentation for one class while cooperative for the other.

### 4. Gradient norm ratio: dominated by an epoch-1 initialization artifact, uninformative once excluded

| Epoch | Mean ratio | Median ratio |
|---|---|---|
| 1 | 52.75 | 28.12 |
| 5 | 2.62 | 0.61 |
| 10 | 5.18 | 1.33 |
| 15 | 5.71 | 1.81 |
| 20 | 4.54 | 1.08 |
| 25 | 5.85 | 1.53 |
| 30 | 6.12 | 1.14 |

Epoch 1's ratio (52.75) is an order of magnitude larger than every other
checkpoint (2.6–6.1) — this is a denominator artifact: `‖g_seg‖` is
still near-zero at the very start of training (the segmentation head has
barely begun learning), not a signal that the margin loss is
"overwhelming" segmentation at epoch 1. Including this point produces a
**spurious, sign-flipping correlation**: `corr(ratio, Dice) = −0.94`
(p=0.002, n=7) with epoch 1 included, but **+0.98** (p=0.001, n=6) with
it excluded — a textbook confound, not two different real findings. With
only 6-7 checkpoint-level data points either way, neither correlation
should be trusted as a real relationship; this is flagged explicitly as
a limitation (see below), not resolved by picking whichever sign looks
better.

**The more decisive number**: `corr(mean_cos, Dice)`, excluding the
epoch-1 outlier, is **−0.057 (p=0.914, n=6)** — essentially flat.
Cosine similarity itself shows no relationship with Dice across the
trajectory, positive or negative. This is the cleaner, outlier-robust
answer to "does more/less gradient agreement track better segmentation,"
and it says no.

## Interpretation: which regime does the project belong to?

Per the pre-specified 3-case framework:

- **Case A (cos ≈ +1, cooperate)**: not observed — mean cosine (~0.09)
  is far too small to represent genuine cooperation; a +1 cosine would
  mean the two gradients point in nearly the same direction, which is
  not the case here.
- **Case B (cos ≈ 0, independent objectives)**: **closest match**, with
  a slight positive lean. Roughly a third of anchors sit in the strict
  |cos|<0.1 "independent" band at any given checkpoint, and the pooled
  distribution is centered at a small positive value (+0.09), not
  exactly zero but far closer to 0 than to either ±1.
- **Case C (cos < 0, systematic interference)**: **not supported**. Mean
  cosine is positive at every single checkpoint measured (all 7), never
  crossing into negative territory even transiently. Genuine conflict
  (cos ≤ −0.1) does occur in a real, non-trivial minority of anchors
  (11–19% depending on checkpoint) — but "some conflict exists at the
  individual-anchor level" is a very different claim from "the two
  losses are systematically fighting," and the trajectory-level mean
  never approaches negative, let alone stays there.

**Overall regime determination: Case B, mildly leaning toward A.**
Segmentation and margin gradients are best described as weakly
cooperative / largely independent on the shared representation, with a
genuine but minority conflicting subpopulation that does not dominate or
grow over training.

## What this does and does not explain

**This result does not support gradient conflict as an explanation for
EGGO-M's Dice null.** If `L_seg` and `L_margin` were systematically
fighting on the shared `dec1` representation, the expected signature
would be Case C (negative mean cosine, ideally growing more negative or
staying persistently negative over training) — that signature is simply
not present in this data, at any of the 7 checkpoints sampled, across
47,607 anchor-level measurements. The margin loss is not actively
undoing the segmentation loss's progress on the shared trunk in any
gross, trajectory-level sense.

This leaves the E12f/E13 puzzle basically where it was: the margin
mechanism is verifiably active (E12e/E12f/E13) and does not appear to be
in gradient conflict with segmentation (E14) — yet Dice still does not
improve. The more likely remaining explanations, per the mildly
cooperative-but-flat relationship found here, point away from
*optimization* as the bottleneck and back toward something in the
*sufficiency of the geometric mechanism itself*: e.g., the margin
signal, even when cooperating with (not fighting) segmentation, may
simply be too weak, too sparse (only ~2000/65,536 voxels per volume are
ever sampled as anchors), or targeting a property (latent margin width)
that is not actually the thing limiting Dice at this architecture's
ceiling — consistent with the E1.3/E7 finding that latent boundary AUC
saturates almost immediately regardless of intervention.

## Limitations

1. **Single seed (seed 0), single checkpoint set.** This entire analysis
   reuses E12f's seed-0 checkpoints only, not all 4 E13 seeds. Given
   E13's own finding that margin-Dice correlation is itself highly
   seed-dependent (ranging from r=−0.60 to r=+0.96 across seeds), it is
   possible gradient-conflict statistics also vary meaningfully by seed
   — not tested here, and would be a natural (cheap) extension if this
   thread is pursued further.
2. **7 checkpoints is a small trajectory-level sample.** The
   epoch-vs-ratio and epoch-vs-cos-vs-Dice correlations are built from
   only 6-7 points; the sign-flipping ratio correlation demonstrates
   directly how fragile conclusions from this few points can be. The
   anchor-level statistics (n=47,607) are far more robust and are what
   this report's conclusions primarily rest on.
3. **8 batches per checkpoint** (not the full epoch) — a computational
   concession, not exhaustive coverage of that epoch's data distribution.
   Chosen to keep this diagnostic phase cheap; the anchor-level sample
   size (thousands per checkpoint) is still large enough for the
   pooled/regime-share statistics to be stable.
4. **Unweighted gradients.** Per the spec, this measures conflict between
   the raw `L_seg` and `L_margin` terms, not their λ/μ-weighted training
   contributions. The applied `λ=0.1` weighting shrinks `g_margin`'s
   actual contribution to the combined gradient further still — the
   norm-ratio numbers here (already order-1 to order-10 unweighted)
   would be roughly 10x smaller in the actually-applied, weighted sense.
   This does not change the cosine-similarity conclusion (weighting a
   vector by a positive scalar does not change its direction), but is
   worth keeping in mind when interpreting the norm-ratio panel.
5. **This measures gradient DIRECTION agreement on dec1 only** — it does
   not test whether the margin loss's gradient, even when cooperative in
   direction, is actually large/frequent enough in practice to move
   Dice (a magnitude/frequency question, partially already addressed by
   E12d/E12e/E12f/E13's active-hinge-percentage tracking, which found
   the mechanism active but sparse — 0.12-0.22% active hinge pairs by
   epoch 30).

## Recommendation

**Is optimization conflict sufficiently supported to justify researching
gradient-conflict-aware optimization methods (PCGrad, CAGrad, or
similar)? No.**

The data shows mean gradient cosine similarity is positive at every
checkpoint measured (range +0.078 to +0.116), with roughly half of
anchors showing meaningful cooperation (cos≥0.1) and only 11-19% showing
meaningful conflict (cos≤−0.1) at any given checkpoint — the opposite of
the profile (persistent, dominant negative cosine) that would justify
investing in gradient-conflict-aware optimization machinery. This
project's own earlier ABO/Phase-B finding that gradient conflict between
the focal and evidential branches was "transient, not persistent" is now
echoed by a second, independent measurement on a different loss pair
(segmentation vs. margin): conflict-resolution methods do not appear to
target a real bottleneck in this codebase's training dynamics, for
either loss pairing investigated so far.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e14_gradient_conflict_analysis.py` | Analysis script |
| `experiments/exp_e12_eggo_m/e14_gradient_conflict_results/` | Raw data (JSON), epoch summary, plots |
| `experiments/exp_e12_eggo_m/e12f_pilot_calibrated_seed0/checkpoints/` | Checkpoints analyzed (same set E12f/E13 already used) |
| `PHASE_E13_MULTISEED_RESULTS.md` | The Dice-null result this phase investigates a possible cause of |

---

**Completed**: 2026-08-07 — Case B (mildly cooperative-to-independent),
not Case C. Gradient conflict does not explain EGGO-M's Dice null;
optimization is not supported as the bottleneck by this data.
