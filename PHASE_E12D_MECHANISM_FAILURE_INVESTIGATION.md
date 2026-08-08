# Phase E12d: Mechanism Failure Investigation

**Status**: ✅ Complete — root cause identified, decisively. Two compounding hyperparameter miscalibrations, not a fundamentally wrong hypothesis (H4-original) or a broken implementation.

**Date**: 2026-08-05

## Purpose

`PHASE_E12B_PILOT_RESULTS.md` found that EGGO-M's pilot run did not grow
the boundary margin — the exact quantity $\mathcal{L}_{margin}$ was
designed to optimize. Per the four candidate explanations set out before
this investigation (H1: hinge rarely active; H2: λ too weak relative to
$\mathcal{L}_{seg}$'s gradient; H3: gate saturates to near-zero; H4:
latent separation already near ceiling, no room to act), this phase
tests all four directly rather than guessing or jumping to a
hyperparameter sweep. All diagnostics are post-hoc analysis of the
existing E12b checkpoints plus one live gradient-measurement pass — no
retraining was needed.

## Result: H1, H3, and H4 are all supported, and they are one coherent story, not three separate problems

| Epoch | Active hinge % | Mean pairwise distance | B_i mean | UB (applied weight) frac near 0 |
|---|---|---|---|---|
| 1 | 0.65% | 34.29 | 0.556 | 0.005% |
| 5 | 8.28% | 19.00 | 0.037 | 0.05% |
| 10 | 1.09% | 31.95 | 0.010 | 0.46% |
| 15 | 0.87% | 30.14 | 0.0044 | 2.14% |
| 20 | 0.87% | 29.35 | 0.0027 | 97.02% |
| 25 | 0.00% | 29.60 | 0.0023 | 98.21% |
| 30 | 0.00% | 31.17 | 0.0018 | 98.60% |

### H4 (margin threshold miscalibrated to the feature space's actual scale) is the root cause

The hinge threshold used was `2×δ_d = 2.0`. The **actual mean pairwise
distance between opposite-class embeddings was 19–34 throughout
training** — more than an order of magnitude larger than the threshold,
from epoch 1 onward. The distance histogram
(`e12d_results/mechanism_diagnosis_dashboard.png`, Panel 3) makes this
visually unambiguous: the `2×δ_d` threshold sits at the extreme left
edge of a distribution whose bulk sits between 20 and 40. **δ_d=1.0 was
never a meaningful threshold for this feature space — it was
miscalibrated from the start**, not something that only broke down after
training progressed.

### H1 (hinge rarely active) is a direct, mechanical consequence of H4

Because almost no sampled opposite-class pairs are ever closer than
`2×δ_d`, the hinge term `[2δ_d − dist]₊²` is zero for the overwhelming
majority of pairs at every checkpoint — 0.65% active at epoch 1,
decaying to exactly 0.00% by epoch 25. This is not evidence the
implementation is broken; it is the correct, expected behavior of a
hinge loss whose threshold is far below the natural scale of the data it
operates on.

### H3 (gate collapse) is a second, independent, compounding problem — not the same issue as H4

This is the important nuance: **`B_i` collapses toward zero on its own**,
independent of the margin/hinge issue. `B_i = exp(-|d_i|/\tau_b)` with
`τ_b ≈ 0.971` (derived in `PHASE_E5_ALGORITHM_DESIGN.md` §2.2 from E1.3's
measured correct/incorrect crossover on a **fully-converged, static**
checkpoint). But the boundary head's logits `|d_i|` grow rapidly in
magnitude as it converges during *live* training (consistent with E7's
finding that boundary AUC saturates almost immediately) — and since
`B_i` decays exponentially in `|d_i|/\tau_b`, a boundary head that's
already highly confident by epoch 10–15 pushes `B_i` toward zero for
nearly all voxels. By epoch 20, **97% of sampled voxels have a
near-zero combined weight** ($\hat U_i \cdot B_i < 0.001$), and by epoch
30 this reaches 98.6%. Panel 6a in the dashboard shows this starkly: the
`B_i` histogram (orange) is compressed almost entirely into the very
first bin, while `U_hat` (teal) remains reasonably spread across its
range — confirming this is specifically a `τ_b`/`B_i` problem, not a
`U_hat`/evidential-head problem.

**τ_b was calibrated on a diagnostic quantity's distribution on a
finished model, not on the live, still-converging boundary head's own
logit-magnitude growth during training** — a mismatch the original
design spec (`PHASE_E5_ALGORITHM_DESIGN.md` §2.2) flagged as a real
possibility ("must be re-derived from the full training set... treat as
a sweep starting point, not a final constant") but which this pilot
confirms is a live, significant issue, not a hypothetical caveat.

## H2 (λ too weak): partially supported, more nuanced than the others

Measured `||∇_dec1 L_seg||` and `||∇_dec1 L_margin||` independently (via
`torch.autograd.grad`, one fixed real training batch, at every
checkpoint) — **before** λ scaling is applied:

| Epoch | ‖∇L_seg‖ | ‖∇L_margin‖ (pre-λ) | Ratio (margin/seg) |
|---|---|---|---|
| 1 | 5.23×10⁻⁵ | 1.27×10⁻⁴ | **2.43** |
| 5 | 2.31×10⁻⁴ | 2.31×10⁻⁵ | 0.10 |
| 10 | 1.73×10⁻⁴ | 8.65×10⁻⁶ | 0.05 |
| 15 | 1.61×10⁻⁴ | 4.10×10⁻⁶ | 0.025 |
| 20 | 1.51×10⁻⁴ | 4.69×10⁻⁶ | 0.031 |
| 25 | 1.43×10⁻⁴ | 5.53×10⁻⁶ | 0.039 |
| 30 | 1.40×10⁻⁴ | 4.98×10⁻⁶ | 0.036 |

**Surprising finding, worth being precise about**: the gradients are
*not* literally zero, and at epoch 1 the margin gradient was actually
**larger** than the segmentation gradient (ratio 2.43) before decaying
to roughly 3–10% of `L_seg`'s magnitude for the rest of training. This
is consistent with H1/H3/H4, not contradictory: since the hinge loss
value is driven to near-zero by the tiny active-pair fraction, the
residual gradient comes from whatever small number of pairs remain
active — a small but genuinely nonzero signal, not a dead loss term in
the strict sense.

Once λ=0.1 is applied, the *effective* contribution to the total
gradient at dec1 is roughly 0.3–1% of $\mathcal{L}_{seg}$'s magnitude by
mid-to-late training (0.1 × 0.03–0.10) — small, and consistent with the
observed lack of margin growth, but this is a **downstream consequence
of H1/H3/H4's near-zero active loss**, not an independently identified
"λ chosen too small in isolation" problem. Increasing λ alone, without
fixing δ_d and τ_b, would mostly just amplify a signal that is itself
nearly zero for the wrong (H1/H4) reason — not a fix on its own.

## Overall verdict: not "the hypothesis targeted the wrong bottleneck" (the more interesting H4-alternative reading) — this is a calibration failure, testable and fixable

This is worth being precise about, since the original four-hypothesis
framing included a version of H4 asking whether "latent margin was not
the limiting factor for segmentation performance" (a stronger,
more publication-interesting negative result). **That is not what was
found here.** What was found is that **two specific constants
(`δ_d=1.0`, `τ_b≈0.971`) were set at scales badly mismatched to this
particular feature space's actual geometry**, causing the loss to be
almost entirely inert through a combination of an inactive hinge and a
collapsed gate — both mechanically explainable, both traceable to
specific numbers, and both plausibly fixable by recalibrating those
constants to the feature space's actual measured scale, rather than
values derived from image-space intuition (`δ_d=1.0`, chosen without a
specific derivation in `PHASE_E5_ALGORITHM_DESIGN.md`) or a
static-checkpoint diagnostic (`τ_b`, correctly flagged as provisional at
the time but not yet re-derived).

**This means the "wrong bottleneck" conclusion cannot yet be drawn.**
The margin-growth prediction failing is consistent with a
mechanism that was never actually given a fair chance to operate in this
pilot, not evidence the mechanism itself is unhelpful. A properly
recalibrated version needs to be tested before the "was margin the
wrong target" question can be answered honestly.

## Recommended next step

Recalibrate both constants from the pilot's own measured data, mirroring
exactly the "derive from measurement, not intuition" discipline used
throughout Phase E and for ABO's own constants:

1. **`δ_d`**: should be set relative to the *actual* observed pairwise
   distance distribution, not an arbitrary value like 1.0. A reasonable
   starting point: set `2×δ_d` near the *lower* tail of the observed
   distance distribution (e.g., a low percentile of same/opposite-class
   pairwise distances at initialization), so a meaningful fraction of
   pairs are initially active and the hinge has real pairs to act on —
   analogous to how `τ_b`/`r_target` were derived from measured
   crossover points elsewhere in this project, not picked from a
   different domain's typical scale.
2. **`τ_b`**: should be re-derived from the boundary head's own
   logit-magnitude distribution as it evolves *during* live EGGO-M
   training (not from the offline, fully-converged E1.3 classifier),
   since this pilot shows the two diverge substantially by epoch 10–15.
   A live-recalibration approach (e.g., an EMA of `|d_i|`'s scale,
   updated during training, rather than one fixed constant) may be
   more robust than a single static value picked once — worth
   considering, not yet decided.
3. Only **after** both constants are recalibrated should E14's λ sweep
   be run — running it now would be sweeping a scalar multiplier on a
   near-dead signal, which per the gradient-norm analysis above would
   not distinguish "λ too small" from "the underlying quantity being
   scaled is itself near zero for unrelated reasons."

**Do not yet proceed to E13 (multi-seed)** — per the original framing,
running more seeds of a mechanism now specifically diagnosed as
miscalibrated (not just "unclear") would not answer a new question.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e12d_mechanism_diagnosis.py` | Post-hoc H1/H3/H4 diagnostic (active hinge %, distance histograms, U/B/UB distributions) |
| `experiments/exp_e12_eggo_m/e12d_gradient_norms.py` | Live gradient-norm measurement (H2) |
| `experiments/exp_e12_eggo_m/e12d_results/` | JSON results + dashboard plot |
| `PHASE_E12B_PILOT_RESULTS.md` | The pilot result this investigation follows up on |
| `PHASE_E5_ALGORITHM_DESIGN.md` | Original δ_d/τ_b derivation (§2.2, §4.1) this investigation revises |

---

**Completed**: 2026-08-05 — recommend recalibrating δ_d and τ_b before any further sweep or multi-seed run
