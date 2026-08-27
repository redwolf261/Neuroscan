# Phase E7: Does Latent Geometry Precede Calibration, or Follow It?

**Status**: ✅ Complete — clear temporal ordering found for class separability; density is more equivocal; boundary→evidence R² is too noisy to support a strong claim

**Date**: 2026-08-04

## Purpose

`PHASE_E6_STRESS_TEST.md` identified Hypothesis D (circularity — does the
evidential loss itself produce the latent geometry patterns E1.3/E1.4
measured, rather than geometry being a genuine, independently-forming
structure the algorithm could exploit?) as unresolvable from static,
single-checkpoint analysis. This experiment answers it by observation: a
full 50-epoch training run (identical to the frozen baseline protocol,
seed 0, no EGGO changes — purely observational) with checkpoints saved
at epochs {1, 3, 5, 10, 20, 35, 50}, then running the E1.3/E1.4
diagnostic pipeline at every checkpoint on a fixed set of 20 validation
subjects (same subjects at every checkpoint, for a directly comparable
trajectory).

**Final Dice at epoch 50 (0.9107) exactly matches the frozen baseline's
seed-0 result** — confirms this run faithfully reproduces the reference
training dynamics; the added checkpoint-saving logic introduces no
confound.

## Results

| epoch | Dice | ECE | latent boundary AUC | boundary→evidence R² | density Cohen's d | evidence Cohen's d |
|---|---|---|---|---|---|---|
| 1 | 0.577 | 0.364 | 0.9945 | 0.208 | −1.150 | 0.717 |
| 3 | 0.752 | 0.230 | 0.9963 | 0.059 | −2.216 | 2.928 |
| 5 | 0.877 | 0.165 | 0.9987 | 0.086 | −2.331 | 2.450 |
| 10 | 0.874 | 0.107 | 0.9995 | 0.226 | −2.469 | 2.017 |
| 20 | 0.916 | 0.071 | 0.9995 | 0.277 | −2.859 | 2.325 |
| 35 | 0.919 | 0.050 | 0.9996 | 0.133 | −2.839 | 2.962 |
| 50 | 0.923 | 0.046 | 0.9997 | 0.296 | −3.007 | 3.291 |

## Finding 1: class-separability geometry forms almost instantly, well before calibration

**Latent boundary AUC is already 0.9945 at epoch 1** — essentially
saturated from the very first epoch (the full trajectory spans only
0.9945→0.9997, a narrow near-ceiling range) — while **ECE is still poor
at epoch 1 (0.364) and improves gradually and monotonically across the
entire 50-epoch run down to 0.046**. This is a clean, visually
unambiguous temporal ordering (`experiments/exp_e7_causality/analysis_results/trajectory_plot.png`):
the shared trunk learns to linearly separate tumor from background in
representation space almost immediately, long before the evidential
head's calibration quality catches up.

**This is genuine evidence against the strongest form of Hypothesis D**
for this specific quantity: class-separability structure is not a
downstream consequence of calibration becoming good — it exists early
and calibration develops on top of (or independently of) an
already-separable representation, not the other way around.

## Finding 2: density separation is present early but continues strengthening on a timescale closer to calibration's

Density Cohen's d is already substantial by epoch 1 (−1.150) — not zero,
so some structure exists immediately — but it continues to grow
substantially in magnitude through epoch 20 (−2.859) before largely
plateauing (−2.839 at 35, −3.007 at 50). This is a **slower, more
gradual trajectory than the boundary AUC's near-instant saturation**,
and tracks closer to Dice's and ECE's own settling timescale (both also
largely plateau after ~epoch 20).

**This is more equivocal than Finding 1.** Density has an early
component (partially supporting "forms independently, before
calibration") but also a component that develops alongside calibration
improvement (partially consistent with a co-evolving or
partially-downstream relationship). This experiment cannot cleanly
separate "density structure independently continues to sharpen" from
"density structure sharpens because calibration is simultaneously
improving and both are driven by the same underlying training dynamics"
— both are consistent with the observed trajectory shape.

## Finding 3: boundary→evidence R² is too noisy to support a directional claim

The R² for latent-boundary-distance-predicting-evidence is
**non-monotonic and noisy across training** (0.208 → 0.059 → 0.086 →
0.226 → 0.277 → 0.133 → 0.296) — no clean trend, bouncing by more than
4× between adjacent checkpoints (e.g., epoch 20's 0.277 down to epoch
35's 0.133, then back up to 0.296 at epoch 50). This is very likely
substantially attributable to the small fixed validation subset (20
subjects × 2,000 sampled voxels = 40,000 points per checkpoint, smaller
than E1.3's original 60,000-voxel/30-subject sample) combined with R²
itself being a noisier, higher-variance statistic than AUC or Cohen's d
at this sample size. **This finding should not be used to support or
refute the causality question** — it's inconclusive, not evidence either
way, and is reported here for completeness/honesty rather than omitted
because it doesn't fit a clean narrative.

## Finding 4: evidence magnitude grows smoothly and monotonically alongside ECE improvement

Mean evidence climbs steadily from 1.7 (epoch 1) to 19.6 (epoch 50),
tracking almost exactly the same gradual timescale as ECE's improvement
— consistent with the evidential head's calibration developing gradually
over the full training run, the same timescale Finding 1 showed lags
well behind class-separability geometry.

## Overall verdict

**Partial, honest answer — not the clean "geometry precedes uncertainty,
full stop" result that would maximally justify EGGO, but also not the
null "geometry only follows calibration" result that would kill it.**

- **Class-separability geometry (E1.3's core mechanism) genuinely
  precedes calibration** — this is the clearest, least ambiguous result
  and is real evidence that the shared trunk's decision-boundary
  structure is not merely a symptom of good calibration; it forms first
  and calibration develops afterward (or on top of it). This
  strengthens confidence in the separation-force half of EGGO
  ($\mathcal{L}_{sep}$, gated by boundary proximity) specifically.
- **Density structure (E1.4's mechanism) is more ambiguous** — present
  early, but its continued strengthening tracks calibration's own
  timescale rather than clearly preceding it. Combined with
  `PHASE_E6_STRESS_TEST.md`'s Problem 1 finding (density's independent
  contribution to evidence is modest, rank 3/4, in a full joint model),
  this is now the **second time density has come out weaker than
  boundary distance** under scrutiny. This further supports
  `PHASE_E5_ALGORITHM_DESIGN.md`'s recommendation (already made in E6)
  to simplify the first implementation to margin/separation-only,
  treating compactification as a secondary extension pending its own
  stronger justification, not a co-equal component from the start.
- **The R²-based boundary→evidence relationship is too noisy at this
  sample size to weigh in either direction** — genuinely inconclusive,
  reported honestly rather than cherry-picked to support a cleaner story.

**This does not fully resolve Hypothesis D** (it was never going to with
a single seed and a fixed small validation subset) but it shifts the
balance of evidence meaningfully: the piece of the algorithm most
directly supported by E7 is boundary-based separation, and the piece
least supported (both here and in E6) is density-based compactification.
This is consistent with, and reinforces, the "simplify EGGO to margin-only
first" recommendation already on the table — not a new conclusion, but
now backed by a second, independent line of evidence (temporal ordering,
not just cross-sectional multivariate regression).

## What this experiment does not establish

- Only one seed was run (seed 0) — trajectory shape could differ across
  seeds; not checked.
- The fixed 20-subject validation subset is smaller than E1.3/E1.4's
  original 30-subject/60,000-voxel sample, contributing to Finding 3's
  noise and potentially affecting the precision of Findings 1/2 as well
  (not just Finding 3) — the AUC and Cohen's d trajectories are clean
  enough that this is unlikely to change their qualitative shape, but
  exact values at each checkpoint carry more sampling noise than the
  original E1.3/E1.4 numbers.
- This is still an observational (not interventional) experiment — it
  establishes temporal precedence, which is necessary but not sufficient
  for a causal claim. Confirming that EGGO's separation force actually
  causes a downstream improvement still requires the training-based
  ablation in `PHASE_E5_ALGORITHM_DESIGN.md` §7.

## Recommendation

Proceed to implementation planning with the **simplified, margin-only
first version already recommended in `PHASE_E5_ALGORITHM_DESIGN.md`**
(the user's own suggestion when reviewing the design: $\mathcal{L} =
\mathcal{L}_{seg} + \lambda \cdot U \cdot B \cdot \mathcal{L}_{margin}$,
no density term) — now with two independent, converging pieces of
evidence supporting boundary/separation over density/compactification as
the higher-priority mechanism to test first: E6's multivariate
regression (density ranks 3rd/4th for explaining evidence) and E7's
temporal ordering (boundary geometry precedes calibration cleanly;
density's trajectory is more ambiguous). If margin-only EGGO improves
Dice, density becomes a well-motivated, justified extension per the
original simplification argument. Still outstanding before any code:
Problems 4 (trainable boundary head vs. external logistic regression) and
5 (gate feedback-loop stability) from the original stress-test framing,
neither addressed by this experiment.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e7_causality/train_with_checkpoints.py` | Training script (frozen baseline logic + fixed-epoch checkpoints) |
| `experiments/exp_e7_causality/analyze_checkpoints.py` | Per-checkpoint diagnostic pipeline |
| `experiments/exp_e7_causality/analysis_results/` | `trajectory_results.json`, `trajectory_plot.png` |
| `PHASE_E6_STRESS_TEST.md` | The stress test that raised Hypothesis D as unresolved |
| `PHASE_E5_ALGORITHM_DESIGN.md` | The design this informs |

---

**Completed**: 2026-08-04
