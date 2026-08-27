# Phase E17: Margin Target Stability Analysis

**Status**: ✅ Complete — measurement-only, no retraining, no changes to model/loss/optimizer/hyperparameters anywhere in this phase. **The target moves. Early training (epochs 1–5) shows severe instability across every measure (centroid direction, gradient direction, and whole-representation rotation). From epoch 10 onward, the representation and gradient direction both stabilize sharply and consistently. But the margin gradient's direction relative to epoch 1 never recovers — it stays permanently rotated (~0.13–0.15 cosine, essentially flat) even once local (consecutive-epoch) stability is high. This is consistent with, and sharpens, E16's finding: a locally-correct, increasingly locally-stable gradient that nonetheless never re-aligns with its own early trajectory, which is exactly the signature of a moving-target/representation-reorganization problem, not a magnitude or conflict problem.**

**Date**: 2026-08-07

## Purpose

E16 found the margin loss's gradient is consistently, correctly oriented
toward the direction E15 proved beneficial (pooled cosine +0.39 with
E15's push direction, positive at every checkpoint) — yet net
accumulated displacement along that direction is small and slightly
negative. The reviewer's hypothesis: this pattern (locally correct, but
failing to integrate into large net displacement) is the signature of a
**moving target** — if the class-separating direction itself rotates
epoch-to-epoch, "move toward separating" means something different every
time, and correct instantaneous gradients cannot accumulate into a large
coherent push. This phase tests that directly, using only existing
checkpoints, no new training.

## Methodology

Same E12f seed-0 checkpoints (epochs 1, 5, 10, 15, 20, 25, 30) and same
fixed 20-subject validation set as every prior phase in this arc
(E12b.5/E12d/E12f/E13/E14/E15/E16).

### Part 1 — Centroid direction stability

For each checkpoint, the tumor-minus-background centroid direction
(unit vector) is computed over a fixed 10,000-voxel tracking set (500
per subject, same sampling scheme/seed as E16's Parts A/B/D, so results
are directly cross-referenceable). `cos(dir_t, dir_{t+1})` is measured
between every consecutive checkpoint pair, plus `cos(dir_1, dir_t)` for
every checkpoint against the earliest available reference (epoch 1;
no true epoch-0 checkpoint exists — same limitation flagged in E16,
carried forward here).

### Part 2 — Margin gradient stability (consecutive)

`cos(g_margin(t), g_margin(t+1))` between consecutive checkpoints, on a
**fixed** anchor set.

**A bug was found and fixed during smoke-testing, before trusting any
result**: the first attempt reused Part 1's small (100–500/subject),
*uniformly random* tracking set for the gradient computation too. This
produced `margin_loss = 0.0` (and therefore a literal zero gradient, not
a computation error) at several checkpoints in the smoke test — traced
to the fact that `compute_margin_loss`'s hinge term is only "active"
for a small fraction of pairs (0.1–3.3%, per every prior phase's own
measurement back to E12e), a rate real training only achieves because of
its **stratified, low-evidence anchor sampling** — a small uniform
random sample essentially never lands on an active pair. **Fixed** by
selecting a much larger (2000/subject, 40,000 total, matching
`ANCHORS_PER_VOLUME` from `train_eggo_m.py` exactly), **stratified**
(low-evidence, using the checkpoint-1 evidence values to decide
inclusion) anchor set, chosen **once** and reused at every subsequent
checkpoint — fixed identity (required for gradient vectors at different
epochs to be directly comparable) combined with stratified selection
(required to actually land on voxels where the hinge fires). This is
deliberately different from E16 Part C, which used each checkpoint's
own *live* stratified sample (correct for E16's different question —
"what does the real training-time gradient look like at this instant" —
but wrong for E17's question, which needs the same voxels tracked
across time).

### Part 3 — Gradient transport (long-range)

Same fixed stratified anchor set as Part 2. `cos(g_margin(5), g_margin(30))`
computed directly (the specific comparison requested), plus the full
`cos(g_margin(1), g_margin(t))` trajectory for every checkpoint, to see
the complete long-range decay/recovery shape, not just the two
endpoints.

### Part 4 (added) — Representation rotation via orthogonal Procrustes

Not in the original request; added to separate two related but distinct
phenomena the reviewer's own explanation list named separately: "the
class-separating axis specifically rotates" (Parts 1–3) vs. "the whole
representation is being reorganized" (this part). For each consecutive
checkpoint pair, `orthogonal_procrustes` finds the best rigid
rotation/reflection `R` mapping the (mean-centered) tracked point cloud
at epoch `t` onto epoch `t+1`. Two summaries: **rotation closeness to
identity** (`trace(R)/32` — 1.0 means no rotation needed, i.e. the point
cloud didn't need to be spun to match; lower means more rotation was
needed) and **residual fraction** (how much of the change from `t` to
`t+1` is *not* explained by a pure rotation — genuine reshaping, not
just the cloud spinning in place).

## Results

### Part 1: Centroid Direction Stability — severe early instability, sharp later stabilization, moderate permanent long-range drift

| Consecutive pair | cos(dir_t, dir_t+1) | dist_t → dist_t+1 |
|---|---|---|
| 1→5 | **0.842** | 25.56 → 11.80 |
| 5→10 | 0.928 | 11.80 → 21.40 |
| 10→15 | 0.987 | 21.40 → 23.67 |
| 15→20 | 0.991 | 23.67 → 22.51 |
| 20→25 | 0.993 | 22.51 → 22.90 |
| 25→30 | **0.999** | 22.90 → 24.09 |

| Epoch | cos(dir_1, dir_t) — long-range vs. epoch 1 |
|---|---|
| 5 | 0.842 |
| 10 | 0.828 |
| 15 | 0.817 |
| 20 | 0.791 |
| 25 | 0.784 |
| 30 | **0.781** |

The consecutive-pair series shows a clean, monotonic story: the biggest
single rotation happens between epochs 1 and 5 (cos=0.842, i.e. roughly
a 33° rotation), then stability improves steadily and monotonically
every subsequent step, reaching near-perfect (0.999, ~2.6°) by epochs
25→30. **This matches the well-documented epoch-5 instability window**
seen independently in E12f's own margin trajectory (27.83→14.16 dip) and
E16's Part A tumor-centroid-drift spike (16.2 at epoch 5) — a third,
independent confirmation via yet another measurement method that
something specifically disruptive happens around epoch 5, not an
artifact of any one pipeline.

The long-range series (vs. epoch 1) tells a different, complementary
story: it does **not** recover back toward 1.0 as the consecutive-pair
series stabilizes — it settles at a **permanently offset** 0.78 by epoch
30. The direction has rotated by a real, fixed amount (~39°) relative to
where it started, and stays there — training does not "come back around"
to the original separating direction even once it stops actively
rotating further.

### Part 2: Margin Gradient Stability (consecutive) — mirrors Part 1's shape almost exactly

| Consecutive pair | n | mean cos | median | std |
|---|---|---|---|---|
| 1→5 | 12,402 | **0.142** | 0.141 | 0.212 |
| 5→10 | 19,094 | 0.537 | 0.595 | 0.280 |
| 10→15 | 16,488 | 0.645 | 0.722 | 0.274 |
| 15→20 | 7,391 | 0.637 | 0.744 | 0.297 |
| 20→25 | 6,890 | 0.709 | 0.802 | 0.245 |
| 25→30 | 6,569 | **0.704** | 0.743 | 0.240 |

Pooled across all consecutive pairs: mean=0.535, std=0.326, n=68,834.

The gradient direction's stability follows the **same qualitative
trajectory** as the centroid direction (Part 1): weakest right after the
epoch-5 instability window (0.142, essentially uncorrelated — a genuine
near-reset), then climbing steadily and substantially to 0.70+ by
epochs 20–30. This is a real, non-trivial degree of local stabilization
— by late training, the gradient direction from one checkpoint to the
next is reasonably (though not perfectly) predictable, not chaotic.

### Part 3: Gradient Transport (long-range) — never recovers, unlike the consecutive-pair trend

`cos(g_margin(5), g_margin(30))`: **mean=0.417, median=0.477, std=0.297,
n=8,692** — a moderate positive long-range correlation across the 25-epoch
gap, meaningfully above zero but far from the 0.70+ consecutive-pair
values at the same late-training point.

| Epoch | n | cos(g_margin(1), g_margin(t)) |
|---|---|---|
| 1 | 14,835 | 1.000 |
| 5 | 12,402 | 0.142 |
| 10 | 10,041 | 0.142 |
| 15 | 9,000 | 0.152 |
| 20 | 5,021 | 0.134 |
| 25 | 5,907 | 0.139 |
| 30 | 4,512 | **0.136** |

This is the most important number in the phase. Once the gradient
direction departs from its epoch-1 orientation (dropping to 0.142 by
epoch 5), **it never recovers** — every subsequent checkpoint measured
against epoch 1 stays flat in a narrow 0.13–0.15 band all the way to
epoch 30, essentially uncorrelated with a large but persistent offset,
never trending back up. This directly parallels Part 1's centroid
finding (settles at 0.78, does not return to 1.0) but is far more
extreme for the gradient (settles at ~0.14, barely above zero) —
**the margin gradient's overall orientation from epoch 5 onward is
almost entirely different from what it was at epoch 1, and stays that
way for the rest of training**, even while it becomes locally
self-consistent (Part 2's rising consecutive-pair cosine).

### Part 4: Representation Rotation (Procrustes) — independently confirms the same two-phase shape

| Consecutive pair | Rotation closeness to identity | Residual after best rotation |
|---|---|---|
| 1→5 | **0.468** | 1.142 |
| 5→10 | 0.930 | 0.487 |
| 10→15 | 0.948 | 0.415 |
| 15→20 | 0.926 | 0.388 |
| 20→25 | 0.986 | 0.314 |
| 25→30 | **0.993** | 0.191 |

A completely independent measurement method (whole point-cloud rigid
alignment, not tied to any specific direction or the margin loss at
all) shows the identical two-phase shape: a large rotation needed
between epochs 1 and 5 (closeness=0.468, meaning a substantial rotation
was required to align the two point clouds), followed by a steady climb
toward near-identity (0.993 by epoch 25→30, almost no rotation needed —
the representation has become close to rigidly stable checkpoint-to-
checkpoint). The residual fraction (how much of the change is *not*
explained by rotation alone — genuine non-rigid reshaping) also
decreases steadily and monotonically (1.14 → 0.19), meaning both the
rotational and non-rotational components of representational change
shrink together as training progresses.

**Three independent methods — geometric centroid direction (Part 1),
actual loss gradient direction (Part 2/3), and whole-point-cloud rigid
alignment (Part 4) — all converge on the same two-phase story, measured
completely independently of each other.** This is a strong triangulation,
not a single fragile measurement.

## Interpretation

The reviewer's specific hypothesis was: does the optimization target
itself drift, such that locally-correct gradients (established in E16
Part C) fail to accumulate into large net displacement (E16 Part B/D)
because they are chasing a moving objective? **This is substantially
supported, with an important refinement the data itself reveals**: the
target does not drift *continuously* throughout training — it drifts
sharply and severely during a specific, identifiable early window
(epochs 1–5), then **stabilizes** (all four measures agree: by epoch
20–30, consecutive-step rotation is minimal and gradient/centroid
directions are becoming locally predictable). What does **not** happen
is a "return to alignment" with the original (epoch-1) target — the
representation and gradient settle into a **new, different, but now
stable** orientation, not the original one.

This reframes E16's puzzle precisely: E16 Part C's positive gradient-vs-
E15-push-direction alignment was measured using **each checkpoint's own
live anchors and its own current opposite-class centroid** (a
"locally correct, judged against the current target" measurement) — and
that finding stands, unmodified by anything here. What E17 adds is that
the *target the gradient is locally correct with respect to* is itself
not the same target from one part of training to another, especially
early on. A gradient that is always locally right, aimed at a target
that rotates substantially in the first several epochs before settling
into a persistently different orientation, is fully consistent with
E16's finding of small/negative *net signed* displacement measured
against the fixed epoch-1 reference direction — that fixed reference
is, per this phase's own measurement, an increasingly poor description
of "the useful direction" as training progresses past epoch 5.

This is meaningfully different from — and more specific than — simply
"segmentation loss overwhelms margin loss" (the explanation this phase
was explicitly asked not to assume, and which the data does not
straightforwardly support): the rotation is a property of the whole
representation (Part 4 shows it independent of the margin loss
specifically), consistent with something like BatchNorm renormalization,
general representation learning dynamics, or the segmentation loss
continually reshaping the feature basis being the driver — all four of
the reviewer's candidate mechanisms remain consistent with what's
measured here, and this phase does not have the evidence to distinguish
between them (see Limitations).

## Limitations

1. **No true epoch-0 checkpoint** — same limitation as E16; `epoch_1` is
   the earliest available reference, so "drift from initialization" is
   understated by whatever rotation happened during epoch 1 itself.
2. **Cannot distinguish which of the reviewer's four candidate causes is
   responsible.** Part 4's finding (whole-representation rotation, not
   specific to the margin direction) rules out an explanation unique to
   the margin loss's own dynamics, but is equally consistent with target
   drift from segmentation-driven reshaping, generic representation
   learning/BatchNorm renormalization, or some combination — this phase
   was scoped as measurement-only and does not attempt to isolate the
   specific mechanism.
3. **Part 2/3's fixed anchor set uses epoch 1's evidence values to decide
   stratified inclusion**, then reuses those same physical voxels at
   every later checkpoint even though each checkpoint's *own* evidence
   distribution would select a somewhat different set. This is the
   correct choice for this phase's question (fixed identity is required
   for gradient-vector comparability across time) but means the tracked
   anchors are specifically "voxels that were high-uncertainty at epoch
   1" — not necessarily still the highest-uncertainty voxels by epoch
   30. A different, live-restratified comparison (like E16 Part C) would
   answer a different, already-answered question.
4. **Single seed, single checkpoint chain** (E12f seed 0), consistent
   with E14/E15/E16 but not yet extended to E13's other 3 seeds.
5. **Procrustes rotation (Part 4) is computed on the mean-centered point
   cloud as a whole** (10,000 tracked voxels, uniform sample, not
   class-conditional) — it answers "does the overall geometric
   arrangement rotate," not specifically "does the tumor-vs-background
   separating axis rotate" (that's Part 1's job); the two parts are
   complementary, not redundant, but a class-conditional Procrustes
   analysis (separately for tumor and background point clouds) was not
   attempted and might reveal finer structure.
6. **The "n" (sample size) for gradient-based statistics shrinks over
   training** (from 14,835 nonzero anchors at epoch 1 down to 4,512–6,569
   by epochs 20–30) because fewer of the fixed, epoch-1-selected
   high-uncertainty anchors remain "active" (within hinge range) as
   training progresses and the model becomes more confident overall —
   expected and consistent with every prior phase's declining
   `active_hinge_pct` trend, but means late-epoch gradient statistics
   rest on a smaller, more volatile sample than early-epoch ones.

## Conclusion

The target moves — substantially and specifically during an identifiable
early-training window (epochs 1–5), independently confirmed by three
separate measurement methods (centroid direction, actual margin
gradient direction, and whole-representation Procrustes rotation). From
roughly epoch 10 onward, the representation and gradient direction both
stabilize sharply and consistently — this is not a case of perpetual,
unresolved chaos throughout all of training. However, the direction never
returns to its original (epoch-1) orientation: long-range gradient
alignment with epoch 1 stays flat at ~0.13–0.15 for the entire remainder
of training, even as local (consecutive-checkpoint) stability climbs to
0.70+. This is consistent with — and sharpens — the moving-target
hypothesis: a locally-correct, increasingly locally-stable gradient
nonetheless fails to accumulate into large net progress along a *fixed*
reference direction, because the useful direction itself relocated
significantly, permanently, and largely during the first several epochs
of training. Per the phase's scope, this result does not identify
*which* of the four candidate mechanisms (target drift specifically,
representation rotation generally, BatchNorm renormalization, or
segmentation continually redefining the feature basis) is the specific
driver, nor does it evaluate whether an optimizer change would help —
consistent with the reviewer's own caution that if the target moves,
changing the optimizer alone is unlikely to resolve a problem rooted in
the objective's own instability, but confirming that is a separate,
not-yet-run investigation.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e17_target_stability_analysis.py` | Analysis script (Parts 1-4) |
| `experiments/exp_e12_eggo_m/e17_target_stability_results/` | Per-part JSON results, plots |
| `experiments/exp_e12_eggo_m/e12f_pilot_calibrated_seed0/checkpoints/` | Checkpoints analyzed |
| `PHASE_E16_MARGIN_REACHABILITY_ANALYSIS.md` | The reachability puzzle this phase investigates a specific hypothesis for |
| `PHASE_E15_DECODER_SENSITIVITY.md` | Source of the decoder-sensitivity finding motivating the whole E15-E17 arc |

---

**Completed**: 2026-08-07 — Target instability confirmed, but time-limited:
severe rotation in epochs 1–5 (three independent methods agree),
stabilizing sharply from epoch 10 onward, yet the gradient/representation
settle into a *new*, permanently-offset orientation rather than returning
to their epoch-1 state — consistent with, and a more specific version of,
the moving-target explanation for E16's "correctly-directed but
non-accumulating" puzzle.
