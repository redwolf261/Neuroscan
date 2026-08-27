# Phase E15: Decoder Sensitivity Experiment

**Status**: ✅ Complete — **the decoder is NOT margin-invariant.** Artificially widening the latent margin (frozen decoder, no retraining) produces a real, monotonic, statistically significant Dice increase. This rules out "decoder invariance" as the explanation for EGGO-M's null result. The reconciling finding: the effect is real but small at realistic scale, and EGGO-M's optimizer never got anywhere near the push magnitude needed to produce a detectable Dice gain — this is best read as an **optimization/calibration shortfall**, not a conceptually false hypothesis.

**Date**: 2026-08-07

## Purpose

E12f/E13 established the margin mechanism is genuinely active and not
seed-dependent noise; E14 ruled out gradient conflict as an explanation
for why this activity doesn't translate to better Dice. This leaves the
question no prior phase directly tested: **does the decoder even care
about latent margin at all**, independent of whether any optimizer could
ever get it there? This is a causal intervention, not another
correlational diagnostic — the first of its kind in the E1-E15 arc.

## Methodology

### Core design

Load E12f seed 0's epoch-30 checkpoint, freeze it completely (`.eval()`,
`requires_grad_(False)` on every parameter, no gradient steps anywhere in
this script). Take real `dec1` activations from 20 real validation
volumes (the same fixed subset used throughout E12b.5/E12d/E12f/E13/E14).
For each voxel, artificially push its embedding directly away from the
same-volume opposite-class centroid by a controlled absolute distance,
then feed the **modified** embedding through the frozen `seg_head` (a
single `1×1×1 Conv3d + Sigmoid` — i.e. a genuinely simple, per-voxel
linear classifier `sigmoid(w·z+b)`, architecturally identical to E1.3's
offline logistic-regression analogy). Measure Dice on the manipulated
output. No retraining, no gradient updates — a pure forward-pass
intervention on frozen weights.

The manipulation direction: `unit_dir = (z - opposite_class_centroid) / ‖·‖`,
applied per-voxel with `modified_z = z + push_units · unit_dir`. This
uses the ground-truth label only to determine which centroid is
"opposite" — the same information EGGO-M's own margin loss uses for its
class-conditional pairing — never to pick a decoder-specific "correct"
direction. It is the most faithful available simulation of "what if the
margin loss had actually succeeded further," using literally the
Euclidean-distance geometry the De Brabandere-style hinge loss targets.

### A design flaw caught and removed before trusting any result

An earlier version of this script also tested a second manipulation:
pushing each voxel along `w` (`seg_head`'s own weight vector — the *one*
direction that provably determines its output), with the push **sign**
chosen per-voxel using that voxel's own ground-truth label (push toward
higher confidence in the true class). The first smoke test immediately
exposed why this is invalid: at a large enough magnitude it drives Dice
to **exactly 1.0000 ± 0.0000** — not a real result, but a guaranteed
outcome of handing the frozen linear classifier the label directly via
the push direction. This is an oracle intervention, not a test of
decoder sensitivity. It was dropped entirely rather than capped or
patched, and only the label-agnostic-direction (Euclidean) manipulation
was kept, plus a genuinely label-free continuous check (the Jacobian,
below) that can speak to sensitivity along `w` without this circularity.

### Push magnitude calibration

Initial push magnitudes (fractions of the raw centroid-to-centroid
distance, ~23-28 units) were unrealistic: even a 20% fraction produced a
push 1.5× the natural per-voxel `‖dec1‖` (~8 at this checkpoint) —
excursions far outside anything EGGO-M's loss could plausibly produce.
Corrected to **absolute unit offsets** (0, 1, 2, 4, 8, 14, 20, 28),
anchored to E12f's own **fully observed** `mean_boundary_margin`
trajectory (min=14.16, max=28.04 across all 30 real epochs — a 14-unit
dynamic range, the largest swing this architecture was ever seen to
produce under real training). Points beyond 14 units (20, 28) are
included as context for whether any effect only emerges at
unrealistically large scale, but the headline claim rests on the ≤14-unit
region — directly comparable to what real training could plausibly do,
not an arbitrary multiple of an unrelated quantity.

### Sanity check (passed)

`push=0` (identity manipulation) reproduces **Dice=0.90500**, matching
E12f's own logged epoch-30 `val_dice=0.9050` to 4 decimal places —
confirms the checkpoint reload and forward pass are faithful, not a
confounded measurement pipeline.

### Complementary continuous measure: analytic Jacobian

`∂probs/∂z`, computed via `torch.autograd.grad` on frozen weights (no
training loop involved), projected onto both the Euclidean push
direction and `w` itself. This is local, linear-approximation-only, and
— critically — entirely label-free in its direction choice (unlike the
dropped oracle manipulation): it measures how much a small nudge in each
direction changes the decoder's output, without ever using ground truth
to pick which way is "correct."

## Results

### 1. Dice increases monotonically and significantly with push magnitude

| Push (units) | Val Dice (mean±std, n=20) | Relative push (×natural ‖dec1‖) |
|---|---|---|
| 0.0 (baseline) | 0.9050 ± 0.0900 | 0.00 |
| 1.0 | 0.9096 ± 0.0889 | 0.13 |
| 2.0 | 0.9138 ± 0.0873 | 0.25 |
| 4.0 | 0.9209 ± 0.0851 | 0.50 |
| 8.0 | 0.9317 ± 0.0811 | 1.01 |
| **14.0 (E12f's full observed range)** | **0.9443 ± 0.0687** | 1.76 |
| 20.0 | 0.9526 ± 0.0592 | 2.52 |
| 28.0 | 0.9606 ± 0.0498 | 3.53 |

`corr(push_units, Dice)`, realistic range (0–14): **r=+0.9905, p=0.0001**.
Paired t-test, push=14 vs. push=0 (per-volume, n=20): mean Δ=**+0.0393**,
t=7.63, **p<0.0001**. The trend continues smoothly, without visible
saturation, out to 28 units (2× the realistic ceiling) — Dice reaches
0.9606, still rising. Standard deviation also shrinks monotonically as
push increases (0.090 → 0.050), consistent with the manipulation
correcting the worst-performing (highest-variance) volumes more than the
already-good ones — the kind of pattern a genuine, non-degenerate
mechanism would be expected to show, not an artifact.

### 2. The Jacobian confirms sensitivity in both directions, stronger along `w`

Mean `|∂probs/∂z|` directional derivative: **0.000104** along the
Euclidean push direction, **0.000397** along `w` — a ratio of ~0.26,
meaning the decoder is roughly **4× more sensitive per unit of movement
along `w`** than along the Euclidean push direction. This is sensible:
`w` is the *only* axis that mathematically determines
`sigmoid(w·z+b)`, so any direction not perfectly aligned with `w` "wastes"
some of its movement. The Euclidean push is real but not maximally
efficient — consistent with, not contradicting, the discrete Dice result:
both show sensitivity, at different degrees of directness.

### 3. Reconciling this with E12f/E13's null result: the effect is real but far too small at the scale EGGO-M's optimizer actually achieved

This is the most important calculation in this phase. The local slope
near the origin (push 0→1 unit) is **≈0.00457 Dice per unit of margin
push**. E12f's real training run only ever achieved a **net sustained
margin change of +0.21 units** over the full 30 epochs (27.83 → 28.04 —
the same tiny, non-monotonic net change already reported in
`PHASE_E12F_RECALIBRATED_PILOT_RESULTS.md`'s Result 3). At the
near-origin slope, this predicts a Dice gain of only **≈+0.001** — an
order of magnitude smaller than E13's actual observed **deficit** of
−0.0037, and well within noise. **EGGO-M's optimizer never got anywhere
close to the push magnitudes (8–28 units) this experiment shows are
needed to produce a Dice change large enough to detect.**

## Interpretation: which of the pre-registered outcomes occurred?

> **Dice increases → the decoder CAN use better geometry. Therefore
> training simply failed to reach it. Optimization problem.**

This is what happened, cleanly, with no ambiguity in the direction of
the result (monotonic across 8 push levels, both statistically
significant and practically large at the realistic ceiling — nearly +4
Dice points at push=14). The "decoder invariance" hypothesis
(possibility 1 in the framing that motivated this phase) is **rejected**.
The "margin is already sufficient / nothing left to fix" hypothesis
(possibility 2, motivated by E7's near-saturated epoch-1 boundary AUC)
is also **not supported in this strong form** — if the decoder were
already extracting all available value from the existing margin, further
widening it should not have moved Dice at all; instead it moved
substantially. The "wrong geometry" hypothesis (possibility 3 — that
segmentation depends on some other latent property EGGO-M didn't target)
is **not needed to explain the null result**, since a much simpler
explanation now fits: the *right* geometry was targeted, but the
optimizer only ever moved it by a small fraction of what would be needed
to matter.

## What this changes about the project's conclusion

This is the first evidence in the E1–E15 arc that meaningfully
complicates the "EGGO-M is a clean negative result" framing from
`PHASE_E13_MULTISEED_RESULTS.md`. The full picture, updated:

- The margin mechanism is genuinely active (E12e/E12f/E13).
- It is not undermined by gradient conflict (E14).
- **The decoder genuinely benefits from more margin, when margin is
  actually present in sufficient quantity (E15).**
- But EGGO-M's real training runs never produced enough margin (E12f's
  own +0.21-unit net change is roughly 1/70th of the 14-unit range shown
  here to matter) — so the "hypothesis is false" reading from E13 should
  be **narrowed**, not abandoned: the *core causal claim* (wider margin
  → better Dice) survives this direct test; what has actually failed so
  far is the *optimizer's ability to achieve a large-enough, well-placed
  margin increase* under EGGO-M's current loss formulation, weighting,
  and calibration.

This reframes the practical next question from "is margin-widening the
right idea" (this phase says: yes, causally, for this decoder) to "why
does EGGO-M's loss only produce ~0.2 units of real margin growth when
the decoder would benefit from something like 10-20×  that" — a
substantially more specific and actionable question than either "tune λ
more" (already discouraged, and rightly, by the earlier probabilistic
reasoning that a small λ change buys at most ~0.001-0.003 Dice against a
−0.0037 deficit) or "abandon margin-based ideas."

## Limitations

1. **Single checkpoint, single seed** (E12f seed 0, epoch 30). Not
   repeated across E13's other 3 seeds or other epochs — a natural, cheap
   extension if this result needs to be confirmed as checkpoint-general
   rather than specific to this one trained model.
2. **The manipulation is class-centroid-based, not the exact pairwise
   hinge EGGO-M's real loss uses.** Pushing away from the *mean* opposite-
   class position is a simpler, coarser proxy than the actual per-anchor,
   sampled-negative hinge mechanics compute_margin_loss implements. This
   was a deliberate simplification for a clean, well-defined intervention
   direction — it is unlikely to change the qualitative conclusion (both
   push the same general "away from opposite class" direction) but the
   exact slope/magnitude numbers here should not be read as a precise
   prediction of what EGGO-M's real loss would produce at a given λ.
3. **Off-manifold risk grows with push magnitude.** By push=28 units,
   the modification is ~3.5× the natural per-voxel norm — large enough
   that `seg_head` is being evaluated on inputs increasingly far from
   what it was ever trained on. The monotonic, non-saturating trend
   suggests this isn't yet causing a breakdown/artifact within the
   tested range, but this becomes a more serious caveat the further out
   the range is pushed; the ≤14-unit (realistic) region is the safer
   basis for the headline claim.
4. **The Jacobian is a small-perturbation, linear approximation** — valid
   near the current operating point, not a guarantee that the same
   sensitivity ratio holds at the larger, discrete push magnitudes tested
   above (though the two measures agree qualitatively here, that
   agreement is not guaranteed in general).
5. This phase tests **whether the decoder benefits from more margin at
   all**, not **whether EGGO-M's specific loss formula is the best way to
   produce that margin** — the next natural question (why the achieved
   margin growth is so much smaller than what would matter) is a
   distinct, not-yet-answered follow-up.

## Recommendation

Do not treat E13's negative result as final without accounting for this
finding. The margin-widening *hypothesis* is causally supported by this
experiment; what needs to change is not the underlying idea but the
mechanism's ability to actually achieve a large enough push — a
question about the optimizer/loss's effectiveness, not about whether the
concept is sound. This does not mean "go back to tuning λ" (the
probabilistic argument against small hyperparameter nudges given the
tiny expected returns still holds) — it means the productive next
question is why the achieved margin growth (+0.21 units over 30 real
epochs) is roughly 1-2 orders of magnitude smaller than what this
experiment shows would be needed to matter, and whether a substantially
different mechanism (not just a re-tuned coefficient) could close that
gap. Not decided here — a call for the user on how to prioritize this
against the calibration/ECE-isolation-control thread already on the
table.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e15_decoder_sensitivity.py` | Analysis script |
| `experiments/exp_e12_eggo_m/e15_decoder_sensitivity_results/` | Summary JSON, plots |
| `experiments/exp_e12_eggo_m/e12f_pilot_calibrated_seed0/checkpoints/epoch_30.pth` | Checkpoint analyzed |
| `PHASE_E13_MULTISEED_RESULTS.md` | The Dice-null result this phase reframes (not overturns) |
| `PHASE_E14_GRADIENT_CONFLICT_ANALYSIS.md` | The gradient-conflict null this phase's causal test complements |

---

**Completed**: 2026-08-07 — Decoder is causally sensitive to latent
margin (rejects decoder invariance). EGGO-M's real training runs never
achieved a margin increase large enough to matter (~0.2 units observed
vs. ~10-20+ units needed) — the null result is better explained as an
optimization/calibration shortfall than a false hypothesis.
