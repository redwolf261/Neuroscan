# Phase E6: Algorithm Stress Test — "Kill the Idea"

**Status**: ✅ Complete — the algorithm survives, but weaker than E1.3/E1.4 alone suggested; two real methodological corrections made along the way

**Date**: 2026-08-04

## Purpose

Before writing any implementation code for EGGO
(`PHASE_E5_ALGORITHM_DESIGN.md`), actively tried to falsify it. The
concern: E1.1–E1.4 established *correlations* between latent geometry
and uncertainty, but the algorithm design assumes a *causal* story
("uncertainty causes bad geometry," or at minimum "fixing the geometry
will fix the uncertainty/calibration"). This phase checks whether that
jump is defensible, using only existing data — no training, no PyTorch.

## Problem 1: Does density survive a full joint (not pairwise) regression?

E1.4's partial correlation only controlled for latent boundary distance
one variable at a time. This fits a single joint model:
`standardized(evidence) ~ image_boundary + latent_boundary + density + feature_norm`,
all four predictors simultaneously, reporting standardized coefficients
(beta weights) so magnitudes are comparable.

| Predictor | Standardized β (evidence) | Rank |
|---|---|---|
| latent_boundary_dist | +0.318 | 1 |
| feature_norm | −0.188 | 2 |
| **density_rho** | **−0.081** | **3** |
| image_boundary_dist | −0.029 | 4 |

Joint R² = 0.156.

**This is an important, honest downgrade from E1.4's headline framing.**
Density's independent contribution, once every other measured predictor
is controlled for simultaneously (not pairwise), is real but **modest** —
about a quarter the size of latent boundary distance's, and smaller than
feature norm's. E1.4's own pairwise partial correlation (−0.222) looked
larger because it only controlled for one other variable at a time;
placed in competition with all four predictors at once, density's
distinct explanatory power for *evidence's continuous value* is more
limited than the earlier framing implied.

For predicting *error* (binary correct/incorrect) via logistic
regression, the ranking differs: image_boundary_dist dominates (β=−16.9,
reflecting the near-tautological "errors are boundary-adjacent" pattern
already flagged in E1.3), followed by density (β=+0.80), latent boundary
(β=+0.74), and feature norm (β=−0.70) — these three are much closer
together than the evidence-regression ranking suggests, so density's
importance depends materially on which target (evidence magnitude vs.
binary correctness) is being predicted.

**Verdict on Problem 1**: density does NOT vanish to zero — it survives
as a real, non-redundant predictor even in the full joint model — but its
practical importance for *evidence specifically* is more modest than
E1.4 alone suggested. Worth carrying into the algorithm design as reduced
confidence in $\lambda_{comp}$'s a priori importance relative to
$\lambda_{sep}$, to be resolved empirically by the ablation sweep
(Section 7 of `PHASE_E5_ALGORITHM_DESIGN.md`) rather than assumed.

## Problem 2: Is the density effect stable across choices of k?

Recomputed ρ at k∈{5,10,20,40,80} from a single shared k-NN index (no
re-fitting overhead), measuring Cohen's d for the correct/incorrect
separation and Pearson r with evidence at each k.

| k | mean ρ (correct) | mean ρ (incorrect) | Cohen's d | Pearson r (evidence) |
|---|---|---|---|---|
| 5 | 0.187 | 2.162 | −3.021 | −0.221 |
| 10 | 0.206 | 2.375 | −3.027 | −0.218 |
| 20 | 0.229 | 2.660 | −3.005 | −0.214 |
| 40 | 0.258 | 3.051 | −2.965 | −0.210 |
| 80 | 0.294 | 3.649 | −2.985 | −0.212 |

**Cohen's d varies by only ~2% across a 16× range of k.** This is
genuinely reassuring and directly answers Problem 2: the density effect
is **not** an artifact of an arbitrarily-chosen k=20 — it is a stable
property of the representation across a wide range of neighborhood
sizes. This is one of the strongest pieces of evidence *for* the
algorithm surviving this stress test.

## Problem 3: Four-group breakdown — the median split was methodologically wrong, corrected below

**First attempt (median split) was uninformative by construction**: since
99.86% of all voxels are correct, splitting at the population median of
ρ just reproduces the overall base rate in both halves and puts **zero**
incorrect voxels in the "dense" bucket purely because the median is far
below where incorrect voxels actually live (their mean ρ=2.66) — this
was a methodological error, not a finding, and is explicitly noted as
such rather than left to imply "density perfectly separates errors."

**Corrected version**: used $\tau_\rho$ = geometric mean of the measured
correct/incorrect group means (0.229, 2.660) = 0.781 — the same
construction already specified in `PHASE_E5_ALGORITHM_DESIGN.md` §2.3 —
as the sparse/dense threshold, since this is the actual diagnostic
crossover point, not an arbitrary population split.

| Group | n | % of all voxels | mean evidence | mean ρ |
|---|---|---|---|---|
| correct, dense | 57,980 | 96.63% | 19.881 | 0.174 |
| correct, sparse | 1,935 | 3.23% | 17.428 | 1.877 |
| incorrect, dense | 4 | 0.007% | 18.290 | 0.416 |
| incorrect, sparse | 81 | 0.135% | 5.724 | 2.771 |

**Two honest, important numbers from this table**:

- **95.3% of incorrect voxels (81/85) are sparse** — density has high
  *sensitivity* for finding errors, consistent with E1.4's headline claim.
- **But 96.0% of sparse voxels (1,935/2,016) are still correct** — density
  has a **high false-positive rate**. Sparsity alone is far from a clean
  discriminator; most sparse regions are unremarkable, correctly-classified
  voxels (plausibly ordinary background/tissue variation, not
  representation "defects").
- A small number of genuine counterexamples exist (4 incorrect-but-dense
  voxels) — real, but rare (4.7% of all errors).

**Verdict on Problem 3**: this directly supports the caution behind
**Hypothesis C** (sparse regions may often be unremarkable or benign
variation, not damaging uncertainty) — sparsity is a real, sensitive
signal but a weak standalone discriminator. This reinforces why the
design already gates density's force by uncertainty ($w_{comp} = g(\hat
U_i) \cdot D_i$, not $D_i$ alone) — pure density-based intervention
without the uncertainty gate would incorrectly target the 96% of sparse-but-correct
voxels the vast majority of the time. **This finding validates a specific
design choice already in the spec, rather than undermining it** — but
only if the gate itself works correctly (see Problem 5's concern below).

## Counterexample inspection (qualitative)

Sampled the first 5 rows of each exception category
(`experiments/exp_e_latent_analysis/e6_results/exception_samples.json`):

- **"Near latent boundary but high evidence"** cases are uniformly
  **background** voxels (ground_truth_tumor=False), correctly classified,
  sitting near the classifier's decision hyperplane geometrically but
  still confidently and correctly predicted. This makes structural sense:
  the *fitted linear boundary* and the *genuinely hard cases* are related
  but not identical — a voxel can sit near the hyperplane in one
  direction while remaining unambiguous in the direction that matters for
  the actual prediction. Not itself alarming, but confirms latent
  boundary distance alone conflates two different notions of "near the
  boundary."
- The other exception samples were dominated by one or two subjects in
  this small (5-row) qualitative sample, likely a sampling artifact
  (first-N-rows-in-file, not a random draw) rather than a real
  volume-specific effect — **not a settled finding**, flagged here so a
  future pass knows to re-sample randomly if this needs a real answer.

## Which hypothesis (A/B/C/D) does this support?

Recall the four hypotheses from the original stress-test framing:

- **A**: sparse regions are genuinely uncertain → algorithm helps.
- **B**: sparse regions are annotation noise → algorithm hurts.
- **C**: sparse regions are rare anatomy → algorithm hurts.
- **D**: sparsity arises from the evidential loss itself (circular) →
  algorithm doesn't make sense.

**This analysis cannot fully distinguish A from C** — that would require
clinical/anatomical review of the specific sparse-but-correct voxels
(are they rare-but-real anatomical variants the model handles fine, or
positions where the model got lucky?), which is out of scope for a
representation-geometry analysis alone. What it **does** establish:

- Density is not epiphenomenal or an artifact of k-choice (Problem 2:
  stable across 16× range of k) — rules out the concern that this is
  measurement noise being over-interpreted.
- Density is not a simple redundant restatement of boundary distance or
  feature norm (Problem 1: survives, with real, non-zero, if modest,
  independent beta in the joint model) — rules out the concern that
  density is "just a proxy" with zero marginal information.
- Density alone is a poor standalone discriminator (Problem 3: 96% false
  positive rate for "sparse ⇒ error") — this is the strongest evidence
  *against* using density unconditionally, and directly supports **why
  the design already requires the uncertainty gate** rather than acting
  on density alone. If the gate is well-calibrated, this is not a fatal
  problem; if the gate is degenerate (Failure Mode 1, already flagged in
  `PHASE_E5_ALGORITHM_DESIGN.md` §6), this becomes a real risk of the
  compactification force firing on mostly-harmless voxels.
- **Hypothesis D (circularity) is not addressed by this analysis** — all
  data here comes from a single frozen, fully-trained checkpoint (no
  training was run), so there is no way to observe whether density
  patterns are a cause or a downstream consequence of the evidential
  loss's own training dynamics from this data alone. This remains a
  genuinely open question, explicitly *not* resolved by Phase E6, and
  should be the first thing checked once/if a pilot training run exists
  (does density's relationship with evidence hold from early epochs, or
  does it only emerge late in training after the evidential loss has
  already shaped the representation? — a question only answerable with a
  live training run, which is exactly why implementation should stay
  paused pending further review, consistent with the "not yet" stance
  this document supports).

## Overall verdict

**The algorithm survives this stress test, with real, honest downgrades
to its confidence level, not a clean pass.**

What got stronger:
- Density's k-stability (Problem 2) is a genuinely reassuring result —
  the measurement itself is robust.
- Problem 3's high false-positive rate for density alone is not a
  contradiction of the design — it's direct empirical support for why the
  design already gates density by uncertainty rather than acting on it
  unconditionally.

What got weaker:
- Density's independent explanatory power for evidence, in a full joint
  model, is smaller than the E1.4 framing implied (rank 3/4, β=0.081) —
  the case for $\lambda_{comp}$ mattering as much as $\lambda_{sep}$ is
  now less certain and should be treated as an open empirical question
  for the ablation sweep, not assumed.
- Hypothesis D (circularity — does the evidential loss itself produce the
  sparsity pattern being measured?) remains completely untested and
  cannot be resolved without a live training run, which creates real risk
  that the diagnostic chain (all measured on one frozen checkpoint) may
  not hold the same way once EGGO's own compactification pressure starts
  actively reshaping the representation during training.

## Recommendation

Proceed toward implementation planning (Problem 4's boundary-classifier
redesign, Problem 5's gate-stability analysis, and Problems 6–8's
hyperparameter sweeps), but **do not treat $\lambda_{comp}$ and
$\lambda_{sep}$ as equally well-justified a priori** — the ablation
design in `PHASE_E5_ALGORITHM_DESIGN.md` §7 already isolates this
question (rows 1 vs. 2 vs. 5), which is now more clearly the load-bearing
experiment for judging whether compactification pulls its weight, rather
than a confirmatory formality.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e_latent_analysis/e6_stress_test.py` | Problem 1, 3 (initial/flawed), counterexample extraction |
| `experiments/exp_e_latent_analysis/e6_fourgroup_fixed.py` | Problem 3, corrected threshold |
| `experiments/exp_e_latent_analysis/e6_k_sensitivity.py` | Problem 2 |
| `experiments/exp_e_latent_analysis/e6_results/` | All JSON results + exception samples |
| `PHASE_E5_ALGORITHM_DESIGN.md` | The design this stress-tests |

---

**Completed**: 2026-08-04
