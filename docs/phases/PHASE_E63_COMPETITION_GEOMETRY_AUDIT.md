# Phase E63 — Local Competition Geometry Audit

## Purpose

E62 established that permuting subcell position inside pool1's 2×2×2 cells (holding
the exact 8-value multiset fixed) causally degrades segmentation, size-specifically.
E63 tested a narrower hypothesis: **is E62's effect driven primarily by the geometry
of the top-two competing activations (winner vs. runner-up) inside each cell**,
rather than the full 8-value arrangement? No training, no architecture/loss change —
pure causal-intervention diagnostic, same discipline as E47/E48/E58/E62.

## Method

- **Checkpoint / data**: identical to E62 — v5/E46 attention-gate checkpoint, same
  125-subject validation split, hooked at `enc1` immediately before `pool1`.
- **Local competition descriptor** per 2×2×2 cell, per channel: sorted values
  `x(1)≥...≥x(8)`, winner `w=x(1)`, runner-up `r=x(2)`, gap `g=w-r`, winner/runner-up
  coordinates `q1,q2`, displacement `Δq=q2-q1`, distance `d=‖Δq‖`.
- **Three nested counterfactual tests**, each a pure permutation preserving the
  cell's exact 8-value multiset:
  - **Test A** (E62-local reference): full 8-value derangement.
  - **Test B** (winner fixed, runner-up moved): winner's value/slot untouched; the
    runner-up's value swaps with one other randomly chosen non-winner,
    non-runner-up slot (exactly 2 of 8 slots change).
  - **Test C** (top-two fixed, weak values rearranged): winner and runner-up slots
    untouched; a derangement is applied to the remaining 6 slots only.
- **Intervention scale — corrected after a first run exposed a confound**: the
  original design sampled a small fixed count of cells per subject (200, 50/gap
  quartile, ~0.02% of the ~1.05M pool1 cells) and intervened on only those. Even
  Test A at that scale produced a near-zero Dice effect (mean drop=-0.00002,
  indistinguishable from noise) — vs. E62's own 0.0226 drop when *every* cell was
  permuted simultaneously. This could not separate "geometry doesn't matter" from
  "0.02% of cells is too small a perturbation to move Dice at all," so it was
  flagged to the user rather than reported as a clean result. **Corrected design**
  (used for the reported results below): every test is applied to **all active
  (gap>0) cells simultaneously** — matching E62's whole-volume scale. The
  gap-quartile breakdown restricts Test B to one quartile's cells at a time, but
  still at whole-stratum scale within that quartile (not a small sample of it).
- **Sparsity handling**: real post-ReLU pool1 activations are ~84% exact `gap==0`
  ties (dead/inactive cells). These are excluded from the gap-quartile-cutoff
  population (computed from a 30-subject pooled sample, frozen before any
  per-subject analysis) — otherwise Q1–Q3 collapse into the same degenerate
  zero-gap bin.
- Six unit tests verify: (1) synthetic single-cell permutation correctness for all
  three tests; (1b) the vectorized whole-volume batch versions match the same
  semantics on 500 synthetic rows; (2) `forward_from_enc1` bit-for-bit reproduces
  `model.forward()`; (3) a single-cell edit changes only its own 8 voxels; (3b) the
  vectorized whole-volume edit-apply function matches the per-cell loop version
  exactly on a scattered 715-cell check.

## Results (n=125 subjects, mean ~169,349 active cells/subject)

| Test | Mean Dice drop | t-test p | Wilcoxon p | Sign-flip permutation p |
|---|---|---|---|---|
| A (full derangement) | **0.0223** | 4.3×10⁻²⁸ | 8.5×10⁻²⁰ | <0.001 |
| B (winner fixed, runner-up moved) | 0.0003 | 0.286 | 0.283 | 0.285 |
| C (top-two fixed, weak values rearranged) | 0.0019 | 0.0025 | 0.0026 | 0.002 |

- **Test A reproduces E62 almost exactly** (0.0223 vs. E62's 0.0226) — confirms the
  whole-volume-scale correction is consistent with the original E62 finding.
- **F_WR = 0.0143** — winner–runner-up relocation (Test B) captures only **1.4%** of
  Test A's effect. Not "substantial" by any reasonable reading of the pre-declared
  bar (>0.5).
- **Test B is not statistically distinguishable from zero.**
- **Test C is significant** and larger than Test B (0.0019 vs. 0.0003) — this is the
  *opposite* of the winner–runner-up hypothesis's prediction, which required Test C
  to be the materially weaker one.
- **Gap-quartile analysis**: nominally "monotonic" in the predicted direction
  (Spearman −0.40) but not remotely significant (p=0.60, only 4 data points), and
  all four quartile means are themselves indistinguishable from zero and from each
  other (−0.0002 to −0.0007) — this is noise, not a gradient.
- **Small-lesion analysis**: Spearman(native_size, drop_B) = −0.175, p=0.050 — right
  at the significance boundary, but this is testing a near-zero effect (drop_B
  itself is not significant) for a size relationship; not meaningful given Test B's
  own null result, and does not meet the pre-declared bar either way.

## Decision (step 15, verbatim rule)

| # | Criterion | Result |
|---|---|---|
| 1 | Test B non-zero, reproducible effect | **False** |
| 2 | F_WR > 0.5 (substantial fraction of Test A) | **False** (0.014) |
| 3 | Strongest at small competition gaps | Nominally true, not significant |
| 4 | Stronger in small-lesion cases | False |
| 5 | Survives subject-level permutation testing | **False** |
| 6 | Magnitude-preserved geometry change sufficient | **False** |
| 7 | Test C materially weaker than Test B | **False** — Test C is *larger* |

## Verdict: **KILL**

The winner–runner-up geometry hypothesis is not supported. Isolating only the
top-two competing activations' spatial relationship and moving just the runner-up
reproduces essentially none of E62's effect. The rearrangement that *does* carry
real signal is the opposite of what the hypothesis predicted: perturbing the six
**weaker** activations (Test C, top-two held fixed) produces a real, if modest,
effect (0.0019, p=0.002) — smaller than the full derangement (0.0223) but larger and
more significant than the top-two-only intervention (0.0003, ns).

## Interpretation

E62's real, robust, size-specific effect is **not concentrated in a top-two
winner–runner-up mechanism**. The signal appears diffusely distributed across the
full local 8-value arrangement — including specifically the sub-maximal activations
Test C perturbed — rather than being reducible to a simple two-point competition
geometry. This rules out the cleanest, most tractable version of a
competition-geometry-based encoder fix (a winner–runner-up relocation operator) and
redirects any future PMD-style design toward a representation of the **full local
distribution** (which is closer to the original mass + moments proposal from before
E62/E63, not a top-2 simplification of it) rather than a top-2 abstraction.

Per the pre-declared rule for E63 (step 15/16): do not respond to this KILL by
testing alternative top-k, rank, or geometric features within this phase. The next
step, if pursued, is a fresh phase deriving a full-distribution representation
(e.g. the original moment-based PMD idea) and subjecting *that* specific
representation to its own causal audit before any operator design.

## Artifacts

- `experiments/exp_e12_eggo_m/e63/run_e63_competition_geometry_audit.py`
- `experiments/exp_e12_eggo_m/e63/E63_competition_geometry_table.json` (125 subject records)
- `experiments/exp_e12_eggo_m/e63/E63_summary.json`
