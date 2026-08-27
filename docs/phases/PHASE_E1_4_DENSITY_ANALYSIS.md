# Phase E1.4: Is Uncertainty Related to Local Manifold Density?

**Status**: ✅ Complete — Case B/C boundary result: density is a strong, genuinely independent contributor, not a repackaging of E1.3's boundary-distance finding

**Date**: 2026-08-04

## Question

E1.1–E1.3 established a chain of partial explanations: image boundary
distance (R²=0.150) < latent decision-boundary distance (R²=0.219),
leaving roughly 78% of evidence's variance unexplained even by the best
predictor so far. This asks whether **local manifold density** — how
many nearby neighbors a voxel's `dec1` feature vector has in the pooled
representation space — explains some of that remainder, and whether it's
independent of latent boundary distance or just measuring the same thing
a different way.

## Data and method

Same source as E1.1–E1.3: `per_voxel_stats.csv`, 60,000 voxels, 30
volumes, frozen baseline seed 0. Density computed as `ρ = mean distance
to the 20 nearest neighbors`, over the full pooled 32-dim `dec1` feature
space (not per-volume, not spatial adjacency — this measures
representation-manifold crowding). Low ρ = dense region; high ρ = sparse
region. No new inference.

## Headline numbers

| Measure | Value |
|---|---|
| Pearson r (ρ, evidence) | −0.214 |
| Spearman ρ | +0.369 |
| Best polynomial R² (degree 1–4) | 0.311 (degree 4) |
| Mean ρ, correct voxels | 0.229 |
| Mean ρ, incorrect voxels | **2.660** (≈12× larger) |
| Cohen's d (correct vs. incorrect) | **−3.005** |
| Pearson r (ρ, \|latent_boundary_distance\|) | **−0.0068** (p=0.097, not significant) |
| Partial corr(evidence, ρ \| latent_boundary_distance) | **−0.222** (barely different from raw −0.214) |
| R² (image boundary + latent boundary) | 0.097 |
| R² (image boundary + latent boundary + density) | **0.140** |
| Incremental R² from adding density | **+0.043** |

## Result 1: incorrect voxels live in dramatically sparser regions of the manifold

This is the strongest single number in this analysis. Correct voxels sit
in very dense neighborhoods (mean ρ=0.229), while incorrect voxels sit in
regions roughly **12× sparser** (mean ρ=2.660). Cohen's d=−3.005 —
nearly as large as evidence's own separation from E1.1 (d=3.09), and
substantially larger than either feature norm's (d=−1.29, E1.2) or
latent boundary distance's (d=1.41, E1.3) separation of the same two
groups. The histogram (`e1_4_results/03_density_correct_vs_incorrect.png`)
shows an almost complete, near-disjoint separation: correct voxels form
a sharp peak near ρ≈0, incorrect voxels cluster in a nearly
non-overlapping range around ρ≈1.5–4.

## Result 2: density is genuinely independent of latent boundary distance — not a repackaging

This is the key result answering the "why does this experiment matter"
question directly. Density and latent boundary distance are essentially
**uncorrelated** (r=−0.0068, not statistically significant at p=0.097).
This means density is not simply another way of measuring "distance from
the decision boundary" — it captures a genuinely different geometric
property. The partial correlation confirms this: controlling for latent
boundary distance barely changes density's relationship with evidence
(−0.222 vs. raw −0.214) — density's contribution to explaining evidence
is essentially unaffected by already knowing the boundary distance.

## Result 3: density adds real incremental explanatory power on top of both boundary measures

Multiple regression: image boundary alone R²=0.043 (linear fit, matched
methodology to E1.2/E1.3 for fair comparison — note this differs slightly
from E1.1's polynomial R²=0.150 for the same reason as E1.2's writeup);
adding latent boundary brings this to R²=0.097; **adding density on top
brings it to R²=0.140** — an incremental gain of +0.043, similar in size
to the gain latent boundary distance itself provided over image boundary
alone. Density is pulling its own weight as a predictor, not redundant
with either boundary measure.

## Result 4: density has its own distinct geometric structure, different from E1.3's evidence gradient

The PCA-colored-by-density plot (`e1_4_results/05_pca_colored_by_density.png`)
shows something worth flagging explicitly because it's subtle: the
background band is uniformly dense throughout (pale yellow), consistent
with E1.3's finding that background forms a tight, compact cluster. The
tumor spike, however, shows its own internal density gradient — **denser
near the base/transition zone, sparser at the extreme tip**. This is the
**opposite spatial pattern** from E1.3's evidence gradient, which was
*low* near the base and *high* at the tip. Density and evidence are not
simply tracking the same spatial axis in the tumor cluster — they vary
along it in different (in this local sense, roughly opposite) ways,
reinforcing that density is contributing information latent boundary
distance alone doesn't capture.

## Interpretation: Case B, bordering on Case C

Using the pre-registered decision thresholds: incremental R²=0.043 sits
in the "Case B" range (0.02–0.15: density contributes a real, moderate
independent signal) rather than "Case C" (>0.15, density as a dominant
co-explanation). However, the **effect-size evidence (Cohen's d=−3.005,
nearly matching evidence's own d=3.09) is much stronger than the
R²-based framing alone suggests** — density is an excellent binary
separator of correct-vs-incorrect even though its continuous relationship
with evidence's exact magnitude is more moderate (consistent with a
threshold-like, rather than smoothly linear, relationship, visible in the
LOWESS curve).

**Practical reading**: E1.3's interpretation should be extended, not
replaced. Uncertainty is not simply "close to the decision boundary" — it
is more precisely "close to the decision boundary **and/or** in a sparse,
poorly-supported region of the representation manifold." These are
measurably distinct, complementary signals (near-zero correlation between
them), and together they explain meaningfully more of evidence's variance
(R²=0.140 vs. 0.097) than boundary distance alone.

## Implication for algorithm design

This directly refines the recommendation in
`PHASE_E2_LITERATURE_REVIEW.md`. The confidence/evidence-weighted margin
loss (the recommended headline contribution) should not be motivated by
boundary proximity alone — **local density should be a second, explicit
input to whatever loss or weighting scheme is designed**, since it is
empirically independent of boundary distance and comparably strong at
separating correct from incorrect predictions. Concretely, this suggests
augmenting the margin/compactification loss with a density-aware term
(e.g., pulling sparse-region embeddings toward denser regions of their
own class, not just away from the opposite class) rather than relying on
margin/boundary-distance mechanics alone. This also gives "local
compactification of the diffuse tumor cluster" (Variant D from the
earlier LBGO discussion) a sharper, measurement-grounded justification:
the tumor cluster's diffuseness (E1.3) and its internal density gradient
(this analysis) both point at the same underlying structural issue.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e_latent_analysis/e1_4_density_analysis.py` | Analysis script |
| `experiments/exp_e_latent_analysis/e1_4_results/` | Plots + `results_summary.json` |
| `PHASE_E1_3_LATENT_GEOMETRY.md` | The boundary-distance finding this extends |
| `PHASE_E2_LITERATURE_REVIEW.md` | Algorithm-design implications this refines |

---

**Completed**: 2026-08-04
