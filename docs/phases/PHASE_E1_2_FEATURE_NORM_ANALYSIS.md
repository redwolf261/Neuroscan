# Phase E1.2: Is Uncertainty Simply Controlled by Latent Feature Magnitude?

**Status**: ✅ Complete — H0 rejected (with an important caveat: pooled
statistics are confounded by class; the true relationship is
class-conditional and strongly nonlinear)

**Date**: 2026-08-04

## Question

Does `Evidence ≈ f(‖z‖)`, where `z` is the shared trunk's `dec1` feature
vector? If feature magnitude alone explains most of evidence's variance,
the evidential head is just converting "how loud is the feature" into
"how confident am I" — a much weaker mechanism than genuinely encoding
uncertainty-relevant structure, and there would be little motivation for
representation-level algorithms.

## Data

Same source as E1.1: `experiments/exp_e_latent_analysis/extracted/per_voxel_stats.csv`,
60,000 voxels from 30 validation volumes (frozen baseline, seed 0). No new
inference.

## Headline numbers

| Measure | Value |
|---|---|
| Pearson r (feature_norm, evidence), pooled | −0.249 |
| Spearman ρ, pooled | −0.434 |
| Best polynomial R² (degree 1–4), pooled | **0.572** (degree 4) |
| R² (boundary only, from E1.1) | 0.150 |
| R² (feature_norm only, linear) | 0.062 |
| R² (boundary + feature_norm, linear, multiple regression) | 0.086 |
| Incremental R² from adding feature_norm to boundary | +0.043 |
| Partial corr(evidence, feature_norm \| boundary) | −0.212 |
| Permutation importance (RandomForest, boundary_distance) | 0.725 ± 0.021 |
| Permutation importance (RandomForest, feature_norm) | **1.144 ± 0.012** |

**On its own, the pooled polynomial R² (0.572) looks like a much bigger
effect than boundary distance's 0.150 — but this number is misleading in
isolation** and needs the class-split and shape analyses below to
interpret correctly.

## What's actually going on: a real, strong, but class-conditional and non-monotonic relationship

**LOWESS curve** (`e1_2_results/02_lowess_evidence_vs_featnorm.png`) is
**U-shaped, not monotonic**: evidence starts high (~19–22) at low feature
norm (5–10), drops sharply to a minimum (~6) around feature_norm≈15, then
rises again for large feature norms (>20) up to ~30. Neither H0's
"magnitude → confidence" story nor a simple linear alternative fits this
shape.

**Tumor vs. background split** (`e1_2_results/05_tumor_vs_background.png`)
explains the U-shape: the two ground-truth classes occupy almost entirely
**non-overlapping feature-norm ranges**, each with its own strong,
opposite-signed relationship:

| Class | n | Pearson r | Feature-norm range |
|---|---|---|---|
| Tumor (GT positive) | 614 | **+0.907** | ~5 to ~65 |
| Background (GT negative) | 59,386 | **−0.680** | ~4 to ~35 (mostly 5–10) |

Within the tumor class specifically, there is a striking, almost linear,
strong positive relationship: feature norm 5→65 tracks evidence 0→33
closely. Within background, evidence and feature norm are weakly
negatively related within a much narrower, low-magnitude band.

**This is a Simpson's-paradox-style confound**: pooling both classes
together produces a nonlinear (U-shaped) curve and an inflated polynomial
R² that don't represent either class's actual, simpler, strongly linear
within-class relationship. The pooled degree-4 polynomial R²=0.572
number should not be read as "feature norm explains 57% of evidence
variance" in any usable sense — it's fitting curvature that is actually
two overlaid linear relationships with different slopes and offsets, not
a genuine single nonlinear law.

## Boundary-controlled and multiple-regression results (the more decisive numbers)

Once boundary distance is already accounted for, feature norm adds only
a **small amount of independent explanatory power**:

- **Multiple linear regression**: boundary alone R²=0.043 (linear;
  differs from E1.1's polynomial 0.150 since this uses a plain linear fit
  for a fair, matched comparison against feature_norm's linear R²=0.062).
  Boundary + feature_norm together: R²=0.086. The incremental gain from
  adding feature_norm is only **+0.043** — feature norm roughly *doubles*
  a small R², but the combined model still explains under 9% of variance
  linearly.
- **Boundary-controlled subgroup** (voxels within ±2 of the true
  boundary, n=1,110, where boundary distance is nearly constant):
  Pearson r=+0.188, Spearman ρ=−0.154 — weak and, notably, the Pearson
  and Spearman signs even disagree with each other here, indicating a
  noisy, non-robust relationship in this specific regime, not a clean
  independent signal.
- **Partial correlation** (evidence, feature_norm | boundary_distance):
  r=−0.212, barely different from the raw uncontrolled r=−0.249 — feature
  norm's relationship with evidence is **largely independent of
  boundary distance** (consistent with feature norm and boundary distance
  capturing different things), but the relationship itself remains modest
  in linear terms.
- **Permutation importance** (RandomForest, nonlinear model): feature_norm
  (1.144) outweighs boundary_distance (0.725) as a predictor of evidence
  — consistent with the class-conditional strong linear relationships
  found above, since a tree-based model can exploit the class-conditional
  structure that a single linear/polynomial fit on pooled data cannot.

## Correct vs. incorrect: feature norm is a partial, not clean, separator

Feature norm does differ between correct and incorrect voxels (mean 7.46
vs. 13.02, Welch t=−9.31, p=1.4×10⁻¹⁴), but the **effect size is
substantially smaller than evidence's own separation**: Cohen's d for
feature_norm = −1.29, vs. Cohen's d for evidence = **3.09** (from E1.1's
same correct/incorrect split). Evidence separates errors from correct
predictions roughly **2.4× more strongly** than feature norm alone does
— confirming evidence is not simply a repackaging of feature magnitude;
it carries additional, stronger separating information. The histogram
(`03_featnorm_correct_vs_incorrect.png`) shows substantial overlap
between the two distributions, not a clean threshold.

## Interpretation

**H0 is rejected**, but the honest, nuanced version of "why" matters more
than the reject/fail-to-reject label:

1. Feature norm has a **real, strong, class-conditional relationship**
   with evidence within each ground-truth class separately (tumor
   r=+0.91, background r=−0.68) — this is not nothing, and it means
   feature magnitude is doing *some* real work.
2. But that relationship is **not usable as a simple global rule**
   (`evidence = f(‖z‖)`) — it flips sign and changes scale depending on
   class, and the pooled/pretend-single-relationship view (the naive
   headline R²=0.572) is a statistical artifact of mixing two
   differently-behaved subpopulations, not a real unified law.
3. Once boundary distance is controlled for, feature norm's **marginal,
   boundary-independent contribution is small** (+0.043 incremental
   linear R², weak and sign-inconsistent Pearson/Spearman results in the
   boundary-controlled subgroup).
4. Feature norm separates correct-vs-incorrect voxels **less than half
   as strongly** as evidence itself does (Cohen's d 1.29 vs 3.09) — so
   evidence is not simply "reading off" feature magnitude to produce its
   calibration signal (the finding from E1.1); it's doing something more.

**Net read against the decision tree in the design brief**: this doesn't
land cleanly in either the "feature norm explains everything" (R²≈0.7,
no algorithm needed) or the "feature norm barely explains anything"
(R²≈0.1, clean handoff to representation geometry) box. It's a genuine
in-between result: feature magnitude carries real, class-dependent
information, but not enough on its own — and not in a form usable without
already knowing the class — to explain evidence, and it explains
meaningfully less of the correct/incorrect calibration signal than
evidence itself carries. **This still points toward representation
geometry** (direction/structure, not just scale, and specifically
*class-conditional* structure) **as the next investigation target**,
consistent with the design brief's decision tree, but with a specific
new detail worth carrying forward: whatever the representation encodes,
it appears to encode it differently, and possibly through a different
mechanism, for tumor voxels versus background voxels. Any representation
diagnostic or algorithm design (e.g., Concept 0/1 in
`PHASE_E3_CANDIDATE_CONCEPTS.md`) should account for this class asymmetry
rather than assuming a single global relationship.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e_latent_analysis/e1_2_feature_norm_analysis.py` | Analysis script (8 analyses) |
| `experiments/exp_e_latent_analysis/e1_2_results/` | Plots + `results_summary.json` |
| `PHASE_E1_1_BOUNDARY_VS_EVIDENCE.md` | E1.1 (boundary distance), the prior filter this builds on |

---

**Completed**: 2026-08-04
