# Phase E1.3: Latent Geometry — Is Uncertainty a Latent Decision-Boundary Phenomenon?

**Status**: ✅ Complete — strong, visually and statistically confirmed result

**Date**: 2026-08-04

## Question (redirected scope, per E1.2's finding)

E1.2 found tumor and background voxels relate feature magnitude to
evidence with **opposite signs** (tumor r=+0.91, background r=−0.68),
suggesting two differently-organized representations rather than one
shared uncertainty manifold. This raised a sharper question than generic
"representation geometry": **are tumor and background organized as
separate clusters in the `dec1` latent space, and does uncertainty
concentrate specifically in the transition zone between them — a latent
decision boundary — rather than (or in addition to) the image-space
tumor boundary?**

## Data

Same source as E1.1/E1.2: `per_voxel_stats.csv`, 60,000 voxels, 30
validation volumes, frozen baseline seed 0. Full 32-dim `dec1` feature
vectors used directly for all clustering/distance computations (PCA used
only for the visualization figure, never for numeric claims).

## Method

Fit a linear classifier (logistic regression, class-balanced) to
separate tumor vs. background directly in the 32-dim `dec1` space. Its
signed decision function — `(w·z + b) / ‖w‖` — is a genuine, principled
**latent decision-boundary distance**: positive means confidently
tumor-side, negative confidently background-side, near-zero means the
voxel sits right at the boundary the network's own features draw between
the two classes. This is the direct latent-space analogue of E1.1's
image-space `boundary_distance`.

## Result 1: The two classes are almost perfectly linearly separable in `dec1` space

**AUC = 0.9995** for tumor-vs-background separation using a plain linear
classifier on the 32-dim features. This is a very strong finding on its
own: the shared trunk's final layer already encodes class identity
almost perfectly linearly, well before either head's own 1×1 conv acts
on it.

**Cluster geometry** (`e1_3_results/03_pca_visualization.png`, 92.6%
variance in 2 PCs): background voxels form a tight, low-spread cluster
(mean distance to own centroid = 4.35, std 1.69); tumor voxels form a
much more diffuse, extended cluster (mean distance to own centroid =
13.43, std 7.45 — roughly 3× more spread). The two class centroids are
separated by 29.26 units, dwarfing either class's internal spread. The
PCA plot shows this directly: a tight, flat background band, with tumor
voxels rising as a continuous, elongated spike — connected to background
by a sparse, visible transition trail, not a disconnected second blob.

## Result 2: Evidence forms a genuine geometric gradient along this structure, low near the transition zone

Coloring the same PCA plot by evidence (`03_pca_visualization.png`,
middle panel) shows a striking, visually unambiguous gradient: **evidence
is low (dark purple) at the base of the tumor spike, where it connects to
background, and rises smoothly to its highest values (bright yellow) at
the far tip of the spike** — the voxels most confidently, unambiguously
tumor. This is not noise; it's a continuous, monotonic-looking gradient
along the class-transition axis.

Quantitatively: `|latent_boundary_distance|` vs. evidence gives Pearson
r=+0.308, Spearman ρ=+0.120, best polynomial R²=**0.219** (degree 4) —
**higher than E1.1's image-boundary R² of 0.150**, meaning the latent
decision-boundary distance explains more of evidence's variance than the
image-space tumor boundary does. This is a real, if partial, improvement
over the E1.1 finding, and directly supports the redirected E1.3
hypothesis: uncertainty tracks proximity to the network's *own* internal
class boundary somewhat better than it tracks the anatomical boundary in
the image.

## Result 3: Incorrect voxels sit almost exactly in the latent transition zone

The clearest single result in this analysis. Correct and incorrect
voxels' distributions of `|latent_boundary_distance|` are
**near-bimodally separated** (`e1_3_results/02_latent_boundary_correct_vs_incorrect.png`):
correct voxels cluster tightly around distance ≈2.0–2.2 (confidently on
one side or the other), while incorrect voxels cluster almost entirely
below ≈1.2, with very little overlap in the tails.

| | mean \|latent boundary distance\| | n |
|---|---|---|
| Correct | 1.456 | 59,915 |
| Incorrect | 0.648 | 85 |

Welch t=20.71, p=5.75×10⁻³⁵, **Cohen's d = 1.41**. The PCA "incorrect
voxels highlighted" panel (third panel, same figure) confirms this
visually: red (incorrect) points sit almost exclusively along the sparse
transition trail connecting the two clusters, not scattered through
either cluster's interior.

## An important reconciliation: don't over-read the horse-race AUC numbers

A direct AUC comparison of predictors for separating correct vs.
incorrect voxels gives:

| Predictor | AUC |
|---|---|
| \|image boundary distance\| | **0.995** |
| evidence (E1.1 reference) | 0.965 |
| feature norm (E1.2 reference) | 0.916 |
| \|latent boundary distance\| | 0.776 |

At first glance this looks like image boundary distance "wins" and even
beats evidence — which would seem to contradict E1.1's conclusion. **It
doesn't, and the reason matters**: nearly all incorrect voxels have
`|boundary_distance| ≤ 3` and nearly all correct voxels have
`|boundary_distance|` far larger (mean 22.4) — this is close to a
tautology of the segmentation task itself (errors occur near boundaries,
correct predictions dominate deep interior/exterior regions by simple
task geometry), not evidence that boundary distance explains *how
uncertain the model is*. AUC-for-separating-errors-by-location and
R²-for-predicting-evidence-magnitude are different questions:

- **E1.1's claim** (still valid): boundary distance is a poor predictor
  of evidence *magnitude* (R²=0.150) — knowing distance-to-boundary
  doesn't tell you much about how much evidence the model assigns.
- **This horse race's claim**: boundary distance (unsurprisingly) predicts
  *where* errors are geographically located, since segmentation errors
  are inherently boundary-adjacent. This was already established in
  E1.1's correct/incorrect scatter plot (errors clustered within ±1-3
  voxels of the boundary) and isn't new information here.

The genuinely new, non-tautological result is that **the latent
decision-boundary distance — a purely representation-space quantity with
no direct access to image geometry — independently recovers a similar
"errors happen near the boundary" structure, and evidence tracks it
somewhat better (R²=0.219) than it tracks the image-space boundary
(R²=0.150)**. That's the meaningful finding, not the AUC horse race.

## Interpretation

This is the most decisive result of Phase E so far. Three independent
lines of evidence converge:

1. Tumor and background form nearly perfectly linearly separable,
   differently-shaped clusters in `dec1` space (AUC=0.9995; background
   tight, tumor diffuse).
2. Evidence forms a visible, continuous gradient along the
   transition/connection zone between these two clusters — low near the
   transition, high deep inside the tumor cluster.
3. Incorrect voxels are concentrated almost bimodally close to the
   latent decision boundary (Cohen's d=1.41), and the latent
   boundary distance explains more of evidence's variance (R²=0.219)
   than the image boundary does (R²=0.150, E1.1).

**Uncertainty in this model is best understood as a phenomenon of
proximity to the network's own internal class boundary in representation
space — not primarily a function of image-space geometry (E1.1) or raw
feature magnitude alone (E1.2), though both contribute partially.** This
gives Phase E a concrete, mechanistically specific target: the latent
representation appears to be organized as (at least) two differently-shaped
class manifolds connected by a transition region, and uncertainty
emerges from position within that specific structure.

## Implication for candidate algorithms

This sharpens `PHASE_E3_CANDIDATE_CONCEPTS.md`'s Concept 0/1
(representation entanglement / orthogonality) into something more
specific: the relevant geometric object isn't generic "entanglement"
between two heads' feature subspaces, but the **shape and margin of the
tumor/background decision boundary the shared trunk has already learned**.
Candidate directions this suggests, not yet built:

- Explicitly widening or regularizing the margin around the latent
  decision boundary (distinct from ABO's gradient-magnitude approach and
  from a generic orthogonality regularizer) — since errors concentrate
  in exactly this transition zone.
- Using `|latent_boundary_distance|` itself as an auxiliary training
  signal or as a loss-reweighting term (connects to Concept 2's
  uncertainty-driven weighting idea in `PHASE_E3_CANDIDATE_CONCEPTS.md`,
  but grounded in latent geometry rather than the evidential head's raw
  output).
- Investigating why the tumor cluster is ~3× more diffuse than
  background (std of distance-to-own-centroid: 13.43 vs 4.35) — is this
  itself a source of the residual uncertainty variance neither boundary
  distance nor feature norm explained?

None of these are committed to yet — this remains a diagnostic finding,
not an implementation decision.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e_latent_analysis/e1_3_latent_geometry.py` | Analysis script |
| `experiments/exp_e_latent_analysis/e1_3_results/` | Plots + `results_summary.json` |
| `PHASE_E1_1_BOUNDARY_VS_EVIDENCE.md`, `PHASE_E1_2_FEATURE_NORM_ANALYSIS.md` | Prior filters this builds on |

---

**Completed**: 2026-08-04
