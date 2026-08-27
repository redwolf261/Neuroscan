# Phase E1.1: Is Uncertainty Just Boundary Ambiguity?

**Status**: ✅ Complete — H0 rejected, H1 supported

**Date**: 2026-08-04

## Question

Is the evidential branch learning anything beyond "pixels near the tumor
boundary are uncertain"? If boundary distance alone explains most of the
variance in evidence, there's little justification for a
representation-learning algorithm — the evidential branch would just be
a boundary detector.

## Data

`experiments/exp_e_latent_analysis/extracted/per_voxel_stats.csv` — 60,000
voxels sampled from 30 validation volumes (frozen baseline, seed 0,
val_dice=0.9107). No new inference; pure analysis of already-extracted
data. 614 tumor voxels (1.02%), 59,386 background — matches the expected
BraTS class imbalance.

## Result: H0 rejected

| Measure | Value |
|---|---|
| Pearson r (boundary_distance, evidence) | −0.206 |
| Spearman ρ | −0.104 |
| Best polynomial R² (degrees 1–4) | **0.150** (degree 4) |

By the pre-registered thresholds (R²≈0.9 → boundary explains almost
everything; R²≈0.1 → boundary barely explains it), **0.15 lands at the
"barely explains it" end**. Boundary distance is a weak-to-moderate
contributor at most, not the dominant driver of evidence.

## What the finer-grained splits show

**LOWESS curve** (`e1_1_results/02_lowess_evidence_vs_boundary.png`): a
real but small effect — evidence dips modestly (~20→17) in the last few
voxels approaching the boundary from the background side — riding on a
large, flat baseline. Not the dominant structure.

**Correct vs. incorrect voxels — the most informative single result**
(`03_correct_vs_incorrect.png`): incorrect voxels (n=85 of 60,000) are
overwhelmingly concentrated within ±1–3 voxels of the true boundary
(confirms errors *are* boundary-adjacent, as expected — the model rarely
gets a voxel wrong anywhere else). But their evidence is dramatically
lower than correct voxels' (mean 6.3 vs. 19.8; Welch t=10.62,
p=3.4×10⁻¹⁷) **despite occupying almost the same narrow boundary-distance
band**. Since boundary distance is nearly constant across the incorrect
group, it cannot be producing this evidence separation — the network
genuinely encodes more uncertainty on voxels it gets wrong, via
something richer than boundary geometry.

**Tumor vs. background split** (`04_tumor_vs_background.png`): within the
tumor class, evidence spans nearly its full 0–33 range at every boundary
distance from 1 to 6 — no visible within-class boundary-distance
structure. Class-conditional Pearson correlations have **opposite signs**
(tumor r=+0.353, background r=−0.160) — a single global
`evidence = f(boundary_distance)` model cannot produce a sign flip like
this; the relationship is genuinely class-dependent, not a universal
boundary law.

## Interpretation

The evidential branch is **not** simply a boundary detector. Boundary
distance predicts roughly *where* errors occur (errors cluster near the
boundary, unsurprising for a segmentation task) but does not predict
evidence *magnitude* well — the strong, real correct-vs-incorrect
evidence separation happens within a boundary-distance band where
boundary distance itself is nearly constant. This means the majority of
evidence variance, and specifically the useful calibration signal (lower
evidence on errors), is carried by something else in the latent
representation.

**This directly motivates continuing into E1.2** (feature magnitude ‖z‖
vs. evidence) and eventually the representation-entanglement diagnostic
(Concept 0 in `PHASE_E3_CANDIDATE_CONCEPTS.md`) — the simplest possible
explanation for uncertainty (boundary geometry) has been falsified, which
is progress regardless of outcome per the pre-registered success
criterion.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e_latent_analysis/e1_1_boundary_vs_evidence.py` | Analysis script (7 analyses) |
| `experiments/exp_e_latent_analysis/e1_1_results/` | Plots + `results_summary.json` |
| `experiments/exp_e_latent_analysis/extracted/per_voxel_stats.csv` | Source data (30 volumes, 60k voxels) |

---

**Completed**: 2026-08-04
