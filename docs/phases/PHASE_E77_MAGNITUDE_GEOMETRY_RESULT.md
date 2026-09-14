# Phase E77 — Natural enc1 Magnitude Geometry: A/B OVERLAP, not clean A

## Origin

Following E76's clean PASS (local causal relationship between enc1
magnitude and translation sensitivity, confirmed in the network's normal
operating regime), this phase measures the ABSOLUTE geometry of the
natural magnitude distribution — per the user's design — before any
intervention formula is chosen. No training, no architecture change.

## Method

125 held-out subjects, enc1's per-voxel feature-vector norm r(p) =
||z(p)||_2, pooled voxel-level (10,000 subsampled voxels/subject,
1.25M pooled) and per-subject. Voxels classified as background / lesion-
boundary (within 2 voxels of the lesion edge, either side) / lesion-
interior via a distance transform. Local per-voxel translation impact
reused E73's own construction (|probs_intact - probs_translated|).

## Results

### (1) Distribution: sharply bimodal, not a smooth long-tail

| Percentile | r(p) |
|---|---|
| p10 | 1.389 |
| p25 | 1.389 |
| p50 | 1.389 |
| p75 | 2.348 |
| p90 | 6.158 |
| p95 | 7.485 |
| p99 | 10.076 |

p10 = p25 = p50 (all 1.389) — the bottom half of ALL voxels sit at
essentially one fixed low value. This is not a smoothly-tailed
distribution; it looks like two populations, a large flat-magnitude mass
and a smaller high-magnitude one.

### (2) Class breakdown: the high-magnitude population IS the lesion

| Class | n (pooled) | mean r |
|---|---|---|
| background | 1,223,709 | 2.350 |
| boundary | 22,420 | 8.084 |
| interior | 3,871 | 10.906 |

Clean, large, monotonic gap. High magnitude at enc1 is, essentially by
construction, lesion salience — not a separate, semantically-empty
population.

### (3) Correlation with translation sensitivity: strong pooled, NOT significant per-subject

- Pooled voxel-level Spearman(r, local translation impact) = **+0.539**
  (p≈0, n=1.25M) — strong within-subject relationship: which VOXELS are
  sensitive tracks magnitude closely.
- Subject-level Spearman(mean r, subject-level translation drop) =
  **+0.132, NOT significant** (p=0.14, n=125) — magnitude does NOT
  reliably distinguish which SUBJECTS are more translation-fragile
  overall. This nuance did not feed the script's automatic outcome
  classifier and is flagged here explicitly.

### (4) Magnitude already carries useful information

r at error voxels: mean 8.05, vs r at correct voxels: mean 2.47 (paired
t, n=125 subjects, p=5.2e-89) — consistent with E73's finding that
error-prone regions are high-magnitude. Confirms the danger case the
user explicitly named: **high magnitude is strongly tied to both lesion
content AND the network's own error-prone regions.**

### (5) Tail analysis

Voxels at/above p90 (r ≥ 6.158): **10.0%** of all voxels, mean local
translation impact **15.5x** higher than the remaining 90%.

## Outcome: A and B simultaneously, not clean A

The script's automatic threshold rule classified this as outcome A
("rare, disproportionately sensitive tail — promising for soft tail
control"). **That classification is incomplete and is corrected here**:
the threshold check for "rare tail" and the threshold check for "tracks
lesion" were computed independently and never cross-referenced against
each other. In fact:

- The high-magnitude tail (top 10%, 15.5x more translation-sensitive)
  and the lesion-tissue population (interior/boundary, means 8.1–10.9)
  are substantially **the same voxels**. p90 = 6.16 sits between the
  boundary mean (8.08) and background mean (2.35) — the tail is not a
  rare anomaly separable from lesion content; it largely **is** lesion
  content.

This is precisely the risk case B warns about: **naive magnitude
suppression in the tail would suppress the tumor signal itself**, not an
incidental artifact. A clean "soft tail-control mechanism" as originally
imagined under outcome A is not warranted without a way to distinguish
"large because it's genuinely salient lesion content" from "large in a
way that specifically drives translation fragility, independent of
lesion salience."

## What this means for intervention design

Any viable mechanism must be **direction-preserving AND lesion-content-
preserving** — not a blanket cap or suppression on r(p) in the high
range, since that range is where the tumor signal itself lives. The
subject-level null (magnitude doesn't distinguish fragile subjects,
only fragile voxels within a subject) also narrows the design: this is
a **within-volume, spatially-local** phenomenon, not a subject-level
property — ruling out any subject-level scalar mechanism (consistent
with E71/E72's own kills of subject-level approaches, for an unrelated
but reinforcing reason).

A defensible next design direction: rather than shrinking magnitude in
the high-r region, the mechanism should aim to make the DECODER less
sensitive to the exact spatial position of high-magnitude features
specifically — e.g. a learned local spatial-smoothing or soft-
registration operation applied ONLY where magnitude is high, rather than
a magnitude rescaling at all. This reframes the candidate intervention
from "control magnitude" to "control the POSITIONAL PRECISION demanded
of high-magnitude features" — closer to what the causal chain (E65
translation dominance + E76's magnitude-gated sensitivity) actually
supports, without touching the magnitude value itself (avoiding the
lesion-suppression risk entirely).

## Artifacts

- `experiments/exp_e12_eggo_m/e77/run_e77_magnitude_geometry.py`
- `experiments/exp_e12_eggo_m/e77/E77_per_subject_geometry.json`
- `experiments/exp_e12_eggo_m/e77/E77_magnitude_geometry_summary.json`
  (note: this file's own `outcome` field says "A" — superseded by this
  document's corrected A/B-overlap reading; kept unedited as a faithful
  record of the script's literal threshold output)
