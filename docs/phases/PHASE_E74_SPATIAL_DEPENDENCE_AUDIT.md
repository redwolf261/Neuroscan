# Phase E74 — Representation Spatial-Dependence Audit

## Origin

User's reframe after E71-E73: stop trying to *act on* self-diagnostic
information directly (routing/reweighting/gating all killed for a
related reason). Instead, measure what representation property makes a
tensor translation-sensitive in the first place — a diagnostic phase,
not a training experiment, before designing any fix.

## Method

Pure inference-time causal intervention, no training, no architecture
change. Reused E65's own validated translation intervention (3-voxel
`torch.roll`, wrapping, preserves every local neighborhood and the full
value distribution — changes only spatial address) and E64's split-
forward isolation design, retargeted to the **multimodal MM-seed0**
checkpoint (E65 used the FLAIR-only checkpoint; E74 moves to the
checkpoint the rest of tonight's chain, E70-E73, has been built on).

Two loci tested: **enc1** (E65's own locus, the skip junction) and the
**bottleneck** (a new, deeper, coarser 8³ locus) — testing whether
translation-sensitivity is a skip-specific phenomenon or general across
depth.

For each of 125 held-out subjects, at each locus: measured the Dice drop
from translating that tensor, plus two properties of the INTACT tensor
— high-frequency spatial-energy fraction (Laplacian-filtered variance /
total variance, sanity-checked against synthetic smooth vs. noisy signals:
noisy scores 31x higher, confirming the metric behaves correctly) and mean
feature magnitude (L2 norm per voxel, averaged). Partial Spearman
correlations control for native lesion size throughout, matching
E48/E65's own confound-control convention.

## Results

### Translation-sensitivity generalizes across depth

| Locus | Mean Dice drop | Significance |
|---|---|---|
| enc1 (skip junction) | **0.317** | p=2.0e-64, permutation p<0.001 |
| bottleneck (new locus) | **0.211** | p=1.4e-23, permutation p<0.001 |

Both large and highly significant. This is new relative to E65: E65 only
established translation-sensitivity at the skip junction. E74 shows it is
**not** confined to that locus — it also appears at a much deeper, coarser
representation. This argues against a skip-connection-specific fix and
toward something more general about how this network encodes position.
(Note: a 3-voxel shift is a much larger *fraction* of the bottleneck's 8³
extent than of enc1's 64³ extent — the two drop magnitudes are not
directly comparable as effect sizes, only as confirmation that the
phenomenon exists at both loci.)

### What predicts sensitivity: magnitude, not frequency — and the frequency story reverses by depth

| Locus | Partial ρ(drop, HF energy \| size) | Partial ρ(drop, magnitude \| size) |
|---|---|---|
| enc1 | +0.054, n.s. (p=0.55) | **+0.301, p=6.4e-4** |
| bottleneck | **−0.589, p=4.8e-13** | **+0.525, p=3.3e-10** |

**Feature magnitude is the consistent predictor** — positive at both loci
(weak at enc1, strong at bottleneck): higher-magnitude features are more
reliant on their exact spatial address, at both depths.

**High-frequency content does NOT predict sensitivity as hypothesized.**
At enc1 it has no relationship at all. At the bottleneck it runs
**backwards**: smoother, lower-frequency bottleneck regions are *more*
translation-fragile than sharp, high-frequency ones — the opposite of the
"coordinate-dependent high-frequency detail is what's fragile" hypothesis
this phase set out to test.

## Interpretation

The simple "high-frequency → coordinate-dependent → fragile" hypothesis
is **not supported** — it fails outright at enc1 and reverses at the
bottleneck. The evidence instead points toward a **magnitude-based**
story: strongly-activated features carry more positionally-specific
information and are therefore more vulnerable to translation, regardless
of depth. At the bottleneck specifically, the reversed frequency
relationship suggests smooth/diffuse regions may encode coarse *global*
spatial context (e.g. rough anatomical location) that a translation
directly corrupts, while sharp, spatially concentrated activations encode
something more translation-robust (distinctive local content that
survives a small positional shift).

This is a genuine update, not a confirmation of the pre-registered
hypothesis as stated — the phase is reported as a characterization per
its own pre-declared framing, not a pass/fail result.

## What this means for next steps

- A future fix should NOT be framed around "reduce high-frequency
  content" (E65's own smoothing arm already showed this addresses the
  smallest of its four effects, and E74 now shows frequency isn't even
  the right axis).
- The magnitude relationship is consistent and worth taking as this
  phase's actionable candidate: something like **magnitude-aware
  positional robustness** (e.g. encouraging high-magnitude features to
  be less sharply localized, or supplying explicit positional
  information proportional to feature magnitude) is a more evidence-
  backed direction than anything tried in E70-E73.
- The bottleneck's reversed frequency-sensitivity relationship is
  unexpected enough to treat as its own open question, not yet explained
  — flagging honestly rather than folding it into the magnitude story
  without further evidence.

## Artifacts

- `experiments/exp_e12_eggo_m/e74/run_e74_spatial_dependence_audit.py`
- `experiments/exp_e12_eggo_m/e74/E74_spatial_dependence_table.json` (125 subject records, both loci)
- `experiments/exp_e12_eggo_m/e74/E74_spatial_dependence_summary.json`
