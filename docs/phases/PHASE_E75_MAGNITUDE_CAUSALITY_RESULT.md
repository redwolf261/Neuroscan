# Phase E75 — Magnitude-Positional Sensitivity Causality Test: SPLIT RESULT (enc1 PASS, bottleneck FAIL)

## Origin

Per the user's design following E74: E74 found feature magnitude
correlates with translation-sensitivity at both enc1 and the bottleneck,
controlling for lesion size. Correlation across naturally-varying subjects
does not establish causation (could be confounded by lesion identity/size,
or "large activations are just important" — a near-tautological
confound the user explicitly flagged). E75 tests causation directly, no
training: decompose each feature vector z(p) = ||z(p)|| · u(p), rescale
ONLY the magnitude by alpha ∈ {0.25, 0.5, 1.0, 2.0}, holding direction
u(p) exactly fixed (verified numerically, max diff < 1e-5 across all
alphas before any subject was processed), then measure whether the SAME
translation intervention (E65/E74's 3-voxel roll) hurts more at higher
alpha and less at lower alpha.

Pre-declared rule: PASS requires monotonic, all-pairwise-significant S(alpha)
increase at BOTH enc1 and bottleneck independently.

## Results

### enc1: PASS, cleanly

| alpha | mean S (translation-induced Dice drop) |
|---|---|
| 0.25 | 0.204 |
| 0.5 | 0.242 |
| 1.0 | 0.317 |
| 2.0 | 0.372 |

Monotonic. Every pairwise comparison significant (S(2.0) vs S(1.0):
p=3.0e-25; S(1.0) vs S(0.25): p=9.2e-35; S(2.0) vs S(0.25): p=5.0e-49,
matching on Wilcoxon). **Artificially inflating magnitude while holding
direction fixed causally increases translation damage; shrinking it
causally decreases it.** This is genuine causal evidence, not
correlation — the earlier E74 finding is confirmed causal at this locus.

### Bottleneck: FAIL, and reverses direction

| alpha | mean S |
|---|---|
| 0.25 | 0.213 |
| 0.5 | 0.247 |
| 1.0 | 0.211 |
| 2.0 | **0.125** |

Not monotonic. S(2.0) is significantly LOWER than S(1.0) (p=1.7e-15) and
S(0.25) (p=7.9e-10) — the opposite of what E74's correlational finding at
this locus predicted, and opposite of enc1's own causal result.

### Confound check (both loci): magnitude rescaling alone damages the network

Even without any translation, rescaling magnitude away from alpha=1.0
degrades the network's own prediction substantially at both loci (e.g.
enc1 intact Dice: 0.902 at alpha=1.0 → 0.867 at alpha=2.0; bottleneck:
0.902 → 0.880). At alpha=2.0 the network is already operating in a
damaged, out-of-distribution regime before translation is even applied —
a real caveat on how cleanly the bottleneck's S(alpha) reversal isolates
the causal question the phase set out to test, though this same confound
is present at enc1 too and did not prevent a clean monotonic result there.

## Verdict

**Per the pre-declared rule (both loci required): OVERALL FAIL.** Magnitude
is not confirmed as a general, cross-depth causal driver of translation
sensitivity — it is causally confirmed at enc1 specifically, and reverses
at the bottleneck. This is a genuine, reportable split, not an ambiguous
result to be waved away in either direction.

## Status: awaiting decision on how to proceed

Two honest readings, put to the user rather than resolved unilaterally:
1. Treat this as "enc1-specific causal confirmation, bottleneck is a
   genuinely different regime" and design a magnitude-decoupling
   intervention scoped to enc1/the skip connection only (where E65's
   original translation finding was strongest and where E75 now confirms
   causation cleanly).
2. Treat the failed bottleneck leg as disqualifying the general magnitude
   hypothesis per the strict pre-declared rule, and do not proceed to any
   decoupling intervention on this basis.

## Novelty context (searched before this design, informs whichever path is chosen)

- **Weight Normalization** (Salimans & Kingma 2016) and its 2026
  "Magnitude-Direction Decoupling" successor decompose LEARNABLE WEIGHTS
  into magnitude x direction — occupied, but a different target
  (parameters, not activations/features).
- **MPLSeg** (magnitude-aware/phase-sensitive segmentation, occupied)
  decomposes FOURIER magnitude/phase of features for semantic-vs-
  localization separation — real, published, conceptually adjacent (both
  treat "magnitude" and positional information as separable), but a
  mathematically different decomposition (spectral, not per-voxel vector
  norm) from what E74/E75 test.
- No hit found for "per-voxel feature-vector-norm causally drives
  positional sensitivity, decouple by rescaling the norm channel
  specifically" — the narrow gap this phase's evidence would motivate,
  if pursued.

## Artifacts

- `experiments/exp_e12_eggo_m/e75/run_e75_magnitude_causality_test.py`
- `experiments/exp_e12_eggo_m/e75/E75_magnitude_causality_table_enc1.json`
- `experiments/exp_e12_eggo_m/e75/E75_magnitude_causality_table_bottleneck.json`
- `experiments/exp_e12_eggo_m/e75/E75_magnitude_causality_summary.json`
