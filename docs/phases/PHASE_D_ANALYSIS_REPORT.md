# Experiment D: Final ABO vs Baseline Comparison — Analysis Report

**Status**: ✅ Complete (50 epochs × 3 seeds, locked config r_target=0.30/gamma=0.10)

**Date**: 2026-08-04

## Purpose

This is the definitive test of ABO (Adaptive Branch Optimizer) against the
frozen baseline, under the exact matching protocol (architecture, losses,
data split, seed logic, 50 epochs, 3 seeds — see `PHASE_A5_BASELINE_FROZEN.md`).
It follows a full controlled-experiment arc: Phase B (diagnosis) → C0
(negative result on static reweighting) → C1 (ABO controller, verified to
move gradient geometry) → C2a/C2b (hyperparameter sensitivity sweeps,
confirming the locked config sits in a broad, stable operating region, not
a fragile cherry-picked point). Experiment D asks the only question left:
**does moving the gradient geometry actually improve segmentation?**

## Result summary

| metric | ABO (n=3) | Baseline (n=3) | Delta |
|---|---|---|---|
| Dice | 0.9100 ± 0.0006 | 0.9100 ± 0.0007 | +0.0000 |
| IoU | 0.8358 ± 0.0010 | 0.8357 ± 0.0011 | +0.0001 |
| Precision | 0.9183 ± 0.0008 | 0.9158 ± 0.0035 | +0.0025 |
| Recall | 0.9031 ± 0.0011 | 0.9053 ± 0.0043 | −0.0022 |
| F1 | 0.9100 ± 0.0006 | 0.9100 ± 0.0007 | +0.0000 |
| HD95 | 1.317 ± 0.171 | 1.302 ± 0.187 | +0.015 (slightly worse) |
| ECE | 0.0381 ± 0.0017 | n/a (no uncertainty head) | — |

Per-seed ABO best-Dice: seed0=0.9101 (epoch 45), seed1=0.9105 (epoch 42),
seed2=0.9093 (epoch 36). Per-seed baseline: 0.9107, 0.9095, 0.9097. All
six values fall within a ~0.001 band — genuinely indistinguishable, not
"close but ABO slightly worse."

**Headline finding: Dice/IoU/F1 are statistically indistinguishable from
baseline, to 4 decimal places.** This is a clean null result on the
segmentation outcome metric, not a partial win or a wash that favors
either side. Precision rose slightly at recall's expense (both within
baseline's own seed-to-seed noise band), and HD95 was marginally worse
(also within noise).

## What DID change: the gradient geometry itself

ABO's `mean_effective_ratio` (per-epoch mean of
`raw_ratio × applied_multiplier` — what the trunk actually received from
the evidential branch, not the branches' own unmodified gradients)
averaged **0.1886 ± 0.0007** across all 50 epochs and 3 seeds. This is
roughly **double** the passive baseline's natural steady-state ratio
(~0.098, measured in Phase B / C1 monitor mode). ABO did exactly what it
was designed to do — this was verified directly in C1 and re-confirmed
here across the full 50-epoch/3-seed protocol, not just the earlier
10-epoch pilot.

**So the causal chain is broken at the last link**: ABO → real, robust,
substantial change in trunk gradient composition → **no** measurable
change in Dice/IoU/F1. Every earlier link in the chain (B, C0, C1, C2a,
C2b) is independently verified and not in question; only the
gradient-geometry → segmentation-quality link fails to hold.

## New finding from re-analyzing existing logs (Step E2, not part of the original Experiment D plan)

The Experiment D epoch-level logs (`expD_active_seed{0,1,2}/epoch_metrics.csv`)
were mined for correlations between `mean_effective_ratio` and downstream
metrics that were not examined in the original Experiment D write-up.

**Naive pooled correlation is confounded and misleading.** Across all 150
epoch-points (3 seeds × 50 epochs), effective_ratio appears strongly
correlated with Dice (r = −0.88). This is an artifact: both
effective_ratio and Dice rise together over early training (α ramps up
as the controller warms up; Dice rises as the model converges) — a
shared training-progress trend, not a causal relationship between the two
quantities.

**Controlling for training progress two ways — restricting to late
epochs (30–49) where effective_ratio has stabilized, and linear partial
correlation controlling for epoch index directly — both give the same,
consistent answer:**

| Target metric | Raw pooled corr | Partial corr (controlling for epoch) | Interpretation |
|---|---|---|---|
| val_dice | −0.88 | **−0.03** | Confound fully explained by epoch; no real relationship. Reconfirms Experiment D's headline finding at finer (within-run) grain. |
| val_ece | +0.71 (pooled, sign artifact of scale) → −0.68 (late epochs) | **−0.65** | **Real, non-confounded relationship survives.** |
| val_hd95 | +0.84 (pooled) → −0.47 (late epochs) | **−0.39** | Weaker but likely real; noisier than the ECE signal. |

**Reading**: within late training, epochs/batches where ABO pushed a
higher effective evidential-gradient contribution into the trunk had
meaningfully *better* (lower) calibration error, independent of how far
training had progressed. This is a mechanistic signal, not simply "having
an uncertainty head helps" (the baseline has no comparator for ECE at
all, so this couldn't have been seen in the headline numbers). The HD95
relationship points the same direction but is weaker and should be
treated as a secondary lead, not a primary claim.

**This survives Experiment D's own null result — it does not contradict
it.** ABO's magnitude control appears to have no causal channel into the
Dice/IoU/F1 objective, but a real one into calibration quality. That is
a meaningfully different, more specific claim than "ABO does nothing."

## What was NOT investigated (scope boundary, explicit)

- **Per-subject analysis.** All Experiment D and Phase B/C0/C1 logs are
  batch-level (8 subjects averaged per training batch) or epoch-level.
  There is no existing per-subject record linking a specific tumor's
  gradient statistics to its own HD95, lesion volume, or boundary-voxel
  count. Answering "does the gradient ratio correlate with tumor size or
  boundary difficulty" would require a new per-subject inference pass
  over the saved `expD_active_seed{0,1,2}/checkpoints/best.pth`
  checkpoints against the validation set — not yet built as of this
  report.
- **Why** the ECE relationship holds (e.g., whether it's simply that the
  evidential head receives more effective training signal when its
  trunk-gradient contribution is larger, a training-budget effect, versus
  something more specific to how magnitude control interacts with the
  Beta-KL loss) is not diagnosed here.

## Interpretation and implication for future work

The working hypothesis going into Phase C ("gradient magnitude imbalance
is the bottleneck limiting Dice") is **falsified** by Experiment D for
this architecture and dataset. The model reaches an equivalent
segmentation optimum regardless of how the trunk composes the two
branches' gradients — magnitude balancing, done correctly and verified
robust, is not where the Dice ceiling lives here.

The calibration finding above is a genuine, non-obvious result that
survived a real confound check, and reframes what ABO is actually good
for: not a segmentation-quality lever, but possibly a **calibration**
lever. This is the strongest concrete thread this analysis surfaced for
motivating "Direction 3: uncertainty-driven optimization" (see
`abo_frozen_lessons_learned` memory / forthcoming candidate-optimizer-concepts
document) over the other candidate directions, pending the literature
review's own independent read on which direction is best supported.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_c1_abo/expD_active_seed{0,1,2}/epoch_metrics.csv` | Full per-epoch metric history (loss/dice/iou/precision/recall/f1/hd95/ece/pct_damped/mean_effective_ratio) per seed |
| `experiments/exp_c1_abo/expD_active_seed{0,1,2}/logs/` | Batch-level gradient norms, cosine similarity, controller diagnostics, loss history |
| `experiments/exp_c1_abo/expD_active_seed{0,1,2}/checkpoints/best.pth` | Best model weights per seed (available for future per-subject inference) |
| `experiments/exp00b_baseline_convergence/seed_{0,1,2}/results.json` | Frozen baseline comparison numbers |
| `experiments/exp_c1_abo/train_abo.py` | Training script (now logs precision/recall/mean_effective_ratio, added for this experiment) |

---

**Completed**: 2026-08-04
