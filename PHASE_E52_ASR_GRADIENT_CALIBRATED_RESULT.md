# Phase E52 — ASR Gradient-Calibrated Re-Attempt: Killed at Smoke Test

## Context

E34's original `[R]` (alpha_c) component-reweighting condition collapsed training catastrophically (best Dice 0.7108 vs. 0.9063 baseline), using a `lambda_cw` calibrated by loss-**value** match only. `calibrate_lambda_cw_v2.py` measured directly that E34's original lambda (~4.75–5.2) would produce a **43× gradient-magnitude blowup** relative to the main segmentation loss — for context, well beyond even the largest blowup this project has previously measured and rejected (E49: 25.4×) — strongly suggesting mis-calibration, not the mechanism itself, was the cause. See `PHASE_E52_ASR_GRADIENT_CALIBRATED_DESIGN.md` for the full design and pre-declared gates.

`lambda_cw` was recalibrated by gradient magnitude to **0.1101** (target ratio 1.0, matching E45's `lambda_d8` treatment), and a 2-epoch smoke test was run per the pre-declared gate before committing to a full 3-seed run.

## A real performance bug was caught and fixed first

The first draft of `train_e52_asr_gradient_calibrated.py` rebuilt the labeled-component cache from scratch every batch (calling `precompute_labeled_64_cache` per-batch instead of once for the full training set) — reproducing, in a new script, the exact ~5–7× slowdown bug E34's own `component_weighted_loss.py` docstring already documented and had already fixed once. The smoke test did not complete a single epoch within 10 minutes under this bug. Fixed by precomputing the labeled-64³ cache for all 1,126 training subjects once, upfront, in `__init__` (507.6s one-time cost) — matching E34's own established convention. This is disclosed here as a real implementation mistake caught and corrected before any result was trusted, not as a scientific finding.

## Smoke test result

| Epoch | Train Dice | Val Dice | Val Precision | Val Recall | `cw_loss` |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0875 | **0.1021** | 0.0538 | 1.0000 | 0.0294 |
| 2 | 0.1596 | **0.4024** | 0.2547 | 0.9761 | 0.0169 |

## Verdict against the pre-declared smoke-test gate

**Gate**: val Dice must reach at least the 0.5–0.6 range other healthy conditions reach by epoch 2 (not E34's own `[R]` pattern: 0.057–0.069 at epoch 1–2). **FAILED.** Epoch 1 (0.1021, precision 0.054, recall 1.0 — the same "predict everything as tumor" degenerate mode E34's original `[R]` showed at epoch 1: 0.069, precision 0.036, recall 1.0) and epoch 2 (0.4024) both land well short of the healthy range every other post-pivot condition reaches at this point (E45's D4+D8: 0.505 at epoch 1; E51's CCAG: 0.415 at epoch 1; both already past 0.4 at epoch 1, where ASRv2 is still at 0.10).

Per this phase's own pre-declared rule, this stops here — no scaling to the full 3-seed run chasing an already-failed result.

## What this result actually shows (the useful finding)

**Gradient calibration alone did not fix the collapse.** This is more informative than a simple repeat of E34's own null, because it isolates the cause:

A direct comparison of epoch-1 metrics across conditions shows `seg_loss` and `bnd` (boundary loss) values are nearly **identical** across ASRv2 (this phase), E45's D4+D8, and E51's CCAG — all in the 0.77–0.78 / 0.56–0.58 range respectively. Since `seg_loss` is the term that most directly drives Dice, and it looks the same across all three, **the `cw_loss` term itself — even at a unity gradient ratio, no longer any kind of magnitude blowup — is actively working against early-training segmentation performance**, not merely adding a manageable amount of noise on top of an otherwise-normal trajectory. Gradient magnitude was the wrong single lever to fix: the *direction* of the component-weighted gradient (which voxels/components it pushes the model toward emphasizing early in training, when large components still dominate correct predictions and small/degraded components are the hardest, noisiest signal in the dataset) appears to be actively counterproductive during the earliest training phase, independent of its magnitude.

This is a plausible, disclosed (not proven) explanation, not a new confirmed mechanism: early in training, before the model has learned to segment large, easy lesions reliably at all, an auxiliary loss that immediately upweights the hardest, most-degraded, often near-vanishing components may be actively fighting the coarse-to-fine curriculum every other successful condition in this project (E45, E46, E49, E51) implicitly relies on — those all supervise coarse/global structure (D4, D8, CCABA, attention routing) rather than directly reweighting toward the hardest fine-grained components from epoch 1.

## Decision

**Killed at the smoke-test gate.** Per the pre-declared rule, this phase does not proceed to a full 3-seed run. The objective-level lever (E34/E52's component-reweighting-by-resize-survival idea) is now closed on its second, more carefully controlled attempt: gradient-magnitude calibration was necessary but not sufficient to fix the original E34 collapse, ruling out miscalibration as the *sole* explanation and pointing instead toward the reweighting mechanism's own interaction with early-training dynamics as the likely deeper cause.

## Implication for the project

This is the seventh mechanism attempt since the strategic pivot (E44 killed, E45 +0.47pp 1-seed, E46 +0.39pp 1-seed, E49/CCABA +0.31pp 3-seed, E50/IECG +0.02pp 3-seed, E51/CCAG +0.32pp 3-seed, E52/ASRv2 killed at smoke test) and the first attempt at an objective-level (not architecture-level) lever. It closes cleanly and cheaply (a 2-epoch smoke test, not a full 3-seed run) — the pre-declared smoke-test gate did its job, saving the compute a full run would have cost on an idea that was already failing by epoch 2.

Combined with E44's own earlier objective-level kill (RCGW, loss-reweighting-by-convergence-gap), this project has now tried and killed **two structurally distinct objective-level mechanisms** in addition to five architecture-level ones. The pattern across all seven is now broad enough (spanning loss scheduling, new heads, attention routing, causal-calibrated amplification, live counterfactual gating, mechanism combination, and component-level loss reweighting) that a further, eighth mechanism attempt on this same architecture/dataset has an even weaker prior of success than before E52. This strengthens, rather than changes, the recommendation standing since E51: the project's defensible contribution is the causal-diagnostic chain (E43→E47→E48) and the multi-seed variance-correction finding (E49→E50→E51), not a further search for the mechanism that finally clears +1pp.

## Artifacts on disk

- `experiments/exp_e12_eggo_m/e52/calibrate_lambda_cw_v2.py`, `E52_lambda_cw_calibration.json`
- `experiments/exp_e12_eggo_m/e52/train_e52_asr_gradient_calibrated.py`
- `experiments/exp_e12_eggo_m/e52/e52_smoke_log.txt` (full 2-epoch smoke test log)
