# Phase E55 — Dual-Resolution Local Refinement: Result

## Context

E54 showed whole-volume 96³ training is flat vs. baseline (0.9066, +0.03pp). E55 built a genuinely novel, fully differentiable mechanism: the existing 64³ global pathway kept completely intact, with a small local pathway examining a native-resolution crop centered on a differentiable soft-centroid derived from the global pathway's own coarse prediction, fused back in via a learned gate. All 6 pre-training verification checks passed with real, measured evidence (coordinate-mapping unit tests, real-subject label-crop alignment on 10 subjects at 99–100% lesion-mass recovery, bit-for-bit ablation-safety, confirmed gradient flow through the crop into the global trunk, memory profiling, clean smoke test).

## Training

Full 30-epoch run, seed 0, `UNet3D_v9`. Physical batch=4, gradient accumulation over 2 steps (effective batch 8). `lambda_local=0.2340`, gradient-matched (target ratio 1.0) — value-matched calibration (1.1823) would have caused a 5.05× gradient blowup, correctly rejected. Clean throughout: no instability, no NaN/Inf, kill condition never triggered.

`fusion_gate` climbed from its 0 init to a peak of ~0.356 by epoch 6, then **declined** over the remainder of training to 0.234 by epoch 30 — the model dialed back its reliance on the local pathway as training progressed, rather than converging to an increasing or stable trust level. `local_loss` decreased steadily throughout (0.66 → 0.20), confirming the local pathway itself learned its own refinement task, independent of whether that refinement ultimately helped the fused output.

## Result

| Metric | Epoch 30 (final, also best) |
|---|---:|
| Val Dice | **0.9007** |
| Val HD95 | 1.30 (best of any condition tracked this session) |
| Val Precision / Recall | 0.9042 / 0.9000 |

## Verdict against success criteria

- Canonical baseline (0.9063) → **−0.56pp**. Worse than baseline.
- D4-only (0.9096) → **−0.89pp**.
- E54/A96 (0.9066, same-resolution-family comparator) → **−0.59pp**.
- Required (≥0.9163): **−1.56pp. NOT MET, not close.**

## Decision

**Per the pre-declared plan and this project's own kill discipline: seed 0 shows no real signal (in fact a negative one), so this does not proceed to seeds 1–2.** Running two more seeds to confirm a result already below baseline would cost ~6-7 more hours without a plausible path to changing the conclusion.

## Honest interpretation

This is a clean, real negative result, not a training failure — every verification check the mechanism was built on passed, the gate learned a genuine intermediate value rather than collapsing to 0 or exploding, and the local pathway demonstrably learned its own task (falling `local_loss`). The mechanism worked exactly as designed; it simply didn't help, and by a specific, measurable margin, actively hurt the fused prediction relative to not having it at all.

A plausible explanation, offered honestly rather than as an excuse: the fusion is additive in logit space (`logit_fused = logit_coarse + gate * scatter(logit_local)`), and the local pathway is trained to minimize its own loss on the native-resolution crop — a genuinely different objective landscape (crop-local Focal Tversky against native-resolution GT) than the fused output's own overall segmentation quality. Nothing in the loss explicitly ties `local_loss`'s own improvement to `probs_fused`'s own Dice; the local pathway can get better at its own task while the fusion still degrades whole-volume Dice, if the scattered correction is well-calibrated to the crop's own boundary but poorly calibrated relative to the global pathway's own (differently-trained, differently-scaled) logit magnitudes at the fusion boundary. The declining `fusion_gate` trajectory is itself suggestive: the model's own optimization, given the freedom to reduce reliance on this pathway, chose to do so as training progressed rather than lean into it further — consistent with the mechanism providing a net-negative contribution to the primary loss it's actually evaluated against.

The falling HD95 (best of any condition this session, 1.30 at final epoch) is a genuine, real signal that the local pathway is doing something related to boundary precision — this is not a null result across every metric, just Dice specifically, which is the metric this project's own success bar is defined against.

## Implication

This is now the **eighth mechanism** tried since the strategic pivot (E44 killed, E45 +0.47pp 1-seed, E46 +0.39pp 1-seed, E49/CCABA +0.31pp 3-seed, E50/IECG +0.02pp 3-seed, E51/CCAG +0.32pp 3-seed, E52 killed at smoke test, E53 incomplete/0.8969 1-seed, E54/A96 +0.03pp 1-seed, E55 **−0.56pp** 1-seed) and the first to land clearly *below* the canonical baseline on a properly-verified, non-collapsed run. The pattern across the whole post-pivot arc — architecture-level and objective-level, single mechanisms and combinations, static and live/learned, whole-volume and dual-resolution — continues to show no mechanism clearing +1.0pp, with more structurally ambitious attempts trending flat-to-negative rather than positive.

## Recommendation

Per this project's own kill/stop discipline: no further single-mechanism attempts on this dataset/architecture family. Combined with the entire E44–E55 arc (`PHASE_POST_E43_SUMMARY.md`, to be updated), this is now strong, repeated, methodologically rigorous evidence that the ceiling on this exact 1,251-subject/FLAIR-only/64³-derived setup is not attributable to any single tested mechanism's quality — it has been probed from essentially every angle this project's own causal-diagnostic chain (E43→E47→E48) and subsequent evidence (E29, E54) could motivate. The defensible contribution for a paper remains the causal-diagnostic methodology and the multi-seed variance-correction finding, now with an even more exhaustive negative-result arc behind it.

## Artifacts on disk

- `neuroscan_3d_v9.py` (git-tracked)
- `experiments/exp_e12_eggo_m/e55/test_v9_coordinate_mapping.py`, `test_v9_label_crop_alignment.py`, `test_v9_ablation_safety.py`, `profile_memory_v9.py`, `calibrate_lambda_local.py`, `train_e55_dual_resolution.py` (disk only, per `experiments/` gitignore convention)
- `experiments/exp_e12_eggo_m/e55/e55_smoke_log.txt`, `e55_seed0_run_log.txt`
- `experiments/exp_e12_eggo_m/e55/runs/DualRes_seed0/epoch_metrics.csv` (30 rows), `checkpoints/` (disk only, gitignored)
