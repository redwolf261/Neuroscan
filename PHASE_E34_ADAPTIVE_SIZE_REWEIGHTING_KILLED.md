# Phase E34 — Adaptive Size-Reweighting (ASR): Killed, Both Conditions

**Note**: this report was reconstructed from raw disk artifacts (`experiments/exp_e12_eggo_m/e34/` — calibration scripts, JSON weight tables, and full training logs) during a documentation audit; no synthesized `.md` report existed for this phase at the time of the audit, unlike every neighboring phase (E33, E35–E43). The underlying experiment was real and complete; only the write-up was missing. This is also the phase referenced elsewhere in project memory as the origin of the "calibrate by gradient magnitude, not loss value alone" mandatory safeguard — see Section 4 below.

## Context

Following E33's novelty audit (identifying Component-Adaptive Tversky, i.e. a native-GT-volume-based inverse weighting, as the closest prior art for size-based component reweighting), E34 implemented and trained **two concrete weighting schemes** derived from the project's own per-component confidence/size signal:

1. **Condition `[S]`** — size-based weighting: `w_c^size = (V_c + eps)^(-gamma)`, gamma calibrated so its mean/variance match the alpha-based weighting below (component native GT volume `V_c`, eps=1.0).
2. **Condition `[R]`** — alpha_c-based weighting: `w_c^alpha = 1 + kappa*(1-alpha_c)_+`, kappa=3 fixed a priori (not tuned to any downstream Dice outcome), where `alpha_c` is the component's own coarse-grid survival fraction (how much of a connected GT component survives the 64³ resize, the project's own recurring "resize-induced shrinkage" concern from E29/E31).

Both weightings multiply a new auxiliary "component-weighted" loss term (`ComponentWeightedLoss`, `component_weighted_loss.py`) added on top of the existing seg_loss + boundary_loss.

## Method — calibration (disclosed, with one caught bug)

**Weight-table construction** (`calibrate_weights.py`): computed `w_alpha` and `w_size` over all 8,722 native-space GT connected components in the full 1,126-subject training set (not a subsample). Triviality check performed before training, as required by this project's own standing safeguard: `Corr(w_alpha, native_size) = -0.720`, KS statistic 0.367 (p<0.001) — confirms `w_alpha` is genuinely correlated with but not identical to a raw size-based weighting (not a trivial duplicate of `w_size`), a real, disclosed distinction.

**A dataloader-caching bug was caught before trusting calibration**: the original `calibrate_lambda_cw.py` precomputed a native-component cache for only a fixed first-32-subject prefix, then silently skipped any randomly-shuffled batch that wasn't entirely drawn from that prefix. With batch_size=8 drawn uniformly from 1,126 subjects, the probability all 8 land in a fixed 32-subject slice is ~(32/1126)^8 ≈ 0 — every batch was silently skipped, `n_batches_tested` stayed 0, and the calibration function was silently returning its fallback default (1.0) rather than a real measurement. **Fixed** by caching subjects on-demand as encountered in real sampled batches, not a fixed prefix — caught and corrected before the calibrated value was used for training, per this project's own "diagnose implementation bugs before interpreting anomalies as findings" safeguard.

**Final calibration** (`E34_lambda_cw_calibration.json`, post-fix): `lambda_alpha_raw=5.278`, `lambda_size_raw=5.192`, both computed by targeting `seg_loss ≈ cw_loss` **at fresh initialization, model in `.train()` mode** (matching E25b's / E12e's own established convention for loss-value calibration). Final `lambda_cw = 5.2` (rounded down to the smaller of the two, used identically for both conditions for a matched comparison).

**Important caveat, disclosed honestly here**: this calibration targeted **loss value** parity (`seg_loss ≈ cw_loss` at init), not **gradient magnitude** parity. This project's later-established mandatory safeguard — calibrate every new loss-term weight by gradient magnitude, not value alone — was not yet standing policy at the time E34 was run; E34 (together with a separate incident referenced in project memory) is the origin of that policy, not an application of it.

## Result

Both conditions trained for the full 30 epochs, seed 0, otherwise-standard protocol (AdamW, cosine LR, batch 8).

| Condition | Best Val Dice | vs. canonical baseline (0.9063) |
|---|---:|---:|
| `[S]` (size-based) | 0.7360 | **−17.03pp** |
| `[R]` (alpha_c-based) | 0.7108 | **−19.55pp** |

Both trajectories show the signature of a destabilized optimization, not a clean-but-unhelpful run: `[R]` in particular starts near-random (val Dice 0.069 at epoch 1, precision 0.036 with recall pinned near 1.0 — a degenerate "predict everything as tumor" collapse mode) and only partially recovers by epoch 30 (0.71), never approaching the 0.85+ range every healthy run in this project reaches within the first 10 epochs. `[S]` shows the same pattern, milder.

## Interpretation

The component-weighted auxiliary loss term, at its value-calibrated weight, destabilized training badly enough that neither weighting scheme came remotely close to the 0.9063 baseline, let alone the +1pp target. This is consistent with — and became a motivating case for — the project's own subsequent discipline (explicitly applied starting with E45's `lambda_d8` calibration and every deep-supervision-adjacent weight since): a loss term whose **value** is matched to the main segmentation loss at initialization can still have a **gradient** magnitude wildly out of proportion, because Dice/Tversky-family losses and this component-weighted term have structurally different value-to-gradient relationships. E45's own report explicitly measured a 2.18× gradient-magnitude mismatch from a value-matched calibration on a *different* auxiliary term and treated it as disqualifying; E34's failure is the more severe version of the same risk, uncaught at the time because gradient-magnitude checking was not yet mandatory practice.

## Decision

**Killed, both conditions.** Neither the size-based nor the alpha_c-based component-reweighting scheme is viable at this calibration. Per this project's own "no post-hoc threshold movement, no rescuing a failed hypothesis with a new metric" rule, this was not re-tuned or re-attempted with a different lambda within E34 itself — the finding was carried forward as the direct motivation for gradient-magnitude calibration becoming mandatory for every subsequent new loss term (E45 onward).

## Artifacts on disk

- `experiments/exp_e12_eggo_m/e34/calibrate_weights.py`, `calibrate_lambda_cw.py` (weight-table and lambda calibration, including the caught-and-fixed dataloader-caching bug)
- `experiments/exp_e12_eggo_m/e34/component_weighted_loss.py`, `test_component_weighted_loss.py`
- `experiments/exp_e12_eggo_m/e34/train_e34_component_weighted.py`, `run_e34_alpha_c_train_set.py`, `run_e34_asr.py`
- `experiments/exp_e12_eggo_m/e34/E34_weight_calibration_results.json`, `E34_lambda_cw_calibration.json`, `E34_weight_table_alpha_c.json`, `E34_weight_table_size.json`, `E34_alpha_c_train_table.json`
- `experiments/exp_e12_eggo_m/e34/alpha_c_train_log.txt`, `asr_train_log.txt` (full 30-epoch logs, both conditions)
- `experiments/exp_e12_eggo_m/e34/asr_runs/` (checkpoints, disk only, gitignored)
