# Phase E12e: Hyperparameter Calibration

**Status**: ✅ Complete — mechanism verified active. A real methodological bug was found and fixed mid-phase, not hidden.

**Date**: 2026-08-05

## Purpose

Per the user's explicit framing: **not** to optimize Dice, only to
verify the margin loss mechanism is actually active before any λ sweep
(E14). `PHASE_E12D_MECHANISM_FAILURE_INVESTIGATION.md` found two
compounding miscalibrations — `δ_d=1.0` mismatched to the feature
space's actual scale, and a fixed `τ_b` that couldn't track the boundary
head's growing confidence during live training. This phase recalibrates
both from measured data and verifies the fix with a short (5-epoch)
run before considering any further sweep or multi-seed work.

## δ_d: measured from a fresh, randomly-initialized model — with a real bug along the way

**First attempt (wrong)**: measured pairwise distances on a freshly
initialized `UNet3D_v2` in `model.eval()` mode, targeting `2×δ_d` at the
20th percentile of the distance distribution (midpoint of the requested
10–30% active-hinge range). Got `δ_d = 0.0148`. **Verified live and it
failed**: `active_hinge_pct` was exactly 0.0% throughout a 5-epoch
verification run, not the targeted ~20%.

**Root cause, found by direct comparison**: `Conv3DBlock` uses
`BatchNorm3d`, which behaves completely differently in `eval()` vs.
`train()` mode. `eval()` mode uses running statistics accumulated over
many batches — for a *freshly initialized* model, these running stats
don't yet reflect anything meaningful (no batches have been seen). The
real training loop always runs in `train()` mode. Direct A/B check on
the identical seed and batch: `dec1` standard deviation in `eval()` mode
= 0.0089; in `train()` mode = 0.6095 — a **~68× difference** at true
initialization. My calibration script measured an regime that never
actually occurs in real training (training starts already in `train()`
mode from the very first optimizer step).

**Second, corrected attempt**: re-measured with `fresh_model.train()`.
Got a very different, much larger `δ_d = 3.6659` (vs. the flawed first
attempt's 0.0148 — **~247× larger**, and vs. the E12b pilot's original,
uncalibrated `δ_d=1.0` — ~3.7× larger). Distances at true initialization
turn out to be in the 5–10 range, not 0.02–0.04.

**Important scope check — this bug did NOT invalidate E12d's earlier
findings.** E12d's mechanism-diagnosis dashboard and E12b.5's mechanism
verification both used **checkpoints from epoch 1 onward** (never a
truly fresh, zero-batches-seen model), and a direct comparison showed
the eval/train gap shrinks to under 10% by epoch 1 of real training and
under 3% by epoch 10 — nowhere near the ~68× gap at true initialization.
E12d's conclusions (margin doesn't grow, hinge rarely active, `B_i`
collapses) stand unaffected. The bug was isolated to this new
calibration script's fresh-init measurement specifically.

## τ_b: confirmed a single fixed value cannot work; implemented as a live EMA

Measured the boundary head's own `|d_i|` (logit magnitude) distribution
across the existing E12b checkpoints (epochs 1–30) — this measurement
was unaffected by the eval/train bug above, since it always used
trained checkpoints, not a fresh model:

| Epoch | Median \|d_i\| | Implied τ_b (for B_i≈0.5 at median) |
|---|---|---|
| 1 | 0.613 | 0.884 |
| 5 | 3.020 | 4.358 |
| 10 | 4.568 | 6.592 |
| 15 | 5.555 | 8.016 |
| 20 | 6.262 | 9.036 |
| 25 | 6.809 | 9.826 |
| 30 | 7.221 | 10.420 |

The implied τ_b grows **~11.8× from epoch 1 to epoch 30** — confirming
directly that no single fixed constant could keep `B_i`'s dynamic range
meaningful throughout training. Implemented `EMATauB`, an exponential
moving average (decay=0.98) of the batch median `|d_i|`, converted to
τ_b via `τ_b = median / ln(2)` each batch (the closed-form value that
keeps `B_i ≈ 0.5` for a "typical" voxel) — with a 20-step warmup
(returning the original static value until enough samples accumulate,
mirroring ABO's own `delta_warmup_steps` cold-start handling).

## Live diagnostic logging added

Per the user's explicit request to monitor the mechanism live, not just
post-hoc: `compute_margin_loss` now returns `active_hinge_pct` per batch
(fraction of sampled pairs with `dist < 2δ_d`), logged to the training
CSV and printed in the epoch summary alongside the live `τ_b` value —
this is the direct, per-epoch answer to "does the mechanism stay
active," not something that requires a separate post-hoc analysis pass
to check.

## Verification: 5-epoch run with both fixes active

| Epoch | Train Dice | Margin loss | Active hinge % | τ_b (end of epoch) | Val Dice |
|---|---|---|---|---|---|
| 1 | 0.198 | 0.02596 | 3.33% | 1.272 | 0.326 |
| 2 | 0.520 | 0.00934 | 1.12% | 1.159 | 0.747 |
| 3 | 0.734 | 0.00922 | 0.99% | 1.403 | 0.759 |
| 4 | 0.833 | 0.00920 | 0.97% | 2.042 | 0.680 |
| 5 | 0.859 | 0.00713 | 0.70% | 2.545 | 0.781 |

**Direct comparison to E12b's original (broken) run**: margin loss stays
in the 0.007–0.026 range throughout, versus E12b's decay to ~0.000016 by
epoch 30 (roughly **500–1800× larger** at comparable points in training).
Active hinge percentage stays comfortably nonzero (0.70–3.33%) across
all 5 epochs, versus E12b's exact 0.00% by epoch 25. τ_b correctly
grows over training (1.27→2.55), tracking the boundary head's increasing
confidence rather than staying fixed and causing `B_i` to collapse.

**This is the qualitative shift needed before E14/E13 can be meaningful**:
the mechanism the algorithm was designed around is now demonstrably
active, not inert. Training remains stable (no NaN, sane epoch times
~135–153s, consistent peak VRAM at 4969MB — identical to E12b's
resource profile, confirming the fix didn't introduce new overhead).

## What this does NOT yet answer

- Whether an *active* margin mechanism actually improves Dice/ECE beyond
  what an inactive one failed to show — this 5-epoch run is too short
  and wasn't designed to test outcome metrics, only mechanism activity.
- Whether `active_hinge_pct` declining over these 5 epochs (3.33%→0.70%)
  will continue declining toward zero again over a longer run (the same
  failure pattern as E12b, just delayed) or stabilize — only a longer
  pilot can tell.
- The δ_d value (3.6659) still targets the *initial* distance
  distribution; as training progresses and the model's geometry changes,
  this may itself need to become adaptive (mirroring τ_b's fix) rather
  than a second fixed constant — not yet addressed, flagged as a
  possible next issue if active% does trend back toward zero in a longer
  run.

## Recommended next step

Run a full pilot (matching E12b's 30-epoch protocol) with both fixes
active, and re-apply the same E12b.5/E12d diagnostic pipeline
(mechanism verification dashboard, gradient norms) to confirm the
mechanism stays active for the full duration, not just these first 5
epochs — before considering E13 (multi-seed) or E14 (λ sweep).

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e12e_calibrate_constants.py` | δ_d/τ_b calibration script (corrected) |
| `experiments/exp_e12_eggo_m/e12e_results/calibration_results.json` | Corrected calibration values |
| `experiments/exp_e12_eggo_m/train_eggo_m.py` | Updated with `DELTA_D_CALIBRATED`, `EMATauB`, live `active_hinge_pct` logging |
| `experiments/exp_e12_eggo_m/e12e_verify_calibrated_v2/` | 5-epoch verification run (successful) |
| `PHASE_E12D_MECHANISM_FAILURE_INVESTIGATION.md` | The investigation this phase responds to |

---

**Completed**: 2026-08-05 — mechanism verified active; recommend a full pilot re-run before E13/E14
