# Phase E12f: Full 30-Epoch Pilot with Recalibrated Constants — Outcome B

**Status**: ✅ Complete — mechanism confirmed active throughout 30 epochs; Dice does not improve; margin does not show sustained growth or correlate with Dice

**Date**: 2026-08-05

## Purpose

`PHASE_E12E_HYPERPARAMETER_CALIBRATION.md` fixed two miscalibrated
constants (`δ_d`, `τ_b`) and verified the margin mechanism was active
over 5 epochs. This phase runs the full 30-epoch protocol (matching
E12b's original pilot exactly, only the two constants changed) and
re-applies the complete E12b.5/E12d diagnostic suite to determine which
of the user's three pre-specified outcomes materialized.

## Setup

Identical to E12b except `δ_d = 3.6659` (recalibrated, was 1.0) and
`τ_b` now live/adaptive via `EMATauB` (was a fixed 0.9713). Same seed
(0), same μ=0.1, λ=0.1, same 30 epochs, same `num_workers=4`.

## Result 1: the mechanism stays active for the full 30 epochs — this is genuinely new, not seen in E12b

| Epoch | Active hinge % | τ_b (end of epoch) | Margin loss |
|---|---|---|---|
| 1 | 3.33% | 1.272 | 0.02596 |
| 5 | 0.70% | 2.545 | 0.00713 |
| 10 | 0.40% | 3.611 | 0.00384 |
| 15 | 0.28% | 4.291 | 0.00239 |
| 20 | 0.18% | 4.604 | 0.00131 |
| 25 | 0.13% | 4.824 | 0.00092 |
| 30 | 0.12% | 5.001 | 0.00079 |

Active hinge percentage declines gradually but **plateaus around
0.12–0.18% from epoch 20 onward — it does not collapse to exactly 0.00%
the way E12b did by epoch 25**. Margin loss similarly stabilizes around
0.0008–0.0013 in late training rather than decaying toward E12b's
~0.000016 (roughly **50–80× larger** at the same late-training point).
τ_b correctly and smoothly tracks the boundary head's growing confidence
throughout (1.27→5.00), never getting stuck. **This directly confirms
E12e's calibration fix worked as intended, sustained over the full
protocol, not just the initial 5 epochs.**

## Result 2: Dice does not improve over the matched baseline

| Epoch | EGGO-M Dice | Baseline Dice (E7, no EGGO) | Δ |
|---|---|---|---|
| 1 | 0.4215 | 0.5662 | −0.1447 |
| 5 | 0.7590 | 0.8669 | −0.1079 |
| 10 | 0.8614 | 0.8602 | +0.0013 |
| 15 | 0.8933 | 0.8871 | +0.0062 |
| 20 | 0.8658 | 0.9050 | **−0.0391** |
| 25 | 0.8991 | 0.9031 | −0.0040 |
| 30 | 0.9050 | 0.9087 | −0.0037 |

**Best Val Dice for this run: 0.9063** (a modest improvement over E12b's
0.9040, and closer to the frozen baseline's 0.9107±0.0005), but the
epoch-matched trajectory tells a less favorable story than that single
number: EGGO-M trails baseline for most of training, including a notable
dip at epoch 20 (−0.039), and only approaches parity (not exceeds it) by
epochs 25–30. **This is not a demonstrated improvement** — if anything,
it's slightly worse than the already-inconclusive E12b comparison at
several matched points, though both remain within plausible single-seed
noise range established by the baseline's own 3-seed spread.

## Result 3: boundary margin does not show sustained growth, and has zero correlation with Dice

Margin trajectory: 27.83 → 14.16 (epoch 5 dip) → 24.12 → 26.74 → 25.73 →
26.43 → 28.04. Net change over 30 epochs: **+0.8%** — essentially flat,
technically "increasing" only because the endpoint happens to be
marginally above the start, not because of any sustained trend. The
same non-monotonic dip-then-partial-recovery shape seen in E12b persists,
just rescaled by the new `δ_d`.

`corr(boundary_margin, Dice) = −0.0016, p=0.997` — statistically
indistinguishable from zero, closely matching E12b's own null result
(`−0.131, p=0.780`). **The margin mechanism being genuinely active
(Result 1) did not translate into the margin actually growing in a
Dice-correlated way (Result 3).** These are two different, separable
findings, and only the first one was fixed by E12e's calibration.

## Result 4: latent boundary AUC still saturates almost immediately, as expected

0.9948 → 0.9960 → 0.9963 → 0.9992 → 0.9994 → 0.9990 → 0.9990 — consistent
with every prior phase (E1.3, E7, E12b.5) and the correctly pre-registered
high-confidence prediction that this metric has little room to move
regardless of intervention.

## Overall verdict: Outcome B

Per the user's own decision tree:

> **Outcome B (still valuable)**: Active hinge remains healthy. Margin
> grows [— partially, not cleanly]. Dice does **not** improve. This is
> still an important scientific result. It suggests that increasing
> latent margins is not sufficient to improve segmentation in this
> architecture.

This is the closest match, with one honest caveat: margin growth is
weaker and more ambiguous than a clean "Outcome B" would have it (net
+0.8%, non-monotonic, zero correlation with Dice) — this sits between
Outcome B and Outcome C, leaning toward B because the mechanism is
unambiguously *active* (Result 1) even though it isn't *effectively
growing the margin in a way that matters* (Result 3).

**The combined reading across E12b, E12d, E12e, and E12f**: the
mechanism failure diagnosed in E12d (inactive hinge, collapsed gate) has
been genuinely fixed — this is a real, verified engineering success. But
fixing the mechanism did not rehabilitate the hypothesis. With the
margin loss now demonstrably participating in optimization throughout
training, Dice still does not improve over the matched baseline, and the
margin itself does not show the kind of sustained, Dice-correlated
growth the original design predicted. This is now a **legitimate,
mechanism-verified negative result** — the earlier E12b null result
could have been dismissed as "we don't know if this ever got a fair
test." That objection no longer holds.

## What this does not yet rule out

- Only one seed. Per the established discipline throughout this project,
  a single-seed null (or near-null) result is not yet a confident
  negative — E13 (multi-seed) is the natural next step now that the
  mechanism-activity confound has been removed, unlike after E12b where
  running more seeds would have been premature.
- `μ` (boundary head weight) and `λ` (margin weight) were never swept —
  it remains possible that a different weighting balance produces a
  different outcome, though the gradient-norm analysis in E12d suggested
  this is a secondary concern relative to the mechanism-activity issue
  now resolved.
- The `ECE` improvement (0.379→0.046) still lacks a controlled comparison
  isolating whether it comes from the margin mechanism specifically or
  would occur from the boundary head/evidential head alone — the
  `μ>0, λ=0` control run flagged as needed since E12b has still not been
  run.

## Recommended next step

Given Outcome B (mechanism active, hypothesis not supported), the
user's own framework suggests this is a legitimate stopping or
pivoting point, not an automatic escalation to E13. Two reasonable
paths:
1. **E13 (multi-seed)**, now legitimately motivated since the mechanism
   confound is resolved — confirms whether this single-seed null holds
   across seeds before writing up the negative result formally.
2. **Treat this as sufficient evidence** that latent margin widening, as
   currently formulated, is not the right lever for this architecture —
   write up the full E8→E12f arc as a rigorous negative result (a
   legitimate, well-evidenced scientific contribution per the project's
   own stated standards) rather than continuing to invest further
   compute in this specific mechanism.

Not decided in this document — a call for the user, not something to
default into.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e12f_pilot_calibrated_seed0/` | Full 30-epoch training run (logs, checkpoints) |
| `experiments/exp_e12_eggo_m/analyze_eggo_m_checkpoints_v2.py` | Mechanism verification script (points at the new pilot) |
| `experiments/exp_e12_eggo_m/e12f_mechanism_results/` | Plots + trajectory JSON |
| `PHASE_E12E_HYPERPARAMETER_CALIBRATION.md` | The calibration fix this pilot verifies |
| `PHASE_E12B_PILOT_RESULTS.md`, `PHASE_E12D_MECHANISM_FAILURE_INVESTIGATION.md` | The original pilot and diagnosis this directly follows up on |

---

**Completed**: 2026-08-05 — Outcome B: mechanism verified active, hypothesis not supported by this single-seed run
