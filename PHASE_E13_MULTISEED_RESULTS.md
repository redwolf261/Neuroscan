# Phase E13: Multi-Seed Confirmation — Scenario 1 (Null Confirmed)

**Status**: ✅ Complete — the E12f single-seed null replicates across 4 seeds; EGGO-M does not improve Dice over baseline, and at the matched epoch (30) is slightly but statistically distinguishably *worse* (paired mean Δ = −0.0037, p=0.032, n=4)

**Date**: 2026-08-06

## Purpose

Per the user's explicit instruction after E12f's Outcome B result: run seeds
1, 2, 3 with **identical** hyperparameters to E12f (δ_d=3.6659, adaptive
τ_b, λ=0.1, μ=0.1, 30 epochs) — no tuning, no redesign — purely to
determine whether the single-seed null is real or an artifact. Combined
with seed 0 (already run as E12f), this gives 4 seeds total.

## Result 1: all 4 seeds land in a tight cluster, none exceeds baseline

| Seed | Best Val Dice | Best epoch | HD95 | ECE | Precision | Recall |
|---|---|---|---|---|---|---|
| 0 | 0.9063 | 23 | 1.671 | 0.0529 | 0.9114 | 0.9022 |
| 1 | 0.9064 | 29 | 1.374 | 0.0464 | 0.9165 | 0.8978 |
| 2 | 0.9028 | 28 | 1.679 | 0.0439 | 0.9224 | 0.8859 |
| 3 | 0.9020 | 28 | 1.646 | 0.0457 | 0.9354 | 0.8726 |

**Cross-seed mean ± std:**

| Metric | Mean ± std |
|---|---|
| Best Val Dice | **0.9044 ± 0.0023** |
| HD95 | 1.592 ± 0.146 |
| ECE | 0.0472 ± 0.0039 |
| Precision | 0.9214 ± 0.0103 |
| Recall | 0.8896 ± 0.0133 |

The spread across seeds is tight (std=0.0023 on Dice, comparable to the
frozen baseline's own 3-seed spread of ~0.0006–0.001 scaled up slightly) —
this is a well-behaved, reproducible result, not a noisy or unstable one.
That reproducibility is exactly what makes it a confident null rather
than an ambiguous one.

## Result 2: at the matched epoch (30), EGGO-M is consistently — and statistically distinguishably — behind baseline

The valid baseline comparator is `exp_e7_causality/seed_0/results.json`,
a **true 50-epoch run** with a correctly-scheduled cosine LR (`T_max=50`
matches its actual length), read at its own epoch-30 checkpoint for a
matched-epoch comparison (see "Correction to E13 code audit" below for
why this matters).

| Seed | EGGO-M @ epoch 30 | Baseline @ epoch 30 (E7) | Δ |
|---|---|---|---|
| 0 | 0.9050 | 0.9087 | −0.0037 |
| 1 | 0.9077 | 0.9087 | −0.0010 |
| 2 | 0.9032 | 0.9087 | −0.0055 |
| 3 | 0.9041 | 0.9087 | −0.0047 |

Mean Δ = **−0.0037 ± 0.0020**. One-sample t-test against 0: **t=−3.80,
p=0.032** (n=4). This is a small effect (well within what most would call
"clinically negligible"), but it is consistently negative across all 4
seeds and *not* noise-indistinguishable from zero at conventional
significance — the honest reading is "EGGO-M is very slightly worse than
baseline at matched epoch 30, reproducibly," not merely "no effect
detected."

The best-Dice-vs-best-Dice comparison tells the same story: EGGO-M's
4-seed best-Dice mean (0.9044) is 0.0043 below the baseline's own
epoch-30 value (0.9087) and 0.0063 below the baseline's true best (0.9107
at epoch 50, not a matched-epoch comparison but included for completeness).

## Result 3: the margin mechanism is active in all 4 seeds, but its relationship to Dice is itself seed-dependent — not just "zero"

Extending the E12f mechanism-verification analysis
(`analyze_eggo_m_checkpoints_v2.py`) to all 4 seeds
(`e13_mechanism_all_seeds.py`) gives a materially more informative
picture than E12f's single-seed "corr ≈ 0" finding:

| Seed | corr(margin, Dice) | p |
|---|---|---|
| 0 | −0.0016 | 0.997 |
| 1 | **+0.9580** | 0.001 |
| 2 | −0.5973 | 0.157 |
| 3 | +0.3521 | 0.439 |

Mean correlation across seeds: **+0.18** (highly seed-dependent, ranging
from a strong positive relationship in seed 1 to a moderate negative one
in seed 2). This is an important refinement of E12f's conclusion: it is
**not** that margin and Dice have *no* relationship — in some seeds
(notably seed 1) they track together quite strongly. But this
relationship is not stable enough across seeds to constitute a reliable
mechanism, and — critically — **even the seed with the strongest positive
margin-Dice correlation (seed 1, r=+0.958) still does not end up beating
baseline** (Δ=−0.0010 at epoch 30). Whatever local within-run relationship
exists between margin growth and Dice improvement, it does not translate
into an absolute improvement over a model that never optimizes margin at
all. This rules out a specific residual hope from E12f — that a "lucky"
seed might reveal the mechanism actually working — cleanly.

All 4 seeds show the same qualitative activity signature as E12f: active
hinge % declines from ~1-12% at epoch 1 to a nonzero plateau (0.12-0.22%)
by epoch 30, never collapsing to E12b's 0.00%; τ_b tracks smoothly upward
(0.18-0.77 at epoch 1 → 5.1-6.1 at epoch 30); margin loss stays
substantial late in training (0.0008-0.0015) rather than decaying toward
E12b's ~0.000016. **The mechanism-is-active finding from E12f fully
replicates.**

Final-epoch (30) independent latent-boundary AUC across seeds: mean
0.9996 ± 0.0001 — saturated in every seed, exactly as predicted since
E1.3, confirming this ceiling is not sensitive to seed variation either.

## Scenario determination

Per the user's own pre-specified 3-scenario framework:

> - **Scenario 1** (confirms null): all 4 seeds cluster near baseline,
>   EGGO-M does not improve segmentation
> - **Scenario 2**: one seed improves significantly — investigate
>   optimization variance/dataset sensitivity
> - **Scenario 3**: all seeds improve — EGGO survives, proceed further

**This is Scenario 1.** All 4 seeds cluster tightly (std=0.0023), none
exceeds the matched-epoch baseline, and the direction of the (small)
difference is consistently unfavorable to EGGO-M. No seed shows anything
resembling Scenario 2's "one seed improves significantly" — the seed-1
outlier is only in the margin-Dice *correlation* (Result 3), not in the
Dice outcome itself, and its final Dice is statistically the same as the
other three.

## Correction to the E13 code audit's Finding 3

`PHASE_E13_CODE_AUDIT.md` originally characterized the `T_max=50` vs.
`--epochs 30` LR-schedule mismatch as **"equally unfair to both"** arms of
the comparison. Re-checking the baseline source directly
(`exp_e7_causality/seed_0/results.json`) while assembling this report
found that characterization was **incorrect**: `results.json` records
`"final_epoch": 50` with 50 full validation epochs logged — this is a
**true 50-epoch run with a correctly-scheduled cosine anneal**
(`T_max=50` matches its actual length exactly). The "baseline @ epoch 30"
values used throughout E12b/E12f/E13's matched-epoch comparisons are this
50-epoch run's trajectory *read at* epoch 30 — a point mid-schedule where
its LR is still relatively high, but not because the run itself stopped
early or had a mismatched `T_max`.

**Corrected framing**: EGGO-M's 30-epoch runs (E7's exact script/config
aside) all actually stop with a badly-annealed LR (T_max=50, training
ends at epoch 30, LR≈0.000139 vs a correctly-scheduled ≈0.000002). The
baseline comparator never has this problem — it runs the full 50 epochs
its schedule is built for. This is a **real asymmetry**, not a
shared-equally condition. Practically, this means:

1. EGGO-M's 30-epoch endpoint numbers are being measured at a
   disadvantage relative to what a matched, correctly-annealed 30-epoch
   EGGO-M schedule would show — the −0.0037 mean deficit could plausibly
   shrink (or, less likely, grow) under a properly-scheduled comparison.
2. This does **not** invalidate Scenario 1's determination on its own —
   the deficit is small and the burden would be on a fixed-schedule rerun
   to show it closes, not on this analysis to assume it does. But it is
   a legitimate caveat any writeup must state plainly, and a
   properly-scheduled confirmation (fresh `T_max` matching actual epoch
   count, both arms) would strengthen the negative result considerably
   if it replicates, or partially undercut it if EGGO-M closes the gap.
3. **Recommended**: if a further, final confirmation run is undertaken
   before writeup, fix the schedule (`T_max` = actual `--epochs` value)
   for both EGGO-M and any freshly-run baseline seed — this is a cheap,
   high-value correction given how close the current margin is (−0.0037,
   right at the edge of "possibly a scheduling artifact").

## What this does and does not settle

**Settled**: the mechanism (margin loss actively participating in
optimization) is real and reproduces across seeds — this is not an
artifact of E12e's calibration fix being seed-0-specific. The hypothesis
that widening the latent margin, as currently formulated, improves
segmentation Dice is **not supported** — reproducibly, across 4 seeds,
with a small but consistent unfavorable direction rather than a
statistical tie.

**Not settled**:
- The `T_max` scheduling asymmetry above — a properly-scheduled rerun
  is the cleanest way to remove this caveat entirely, and is now the
  highest-value remaining code-level fix given how small the observed
  gap is.
- `μ`/`λ` were still never swept (deliberately, per the user's "don't
  tune" instruction for this phase) — remains theoretically possible a
  different weighting changes the outcome, though E12d's gradient-norm
  analysis suggested this is secondary to the (now-resolved)
  mechanism-activity issue.
- The `μ>0, λ=0` isolation control (does the ECE improvement come from
  the margin loss specifically, or the boundary/evidential heads alone)
  is still unrun — still the most promising unexplored angle raised in
  the Q1-publication-strategy discussion, independent of this result.

## Recommendation

This is now a defensible, mechanism-verified, multi-seed-confirmed
negative result for the margin-widening hypothesis as implemented. Two
honest paths forward, unchanged from what was flagged after E12f, now
with stronger evidence behind option 1:

1. **Write up E8→E13 as the primary negative-result contribution** —
   the arc from hypothesis (E1-E7 diagnostics) → design (E8-E11.5) →
   implementation with full gradient/loss audit (E12a) → mechanism
   failure and root-cause diagnosis (E12d) → calibration fix and
   verified reactivation (E12e-f) → 4-seed statistical confirmation
   (E13) is a genuinely rigorous negative-result narrative, which is a
   legitimate (if harder-to-place) scientific contribution.
2. **Pivot to the calibration/ECE angle** as the primary positive
   finding — EGGO-M's ECE improves substantially and consistently
   (E12f: 0.379→0.046; all 4 E13 seeds land in 0.043-0.053) even though
   Dice does not — but this requires the still-unrun `μ>0,λ=0` isolation
   control to know whether that improvement is attributable to the
   margin loss or just the added boundary/evidential supervision.

Not decided here — remains a call for the user, informed now by
confirmed statistics rather than a single-seed result.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e13_pilot_calibrated_seed{1,2,3}/` | New E13 training runs (seed 0 reuses E12f's) |
| `experiments/exp_e12_eggo_m/e13_mechanism_all_seeds.py` | Cross-seed mechanism verification (margin, AUC, correlation) |
| `experiments/exp_e12_eggo_m/e13_mechanism_results_all_seeds/` | Trajectory JSON, per-seed correlations |
| `PHASE_E12F_RECALIBRATED_PILOT_RESULTS.md` | Seed-0 single-seed result this confirms |
| `PHASE_E13_CODE_AUDIT.md` | Code audit; Finding 3 corrected by this document |
| `experiments/exp_e7_causality/seed_0/results.json` | Baseline comparator (true 50-epoch run) |

---

**Completed**: 2026-08-06 — Scenario 1 confirmed: EGGO-M's margin-widening
mechanism is real and reproducible but does not improve segmentation
Dice; at matched epoch 30 it is slightly, consistently, and statistically
distinguishably worse than baseline (mean Δ=−0.0037, p=0.032, n=4),
though a T_max scheduling asymmetry favoring the baseline (newly found
while writing this report) means a properly-scheduled rerun is the
recommended final check before treating the gap as fully settled.
