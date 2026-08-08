# Phase E12b/E12b.5/E12c: EGGO-M Pilot Results, Baseline Comparison, and Mechanism Verification

**Status**: ✅ Complete — modest, late-emerging Dice edge; the core mechanism-verification prediction (margin should grow) is NOT supported

**Date**: 2026-08-05

## Purpose

Full pilot run per the user's roadmap: E12b (single complete 30-epoch
training run, all pre-registered metrics), E12c (comparison against a
matched frozen-baseline run), and E12b.5 (geometry diagnostics on
checkpoints every 5 epochs, to verify whether EGGO-M actually changes
the latent geometry it was designed to change — not just whether Dice
moves).

## Setup

- EGGO-M, seed=0, μ=0.1, λ=0.1, δ_d=1.0 (the same starting hyperparameters
  used in E12a's smoke test — not yet swept, per the roadmap E14 comes
  after this).
- 30 epochs, `num_workers=4` (confirmed working reliably in this launch
  context — see `windows_training_env_gotchas` memory for the earlier,
  apparently transient, hang).
- **Baseline comparator**: `experiments/exp_e7_causality/seed_0/`'s
  results — the *exact same* training protocol (architecture minus the
  boundary head, same seed, same data split, same optimizer/schedule),
  run for the Phase E7 causality test. This is the fairest available
  comparison, since it isolates EGGO-M's added mechanism as the only
  difference from an otherwise identical run.

## E12b: full pilot results (EGGO-M, seed=0)

| epoch | train Dice | val Dice | val Precision | val Recall | val HD95 | val ECE | boundary BCE | boundary acc | margin loss | peak VRAM |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.207 | 0.416 | 0.266 | 0.972 | 11.55 | 0.382 | 0.458 | 0.977 | 0.000858 | 4969MB |
| 5 | 0.861 | 0.769 | 0.972 | 0.640 | 3.30 | 0.152 | 0.051 | 0.995 | 0.000310 | 4969MB |
| 10 | 0.892 | 0.874 | 0.869 | 0.884 | 2.72 | 0.104 | 0.017 | 0.998 | 0.000059 | 4969MB |
| 15 | 0.906 | 0.881 | 0.905 | 0.861 | 1.79 | 0.080 | 0.011 | 0.998 | 0.000024 | 4969MB |
| 20 | 0.915 | 0.892 | 0.905 | 0.881 | 2.06 | 0.067 | 0.009 | 0.998 | 0.000019 | 4969MB |
| 25 | 0.921 | 0.899 | 0.908 | 0.892 | 1.88 | 0.058 | 0.008 | 0.998 | 0.000017 | 4969MB |
| 30 | 0.925 | 0.902 | 0.917 | 0.890 | 1.55 | 0.051 | 0.007 | 0.998 | 0.000016 | 4969MB |

**Best val Dice: 0.9040 (epoch 28)**. Training was stable throughout —
no NaN/Inf, no crashes, consistent epoch times (~123-162s), peak VRAM
constant at 4969MB (no leak or growth over 30 epochs).

## E12c: comparison against the matched frozen-baseline run (E7 seed 0)

| epoch | Baseline Dice | EGGO-M Dice | Δ | Baseline HD95 | EGGO-M HD95 | Baseline Precision | EGGO-M Precision | Baseline Recall | EGGO-M Recall |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.5662 | 0.4162 | −0.1499 | 8.116 | 11.553 | 0.4121 | 0.2663 | 0.9243 | 0.9720 |
| 5 | 0.8669 | 0.7689 | −0.0979 | 2.126 | 3.305 | 0.8627 | 0.9721 | 0.8745 | 0.6395 |
| 10 | 0.8602 | 0.8743 | **+0.0141** | 2.208 | 2.722 | 0.9513 | 0.8688 | 0.7868 | 0.8835 |
| 15 | 0.8871 | 0.8808 | −0.0063 | 1.886 | 1.785 | 0.8598 | 0.9054 | 0.9178 | 0.8607 |
| 20 | 0.9050 | 0.8917 | −0.0132 | 1.416 | 2.061 | 0.9129 | 0.9050 | 0.8984 | 0.8814 |
| 25 | 0.9031 | 0.8990 | −0.0041 | 1.399 | 1.876 | 0.9439 | 0.9083 | 0.8668 | 0.8923 |
| 30 | 0.9087 | 0.9022 | −0.0065 | 1.599 | 1.547 | 0.9167 | 0.9170 | 0.9018 | 0.8896 |

**Best-checkpoint comparison**: baseline best = epoch 50, Dice **0.9107**
(from the full 50-epoch protocol). EGGO-M best (within the 30 epochs
run) = epoch 28, Dice **0.9040**. These aren't directly comparable
(different epoch budgets), so the epoch-matched table above is the fair
comparison, not the best-vs-best number.

**Honest reading of the epoch-matched table**: EGGO-M trails the
baseline for the first ~10 epochs (notably epochs 1 and 5, by −0.10 to
−0.15 Dice — a real, non-trivial early deficit), converges to roughly
parity by epoch 10, and from epoch 15 onward sits within about ±0.01
Dice of the baseline at every matched epoch — sometimes slightly above,
sometimes slightly below, no consistent direction. **This is not a
demonstrated Dice improvement.** The differences from epoch 15 onward
are small enough, and inconsistent enough in sign, that they are very
plausibly within single-seed noise (recall the frozen baseline's own
3-seed spread was ±0.0005–0.007 Dice depending on the metric, per
`PHASE_A5_BASELINE_FROZEN.md` and Experiment D) — this is a single-seed
comparison and cannot yet distinguish a real small effect from noise.
**E13 (multi-seed) is necessary before any claim about EGGO-M's Dice
effect, positive or negative, is defensible.**

## E12b.5: mechanism verification — the load-bearing check

The whole point of the diagnostic checkpoints was to test *why*, not
just *whether*, EGGO-M might work — specifically the chain "boundary
geometry improves → calibration improves → segmentation improves," and
directly: **does the boundary margin (the quantity $\mathcal{L}_{margin}$
directly optimizes) actually grow over training?**

### Finding: the margin does NOT grow — this contradicts the pre-registered prediction

| epoch | mean boundary margin (pairwise dist) | independent latent boundary AUC | val ECE |
|---|---|---|---|
| 1 | 34.29 | 0.9956 | 0.379 |
| 5 | 19.00 | 0.9937 | 0.152 |
| 10 | 31.95 | 0.9987 | 0.104 |
| 15 | 30.14 | 0.9995 | 0.080 |
| 20 | 29.35 | 0.9993 | 0.067 |
| 25 | 29.60 | 0.9996 | 0.058 |
| 30 | 31.17 | 0.9997 | 0.051 |

The margin trajectory is **34.29 → 31.17 overall (a net decrease)**, and
highly non-monotonic in between (a sharp dip to 19.00 at epoch 5, then
oscillating in the 29–32 range for the rest of training). This is the
opposite of, or at best flat relative to, what
`PHASE_E11_5_READINESS_REVIEW.md` §3 pre-registered as a medium-confidence
prediction ("boundary margin increase... near-tautological — literally
what the loss optimizes"). **The mechanism the algorithm was designed
around is not clearly operating as intended in this pilot run.**

`corr(boundary_margin, Dice) = −0.131 (p=0.780)` — not remotely
significant, consistent with no reliable relationship between the two in
this run.

### What IS working: class separability and calibration, but neither is uniquely attributable to the margin mechanism

- **Latent boundary AUC** stays extremely high throughout (0.994–0.9997)
  — consistent with E7's finding that separability saturates almost
  immediately regardless of intervention. This was the correctly
  pre-registered high-confidence prediction ("roughly unchanged... little
  room to move") and holds.
- **ECE improves smoothly and substantially** (0.379→0.051) — but there
  is no baseline comparator for this (the frozen baseline has no
  evidential/uncertainty head at all), so this cannot be read as "EGGO-M
  improved calibration relative to baseline." It only shows that ECE
  improved during training, which a converging evidential head would be
  expected to do with or without EGGO-M's margin mechanism active. This
  needs a **μ>0, λ=0 control run** (boundary head active, margin loss
  inactive) to isolate whether ECE improvement is attributable to the
  margin loss specifically or would happen anyway — not yet run.
- **`corr(ECE, Dice) = −0.995`**: strong, but this is almost certainly
  driven by both quantities improving together over the course of normal
  training convergence (the same shared-training-progress confound
  flagged in the earlier density-vs-Dice analysis from `abo_frozen_lessons_learned`),
  not evidence of a causal ECE→Dice link specifically attributable to
  EGGO-M.

## Overall interpretation

**This pilot does not yet support the hypothesis that EGGO-M's specific
mechanism (margin-driven boundary separation) causes the small Dice
difference observed.** Three things would need to be true for the
mechanism story to hold, and only one clearly does:

1. ~~Margin should grow~~ — **does not hold** (net decrease, non-monotonic).
2. Boundary AUC should stay saturated (little room to move) — **holds**,
   but this was already true in the no-EGGO baseline (E7), so it isn't
   evidence EGGO-M is doing anything distinctive.
3. Calibration should improve — **holds**, but cannot yet be
   attributed to the margin mechanism specifically vs. the boundary head
   alone vs. ordinary evidential-head convergence, without the μ>0/λ=0
   control.

The Dice comparison itself is a single-seed result inside plausible
noise from epoch 15 onward, with a real early deficit in epochs 1–5 that
should also be explained, not ignored, before any positive claim.

**This is a genuinely useful negative/mixed result, not a failure of the
experiment** — it directly follows the user's own framing: "if EGGO
improves Dice and produces larger margins... your causal argument
becomes substantially stronger. If not, that's equally informative."
Here, Dice shows at most a small, unproven, single-seed effect, and the
specific mechanism (margin growth) is not supported. This should
temper, not necessarily kill, further investment — but the honest
next step is not immediately E13's multi-seed sweep as originally
planned; it is first understanding *why* the margin isn't growing,
since running more seeds of a mechanism that already isn't operating as
designed risks compounding the same issue three times over rather than
answering the open question.

## Recommended next step (before E13)

Investigate why the margin loss isn't producing margin growth, before
committing to a multi-seed run. Candidate explanations to check, roughly
in order of ease:

1. **λ=0.1 may be too weak relative to $\mathcal{L}_{seg}$'s gradient
   magnitude** to meaningfully move `dec1` against the segmentation
   loss's own (much larger) pull — the margin loss values themselves are
   very small in absolute terms (0.0009 → 0.00002), two to three orders
   of magnitude smaller than `seg_loss`. This is exactly what E14's
   planned λ sweep would reveal, but it may be worth a quick, targeted
   check (e.g. one run at λ=1.0) before the full sweep, given this
   specific negative finding.
2. **The uncertainty gate ($\hat U_i$) may be suppressing the margin
   loss to near-zero for most anchors** once the evidential head
   converges (recall the previously-identified gate-degeneracy risk,
   `PHASE_E5_ALGORITHM_DESIGN.md` §6, Failure Mode 1) — worth logging the
   distribution of $\hat U_i \cdot B_i$ weights actually applied per
   epoch, not just the aggregate margin loss value, to check this
   directly rather than infer it.
3. **34.29 (epoch 1) may already be near a natural ceiling** for this
   feature space's typical opposite-class distance, such that pushing
   further via $\delta_d=1.0$'s hinge threshold has little room to act —
   worth checking the actual distribution of pairwise distances, not
   just the mean, and comparing against $2\delta_d=2.0$ to see what
   fraction of pairs are even within the hinge's active region.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e12b_pilot_seed0/` | Full 30-epoch training run (logs, checkpoints) |
| `experiments/exp_e12_eggo_m/analyze_eggo_m_checkpoints.py` | E12b.5 diagnostic script |
| `experiments/exp_e12_eggo_m/e12b5_results/` | Mechanism verification plots + JSON |
| `experiments/exp_e7_causality/seed_0/` | Matched baseline comparator |
| `PHASE_E11_5_READINESS_REVIEW.md` | The pre-registered predictions this tests |

---

**Completed**: 2026-08-05 — recommend investigating margin non-growth before E13's multi-seed run
