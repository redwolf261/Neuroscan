# Phase E25, C6-2.8 (real-trajectory version): Confusion Dynamics Across Actual Checkpoints — Outcome A, With an Important Measurement-Scale Caveat

**Status**: ✅ Complete. Measures the real, cumulative TP/FN/FP/TN transition dynamics directly from consecutive real saved checkpoints of both condition A (baseline) and C6-2 (SC-TAM) — no perturbation, no gradient replay, no extrapolation. **This is Outcome A from the user's own three-way framework: `L_seg`'s continuous corrective pressure prevents the collateral damage predicted by the one-shot extrapolation from compounding. In real, cumulative late-training (epochs 10→30), both A and C6-2 show comparable, healthy positive net flux (+670 and +638 voxels respectively out of ~9,000 total transitions) — collateral damage does not dominate or cancel benefit in either condition.** A real, honestly-flagged discrepancy: on this experiment's own 8-subject pooled-voxel Dice basis, C6-2 finishes marginally *above* A (0.9235 vs. 0.9211) — the opposite direction from H4's own headline full-validation-set result (C6-2 0.9030 vs. A 0.9063) — attributed to a genuine measurement-scale difference (8-subject voxel-pooled vs. full-125-subject volume-averaged Dice), not a contradiction of H4, and explicitly not resolved by this report.

**Date**: 2026-08-12

---

## Method

For both condition A and C6-2, at every pair of consecutive real saved checkpoints sharing the same schedule (1→5, 5→10, 10→15, 15→20, 20→25, 25→30), both real trained models were loaded and run in `.eval()` mode (matching how Dice is actually scored, not the `.train()` mode used throughout the mechanism-diagnostic phases) on the same fixed 8 validation subjects (matching H1's own established scope). Each voxel's TP/FN/FP/TN state was computed independently at checkpoint `t` and `t+1` from real predictions vs. ground truth, and accumulated into a real 4×4 transition matrix per checkpoint pair, aggregated across all 8 subjects — no perturbation, no synthetic update, no single-step extrapolation. This directly sidesteps the problem `PHASE_E25_C62_POPULATION_WEIGHTED_ACCOUNTING.md` flagged: that one-shot `L_margin`-only dynamics cannot be treated as a linear proxy for the cumulative, real `L_seg+L_margin` training trajectory.

---

## Full trajectory, both conditions

| Epoch | A Dice (8-subject pooled) | C6-2 Dice (8-subject pooled) |
|---|---:|---:|
| 1 | 0.390 | 0.147 |
| 5 | 0.721 | 0.693 |
| 10 | 0.905 | 0.912 |
| 15 | 0.899 | 0.913 |
| 20 | 0.908 | 0.920 |
| 25 | 0.919 | 0.923 |
| 30 | **0.921** | **0.924** |

Both conditions follow a normal, healthy training curve — rapid early gains (epoch 1→10), a small dip or plateau around epoch 15 for A, then steady late-training improvement for both. Neither shows the collapse the one-shot extrapolation predicted.

---

## Late-training (epochs 10→30) transition accounting

| Condition | FN→TP (benefit) | FP→TN (benefit) | TP→FN (damage) | TN→FP (damage) | **Net flux** | Dice change |
|---|---:|---:|---:|---:|---:|---:|
| A (baseline) | 3,400 | 1,441 | 2,418 | 1,753 | **+670** | +0.0162 |
| C6-2 (SC-TAM) | 1,902 | 2,798 | 1,850 | 2,212 | **+638** | +0.0118 |

**Both conditions show a healthy positive net flux of comparable magnitude** — benefit consistently and substantially exceeds damage in both A and C6-2, at roughly the same overall scale (net +670 vs. +638). Collateral damage is real (nonzero in both) but does not dominate or approach cancellation in either condition. This directly answers the user's Outcome A/B/C question: **Outcome A** — `L_seg`'s continuous presence prevents the one-shot extrapolation's predicted damage from compounding into a real, observed degradation.

**A genuinely interesting compositional difference, reported without over-interpreting it**: A's benefit is dominated by FN→TP corrections (3,400 vs. 1,441 FP→TN), while C6-2's benefit leans more toward FP→TN (2,798 vs. 1,902 FN→TP) — a real, measured difference in *how* each condition achieves its comparable net-positive flux, though this experiment does not establish why, and it is not obviously connected to the FP-specific transition-rate puzzle from `PHASE_E25_C62_REPRESENTATION_TO_LOGIT.md` (that experiment measured one-shot transition *probability per voxel*, not real cumulative transition *counts* — the two are not directly comparable without further work, not attempted here).

---

## The discrepancy that must be flagged, not resolved here

**On this experiment's own 8-subject, voxel-pooled Dice measure, C6-2 finishes marginally above A at epoch 30 (0.9235 vs. 0.9211, +0.0024).** This is the opposite direction from H4's own headline, already-reported result (`PHASE_E25_C62_RESULTS.md`): C6-2's best validation Dice (0.9030) is *below* A's (0.9063), on the full 125-subject validation set using per-volume Dice averaged across subjects — a different aggregation method entirely (pooling every voxel across only 8 subjects into one confusion matrix, as this experiment does, is not the same statistic as averaging 125 independently-computed per-volume Dice scores, and the two can legitimately disagree, especially with an 8-subject sample small enough that a handful of easy or hard cases can shift the pooled number without being representative of the full 125-subject average).

**This report does not attempt to resolve which number is "more correct"** — H4's full-validation-set, per-volume-averaged Dice remains the project's own established, locked outcome metric (the one the 0.9183 hard bar is defined against), and this experiment's 8-subject pooled-voxel figure was never intended as a competing Dice measurement; it exists here only to support the transition-matrix accounting, which is the actual point of this experiment. The discrepancy is reported honestly as a real, unresolved methodological gap between the two measures, not smoothed over or used to argue against H4's own established result.

---

## Synthesis: Outcome A, with the real question now narrowed further

Per the user's own three-way framework:

- **Outcome A** (`L_seg` cancels the one-shot-predicted collateral damage; FN/FP steadily decrease; Dice improves but plateaus) — **this is what the data shows.** Both conditions show healthy positive net flux across late training, with no sign of the catastrophic TN→FP compounding the one-shot extrapolation predicted.
- **Outcome B** (damage survives, cancels benefit) — **not supported.** Damage is present but consistently smaller than benefit in both conditions.
- **Outcome C** (neither population effect explains the plateau; look elsewhere spatially) — **not directly supported either**, since a real, comparable-to-baseline positive flux is observed, not a wash.

**The real, standing finding**: SC-TAM's real, cumulative training produces a healthy net-positive confusion-matrix flux, comparable in scale to the baseline's own — **collateral damage, real as it is in the one-shot mechanism (C6-2.6, `PHASE_E25_C62_REPRESENTATION_TO_LOGIT.md`), is not the explanation for C6-2's Dice shortfall against the 1.2pp bar.** Per Outcome A's own framing, the remaining, now-sharpened question is: **SC-TAM's corrective mechanism is real, correctly-directed, and not being cancelled by collateral damage — so why does it not produce the additional ~1.5 percentage points of Dice needed to clear the bar, when A's own baseline mechanism (pure `L_seg`) already achieves comparable net flux without any margin term at all?** This reframes the investigation away from "is SC-TAM broken or self-cancelling" (it is not) toward "is SC-TAM's correction simply redundant with, rather than additive to, what `L_seg` already accomplishes on its own" — a genuinely different question, not yet investigated.

---

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e25/run_c62_real_trajectory_confusion.py` | Full implementation |
| `experiments/exp_e12_eggo_m/e25/real_trajectory_confusion_results/real_trajectory_confusion_C62_vs_A.json` | Raw transition matrices, both conditions, all 6 checkpoint pairs |
| `PHASE_E25_C62_POPULATION_WEIGHTED_ACCOUNTING.md` | The one-shot extrapolation whose predicted collapse this experiment shows did not occur in real training |
| `PHASE_E25_C62_RESULTS.md` | H4's own established, locked full-validation-set Dice figures this report's 8-subject figures should not be confused with |
