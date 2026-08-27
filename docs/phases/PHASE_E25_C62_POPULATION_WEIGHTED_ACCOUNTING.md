# Phase E25, C6-2.8: Population-Weighted Benefit vs. Collateral Damage — A Real, Large Discrepancy That Needs Honest Treatment

**Status**: ✅ Complete, with an important unresolved discrepancy flagged rather than hidden. Using C6-2.6's real, measured transition probabilities (with tight confidence intervals) applied to the real population counts established in `PHASE_E25_C62_CORRECTED_REANALYSIS.md`'s Level 4 table, the population-weighted extrapolation predicts a **severe** net Dice collapse (0.86 → 0.64 on this experiment's own voxel-confusion-count basis) — far more damage than the actual training run shows (real validation Dice held at 0.9030, only modestly below A's 0.9063). **This is not necessarily evidence the collateral-damage hypothesis is wrong; it is evidence that a single-step, linear extrapolation of one realized 15-step update's effect across an entire epoch of training is not how the real training dynamics actually work**, and the discrepancy itself is reported as the main finding of this experiment, not resolved by adjusting the numbers to fit.

**Date**: 2026-08-12

---

## Method

Real population counts, aggregated across all 6 checkpoints × 4 batches (matching Level 4's own scope, `PHASE_E25_C62_CORRECTED_REANALYSIS.md`):

| Category | Count | Fraction |
|---|---:|---:|
| TP | 119,080 | 0.946% |
| FN | 21,440 | 0.170% |
| FP | 17,163 | 0.136% |
| TN | 12,425,229 | 98.747% |

Real transition probabilities, measured directly from C6-2.6's `p_after` field (not the earlier `transitioned` field, which was `None` for TP/TN by the original script's design — recomputed here directly from whether `p_after` crosses 0.5, for all four categories uniformly), with Jeffreys 95% confidence intervals:

| Category | Transition | k/n | p̂ | 95% CI |
|---|---|---:|---:|---:|
| FN | → TP (beneficial) | 3,923/5,602 | 70.03% | [68.82%, 71.22%] |
| FP | → TN (beneficial) | 1,632/6,822 | 23.92% | [22.92%, 24.95%] |
| TP | → FN (damaging) | 899/7,200 | 12.49% | [11.74%, 13.27%] |
| TN | → FP (damaging) | 57/7,200 | 0.79% | [0.61%, 1.02%] |

All four sample sizes are large (5,600–7,200), giving tight CIs — the *measurement* of each transition rate is precise, whatever its extrapolation validity turns out to be.

---

## The four scenarios

Applying these transition rates to the real population counts (net flux, both directions applied simultaneously in Scenario C):

| Scenario | TP' | FP' | FN' | Confusion-count Dice | Δ vs. baseline |
|---|---:|---:|---:|---:|---:|
| Baseline (pre-perturbation) | 119,080 | 17,163 | 21,440 | **0.8605** | — |
| **A: correction only** (FN→TP, FP→TN; damage off) | 134,094 | 13,057 | 6,426 | **0.9323** | +0.0718 |
| **B: damage only** (TP→FN, TN→FP; correction off) | 104,212 | 115,571 | 36,308 | **0.5785** | −0.2821 |
| **C: both, predicted net** | 119,226 | 111,465 | 21,294 | **0.6424** | −0.2182 |
| **D: actual C6-2 (real validation Dice)** | — | — | — | **0.9030** | (different basis, see below) |

**Net voxel flux** (Scenario C): FN→TP corrects 15,014 voxels; FP→TN corrects 4,106; TP→FN damages 14,868; **TN→FP damages 98,408** — nearly **24× larger** than TP's own damage flux, purely because TN's population (12.4M) is so enormous that even its small 0.79% flip rate produces an absolute voxel count that dwarfs the entire original FP population (17,163) by 5.7×.

---

## The discrepancy, stated plainly

**Scenario C predicts a catastrophic net Dice collapse (0.64) that did not happen.** The real, actual training run's validation Dice was 0.9030 — comfortably close to A's baseline (0.9063), not anywhere near the 0.22-point collapse this extrapolation predicts. Two things must both be true simultaneously, and reconciling them is the actual finding of this experiment, not a footnote:

1. **The per-voxel transition-rate measurements themselves are real and precise** — large samples, tight CIs, directly measured from C6-2.6's real forward passes on real checkpoints. There is no reason to doubt that *if* a single 15-step realized update were applied once, in isolation, and its effect were linearly extrapolated across the full population, the numbers above would follow arithmetically (the Dice formula itself is exact, not estimated).
2. **The real training run did not collapse this way.** Over 30 real epochs, actual Dice stayed close to baseline.

**The reconciliation is almost certainly that a single 15-step accumulated update's measured effect is not a valid proxy for "the cumulative effect of applying this mechanism repeatedly across an entire 30-epoch training run."** Real training does not apply one fixed `Δθ_margin` once and stop — it continuously re-estimates gradients against a constantly-changing parameter state, with `L_seg`'s own gradient present at every step (not isolated out, as H2's/C6-2.6's own measurement mechanism deliberately does to isolate the margin loss's effect), the optimizer's moment state evolving continuously rather than restarting from zero every 15 steps, and — most importantly — **the TN population that would newly flip to FP under one update's damage is not the same population that gets a chance to flip back** under `L_seg`'s own strong, continuously-present corrective pressure at the next step. This experiment's own methodology (isolating `L_margin`'s realized effect via a fresh, from-checkpoint 15-step replay) is well-suited to answering "what does the margin loss alone do, mechanistically, at one snapshot" — which is exactly what H1–H4, the mechanism audit, and C6-2.6 itself have been measuring throughout this whole phase — but it is not the right tool to predict the *cumulative*, *equilibrium* behavior of a full training trajectory where this update recurs every step alongside `L_seg`'s own continuous counter-pressure.

**This is not a failure of the extrapolation methodology in the sense of being wrong to attempt** — the user's own request to "determine whether the measured mechanism's beneficial corrections could quantitatively account for the Dice shortfall" is answered here, honestly: **no, not on a naive single-step linear extrapolation basis, because that basis overstates the damage by an amount large enough to be a genuine, reportable mismatch, not a rounding difference.** The correct interpretation is that the real training dynamics involve a continuous equilibrium between `L_margin`'s damage and `L_seg`'s own counter-corrective pressure that this snapshot-based extrapolation cannot capture, not that the underlying per-voxel measurements (FN 70% corrective, TN 0.79% damage rate, etc.) are themselves wrong.

---

## What this experiment does establish, despite the extrapolation mismatch

- **The relative scale finding is real and likely still informative even though the absolute Dice prediction is not trustworthy**: TN's damage flux is measured to be nearly 24× TP's, purely from population-size arithmetic on real, precisely-measured transition rates. This directional imbalance (TN's sheer size makes even a tiny per-voxel damage rate consequential) is a real property of this problem's class imbalance (98.75% TN), independent of whether the absolute extrapolated Dice number is trustworthy.
- **Scenario A (correction only) predicts a substantial real gain (+0.072)** if collateral damage could be eliminated entirely — this is a meaningful, quantified upper bound on what fixing the collateral-damage problem (not adding error weighting) could plausibly buy, on this experiment's own voxel-confusion-count Dice basis. It is directionally consistent with, though not the same number as, the volume-averaged validation Dice metric used elsewhere in this project (baseline note: this experiment's `0.8605`/`0.9323` figures use a pooled voxel-level confusion-count Dice from a stratified sample, not the volume-averaged validation-set Dice reported in H4 — the two are not on the same scale and should not be directly subtracted or compared as if they were).

---

## What this experiment does not establish

- **Not established**: whether collateral damage, properly accounted for at the true continuous-training equilibrium (not a single-step extrapolation), is sufficient, insufficient, or irrelevant to explaining the real 0.9030 vs. 0.9183 gap. The honest answer is that this specific methodology cannot resolve that question, and a different approach would be needed (e.g., directly tracking real TP/FN/FP/TN confusion counts across real training epochs from the already-saved checkpoints, which does not require this experiment's isolated-update extrapolation at all).
- **Not established**: whether the FP-specific puzzle from `PHASE_E25_C62_REPRESENTATION_TO_LOGIT.md` (flat transition rate despite growing corrective `Δlogit`) is related to or independent of the collateral-damage picture here.

---

## Files

| File | Purpose |
|---|---|
| `PHASE_E25_C62_REPRESENTATION_TO_LOGIT.md` | Source of the transition-rate measurements used here |
| `PHASE_E25_C62_CORRECTED_REANALYSIS.md` | Source of the real population counts used here |
| `experiments/exp_e12_eggo_m/e25/representation_to_logit_results/representation_to_logit_C62.json` | Raw per-voxel `p_after` values this accounting recomputed transition rates from |
