# Phase E53 — ASR with Curriculum Warmup: Design

## Motivation, directly from E52's own finding

E52 (gradient-calibrated re-attempt of E34's `[R]`/alpha_c component-reweighting) fixed the 43× gradient-magnitude blowup that likely caused E34's original collapse, but **still failed** at the smoke-test gate: val Dice 0.10 at epoch 1 (vs. 0.41–0.51 for every other healthy post-pivot condition at the same point), the same "predict everything as tumor" degenerate pattern E34's own `[R]` condition showed. E52's own analysis isolated the cause more precisely than E34 alone could: `seg_loss` and `boundary_loss` values at epoch 1 were nearly identical to healthy conditions, but Dice was far lower — meaning the component-weighted loss term's **direction** (not magnitude) was actively working against the model at a point in training when it has not yet learned to segment easy, large lesions reliably. E52's disclosed (not proven) explanation: upweighting the hardest, most-degraded components from epoch 0 fights the implicit coarse-to-fine curriculum every other successful mechanism in this project (D4, D8, CCABA, attention gate) implicitly relies on, by never supervising basic large-lesion competence before disproportionately punishing errors on the hardest small/degraded ones.

**E53's idea, directly motivated by that diagnosis**: keep E34/E52's mechanism and gradient-matched calibration exactly as-is, but **warm it in gradually** rather than applying it at full strength from epoch 0 — let the model establish baseline large-lesion competence first (matching every other successful condition's own implicit behavior), then introduce the component-reweighting signal once that competence exists.

## Literature positioning (Gate D)

A 2025–2026 literature scan found curriculum-learning approaches to loss weighting exist (radiomics-guided curriculum initialization, hardness-weighted losses, progressive patch-size curricula) and the closest static prior art for the underlying reweighting formula itself is **Universal Loss Reweighting** (Shirokikh et al., MICCAI 2020 — `weight = N / ((K+1) * component_size)`, inverse-size weighting, no time-varying schedule), which is the same prior art E33's original novelty audit already identified and distinguished from `alpha_c` (resize-survival-based, not raw-size-based). **No source found combines a time-varying warmup ramp with a resize-survival-based (not raw-size-based) component reweighting signal specifically** — this is a genuinely narrow, disclosed gap, not a claim of first-ever curriculum-weighted loss.

## Mechanism

Reuses, unchanged: `component_weighted_loss.py` (E34's own tested module), `E34_weight_table_alpha_c.json` (the `[R]`/alpha_c weight table), `E52_lambda_cw_calibration.json`'s gradient-matched `lambda_cw=0.1101` (the magnitude is already correctly calibrated per E52's own finding — only the *schedule* is new).

**New**: a fixed, pre-declared linear ramp multiplier, applied to `lambda_cw` at each epoch:

```
multiplier(epoch) = clip((epoch - E_START) / (E_END - E_START), 0, 1)
lambda_cw_effective(epoch) = LAMBDA_CW * multiplier(epoch)
```

`E_START=8`, `E_END=20` (0-indexed epochs). **Chosen from real data, not tuned on this phase's own results**: inspected E51/CCAG's own healthy training-Dice trajectory (a representative post-pivot condition) before picking these numbers — train Dice is still volatile through epoch 3 (0.22→0.75→0.85) and stabilizes into a smooth, high-90s climb by epoch 8 onward (0.89 and climbing steadily). `E_START=8` keeps the component-weighted term fully off through the entire volatile early-training window (including E52's own smoke-tested epochs 1–2, where the collapse was observed) and `E_END=20` leaves a full 10 epochs (of the standard 30-epoch schedule) at full calibrated strength before training ends, matching the general shape of every other auxiliary term in this project (D4/D8/CCABA are active from epoch 1, but their own value contribution is naturally small early since the model's predictions are themselves poor and uninformative at that point — this mechanism differs specifically because component *reweighting*, unlike an added prediction head, actively redirects gradient toward specific already-hard voxels from the first step it's active, which is exactly the interaction E52 isolated as the problem).

## Pre-declared gates (same discipline as every prior phase)

- **Smoke-test gate** (2 epochs): since the ramp multiplier is 0 for epochs 1–2 by construction, this smoke test is expected to reproduce **exactly** D4-only/v3's own baseline behavior at epochs 1–2 (the component-weighted term contributes nothing until epoch 8) — this is itself a verification the ramp is wired correctly, not a test of the new mechanism's own effect (which cannot appear before epoch 8). A **10-epoch** smoke test is used instead, to observe the ramp's actual onset (epoch 8) and confirm training doesn't collapse as the term switches on, before committing to the full 30-epoch schedule.
- **Kill condition**: val Dice < 0.5 after epoch 5 (unchanged), PLUS a new check specific to this phase — val Dice must not drop by more than 10 relative percentage points in the 3 epochs immediately following `E_START` (i.e., the ramp-onset window) compared to its pre-ramp trajectory, to catch a delayed version of E52's own collapse pattern without waiting for a full kill-threshold breach.
- **Success criterion**: mean ≥1.0pp over canonical baseline (≥0.9163) across 3 seeds, 95% CI excluding D4-only (0.9096) — same bar as every post-E49 condition.

## What this phase would tell us either way

- **If the ramp avoids the collapse and produces a real gain**: confirms E52's own curriculum-interaction diagnosis was correct and actionable, and reopens component-reweighting (previously twice-killed) as a viable direction — a genuinely new result building directly on this project's own two most recent findings.
- **If the ramp avoids the collapse but the result is flat-to-modest**: consistent with the now-familiar +0.3–0.5pp band; closes component-reweighting definitively (three attempts: E34 value-matched, E52 gradient-matched, E53 gradient-matched+curriculum, spanning the calibration and scheduling axes both).
- **If the collapse recurs even with the ramp** (at or after epoch 8): would indicate the mechanism's failure is not a pure early-training curriculum-timing issue either, ruling out both hypotheses tested so far (magnitude in E52, timing in E53) and leaving the reweighting-by-degradation idea family without a clear remaining fix to try.
