# Phase E25, C6-2: SC-TAM Results (H1 → H2 → H3 → H4)

**Status**: ✅ Complete — full unit-test → calibration → smoke-test → training → H1 → H2 → H3 → H4 sequence run to completion for C6-2 (Signed Class-Conditional Task-Aligned Margin, unweighted core variant). **C6-2 produces the strongest, most correctly-signed mechanistic evidence of any EGGO-M margin candidate tested to date (H1, H2, and H3's Q5), but does not clear the project's hard 1.2-percentage-point Dice bar, and its best Dice (0.9030) is below both the unmodified baseline A (0.9063) and the random-projection control E (0.9058).** H3 additionally surfaces a genuine, unresolved discrepancy between the *realized, AdamW-accumulated parameter update* (H2: strongly correctly-signed) and the *immediate loss gradient* at a real training step size (Q3/Q4: near/below chance, net-unfavorable FP/FN transitions) — reported here as an open finding, not smoothed over.

**Date**: 2026-08-10

---

## Executive result

**Does SC-TAM fix the margin mechanism the prior EGGO-M/E24 candidates (A/B/E) could not?** Mechanistically, largely yes — SC-TAM is the first candidate whose local loss-descent direction correlates with Dice improvement in the *correct* sign, and whose *realized* parameter-induced representation displacement carries the predicted class-conditional signed structure at every one of 24 measured checkpoint-batches. But this does not translate into a segmentation-outcome win: best validation Dice (0.9030, epoch 29, seed 0) falls 1.53 percentage points short of the required 0.9183 bar, and is itself below the untouched baseline. H3's decisive concentration finding (Q5: SC-TAM's real displacement is 9.4x larger on originally-misclassified voxels than on correct ones, 48/48 records, p=7.1e-15) is a genuinely strong, favorable signal — but sits alongside a net-unfavorable FP/FN transition count at the immediate-gradient step size, an internal tension this report does not attempt to resolve prematurely.

Per the project's own locked decision rule (`PHASE_E25_CANDIDATE6_SC_TAM_DESIGN.md` Section 11): a candidate that fixes real mechanism (H1/H2) but misses the Dice bar (H4) is evidence for investigating error/uncertainty or boundary weighting (C6-3/C6-4) as a *hypothesis-driven*, not arbitrary, next step. That decision is not made in this report.

---

## Pipeline summary (pre-registered, all gates passed before training)

1. **Unit tests** (`experiments/exp_e12_eggo_m/e25/test_sc_tam.py`): 6/6 passing, including `test_orientation_invariance` — corrected after an initial test-construction flaw (see Implementation Note below) — which directly proves loss/gradient sign and magnitude are invariant to array position for the identical physical voxel pair, and `test_sc_tam_sign_correctness`, which confirms tumor-voxel gradients project negatively onto `w_hat` and background-voxel gradients project positively, on synthetic data.
2. **`m_ij` calibration** (`e25_calibrate_m_ij.py`, Section 12 methodology — fresh-init model, `.train()` mode, same 10–30%-active-hinge-at-init criterion as E12e/E24): `m_ij = 0.3089`, giving 20.04% active-hinge rate at init. Cross-checked against E24's `delta_d_w` distribution (identical p50 unsigned scale, as expected — same underlying `dec1`/`w_hat` construction). Noted: 85.62% of FB pairs already have positive sign at fresh init (higher than a 50/50 null), carried forward as context for interpreting H1/H2, not treated as a calibration defect.
3. **Gate 5 smoke test** (`e25_smoke_train.py`): both baseline and SC-TAM paths pass all 8 criteria carried over from E24's Gate 5, plus a new criterion 9 — tumor/background gradient sign correctness on **real batches**, which held at **exactly 100%** on every one of 6 batches (a hard mechanistic guarantee, not a statistical tendency). Orthogonality to `w_hat` at numerical-precision levels (~1e-7), matching E24's task-aligned result.
4. **Initial-parameter-equality gate** (`run_c62.py`): C6-2's fresh model hash-verified identical to condition A's own fresh instance (`theta_A^(0) == theta_C62^(0)`, SHA-256 exact match), optimizer metadata identical — any downstream Dice/mechanism difference is attributable to the loss term alone.
5. **Training** (`train_c62.py`, `run_c62.py`): 30 epochs, seed 0, `mu=0.1`, `lambda_margin=0.1`, same `configs/brats.yaml`, same checkpoint schedule as Gate 6. ~127–159s/epoch throughout (no `num_workers` regression). No NaN/Inf, no crashes.

---

## Implementation note: a test-construction flaw found and fixed before trusting the unit suite

The initial `test_orientation_invariance` reordered voxels *within* a larger synthetic batch under an identical `torch.manual_seed()` and compared losses, expecting near-identical values. It failed (~1–5% relative discrepancy, not shrinking as `max_negatives` increased from 500→5000). Diagnosis: reordering voxels changes which *physical* voxel a given array *index* refers to; `torch.randint`'s negative-sampling draw is index-based, so an identical seed after reordering samples the same index positions but those positions now hold *different* underlying embedding vectors — the "same seed" comparison was silently comparing genuinely different sampled pairs, not the same pairs restated. This was a flaw in the test's own premise, not a bug in `compute_margin_loss`. Confirmed by ruling out the alternative explanation directly: reordering does not change class-pool composition or size (`tumor_local`/`bg_local` index sets verified identical in both cases). Fixed by rewriting the test around a fully deterministic, single-pair 2-voxel case (no stochastic sampling involved), which directly isolates "does array position affect the computed sign for the identical physical pair" — confirmed: loss bit-identical (3.527390 == 3.527390) and gradient sign/magnitude on each voxel invariant to array position, regardless of which index the tumor/background voxel occupies.

A second, smaller implementation-adjacent fix: extending `run_counterfactual.py`'s `ObjectiveConfig` with a 4th mode (`"sc_tam"`) required adding an explicit `margin_mode` property and threading it through all 4 `compute_margin_loss` call sites — because `"task_aligned"` and `"sc_tam"` both pass a non-`None` `w_hat` and are otherwise indistinguishable to `compute_margin_loss`'s own auto-inference (`margin_mode=None` → `"task_aligned"` whenever `w_hat is not None`). Since `run_counterfactual.py` is the E22-regression-locked reference implementation, this change was re-verified against the regression gate after editing — **PASS, 0.00e+00 max diff on all fields, 1536/1536 rows, unchanged headline correlation** — confirming the addition is a true no-op for the three pre-existing modes.

---

## H1: mechanistic (loss-level local descent direction vs. Dice)

Exact counterfactual rerun (`run_h1_c62.py`, reusing `run_counterfactual()` unchanged) on C6-2's own 6 checkpoints (epochs 5/10/15/20/25/30) × 8 validation subjects × full epsilon sweep, scored under C6-2's own objective (never cross-evaluated against A/B/E's metrics). 1536 rows, matching A/B/E's own H1 scope exactly.

**`rho_C62 = corr(ΔL_margin^SC, ΔDice)` for the `B_margin` direction = −0.4924 (Spearman), p = 4.06×10⁻¹³, n = 192.**

This is the *correct* sign — decreasing SC-TAM's own margin loss correlates with Dice *increasing*, the relationship the whole EGGO-M/E24 arc has been trying and failing to establish. Reported alongside the historical (non-threshold) reference values from `PHASE_E24_GATE6`:

| Condition | rho (B_margin direction) | p | Sign |
|---|---:|---:|---|
| A (baseline Euclidean) | +0.2432 | 6.75×10⁻⁴ | wrong |
| B (task-aligned, unsigned) | +0.0286 | 0.694 | wrong, n.s. |
| E (random-projection control) | +0.1008 | 0.164 | wrong, n.s. |
| **C6-2 (SC-TAM)** | **−0.4924** | **4.06×10⁻¹³** | **correct** |

The Pearson coefficient agrees in sign and significance (`r = −0.3465, p = 8.54×10⁻⁷`), ruling out a rank-only artifact.

---

## H2: representation (realized parameter-induced displacement, class-conditional)

Reuses E21's exact perturb-and-forward mechanism (`ShadowAdam`, 15 real-batch replay steps, accumulated `Δθ_margin` applied out-of-place to a deep-copied model, re-forward-passed on the same batch to get the realized `Δz`) — same 6 checkpoints, same `N_TRANSPORT_BATCHES=4` as B/E's own H2. **Adapted with the one substantive change the design doc requires**: class-conditional, signed reporting (`run_h2_c62.py`), since SC-TAM's defining mechanistic claim is directional per class (tumor → `-w_hat`, background → `+w_hat`), unlike B/E's class-agnostic axis-alignment question.

**Result: 24/24 checkpoint-batches show the correct mean sign for both classes.**

| Epoch | Tumor cos_mean (expect <0) | Tumor frac_correct_sign | Background cos_mean (expect >0) | Background frac_correct_sign |
|---|---:|---:|---:|---:|
| 5 | −0.11 (range −0.23 to −0.002) | 0.63–0.92 | +0.40 | 0.91–0.93 |
| 10 | −0.33 | 0.86–0.96 | +0.37 | 0.87–0.89 |
| 15 | −0.34 | 0.93–0.95 | +0.31 | 0.81–0.91 |
| 20 | −0.31 | 0.80–0.92 | +0.17 | 0.62–0.76 |
| 25 | −0.35 | 0.89–0.93 | +0.27 | 0.82–0.85 |
| 30 | −0.37 | 0.92–0.97 | +0.16 | 0.57–0.71 |

Honestly reported, not smoothed over: **background's `frac_correct_sign` degrades over training** (~0.91 at epoch 5 down to ~0.57–0.71 by epoch 30) while **tumor's stays high or improves**. This is a partial echo of the original Q3 failure-mode finding that motivated SC-TAM (background was the weaker-calibrated class) — the *mean* sign stays correct for background throughout, but less uniformly across voxels as training proceeds.

---

## H3: error-correction mechanism / failure-analysis machinery

Reuses `run_failure_analysis.py`'s exact Q2/Q3/Q4/Q5 pipeline (`run_h3_c62.py`), same 6 checkpoints × 8 subjects, same `epsilon=0.25` (H1's smallest tested step, representing "C6-2's real local update"). 48 records, matching A/B/E's own scope.

**One required correction, made explicit rather than silently inherited**: E24's original Q3 sign convention (`tumor correct = positive projection onto w_hat`) was written for the *unsigned* task-aligned case. SC-TAM's own design predicts the *opposite* for its real displacement (verified directly by H2 above: tumor → `-w_hat`). This script flips Q3's convention to match SC-TAM's own prediction, consistent with what H2 already measured — not a new, unverified assumption.

### Q5 (the decisive question): does geometric movement concentrate on originally-misclassified voxels?

**Yes, strongly.** `mean(‖Δz‖ | misclassified) = 2.781`, `mean(‖Δz‖ | correct) = 0.295` — a **9.43x** pooled ratio (9.79x mean / 8.39x median per-record). **48/48 records** show misclassified > correct (Wilcoxon signed-rank, p = 7.11×10⁻¹⁵). SC-TAM's real displacement is genuinely, overwhelmingly concentrated where correction could actually help Dice, not on already-confident-correct voxels where it would be functionally inert.

### Q3/Q4: a real, unresolved discrepancy with H2

At the **immediate loss-gradient direction** (not the realized 15-step AdamW-accumulated update H2 measured), sign correctness is **not** favorable:

- Tumor: mean `frac_correct_sign = 0.390` (below chance; one-sample t-test vs. 0.5: t=−2.19, p=0.034).
- Background: mean `frac_correct_sign = 0.452` (below chance, not significant: t=−1.90, p=0.063).
- Per-epoch breakdown shows this is not a transient effect — tumor sign-correctness hovers near or below 0.5 at every checkpoint (0.015 at epoch 5, 0.39–0.56 at epochs 10–30), never approaching H2's realized-update levels.

**Q4 (TP/TN/FP/FN transitions) is net unfavorable**: 14,443 favorable transitions (FP→TN: 925, FN→TP: 13,518) vs. **38,283 unfavorable** (TP→FN: 437, TN→FP: **37,846**, the dominant term) — a 0.377x favorable/unfavorable ratio. The bulk of the unfavorable shift is background pixels flipping to false positives.

**This is a genuine tension, not a contradiction, and is reported as an open finding**: H2 measures the *realized, AdamW-accumulated* parameter update over 15 real training steps; Q3/Q4 measure the *immediate, single-step gradient direction* at a fixed small step size. The project's own established distinction (E19/E20's original motivation — loss-level gradient constraint vs. realized parameter-space movement are not the same thing) applies here with unusual force: the two measurements of "does SC-TAM's update move things the right way" disagree in direction. This report does not adjudicate which one is more decision-relevant for Dice; both are real, verified measurements of different quantities, and the disagreement between them is itself the most interesting open question this phase raises.

### Q2 (boundary vs. interior)

`boundary_delta_prob_mean = +0.205`, `interior_delta_prob_mean = +0.002` — consistent with all three of A/B/E's own Q2 findings (movement concentrates at boundary voxels, not interior), with SC-TAM's boundary effect ~100x larger than its interior effect.

---

## H4: segmentation outcome vs. the hard bar

| Condition | Best val Dice | Δ vs. A |
|---|---:|---:|
| A (baseline, locked reference) | 0.9063 | — |
| B (task-aligned, E24, failed) | 0.9017 | −0.46pp |
| E (random-projection control) | 0.9058 | −0.05pp |
| **Required bar (A + 1.2pp)** | **0.9183** | **+1.20pp** |
| **C6-2 (SC-TAM)** | **0.9030** (epoch 29) | **−0.33pp** |

**C6-2 does not clear the hard bar.** Shortfall: 1.53 percentage points. C6-2's best Dice is also below A's own untouched baseline and below E's random-projection control. Final-epoch (30) Dice is lower still (0.8995), suggesting the best checkpoint (epoch 29) may not reflect a stable late-training optimum — not further investigated in this report. Single seed (0) only; per the project's own locked rule, a *strong* claim in either direction would require a multi-seed replication, which this report does not attempt.

---

## Summary table: H1 → H4 across all tested conditions

| Condition | H1 rho (sign) | H2 (realized alignment) | H3 Q5 (concentration ratio) | H4 best Dice | Clears 1.2pp bar? |
|---|---|---|---|---:|---|
| A (baseline) | +0.24 (wrong) | n/a (no fixed axis) | not computed for A in this phase | 0.9063 | reference |
| B (task-aligned) | +0.03 (wrong, n.s.) | favorable but weaker than C6-2 | not the focus of this report | 0.9017 | No |
| E (random control) | +0.10 (wrong, n.s.) | — | — | 0.9058 | No |
| **C6-2 (SC-TAM)** | **−0.49 (correct)** | **24/24 correct mean sign, per-class** | **9.4x, 48/48 records, p=7e-15** | **0.9030** | **No** |

---

## What this report does not decide

Per `PHASE_E25_CANDIDATE6_SC_TAM_DESIGN.md` Section 11's locked decision rule: C6-2 shows real mechanism-level improvement (H1, H2, H3's Q5) without clearing H4's outcome bar, which is the specific evidence pattern the design doc identifies as grounds for a *hypothesis-driven* (not arbitrary) investigation of error/uncertainty weighting (C6-3) or boundary weighting (C6-4) — explicitly **not started here**, gated on this report being reviewed first. The H2-vs-Q3/Q4 discrepancy (realized update vs. immediate gradient) is left as an open, unresolved finding rather than a conclusion either direction.

---

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e25/test_sc_tam.py` | 6/6 unit tests, including the corrected `test_orientation_invariance` |
| `experiments/exp_e12_eggo_m/e25/e25_calibrate_m_ij.py` | `m_ij` calibration (0.3089) |
| `experiments/exp_e12_eggo_m/e25/e25_smoke_train.py` | Gate 5 smoke test, both paths pass all 9 criteria |
| `experiments/exp_e12_eggo_m/e25/train_c62.py`, `run_c62.py` | Training class + orchestrator, initial-parameter-equality gate vs. condition A |
| `experiments/exp_e12_eggo_m/e25/c62_runs/C62_sc_tam_seed0/` | 30-epoch training run, checkpoints, `epoch_metrics.csv` |
| `experiments/exp_e12_eggo_m/e24/run_counterfactual.py` | Extended with `ObjectiveConfig(mode="sc_tam")` + explicit `margin_mode` property (regression-verified no-op for A/B/E) |
| `experiments/exp_e12_eggo_m/e25/run_h1_c62.py` | H1 counterfactual rerun, `h1_results/h1_results_C62.json` (1536 rows) |
| `experiments/exp_e12_eggo_m/e25/run_h2_c62.py` | H2 class-conditional signed alignment, `h2_results/h2_results_C62.json` (24 records) |
| `experiments/exp_e12_eggo_m/e25/run_h3_c62.py` | H3/failure analysis, `failure_analysis_results/failure_analysis_C62.json` (48 records) |
| `PHASE_E25_CANDIDATE6_SC_TAM_DESIGN.md` | The locked C6-2 design and decision rule this report evaluates against |
