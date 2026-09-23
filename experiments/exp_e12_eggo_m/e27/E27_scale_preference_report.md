# E27: Component-Specific Decoder-Scale Preference Audit

**Status**: Complete. No training performed. Inference-only, on already-trained checkpoints (A, D4-only, D2-only, Both), full 125-subject validation set (development-contaminated — see Section 8).

**Date**: 2026-08-13

---

## Executive conclusion

Component-level scale preference (S_c) exists only for a small minority (8.5%) of the 363 GT lesion components in the validation set; the other 91.5% show no meaningful difference between D4 and D2 supervision at all. Among the 31 components that *do* show a clear preference, the split is almost exactly even — 48.4% D4-preferred vs. 51.6% D2-preferred, 95% CI [0.30, 0.67], statistically indistinguishable from a coin flip (binomial p=1.00 vs. p=0.5) — meaning even Gate A's weakest form ("both directions occur in non-trivial proportion") is technically satisfied, but there is no dominant, exploitable D4-vs-D2 signal to route on. Baseline difficulty and component size explain essentially none of this variance (R²=0.008, all confounder correlations non-significant). Critically, early-training trajectory features (epochs ≤10 of 30) **fail to predict the final preference direction at all** — the cross-validated logistic model scores AUROC=0.43 (worse than a coin flip) and underperforms every trivial baseline, including "always guess." **Gate B fails decisively.** Combined with Gate C evidence that "Both" ties with max(D4,D2) for 91.5% of components and shows no consistent incremental value, the verdict is unambiguous:

```text
NO-GO — insufficient evidence for Adaptive Decoder Credit Assignment.
```

---

## 1. Data and matching protocol

Full 125-subject validation set, 363 GT connected components (`scipy.ndimage.label`, 6-connectivity). Matching rule (identical across all four models and all checkpoint epochs, documented in `run_e27_collect_component_data.py`): for each GT component, union all predicted connected components that overlap any of its voxels into one matched-prediction region; compare that region to the GT component. A GT component is "missed" iff no predicted component overlaps it at all (matched region is empty). This is a many-predicted-to-one-GT rule, well-defined under both fragmentation (one lesion split into several predicted pieces) and under-segmentation. Missed-component convention: dice=0.0, iou=0.0, coverage=0.0, precision=undefined (`None`, not defaulted to 0 or 1, since no predicted region exists to compute precision over).

Best-checkpoint inference was run once per model (A/D4/D2/Both) per subject; periodic-checkpoint inference (epochs 1, 5, 10, 15, 20, 25, 30 — every epoch checkpoint that exists on disk for these four training runs) was run for the early-predictability analysis, 4 × 7 × 125 = 3,500 additional forward passes, 10,164 total trajectory records.

## 2. Component-level scale preference

Raw, untruncated distribution of $S_c$ = (component Dice gain, D4 vs A) − (component Dice gain, D2 vs A), n=363:

| Stat | Value |
|---|---:|
| Mean | 0.0003 |
| Median | 0.0000 |
| Std | 0.1512 |
| IQR | 0.00006 |
| Min | −1.000 |
| Max | 1.000 |
| p5 / p25 / p50 / p75 / p95 | −0.038 / −0.00003 / 0.000 / 0.00003 / 0.030 |

The distribution is extremely concentrated at zero (median exactly 0, IQR essentially 0) with a small number of large-magnitude outliers driving the mean/std/percentile tails — a classic "mostly nothing happens, occasionally something dramatic happens" shape, not a broad, exploitable continuum.

**Epsilon derivation (Section 6 requirement)**: rather than pick an arbitrary threshold, I estimated a noise floor from the median within-subject standard deviation of $S_c$ among the 66 subjects with ≥2 components (0.0075) — a real, measured dispersion. This value is far too small to use directly as a categorical threshold given the distribution's near-zero IQR (it would flag almost every component as "non-neutral" on pure noise), so I used a rounded, explicitly-labeled **exploratory** threshold of ε=0.05 (roughly the p95/|p5| magnitude) for the categorical breakdown below — six-and-a-half times the measured noise floor, a conservative choice that undercounts rather than overcounts "preference."

## 3. D4 vs D2 heterogeneity

Preference categories (ε=0.05, exploratory):

| Subset | n | D4-preferred | D2-preferred | Neutral |
|---|---:|---:|---:|---:|
| All components | 363 | 15 (4.1%) | 16 (4.4%) | 332 (91.5%) |
| Detected by all four | 160 | 8 (5.0%) | 8 (5.0%) | 144 (90.0%) |
| 1–50 vox | 223 | 11 (4.9%) | 10 (4.5%) | 202 (90.6%) |
| 50–150 vox | 8 | 0 (0%) | 5 (62.5%) | 3 (37.5%) |
| 150–400 vox | 6 | 0 (0%) | 0 (0%) | 6 (100%) |
| 400–1000 vox | 15 | 2 (13.3%) | 1 (6.7%) | 12 (80%) |
| >1000 vox | 111 | 2 (1.8%) | 0 (0%) | 109 (98.2%) |

(Per instruction, these size bins are descriptive only, not evidence of novelty — the 50–150 bin's 62.5% D2-preferred figure is n=8, not a reliable signal.)

**Heterogeneity test**: among the 31 non-neutral components, D4-preferred=15 (48.4%), D2-preferred=16 (51.6%). Binomial test against p=0.5 (the "D4 always wins" degenerate null): p=1.00 — indistinguishable from a coin flip. 95% CI on the D4 fraction: [0.302, 0.669]. **Subject-clustered bootstrap** (2000 resamples, resampling subjects not components, respecting non-independence within subject): mean D4-fraction=0.487, 95% CI [0.300, 0.658] — the same conclusion survives clustering.

So: heterogeneity in direction genuinely exists (not "D4 always wins"), but the *base rate* of any preference at all is only 8.5% of components, and the direction, when it occurs, is close to a coin flip rather than a strong, structured signal.

## 4. Baseline-difficulty and size confounding

Raw Spearman correlations with $S_c$ (dice basis):

| Feature | ρ | p |
|---|---:|---:|
| Baseline (A) component Dice | +0.016 | 0.758 |
| Baseline (A) coverage | −0.005 | 0.931 |
| GT component size | +0.023 | 0.659 |

All non-significant, essentially zero. Following the E26 lesson explicitly (do not linearly control a nonlinearly-related size variable), the partial correlation of baseline Dice vs. $S_c$ controlled for `size, size^(-1/3), log(size)` simultaneously: partial r=−0.079, p=0.134 — still non-significant. A joint model of $S_c$ on nonlinear size terms + baseline Dice explains R²=0.008 — essentially none of the variance. **Unlike E26 (where baseline Dice explained 51% of the D4-vs-A Dice gain), here neither size nor baseline difficulty explains the D4-vs-D2 *relative* preference at all.** This is an important distinction: baseline difficulty predicts *whether deep supervision helps in general*, but not *which scale* helps.

## 5. Comparison against Both

$Q_{Both} - \max(Q_{D4}, Q_{D2})$ (component Dice gain basis): mean=−0.0097, median=0.0000.

| | n | % |
|---|---:|---:|
| Both clearly dominates max(D4,D2) | 12 | 3.3% |
| A single scale clearly beats Both | 19 | 5.2% |
| Tied within ε | 332 | 91.5% |

Both does **not** consistently dominate — if anything the median-zero, slightly-negative-mean pattern suggests combining the two auxiliary losses is, on average, a wash or mild dilution relative to whichever single scale would have been better for that specific component, consistent with the 4-way comparison's earlier finding (`PHASE_E25_STRUCTURAL_PIVOT_1B_4WAY_MECHANISM_COMPARISON.md`) that D4-only vs Both were statistically indistinguishable in aggregate. This is not strong evidence *for* a routing mechanism either — the "tied within noise" bucket dominates everything (91.5%), meaning for the overwhelming majority of components, none of D4/D2/Both/A differ meaningfully.

## 6. Early predictability

Early window: epochs ≤10 (of 30), fixed and documented before running this analysis, ~33% of training. Features (Section 10/11 list) built exclusively from checkpoints at epochs 1/5/10 for A/D4/D2 (slope, recent mean improvement, stagnation count, regression count, stability/std, last-observed dice/coverage, early detection rate), strictly excluding any epoch >10 or the final/best checkpoint.

31 components had both valid early features (≥2 early checkpoints per condition) and a non-neutral final label (the classification target population, per Section 12).

- **Spearman(early $S_c$ proxy, final $S_c$)**: ρ=+0.196, p=0.290 — weak, non-significant.
- **Grouped 5-fold cross-validated logistic regression** (subject-grouped folds, full feature set, standardized within-fold):
  - AUROC: **0.433 ± 0.207** (worse than chance)
  - Balanced accuracy: 0.350 ± 0.168
  - F1: 0.391 ± 0.106
- **Trivial baselines**:
  - Always-D4: balanced accuracy 0.500
  - Always-D2: balanced accuracy 0.500
  - Size-only AUROC: 0.467
  - Baseline-Dice-only AUROC: 0.602 (best direction) — **this trivial single-feature baseline outperforms the full early-trajectory logistic model.**

**The trained model does not beat "always guess," and does not beat the simplest trivial baseline (baseline-Dice-only).** This is a clean, decisive failure of Gate B, not an ambiguous one. (n=31 is small, a known limitation — see Section 8 — but the direction and magnitude of the failure, underperforming even coin-flip baselines, is not a sample-size artifact that a larger n would plausibly reverse into a strong positive signal.)

## 7. Statistical robustness

Cross-endpoint stability (does the preference direction hold across metrics, or is it metric-dependent — Section 14):

| Comparison | Agreement |
|---|---:|
| Dice-preference vs IoU-preference | 98.1% |
| Dice-preference vs coverage-preference | 89.0% |

High agreement between Dice and IoU (expected, closely related metrics); somewhat lower with coverage (recall-only, ignores false positives, so some divergence is expected and not concerning). No evidence of metric-shopping-dependent conclusions.

Subject-clustered bootstrap (Section 3) confirmed the heterogeneity-test conclusion is not an artifact of treating non-independent within-subject components as independent samples.

## 8. Failure modes / limitations

- **The 125-subject validation set is not an independent test set.** It has been the sole basis for every checkpoint-selection and prior go/no-go decision across the E12–E26 arc. All numbers above should be read as "best available signal from a repeatedly-used development set," not as a held-out generalization estimate.
- **n=31 non-neutral components is small** for the classification analysis (Section 6/12); a larger validation cohort could in principle sharpen the AUROC estimate, but the direction of the current result (below-chance, below-trivial-baseline) gives no reason to expect a reversal, only tighter confidence around a null/negative finding.
- **Only periodic checkpoints (epochs 1,5,10,...,30) were available**, not every epoch — the early-window features are coarser (3 points at most within the ≤10-epoch cutoff: epochs 1, 5, 10) than a per-epoch trajectory would allow. This is reported as a real limitation, not worked around by inventing denser synthetic trajectory data.
- **Missed-component precision is undefined by convention** (not zero), which slightly reduces the effective n for precision-based S_c comparisons versus the dice/iou/coverage versions — this was handled by only computing `S_precision` where both models have a valid (non-None) precision value, documented in the code.
- $\epsilon$=0.05 is explicitly exploratory, chosen post-hoc-but-documented as a conservative multiple of the measured noise floor — a different (larger or smaller) ε would shift the exact percentages in Section 3's table but would not change Section 6's headline finding (AUROC below chance), which does not depend on ε at all.

## 9. Decision gate

**Gate A (heterogeneous scale preference)**: **marginal pass on the narrowest technical reading** (both directions occur, 48.4%/51.6%, CI doesn't exclude either extreme cleanly but centers near 0.5 not near 0/1) — but the base rate is only 8.5% of components, and neither baseline difficulty nor size explains the variance (consistent with Gate A condition 4). This is a weak, not a strong, pass.

**Gate B (early predictability)**: **FAIL.** Cross-validated AUROC=0.43, below chance and below every trivial baseline including "always guess." This is the decisive gate — Section 17 of the execution prompt is explicit: *"If not: KILL the online adaptive controller. Do not train it."*

**Gate C (incremental value over Both)**: **FAIL/inconclusive-negative.** "Both" ties with max(D4,D2) for 91.5% of components and shows a slightly negative mean delta relative to the better single scale — no evidence dynamic routing would outperform simply using "Both" or even D4-only (which already has the best aggregate Dice of the three deep-supervision variants per the E25 4-way comparison).

Per the execution prompt's own logic (Section 17: "if Heterogeneity passes but predictability fails → PARTIAL GO" is available as an option, but predictability here fails so decisively, and so far below not just "no signal" but below trivial baselines, that describing this as a partial success would overstate the evidence):

```text
NO-GO — insufficient evidence for Adaptive Decoder Credit Assignment.
```

The premise that individual lesion components have a genuinely different, *predictable-in-advance* response to D4 vs D2 supervision is not supported. A weak, noisy, roughly-coin-flip preference direction exists for a small (8.5%) minority of components, but it cannot be anticipated from early training dynamics, and "Both" (the simplest possible non-adaptive combination) already captures most of the achievable benefit without any routing logic. Building an online adaptive controller on this evidence would not be justified — per the execution prompt's explicit instruction, this idea should be abandoned rather than carried into implementation.

---

## Files

| File | Purpose |
|---|---|
| `run_e27_collect_component_data.py` | Inference + component matching, best-checkpoint (363 components) and periodic-checkpoint (10,164 records) tables |
| `analyze_e27_scale_preference.py` | Sections 4-9, 13-15 analysis; figures 1-4 |
| `analyze_e27_trajectory.py` | Sections 10-12 early-predictability analysis |
| `E27_component_table.json`, `E27_component_full_table.json` | Per-component metrics/gains/S_c, best checkpoints |
| `E27_trajectory_table.json` | Per-component metrics at every periodic checkpoint, all 4 conditions |
| `E27_scale_preference_results.json` | Statistical summary (Sections 6-9, 13-15) |
| `E27_trajectory_analysis_results.json` | Early-predictability results (Section 10-12) |
| `figures/figure1_S_c_histogram.png` … `figure5_trajectories.png` | Required visualizations |
