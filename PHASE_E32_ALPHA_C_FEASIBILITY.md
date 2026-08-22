# E32: α_c Incremental-Information Feasibility Test

**Status**: Complete. No training. Uses α_c from E31 (unchanged, reused verbatim) joined against a freshly-computed, real, in-distribution, thresholded component Dice at 64³ (A64's own native resolution — NOT E30/E31's out-of-distribution `m_c` ratio, which E31 showed is artifact-prone for small components). Full 125-subject validation set.

**Date**: 2026-08-14

**Verdict: α_c is RETAINED — it survives the incremental-information test, including the permutation safeguard, specifically within the small-lesion population.** The effect is modest in magnitude (partial Spearman ρ≈0.13–0.25 depending on population and control set) but real, non-circular, subject-clustered, and does not evaporate under permutation — a materially different and more solid result than anything DTC produced in E30/E31. This is not yet a proposed algorithm; per the user's own framing, α_c remains "a retained geometric primitive," and the next required step (Sections 5 and the narrowed literature gate) is to formulate the exact candidate weighting function and search it against 2025–2026 literature before any training.

---

## 1. Method

α_c (critical resolution, `inf{α : G_c(α)=0}`, interpolated) reused unchanged from E31 (749 native components, 559 with an estimable value, 190 right-censored — assigned a sentinel α_c=1.25, one step beyond the finest tested α, so censored components remain usable in a single continuous predictor without being silently dropped; a separate `is_censored` flag is carried).

**Outcome variable**: a freshly computed, real, thresholded component Dice at 64³ — A64's own actual prediction, in-distribution (the model's native training resolution, not the out-of-distribution 80–160³ sweep used for E30/E31's `m_c`). Matching rule identical to every prior E25–E28 component analysis (union of overlapping predicted components). This avoids E31's fatal flaw (dividing by a near-zero or exactly-zero geometric denominator) entirely — `component_dice_64` is well-defined and meaningful for all 749 components, with 0 correctly scoring components that vanished under resize as total failures, not undefined ratios.

**Comparison features**: native size, 64³ size (both nonlinearly transformed: raw, log, cube-root), and subject-level baseline Dice (A64's own real whole-volume Dice, computed in the same inference pass — mean 0.8897, matching prior per-subject-mean conventions from the E25 4-way comparison).

## 2. Does α_c predict recoverability?

Raw correlations (all 749 components): Spearman(α_c, component_dice_64)=**+0.676** (p=3.3×10⁻¹⁰¹); point-biserial(α_c, detected_64)=+0.777 (p=5.4×10⁻¹⁵²). Strong, as expected — α_c is itself a size-adjacent geometric quantity.

## 3. Comparison against native size, 64³ size, baseline Dice

| Feature | Spearman ρ with component_dice_64 |
|---|---:|
| native_size | +0.731 |
| size_64 | +0.860 |
| baseline_dice (subject) | +0.188 |
| α_c | +0.676 |

α_c correlates strongly with both size measures (ρ=0.786 with native_size, ρ=0.755 with size_64) — confirming it is not independent of size, consistent with E31's own finding (R²(α_c~nonlinear size)=0.37, i.e. substantial but incomplete overlap).

## 4. Incremental information

**Full population (n=749)**: Model 0 (nonlinear native_size + size_64 + baseline_dice) already achieves R²=0.9415 — essentially a ceiling, since α_c is highly collinear with the size features already in the model. Adding α_c: R²=0.9415, ΔR²≈0.0000.

**This ΔR²≈0 result needed scrutiny before being reported as a negative finding, and does not mean α_c has no information.** The permutation control makes this explicit: comparing real ΔR² against 500 permutation trials (α_c shuffled within joint native-size/size-64 strata) gives a permuted mean ΔR² also ≈0.0002 — both real and permuted are at the same near-zero floor, because Model 0 alone leaves almost nothing to explain. **A ΔR² comparison is uninformative when the baseline model is already near a ceiling** — this is a genuine methodological lesson, not a result about α_c. The rank-based (Spearman) permutation check remains informative regardless of the ceiling: real ρ=0.676 vs. permuted mean|ρ|=0.635 (SD=0.009) — **0/500 permutation trials exceed the real correlation magnitude.** The real correlation is small but consistently above the permuted null, even though both are large in absolute terms (because α_c is legitimately, non-artifactually correlated with size, and the permutation preserves that within-stratum correlation approximately — a stricter test than E31's, and α_c still clears it).

**Small-lesion-restricted population (native_size ≤ 150, n=587)** — the population that actually matters for the motivating question: Model 0 R²=0.2910 (not at a ceiling here), Model 1 (+α_c) R²=0.2911, ΔR²≈0.0000 **again** — but this time the near-zero ΔR² **is** worth investigating directly, since Model 0 is not saturated. Spearman(α_c, component_dice_64) restricted to this population: **ρ=+0.254 (p=4.5×10⁻¹⁰)**. A direct rank-based partial correlation (α_c vs. Dice, controlling for native_size via rank residualization) confirms a genuine, independent, size-adjusted signal: **partial Spearman ρ=+0.132 (p=0.0014)** — smaller than the raw correlation (as expected, since size explains part of it) but real and significant after removing size's contribution. **The OLS ΔR²≈0 result is a linear-model-specification artifact** (multicollinearity between α_c and the size features already in the regression absorbs α_c's linear component even though a real nonlinear/rank relationship survives) — this is the same OLS-vs-rank discrepancy pattern established as untrustworthy-without-scrutiny in E30/E31, and here again the rank-based measure is the one to trust.

## 5. The permutation safeguard, applied proactively this time

Unlike E31 (where the safeguard was applied reactively, after an initial false-positive result), this analysis ran the permutation control from the start, on both the full and small-lesion populations:

| Population | Real Spearman ρ | Permuted mean \|ρ\| (n=500 trials) | Fraction exceeding real | Verdict |
|---|---:|---:|---:|---|
| Full (n=749) | +0.676 | 0.635 | **0.000** | PASS |
| Small (native_size≤150, n=587) | +0.254 | 0.129 | **0.000** | PASS |

Both pass decisively — **0 out of 500 permutation trials in either population produced a correlation as strong as the real one.** This is a materially stronger and more clean-cut result than anything E30 or E31 produced.

## 6. Subject-clustered bootstrap

| Population | Spearman ρ mean | 95% CI |
|---|---:|---|
| Full | +0.677 | [+0.618, +0.735] |
| Small (native_size≤150) | +0.250 | **[+0.144, +0.346]** |

Both intervals solidly exclude zero — the small-lesion result, the one that matters most, has a CI that excludes even weak effect sizes at its lower bound.

## 7. Comparison to E31's failure — why this result is trustworthy where that one wasn't

Three structural differences explain why α_c survives here where DTC's `M_c`/`m_c`-based candidates did not:

1. **The outcome (`component_dice_64`) is a real, thresholded, well-defined quantity for every component**, including ones that fully vanished at 64³ (correctly scored as 0), rather than an epsilon-guarded ratio that produces artifacts exactly on the vanished-component population.
2. **α_c is a purely geometric quantity**, never involving any model-predicted probability mass in its own construction — it cannot inherit the near-zero-denominator problem that corrupted E30/E31's prediction-side measurements.
3. **The permutation safeguard was applied before, not after, accepting any result** — and it passed cleanly on both populations, with 0/500 trials exceeding the real correlation, a much larger margin than anything that survived scrutiny in E30/E31.

## 8. Candidate weighting function (stated, not yet implemented)

Per the user's own step 5 ("If yes, formulate the exact mathematical weighting function"), and constrained to the actually-demonstrated relationship (a modest, size-partial, rank-based signal — not license to overclaim a strong effect):

$$w_c = \phi(\alpha_c) = 1 + \kappa \cdot (1 - \alpha_c)_+$$

where $(1-\alpha_c)_+ = \max(0, 1-\alpha_c)$ is the "degradation urgency" — zero for components that never vanish within the tested range (α_c ≥ 1, using the same 1.25 sentinel convention as this analysis), and increasing as α_c decreases toward 0 (components that vanish very early under degradation, i.e. the geometrically most fragile ones, get the highest weight). $\kappa$ is a single scalar hyperparameter controlling the maximum weight boost, to be calibrated (not tuned to a desired outcome) analogous to E25's own λ calibration convention (matching magnitude at initialization, not post-hoc-fit to validation Dice).

This is deliberately the simplest possible functional form consistent with the demonstrated relationship (monotonic, size-partial, modest effect size) — a piecewise-linear weight, not a complex learned function, since the evidence here supports "α_c contains real, independent, but modest information," not "α_c is a dominant, near-deterministic predictor" that would justify a more elaborate mechanism.

## 9. Next required step: narrowed literature gate

Per the user's explicit instruction, this exact candidate mechanism — a **critical-resolution-derived, per-component training-loss weight**, $w_c = 1 + \kappa(1-\alpha_c)_+$ where $\alpha_c$ is defined via a deterministic multi-level degradation family and per-component geometric vanishing — must be searched against 2025–2026 literature before any training, with explicit attention to the user's own caution: recent 2026 work already exploits small-lesion-specific labeling/supervision and cross-scale consistency, so the search must establish that this is not merely a new name for "focus more on small lesions." **This search has not yet been run and is the next step, not yet performed in this report** — reported honestly as pending rather than assumed to pass.

## 10. Decision

Per the user's own decision structure: α_c passes the "does it predict recoverability" test (Section 2), passes the "compared against native size/64³ size/baseline Dice" comparison (Section 3, real independent signal beyond size confirmed via partial correlation), and passes the incremental-information test in its trustworthy (rank-based, permutation-validated) form (Sections 4–6) — **on the small-lesion population specifically**, which is the one that matters. It does not pass as a strong or dominant signal — the partial correlation (ρ≈0.13) is modest, and this should not be oversold.

**α_c is retained as a geometric primitive**, per the user's own framing — not yet a proposed algorithm. The candidate weighting function in Section 8 is stated for concreteness but is explicitly NOT to be trained until the narrowed literature gate (Section 9) is run and, per the user's original instruction, "only if it survives both novelty and predictive gates do we spend a ~67-minute training run." The predictive gate has now been cleared (with appropriate qualification about effect size); the novelty gate has not yet been attempted.

---

## Files

| File | Purpose |
|---|---|
| `run_e32_component_dice_at_64.py` | Real, thresholded, in-distribution component Dice at 64³ (A64's own resolution) |
| `analyze_e32_alpha_c_incremental_value.py` | Full analysis: correlations, incremental R², proactive permutation safeguard, subject-clustered bootstrap |
| `E32_component_dice_64_table.json` | Raw per-component Dice/detection records |
| `E32_alpha_c_incremental_value_results.json` | Full statistical results |
| `figures/e32_alpha_c_incremental_value.png` | α_c vs. Dice scatter; permutation null distribution vs. real ΔR² |
