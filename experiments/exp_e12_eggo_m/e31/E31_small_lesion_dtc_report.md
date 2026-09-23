# E31: Small-Lesion-Safe DTC Reformulation — Feasibility Verdict

**Status**: Complete. No training. Reuses E30's already-collected geometric and raw-probability-mass data; no new model inference. Full 125-subject validation set, native BraTS masks, A64 checkpoint.

**Date**: 2026-08-14

**Decision: KILL DTC (in both its original and reformulated forms).** The candidate signal that survived every check in the large-component population (E30) does **not** survive contact with the actual small-lesion population once measured honestly, and the version that initially appeared to show a small-lesion effect turned out to fail the permutation safeguard decisively — the observed correlation is smaller than what pure chance produces under the same size-stratified construction. This is reported as a clean kill, not a partial or ambiguous one, per Section 12's own kill-condition discipline (KILL 3: signal survives prediction permutation).

---

## 1. Section 10 first: GT-only resolution survival curves (model-independent)

Reused E30's own geometric survival table (749 native components) — no new computation needed for this part.

**Vanishing is abrupt, not gradual.** 57.3% of components that eventually vanish go from full/proportional survival (≥1.0× the theoretical volumetric-compression floor) directly to **zero voxels in a single α-step** — not a multi-step decay. This confirms and sharpens E30's Section 3 finding using an independent measure (transition sharpness rather than the raw survival-vs-size regression).

**Critical resolution α_c** = inf{α : G_c(α)=0}, estimated via linear interpolation between the last-nonzero and first-zero α points: 559/749 components (74.6%) vanish within the tested range (α_c estimable); 190/749 (25.4%) never vanish even at 64³ (right-censored). For the 559 with an estimable α_c: mean=0.231, median=0.250 (i.e. most components that do vanish, vanish quite early — around the 128³ level — not gradually approaching 64³).

**α_c is genuinely not just a re-expression of size**: R²(α_c ~ nonlinear size)=0.372 — real, substantial, but far from 1.0. This means two components of similar native size can have meaningfully different critical resolutions, which was the motivation for treating α_c as a potentially useful, non-redundant weighting quantity (Section 10's own proposed `w_c = φ(α_c)`). This finding stands on its own, independent of everything below, and is worth keeping in mind for a future, differently-scoped experiment — but it is not sufficient by itself to justify DTC, since it says nothing about whether the MODEL's behavior tracks or diverges from this geometric quantity in a way that's actionable (that requires the prediction-side analysis below, which is where the kill occurs).

## 2. Abandoning the ratio: new quantities

Following the execution prompt exactly: `M_c(α)` (raw predicted probability mass, reused verbatim from E30's `v_hat_alpha`), `V_c(α)` (component's own resized voxel footprint, E30's `size_alpha`), `m_c(α) = M_c(α)/(V_c(α)+ε)` (mean predicted probability inside the GT component — normalized by a model-independent quantity), and `G_c(α)` (geometric survival ratio, reused from E30). None of these divide by the model's own reference prediction — E30's root problem is structurally avoided.

## 3. Size stratification, S1–S6

| Bin | Range (post-64³ voxels) | n | Mean native size | Mean size@64³ | Mean M_c@64³ | Mean m_c@64³ | Detect rate |
|---|---|---:|---:|---:|---:|---:|---:|
| S1 | 1–5 | 68 | 37.8 | 1.59 | 0.330 | 0.172 | 100% |
| S2 | 6–10 | 3 | 347.0 | 9.33 | 1.324 | 0.147 | 100% |
| S3 | 11–25 | 11 | 618.4 | 17.55 | 4.125 | 0.269 | 100% |
| S4 | 26–50 | 7 | 1518.1 | 40.14 | 11.200 | 0.246 | 100% |
| S5 | 50–150 | 5 | 2479.6 | 77.20 | 8.355 | 0.116 | 100% |
| S6 | >150 | 133 | 88237.1 | 2499.22 | 2220.79 | 0.862 | 100% |

**Important note on "detect rate=100%" across every bin**: this reflects the threshold used (`M_c(64) > 1e-6`), which is extremely permissive — essentially "any nonzero predicted mass at all." This is not the same as a real, thresholded, clinically meaningful detection (as used in E25–E28), and should not be read as "the model detects 100% of small lesions" — it means the model assigns *some* nonzero probability almost everywhere, which is expected and uninformative on its own.

`m_c` does vary meaningfully within S1–S4 (not collapsed to all-zero): S1 SD=0.342, S3 SD=0.439, ranges spanning [0.00001, 1.0] — confirming the small-lesion-safe quantities are at least well-defined and non-degenerate, unlike E30's broken ratio.

## 4. Candidate signals and the size-control test

Three candidates constructed strictly from EARLY α levels (0.00–0.75), holding out α=1.0's `m_c` as the independent outcome — the same non-circularity discipline established in E30:

| Candidate | Full population ΔR² | Full population Spearman | S1–S4 (naive, n=611) ΔR² | S1–S4 (naive) Spearman |
|---|---:|---:|---:|---:|
| A (absolute evidence decay) | +0.002 | ρ=+0.469 | +0.015 | ρ=−0.137 |
| B (geometry-fitted residual) | +0.005 | ρ=+0.062 (ns) | +0.006 | ρ=+0.391 |
| C (evidence/geometry mismatch) | +0.022 | ρ≈0 (ns) | +0.050 | ρ=−0.245 |

Candidate C had the largest ΔR² in both populations and is, per the execution prompt's own framing, "the closest analogue to the original DTC idea" — it received the deepest scrutiny below.

## 5. The permutation safeguard (Section 8) — and why it matters here

**Full population**: Candidate C's real ΔR²=+0.022 clearly exceeds its permuted-control ΔR²=+0.0001 — passes cleanly. Candidates A and B do **not** clear this bar (real vs. permuted are statistically indistinguishable) and are set aside as not model-specific in the full population.

**But a direct OLS-vs-rank check on Candidate C's full-population result revealed it is a weak, non-monotonic, linear-only artifact**: Pearson r=+0.18 is stable under outlier removal (r=+0.20 without the top 5), but Spearman ρ≈0 (p=0.97) both with and without those outliers. A real, monotonic relationship should show up in rank correlation; this doesn't. The ΔR²=0.022 in the full population is a small, real-but-weak linear effect, not the strong signal the raw number might suggest.

**The S1–S4-restricted result initially looked like the opposite, and more promising, pattern** — Spearman ρ=−0.245 (p=8×10⁻¹⁰), stable under outlier removal (ρ=−0.264 without the top 5), subject-clustered bootstrap CI **[−0.354, −0.132]**, solidly excluding zero. This looked like a genuine, robust, small-lesion-specific finding.

**It was not.** A critical flaw was found before accepting this: **522 of the 611 "S1–S4" components have `V_c(α=1.0)=0`** — i.e., the component's own geometric footprint has *already vanished entirely* at 64³ (consistent with E30/Section 1's near-binary vanishing finding). For these, `m_c(α=1.0) = M_c(1.0)/(0+ε)` is not measuring "mean predicted probability inside the lesion" — it's dividing an arbitrary small numerator by an artificially tiny epsilon, producing a value driven by numerical noise, not by any real geometric or model quantity. This directly explains why the permutation control (built on the full S1–S4 population including these 522 artifact-prone components) still showed a strong opposite-signed correlation (ρ=+0.238, p=2.4×10⁻⁹) — **the "signal" was partly an artifact of the ε-guarded division activating on geometrically-empty components**, not a real model-vs-geometry relationship.

**Corrected analysis, restricting S1–S4 to components with actual nonzero footprint at 64³ (n=89, the honest small-lesion population where `m_c(1.0)` is a meaningful quantity)**: Spearman ρ=+0.255 (p=0.016) — real-looking on its own, but:
- **Fails the outlier check**: drops to ρ=0.19 (p=0.08, non-significant) after removing just the 3 most extreme points, out of only 89 components.
- **Fails the permutation safeguard decisively**: 200 permutation trials (shuffling `M_c` within native-size strata, same population, same construction) produce a null distribution with mean |ρ|≈0.39 — **larger in magnitude than the real observed ρ=0.255**. 96.5% of random permutation trials produce a correlation at least as strong as the real one.

**This is KILL 3, unambiguously**: "signal survives prediction permutation" is exactly the condition observed — the apparent small-lesion evidence-vs-geometry mismatch signal is statistically indistinguishable from what the same size-stratified construction produces under a randomized model.

## 6. Why the permutation control matters here (mechanistic explanation)

`D_C` is constructed from `m_c(α)` and a fitted function of `G_c(α)` and `log(V_c(0))` — both of the latter are pure geometry, shared identically between the real and permuted versions. `final_quality = m_c(1.0)` is *also* partly a function of geometry (via its own `V_c(1.0)` denominator). Within the small-component population — where `V_c` values are tiny, near-zero, or exactly zero for the majority — small differences in geometry-derived normalization dominate the resulting correlation regardless of what `M_c` (the actual model-specific quantity) contains. This is precisely the scenario the permutation safeguard exists to catch, and it did.

## 7. Kill conditions checked

- **KILL 1** (no usable signal in S1–S4): not applicable as stated — a signal was found, but it fails KILL 3, which supersedes.
- **KILL 2** (signal disappears after nonlinear size correction): the S1–S4 signal was measured *within* a nonlinear-size-controlled model (ΔR² framing) and still appeared — this alone did not kill it.
- **KILL 3 (signal survives prediction permutation): TRIGGERED.** Real ρ=0.255 vs. permuted null mean |ρ|≈0.39 — the real signal does not exceed, and is in fact smaller than, the permuted control.
- KILL 4/5/6: not reached — KILL 3 alone is sufficient per Section 12's "any one of these can kill DTC."

## 8. What survives from this analysis, worth keeping

- **The critical-resolution quantity α_c (Section 1 above) is real, non-redundant with size (R²=0.37, not near 1.0), and was never tested against the model at all** — it is a purely geometric object. A future, differently-designed experiment could test whether α_c (not the M_c/m_c ratio machinery, which is what failed here) has model-relevant predictive value, without inheriting the small-footprint-denominator artifact that killed this analysis.
- **The abrupt/near-binary vanishing characterization (Section 1) is now confirmed by two independent measures** (E30's raw survival distribution, and this analysis's transition-sharpness statistic) — a robust, well-established fact about this project's preprocessing pipeline, useful for any future resolution-focused work regardless of DTC's fate.
- **The permutation-safeguard methodology itself worked exactly as intended** — it caught a real, non-obvious artifact (the epsilon-division-on-vanished-components issue) that neither the size-control test nor the subject-clustered bootstrap alone would have caught. This is a useful, reusable check for any future small-lesion-quantity analysis in this project.

## 9. Decision

Per Section 13's success condition, all four criteria are required; KILL 3 fails outright, so the conjunction fails regardless of the other three:

```
KILL — DTC does not survive the small-lesion-safe reformulation. The apparent
signal within the actual small-lesion population is not distinguishable from
what the same construction produces under prediction randomization (KILL 3),
traced to an epsilon-guarded division activating on geometrically-vanished
components (V_c(alpha=1.0)=0 for 522/611 nominal "S1-S4" components), not a
genuine model-specific evidence-vs-geometry mismatch.
```

This is not a rejection of the broader research direction (small-lesion information loss under resize remains real, well-established, and unaddressed by anything trained so far) — it is a rejection of this specific mathematical mechanism (DTC, in both its original and reformulated forms) as currently defined. Per the user's own strategic framing, the next step should not be a further DTC variant, but a genuinely different mathematical object — and α_c (Section 1/8) is the one candidate from this analysis that has not yet been tested against the model and does not share the failure mode that killed the M_c/m_c-based formulations.

---

## Files

| File | Purpose |
|---|---|
| `analyze_e31_critical_resolution.py` | Section 10: GT-only survival curves, critical resolution α_c |
| `analyze_e31_small_lesion_signal.py` | Sections 1-9: small-lesion-safe quantities, candidates A/B/C, permutation safeguard, the artifact discovery and corrected re-analysis |
| `E31_alpha_c_table.json` | Per-component critical resolution values |
| `E31_critical_resolution_results.json` | Section 10 statistics |
| `E31_small_lesion_signal_results.json` | Sections 1-9 statistics (includes both the naive and artifact-corrected S1-S4 results) |
| `figures/e31_critical_resolution.png`, `figures/e31_candidates_and_permutation.png` | Visualizations |
