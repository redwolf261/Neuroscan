# E30: Degradation-Trajectory Constraint (DTC) — Feasibility & Novelty Gate

**Status**: Complete. No DTC training, no λ_DTC tuning, no architecture changes — pure feasibility/measurement/literature analysis, per the explicit restrictions. Full 125-subject validation set, native BraTS multiclass masks, existing A64 checkpoint (not retrained).

**Date**: 2026-08-14

---

## Executive conclusion

DTC's core premise survives the feasibility gates, but only in a **narrower, more precisely scoped form than originally proposed**, and two real methodological bugs were caught and fixed during this analysis rather than silently producing misleading numbers — both are reported here in full, since they materially shape what can and cannot be claimed. **Geometrically (Gate A), native lesion survival under resize is real and reproducible, but it is a near-binary threshold effect (a component either survives near a theoretical volumetric-compression floor, or vanishes to exactly zero) rather than a smooth function of size — this is itself a more precise and more useful characterization than "small lesions shrink."** On the prediction side (Gate B), a genuine, subject-clustered, outlier-robust, non-circular signal was found: the degradation slope G_c (excess in how fast the model's predicted mass falls relative to GT geometric survival, measured only over early/coarser degradation levels) predicts final component quality with subject-clustered bootstrap CI [0.49, 0.77] on Spearman ρ — but **this entire prediction-side result applies only to the 25.8% of components (193/749) with well-defined, non-degenerate reference probability mass, which are overwhelmingly the LARGE components (median native size ~51,000 voxels), not the small-lesion population (median native size 1 voxel among the excluded 74%) that motivated DTC in the first place.** Gate D (literature) found no close prior art (N2). **Given this scoping mismatch between where the predictive signal was measured and where the small-lesion problem actually lives, the honest verdict is a qualified GO — DTC has real, non-trivial information content and no close prior art, but the evidence gathered here does not yet demonstrate it addresses the specific small-lesion failure mode it was designed for, and that gap must be closed before implementation.**

---

## 1. Exact degradation construction

Deterministic family, identical for every subject, monotonic:

```
alpha=0.00 -> 160^3 (finest tested resampled grid, reference)
alpha=0.25 -> 128^3
alpha=0.50 -> 96^3
alpha=0.75 -> 80^3
alpha=1.00 -> 64^3 (A64's actual training/deployment resolution)
```

Component **identity** is defined exclusively in native resolution (`scipy.ndimage.label` on the native, un-resized binary WT mask), per the explicit requirement not to use the 64³ mask to define whether a native lesion exists. All per-α measurements resample the native INTEGER component-ID volume (nearest-neighbor, order=0) to each α's grid — geometric survival, fragmentation, and boundary retention are all derived from this one consistent resampling convention, never re-deriving component identity at a coarser resolution.

**Boundary retention** (Section 4D): implemented as a real, disclosed approximation — native boundary voxel locations (surface layer of the native mask) are resampled to each α level and back to native coordinates (round-trip, nearest-neighbor both directions), and the fraction of original native boundary voxels that still register as positive after the round-trip is reported. This is a real geometric measurement, not a distance-transform-based boundary metric — judged adequate to keep rather than omit, per the instruction to omit only if forced into "inventing something weak."

## 2. Native component survival

749 native-space GT components (full 125-subject validation set; consistent with E29's independent count of 749, a useful cross-check).

| Quantity | Mean | Median | SD |
|---|---:|---:|---:|
| Slope (survival vs α, linear fit) | −0.421 | −0.414 | 0.376 |
| AUC (trapezoidal, α∈[0,1]) | 0.155 | 0.157 | 0.111 |
| Final survival at α=1.0 (64³) | 0.026 | **0.000** | — |

76.5% of components (573/749) show a "major drop" (>0.2 survival loss in one α-step). AUC variance is real and not explained by cross-subject noise alone (within-subject AUC variance 0.0079 < overall AUC variance 0.0123).

**GATE A: PASS.** Survival curves vary meaningfully across components, the behavior is reproducible within subjects, and it is not numerical interpolation noise (verified: the theoretical volumetric-compression floor for a 240×240×155→64³ resize is 0.0294, and well-resolved components empirically land almost exactly there — confirming the measurement is capturing a real physical resampling process, not an artifact).

## 3. Nonlinear dependence on lesion size (the mandatory E26/E28-style test)

**A methodological bug was caught here before being reported, not after.** An initial OLS fit of `final_survival_64 ~ nonlinear_size_features` produced R²=0.003 — near zero — despite a large, highly significant Spearman correlation (ρ=+0.72, p≈2.6×10⁻¹²¹). This contradiction was investigated rather than either number being reported alone.

**Root cause**: `final_survival_64` is not a smooth function of native size — it is **close to binary**. A component either survives near the theoretical volumetric-compression floor (≈0.029), or collapses to **exactly zero voxels**. A smooth OLS polynomial/log/cube-root basis cannot fit a step-function-like relationship, while Spearman (rank-based, monotonic) correctly detects the real underlying pattern. The correct characterization, verified directly:

| Native size bin | n | P(fully vanishes) | Mean relative survival among survivors (1.0 = exactly at the compression floor) |
|---|---:|---:|---:|
| 1–5 | 469 | **95.5%** | 19.1× *(few survivors, small denominators — noisy)* |
| 5–10 | 54 | 85.2% | 5.1× |
| 10–50 | 48 | 54.2% | 2.0× |
| 50–150 | 16 | 6.2% | 0.96× |
| 150–500 | 7 | 0.0% | 0.91× |
| 500–2000 | 18 | 5.6% | 0.93× |
| 2000–10000 | 10 | 0.0% | 1.00× |
| 10000–300000 | 127 | 0.0% | 0.96× |

Point-biserial correlation (survives-at-all vs. native size): r=+0.599, p=5.2×10⁻⁷⁴. **This — the probability of survival at all, not a smooth regression — is the honest, correct characterization of the size relationship**, and it is a threshold/digital-sampling effect, not a gradual decay. AUC and slope (the two continuous trajectory summaries) show weak-to-no size dependence (R²(size only)≈0.0001 for both, Spearman ρ=+0.18 and −0.05 respectively) — meaning that among components that DO survive somewhat, the shape of their trajectory is fairly size-independent, which is a genuinely informative finding: size mostly determines *whether* a component appears at all post-resize, not the shape of its degradation curve once it does.

## 4. Prediction survival (A64, out-of-distribution inference)

**Explicit caveat, stated once and applying throughout this section**: A64 was trained only at 64³. Running it at 80/96/128/160³ is out-of-distribution inference — this measures how A64's already-learned features respond to unfamiliar input scales, not what a model trained at that resolution would predict. Confirmed technically feasible (fully convolutional, no fixed-size layers; direct probe succeeded at all 5 resolutions, peak memory 5.15GB at 160³, well within the 8GB budget) before proceeding, per the instruction not to fake this if infeasible.

**A second, more consequential problem was found and fixed here.** The originally-defined `s_hat_c(α) = v_hat_c(α) / (v_hat_c(α=0) + ε)` produced mean values in the **millions** at α∈{0.25,0.5,0.75,1.0} — clearly broken. Root cause: A64's raw predicted probability mass is near-zero for the large majority of small components at **every** tested resolution, including its own native 64³ (median `v_hat_c` at α=1.0 is exactly 0.0 across all 749 components) — itself a real, independently-confirming finding consistent with everything E25–E29 established about small-lesion failure, but it makes a ratio against a near-zero reference denominator numerically meaningless (near-zero-over-near-zero noise).

**Fix**: a pre-declared floor (`v_hat_c(α=0) ≥ 1e-3`, chosen before checking how it affected any downstream result) defines which components have a well-defined survival ratio. **193/749 (25.8%) pass this floor; 556/749 (74.2%) are excluded** as "reference too small to define a ratio," rather than silently reporting huge or meaningless numbers. Checking what distinguishes the two groups: the valid (ratio-computable) subset has median native size ~51,038 voxels; the excluded subset has median native size **1 voxel**. **The ratio-based analysis below is therefore restricted, by construction, to large, easily-resolved components — not the small-lesion population that motivated DTC.** This is the single most important scoping fact in this report and is carried through every subsequent section.

Raw predicted mass `v_hat_c(α)` (well-defined for all 749 components, unaffected by the floor) declines monotonically with α: median exactly 0 by α=0.5, with the fraction of exactly-zero components rising from 32.7% (α=0) to 69.7% (α=1.0) — consistent with, and a direct confirmation of, the geometric near-vanishing pattern in Section 3.

## 5. Degradation excess

Within the valid 193-component subset, `A_c` (excess area) and `G_c` (excess slope) were computed using **only α∈{0.00, 0.25, 0.50, 0.75}** — explicitly excluding α=1.0 from their construction. This was a deliberate, disclosed fix (see Section 6) to prevent circularity with the α=1.0-based prediction target used in Section 6's own predictive test.

| Quantity | Mean | Median | SD |
|---|---:|---:|---:|
| A_c | 0.0060 | 0.0000 | 0.0218 |
| G_c | 54.52 | −0.51 | 565.27 |

## 6. Predictive value beyond size — including a caught circularity bug

**A third methodological issue was caught here before being reported as a result.** The first version of this analysis used `s_hat_c(α=1.0)` as both the outcome being predicted AND (via A_c/G_c's construction from the full 5-point trajectory including α=1.0) an ingredient of the predictor itself. This produced an apparent Model 2 (size+G_c) R²=0.87 — which, per Section 20's explicit restriction ("do not use the final prediction to construct a predictor of itself"), was investigated rather than trusted. Diagnostic: `corr(G_c, final_quality)` remained ≈0.93–0.99 even after removing the 5 most extreme points — too strong and too outlier-*insensitive* to be a real relationship, and mathematically expected since the α=1.0 point was literally shared between predictor and target.

**Fix**: A_c and G_c were redefined to use only α∈{0.00,0.25,0.50,0.75} (Section 5), holding out α=1.0's `s_hat_c` as a genuinely independent target — the same early/late separation discipline established in E27.

With this fix, the honest predictive comparison (all within the 193-component valid subset, nonlinear size features standardized before OLS):

| Model | R² | ΔR² vs. size-only |
|---|---:|---:|
| Model 0 (nonlinear size only) | 0.0851 | — |
| Model 1 (size + A_c) | 0.1061 | +0.0210 |
| Model 2 (size + G_c) | 0.2555 | +0.1704 |
| Model 3 (size + A_c + G_c) | 0.2647 | +0.1796 |

**A fourth check, an outlier-sensitivity audit, was run on Model 2 before accepting ΔR²=0.17 at face value** (per this project's established discipline of stress-testing any surprisingly large effect, e.g. as in E27/E28/E29). Result: Pearson r between G_c and final quality collapses from 0.46 to 0.11 (non-significant, p=0.14) when the 5 most extreme |G_c| points (2.6% of the subset) are removed — meaning the **linear/OLS ΔR² is inflated by a small number of heavy-tailed outliers and should not be reported as a clean, robust effect size on its own.** However, the **rank-based (Spearman) relationship is genuinely robust**: ρ=0.621 (all points) vs. ρ=0.629 (excluding the same 5 outliers) — nearly unchanged, strongly significant either way (p<10⁻²¹). The underlying monotonic signal is real; the OLS/linear-model summary overstates its magnitude.

**GATE B, dual criterion**: OLS-based (ΔR²≥0.05): **PASS** (0.17). Robust/rank-based (outlier-excluded Spearman |ρ|≥0.3, p<0.01): **PASS** (0.629, p=4.5×10⁻²²). **Both pass — Gate B clears on the stricter combined standard**, not merely the fragile OLS number.

## 7. Subject-clustered statistical analysis

Subject-clustered bootstrap (1000 resamples, resampling by subject not component, n=125 subjects):

| Quantity | Mean | 95% CI |
|---|---:|---|
| ΔR² (Model 1: size+A_c vs. size-only) | +0.031 | [+0.007, +0.103] |
| Spearman ρ(G_c, final_quality) | +0.624 | **[+0.489, +0.767]** |

The G_c-based signal's confidence interval excludes zero by a wide margin and excludes even weak effect sizes — this is the strongest, most trustworthy result in the entire analysis.

## 8. Triviality check

Is A_c merely a re-expression of size or final outcome? Spearman(A_c, final_quality)=−0.462; Spearman(A_c, native_size)=−0.402; Spearman(A_c, GT survival at 64³)=+0.013 (essentially zero); R²(A_c ~ nonlinear size alone)=0.226. **None of these approach the "A_c ≈ f(size)" or "A_c ≈ f(outcome)" collapse thresholds** (>0.8 correlation or >0.7 R² was the pre-declared concern threshold) — A_c carries real information not reducible to size or to the outcome it's used to predict.

## 9. Prior-art audit (Gate D)

Full search results and per-paper mechanism analysis: `E30_literature_novelty_matrix.md`. Summary: 12 required search terms run against 2025-2026 literature; three close-but-distinct records found (SDM/label-smoothing, GRCSF, soft-labeling-for-downsampling) — none combine a continuous α-indexed degradation family with per-component geometric-vs-predicted survival trajectory comparison as a training loss. **Novelty classification: N2** (no close mathematical prior art found in the searched literature — not a claim of being "first," and a deeper targeted search of MICCAI/TMI/MIDL full proceedings would strengthen this before a stronger claim).

## 10. Limitations

- **The single most important limitation**: the entire prediction-side analysis (Sections 4-8, everything about A_c/G_c/Gate B) is restricted to the 25.8% of components with a well-defined reference (median native size ~51,000 voxels) — essentially the opposite population from the small-lesion failure mode (median native size 1 voxel among the excluded 74%) that motivated DTC. **Gate B's pass should not be read as "DTC helps small lesions" — it has not yet been tested there, because A64's near-zero raw probability mass for small components makes the current ratio formulation undefined for exactly the population that matters most.**
- A64 is used far outside its training distribution (80–160³ vs. its 64³ training resolution) for the entire prediction-side analysis — a genuinely different, weaker inference regime than what a resolution-aware model would actually see.
- Two real bugs (OLS-on-a-binary-target near-zero-R², and predictor/target circularity) were found and fixed during this analysis, not before. This is reported transparently rather than only presenting the corrected numbers, per this project's established discipline — but it also means a first, unscrutinized pass at this same analysis would have reported a materially wrong conclusion in both cases (an artificially weak nonlinear-size relationship, and an artificially strong, circular G_c-predictiveness result).
- Boundary retention (Section 4D) is a round-trip-resample approximation, not a distance-transform boundary metric — adequate but not the strongest possible measurement.
- The 125-subject validation set is not an independent test set (used repeatedly across this project's entire diagnostic history) — this analysis's own findings should not be treated as validated on unseen data.
- G_c's OLS-based effect size is inflated by outliers (Section 6); only the rank-based/bootstrap-CI characterization should be treated as robust.

## 11. Decision gate

- **Gate A (nontrivial degradation behavior)**: PASS — real, reproducible, non-noise variance in survival trajectories.
- **Gate B (information beyond static size)**: PASS on both the fragile OLS criterion and the stricter robust/subject-clustered criterion, but **only within the large-component subset** where the ratio is well-defined.
- **Gate C (predictive degradation excess)**: PASS in the same restricted sense as Gate B — G_c predicts final quality beyond size and baseline difficulty, subject-clustered CI solidly excludes zero.
- **Gate D (mathematical prior-art audit)**: PASS — N2, no close prior art found in the searched literature.

All four gates technically pass, but Gates B/C pass on a population that is the near-opposite of DTC's motivating failure mode. Per Section 17's own honesty requirement ("do not report a GO unless all three components pass" — read here as requiring the *substance*, not just the letter, of each gate):

```
GO — DTC has measurable information beyond lesion size, predicts segmentation failure, and no close mathematical prior art was identified in the searched literature.
```

**This GO is qualified and should not be acted on without first closing one specific gap**: before any implementation, the prediction-side measurement needs to be redone in a way that produces a well-defined signal for SMALL components specifically — either by using a model trained closer to native resolution as the reference (so its raw probability mass doesn't collapse to near-zero for small lesions), by redefining the reference denominator to not require A64's own confidence, or by directly testing whether G_c (or an analogous early-trajectory quantity) has predictive value within the small-component population using a genuinely appropriate reference. Until that is done, the mathematically minimal implementation recommended by Section 20 should target this specific, disclosed gap as its first validation step — not proceed directly to training against the full small-lesion population on the assumption that the large-component finding generalizes down.

---

## Files

| File | Purpose |
|---|---|
| `run_e30_survival_trajectory.py` | Native-space GT component geometric survival, fragmentation, boundary retention across 5 α levels |
| `run_e30_prediction_trajectory.py` | A64 out-of-distribution inference at each α level, per-component predicted mass |
| `analyze_e30_geometric_survival.py` | Sections 2-3 statistics (including the caught OLS/binary-target bug and its fix) |
| `analyze_e30_prediction_and_excess.py` | Sections 4-8 statistics (including the caught reference-floor bug, the circularity bug, and the outlier-sensitivity audit) |
| `E30_component_survival_table.json` | Raw per-component geometric records (749 components) |
| `E30_prediction_degradation_table.json` | Raw per-component prediction records (749 components) |
| `E30_statistical_results_geometric.json`, `E30_statistical_results_prediction.json` | Statistical summaries |
| `E30_literature_novelty_matrix.md` | Gate D full search results and mechanism comparison table |
| `figures/figure1-6*.png` | Required visualizations |
