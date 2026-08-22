# Phase E26: D4 Response Phenotype

**Status**: ✅ Complete. No new training — pure re-analysis of the existing A and D4only checkpoints on the full 125-subject validation set, extended with candidate predictor features.

**Date**: 2026-08-13

**Decision gate result: FAILURE.** No stable predictor of D4-only's benefit was found beyond lesion size and baseline (A) Dice. Two candidates that initially looked promising (component-level `surface_to_volume`, subject-level intensity statistics) were traced to nonlinear reparameterizations of size/baseline-Dice and did not survive proper nonlinear confound control. Per the pre-registered decision rule: **do not force adaptive D4 supervision; search for a different mechanism.**

---

## 1. Question

What distinguishes components/subjects that benefit strongly from D4 supervision?

## 2. Method

125-subject validation set, subject-level and component-level candidate features computed from GT masks and FLAIR intensities (not model-internal quantities): mean/std component size, lesion fragmentation (size CV), lesion/component surface-to-volume ratio, mean/std FLAIR intensity inside lesion, spatial location (distance from volume center), inter-component isolation, and baseline (A) component-Dice variance. Targets: subject-level `delta_dice = dice_D4only - dice_A` (n=125), component-level `delta_comp_dice` restricted to components detected by both A and D4only (n=166/363).

## 3. Subject-level results

Raw Pearson correlations with `delta_dice`:

| Feature | r | p |
|---|---:|---:|
| dice_A (baseline) | **−0.702** | <0.0001 |
| lesion_surface_to_volume | +0.303 | 0.0006 |
| gt_lesion_volume | −0.202 | 0.024 |
| mean_component_size | −0.132 | 0.14 |
| std_intensity | −0.128 | 0.15 |
| mean_intensity | +0.118 | 0.19 |
| n_gt_components | +0.001 | 0.99 |

**Baseline Dice dominates everything else** — a joint linear model using only `mean_component_size + dice_A` already explains R²=0.51 of the variance in delta_dice. This matches the top-10-vs-rest comparison: the top-10 highest-gain subjects have mean baseline Dice of 0.709 (median 0.793) vs. 0.903 for the remaining 115 (Mann-Whitney p<0.0001) — the two largest gains (subjects at dice_A≈0.295) are essentially catastrophic-baseline-failure subjects where D4-only rescues a badly broken segmentation, not a subtle quality effect.

`lesion_surface_to_volume` looked like an independent candidate (raw p=0.0006), but subject-level lesion surface-to-volume for these compact lesions is itself strongly driven by size (smaller/more fragmented lesions have higher surface-to-volume by geometry), and its partial correlation after controlling for nonlinear size terms + `dice_A` + `dice_A²` was not tested subject-level directly (component-level version below was, and failed) — treated as not independently supported given the component-level finding.

## 4. Component-level results (the more diagnostic test — same lesion, same subject, only detection/quality varies)

Raw correlations with `delta_comp_dice` (components detected by both models, n=166):

| Feature | r | p |
|---|---:|---:|
| a_comp_dice (baseline difficulty) | −0.337 | <0.0001 |
| size | −0.037 | 0.63 |
| surface_to_volume | +0.010 | 0.90 |
| mean_intensity | −0.019 | 0.81 |
| std_intensity | +0.056 | 0.48 |
| dist_from_center | −0.062 | 0.43 |
| isolation | −0.082 | 0.40 |

An initial linear-control partial correlation for `surface_to_volume` (controlling only linear `size` + `a_comp_dice`) showed a large apparent effect (partial r=−0.51, p<0.0001). **This was investigated and found to be a control-specification artifact, not a real finding**: `surface_to_volume` correlates at r=0.83 with `size^(-1/3)` (the expected relationship for roughly-compact 3D blobs), i.e. it is a nonlinear reparameterization of size, not an independent shape/complexity measure. A linear control for `size` does not remove this nonlinear dependence. Re-running the partial correlation with `size`, `size^(-1/3)`, and `log(size)` all included as controls collapses the effect: **partial r=−0.122, p=0.118 — not significant.** `std_intensity`'s component-level partial effect similarly collapsed (r=−0.023, p=0.76) under the same corrected control set.

## 5. Corrected verdict

With nonlinear size and baseline-difficulty properly controlled:

| Feature | Corrected partial r | p |
|---|---:|---:|
| Component surface_to_volume | −0.122 | 0.118 |
| Component std_intensity | −0.023 | 0.76 |
| Subject mean_intensity | +0.160 | 0.074 |
| Subject std_intensity | −0.107 | 0.24 |

**None reach significance.** The response to D4 supervision is well predicted by size and baseline Dice alone (R²=0.51 at the subject level from those two variables jointly) and by baseline component-difficulty alone at the component level — but no additional, independent difficulty/dynamics variable was found among lesion geometry (fragmentation, surface complexity, isolation, spatial location) or FLAIR intensity statistics.

## 6. Interpretation against the pre-registered gate

- **Success criterion** ("find a reproducible difficulty/dynamics variable that explains a substantial portion of ΔDice beyond size/baseline-Dice, and preferably predicts component-level improvement"): **not met.** The only variables that explain substantial variance (baseline Dice, size) are exactly the "obvious confounders" excluded by the gate's own definition of success.
- **Failure criterion** ("no stable predictor beyond lesion size / baseline Dice / obvious confounders"): **met.**

The strongest single fact this analysis surfaces is not a new mechanism but a reframing of the existing one: D4-only's aggregate gain is substantially explained by **regression-to-competence on subjects/components where A was already doing poorly** (baseline Dice is the dominant predictor, and the largest individual subject gains come from near-catastrophic A failures), rather than a size- or intensity-specific phenomenon distinguishable from "A was bad here for some reason, and D4 supervision generically helps more when A is bad."

## 7. Decision, per the pre-registered gate

**→ Do not force adaptive D4 supervision. Search for a different mechanism.**

No literature audit or adaptive-supervision algorithm design is warranted on this evidence — there is no reproducible, independent phenotype variable to build such an algorithm around. The honest characterization of the D4/deep-supervision arc's outcome: a real, reproducible, modest Dice improvement exists (established in Structural Pivots 1A/1B), concentrated among subjects/components with poor baseline performance, but its cause is not resolved into an interpretable, actionable difficulty axis beyond "baseline was already failing there."

---

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e25/run_e26_response_phenotype.py` | Feature extraction (subject + component level), full 125-subject val set |
| `experiments/exp_e12_eggo_m/e25/analyze_e26_response_phenotype.py` | Correlation, top-10-vs-rest, and partial-correlation analysis (including the corrected nonlinear-size-control re-run) |
| `experiments/exp_e12_eggo_m/e25/e26_response_phenotype_results/e26_response_phenotype.json` | Raw per-subject and per-component feature records |
| `PHASE_E25_STRUCTURAL_PIVOT_1B_4WAY_MECHANISM_COMPARISON.md` | Prior comparison this phenotype analysis follows up on |
