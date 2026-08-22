# E28: Subregion Information Audit

**Status**: Complete. No training. Inference-only on A's existing best.pth checkpoint, using the **original 4-class BraTS labels** (loaded for the first time anywhere in this project — `Dataset/brats_dataset.py` immediately binarizes them and no prior E-series analysis has touched the raw `-seg.nii.gz` class structure). Full 125-subject validation set, same deterministic split as every prior analysis.

**Date**: 2026-08-13

**Strong-gate verdict: KILL.** Subregion composition (%NCR, %ED, %ET) shows large raw correlations with A's detection/quality outcomes (Spearman ρ up to 0.80), but these collapse to near-zero and non-significant (partial r=0.02–0.04, all p>0.48) once component size is properly controlled with nonlinear terms. The incremental R² from adding subregion composition on top of a nonlinear-size-only model is **0.0002** — subregion identity explains essentially nothing beyond what size alone already explains. The apparent relationship is a confound: tiny post-resize components are overwhelmingly classified as edema-dominant almost by construction (median size of ED-dominant components is 3 voxels vs. 47 for ET-dominant and 2561 for NCR-dominant; ρ(size, %ED)=−0.70), not because A has a specific weakness segmenting edema tissue.

---

## 1. Method

For every GT whole-tumor (WT) connected component (n=363, same population as E25–E27, defined in the same 64³ resized voxel space every model actually sees), subregion composition was computed by restricting the original 4-class label (1=NCR/NET, 2=ED, 3=ET), resized with the same nearest-neighbor `zoom` convention used for the binary mask, to the component's own voxels. A sanity check confirmed the multiclass-derived WT mask (labels>0) agrees with the binarized WT mask used everywhere else in this project in every one of the 125 subjects (0/125 subjects showed >2% disagreement) — the composition measurement is trustworthy and consistent with the rest of the project's component definitions.

## 2. Overall composition

Population-mean composition across all 363 components: **%NCR=5.3%, %ED=83.6%, %ET=11.1%**, %other (resampling-edge, no matching multiclass label)=0.0%. Dominant-subregion breakdown (which class occupies the largest share of each component):

| Dominant | n | % of components |
|---|---:|---:|
| NCR | 11 | 3.0% |
| ED | 329 | 90.6% |
| ET | 23 | 6.3% |

The overwhelming majority of WT components are edema-dominant — expected, since edema is the largest and most diffuse subregion in glioma anatomy, and is disproportionately what small satellite/scattered lesions actually are once resized to 64³.

## 3. Raw detection-rate relationship (before controlling for anything)

| Dominant | n | A detect rate |
|---|---:|---:|
| NCR | 11 | 90.9% |
| ED | 329 | 49.5% |
| ET | 23 | 52.2% |

Chi-square test (detection × dominant subregion): χ²=7.30, **p=0.026** — nominally significant. Continuous-fraction correlations are large: %NCR vs dice ρ=+0.796, %ED vs dice ρ=−0.687, %ET vs dice ρ=+0.692 (all p<0.0001). On the surface, this looks like exactly the signal the hypothesis predicted — NCR-dominant components are detected far more reliably, ED-dominant components are detected worst.

## 4. The confound: this is almost entirely a size effect

Before accepting the raw result, I checked what drives dominant-subregion assignment. Size differs enormously by dominant class:

| Dominant | n | Mean size | Median size |
|---|---:|---:|---:|
| NCR | 11 | 2518.5 | 2561.0 |
| ED | 329 | 842.5 | **3.0** |
| ET | 23 | 1239.1 | 47.0 |

**Median size for ED-dominant components is 3 voxels** — these are almost all tiny post-resize remnants, not "genuine, substantial edema regions the model fails to segment." Spearman correlation of size with %ED = **−0.704** (p=1.2×10⁻⁵⁵) and with %ET = **+0.706** (p=4.6×10⁻⁵⁶) — both enormous. This makes physical sense: at very small post-resize volumes (1–5 voxels), a lesion is likely to register as whatever subregion happens to occupy the largest fraction of that tiny voxel footprint, and edema — being the most spatially extensive class in the original label before resizing collapses it — is the class most likely to "win" the dominant-subregion assignment purely by chance/geometry at small sizes, independent of any model behavior.

**Size-stratified detection rates make this explicit** (the confound resolves almost completely within size bins):

| Size bin | NCR detect | ED detect | ET detect |
|---|---:|---:|---:|
| 1–50 vox | 0.0% (n=1) | 22.5% (n=209) | 15.4% (n=13) |
| 50–150 vox | — | 75.0% (n=8) | — |
| 150–400 vox | — | 83.3% (n=6) | — |
| 400–1000 vox | 100.0% (n=1) | 100.0% (n=14) | — |
| >1000 vox | 100.0% (n=9) | 98.9% (n=92) | 100.0% (n=10) |

Within the 1–50 voxel bin (where nearly all of the ED-dominant components live), detect rates are uniformly low regardless of dominant subregion (22.5% ED vs 15.4% ET — comparable, both dominated by the small-size effect already established in E25/E26). At every larger size bin, detection is near-ceiling (83–100%) for whichever subregion happens to be present. **The dominant-subregion "effect" in Section 3 is size acting through a labeling artifact, not a genuine subregion-specific segmentation weakness.**

## 5. Confirmatory statistics: partial correlation and incremental R²

Controlling for nonlinear size (`size`, `size^(-1/3)`, `log(size)` simultaneously, per the E26/E27 lesson about not linearly-controlling a nonlinearly-related confound):

| Feature | Raw Spearman ρ (vs dice) | Partial r (size-controlled) | p |
|---|---:|---:|---:|
| %NCR | +0.796 | +0.024 | 0.646 |
| %ED | −0.687 | −0.037 | 0.487 |
| %ET | +0.692 | +0.029 | 0.579 |

All three collapse from large-and-highly-significant to essentially zero and non-significant. A joint model:

- R²(dice_A ~ nonlinear size alone) = **0.8680**
- R²(dice_A ~ nonlinear size + %NCR + %ED + %ET) = **0.8681**
- **Incremental R² from subregion composition = 0.0002**

Subregion composition adds no measurable explanatory power over size alone.

## 6. Subject-clustered bootstrap

To make sure the (already-null) result isn't an artifact of treating non-independent within-subject components as independent samples: subject-clustered bootstrap (2000 resamples) of the ED-minus-ET detect-rate gap gives mean=−0.038, 95% CI **[−0.285, +0.182]** — a wide interval spanning zero, consistent with no reliable subregion-specific effect once uncertainty is honestly propagated.

## 7. Strong gate

Per the user's own stated criterion: *"If subregion composition has no meaningful relationship with ΔDice, missed/detected status, or component quality [after appropriate control], then kill the idea."*

**Result: KILL.** The raw relationship exists and is large, but it is fully explained by the already-established small-size failure mode (E25 Structural Pivot 1A, E26) acting through a labeling artifact (tiny components are geometrically likely to register as edema-dominant), not by any subregion-specific segmentation weakness. This is a **different hypothesis than E26 tested** (E26 tested lesion-intrinsic geometric/intensity properties; this tests biological subregion identity) — but it converges on the **same underlying explanation**: component size is the dominant, well-established confound, and once it's controlled, no new discriminative axis survives. Subregion-composition-aware WT segmentation mechanisms (contrastive subregion structure, nested ET⊆TC⊆WT constraints, or otherwise) are not supported by this evidence as a route to the small-component failure mode — the failure is a resolution/size problem, not a semantic-collapse-of-subregions problem, at least as measurable from A's existing predictions.

**One caveat worth stating plainly**: this audit only asks whether subregion composition predicts A's *existing* error pattern — it does not test whether training with subregion-aware supervision would change that error pattern (e.g., by giving the model class-specific texture cues it currently has no reason to learn, since the current binary loss treats ET and ED voxels identically as `y=1`). A null finding in a *post-hoc audit of a binary-trained model* is not proof that subregion information is valueless during *training* — but it does mean the originally proposed diagnostic justification ("components A misses are disproportionately one subregion") is not supported, so a training intervention motivated specifically by that claim would not currently be evidence-backed. If subregion-aware training is still of interest on other grounds (e.g., theoretical/architectural, independent of this specific missed-component correlation), that would need to be argued on its own terms, not on the basis of this audit.

---

## 8. Recommendation

Per the user's stated dual-track plan, this closes the E28 diagnostic branch: **do not build subregion-contrastive WT refinement on the basis of the missed-component-composition hypothesis** — it was the specific empirical justification proposed, and it does not survive the size confound. The performance track (P1: augmentation, patch/crop training, lesion-aware sampling, resolution comparison, alternative segmentation objectives, multi-seed optimization — none of which have been tested anywhere in this project, per the E27 audit) remains the well-evidenced, untested lever and is the more directly justified next step.

---

## Files

| File | Purpose |
|---|---|
| `run_e28_subregion_audit.py` | Data collection: subregion composition per WT component + A's detection/dice/coverage |
| `analyze_e28_subregion_audit.py` | Statistical analysis (raw correlation, size-stratified, partial correlation, bootstrap) |
| `E28_subregion_component_table.json` | Raw per-component records (363 components) |
| `E28_subregion_audit_results.json` | Statistical summary |
| `figures/e28_figures.png` | Detection rate by dominant subregion; %ET vs component Dice scatter |
