# E29 (Part 1): Resize-Survival Analysis

**Status**: Complete. No training, no model inference — pure data-geometry analysis on the original native-resolution BraTS labels (`-seg.nii.gz`), full 125-subject validation set. This is the pre-training diagnostic requested before committing GPU time to the A64/A96/A128 resolution-ceiling training experiment.

**Date**: 2026-08-13

**Headline finding: the resolution hypothesis is supported, but with an important, honestly-disclosed complication — most "GT components" are themselves single-voxel-scale annotation specks that no reasonable resolution will recover. Once those are set aside, the remaining real small lesions show a clear, substantial, monotonic resolution-recovery effect.**

---

## 1. Method

For every native-resolution GT lesion connected component (n=749, `scipy.ndimage.label`, 6-connectivity, computed directly on the native-resolution binary mask before any resizing), I resized the same integer component-ID volume independently to 64³, 96³, and 128³ using the identical nearest-neighbor `scipy.ndimage.zoom` convention `BraTSDataset` already uses for its binary mask, then counted how many voxels of each native component survive at each target resolution. This is model-independent — no checkpoint, no training, no inference — a pure question about what information the preprocessing pipeline discards before the network ever sees it.

## 2. Overall survival rates

| Resolution | % of native components with >0 surviving voxels | Mean surviving size | Median surviving size |
|---|---:|---:|---:|
| 64³ | 30.3% | 445.1 | **0.0** |
| 96³ | 39.3% | 1526.1 | 0.0 |
| 128³ | 52.6% | 3661.7 | **1.0** |

At 64³, the **median** native component vanishes entirely (0 voxels) — more than half of all 749 native-space GT lesions are completely erased by the current preprocessing before the model ever trains on them. Survival improves monotonically with resolution (30.3% → 39.3% → 52.6%), consistent with the resolution hypothesis, but even at 128³ the majority (47.4%) of native components still vanish completely.

## 3. The critical complication: most "components" are single-voxel annotation specks

Native lesion component size distribution: **mean=15,715.6, median=2.0** (note the massive mean/median gap — a few huge tumors dominate the mean while the population is overwhelmingly tiny). Directly checked: **65.3% of all 749 native-space components are ≤5 voxels**, and 332/749 (44.3%) are **exactly 1 voxel**. These are very likely isolated annotation noise/single-pixel labeling artifacts in the original BraTS masks, not clinically meaningful lesions — no resize strategy at any of the tested resolutions (or realistically, any resolution short of native) can recover a genuinely 1-voxel-native object as a learnable target, and it would be dishonest to attribute their disappearance to "the bottleneck" the same way as genuinely-sized small lesions that get crushed by resizing.

**Excluding native components ≤5 voxels** (260 remaining, the population where "resolution recovery" is actually a meaningful question):

| Native size bin | n | Median size @64³ | Median size @96³ | Median size @128³ |
|---|---:|---:|---:|---:|
| 5–50 | 83 | **0.0** | **1.0** | **3.0** |
| 50–150 | 15 | 2.0 | 11.0 | 23.0 |
| 150–500 | 7 | 9.0 | 32.0 | 72.0 |

This is the clean, honest version of the resolution story: a lesion that's typically **already gone (median 0 voxels) at 64³** becomes marginally present (median 1 voxel) at 96³ and modestly present (median 3 voxels) at 128³ in the 5–50 native-voxel bin — real, monotonic, meaningful recovery, though still small in absolute terms even at 128³. Larger small-lesion bins (50–150, 150–500 native voxels) show a strong, clearly usable recovery: roughly 2× voxel count per resolution step, going from single-digit voxel counts at 64³ to genuinely segmentable sizes (23–72 voxels) at 128³.

## 4. Full size-bin breakdown (all components, including the ≤5-voxel specks, for completeness)

| Native size bin | n | Survive % @64³ | Survive % @96³ | Survive % @128³ |
|---|---:|---:|---:|---:|
| 1–10 | 525 | 5.7% | 15.4% | 32.8% |
| 10–50 | 47 | 44.7% | 78.7% | 95.7% |
| 50–150 | 15 | 100.0% | 100.0% | 100.0% |
| 150–500 | 7 | 100.0% | 100.0% | 100.0% |
| >500 | 155 | 99.4% | 99.4% | 100.0% |

The 10–50 native-voxel bin shows the clearest, most decision-relevant signal: survival jumps from 44.7% (64³) to 95.7% (128³) — more than doubling the fraction of real small lesions that have *any* representation at all for the model to learn from.

## 5. Direct 64→128 comparison for native lesions ≤50 voxels (n=572, includes the ≤5-voxel specks)

Mean voxel gain (128³ minus 64³): +0.78 voxels; median gain: 0.00 (dominated by the many single-voxel specks that stay at 0 regardless of resolution). Mean ratio 128³/64³ where 64³>0: **2.62×**. Of the components that were near-vanished (≤3 voxels) at 64³, only **26/572 (4.5%)** are "rescued" to a meaningfully present size (≥5 voxels) at 128³ — a real but modest absolute rescue rate, heavily diluted by the ≤5-voxel-native population that cannot be rescued at any tested resolution.

## 6. Interpretation

Both things are true simultaneously, and neither should be hidden behind the other:

1. **A large fraction of the "missed component" problem documented throughout E25–E28 is not solvable by resolution alone** — a substantial share of native-space GT components are single-voxel or near-single-voxel annotation artifacts that will not exist as learnable targets at any of the tested resolutions. If the training experiment (A64/A96/A128) shows a smaller Dice gain than the optimistic scenario in the E29 proposal, part of the explanation will be this irreducible floor, not necessarily a failure of the resolution hypothesis.
2. **For the genuinely-small-but-real lesions (roughly 5–150 native voxels, ~98/749 components, ~13% of the population but likely a meaningfully larger share of the *clinically relevant* small-lesion population once true annotation noise is excluded)**, the resolution-recovery effect is real, monotonic, and substantial — median voxel counts increase 2–8× per resolution doubling, and 128³ recovers genuinely segmentable component sizes (23–72 voxels) from what is often a complete or near-complete wipeout at 64³ (median 0–9 voxels).

**This is exactly the kind of result that should inform, not just confirm, the training experiment**: the A64/A96/A128 comparison should expect a real but likely moderate Dice gain (not a dramatic one), concentrated specifically in the mid-small lesion range, with a portion of the currently-missed-component population remaining structurally unrecoverable regardless of resolution. This is a more precise, honestly-caveated version of the hypothesis than "resolution explains everything," and it still fully justifies proceeding to the training experiment — the signal here is real and directionally exactly as predicted, just with a quantified, non-trivial ceiling.

---

## Files

| File | Purpose |
|---|---|
| `run_e29_resize_survival_analysis.py` | Native-space component labeling + independent resize to 64/96/128³ |
| `analyze_e29_resize_survival.py` | Survival-rate, size-bin, and near-vanishing analysis |
| `E29_resize_survival_table.json` | Raw per-component records (749 native components) |
| `E29_resize_survival_results.json` | Statistical summary |
| `figures/e29_resize_survival_figures.png` | Native vs resized size scatter; vanishing rate by bin and resolution |
