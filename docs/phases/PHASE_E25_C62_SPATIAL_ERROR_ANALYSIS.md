# Phase E25, C6-2.9: Spatial Error Concentration & Subject-Level Failure Analysis — No Dominant Structural Pattern Found

**Status**: ✅ Complete. Compares A's and C6-2's real best checkpoints (verified directly from each checkpoint's own saved metadata: A epoch 23, `best_val_dice=0.9063020758330822`; C6-2 epoch 28, `best_val_dice=0.9030254185199738` — both exactly matching H4's own reported figures) across the full 125-subject validation set, no new training, no perturbation. **The honest finding: no single spatial, boundary-distance, lesion-size, or fragmentation category cleanly explains the Dice gap.** Slice-position error rates track each other closely (ratios 0.96–1.07 across every non-empty bin). Boundary-distance error rates show a real but modest relative increase specific to the near-boundary region (2–4 voxels: ratio 1.30) that does not scale up further at intermediate distances (4–8 voxels: ratio 1.12) the way a purely boundary-driven story would predict. The 4-way spatial classification shows C6-2 makes both more new errors and fixes more of A's errors than either "pure improvement" or "pure regression" would suggest (15,009 improved vs. 17,432 regressed voxels — close to balanced, net slightly unfavorable). Connected-component counts are nearly identical (183 vs. 178 missed lesions, 96 vs. 97 false-positive components, out of 363 total). Per-subject Dice deltas show no significant correlation with lesion volume (r=−0.175, p=0.051, borderline but not significant) or lesion count/fragmentation (r=−0.017, p=0.85). **The deficit looks genuinely diffuse — small, real, subject-specific fluctuations without an identifiable common structural cause — rather than concentrated in one interpretable category.**

**Date**: 2026-08-12

---

## Method

A's and C6-2's real `best.pth` checkpoints were loaded directly (not by an assumed epoch number — verified at runtime against each checkpoint's own saved `epoch`/`best_val_dice` fields, both confirmed to exactly match H4's own established headline numbers). Both models were run in `.eval()` mode on the same full 125-subject validation set, identical preprocessing (`target_shape=(64,64,64)`), identical thresholding (`p≥0.5`). One prediction-generation pass per subject per model; every downstream analysis below is computed from that single set of predictions.

---

## Result 1: slice-position error rates — no depth-specific pattern

| Z-bin | A error rate | C6-2 error rate | Ratio (C6-2/A) |
|---|---:|---:|---:|
| 0–20%, 80–100% | 0.0000% | 0.0000% | — (no lesion tissue at these depths in this cohort) |
| 20–30% | 0.131% | 0.125% | 0.957 |
| 30–40% | 0.340% | 0.346% | 1.019 |
| 40–50% | 0.361% | 0.384% | 1.063 |
| 50–60% | 0.416% | 0.444% | 1.067 |
| 60–70% | 0.453% | 0.478% | 1.056 |
| 70–80% | 0.156% | 0.153% | 0.979 |

Error rates track each other closely across every populated bin, with C6-2 modestly higher (5–7%) only in the mid-volume bins (40–70%) where lesion tissue and error counts are both highest in absolute terms — not a depth-specific failure mode, just where most of the data (and hence most of the small residual gap) naturally lives.

---

## Result 2: boundary-distance error rates — a real but non-monotonic signal

| Distance from lesion boundary | A error rate | C6-2 error rate | Ratio | n voxels in bin |
|---|---:|---:|---:|---:|
| 0–1 | — | — | — (no voxels; EDT convention places the boundary layer itself at distance exactly 1, see note) |
| 1–2 | 11.21% | 11.44% | 1.021 | 502,740 |
| 2–4 | 0.553% | 0.719% | **1.299** | 676,511 |
| 4–8 | 0.046% | 0.051% | 1.124 | 1,701,219 |
| >8 | 0.0008% | 0.0008% | 1.080 | 29,887,530 |

**Note on the empty 0–1 bin**: this is a property of the Euclidean-distance-transform construction, not missing data — voxels immediately adjacent to the boundary land at distance exactly 1 under this scheme (there is no fractional-distance layer between 0 and 1 for integer-grid EDT), so the `1–2` bin is effectively the true boundary layer, correctly capturing the overwhelming majority of both models' errors (11%, by far the highest error rate of any bin, exactly as expected — boundary voxels are the hardest for any segmentation model).

**The real signal**: at the boundary layer itself (1–2 voxels), C6-2's relative excess is small (2.1%). It is actually *largest* one bin further out (2–4 voxels: 29.9% relative excess) and then *shrinks* again at 4–8 voxels (12.4%) before flattening at long range. **This is not the clean "SC-TAM specifically damages boundary-adjacent voxels" pattern the mechanism's own design would most naturally predict** — if that were the whole story, the relative excess should be largest immediately at the boundary and monotonically decay outward, not peak one bin removed from it. This is reported as a real, modest, but structurally puzzling signal, not dismissed and not overinterpreted into a clean boundary story.

---

## Result 3: the 4-way spatial classification — C6-2 makes different errors, not simply more of them

| Category | Voxel count |
|---|---:|
| Both correct | 32,689,491 |
| **A wrong, C6-2 correct (improvement)** | **15,009** |
| **A correct, C6-2 wrong (regression)** | **17,432** |
| Both wrong | 46,068 |

Improvement/regression ratio: 0.861. Net: −2,423 voxels (mildly unfavorable, consistent with C6-2's slightly lower overall Dice). **The key qualitative finding**: C6-2 is not simply "A plus extra errors" — it corrects 15,009 voxels A got wrong while introducing 17,432 new errors of its own. This is a real, substantial two-way churn (32,441 voxels changed state in either direction), not a small perturbation on top of an otherwise-identical prediction. The net effect is close to balanced, tilted slightly unfavorable.

---

## Result 4: connected components — comparable lesion-level detection

| | A | C6-2 | Δ |
|---|---:|---:|---:|
| Missed GT components (out of 363 total) | 178 | 183 | +5 |
| False-positive components | 97 | 96 | −1 |
| Mean component Dice (subject-averaged) | 0.6076 | 0.6052 | −0.0024 |

Nearly identical at the lesion level. C6-2 misses 5 more whole lesions than A (out of 363), essentially unchanged FP component counts, and a negligible difference in mean component-level Dice. **No evidence that C6-2 disproportionately fails to detect small or fragmented lesions as distinct connected components** — the aggregate lesion-detection performance of the two models is close to indistinguishable at this level.

---

## Result 5: per-subject Dice delta — real variance, no significant structural correlate

- **Median Δ Dice**: −0.0002 (essentially zero)
- **Mean Δ Dice**: +0.0025
- **Fraction C6-2 better**: 60/125 (48.0%)
- **Fraction A better**: 65/125 (52.0%)

Nearly an even split, consistent with the near-zero median — this is not a story of C6-2 being uniformly worse across most subjects; it wins almost as often as it loses, but the subjects where it loses tend to lose slightly more (net mildly unfavorable, consistent with Result 3).

**Individual subjects show large swings in both directions**, far larger than the aggregate Dice gap:

| Top regressions | Δ Dice | Lesion vol | n components |
|---|---:|---:|---:|
| BraTS-GLI-00425-000 | −0.1721 | 729 | 2 |
| BraTS-GLI-01530-000 | −0.0931 | 489 | 1 |
| BraTS-GLI-01293-000 | −0.0832 | 3,713 | 3 |

| Top improvements | Δ Dice | Lesion vol | n components |
|---|---:|---:|---:|
| BraTS-GLI-01349-000 | **+0.4073** | 266 | 5 |
| BraTS-GLI-00731-001 | **+0.3595** | 661 | 2 |
| BraTS-GLI-00016-000 | +0.1090 | 2,127 | 4 |

The single largest swing in the entire cohort (+0.41) is an *improvement*, not a regression — C6-2's worst single-subject outcome (−0.17) is smaller in magnitude than its best (+0.41). This further undercuts a simple "C6-2 systematically damages certain subjects" narrative.

**Correlations, both Pearson and Spearman** (neither reaches significance):

| | Pearson r | p | Spearman ρ | p |
|---|---:|---:|---:|---:|
| Δ Dice vs. lesion volume | −0.175 | 0.051 | −0.112 | 0.213 |
| Δ Dice vs. lesion count (fragmentation) | −0.017 | 0.852 | +0.069 | 0.445 |

The lesion-volume correlation is borderline (p=0.051, just outside conventional significance) and weak even if real (r=−0.175 explains ~3% of variance) — a hint that C6-2 may regress *very slightly* more on smaller-lesion subjects, not a confirmed or strong effect. Fragmentation (lesion count) shows no relationship at all.

---

## Synthesis: the deficit is diffuse, not concentrated

Per the explicit instruction not to force a tidy story: **none of the five structural categories tested (slice depth, boundary distance, lesion detection at the component level, lesion size, lesion fragmentation) shows a strong, clean, monotonic relationship with where C6-2 underperforms A.** The one real signal — a boundary-adjacent (but not boundary-*immediate*) relative error excess (Result 2) — is genuine and worth carrying forward, but its non-monotonic shape (peaking one bin removed from the boundary, not at it) does not cleanly match the most natural mechanistic story ("SC-TAM's representation-geometry manipulation specifically damages boundary voxels"), and this report does not stretch it into one.

**What this experiment does establish, positively**: the Dice gap is small (about 0.33pp on this exact-checkpoint comparison), and its sources — to the extent they are visible at all in this analysis — are spread thinly across many individual subjects and voxel populations rather than concentrated in an identifiable failure mode. The per-subject variance is large relative to the aggregate effect (individual swings of ±0.17 to +0.41 Dice against a headline gap of 0.0033), meaning subject-to-subject idiosyncrasy dominates over any systematic structural pattern this analysis could detect.

---

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e25/run_c62_spatial_error_analysis.py` | Full implementation |
| `experiments/exp_e12_eggo_m/e25/spatial_error_analysis_results/spatial_error_analysis_C62_vs_A.json` | Raw per-subject records, slice/boundary aggregates, 4-way classification |
| `PHASE_E25_C62_REAL_TRAJECTORY_CONFUSION_FULL.md` | The confusion-flux analysis this experiment follows up on at the lesion/spatial level |
