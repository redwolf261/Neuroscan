# E147 Representational-Demand Mining

**Date**: 2026-09-14
**Status**: Existing-artifact analysis only. No model inference, intervention, or training was run.

**Methodology recovery update (2026-09-14)**: The related E126 causal protocol has been recovered, but it is not the generator of the saved `Rstar_self` fields. E126 validates graded rank reduction as a causal perturbation of enc3 windows and bottleneck necessity; it does not define the E147 minimum-rank threshold.

## Question

Does the minimum saved rank requirement vary across subjects, and is that variation explained by observable information properties rather than only by baseline difficulty?

## Artifacts mined

- `experiments/exp_e12_eggo_m/E147_repdemand.json`
- `experiments/exp_e12_eggo_m/E148_probes.json`

The E147 artifact contains 88 subjects with `Rstar_self`, `Rstar_gt`, `base_gt`, `ap`, and rank-response curves. E148 contains matched subject-level `model_ap` and probe metrics for `dec1`, `dec2`, and `dec3`.

The recovered E126 protocol uses the v5/E46 checkpoint and blends every enc3 2x2x2 window as `x' = alpha*x + (1-alpha)*window_mean` for `alpha={0.95,0.85,0.70,0.50}`. It defines bottleneck necessity as Dice(intact bottleneck) minus Dice(ablated bottleneck), checks intervention-alone Dice cost against `0.02`, and confirms a monotonic dose response. E126 is a causal rank-reduction validation, not the E147 `Rstar` thresholding script.

## Distribution

### `Rstar_self`

| Rank | Subjects |
|---:|---:|
| 1 | 14 |
| 2 | 17 |
| 4 | 43 |
| 8 | 7 |
| 16 | 4 |
| 32 | 3 |

- `n = 88`
- mean `= 4.95`
- median `= 4`
- range `= 1..32`
- 74/88 subjects are at rank `<=4`; 14/88 are at rank `>=8`.

### `Rstar_gt`

- counts: rank 1 = 16, rank 2 = 24, rank 4 = 39, rank 8 = 4, rank 16 = 3, rank 32 = 2
- mean `= 4.14`
- median `= 4`
- range `= 1..32`

The saved data therefore rejects the homogeneous-demand premise at face value, although the rank grid is discrete and metadata for the exact target tolerance is not present in the JSON artifact.

## Difficulty and observability checks

| Relationship with `Rstar_self` | Statistic |
|---|---:|
| Baseline Dice, Spearman | `rho = -0.437`, `p = 2.08e-5` |
| Image-only AP `ap`, Spearman | `rho = -0.638`, `p = 2.20e-11` |
| Image-only AP, partial Spearman controlling baseline Dice | `rho = -0.496`, `p = 8.74e-7` |
| Model AP, Spearman | `rho = -0.387`, `p = 1.94e-4` |
| Model AP, partial Spearman controlling baseline Dice | `rho = 0.009`, `p = .934` |
| `dec1` probe AP, Spearman | `rho = -0.371`, `p = 3.75e-4` |
| `dec2` probe AP, Spearman | `rho = -0.307`, `p = .00364` |
| `dec3` probe AP, Spearman | `rho = -0.314`, `p = .00286` |

The key result is that rank demand is not merely a restatement of baseline Dice: image-only observability retains a strong association after baseline control. The corresponding model-AP association disappears after baseline control, which is consistent with image observability being the more specific signal in this saved analysis.

## Preliminary decision

**GREEN, provisional**: the existing artifacts support heterogeneous representational demand that is systematically associated with an image-only observability measure beyond baseline difficulty.

This is evidence for a conditional computational-demand hypothesis, not yet evidence for a deployable adaptive architecture. It earns the next question:

> Can the required representational contract be predicted before the expensive representation is computed?

## Limitations blocking a final claim

1. Only 88 subjects are present in E147, not the full 125-subject validation set.
2. The JSON does not record the exact epsilon, target definition, rank grid semantics, or the script that produced `Rstar_self` and `Rstar_gt`. The E126 script is related evidence, but it does not resolve these missing E147 details.
3. `Rstar` is a representational rank intervention, not a direct FLOP or dynamic-depth measurement. It should be called a **minimum retained-rank requirement**, not minimum computation, until that distinction is addressed.
4. The analysis uses saved summaries rather than recomputing subject-level curves, so threshold sensitivity cannot be audited here.

## Current conclusion

The fixed-contract hypothesis is **not killed**. It has a real, artifact-backed signal: rank requirement spans 1 to 32 and remains associated with image-only observability after baseline control. E126 additionally provides causal dose-response evidence that graded enc3 rank reduction changes bottleneck necessity, with intervention-alone cost below 0.02 at alpha 0.95, 0.85, and 0.70. The broader adaptive-contract hypothesis remains **YELLOW** until the exact E147 generator, threshold, and rank semantics are recovered.

## Recovered E126 artifacts

- [E126 protocol script](../../experiments/exp_e12_eggo_m/e126/run_e126_graded_rank_reduction_causal_test.py)
- [E126 summary](../../experiments/exp_e12_eggo_m/e126/E126_summary.json)
- [E126 per-subject table](../../experiments/exp_e12_eggo_m/e126/E126_per_subject_table.json)
- [E126 run log](../../experiments/exp_e12_eggo_m/e126/run_log.txt)
