# E147 Representational-Demand Mining

**Date**: 2026-09-14
**Status**: Existing-artifact analysis only. No model inference, intervention, or training was run.

**Methodology recovery update (2026-09-14)**: The related E126 causal protocol has been recovered, but it is not the generator of the saved `Rstar_self` fields. E126 validates graded rank reduction as a causal perturbation of enc3 windows and bottleneck necessity; it does not define the E147 minimum-rank threshold.

**THRESHOLD RULE RECOVERED (2026-09-15)** — *partial recovery; see the caveat below.* `Rstar_self` is the first point on the dyadic grid `[1,2,4,8,16,32,64,128,256]` at which the stored `self_ag` agreement curve reaches **0.90**. This matches **88/88** subjects at threshold 0.90 and at no other threshold (0.85 → 66/88; 0.92 → 81/88; 0.95 → 60/88), and recomputing this doc's headline statistics from the recovered rule reproduces them to 4 decimals (`rho(ap,R*) = -0.6385` vs `-0.638`; `rho(baseDice,R*) = -0.4369` vs `-0.437`). Limitation 4 is therefore **resolved** and limitation 2 **partially** so.

**CAVEAT — what is still missing.** Only the *last step* of the pipeline is recovered. A repo-wide search for `self_ag`/`Rstar` across all `*.py` returns **zero hits**; there is no `e146`/`e147`/`e148` directory; and `git log --all` matching `e14[0-9]|e15[0-9]` returns only this very doc. **The generator was never committed and does not exist on disk.** So it remains UNKNOWN what `self_ag` actually measures (agreement between what and what), where the truncation is applied, and by what metric. `Rstar_gt` has no stored curve at all (`gt_ag` absent) and is recovered only by analogy — it should carry no weight. Consequently `R*` **cannot currently be recomputed** on any other checkpoint, which blocks the checkpoint-invariance test. See `PHASE_E154_RSTAR_RESPECIFICATION_PREREG.md` Parts 0/0b for the full audit, the threshold-sensitivity numbers, and two new facts it surfaces (`R*` is a 6-valued ordinal, not a continuous rank; 31/88 agreement curves are non-monotonic, max dip 0.179).

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
2. ~~The JSON does not record the exact epsilon, target definition, rank grid semantics, or the script that produced `Rstar_self` and `Rstar_gt`.~~ **RESOLVED 2026-09-15** — the rule (dyadic grid, first crossing of 0.90) is recovered and verified 88/88 against the stored curves. Note `Rstar_gt` has no stored curve (`gt_ag` absent), so it is recovered by analogy only and should not carry weight. **New limitation in its place**: `R*` is a **6-valued ordinal** on a dyadic grid — the apparent `1..32` range is five steps, so all statistics must be rank-based, and the spread is weaker evidence of heterogeneity than a continuous `1..32` range would be.
3. `Rstar` is a representational rank intervention, not a direct FLOP or dynamic-depth measurement. It should be called a **minimum retained-rank requirement**, not minimum computation, until that distinction is addressed.
4. ~~The analysis uses saved summaries rather than recomputing subject-level curves, so threshold sensitivity cannot be audited here.~~ **RESOLVED 2026-09-15** — threshold sensitivity audited: perturbing the threshold to 0.88/0.92 changes `R*` for only 8/88 and 7/88 subjects, with Spearman 0.949 and 0.906 against the 0.90 assignment. The exact value 0.90 is not load-bearing.

5. **NEW (2026-09-15)**: `R*` has not been shown to be **checkpoint-invariant**. E129 established that `N_k` magnitudes are run-dependent across equally-good checkpoints; if `R*` inherits that, it is a trajectory property, not a subject property, and Gate A's GREEN is unearned. This is now the blocking test — see `PHASE_E154_RSTAR_RESPECIFICATION_PREREG.md`. Until it passes, **Gate A should be read as YELLOW, not GREEN**, and Gate B inherits that qualification because `rho_partial = -0.496` is a correlation with the unvalidated half of the pair (`O_i` itself is solid — E143/E144 cross-fitted, pairwise rho 0.786–0.974).

## Current conclusion

The fixed-contract hypothesis is **not killed**. It has a real, artifact-backed signal: rank requirement spans 1 to 32 and remains associated with image-only observability after baseline control. E126 additionally provides causal dose-response evidence that graded enc3 rank reduction changes bottleneck necessity, with intervention-alone cost below 0.02 at alpha 0.95, 0.85, and 0.70. The broader adaptive-contract hypothesis remains **YELLOW** until the exact E147 generator, threshold, and rank semantics are recovered.

## Recovered E126 artifacts

- [E126 protocol script](../../experiments/exp_e12_eggo_m/e126/run_e126_graded_rank_reduction_causal_test.py)
- [E126 summary](../../experiments/exp_e12_eggo_m/e126/E126_summary.json)
- [E126 per-subject table](../../experiments/exp_e12_eggo_m/e126/E126_per_subject_table.json)
- [E126 run log](../../experiments/exp_e12_eggo_m/e126/run_log.txt)
