# Phase E69 — E54 Offline Mining (Staged Compute Gate, Step 1)

## Purpose

Per explicit user instruction: before spending ~24 GPU-hours on 7 additional
E54 seeds to resolve whether E54's marginal 3-seed per-subject-Dice "pass"
(+0.0002 over the corrected threshold, ~1/10th of its own between-seed
standard deviation) is real, mine the **already-trained** 3 seeds for whether
a coherent phenomenon exists. Zero new training or inference — pure analysis
of existing artifacts (`E56_per_subject_rescoring.json`,
`E45_E54_seed12_rescoring.json`, E48/E35's existing per-subject tables,
existing training logs).

## E69: cross-seed consistency and size-dependence

- **Per-seed summary**: all 3 seeds individually show a modest positive mean
  delta (+0.008 to +0.014) with wide per-subject spread (std ≈0.067–0.069) —
  consistent with the project's own earlier finding that seed-level noise
  dominates the small mean effect.
- **Cross-seed sign consistency** (the key new analysis): 52/125 subjects
  (41.6%) **consistently improve across all 3 independently-trained seeds**,
  versus the ~25% expected under pure chance (independent per-seed coin
  flips) — binomial test p<0.0001. 13/125 (10.4%) consistently degrade. This
  is a genuine, non-trivial signal: a mechanism with zero true effect would
  not be expected to produce this much cross-seed agreement at the individual
  subject level.
- **Size-dependence (pooled, n=375)**: Spearman(native_size, delta) = −0.118,
  p=0.023 — nominally significant, matching the E48/E65 size-specificity
  direction (smaller lesions benefit more). **However, this does not hold up
  under scrutiny**: none of the 3 seeds shows a significant size correlation
  individually (p=0.075, 0.546, 0.133) — the pooled significance is a
  sample-size artifact of tripling n, not a robust per-seed effect. A direct
  check (E69b) found the consistently-improved group's median native size
  (97,805) is actually **larger**, not smaller, than the consistently-degraded
  group's (77,815) — the opposite of the size-specificity story — and the
  difference is not significant either way (p=0.64).
- **Training trajectories**: E54's 3 seeds and the matched baseline converge
  at similar speed (first val_dice>0.85 at epoch 4–5 for all) and reach
  similar final val_dice (0.905–0.907), with comparable late-training
  epoch-to-epoch variance. No visibly distinct optimization signature at the
  trajectory level.

## E69b: covariate search for what DOES distinguish the consistent groups

Since lesion size does not explain the cross-seed-consistent effect, four
pre-registered candidate covariates (native size, baseline Dice, bottleneck
causal-ablation drop from E48, lesion component count from E35) were tested
via Kruskal-Wallis (3-group) and Mann-Whitney (improved vs. degraded), all
reused from existing per-subject tables, zero new computation.

| Covariate | Improved (median) | Degraded (median) | KW p | MWU (imp. vs deg.) p |
|---|---|---|---|---|
| Native lesion size | 97,805 | 77,815 | 0.801 | 0.640 |
| **Baseline Dice (dice_intact)** | **0.927** | **0.898** | 0.114 | **0.020** |
| Bottleneck causal drop | 0.323 | 0.359 | 0.504 | 0.321 |
| Component count | 3 | 4 | 0.181 | 0.495 |

**Baseline Dice is the only covariate showing a nominal difference**:
subjects already easy for the baseline (higher `dice_intact`) are more likely
to consistently improve with A96; subjects already hard for the baseline are
more likely to consistently degrade. This is an interpretable pattern — it
would suggest A96 helps refine already-good segmentations rather than
rescuing hard cases, the reverse of what a "helps small/difficult lesions"
story would predict.

**This does not survive multiple-comparisons correction.** 4 covariates × 2
tests = 8 comparisons; Bonferroni-corrected α = 0.00625. The baseline-Dice
MWU p=0.020 does not clear this bar. Reported honestly as **suggestive, not
confirmed**.

## Decision (staged compute gate)

Neither a clean **A** (pure noise, no coherent pattern at all) nor a clean
**C** (strong, well-supported, size-structured pattern justifying full
replication) applies. The honest classification is **B**: a real, statistically
non-trivial cross-seed-consistent subject-level effect exists (41.6% vs. 25%
chance, p<0.0001 — this part is solid), but:

1. It is **not explained by lesion size** (the a priori hypothesis from
   E48/E65), contrary to the initial pooled-correlation read.
2. The one covariate that shows a directional difference (baseline Dice)
   does not survive correction for multiple comparisons.
3. Training trajectories show nothing distinctive.

## Recommendation

**Do not launch the full 7-seed (~24 GPU-hour) replication yet.** The
existing evidence supports a narrower, cheaper next step if this thread is
pursued further: a **pre-registered, single-hypothesis test of the baseline-
Dice pattern specifically** (not a fishing expedition across 4 covariates) —
either on the existing 3 seeds' finer-grained data (e.g., a continuous
correlation rather than a group split) or, if a 4th data point is wanted, a
single additional seed (~3.4 GPU-hours, not 24) to see whether the baseline-
Dice pattern replicates before committing to the full 7-seed run.

## Artifacts

- `experiments/exp_e12_eggo_m/e69/run_e69_e54_offline_mining.py`
- `experiments/exp_e12_eggo_m/e69/E69_summary.json`
- `experiments/exp_e12_eggo_m/e69/run_e69b_covariate_search.py`
- `experiments/exp_e12_eggo_m/e69/E69b_covariate_search_summary.json`
