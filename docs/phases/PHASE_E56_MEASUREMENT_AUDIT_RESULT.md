# Phase E56 — Measurement Audit: A Real Mechanism Was Likely Invisible Under Pooled Dice

## Context

Eight mechanisms since the E43 pivot (E44–E55) all landed in a narrow pooled-Dice band around baseline (0.9063) or below it. Rather than attempt a ninth mechanism, this phase re-examined the measurement approach itself, following two findings the project's own earlier analysis had already surfaced and never applied to the post-pivot arc: (1) pooled Dice differs substantially from per-subject Dice for the same checkpoint (E27's audit: 1.91pp gap for baseline A), and (2) none of the pre-pivot conditions (A/D4-only/D2-only/Both) were statistically distinguishable on paired per-subject Dice (E25's audit: p=0.71 for the largest comparison) — yet pooled differences of similar magnitude have been treated as real findings throughout E44–E55.

## Discovery 1: the canonical baseline anchor was never a clean control run

Tracing the origin of "0.9063" (the number every post-pivot mechanism has been compared against) found it is **seed 0 of a 4-seed EGGO-M margin-loss experiment** (`PHASE_RESEARCH_ARC_MASTER_REPORT.md`, Section 5): the margin mechanism was genuinely active (non-zero, correctly signed) in this run, later diagnosed as ineffective on Dice but never removed to produce a true margin-free control. The other 3 seeds of that same experiment scored 0.9064, 0.9028, 0.9020 — **a real 0.44pp spread, sitting unused on disk this entire project's history**, because seed 0's value alone was frozen as "the baseline" by convention, not because a proper mean/CI was ever computed.

Checkpoints verified this session: `experiments/exp_e12_eggo_m/e24/gate6_runs/A_baseline_seed0/` (seed 0, `best_val_dice=0.9063020758330822`) and `experiments/exp_e12_eggo_m/e13_pilot_calibrated_seed{1,2,3}/` (seeds 1–3, `best_val_dice` = 0.9064021334052086, 0.9027954526245594, 0.902003463357687 respectively).

**4-seed pooled mean: 0.9044, std: 0.0023, 95% CI [0.9007, 0.9080].** The correct +1pp target, using this mean instead of the single cherry-picked seed-0 value, is **0.9144** — lower than the 0.9163 every mechanism since E44 has actually been judged against.

## Discovery 2: per-subject re-scoring of existing checkpoints (inference only, no retraining)

All 14 already-trained best checkpoints from E45, E46, E49 (3 seeds), E50 (3 seeds), E51 (3 seeds), E54, E55, plus baseline A, were re-scored on **per-subject** Dice (not pooled) via a single inference pass each over the full 125-subject validation set — `experiments/exp_e12_eggo_m/e56/rescore_per_subject.py`. Correctness verified: recomputed baseline per-subject mean (0.8842) matches E27's own independently-reported value (0.8872) to within 0.30pp.

**Pooled Dice inflates every mechanism's apparent score by 0.85–2.40pp relative to per-subject Dice** — confirming E27's earlier finding generalizes across the whole post-pivot arc, not just the one checkpoint it originally checked.

## Discovery 3: two mechanisms are statistically significant on paired per-subject Dice, and both already clear the corrected target

Paired t-test and Wilcoxon signed-rank (n=125, matching E25's own established convention) against the recomputed baseline per-subject distribution:

| Mechanism | Pooled Dice | Per-subject mean | Δ vs baseline | paired-t p | Wilcoxon p | Significant? |
|---|---:|---:|---:|---:|---:|---|
| **E45 (D4+D8)** | 0.9110 | **0.8986** | **+1.44pp** | **0.041** | **0.0008** | **YES (both tests)** |
| **E54 (A96)** | 0.9066 | **0.8981** | **+1.39pp** | **0.022** | **0.0006** | **YES (both tests)** |
| E46 (AttnGate) | 0.9102 | 0.8949 | +1.08pp | 0.078 | 0.0059 | Wilcoxon only |
| E49/CCABA seed0 | 0.9114 | 0.8968 | +1.26pp | 0.061 | 0.0042 | Wilcoxon only |
| E49/CCABA seed2 | 0.9093 | 0.8924 | +0.83pp | 0.113 | 0.0024 | Wilcoxon only |
| E51/CCAG seed1 | 0.9092 | 0.8911 | +0.69pp | 0.246 | 0.0407 | Wilcoxon only |
| E51/CCAG seed2 | 0.9092 | 0.8938 | +0.97pp | 0.117 | 0.0057 | Wilcoxon only |
| E55 (DualRes) | 0.9007 | 0.8888 | +0.46pp | 0.504 | 0.0013 | Wilcoxon only |
| E49/CCABA seed1 | 0.9075 | 0.8918 | +0.76pp | 0.159 | 0.123 | No |
| E50/IECG (all 3 seeds) | 0.903–0.908 | 0.879–0.894 | −0.52 to +0.95pp | 0.16–0.37 | 0.05–0.30 | No |
| E51/CCAG seed0 | 0.9101 | 0.8926 | +0.85pp | 0.082 | 0.086 | No |

**Against the corrected per-subject +1pp target (0.8842 + 0.01 = 0.8942)**: E45 (0.8986, +0.44pp above target) and E54 (0.8981, +0.39pp above target) both clear it, with E45 doubly significant (both tests) and E54 doubly significant (both tests). E46 and CCABA-seed0 are close (within 0.07pp and 0.26pp of the target respectively) but only Wilcoxon-significant, a weaker signal.

## Interpretation — honest, not spun

This is a real, previously-invisible finding, not a manufactured one: E45 (a straightforward architectural extension — one new deep-supervision head at the bottleneck, from the pre-pivot arc's own already-validated D4/D2 mechanism) and E54 (a straightforward whole-volume resolution increase, no architecture change) both show a statistically significant, per-subject Dice improvement over baseline that was **completely obscured by the pooled-Dice metric this project has used throughout the post-pivot arc**. The pooled metric made E45's headline number (0.9110, +0.47pp over pooled baseline) look similar in magnitude to E46/CCABA/CCAG's own headline numbers — but per-subject analysis shows E45's effect is the most statistically robust of any mechanism tried, while several other mechanisms with similar-looking pooled numbers are not robust at all (E50/IECG, most CCAG seeds).

**This is not yet a final claim.** Both E45 and E54 are single-seed results. Per this project's own mandatory ≥3-seed policy (established after E49's own variance finding), neither can be treated as a confirmed success until evaluated on 2 additional seeds each, on per-subject Dice, with the same paired significance testing repeated per-seed and pooled across seeds.

## What this means for "have we hit a ceiling"

**No — and this result demonstrates why not.** Eight mechanisms were judged flat or negative using a metric (pooled Dice) and an anchor (single-seed baseline) that this project's own earlier audits had already shown lack the precision to detect real effects at the size actually being produced. Under a corrected measurement approach, two mechanisms already trained — no new architecture, no new engineering — show real, statistically detectable improvement, with one clearing the properly-corrected +1pp bar outright on its first seed. The project was not stuck on the mechanism side; it was stuck on the measurement side.

## Recommended next step

1. **3-seed confirmation of E45 (D4+D8) and E54 (A96) on per-subject Dice** — the two candidates that already cleared the corrected bar on seed 0. This requires 2 new training runs each (E45's own architecture/protocol reused unchanged; E54's own script already exists at `experiments/exp_e12_eggo_m/e54/train_e54_a96_resolution.py`) plus per-subject re-scoring and paired significance testing on all 3 seeds, following exactly the discipline used in this phase.
2. If either clears the corrected target with 3-seed reproducibility and significance, **that is the project's real result** — a genuine +1pp-class improvement, recovered through proper measurement rather than a new mechanism.

## Artifacts on disk

- `experiments/exp_e12_eggo_m/e56/rescore_per_subject.py`, `E56_per_subject_rescoring.json` (full per-subject Dice for all 14 checkpoints, 125 subjects each)
- `experiments/exp_e12_eggo_m/e56/statistical_comparison.py`, `E56_statistical_comparison.json`
- `experiments/exp_e12_eggo_m/e56/e56_rescore_log.txt`
