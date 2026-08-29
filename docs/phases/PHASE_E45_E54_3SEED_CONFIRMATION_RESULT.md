# Phase E45/E54 — 3-Seed Confirmation: Neither Mechanism Clears the Bar

## Context

`PHASE_EVIDENCE_MAP.md` identified E45 (D4+D8) and E54 (A96) as the only two mechanisms in the entire post-pivot arc with a doubly-significant (paired-t AND Wilcoxon) per-subject Dice result — both on a single seed each, never confirmed on 3 seeds per this project's own mandatory policy. This phase runs that confirmation: seeds 1 and 2 trained identically to the existing seed 0, all 3 seeds re-scored on per-subject Dice, and evaluated against two pre-declared, non-negotiable criteria — the corrected per-subject target (baseline per-subject mean 0.8842 + 1pp = **0.8942**) and the original stronger pooled target (**0.9163**).

## Result

| Mechanism | Seed | Pooled Dice | Per-subject mean | Paired-t p | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| **E45 (D4+D8)** | 0 | 0.9110 | 0.8986 | 0.041 | 0.0008 |
| | 1 | 0.9058 | 0.8878 | 0.588 | 0.085 |
| | 2 | 0.9089 | 0.8952 | 0.088 | 0.0010 |
| **E54 (A96)** | 0 | 0.9066 | 0.8981 | 0.022 | 0.0006 |
| | 1 | 0.9061 | 0.8930 | 0.160 | 0.0019 |
| | 2 | 0.9060 | 0.8922 | 0.197 | 0.0073 |

**3-seed summary:**

| Mechanism | 3-seed mean per-subject Dice | 95% CI | 3-seed mean pooled Dice | Pooled-across-seeds paired-t p (n=375) | Pooled-across-seeds Wilcoxon p |
|---|---:|---|---:|---:|---:|
| E45 | **0.8939** | [0.8802, 0.9076] | 0.9086 | 0.013 | 1.3×10⁻⁶ |
| E54 | **0.8944** | [0.8864, 0.9024] | 0.9062 | 0.004 | 1.1×10⁻⁷ |

**Patient-level ΔDice distribution** (pooled across all 3 seeds × 125 subjects, n=375 pairs):
- E45: mean +0.0097, median +0.0041, std 0.0748, 5th–95th percentile [−0.0348, +0.0395]
- E54: mean +0.0103, median +0.0042, std 0.0685, 5th–95th percentile [−0.0366, +0.0497]

## Verdict against the pre-declared criteria (checked exactly as specified, not adjusted)

| Criterion | E45 | E54 |
|---|---|---|
| Corrected target: 3-seed mean per-subject Dice ≥ 0.8942 | 0.8939 — **NOT MET** (by 0.0003) | 0.8944 — **MET** (by 0.0002) |
| Original target: 3-seed mean pooled Dice ≥ 0.9163 | 0.9086 — **NOT MET** | 0.9062 — **NOT MET** |

## Honest interpretation

**Neither mechanism clears the original, stronger target** — both land firmly in the 0.906–0.909 pooled range, nowhere near 0.9163.

**On the corrected, per-subject target, the two mechanisms land on opposite sides of the threshold — but the margin (0.0003 for E45, 0.0002 for E54) is smaller than a tenth of the mechanisms' own between-seed standard deviation (0.0055 and 0.0032 respectively).** This is not a meaningful pass/fail distinction; it is a coin-flip-level tie against the threshold for both. Reporting E54 as "confirmed" and E45 as "not confirmed" on this basis would overstate the precision of the underlying measurement — the honest reading is that **both mechanisms are statistically indistinguishable from the corrected target itself**, not that one succeeded and one failed.

**Seed 0 was the most favorable seed for both mechanisms**, exactly the risk flagged before this confirmation was run. Neither mechanism's doubly-significant seed-0 signature reproduced cleanly on seeds 1 or 2 — both show a weaker, more mixed pattern (Wilcoxon-significant but not paired-t-significant on at least one of the two new seeds each). The only test that comes back cleanly significant for both mechanisms is the pooled-across-all-375-pairs test — but that establishes a much weaker claim ("some non-zero average effect exists across 3 seeds combined") than the pre-declared target ("a reproducible, ≥1pp-class effect on 3 independently-run seeds"), and should not be substituted for the actual criterion.

## Decision

**Per the pre-declared, non-adjusted criteria: neither E45 nor E54 is confirmed.** Both fail the original target outright. Neither clears the corrected target with a margin distinguishable from noise. This closes the last open empirical thread identified in `PHASE_EVIDENCE_MAP.md` — the question "did we already have a reproducible >1pp improvement hiding in prior experiments?" is answered: **no.**

## What this means for the project

Combined with the full E44–E60 arc, this result completes a genuinely thorough search: architecture-level mechanisms (E45, E46, E49–E51, E55), objective-level mechanisms (E44, E52, E53), a resolution-level intervention (E54), and two rounds of causal-diagnostic-derived hypotheses (E58's compatibility test, E59/E60's synergy test) have all been tried, and the two candidates that looked most promising under the corrected measurement framework do not survive 3-seed confirmation with a defensible margin. This is not a failure of search effort — it is a real, negative result produced with the discipline (3 seeds, per-subject Dice, paired significance testing, pre-declared non-adjusted thresholds) this project's own methodology work (E56) established was necessary to trust any claim in either direction.

The project's defensible contribution remains the causal-diagnostic chain (E43→E47→E48→E58→E59→E60) and the measurement-methodology finding (E56) — not a confirmed algorithmic improvement over baseline.

## Artifacts on disk

- `experiments/exp_e12_eggo_m/e56/rescore_e45_e54_confirmation.py`
- `experiments/exp_e12_eggo_m/e56/E45_E54_seed12_rescoring.json`
- `experiments/exp_e12_eggo_m/e56/E45_E54_3seed_confirmation_summary.json`
- `experiments/exp_e12_eggo_m/e45/runs/D4_D8_seed{1,2}/`, `experiments/exp_e12_eggo_m/e54/runs/A96_seed{1,2}/` (checkpoints, disk only, gitignored)
