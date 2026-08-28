# Phase E60 — DBC Feasibility Test: Killed, Phenomenon Does Not Reproduce Under the Training-Time Formulation

## Context

E59 found real, significant, size-specific interaction in the bottleneck using **pairwise singleton interactions** (I_ij across all 28 pairs of 8 octant groups): small lesions showed significant positive synergy (mean I_ij=+0.00104, p=0.049), large lesions showed significant negative interaction/redundancy (mean I_ij=−0.00192, p<0.0001). This motivated Distributed Bottleneck Coding (DBC), a proposed training objective that would preserve this synergy using a cheaper, training-time-feasible sampling scheme: **balanced complementary partitions** (`A`, `B=A^c`, each of size 4) rather than all 28 pairs.

Per the user's own explicit instruction, before any training was attempted, this phase verified E59's phenomenon reproduces under DBC's **exact intended formulation** — `Γ(A,B) = L(A) + L(B) − L(A∪B) − L(∅)`, evaluated on balanced 4-vs-4 partitions, using the actual loss (not the utility/−loss convention E59 used). If it did not reproduce, DBC was to be killed with zero training cost. Bit-for-bit sanity check passed (max abs diff 0.0) before trusting any result.

## Result

Six random balanced partitions sampled per subject (of 35 possible), same 125 validation subjects, same checkpoint as E59.

| Group | n | Mean Γ | vs. 0 (t-test) |
|---|---:|---:|---:|
| Small lesions (below median) | 62 | −0.00615 | t=−0.632, **p=0.530** |
| Large lesions (at/above median) | 63 | −0.03809 | t=−10.028, **p<0.0001** |

Size-correlation: Spearman(native_size, mean Γ) = **−0.287**, p=0.001 (parametric), p=0.001 (permutation) — real and significant, but **weaker** than E59's own ρ=−0.498/−0.498 (interaction/residual) and, critically, **not accompanied by a significant small-lesion effect on its own**.

**The small-lesion synergy signature does not reproduce.** Mean Γ for small lesions is not significantly different from zero (p=0.530) — the entire premise DBC's loss term was built on. Large lesions show a strong, significant negative Γ, qualitatively consistent with E59's own redundancy finding for large lesions, but that is not the branch DBC's first implementation was designed to exploit.

## Decision

**Per the pre-declared rule: DBC is killed here.** The phenomenon that motivated it — measurable, exploitable, small-lesion-specific positive synergy — does not survive the switch from E59's exhaustive pairwise-singleton sampling to DBC's own intended cheaper balanced-partition sampling. No training was run; this is a zero-cost, purely diagnostic result, exactly as designed.

## Why this happened — honest speculation, not an excuse

Two candidate explanations, disclosed rather than adjudicated:

1. **Sampling-scheme sensitivity, not a real discrepancy in the underlying phenomenon.** Pairwise singleton interactions (E59) isolate the *marginal* contribution of adding one octant to another single octant, holding the rest at zero. Balanced 4-vs-4 partition Γ measures something coarser — the interaction between two already-large coalitions (4 groups each), where each half already contains substantial information on its own (per E58 Stage 1, even a single octant does *some* real work). The small-lesion synergy E59 found may be a genuinely *fine-grained* phenomenon (specific pairs of octants complementing each other) that gets washed out when measured at the coarser 4-vs-4 granularity — six sampled 4-vs-4 partitions, drawn at random, may simply not isolate the specific complementary pairs E59's exhaustive pairwise sweep found.
2. **The E59 finding itself, while statistically real at the significance levels reported, may be a smaller, more fragile effect than its headline p-values suggested** — real enough to detect with an exhaustive 28-pair sweep and large aggregate sample, but not robust enough to survive a coarser, lower-resolution measurement of essentially the same underlying quantity.

Distinguishing these would require testing more/all 35 balanced partitions (not just 6) and/or partition sizes closer to E59's own granularity (e.g. 1-vs-7, 2-vs-6) before concluding the phenomenon is entirely illusory — but per the user's own explicit stop-rule ("if it doesn't reproduce, KILL DBC here, no training"), that further tuning is exactly the kind of post-hoc rescue this project's own discipline exists to prevent. The pre-declared rule is honored as written.

## Recommendation

DBC (Distributed Bottleneck Coding), as specified, is closed. This is the ninth mechanism-adjacent hypothesis killed or nulled since the strategic pivot (counting E44–E55, E59→E60 as the ninth and tenth diagnostic/mechanism steps respectively), and the second time in two consecutive phases (E58 Stage 3, E60) that a promising-looking causal signal did not survive its own direct confound/feasibility check. This continues to support the project's standing position: the defensible contribution is the causal-diagnostic methodology itself (E43→E47→E48→E58→E59→E60, a genuine, rigorous chain of hypothesis-and-test even where most hypotheses were killed) and the measurement-correction finding (E56), not a working novel algorithm recovered from this line of investigation.

## Artifacts on disk

- `experiments/exp_e12_eggo_m/e60/run_e60_dbc_feasibility.py`
- `experiments/exp_e12_eggo_m/e60/E60_feasibility_table.json`
- `experiments/exp_e12_eggo_m/e60/E60_summary.json`
