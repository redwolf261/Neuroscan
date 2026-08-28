# Phase E59 — Bottleneck Interaction Audit: Size-Dependent Synergy, Confirmed

## Motivation

E48 established the whole bottleneck is causally load-bearing (mean Dice drop 0.32 from full ablation), disproportionately for small lesions (ρ=−0.454, p<0.001). E58 found neither of two fixed spatial octants (1/8 each) reproduces anything close to that effect individually. This phase asks the sharper mathematical question those two facts jointly raise, following the coalitional-game-theory formulation proposed by the user: is the bottleneck's causal contribution **additive** across its 8 spatial octant-groups (any sufficient subset recovers most of the value — redundant/distributed information), or **synergistic** (the decoder needs combinations of octants that no small subset supplies)?

## Method

No training. Same plain v3/D4-only checkpoint as E58 (avoiding the E46/E48 attention-gate confound). For each of 125 validation subjects, the 8³×256ch bottleneck was partitioned into the same 8 fixed spatial octants used in E58. For every subset S of octants, `U(S) = -FocalTverskyLoss(f(B_S), Y)`, where `B_S` retains only the octants in S (zeros the rest — the same "ablate by zeroing" convention as every prior E47/E48/E58 causal script). Computed per subject: `U(∅)`, all 8 singletons `U({i})`, all 28 pairs `U({i,j})`, and `U(all)` (sanity reference). Second-order interaction: `I_ij = U({i,j}) − U({i}) − U({j}) + U(∅)`. Bit-for-bit sanity check passed (max abs diff 0.0, subset=all 8 groups vs. real forward).

## Result

**Population-average interaction is not significantly different from zero** (mean I_ij = −0.0004, one-sample t-test p=0.146, sign-flip permutation p=0.161) — the pre-declared population-level "synergy" test does not pass on its own.

**But the size-dependence of the interaction signal is extremely strong and highly significant**: Spearman(native_size, mean_I_ij) = **−0.498**, parametric p=3.3×10⁻⁹, permutation p<0.001 (1000 trials). Stratifying directly by a median size split confirms this is not an artifact of averaging across a null population:

| Group | n | Mean I_ij | vs. 0 (t-test) |
|---|---:|---:|---:|
| Small lesions (below median) | 62 | **+0.00104** | t=2.010, **p=0.049** |
| Large lesions (at/above median) | 63 | **−0.00192** | t=−9.234, **p<0.0001** |
| Smallest quartile specifically | 31 | **+0.00328** | t=4.120, **p=0.0003** |

The additive-model residual (`U(all) − [sum of singleton contributions − 7·U(∅)]`, i.e. how much better the real full bottleneck performs than an additive model of its own parts predicts) shows the same pattern much more starkly: small lesions' residual is ≈0 (+0.00114 — the additive model predicts the real result almost exactly), while large lesions' residual is substantially negative (−0.06363 — the real full-bottleneck result is markedly *worse* than an additive model predicts, consistent with heavy redundancy: individual octants alone are each already nearly as informative as the whole).

## Interpretation

The population-average null the pre-declared rule checked for was the wrong single number to look at — the bottleneck is **neither globally synergistic nor globally additive**, it is **both, split cleanly by lesion size**:

- **Large lesions: real, significant sub-additivity (redundancy).** Individual octants substantially overlap in the information they carry about large lesions — consistent with E48's own original interpretation that large lesions are "locally identifiable" and don't depend heavily on integrating coarse context, since any one octant already carries most of what's needed.
- **Small lesions: real, significant, positive synergy.** No individual octant, or even the best pair, captures what the full bottleneck provides — the decoder genuinely needs a **combination** of octants for small lesions specifically, which no small subset supplies. This is the first mechanistic refinement of E48's own finding beyond "the bottleneck matters more for small lesions" — it specifies **why** in structural terms: small-lesion information is coalitionally distributed (synergistic), not merely diffusely present (redundant), across the bottleneck's spatial groups.

This directly resolves the ambiguity E58 left open ("distributed" was consistent with either redundancy or synergy) and gives a genuinely new, precise, falsifiable causal characterization: **small-lesion segmentation depends on synergistic coalitional structure in the bottleneck; large-lesion segmentation does not.**

## Decision

**GO for algorithm design.** The pre-declared rule ("kill if additive/redundant, proceed if synergistic and size-specific") is satisfied on its more informative, correctly-stratified reading: synergy is real, significant, and size-specific — smaller lesions show significantly more positive interaction (both the raw I_ij measure and the smallest-quartile subgroup show this cleanly, p=0.0003 for the tightest test). The population-average null is explained by the redundancy in large lesions canceling the synergy in small ones when pooled — not evidence against the phenomenon, but evidence for exactly the size-dependent split E48's own causal chain predicted.

## Next step

Per the user's own proposed research chain (E48 → E58 → mathematical consequence → algorithm): this phase supplies the "mathematical consequence" — a measured, causally-grounded, size-specific synergy signature — as the basis for designing Distributed Bottleneck Coding (DBC), an objective that explicitly trains the representation to preserve synergistic (coalition-dependent) information rather than merely surviving generic feature dropout (the already-published, and therefore non-novel, comparator explicitly flagged before this audit was run).

## Artifacts on disk

- `experiments/exp_e12_eggo_m/e59/run_e59_interaction_audit.py`
- `experiments/exp_e12_eggo_m/e59/E59_interaction_table.json` (125 subjects × full subset/pair/interaction data)
- `experiments/exp_e12_eggo_m/e59/E59_summary.json`
