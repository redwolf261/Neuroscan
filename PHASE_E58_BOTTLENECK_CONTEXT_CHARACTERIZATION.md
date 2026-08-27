# Phase E58 — Characterizing E48's Bottleneck Dependence: Diagnostic Chain (Not a Novelty Claim)

## Purpose and framing (explicit, per pre-declared instruction)

E48 established, causally, that small lesions depend more on the bottleneck than large lesions (full-bottleneck-zeroing ablation, rho(native_size, drop)=−0.454, p<0.001, on E46's attention-gate checkpoint). E48 never established *why*, or *what kind* of information the bottleneck carries that produces this effect. This phase runs a sequence of targeted causal interventions to characterize that content precisely. **None of the results below are themselves a novelty claim.** Per explicit instruction, this phase determines *where to search next* (or whether the line of investigation should be closed), not what to build. A separate literature-novelty search is required before any mechanism is designed around any of these findings.

**Checkpoint used for all three stages**: `experiments/exp_e12_eggo_m/e25/deep_sup_runs/DeepSup_D4only_seed0` (plain `UNet3D_v3`, no attention gate) — deliberately different from E48's own E46/v5 checkpoint, to avoid the confound where E48's attention gate re-derives itself from the (possibly ablated) bottleneck, conflating the bottleneck's direct decoder contribution with its indirect gate-modulation contribution. **This means rho values in this phase are not perfectly comparable to E48's own −0.454 reference** (different checkpoint, different architecture) — reported as a real limitation, not glossed over.

## Stage 1 — Fixed spatial-octant ablation

Two fixed, diagonally-opposite octants (1/8 each) of the 8³ bottleneck grid were zeroed independently, same subjects, same checkpoint. Bit-for-bit sanity check passed (max abs diff 0.0) before trusting any result.

| Condition | Mean Dice drop | Spearman rho(size, drop) | Permutation p | 95% CI | Overlaps E48 ref (−0.454)? |
|---|---:|---:|---:|---|---|
| Full ablation (E48 reference, different checkpoint) | 0.3205 | −0.454 | <0.001 | — | — |
| Octant A | 0.0195 | −0.186 | 0.031 | [−0.352, +0.004] | No |
| Octant B | 0.0161 | −0.081 | 0.377 | [−0.250, +0.107] | No |

**Neither fixed octant reproduces the full-ablation signature.** Octant A shows a real but much weaker effect; Octant B shows none. Six additional octants were deliberately not tested (per explicit discussion) — two diagonally-opposite corners already answer the first-order question ("is the effect concentrated in one arbitrary corner? No") without needing a full 8-way partition to be informative for the next, sharper question.

## Stage 1b — Lesion-conditioned octant ablation

For each subject, the octant containing their own lesion's centroid was zeroed, versus the octant geometrically farthest from it (per-subject, not a fixed pair) — isolating whether the bottleneck's relevant content is organized *relative to* lesion location, distinct from Stage 1's absolute-coordinate test. Bit-for-bit sanity check passed.

| Condition | Mean Dice drop | Spearman rho(size, drop) | Permutation p |
|---|---:|---:|---:|
| Lesion-containing octant | 0.0284 | +0.126 | 0.175 |
| Far octant | 0.0215 | −0.089 | 0.314 |

Paired comparison: lesion-octant drop is not significantly larger than far-octant drop (paired-t p=0.271, Wilcoxon p=0.345). The lesion-octant's own rho is not closer to the E48 reference — if anything, it's the wrong sign.

**Lesion-relative organization is not supported.** Combined with Stage 1, the bottleneck's causal content does not concentrate in any spatial subset tested — absolute or lesion-relative — consistent with (but not proof of) distributed representation.

## Stage 2 — Cross-subject bottleneck substitution

For each subject, three interventions compared against intact: bottleneck zeroed (reference, matching E48's own method), substituted with a size-matched different subject's own cached bottleneck, and substituted with a random different subject's bottleneck. Two independent bit-for-bit sanity checks passed: (1) split encoder/decoder reimplementation matches real forward exactly, (2) self-substitution (a subject's own cached bottleneck spliced back in) matches real forward exactly.

| Condition | Mean Dice drop | Spearman rho(size, drop) | Permutation p |
|---|---:|---:|---:|
| Zero | 0.2554 | −0.312 | 0.001 |
| Size-matched donor | 0.2700 | −0.001 | 0.996 |
| Random donor | 0.3098 | −0.143 | 0.106 |

Paired comparisons (all vs. zero, and against each other):

| Comparison | Mean difference | paired-t p | Wilcoxon p |
|---|---:|---:|---:|
| Zero vs. size-matched donor | −0.0146 (donor worse) | 0.560 | 0.627 |
| Zero vs. random donor | −0.0545 (donor worse) | **0.026** | 0.061 |
| Size-matched vs. random donor | −0.0399 (random worse) | 0.133 | 0.126 |

**A random foreign subject's bottleneck produces significantly worse Dice than zeroing the bottleneck entirely** (p=0.026, paired t-test; Wilcoxon p=0.061, a weaker but directionally consistent secondary test). Size-matched donor sits between zero and random donor but is not significantly different from either.

**Critical limitation, disclosed plainly**: none of the three substitution conditions reproduce E48's original size-dependent signature as strongly as the zero condition's own rho (−0.312, itself weaker than E48's −0.454 reference on the different checkpoint). Size-matched donor shows essentially zero size-correlation (rho≈0, p=0.996); random donor shows a non-significant weak correlation (rho=−0.143, p=0.106). **The "actively misleading" effect (random donor worse than zero) is real and significant, but it does not carry the same small-lesion-specific signature that made E48's original finding notable.**

## Honest overall interpretation

1. The bottleneck's causal contribution to Dice does not localize to any tested spatial subset (absolute or lesion-relative octant) — a real, consistent null across two independent tests.
2. Wrong context (a foreign subject's bottleneck) is causally worse than no context (zero) — a real, significant, and mildly surprising finding: the decoder does not treat a plausible-but-wrong coarse representation as "no information," it treats it as actively misleading signal.
3. This "actively misleading" effect, however, is **not itself the small-lesion-specific phenomenon** E48 established — it appears to be a general property of the bottleneck pathway (wrong context hurts), not a size-dependent one specifically. The two causal findings (E48's size-dependence, E58's misleading-donor effect) are both real but do not yet compose into one clean, unified story.

## What this means for next steps (explicitly not a novelty claim)

Per the pre-declared framing, this phase's job was to determine *where to search*, not to produce a mechanism. The result is genuinely informative but doesn't point at one single clean target:

- The "distributed, not localized" finding (Stages 1/1b) rules out spatially-targeted mechanisms (crop/attend/route to a specific coarse region) as a promising direction — consistent with, and further explaining, why E46's attention gate (spatial routing) and E55's dual-resolution crop (spatially-targeted local refinement) both landed flat-to-negative.
- The "wrong context is worse than none" finding (Stage 2) is real but doesn't decompose along the size axis the way E48's original finding did, so it does not yet sharpen the small-lesion question specifically — it may be a separate, general property of this architecture's bottleneck pathway worth investigating on its own terms, decoupled from the small-lesion framing that has driven the entire post-pivot arc since E48.

**Recommendation**: before any further mechanism design, this asymmetry between Stage 2's result and E48's original size-signature should itself be investigated — specifically, whether the "actively misleading donor" effect is present broadly across all lesion sizes or concentrated somewhere unexpected (e.g. large lesions, not small ones), since the current data doesn't resolve that. This is a cheap, disclosed next step, not a new architecture.

## Artifacts on disk

- `experiments/exp_e12_eggo_m/e58/run_e58_octant_ablation.py`, `E58_octant_ablation_table.json`, `E58_octant_summary.json`
- `experiments/exp_e12_eggo_m/e58/run_e58b_lesion_conditioned_ablation.py`, `E58b_lesion_conditioned_table.json`, `E58b_summary.json`
- `experiments/exp_e12_eggo_m/e58/run_e58c_bottleneck_substitution.py`, `E58c_substitution_table.json`, `E58c_summary.json`
