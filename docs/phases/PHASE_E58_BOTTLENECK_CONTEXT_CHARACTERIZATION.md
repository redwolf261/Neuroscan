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

## Stage 2b — Checkpoint-artifact control (re-run Stage 2 on E48's own checkpoint)

Per explicit instruction, before interpreting Stage 2's finding (wrong donor worse than zero, size-signature vanishing under substitution) as meaningful, the same protocol was re-run on E48's own original checkpoint (E46/v5, attention-gate model) — identical donor-selection logic and seed, only the checkpoint (and correspondingly the gate's own recomputation from the substituted bottleneck, matching E48's own established convention) differs. Both bit-for-bit sanity checks passed again (max abs diff 0.0).

| Condition | Mean Dice drop | Spearman rho(size, drop) | Permutation p |
|---|---:|---:|---:|
| Zero (this run) | 0.2717 | −0.383 | <0.001 |
| Zero (E48's own original reference, same checkpoint) | 0.3205 | −0.454 | <0.001 |
| Size-matched donor | **0.5484** | **−0.215** | **0.015** |
| Random donor | 0.5730 | −0.016 | 0.869 |

Paired comparisons: both donor conditions are dramatically and significantly worse than zero (p<0.0001 for both, paired t-test and Wilcoxon) — donor substitution roughly **doubles** the Dice damage relative to zeroing on this checkpoint, a much larger effect than seen on the plain v3 checkpoint (where the zero-vs-random-donor difference was 0.055 Dice, p=0.026; here it's 0.301 Dice, p<0.0001).

**The checkpoint-artifact control confirms the core finding is real, not an artifact**: "wrong context is worse than no context" reproduces cleanly on both checkpoints, and more strongly on E48's own.

**A new, more specific finding emerged from this control that Stage 2 alone could not show**: on this checkpoint, the size-matched donor condition **retains a statistically significant, if attenuated, size-dependent signature** (rho=−0.215, p=0.015) — unlike Stage 2's own result on the plain v3 checkpoint, where size-matching made no detectable difference (rho≈0, p=0.996). Only the *random* donor's size-signature fully vanishes here (rho=−0.016, p=0.869). This asymmetry between the two checkpoints — one has an attention gate reading the bottleneck, one doesn't — is a genuinely new, unplanned observation, not one of the pre-declared hypotheses, and is reported honestly as such rather than folded into either pre-declared decision rule.

**This raises a specific, previously unasked question**: does the attention gate's own presence (not just the bottleneck's raw content) mediate whether donor substitution preserves or destroys the size-dependent signature? This was not tested by any prior E58 stage and would require deliberately isolating the gate's own contribution (e.g. repeating Stage 2b's substitution but with the gate forced to a fixed/frozen state rather than recomputed from the substituted bottleneck) — a natural next diagnostic step, not a mechanism design.

## Stage 3 — Out-of-distribution ablation confound test

Before treating the donor-substitution finding (Stages 2/2b) as a genuine "representation compatibility" phenomenon worth a novelty search, a literature check found this exact pattern — a plausible-but-wrong substituted activation being more damaging than zeroing — is already a named, actively-studied confound in the mechanistic-interpretability literature: the "out-of-distribution ablation problem" (Li & Janson, NeurIPS 2024). Resampling/substitution-style ablations are explicitly documented as unreliable because they can push a model into activation regimes never seen during training, so the resulting damage may reflect generic OOD confusion rather than the causal importance of the substituted content.

This was tested directly, not just accepted from the literature: **mean-bottleneck ablation** (replacing each subject's bottleneck with the mean bottleneck across all 125 validation subjects — a smooth, in-distribution-aggregate, non-subject-specific signal, distinct from both zero and a real donor) was run on the same checkpoint as Stage 2. Sanity check passed (max abs diff 0.0).

| Condition | Mean Dice drop |
|---|---:|
| Zero | 0.2554 |
| **Mean bottleneck (new, this stage)** | **0.2119** |
| Size-matched donor (Stage 2) | 0.2700 |
| Random donor (Stage 2) | 0.3098 |

Paired comparisons: mean-bottleneck ablation is significantly **less** damaging than both donor conditions (vs. random donor: p<0.0001; vs. size-matched donor: p=0.002), and its damage sits close to — in fact numerically slightly below — zero's own (difference not quite significant, p=0.071).

**Verdict: OOD_CONFOUND_SUPPORTED.** A generic, non-informative, in-distribution-aggregate signal (mean) does comparatively little damage, similar to zero. Donor substitution's *extra* damage over zero appears specific to injecting a real, structured, wrong subject's signal — exactly the pattern the out-of-distribution ablation confound predicts. This is evidence *against* treating the donor-substitution finding as a genuine, novel "cross-subject representation compatibility" phenomenon, and evidence *for* the more mundane explanation: substituting a real donor pushes the decoder into an activation regime it never trained on, and the degraded output reflects that novelty, not a real causal property worth building a mechanism around.

## What this means for next steps (explicitly not a novelty claim)

Per the pre-declared framing, this phase's job was to determine *where to search*, not to produce a mechanism. With Stage 3's result, the picture is now clearer than it was after Stage 2b alone:

- **The "distributed, not localized" finding (Stages 1/1b) stands.** Neither absolute nor lesion-relative spatial subsets of the bottleneck reproduce the full-ablation signature — this rules out spatially-targeted mechanisms (crop/attend/route to a specific coarse region) as a promising direction, consistent with why E46's attention gate and E55's dual-resolution crop both landed flat-to-negative.
- **The "wrong context is worse than none" finding (Stages 2/2b) is real, reproduces across two checkpoints, but is most parsimoniously explained by a known measurement confound (Stage 3), not a genuine causal "compatibility" phenomenon.** Mean-bottleneck ablation — a generic, non-informative, in-distribution-aggregate signal — does comparatively little damage, close to zero's own. This shows the extra damage from donor substitution specifically requires injecting a real, structured, wrong subject's content, exactly the signature the out-of-distribution ablation confound predicts (Li & Janson, NeurIPS 2024). The literature check that should have preceded treating this as a novel finding was run only after the effect was already observed, and it correctly identified the mundane explanation before any mechanism was designed around the alternative.

**Recommendation: close this diagnostic line.** The E48→E58 causal chain is now complete and consistent: the bottleneck causally matters (E48), its relevant content does not localize spatially in any tested sense (Stages 1/1b), and the one candidate "genuinely new phenomenon" this phase surfaced (donor incompatibility) does not survive a direct confound check (Stage 3) — it is very likely the same out-of-distribution ablation artifact already documented elsewhere, not a novel, actionable causal property. No fresh novelty search or mechanism design is warranted from this specific line of investigation. The open question left standing from E48 itself — small lesions depend disproportionately on the bottleneck, and that dependence is real, causal, and distributed rather than localized — remains true and unexplained at the mechanistic level; this phase narrowed what it is *not* (not spatial routing, not a donor-compatibility effect) without identifying what it *is*.

## Artifacts on disk

- `experiments/exp_e12_eggo_m/e58/run_e58_octant_ablation.py`, `E58_octant_ablation_table.json`, `E58_octant_summary.json`
- `experiments/exp_e12_eggo_m/e58/run_e58b_lesion_conditioned_ablation.py`, `E58b_lesion_conditioned_table.json`, `E58b_summary.json`
- `experiments/exp_e12_eggo_m/e58/run_e58c_bottleneck_substitution.py`, `E58c_substitution_table.json`, `E58c_summary.json`
- `experiments/exp_e12_eggo_m/e58/run_e58d_substitution_e48_checkpoint.py`, `E58d_substitution_e48ckpt_table.json`, `E58d_summary.json`
- `experiments/exp_e12_eggo_m/e58/run_e58e_ood_confound_test.py`, `E58e_ood_confound_table.json`, `E58e_summary.json`
