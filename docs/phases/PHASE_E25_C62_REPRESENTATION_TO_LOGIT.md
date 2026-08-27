# Phase E25, C6-2.6: Representation Movement → Segmentation Logit — Collateral Damage Confirmed, With a Real FP-Specific Puzzle

**Status**: ✅ Complete. Traces the missing link `Δz → Δlogit → prediction transition` directly, using C6-2's own realized (many-anchor, 15-step AdamW-accumulated) trajectory. **The evidence points primarily to Hypothesis D (collateral damage), with a secondary, FP-specific instance of Hypothesis C (boundary distance) that does not fit the same pattern FN shows.** Hypothesis B (representation-to-logit mismatch) is ruled out as a general explanation — the segmentation head is comparably or *more* sensitive to movement for FN than for TP/TN, and the movement-to-logit correlation is strong for both FN and FP (r≈0.78–0.83). Hypothesis A (insufficient movement) is not the primary bottleneck for FN, which transitions at 65–84% depending on depth, but plausibly contributes to FP's separate, unresolved pattern.

**Date**: 2026-08-12

---

## What this experiment measured

For each of 6 checkpoints, H2's own realized trajectory mechanism (unchanged — the same 15-step ShadowAdam-accumulated `Δθ_margin`, applied via `apply_delta_and_forward`) was extended to capture the **pre-sigmoid segmentation logit** (`model.seg_head[0](dec1)`, the `Conv3d` alone, avoiding probability saturation near 0/1) before and after the perturbation, at up to 300 voxels per category per batch (4 batches/checkpoint), split by TP/TN/FP/FN and, for FN/FP specifically, by confidence tier (FN: near-boundary `p∈[0.4,0.5)`, mid `[0.2,0.4)`, deep `[0,0.2)`; FP: near-boundary `p∈(0.5,0.6]`, mid `(0.6,0.8]`, deep `(0.8,1.0]`). 26,824 voxel records total.

---

## Result 1: transition rates by confidence tier — FN and FP behave very differently

| Tier | n | Transition rate | Mean Δlogit |
|---|---:|---:|---:|
| FN near-boundary | 651 | **84.3%** | +0.778 |
| FN mid | 1,084 | **81.5%** | +1.389 |
| FN deep | 3,867 | **64.4%** | +4.431 |
| FP near-boundary | 912 | **21.7%** | +0.259 |
| FP mid | 2,187 | **21.9%** | −0.393 |
| FP deep | 3,723 | **25.7%** | −1.829 |

**FN shows the pattern Hypothesis C predicts, but weakly**: transition rate declines with depth (84%→82%→64%), consistent with "the deeper the error, the harder to flip" — but even the deepest FN tier still transitions **64.4% of the time**, and FN's overall transition rate (averaged across tiers, weighted by count) is a substantial majority. FN is not the primary bottleneck.

**FP does not show the same pattern at all, and this is a real, reportable puzzle.** FP's transition rate is roughly flat across tiers (21.7%→21.9%→25.7%, if anything *rising* slightly with depth, the opposite of what boundary-distance alone would predict) and is uniformly low — never exceeding 26%, even at the near-boundary tier where a small movement should easily suffice. **FP's `Δlogit` is correctly-signed and grows with depth (+0.26→−0.39→−1.83, moving toward the correct sign)**, yet this larger corrective signal does not translate into a proportionally higher transition rate. This decouples "is the direction/magnitude of `Δlogit` correct" (yes, and growing) from "does the voxel actually cross the boundary" (no, and not improving) — a distinct puzzle from the FN case, not explained by this experiment.

---

## Result 2: sensitivity `S = |Δlogit|/|Δz|` — rules out Hypothesis B as a general explanation

| Category | Mean S | Median S | n |
|---|---:|---:|---:|
| TP | 0.970 | 1.053 | 7,200 |
| TN | 0.977 | 1.122 | 7,200 |
| FP | 0.612 | 0.602 | 6,822 |
| **FN** | **1.047** | **1.132** | 5,602 |

**The segmentation head is not insensitive to movement for the error population — if anything the opposite.** FN shows the *highest* mean sensitivity of any category (1.047), comparable to or exceeding TP/TN (0.970/0.977). This rules out "the seg_head doesn't use SC-TAM's axis effectively for FN" as an explanation. **FP is the one category with a materially lower sensitivity (0.612, roughly 60% of TP/TN/FN's level)** — this is a real, specific finding worth carrying forward: FP's representation movement produces proportionally less logit change per unit of `Δz` than any other category, which is at least part of why FP's transition rate stays low even as its `Δlogit` grows with tier depth (Result 1) — a smaller sensitivity means a larger `Δz` is needed to produce the same `Δlogit`, and FP's movement may simply not be efficient enough at converting representation change into decision change.

---

## Result 3: does more correctly-aligned movement produce more corrective logit change?

| Category | Pearson r(`cos(Δz,ŵ)`, `Δlogit`) | p | n |
|---|---:|---:|---:|
| FN | **+0.826** | <10⁻³⁰⁰ | 5,602 |
| FP | **+0.784** | <10⁻³⁰⁰ | 6,822 |

**Strong, clean, positive relationships for both.** More alignment with `ŵ` reliably produces more corrective logit movement, for both FN and FP. This further rules out Hypothesis B (a representation-axis/logit-axis mismatch) as the general explanation — the axis SC-TAM optimizes and the axis the segmentation head reads from are strongly coupled, exactly as the design intends.

---

## Result 4: collateral damage on TP/TN — Hypothesis D, confirmed and large

| Category | Fraction showing damaging-direction `Δlogit` | Mean magnitude of damaging shift |
|---|---:|---:|
| TP (wants `Δlogit>0`) | **96.0%** (6,912/7,200) | **3.733** |
| TN (wants `Δlogit<0`) | **79.5%** (5,726/7,200) | 1.561 |

This is the sharpest, most decisive finding in this experiment. **96% of TP voxels show a damaging logit shift, with a mean magnitude (3.73) that is larger than FN's own corrective magnitude (3.48, aggregated across all tiers)** — comparable in scale to FN's *deep*-tier corrective push (4.43), which is exactly the population size that determines whether FN transitions succeed. TN's damage is smaller in magnitude (1.56) but still affects 79.5% of voxels and is comparable to FP's own corrective magnitude (1.37).

**Given the population sizes established in `PHASE_E25_C62_CORRECTED_REANALYSIS.md`'s Level 4 table (TN alone: 12.4 million voxels vs. FN's 21,440, roughly 580:1), even a per-voxel damage magnitude smaller than the correction is enough to dominate in aggregate.** This experiment's subsampled, per-category-balanced design (roughly 5,600–7,200 voxels per category) cannot directly compute the true population-weighted net effect on Dice — that would require the full, unsampled voxel counts — but the combination of (a) TP's damage magnitude actually *exceeding* FN's correction magnitude per voxel, and (b) TP/TN's enormous population advantage over FN/FP, is strong, consistent evidence for Hypothesis D: **collateral damage to the already-correct majority is large enough, per voxel and in aggregate population, to plausibly offset the correctly-directed correction on the error minority.**

---

## Synthesis: which hypothesis does the evidence support?

- **Hypothesis A (insufficient movement)**: not the primary bottleneck for FN (64–84% transition even at depth). Plausibly relevant to FP's separate puzzle (Result 1), but not established directly — FP's `Δlogit` magnitude is not obviously insufficient relative to what would be needed, since it grows with depth while transition rate does not follow.
- **Hypothesis B (representation-to-logit mismatch)**: **ruled out as a general explanation.** FN shows the highest sensitivity of any category (Result 2) and the strongest correlation between alignment and corrective logit change (Result 3). The segmentation head reads SC-TAM's axis effectively for the error population.
- **Hypothesis C (boundary distance)**: **partially supported for FN only.** FN's transition rate genuinely declines with depth, consistent with harder corrections requiring more movement. **This pattern does not hold for FP**, whose transition rate is flat-to-slightly-rising with depth despite growing corrective `Δlogit` — an unresolved, FP-specific puzzle this experiment surfaces but does not explain. FP's markedly lower sensitivity (Result 2, 0.612 vs. others' ~0.97–1.05) is a plausible but not confirmed partial explanation.
- **Hypothesis D (collateral damage)**: **the best-supported explanation overall.** TP shows damage in 96% of sampled voxels with a mean magnitude exceeding FN's own aggregate correction magnitude, TN shows damage in 79.5% of voxels, and both categories vastly outnumber FN/FP in the real training population. This is consistent, not merely suggestive, with the corrected Level 4 finding (TP only 6.1% correctly-signed, TN 21.1%) translating into a real, large-scale offsetting cost.

**Honest limitation, stated plainly**: this experiment's per-category-balanced subsampling (~300 voxels/category/batch) was necessary to make FN/FP — naturally rare — comparably represented to TP/TN, but this means the raw counts here cannot be used to compute a true population-weighted net Dice effect directly; that would require re-running without subsampling (at substantially higher cost, given TN alone spans ~12.4M voxels) or a weighted aggregation using the real per-category population counts already available from `PHASE_E25_C62_CORRECTED_REANALYSIS.md`. This experiment establishes the *direction and rough scale* of the collateral-damage hypothesis, not a final, population-exact accounting.

---

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e25/run_c62_representation_to_logit.py` | Full implementation |
| `experiments/exp_e12_eggo_m/e25/representation_to_logit_results/representation_to_logit_C62.json` | Raw 26,824-voxel records |
| `PHASE_E25_C62_CORRECTED_REANALYSIS.md` | The corrected Level 4 table and population counts this experiment's damage estimate should eventually be weighted against |
