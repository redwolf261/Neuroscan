# Phase E48 — Bottleneck Encoding-Deficiency Audit (Reversed, Significant Finding)

## Context

E47's causal routing audit (see `PHASE_E47_CAUSAL_ROUTING_AUDIT_NULL.md`) found that the bottleneck→enc1 attention gate's effect is not concentrated at boundary voxels — a clean null, causally confirming E43's earlier correlational null. This ruled out "boundary-routing failure" as the mechanism behind the project's Dice ceiling.

E48 tested a different, previously-untested hypothesis: perhaps the bottleneck fails to *encode* useful discriminative signal for **small** lesions specifically (an encoding deficiency, distinct from routing). The pre-declared, falsifiable prediction: if true, severing the bottleneck entirely should hurt **large**-lesion subjects more than small-lesion subjects (large lesions have more real signal there to lose; small lesions have little to lose).

## Method

No training. Loaded E46's checkpoint. Causal intervention: the bottleneck tensor is **completely zeroed** (`torch.zeros_like`) before reaching `upconv3` — a full severing of the coarse pathway, stronger than E47's per-voxel gate clamp, testing the bottleneck's own causal contribution directly. Manual trunk reimplementation verified bit-for-bit identical to the real `forward()` (max abs diff = 0.0) before trusting any ablated result.

Outcome: per-subject Dice drop (`dice_intact - dice_ablated`) vs. `native_size` (native-resolution lesion voxel count, reused from this project's established e30 convention), all 125 validation subjects. Spearman correlation (rank-based, robust to the nonlinear size relationships this project has repeatedly found) + 1000-trial permutation test.

## Result

| Metric | Value |
|---|---|
| n subjects | 125 |
| Mean dice_intact | 0.8915 |
| Mean dice_ablated (bottleneck zeroed) | **0.5710** |
| Mean drop | **0.3205** (±0.1741) |
| native_size range | 7,285 – 225,535 voxels (median 90,377) |
| Spearman rho(native_size, drop) | **−0.4541** |
| Parametric p | 1.04×10⁻⁷ |
| Permutation p (1000 trials) | **<0.001** |

## Interpretation — the hypothesis was REJECTED, but with a strong, opposite, significant effect

1. **The bottleneck matters enormously overall.** A 0.32 mean Dice drop from full ablation (0.89 → 0.57) confirms the coarse pathway is not a minor contributor — it is load-bearing for the whole model, consistent with why every deep-supervision variant tried in this project (E25, E44, E45) that touches this pathway has produced real (if sub-1pp) effects.

2. **The correlation is strongly NEGATIVE, not positive.** This *rejects* the specific "encoding deficiency for small lesions" hypothesis as originally framed (which predicted positive correlation) — but it does so by finding the **opposite, equally strong effect**: **smaller lesions depend MORE on the bottleneck, not less.** Severing the coarse pathway hurts small-lesion subjects disproportionately more than large-lesion subjects.

This is mechanistically sensible on reflection: a large, texturally distinctive lesion may be locally identifiable from fine-resolution features alone (the skip connections carry enough local evidence). A small lesion is more likely to be locally ambiguous — indistinguishable from noise/normal tissue at the voxel-neighborhood scale — and may depend on *global* anatomical context (what the bottleneck encodes) to be correctly localized at all. This reframes the project's small-lesion problem: not "the bottleneck fails to encode small-lesion-relevant signal," but **"small lesions are disproportionately reliant on exactly the pathway (global/coarse context) that current architectures under-serve relative to how much small lesions need it."**

## Decision (per pre-declared rule)

The pre-declared GO rule required `rho > 0`. Observed `rho = -0.454`, so by the literal pre-declared rule: **NULL** for the originally-stated hypothesis. This is reported honestly — the one-directional hypothesis as written was falsified, not rescued or reframed after the fact.

However, the underlying causal question ("does lesion size predict differential dependence on the bottleneck?") has a **clear, significant, opposite-direction answer**, which is scientifically the more important outcome here and is reported in full rather than discarded because it doesn't match the pre-declared sign.

## Implication for the fix direction

This is now a genuinely new, causally-demonstrated, previously-undiagnosed mechanism in this project's arc (distinct from E43's null, E47's null): **small lesions are disproportionately dependent on global/bottleneck context relative to large lesions.** A defensible diagnosis-driven fix follows directly: **strengthen or specialize the coarse-context pathway's contribution specifically when the input is likely to contain a small lesion**, rather than uniformly (as E45's D4+D8 auxiliary heads and E46's global attention gate both did, which may be exactly why both landed in the same modest +0.3–0.5pp band — neither concentrated capacity where the size-dependent analysis now shows it's disproportionately needed).

Candidate mechanism for the next phase: a size-adaptive (or difficulty-adaptive, using an easily-computed cheap proxy since ground-truth size isn't available at inference) reweighting or amplification of the bottleneck's contribution to the decoder — conceptually simple, directly motivated by this specific causal finding, and testably falsifiable via the same instrumentation used here (re-running this exact ablation audit on the new model should show the size-drop correlation weaken if the fix works as intended).

## Artifacts on disk

- `experiments/exp_e12_eggo_m/e48/run_e48_bottleneck_encoding_audit.py` (disk only, per `experiments/` gitignore convention)
- `experiments/exp_e12_eggo_m/e48/E48_encoding_audit_table.json` (125 per-subject records)
- `experiments/exp_e12_eggo_m/e48/E48_summary.json`
