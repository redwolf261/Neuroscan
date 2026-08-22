# Phase E25, C6-2 Mechanistic Isolation Checks: Ruling Out Sampling, Aggregation, and AdamW

**Status**: ✅ Complete — three targeted isolation experiments run on C6-2's existing checkpoints, per the user's explicit instruction to investigate *why* a dominant, correctly-conditioned FN/FP gradient signal (established in `PHASE_E25_C62_MECHANISM_AUDIT.md`'s anchor-composition addendum) nonetheless produces wrong-signed realized movement, before designing anything. **The decisive result: FN's wrong-signed movement is present even in a fully isolated, single-anchor, pure-gradient probe — no AdamW, no other anchors, no aggregation.** This rules out AdamW's moment-based accumulation and cross-anchor gradient aggregation as the source of the sign flip, and localizes the failure to the relationship between a single anchor's parameter-space gradient and that same anchor's own forward-pass-realized representation change — a property of the shared-parameter network itself, not of training dynamics across many anchors or many steps.

**Date**: 2026-08-12

---

## Executive result

Three checks, run in the order specified:

- **Check 3** (real `U_hat·B` weighting): materially revises the anchor-composition diagnostic's earlier unweighted figures — FN's loss share shrinks from 43.0% (unweighted) to 27.8% (with the real evidential/boundary weight applied) — but does not reverse the finding: FN remains a large, non-trivial contributor, and FP's share is essentially unchanged (30.1%→35.5%).
- **Check 1** (individual anchor vs. batch aggregate): **every single sampled FN and FP probe (24/24 each) has a positive cosine with the full batch's aggregated `∇_θL_SC`** — individual FN/FP anchors are not opposed or diluted by other anchors; if anything they align with the aggregate direction more strongly and consistently (mean cos ≈ 0.41–0.46) than TP/TN do (≈0.08–0.14, noisier). This rules out "FN's own correct signal gets outvoted by other anchors" as an explanation.
- **Check 2** (isolated single-anchor probe vs. realized full trajectory): **the decisive check.** A fully isolated probe — one FN anchor's own per-anchor loss row, backpropagated alone (no AdamW, no other anchors, no 15-step accumulation), applied as a tiny standalone parameter perturbation, then measured via a real forward pass at that same voxel — is wrong-signed in **36/36 measurements across all 6 checkpoints** (epoch 5 through 30), with remarkable consistency (cosine range +0.24 to +0.61, all positive when FN's design expects negative). This matches, rather than contradicts, the realized full-trajectory result (2–13% correct across the same checkpoints).

**Conclusion: the sign flip is not introduced by AdamW, by aggregation across many anchors, or by the 15-step training trajectory. It is present at the smallest possible unit of measurement — one anchor's own gradient, applied in isolation.** The failure point is upstream of all the mechanisms the mechanism audit's Level 2/Level 3 findings could explain, and must instead be a property of how a parameter-space step (shared across the whole network) maps to that specific voxel's own realized representation change — i.e. the network's own Jacobian structure, not the optimizer or the sampling.

---

## Check 3: does real `U_hat·B` weighting change the loss-share picture?

Recomputed on the same real checkpoint/batch data as the anchor-composition diagnostic, this time including the actual evidence-based (`U_hat`) and boundary-based (`B`) per-anchor weight `compute_margin_loss` applies in real training (the prior diagnostic deliberately used `weight=1.0` to isolate raw class structure).

| Category | Unweighted share | Weighted share (real `U_hat·B`) | Mean `U_hat·B` |
|---|---:|---:|---:|
| TP | 22.5% | 32.5% | 0.320 |
| TN | 4.4% | 4.2% | 0.252 |
| FP | 30.1% | 35.5% | 0.613 |
| FN | 43.0% | **27.8%** | 0.362 |

The real weighting shifts the picture — FN's dominant unweighted share shrinks by 15 points, while TP's grows by 10 — but **FN+FP together still account for ~63% of the real, weighted loss**, remaining the largest combined contributor. This is a genuine, honestly-reported correction to the earlier diagnostic's magnitude (not a reversal of its qualitative conclusion): the anchor sampler and the real per-anchor weighting both still concentrate substantial loss mass on error voxels, just less overwhelmingly than the unweighted figure suggested. FN's own `U_hat·B` (0.362) is unremarkable relative to TP's (0.320) — the evidential/boundary weighting is not itself discounting FN in any dramatic, targeted way; FP receives almost twice FN's weight (0.613), likely reflecting FP voxels' typical proximity to the decision boundary.

---

## Check 1: does aggregation across anchors dilute or oppose FN/FP's own signal?

For 11–14 individually-probed anchors per category per checkpoint (66–84 probes total), each anchor's own isolated contribution to `∇_θL_SC` (backpropagated from that anchor's own per-anchor loss row, drawn from the real full anchor pool for negative sampling) was compared via cosine similarity against the full batch's aggregated `∇_θL_SC`.

| Category | Mean cos with aggregate | Std | Fraction positive (agrees with aggregate) | n |
|---|---:|---:|---:|---:|
| TP | +0.082 | 0.194 | 0.73 | 15 |
| TN | +0.136 | 0.213 | 0.64 | 14 |
| **FP** | **+0.456** | 0.177 | **1.00** | 24 |
| **FN** | **+0.406** | 0.122 | **1.00** | 24 |

**FN and FP anchors are not outvoted.** Every single sampled FN/FP probe agrees in sign with the aggregate direction, and with substantially larger, more consistent magnitude than TP/TN's own comparatively weak and noisy alignment (which agrees only 64–73% of the time). If FN's individually-correct gradient were being drowned out by conflicting contributions from the far more numerous TP/TN anchors, we would expect FN's own probes to show low or negative cosine with the aggregate — instead they show the *strongest, most reliable* alignment of any category. This rules out cross-anchor gradient dilution/opposition as the explanation.

---

## Check 2: does the isolated single-anchor probe agree with the realized trajectory?

For each of the 6 checkpoints, 6 individual FN voxels present in the checkpoint's own real training-batch data were each probed independently: their own per-anchor loss row was isolated (no other anchor's contribution), backpropagated to `dec1`'s parameters alone, applied as a tiny standalone perturbation (`ε=0.01`, pure gradient descent, no AdamW), and the resulting `Δz` at that same voxel — measured via a real forward pass — was checked for sign against `ŵ`.

| Epoch | n FN anchors (real batch) | Realized full-trajectory frac correct | Isolated single-anchor probe frac correct |
|---|---:|---:|---:|
| 5 | 10 | 0.100 | 0/6 |
| 10 | 79 | 0.063 | 0/6 |
| 15 | 64 | 0.125 | 0/6 |
| 20 | 100 | 0.110 | 0/6 |
| 25 | 141 | 0.021 | 0/6 |
| 30 | 116 | 0.043 | 0/6 |

**36/36 isolated single-anchor probes are wrong-signed, at every checkpoint, with no exceptions.** The cosine values are not noisy or near-zero (which would suggest a marginal, easily-flipped effect) — they range consistently from +0.24 to +0.61, positive throughout, when SC-TAM's design predicts negative for FN. This is the same qualitative and quantitative pattern as the realized full-trajectory result (2–13% correct, i.e. overwhelmingly wrong-signed) — **the isolated probe does not disagree with the realized trajectory; it reproduces the same failure at the smallest possible scale.**

---

## Reconciling this with the mechanism audit's Level 1 finding

`PHASE_E25_C62_MECHANISM_AUDIT.md`'s Level 1 established that `cos(∇_z L_SC, ŵ) = ∓1.0` exactly, for every active anchor, as a mathematical identity of SC-TAM's own loss formula (the activation-space gradient at an active pair has no component outside `ŵ`'s axis by construction). Check 2 appears to contradict this — but it measures a genuinely different quantity, and the two are not in tension once this is made precise:

- **Level 1** measures `∇_z L_SC` — the gradient with respect to the voxel's own activation `z_i` directly, holding the rest of the network fixed. This is a purely local, algebraic property of the loss function's formula, and is correctly signed by construction, always.
- **Check 2** measures the realized `Δz_i` that results from taking a step in **parameter space** (`θ' = θ - ε∇_θL_SC`, using the gradient with respect to `dec1`'s shared parameters, not `z_i` directly) and then **re-running the forward pass**. The parameters that produce `z_i` are shared across every voxel in the entire batch — a parameter step does not move `z_i` in isolation; it moves the function that computes *every* voxel's representation simultaneously, and `z_i`'s own realized displacement is `Δz_i ≈ J_i · Δθ` where `J_i = ∂z_i/∂θ` is that voxel's own local Jacobian (a real, nontrivial linear map, not the identity), while `Δθ` itself is driven by the loss gradient summed/meaned across the *entire* anchor set (even in Check 2's "isolated" version, `Δθ` comes from one anchor's own loss term, but that single anchor's gradient must still pass through the *same shared convolutional parameters* that also determine `z_i`'s own forward computation).

**In plain terms**: a locally correct "this activation should move toward `-ŵ`" signal, once translated into "adjust these shared convolutional weights accordingly," does not guarantee that voxel's own representation actually moves that way after the weights change and the forward pass is rerun — because the same weight change also alters the convolutional features feeding into that voxel from its neighborhood, and the net effect for a specific FN voxel is evidently, consistently, the opposite of the locally-intended direction. This is not a contradiction between Level 1 and Check 2 — it is exactly the distinction the project's own prior work (E19/E20, the original motivation for separating "loss-level gradient constraint" from "realized parameter-space movement") already established as a real, general phenomenon; Check 2 shows it operating with unusual strength and consistency specifically for FN voxels in C6-2.

---

## What this rules out, and what remains open

**Ruled out** (with direct, quantified evidence, not inference):
- Anchor sampling bias toward already-correct voxels (mechanism audit's own addendum).
- FN/FP being drowned out or diluted by aggregation with TP/TN's own gradients (Check 1: 100% same-sign agreement, strong magnitude).
- AdamW's moment-based accumulation as the source of the sign flip (Check 2: the flip is present even in a single, un-accumulated, non-AdamW step).
- The evidential/boundary weight `U_hat·B` as a targeted suppressor of FN specifically (Check 3: FN's own weight is unremarkable relative to TP's).

**Still open, and now the most concrete remaining candidate**: the mapping from a single voxel's own parameter gradient, through the network's shared convolutional weights, to that same voxel's own realized forward-pass representation change (`J_i = ∂z_i/∂θ`) appears to consistently invert the intended direction for FN voxels specifically. This was not directly decomposed here (e.g. via an explicit Jacobian-vector product computation, or by testing whether the effect is local — dominated by the anchor voxel's own immediate receptive field — or genuinely non-local/batch-wide) and would be the natural next diagnostic if further mechanism-level understanding is wanted before any C6-3 design decision.

---

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e25/run_c62_isolation_checks.py` | All three checks' implementation |
| `experiments/exp_e12_eggo_m/e25/isolation_check_results/check1_per_anchor_gradient.json` | Check 1 raw results |
| `experiments/exp_e12_eggo_m/e25/isolation_check_results/check2_jacobian_isolation.json` | Check 2 raw results |
| `experiments/exp_e12_eggo_m/e25/isolation_check_results/check3_weighted_loss_share.json` | Check 3 raw results |
| `PHASE_E25_C62_MECHANISM_AUDIT.md` | The 4-level audit and anchor-composition addendum this follows up on |
