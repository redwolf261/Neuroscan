# Phase E25, C6-2 Mechanism Audit: Where Does the Favorable Signal Break Down?

**Status**: ✅ Complete — diagnostic decomposition of the existing, already-trained C6-2 checkpoints into four levels of the causal chain `L_SC → ∇_θL_SC → Δθ → Δz → segmentation outcome`. No new training, no algorithm change, per the explicit instruction to audit before designing C6-3. **The audit localizes the failure precisely: SC-TAM's realized representation movement is correctly signed for voxels the model already classifies correctly (TP, TN), and WRONG-signed for the voxels that actually need correction (FN: 98.5% wrong sign; FP: 75% wrong sign), consistently across every checkpoint from epoch 5 through 30.** This directly reconciles H2's favorable pooled tumor/background result, H3/Q5's large-but-undirected error-concentration finding, and H4's Dice shortfall into a single, coherent picture — not three separate puzzles.

**Date**: 2026-08-10

---

## Executive result

The question posed: why does SC-TAM produce a correctly-signed *realized* representation movement (H2) while the segmentation outcome is unfavorable (H4), and where does the disconnect enter?

**Answer, established directly rather than inferred**: it enters at the point H2's own class-conditional (tumor/background) split was too coarse to see. Splitting further by the model's *own current prediction* (TP/FP/FN/TN, not just ground-truth class) reveals that SC-TAM's real movement has learned to reinforce voxels that are *already correct* and to push already-*incorrect* voxels in the wrong direction. This is not a subtle statistical trend — on false negatives (tumor voxels the model currently misses), the movement has the wrong sign **98.5% of the time**, with a *larger* mean magnitude (+0.42) than the correctly-signed true-positive movement (−0.32). The mechanism is doing something coherent, just not the thing needed to fix errors: it is deepening the model's existing decision boundary rather than moving it.

---

## Method

Four levels computed on C6-2's own 6 checkpoints (epochs 5/10/15/20/25/30), reusing established machinery at every level rather than reinventing measurement infrastructure (`run_c62_mechanism_audit.py`):

- **Level 1** (activation-space loss gradient): `cos(∇_z L_SC, ŵ)`, per class, at active-pair anchor voxels, computed fresh on all 15 real ShadowAdam replay steps per checkpoint (same batches H2 already used).
- **Level 2** (parameter-space total gradient): `cos(∇_θL_seg, ∇_θL_SC)` — **reused directly from H1's already-saved data** (the `A_seg` direction's `cos_with_g_margin` field, verified epsilon-invariant — a true per-checkpoint/subject constant — before trusting the shortcut), not recomputed.
- **Level 3** (optimizer update): `cos(Δθ_margin, ∇_θL_SC^{last step})`, where `Δθ_margin` is H2's own real 15-step ShadowAdam-accumulated update and the gradient is freshly computed at the same checkpoint's final replay step (H2 did not save per-step raw gradients, only the accumulated delta).
- **Level 4** (representation update): `cos(Δz, ŵ)`, using H2's own real `Δz` mechanism, split into **six** categories — tumor, background (H2's own split, for continuity) plus the four confusion categories TP/FP/FN/TN (new — requires the model's own prediction at the pre-perturbation checkpoint, split by whether that prediction already agrees with ground truth).

---

## Level 1: instantaneous activation-space gradient (uninformative by construction — reported for completeness)

`cos(∇_z L_SC, ŵ) = −1.0000` for every active tumor anchor, `+1.0000` for every active background anchor, at every one of 90 measured (checkpoint, replay-step) instances (`cos_std ≈ 1e-7`, i.e. float32 precision noise around an exact value, not approximate agreement).

**This is not a new finding — it is a mathematical identity, not an empirical result.** SC-TAM's own derivative (`∂L_SC/∂z_i = ∓2[m−d^SC]·ŵ` for an active pair) has no component outside `ŵ`'s own axis by construction — the activation-space gradient for any active anchor is *always* an exact scalar multiple of `±ŵ`. This is the same property Gate 5's orthogonality check already verified (relative orthogonal component ~1e-7). **Level 1 rules out the instantaneous per-voxel gradient direction as a candidate explanation for the discrepancy** — it cannot be the source, since it is correct by mathematical necessity at every single active voxel, with no exceptions. The failure, wherever it is, must enter downstream of this point.

---

## Level 2: parameter-space total-gradient alignment (reused from H1)

| Epoch | `cos(∇_θL_seg, ∇_θL_SC)` | std | n |
|---|---:|---:|---:|
| 5 | +0.735 | 0.086 | 8 |
| 10 | −0.237 | 0.485 | 8 |
| 15 | −0.088 | 0.506 | 8 |
| 20 | +0.007 | 0.332 | 8 |
| 25 | +0.099 | 0.464 | 8 |
| 30 | +0.036 | 0.450 | 8 |

Starts strongly aligned (epoch 5: +0.74, consistent with the project's own established epoch-5 gradient-alignment spike, seen in E19 and E22 as well), then collapses to near-zero/mildly negative and stays there with high variance (std 0.33–0.51) for the rest of training. The segmentation and margin gradients are, on average, close to orthogonal from epoch 10 onward — meaning `L_seg` provides little consistent steering on the margin direction after the earliest training phase, and the two objectives are not straightforwardly reinforcing or opposing each other in parameter space at this level.

---

## Level 3: does AdamW substantially rotate the update?

| Epoch | `cos(Δθ, ∇_θL_SC)` | `cos(Δθ, −∇_θL_SC)` | ‖Δθ‖ | ‖∇_θL_SC‖ (last step) |
|---|---:|---:|---:|---:|
| 5 | −0.386 | +0.386 | 0.994 | 2.966 |
| 10 | −0.344 | +0.344 | 0.852 | 0.302 |
| 15 | −0.452 | +0.452 | 0.796 | 0.749 |
| 20 | −0.443 | +0.443 | 0.691 | 0.510 |
| 25 | −0.411 | +0.411 | 0.417 | 0.379 |
| 30 | −0.239 | +0.239 | 0.333 | 0.195 |

**Mean `cos(Δθ, −∇_θL_SC) = +0.379` (std 0.072) — moderate alignment with naive single-step descent, not close to 1.0 at any checkpoint.** If AdamW's 15-step accumulated update simply tracked the instantaneous gradient direction, this cosine would sit close to 1.0; instead it sits at roughly 0.24–0.45 throughout training. This confirms real, substantial rotation: AdamW's moment-based accumulation, interacting with the interleaved segmentation-loss gradient (Level 2) and the changing gradient direction across 15 real steps, produces a materially different trajectory than "descend the current margin gradient." This gives a concrete parameter-space mechanism for why H2's *realized* signal could plausibly differ from a naive gradient-only prediction — the update is not what the instantaneous loss gradient alone would suggest.

---

## Level 4: the decisive result — where H2's favorable pooled signal breaks down

Pooled across all 6 checkpoints × 4 batches (24 records):

| Category | GT class | Model's current prediction | `cos(Δz, ŵ)` mean | `frac_correct_sign` | n (pooled) |
|---|---|---|---:|---:|---:|
| Tumor (pooled) | tumor | — | −0.228 | 0.812 | 140,520 |
| Background (pooled) | background | — | +0.277 | 0.789 | 12,442,392 |
| **TP** | tumor | correct (tumor) | **−0.318** | **0.920** | 119,080 |
| **FN** | tumor | **wrong (background)** | **+0.420** | **0.015** | 21,440 |
| **FP** | background | **wrong (tumor)** | **−0.159** | **0.246** | 17,163 |
| **TN** | background | correct (background) | **+0.278** | **0.790** | 12,425,229 |

Per-epoch TP/FN breakdown (both ground-truth tumor, both "expect toward `−ŵ`"):

| Epoch | TP `cos_mean` | FN `cos_mean` | FP `cos_mean` | TN `cos_mean` |
|---|---:|---:|---:|---:|
| 5 | −0.317 | +0.427 | −0.200 | +0.418 |
| 10 | −0.375 | +0.441 | −0.286 | +0.378 |
| 15 | −0.347 | +0.414 | −0.219 | +0.321 |
| 20 | −0.367 | +0.368 | −0.172 | +0.182 |
| 25 | −0.189 | +0.415 | −0.032 | +0.199 |
| 30 | −0.314 | +0.455 | −0.047 | +0.171 |

**This is the answer.** Splitting H2's tumor/background split further by the model's own current correctness reveals a consistent, large, *opposite-signed* pattern between "voxels the model already gets right" and "voxels the model currently gets wrong," within the *same* ground-truth class:

- **TP** (correct tumor): strongly and consistently correctly signed (`−0.32` mean, 92% correct-sign fraction) — reinforcing an already-correct classification.
- **FN** (tumor the model currently misses — exactly the voxels that would need to flip to TP to help Dice): **wrong sign 98.5% of the time**, every single epoch, with a *larger* magnitude (+0.42) than TP's own correct-signed movement. SC-TAM's real update pushes these voxels *away* from the tumor side of the decision axis, not toward it.
- **FP** (background misclassified as tumor): also predominantly wrong-signed (75% wrong), pushed toward `−ŵ` ("more tumor-like"), the opposite of what would convert it to TN.
- **TN**: correctly signed (79%), consistent with the pooled background result.

This pattern is **stable across all 6 checkpoints** (epoch 5 through 30) — not a transient early-training artifact, not something that resolves with more training.

---

## Synthesis: reconciling H2, H3/Q5, and H4 into one picture

These are not three separate puzzles — they are three views of the same underlying mechanism:

- **H2** (pooled tumor/background split) looked favorable because **TP and TN vastly outnumber FN and FP** (119,080 + 12,425,229 vs. 21,440 + 17,163 in this sample — the model is already correct on the overwhelming majority of voxels by the time these checkpoints are reached). The pooled average is dominated by the already-correct majority, masking the wrong-signed minority.
- **H3/Q5** found 9.4x more movement magnitude on misclassified vs. correct voxels — **true and unexplained by this audit alone**, but Q5 only measured `‖Δz‖`, never its *direction* relative to `ŵ`. This audit shows that large movement on error voxels is not inherently helpful: FN's movement is both large *and* wrong-directioned. Q5's magnitude finding and this audit's directional finding are complementary, not contradictory — together they show SC-TAM concentrates substantial movement on error voxels, but in the wrong direction for a majority of the FN population specifically.
- **H3/Q4**'s net-unfavorable FP/FN transition count (TN→FP dominating at 37,846) is the direct behavioral consequence of this audit's FP/TN finding: TN voxels are being pushed correctly (toward `+ŵ`, reinforcing background), but with a `frac_correct_sign` of only 0.79 — meaning roughly 1 in 5 TN voxels move the wrong way at any given step, and given the sheer size of the TN population (12.4M voxels here vs. 17K FP), even a modest wrong-signed fraction of TN produces a large absolute TN→FP count. FN's own poor conversion rate (only 13,518 FN→TP transitions against 21,440 FN voxels showing wrong-signed movement) is the direct consequence of the 98.5% wrong-sign finding above.
- **H4**'s Dice shortfall is the aggregate, downstream consequence: a mechanism that reinforces the existing decision boundary rather than correcting it cannot improve Dice, regardless of how strongly and consistently it does so.

**The clearest single-sentence statement of the failure point**: SC-TAM's class-conditional signed constraint is defined purely in terms of ground-truth class identity (tumor vs. background), with no dependence on whether the model's *current* prediction is already correct — so nothing in the objective distinguishes "reinforce an already-correct boundary" from "correct an existing error," and the realized dynamics have settled into predominantly doing the former.

---

## Addendum: active-anchor composition diagnostic (run before any C6-3 design, per explicit instruction)

The open question flagged above — whether the Level 4 pattern traces to anchor-sampling bias toward already-correct (high-confidence) voxels — is directly testable and was tested (`run_c62_anchor_composition.py`), on the same 6 checkpoints, 6 real training batches each (36 records). For every sampled anchor and every active `(i,j)` pair, anchors were classified TP/TN/FP/FN by the model's own current prediction (eval-mode, matching how Dice is actually scored) vs. ground truth, and each category's sampled fraction, active-pair share, and raw loss-share (fraction of `L_SC`'s actual summed value) were measured.

**Result: the sampling-bias hypothesis is falsified.** `sample_stratified_anchors`' evidence-based (highest-uncertainty) selection already concentrates heavily on error voxels — FN sampled at **38.4x** its true volume-wide frequency, FP at **126.2x**, while the overwhelmingly common, easy TN class is *undersampled* to 0.22x. Within `L_SC` itself: **FN alone contributes a mean 39.7% of the total loss share** (the single largest category, exceeding even TP's 23.9%), and FN+FP together account for ~74% of the loss, despite `compute_margin_loss`'s sc_tam branch (per C6-2's own locked scope, Section 11 decision 4) having no explicit error-weighting term at all. Active-pair counts confirm the same picture: FN anchors participate in 17.0% of all active pairs, FP in 31.3%, vs. TN's 3.6%.

**This means the failure is not "the objective doesn't see enough error voxels."** It sees them constantly — FN and FP together dominate both the sampled-anchor pool and the loss's own gradient budget — and still, per Level 4, moves the majority of FN voxels in the wrong direction. Simply adding an explicit error-weighting term on top (the original, generic framing of C6-3: `ω_i = 1 + α·e_i`) would be reweighting a category that is already the largest contributor to the loss, which gives little reason to expect it would flip Level 4's wrong sign. **The bottleneck is not sampling or gradient allocation — it is what happens to that already-large FN/FP gradient signal downstream, consistent with Level 2's near-zero/collapsing `cos(∇_θL_seg, ∇_θL_SC)` and Level 3's substantial `Δθ` rotation away from the naive descent direction.** This redirects the open question from "is FN/FP underrepresented" (answered: no) to "why does a dominant, correctly-conditioned FN/FP gradient signal net out wrong-signed after passing through the optimizer and the network's own Jacobian" — a question this audit and its addendum surface precisely but do not resolve further.

---

## What this audit does not resolve

- **Why** a dominant FN/FP loss contribution nonetheless produces wrong-signed realized movement is not explained at the mechanism level by either this audit or its anchor-composition addendum — the sampling-bias explanation is now ruled out, but the remaining candidates (interaction with the evidential-uncertainty/boundary reweighting still present in the real `compute_margin_loss` call, per-anchor gradient magnitude vs. direction effects, or genuine network-Jacobian/AdamW-moment dynamics as Level 2/3 suggest) are not disentangled here.
- Level 2 and Level 3's own moderate/collapsing alignment values are reported as real, measured facts; this audit does not claim a single quantitative causal accounting of exactly how much of the Level 4 pattern is attributable to Level 2's near-zero seg/margin alignment vs. Level 3's AdamW rotation vs. some other factor.
- This audit does not evaluate whether a design change would fix this — that is explicitly a C6-3-scoped question, not attempted here per the standing instruction to audit before designing.

---

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e25/run_c62_mechanism_audit.py` | Full 4-level audit implementation |
| `experiments/exp_e12_eggo_m/e25/mechanism_audit_results/mechanism_audit_C62.json` | Raw per-checkpoint/per-batch results, all 4 levels |
| `experiments/exp_e12_eggo_m/e25/run_c62_anchor_composition.py` | Active-anchor composition diagnostic (addendum) |
| `experiments/exp_e12_eggo_m/e25/anchor_composition_results/anchor_composition_C62.json` | Raw per-checkpoint/per-batch composition results |
| `PHASE_E25_C62_RESULTS.md` | The H1–H4 result this audit follows up on |
| `PHASE_E25_CANDIDATE6_SC_TAM_DESIGN.md` | The locked C6-2 design and decision rule |
