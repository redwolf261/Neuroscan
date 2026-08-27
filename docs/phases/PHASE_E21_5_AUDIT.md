# Phase E21.5: EGGO-M Implementation + Mathematical Integrity Audit (E12–E21)

**Status**: Complete — audit only. No training performed. No model, loss,
or experiment script modified. No E22 design work done. Four standalone,
read-only verification scripts were written to the scratchpad directory
(finite-difference gradient check, E16 sign-convention synthetic test,
BatchNorm mutation checksum, checkpoint/replay reproducibility check, and
an A/B/C geometry-aggregation comparison reusing E20/E21's real code on
real checkpoints) — none of them touch `experiments/exp_e12_eggo_m/` or
any checkpoint file.

**Date**: 2026-08-09

---

## 1. Executive verdict

**YES, WITH SPECIFIC CAVEAT.**

The E12–E21 implementation is mathematically sound at every layer this
audit could verify directly: the margin loss's analytical gradient
matches PyTorch autograd and finite differences to high precision; label
semantics are unambiguous binary tumor/background with no soft-label
leakage; E15's causal intervention is genuinely label-agnostic at
application time; E16's sign convention is correct; E19's 68/68
parameter alignment and 4 intentionally-excluded params are confirmed
directly; E20's ShadowAdam replicates PyTorch's real decoupled AdamW
update rule; E21's BatchNorm mechanics do NOT contaminate the
Delta_theta -> Delta_z transport measurement (verified by direct
checksum: running stats mutate but are never read during `.train()`-mode
forward passes, so Delta_z is bit-for-bit unaffected); and E21's
negative-cosine finding is **not** an artifact of its specific
"mean-per-voxel" aggregation choice — an independently-run global-pooled
cosine (Version A) reproduces the same sign and similar magnitude
(-0.23 to -0.26 vs E21's own -0.24 to -0.26), and a class-centroid-level
aggregation (Version C) reproduces the same sign with a much *larger*
magnitude (-0.80 to -0.97).

The caveat: this audit surfaced **one real, previously-undocumented
methodological ambiguity** in the E20/E21 chain (the ShadowAdam replay's
absolute delta magnitudes are not comparable to real historical training
deltas, which the scripts' own docstrings already state explicitly and
correctly — so this is a pre-existing, self-disclosed limitation, not a
newly-discovered bug) and **one genuine interpretive risk** already
half-flagged by the spec itself: E21's raw amplification ratio
(`||Delta_z|| / ||Delta_theta||`) mixes an ~83K-dimensional parameter
space with an ~67M-scalar pooled activation space (an 807x raw
dimensionality gap, ~28x on an RMS basis) and must not be read as a
physically meaningful "gain" without that normalization — this was
already the audit's own instruction, and no report to date has violated
it (E21's own script never states a raw interpretation of this ratio as
"amplification" in a physical sense; PHASE_*.md reports for E19-E21 do
not yet exist, so there is no prior claim to correct here). No fatal
implementation error, autograd bug, checkpoint-corruption, sign-flip, or
train/val leakage was found anywhere in E12–E21.

---

## 2. Critical findings

None at severity D or E. The findings below are the only items rated C
or worse; all are documented in full in Section 9's matrix.

1. **(C) E21's "amplification" ratio compares incompatible dimensionalities** (Phase 13). `||Delta_z||/||Delta_theta||` pools ~67M activation scalars against ~83K parameter scalars (807x raw dimension gap, ~28.4x on sqrt-N/RMS grounds). The conclusion this ratio is used to support ("parameter movement is highly amplified in representation space") is directionally still plausible but the raw magnitude is not interpretable as a physical gain factor without RMS normalization, which neither E21's script nor any existing report computes. Conclusion survives; interpretation needs a caveat before being quoted as a number.
2. **(C) The "pooled d_useful reference" for a class-centroid-level (Version C) cosine is ambiguous and sign-sensitive** (Phase 15). Using the mean of ALL voxels' `d_useful` (tumor-pointing and background-pointing vectors averaged together) flips the sign of the class-conditional cosine relative to using the mean of tumor-only `d_useful`. This is not a bug in any existing script (E21 never computes a Version-C metric at all — this ambiguity only arises in this audit's own new Version-C construction, built fresh for Phase 15), but it is a real interpretive trap for any future analysis that tries to build a class-pooled reference direction naively.
3. **(C, inherited/self-disclosed, not new) E20's shadow-replay update magnitudes are not directly comparable to real historical AdamW updates** — E20's and E21's own docstrings state this explicitly and correctly (fresh accumulators, small bias-correction sample, local-neighborhood-only validity). This audit re-confirms the limitation is real and correctly disclosed, not silently swept under the rug; no script anywhere claims otherwise. Downgraded from a "finding" to a confirmation that the existing self-disclosure is accurate and sufficient.

No D or E severity issues were found in E12–E21.

---

## 3. Mathematical audit

| # | Component | Implemented equation | Expected equation (per PHASE_E10/E11 spec) | Match? | Evidence |
|---|---|---|---|---|---|
| 1 | Margin loss | `L = mean_i [ U_hat_i * B_i * mean_j hinge(z_i,z_j) ]`, `hinge = relu(2*delta_d - \|\|z_i-z_j\|\|)^2`, Euclidean, unnormalized, uncentered, per-voxel (channel-pooled via the norm), no spatial coordinates involved, hinge applied per-pair then averaged before the outer weight multiply and outer mean (hinge is NOT averaged before the ReLU/square) | De Brabandere-style pairwise squared hinge, `(1/\|B\|) Σ weight_i (1/\|N(i)\|) Σ_j hinge(i,j)` | **Yes** — `compute_margin_loss` in `train_eggo_m.py:169-248` implements exactly this; independently confirmed by `PHASE_E13_CODE_AUDIT.md`'s prior narrower audit and re-verified line-by-line here | `train_eggo_m.py:230-235`; standalone finite-diff script (Section "Phase 2" below) |
| 2 | `dL_margin/dz` | Analytical: `d(hinge)/dz_i = -2*relu(2*delta_d - d_ij)*(z_i-z_j)/d_ij` (subgradient 0 at/beyond the hinge boundary) | Same, standard hinge-squared derivative | **Yes** — autograd vs. finite-difference max abs error 8.7e-6 at eps=1e-2 (the numerically well-conditioned regime; smaller eps values suffer float32 subtractive-cancellation noise, a known FD artifact, not a gradient bug — confirmed by sweeping eps from 1e-2 down to 1e-7 and observing error *grow* below 1e-4, the signature of numerical noise, not a real discrepancy) | `audit_phase2_margin_grad_v2.py` (scratchpad) |
| 3 | tau_b (EMATauB) | `tau_b = EMA(median\|d_i\|) / ln(2)`, detached inputs, warmup fallback to a static init value, updated with the batch's own `abs_boundary_logit` (detached, no grad path) | Adaptive EMA-based tau_b per PHASE_E12D/E12E | **Yes** — `EMATauB.update()` operates on a `.detach()`'d tensor (`train_eggo_m.py:119-128`, called with `margin_diag["abs_boundary_logit"]` which is itself `torch.abs(anchors_boundary)` where `anchors_boundary = boundary_logit_flat[anchor_idx].detach()`, `train_eggo_m.py:197`) — no gradient path exists from `tau_b` back into the boundary head | `train_eggo_m.py:93-137, 196-197, 246` |
| 4 | Reachability sign convention (E16) | `signed_proj = sum(delta_z * margin_unit)`, `margin_unit = unit(z0 - opposite_class_centroid_0)` | `+` = moved away from opposite class (margin widened), `-` = moved toward it | **Yes** — synthetic test with known ground truth: `v=+d -> +2.0`, `v=-d -> -2.0`, `v⊥d -> 0.0`, all exact to 1e-9 | `audit_phase8_e16_sign_convention.py` (scratchpad) |
| 5 | AdamW update (E19/E20/E21) | `m_hat = exp_avg/(1-b1^t)`, `v_hat = exp_avg_sq/(1-b2^t)`, `delta = -lr * m_hat/(sqrt(v_hat)+eps)`, weight_decay omitted from delta (decoupled, acts directly on theta) | PyTorch `AdamW` (decoupled weight decay per Loshchilov & Hutter 2019) | **Yes** — checkpoint's own `optimizer_state["param_groups"][0]` contains the key `"decoupled_weight_decay": True`, confirming the saved optimizer genuinely used decoupled AdamW, not Adam+L2; E19/E20's manual reconstruction (`e19_layerwise_gradient_attribution.py:169-189`, `e20_dec1_update_decomposition.py:169-181`) implements the textbook bias-corrected AdamW step exactly, correctly omitting weight_decay from the gradient-driven delta they report (a deliberate, disclosed scope choice, not an error, since decoupled WD is applied by the optimizer directly to theta outside of `delta_theta`'s gradient-driven definition) | Direct checkpoint inspection (Section "Checkpoint facts" below); `e19_...py:169-189`, `e20_...py:158-181` |

---

## 4. Autograd audit

**Model graph (Phase 1), traced directly from `neuroscan_3d_v2.py` and `neuroscan_3d_fixed.py`:**

```
input (B,1,D,H,W), float32, device=cuda/cpu per config
  -> enc1/enc2/enc3/bottleneck (encoder, 8 BatchNorm3d total)
  -> upconv3/dec3/upconv2/dec2/upconv1/dec1 (decoder, 6 BatchNorm3d total)
  -> dec1: (B,32,D,H,W), POST-activation (output of Conv3d -> BatchNorm3d -> ReLU,
            the second of two Conv3DBlocks in nn.Sequential dec1) -- confirmed by
            reading Conv3DBlock.forward() (neuroscan_3d_fixed.py:24-25): relu(bn(conv(x)))
     |
     +--> seg_head = Conv3d(32,1,k=1) -> Sigmoid   [reads dec1 DIRECTLY, no detach, no BN/ReLU between]
     +--> evidential_head = Conv3d(32,2,k=1)         [reads dec1 DIRECTLY, no detach]
     +--> boundary_head = Conv3d(32,1,k=1)            [reads dec1.detach() -- neuroscan_3d_v2.py:96]
```

Confirmed facts (all by direct code inspection, not documentation):

- dec1 shape at 64^3 input: `(B, 32, 64, 64, 64)`, dtype float32.
- `seg_head` reads dec1 directly (`probs = self.seg_head(dec1)`, `neuroscan_3d_v2.py:87` inherited path / `train_eggo_m.py` uses `outputs["probs"]` from exactly this).
- The margin loss reads the SAME `dec1` tensor node: `train_eggo_m.py:351` `dec1 = outputs["dec1"]`, then `dec1_perm = dec1.permute(...).reshape(...)` (`:373`) — a view, not a copy, same autograd graph node. `anchors_z = dec1_flat[anchor_idx]` (`compute_margin_loss`, no `.detach()`).
- No `.detach()` occurs before `L_seg` or `L_margin`'s path back through `dec1`. `.detach()` occurs ONLY at: (a) `evidence_flat`/`boundary_flat` inputs to `U_hat`/`B` inside `compute_margin_loss` (`train_eggo_m.py:196-197`), and (b) `boundary_head`'s own input (`dec1.detach()`, `neuroscan_3d_v2.py:96`).
- **Confirmed**: `dL_margin/d(theta_seg_head) = 0` is structurally correct — `L_margin` is a function of `dec1` (via sampled anchors) and detached `evidence`/`boundary_logit` only; it never reads `probs` or any `seg_head` parameter, so there is no computational path for gradient to flow there. Verified directly in E19's own gradient computation: `g_margin_all` for the `seg_head` block is computed via `torch.autograd.grad(margin_loss, all_params_ordered, ..., allow_unused=True)` and is legitimately `None`/zero (`e19_layerwise_gradient_attribution.py:296-298`, `283-284` comment block explicitly notes this is an expected, not-a-bug zero).
- Boundary/evidential heads cannot affect `dec1` through backward gradient paths from their OWN loss terms in the E14/E19 raw-gradient measurements, since those measurements isolate `L_seg` and `L_margin` specifically (not the boundary BCE or evidential loss) — but this is a scope statement about what E14/E19 measure, not a claim that boundary/evidential losses never influence `dec1` during REAL joint training (they do, via `total_loss.backward()` in `train_eggo_m.py:403`, which is the correct, intended, and undisputed behavior — real training's combined gradient includes all three terms). E19/E20's own gradient computations correctly isolate `seg_loss`/`margin_loss` from this combined signal by using separate `torch.autograd.grad()` calls, which is the right tool for the question each phase asks.
- Skip connections (`torch.cat([upconvN, encN], dim=1)`) do not alter dec1's interpretation as "the shared trunk representation" — dec1 is the LAST decoder block's output, downstream of all skip connections, so its role as the single, unambiguous branch point for all three heads is unaffected by the U-Net's skip topology.

---

## 5. Replay audit

**E20 (`e20_dec1_update_decomposition.py`)**: For each of 6 checkpoints (epochs 5,10,15,20,25,30 — epoch 1 deliberately excluded per the script's own documented rationale, the pathological early-reorganization window), THREE fresh `ShadowAdam` accumulators (`m=0,v=0,step=0`) are stepped over the SAME 15 real training batches, using the checkpoint's own real `lr/beta1/beta2/eps`. `dec1`'s actual weights are held fixed throughout (never updated in place) — each of the 15 steps computes a gradient at (approximately) the same point and accumulates a bias-corrected delta into a running sum. This is explicitly, and correctly per its own docstring, **NOT** a reconstruction of the real historical optimizer trajectory — it is a hypothetical "if training continued from exactly this checkpoint using only this loss term" probe. The docstring states this distinction in three separate places (module docstring "Limitations" 1-3, `ShadowAdam`'s own docstring, and the "NOTE on not applying updates" block) — this audit confirms the disclosure is accurate: the ratio `||delta_margin||/||delta_seg||` (both equally fresh/biased accumulators) is a fair comparison; the absolute magnitudes are not comparable to E19's real `|delta_theta_l|` (mature, thousands-of-steps-old accumulator).

**E21 (`e21_parameter_to_representation_transport.py`)**: Reuses E20's exact `delta_theta_margin/seg/total` (imported via `compute_delta_theta_for_checkpoint`, re-running the identical 15-step replay logic, not a separate reimplementation). For each of 4 additional real training batches: (1) forward the UNPERTURBED model to get `dec1_orig` and the batch's ground truth; (2) for each of {margin, seg, total}, deep-copy the model, add that objective's `delta_theta` to the COPY's `dec1.parameters()` in place, forward the SAME input batch through the copy, subtract to get `Delta_z = dec1_perturbed - dec1_orig`. `d_useful` is recomputed fresh per batch from `dec1_orig` and that batch's own ground truth, using E15's exact formula (verified by direct code comparison between `compute_useful_direction` in E21 and `compute_manipulated_dice`'s direction construction in E15 — byte-identical formula: `direction[tumor] = z[tumor] - bg_centroid`, `direction[bg] = z[bg] - tumor_centroid`, unit-normalized).

Mathematically, this is a first-order (but NOT linearized/Jacobian-approximated — it is a genuine, exact nonlinear forward pass through the perturbed weights) measurement of how a given parameter-space displacement translates into an activation-space displacement, evaluated at one specific point in parameter space (the checkpoint) and one specific batch of inputs. It correctly isolates the THREE perturbations from each other (verified: three independent `copy.deepcopy()` calls, confirmed no cross-contamination — see Section 6).

---

## 6. BN/state audit

**This was the single most safety-critical check in this audit, and the result is reassuring.**

Directly checksummed (`running_mean.sum()`, `running_var.sum()`, `num_batches_tracked`) all 14 `BatchNorm3d` layers in the model, before and after running E21's real 15-step `compute_delta_theta_for_checkpoint` replay loop on the real epoch-30 checkpoint with real training batches:

- **The ORIGINAL model's BN running statistics DO mutate during the 15-step replay** — all 14 layers changed; `num_batches_tracked` advanced by exactly 15 (e.g. `enc1.0.bn`: 4230 -> 4245), confirming each of the 15 real `model(images)` forward calls (in `.train()` mode) triggered a genuine running-stat update, exactly as `nn.BatchNorm3d.forward()`'s standard behavior dictates when `self.training=True` and `track_running_stats=True`.
- **But this mutation is causally inert for the forward computation itself.** Direct inspection of PyTorch's `_BatchNorm.forward()` source confirms: when `self.training=True`, `bn_training=True` is passed to `F.batch_norm`, which normalizes using the **current input's own live batch statistics**, never `self.running_mean`/`self.running_var` — the running buffers are updated as a side effect (for later `.eval()`-mode use) but are never READ during a `.train()`-mode forward pass.
- **Direct numerical confirmation**: `dec1` computed by the model AFTER the 15-step replay (with its now-mutated running stats) is **bit-for-bit identical** (`max_abs_diff = 0.0`, `mean_abs_diff = 0.0`) to `dec1` computed by a completely fresh reload of the same checkpoint on the same batch. The BN running-stat drift accumulated during replay has **zero effect** on `dec1_orig`, the "before" reference E21's entire Delta_z/cosine analysis is built on.
- **Cross-contamination check between the three perturbed copies (margin/seg/total)**: three sequential `copy.deepcopy()` + zero-perturbation forward passes reproduce `dec1_original` exactly (`max_abs_diff = 0.0` in all pairwise comparisons among copy-A/copy-B/copy-C/original). Running a real nonzero perturbation on one copy and re-running a zero-perturbation copy afterward shows **zero** difference in the zero-perturbation copy's output (`0.0` exactly) — confirming Python object-identity/`deepcopy` semantics correctly isolate the three branches; there is no shared mutable BN state leaking between them.
- **Conclusion: E21's Delta_z measurement is NOT contaminated by BatchNorm running-statistic drift, at any of the checkpoints it uses.** This holds specifically because the model stays in `.train()` mode throughout (deliberately, to match live training-time behavior per the E12e lesson every phase since E14 has followed) — had any part of this pipeline switched to `.eval()` mode mid-replay, this conclusion would NOT hold (eval mode reads the running buffers directly, and the 15-step-replay-induced drift would then leak into the comparison). This was checked directly, not assumed.

Diagnostic scripts: `audit_phase12_bn_mutation_check.py`, `audit_phase12b_bn_train_mode_normalization.py` (both in scratchpad; standalone, read-only, no checkpoint files modified).

---

## 7. E15/E21 geometry compatibility (special attention, per spec)

**Both `d_useful` and `Delta_z` are confirmed, directly from the code, to live in the exact same per-voxel `(N_voxels, 32)` raw dec1 activation space** — this is not assumed, it is verified: `compute_useful_direction()` in E21 (`e21_...py:219-236`) reshapes `dec1` via `dec1.permute(0,2,3,4,1).reshape(-1,C)` (identical to every other phase's flattening convention) and returns a `(N_voxels, 32)` unit-direction tensor; `Delta_z` is computed the same way (`dec1_perturbed...reshape(-1,C) - dec1_orig_flat`, `e21_...py:282`). Both tensors index the same voxel ordering from the same batch, same permute/reshape convention, no transpose or channel-order mismatch found.

**Testing the three aggregation versions from the spec, on real data (epochs 15 and 30, E20/E21's own real replay pipeline, reused not reimplemented):**

| Epoch | A: global pooled cosine | B: mean per-voxel cosine (E21's actual metric) | C: class-centroid displacement cosine (tumor-referenced) | C: class-centroid displacement cosine (all-voxel-mean-referenced) |
|---|---|---|---|---|
| 15 | **-0.2302** | **-0.2393** | **-0.9744** | +0.9673 |
| 30 | **-0.2445** | **-0.2552** | **-0.8050** | +0.8106 |

**Key findings:**

1. **A and B agree closely** (within 0.01-0.02 of each other at both epochs) — E21's "mean per-voxel cosine" is NOT an artifact of averaging noisy individual cosines that would look different if pooled globally first. Both answer materially the same question here, empirically, on this data (they are not logically guaranteed to agree in general — averaging normalized per-voxel cosines vs. normalizing a pooled sum can diverge sharply when voxel-level norms are highly heterogeneous — but on the actual E12f checkpoints they do not diverge).
2. **C (class-centroid-level) reproduces the SAME sign as A/B, with a much stronger magnitude** (-0.80 to -0.97) when the reference direction is the mean of TUMOR voxels' `d_useful` (the principled choice, since the class-conditional shift `(mu_T_after - mu_T_before) - (mu_B_after - mu_B_before)` is itself implicitly tumor-vs-background framed). **This means E21's negative-cosine finding does not weaken under a coarser, class-pooled aggregation — if anything it strengthens.** This is a materially important confirmation: the "margin update moves dec1 away from the useful direction" finding is robust across at least three different, independently-reasonable ways of asking the same geometric question.
3. **A genuine ambiguity surfaced by this audit's own new Version-C construction** (not a pre-existing bug in any script): pooling `d_useful` across BOTH classes before computing C's cosine (rather than restricting to the tumor-only reference) flips the sign entirely (+0.81 to +0.97). This is because tumor voxels' `d_useful` points one way and background voxels' points the opposite way (by construction — each class's useful direction points AWAY from the OTHER class), so a naive all-voxel average of `d_useful` is a poorly-defined, nearly-cancelling reference vector, NOT a meaningful "pooled useful direction." This is documented here as a genuine interpretive trap for any FUTURE class-pooled analysis (e.g., a possible E22 ingredient) — it does not affect any existing E12-E21 script, since none of them compute a Version-C-style metric today.

**Downgrade/upgrade assessment**: E21's headline finding ("the actual EGGO update moves dec1 away from the useful direction, mean cosine consistently negative") is **not overstated** — if anything, this audit's independent replication under two additional aggregation schemes shows the finding is *more* robust than a single-metric result would suggest, provided the class-pooled reference is constructed sensibly (tumor-referenced, not naively class-agnostic-averaged).

---

## 8. Claim audit

| # | Claim | Verdict | Evidence / caveat |
|---|---|---|---|
| 1 | "EGGO-M's margin mechanism is active" | **SUPPORTED** | Direct code inspection confirms non-zero `lambda_margin`, real `active_hinge_pct` tracking, non-trivial hinge activity across all 30 real epochs (E12f/E13 logs); independently re-confirmed by E18's `freeze_decoder` config showing 64x higher margin activity, itself only possible if the mechanism is genuinely wired into the loss and gradient graph. |
| 2 | "EGGO-M does not improve Dice" | **SUPPORTED** | E12f/E13's own trajectory data (not re-derived here, but the training script producing it, `train_eggo_m.py`, was audited line-by-line and found faithful to its stated design — no bug found that would flip this outcome). |
| 3 | "The margin gradient is not persistently in conflict with segmentation" | **SUPPORTED** | E14's methodology (independent `torch.autograd.grad` calls, same forward pass, anchor-restricted cosine) is sound; code matches the reported methodology exactly; no accidental gradient accumulation or stale-gradient bug found (verified: `torch.autograd.grad` never writes to `.grad` buffers, `retain_graph` used correctly, no `.backward()`/`zero_grad()` interaction in this measurement path). |
| 4 | "The decoder is sensitive to margin changes" | **SUPPORTED** | E15's Euclidean push intervention is label-agnostic at application time (ground truth used only to select which centroid is "opposite," identical information EGGO-M's own loss uses); the rejected oracle variant (push-sign-by-own-label) is confirmed absent from the final script (only the Euclidean manipulation and the label-free Jacobian remain); push=0 sanity check reproduces E12f's own logged Dice to 4 decimal places, confirming a faithful, non-confounded pipeline. |
| 5 | "The margin gradient points toward the E15 useful direction" | **SUPPORTED, WITH THE PRE-EXISTING CAVEAT E16 ITSELF ALREADY STATES** | E16 Part C's methodology is sound (same anchor sampling as real training, independent `torch.autograd.grad`, correct cosine definition). This audit adds no new caveat beyond what E16 already discloses (Part C measures LOCAL, per-step, live-reference alignment; it does not by itself imply large net accumulated movement — that is a separate, correctly-distinguished question E16 Part B/D already addresses). |
| 6 | "The representation undergoes substantial early rotation" | **SUPPORTED** | Confirmed by three independently-coded methods in E17 (centroid direction, margin gradient direction, Procrustes) — this audit re-verified the Procrustes code (`orthogonal_procrustes`, correct mean-centering, correct `trace(R)/dim` closeness metric) and the fixed vs. live reference-frame distinction (correctly implemented and correctly distinguished in the Pre-E19 Review). |
| 7 | "The decoder region is the strongest experimentally identified source of that rotation" | **PARTIALLY SUPPORTED, per the project's own already-issued correction** | E18's ablation code is faithful to its documented design (frozen params correctly excluded from the optimizer, not just `requires_grad=False`); the BN-confound correction already issued in `PHASE_E18_FOLLOWUP_PRE_E19_REVIEW.md` (symmetric BN-running-stat-still-updating confound in both `freeze_encoder` and `freeze_decoder`) is verified accurate by this audit's own reading of `apply_ablation`/`freeze_bn_running_stats` — the downgraded claim ("strongest experimentally identified source," not "decoder learning causes") is the correct, code-supported framing and this audit finds no reason to revise it further. |
| 8 | "The margin objective receives substantial parameter-space update magnitude" | **SUPPORTED, WITH THE PRE-EXISTING SCOPE CAVEAT** | E19/E20's docstrings correctly and explicitly limit this claim to a "hypothetical, objective-isolated, fresh-shadow-accumulator" sense, not literal historical optimizer state (E19's real `|delta_theta_l|` IS literal historical state, but does not by itself decompose by loss term — E20 is the term specifically built, with disclosed limitations, to approximate that decomposition). This audit confirms the disclosure is accurate and the ShadowAdam implementation matches real AdamW's math exactly. |
| 9 | "The margin objective produces substantial dec1 representation movement" | **SUPPORTED** | E21's Delta_z computation is confirmed free of BN contamination (Section 6) and uses the correct, unperturbed `dec1_orig` reference; `dz_pooled_norm` values are real, non-zero, and causally attributable only to the weight perturbation (verified: zero-perturbation copies reproduce the original exactly). |
| 10 | "The actual EGGO update moves dec1 away from the E15 useful direction" | **SUPPORTED, AND THIS AUDIT'S OWN INDEPENDENT REPLICATION STRENGTHENS RATHER THAN WEAKENS IT** | Section 7's A/B/C comparison: all three aggregation methods agree in sign (negative) at both tested epochs; the class-centroid-level version (C, tumor-referenced) shows an even larger-magnitude negative cosine than E21's own per-voxel-mean metric. This is the single most reassuring result of this entire audit for the arc's current headline finding. |
| 11 | "EGGO's failure is caused by an optimization bottleneck" | **NOT SUPPORTED (unchanged from the project's own E18-follow-up conclusion)** | This audit does not re-litigate this claim (it is out of scope — no new experiment was run), but confirms nothing found here contradicts the project's own existing conclusion that rotation/optimization framing was already downgraded by `PHASE_E18_FOLLOWUP_PRE_E19_REVIEW.md`, and E19-E21's own chain (parameter update exists and is substantial, AND it demonstrably moves dec1 AWAY from the useful direction, not merely insufficiently toward it) is, if anything, further evidence AGAINST a pure "not enough optimization" story and toward a "wrong-direction" or "target-instability" story — consistent with, not contradicting, E17's moving-target finding. |
| 12 | "The current evidence justifies an optimization intervention" | **NOT SUPPORTED** | Per claim 11's chain: if the margin-objective's own parameter update moves dec1 AWAY from the useful direction (claim 10, now independently re-confirmed under two additional aggregations), then a pure optimization fix (larger steps, better conditioning, PCGrad, etc.) would plausibly move the representation FURTHER in the wrong direction, faster — this is a reason for caution about defaulting to an optimizer-centric E22, not new evidence either way about what the right E22 design is. This audit does not recommend or evaluate any specific E22 design, per its explicit scope restriction. |

---

## 9. Bug severity matrix

| Phase | Component | Finding | Severity | Affected conclusions | Required action |
|---|---|---|---|---|---|
| 1 | Model graph | dec1 is post-BN-post-ReLU activation (not pre-activation); confirmed structurally correct autograd isolation of `dL_margin/d(theta_seg_head)=0` | A (no issue) | None | None |
| 2 | Margin loss math | Analytical/autograd/finite-difference gradients agree (max abs err 8.7e-6 at well-conditioned eps) | A | None | None |
| 2 | Margin loss math | FD comparison at very small eps (<=1e-5) shows spurious large error due to float32 subtractive cancellation, NOT a real gradient discrepancy — confirmed by eps-sweep showing error minimized at moderate eps, not small eps (inverse of the expected FD truncation-error pattern if the gradient were wrong) | B (cosmetic — an FD methodology pitfall for anyone re-running this check carelessly, not a code bug) | None (would only mislead a future careless re-verification) | None; documented here for any future re-auditor |
| 3 | Anchor sampling | `sample_stratified_anchors` deterministic top-k given a fixed evidence tensor; negative sampling uses `torch.randint` on the global torch RNG (not the passed-in `rng` numpy object) — the `rng` parameter to `compute_margin_loss` is accepted but not actually used for negative sampling (checked: `torch.randint(0, opp_local.numel(), (n_pos, n_neg), device=device)` uses no `rng` argument) | B (cosmetic/reporting) | None — this does not affect correctness, since `torch.randint`'s own global RNG is a valid, real random source; it just means the `rng` parameter name is slightly misleading (dead/unused for that specific call site) and any attempt to force full negative-sampling reproducibility via the numpy `rng` object alone (without also fixing `torch.manual_seed`) would fail | None required for existing results (all downstream reproducibility checks in this audit controlled `torch`'s global RNG directly, not the numpy `rng`, and confirmed full determinism); worth a one-line comment fix if the code is touched again, not urgent |
| 4 | Label semantics | Binary tumor(1)/background(0), `order=0` nearest-neighbor resize preserves binary-ness exactly, no soft labels possible from this pipeline | A | None | None |
| 5-6 | E15 direction / oracle leakage | Direction is per-voxel, computed fresh per batch, uses ground truth only to select which centroid is "opposite" (same information the margin loss itself uses) — genuinely label-agnostic AT APPLICATION TIME, but NOT label-free in direction ESTIMATION (ground truth is used to construct the centroids) — this distinction is already explicitly drawn in E15's own module docstring and PHASE_E15's report | A (already correctly disclosed, not a new finding) | None | None |
| 7 | E14 gradient conflict | Independent `torch.autograd.grad` calls, correctly restricted to anchor voxels, no shared mutable state, no gradient-clipping/AMP contamination (this codebase does not use AMP anywhere in the audited scripts) | A | None | None |
| 8 | E16 sign convention | Verified correct via synthetic test (+1/-1/0 all exact) | A | None | None |
| 9 | E17/E18 rotation | Procrustes correctly mean-centers before alignment; moving vs. fixed reference frames correctly and separately reported (per the Pre-E19 Review's own correction); Procrustes CAN mask class-specific geometric changes since it operates on the pooled (not class-conditional) point cloud — already flagged as a limitation in E17's own report (Limitation 5) | B (already disclosed limitation, re-confirmed accurate) | Slight — a class-conditional Procrustes was never run; this remains a genuinely open, not-yet-answered question, not a bug | None required by this audit; a natural (not urgent) extension if class-specific rotation ever becomes decision-relevant |
| 10 | E19 gradient attribution | 68/68 parameter alignment confirmed directly against a real checkpoint; 4 excluded params (evidential_head x2, boundary_head x2) confirmed intentional and structurally correct (`g_margin` has no path to those params' gradients since boundary_head reads `dec1.detach()` and evidential_head is never involved in `L_margin`'s formula) | A | None | None |
| 11 | E20 AdamW reconstruction | Checkpoint confirmed `decoupled_weight_decay: True` (genuine AdamW); ShadowAdam's math matches PyTorch's real update rule exactly; docstring's own disclosure of the fresh-accumulator/local-neighborhood limitation independently confirmed accurate | A (limitation already correctly self-disclosed) | None new | None |
| 12 | E21 BN mutation | Running stats DO mutate during replay (15 real forward passes); this has ZERO effect on any forward output because `.train()`-mode BN never reads running buffers — confirmed by exact (0.0) numerical comparison | A (verified non-issue) | None | None |
| 13 | E21 amplification ratio | Raw `\|\|Delta_z\|\|/\|\|Delta_theta\|\|` mixes ~83K-parameter-dim and ~67M-activation-scalar-dim spaces (807x raw dimension gap, ~28.4x on RMS/sqrt(N) grounds) — not yet normalized in any existing script or report | C (methodological limitation, conclusion likely survives but the specific magnitude should not be quoted without this caveat) | Any future write-up that quotes E21's raw amplification number as a physically meaningful "gain" without this normalization would be overstating precision | If E21's amplification numbers are ever included in a report, compute and report RMS_activation/RMS_parameter alongside the raw ratio, per Phase 13's own instruction |
| 14 | E21 cosine definition | Confirmed E21 computes Version B (mean of per-voxel cosines), not Version A (cosine of pooled vectors) — `cos_per_voxel = F.cosine_similarity(dz[valid], d_useful[valid], dim=1)` then `.mean()`, `e21_...py:285-286` | A (correctly matches what this audit independently verified is a robust choice — see Phase 15) | None | None |
| 15 | E15/E21 geometry compatibility | A, B, C all agree in sign on real data; C (tumor-referenced) shows the strongest magnitude; a naive all-class-pooled reference direction for C is ambiguous/sign-flippable (new observation from this audit's own construction, not a pre-existing bug) | C (genuine interpretive trap for any FUTURE class-pooled metric, not an issue in any current script) | None to existing E21 results; relevant only if a future phase (e.g. E22) builds a class-pooled reference direction | Document (done, here) that a class-pooled `d_useful` reference must be constructed carefully (e.g. tumor-only, or with an explicit sign convention), not naively averaged across both classes |
| 16 | Checkpoint reproducibility | Two independent loads of the same checkpoint produce EXACTLY (0.0) identical parameters, BN buffers, and forward outputs; E20/E21's replay is exactly deterministic given a fixed seed | A | None | None |
| 17 | Data leakage | E14/E19/E20/E21 confirmed to use ONLY the training split (`create_brats_loaders`'s `train_loader`); E15/E16/E17 confirmed to use ONLY the validation split (`BraTSDataset(split="val")`), consistent with each phase's own documented intent; the train/val split itself is a fixed-seed (42), patient-disjoint partition, identical across every script that calls `create_brats_loaders`/`BraTSDataset` | A | None | None |

**No D or E severity findings exist anywhere in this audit.**

---

## 10. Final decision

**PROCEED TO E22 UNCHANGED.**

No implementation, tensor-selection, autograd, checkpoint-replay,
normalization, masking, sampling, reference-direction, or
mathematical-definition error was found anywhere in E12–E21 that would
invalidate any current conclusion. The one C-severity item with any
forward-looking relevance (Phase 13's amplification-ratio
dimensionality mismatch) is an interpretation caveat for reporting, not
a defect requiring rework, and does not change any existing claim's
SUPPORTED/NOT SUPPORTED status. The E15/E21 geometry-compatibility
question (Phase 15), which this audit treated as the central item to
resolve, comes back **more** favorable to the arc's current conclusions
than before this audit began: the "margin update moves dec1 away from
the useful direction" finding replicates under two independently
constructed alternate aggregations, with the class-centroid version
showing an even stronger effect than E21's own reported per-voxel-mean
metric.

---

## Appendix: checkpoint facts used throughout this audit

Directly read from `experiments/exp_e12_eggo_m/e12f_pilot_calibrated_seed0/checkpoints/epoch_30.pth`:

- `optimizer_state["param_groups"][0]`: `lr=1.38851e-4`, `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=1e-5`, `decoupled_weight_decay=True`, `params` length 68.
- `optimizer_state["state"]`: 68 entries, each with `step` (a tensor, e.g. `4230.0` at epoch 30), `exp_avg`, `exp_avg_sq` matching that parameter's shape.
- `model.named_parameters()` on a fresh `UNet3D_v2()`: exactly 68 tensors (enc1=8, enc2=8, enc3=8, bottleneck=8, upconv3=2, dec3=8, upconv2=2, dec2=8, upconv1=2, dec1=8, seg_head=2, evidential_head=2, boundary_head=2) — confirms E19's 68/68 alignment claim and the "4 excluded params" (evidential_head + boundary_head) directly.
- 14 `BatchNorm3d` layers total (8 encoder + 6 decoder, 0 in any head) — matches the Pre-E19 Review's own corrected count exactly.
- `dec1` module: 8 parameter tensors, 83,136 scalar parameters total.
- `config["training"]["epochs"] = 50` still present in this checkpoint's saved config (the pre-existing `PHASE_E13_CODE_AUDIT.md` LR-schedule finding remains true of this checkpoint; out of scope for this audit to re-litigate, noted only for completeness).

## Appendix: standalone audit scripts (scratchpad, read-only, not part of the experiment suite)

| Script | Purpose | Result |
|---|---|---|
| `audit_phase2_margin_grad.py`, `audit_phase2_margin_grad_v2.py` | Analytical vs. autograd vs. finite-difference gradient check on a tiny synthetic margin-loss example | PASS (max abs err 8.7e-6 at well-conditioned eps) |
| `audit_phase8_e16_sign_convention.py` | Synthetic v=+d/-d/perp test of E16's signed-projection formula | PASS (exact) |
| `audit_phase12_bn_mutation_check.py` | BatchNorm running-stat checksum before/after E21's 15-step replay, on the real epoch-30 checkpoint | Running stats mutate; dec1 output does NOT (0.0 diff) |
| `audit_phase12b_bn_train_mode_normalization.py` | Cross-contamination check between E21's three (margin/seg/total) deepcopy branches | PASS (0.0 diff on zero-perturbation controls) |
| `audit_phase15_ABC_geometry.py` | A/B/C cosine-aggregation comparison, reusing E20/E21's real replay code on real checkpoints (epochs 15, 30) | A/B agree closely; C agrees in sign, stronger magnitude (tumor-referenced) |
| `audit_phase16_reproducibility.py` | Checkpoint double-load and E20/E21 replay determinism check | PASS (exact 0.0 in all cases) |
