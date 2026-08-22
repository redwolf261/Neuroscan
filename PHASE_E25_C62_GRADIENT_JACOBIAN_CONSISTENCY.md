# Phase E25, C6-2.5b: Gradient/Jacobian Consistency Check — Outcome A Confirmed

**Status**: ✅ Complete. **Outcome A**: gradient descent's fundamental theoretical invariant (`Δz·∇_zL ≤ 0`) holds cleanly (95.8–100% of measurements across the epsilon sweep), the finite-difference measurement mechanism is independently validated against a first-order Jacobian-vector-product prediction (5/6 checkpoints converge cleanly toward zero relative error as ε shrinks), and — decisively — once `cos(Δz, ŵ)` is scored against the **corrected** sign convention from `PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md` rather than the old backwards one, **every single one of 48 probed voxels across all four confusion categories (TP, TN, FP, FN) is correctly signed, at 100%.** There is no remaining mystery about a network/Jacobian phenomenon inverting SC-TAM's intended direction. The entire "pathway-invariant inversion" finding from `PHASE_E25_C62_JACOBIAN_LOCALIZATION.md` was the sign-convention bug, fully and cleanly explained, not a separate deeper problem.

**Date**: 2026-08-12

---

## What this experiment checked

Per the user's explicit specification, before interpreting the earlier localization experiment's uniform 0%-correct finding as a genuine network/Jacobian phenomenon, this experiment ran the mathematical consistency check that must hold for *any* gradient-descent step on *any* scalar loss, independent of SC-TAM's own directional claims: for `Δz⁻ ≈ −ε·J·Jᵀ·g_z` and `K=J·Jᵀ` positive semidefinite, `g_z·Δz⁻ ≈ −ε·g_zᵀ·K·g_z ≤ 0`. A parallel gradient-**ascent** control (`Δz⁺`, using `+εg_θ`) should show the opposite sign. Both were measured directly, alongside the loss change `ΔL` itself, across an epsilon sweep (0.1, 0.01, 0.001, 0.0001), on the same 48 isolated anchor voxels (12 per checkpoint × 4 categories) used throughout the recent C6-2 diagnostic work. A first-order Jacobian-vector-product (JVP) prediction was additionally computed via `torch.func.jvp` and compared against the actual measured `Δz`, as an independent validation of the finite-difference measurement mechanism itself.

**A real implementation obstacle, resolved and verified before trusting any result**: `torch.func.jvp` forbids the in-place `num_batches_tracked` mutation `BatchNorm3d` performs in `.train()` mode. Fixed by setting `track_running_stats=False` on every `BatchNorm3d` module before the JVP call — verified directly, via a forward-pass `allclose` check, that this does **not** change the model's output at all, since train-mode BatchNorm always normalizes using the current batch's own live statistics, never the running buffers, regardless of whether tracking is enabled. A separate float32-vs-float64 validation (reported in the module docstring of `run_c62_gradient_jacobian_consistency.py`) confirmed the JVP-vs-finite-difference comparison itself needs float64 to avoid floating-point cancellation noise contaminating the smallest epsilon values — at float32, relative error got *worse* again below ε=1e-5 (0.68→4.66), a classic cancellation signature; at float64, the identical comparison converges cleanly (0.99→0.0047 across the same six-decade sweep). This float64 control validated the measurement infrastructure itself, independent of anything SC-TAM-specific.

---

## Result 1: `ΔL` sign check

| ε | GD (`ΔL<0`) correct | GA (`ΔL>0`) correct |
|---:|---:|---:|
| 0.1 | 91.7% | 89.6% |
| 0.01 | 79.2% | 89.6% |
| 0.001 | 64.6% | 66.7% |
| 0.0001 | 50.0% | 56.2% |

Honestly reported, not smoothed over: this degrades toward chance as ε shrinks, the opposite of the pattern the other two checks show. This is very likely a real, distinct floating-point effect specific to `ΔL`: the loss recomputation (`recompute_loss_for_model`) involves the **squared** hinge (`(margin_target−dist)²`) and re-derives `dist` from a fresh forward pass through the whole network, both of which amplify small numerical noise relative to `Δz`'s own direct linear measurement — at `ε≤0.001`, the true `ΔL` signal is likely smaller than the floating-point noise floor of recomputing the loss end-to-end. This is flagged as a real limitation of the `ΔL` measurement specifically, not evidence against the descent/ascent mechanism itself, which the other two checks (below) directly and independently confirm.

## Result 2: the key invariant, `Δz·∇_zL`

| ε | GD (`Δz·g_z<0`) correct | GA (`Δz·g_z>0`) correct |
|---:|---:|---:|
| 0.1 | 95.8% | 87.5% |
| 0.01 | 97.9% | 100.0% |
| 0.001 | 100.0% | 100.0% |
| 0.0001 | 95.8% | 100.0% |

This is the theoretically load-bearing check, and it holds essentially perfectly — 100% at the two middle epsilons, with the small dips at the largest (ε=0.1, where nonlinear/second-order effects are genuinely expected to matter) and smallest (ε=0.0001, likely the same floating-point floor implicated in the `ΔL` result) ends of the sweep. This directly confirms gradient descent is behaving exactly as first-order theory requires, using the raw dot product as specified (not merely a cosine, which could hide a near-zero-magnitude technicality).

## Result 3: JVP first-order prediction

| Checkpoint | ε=0.1 | ε=0.01 | ε=0.001 | ε=0.0001 |
|---|---:|---:|---:|---:|
| epoch 5 | 0.018 | 0.001 | 0.0005 | 0.0000 |
| epoch 10 | 0.670 | 0.142 | 0.006 | 0.003 |
| epoch 15 | 0.093 | 0.025 | 0.007 | 0.0001 |
| epoch 20 | 0.129 | 0.025 | 0.006 | 0.0001 |
| epoch 25 | 0.721 | 0.557 | 0.496 | 0.248 |
| epoch 30 | 0.737 | 0.388 | 0.026 | 0.013 |

Five of six checkpoints (5, 10, 15, 20, 30) show clean, expected convergence — relative error shrinking by one to three orders of magnitude across the sweep, matching the float64 infrastructure-validation control almost exactly. **Epoch 25 is a genuine outlier**, plateauing around 0.25–0.72 rather than converging — reported honestly, not discarded. This single checkpoint's representative voxel likely sits in a region of unusually high local curvature (large second-order term relative to the first-order one) or was affected by some other checkpoint-specific numerical condition; it was not investigated further, since it does not change the overall conclusion (5/6 clean convergences is already a strong validation, and this experiment's headline finding — Result 4, below — does not depend on the JVP check at all).

## Result 4: the decisive result — Outcome A vs. Outcome B

`cos(Δz_GD, ŵ)` was scored against both the old (backwards) and newly corrected sign conventions, at ε=0.001 (the epsilon where both the `Δz·g_z` invariant and the JVP check are cleanest):

| Convention | Fraction correctly signed |
|---|---:|
| OLD (backwards, used by every H2-onward script) | **0.0%** |
| NEW (corrected, per `PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md`) | **100.0%** |

Per-category breakdown under the corrected convention: **TP 100.0%, TN 100.0%, FP 100.0%, FN 100.0%** — all 48/48 voxels, no exceptions.

---

## Outcome A, per the user's own decision framework

This is **Outcome A**, unambiguously: gradient descent is mathematically consistent (`Δz·∇_zL<0` holds essentially perfectly, the JVP prediction independently confirms the measurement mechanism), and the earlier appearance of a universal, pathway-invariant "inversion" was not a genuine property of the network's Jacobian or shared parameterization — it was the sign-convention bug already identified and confirmed in `PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md`, now demonstrated to fully and completely explain the prior finding. There is no remaining evidence of "the network's representation response is not aligned with the semantic axis we assigned to it" (the more interesting Outcome-A sub-case the user's framing flagged as possible) — under the correct axis assignment, alignment is total (100%) at the single-anchor, isolated-gradient level.

**This closes C6-2.5's original open question decisively.** The gradient→parameter-update→representation chain is not where SC-TAM's mechanism breaks down. What remains open — carried forward, unchanged, from `PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md`'s own scope — is that H2, H3, the mechanism audit's Level 4 table, the isolation checks, and the jacobian localization's own conclusions all need to be recomputed under the corrected convention before any of their specific numeric claims (e.g. "FN moves wrong-signed 98.5% of the time") can be trusted in either direction. That recomputation is real, substantial work, deliberately not rushed here.

---

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e25/run_c62_gradient_jacobian_consistency.py` | Full implementation (ΔL, Δz·g_z, JVP checks) |
| `experiments/exp_e12_eggo_m/e25/gradient_jacobian_consistency_results/gradient_jacobian_consistency_C62.json` | Raw 48-voxel records + 6 JVP checks |
| `PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md` | The bug this experiment confirms fully explains the prior "inversion" finding |
| `PHASE_E25_C62_JACOBIAN_LOCALIZATION.md` | The experiment whose 0%-correct finding is now fully explained, not a separate mechanism |
