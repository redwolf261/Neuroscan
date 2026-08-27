# Phase E25, C6-2.5: Parameter-Group Localization — The Inversion Is Not Localized

**Status**: ✅ Complete — parameter-group intervention experiment run per the user's explicit direction, with a finite-difference control validated before trusting any result. **The finding is not what the experiment was designed to find, and is reported as such: the sign inversion is not confined to any parameter subset. It is present, with near-identical sign and magnitude (per-voxel cross-pathway correlation r=0.98), whether the update is restricted to `dec1` alone, the full decoder, the encoder alone, or all of them together — and, more consequentially, it is present in ALL FOUR confusion categories (TP, TN, FP, FN), not only the FN/FP population the mechanism audit's Level 4 flagged.** This is a materially different and more fundamental finding than a layer-localized bug, and it changes what the isolation checks' TP/TN result meant.

**Date**: 2026-08-12

---

## What this experiment set out to answer

Per the user's own framing, after `PHASE_E25_C62_ISOLATION_CHECKS.md` ruled out sampling bias, cross-anchor aggregation, and AdamW's moment accumulation as the source of FN's wrong-signed realized movement, the remaining suspect was "the network's shared parameterization/Jacobian" — specifically, whether a parameter update intended to move one voxel's embedding along `ŵ` is corrupted somewhere between the loss gradient and the realized representation change. The proposed test: isolate the update to specific parameter groups (`dec1` only, decoder, encoder, encoder+decoder, all) and see whether FN/FP become correctly signed under a restricted pathway, localizing where the inversion enters.

---

## Method

For each of 6 checkpoints, up to 3 real, active (nonzero-loss), individually-isolated anchor voxels were sampled per confusion category (TP/TN/FP/FN — determined by the model's own current prediction vs. ground truth, same convention as the mechanism audit). For each voxel, the full-model parameter gradient of that single anchor's own SC-TAM loss row was computed (reusing `_sc_tam_per_anchor_losses`, the same isolation mechanism validated in the prior isolation checks — real negatives drawn from the full anchor pool, loss/gradient attributed to exactly one anchor). This gradient was then masked to 5 conditions and, for each, applied as a tiny (`ε=0.01`) standalone perturbation with a real forward pass measuring `Δz` at the same voxel:

- **`dec1_only`** — only the final decoder block's own parameters (whose direct output *is* `z`).
- **`decoder`** — every decoder-side parameter (`upconv3, dec3, upconv2, dec2, upconv1, dec1`).
- **`encoder`** — every encoder-side parameter (`enc1, enc2, enc3, bottleneck`).
- **`encoder_and_decoder`** — both combined.
- **`all`** — every model parameter (the reference condition, matching what a real training step actually updates).
- **`seg_head_only`** — a null control. Verified by direct code inspection before running anything: SC-TAM's loss has no gradient path into `seg_head`/`evidential_head`/`boundary_head` at all (confirmed empirically: `0/8` output-head parameters receive nonzero gradient from `L_SC`, consistent with the E11.5 detach audit). This condition is expected, by construction, to produce exactly zero `Δz` — included purely to verify the masking harness itself works.

**Finite-difference control**, run and verified before trusting the sweep (per the user's explicit requirement): for a real control voxel, `|Δz|/ε` was measured at `ε ∈ {0.1, 0.01, 0.001}` and found to converge toward a stable value (86.3 → 174.6 → 186.8) rather than diverge or behave erratically — confirming `Δz` behaves as a genuine first-order linear response to the parameter perturbation, and the measurement mechanism (`apply_delta_and_forward`, reused unchanged from H2/the isolation checks) is not producing an artifact. The same control voxel's measured sign matched the correctly-predicted sign for its ground-truth class, a real physical sanity check, not just a numerical-stability one.

---

## Result

### The null control passed exactly as predicted

All 72 `seg_head_only` records show `g_masked_norm=0.0` and `delta_z_norm=0.0` exactly — confirming the masking harness correctly finds zero gradient precisely where the detach audit says it must.

### Fraction of voxels with the *correct* sign, per pathway × category

| Pathway | TP | TN | FP | FN |
|---|---:|---:|---:|---:|
| `dec1_only` | 0.0% | 0.0% | 0.0% | 0.0% |
| `decoder` | 0.0% | 0.0% | 0.0% | 0.0% |
| `encoder` | 0.0% | 0.0% | 0.0% | 0.0% |
| `encoder_and_decoder` | 0.0% | 0.0% | 0.0% | 0.0% |
| `all` | 0.0% | 0.0% | 0.0% | 0.0% |

**Every single one of 72 probed voxels, across every category and every pathway, is wrong-signed.** This is a clean, unambiguous 0% across the board — not a marginal or noisy result.

### Mean signed cosine, per pathway × category

| Pathway | TP (expect &lt;0) | TN (expect &gt;0) | FP (expect &gt;0) | FN (expect &lt;0) |
|---|---:|---:|---:|---:|
| `dec1_only` | +0.361 | &minus;0.600 | &minus;0.402 | +0.525 |
| `decoder` | +0.359 | &minus;0.605 | &minus;0.401 | +0.515 |
| `encoder` | +0.310 | &minus;0.612 | &minus;0.521 | +0.409 |
| `encoder_and_decoder` | +0.323 | &minus;0.620 | &minus;0.508 | +0.376 |
| `all` | +0.323 | &minus;0.620 | &minus;0.508 | +0.376 |

Magnitudes are substantial (0.3–0.6), not near-zero-and-noisy, and the sign is uniformly wrong for every category at every pathway.

### The inversion is pathway-invariant, not layer-localized

Per-voxel cross-pathway correlation between `dec1_only`'s cosine and `encoder`'s cosine (the two most different interventions tested — the most local possible update vs. an update confined entirely to the far end of the network) is **r=0.98** (n=72). Mean `|cos|` shrinks only mildly and monotonically as more upstream parameters are included (0.472 → 0.470 → 0.463 → 0.457 from `dec1_only` to `encoder_and_decoder`) — a small magnitude attenuation, not a sign change, not even a partial one.

**There is no parameter subset tested, from the most local (`dec1` alone) to the most upstream (`encoder` alone), where the sign flips to correct for any category.** The experiment did not find a layer where the inversion "enters" — it found that the inversion is already fully present at the most local possible intervention and stays essentially unchanged in sign and magnitude no matter how much more of the network's shared parameterization is included.

---

## This changes what the isolation checks' earlier TP/TN result meant

`PHASE_E25_C62_MECHANISM_AUDIT.md`'s Level 4 found TP and TN move with the *correct* sign in the **realized, 15-step AdamW-accumulated, full-batch (many-anchor) trajectory** (TP: 92% correct, TN: 79% correct). This experiment tests a different, more isolated quantity — a single anchor's own gradient, applied alone, with no AdamW and no other anchors — and finds TP/TN's isolated single-anchor probe is **wrong-signed 0% of the time**, the same as FN's already-established isolated-probe result from the prior isolation checks.

These are not contradictory findings about the same measurement — they are two different measurements that had never been directly compared before this experiment, because the isolation checks only ever ran the single-anchor probe on FN. **This experiment is the first time TP/TN's own isolated single-anchor probe was measured, and it reveals the wrong-sign phenomenon is not FN/FP-specific at all — it is a property of the isolated single-anchor gradient-to-representation mapping for every voxel, regardless of class or current prediction correctness.** What differs between categories is not whether the isolated probe is wrong-signed (it always is) but how that isolated-probe signal combines, across many simultaneously-active anchors and 15 real AdamW steps, into a realized trajectory — and that aggregation process evidently rescues TP/TN's sign (Level 4) while failing to rescue FN's (also Level 4, and the isolation checks' Check 2).

---

## Honest assessment: what this experiment does and does not establish

**Established directly**: the sign inversion, at the single-anchor, single-step, no-AdamW level of measurement, is a general property of this network's parameter-to-representation mapping for SC-TAM's construction — not something introduced by any specific layer or parameter subset among those tested, and not something specific to error voxels.

**Not established, and this report does not force a conclusion either way**:

- **Whether a genuinely different pathway split (not tested here) would localize it.** The groups tested were coarse (whole-encoder, whole-decoder, `dec1` alone) — a finer split (e.g. per individual encoder/decoder block, or isolating `BatchNorm` parameters specifically, which are a structurally different kind of parameter from convolution weights and were not tested as their own separate condition) might behave differently. This was not run.
- **Whether the direction `ŵ` itself, or the sign convention SC-TAM assigns to each ground-truth class, needs reconsidering** — the user's own alternative framing, raised explicitly as a possibility this result should not foreclose. This experiment's design (measuring `cos(Δz, ŵ)` against a fixed, pre-decided "correct" sign) cannot by itself distinguish "the network's Jacobian genuinely inverts a correctly-targeted direction" from "the assumed correct direction for at least some of these categories was never right in the first place." Both are consistent with a uniformly wrong-signed isolated-probe result across all four categories. This is a real, open interpretive question this report flags rather than resolves.
- **Why aggregation (many anchors + AdamW) rescues TP/TN's sign but not FN's**, given that all four categories show the identical wrong-signed pattern at the isolated single-anchor level — this is now the sharper, better-defined open question the investigation has converged on, not "which layer is broken."

---

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e25/run_c62_jacobian_localization.py` | Full parameter-group localization implementation |
| `experiments/exp_e12_eggo_m/e25/jacobian_localization_results/jacobian_localization_C62.json` | Raw per-voxel, per-pathway results (72 records) |
| `PHASE_E25_C62_ISOLATION_CHECKS.md` | The isolation checks this experiment follows up on |
| `PHASE_E25_C62_MECHANISM_AUDIT.md` | Level 4's original realized-trajectory TP/TN/FP/FN finding, now reconciled against this experiment's isolated-probe result |
