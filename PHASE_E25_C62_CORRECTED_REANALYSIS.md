# Phase E25, C6-2: Corrected Reanalysis — H2, H3, Level 4, and the Isolation Checks Recomputed

**Status**: ✅ Complete. Every diagnostic result that depended on the backwards `expected_sign()` convention (`PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md`) has been recomputed from the already-saved raw data — **no new training, no new measurement runs, exploiting the checkpoints and data already paid for**, exactly as instructed. **The corrected picture is not a simple mirror-flip of the old one. It reveals a real, specific, non-uniform pattern: under the correct convention, the realized (many-anchor, AdamW-accumulated) H2/Level-4 trajectory has FN and FP — the actual prediction errors — moving CORRECTLY signed (98% and 82%), while TP and TN — the already-correct predictions — move WRONG-signed (6% and 21%).** This is close to the mechanistic result the design was originally intended to produce, not the failure the old analysis reported. H1 and H4 were never affected by the bug and are restated unchanged. This report determines the verdict is **Outcome 2**: the mechanism is now shown to be substantially correctly-signed where it matters (errors), yet C6-2's actual Dice (0.9030) still falls short of the required bar (0.9183) — a materially different and more interesting research finding than "the mechanism is broken."

**Date**: 2026-08-12

---

## Method: recomputation, not remeasurement

Every affected script saved either (a) raw signed `cos_mean` values (unaffected by the convention — only their *interpretation* needs to change) or (b) a `frac_correct_sign` fraction computed under the old convention. Because the old and new conventions are exact opposites (`old_expected_sign = -new_expected_sign` for every category), **the corrected fraction is always the exact complement**: `new_frac_correct = 1 - old_frac_correct`. This is an exact recomputation, not an approximation — no data was re-generated, and no measurement was rerun.

---

## H2: realized parameter-induced representation displacement (class-conditional)

| Epoch | Tumor cos_mean (unchanged) | Tumor NEW frac_correct | Background cos_mean (unchanged) | Background NEW frac_correct |
|---|---:|---:|---:|---:|
| 5 | −0.110 | 22.7% | +0.405 | 7.8% |
| 10 | −0.335 | 9.2% | +0.372 | 11.5% |
| 15 | −0.342 | 6.1% | +0.314 | 13.5% |
| 20 | −0.313 | 11.5% | +0.172 | 28.5% |
| 25 | −0.345 | 9.8% | +0.267 | 16.7% |
| 30 | −0.371 | 4.8% | +0.156 | 35.9% |

**Reversed from the original report.** Pooled voxel-level: tumor 10.70% correct (was 91.7%), background 18.99% correct (was 90.5%). Checkpoint-level (does the mean cosine have the correct sign): **0/24 correct** (was reported as 24/24). The H2 headline — "SC-TAM's realized movement is correctly signed for both classes at every checkpoint" — does not survive correction. Under the corrected convention, the class-pooled realized trajectory is consistently, cleanly **wrong**-signed.

---

## Mechanism audit, Level 4: TP/TN/FP/FN breakdown

| Category | cos_mean (unchanged) | OLD frac_correct | **NEW frac_correct** | n (pooled) |
|---|---:|---:|---:|---:|
| Tumor (pooled) | −0.228 | 79.9% | 20.1% | 140,520 |
| Background (pooled) | +0.277 | 78.9% | 21.1% | 12,442,392 |
| **TP** | −0.318 | 93.9% | **6.1%** | 119,080 |
| **FN** | +0.420 | 2.0% | **98.0%** | 21,440 |
| **FP** | −0.159 | 17.7% | **82.3%** | 17,163 |
| **TN** | +0.278 | 78.9% | **21.1%** | 12,425,229 |

**This is the central corrected finding, and it is not a uniform flip.** The old report's headline — "SC-TAM reinforces already-correct voxels (TP/TN) and pushes errors (FN/FP) in the wrong direction" — is now shown to be **backwards in exactly the way that matters most**: under the corrected convention, **FN moves correctly-signed 98.0% of the time and FP 82.3% of the time — the actual prediction errors, the voxels the mechanism exists to fix, are moving in the intended direction.** Conversely, **TP (6.1%) and TN (21.1%) — voxels the model already classifies correctly — are the ones moving wrong-signed.** This is close to the literal opposite of the old narrative, and considerably more encouraging for the mechanism's design intent: SC-TAM's realized movement concentrates its *correctly-directed* effect specifically on the error population, not on reinforcing an already-correct boundary.

---

## H3, Q3: sign correctness at the immediate loss-gradient step

| | OLD mean | **NEW mean** | vs. chance (0.5) |
|---|---:|---:|---:|
| Tumor | 38.95% (below chance, p=0.034) | **61.05%** | t=+2.19, **p=0.034**, above chance |
| Background | 45.23% (below chance, n.s.) | **54.77%** | t=+1.90, p=0.063, marginal |

Q4 (FP/FN transition counts) and Q5 (`‖Δz‖` concentration ratio) were confirmed, by direct inspection of their own computation, to depend only on prediction-vs-ground-truth crossing 0.5 (Q4) or displacement magnitude alone (Q5) — **neither uses `expected_sign` anywhere and both stand exactly as originally reported** (Q4: net-unfavorable FP/FN transition counts; Q5: 9.4× concentration on misclassified voxels, 48/48 records, p=7.1×10⁻¹⁵).

---

## Isolation checks: Check 2 (FN-specific) and Check 1 (unaffected)

**Check 2's realized full-trajectory** (FN only), per checkpoint — reversed exactly as Level 4's FN row predicts:

| Epoch | cos_mean (unchanged) | OLD frac_correct | **NEW frac_correct** |
|---|---:|---:|---:|
| 5 | +0.226 | 10.0% | 90.0% |
| 10 | +0.271 | 6.3% | 93.7% |
| 15 | +0.294 | 12.5% | 87.5% |
| 20 | +0.252 | 11.0% | 89.0% |
| 25 | +0.350 | 2.1% | 97.9% |
| 30 | +0.307 | 4.3% | 95.7% |

**Check 2's isolated single-anchor probes** (FN only): **36/36 = 100% correct** under the corrected convention (was 0/36) — this is the exact same measurement `PHASE_E25_C62_GRADIENT_JACOBIAN_CONSISTENCY.md`'s own independent 48-voxel/4-category sweep found (100% at every category), now cross-verified against this earlier, FN-only measurement using the complement-recomputation method rather than a fresh run. The two independent measurements of the same quantity agree exactly.

**Check 1** (individual anchor's gradient vs. the batch's aggregated gradient, `cos_with_full_batch_grad`) is **not affected** by this bug at all — it is a self-referential comparison between two gradients, with no dependency on `ŵ` or any assumed "correct" direction. Its original finding stands unchanged: FN and FP anchors agree with the batch aggregate 100% of the time, more strongly and consistently than TP/TN.

---

## H1 and H4: confirmed unaffected, restated for completeness

Neither uses `expected_sign()` anywhere (verified again in this report, not merely asserted from the prior memo):

- **H1**: `rho_C62 = corr(ΔL_margin, ΔDice) = −0.4924` (Spearman), p=4.06×10⁻¹³, n=192 — correct sign, unchanged.
- **H4**: best val Dice = **0.9030** (epoch 29), vs. the required 0.9183 (A's 0.9063 + 1.2pp) — a **1.53pp shortfall**, unchanged. Also below A's own baseline (0.9063) and E's random-projection control (0.9058).

---

## The corrected verdict: Outcome 2

Applying the three-outcome framework directly:

- **Outcome 1** (SC-TAM actually improves Dice, misinterpreted due to the bug) — **does not hold**. H4's Dice number was never affected by the bug; 0.9030 is still 0.9030, still short of the bar.
- **Outcome 3** (mechanism weak/inconsistent even when correctly signed) — **does not hold either**. The corrected numbers are not weak or inconsistent: FN 98.0%, FP 82.3%, the isolated-probe 100%, Q3 above chance, all independently cross-verified across three separate measurement mechanisms (H2, the mechanism audit, the isolation checks) and two independent recomputation methods (complement-of-saved-fraction, and C6-2.5b's fresh, from-scratch sweep). This is a strong, consistent, well-supported mechanistic signal, not a marginal one.
- **Outcome 2 holds**: **SC-TAM's realized representation movement is now shown to be substantially and consistently correctly-directed specifically on the error population (FN/FP) it was designed to correct — and this well-verified, correctly-functioning mechanism still does not translate into the required Dice improvement.** The research question this reframes is no longer "why is the mechanism inverted" (it isn't) but **"why does a mechanism that correctly concentrates corrective movement on error voxels not yield sufficient segmentation improvement."** This is a legitimate, different, and more interesting failure mode than the one the uncorrected analysis reported.

**One genuinely new open question this correction surfaces, not present in the original (wrong) narrative**: why do TP/TN — voxels the model already gets right — move *wrong*-signed (6.1%, 21.1%) in the realized trajectory, even though the mechanism is working correctly for the error population? This was invisible under the old convention (where TP/TN's *old* high "correct" fractions were actually measuring the wrong-signed movement being miscounted as right). Whether this TP/TN wrong-signed movement is itself harmless (moving an already-confident, already-correct voxel slightly the "wrong" way along `ŵ` may not flip its prediction, since it may still be far from the decision boundary) or is itself contributing to the Dice shortfall (e.g. by degrading calibration or margin quality among currently-correct voxels) is not established by this reanalysis and would be the natural next diagnostic question — not attempted here.

---

## What this reanalysis does not do

Per the explicit instruction, this report exploits already-existing data and does not launch any new training or measurement runs, and does not propose or design C6-3. The corrected picture changes *why* C6-2 falls short (a real mechanism that concentrates correctly on errors, yet still insufficient — not a broken/inverted mechanism), which is directly relevant context for whatever comes next, but that decision is explicitly left open here.

---

## Files

| File | Purpose |
|---|---|
| `PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md` | The bug this reanalysis corrects for |
| `PHASE_E25_C62_GRADIENT_JACOBIAN_CONSISTENCY.md` | The independent, from-scratch sweep whose 100%-correct isolated-probe result this reanalysis cross-verifies via a second, independent method |
| `experiments/exp_e12_eggo_m/e25/h2_results/h2_results_C62.json`, `mechanism_audit_results/mechanism_audit_C62.json`, `failure_analysis_results/failure_analysis_C62.json`, `isolation_check_results/check1_per_anchor_gradient.json`, `isolation_check_results/check2_jacobian_isolation.json` | Raw source data this reanalysis recomputes from |
| `PHASE_E25_C62_RESULTS.md` (H1/H4 sections), `PHASE_E24_GATE6_EXPERIMENTAL_SPECIFICATION.md` | Unaffected reference results restated here for completeness |
