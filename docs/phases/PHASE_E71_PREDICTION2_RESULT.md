# Phase E71 — Prediction 2 Result: FAIL (200 labels) → PASS (1126 labels, marginal)

## UPDATE: scaled re-test PASSES, confirming the diagnosis below

The diagnosis in this document (undertrained head, 200 labels) was tested
directly: re-ran with ALL 1126 training subjects (not 200) and 150 epochs (not
30), same architecture, same seed, same held-out validation population, per
this document's own proposed next step.

| Quantity | 200-label run (FAIL) | 1126-label run (this update) |
|---|---|---|
| Spearman(predicted, measured), held-out | +0.540 to +0.701 | **+0.866** (p=7.4×10⁻³⁹) |
| Predicted range (span) | 0.43 (compressed) | **0.83** (matches measured 0.81) |
| Spearman(predicted, native_size) | **+0.395** (wrong sign) | **−0.190** (correct sign, p=0.035) |
| E48's own reference | −0.454 | −0.454 |
| Partial Spearman (predicted, measured \| size) | +0.821 | **+0.855** |

**Verdict: PASS, but the size-direction match is marginal, not strong.**
ρ=−0.190 (permutation p=0.035) is the correct sign and clears the pre-declared
p<0.05 bar, but it is noticeably weaker than both E48's own −0.454 and the
direct measured-vs-size relationship on this same population (−0.353). The
partial correlation (0.855) is strong and confirms the predictor carries real
signal beyond a simple size proxy — this is the more decisive number.

The size-binned breakdown now tracks the measured pattern reasonably well
(predicted: 0.411 → 0.366 → 0.279 → 0.294 → 0.312 vs. measured:
0.475 → 0.346 → 0.255 → 0.300 → 0.231), though the largest-lesion bin still
slightly overshoots — a residual version of the original compression problem,
much reduced but not fully gone.

**Honest reading**: the original FAIL was substantially a data-scale artifact,
as diagnosed. CDCG clears prediction 2 on this evidence, but "clears" here
means a real, direction-correct, partially-independent signal — not an
emphatic confirmation. Per the design doc's own decision rule this is enough
to proceed toward the gating-mechanism design (Section 3.3) in a future phase,
but any eventual Dice claim from that mechanism still needs the full 3-seed
discipline this project has applied consistently (and which just correctly
killed CAS).

Artifacts from the scaled run: `E71_prediction2_scaled_summary.json`,
`E71_val_per_subject_table_scaled.json`, `E71_aux_head_state_scaled.pt`.

---

## Original result and diagnosis (200-label run, superseded above but kept for the record)

## Result

| Quantity | Value |
|---|---|
| Reconfirm: Spearman(predicted, measured), held-out | +0.540 (p=8.2×10⁻¹¹) — overnight run: +0.701 |
| **Spearman(predicted d̂ᵢ, native_size)** | **+0.395** (p=5.2×10⁻⁶) |
| E48's own reference (measured drop vs. size) | **−0.454** |
| Reference check: Spearman(measured dᵢ, size) on this exact val population | **−0.353** (p=5.5×10⁻⁵) — reproduces E48's finding closely |
| Partial Spearman(predicted, measured \| size), re-verified | +0.821 (p=1.0×10⁻³¹) — overnight: +0.904 |

**Verdict: FAIL.** The predicted sensitivity d̂ᵢ correlates with native lesion size
in the **opposite direction** from E48's own established finding, and from the
measured label itself (which correctly reproduces E48's ρ≈−0.45 on this exact
population, confirming the labeling pipeline is correct).

## Diagnosis: this looks like an undertrained-head artifact, not a fundamental flaw

Two observations, both computed from the persisted per-subject predictions
(`E71_val_per_subject_table.json`), point to a specific, correctable cause:

1. **Predicted range is severely compressed relative to measured**: predicted
   spans [0.106, 0.532] (range 0.43) vs. measured [−0.028, 0.781] (range 0.81) —
   classic regression-to-the-mean behavior from a small model trained on only
   200 labels for 30 epochs.
2. **Size-binned means show the disagreement is concentrated at small lesion
   sizes specifically**:

| Size bin (native voxels) | n | Mean measured | Mean predicted |
|---|---|---|---|
| 7,285–49,858 (smallest) | 25 | **0.475** | 0.255 |
| 50,014–75,231 | 25 | 0.346 | 0.288 |
| 76,330–103,083 | 25 | 0.255 | 0.286 |
| 104,014–137,297 | 25 | 0.300 | 0.342 |
| 138,051–225,535 (largest) | 25 | 0.231 | **0.344** |

The predictor's largest error is exactly where it matters most for this
project's whole causal thread: it substantially **underestimates** sensitivity
for the smallest lesions (predicts 0.255 vs. true 0.475) and **overestimates**
for the largest (0.344 vs. true 0.231). With only 25 subjects per bin and 200
total training labels, the small 2-layer MLP head likely has not seen enough
examples of the high-sensitivity small-lesion regime to learn it — consistent
with, though not proof of, a data/capacity limitation rather than a conceptual
failure of the underlying idea (the raw correlation with the measured label,
+0.54 to +0.70 across two independent training runs, is real and substantial;
the head is predicting *something* real, just not calibrated correctly across
the size range).

## What this means for CDCG

Per the design doc's own pre-registered rule: **do not proceed to implementing
the gating mechanism (Section 3.3) with this result standing as-is.** A head
that inverts the size relationship on the population where the effect matters
most cannot be trusted to gate coarse/fine mixing sensibly.

This is not necessarily the same kind of failure CAS had (a coherent, real,
alternative mechanism replacing the intended one). This looks more like a
**data-scale problem**: 200 labeled subjects, 30 epochs, is a small
first-pass, chosen for the cheapest possible test per the design's own
"cheapest, first, most falsifiable test" instruction — and it may simply be
undertrained.

## Honest next step, not yet run

Before concluding CDCG is dead, the cheap, disciplined thing to check is
whether the size-direction failure is a labeled-data-scale artifact: retrain
the same auxiliary head with (a) more labeled training subjects (e.g. all
1126 training subjects, not 200) and/or (b) more epochs, and re-run this
exact prediction-2 test. If the size-direction mismatch persists with more
data, that would be a real, structural finding — the network's bottleneck
representation predicts *ablation impact* well but does NOT organize that
information along the size axis the causal chain (E48) says matters, which
would itself be a genuinely interesting (if negative) result. If it resolves
with more data, prediction 2 should be re-run and re-judged before any
further step.

This has NOT been run yet — reporting the FAIL honestly first, per this
project's own discipline, rather than immediately re-running until a
preferred answer appears.

## Artifacts

- `experiments/exp_e12_eggo_m/e71/check_cdcg_prediction2.py`
- `experiments/exp_e12_eggo_m/e71/E71_prediction2_summary.json`
- `experiments/exp_e12_eggo_m/e71/E71_val_per_subject_table.json` (persisted
  per-subject predicted/measured/size, for any future re-analysis without
  re-running the ablation labeling)
- `experiments/exp_e12_eggo_m/e71/E71_aux_head_state.pt` (trained head weights)
