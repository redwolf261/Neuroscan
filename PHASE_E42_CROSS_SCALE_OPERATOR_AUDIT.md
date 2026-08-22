# E42 — Cross-Scale Operator Audit

**Verdict: KILL** (structural/definitional — decided before any residual was computed, before any real audit compute was spent)

No model was trained. No loss was proposed. No literature search was performed. This document is self-contained.

---

## 1. Exact operator recovery (the step that ends this phase)

The proposal's central object, `R4 = P4 - D4(P1)`, assumes a specific architectural fact: that the model produces one full-resolution prediction, and that its coarser predictions are (or should be) a resized version of that same prediction. This was checked directly against the real model code before anything else was done, per this phase's own explicit first instruction not to assume the operator.

**That assumption does not hold in this architecture.** The model produces three predictions, and they come from three entirely separate small sub-networks, not from one prediction being resized:

| Prediction | Resolution | Exact source |
|---|---|---|
| `P1` (`probs`) | 64³ | `seg_head(dec1)` — its own 1×1×1 convolution + sigmoid, reading `dec1` (32 channels) |
| `P2` (`aux_probs2`) | 32³ | `aux_head2(dec2)` — its own, separately-learned 1×1×1 convolution + sigmoid, reading `dec2` (64 channels) |
| `P4` (`aux_probs3`) | 16³ | `aux_head3(dec3)` — its own, separately-learned 1×1×1 convolution + sigmoid, reading `dec3` (128 channels) |

Sigmoid is applied inside each of these three heads, independently, at that head's own native resolution — there is no point anywhere in the model's computation where a probability map is resized to a different resolution. `D4(P1)` and `D2(P1)` as literally written in this phase's proposal — "downsample the full-resolution prediction" — describe an operation that **does not exist anywhere in the real forward pass**. It would have to be invented and computed after the fact, purely for this audit, comparing it against `P4`/`P2`, which come from a completely different computational path: different decoder stage, different channel count, different learned weights, trained against a different (though related) loss term.

The target-side operator, separately, is confirmed exactly as it was in the previous phase's own audit: `mask_d4 = avg_pool3d(masks, kernel_size=4, stride=4)`, `mask_d2 = avg_pool3d(masks, kernel_size=2, stride=2)`, applied directly to the binary ground-truth mask, with no thresholding step anywhere. This part of the operator recovery is solid and was not the source of the problem.

## 2. What this means for the proposed residual

`R4 = P4 - D4(P1)`, computed as specified, would not measure "does the model's prediction stay internally consistent when resized" — because the model never resizes a prediction, so there is no such consistency for it to break or preserve in the first place. What it would actually measure is: **the disagreement between two independently-parameterized sigmoid heads**, each with its own separately-learned bias, threshold behavior, and calibration, applied to two different feature representations that are related (one feeds into the other, causally, through the shared trunk) but not identical.

This distinction was investigated directly, not assumed, before deciding what to do about it. A search was made for a way to isolate the trunk's own scale-consistency behavior from this head-level bias — for instance, by reusing one single prediction head across multiple decoder stages, so that any measured disagreement could only come from the underlying features, not from three separately-trained final layers disagreeing for their own, unrelated reasons. This is not possible without new training: `dec1`, `dec2`, and `dec3` have different channel counts (32, 64, 128 respectively), so no existing head can be applied to more than one of them without first training a new, compatible projection — which would itself be new model training, explicitly forbidden for this phase.

## 3. Why this is decided now, before any residual is computed

The remaining choice was between two paths, both considered directly: compute `R4`/`R2` exactly as specified and attempt to statistically separate genuine trunk-level scale-consistency information from head-level bias after the fact, or conclude the audit here on structural grounds. The first path was rejected, for a reason specific to this project's own history: this project has repeatedly found — and been explicitly warned by its own accumulated experience — that a quantity built from two things that are trivially, structurally different for reasons unrelated to the phenomenon of interest tends to produce exactly the kind of misleading, hard-to-fully-explain-away positive result this project has spent many prior phases chasing down and killing. Attempting to statistically "control for" head bias after the fact, on a quantity that conflates two genuinely different sources of disagreement by construction, would very likely produce ambiguous results that could be argued either way — precisely the situation this phase's own instructions were written to prevent ("be aggressively skeptical," "do not spend another training run on a mathematically interesting but useless quantity").

Given that the clean version of the question (does the shared trunk's own resolution-reduction stay internally consistent) cannot be answered with the tools currently available without new training, and the available substitute conflates that question with an unrelated one (do three separately-trained heads happen to agree), this phase concludes here rather than spending real audit compute chasing a quantity whose interpretation would already be compromised before a single number is measured.

## 4. Sections not reached

Per this phase's own explicit instruction ("do not continue to later algorithm design if the verdict is KILL"), and because the kill here is structural rather than empirical, the following sections of the original audit plan were not executed, and are listed here only to state clearly what was not attempted, not to imply they were tried and failed: sanity checks on a computed residual, residual distributions, spatial analysis, the error-relationship test, the D4/D2/baseline comparison, the permutation safeguard, outlier and population checks, the detection-vs-quality split, and the redundancy/negative-control test. None of these were run, because the object they would have been run on — a cleanly-defined cross-scale residual — does not exist in a form this phase's own tools can isolate from head-level noise.

## 5. Verdict

**KILL.**

- The residual as specified conflates two different sources of disagreement (genuine trunk-level scale inconsistency, and independently-trained-head bias) that cannot be separated without new model training.
- No literal implementation of "downsample the full-resolution prediction" exists anywhere in the actual model; it would have to be manufactured for this audit alone, comparing it against outputs from a materially different computational path.
- A structurally compromised quantity was judged, directly and before computing it, not worth spending real audit compute on, consistent with this phase's own explicit warning against this exact failure mode.

## 6. What would need to be true for a future revisit

Not a next step for this phase, but recorded for completeness: a genuinely clean version of this idea would require either (a) training a new, shared, resolution-agnostic prediction head that all three decoder stages route through (a real architecture change, and a real training run), or (b) a different, non-architectural way of defining "scale consistency" that does not rely on comparing outputs of two separately-parameterized heads. Neither was attempted here, consistent with this phase's explicit prohibition on new training, new losses, and architectural changes.

## 7. Limitations of this document

This document itself is unusually short relative to the rest of this project's audits, because the kill decision was made at the structural/definitional stage, before any data was collected — consistent with the instruction to prevent wasted training/compute, but meaning this document cannot report empirical distributions, permutation results, or population checks, because none were computed. This is disclosed explicitly rather than padded with placeholder analysis.
