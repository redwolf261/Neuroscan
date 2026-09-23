# E41 — Target-Transformation Error Audit

**Verdict: KILL.** The model's actual residual error against the true, correctly-defined coarse target does not predict where coarse (D4) supervision helps — the central, decisive test comes back essentially exactly null, and survives no rescue attempt.

No model was trained. This document is self-contained.

---

## 0. A correction made before any measurement was taken

The idea motivating this phase started from a mathematically real observation: shrinking an image down to a coarser grid converts each coarse cell into a fractional "how much of this cell is really tumor" value, rather than a clean yes/no answer, and a model trained against a crudely thresholded (yes/no) version of that value would be throwing away real information. A short derivation also confirmed something genuinely useful: under the standard way of scoring a probability prediction, the mathematically correct target for a coarse cell is exactly that fractional occupancy value itself — not a thresholded approximation of it.

Before building anything, this project's own training code was checked directly — not assumed — to see whether it actually has this thresholding problem. It does not. The coarse-resolution training target this project already uses is the exact, correct fractional occupancy value; there is no thresholding step anywhere in the training pipeline for either the coarse or medium-resolution auxiliary supervision. The derivation above is correct, but it describes a fix for a gap that turned out not to exist in this codebase's real training loop.

Given that, this phase was reframed, before any measurement was taken, around the real remaining question: even with the mathematically correct target, the model still doesn't predict it perfectly — does the shape of that remaining, real prediction error explain where coarse supervision actually helps, once tumor size is properly accounted for?

## 1. What was measured

Using the model version that has both the coarse and medium-resolution supervision heads active together (matching this project's own established convention for a fair, same-model comparison), every one of the 749 real tumor fragments in the held-out evaluation set was checked against the model's actual predictions at both resolutions. For the 227 fragments that survive the model's mandatory image-shrinking step at all, this measured: the true fractional occupancy at each resolution (no thresholding, matching the real training target exactly), and the model's own prediction error against that true value, using the same style of scoring the model is actually trained with.

## 2. Does the coarse resolution have a harder residual-error floor than the medium one?

A modest, borderline-significant difference (p=0.057, just short of the conventional cutoff): the coarse resolution's typical error is very slightly lower on average than the medium resolution's, which on its own is a mild surprise (the earlier project finding was that the coarse task is *harder* to fully solve, not that its *typical* per-cell error is larger) — but roughly 71% of the difference between the two resolutions' error levels is not explained by tumor size at all. This part of the audit is not, by itself, either a clear support or a clear contradiction of the earlier finding; it measures a different quantity (average prediction accuracy per cell) than what the earlier phase measured (how much the loss's own optimization signal stays alive throughout training).

## 3. The decisive test: does this error predict where coarse supervision actually helps?

This is the test this whole phase existed to run, and the answer is unambiguous.

**Among tumor fragments the model already detects both with and without coarse supervision** (146 of the 227 measurable fragments — the population where "did coarse supervision improve the *quality* of an already-successful detection" is a well-posed, real question, matching this project's own earlier, established distinction between "found it at all" and "found it well"): the model's coarse-resolution residual error and the actual, real improvement in that fragment's segmentation quality from coarse supervision have **no relationship at all** — a raw correlation of essentially zero (rho = −0.002, on a scale where 0 means no relationship whatsoever), and adding this error measure to a model that already accounts for tumor size improves that model's explanatory power by two hundredths of one percent — a rounding error, not a real gain.

**Among tumor fragments the baseline model missed entirely**, whether the model's coarse-resolution error predicted which of those missed fragments coarse supervision went on to successfully rescue: also no relationship (rho = +0.09, not distinguishable from no relationship at this sample size).

**After properly removing tumor size's own effect** (using this project's own established, multi-transform nonlinear size-control method) before comparing the two remaining quantities directly: still no relationship (partial correlation = −0.03).

**Checked against 500 randomized versions of the same comparison**, specifically designed to preserve each fragment's own tumor size while breaking any real link between its residual error and its actual improvement: the real, observed result was **less** extreme than the typical randomized result (sitting at the 23rd percentile, not even approaching the far tail that would indicate a real, non-chance relationship). A genuine finding would need to sit near the very top or bottom of this distribution; this one sits comfortably inside the ordinary, expected range of pure noise.

**Removing the five most extreme error measurements and re-checking**: the result stays at essentially zero (rho = −0.017), confirming this null is not a fragile result being masked by a handful of unusual data points — it is a stable, consistent absence of any relationship.

![figure placeholder — see figures/e41_transformation_error.png]

## 4. The specific small-vs-large test could not actually be run

This phase's own design called for testing whether the residual-error relationship differs specifically between small and large tumor fragments — the single most important distinction this whole line of research keeps trying and failing to establish. In practice, once restricted to fragments the model detects both with and without coarse supervision, only 7 small (S1–S4 size range) fragments remained — far too few for any meaningful statistical statement. This is disclosed honestly rather than worked around: the small-lesion-specific version of this question remains genuinely untested by this phase, not because it was found to be null, but because the available data, once filtered down to the fragments this specific test requires, does not contain enough small examples to ask it. The large-fragment version of the same test (139 fragments) was fully testable, and also came back null (rho = −0.01).

## 5. Verdict

**KILL.** Not because the underlying mathematics was wrong — the derivation that the correct coarse target is the fractional occupancy value itself is correct, and this project's training pipeline already implements it correctly, which was itself worth confirming directly rather than assuming. The kill is because the natural next question this correct mathematical foundation raises — does the model's own remaining prediction error against that correct target explain where coarse supervision actually helps — has now been tested as thoroughly as this project's own established methodology allows (raw correlation, size-controlled partial correlation, a proper randomized safeguard, and an outlier-robustness check), on both of the two ways "helps" can reasonably be defined, and every single one of those tests came back at or extremely close to exactly zero.

## 6. What this leaves the project with

Two real, established facts survive this phase, from earlier work, unweakened: the model's training target at each resolution is already the mathematically correct one (this phase's own audit, Section 0), and coarse supervision does measurably help overall, by a modest amount (an earlier, independent project result, not retested here). What does not survive, after this phase and several before it, is any attempt to find a single per-tumor-fragment number — size, a resolution-derived ambiguity measure, a gradient-based quantity, or now a direct residual-error measure — that predicts, fragment by fragment, exactly where and how much that overall benefit comes from. The overall effect appears to be real without being explainable at the level of any individual fragment's own measurable properties tried so far.

## 7. Limitations

- All error measurements come from a single trained model, at its final checkpoint, using a single fixed evaluation pass — not repeated across multiple training runs or random seeds.
- The small-fragment version of the decisive test (Section 4) was not actually run, for the reason stated there — this is a real gap in coverage, not a tested-and-negative result, and should not be read as evidence either way about small fragments specifically.
- This phase measures a form of prediction *accuracy* (how close the model's number is to the true fractional occupancy); it does not re-measure the training *optimization signal* (how much active, ongoing gradient a loss term supplies), which is what an earlier, separate phase examined and which is not contradicted by anything found here — the two are related but genuinely different quantities.
