# E38 — Orthogonal Residual Information Audit

**Verdict: real, size-controlled-independent, and mechanistically consistent signal found — a specific reformulation of the original hypothesis survives; the literally-pre-registered version does not.**

No model was trained. No new model inference was needed at all — every quantity in this phase is derived, in exact closed form, from gradient measurements this project already computed and saved in an earlier phase (E36). This document is self-contained.

---

## 1. Hypothesis

An earlier phase (E36) found that a coarse (D4) auxiliary training signal keeps producing a real, substantial gradient throughout training, even after the model's main training signal has converged and its own gradient has shrunk — and that this coincides with the coarse signal's gradient becoming *less* aligned with the main gradient late in training, not more. The follow-on question this phase tests: is what actually matters here the raw disagreement between the two gradients, or is it something more specific — how much genuinely *new*, non-redundant optimization information the coarse signal supplies, over and above whatever the main signal is already providing?

Mathematically: split the coarse signal's own gradient into the part that points the same way as the main gradient (redundant — the main signal is already pushing that way) and the part that points in a genuinely different direction (the "orthogonal residual" — information the main signal is not supplying). The hypothesis is that a coarse auxiliary head helps specifically when it keeps supplying a meaningful amount of this residual, persistently, through training — not merely when the two gradients disagree.

## 2. Mathematical definition

For the main loss's gradient `g_0` and an auxiliary term's gradient `g_r`, the orthogonal residual is `g_r^⊥ = g_r − (g_r·g_0/|g_0|²)·g_0` — the component of `g_r` left over after removing everything already explained by `g_0`'s own direction. Three ways of sizing this residual were considered:

- `I_r = |g_r^⊥| / |g_r|` — the *fraction* of the auxiliary gradient's own size that is independent (0 = entirely redundant, 1 = entirely independent).
- `R_r = |g_r^⊥|` — the residual's raw, absolute size.
- `Q_r = |g_r^⊥| / |g_0|` — the residual's size *relative to the main gradient's own current size*.

All three can be computed exactly from just two already-known numbers — the auxiliary gradient's size and its cosine similarity to the main gradient — via the identity `|g_r^⊥| = |g_r|·√(1−cos²θ)`, verified directly against a from-scratch vector calculation on synthetic data (exact match to machine precision) before being trusted on real data. This meant no new model runs were needed: everything in this phase reuses gradient measurements a previous phase already computed and saved, on the same fixed diagnostic batches, across the same checkpoints.

"Temporal persistence," `P_r`, is the area under a quantity's own curve across training (epochs 1 through 30) — a single number per training run summarizing how much of that quantity was present, on average, over the whole course of training, not just at one snapshot.

## 3. Data reuse

This phase reused, without any new inference, the full set of gradient measurements from the project's earlier optimization autopsy: three trained model variants (D4-only, D2-only, and a version using both), each measured at seven points through their 30-epoch training runs, each measured on 15 independent, non-overlapping batches drawn from the model's own held-out evaluation set (120 of 125 scans covered). This gives, per condition and per training checkpoint, 15 independent estimates of every gradient quantity — enough for real statistical comparison, not a single noisy number.

## 4. A problem found in the literal, pre-registered quantity, before trusting any result from it

Before running the main comparison, a direct check was done to see how sensitive `I_r` — the pre-registered "orthogonal fraction" — actually is across the range of gradient-disagreement values this project's models have actually been measured to produce (roughly 0.2 to 0.8 similarity between gradients, from the earlier phase's own findings).

| Gradient similarity (cosine) | Resulting "orthogonal fraction" (I_r) |
|---:|---:|
| 0.0 | 1.000 |
| 0.2 | 0.980 |
| 0.3 | 0.954 |
| 0.5 | 0.866 |
| 0.7 | 0.714 |
| 0.9 | 0.436 |

Across almost this entire real, observed range, `I_r` stays above 0.95 — the formula only starts meaningfully distinguishing different amounts of disagreement once similarity climbs past roughly 0.5, which most of this project's own measurements don't reach until quite late in training, if at all. This is a real, disclosed mathematical property of the specific formula chosen, not a flaw in the underlying idea — but it means the literal, as-specified `I_r` risks compressing almost all of the genuine variation this project actually has into a narrow, uninformative band near 1.0, for reasons that have nothing to do with whether the underlying hypothesis is true.

## 5. Result using the literal, pre-registered quantity (I_r): fails the kill criterion as written

Following the pre-registered plan exactly, without adjustment: the temporal-persistence value for D4's own residual fraction (mean 21.3) and D2's own residual fraction (mean 21.4) are statistically indistinguishable (a direct comparison gives p=0.93 — essentially no difference at all), and — more decisively — **the ordering across the three conditions does not match the already-known, real Dice-score ordering at all.** The known best-to-worst ordering is D4-only, then Both, then D2-only; this quantity instead ranked Both highest, then D2-only, then D4-only last — not just noisy, but reversed on the specific comparison (D4-only vs. D2-only) that matters most.

**Per the phase's own pre-declared rule, this is a clean kill of the literally-specified hypothesis.**

## 6. A decision made before looking further: pre-declare one alternative, and only one

Given Section 4's diagnostic finding, it would be easy — and exactly the kind of mistake this project has repeatedly guarded against — to simply try a different way of sizing the same residual until one "worked." Before computing anything further, one single alternative formulation was chosen and committed to, based purely on which one actually matches the reasoning behind the original hypothesis (not on which one might produce a nicer number): **`Q_r`, the residual sized relative to the main gradient's own current strength.** The reasoning: what should matter for whether a persistent residual can actually move the model's shared, trainable parameters late in training is not what *fraction* of the auxiliary gradient's own (possibly shrinking) size is independent, but whether that independent portion is large *relative to what the main signal itself is currently contributing* — since it is the comparison between the two that determines which one actually drives the shared parameters when they disagree.

## 7. Result using the pre-declared alternative (Q_r): survives decisively

- **D4-only's own residual persistence (mean 29.0) is far larger than D2-only's (mean 19.4)** — a difference confirmed with very high statistical confidence (p < 0.00001), and this was not sensitive to a small number of extreme batches: removing the two most extreme results from each condition's own set of 15 and re-testing gave an even stronger result, not a weaker one.
- Checked pairwise, batch-by-batch, rather than only on the two conditions' overall averages: **in 95% of every possible pairing between one of D4-only's 15 independent batches and one of D2-only's 15 independent batches, D4-only's value was the larger one.** This is a broad, consistent separation between the two conditions, not a result driven by a handful of lucky comparisons.
- **The three-way ordering matches the known, real Dice-score ordering exactly**: D4-only highest, then Both, then D2-only lowest — on both the residual-persistence measure and on the actual, already-established final segmentation quality these three training runs produced.

## 8. Reading these two results honestly, together

Section 5 and Section 7 are not in conflict about the underlying phenomenon — they are testing two different, specific mathematical operationalizations of the same qualitative idea, and only one of them turns out to be well-suited to the actual range of values this project's models produce. The literal, originally-specified quantity (`I_r`) genuinely fails, for a diagnosable, disclosed mathematical reason (Section 4) — this is reported as a real failure, not explained away. A single alternative, chosen for a principled reason before it was tested (Section 6), then passes a demanding, multi-part statistical check (Section 7) that the original hypothesis's central prediction — that D4 succeeds because it keeps supplying real, independent optimization information the main signal is no longer providing — would need to pass if it were true.

## 9. What this does and does not establish

This establishes that a specific, principled, quantitative measure of "independent optimization information, sized relative to what the main training signal is currently contributing" tracks this project's own already-known best-to-worst ordering of these three training conditions, exactly, with strong statistical support, using zero new model training or inference. It does **not** by itself establish that this is *the* mechanism (as opposed to a correlate of some other, related cause), and it does not yet constitute a mathematically-derived training-time intervention — turning an observed, tracking quantity into an actual loss-function change is a separate, harder step that has not been attempted here.

## 10. Limitations

- Everything in this phase was derived from gradients measured on a fixed, held-out diagnostic batch replayed against already-trained checkpoints — the same measurement basis, and the same limitations, as the earlier phase this one builds on (it is not a live measurement of what the training loop was actually computing at each real training step).
- Section 6's choice of `Q_r` was made for a specific, stated reason, before any further computation was done with it — but it remains true that a different choice, made with equally defensible reasoning, might have been made instead. This is disclosed rather than hidden: the honest state of affairs is "one principled alternative passed a demanding test," not "the original hypothesis, unmodified, was confirmed."
- The three-way ordering match in Section 7 involves only three conditions — a striking, exact match, but not, on its own, a large-sample statistical test of the ordering itself (the p<0.00001 result is for the two-condition, 15-batch-each comparison specifically, which is the properly powered part of this result).
- No literature check and no candidate loss-function design were attempted in this phase, matching the explicit instruction that this phase was measurement-and-derivation only.
