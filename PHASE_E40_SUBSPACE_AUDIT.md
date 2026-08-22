# E40 — D4 Optimization Subspace Audit

**Verdict: KILL.** D4's independent-optimization-direction diversity is real but confined to a single condition, does not survive a permutation safeguard, and does not distinguish the model's actual, known performance ordering. Per the project's own pre-declared rule, a partial, non-surviving positive result is not rescued by reporting only its most favorable subset.

No model was trained. This document is self-contained.

---

## 0. A necessary scope note, decided before any analysis began

The three prior phases' saved data (E36, E38, E39) contain only scalar gradient sizes and angles-to-the-main-gradient at each individual measurement point — never the raw gradient directions themselves, and never any comparison between one measurement and another. Building the directional-covariance/effective-rank quantities this audit requires is not possible from that data alone. This was checked directly, not assumed, before concluding that a real, disclosed scope expansion was necessary: reloading the already-trained models (no training, inference only) to compute fresh, additional gradient measurements this phase specifically needs. This was agreed before proceeding.

Because the model has roughly 5.6 million trainable numbers, a literal covariance matrix over gradient directions would be about 5.6 million by 5.6 million — computationally impossible. This audit instead uses a standard, exact mathematical identity: for a modest number of direction vectors in an enormous space, the space's true "effective number of independent directions" can be computed exactly from just the pairwise angles between those vectors, without ever needing the full high-dimensional space itself. This was verified directly, on a small synthetic test case, before being trusted on real data — the two methods gave identical answers to eight decimal places.

## 1. Hypothesis

The previous phase (E39C) found that the raw *size* of D4's independent optimization contribution does not predict how much D4 supervision actually helps — a larger auxiliary weight always produced a larger contribution, but the real outcome (segmentation quality) rose and then fell, a shape the contribution's size never matched. This phase tests a different, more specific idea: perhaps it isn't how *much* independent information D4 supplies, but how *diverse* and *reproducible* that information is — does D4 repeatedly point in a rich variety of genuinely different, useful directions across different batches of data, in a way D2 does not?

## 2. What was measured

For every checkpoint tested, and for 15 independent, held-out batches of scans, this phase measured — for both D4's and D2's own training signal — the specific "leftover" direction each one points in after removing whatever part of it simply repeats the model's main training signal (this project's own established residual-projection method, unchanged from the previous phase). From those 15 leftover directions, two properties were computed: how many genuinely *different* directions those 15 examples effectively represent (an "effective rank" — a low number means the 15 batches all point roughly the same way; a high number means they point in many different ways), and how *similar* those 15 directions are to each other on average (a separate "stability" measure — do independent batches keep landing on the same leftover direction, or a different one each time).

## 3. D4 vs D2 effective rank

Pooled across all three of the pre-existing conditions this project has (a model using only D4 supervision, one using only D2, and one using both together) and all seven training checkpoints tested: D4's effective rank exceeds D2's in two-thirds of the comparisons, but this is **not** statistically distinguishable from chance (paired test, p=0.17).

A cleaner, narrower comparison is available inside the one condition where D4 and D2 are genuinely both active in the same model at the same time — the only condition where the two are directly, fairly comparable rather than one being architecturally forced to a value of zero. There, D4's effective rank is higher than D2's at every single one of the seven checkpoints tested, and this is statistically significant on its own (p=0.016). This is a real, disclosed finding, not hidden — but it comes from a single condition, seven data points, and does not, on its own, establish that this property is what makes D4 useful. Whether it does is what the next two sections test directly.

## 4. Cross-batch stability

The opposite of what a "more reproducible" story would predict: pooled across all three conditions, D4 is actually the *less* stable of the two two-thirds of the time (not statistically distinguishable from chance either, p=0.10) — independent batches do not agree with each other on D4's leftover direction any more reliably than they do for D2's.

## 5. Outcome test: does this actually predict the real result?

This is the test that matters most, and it is where the hypothesis fails decisively. Using the six controlled training runs from the prior weight-sweep phase — the only place this project has a real, ground-truth measurement of how much a given amount of D4 supervision actually improved the model — this phase asked whether the total amount of directional diversity accumulated during each of those six runs predicts how much that run's segmentation quality actually improved.

It does not, at a level any reasonable standard would accept. Using an exact test (checking every one of the 720 possible ways the six diversity measurements could have been paired with the six real outcomes, not a large-sample approximation), the real pairing's relationship is no more extreme than 30% of all possible random pairings would produce. The reproducibility/stability measure does even worse — no relationship at all (and its own leave-one-condition-out check swings wildly, from a strongly negative reading to exactly zero depending on which single condition is set aside — a hallmark of a result with no real, stable substance behind it).

## 6. Permutation safeguard on the D4-vs-D2 structure itself

The one interesting number from Section 3 — that D4's effective rank exceeds D2's, pooled across everything measured — was tested directly against 500 randomized versions of the same comparison, where which measurement belonged to "D4" versus "D2" was shuffled while keeping everything else about the data the same. The real difference between D4 and D2 does not clear this randomized comparison at a level worth trusting (the real result sits at the 87th percentile of the random shuffles — noticeably above the middle, but well short of anything that would count as a reliable, non-chance finding).

![figure placeholder — see figures/e40_subspace_audit.png]

## 7. Size confound

This project has, more than once, found an apparently interesting result that turned out to be secretly explained by plain tumor size. That specific failure mode does not apply here, for a structural reason rather than a statistical one: every quantity in this audit is computed from fixed batches of eight scans at a time, and the exact same fifteen batches — the same scans, in the same groupings — were reused for every single comparison in this document. Since tumor size, by construction, cannot vary between conditions when the underlying scans being measured never change, it cannot be responsible for any difference reported here. This is stated explicitly rather than left as an unaddressed gap.

## 8. Kill criteria, checked one at a time

- **D4 has no reproducible directional-structure advantage over D2** — partially true: real within the one condition where the comparison is architecturally fair, but not present in the pooled, honest comparison.
- **The apparent advantage disappears under permutation** — **yes.** Section 6.
- **Explained by gradient magnitude alone** — not directly tested as a separate factor here, but the prior phase (E39C) already closed this specific door for the related "size of contribution" quantity; this phase's own contribution is about direction, not size, and the negative result stands on its own regardless.
- **Explained by lesion size** — ruled out structurally (Section 7), not just statistically.
- **Does not distinguish the known D4/D2 performance ordering** — **yes, this is the decisive failure.** Section 5.
- **The result depends on a handful of batches/epochs** — **yes.** The only statistically real signal in this entire audit comes from a single condition's seven checkpoints, and does not survive being combined with the other two conditions this project actually has.

Per the explicit standing instruction not to keep tuning or re-selecting a measurement after seeing which version of it looks best: the pre-specified effective-rank measure is reported as a kill, in full, rather than searched for a better-looking alternative.

## 9. Verdict and next step, per the pre-agreed decision structure

**KILL.** This closes the specific question this phase was built to test: whether D4 supplies a reproducible, diverse set of independent optimization directions, and whether that diversity explains why D4 supervision helps. It does not — not because the underlying measurement failed technically (every check ran cleanly, matched its own verification tests, and behaved exactly as its mathematics predicted), but because the one place a real, checkable difference between D4 and D2 exists does not carry over to explaining the actual, measured outcome this whole research line cares about.

Per the decision tree agreed before this phase began: this result calls for stepping away from further gradient-level investigation. The next step is not another gradient measurement, and not another loss-weighting idea derived from gradient behavior — it is a return to the more basic, structural question of how the target itself changes shape under the model's own resolution-reduction step, and what a loss built directly around that transformation (rather than around anything to do with gradients) would need to look like.

## 10. Limitations

- Every quantity in this document uses a single, fixed seed and a single, fixed set of fifteen diagnostic batches — reused deliberately, for direct comparability with the two prior phases this one builds on, but not independently re-verified with a different random seed or a different batch selection.
- The one statistically real result in this document (Section 3's within-condition comparison) is disclosed in full because it is real and because hiding an inconvenient positive result would be exactly the kind of selective reporting this project has repeatedly guarded against — but it should not be read as evidence the underlying hypothesis is "almost right." Sections 5 and 6 test the two questions that actually matter (does this explain the real outcome; does it survive a fair randomized comparison), and both come back negative.
