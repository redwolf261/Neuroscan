# What Not To Do — Lessons Established This Session

**Purpose**: a plain checklist of mistakes, traps, and bad habits this project has already fallen into and corrected. Self-contained — no other file needs to be read to understand any item.

---

## Statistical / analysis traps

1. **Don't trust a single summary statistic (especially a linear-regression R²) without checking whether the underlying relationship is even shaped like something that statistic can detect.** Several times, a near-zero R² or correlation was initially reported as "no relationship," when the real relationship was a sharp on/off (binary) effect that linear regression is mathematically blind to. Always cross-check a surprising "no effect" result against a different, non-linear-assuming statistic (e.g. Spearman rank correlation, or directly binning and looking at the data) before accepting it.

2. **Don't trust a large effect size without checking whether it's driven by a handful of outliers.** More than once, a seemingly strong correlation or R² collapsed to near-zero (or reversed) once the top few extreme data points were removed. Always re-run the same test excluding the most extreme few points and see if the conclusion survives.

3. **Don't accept a positive-looking result without a randomization/permutation control, especially when the "signal" involves dividing by a small or noisy denominator.** One idea (the degradation-trajectory mechanism) looked genuinely promising — passed outlier checks, passed a bootstrap confidence interval — and still turned out to be a pure mathematical artifact, caught only when the model's real predictions were replaced with randomly shuffled ones and the "signal" didn't go away. If a result matters, deliberately try to break it with a randomization test before trusting it.

4. **Don't divide by a small or model-dependent number without checking how often that number is actually close to zero.** Multiple bugs in this project were exactly this shape: a ratio's denominator was a model's own predicted confidence, which is frequently near-zero for hard/small cases — producing meaningless, huge, or NaN-adjacent values silently. Always check the denominator's distribution before trusting a ratio built from it, especially in the population you care about most.

5. **Don't treat "detection" and "quality of what was detected" as the same question.** An improvement in one does not imply an improvement in the other, and conflating them produces wrong mechanistic explanations. Measure both explicitly when auditing why something worked.

6. **Don't let a small-sample or single-subject result stand in for the full population.** An early finding based on a handful of examples reversed once measured on the complete validation set. Prefer the full dataset over a promising-looking sample whenever it's affordable to compute.

7. **Don't confuse two different definitions of the same-named metric.** This project has both a "pooled" Dice score (computed by lumping several scans together first) and a "per-subject" Dice score (computed per scan, then averaged) — they differ by a couple of percentage points and are not interchangeable. Always state which one is being reported, and never let a conclusion depend on silently switching between them.

---

## Experimental design traps

8. **Don't accept an aggregate/softened justification when an explicit, strict numeric criterion was set in advance.** If a threshold was agreed beforehand (e.g. "must retain most of the corrective signal"), a result that technically improves things on average but fails the literal stated criterion should be treated as failing — not rescued by re-framing the criterion after seeing the result.

9. **Don't assume a component's "size" and a more sophisticated derived quantity (like a resolution-based measurement) are automatically different things — check the correlation directly, and check whether the derived quantity still adds anything after size is properly (non-linearly) controlled for.** A cheap, sophisticated-sounding idea can still turn out to be "just size wearing a costume" if this check isn't done explicitly.

10. **Don't declare a training instability or an anomalous run "a finding" without first ruling out a mundane implementation cause.** A model that produced wildly unstable validation scores at one resolution turned out to be a well-known technical artifact (batch normalization breaking down at batch size 1, forced by GPU memory limits) — not evidence about the resolution itself. Stop and diagnose before reporting a surprising result as real.

11. **Don't calibrate a new loss term's weight by matching its typical *value* to the existing loss's typical value, without also checking its *gradient* magnitude.** A new loss term computed over a small subset of voxels can have a totally reasonable-looking value but a wildly disproportionate effect on training, because its gradient is much steeper than a loss averaged over an entire large volume. This exact mismatch (value looked fine, gradient was ~75x too large) caused a full training collapse. Always check gradient magnitude, not just loss magnitude, when calibrating a new loss term's weight.

12. **Don't build a fairness comparison (e.g. "method A vs. a matched control method B") around only one matched statistic (e.g. matching the mean) if a second one (e.g. matching the variance) can't also be reasonably matched — and if it can't, say so honestly rather than silently proceeding as if the comparison were perfectly fair.**

13. **Don't skip a literature/novelty check before spending a large amount of compute on an idea that's meant to be a genuine contribution, not just a Dice-score improvement.** And when doing that check, actually read the mathematical mechanism of every close-looking paper — not just its title or abstract — since two ideas can sound identical in plain English while being mathematically unrelated, or vice versa.

---

## Engineering / performance traps

14. **Don't assume you've found the real performance bottleneck just because you fixed *something* that seemed slow — measure again after the fix, and profile properly (not just guess) if the fix doesn't actually help.** One slowdown was wrongly attributed to a resize operation at first (fixing it did nothing); the real cause (many small individual operations issued one at a time instead of batched together) was only found via a proper profiler.

15. **Don't trust wall-clock timing measurements on GPU code without an explicit synchronization point.** GPU work runs asynchronously; a timing measurement can silently include queued-up work from a previous step, making costs look like they're mysteriously "growing" across iterations when they're actually just being measured at the wrong moment.

16. **Don't rewrite a core computation for performance without directly verifying, numerically, that the fast version produces the same output as the slow-but-known-correct version.** ("It runs without crashing" is not the same check as "it computes the same number.")

17. **Don't write a synthetic/unit test with two pieces of test data that were constructed independently of each other, and then assert a relationship between them that would only be guaranteed to hold for *real*, correctly-paired data.** A background-leak test failed once purely because the hand-made mask and the hand-made label volume in the test weren't actually built to correspond to each other — a flaw in the test, not the code under test.

18. **Don't let a caching or lookup step silently fall back to a default value without surfacing that it did so.** A calibration script silently returned a meaningless fallback number for over an hour of implied "calibration" because its data cache never actually matched anything in the real batches — nothing crashed, it just quietly did nothing useful. Any such fallback path should be loud (a counter, a warning, an assertion) not silent.

---

## Reporting / process discipline

19. **Don't let a diagnostic script or "novel finding" bypass the same skepticism applied to everything else, just because it's a diagnostic rather than a training run.** Several of the worst near-misses in this project (the degradation-trajectory artifact, the sign-convention bug) were in analysis code, not training code — analysis code is not automatically trustworthy just because it doesn't train a model.

20. **Don't report a failed pre-registered gate as a "finding" without first checking for a mundane bug.** When a controlled experiment collapses dramatically and unexpectedly, the right first move is suspicion and root-cause diagnosis, not writing up the collapse as "the mechanism doesn't work."

21. **Don't keep tuning a null result indefinitely hoping it flips.** Several ideas were explicitly, deliberately frozen as null once hyperparameter sweeps or multi-seed checks confirmed the null was robust — continuing to tune at that point would be trial-and-error, not research.

22. **Don't write a report for insiders when the audience doesn't have repo access.** A report that says "see PHASE_X.md" or references internal script/variable names is not self-contained — assume the reader has nothing but the document itself.
