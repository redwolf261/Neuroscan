# E37 — Grid-Entropy Feasibility Audit

**Verdict: KILL** (the mechanism as hypothesized — grid-induced ambiguity, measured as binary entropy, increasing systematically at coarser resolution — does not hold for the small-lesion population this project cares about, and the failure is a genuine, provable mathematical property, not a measurement artifact).

No model was trained in this phase. This document is self-contained.

---

## 1. Hypothesis

An earlier phase of this project (E36) found that a coarse (16×16×16, "D4") auxiliary prediction head, used to give the model extra training signal at a coarse internal resolution, kept producing a substantial, unsolved training signal throughout training — while a medium-resolution (32×32×32, "D2") version of the same idea converged to a much smaller residual signal. E36 traced this, qualitatively, to boundary geometry: shrinking a tumor's outline down to a coarser grid turns more of that outline into ambiguous, partially-covered grid cells rather than clean, fully-covered ones.

This phase, E37, tests whether that qualitative story can be captured by one precise, per-tumor, per-resolution number — a measure of how "ambiguous" a tumor's outline becomes at a given grid resolution — and whether that number, specifically, is what predicts the model's actual difficulty at that resolution, independent of a tumor's plain size (a confound this project has been burned by before, in more than one earlier phase).

## 2. Mathematical definition

For every grid cell `Q` a tumor component `c`'s outline overlaps, its coverage fraction is `π_Q = |Y_c ∩ Q| / |Q|` (what share of that cell's volume is really tumor). The binary entropy of that coverage fraction, `H(π_Q) = -π_Q·ln(π_Q) - (1-π_Q)·ln(1-π_Q)`, is 0 when a cell is unambiguously all-tumor or all-background, and reaches its maximum when a cell is split exactly half and half. The component-level grid ambiguity is the entropy of every cell the component touches, weighted by how much of that component's own mass sits in that cell, averaged over the component's total mass:

`H_{c,r} = (1/|Y_c|) · Σ_Q |Y_c ∩ Q| · H(π_{c,Q})`

computed separately at the D4 (16³) and D2 (32³) grid resolutions, with the full 64³ resolution ("D1") computed as a reference (at D1, every grid cell is exactly one voxel of the component's own already-resized mask, so its entropy is always exactly 0 — verified directly before trusting anything downstream).

## 3. Data construction

All 749 tumor fragments across the project's full 125-scan held-out validation set were used — the same, already-verified fragment inventory this project has built up and cross-checked across several earlier phases, reused here rather than rebuilt from scratch. 227 of the 749 fragments survive the model's mandatory shrink-to-64³ preprocessing step at all (the rest vanish to zero voxels before the model or this measurement ever sees them, a repeatedly-confirmed finding from this project's earlier work) — only those 227 have a meaningful grid-ambiguity or model-error value to measure, and all statistics below are computed on that population.

Both the D4 and D2 entropy measurements, and the model's actual prediction error at each resolution, were computed from **one single trained model** — the version that has both the D4 and D2 auxiliary prediction heads active together — specifically so that the D4-vs-D2 comparison at the heart of this audit compares one consistent model's own behavior at two resolutions, not two different models that happen to have been trained separately.

A genuine subtlety was caught and verified directly, before it could distort any downstream result: a first check on a single, isolated one-voxel tumor fragment showed its entropy was **lower** at the coarser D4 resolution than at the medium D2 resolution — the opposite of what the hypothesis expects. This was not assumed to be a bug; it was checked by hand, and it is mathematically correct. When a very small amount of tumor mass is spread out (diluted) into a much larger coarse grid cell, its coverage fraction gets pushed *closer to zero* — meaning the cell reads as "basically background," which is **unambiguous**, low-entropy — rather than pushed toward the maximally-ambiguous halfway point. This single observation turned out to be the central finding of the entire audit, confirmed at full scale in Section 4.

## 4. D4 vs D2 entropy (Test A)

This is the test the hypothesis most needs to pass, and it fails as literally stated.

Across all 227 measurable fragments, only **59.5%** show higher entropy at D4 than at D2 — barely better than a coin flip, and a paired statistical test comparing the two only reaches borderline significance (p=0.05). Far more decisively: **broken down by size, the pattern is not noisy — it is a sharp, near-deterministic threshold effect.**

| Fragment size (post-resize, in voxels) | Fragments | Fraction with D4 more ambiguous than D2 |
|---|---:|---:|
| exactly 1 voxel | 47 | **0%** |
| 2–5 voxels | 21 | **0%** |
| 6–20 voxels | 11 | **0%** |
| 21–100 voxels | 15 | 20% |
| more than 100 voxels | 133 | **99%** |

For every fragment 20 voxels or smaller (post-resize) — which is most of the small-tumor population this whole line of work exists to help — **the coarser grid is never more ambiguous than the medium grid; it is always less.** This was investigated directly rather than treated as a puzzling anomaly to explain away: it follows immediately and unavoidably from the definition in Section 2. A small, fixed amount of tumor mass, diluted into an ever-larger grid cell, has its coverage fraction pushed toward zero, and entropy is lowest near zero. No reformulation of the entropy formula — a directional version, an asymmetric one, or any other monotonic reshaping of the same coverage-fraction number — escapes this, because the problem is not in the shape of the entropy function; it is in the arithmetic of averaging a small, fixed amount of stuff over a much bigger space. This was checked directly before finalizing the verdict, specifically to make sure a fixable formula choice wasn't being mistaken for a fundamental one.

## 5. Entropy → error (Test B)

Without controlling for size, the raw relationship between entropy and the model's actual measured error at that resolution is unstable across the two scales — negative at D4 (rho = −0.37), strongly positive at D2 (rho = +0.87). This instability is itself a red flag pointing straight at a size confound, investigated properly in Section 6.

## 6. Nonlinear size-controlled analysis (Test C)

Using this project's own already-established, standing methodology for this exact situation (raw size, log size, and cube-root size — not just a straight linear correction, exactly matching the caution this phase was explicitly told to apply) as a set of "size" predictors, and asking how much extra explanatory power entropy adds on top of that size model:

- At D4: adding entropy only improves the fit from R²=0.924 to R²=0.930 (a real but small gain) — and once size's effect is properly removed from both entropy and error before comparing them, the sign of the relationship **flips**, from raw rho=−0.37 to a genuinely positive, size-independent rho=+0.34 (p≈1×10⁻⁷).
- At D2: a similar, slightly larger effect — R² improves from 0.941 to 0.949, and the size-controlled relationship is rho=+0.45 (p≈1×10⁻¹²).

Both size-controlled relationships are real, positive, and statistically solid on their own. Native size alone is a much stronger predictor of error than entropy is (R²≈0.92–0.94 from size alone, versus a roughly 0.6–0.7 percentage-point improvement from adding entropy) — entropy is a small, secondary signal riding on top of an overwhelmingly size-dominated relationship, not a comparably strong one.

## 7. Entropy vs boundary fraction (Test D)

This project's own earlier phase (E36) had already identified a much simpler quantity — the plain fraction of a tumor's mass sitting in partially-covered ("boundary") cells, with no entropy weighting at all — as the practical explanation for the coarse-resolution difficulty gap. This test asks whether the fancier entropy measure adds anything beyond that simpler one.

- At D2, entropy and the plain boundary fraction are **almost the same thing** (rho=0.94, and the boundary fraction alone explains 80% of entropy's own variation). At this resolution, entropy is close to a relabeling of a quantity this project already had.
- At D4, the two are much more independent (rho=−0.19, boundary fraction explains only 21% of entropy). Here entropy genuinely carries information the simple boundary indicator does not.

This is a real, disclosed warning specifically for D2: the more elaborate entropy quantity is largely redundant with a much simpler one already available, at that resolution.

## 8. D4−D2 differential test

This is the strongest, cleanest result in the whole audit, and it is the one genuinely mechanistic (not just correlational) test the phase specifies. For each of the 227 measurable fragments, how much its ambiguity increases going from D2 to D4 (`ΔH_c = H_{c,D4} − H_{c,D2}`) was compared against how much its actual model error increases over the same change (`ΔE_c = E_{c,D4} − E_{c,D2}`).

**Result: rho = +0.78 (p ≈ 6×10⁻⁴⁸).** Fragments whose grid-ambiguity increases more sharply when the resolution coarsens are, very reliably, exactly the fragments whose model error also increases more sharply. This survives on its own as a genuinely strong, mechanistically meaningful finding — but see Section 12 for why it cannot rescue the overall verdict on its own.

## 9. 500-trial permutation safeguard (Test E)

Mandatory per this project's own standing practice, and applied here before trusting Section 6's positive, size-controlled finding: the model's actual error values were shuffled 500 times, always within matched size strata (so the shuffle can never accidentally recreate a size-driven correlation), and the same size-controlled correlation was recomputed each time.

**Result: the real correlation exceeds every single one of the 500 shuffled versions, at both resolutions** (empirical p < 0.002 at both D4 and D2 — conservatively reported as the tightest value measurable at 500 trials, since not one shuffle came close). This is a clean, decisive pass — the size-controlled entropy-error relationship found in Section 6 is real, not a mathematical artifact of the size-control procedure itself.

## 10. Outlier sensitivity

Removing the five most extreme entropy values before recomputing the D4 entropy-error correlation changes it from rho=−0.37 to rho=−0.39 (raw, unadjusted for size) — a negligible shift. The relationship is not being driven by a small number of extreme fragments.

## 11. Subject-clustered bootstrap CI

Because more than one tumor fragment can come from the same scan, and fragments from the same scan are not independent data points, this project's own standing bootstrap method (resampling whole scans, not individual fragments, 2000 times) was used to build a confidence interval around the size-controlled correlation specifically (not the raw, size-confounded one — an inconsistency that was caught and fixed during this analysis: an earlier draft of this bootstrap accidentally reported a confidence interval for the *raw* correlation, which has the opposite sign from the *size-controlled* one Section 6 actually reports, and would have been a misleading, self-contradictory result if published as-is).

- D4: mean +0.35, 95% interval [+0.18, +0.51] — entirely positive, excludes zero.
- D2: mean +0.44, 95% interval [+0.23, +0.61] — entirely positive, excludes zero.

## 12. Kill/GO decision

The phase's own pre-declared bar for treating this as a candidate for an actual new loss requires **all four** of: a real, size-independent signal; survival of the permutation safeguard; a genuine D4-more-ambiguous-than-D2 mechanism; and a genuine differential (ΔH-predicts-ΔE) relationship.

| Requirement | Result |
|---|---|
| Size-independent signal | **Pass** — positive, significant, bootstrap-confirmed at both resolutions |
| Permutation safeguard | **Pass** — decisive, real correlation exceeds all 500 shuffles |
| D4 > D2 mechanism | **Fail** — only holds for fragments over ~100 post-resize voxels; is 0% for the small-fragment population (20 voxels or fewer) this entire research line is about |
| Differential (ΔH→ΔE) prediction | **Pass** — strong, rho=0.78 |

Per the explicit standing rule for this phase — do not invent a positive interpretation if only one statistic passes, and by direct extension, do not treat a 3-out-of-4 pass as a pass when the failing piece is the one that defines *why the mechanism would matter for the population this project cares about* — **this is a KILL.** The entropy-as-hypothesized mechanism does not explain why coarse supervision would specifically help small lesions, because for small lesions specifically, the entropy quantity moves in the wrong direction by mathematical necessity, not by chance. The three passing tests are real, and are disclosed in full above rather than discarded, but they describe a relationship that is strongest and most self-consistent for **larger** fragments — not the small-lesion population this whole research arc exists to help.

## 13. What would constitute a valid training-time intervention

Not attempted or endorsed here, since the mechanism did not survive; noted only because a future phase, if one is undertaken, should not restart from scratch:

- Any future version of this idea would need to define "ambiguity" in a way that does **not** get driven toward zero when a small, fixed amount of tumor mass is diluted into a large coarse cell — for instance, a measure conditioned on the fragment already being detected/present, rather than one that conflates "this cell is ambiguous" with "this fragment is tiny relative to the grid."
- The differential (ΔH → ΔE) result in Section 8, despite not rescuing the overall verdict, is a genuinely strong, real, mechanistic finding on its own — it may be worth asking, separately, whether it survives using E36's own simpler boundary-fraction measure in place of entropy, since Section 7 already showed the two are close to interchangeable at D2.
- Any revised version would need its own fresh pass through Sections 4–11 in full — surviving three of four tests once is not a basis for assuming a revised formula would pass the fourth.

## 14. Limitations

- All measurements use one fixed, already-trained model (the version with both auxiliary heads active) at one single checkpoint (the end of its 30-epoch training run) — this was a deliberate choice, made explicitly so the D4-vs-D2 comparison is apples-to-apples within one model, but it means this audit does not speak to whether the same pattern holds earlier in training or in a model trained with only one auxiliary head active.
- The 227-fragment population this entire audit is built on already excludes every fragment that vanishes under the model's own mandatory image-shrinking step — a large majority (roughly 70%) of all 749 real tumor fragments in this dataset. Nothing in this report can speak to those fragments, because there is no model prediction to measure error against for something that was never given any surviving pixels to predict from.
- A genuine internal inconsistency (the bootstrap-CI sign mismatch described in Section 11) was caught and corrected during this analysis, not before it — a reminder that even a project with this many established safeguards can still introduce a new bug when reusing a proven method in a new context, and that every new statistic computed this way should be spot-checked against the other statistics already computed for internal consistency, not trusted purely because the underlying method has been validated before.
