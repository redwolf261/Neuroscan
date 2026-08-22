# E34 — Scale-Conditioned Context Supervision: Feasibility Audit

**Verdict: KILL**

No model was trained in this phase. This is a read-only feasibility audit of a proposed new loss-supervision mechanism for a 3D U-Net brain-tumor segmentation model (BraTS 2023 GLI, FLAIR-only, whole-tumor binary segmentation). It was stopped after Section C because that section's result fails the first pre-registered "Go" condition decisively, and the project's own standing rule is not to keep tuning or extending a result once a pre-registered gate fails cleanly.

---

## 1. Hypothesis

The model has three prediction resolutions inside its decoder: a coarse one (16³ voxels, called "D4"), a medium one (32³, "D2"), and the full one (64³, "D1"). Extra training supervision is already applied at D4 and D2 in a prior, separately-validated experiment ("Deep Supervision"), which gave the project's only confirmed positive result so far (a small, real improvement over the baseline).

This audit asked a follow-up question about *what* that D4/D2 supervision should look like. Instead of just supervising the model with a shrunk-down, exact copy of the tumor outline at each resolution (the obvious choice, and what Deep Supervision already does), the idea was to supervise it with a *softer, spatially spread-out* version of the tumor outline — a "contextual field" built by blurring the true outline with a Gaussian kernel — on the theory that a small tumor's *exact* outline can become vanishingly small or vanish entirely once compressed down to 16³ or 32³, while a *blurred* version of that same tumor might still leave a meaningful trace at those resolutions, giving the model something to learn from where the exact-outline signal has nothing left to offer.

Concretely, the proposal was a blend: `T_s = (1 - η_s)·Y_s + η_s·C_s`, where `Y_s` is the exact downsampled outline at decoder scale `s`, and `C_s` is the Gaussian-blurred contextual field at that same scale, with the blend weight `η` set higher at the coarsest scale (D4) and zero at full resolution (D1) — matching the architectural motivation that only the coarse scales have an impoverishment problem.

## 2. Mathematical definition

`C_σ(x) = (K_σ * Y)(x)`, a normalized 3D Gaussian kernel of width `σ` convolved with the binary ground-truth tumor mask `Y`. A second candidate, simple morphological dilation of the mask by a fixed radius, was constructed alongside it as a control, to check whether the Gaussian's specific shape matters or whether any kind of "make the region bigger" transformation would do just as well.

Sigma values (0.5, 1.0, 1.5, 2.0 voxels, measured at each decoder's own resolution) and matching dilation radii (1, 2, 3, 4 voxels) were fixed before any data was generated or any result was seen, as required.

## 3. Data construction

All 125 scans held out for evaluation in this project (never used for any training decision) were used, no smaller sample. For every real tumor fragment ("connected component") in every scan — 749 in total — the following was recorded: its true size (in voxels, at native scan resolution, before any resizing), its size after the standard 64³ resize the model always applies, whether the current baseline model and two Deep-Supervision variants successfully detect it, and (where previously measured) a separate "critical resolution" statistic from an earlier phase of this project.

Two consistency problems with prior data from earlier phases of this project were found and resolved before any new work relied on them: an older component table used a different, incompatible way of identifying tumor fragments (labeling them after resizing rather than before, which merges or erases fragments that a before-resizing labeling keeps distinct — the counts differed by more than 2×, 363 vs. 749, for the same 125 scans), and a separate old table's "baseline model" numbers turned out to come from a different, non-representative training run of the baseline than the one used everywhere else in this project's records. Both were corrected: a fresh, verified table was built using the correct component-identification method and the correct, canonical baseline checkpoint, and every number was independently cross-checked before being trusted.

The three supervision targets (exact, dilated, Gaussian-blurred) were then built for all 749 fragments at all three decoder resolutions. A first version of this construction contained a fundamental design mistake, caught before it could contaminate any downstream result: it blurred the *already-shrunk-down* tumor outline rather than blurring the outline *first* and *then* shrinking it down. Blurring something that has already been erased by resizing cannot produce anything except more erasure — so this version could never have shown the "the blurred version survives when the exact version doesn't" effect the whole hypothesis depends on, by construction, regardless of what the real data looked like. This was fixed by blurring at the scan's original, full resolution first and only then shrinking the result down to each decoder's resolution, which is also what the proposed mathematical formula actually specifies.

A second, purely computational problem was also found and fixed: an early size limit placed on how much surrounding context each blur calculation was allowed to look at (added purely to keep the computation fast) was, on inspection, cutting the blur off well before its natural extent for three-quarters of the pre-declared blur widths at the coarsest resolution — silently and uniformly across every single tumor fragment, not just unusually large ones as first assumed. This was corrected by enlarging that limit to the true requirement and re-running the full computation; the corrected run confirmed the limit no longer cuts off any blur calculation.

## 4. Small-lesion analysis (Section C — where this audit stopped)

Tumor fragments were grouped by their true, native size into five pre-declared bins: S1 (1–5 voxels), S2 (6–20), S3 (21–50), S4 (51–150), S5 (over 150). For reference, S1 makes up 65% of every fragment in the dataset (489 of 749) — the population is heavily weighted toward extremely small fragments, many of which are plausibly closer to annotation noise than clinically real tumor pieces, a caveat noted in this project's other work as well.

**Finding 1 — the exact-outline collapse is confirmed and severe.** At the coarsest decoder resolution (D4), 100% of S1–S4 fragments (every fragment 150 voxels or smaller at native resolution) have zero surviving voxels in their exact-outline target. This matches and reconfirms what an earlier, independent phase of this project already found using a different method.

**Finding 2 — the contextual field does NOT meaningfully survive there either, once measured honestly.** This is the audit's core result, and it required a mid-analysis correction of its own. The first pass measured "did the contextual field survive?" using the loosest possible test — is there any voxel anywhere with a value greater than exactly zero after blurring — and got a suspiciously perfect answer: 100% of fragments in every size bin, at every blur width, "survived." Because a result that clean is itself a signal to check for a measurement mistake before trusting it (per this project's own established practice), it was checked directly, and it was indeed an artifact: a Gaussian blur mathematically never produces an exact zero except where deliberately cut off, so "any nonzero value" is satisfied almost automatically and says nothing about whether the surviving signal is big enough to matter. Re-measured against a real, pre-declared threshold (already computed as one of the standard summary statistics from Section B, not selected after seeing this problem) — a context value has to exceed 0.1 somewhere to count as meaningful — the true picture reverses almost completely:

| Native-size bin | Fragments | % with zero exact target (D4) | % of those "rescued" by meaningful (>0.1) context, any blur width |
|---|---:|---:|---:|
| S1 (1–5 voxels) | 489 | 100% | 0% |
| S2 (6–20) | 61 | 100% | 0% |
| S3 (21–50) | 22 | 100% | 0% |
| S4 (51–150) | 15 | 100% | 0% (except a 40% rate on the loosest fallback threshold, at the narrowest blur width only) |
| S5 (over 150) | 162 | 12% (19 fragments) | up to 26%, only at the narrowest blur width, dropping to 0% at wider blur widths |

The same pattern holds, only slightly less severely, at the medium decoder resolution (D2): meaningful rescue is essentially zero for S1–S3, and only appears at all for S4/S5 with the narrowest blur width.

In plain terms: the exact tumor outline reliably disappears for small tumors at the coarse decoder resolutions, exactly as the hypothesis assumed — but the proposed fix (a blurred, spread-out version of the same outline) does not leave behind anything strong enough to plausibly teach the model something useful, for precisely the population of small tumors this mechanism was designed to help. It is real (nonzero) in a purely mathematical sense, but too faint, by the audit's own pre-declared "meaningful" threshold, to be worth the added complexity of a new loss term — the same failure shape as this project's own margin-loss line of research (mathematically real but too weak to move the needle).

## 5. Nonlinear size controls

Not reached — the pre-registered protocol calls for stopping once a Go criterion fails decisively, rather than continuing to spend analysis effort validating a mechanism whose central claim (contextual support survives where exact support vanishes) already failed its own most direct test.

## 6. Permutation safeguard

Not reached, for the same reason. Given Section 4's clean numeric answer, the more elaborate randomize-and-retest safeguard this project uses for less clear-cut positive results is not needed here — there is no positive result to falsify.

## 7. Gaussian-vs-dilation comparison

Not reached. The dilation-family targets were successfully constructed (Section 3) but never analyzed, since the more fundamental question — does *either* soft-target family survive meaningfully at coarse scale — was already answered no for the Gaussian version.

## 8. Spatial information test

Not reached, for the same reason.

## 9. Literature audit

Not performed. This project's own policy is to run a literature-novelty check before committing to a training run, not before an early feasibility check — and no training run is now warranted, so this step is moot.

## 9b. Post-hoc code audit (performed after the verdict, to check for coding errors)

A full re-check of the pipeline was performed after this report's first draft, specifically hunting for coding errors that could have produced the KILL verdict artificially. One genuine, previously-undetected issue was found and is disclosed here for completeness, along with why it does not change the verdict.

`scipy`'s resize function (`zoom`, used to shrink the blurred/dilated fields down to each decoder resolution) works by sampling at a sparse grid of points, not by averaging over areas. For a very large shrink factor (roughly 10x or more per axis, which happens at the coarsest decoder resolution), this sampling can step entirely over an isolated small object between two sample points and report it as completely absent, even though the object is genuinely present in the data. This was confirmed directly: a single isolated voxel, resized by this method with no blur applied, disappeared completely.

This was checked against every one of the saved measurements. It affected only the **dilation** family of targets (not the Gaussian family the verdict is based on), and only at the narrowest, smallest setting in the pre-declared grid (the smallest blur/dilation width, at the coarsest and full-resolution scales) &mdash; 33 out of 8,988 individual measurements, none of them at the resolution or setting most relevant to the small-lesion question. The Gaussian-context family, which Section 4's finding is entirely based on, had zero such cases anywhere in the dataset, because a Gaussian blur spreads its mass too widely for every sample point to miss it at once.

As a final, independent check, one specific single-voxel tumor fragment's Gaussian-context value was recomputed completely from scratch &mdash; on the full original scan volume, with no cropping or any of the performance optimizations used elsewhere in this audit &mdash; using the widest, most favorable blur width tested. Its highest value anywhere in the coarse 16&sup3; grid was about 0.0000035, several orders of magnitude below the 0.1 threshold used to call something "meaningfully present." This matches the pipeline's own saved result for that fragment exactly, confirming Section 4's central finding independently of any of this audit's own code.

## 10. Limitations

- This audit only reached Section C of a nine-part pre-registered protocol; Sections D through I (nonlinear-size controls, the permutation safeguard, the Gaussian-vs-dilation comparison, availability-at-coarse-scale, the spatial-concentration test, and the literature search) were not executed, on the explicit reasoning that a decisive early-gate failure doesn't need the later, more expensive gates to also be checked before stopping.
- The "meaningful" rescue threshold (context value > 0.1) was one of three pre-declared thresholds computed automatically in Section B (alongside > 0.01 and > 0.5); it was not chosen after seeing this section's numbers, but a different, defensible threshold choice could in principle shift the exact percentages reported here, though not the qualitative conclusion (the >0.01 threshold, one notch looser, still shows near-total collapse for S1–S3 once the blur width grows past the very narrowest setting).
- The S5 (large-fragment) population's crop-based construction used a size cap to keep computation time tractable, which very occasionally trades a small amount of blur accuracy for large fragments; this does not affect the S1–S4 conclusion, which is the one the verdict rests on.
- Two real construction bugs (described in Section 3) were caught and fixed before they could distort this result, but their discovery is itself worth flagging: this general class of mechanism (spread the ground truth out spatially, hoping it survives resizing) is easy to accidentally implement in a way that's mathematically guaranteed to fail regardless of whether the real underlying idea has merit, and any future revisit of this class of idea should re-verify the blur-before-resize ordering from scratch rather than reuse this codebase's assumptions uncritically.

## 11. Exact next experiment if GO

Not applicable — the verdict is KILL.

## 12. What, if anything, is worth carrying forward

The underlying diagnosis this idea was built on — small tumors lose their exact-outline signal at coarse decoder resolutions — is confirmed again here, consistent with multiple earlier, independent measurements in this project. What this audit adds is a specific, now-closed answer to the natural next question ("can a Gaussian-blurred version of the outline recover a useful training signal there?"): no, not at a threshold anyone would call meaningful, for the small-lesion population this whole line of reasoning was meant to help. This narrows, rather than expands, the space of remaining ideas worth trying for that underlying problem.
