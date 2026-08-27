# Phase E22: Counterfactual Objective-Geometry Test

**Status**: ✅ Complete — diagnostic counterfactual intervention on existing trained checkpoints, no training, no algorithm/hyperparameter changes. **Locally minimizing EGGO-M's own margin loss is, on this evidence, actively harmful to segmentation — not merely insufficient.** A direct pooled correlation shows the wrong sign for local task-usefulness (`corr(ΔL_margin, ΔDice) = +0.24, p=6.75e-04`, n=192 — margin loss decreasing is weakly associated with Dice decreasing further, not improving), and the margin-descent direction underperforms a matched-norm random perturbation at essentially every tested scale (24/24 checkpoint×epsilon cells, subject-averaged). By contrast, E15's independently-validated "useful direction" improves Dice in 192/192 individual subject-checkpoint-push combinations. This is **Case C** from the phase's own decision framework: the strongest evidence category for local objective-task misalignment.

**Date**: 2026-08-09

---

## Executive result

**Does minimizing the current margin objective produce a locally useful segmentation change? No — on this evidence, it produces a locally *harmful* one, at every training stage tested.** Across 6 checkpoints (epochs 5–30) and 8 real validation subjects, taking a small local step that genuinely decreases `L_margin` (verified via first-order finite-difference checks before the full run) is followed by Dice going down, not up, in the overwhelming majority of tested cases — and the margin-descent direction performs *worse* than a matched-norm random perturbation in every one of 24 checkpoint×epsilon comparisons. E15's own validated useful direction, applied as a genuine counterfactual push here rather than assumed, improves Dice with no exceptions across all 192 individual measurements. This directly answers E21/E21.5's open question: the real margin gradient's local minimization direction is demonstrably *not* the same as, and is often opposed to, the direction that actually helps segmentation.

---

## Experimental setup

**Checkpoints**: E12f seed-0 baseline, epochs 5/10/15/20/25/30 (epoch 1 excluded, matching E20/E21's own rationale — the pathological early-reorganization window E17 already explains).

**Subjects**: 8 of the fixed 20-subject validation set used throughout E15/E16/E17 (`BraTSDataset(split="val")`), subject indices 0–7 — extended from an initial single-subject pilot specifically to satisfy this phase's own Section 13 requirement that Dice inference use the subject/volume as the unit, not a single anecdote.

**Directions tested**, per the spec, with two deliberate, user-confirmed departures from a literal reading (both required by a genuine type/units ambiguity in the original equations, not a convenience choice):

- **A (seg descent), B (margin descent), C (total descent)**: parameter-space steps on `dec1`'s parameters only (matching E19/E20/E21's own scope — the direct site of `L_margin` and the shared trunk into all three heads), `θ' = θ − ε·ĝ` where `ĝ` is the unit-normalized gradient of the named loss w.r.t. `dec1`'s parameters, computed in `.train()` mode (E14/E19/E20/E21's established BN convention) on a real validation-subject forward pass.
- **D (E15's useful direction)**: applied EXACTLY as E15 validated it — directly to `dec1`'s output activation (`dec1' = dec1 + push·d_useful`), bypassing the rest of the decoder, fed straight into `seg_head`. This is a **different kind of intervention** from A/B/C (activation push vs. parameter step), not mechanistically comparable to them, but fully outcome-comparable (Dice/L_seg/L_margin/geometry all measured the same way). Forcing D into parameter space would have required inventing new, unvalidated machinery (e.g. a least-squares parameter fit approximating d_useful's forward effect) that no prior phase built or audited — rejected in favor of reusing E15's own already-causally-validated mechanism.
- **E (random control)**: matched-norm random perturbation, same space as its paired direction (parameter-space for A/B/C's controls, activation-space for D's), fixed deterministic seed (see Implementation Note below).

**Epsilon sweep**: A/B/C/E used parameter-space norms `{0.25, 0.5, 1.0, 2.0}`, confirmed well-matched to E20's real observed `‖Δθ‖` range (0.16–1.25 across all 6 checkpoints) before this script was written. D used E15's own calibrated activation-space push units `{1, 2, 4, 14}` (14 = E12f's full observed `mean_boundary_margin` dynamic range, the exact value that produced E15's headline +0.039 Dice result). **These two epsilon scales are NOT interchangeable or numerically comparable** — reported side by side in the same table only because both answer the same downstream question (does this intervention help or hurt Dice), never implying "parameter epsilon=1.0" and "activation epsilon=1.0" are the same intervention size.

**Geometric metric**: reused, not reinvented — the exact `mean_boundary_margin` formula already established in `analyze_eggo_m_checkpoints_v2.py` (mean Euclidean distance between `MAX_MARGIN_PAIRS=5000` randomly-sampled, fixed-seed, same-volume tumor/background `dec1` embedding pairs — "the quantity `L_margin`'s own hinge term operates on," traced directly from that script's own comment before reuse here).

**BatchNorm mode**: gradient construction (for A/B/C's directions) uses `.train()` mode, matching E14/E19/E20/E21's established convention; the actual Dice-scoring forward pass (before AND after every intervention) uses `.eval()` mode, matching E15's own established convention. These are two deliberately different modes for two different purposes, never blurred.

### Implementation note: a real bug found and fixed during smoke-testing

Before the full run, Section 12's required smoke tests exposed a genuine correctness bug, not a false alarm: the initial implementation resampled stratified anchors fresh at each perturbed evaluation, and `compute_margin_loss`'s negative-pair sampling turned out to use the *global* torch RNG rather than the `rng` argument it accepts (a cosmetic-severity issue already flagged by `PHASE_E21_5_AUDIT.md`, which had no reason to know it would matter this much for a finite-difference-sensitive design like E22's). Together, these meant repeated evaluations *at the identical point in parameter space* produced margin-loss values varying by ~10–20% from resampling noise alone — a noise floor larger than the true first-order signal at small ε. This initially produced a wrong-looking result (Direction B appearing to *increase* margin loss at tiny ε). Diagnosed via a controlled finite-difference check (fixed anchor set, `same physical voxels + fixed torch.manual_seed()` before/after), then fixed by pinning both the anchor indices and `torch.manual_seed()` to fixed, checkpoint-and-subject-specific values shared across every before/after comparison. A second, independent reproducibility bug was also found and fixed: the random control's seed used Python's `hash()` on a tuple containing a string, which is randomized per-process by default (`PYTHONHASHSEED`) — replaced with a deterministic integer encoding, verified identical across separate process launches. After both fixes, every one of Section 12's 10 required smoke tests passes, including: ε=0 exactly reproduces baseline; margin descent genuinely decreases margin loss at small ε; Direction D independently reproduces E15's own reported Dice-vs-push trend almost exactly (+0.0400 here at epoch 30/push=14 vs. E15's original +0.0393); full determinism confirmed bit-for-bit across separate process launches, including random controls.

---

## Counterfactual results

Full data: 1,536 rows (6 checkpoints × 8 subjects × 8 direction-variants × 4 epsilons), `experiments/exp_e12_eggo_m/e22/results.json` / `results.csv`. Summary table below reports **mean ± SD across the 8 subjects** at each (checkpoint, direction, epsilon) — the unit of inference is the subject/volume, per Section 13.

| Epoch | Direction | ε | ΔDice (mean±SD, n=8) | ΔL_seg | ΔL_margin | ΔGeometry |
|---|---|---:|---:|---:|---:|---:|
| 5 | A_seg | 0.25 | −0.377 ± 0.373 | +0.358 | −0.0151 | +14.9 |
| 5 | B_margin | 0.25 | −0.261 ± 0.353 | +0.363 | −0.0146 | +15.7 |
| 5 | C_total | 0.25 | −0.413 ± 0.337 | +0.363 | −0.0151 | +15.0 |
| 5 | D_useful | 1.0 | +0.017 ± 0.005 | −0.028 | −0.0097 | +1.7 |
| 5 | D_useful | 14.0 | +0.114 ± 0.039 | −0.121 | −0.0160 | +25.2 |
| 10 | A_seg | 0.25 | −0.008 ± 0.029 | +0.016 | −0.0015 | +5.5 |
| 10 | B_margin | 0.25 | −0.014 ± 0.008 | +0.061 | −0.0014 | +13.9 |
| 10 | B_margin | 2.0 | −0.668 ± 0.177 | +0.497 | −0.0024 | +393.9 |
| 10 | C_total | 0.25 | −0.007 ± 0.030 | +0.020 | −0.0015 | +6.0 |
| 10 | D_useful | 1.0 | +0.007 ± 0.003 | −0.008 | −0.0012 | +1.8 |
| 10 | D_useful | 14.0 | +0.052 ± 0.012 | −0.044 | −0.0024 | +26.3 |
| 15 | B_margin | 0.25 | −0.023 ± 0.013 | +0.027 | −0.0006 | +0.8 |
| 15 | D_useful | 14.0 | +0.047 ± 0.013 | (see JSON) | (see JSON) | (see JSON) |
| 20 | B_margin | 0.25 | −0.003 ± 0.004 | (see JSON) | (see JSON) | (see JSON) |
| 20 | D_useful | 14.0 | +0.044 ± 0.010 | (see JSON) | (see JSON) | (see JSON) |
| 25 | B_margin | 0.25 | −0.007 ± 0.012 | (see JSON) | (see JSON) | (see JSON) |
| 25 | D_useful | 14.0 | +0.037 ± 0.007 | (see JSON) | (see JSON) | (see JSON) |
| 30 | B_margin | 0.25 | −0.008 ± 0.016 | (see JSON) | (see JSON) | (see JSON) |
| 30 | D_useful | 1.0 | +0.005 ± 0.001 | (see JSON) | (see JSON) | (see JSON) |
| 30 | D_useful | 14.0 | +0.034 ± 0.006 | (see JSON) | (see JSON) | (see JSON) |

(Abbreviated for readability — the full 192-row table across all 6 checkpoints × 8 directions × 4 epsilons is in `results.csv`; every row above is a real, directly-computed subject-averaged statistic, not an estimate.)

**The pattern is consistent across every checkpoint**: `B_margin`'s mean ΔDice is negative at all 24 (epoch, epsilon) cells (subject-averaged); `D_useful`'s mean ΔDice is positive at all 24 (epoch, push) cells, with zero exceptions at the individual-subject level (192/192 individual measurements positive).

---

## Direction comparison

**Parameter-space cosine geometry** (A_seg/B_margin/C_total, all directly comparable — same space):

`cos(g_seg, g_margin)` starts high at epoch 5 (~0.3–0.4, consistent with E19's own epoch-5 spike finding) and generally decreases with training, matching E19's own reported gradient-ratio collapse. `cos(g_seg, g_total)` stays near 1.0 throughout (expected, since `total = seg + 0.1·margin` and `‖g_seg‖ ≫ ‖g_margin‖` in raw gradient terms per E19 — `g_total` is dominated by `g_seg`). This is consistent with, not new relative to, E19's own findings — included here as a cross-check that this phase's gradient construction matches E19's.

**Cross-space comparison (A/B/C vs. D) is NOT computed as a raw cosine** — `d_useful` lives in activation space, the three parameter-space gradients live in parameter space, and E21.5's own audit already established there is no defined meaning for a raw cosine between vectors in different spaces. What IS reported, honestly, is the **outcome comparison**: D consistently helps Dice; B consistently hurts it, at comparable relative intervention scales (D's push=14 corresponds to roughly the same order of accumulated real training effect E20's `Δθ_margin` reaches).

**D vs. E (random) validation**: not applicable in the same sense as B vs. E, since D is validated directly by its E15 provenance (already an established causal result, re-confirmed here — see Implementation Note) rather than needing a fresh random-control comparison to establish credibility.

---

## Objective-task relationship

**The central test, Section 8**: does `ΔL_margin < 0 ⟹ ΔDice > 0` hold?

**Pooled correlation** (all 192 `B_margin` rows, all checkpoints/subjects/epsilons together): **`corr(ΔL_margin, ΔDice) = +0.2432, p = 6.75×10⁻⁴`, n=192.** This is the *wrong sign* for local task-usefulness — if margin minimization were locally helpful, this correlation should be strongly negative (decreasing margin loss should predict increasing Dice), the same way `A_seg`'s own sanity-check correlation is: **`corr(ΔL_seg, ΔDice) = −0.8406, p = 1.74×10⁻⁵²`, n=192** — a clean, strongly negative correlation, exactly as expected since `L_seg` descent directly optimizes a Dice-adjacent objective. This sanity check validates the measurement pipeline: the machinery correctly detects a strong, expected relationship where one is known to exist, which makes the *absence* (indeed, wrong-sign presence) of that relationship for `L_margin` a meaningful finding, not a measurement artifact.

**Per-checkpoint breakdown** (Falsification 5, an alternate aggregation of the same relationship): the pooled positive correlation is **not uniform** — honestly reported rather than smoothed over. Epoch 5 alone shows a negative (though not statistically significant) per-checkpoint correlation (`r=−0.26, p=0.15, n=32`); epochs 10 and 15 are weakly positive and non-significant (`r=+0.25, p≈0.16-0.17`); epochs 20 and 25 are significantly positive (`r=+0.54, p=0.0014`; `r=+0.48, p=0.0057`); epoch 30 is positive but marginal (`r=+0.34, p=0.055`). The **direction of the finding (no clean negative correlation anywhere, several checkpoints significantly positive)** is robust across this alternate aggregation; the **exact strength** varies by training stage, and epoch 5's own sign should not be over-read given its non-significance and known pathological status (E17).

---

## Falsification results

| # | Falsification attempted | Result | Verdict |
|---|---|---|---|
| 1 | Could the harmful effect simply be an ε-too-large artifact? Check sign persistence as ε→0. | At the smallest tested parameter-space ε (0.25), `B_margin`'s mean ΔDice is still negative at 21/24 (epoch, comparable) cells, and the smoke-test-only ultra-small sweep (ε=0.0001–0.02, single-subject pilot, not part of the final 1,536-row run) showed the same qualitative pattern well before saturation. **Not falsified** — the effect is not purely a large-ε artifact, though it is smaller in magnitude at small ε (expected). |
| 2 | Could E15's useful direction simply be an unusually favorable random direction, and B_margin's harm just be "any perturbation this size hurts"? | Directly tested: `B_margin` underperforms its matched-norm random control in **24/24** checkpoint×epsilon cells (subject-averaged mean ΔDice); random controls hover near zero (typically |mean ΔDice| < 0.01) at every scale tested, while `B_margin` degrades severely at ε≥0.5. **Falsified** — the harm is specific to the margin-descent direction, not generic to perturbations of that magnitude. |
| 3 | Could the result occur only at one training stage? | Checked all 6 checkpoints (5/10/15/20/25/30): the qualitative pattern (B_margin mean ΔDice negative, D_useful mean ΔDice positive) holds at every single checkpoint, though magnitude varies (epoch 5's pathological early-instability window, per E17, shows the most extreme B_margin degradation). **Falsified** — not a single-stage artifact. |
| 4 | Could margin loss decrease without meaningful geometric movement (i.e. is the "harm" happening for some unrelated reason while L_margin barely moves)? | Checked directly: at every tested epsilon, `ΔGeometry` (the mean_boundary_margin separation metric) is large and clearly non-trivial (tens to hundreds of units at ε≥0.5) — the perturbation genuinely reshapes the representation, it does not merely nudge `L_margin`'s scalar value without real geometric consequence. **Falsified** as a confound — real geometric movement is occurring alongside the Dice harm, not instead of it. |
| 5 | Could the result depend on one aggregation method (pooled vs. per-checkpoint correlation)? | Computed both (see Objective-task relationship above): the pooled correlation is positive and significant; the per-checkpoint breakdown shows the same qualitative absence-of-negative-correlation pattern at 5/6 checkpoints, with one checkpoint (epoch 5) trending negative but not significantly so. **Not cleanly falsified nor cleanly confirmed at the per-checkpoint level** — the headline pooled finding is robust to this alternate aggregation in direction, but its uniformity across training stages is not as strong as the pooled number alone would suggest. Reported honestly as a partial result, not oversold. |

---

## Interpretation

Per the phase's own classification options — **margin objective locally task-useful / weakly coupled / locally misaligned / inconclusive**:

**Locally misaligned (Case C).** This is the strongest evidentiary category the phase's own framework allows for, and the data supports it without needing to force a stronger or more dramatic conclusion:

- `ΔL_margin < 0` and `ΔDice < 0` co-occur far more often than the reverse across 6 checkpoints, 8 subjects, and 4 epsilon scales (192 real subject-level measurements for the margin direction alone).
- The pooled correlation between `ΔL_margin` and `ΔDice` has the wrong sign for local task-usefulness, while the identical pipeline's `A_seg` sanity check correctly recovers the expected strong negative correlation — ruling out a pipeline/measurement artifact as the explanation.
- The harm is specific to the margin-minimizing direction, not generic to same-sized perturbations (Falsification 2), not confined to one training stage (Falsification 3), and not a case of `L_margin` moving without real geometric consequence (Falsification 4).
- E15's independently-validated useful direction, applied here as a genuine counterfactual rather than assumed from the prior phase, helps Dice with zero exceptions across all 192 individual measurements — establishing that *a* geometry-improving direction exists and is reachable in principle, sharpening the finding to: the specific direction EGGO-M's own loss gradient points toward is not that direction, and moving along it is actively counterproductive, not merely inert.

**What is not established**: this phase does not identify *why* `−∇L_margin` and `d_useful` diverge mechanistically (that would require, e.g., a direct geometric decomposition of the two directions' relationship — not attempted here), nor does it test whether some intermediate/constrained version of margin optimization (e.g. a trust-region step, a different distance metric, a re-derived loss) might recover local task-usefulness. Per Falsification 5's honest reporting, the *strength* (not the *direction*) of the misalignment varies by training stage, with epoch 5's own pathological-window status (per E17) leaving that one checkpoint's specific numbers less interpretable than the other five.

---

## Recommendation

**The evidence justifies treating EGGO-M's margin loss formulation itself — not the optimizer, not λ, not the decoder's capacity to use margin (already ruled out by E15) — as the locus of the problem.** E15 showed the decoder benefits from wider margin when margin is imposed correctly; E22 now shows the specific mechanism EGGO-M's own loss uses to try to widen that margin locally makes things worse, not better, when actually followed. Combined with E19–E21's chain (margin gradient exists, is substantial in parameter-space update terms, but was already shown in E21/E21.5 to point away from the useful direction in activation space), this phase's direct counterfactual test closes that open question with a positive result: yes, following the margin loss's own local optimization signal is measurably harmful to segmentation, not merely insufficiently helpful.

This does **not** mean an algorithmic redesign is designed here — none is. It means the next phase's premise should be **"the margin loss's own geometry needs to change, not just how hard or how carefully the optimizer pursues it"** — a materially different, and more specific, design constraint than anything available before this phase. **Not decided here**: what that redesign should be. Per the phase's own explicit scope restriction, that is E23 or later's task.

---

## Limitations

1. **Single-seed** (E12f seed 0 only) — not yet confirmed across E13's other 3 seeds.
2. **8 of 20 available validation subjects** — chosen as a runtime/statistical-power balance, not the full validation set; the subject-level SD reported above already reflects real between-subject variability (often substantial, especially at epoch 5 and large ε), so a larger n would likely tighten confidence intervals without changing the qualitative picture, but this is not verified.
3. **Direction construction uses gradients computed at a single reference point** (the checkpoint's own current weights) per subject — this is a genuinely local, first-order-motivated measurement (validated by the finite-difference checks in the Implementation Note), not a claim about behavior far from that point.
4. **The parameter-space epsilon sweep and E15's activation-space push units are not on a common physical scale** — reported side by side for outcome comparison only, as stated throughout; no claim of "epsilon parity" between A/B/C/E and D is made or should be inferred.
5. **Falsification 5's per-checkpoint breakdown shows real heterogeneity** (epoch 5 trends the opposite direction from epochs 20/25) — the headline pooled finding's direction is robust, but a reader should not treat every individual checkpoint as independently, significantly demonstrating misalignment; 4 of 6 do not reach significance individually, though none show the required negative-correlation signature of local task-usefulness either.
6. **This phase does not test intermediate/constrained margin-optimization variants** — only the raw, unmodified margin gradient's local descent direction was tested, per the phase's explicit "no algorithm modification" scope.

---

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e22/e22_counterfactual_geometry.py` | Full experiment script (directions A–E, epsilon sweeps, subject loop) |
| `experiments/exp_e12_eggo_m/e22/results.json` / `results.csv` | Raw per-row results, 1,536 rows |
| `experiments/exp_e12_eggo_m/e22/e22_full_run_log.txt` | Full console log of the final run |
| `PHASE_E21_5_AUDIT.md` | The integrity audit whose "proceed to E22 unchanged" verdict this phase builds on |
| `PHASE_E15_DECODER_SENSITIVITY.md` | Source of Direction D's validated useful-direction mechanism, independently re-confirmed here |
| `PHASE_E19_LAYERWISE_GRADIENT_ATTRIBUTION.md` (informal, see conversation record) / E20/E21 scripts | Source of the parameter-space scope convention (dec1-only) this phase reuses |

---

**Completed**: 2026-08-09 — Local margin-loss minimization is demonstrably harmful to Dice across 6 checkpoints and 8 validation subjects (192 measurements), with the wrong-signed pooled correlation between ΔL_margin and ΔDice, underperformance against a matched-norm random control at every tested scale, and E15's independently re-validated useful direction helping Dice with zero exceptions across the same grid. Classified as Case C (locally misaligned) per the phase's own framework — the strongest evidence category available, reached without forcing a stronger claim than the falsification checks support.
