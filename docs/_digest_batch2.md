# Digest Batch 2: E25 SC-TAM Deep-Dive Sub-Arc + E26-E43

Compact digest entries for each source file, covering claims, self-contradictions/hedges, caught bugs, and cross-references to other phase docs.

---

## PHASE_E25_CANDIDATE6_SC_TAM_DESIGN.md

**Summary**: Locks the design of "Candidate 6" (SC-TAM — Signed Class-Conditional Task-Aligned Margin), derived from E24's Q3 finding that condition B's background voxels moved incorrectly toward the tumor side (39.1% correct, p=0.001, significantly below chance) while tumor voxels moved correctly (64.5%, p=0.007). Defines the signed loss `d^SC_ij = (z_i-z_j)^T w_hat` for FB pairs only, squared hinge matching existing `L_margin`, and a staged plan C6-0 through C6-4. No code written, no training run — pure specification.

**Self-contradiction/hedging**: None — this is explicitly a pre-implementation design doc and repeatedly states it makes no result claims. Notably transparent about three corrections made to an *earlier draft* of this same document (same-class-pair scoping, `p_i` availability, dropped "ante-hoc interpretability" framing) before finalizing.

**Bug found/fixed**: No runtime bug (nothing executed yet), but documents correcting a documentation/design error from an earlier draft of itself (incorrectly believed the pair-sampling loop needed new code to exclude same-class pairs; verified it already only formed cross-class pairs).

**Cross-references**: Builds directly on `PHASE_E24_GATE6_EXPERIMENTAL_SPECIFICATION.md` (reuses locked A/B/E protocol) and follows the novelty-audit discipline set by `PHASE_E23_ALGORITHMIC_REDESIGN.md`.

---

## PHASE_E25_C62_RESULTS.md

**Summary**: Reports full H1→H4 pipeline for C6-2 (SC-TAM core, unweighted). H1: rho=-0.4924 (correct sign, p=4.06e-13) — first EGGO-M candidate with correctly-signed loss-to-Dice correlation. H2: reports "24/24 checkpoint-batches show correct mean sign" for class-conditional realized displacement. H3/Q5: 9.4x concentration of movement on misclassified voxels (p=7.1e-15). H4: best Dice 0.9030, **below** both baseline A (0.9063) and random-control E (0.9058), missing the 0.9183 bar by 1.53pp.

**Self-contradiction/hedging**: The document itself flags an internal tension it does not resolve: H2 (realized 15-step AdamW update) shows favorable signed alignment, but Q3/Q4 (immediate single-step gradient) shows near/below-chance sign correctness and a net-unfavorable FP/FN transition count (14,443 favorable vs 38,283 unfavorable). Explicitly says "this report does not adjudicate" — flagged as its own open question, not smoothed over. **This entire H2/H3 sign-correctness narrative is later found to rest on a backwards convention** (see `PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md`) — meaning the "24/24 correct sign" headline is eventually shown to be scored under the wrong convention.

**Bug found/fixed**: Yes — a test-construction flaw in `test_orientation_invariance` (comparing reordered voxels under identical seed, not realizing index-based negative sampling meant "same seed" ≠ "same pairs"). Fixed by rewriting to a deterministic 2-voxel case. Also extended `run_counterfactual.py` with a new `margin_mode` property; re-verified as a true no-op via regression gate (0.00e+00 max diff).

**Cross-references**: Evaluates against decision rule in `PHASE_E25_CANDIDATE6_SC_TAM_DESIGN.md` Section 11. Superseded/corrected later by `PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md` and `PHASE_E25_C62_CORRECTED_REANALYSIS.md` (H1/H4 sections confirmed unaffected and stand).

---

## PHASE_E25_C62_MECHANISM_AUDIT.md

**Summary**: Four-level causal decomposition of C6-2's already-trained checkpoints. Headline: splitting H2's tumor/background pooled result by the model's own current prediction (TP/FP/FN/TN) reveals FN moves with the WRONG sign 98.5% of the time (with larger magnitude than TP's correct movement), while TP/TN move correctly — concluding SC-TAM "reinforces an already-correct boundary" rather than fixing errors. An addendum diagnostic (anchor-composition) rules out sampling bias as an explanation (FN/FP are heavily oversampled and dominate loss share, yet still net wrong-signed).

**Self-contradiction/hedging**: None internally, but this document's entire central finding (the 98.5%-wrong-sign FN claim) is **later completely reversed** by `PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md` and `PHASE_E25_C62_CORRECTED_REANALYSIS.md`, which show the opposite (FN moves correctly 98.0% of the time) once the sign-scoring bug is fixed. This document itself has no self-awareness of the bug — it is a downstream casualty, not a self-correction.

**Bug found/fixed**: No bug caught in *this* document (the sign-convention bug it unknowingly relies on isn't discovered until two documents later), though it does note Level 1 is "not a new finding — a mathematical identity."

**Cross-references**: Follows up on `PHASE_E25_C62_RESULTS.md`. Its Level 4 table is directly superseded by `PHASE_E25_C62_CORRECTED_REANALYSIS.md`.

---

## PHASE_E25_C62_ISOLATION_CHECKS.md

**Summary**: Three checks to isolate why FN's realized movement is wrong-signed (per the mechanism audit's then-current, later-overturned framing). Check 3: real evidential/boundary weighting reduces FN's loss share from 43.0%→27.8% but FN+FP still dominate (~63%). Check 1: individual FN/FP anchor gradients agree 100% with the batch-aggregate direction (rules out dilution/opposition). Check 2 (decisive): isolated single-anchor FN probes (no AdamW, no aggregation) are wrong-signed 36/36 — same failure at the smallest possible scale, ruling out AdamW and cross-anchor aggregation as causes.

**Self-contradiction/hedging**: None in isolation, but — like the mechanism audit before it — its "36/36 wrong-signed" finding is later shown to be an artifact of the same sign-convention bug; the corrected reanalysis shows the identical measurement is 36/36 **correctly** signed once fixed.

**Bug found/fixed**: No bug caught here (again a downstream casualty of the not-yet-discovered sign bug).

**Cross-references**: Follows up on `PHASE_E25_C62_MECHANISM_AUDIT.md`. Superseded by `PHASE_E25_C62_CORRECTED_REANALYSIS.md`.

---

## PHASE_E25_C62_JACOBIAN_LOCALIZATION.md

**Summary**: Tests whether the (at-the-time-believed-real) sign inversion is localized to a specific parameter subset (dec1/decoder/encoder/combined). Finding: the inversion is NOT localized — it's present with near-identical magnitude/sign (r=0.98 cross-pathway) everywhere, including in ALL FOUR confusion categories (TP/TN/FP/FN), not just FN/FP — 0% correct sign across every pathway × category (72/72 wrong).

**Self-contradiction/hedging**: The document is honest that this "changes what the isolation checks' earlier TP/TN result meant" — since prior phases never measured TP/TN's isolated single-anchor probe, this is the first time it's shown wrong-signed too. Explicitly flags an open question it does NOT resolve: whether the "network's Jacobian genuinely inverts a correctly-targeted direction" vs. "the assumed correct direction was never right in the first place" — this hedge turns out to be exactly correct (it's the latter). This entire document's 0%-correct headline is **fully and completely retracted** by the next two documents in the chain.

**Bug found/fixed**: No bug caught here — but this document's own uniform, suspiciously clean 0% result (rather than being accepted) triggers the very next audit, which finds the actual root cause (a sign-convention scoring bug).

**Cross-references**: Follows up on `PHASE_E25_C62_ISOLATION_CHECKS.md`. Directly and fully superseded by `PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md` and `PHASE_E25_C62_GRADIENT_JACOBIAN_CONSISTENCY.md`, which show the "pathway-invariant inversion" was 100% a scoring-convention bug, not a real network phenomenon.

---

## PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md — KEY CORRECTION DOCUMENT

**Summary**: Root-causes the entire preceding chain's anomalies. Every diagnostic script from H2 onward (`run_h2_c62.py`, `run_h3_c62.py`, mechanism audit, isolation checks, jacobian localization) scored directional correctness against `tumor→−ŵ, background→+ŵ`, which is the OPPOSITE of what SC-TAM's own loss formula produces under gradient descent (correct: `tumor→+ŵ, background→−ŵ`). Found via the theoretical sanity check `cos(Δz_GD, ∇_zL) ≤ 0` (must hold for any genuine descent step regardless of SC-TAM's directional claims). Confirms this matches the project's own already-passing unit test `test_sc_tam_sign_correctness`, which had the correct convention all along.

**Self-contradiction/hedging**: This document IS the self-correction — it explicitly states this is "a scoring bug in the diagnostic scripts, NOT in compute_margin_loss itself, not in C6-2's training run" and explicitly does NOT affect H1 or H4. It deliberately stops short of recomputing downstream conclusions ("real, substantial follow-up work... not rushed inline here").

**Bug found/fixed**: THE central bug of this whole sub-arc — a backwards `expected_sign()` convention baked into 5 separate diagnostic scripts (H2, H3, mechanism audit, isolation checks, jacobian localization), while the actual training code (`compute_margin_loss`) and its unit test were correct the entire time.

**Cross-references**: Explicitly identifies `PHASE_E25_C62_JACOBIAN_LOCALIZATION.md`, `PHASE_E25_C62_MECHANISM_AUDIT.md`, `PHASE_E25_C62_ISOLATION_CHECKS.md`, and the H2/H3 sections of `PHASE_E25_C62_RESULTS.md` as all needing reinterpretation (H1/H4 sections of the latter unaffected).

---

## PHASE_E25_C62_GRADIENT_JACOBIAN_CONSISTENCY.md

**Summary**: Independent, from-scratch validation of the sign-convention fix. Confirms gradient descent's `Δz·∇_zL ≤ 0` invariant holds 95.8-100% across an epsilon sweep; validates the finite-difference measurement mechanism against a first-order JVP prediction (5/6 checkpoints converge cleanly; epoch 25 is a flagged outlier that "does not change the overall conclusion"). Decisive: under the corrected sign convention, ALL 48/48 probed voxels across all 4 categories (TP/TN/FP/FN) are correctly signed at 100% — confirms "Outcome A": no genuine network/Jacobian inversion exists; the entire prior "pathway-invariant inversion" finding was purely the sign bug.

**Self-contradiction/hedging**: Honestly reports that the `ΔL` sign check (a secondary measure) degrades toward chance as epsilon shrinks (91.7%→50.0%), attributed to floating-point noise in the squared-hinge loss recomputation — explicitly flagged as a limitation of that specific measurement, not evidence against the main finding.

**Bug found/fixed**: A real implementation obstacle (not a logic bug): `torch.func.jvp` incompatible with BatchNorm3d's in-place `num_batches_tracked` mutation in train mode; fixed by disabling `track_running_stats`, verified via forward-pass `allclose` to not change model output. Also found/fixed a float32 vs float64 precision issue in the JVP comparison.

**Cross-references**: Confirms and fully explains `PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md`'s hypothesis; declares `PHASE_E25_C62_JACOBIAN_LOCALIZATION.md`'s 0%-correct finding "fully explained, not a separate mechanism."

---

## PHASE_E25_C62_CORRECTED_REANALYSIS.md

**Summary**: Recomputes every affected prior result (H2, mechanism-audit Level 4, isolation-check Check 2) via exact complement (`new = 1 - old`), no new runs. The corrected picture is NOT a simple mirror-flip: FN now moves correctly-signed 98.0% of the time and FP 82.3% (the actual errors are moving in the intended direction), while TP (6.1%) and TN (21.1%) — the already-correct voxels — move WRONG-signed. This is "close to the literal opposite" of the mechanism audit's original narrative. Declares verdict "Outcome 2": the mechanism is correctly-directed on errors, yet C6-2's Dice (0.9030) still misses the bar — reframing the open question from "why is the mechanism inverted" to "why doesn't a correctly-functioning mechanism yield enough Dice."

**Self-contradiction/hedging**: Explicit and central to the document's purpose — it states plainly that "the old report's headline... is now shown to be backwards in exactly the way that matters most." Surfaces a genuinely new open question (invisible under the old convention): why do TP/TN move wrong-signed at all, and whether that's harmless or a real contributor to the Dice shortfall — left unresolved.

**Bug found/fixed**: No new bug — this is the recomputation following the bug fix in `PHASE_E25_C62_SIGN_CONVENTION_CORRECTION.md`.

**Cross-references**: Recomputes `PHASE_E25_C62_RESULTS.md` (H2), `PHASE_E25_C62_MECHANISM_AUDIT.md` (Level 4), `PHASE_E25_C62_ISOLATION_CHECKS.md` (Check 2); cross-verifies against `PHASE_E25_C62_GRADIENT_JACOBIAN_CONSISTENCY.md`'s independent sweep (exact agreement, 100% at every category).

---

## PHASE_E25_C62_REPRESENTATION_TO_LOGIT.md

**Summary**: Traces the corrected-convention finding downstream: does correctly-signed representation movement (FN/FP) actually flip predictions? Tests four hypotheses (A: insufficient movement, B: representation-logit mismatch, C: boundary distance, D: collateral damage). FN transitions at 64-84% depending on confidence depth; FP shows a puzzling FLAT transition rate (~22-26%) despite growing corrective Δlogit with depth — an unresolved, FP-specific anomaly. Decisive finding: TP shows damaging-direction logit shift in 96.0% of voxels (magnitude 3.73, exceeding FN's own aggregate correction magnitude of ~3.48) and TN in 79.5% — strong support for Hypothesis D (collateral damage to the numerically enormous already-correct majority).

**Self-contradiction/hedging**: Explicitly states its own per-category-balanced subsampling means "raw counts here cannot be used to compute a true population-weighted net Dice effect directly" — flagged honestly as a scope limitation, not resolved in this document (deferred to the next one).

**Bug found/fixed**: None new — builds on the already-corrected convention.

**Cross-references**: Builds on `PHASE_E25_C62_CORRECTED_REANALYSIS.md`'s Level 4 table and population counts; its own damage estimate is later population-weighted in `PHASE_E25_C62_POPULATION_WEIGHTED_ACCOUNTING.md`.

---

## PHASE_E25_C62_POPULATION_WEIGHTED_ACCOUNTING.md

**Summary**: Applies real transition-rate measurements from C6-2.6 (representation-to-logit) to real population counts (TP 119,080; FN 21,440; FP 17,163; TN 12,425,229). The naive single-step linear extrapolation predicts a catastrophic Dice collapse (0.86→0.64) — but real training's actual validation Dice held near baseline (0.9030 vs 0.9063). Concludes this is NOT evidence collateral damage is wrong, but evidence that single-step extrapolation is invalid for predicting cumulative multi-epoch training dynamics (continuous `L_seg` counter-pressure at every step isn't captured by a one-shot 15-step replay).

**Self-contradiction/hedging**: The entire document is built around openly flagging a "real, large discrepancy" rather than resolving it — explicitly states "this is not resolved by adjusting the numbers to fit" and repeatedly emphasizes what it does NOT establish (whether collateral damage matters at true training equilibrium).

**Bug found/fixed**: No coding bug — a methodological/conceptual limitation identified (extrapolation validity), not a bug per se.

**Cross-references**: Uses `PHASE_E25_C62_REPRESENTATION_TO_LOGIT.md` (transition rates) and `PHASE_E25_C62_CORRECTED_REANALYSIS.md` (population counts). Its predicted collapse is directly tested and refuted by `PHASE_E25_C62_REAL_TRAJECTORY_CONFUSION.md`.

---

## PHASE_E25_C62_REAL_TRAJECTORY_CONFUSION.md

**Summary**: Measures REAL cumulative TP/FN/FP/TN transitions directly from consecutive saved checkpoints (no perturbation/extrapolation) for both A and C6-2, 8 validation subjects. Finding: "Outcome A" — both conditions show healthy, comparable positive net flux in late training (A: +670, C6-2: +638) — the one-shot extrapolation's predicted collapse does NOT occur in real training. Flags a real discrepancy: on this experiment's own 8-subject pooled Dice, C6-2 finishes marginally ABOVE A (0.9235 vs 0.9211) — opposite direction from H4's headline (C6-2 0.9030 < A 0.9063).

**Self-contradiction/hedging**: Explicitly discloses the Dice-direction reversal as "a genuinely interesting... discrepancy that must be flagged, not resolved here," attributing it to 8-subject small-sample variance vs. the 125-subject full validation set, and explicitly defers to H4 as the "established, locked" metric. Does not let its own contrary result stand unchallenged.

**Bug found/fixed**: None — sidesteps the extrapolation problem by design rather than debugging it.

**Cross-references**: Directly tests and refutes the predicted collapse from `PHASE_E25_C62_POPULATION_WEIGHTED_ACCOUNTING.md`. Its own 8-subject discrepancy is resolved by `PHASE_E25_C62_REAL_TRAJECTORY_CONFUSION_FULL.md`.

---

## PHASE_E25_C62_REAL_TRAJECTORY_CONFUSION_FULL.md

**Summary**: Full 125-subject-scale rerun of the prior 8-subject experiment. Resolves the discrepancy: at full scale, A (0.8899) again exceeds C6-2 (0.8889), matching H4's direction — confirming the 8-subject reversal was a small-sample artifact. The residual gap to H4's exact numbers (0.9063/0.9030) is explained by checkpoint-schedule coarseness (true best epoch 29 falls between tested checkpoints 25/30). New finding: C6-2's net flux (+18,687) is actually 1.76x LARGER than A's (+10,614), with a better benefit/damage ratio (1.326 vs 1.166) — meaning SC-TAM's real dynamics are MORE favorable than baseline's, yet still yields slightly lower final Dice. Concludes per decision tree: net flux and final Dice are not simply proportional; recommends moving from gradient-mechanics investigation to spatial error-concentration analysis.

**Self-contradiction/hedging**: States plainly "this makes C6-2's Dice shortfall genuinely more interesting, not less" — a mechanism with better flux still yields worse Dice, an explicitly unresolved paradox the document does not force an explanation for.

**Bug found/fixed**: None — a scale-up confirmation, no new bugs.

**Cross-references**: Confirms and corrects `PHASE_E25_C62_REAL_TRAJECTORY_CONFUSION.md`; reconciles exactly against `PHASE_E25_C62_RESULTS.md`'s H4 headline numbers (best.pth epoch 29 confirmed).

---

## PHASE_E25_C62_SPATIAL_ERROR_ANALYSIS.md

**Summary**: Compares A's and C6-2's real best checkpoints (A epoch 23, 0.9063; C6-2 epoch 28, 0.9030) across full 125-subject set for spatial/structural patterns. Honest null: no single spatial category (slice depth, boundary distance, lesion size, fragmentation) cleanly explains the gap. One real-but-puzzling signal: boundary-distance error excess peaks one bin AWAY from the boundary (2-4 voxels: ratio 1.30) rather than monotonically decaying from the boundary itself — explicitly does not match the "SC-TAM specifically damages boundary voxels" story. 4-way classification shows real two-way churn (15,009 improved, 17,432 regressed voxels) — not simple degradation. Per-subject correlations with lesion volume/fragmentation are weak/non-significant.

**Self-contradiction/hedging**: Explicitly instructed "not to force a tidy story" and does not — states the deficit "looks genuinely diffuse... rather than concentrated in one interpretable category," an honest non-finding presented as the final word on this specific comparison (though the NEXT document reframes the same underlying data to find a real pattern in baseline A's own absolute error, not the A-vs-C6-2 delta).

**Bug found/fixed**: None.

**Cross-references**: Follows up on `PHASE_E25_C62_REAL_TRAJECTORY_CONFUSION_FULL.md`. Its own raw data is re-mined by `PHASE_E25_NEXT_STRUCTURAL_PIVOT.md` to find a real small-lesion-detection signal in A's absolute (not comparative) error.

---

## PHASE_E25_C63_GATE_RETROSPECTIVE_TEST.md

**Summary**: Tests a confidence gate to filter SC-TAM's collateral damage before committing to a real C6-3 training run. The original symmetric gate `w(x)=4p(1-p)` FAILS the user's strict criterion (retains only 23.4% of FN/FP corrective signal, not "most"). A redesigned ground-truth-conditioned asymmetric gate `w(x)=|p(x)-target(x)|` PASSES decisively: 92.3% corrective retention, 99.2% damage removal, 122.55x ratio improvement. Verdict: GO — train Confidence-Gated SC-TAM (C6-3) once.

**Self-contradiction/hedging**: Explicitly refuses to accept the symmetric gate's large aggregate ratio gain (8.68x) as sufficient, calling out that the improvement "comes from the sheer size of the TN-damage pool being crushed, not from the gate doing its intended job on FN specifically" — a self-imposed stricter bar than the raw number would suggest, i.e., resisting a tempting-looking positive.

**Bug found/fixed**: None — a pure re-weighting of already-collected data.

**Cross-references**: Builds on `PHASE_E25_C62_REPRESENTATION_TO_LOGIT.md` (collateral damage finding) and `PHASE_E25_C62_CORRECTED_REANALYSIS.md` (population imbalance). Its GO verdict is later shown in `PHASE_E25_NEXT_STRUCTURAL_PIVOT.md` to have produced C6-3 at 0.9026 Dice — still below A and the bar — closing the whole C6 family.

---

## PHASE_E25_NEXT_STRUCTURAL_PIVOT.md

**Summary**: Pre-registered pivot specification away from the exhausted SC-TAM/margin-loss family (A=0.9063, C6-2=0.9030, C6-3=0.9026, all below the 0.9183 bar) toward Deep (multi-scale) Supervision. Builds a NEW failure model from A's own absolute error (not the A-vs-C6-2 delta): small-lesion whole-component detection failure accounts for 58.9% of A's total Dice shortfall (corr(component size, miss rate)=-0.601, p<0.0001; 44.0% of subjects have ≥1 fully missed component). Proposes adding two auxiliary segmentation heads at dec3 (D/4) and dec2 (D/2) resolutions with explicit falsifiable predictions and a 4-gate decision structure. No training run yet at time of writing.

**Self-contradiction/hedging**: None — a forward-looking spec; explicitly labels itself "not yet started, not authorized here."

**Bug found/fixed**: None.

**Cross-references**: Explicitly built by re-mining `PHASE_E25_C62_SPATIAL_ERROR_ANALYSIS.md`'s raw JSON data with a different question (A's absolute failure, not A-vs-C6-2 comparison). Closes the C6 family referencing `PHASE_E25_C63_GATE_RETROSPECTIVE_TEST.md`. Its predictions are tested next in `PHASE_E25_STRUCTURAL_PIVOT_1A_MECHANISM_AUDIT.md`.

---

## PHASE_E25_STRUCTURAL_PIVOT_1A_MECHANISM_AUDIT.md

**Summary**: Tests Deep Supervision's actual best checkpoint (0.9091, +0.28pp vs A's 0.9063) against its own originally-hypothesized mechanism. Result: "Possibility B" — the ORIGINAL hypothesis (deep supervision improves small-component whole-lesion DETECTION) is **not supported** (detection rate flat/slightly worse: 20.6% vs A's 22.0%; missed components 181 vs 178, slightly worse). A follow-up measurement (not in the original request) finds the REAL mechanism: for lesions that ARE detected, DS substantially improves segmentation QUALITY at small sizes (1-50 voxel component Dice: 0.186→0.261, +40% relative; effect shrinks monotonically with size, vanishing above ~1000 voxels).

**Self-contradiction/hedging**: The document's title finding is explicitly reframed mid-document — the originally hypothesized mechanism (detection) fails, but rather than declaring the whole pivot a failure, it pivots the explanation itself to a materially different, more precise mechanism (quality-of-detected, not detection-of-new). This is presented transparently as a correction of the initial hypothesis, not a hidden walkback.

**Bug found/fixed**: None.

**Cross-references**: Tests hypothesis from `PHASE_E25_NEXT_STRUCTURAL_PIVOT.md`. Extended to 4 conditions (D4-only, D2-only, Both) in `PHASE_E25_STRUCTURAL_PIVOT_1B_4WAY_MECHANISM_COMPARISON.md`.

---

## PHASE_E25_STRUCTURAL_PIVOT_1B_4WAY_MECHANISM_COMPARISON.md

**Summary**: Compares A vs D4-only (0.9096 training-log Dice) vs D2-only (0.9080) vs Both (0.9091) — 4-way analysis, full 125-subject set. Discloses a metric-definition discrepancy upfront: this analysis's per-subject-mean Dice values (~0.887-0.898) are systematically ~1.7-1.9pp lower than each checkpoint's own recorded batch-pooled `best_val_dice`, but rankings/relative gaps are consistent across both metrics. D4-only's small-lesion gain is concentrated in ~10/125 subjects (73.4% of total positive delta). Crucially: NONE of the 6 pairwise comparisons (including A vs any DS variant) reach statistical significance on whole-volume Dice (all p>0.07) — a real limitation. D2-only and D4-only are NOT interchangeable at the component level (D2-only stronger at 50-400 voxels; D4-only more balanced 50-1000 + boundary benefit; only "Both" shows a clear positive effect at 1-50 voxels).

**Self-contradiction/hedging**: Explicitly flags that the earlier document's "D4-only reproduces Both's small-lesion mechanism" story "does not hold at the component level — it only held at the subject-level aggregate," calling the evidence here "genuinely mixed, not a tidy confirmation." Also explicitly notes that none of the headline Dice comparisons are statistically significant, undercutting confidence in all of Structural Pivot 1/1A's numeric claims (though the size-stratified/component-level pattern is treated as more informative than the failed significance test).

**Bug found/fixed**: No code bug — a metric-definition mismatch identified and explained (batch-pooled vs per-subject-mean Dice), not a bug but a documented discrepancy.

**Cross-references**: Extends `PHASE_E25_STRUCTURAL_PIVOT_1A_MECHANISM_AUDIT.md` from 2 to 4 conditions. Feeds into `PHASE_E26_D4_RESPONSE_PHENOTYPE.md`.

---

## PHASE_E26_D4_RESPONSE_PHENOTYPE.md

**Summary**: Searches for a predictor of which subjects/components benefit most from D4 supervision, beyond size/baseline-Dice. Decision-gate result: FAILURE. Two initially-promising candidates (`surface_to_volume`, intensity statistics) are traced to nonlinear reparameterizations of size and collapse under proper nonlinear confound control (partial r=-0.122, p=0.118 after correction, vs an initial misleading linear-control result of partial r=-0.51). Baseline (A) Dice dominates: joint model (size+baseline Dice) explains R²=0.51 of subject-level variance. Top-10 highest-gain subjects are near-catastrophic baseline failures being "rescued," not a subtle quality phenomenon.

**Self-contradiction/hedging**: The document explicitly walks back its OWN initial finding within the same document: an initial partial correlation (linear-control only) showed a large apparent effect that was then "investigated and found to be a control-specification artifact, not a real finding" once nonlinear size terms were properly included.

**Bug found/fixed**: A methodological/statistical error (insufficient confound control), not a code bug — caught and corrected within the same document before being reported as final.

**Cross-references**: Follows up on `PHASE_E25_STRUCTURAL_PIVOT_1B_4WAY_MECHANISM_COMPARISON.md`. Decision: "do not force adaptive D4 supervision, search for a different mechanism" — sets up the subsequent E27 audit / E28+ line of investigation.

---

## PHASE_E27_PROJECT_AUDIT.md

**Summary**: A comprehensive, from-source, question-and-answer style project audit (architecture, training config, data preprocessing, sampling, augmentation, dataset split, GPU, existing experiments). Key findings: exact architecture (5.6M params UNet3D variants), exact losses (FocalTversky+EvidentialBeta, 0.5/0.5), **no data augmentation exists anywhere in the pipeline**, **no patch-based/cropped training — whole-volume resize only** (64³ from native ~240×240×155, a ~3.75-3.75-2.4x downsampling), no lesion-aware sampling, no held-out third test split (the 125-subject validation set has been reused for every decision across the whole E12-E26 arc, not truly independent), and confirms the actual GPU is an RTX 5050 8GB (correcting an assumed RTX 2050 4GB). Concludes with "the single most important audit finding": small lesions may be losing information at the PREPROCESSING stage (before training even starts), reframing months of margin-loss and deep-supervision mechanism hunting as possibly working around, not on, the real problem.

**Self-contradiction/hedging**: Multiple fields explicitly marked `unknown` rather than guessed (target journal, exact dataset citation scope for novelty, architecture-change latitude, professor's novelty bar) — a document that is transparent about the limits of what it could verify vs. assume.

**Bug found/fixed**: None new, but surfaces that project memory's assumed GPU (RTX 2050 4GB) was WRONG — actual hardware is RTX 5050 8GB, materially changing the compute-budget assumptions used in prior planning.

**Cross-references**: Synthesizes and cites `PHASE_E25_STRUCTURAL_PIVOT_1A_MECHANISM_AUDIT.md` (G2), `PHASE_E25_C62_SPATIAL_ERROR_ANALYSIS.md` (G3), `PHASE_E25_STRUCTURAL_PIVOT_1B_4WAY_MECHANISM_COMPARISON.md` (metric caveat, L4). Motivates the entire E29+ resolution-focused investigation line.

---

## PHASE_E28_SUBREGION_INFORMATION_AUDIT.md

**Summary**: Tests whether BraTS subregion composition (NCR/ED/ET) explains A's missed-component pattern, using the original 4-class labels for the first time in the project. Raw correlations look large (%NCR vs Dice ρ=0.796, %ED ρ=-0.687, both p<0.0001; chi-square p=0.026) but collapse to near-zero and non-significant (partial r=0.02-0.04, all p>0.48) once component size is properly controlled — incremental R² from adding subregion composition is 0.0002. Root confound: tiny post-resize components are geometrically almost always classified as edema-dominant (median ED-dominant size = 3 voxels) purely by chance/geometry at small sizes, not because A has a specific edema-segmentation weakness. Strong-gate verdict: KILL.

**Self-contradiction/hedging**: None internally — the document is itself structured explicitly as "raw relationship looks real → confound check → kill," a clean demonstration of the project's standing discipline rather than a document contradicting its own verdict.

**Bug found/fixed**: None — a statistical confound identified, not a code bug.

**Cross-references**: Uses the same 363-component population as E25-E27. Explicitly says its null finding converges on "the same underlying explanation" as `PHASE_E26_D4_RESPONSE_PHENOTYPE.md` (component size is the dominant confound). Recommends the "performance track" (augmentation, resolution, sampling — per E27's audit) as the better-evidenced next step.

---

## PHASE_E29_RESIZE_SURVIVAL.md

**Summary**: Model-independent geometric analysis of how native BraTS lesion components survive resize to 64³/96³/128³. Headline: at 64³, the MEDIAN native component vanishes entirely (0 surviving voxels) — over half of all 749 native lesions are erased by preprocessing before training. But 65.3% of all components are ≤5 voxels natively (44.3% exactly 1 voxel) — likely annotation noise/specks that no resolution can recover. For the genuinely-small-but-real lesion band (5-150 native voxels, ~13% of population), resolution recovery is real, monotonic, and substantial (2-8x per doubling; 128³ recovers segmentable sizes of 23-72 voxels from near-total wipeout at 64³).

**Self-contradiction/hedging**: Explicitly frames its own finding as having "an important, honestly-disclosed complication" — does not let the headline resolution-hypothesis-support stand without immediately qualifying it with the annotation-noise caveat.

**Bug found/fixed**: None — pure geometric measurement.

**Cross-references**: Directly informs and is cross-checked by `PHASE_E30_DTC_FEASIBILITY.md` (same 749-component count, independently confirmed). Explicitly frames itself as the "pre-training diagnostic" before an A64/A96/A128 resolution-ceiling training experiment (not covered in this batch).

---

## PHASE_E30_DTC_FEASIBILITY.md

**Summary**: Feasibility/novelty gate for "Degradation-Trajectory Constraint" (DTC) — a proposed loss weighting based on how a component's geometric survival degrades across a resolution sweep (160³→64³). Gate A (geometric behavior): PASS — survival is a near-binary threshold effect, not smooth. Gate B (predictive information beyond size): the degradation-excess signal `G_c` predicts final quality (subject-clustered bootstrap CI [0.49,0.77] on Spearman ρ) — BUT this entire predictive result applies ONLY to the 25.8% of components with well-defined reference probability mass, which are overwhelmingly LARGE components (median native size ~51,000 voxels) — nearly the opposite of the small-lesion population that motivated DTC. Gate D (literature): N2, no close prior art. Overall: qualified GO, but flags the scoping mismatch as the single most important limitation, requiring resolution before implementation.

**Self-contradiction/hedging**: Extremely explicit: "All four gates technically pass, but Gates B/C pass on a population that is the near-opposite of DTC's motivating failure mode" — the document itself insists the GO should not be read as "DTC helps small lesions... it has not yet been tested there."

**Bug found/fixed**: THREE bugs caught and fixed within this single document: (1) an OLS fit producing a misleadingly near-zero R²=0.003 despite a large Spearman correlation, traced to the outcome being near-binary not smooth (measurement lesson, not a code bug); (2) a broken ratio (`s_hat_c`) producing mean values in the millions, fixed via a pre-declared reference floor excluding 74.2% of components; (3) a circularity bug where the predictor (`G_c`) and the outcome shared the same α=1.0 data point, inflating apparent R² to 0.87 — fixed by restricting `G_c`'s construction to α∈{0,0.25,0.5,0.75} only.

**Cross-references**: Confirms E29's 749-component count independently. Its DTC mechanism is later killed in its reformulated form by `PHASE_E31_SMALL_LESION_DTC.md`.

---

## PHASE_E31_SMALL_LESION_DTC.md

**Summary**: Attempts a small-lesion-safe reformulation of DTC, abandoning the broken ratio for new quantities (`M_c`, `V_c`, `m_c`, `G_c`) that don't divide by the model's own reference prediction. Decision: **KILL DTC in both original and reformulated forms.** A candidate signal (Candidate C) that initially looked promising in the small-lesion population (S1-S4, ρ=-0.245, bootstrap CI excluding zero) is traced to an artifact: 522/611 "S1-S4" components have `V_c(α=1.0)=0` (geometrically vanished), making `m_c` an epsilon-guarded division producing numerical noise, not a real quantity. After restricting to the honest 89-component population with real nonzero footprint, the apparent signal (ρ=0.255) FAILS the permutation safeguard decisively — the real correlation is SMALLER than 96.5% of permutation trials. One thing survives: the critical-resolution quantity α_c is real and not fully redundant with size (R²=0.37) — carried forward, untested against the model, to E32.

**Self-contradiction/hedging**: The document explicitly narrates its own near-miss: "This looked like a genuine, robust, small-lesion-specific finding... It was not" — walking back its own promising intermediate result within the same document once the permutation safeguard is applied.

**Bug found/fixed**: A conceptual/measurement artifact (epsilon-guarded division activating on geometrically-vanished components) — caught via the permutation safeguard methodology itself, explicitly praised as "worked exactly as intended."

**Cross-references**: Builds on and kills the DTC mechanism from `PHASE_E30_DTC_FEASIBILITY.md`. Preserves α_c as a candidate, tested next in `PHASE_E32_ALPHA_C_FEASIBILITY.md`.

---

## PHASE_E32_ALPHA_C_FEASIBILITY.md

**Summary**: Tests α_c (critical resolution) against a freshly-computed, real, in-distribution, thresholded component Dice at 64³ (avoiding E31's fatal out-of-distribution ratio artifact). Verdict: α_c RETAINED. Full population: real correlation (ρ=0.676) exceeds ALL 500 permutation trials. Small-lesion population (native_size≤150, n=587): naive OLS ΔR²≈0 is shown to be a multicollinearity/linear-model artifact — the real signal is a genuine, significant partial Spearman ρ=+0.132 (p=0.0014) after removing size's contribution, also passing permutation (0/500 trials exceed it) and subject-clustered bootstrap (CI [0.144, 0.346], excluding zero). Effect is modest but real. Proposes candidate weighting function `w_c = 1 + κ(1-α_c)_+` for the NEXT step (literature audit), explicitly not yet implemented/trained.

**Self-contradiction/hedging**: Explicitly diagnoses its own initial ΔR²≈0 result as potentially misleading before investigating further ("this ΔR²≈0 result needed scrutiny before being reported as a negative finding") — an internal self-check, not an external correction.

**Bug found/fixed**: A methodological point flagged as a "genuine methodological lesson" — ΔR² is uninformative when the baseline model is already near a ceiling (full-population case); not a code bug.

**Cross-references**: Reuses α_c unchanged from `PHASE_E31_SMALL_LESION_DTC.md`. Explicitly contrasts its own trustworthiness against E31's failure (Section 7). Feeds into the novelty audit in `PHASE_E33_ALPHA_C_NOVELTY_AUDIT.md`.

---

## PHASE_E33_ALPHA_C_NOVELTY_AUDIT.md

**Summary**: Literature novelty audit for the α_c weighting mechanism. No exact prior-art collision found. Closest paper: Component-Adaptive Tversky (arXiv 2604.08015, 2026) — overlap score 3/5, same topic (small-structure brain MRI segmentation) but mechanistically pure size-inverse-power weighting with no degradation process. Persistent-homology/topological-loss family is the only formally analogous construct (a "critical value" concept) but filters over intensity, not spatial resolution — a genuinely different axis. Novelty score: MEDIUM-HIGH. Decision: GO — proceed to the controlled comparison experiment (A → A+size-weighting → A+α_c-weighting), not skip directly to a full training campaign.

**Self-contradiction/hedging**: None significant — a careful, appropriately-hedged novelty audit that explicitly caveats it is not an exhaustive full-text MICCAI/TMI/MIDL search.

**Bug found/fixed**: None (literature search only).

**Cross-references**: Builds on `PHASE_E32_ALPHA_C_FEASIBILITY.md`'s empirical distinction (α_c ≠ f(size)). Its recommended controlled comparison is executed (and both conditions fail) in `PHASE_E34_ADAPTIVE_SIZE_REWEIGHTING_KILLED.md`.

---

## PHASE_E34_ADAPTIVE_SIZE_REWEIGHTING_KILLED.md

**Summary**: Note: this document was reconstructed during a later documentation audit from raw disk artifacts — no synthesized report existed at the time of the original experiment, unlike neighboring phases. Implements and trains BOTH weighting schemes from E33's controlled-comparison requirement: `[S]` size-based (`w_c=(V_c+eps)^-gamma`) and `[R]` alpha_c-based (`w_c=1+κ(1-α_c)_+`). BOTH FAIL CATASTROPHICALLY: `[S]` reaches only 0.7360 Dice (-17.03pp vs baseline 0.9063), `[R]` reaches 0.7108 (-19.55pp), with `[R]` showing signs of a degenerate "predict everything as tumor" collapse early in training. Root cause: the new auxiliary loss term was calibrated by LOSS VALUE parity only (not gradient magnitude), destabilizing training. This failure is identified as the direct origin of the project's later-mandatory "calibrate by gradient magnitude, not loss value alone" safeguard.

**Self-contradiction/hedging**: None — a clean, disclosed kill. Notably transparent about being a retroactively-written report for a real, previously-undocumented experiment.

**Bug found/fixed**: A dataloader-caching bug caught BEFORE training: `calibrate_lambda_cw.py` cached components for only a fixed first-32-subject prefix, silently skipping virtually all shuffled batches (probability all 8 land in that 32-subject slice ≈0), so calibration was silently returning its fallback default (1.0) rather than a real measurement. Fixed by caching on-demand.

**Cross-references**: Directly executes the controlled comparison recommended by `PHASE_E33_ALPHA_C_NOVELTY_AUDIT.md`. Explicitly identified as the origin of a project-wide calibration policy applied "starting with E45's lambda_d8 calibration." **Note**: there is a second, unrelated document also named "E34" — `PHASE_E35_SCALE_CONTEXT_FEASIBILITY.md`'s own title says "E34" internally (see next entry) — a filename/numbering ambiguity in the project's own phase numbering, not this document's fault.

---

## PHASE_E35_SCALE_CONTEXT_FEASIBILITY.md

**Note on numbering**: This file is named PHASE_E35 but its own document header reads "E34 — Scale-Conditioned Context Supervision" — an internal phase-numbering inconsistency in the project's own records (there are now two documents both self-labeled "E34": this one and `PHASE_E34_ADAPTIVE_SIZE_REWEIGHTING_KILLED.md`), flagged here as found, not resolved.

**Summary**: Feasibility audit (no training) for supervising coarse decoder heads with a Gaussian-blurred "contextual field" version of the tumor outline, instead of an exact downsampled copy, on the theory that blurred small-lesion signal might survive coarsening where the exact outline vanishes. Stopped early after Section C fails the first pre-registered Go condition decisively. Finding 1: exact-outline collapse at D4 is confirmed severe (100% of S1-S4 fragments ≤150 voxels have zero surviving voxels). Finding 2 (core, negative): the contextual (blurred) field does NOT meaningfully survive either — an initial "100% survival" result is caught as a measurement artifact (any nonzero value trivially satisfied by Gaussian blur math); against a real >0.1 threshold, meaningful rescue is 0% for S1-S4 at every blur width. Verdict: KILL.

**Self-contradiction/hedging**: The document catches its own first-pass measurement (100% survival) as "suspiciously perfect" and investigates before trusting it — an internal self-correction within the same document, not an external one. Also performs an unusual post-hoc "9b" section specifically hunting for coding errors that might have produced the KILL artificially, finding one real issue (scipy zoom sparse-sampling can skip small isolated objects) but confirming it doesn't affect the Gaussian-family verdict (only affected 33/8,988 dilation-family measurements at settings unrelated to the small-lesion question).

**Bug found/fixed**: Two real construction bugs caught before trusting results: (1) blurring the already-shrunk-down outline instead of blurring at full resolution first then shrinking (mathematically guaranteed to fail by construction, unrelated to whether the real idea has merit); (2) a computational size limit on blur calculations was silently cutting off 3/4 of pre-declared blur widths at the coarsest resolution. Plus the post-hoc scipy-zoom sparse-sampling issue described above.

**Cross-references**: Follows directly from the D4-supervision line of work (implicitly `PHASE_E25_STRUCTURAL_PIVOT_1A/1B`). Its confirmed exact-outline-collapse finding is consistent with `PHASE_E29_RESIZE_SURVIVAL.md`.

---

## PHASE_E36_DEEP_SUPERVISION_AUTOPSY.md

**Summary**: Explains, mechanistically, why D4-only supervision (best DS variant, +0.33pp) outperforms D2-only and Both — no new training. Finding 1 (counter-intuitive but confirmed): D4's gradient becomes LESS aligned with the main loss gradient over training (0.26→0.84→0.25 by epoch 30), not more — opposite of the "alignment is good" intuition. Finding 2 (the real explanation): D4's own loss gets "stuck" around 0.15 and never converges like the main loss does (D4 loss÷main loss grows from 0.69→2.74) — D4 keeps supplying real gradient signal exactly where the main loss has already converged and stopped exploring. Finding 3: this is a boundary/partial-coverage problem, not primarily a small-lesion-SIZE problem — partial-coverage voxels (boundary cells) have ~10x the prediction error of full-coverage voxels (0.30 vs 0.03). Finding 4: 61% of D4's supervised mass sits in partial-coverage cells vs only 39% for D2 — a geometric, not size-based, explanation matching the 1.86x stuck-loss gap ratio. Finding 5: rules out a candidate explanation (D4's gradient reshaping shared layers more than D2's) — both are 99%+ concentrated in shared early layers, not a D4-specific property.

**Self-contradiction/hedging**: None — internally consistent, and explicitly labels Finding 5 "ruled out one candidate explanation" without hedging.

**Bug found/fixed**: None (measurement-only).

**Cross-references**: Explains the mechanism first established (without explanation) in `PHASE_E25_STRUCTURAL_PIVOT_1B_4WAY_MECHANISM_COMPARISON.md`. Lists candidate future directions (not implemented) that are explored in `PHASE_E37_GRID_ENTROPY_FEASIBILITY.md`, `PHASE_E38_ORTHOGONAL_RESIDUAL.md`, and later phases.

---

## PHASE_E37_GRID_ENTROPY_FEASIBILITY.md

**Summary**: Tests whether a precise per-tumor grid-ambiguity (binary entropy of coverage fraction) metric captures E36's qualitative boundary-geometry story. Verdict: KILL. Test A (D4 more ambiguous than D2) fails as literally stated for the small-lesion population that matters most — for fragments ≤20 voxels post-resize, coarsening NEVER increases ambiguity (0% show D4>D2 entropy); only large fragments (>100 voxels) show the expected pattern (99%). This is proven as a mathematical necessity (diluting small mass into a larger cell pushes coverage fraction toward zero = LOW entropy), not a fixable formula issue. Three other tests (size-controlled signal, differential ΔH→ΔE prediction rho=0.78, permutation safeguard) all PASS, but per the pre-declared rule, failing the one test that "defines why the mechanism would matter for the population this project cares about" is a decisive kill, not a 3-of-4 pass.

**Self-contradiction/hedging**: Very explicit and disciplined: acknowledges 3/4 tests pass with strong statistics but insists "do not treat a 3-out-of-4 pass as a pass when the failing piece is the one that defines why the mechanism would matter" — actively resists over-interpreting a mostly-positive result.

**Bug found/fixed**: An internal inconsistency caught and fixed during the analysis: an earlier draft of the subject-clustered bootstrap CI accidentally reported a CI for the RAW (size-confounded) correlation instead of the size-controlled one, which "would have been a misleading, self-contradictory result if published as-is."

**Cross-references**: Directly follows up on `PHASE_E36_DEEP_SUPERVISION_AUTOPSY.md`'s boundary/partial-coverage explanation, attempting to quantify it precisely. Notes E36's simpler "boundary fraction" measure is nearly redundant with entropy at D2 (rho=0.94) but not at D4 (rho=-0.19).

---

## PHASE_E38_ORTHOGONAL_RESIDUAL.md

**Summary**: Follow-on to E36, using only already-computed gradient data (no new training/inference). Tests whether D4's "orthogonal residual" (the part of its gradient not explained by the main gradient) predicts benefit. The literally pre-registered quantity `I_r` (fraction of auxiliary gradient that's independent) FAILS the kill criterion — its cross-condition ordering (Both > D2-only > D4-only) is REVERSED from the known real Dice ordering (D4-only best). Diagnosed as a mathematical property of the formula (saturates near 1.0 across most of the real observed similarity range 0.2-0.8). A single pre-declared alternative, `Q_r` (residual sized RELATIVE TO the main gradient's current strength), is tested instead and SURVIVES: matches the exact known Dice ordering (D4-only > Both > D2-only), p<0.00001, robust to removing extreme batches, 95% pairwise dominance.

**Self-contradiction/hedging**: The document explicitly frames both a failure (`I_r`) and success (`Q_r`) of two different operationalizations of "the same qualitative idea" — states plainly "the honest state of affairs is 'one principled alternative passed a demanding test,' not 'the original hypothesis, unmodified, was confirmed.'" Explicitly flags the alternative was chosen for principled reasons BEFORE testing (to preempt an accusation of post-hoc metric shopping), while still disclosing that "a different choice, made with equally defensible reasoning, might have been made instead."

**Bug found/fixed**: No code bug — a formula-sensitivity issue diagnosed and disclosed before it corrupted the conclusion.

**Cross-references**: Directly builds on `PHASE_E36_DEEP_SUPERVISION_AUTOPSY.md`'s gradient measurements (reuses saved data, no new inference). Its `Q_r`/`P4` quantity is tested in a genuinely different, controlled setting (a lambda sweep) in `PHASE_E39_MECHANISM_TEST.md`, where it is then KILLED.

---

## PHASE_E39_MECHANISM_TEST.md

**Summary**: (Titled "E39C" internally) Tests whether E38's surviving residual-information quantities (`P4`, `M4`) predict the outcome of a NEW controlled λ-sweep experiment (6 conditions, λ=0 to 2.0; best result λ=0.25 at +0.28pp, still below the +1pp bar). Verdict: KILL. `P4`/`M4` climb MONOTONICALLY with λ, while the real outcome (Dice) shows an inverted-U (rises then falls, peaking at λ=0.25) — the two curves have "fundamentally different shapes." λ=0.25 (best Dice) actually has the LOWEST `P4`/`M4` of the three λ values that beat baseline comparably. A denominator-confound risk (M4 driven by `|g0|` shrinking rather than genuine residual growth) is explicitly checked and ruled out — `M4`'s growth tracks the residual's own raw size (rank correlation +1.00), not a shrinking denominator.

**Self-contradiction/hedging**: Explicitly separates what this phase kills from what it does NOT kill — clarifies that E38's finding (which held across three pre-existing, architecturally different conditions) is a "separate result, on a separate comparison, and is not contradicted by this phase." A careful, non-overreaching scoping of a negative result.

**Bug found/fixed**: None — a confound was suspected, checked, and ruled out (not found to be present).

**Cross-references**: Directly tests `PHASE_E38_ORTHOGONAL_RESIDUAL.md`'s `P4`/`Q_r` quantity in a new context; explicitly restates E39A/B's standing verdict (none of 6 λ conditions reach +1pp) as not reopened here.

---

## PHASE_E40_SUBSPACE_AUDIT.md

**Summary**: Tests whether D4's DIVERSITY/reproducibility of independent optimization directions (not just magnitude, per E39C's kill) explains its benefit. Uses a new "effective rank" measure derived via reloading trained models for fresh gradient measurements (E36/38/39's saved data was insufficient — a real, disclosed scope expansion). Pooled across all 3 conditions, D4's effective rank exceeds D2's in 2/3 comparisons but NOT statistically distinguishable from chance (p=0.17); within the single condition where both are architecturally comparable ("Both"), D4 IS significantly higher (p=0.016) — but this narrower positive result does NOT survive a permutation safeguard (87th percentile of 500 shuffles, not decisive) and does NOT predict the actual six-condition λ-sweep outcome (exact test: real pairing no more extreme than 30% of all 720 possible random pairings). Verdict: KILL.

**Self-contradiction/hedging**: Explicitly discloses the one positive-looking result (Section 3's within-condition finding) "in full because it is real... but it should not be read as evidence the underlying hypothesis is 'almost right'" — actively pre-empting a reader's temptation to over-weight the partial positive.

**Bug found/fixed**: None — verifies its own novel effective-rank computation method against synthetic data (matched to 8 decimal places) before trusting it on real data, a validation step rather than a bug fix.

**Cross-references**: Follows directly from `PHASE_E39_MECHANISM_TEST.md`'s finding that magnitude alone doesn't explain the outcome. Its KILL verdict explicitly redirects the project's next step away from further gradient-level investigation toward "the more basic, structural question of how the target itself changes shape under the model's own resolution-reduction step" — setting up E41/E42.

---

## PHASE_E41_TRANSFORMATION_ERROR.md

**Summary**: Tests whether the model's own residual prediction error (against the mathematically-correct fractional-occupancy coarse target) predicts where D4 supervision helps. First confirms (Section 0) that the project's training code already uses the mathematically correct fractional target — no thresholding bug exists, contrary to what might have been assumed. The decisive test: among fragments detected both with/without coarse supervision (146/227), correlation between residual error and actual quality improvement is essentially zero (rho=-0.002); among missed-then-rescued fragments, also null (rho=+0.09); size-controlled partial correlation also null (-0.03); permutation check shows the real result at only the 23rd percentile (not even close to a tail). The small-vs-large split this phase was specifically designed to test COULD NOT be run for small fragments — only 7 remained after filtering, "too few for any meaningful statistical statement" — explicitly disclosed as an untested gap, not a negative finding. Verdict: KILL.

**Self-contradiction/hedging**: Very careful to distinguish "tested and null" from "untested due to insufficient data" for the small-fragment case — explicitly states this "remains genuinely untested by this phase... not because it was found to be null."

**Bug found/fixed**: None — confirms an assumed bug (thresholding) does NOT exist in the actual codebase.

**Cross-references**: Follows the redirect from `PHASE_E40_SUBSPACE_AUDIT.md`. Explicitly summarizes the project's now-exhausted attempts to find ANY per-fragment predictor (size, entropy, gradient-based, now residual-error) — all failed.

---

## PHASE_E42_CROSS_SCALE_OPERATOR_AUDIT.md

**Summary**: A structural (not empirical) KILL decided BEFORE any data was computed. The proposed residual `R4 = P4 - D4(P1)` assumes the model resizes one full-resolution prediction to get its coarse predictions — checked directly against the code and found FALSE: the model has three entirely separate, independently-parameterized prediction heads (`seg_head`, `aux_head2`, `aux_head3`), each reading a different decoder stage with its own learned weights; there is no resize operation anywhere in the actual forward pass. Any computed residual would conflate genuine trunk-level scale inconsistency with independently-trained-head bias — a confound that cannot be separated without new training (explicitly forbidden for this phase). Concludes here rather than computing a "structurally compromised quantity."

**Self-contradiction/hedging**: None — an unusually short, self-aware document that explicitly justifies its own brevity ("this document is unusually short... because the kill decision was made at the structural/definitional stage, before any data was collected").

**Bug found/fixed**: No code bug — a design/assumption error in the PROPOSAL itself, caught before any implementation.

**Cross-references**: A separate, self-contained investigation from the OTHER document also filed under "E42" in this batch (`PHASE_E42_CURVATURE_COLLECTION_INCOMPLETE.md`) — explicitly distinguished as different sub-investigations sharing the same phase number.

---

## PHASE_E42_CURVATURE_COLLECTION_INCOMPLETE.md

**Summary**: A short disclosure note (explicitly written retroactively during a documentation audit) describing an INCOMPLETE investigation — a Hessian-curvature/Lanczos spectral analysis of the E39 lambda-sweep conditions. Method verification passed (Lanczos matched direct eigendecomposition to 1e-2; HVP matched finite-difference to 3.4e-5). But the actual data-collection run STOPPED mid-execution after 57 of a planned 72 primary records, with no error message (likely a manual interruption). No scientific conclusion is drawn — explicitly states the partial data is insufficient to support any claim.

**Self-contradiction/hedging**: N/A — this document exists purely to disclose an incomplete/abandoned run rather than claim any finding; it takes pains to state "no conclusion is drawn."

**Bug found/fixed**: No bug — process simply stopped without error, presumed manual interruption; not resumed.

**Cross-references**: Distinguished from the OTHER E42 document (`PHASE_E42_CROSS_SCALE_OPERATOR_AUDIT.md`, a self-contained structural kill) as a separate sub-investigation. Compares itself favorably to `PHASE_E34_ADAPTIVE_SIZE_REWEIGHTING_KILLED.md` (which WAS a completed experiment simply never written up) — this one genuinely never finished.

---

## PHASE_E43_REPRESENTATION_CHANGE.md

**Summary**: Tests E36's boundary-localization explanation directly at the representation level across baseline, D4-only, D2-only, and Both models (125 scans, forward passes only). Confirms a real, large, reproducible effect first: each model changes its OWN supervised resolution's representation far more than the other model does (D4-only nearly doubles D2-only's change at the coarse resolution, and vice versa at medium resolution) — strong statistical support. But the decisive boundary-localization test initially shows the OPPOSITE of the hypothesis (interior changes MORE than boundary) — investigated and found to be a measurement confound: for small lesions, a coarse "interior" region can be as few as 3-9 cells, meaning "interior" mislabels boundary-adjacent regions for small tumors. After correcting (restricting to scans with ≥9 interior cells, a genuinely reliable label), interior vs boundary change is statistically indistinguishable (48%/52% split, "no better than a coin flip"). Verdict: KILL specifically for E36's boundary-localization claim, while the broader "coarse supervision reshapes its targeted representation" finding is kept as real.

**Self-contradiction/hedging**: The document explicitly narrates catching its own initial wrong-direction result and correcting it within the same document — "before accepting that at face value: a real confound was found and corrected." A textbook example of self-correction happening inline rather than across a chain of documents (unlike the C6-2 sign-bug saga, which took 3 separate documents to resolve).

**Bug found/fixed**: A measurement confound (not a code bug per se) — the "interior" spatial label is unreliable for small lesions at coarse resolution, caught and corrected via a size-based split before reporting the final comparison.

**Cross-references**: Directly tests and ultimately REJECTS E36's specific boundary-ambiguity mechanistic explanation (`PHASE_E36_DEEP_SUPERVISION_AUTOPSY.md`) for WHY D4 supervision reshapes representations, while preserving E36's separate finding that D4 supervision does measurably reshape its targeted resolution. Explicitly closes this line of investigation: "not as a prompt for a fifth representation-level phase."

