# Digest Batch 1b (E14–E24 phase docs)

Resume batch covering the files listed below (E1–E13 already covered in a prior batch). Each section reflects a full read of the source file, not a filename-only inference.

---

## PHASE_E14_GRADIENT_CONFLICT_ANALYSIS.md

**Summary**: Diagnostic-only phase testing whether `L_seg` and `L_margin` gradients conflict on the shared `dec1` representation in EGGO-M. Using `torch.autograd.grad` on 7 checkpoints (epochs 1–30) restricted to anchor voxels, mean cosine similarity between the two gradients stayed small and positive at every checkpoint (+0.078 to +0.116, pooled mean +0.0938 across 47,607 anchor samples). Regime classified as "Case B, mildly leaning A" (weakly cooperative/independent) — explicitly NOT Case C (systematic interference). Concludes gradient conflict does not explain EGGO-M's Dice null.

**Self-contradiction/hedging**: None — verdict is stated once and held consistently throughout; the "mildly cooperative, not conflicting" framing in the interpretation section matches the header status exactly.

**Bugs caught/fixed**: Yes — an initial attempt to build a per-voxel decomposition of `L_seg` used a wrong loss formula entirely (digamma-based NLL instead of the codebase's real mean-based BCE, plus incorrect Tversky hyperparameters); caught by re-reading `neuroscan_3d_fixed.py`'s actual loss classes before trusting results, and the per-voxel decomposition approach was abandoned in favor of using `torch.autograd.grad` directly on the global loss scalar.

**Cross-references**: Explicitly builds on `PHASE_E13_MULTISEED_RESULTS.md` (the Dice-null result this phase investigates a cause of). Notes the epoch-1 gradient-norm-ratio correlation flips sign depending on whether the epoch-1 outlier is included (r=−0.94 vs +0.98) — flagged as a confound, not a real finding, and not attributed to another document.

---

## PHASE_E15_DECODER_SENSITIVITY.md

**Summary**: Causal intervention (frozen decoder, no retraining) testing whether artificially widening the latent margin between `dec1` embeddings and the opposite-class centroid improves Dice. Found a monotonic, significant Dice increase with push magnitude (0.9050 baseline → 0.9443 at push=14 units, matching E12f's own observed dynamic range; r=+0.9905, p=0.0001 in the realistic 0–14 range). Rejects "decoder invariance" as an explanation for EGGO-M's null. Reconciling calculation: EGGO-M's real training only ever achieved +0.21 units of net margin growth, predicting only ~+0.001 Dice gain at the measured local slope — far too small to detect, and consistent with E13's observed −0.0037 deficit. Reframes the null as an optimization/calibration shortfall rather than a false hypothesis.

**Self-contradiction/hedging**: The document itself narrates a walk-back mid-text: an earlier version of the manipulation (pushing along `seg_head`'s own weight vector `w`, sign chosen by ground truth) produced a "guaranteed" Dice of exactly 1.0000 — recognized as an oracle/label-leakage artifact, not a real result, and explicitly dropped rather than kept or reported. This is presented transparently as a design flaw caught before trusting results, not as a reversed verdict on the final finding.

**Bugs caught/fixed**: Yes, two: (1) the oracle-direction manipulation described above, dropped entirely; (2) initial push magnitudes (as fractions of raw centroid-to-centroid distance) were unrealistic (a 20% push exceeded natural `‖dec1‖` by 1.5x) and were corrected to absolute unit offsets anchored to E12f's real observed margin range.

**Cross-references**: Builds directly on E12f/E13 (margin mechanism active) and E14 (rules out gradient conflict). States it "reframes (not overturns)" `PHASE_E13_MULTISEED_RESULTS.md`'s null result — narrows rather than contradicts it.

---

## PHASE_E16_MARGIN_REACHABILITY_ANALYSIS.md

**Summary**: Measurement-only phase (no retraining) decomposing whether the optimizer moves `dec1` enough, in the right direction, toward margin. Six-part analysis: Part A shows a real, substantial movement budget (6.47 units mean displacement by epoch 30) ruling out "no movement" outcomes. Part B shows 15.7–35.4% of movement energy is margin-aligned but net *signed* displacement is small and increasingly negative (−0.86 to −1.59 across epochs), confirmed by an independently-added live-reference cross-check that also stays negative. Part C shows the margin loss gradient is consistently, positively aligned with E15's proven-useful push direction (pooled cos=+0.39, positive at every checkpoint). Part D computes reachability ratio R=−0.114 (signed) or ~0.21 (RMS-only) against the 14-unit requirement from E15. Part F finds no alternate margin metric that shows sufficient movement. Conclusion: none of pre-registered outcomes A/B/C cleanly fit; closest is a specific version of "D" (Other) — gradient correctly directed and a real movement budget exists, but only a modest, largely incoherent fraction of that budget nets out as sustained margin growth.

**Self-contradiction/hedging**: The status line itself is explicitly hedged ("does not cleanly match any single pre-registered outcome... closest fit is a mixture of B and D"), and the document is transparent that this is a genuinely ambiguous, synthesized result rather than a clean verdict — this is disclosed hedging by design, not an internal contradiction of an initially stronger claim.

**Bugs caught/fixed**: No explicit bug-fix described in this phase (it is measurement-only); the phase does document a "design choice surfaced during development, not resolved unilaterally" (the fixed-vs-live reference-frame discrepancy), reported as a finding rather than a bug.

**Cross-references**: Uses E15's `M_required=14.0` directly. Cites E14's L_seg/L_margin cosine (+0.09) as context for why margin push doesn't "compound coherently." References E12F's epoch-5 dip as independently reproduced via a different measurement pipeline (Part A's tumor-centroid drift spike).

---

## PHASE_E17_MARGIN_TARGET_STABILITY.md

**Summary**: Measurement-only follow-up testing whether the "target moves" (class-separating direction rotates), which would explain E16's "correctly-directed but non-accumulating" gradient puzzle. Four independent methods (centroid direction, margin gradient consecutive-pair stability, margin gradient long-range transport vs. epoch 1, whole-representation Procrustes rotation) all converge: severe rotation during epochs 1–5 (e.g., centroid cos 1→5 = 0.842; Procrustes closeness 1→5 = 0.468), then sharp stabilization from epoch 10 onward (consecutive-pair metrics reach >0.99 by epoch 25–30). Critically, none of the metrics ever return to their epoch-1 orientation: long-range gradient alignment vs. epoch 1 stays flat at ~0.13–0.15 for the rest of training even as local (consecutive) stability climbs past 0.70. Conclusion: target instability is real but time-limited — the representation settles into a new, permanently offset configuration, not the original one, supporting a refined "moving-target" explanation for E16's puzzle.

**Self-contradiction/hedging**: None found — the finding is nuanced (time-limited, not perpetual) but this nuance is presented consistently from the abstract onward, not walked back.

**Bugs caught/fixed**: Yes — Part 2's first attempt reused Part 1's small, uniformly-random tracking set for gradient computation, producing `margin_loss=0.0` (a true zero gradient, not a computation error) at several checkpoints because the hinge is only active for 0.1–3.3% of pairs and small uniform samples rarely land on active pairs. Fixed by switching to a larger (40,000-anchor), stratified, fixed-identity anchor set matching real training's `ANCHORS_PER_VOLUME`.

**Cross-references**: Directly investigates a hypothesis raised in response to E16's Part C/B tension. Independently corroborates E12f's epoch-5 dip and E16's Part A tumor-centroid-drift spike via a third, unrelated measurement method (a explicitly noted "third, independent confirmation").

---

## PHASE_E18_REPRESENTATION_ROTATION_SOURCE.md

**Summary**: Ablation study (baseline + 5 single-component freezes: BN, encoder, decoder, seg_head, lambda_zero) identifying which architectural component drives the early representation rotation found in E17. Four independent measures (centroid direction, Procrustes rotation, margin-gradient stability, and less-discriminating principal angles) all converge: `freeze_decoder` shows dramatically reduced rotation (e.g., Procrustes mean 0.9468 vs baseline 0.8751) while still training normally (Dice climbs to 0.843, not degenerate). `freeze_encoder` tracks close to baseline. `freeze_seg_head` shows increased/non-converging rotation, suggesting seg-head adaptation may help stabilize rather than destabilize. `freeze_bn` never produced a clean 30-epoch trajectory despite three escalating attempts (warmup 0, 1, 10 epochs) — all eventually collapsed, itself treated as informative (representation keeps moving past epoch 10, contrary to attempt 3's assumption). Original verdict: "decoder learning is the dominant source" of rotation (Success Criterion C).

**Self-contradiction/hedging — THIS IS THE KEY REVERSAL DOCUMENT PAIR (with the follow-up review)**: The document's own header explicitly states it is corrected the same day by `PHASE_E18_FOLLOWUP_PRE_E19_REVIEW.md` on two points: (1) "decoder learning causes the rotation" is walked back to "the decoder block is the strongest experimentally identified source" (a weaker, correlational-not-causal claim) because of a BN confound (both freeze_encoder and freeze_decoder leave that region's BatchNorm running-stats still updating even though the affine params are frozen — a symmetric confound neither ablation isolates away). (2) More importantly, per the header, the follow-up review checked whether reduced rotation actually restores effective margin-driven optimization and found **no**: freeze_decoder achieves the lowest rotation of any config but ALSO the lowest Dice among the three real freeze ablations (0.8432 vs baseline 0.9062), and a weaker, non-significant margin-Dice correlation (r=−0.354, p=0.126) than baseline's own significant correlation (r=−0.683, p=0.0009), despite the margin mechanism being 64x more active. The document's status line states plainly: "Decoder rotation looks like an associated phenomenon, not the bottleneck." The body of this document (methodology, incident log, Parts 1–5 measurements) is preserved as accurate; only the interpretive/causal framing is superseded.

**Bugs caught/fixed**: Four separate incidents, all described in detail: (1) `freeze_bn` attempt 1 froze running stats before any real data touched them, causing `val_dice≈1e-11` while `train_dice` looked healthy — caught via the train/eval BN divergence signature. (2) An unrelated tooling bug: a watcher script's `pgrep`-based completion check failed silently in Git Bash, launching `freeze_bn` in parallel with `freeze_encoder`, causing a CUDA OOM that killed `freeze_encoder` at epoch 27/30 — recovered by discarding corrupted runs and rerunning sequentially. (3) `freeze_bn` attempt 2 (1-epoch warmup) still collapsed progressively from epoch 2, traced to an immature running-stat snapshot going stale. Attempt 3 (10-epoch warmup) also eventually collapsed at epoch 16, and was stopped by explicit user decision rather than continuing to chase longer warmups — treated as itself the finding (no warmup length survives the instability window). (4) A stale-cache bug in the measurement script: a resume/skip-if-exists optimization reused an invalid pre-path-fix `freeze_bn_results.json`, caught because it was byte-identical to baseline's results to 15+ decimal places (impossible for genuinely different runs) — fixed by deleting the stale file and re-measuring.

**Cross-references**: Investigates the source of E17's rotation finding. Explicitly superseded/corrected by `PHASE_E18_FOLLOWUP_PRE_E19_REVIEW.md` (cited by name multiple times, including "Read this too" in the Files table).

---

## PHASE_E18_FOLLOWUP_PRE_E19_REVIEW.md

**Summary**: A same-day diagnostic review answering 7 reviewer questions about E18's implementation plus a consolidated table, using direct code/data inspection (not memory). Confirms E18's freeze mechanics (params excluded from optimizer entirely, not just `requires_grad=False`), confirms `dec1` is literally the decoder's own final block output (so freeze_decoder directly freezes the space margin is measured in, an asymmetry vs. freeze_encoder). Tabulates full 30-epoch Dice trajectories for all 6 configs. Central finding: `freeze_decoder`'s margin mechanism is ~64x more active (mean margin loss 0.09972 vs baseline 0.00155; active hinge 7.54% vs 0.12%) yet its margin-Dice correlation is weaker and non-significant (r=−0.354, p=0.126) versus baseline's real, significant correlation (r=−0.683, p=0.0009) — directly refuting the proposed causal chain "decoder adaptation → rotation → poor margin accumulation → weak Dice effect," since reducing rotation via freeze_decoder did NOT strengthen the margin-Dice relationship or lift Dice. Also corrects the earlier BatchNorm count claim (encoder actually has MORE BN layers than decoder, 8 vs 6) and formally downgrades "decoder learning causes rotation" to "decoder block is the strongest experimentally identified source."

**This IS the reversal document**: per the user's specific interest — the reversal is that `freeze_decoder`, which looked most promising in E18 (lowest rotation, cleanest stabilization by every geometric measure), is revealed here to have the **lowest Dice among the three real freeze ablations** (0.8432, vs freeze_encoder's 0.8751 and baseline's 0.9062 — worse than baseline by 0.063) and a *weaker, non-significant* margin-Dice coupling despite far higher margin activity. The document states explicitly: "we successfully stabilized a representation that simply isn't good enough" was the concerning pattern flagged, and "this was reported in E18's own numbers but not framed this way — it should have been the headline caveat, not a footnote, and I'm correcting that now." The document's own conclusion: "No, not by this evidence" — reducing rotation does not restore effective margin-driven optimization; decoder rotation is "an associated phenomenon, not the bottleneck."

**Self-contradiction/hedging**: This entire document IS a self-correction of E18's causal framing — it doesn't hedge on itself internally, but its function is to hedge/correct the prior document.

**Bugs caught/fixed**: No new code bugs; this is a re-analysis correcting an interpretive/causal-inference error (over-claiming causality from correlational ablation data) and a factual error (BN layer count).

**Cross-references**: Directly corrects `PHASE_E18_REPRESENTATION_ROTATION_SOURCE.md`. Notes `PHASE_E17_MARGIN_TARGET_STABILITY.md`'s rotation-reference-frame distinction (moving vs. fixed) was present in code all along but not surfaced clearly in either the E17 or E18 writeups — a gap attributed to both documents. Flags that E18's own rotation-magnitude summary table used only the moving-reference metric, not a fixed-reference version, calling this "a real gap, not silently assumed to be covered."

---

## PHASE_E21_5_AUDIT.md

**Summary**: A comprehensive, read-only mathematical/implementation integrity audit of E12–E21 (margin loss formula, gradient correctness via finite differences, tau_b, AdamW reconstruction, autograd graph structure, BN mutation effects, and E15/E21 geometry compatibility). Verdict: "YES, WITH SPECIFIC CAVEAT" — implementation is mathematically sound everywhere checked; no fatal bugs, sign-flips, or leakage found across 17 audited components. Key reassuring finding: E21's "margin update moves dec1 away from the useful direction" result (negative cosine) is independently replicated under two additional aggregation methods (global-pooled and class-centroid-level), with the class-centroid version showing an even stronger negative effect (−0.80 to −0.97 vs E21's own −0.24 to −0.26). Final decision: "PROCEED TO E22 UNCHANGED."

**Self-contradiction/hedging**: None on its own headline claims; it is itself a hedge-detection audit and lists graded caveats deliberately (three "C-severity" items) rather than overstating confidence, consistent with its own conclusion throughout.

**Bugs caught/fixed**: No implementation bugs found in E12-E21 (explicitly: "No fatal implementation error... was found anywhere in E12–E21" and "No D or E severity findings exist"). It documents three C-severity *interpretive* issues (not code bugs): (1) E21's amplification ratio mixes incompatible dimensionalities (807x raw parameter-vs-activation space gap) and shouldn't be quoted as a physical "gain" without RMS normalization; (2) a newly-discovered ambiguity in how this audit's own new "Version C" class-centroid metric could be constructed (sign-flippable if pooled across both classes naively) — not a bug in any existing script; (3) reconfirms E20's shadow-replay magnitude-incomparability limitation was already correctly self-disclosed. Also notes (severity B, cosmetic) that `compute_margin_loss`'s `rng` parameter is accepted but not actually used for negative sampling (uses global torch RNG instead) — flagged as misleading naming, not a correctness bug.

**Cross-references**: Audits and reconfirms accuracy of `PHASE_E18_FOLLOWUP_PRE_E19_REVIEW.md`'s BN-confound correction (found accurate). Re-verifies E15's, E16's, E17's, E19's, E20's, and E21's methodologies and formulas directly against code. Explicitly states E19–E21's chain is "further evidence AGAINST a pure 'not enough optimization' story," consistent with E17's moving-target finding.

---

## PHASE_E22_COUNTERFACTUAL_OBJECTIVE_GEOMETRY.md

**Summary**: Direct causal counterfactual test (no training) of whether locally minimizing EGGO-M's own margin loss is beneficial or harmful to Dice. Across 6 checkpoints × 8 subjects × 4 epsilons (192 measurements), local descent along `-∇L_margin` (Direction B) is followed by Dice decreasing in the overwhelming majority of cases, underperforms a matched-norm random perturbation in 24/24 checkpoint×epsilon cells, and shows the wrong-signed pooled correlation `corr(ΔL_margin, ΔDice) = +0.2432, p=6.75e-04` (should be strongly negative if locally useful) — while the pipeline's own sanity check (`A_seg` direction) correctly recovers a strong expected negative correlation (r=−0.8406, p=1.74e-52), ruling out a measurement artifact. By contrast, E15's independently re-validated useful direction (Direction D) improves Dice in 192/192 individual measurements with zero exceptions. Classified as "Case C" (locally misaligned) — the strongest evidence category in the phase's own framework.

**Self-contradiction/hedging**: The document is honest about heterogeneity: Falsification 5 shows the per-checkpoint correlation is NOT uniform — epoch 5 alone trends negative (though non-significant, r=−0.26, p=0.15) while epochs 20/25 are significantly positive — explicitly stated as "not cleanly falsified nor cleanly confirmed at the per-checkpoint level," a genuine internal hedge on the pooled headline's uniformity, though not a reversal of the pooled sign itself.

**Bugs caught/fixed**: Yes — during smoke-testing, resampling stratified anchors fresh at each perturbed evaluation combined with `compute_margin_loss`'s negative-sampling using the global torch RNG (already flagged cosmetically by E21.5) caused repeated evaluations at the *same* point in parameter space to vary by 10–20% from resampling noise, large enough to swamp the true first-order signal and initially make Direction B appear to *increase* margin loss at tiny epsilon (a wrong-looking result). Fixed by pinning anchor indices and `torch.manual_seed()` per checkpoint/subject. A second bug: the random control's seed used Python's `hash()` on a string-containing tuple, which is randomized per-process by default (`PYTHONHASHSEED`) — replaced with a deterministic integer encoding.

**Cross-references**: Builds on and reuses `PHASE_E21_5_AUDIT.md`'s "proceed unchanged" verdict as its foundation. Independently re-confirms `PHASE_E15_DECODER_SENSITIVITY.md`'s Dice-vs-push trend almost exactly (+0.0400 here vs E15's original +0.0393 at epoch 30/push=14). Explicitly notes E21.5 "had no reason to know" its cosmetic RNG finding would matter this much for E22's finite-difference-sensitive design.

---

## PHASE_E23_ALGORITHMIC_REDESIGN.md

**Summary**: Design-only phase (no code/training) proposing a specific algorithmic fix to EGGO-M's margin loss in response to E22's Case-C finding. Selected method: "task-aligned projected margin loss" (`L_margin^w`), replacing the isotropic Euclidean pairwise distance with a projection onto `seg_head`'s own detached decision-direction vector `w_hat`. Compares 5 candidates, selects Candidate 2, retains Candidate 3 as a named fallback. Runs two rounds of novelty audit: Round 1 finds the method is algebraically, for this project's linear `seg_head`, exactly a rescaled pairwise logit-margin hinge (`d_ij^w = |ℓ_i-ℓ_j|/‖w‖`) — not novel embedding geometry. Round 2 finds the natural generalization of the method (a Jacobian pullback metric `M_z = J_z^T J_z`) is itself an established differential-geometry construction, not new. The defensible novelty claim narrows twice, ultimately resting on the specific combination plus its derivation from the E15→E22 diagnostic chain, not the projection mechanism itself.

**Self-contradiction/hedging**: The document's own framing is repeatedly, deliberately self-narrowing (not a surprise reversal but an explicit, disclosed two-round downgrade): Section 0 lists "three precision corrections" applied to an original draft (that `w` is not proven universally useful, that embedding-space and parameter-space constraints are distinct, that Candidate 3's gate should be continuous not discontinuous). Section 10 shows the original claim wording struck through and replaced twice, with the document stating plainly it must not claim "novel segmentation-head-aligned margin loss" or "novel task-aligned embedding geometry" — both explicitly rejected as too strong given LMNN/angular-margin/pullback-metric prior art.

**Bugs caught/fixed**: None (no code was written in this phase).

**Cross-references**: Directly responds to `PHASE_E22_COUNTERFACTUAL_OBJECTIVE_GEOMETRY.md`'s Case-C finding. Built on `PHASE_E21_5_AUDIT.md`'s verified facts. Cites `PHASE_E15_DECODER_SENSITIVITY.md`'s Jacobian sensitivity ratio (~4x stronger along `w`) as motivation, explicitly stating this is used as a design proxy, not proof of universal optimality (a correction relative to how it might have been read from E15 alone).

---

## PHASE_E24_CALIBRATION_AND_IMPLEMENTATION_SPEC.md

**Summary**: Implementation specification and calibration phase (no training yet) for E23's selected method. Calibrates `delta_d_w = 0.2553` using the same methodology as E12e's original `delta_d` calibration (fresh-init, `.train()` mode discipline), verified against two independent sanity checks (20.04% active-hinge rate within the 10–30% target band; projected distance never exceeds full Euclidean distance). Specifies an exact, minimal diff to `compute_margin_loss` (one line changes: distance computation switches between `torch.norm` and a projection when `w_hat` is provided, with `w_hat=None` guaranteed byte-identical to baseline). Defines 6 required unit tests (baseline-preservation, projection-formula correctness, sanity bound, gradient correctness via finite differences, zero-gradient-to-seg_head verification, delta_d_w/delta_d consistency) — none of which have been written or run yet as of this document.

**Self-contradiction/hedging**: None — status is honestly and consistently described as "in progress," explicitly stating no training-script code has been modified yet and no unit tests have been run.

**Bugs caught/fixed**: No new bug in this phase; it explicitly cites the precedent of E12e's own original calibration bug (using `.eval()` mode produced a `delta_d` giving 0.0% active hinge in real training, due to a ~68x BN train/eval discrepancy at fresh init) as the reason this phase's calibration script deliberately avoids `.eval()` mode.

**Cross-references**: Implements `PHASE_E23_ALGORITHMIC_REDESIGN.md`'s design. Replicates `PHASE_E12E_HYPERPARAMETER_CALIBRATION.md`'s methodology exactly (named explicitly, including its cautionary bug). Reuses E19/E20/E21's verification patterns and E21.5's audit discipline for the planned unit tests.

---

## PHASE_E24_GATE6_EXPERIMENTAL_SPECIFICATION.md

**Summary**: A locked, audit-corrected experimental specification for training-based validation of E23/E24's method (no training launched yet). Defines a condition matrix (A=baseline, B=proposed `L_margin^w`, C=segmentation-only reused from E18's `lambda_zero`, E=random-projection specificity control; D explicitly not run since it duplicates existing unit-test coverage). Introduces a strict four-level interpretation hierarchy (H1 mechanistic → H2 representation → H3 outcome → H4 specificity) explicitly designed to block the "Dice improved therefore the mechanism worked" anti-pattern. Documents seven methodological corrections from a design audit, including a further correction found during the pre-launch audit itself: condition E's hinge threshold was initially, incorrectly set to reuse B's `delta_d_w=0.2553`, when it needed its own separate calibration — corrected to `delta_d_r=0.2022` (a ~21% difference), confirming the two constants were never interchangeable. States clearly that as of this document, no training has been launched — only the diagnostic refactor and calibration groundwork are complete.

**Self-contradiction/hedging**: The document's status line itself narrates a caught-and-fixed pre-launch error (the E/`delta_d_r` calibration mixup) as part of its own status description — this is presented as a correction applied before any run, not a walked-back result, since no training had occurred yet. No walked-back *result* exists because no result exists yet.

**Bugs caught/fixed**: Yes — the E-condition `delta_d_r` miscalibration described above, caught during "pre-launch launch-configuration audit" before any training was run: reusing `delta_d_w` for E would have confounded "is task-alignment useful" with "is a miscalibrated hinge threshold useful," undermining the H4 specificity comparison. Fixed via a dedicated calibration script verified against the actual frozen random vector `r` (not merely assumed from a matching seed).

**Cross-references**: Extends `PHASE_E22_COUNTERFACTUAL_OBJECTIVE_GEOMETRY.md`'s protocol (to be parameterized/reused, confirmed NOT usable unmodified without refactoring). Trains the implementation from `PHASE_E24_CALIBRATION_AND_IMPLEMENTATION_SPEC.md`. Tests the hypothesis from `PHASE_E23_ALGORITHMIC_REDESIGN.md`. References `PHASE_E18_FOLLOWUP_PRE_E19_REVIEW.md`'s data-order reproducibility caveat, explicitly carrying forward the same "asserted-by-construction, not independently re-verified" caveat rather than claiming stronger confidence here. Notes condition C's reuse of E18's `lambda_zero` config requires a byte-for-byte config match check, citing `PHASE_E13_CODE_AUDIT.md`'s precedent of catching an LR-schedule mismatch as the rigor standard to apply.

