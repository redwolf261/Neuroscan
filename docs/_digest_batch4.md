# Digest — Batch 4 of 4 (E70-E84 arc + synthesis documents)

---

## PHASE_E70_CAS_DESIGN_AND_PROTOCOL.md

**Summary**: Pre-registered design doc (written before results) for Correspondence-Aware Skip (CAS), a learned, bounded, gated deformable-offset correction at the enc1→dec1 skip junction (UNet3D_v10 vs baseline UNet3D_v3). Verified exact identity at init (max diff 5.96e-8) and bounded displacement (≤4.0 voxels), +0.52% param overhead. Design is justified by E65's causal decomposition (3-voxel translation caused 0.214 mean Dice drop vs 0.101/0.028/0.027 for other interventions) and E64's finding the effect lives entirely in the skip path. Declares comparison 2 (MM_CAS vs MM) as the only valid ablation for a CAS-specific claim, and pre-declares that single-seed results are exploratory only (≥3 seeds required per project's own multi-seed policy).

**Self-contradiction/hedging**: None — this is a pure pre-registration document, no results yet.

**Bug fix mentioned**: No.

**Cross-references**: Builds directly on E65 (causal decomposition) and E64 (split intervention, corrected E62/E63's attribution). Cites CCABA and E45/E54 multi-seed policy as precedent for requiring ≥3 seeds.

---

## PHASE_E70_RESULTS_SEED0.md

**Summary**: Seed-0 results for CAS. Headline: MM_CAS reaches 0.9200 pooled / 0.9052 per-subject Dice, clearing both the corrected (E56) and original pre-E56 pooled targets. The three pre-registered comparisons: MM vs FLAIR-only +1.81pp (CI excludes 0, not a contribution per protocol), MM_CAS vs MM (the CAS claim) +0.30pp (CI [+0.07,+0.53] excludes 0 but flagged as inside this project's own known between-seed noise band), MM_CAS vs FLAIR-only +2.11pp. A gate diagnostic (pre-registered) tested whether CAS's mechanism does what it claims: gate is non-trivial (mean 0.722, not collapsed) but **anti-correlates** with E65's causal sensitivity map (mean ρ=-0.388, 0/125 subjects positive, p=8.2e-114) and is concentrated in background, not lesion (paired Δ=-0.231, p=3.6e-134) — the opposite of the intended correspondence-correction behavior. A follow-up investigation found the gate actually tracks local activation magnitude/variance (ρ=-0.566/-0.555, 100% unanimous), and that this fully explains the anti-correlation (partial correlation controlling for magnitude shrinks -0.388 to -0.087, ~78% reduction) — i.e., CAS is really a magnitude-gated smoothing operator, not correspondence correction.

**Self-contradiction/hedging**: The document explicitly hedges its own headline result: "Comparison 2 is the CAS claim, and on this evidence it is suggestive but NOT confirmed" — despite the CI excluding zero and both significance tests passing, because the effect size (+0.30pp) sits inside the project's own demonstrated noise range (citing E45 and E54 as prior single-seed effects with stronger stats that did not replicate). The document's own summary table marks "CAS improves Dice" as "single seed, inside noise range, unreplicated" and "CAS works by correspondence correction" as "❌ falsified by the gate diagnostic" — i.e. it walks back its own title/headline before the reader even reaches the end.

**Bug fix mentioned**: No (this is a diagnostic/interpretive correction of an assumed mechanism, not a code bug).

**Cross-references**: Explicitly built on/tests predictions from PHASE_E70_CAS_DESIGN_AND_PROTOCOL.md's pre-registered rules. References E45 and E54 as prior cases of single-seed effects that later failed multi-seed confirmation (setting up the caution about CAS's own single-seed result).

---

## PHASE_E70_E71_OVERNIGHT_RESULTS.md

**Summary**: Overnight unattended pipeline results covering three items. (1) MM_A96 (multimodal at 96³): +2.19pp vs FLAIR-only baseline (CI excludes 0) but resolution's own marginal contribution over 64³ multimodal is NOT confirmed (+0.38pp, CI [-0.00,+0.79] includes 0). (2) CAS 3-seed confirmation: **DOES NOT REPLICATE** — CAS vs MM per seed is +0.30pp/+0.42pp(barely, p=0.068)/-0.01pp(p=0.98), 3-seed mean +0.24pp with std 0.22 (nearly as large as the mean); explicitly called "the third time this exact pattern has appeared" (after CCABA and E45/E54). Modality gain (MM vs FLAIR) IS confirmed on 3 seeds (+1.75pp, std 0.27) as the project's one solid reproducible number, though not novel. (3) CDCG prediction 1 passes clearly: Spearman(predicted, measured) = +0.701 held-out (permutation p<0.001), partial Spearman controlling for lesion size = +0.904 (p=4.1e-47) — stronger than the raw correlation.

**Self-contradiction/hedging**: None beyond what it inherits from E70 seed-0 (it explicitly reconfirms/finalizes the CAS-does-not-replicate verdict rather than contradicting itself). It also cautions that CDCG's own gating mechanism (Section 3.3) "should not be assumed to work just because prediction 1 passed."

**Bug fix mentioned**: No.

**Cross-references**: Directly finalizes/extends PHASE_E70_RESULTS_SEED0.md's CAS finding by adding seeds 1-2, and formally closes CAS ("should now be considered fully closed"). References CCABA and E45/E54 as the two prior instances of the same single-seed-then-fails pattern. Sets up PHASE_E71_CDCG_DESIGN.md's prediction 1 as passed and prediction 2 as the next still-to-be-designed step.

---

## PHASE_E71_CDCG_DESIGN.md

**Summary**: Design-only document (explicitly "no code written, no training run") for Causally-Distilled Capacity Gating (CDCG): train an auxiliary head to predict the network's own bottleneck-ablation-derived causal sensitivity d_i from its bottleneck representation, then (proposed) use that self-prediction at inference to gate a coarse/fine decoder mixing coefficient at the enc1 junction — no ablation ever run at test time. Distinguishes itself from CFKD-family counterfactual distillation (input-space perturbation) by being model-space (ablate the model, not the input). Pre-registers three falsifiable predictions in strict order: (1) self-prediction fidelity (kill gate if it fails), (2) mechanism sensibility check (gate should track E48's known size-dependence), (3) actual ≥1pp Dice performance check on ≥3 seeds — explicitly states 1 AND 2 must both hold for 3's result to mean what it appears to.

**Self-contradiction/hedging**: None — pure design document, describes itself as moderately-high but not certain confidence on novelty.

**Bug fix mentioned**: No.

**Cross-references**: Explicitly reuses E48's own `forward_with_bottleneck_ablation` construction "bit-for-bit... reused, not reimplemented." Deliberately avoids reusing CAS's deformable-offset machinery "to keep this a clean single-variable test... not conflated with CAS's (falsified) correspondence-correction idea" — directly referencing PHASE_E70 results as already falsified at the time this was written.

---

## PHASE_E71_PREDICTION2_RESULT.md

**Summary**: Two-part document. Original result (200 labeled subjects, 30 epochs): prediction 2 **FAILS** — predicted sensitivity d̂ᵢ correlates with lesion size in the *opposite* sign (+0.395) from E48's own established reference (-0.454) and the measured label itself (-0.353, which correctly reproduces E48). Diagnosed as likely an undertrained-head/data-scale artifact: predicted range severely compressed (0.43 vs measured 0.81) and the disagreement concentrates specifically at small lesions (predicted 0.255 vs true 0.475). An UPDATE section (placed at the top) reports the proposed scaled re-test (all 1126 training subjects, 150 epochs) **PASSES, but marginally**: Spearman jumps to +0.866, size-direction sign corrects to -0.190 (p=0.035, "correct sign... but noticeably weaker" than E48's -0.454 and the direct measured-vs-size -0.353), and partial correlation controlling for size stays strong (0.855). Explicitly labeled "the more decisive number" is the partial correlation, not the marginal size-sign match.

**Self-contradiction/hedging**: The document itself frames the eventual PASS as heavily hedged: "clears prediction 2 on this evidence, but 'clears' here means a real, direction-correct, partially-independent signal — not an emphatic confirmation." It also notes the largest-lesion bin "still slightly overshoots — a residual version of the original compression problem, much reduced but not fully gone," i.e., the original FAIL's underlying issue is not fully resolved even in the PASS.

**Bug fix mentioned**: No coding bug — this is a diagnosed data-scale/undertraining artifact, explicitly distinguished in the text from "the same kind of failure CAS had (a coherent, real, alternative mechanism)."

**Cross-references**: Directly tests/extends PHASE_E71_CDCG_DESIGN.md's prediction 2. Uses E48's own reference number (-0.454) as ground truth to check against. Sets up PHASE_E71_GATE_MECHANISM_KILLED.md as the next (and final) step in the CDCG line.

---

## PHASE_E71_GATE_MECHANISM_KILLED.md

**Summary**: Tests CDCG's gating mechanism's core premise directly (no training): does bottleneck-ablation sensitivity d_i predict tolerance to enc1-skip suppression, as the mechanism requires? Result: Spearman(d_i, suppression-tolerance slope) = **-0.290** (parametric p=1.0e-3, permutation p=0.001) — significant but in the **opposite** direction required (mechanism needs positive). Diagnosis: bottleneck-dependence and skip-dependence are correlated markers of overall subject difficulty, not substitutable/tradeable resources — subjects fragile to bottleneck ablation are also more fragile to skip suppression, so there's no compensatory relationship for a gate to exploit. Gate mechanism (CDCG Section 3.3) formally KILLED; prediction 1 (self-prediction of causal sensitivity, ρ=0.87) is explicitly stated to remain a valid, standing finding independent of this kill.

**Self-contradiction/hedging**: None — the verdict is stated cleanly and is not walked back; document explicitly frames this as "a good result, not just a null."

**Bug fix mentioned**: No.

**Cross-references**: Directly kills the mechanism designed in PHASE_E71_CDCG_DESIGN.md Section 3.3, independent of PHASE_E71_PREDICTION2_RESULT.md's earlier pass/fail. Explicitly separates what "survives" (prediction 1) from what's dead (the gating use of it) — this framing carries forward into E72.

---

## PHASE_E72_FWL_DESIGN.md

**Summary**: Fragility-Weighted Loss (FWL) — use the frozen, validated E71 aux-head prediction ĥat{d}_i as a per-subject loss weight during training, rather than for routing/gating (which E71 killed). Stage 1 sanity check passes (weighting isn't a disguised size proxy, ρ=-0.190 vs E48's -0.454 reference). Stage 2 pilot training, 12 epochs: **v1 (softmax T=1.0)** gives -0.75pp Dice, statistically significant (p=0.0001), but with a significant, correctly-signed mechanism-consistency check (Spearman(ĥat{d}_i, delta)=+0.21, p=0.019) — i.e., it does help fragile subjects as designed, but at too high a cost to easy subjects. **v2 (gentler linear floor=0.7)**: -0.10pp, not significant (p=0.53), but the mechanism-consistency check also disappears (ρ=+0.07, p=0.45) — gentler weighting loses the intended targeting entirely. Both weightings tested; project's "one re-test" discipline invoked; **FWL formally KILLED**, no further weighting variants to be attempted.

**Self-contradiction/hedging**: None — verdict is stated and held consistently through the document; it explicitly frames the finding ("cost and mechanism-targeting shrink together, not independently") as reinforcing rather than undercutting the kill decision.

**Bug fix mentioned**: No.

**Cross-references**: Directly follows from E70 (CAS) and E71 (CDCG gating) both being killed the same night; treats FWL's failure as evidence reinforcing E71's gate-sweep finding (difficulty is a shared, non-partitionable subject property) via "a related diagnosed reason." Explicitly distinguishes its own failure mode from CAS's (CAS had a coherent alternative mechanism; FWL's failure is a real cost/benefit tradeoff).

---

## PHASE_E73_SDLR_RESULT.md

**Summary**: Self-Diagnostic Localized Refinement (SDLR) — reframes the fragility signal from a scalar (E70-E72, all killed) to a spatial map: can the bottleneck predict WHERE it's causally vulnerable? Stage 1 cheap probe PASSES: real causal sensitivity map S correlates strongly with the network's own errors (22.6x enrichment ratio, p<1e-9), survives boundary-distance confound control (partial ρ=+0.88). Stage 3 trains a small predictor head: pooled ρ=0.394-0.414 held-out across two data scales, both pass their design-time bar but downstream error-localizing power collapses sharply when distilled (error/correct ratio drops from real map's 22.6x to only 1.07x-1.19x for the predictor, and scaling up data barely helps, unlike E71's prediction 2) — described as "a genuine ceiling," not a data-scale artifact. Stage 4 refinement pilot (UNet3D_v11, verified bit-identical to v3 at init): **KILLED**, mean delta +0.0005 (not significant, p=0.78), and the gate doesn't even preferentially help hard subjects (ρ=-0.056, p=0.53, essentially inert).

**Self-contradiction/hedging**: None — stages are reported honestly as they progressively weaken (probe strong → predictor weak → refinement dead), consistent with the document's own framing.

**Bug fix mentioned**: No.

**Cross-references**: Explicitly frames itself as the third of three mechanistic verbs (routing E71, reweighting E72, spatial gating E73) all applied to the same underlying causal signal and all failing "for a specific, diagnosed reason." Deliberately structured to avoid E71's routing mistake and E72's subject-level reweighting mistake.

---

## PHASE_E74_SPATIAL_DEPENDENCE_AUDIT.md

**Summary**: Pure diagnostic (no training) reframing after E71-E73's kills: measure what representation PROPERTY makes a tensor translation-sensitive, before designing any fix. Uses E65's translation intervention and E64's split-forward design, now on the MM-seed0 checkpoint, tested at two loci: enc1 (skip, E65's original locus) and bottleneck (new). Translation-sensitivity generalizes across depth (enc1 drop 0.317, p=2.0e-64; bottleneck drop 0.211, p=1.4e-23), arguing against a skip-specific fix. Feature magnitude is the consistent positive predictor at both loci (partial ρ=+0.301 at enc1, +0.525 at bottleneck, controlling for size), but high-frequency spatial content does NOT predict sensitivity as hypothesized — null at enc1 (ρ=0.054, n.s.), and **reverses** at bottleneck (ρ=-0.589, p=4.8e-13, smoother regions more fragile). Explicitly states: "the simple 'high-frequency → coordinate-dependent → fragile' hypothesis is not supported... a genuine update, not a confirmation of the pre-registered hypothesis."

**Self-contradiction/hedging**: The document is itself upfront that its own pre-registered hypothesis was not confirmed and reports this as a "characterization... not a pass/fail result" rather than trying to rescue the original framing.

**Bug fix mentioned**: No.

**Cross-references**: Reuses E65's translation intervention and E64's split-forward isolation design (explicitly, not reimplemented). Notes E65's own smoothing arm already showed limited effect, reinforcing this phase's conclusion against a frequency-based fix. Feeds directly into E75 (causality test of the magnitude finding).

---

## PHASE_E75_MAGNITUDE_CAUSALITY_RESULT.md

**Summary**: Causal test (no training) of E74's magnitude correlation, decomposing z(p)=r(p)·u(p) and rescaling ONLY magnitude (direction held exactly fixed, verified numerically) across alpha∈{0.25,0.5,1.0,2.0}. Pre-declared rule requires monotonic, significant S(alpha) increase at BOTH loci. **Enc1: PASS cleanly** — monotonic (0.204→0.242→0.317→0.372), all pairwise comparisons significant (p as low as 5.0e-49) — genuine causal confirmation. **Bottleneck: FAIL, reverses** — not monotonic (0.213→0.247→0.211→0.125), S(2.0) significantly LOWER than S(1.0) (p=1.7e-15), opposite of enc1's result. A confound is flagged: rescaling magnitude alone (even without translation) damages the network substantially at high alpha (e.g. enc1 intact Dice 0.902→0.867 at alpha=2.0), meaning the network is already out-of-distribution at the extremes — noted as present at both loci but not preventing enc1's clean result. **Overall verdict per the strict pre-declared both-loci rule: OVERALL FAIL.**

**Self-contradiction/hedging**: The document is explicit and disciplined about this being a genuine mixed/split result rather than picking a convenient reading — it puts two "honest readings" to the user rather than resolving unilaterally, and states the overall verdict is FAIL per the pre-declared rule even though enc1 alone passed cleanly.

**Bug fix mentioned**: No.

**Cross-references**: Directly tests E74's correlational finding causally. Its enc1-only "PASS" and bottleneck "FAIL/reversal" set up E76 (which re-tests enc1 specifically at a tighter, non-OOD alpha range to address the confound flagged here).

---

## PHASE_E76_ENC1_OPERATING_REGIME_RESULT.md

**Summary**: Follow-up to E75's split result, addressing the flagged OOD confound at wide alpha ranges by testing a tight neighborhood around the network's real operating point (alpha ∈ {0.70...1.30}), scoped to enc1 only per E75's split (bottleneck already killed). Pre-declared PASS requires (a) D_intact stability within 0.02 across the whole range, and (b) monotonic S(alpha) with significant positive local slope at alpha=1. **Both conditions PASS**: max D_intact deviation 0.0101 (well under 0.02 threshold); S(alpha) strictly monotonic across all 7 points, local slope S(1.10)-S(0.90)=+0.0211, p=7.4e-42. Verdict: the causal chain enc1 magnitude → translation sensitivity → segmentation vulnerability is confirmed as a genuine LOCAL property at the real operating point, not an artifact of E75's wide-range OOD test.

**Self-contradiction/hedging**: None — clean pass on both pre-declared conditions.

**Bug fix mentioned**: No.

**Cross-references**: Directly addresses the confound flagged in PHASE_E75_MAGNITUDE_CAUSALITY_RESULT.md. Confirms the precondition the user set before intervention design ("if E76 is clean, then — and only then — design the enc1 intervention"), explicitly deferring the intervention formula to a later phase and stating it should NOT be a spatial gate (referencing E73's kill).

---

## PHASE_E77_MAGNITUDE_GEOMETRY_RESULT.md

**Summary**: Measures the absolute geometry of enc1's natural magnitude distribution across 125 subjects (no training). Distribution is sharply bimodal (p10=p25=p50=1.389, jumping to p90=6.158, p99=10.076) rather than smoothly tailed. High magnitude essentially IS lesion tissue (background mean r=2.350, boundary 8.084, interior 10.906 — clean monotonic gap). Pooled voxel-level correlation with local translation impact is strong (ρ=+0.539, p≈0, n=1.25M) but subject-level correlation (mean r vs subject-level translation drop) is **NOT significant** (ρ=+0.132, p=0.14) — flagged explicitly as a nuance the automated classifier missed. The top 10% magnitude tail is 15.5x more translation-sensitive than the rest, but that tail substantially overlaps with lesion tissue itself. The document explicitly **corrects its own auto-generated classification**: the script's automatic threshold rule labeled this outcome "A" (a rare, separable sensitive tail — promising for tail suppression), but the human-written text overrides this as "A and B simultaneously, not clean A," warning that naive magnitude suppression in the tail would destroy tumor signal itself.

**Self-contradiction/hedging**: Explicit and direct — the document states "That classification is incomplete and is corrected here" regarding its own script's automatic "outcome A" label, and the summary JSON's `outcome` field is noted as "superseded by this document's corrected A/B-overlap reading; kept unedited as a faithful record of the script's literal threshold output." This is one of the clearest self-flagged verdict-vs-evidence mismatches in the batch.

**Bug fix mentioned**: Not a code bug per se, but an incorrect/incomplete automatic interpretation logic that was caught and manually corrected before being trusted.

**Cross-references**: Builds on E76's clean pass. Sets up E78 by reframing the intervention target away from "control magnitude" (unsafe per this phase's finding) toward "control positional precision of high-magnitude features."

---

## PHASE_E78_MAGNITUDE_DIRECTION_DECOMPOSITION.md

**Summary**: Causal decomposition (no training) of z(p)=r(p)·u(p) at enc1 into magnitude-only-translated vs direction-only-translated hybrid tensors, to determine which channel carries the positional information translation disrupts. Verified exact decomposition/recomposition (2.4e-07 diff) and that translation commutes with the r/u split. Results: S_full=0.3174 (100%), **S_r (magnitude-only) = 0.0223 (~7%)**, **S_u (direction-only) = 0.2134 (~67%)** — direction-only is ~9.6x larger than magnitude-only (p=2.2e-55). Additivity check shows super-additive interaction: S_r+S_u=0.236 vs S_full=0.317, a real +0.082 gap. Conclusion: **direction carries the positional structure; magnitude amplifies it** (not an independent carrier). Motivates preserving magnitude (lesion salience) while making direction more translation-robust specifically.

**Self-contradiction/hedging**: None — clean, decisive result consistent throughout.

**Bug fix mentioned**: No.

**Cross-references**: Directly resolves the ambiguity left by E75/E76 (which only established magnitude rescaling changes sensitivity, not that magnitude itself carries position). Builds on E77's finding that magnitude suppression is unsafe. Presents a full "causal chain, complete" summary spanning E65, E73-E78.

---

## PHASE_E79_DIRECTIONAL_AUDIT_RESULT.md

**Summary**: Tests two competing hypotheses for WHY direction is vulnerable (from E78): A) locally noisy/oversharp (fixable by smoothing) vs B) bound to absolute spatial coordinates (not fixable by smoothing). No training. Direction found to be already locally smooth/coherent (cos(theta) 0.88-0.95 across classes) — little local noise to remove. Smoothing direction before translating it recovers only 11.0% of the damage (S_u_after_smooth=0.1899 vs raw S_u=0.2134). Local direction permutation (2x2x2 scramble, position preserved) causes only 17.4% as much damage as translation (0.0372 vs 0.2134) — the decisive result. **Verdict: Hypothesis B confirmed** — the decoder uses enc1's direction essentially as a positional lookup key bound to absolute coordinates, not local directional texture. Rules out local-smoothing/Lipschitz interventions; motivates registration/correspondence-family mechanisms instead, "closer in spirit to E68's original SC-DCU correspondence-correction idea (retracted then for an unrelated design flaw)."

**Self-contradiction/hedging**: None — clean confirmation of one of the two pre-declared hypotheses.

**Bug fix mentioned**: No.

**Cross-references**: Explicitly references E68's SC-DCU idea as retracted for an unrelated reason, distinguishing this new motivation from that old (different) failure. Presents an updated 8-phase "causal chain, complete" (E65, E73-E79). Directly sets up E80 (decoder-readout / correspondence test), with the phase stating the novelty search comes next before any training code.

---

## PHASE_E80_DECODER_READOUT_RESULT.md

**Summary**: Tests whether E79's absolute-coordinate binding is an encoder-only property or an encoder-decoder correspondence property, using 4 arms of split-forward translation: Arm 1 (encoder-only, =E78/E79's S_u=0.2134), Arm 2 (decoder-only upconv1, S=0.2615), Arm 3 (coherent joint shift, both same offset, S=0.4386 — WORSE than Arm 1, not better), and Arm 4 (mismatch sweep, monotonic in upconv1's own absolute offset, not troughing at zero relative mismatch as the correspondence hypothesis predicted). The document itself explicitly catches and corrects an **auto-generated misread**: the script's automatic reading function fell through to hypothesis (i) ("encoder representation property") by default because it had no branch for a negative recovery fraction — the document states plainly "That auto-generated conclusion is WRONG and is corrected here." The corrected reading concludes BOTH encoder (enc1 direction) and decoder (upconv1) paths are independently, additively position-bound (Arm1+Arm2=0.475 ≈ Arm3=0.439), ruling out correspondence/alignment mechanisms as a fix.

**Self-contradiction/hedging — CRITICAL, per explicit task focus**: This document carries its own **RETRACTION NOTICE at the very top**, added after E81. It states: "E81... found that E80's Arm 3... result below is confounded: coherently shifting both sides by the same offset causes the network's OUTPUT to also shift by approximately the same offset (near-perfect translation-equivariance, verified directly: relative equivariance error 0.096). Undoing that shift... recovers 93.5% of Arm 3's apparent damage. This retracts this document's 'both sides independently, additively position-bound' conclusion." The notice further specifies precisely what SURVIVES: Arm 1 (encoder-only, S=0.213, reproduces E65/E78/E79) and Arm 2 (decoder-only, S=0.262) are NOT subject to the confound (asymmetric shifts give the network no compensating output-shift option) and remain standing findings. Only the joint-shift (Arm 3) interpretation and the mismatch-sweep shape (driven by the same confound) are retracted. The rest of the original document is "kept unedited below as a faithful record of what was concluded before E81's correction."

**Bug fix mentioned**: Yes — two: (1) the auto-generated reading-function bug described above (no branch for negative recovery fraction), corrected manually within the same document before E81 even ran; (2) the deeper conceptual confound (coordinate-frame/output-equivariance artifact) that E81 subsequently caught and that necessitated the retraction notice.

**Cross-references**: This document is explicitly superseded/partially retracted by PHASE_E81_REALIGNMENT_CONTROL_RESULT.md — the retraction notice names that file directly. Also references E67/E67b (novelty search ruling out generic deformable/alignment mechanisms) preceding this phase.

---

## PHASE_E81_REALIGNMENT_CONTROL_RESULT.md

**Summary**: Directly tests the confound the user flagged in E80: does coherently shifting both sides of the skip concatenation simply produce a globally-displaced output, making Arm 3's large Dice drop a trivial coordinate-frame artifact rather than evidence of independent absolute binding? Reuses E80's own `forward_dual_shift` exactly. Realignment check: undoing the same shift on the prediction before scoring recovers D from 0.4637 (E80's Arm 3, reproduced) to 0.8737 — **93.5% recovery** of S (0.439→0.029), p=3.0e-84. Independent feature-level equivariance check confirms: relative equivariance error 0.096, meaning the coherently-shifted output is, within ~10%, exactly what a pure translation of the clean output would look like (near-equivariant). **Verdict: Outcome A — E80's Arm 3 interpretation is retracted.**

**What E81 precisely retracts from E80** (per explicit task focus): E80's conclusion that "both encoder and decoder features are INDEPENDENTLY, additively coordinate-bound" (drawn from Arm 3 and the mismatch sweep) is retracted — that damage was substantially a coordinate-frame displacement artifact, not evidence of independent absolute-coordinate keys on both sides.

**What survives from E80, per E81's own text**: (1) E65/E78/E79's own results (enc1 direction translated alone, asymmetric, no coordinate-frame escape hatch available) are explicitly stated to be untouched by this confound and remain standing. (2) E80's Arm 2 (decoder-only shift, S=0.262) is explicitly stated to also be asymmetric and therefore not subject to the confound — "This remains a genuine, still-standing finding: the decoder's own upconv1 pathway is independently sensitive to positional disruption." The corrected standing picture: enc1's direction (E65/E78/E79) and upconv1 (E80 Arm 2) are each independently sensitive to being moved RELATIVE TO THEIR PARTNER — not the stronger claim of encoding an absolute global coordinate. The document even reframes the near-equivariance finding as "actually more consistent with a correspondence-based mechanism than the retracted 'independent absolute binding' reading was" — while also noting this creates "a genuine tension to sit with" since direct correspondence/alignment fixes were already flagged as occupied territory by the earlier novelty search.

**Self-contradiction/hedging**: The document is itself the retraction of another phase, and is transparent and precise about scope — it does not overclaim beyond what the two checks show, and explicitly flags the resulting interpretive tension rather than resolving it conveniently.

**Bug fix mentioned**: No new bug in this document (it is diagnosing/resolving E80's confound, described above).

**Cross-references**: Explicitly retracts part of PHASE_E80_DECODER_READOUT_RESULT.md (named directly). Produces an updated "causal chain, corrected" table spanning E65-E81, marking each item STANDS/RETRACTED explicitly.

---

## PHASE_E82_LOCAL_DONOR_IDENTITY.md

**Summary**: Naming note: user called this "E81" but that name was taken, renumbered E82 for ledger consistency. Tests, following E81, whether correspondence-breaking damage is isotropic/distance-only or anisotropic/direction-specific, and whether nearby donor directions are interchangeable. Part 1 (directional/radial sweep): anisotropic — axis range 0.090 vs mean 0.155 (z-axis hurts less than x-axis at same magnitude); magnitude sweep confirms smooth distance-scaling (0.075 at mag=1 to 0.239 at mag=5). Part 2 (local donor swap): **a real measurement bug was caught and fixed mid-analysis** — the initial whole-volume-Dice metric returned near-zero at every radius because only ~10 of 262,144 voxels (0.004%) were perturbed, far too small a fraction for Dice to register; corrected to measure local probability change at swap sites directly. **A second correction** caught the corrected script's own auto-classifier mislabeling radius=1 as "near-zero"/"local equivalence" against an arbitrary 0.05 threshold — checked directly via paired t-tests and found the radius trend is real and significant, not flat (r1=0.0274, r2=0.0412, r3=0.0450, all pairwise significant, ratio r1/r3=0.61). Verdict: neither clean hypothesis (full local equivalence nor sharp address-identity) — a real, moderate, distance-graded, anisotropic penalty.

**Self-contradiction/hedging**: The document explicitly flags that its own underlying JSON summary's `local_reading` field says "LOCAL EQUIVALENCE," calling this "INCORRECT per an arbitrary threshold, superseded by this document's corrected reading based on direct pairwise significance tests; kept unedited as a faithful record of the script's literal, flawed threshold logic." This is a second clear instance (after E77 and E80) of an auto-generated verdict being caught and overridden within the same document.

**Bug fix mentioned**: Yes, explicitly two: (1) the Dice-insensitivity measurement bug (wrong metric for a sparse intervention), fixed by switching to local probability-change measurement; (2) the auto-classifier's arbitrary/flawed threshold mislabeling the radius=1 result, corrected via direct significance testing.

**Cross-references**: Follows directly from E81's finding that relative encoder-decoder displacement (not global shift) is the real failure mode. Produces an updated causal-chain summary (E65→E82). Sets up E83 (anisotropy confound test).

---

## PHASE_E83_CORRESPONDENCE_KERNEL_RESULT.md

**Summary**: The user flagged, before interpreting E82's anisotropy finding, that BraTS native volumes (240×240×155, 1mm isotropic) are resized to a 64³ cube non-uniformly (x/y zoom 3.750mm/voxel, z zoom 2.422mm/voxel — verified against a real NIfTI header), so a fixed VOXEL offset is NOT a fixed physical distance across axes. Re-tests E82's anisotropy under physical-distance matching. (1) Voxel-matched (reproduces E82): real, significant (effect size 0.0191, p=1.8e-3). (2) **Physical-distance-matched** (x at 2 voxels/7.50mm vs z at 3 voxels/7.27mm): effect **shrinks ~48%** to 0.0100 and **loses significance** (p=0.142). (3) Reflection-symmetry check: unplanned finding — only z-axis shows significant direction asymmetry (p=0.041), not predicted by either hypothesis, explicitly flagged as unpursued this session. **Verdict: AMBIGUOUS** — the resampling-geometry confound explains roughly half the original anisotropy effect, but the residual (0.0100) is itself underpowered to confirm or rule out (p=0.14). E82's "anisotropic learned representation" claim as stated "cannot be confirmed."

**Self-contradiction/hedging**: The document is explicit that this is neither a clean confirmation nor a clean refutation, and states directly that E82's headline finding "was substantially explained by ordinary preprocessing geometry, not a novel representational phenomenon" — a direct partial walk-back of E82's own anisotropy claim, though the document is careful to say the isotropic, distance-graded core of E82 (radius 1→2→3 within a single axis) is unaffected and still stands.

**Bug fix mentioned**: No code bug — this is a discovered measurement confound (voxel-space vs physical-space anisotropy from non-uniform resampling), not a bug per se, though functionally similar in effect (caught before trusting a conclusion).

**Cross-references**: Directly qualifies/partially retracts PHASE_E82_LOCAL_DONOR_IDENTITY.md's cross-axis anisotropy claim (explicitly: "E82's 'anisotropic learned representation' claim cannot be confirmed as stated"), while explicitly preserving E82's single-axis (dim2-only) radius-graded finding as unaffected. Also reaffirms E65, E78, E79, E81 as still standing.

---

## PHASE_E84_CONTRIBUTION_AUDIT.md

**Summary**: A no-experiment research-design decision phase, freezing experimentation at E83. Reviews the surviving E48-E83 evidence table and the full list of failed algorithmic branches (DCU/generic deformable — occupied; SC-DCU — not justified/E68; CAS — killed/E70; CDCG gate — killed/E71; FWL — killed/E72; SDLR — killed/E73; E54/A96 — unconfirmed). Frames two standing positive results: (1) 4-modality input gain, +1.75pp 3-seed confirmed, real but not novel; (2) bottleneck causal self-knowledge (ρ≈0.87 held-out), a real validated scientific phenomenon but never successfully turned into an algorithm. Poses and answers three questions: existing literature already covers "correspondence repair" generically (deformable/attention/spatial-transformer families); this project's causal chain adds real, non-obvious CAUSAL SHAPE characterization (not magnitude, not local noise, not independent per-side binding, not primarily axis-structured, but smoothly graded with distance) that literature search alone wouldn't produce; but Question 3 — is there a genuinely missing operation the evidence demands — is answered **NO**: nothing in the chain identifies a functional form outside what existing occupied deformable/attention mechanisms could already express. **Verdict: the algorithm hunt on this line is stopped, not because evidence is weak but because it doesn't license a new architectural operation.** No E85 is scheduled.

**Self-contradiction/hedging**: None in the sense of walking back its own verdict — but the document is explicit that this is a deliberate stop, not a failure, and it plainly lists the multiple real bugs its own preceding chain caught (see below) as a methodological throughline/contribution in its own right.

**Bug fix mentioned**: Yes — this document explicitly enumerates, as a retrospective list, the bugs caught across the whole E48-E83 chain: "E62/E63's shared-tensor bug, E80's coordinate-frame confound, E82's Dice-insensitivity bug, E83's physical-spacing confound."

**Cross-references**: Synthesizes and closes out the entire E48-E83 chain; explicitly references and tabulates outcomes from E48, E65, E67, E67b, E68, E70 (CAS), E71 (CDCG), E72 (FWL), E73 (SDLR), E74-E76, E77, E78, E79, E80 (corrected by E81), E81, E82, E83 — the most cross-referencing document in this batch by far.

---

## PHASE_EVIDENCE_MAP.md

**Summary**: A synthesis/reference document (covering E1-E58, dated before the E70-E84 arc) distinguishing what's proven (causal diagnostics, measurement corrections) from what's been tried-and-failed (working algorithms). States the precise, corrected summary claim about the bottleneck: causally important, size-dependent (E48 ρ=-0.454, p<0.001), but the mechanism is unresolved — explicitly warns "do not use 'global context,' it overreaches the evidence." Documents E43→E47 routing hypothesis as CLOSED/null (E47: p=0.808, no differential boundary effect). Documents E48→E58 bottleneck hypothesis: causal effect real, but spatial localization (E58 Stage 1/1b) and cross-subject "compatibility" (E58 Stage 2/2b/3) were tested and shown NOT to be the explanation — Stage 3 shows the donor-substitution effect matches a known published measurement confound (OOD ablation, Li & Janson 2024), not a genuine finding. Documents three measurement corrections (pooled vs per-subject Dice gap of 0.85-2.4pp; baseline anchor variance across 4 seeds; single-seed unreliability established by CCABA's 3-seed check). Part 3 gives a full mechanism-attempt audit table E44-E55. Part 4 flags E45 (D4+D8) and E54 (A96) as the only two mechanisms with doubly-significant per-subject results, neither yet confirmed on 3 seeds at time of writing — named as the one genuinely open, low-cost next step.

**Self-contradiction/hedging**: The document itself flags an internal correction it made: an "E58 Stage 2/2b" donor-substitution effect initially "looked like a real 'wrong context is misleading' effect" but Stage 3's mean-ablation confound check showed it "matches the signature of a known, already-published measurement confound... not a genuine semantic compatibility finding" — the document walks back its own earlier-stated interpretation within the same table.

**Bug fix mentioned**: No code bug described here, but a measurement-confound diagnosis (OOD-ablation artifact) is treated with the same rigor.

**Cross-references (headline conclusions per explicit task focus)**: Explicitly lists killed/closed mechanisms: E43→E47 routing (null), E58 spatial-locality and cross-subject-compatibility hypotheses (both closed/ruled out). States "what's still open": why the bottleneck's causal, size-dependent effect exists — "No architecture tried has actually targeted this correctly, because no diagnostic has identified what 'correctly' would mean." Explicitly rules out guessing a new "E59" architecture without first resolving the open why-question. Full mechanism audit table (Part 3) covers E44 (RCGW, killed), E45 (D4+D8, real but unconfirmed on 3 seeds), E46 (attention gate, mixed significance), E47 (causal null), E48 (causal finding), E49 (CCABA, 3-seed mean tied with D4-only, established mandatory 3-seed policy), E50 (IECG, worse than CCABA on all 3 seeds), E51 (CCAG, didn't compose additively), E52 (ASR, killed at smoke test), E53 (incomplete/no valid conclusion), E54 (A96, real but unconfirmed on 3 seeds), E55 (dual-resolution crop, below baseline despite passing all verification checks).

---

## PHASE_LESSONS_WHAT_NOT_TO_DO.md

**Summary**: A self-contained 22-item checklist of mistakes/traps this project has fallen into and corrected, grouped into four categories: Statistical/analysis traps (7 items — e.g., don't trust R²/correlation without checking shape of relationship, don't trust effect sizes without outlier checks, don't skip randomization/permutation controls especially with noisy denominators, don't confuse pooled vs per-subject Dice), Experimental design traps (6 items — e.g., don't soften a pre-agreed strict numeric criterion after seeing results, don't assume a derived quantity differs from size without checking directly, don't declare training instability a finding without ruling out mundane causes like BatchNorm-at-batch-1, don't calibrate loss weight by value alone without checking gradient magnitude, don't skip literature/novelty checks before big compute spends), Engineering/performance traps (5 items — e.g., don't assume you found the bottleneck without re-measuring/profiling, don't trust unsynchronized GPU wall-clock timing, don't rewrite core computation without numerically verifying equivalence, don't write mismatched synthetic test data, don't let caching silently fall back to defaults), and Reporting/process discipline (4 items — e.g., don't exempt diagnostic/analysis code from the same skepticism as training code, don't report a collapsed pre-registered gate as a finding without first checking for a bug, don't keep tuning a null result hoping it flips, don't write reports assuming the reader has repo access).

**Self-contradiction/hedging**: N/A — this document IS the project's own explicit meta-lesson list; it does not carry a verdict of its own to contradict, but item 19 explicitly names "the degradation-trajectory artifact" and "the sign-convention bug" as examples of exactly the auto-generated/premature-verdict problem the whole digest task is checking for.

**Bug fix mentioned**: Yes, multiple, described generically without phase numbers: a degradation-trajectory mathematical artifact (caught via randomization test), a sign-convention bug in diagnostic scripts, a BatchNorm-at-batch-1 instability misattributed as a real finding, a resize-operation misdiagnosis of a slowdown (real cause was unbatched operations), a mismatched synthetic test, a silently-failing cache fallback.

**Cross-references**: Does not name specific phase documents (explicitly designed to be self-contained, "no other file needs to be read"), but items generically describe patterns that recur across E67/E68 (loss-weight calibration/gradient magnitude — item 11, matching PHASE_RESEARCH_ARC_MASTER_REPORT's Section 14 collapse), the degradation-trajectory idea (Section 11 of the master report), and the pooled-vs-per-subject Dice distinction (recurring throughout E44-E58).

---

## PHASE_RESEARCH_ARC_MASTER_REPORT.md

**Summary**: Self-contained narrative history (dated 2026-08-15) of the earlier E1-E27-ish arc, describing the task (BraTS 2023 GLI whole-tumor segmentation, FLAIR-only, 64³ resize), baseline (pooled Dice 0.9063, "condition A"), and every idea tried in sequence: Idea #1 ABO (adaptive boundary optimization) — null, closed after hyperparameter sweep ruled out tuning as the explanation. Idea #2 EGGO-M margin loss — mechanism verified active/correctly-signed via an extensive 8+-step diagnostic chain (gradient-conflict, decoder-sensitivity, margin-reachability, target-stability, freezing ablations, mathematical/gradient audits) but Dice never improved; explicitly corrects itself mid-narrative when a "promising lead" (freezing the decoder reduces representation "rotation") is reviewed further and found to actually have the LOWEST Dice of all conditions tested — "rotation is associated with the problem, but is not causing it." Idea #3 SC-TAM — also null, but a major sign-convention bug was caught (JVP consistency check revealed diagnostic scripts had been backwards, corrected: 98% vs previously-computed 2% correction accuracy) before accepting a "strange finding" at face value; confidence-gated version passed its strict acceptance criterion (92% signal retained, 99% damage removed) but STILL didn't improve Dice. Idea #4 Deep Supervision — the project's first and only real positive result (+0.28-0.33pp, best 0.9096 with D4-only), with a full mechanism audit showing the true mechanism is improved completion quality on already-detected small tumors, not improved detection rate — but explicitly caveats that none of the six pairwise comparisons among conditions reached significance at n=125. Two follow-up "does X explain it" searches both closed as honest nulls (one found a real R²=0.51 baseline-explains-half finding; one initially found an apparent "edema subregion" effect that collapsed to near-zero once tumor size was controlled for — explicitly flagged as a confound, not a real finding). A full project audit revealed no augmentation ever used, no cropping/patch-training, only one loss function type ever tried, and validation-set reuse across all model-selection decisions. Tests the "resolution" hypothesis directly: 64³ Dice 0.9038, 96³ worse (0.8986), 128³ invalidated (BatchNorm breaks at batch size 1, erratic 0.53-0.75 range, explicitly recognized as an artifact and not reported as a real finding). The "Degradation-Trajectory Constraint" idea is designed, hits two real bugs (a linear-regression-blind-to-binary-effect bug, and a near-zero-denominator blowup bug), and even after both fixes, its final "positive" result is proven to be a pure mathematical artifact via a randomization test (200 shuffle trials beat the real result 96.5% of the time). A narrower "critical resolution" (α_c) idea is designed to avoid that exact flaw and survives every falsification check (partial correlation beyond size, 500/500 shuffle trials beaten, bootstrap CI excludes zero) — the most rigorously validated positive finding in this arc, though explicitly noted as "only a validated raw measurement — not yet a proven training algorithm." A literature check finds no exact prior-art match. The decisive training experiment (conditions A/S/R) is designed carefully but both new conditions (S, R) collapse dramatically (Dice 0.7360, 0.7108 vs baseline 0.9038) — investigated rather than accepted, and traced to a real bug: the new loss term's gradient was ~75x larger than intended on the very first batch (calibrated by value, not by gradient magnitude), destabilizing training. This result is explicitly marked "invalid and not usable as evidence... at the time of writing" — fix proposed but not yet re-tested.

**Self-contradiction/hedging**: This document contains one of the most explicit self-corrections in the whole batch: Idea #2's "decoder freezing reduces rotation" lead is described as initially "treated as a promising lead" and then explicitly reversed: "a careful follow-up review found the opposite of what was hoped... This was explicitly corrected in the record." Also explicit about Idea #4's headline number not being statistically bulletproof despite being the project's best result, and about the α_c training experiment's result being currently "invalid... not usable as evidence for or against" either hypothesis.

**Bug fix mentioned**: Yes, multiple and explicit: (1) sign-convention bug in SC-TAM diagnostic scripts (caught via JVP check, fixed, all affected analyses rerun); (2) linear-regression-blind-to-binary-relationship bug in the degradation-trajectory analysis; (3) near-zero-denominator blowup bug in the same analysis, fixed by exclusion; (4) slow loss-weighting code traced via profiling to unbatched per-tumor operations, rewritten and verified numerically equivalent; (5) the ~75x gradient-magnitude miscalibration bug in the A/S/R training collapse.

**Cross-references**: This document is itself described in docs/README.md as covering "the earlier E1-E27 arc narratively" — i.e., it predates and is distinct from (not contradicting) the E44+ arc covered by PHASE_EVIDENCE_MAP.md and the E70-E84 files above. Its final "where the project stands" section states Deep Supervision (+0.28-0.33pp) is the only positive result so far and α_c is a validated-but-unexploited measurement — setting up the later pivot documented in PHASE_EVIDENCE_MAP.md and beyond.

---

## PHASE_README.md (docs/README.md)

**Summary**: A short index/navigation file, not a phase report. States this `docs/` folder covers the "current, active" research line (BraTS 2023 GLI, FLAIR-only, 3D U-Net family, phases E1-E58+), distinct from the top-level README describing an earlier/different project iteration (2.5D HybridMiniSwin MS-lesion webapp). Points to `phases/` (all PHASE_*.md reports, naming PHASE_POST_E43_SUMMARY.md as the best entry point for the post-pivot arc and PHASE_RESEARCH_ARC_MASTER_REPORT.md for the earlier E1-E27 arc) and `setup/` (environment/dataset docs). Notes architecture files (neuroscan_3d_v1..v9.py) stay at project root by convention (imported by bare module name), and that the authoritative running index of findings lives in Claude's own project memory (MEMORY.md), not in this repo.

**Self-contradiction/hedging**: None — pure navigation stub, no verdicts.

**Bug fix mentioned**: No.

**Cross-references**: Names PHASE_POST_E43_SUMMARY.md and PHASE_RESEARCH_ARC_MASTER_REPORT.md as entry points; distinguishes the current docs/ folder from an unrelated top-level project iteration.
