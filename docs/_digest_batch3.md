# Digest Batch 3 — Post-Pivot Arc E44–E69 (plus synthesis docs)

Compact per-file digest for the post-E43 pivot arc. Each entry: summary, self-contradiction/hedge check, bug notes, cross-references to other phase docs (as stated by the document itself).

---

## PHASE_POST_E43_SUMMARY.md

**Summary**: Self-contained synthesis of the entire E44–E53 arc. Documents the causal-diagnostic chain (E44–E48) and mechanisms built on E48's finding (E49–E51), plus an objective-level detour (E52–E53). Seven mechanisms fully evaluated/closed (E44, E45, E46, E49, E50, E51, E52); E53 incomplete (stopped by user before 3-seed evaluation finished). None cleared the +1.0pp Dice bar against canonical baseline 0.9063. Table summarizes all results: E45 +0.47pp (1 seed), E46 +0.39pp (1 seed), E49/CCABA +0.31pp (3-seed mean), E50/IECG +0.02pp (3-seed mean), E51/CCAG +0.32pp (3-seed mean), E52 killed, E53 0.8969 (1 of 3 seeds, incomplete, below baseline).

**Self-contradiction/hedging**: None — this is itself a synthesis/summary document reporting other phases' verdicts faithfully, including E53's own incompleteness ("should not be cited as a Dice number for the mechanism").

**Bugs**: References E45's gradient-magnitude recalibration (2.18× mismatch caught) and E49's (25.4× blowup caught), but these are reported in the underlying phase docs, not discovered here.

**Cross-references**: Explicitly builds on/summarizes E44, E45, E46, E47, E48, E49 (+ its multiseed variance finding), E50, E51, E52, E53. States the recommendation "made after E51 and reaffirmed after E52" to stop mechanism-hunting and write up the causal-diagnostic methodology instead.

---

## PHASE_E44_E45_POST_PIVOT_ALGORITHMIC_ATTEMPTS.md

**Summary**: Records E44 (RCGW — Relative-Convergence-Gap Weighting) killed in two formulations: Formulation 1 (raw ratio) caused real training collapse at epoch 6 (val Dice 0.7135 vs reference 0.8463); Formulation 2 (saturating transform) fixed instability but gave flat-to-slightly-negative result (−0.35pp mean). E45 (D4+D8 combined deep supervision, new `UNet3D_v4`) trained cleanly, best val Dice 0.9110 (+0.47pp over 0.9063 baseline, +0.14pp over D4-only), below the +1.0pp bar.

**Self-contradiction/hedging**: None found — verdicts (both RCGW formulations killed, E45 real-but-sub-threshold) are stated plainly and consistently throughout, no walkback.

**Bugs**: E45's lambda_d8 value-matched calibration (1.0010) was rejected after gradient-magnitude check revealed 2.18× blowover vs tolerance band; recalibrated to 0.4581. Not exactly a "bug" but a caught-and-corrected miscalibration before trusting the result.

**Cross-references**: References the pre-pivot arc (E1–E43) and `PHASE_RESEARCH_ARC_MASTER_REPORT.md` / `project_requirements_and_status_v2` memory for context. States this document + phase-specific memory summaries form the complete record as of 2026-08-19.

---

## PHASE_E45_E54_3SEED_CONFIRMATION_RESULT.md

**Summary**: Runs the mandatory 3-seed confirmation for E45 (D4+D8) and E54 (A96), the only two mechanisms in the whole post-pivot arc with doubly-significant single-seed per-subject Dice results. 3-seed mean per-subject Dice: E45 = 0.8939 (corrected target 0.8942, NOT MET by 0.0003); E54 = 0.8944 (MET by 0.0002). Both fail the original stronger pooled target (0.9163) outright (E45: 0.9086, E54: 0.9062).

**Self-contradiction/hedging**: Explicit and central to the document — it directly undercuts the appearance of a pass/fail split: "Reporting E54 as 'confirmed' and E45 as 'not confirmed' on this basis would overstate the precision... the honest reading is that both mechanisms are statistically indistinguishable from the corrected target itself." The margins (0.0003, 0.0002) are called "smaller than a tenth of the mechanisms' own between-seed standard deviation." Final decision states plainly: "neither E45 nor E54 is confirmed" — despite E54 technically "passing" a threshold in the table.

**Bugs**: None reported.

**Cross-references**: Says this closes the last open thread from `PHASE_EVIDENCE_MAP.md`. Cites E44–E60 arc broadly, and calls out E56 for establishing the per-subject/paired-significance methodology this confirmation follows.

---

## PHASE_E46_ATTENTION_GATE_RESULT.md

**Summary**: E46 built `UNet3D_v5`, an Attention-U-Net-style gate on the `enc1` skip conditioned on the bottleneck, motivated by E43's unresolved routing question. Ablation-safety verified bit-for-bit. Result: best val Dice 0.9102 (+0.39pp over baseline, +0.06pp over D4-only — a statistical tie), psi diagnostic showed a real, non-degenerate spatial pattern (not collapse). Below the +1.0pp bar.

**Self-contradiction/hedging**: None — verdict "NOT MET" stated plainly and consistently; "Not reframed as a success."

**Bugs**: None reported in this document (the shared-tensor bug affecting E62/E63 is a distinct, later issue in a different script, not present here).

**Cross-references**: Follows E44 (killed on instability) and E45 (+0.47pp, below target). Motivates E47 (uses E46's trained gate as a causal instrument).

---

## PHASE_E47_CAUSAL_ROUTING_AUDIT_NULL.md

**Summary**: Uses E46's trained attention gate as a causal instrument: clamps psi=1 (gate disabled) at boundary vs. matched-count interior voxels, measures Dice-drop difference. Result: drop_difference = +0.00025 (~zero), permutation p=0.808 — NULL. Boundary-routing is not the mechanism behind the Dice ceiling; the gate's effect (where it exists) is diffuse, not boundary-concentrated.

**Self-contradiction/hedging**: None — stated as a clean, confirmed null throughout.

**Bugs**: Explicitly caught and fixed: "a first implementation bug used nearest-neighbor resize and produced zero boundary voxels for every subject, caught before trusting any result" (fixed via fractional/area interpolation).

**Cross-references**: Confirms E43's earlier correlational null causally. Motivates E48 ("is the bottleneck's information failing to be encoded at all... rather than failing to be routed?").

---

## PHASE_E48_BOTTLENECK_ENCODING_AUDIT_REVERSED_FINDING.md

**Summary**: Tests whether the bottleneck fails to encode signal for small lesions (predicted: severing bottleneck should hurt large lesions more). Full bottleneck-zeroing ablation: mean Dice drop 0.32 (0.89→0.57) overall; Spearman rho(native_size, drop) = −0.454, p<0.001, n=125 — the OPPOSITE of the hypothesized direction. Small lesions depend MORE on the bottleneck. Interpreted as the first genuinely new causal mechanism in the diagnostic chain.

**Self-contradiction/hedging**: The document itself flags its own literal pre-declared rule was violated in form but not spirit: "the pre-declared GO rule required rho > 0... by the literal pre-declared rule: NULL for the originally-stated hypothesis." It then argues the opposite-direction finding is "scientifically the more important outcome" and reports it in full — this is a disclosed, explicit reframing (not a silent walkback), presented transparently as such.

**Bugs**: None reported (bit-for-bit sanity check passed before trusting results).

**Cross-references**: Builds on E47's null. Motivates E49 (CCABA) directly.

---

## PHASE_E49_CCABA_DESIGN.md

**Summary**: Design doc for CCABA (Causally-Calibrated Adaptive Bottleneck Amplification) — a bottleneck amplification mechanism whose size-conditioning function is fit directly to E48's real causal-ablation data (log-linear OLS, R²=0.260), not assumed. Literature scan found no prior work uses a measured causal dependency curve as a design specification. Verified ablation-safety and shape correctness before training.

**Self-contradiction/hedging**: None (design-only document, no result claimed yet).

**Bugs**: None reported.

**Cross-references**: Explicitly built on E43 (null), E47 (causal null), E48 (reversed finding). Distinguishes itself mechanistically from E46's attention gate.

---

## PHASE_E49_CCABA_RESULT.md

**Summary**: CCABA (`UNet3D_v6`) trained cleanly; single-seed best val Dice 0.9114 (+0.51pp over baseline, +0.18pp over D4-only) — the best of four post-pivot attempts at the time. Lambda_frac gradient-calibrated (0.0203) after value-matching would have caused a 25.4× gradient blowup (rejected).

**Self-contradiction/hedging**: The document itself pre-emptively flags the result may not hold: "Two candidate explanations, not yet distinguished: 1. Seed noise... 2. A true architectural ceiling." Explicitly recommends a multi-seed check before trusting the headline number — a hedge built into the same document, later borne out by PHASE_E49_CCABA_MULTISEED_VARIANCE.md.

**Bugs**: Gradient-magnitude calibration caught and rejected a 25.4× blowup (value-matched calibration).

**Cross-references**: Compares directly against E44 (killed), E45 (+0.47pp), E46 (+0.39pp). Recommends the multi-seed check that becomes its own follow-up document.

---

## PHASE_E49_CCABA_MULTISEED_VARIANCE.md

**Summary**: First multi-seed check in the project's history. CCABA on seeds 0/1/2: 0.9114, 0.9075, 0.9093 — mean 0.9094 (+0.31pp over baseline, not +0.51pp), std ≈0.19pp, 95% CI [0.9046, 0.9142] (crosses D4-only). Conclusion: the single-seed headline (+0.51pp) was favorable seed noise.

**Self-contradiction/hedging**: This document IS the explicit, formal walkback of E49's own single-seed headline claim — "Seed 0's headline +0.51pp result was favorable seed variance, not a stable effect." It also retroactively casts doubt on E44/E45/E46's single-seed numbers: "none of E45's 0.9110, E46's 0.9102, or E49's own 0.9114 can be treated as precise point estimates."

**Bugs**: None reported (methodological finding, not a bug).

**Cross-references**: Directly retracts/qualifies PHASE_E49_CCABA_RESULT.md's headline number. Establishes the ≥3-seed policy applied in E50, E51 onward. Recommends CCABA+E46 attention gate combination as next step (becomes E51).

---

## PHASE_E50_IECG_DESIGN.md

**Summary**: Design for IECG (Internally-Estimated Counterfactual Gating), `UNet3D_v7` — a live, jointly-trained causal-sensitivity estimate via on-the-fly bottleneck ablation replayed through the decoder during training (ablation branch skipped at inference). Literature scan confirmed genuinely differentiated from existing post-hoc counterfactual methods (MoE routing analysis, TRACE-Seg3D). Lambda_sens gradient-calibrated to 0.0841 (value-matching would give 221× blowup, rejected). Pre-declared: evaluate on 3 seeds from the start per the new post-E49 policy.

**Self-contradiction/hedging**: None (design-only).

**Bugs**: 221× gradient blowup caught and rejected during calibration.

**Cross-references**: Directly motivated by PHASE_E49_CCABA_MULTISEED_VARIANCE.md's finding that CCABA's headline was largely noise; explicitly says CCABA's core limitation is a "fixed, externally-fit curve."

---

## PHASE_E51_CCAG_COMBINED_RESULT.md

**Summary**: CCAG (`UNet3D_v8`) combines CCABA (E49) and the attention gate (E46) — the pre-declared "last single-architecture-family attempt." All ablation-safety checks passed bit-for-bit (full-off, CCABA-only, gate-only), param overhead exactly additive. Evaluated on 3 seeds from the start: 0.9101, 0.9092, 0.9092 — mean 0.9095, std ≈0.05pp (tightest of any condition), 95% CI [0.9082, 0.9108]. Result: +0.32pp over baseline, but a statistical dead tie with both D4-only (0.9096) and CCABA alone (0.9094, paired t-test p=0.919).

**Self-contradiction/hedging**: None — verdict "did not compose additively, or even improve on either mechanism alone" stated plainly and consistently; explicitly not spun as a partial success despite the very tight variance.

**Bugs**: None reported — three separate ablation-safety checks (full-off vs v3, CCABA-only vs v6, gate-only vs v5) and additive param-count check all passed bit-for-bit before training.

**Cross-references**: Directly combines E49 (CCABA) and E46 (attention gate). Cites E47's causal finding (gate's effect is diffuse, not boundary-localized) as the explanation for why the combination didn't synergize. Declares itself "the final single-architecture-family attempt for this project," per the user's explicit pre-declared stop condition — closing out the line continued in PHASE_POST_E43_SUMMARY.md's synthesis.

---

## PHASE_E52_ASR_GRADIENT_CALIBRATED_DESIGN.md

**Summary**: Design for re-attempting E34 (Adaptive Size-Reweighting), which had catastrophically collapsed pre-pivot (best Dice 0.7108–0.7360) using a loss-value-only calibrated lambda. This phase re-calibrates by gradient magnitude to test whether E34's original failure was a calibration artifact vs. a mechanism failure. Only the `[R]` (alpha_c) condition re-tested. Pre-declares a 2-epoch smoke-test gate before any full run.

**Self-contradiction/hedging**: None (design-only, framed as an open, falsifiable question with three possible outcomes disclosed in advance).

**Bugs**: None yet (design stage).

**Cross-references**: References `PHASE_E34_ADAPTIVE_SIZE_REWEIGHTING_KILLED.md` (E34's original kill) and ties the objective-level lever explicitly back to E43/E47/E48's causal chain.

---

## PHASE_E52_ASR_GRADIENT_CALIBRATED_RESULT.md

**Summary**: Recalibrated lambda_cw to 0.1101 (E34's original lambda would have caused a 43× gradient blowup). Smoke test still collapsed: val Dice 0.10 at epoch 1 (same "predict everything as tumor" pattern as E34's original failure). Killed at the smoke-test gate — gradient calibration alone did not fix the collapse. Diagnosis: the component-weighted loss term's *direction*, not magnitude, actively works against early training.

**Self-contradiction/hedging**: None — the killed verdict stands consistently; the document explicitly frames its finding as "more informative than a simple repeat of E34's own null" rather than claiming success.

**Bugs**: Explicitly caught and fixed: a real performance bug where the labeled-component cache was rebuilt from scratch every batch (reproducing a previously-documented ~5–7× slowdown bug from E34's own code) — fixed by precomputing the cache once upfront (507.6s one-time cost).

**Cross-references**: Directly follows PHASE_E52_ASR_GRADIENT_CALIBRATED_DESIGN.md and E34 (`PHASE_E34_ADAPTIVE_SIZE_REWEIGHTING_KILLED.md`). Motivates E53 (curriculum warmup) as the direct next step.

---

## PHASE_E53_ASR_CURRICULUM_WARMUP_DESIGN.md

**Summary**: Design for E53, directly motivated by E52's diagnosis (direction, not magnitude, of the component-weighted loss fights early training). Proposes a linear ramp (epochs 8→20) to delay the loss term until baseline competence is established. Literature scan found no source combining a time-varying warmup with resize-survival-based reweighting specifically.

**Self-contradiction/hedging**: None (design-only).

**Bugs**: None reported.

**Cross-references**: Directly built on E52's own diagnosis; reuses E52's calibrated lambda_cw=0.1101 and E34's weight table unchanged. (Its result appears only in PHASE_POST_E43_SUMMARY.md — no standalone E53 result file was in this reading list, and PHASE_POST_E43_SUMMARY.md notes the run was incomplete/stopped by the user after only seed 0 finished at 0.8969, below baseline.)

---

## PHASE_E54_A96_RESOLUTION_DESIGN.md

**Summary**: Design to test E29's proposed but never-run "A96" resolution experiment: train `UNet3D_v3` unchanged at 96³ instead of 64³. Memory profiling found only batch=2 (with gradient accumulation ×4 for effective batch 8) fits the GPU budget; batch=8 at 96³ measured 15.626GB (OOM). Pre-declares seed-0-first, then 3-seed policy only if seed 0 shows a real signal.

**Self-contradiction/hedging**: None (design-only).

**Bugs**: None reported (a real, disclosed limitation — BatchNorm sees only physical batch=2 — but not framed as a bug).

**Cross-references**: Directly follows up on E29's own unexecuted proposal (`PHASE_E29_RESIZE_SURVIVAL.md`).

---

## PHASE_E54_A96_RESOLUTION_RESULT.md

**Summary**: Seed-0 result: best val Dice 0.9066 (+0.03pp over baseline, essentially a dead tie; −0.30pp vs D4-only). Per pre-declared plan, did not proceed to seeds 1–2 since the gap to D4-only was outside the project's measured seed-noise band. Clean training throughout.

**Self-contradiction/hedging**: None — consistent "NOT MET, not close" verdict throughout. (Note: this pooled-Dice-based "NOT MET" verdict is later partially revisited by E56's per-subject re-scoring, which found E54's per-subject Dice WAS statistically significant vs baseline — but that reversal is disclosed in E56's own document, not this one; this document does not contradict itself internally.)

**Bugs**: None reported.

**Cross-references**: Confirms E29's own prediction of "moderate, not dramatic" gain. Motivates E55 (dual-resolution local refinement) as the next, more targeted step. References an "approved plan" document (`ok-what-would-you-iridescent-rossum.md`).

---

## PHASE_E55_DUAL_RESOLUTION_DESIGN.md

**Summary**: Design for E55 — keep the 64³ global pathway (`UNet3D_v3`) intact, add a local pathway examining a native-resolution crop centered on a differentiable soft-centroid from the global prediction, fused back via a learned gate near-zero init. Details coordinate-mapping and grid_sample conventions verified this session. Lists a 6-step pre-training verification order and success criteria (≥0.9163, 3-seed if signal found).

**Self-contradiction/hedging**: None (design-only).

**Bugs**: None reported (design stage; notes several implementation facts "verified this session, not assumed").

**Cross-references**: Directly motivated by E54 (flat whole-volume result) and E48 (small lesions depend on global context) jointly — states the mechanism must recover local detail "without discarding global context."

---

## PHASE_E55_DUAL_RESOLUTION_RESULT.md

**Summary**: All 6 pre-training verification checks passed (label-crop alignment 99–100% mass recovery, ablation-safety bit-for-bit, gradient flow confirmed). Trained cleanly; result: val Dice 0.9007 — WORSE than baseline (−0.56pp), D4-only (−0.89pp), and E54 (−0.59pp). fusion_gate rose then declined over training (0.356 peak → 0.234 final), suggesting the model dialed back reliance on the local pathway. HD95 improved (1.30, best of any condition), showing the local pathway did something useful for boundary precision specifically, just not for Dice.

**Self-contradiction/hedging**: None — the negative result is stated plainly and not spun; explicitly labeled "the first to land clearly below the canonical baseline on a properly-verified, non-collapsed run."

**Bugs**: None reported — explicitly noted "not a training failure," every verification passed.

**Cross-references**: Directly follows E54. Cites `PHASE_POST_E43_SUMMARY.md` as "to be updated" with this result.

---

## PHASE_E56_MEASUREMENT_AUDIT_RESULT.md

**Summary**: Major reframing phase. Discovers (1) the canonical baseline "0.9063" was never a clean control — it's seed 0 of a 4-seed margin-loss experiment with margin genuinely active; true 4-seed mean is 0.9044 (CI [0.9007, 0.9080]), making the correct +1pp target 0.9144, not 0.9163. (2) Pooled Dice inflates every mechanism's score by 0.85–2.40pp relative to per-subject Dice. (3) Re-scoring 14 existing checkpoints on per-subject Dice with paired significance testing finds E45 (+1.44pp per-subject, doubly significant) and E54 (+1.39pp per-subject, doubly significant) both already clear the corrected per-subject +1pp target (0.8942) on their single seed.

**Self-contradiction/hedging**: The document explicitly self-qualifies its own big claim: "This is not yet a final claim. Both E45 and E54 are single-seed results. Per this project's own mandatory ≥3-seed policy... neither can be treated as a confirmed success until evaluated on 2 additional seeds." This hedge is exactly what PHASE_E45_E54_3SEED_CONFIRMATION_RESULT.md later tests and finds NOT confirmed on the stronger reading.

**Bugs**: None reported — a measurement/analysis correction, not a code bug, though it does correct a long-standing convention error (treating a single seed of a different experiment as "the baseline").

**Cross-references**: Explicitly traces the canonical baseline back to `PHASE_RESEARCH_ARC_MASTER_REPORT.md` Section 5. Cites E27's earlier pooled-vs-per-subject gap finding and E25's earlier paired-significance finding as prior, previously-unapplied results. Directly sets up PHASE_E45_E54_3SEED_CONFIRMATION_RESULT.md as the recommended next step (which later found neither mechanism robustly confirmed).

---

## PHASE_E58_BOTTLENECK_CONTEXT_CHARACTERIZATION.md

**Summary**: Multi-stage causal characterization of E48's bottleneck-dependence finding. Stage 1 (fixed octant ablation): neither of two octants reproduces the full-ablation size-signature — no spatial localization. Stage 1b (lesion-conditioned octant): no support for lesion-relative organization. Stage 2 (cross-subject substitution): a random foreign subject's bottleneck is significantly WORSE than zeroing (p=0.026) — "wrong context is worse than no context." Stage 2b (checkpoint-artifact control on E48's own checkpoint): confirms this more strongly, plus finds a new asymmetry (size-signature survives size-matched donor substitution on the gated checkpoint but not the ungated one). Stage 3 (OOD confound test): mean-bottleneck ablation does comparatively little damage (similar to zero) — supports the "out-of-distribution ablation" confound explanation (Li & Janson 2024) rather than a genuine "representation compatibility" phenomenon.

**Self-contradiction/hedging**: The document is explicitly self-correcting in real time — it flags its own Stage 2 finding as a "candidate 'genuinely new phenomenon'" but then Stage 3 directly undercuts it: "evidence against treating the donor-substitution finding as a genuine, novel 'cross-subject representation compatibility' phenomenon." This is disclosed transparently within the same document, not contradicted elsewhere.

**Bugs**: None reported (multiple bit-for-bit sanity checks passed).

**Cross-references**: Explicitly built on E48. Explicitly frames itself as diagnostic, not a novelty claim, per "pre-declared framing" instruction. Recommends closing this diagnostic line; the still-open question (why small lesions depend on bottleneck) is left for E59.

---

## PHASE_E59_BOTTLENECK_INTERACTION_AUDIT.md

**Summary**: Tests additive vs. synergistic bottleneck coalition structure via octant subset ablation. Population-average interaction is NOT significant (p=0.146) but stratifying by lesion size reveals a clean split: small lesions show significant positive synergy (mean I_ij=+0.00104, p=0.049; smallest quartile p=0.0003), large lesions show significant negative interaction/redundancy (mean I_ij=−0.00192, p<0.0001). Spearman(native_size, mean_I_ij) = −0.498, p=3.3e-9.

**Self-contradiction/hedging**: The document itself explicitly notes the pre-declared population-level test initially failed and required a re-read: "The population-average null the pre-declared rule checked for was the wrong single number to look at." This is presented as a correct refinement, not a reversal, but is worth flagging as the document narrating its own initial-test failure before finding the real signal via stratification.

**Bugs**: None reported (bit-for-bit sanity check passed).

**Cross-references**: Directly builds on E48 and E58 ("resolves the ambiguity E58 left open"). Motivates E60 (Distributed Bottleneck Coding, DBC) as the next mechanism design.

---

## PHASE_E60_DBC_FEASIBILITY_KILLED.md

**Summary**: Zero-training feasibility check for DBC (motivated by E59's synergy finding), using DBC's actual intended balanced 4-vs-4 partition sampling instead of E59's exhaustive pairwise sampling. Result: the small-lesion synergy signature does NOT reproduce under this formulation (small-lesion mean Γ not significantly different from zero, p=0.530). Large lesions show a strong negative Γ (consistent with E59's redundancy finding) but that's not the branch DBC needed. Killed with zero training cost, per pre-declared rule.

**Self-contradiction/hedging**: None — killed verdict stated plainly, with two disclosed (not adjudicated) candidate explanations for why E59's finding didn't reproduce (sampling-scheme sensitivity vs. E59's own effect being more fragile than its p-values suggested) — explicitly declines to further tune/rescue per the project's own stop-rule.

**Bugs**: None reported (bit-for-bit sanity check passed).

**Cross-references**: Directly follows and narrows E59's finding; explicitly declares itself the "ninth mechanism-adjacent hypothesis killed or nulled since the strategic pivot."

---

## PHASE_E61_BOTTLENECK_DIMENSIONALITY_AUDIT.md

**Summary**: Tests whether small-lesion segmentation needs a higher-effective-dimension bottleneck representation. All three primary dimensionality metrics (effective rank, stable rank, participation ratio) fail to show "small > large" — in fact effective rank shows the REVERSE (large > small, p not given as significant value but described as the only metric showing any effect, in the wrong direction). After controlling for lesion size, the small/large gap disappears entirely (p=0.45). The correlation with E48's CausalDrop also loses significance after controlling for size (partial rho=+0.081, p=0.368, flipped sign). All 7 pre-declared decision criteria FAIL. Explicit KILL verdict.

**Self-contradiction/hedging**: None — consistently and explicitly a clean kill across every criterion; document is unusually thorough in disclosing every control test (activation-scale, lesion-burden, randomized-control) that fails to rescue the hypothesis.

**Bugs**: None reported.

**Cross-references**: Explicitly notes a checkpoint mismatch limitation: E61 uses the v5/E46 checkpoint (to match E48) while E58/E59 used the plain v3/D4-only checkpoint — so its qualitative cross-check against E59 is disclosed as "cross-architecture, informal only." States this is "the fourth distinct framing... to return a clean null or reversed-direction result" (after E47 routing, E58 localization, E59/E60 synergy).

---

## PHASE_E61_NOVELTY_GAP_SEARCH.md

**Summary**: Search-first literature review across 5 candidate directions implied by the E43–E60 causal chain, before any mechanism design. 4 of 5 directions found OCCUPIED by 2025–2026 papers (cascade/coarse-fine, cross-scale arbitration, representation-level causal intervention, conditional gating). One direction (representation-collapse theory for small objects) is "partially open" but not itself a mechanism.

**Self-contradiction/hedging**: None — the "honest assessment" section explicitly concludes no candidate satisfies the novelty filters: "The searches did not surface a gap; they surfaced further confirmation that this neighbourhood is saturated."

**Bugs**: N/A (literature search only).

**Cross-references**: References the "pattern already established across E55/E57/E58/E59-E60 novelty checks" of a saturated design space.

---

## PHASE_E62_E63_INVALIDATED_SKIP_CONNECTION_BUG.md

**Summary**: This is the bug-report/invalidation document itself. States both `run_e62_maxpool_position_audit.py` and `run_e63_competition_geometry_audit.py` share a `forward_from_enc1()` function that reuses a single `enc1` tensor for two roles: `pool1`'s input AND the decoder skip connection. Because `nn.MaxPool3d(2)` is non-overlapping (k=2,s=2), `pool1(enc1) = pool1(π(enc1))` for any within-cell derangement — provably, and independently reverified in this session. Therefore 100% of both phases' measured Dice effects came through the skip-connection path (`attn_gate1`+`cat1`), not through any information lost by MaxPool3d. An inline code comment asserting the opposite ("skip = enc1 — reads the REAL, un-permuted enc1") was an unverified, false claim. Explains why the existing bit-for-bit sanity checks (identity-forward reproduction) could not have caught this bug (a no-op permutation is unaffected by the shared-tensor issue). Prescribes the correction path (two independent tensors, pool-only and skip-only tests) — explicitly NOT yet run as of this document, later executed as E64.

**Self-contradiction/hedging**: This document is itself the correction/retraction of E62 and E63's prior GO/KILL verdicts — it explicitly states "E62's GO verdict... is not supported by this data" and "E63's KILL verdict... may still be internally consistent as a statement about the skip-connection effect, but its framing throughout... inherits E62's mischaracterization." It also explicitly states the numeric results themselves were "numerically real, not a measurement bug or noise" — only the causal attribution (pooling vs. skip) was wrong. Includes a status banner at top: "RESOLVED by PHASE_E64_SPLIT_INTERVENTION_AUDIT.md" confirming E64 later validated exactly the predicted resolution (Δ_pool=0.0, Δ_skip real and size-specific).

**Bugs**: This entire document IS the bug report — a shared-tensor confound (`enc1` used for two logically distinct roles) affecting both E62 and E63. Explicitly documents a new methodological safeguard for future work: "an explicit differential check: intervene on ONE role while asserting the other role's downstream output is unchanged."

**Cross-references**: Explicitly retracts/invalidates PHASE_E62_MAXPOOL_POSITION_AUDIT.md's GO verdict and PHASE_E63_COMPETITION_GEOMETRY_AUDIT.md's causal framing (though not necessarily its internal KILL conclusion about winner/runner-up geometry). States it is resolved by PHASE_E64_SPLIT_INTERVENTION_AUDIT.md.

**RELATIONSHIP SUMMARY (E62/E63/invalidation chain, as stated by the docs themselves)**: E62 originally claimed GO — that MaxPool3d's subcell-position loss at pool1 is causally task-relevant and small-lesion-specific (mean Dice drop 0.0226, size correlation rho=−0.355, both significant). E63 then tested whether this effect was driven by winner/runner-up (top-2) competition geometry specifically and found KILL — the effect is diffuse across the full local value distribution, not reducible to top-2 relocation. Both phases used the same shared `forward_from_enc1()` helper, which had a bug: `enc1` was passed as a single argument used for two distinct roles (feeding `pool1` AND feeding the decoder's skip connection), so any permutation applied for the "pooling" intervention also silently reached the skip connection. Because non-overlapping MaxPool3d(2) is mathematically invariant to any within-cell value permutation, `pool1`'s output could never have changed under either phase's intervention — meaning 100% of both phases' measured effects necessarily came through the untouched skip-connection path, not through pooling as claimed. The bug was caught not by the existing bit-for-bit sanity checks (which only test the no-op/identity case and are blind to dual-role tensor reuse) but by an independent mathematical proof plus a differential check design. E64 then re-ran the audit correctly, using two fully independent tensors (`E_pool` for pooling only, `E_skip` for the skip connection only), on both the v5 (gated) and v3 (ungated) checkpoints. Results: Δ_pool = exactly 0.00000 (p=1.0) on both checkpoints — confirming pooling truly has zero causal effect; Δ_skip reproduced E62's original 0.0226 number almost exactly on v5, and was even larger (0.0270) on v3 without any attention gate. Final resolved state: the real, robust, size-specific effect originally misattributed to MaxPool3d information loss is in fact a skip-connection representation effect (sensitivity to `enc1`'s raw subcell spatial arrangement as it reaches `dec1`), independent of and not requiring the attention gate. E62's "pooling matters" causal claim and any pooling-operator redesign (PMD) motivated by it are formally discarded; E63's specific "not winner-runner-up geometry" conclusion is retained as informative about the (correctly relocated) skip-connection effect, though its original framing as "the local analogue of the E62 pooling intervention" is acknowledged as incorrect. This correctly-diagnosed skip-connection effect then becomes the subject of the further decomposition in E65.

---

## PHASE_E62_MAXPOOL_POSITION_AUDIT.md

**Summary**: (See relationship summary above.) Tested whether MaxPool3d's discarding of subcell spatial position is causally task-relevant. Original claimed result: mean Dice drop 0.0226 (p=2.0e-28), Spearman(native_size, drop)=−0.355 (p<0.001) — both pre-declared GO criteria met. Original verdict: GO, motivating a Positional-Moment Downsampling (PMD) design.

**Self-contradiction/hedging**: The document itself now carries a prominent top-of-file warning banner: "⚠️ INVALIDATED — see PHASE_E62_E63_INVALIDATED_SKIP_CONNECTION_BUG.md" stating the GO verdict "is not supported" and that the numbers, while real, measured a different mechanism (skip connection, not pooling) than claimed. This is the clearest case in the batch of a document's own displayed verdict being retracted by a companion document, with the retraction physically inserted at the top of the original file.

**Bugs**: The shared `forward_from_enc1()` dual-role tensor bug (see above), affecting this file's core causal claim.

**Cross-references**: Invalidated by PHASE_E62_E63_INVALIDATED_SKIP_CONNECTION_BUG.md; the underlying effect is correctly re-measured in PHASE_E64_SPLIT_INTERVENTION_AUDIT.md.

---

## PHASE_E63_COMPETITION_GEOMETRY_AUDIT.md

**Summary**: Tested whether E62's effect is driven specifically by winner/runner-up (top-2) competition geometry inside each pooling cell. Three nested tests (A: full derangement reproduces E62 almost exactly at 0.0223; B: winner fixed/runner-up moved — only captures 1.4% of Test A's effect, not significant; C: top-two fixed/weak values rearranged — significant and LARGER than Test B, opposite of the hypothesis's prediction). All 7 pre-declared decision criteria fail. Verdict: KILL — the winner-runner-up hypothesis is not supported; signal is diffusely distributed across the full local value arrangement.

**Self-contradiction/hedging**: Also carries the top-of-file warning banner noting the shared bug means "Tests A/B/C all measured skip-connection sensitivity, not a property of information MaxPool3d discards" and that its framing as "the local analogue of the E62 [pooling] intervention" is incorrect, even though the internal KILL conclusion about winner/runner-up geometry not explaining the effect may still be informative about the (differently-attributed) skip effect. Also separately documents an internally-caught issue: the original intervention-scale design (200 cells/subject, ~0.02% of cells) produced a near-zero effect even for Test A, which "could not separate 'geometry doesn't matter' from '0.02% of cells is too small a perturbation'" — flagged to the user and corrected to whole-volume scale before reporting final results.

**Bugs**: Inherits the shared `forward_from_enc1()` bug from E62. Also documents its own separately-caught scale-confound issue (too few cells intervened on in the first design attempt), fixed before the reported results.

**Cross-references**: Invalidated (in its causal framing, not necessarily its internal KILL logic) by PHASE_E62_E63_INVALIDATED_SKIP_CONNECTION_BUG.md. Directly follows PHASE_E62_MAXPOOL_POSITION_AUDIT.md.

---

## PHASE_E64_SPLIT_INTERVENTION_AUDIT.md

**Summary**: The corrected re-run of E62/E63 using two fully independent tensors (`E_pool`, `E_skip`). Ran on both v5 (gated) and v3 (ungated) checkpoints. Results: Δ_pool = exactly 0.00000 (p=1.0) on both checkpoints; Δ_skip = 0.02262 (v5, matches E62's 0.0226) and 0.02703 (v3, larger without the gate), both highly significant and size-specific (rho=−0.355 and −0.404 respectively). Verdict: SKIP-CONNECTION REPRESENTATION EFFECT CONFIRMED — NOT A POOLING EFFECT. Explicitly rules out the attention gate as the mechanism (effect is present, and larger, without it).

**Self-contradiction/hedging**: None — this document is itself the resolution/correction of the prior E62/E63 confusion, stated cleanly and without further hedge.

**Bugs**: None new; this phase exists specifically to fix the E62/E63 bug via a differential-check design, and explicitly verifies the fix (pool1 output bit-identical under permutation; pool-only intervention produces zero output change; skip-only intervention produces large output change).

**Cross-references**: Directly supersedes/corrects PHASE_E62_MAXPOOL_POSITION_AUDIT.md and PHASE_E63_COMPETITION_GEOMETRY_AUDIT.md per PHASE_E62_E63_INVALIDATED_SKIP_CONNECTION_BUG.md's stated resolution path. States findings do "not support PMD, RCD, TSQL, or any other pooling-operator redesign." Motivates E65 (skip-property decomposition).

---

## PHASE_E65_SKIP_PROPERTY_DECOMPOSITION.md

**Summary**: Decomposes "sensitivity to enc1's spatial arrangement" (confirmed by E64) into four causal properties on the skip tensor alone (v3/ungated checkpoint): translation (+0.2143 Dice drop, dominant), channel permutation (+0.1009, secondary), smoothing (+0.0284), local permutation (+0.0270, = E64's Δ_skip). All four are significant and size-specific, but translation is ~8× larger than local permutation/smoothing, and channel permutation ~4× larger. Conclusion: absolute spatial correspondence (registration) between skip and decoder-upsampled path is the dominant causal factor, channel identity secondary, fine local detail/frequency tertiary.

**Self-contradiction/hedging**: None — findings presented consistently; explicitly notes "nothing is ruled out" (all four arms significant) but ranks them by magnitude without walking back any of E64's findings.

**Bugs**: None reported (differential check from E64 reused and re-verified; unit tests for each intervention's invariant properties passed).

**Cross-references**: Directly builds on and extends PHASE_E64_SPLIT_INTERVENTION_AUDIT.md's finding. Explicitly notes this "has NOT yet been translated into an algorithm design" and recommends a novelty search next — this becomes E67/E67b.

---

## PHASE_E66_GRADIENT_TOPOLOGY_AUDIT.md

**Summary**: Tests whether gradient conflict between the project's live loss terms (FocalTversky vs. EvidentialBeta — the only pair both undetached through the full shared trunk) varies structurally by network region (encoder/bottleneck/decoder/heads), using 7 existing checkpoints, zero new training. Result: the two losses are cooperative everywhere (positive cosine at every checkpoint, every block: encoder ~0.845, bottleneck ~0.813, decoder ~0.463) — no conflict, no sign flip anywhere. Real magnitude variation exists across blocks but is a smooth gradient (Outcome B), not the sign-flip pattern (Outcome C) needed to motivate "Gradient-Conditioned Objective Routing" as a new mechanism. Verdict: KILL.

**Self-contradiction/hedging**: None — verdict stated plainly and matched against a pre-declared three-way decision tree (A/B/C), landing cleanly on B (kill).

**Bugs**: None reported; instead documents corrections to the *original proposal's assumptions* — the proposed L1–L4 objective set doesn't match the live pipeline (consistency/pseudo-label losses only exist in an unrelated frozen pipeline); boundary loss is architecturally excluded (detached gradient by design, not by discovery); heads block is geometrically forced to cos=0 (disjoint parameter subsets) and excluded from the decision.

**Cross-references**: Explicitly returns to "the project's original stated axis" after "the E62–E65 causal-diagnostic branch (retained as completed side-branch evidence)" and after "a self-corrected multimodal detour (CMRR, killed on novelty grounds)" — referencing prior undocumented-in-this-batch phases. Notes the dormant `lambda_margin` term "was tested extensively in the EGGO-M arc and found null on Dice at E13."

---

## PHASE_E67_SKIP_CORRESPONDENCE_NOVELTY_SEARCH.md

**Summary**: Literature search (before any design) on E65's finding (spatial correspondence dominant at the skip connection). Direction 1 (registration-aware/deformable skip alignment) found OCCUPIED by Dynamic U-Net's DCU module (arXiv:2403.07303, 2024) — a very close direct precedent. Direction 2 (offset-learning for alignment) adjacent-occupied (OffSeg). Direction 3 (size-conditional deformable correction) and Direction 4 (causal-audit-derived design methodology) both NOT FOUND. Verdict: the base mechanism is occupied (domain port only); what's not occupied is size-conditional calibration derived from the causal chain.

**Self-contradiction/hedging**: None — honest assessment explicitly states this "does not support a 'novel algorithm' claim" and frames the real contribution narrowly (methodology + size-conditioning), consistent throughout.

**Bugs**: N/A (literature search only).

**Cross-references**: Directly follows E65's causal decomposition. Proposes SC-DCU design (becomes E68) as next step.

---

## PHASE_E67B_MECHANISM_LEVEL_NOVELTY_SEARCH_EXTENDED.md

**Summary**: Extended novelty search (4 more search rounds, 8 total across E67/E67b) for a structurally new mechanism. Direction 5 (position/semantic decoupling) OCCUPIED by MPLSeg (Fourier magnitude/phase decomposition). Directions 6–7 (scale-conditioned correction magnitude; position as its own uncertainty-quantified signal) NOT FOUND but assessed as likely "thin" research areas rather than genuine gaps. Direction 8 (coordinate-confidence positional embedding) found ADJACENT (HELP/HPE, transformer detection, different architecture class). Conclusion: no genuinely unoccupied, directly actionable mechanism-level idea found after 8 searches; the space is densely worked from multiple angles ("in the air").

**Self-contradiction/hedging**: None — conclusion consistent with and reinforces E67's own conclusion, explicitly stating "does not overturn E67's conclusion."

**Bugs**: N/A (literature search only).

**Cross-references**: Directly continues PHASE_E67_SKIP_CORRESPONDENCE_NOVELTY_SEARCH.md; references "prior novelty-search conclusions at E55/E57/E58/E59-E60/E61/E67" as a repeated pattern.

---

## PHASE_E68_SC_DCU_DESIGN.md

**Summary**: Long, multi-stage design document for SC-DCU (Self-supervised Correspondence-verified DCU). Original Sections 1–5 proposed a hard-equality offset-fidelity loss (synthetic shift → supervised target). A "follow-up targeted novelty search" (conducted by a different process, independently checked here) found this claim too strong: synthetic-deformation-as-ground-truth and transformation-consistency supervision are BOTH well-established prior art (registration literature, semi-supervised segmentation), and — more fundamentally — nothing justified assuming the task-optimal offset should equal the literal geometric inverse of an imposed shift. Section 7 replaces the retracted content with a corrected, three-way falsifiable hypothesis (H68-A: geometric≈task-optimal; H68-B: systematic discrepancy, structured; H68-C: unstructured noise) and a cheap discriminating experiment. Section 8 imposes a stricter novelty bar: even a confirmed H68-B discrepancy characterization would NOT itself clear the bar — only a new operator built from that structure would.

**Self-contradiction/hedging**: This document is itself an extensive, explicit, in-place retraction of its own earlier sections. It states outright: "Do not cite Sections 1–5's novelty claim; it is retracted" and includes a "Correction record" table showing every original novelty claim downgraded to "No" or "Open." This is one of the most self-critical documents in the batch — the retraction happens within the same file, not via a separate companion doc.

**Bugs**: None yet (design stage; implementation bugs occur in the follow-up result document).

**Cross-references**: Explicitly retracts/supersedes its own Sections 1–6 via Section 7. Builds on E65's causal decomposition and E67/E67b's novelty searches. Its Section 7.3 discriminating experiment becomes PHASE_E68_OFFSET_DISCREPANCY_MEASUREMENT_RESULT.md.

---

## PHASE_E68_OFFSET_DISCREPANCY_MEASUREMENT_RESULT.md

**Summary**: Runs Section 7.3's discriminating experiment (SC-DCU module trained on `L_seg` alone, no offset-fidelity term, 3 pilot epochs). Two implementation bugs found and fixed: (1) a sign-convention error (grid_sample's offset semantics are inverse to torch.roll's — target should be +t not −t); (2) a train/test distribution mismatch (module trained only on unperturbed enc1, evaluated on shifted enc1, producing a spuriously "significant" H68-B correlation that was actually a frozen zero-init artifact). After both fixes, an added effectiveness check FAILED: the module's predicted offset is essentially constant regardless of injected shift magnitude — meaning the module never learned to track shift at all. Conclusion: the experiment is INCONCLUSIVE, not a valid H68-C null, because "a module that isn't tracking shift magnitude at all isn't doing the thing the experiment is designed to measure."

**Self-contradiction/hedging**: The document explicitly distinguishes and corrects an initial mis-framing within itself: after the second bug fix, the naive reading would report H68-C (Spearman ρ=+0.174, p=0.051, "not significant, but only just") as a clean null — but the document explicitly rejects this framing: "this experiment did not produce usable evidence for or against H68-A, H68-B, or H68-C. It is reported as inconclusive, not as a null result." This is a deliberate, careful self-correction distinguishing "no structure found" from "test wasn't sensitive enough."

**Bugs**: Two explicitly caught and fixed: (1) grid_sample/torch.roll sign-convention mismatch; (2) train/test distribution mismatch (module never trained on shifted inputs). A third issue (effectiveness check failure — offset not tracking shift magnitude) was caught but NOT fixed — the phase was stopped instead, per explicit user decision not to keep tuning.

**Cross-references**: Directly executes Section 7.3 of PHASE_E68_SC_DCU_DESIGN.md. States the SC-DCU thread "is closed for now" (per PHASE_E68_SC_DCU_DESIGN.md's own updated status banner referencing this result).

---

## PHASE_E69_E54_OFFLINE_MINING_RESULT.md

**Summary**: Before spending ~24 GPU-hours on 7 more E54 seeds, mines the already-trained 3 E54 seeds for a coherent phenomenon (zero new compute). Finds a real, non-trivial cross-seed-consistency signal: 52/125 subjects (41.6%) consistently improve across all 3 seeds vs. ~25% expected by chance (p<0.0001). However, the pooled size-correlation (rho=−0.118, p=0.023) does NOT hold up: no individual seed shows significant size correlation, and the consistently-improved group's median lesion size (97,805) is actually LARGER, not smaller, than the consistently-degraded group's (77,815) — opposite of the E48/E65 size-specificity story. A follow-up covariate search (E69b) finds baseline Dice is the only covariate showing a nominal difference (p=0.020) but does NOT survive Bonferroni correction (needed α=0.00625). Final classification: "B" — a real cross-seed effect exists but is unexplained by lesion size and only weakly, non-significantly suggestive on baseline difficulty.

**Self-contradiction/hedging**: Explicit and central: "this does not hold up under scrutiny" regarding the initial pooled size-correlation; and "this does not survive multiple-comparisons correction" regarding the baseline-Dice covariate. The document is careful to distinguish "solid" findings (cross-seed consistency itself) from "suggestive, not confirmed" ones (baseline Dice pattern) within its own text.

**Bugs**: None reported (pure re-analysis of existing artifacts, zero new training/inference).

**Cross-references**: Directly follows up on E54 and the corrected-target reasoning from E56/PHASE_E45_E54_3SEED_CONFIRMATION_RESULT.md (mentions E54's "marginal 3-seed per-subject-Dice 'pass'... ~1/10th of its own between-seed standard deviation" as the reason for this cheaper mining step instead of full replication). Cites E48 and E35's existing per-subject tables as reused data sources. Recommends a narrower single-hypothesis follow-up (not the full 7-seed replication) as next step.

