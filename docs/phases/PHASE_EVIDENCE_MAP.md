# Project Evidence Map — What's Actually Been Established (E1–E58)

**Purpose**: a single reference distinguishing what this project has *proven* (causal diagnostics, measurement corrections) from what it has *tried and failed to produce* (a working algorithmic mechanism), and precisely which regions of the design space are closed vs. genuinely open. Built to answer one question before any further mechanism attempt: **is there real, unexploited signal in the existing evidence, or would a new attempt be another blind guess?**

---

## Part 1 — Causal diagnosis: what is actually known about why this model fails

```
                              PROJECT
                                 │
                ┌────────────────┴────────────────┐
                │                                  │
          MODEL DIAGNOSIS                    MEASUREMENT
                │                                  │
       ┌────────┴────────┐              pooled Dice → per-subject Dice
       │                 │              single-seed → 3-seed (mandatory)
   ROUTING           BOTTLENECK          single anchor → 4-seed baseline mean
   (E43→E47)          (E48→E58)
       │                 │
     NULL          DISTRIBUTED, CAUSAL, SIZE-DEPENDENT
  (boundary-      (large effect, doesn't localize spatially,
   routing not     doesn't localize to lesion-relative octant,
   the mechanism)  donor-substitution "compatibility" effect
                    is most likely an OOD-ablation artifact,
                    NOT a genuine semantic finding)
                        │
                        └── MECHANISM UNRESOLVED — this is the honest
                            state, not a failure to search hard enough
```

**Precise, defensible summary statement** (per explicit correction — do not use "global context," it overreaches the evidence):

> The bottleneck makes a distributed causal contribution to segmentation, with disproportionately large effects on small lesions (E48: ρ=−0.454, p<0.001), but the interventions run so far (E58: fixed-octant, lesion-relative-octant, cross-subject substitution, mean-ablation control) do not identify the informational basis of that dependence. Two plausible explanations (spatial localization, cross-subject content incompatibility) were tested directly and ruled out or shown to be confounded, not merely left unexamined.

### E43 → E47: routing hypothesis — CLOSED (null)

| Phase | Test | Result |
|---|---|---|
| E43 | Correlational: does D4 supervision's representation change localize to boundary-ambiguous cells? | Real representation change exists, but not boundary-localized after confound correction |
| E47 | Causal: clamp E46's learned attention gate at boundary vs. matched-count interior voxels | No differential effect (p=0.808) — boundary-routing is not the mechanism |

### E48 → E58: bottleneck-encoding hypothesis — CAUSAL EFFECT REAL, MECHANISM UNRESOLVED

| Phase | Test | Result |
|---|---|---|
| E48 | Causal: zero entire bottleneck, correlate Dice drop with native lesion size | Large effect (mean drop 0.32), significantly size-dependent (ρ=−0.454, p<0.001) — small lesions depend **more** |
| E58 Stage 1 | Causal: zero one of two fixed, diagonally-opposite octants (1/8 each) of the 8³ bottleneck | Neither reproduces the full-ablation signature (octant A ρ=−0.186 p=0.031, CI doesn't overlap reference; octant B ρ=−0.081 p=0.377) — **not spatially localized** |
| E58 Stage 1b | Causal: zero the octant containing each subject's own lesion centroid vs. the farthest octant | No significant paired difference (p=0.271) — **not lesion-relative either** |
| E58 Stage 2 | Causal: substitute a different subject's real bottleneck (size-matched or random) instead of zeroing | Random donor significantly **worse** than zero (p=0.026) — initially looked like a real "wrong context is misleading" effect |
| E58 Stage 2b | Same substitution protocol, re-run on E48's own checkpoint (ruling out a checkpoint artifact) | Effect reproduces, more strongly (donor ~2× as damaging as zero, p<0.0001) — confirms the substitution effect itself is real, not an artifact of which checkpoint was used |
| E58 Stage 3 | Confound test: mean-bottleneck ablation (generic, in-distribution-aggregate, no wrong subject-specific content) | Mean ablation ≈ zero, significantly **less** damaging than donor substitution (p<0.0001) — matches the signature of a known, already-published measurement confound (out-of-distribution ablation, Li & Janson NeurIPS 2024), **not** a genuine semantic compatibility finding |

**What's ruled out**: spatially-targeted mechanisms (crop/route/attend to a specific coarse region) — consistent with why E46 (attention routing) and E55 (dual-resolution crop) both landed flat-to-negative. Cross-subject "representation compatibility" as a genuine, novel, exploitable phenomenon — the direct confound check did not survive.

**What's still open**: *why* the bottleneck's real, causal, size-dependent effect exists. No architecture tried has actually targeted this correctly, because no diagnostic has identified what "correctly" would mean.

---

## Part 2 — Measurement corrections: real, orthogonal to the mechanism question

| Correction | Established in | Effect |
|---|---|---|
| Pooled Dice inflates apparent scores vs. per-subject Dice | E27 (0.9063 vs 0.8872 for baseline A), E56 (confirmed across all 6 re-scored mechanisms, gap 0.85–2.4pp) | Every headline number in E44–E55 was measured in the more favorable of two conventions |
| The canonical baseline anchor (0.9063) was never itself variance-checked | E56 (traced to seed 0 of a 4-seed EGGO-M run; other 3 seeds: 0.9064, 0.9028, 0.9020, mean 0.9044) | The +1pp target should be 0.9144 (per-subject-corrected: 0.8942), not the originally-used 0.9163 |
| Single-seed Dice claims are unreliable at the effect sizes this project produces | E49's own multi-seed check (CCABA: seed 0 alone looked like +0.51pp; 3-seed mean was +0.31pp, tied with D4-only) | Established the mandatory ≥3-seed policy, applied from E50 onward |

These are real, disclosable, and improve every future measurement — but per the explicit point already made, **none of them produce a novel algorithm on their own.**

---

## Part 3 — Every mechanism attempt, E44–E55: full audit table

| Phase | Mechanism | Level | Novelty status | Checkpoint(s)/seeds | Pooled Dice | Per-subject Dice | Paired sig. vs. baseline | Why it failed / what it ruled out |
|---|---|---|---|---|---|---|---|---|
| E44 | RCGW (dynamic auxiliary-loss weight, 2 formulations) | Objective | Adjacent to GradNorm-family MTL literature (disclosed) | 1 seed (pilot) | Killed (instability, then flat) | not measured | — | Raw-ratio formulation unstable; saturating-fixed formulation flat (−0.35pp vs D4-only reference). Closed the "adaptive auxiliary weighting" family. |
| E45 | D4+D8 (new bottleneck-resolution deep-supervision head) | Architecture | Standard technique (deep supervision, populated in literature) | 1 seed | 0.9110 | **0.8986** | **YES (paired-t p=0.041, Wilcoxon p=0.0008)** — the only cleanly double-significant result | Real effect, but never reconfirmed on 3 seeds after the mandatory policy was established. Genuinely still open (see Part 4). |
| E46 | Attention gate on enc1 skip, conditioned on bottleneck | Architecture | Standard technique (Attention U-Net, disclosed) | 1 seed | 0.9102 | 0.8949 | Wilcoxon only (p=0.0059), paired-t p=0.078 | Gate learned a real, non-degenerate spatial pattern (psi diagnostic) but effect not robust across both tests. Motivated E47's causal audit, which found the gate's own effect is not boundary-localized. |
| E47 | Causal audit of E46's gate (boundary vs interior clamp) | Diagnostic, not a mechanism | — | E46's checkpoint | — | — | — | NULL (p=0.808). Closed the boundary-routing hypothesis causally. |
| E48 | Causal audit of full bottleneck (zero ablation vs. lesion size) | Diagnostic, not a mechanism | — | E46's checkpoint | — | — | — | Reversed, significant finding (ρ=−0.454, p<0.001) — small lesions depend more on the bottleneck. Motivated E49–E55. |
| E49 | CCABA (bottleneck amplification, conditioning function fit to E48's real causal data) | Architecture | Checked against 2025-2026 literature (S³-Mamba, SvANet, CausalX-Net) — no direct match for calibrating a gate to measured causal data | 3 seeds | 0.9114/0.9075/0.9093 (mean 0.9094) | 0.8968/0.8918/0.8924 (mean **0.8937**) | Mixed: seed0 Wilcoxon-only, seed1 none, seed2 Wilcoxon-only | 3-seed mean tied with D4-only on pooled Dice; per-subject confirms no seed reaches the E45-level significance pattern. Established the mandatory 3-seed policy via its own seed-0-was-noise discovery. |
| E50 | IECG (live, jointly-trained causal-sensitivity gate, replacing CCABA's static curve) | Architecture | Checked directly — no prior work trains a live causal-sensitivity gate jointly with the task (closest candidates were post-hoc frozen-model diagnostics) | 3 seeds | 0.9081/0.9030/0.9084 (mean 0.9065) | 0.8937/0.8789/0.8899 (mean **0.8875**) | None of 3 seeds significant | Worse than CCABA on all 3 matched seeds despite being bigger/more novel/more expensive. First clean evidence that "more ambitious" ≠ "more effective" on this setup. |
| E51 | CCAG (CCABA + attention gate combined) | Architecture | Combination of two independently-verified mechanisms | 3 seeds | 0.9101/0.9092/0.9092 (mean 0.9095) | 0.8926/0.8911/0.8938 (mean **0.8925**) | Mixed: seed0 none, seed1 Wilcoxon-only, seed2 Wilcoxon-only | Did not compose additively — statistically indistinguishable from CCABA alone (p=0.919 on pooled). Consistent with E47's own finding that the gate's effect is diffuse. |
| E52 | ASR gradient-calibrated (re-attempt of a pre-pivot killed mechanism, E34) | Objective | Not a new idea — re-testing an old one with corrected calibration | 0 (killed at smoke test) | — | — | — | Confirmed a 43× gradient blowup explained E34's original collapse, but fixing the magnitude alone was insufficient — still collapsed. Isolated the cause to the term's *direction*, not magnitude. |
| E53 | ASR + curriculum warmup (delay the E52 term until after early training) | Objective | Direct fix motivated by E52's own diagnosis | 1 of 3 seeds (stopped by user before completion) | 0.8969 (seed 0 only) | not measured | — | Incomplete — smoke test passed (mechanism no longer collapses), but the one completed seed landed below baseline. No valid conclusion; would need 2 more seeds to judge. |
| E54 | A96 (whole-volume training at 96³, no architecture change) | Data/resolution | Not novel by itself — the "obvious next experiment" E29 itself proposed and never ran | 1 seed | 0.9066 | **0.8981** | **YES (paired-t p=0.022, Wilcoxon p=0.0006)** — the other cleanly double-significant result | Confirmed E29's own prediction (moderate, not dramatic gain) on pooled Dice, but the per-subject/E56 audit shows this may be a real, significant effect that was judged "flat" only because of the pooled-Dice/single-seed measurement problem. Never reconfirmed on 3 seeds. |
| E55 | Dual-resolution local refinement (differentiable soft-centroid crop + fusion, gradient verified end-to-end) | Architecture | Checked directly against 2025-2026 literature — found 3 close-to-exact matches during scoping (uncertainty-crop-refine, foveated acquisition, invertible downsampling) before landing on this specific design; still judged narrowly defensible at the time | 1 seed | 0.9007 | 0.8888 | Wilcoxon only (p=0.0013), paired-t not significant (p=0.504) | Below baseline on pooled Dice. All 6 pre-training verification checks passed (the mechanism itself worked as designed) — it simply didn't help, and the per-subject data doesn't rescue it either. |

---

## Part 4 — The one genuinely open thread this map surfaces

**E45 and E54 are the only two mechanisms in the entire post-pivot arc with a doubly-significant (both paired-t and Wilcoxon) per-subject result — and neither has ever been confirmed on 3 seeds.** Every other mechanism either failed outright or showed at most a single-test-significant, noisier signal. This was flagged once (E56's own recommendation) and not yet acted on, pending the novelty question this session worked through separately (both mechanisms are individually non-novel — D4+D8 is standard deep supervision, A96 is a whole-volume resolution increase — so a positive 3-seed result would be a measurement-driven confirmation of an existing technique, not a new algorithm).

This is the actual, concrete, low-cost next step this evidence map identifies — not a blind E59 guess, but finishing an already-flagged, already-partially-run analysis:

1. **3-seed confirmation of E45 (D4+D8) on per-subject Dice** — architecture and training script already exist, just needs 2 more seeds + re-scoring.
2. **3-seed confirmation of E54 (A96) on per-subject Dice** — same; script already exists at `experiments/exp_e12_eggo_m/e54/train_e54_a96_resolution.py`.

If either survives 3-seed confirmation with significance, that is a real, defensible, properly-measured finding — the project's actual empirical result, recovered through correcting measurement rather than inventing a ninth mechanism. If neither survives, that closes the last dangling thread from the entire E44–E55 arc, and the project's remaining defensible contribution is unambiguously the causal-diagnostic chain (E43→E47→E48→E58) plus the measurement-methodology finding (E56) — not an algorithmic result.

**What this map rules out as a next step**: any further single mechanism targeting "the bottleneck matters for small lesions" without first resolving *why* — E58 closed both tested explanations (spatial locality, cross-subject compatibility) and no third hypothesis has yet been proposed with comparable rigor. Guessing an E59 architecture now would repeat the exact pattern (E49→E55) that this whole audit was built to avoid.
