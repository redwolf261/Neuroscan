# Post-E43 Summary: The Complete Diagnosis-to-Mechanism Arc (E44–E53)

**Purpose**: self-contained account of everything that happened after E43 (the last item in the pre-pivot mechanistic-investigation arc). Written as a single reference so nothing has to be reconstructed from scattered PHASE docs later.

**Where E43 left off**: the bottleneck carries real, reproducible representation change tied to the coarse-resolution boundary problem — but that change was **not** shown to be spatially localized to boundary regions after controlling a confound. A correlational null on localization, not a mechanism.

**The user's explicit instruction after E43**: stop open-ended mechanism-hunting. Take the one validated finding (D4 deep supervision works, +0.33pp) and go: **one algorithmic modification → controlled training → ≥1.0 percentage point Dice gain, or stop.**

---

## Part 1: The causal-diagnostic chain (E44–E48)

### E44 — Relative-Convergence-Gap Weighting (RCGW): killed, twice

First real post-pivot attempt: replace D4's fixed auxiliary loss weight with a dynamic, per-epoch weight computed from the model's own training trajectory.

- **Formulation 1** (raw ratio of relative convergence): grew unboundedly, caused a real training collapse at epoch 6 (val Dice 0.71 vs. reference 0.85). Killed.
- **Formulation 2** (saturating/bounded transform of the same signal): fixed the instability completely, but produced a flat-to-slightly-negative result (−0.35pp mean vs. reference). Killed.

**Verdict**: the "adaptive auxiliary weighting" family closed. No signal beyond the already-tuned fixed weight.

### E45 — D4+D8 combined deep supervision: real but sub-threshold

Added a new auxiliary head at the bottleneck itself (8³ resolution), extending the same geometric progression D4 already established (coarser supervision helps). New architecture `UNet3D_v4`. Calibrated by gradient magnitude (caught a 2.18× mismatch from value-only calibration, rejected, recalibrated).

**Result**: 0.9110 best Dice, single seed. **+0.47pp** over canonical baseline (0.9063), **+0.14pp** over D4-only (0.9096). Real, clean training, but below the +1.0pp bar.

### E46 — Bottleneck-conditioned attention gate: real but sub-threshold

Different lever: architecture-side (not supervision-side) change. An Attention-U-Net-style gate on the `enc1` skip connection, conditioned on the bottleneck, so coarse context could modulate which fine-resolution features the decoder emphasizes. New architecture `UNet3D_v5`. Ablation-safety verified bit-for-bit (gate forced to identity reproduces v3 exactly).

**Result**: 0.9102 best Dice, single seed. **+0.39pp** over baseline, **+0.06pp** over D4-only (statistical tie). The gate demonstrably learned a real, non-degenerate spatial pattern (psi diagnostic), but that didn't translate into a decisive Dice gain.

**Pattern emerging**: two structurally distinct mechanisms (new head, attention routing) both landed in the same **+0.3–0.5pp band**.

### E47 — Causal routing audit: NULL (confirms E43 causally)

Used E46's trained gate as a genuine causal instrument (not a new training run): clamped the gate at boundary voxels vs. matched-count interior voxels, measured the Dice-drop difference.

**Result**: drop_difference = +0.00025 (essentially zero), permutation p=0.808. **NULL.** Clamping the routing gate at the boundary does not hurt more than clamping it at the interior. This causally confirms E43's earlier correlational null: boundary-routing is not the mechanism behind the ~9pp ceiling below perfect Dice.

### E48 — Bottleneck encoding-deficiency audit: REVERSED, significant finding

Tested a different hypothesis: does the bottleneck fail to *encode* useful signal for small lesions specifically? Pre-declared prediction: severing the bottleneck should hurt **large**-lesion subjects more (they have more signal there to lose).

**Result**: the bottleneck matters enormously overall (severing it drops mean Dice from 0.89 to 0.57). But the correlation with lesion size is **strongly negative, opposite the hypothesized direction**: Spearman ρ(native_size, drop) = **−0.454**, p<0.001, n=125, permutation-confirmed. **Small lesions depend MORE on the bottleneck, not less.**

This is the first genuinely new causal mechanism found in the entire diagnostic chain — a real, statistically strong, counterintuitive result, not another null. Interpretation: large lesions are locally identifiable from fine features alone; small lesions are locally ambiguous and depend on global/coarse context the current architecture under-serves relative to how much small lesions need it.

**This finding motivates every mechanism that follows.**

---

## Part 2: Mechanisms built on E48's causal finding (E49–E51)

### E49 — CCABA (Causally-Calibrated Adaptive Bottleneck Amplification)

The first mechanism whose conditioning function is a **measured causal curve**, not an assumed prior — fit directly to E48's real 125-subject ablation data. A literature scan (2025–2026) confirmed no prior work does this: every size-aware mechanism found assumes size matters and designs around that assumption; every causal method found explains/attributes but doesn't feed back into architecture design. New architecture `UNet3D_v6`. Gradient-calibrated (`lambda_frac`, rejected a 25.4× value-matched blowup).

**Single-seed result**: 0.9114 best Dice. +0.51pp over baseline — the best of four post-pivot attempts.

**Then: the project's first multi-seed variance check** (2 additional seeds trained). Result: **[0.9114, 0.9075, 0.9093], mean 0.9094, std ≈0.19pp, 95% CI [0.9046, 0.9142]**. The headline +0.51pp was **favorable seed noise**. True effect: **+0.31pp, statistically tied with D4-only.**

**This is the single most consequential methodological finding in the post-pivot arc**: it retroactively casts doubt on every prior single-seed result (E44–E46) and establishes a new mandatory policy — **≥3 seeds required for any future GO/KILL decision or Dice claim.**

### E50 — IECG (Internally-Estimated Counterfactual Gating)

A qualitatively bigger, more novel mechanism than CCABA: instead of a static externally-fit curve, a **live, jointly-trained** causal-sensitivity estimate — an on-the-fly bottleneck ablation replayed through the decoder every training step, with a small head learning to predict the ablation's own effect, used to gate the real (non-ablated) bottleneck. Genuinely differentiated from all 2025–2026 literature checked (counterfactual MoE routing analysis and TRACE-Seg3D, both confirmed to be post-hoc frozen-model diagnostics, not live trainable components). New architecture `UNet3D_v7`. Evaluated on 3 seeds **from the start**, per the new policy.

**Result**: **[0.9081, 0.9030, 0.9084], mean 0.9065**, +0.02pp over baseline — essentially zero, and **worse than CCABA on all 3 matched seeds** despite being bigger, more novel, and ~2.5× more expensive to train per step. The mechanism's internal consistency check passed (the sensitivity head did learn to predict real causal sensitivity accurately) — it worked as designed internally, it just didn't help the actual task.

**Strong evidence that "more novel/more ambitious" does not track "more effective" on this setup.**

### E51 — CCAG (CCABA + Attention Gate combined)

The pre-declared **last single-architecture-family attempt**: combine the two mechanisms already independently verified as real and non-conflicting (CCABA amplifies the bottleneck; the attention gate routes the `enc1` skip — different tensors, designed without awareness of each other). New architecture `UNet3D_v8`. All three ablation-safety checks (full-off, CCABA-only, gate-only) passed bit-for-bit; param overhead exactly additive.

**Result** (3 seeds from the start): **[0.9101, 0.9092, 0.9092], mean 0.9095**, std ≈0.05pp (tightest of any condition), 95% CI [0.9082, 0.9108]. **+0.32pp over baseline, −0.01pp vs. D4-only (dead tie), +0.01pp vs. CCABA alone (p=0.919, statistically indistinguishable).**

**The two mechanisms did not compose additively.** Consistent with E47's own causal finding that the gate's effect is diffuse, not localized — it had no obvious lever to synergize with CCABA's size-conditioned amplification.

**This closes the architecture-level lever.** Six mechanisms tried (E44 killed, E45/E46 single-seed +0.3–0.5pp, E49/E50/E51 properly multi-seed evaluated), with a clear trend: more novelty and combination did not beat the simplest validated approach (D4-only), and going bigger (IECG) or combining (CCAG) trended flat-to-worse, not better.

---

## Part 3: The objective-level detour (E52–E53)

At this point the recommendation was to stop and write up the causal-diagnostic chain + variance-correction finding as the paper's contribution. The user instead asked for one more genuinely different lever: **objective-level (loss-function), not architecture-level.**

### E52 — ASR gradient-calibrated re-attempt: killed at smoke test

Revisited a **pre-pivot** experiment, E34 (Adaptive Size-Reweighting), which had collapsed training catastrophically (0.71–0.74 Dice) back in the E1–E43 arc. E34's `lambda_cw` had been calibrated by **loss value** only, never gradient-checked — E34 is the likely origin of this project's mandatory gradient-calibration safeguard, but that safeguard was never applied retroactively to E34's own failure.

Direct measurement: E34's original lambda would have caused a **43× gradient-magnitude blowup** — likely the actual cause of the original collapse. Recalibrated to the gradient-matched value (0.1101, target ratio 1.0) and re-ran the pre-declared 2-epoch smoke test.

**Result**: still collapsed — same degenerate "predict everything as tumor" pattern (val Dice 0.10 at epoch 1, precision 0.05, recall 1.0). **Fixing the gradient magnitude was not sufficient.** Direct comparison showed `seg_loss`/`boundary_loss` values nearly identical to healthy conditions at epoch 1, isolating the cause: the component-weighted term's **direction**, not its magnitude, was fighting the model before it had learned basic large-lesion competence.

**Killed at the smoke-test gate** — no full run on an already-failing result. This is the seventh mechanism killed/null since the pivot, and the first at the objective (loss) level rather than architecture level.

### E53 — ASR with curriculum warmup: mechanism fixed, result still sub-baseline

Directly motivated by E52's own diagnosis: keep the exact same gradient-matched loss term, but **ramp it in gradually** (linear multiplier, 0 through epoch 8, full strength by epoch 20 — window chosen from a real reference training curve, not tuned on this phase's results) rather than applying it from epoch 0.

**10-epoch smoke test**: passed cleanly. Epochs 1–9 behaved exactly like a healthy baseline run (no collapse), and the ramp's onset at epoch 10 caused no disruption — the diagnosed fix worked as intended. Cleared to commit to the full 3-seed run.

**Full run**: only **seed 0 completed** before the user stopped the remaining runs (seeds 1 and 2 were killed mid-training, no result). Seed 0's result: **best val Dice 0.8969** — no instability anywhere, the curriculum ramp completed cleanly to full strength — but **below both the canonical baseline (0.9063) and D4-only (0.9096)**.

**Status: incomplete, stopped by user request.** The curriculum fix appears to have solved the *instability* problem (no more collapse) but, on the one seed that finished, did not produce a *competitive* result — "stable but mediocre" rather than "stable and better." No 3-seed mean, no confidence interval, and no final verdict can be drawn — this is explicitly not a completed evaluation and should not be cited as a Dice number for the mechanism.

---

## Where this leaves the project

**Numbers, all against canonical baseline 0.9063 / D4-only 0.9096:**

| Phase | Mechanism | Level | Seeds | Result | vs. baseline |
|---|---|---|---|---|---|
| E44 | RCGW (adaptive λ schedule) | Objective | 1 (pilot) | Killed (instability, then null) | — |
| E45 | D4+D8 (new bottleneck head) | Architecture | 1 | 0.9110 | +0.47pp |
| E46 | Attention gate (enc1 routing) | Architecture | 1 | 0.9102 | +0.39pp |
| E49 | CCABA (causal-calibrated amplification) | Architecture | 3 | 0.9094 mean | +0.31pp (tied w/ D4-only) |
| E50 | IECG (live counterfactual gate) | Architecture | 3 | 0.9065 mean | +0.02pp (worse than CCABA) |
| E51 | CCAG (CCABA + attention combined) | Architecture | 3 | 0.9095 mean | +0.32pp (tied w/ CCABA, D4-only) |
| E52 | ASR gradient-calibrated | Objective | 0 (smoke-killed) | Killed | — |
| E53 | ASR + curriculum warmup | Objective | 1 of 3 (incomplete) | 0.8969 | **−0.94pp (below baseline)** |

**Seven mechanisms fully evaluated and closed** (E44, E45, E46, E49, E50, E51, E52); **one incomplete** (E53, stopped by user before 3-seed evaluation finished). None has cleared the +1.0pp bar. None of the three properly multi-seed-evaluated mechanisms (E49, E50, E51) beats D4-only with statistical confidence. The clearest trend across all seven completed attempts: bigger, more novel, or combined mechanisms trended flat-to-worse, not better, than the simplest validated baseline (D4-only, +0.33pp, established pre-pivot).

**What is solid and durable regardless of the Dice outcome:**

1. **The causal-diagnostic chain (E43→E47→E48)**: two genuine causal nulls, using real interventions on a trained model (not correlational proxies), followed by a real, statistically strong, initially counterintuitive finding — small lesions depend disproportionately on global/bottleneck context, not less.
2. **The multi-seed variance-correction finding (E49→E50→E51)**: three separate, honestly-reported instances of a single-seed headline number overstating a real effect once measured properly — a legitimate, disclosable methodological contribution, not just internal practice.

**Recommendation status**: unchanged and further strengthened by E52/E53's own results. The standing recommendation, made after E51 and reaffirmed after E52, is to stop searching for the mechanism that clears +1pp and write up the causal-diagnostic methodology and the variance-correction finding as the paper's actual contribution. E53 remains an open thread only in the narrow sense that its 3-seed evaluation was not completed — if resumed later, it would need seeds 1 and 2 re-run from scratch (they were killed, not paused) before any conclusion about the curriculum-warmup mechanism specifically could be drawn.
