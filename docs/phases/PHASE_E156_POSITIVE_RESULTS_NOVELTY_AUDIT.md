# E156 — Positive-Results Novelty Audit

**Date**: 2026-09-15
**Status**: Step 1 (recovery) complete from existing artifacts. No compute run.
**Supersedes as priority**: E154/E155 (Gate C) are parked pending this audit.

---

## Purpose

The project has already produced results exceeding the +1pp Dice bar. The question is no
longer "can we reach +1pp" but:

> Does any existing positive result contain a genuinely novel, claimable algorithmic principle?

This document recovers the exact provenance of each, per the Step-1 requirement.

---

## The three positive results — recovered

### Source of truth

`docs/phases/PHASE_E84_CONTRIBUTION_AUDIT.md` § "Strong positive results, and their honest
status" is the project's own authoritative list. It names **two**. The third candidate (E15) is
positive-but-oracle and is included because it carries the largest raw number.

---

## Master table

| | Experiment | Exact gain | Baseline | Task | Mechanism (principle, not module) | Robust? | Closest prior art | Novelty |
|---|---|---|---|---|---|---|---|---|
| **#1** | **E70 MM (4-modality input)** | **+1.75 pp** (3-seed mean, std 0.27) | FLAIR-only single-modality (per-subject **0.8842**) | BraTS, per-subject mean Dice (E56 protocol) | Supply T1/T1ce/T2/FLAIR instead of FLAIR alone — information added at the **input**, no architectural change | **YES** — 3 seeds, +1.81/+1.45/+1.98, all CIs exclude 0 | Multimodal MRI fusion; standard in every BraTS paper since ~2014 | 🔵 **Clearly established** |

**CRITICAL CONTEXT (added 2026-09-15 after full-corpus read)**: the FLAIR-only baseline was
a **self-imposed project constraint, not a scientific baseline.** The master report §1 is
explicit: the project deliberately used "only the FLAIR MRI sequence as input... to keep the
model and problem simple." E70's own results doc states the consequence plainly, and it was
pre-registered *before* the numbers existed:

> "Comparison 1 is real but is not a contribution. Adding T1/T1ce/T2 to a FLAIR-only model is
> expected to help; every published BraTS method uses all four modalities. The single-modality
> restriction was a self-imposed project constraint, not a scientific baseline."

So #1 is not merely "a known technique that happens to work." It is **the removal of an
artificial handicap the project had imposed on itself.** The +1.75pp measures how much was
being given up by the simplification, not a discovery. This is a stronger disqualification
than "not novel," and it should be stated that way.

**It also means every pre-E70 percentage-point claim in the project was measured against a
deliberately weakened baseline.** E45's +1.44pp and E54's +1.39pp are both measured against
FLAIR-only per-subject 0.8842 (verified directly from
`E45_E54_3seed_confirmation_summary.json`: seed-0 per-subject 0.8986 − delta 0.0144 =
0.8842). A mechanism that adds +1.4pp on a handicapped single-modality model has **not** been
shown to add anything on a proper 4-modality baseline — and the later 4-modality work
(E128–E141) reaches a mean Dice around 0.89 on a 3-region task, a different measurement
regime entirely.
| **#2** | **E15 centroid push** | **+3.93 pp** (0.9050→0.9443) | push=0, same frozen decoder | 20 subjects, frozen-decoder causal intervention | Move `dec1` away from the **ground-truth-selected** opposite-class centroid | **NO** — not a training method; n=20; single condition | Prototype/metric learning; decoder-aware repr. learning; Jacobian-guided optimization — all listed as occupied in `PHASE_E15_PRIOR_ART_AUDIT.md` | 🔴 **Contradicted — oracle** |
| **#3** | **E54/A96 (96³ resolution)** | **+0.38 pp** marginal, CI [−0.00, +0.79] | MM at 64³ | BraTS, per-subject | Increase input resolution 64³→96³ | **NO** — CI includes 0; 3-seed A96 0.8944 vs 0.8942 | Resolution scaling / patch-size tuning — routine engineering | 🔵 **Established + not robust** |

### #4 — Deep supervision (added 2026-09-15 at user prompt)

**Two different experiments are both loosely called "deep supervision." They must not be
conflated, and the earlier draft of this audit wrongly omitted both.**

| | Experiment | Exact gain | Baseline | Task | Mechanism | Robust? | Prior art | Novelty |
|---|---|---|---|---|---|---|---|---|
| **#4a** | **E25 D4-only deep supervision** | **+0.33 pp** (0.9096 vs 0.9063), pooled | A (no aux heads) | BraTS 64³, single seed | Aux head at 1/4-res decoder stage, discarded at inference | **NO** — single seed; **none of 6 pairwise comparisons reach significance (all p>0.07)** | Deep supervision — standard published technique | 🔵 **Established** |
| **#4b** | **E45 D4+D8 supervision** | seed 0: **+1.44 pp per-subject** (+0.47 pp pooled); 3-seed per-subject **0.8939** | corrected per-subject 0.8842 | BraTS 64³, 3 seeds | Deeper aux supervision (D4+D8) | **NO** — seed 0 doubly significant, but 3-seed mean **misses** the 0.8942 target | Same family | 🔵 **Established** |

**CORRECTION (2026-09-15)**: an earlier draft of this table listed E45 only at its **pooled**
+0.47 pp and omitted that its **seed-0 per-subject result was +1.44 pp, doubly significant**
(paired-t p=0.041, Wilcoxon p=0.00075). That omission understated the result and was wrong.
E56's re-score found **both E45 and E54 already clear the corrected +1pp per-subject target on
their single seed** — this is the "comfortable 1%" reading, and it is correct as far as seed 0
goes.

**What the 3-seed confirmation then showed** (verified from
`e56/E45_E54_3seed_confirmation_summary.json`):

| E45 seed | per-subject mean | Δ vs baseline | paired-t p | Wilcoxon p |
|---|---:|---:|---:|---:|
| 0 | 0.8986 | **+1.44 pp** | **0.041** | **0.00075** |
| 1 | 0.8878 | +0.36 pp | 0.588 | 0.085 |
| 2 | 0.8952 | +1.11 pp | 0.088 | 0.0010 |
| **3-seed mean** | **0.8939** | **+0.97 pp** | — | — |

`"confirmed_corrected_target": false`, `"confirmed_original_target": false` — both recorded in
the artifact itself.

So the honest statement is **not** "deep supervision gave +0.47pp." It is: **deep supervision
gave a comfortable, doubly-significant +1.44 pp on seed 0, which did not hold up on seeds 1–2**
(seed 1 collapsed to +0.36 pp at p=0.588), leaving a 3-seed mean of +0.97 pp that lands
0.0003 below the pre-declared target.

The 3-seed mean missing by 0.0003 is a **coin flip against the threshold**, not a clean failure
— the project's own confirmation doc says exactly this, and refuses to call E54 "confirmed" and
E45 "not confirmed" when they straddle the line by ±0.0002. The disqualifying fact is not the
0.0003; it is **seed 1**, which shows the effect is not reliably reproducible.

**Why this does not change the verdict:**

1. **#4a was never multi-seed tested**, and its own source doc (`PHASE_E25...1B_4WAY`) states
   that *none* of the six pairwise whole-volume Dice comparisons reached significance
   (all p>0.07). The master report calls it "the only real positive result" because it was the
   **first** idea to beat baseline — chronological primacy, not statistical strength. At
   +0.33 pp it is **well below the +1pp bar** in any case.
2. **#4b (E45) was 3-seed tested and failed.** From
   `PHASE_E45_E54_3SEED_CONFIRMATION_RESULT.md`: 3-seed mean per-subject 0.8939 vs target
   0.8942 — misses by 0.0003, a margin smaller than a tenth of its own between-seed std
   (0.0055). Seed 0 was the most favourable seed; seeds 1 and 2 did not reproduce its
   doubly-significant signature (seed 1: paired-t p=0.588).
3. **Deep supervision is explicitly acknowledged as prior art** inside the project's own
   autopsy: *"This 'deep supervision' idea is a standard, published technique — using it at all
   is not the novel part of this project."*

**However — E36's autopsy contains a genuinely non-obvious mechanistic finding**, and it is the
most interesting thing in this entire audit besides the self-knowledge phenomenon:

- The intuition "D4 helps because its gradient reinforces the main gradient" is **false**.
  D4-only (best result) ends training with the **least**-aligned gradient (cos 0.25); D2-only
  (worse result) ends **most** aligned (0.82). p<0.0001 across 15 independent batches, and
  checked that it is not a gradient-shrinkage artifact.
- The explanation: D4's own loss **plateaus at ~0.15 while the main loss reaches 0.05** — the
  D4 task stays 2.74× "unsolved." So D4 keeps supplying substantial gradient in exactly the
  region the main loss has stopped attending to.
- **What is unsolved is partial-coverage voxels** — coarse cells a boundary passes through —
  at **~10× the error of full-coverage voxels** (0.30 vs 0.03).

This is a real, measured, non-obvious characterization: *auxiliary supervision helps by staying
unsolved, not by agreeing.* It is a **mechanistic finding about why a known technique works**,
not a new technique. It is closer to publishable as an analysis contribution than as an
algorithmic one, and it is **not** a positive Dice result — the Dice result it explains is
+0.33 pp, single-seed, non-significant.

### Ranking by novelty-plausibility × robustness × mechanistic clarity

| Rank | Result | Novelty | Robust | Clarity | Product |
|---|---|---|---|---|---|
| 1 | E70 MM | 0 (known) | **high** | high | **0** |
| 2 | E15 | 0 (oracle) | low | medium | **0** |
| 3 | E54/A96 | 0 (known) | none | high | **0** |
| 4a | E25 D4-only DS | 0 (known) | none (1 seed, p>0.07) | **high** | **0** |
| 4b | E45 D4+D8 | 0 (known) | none (3-seed miss) | high | **0** |

**All products are zero.** Not because the numbers are wrong — #1 is the most solid number in
the project — but because each fails a *different* gate.

Note the pattern across the whole table: **the results with real statistical backing (MM and deep supervision) are both textbook techniques, while those with interesting-sounding mechanisms (E15, and the deep-supervision autopsy) are either oracle-driven or explain a sub-1pp effect.** Robustness and novelty are perfectly anti-correlated across the project's positive results.

---

## Per-result detail

### #1 — E70 MM, the only robust number

Verified from `PHASE_E70_E71_OVERNIGHT_RESULTS.md`:

| Seed | MM vs FLAIR |
|---|---|
| 0 | +1.81 pp, CI excludes 0 |
| 1 | +1.45 pp, CI excludes 0 |
| 2 | +1.98 pp, CI excludes 0 |
| **mean** | **+1.75 pp (std 0.27)** |

Endpoint is **per-subject mean Dice** under E56's corrected protocol (principle 6 satisfied).
The system reaches 0.9200 pooled / 0.9052 per-subject; against the corrected bar (≥0.8942) it
clears by 1.10 pp.

E84's own verdict: *"a real, robust performance result — but not novel (multimodal input is a
well-established technique)."* This audit concurs. **Changed computational assumption: none.**
Adding input channels alters what information enters the network, not how any computation is
organized. There is no principle here to claim.

**Important**: this is the correct baseline for any future claim, and it is already banked. A
paper can *use* it; it cannot be *about* it.

### #2 — E15, why the largest number is the weakest candidate

`PHASE_E15_EVIDENCE_FREEZE.md` is explicit and self-critical, and its analysis is correct:

- The decoder head is `D(z) = sigmoid(wᵀz + b)`, single output channel. Therefore
  `∇_z D = D(1−D)·w` — the decoder sees **exactly one direction**, `w`.
- The 3.83× Jacobian ratio is a *local sensitivity observation*, not evidence of a
  geometry–decoder interaction. With a rank-1 gradient there is no second geometry to interact
  with.
- The opposite-class centroid was chosen **using ground-truth labels**. The intervention is
  therefore label-informed movement along the decoder's own sensitive direction.

So the honest reading — already frozen in the project's own doc — is: *supervised latent
separation can causally improve a frozen decoder's output.* That is a true statement and a
**tautology-adjacent** one: pushing representations along `w` in the correct class direction
raises the sigmoid. It is not a deployable rule, because the direction cannot be discovered
without the label.

E151 independently confirmed the decoder's large null space and failed to support the geometry
mechanism at dec3.

**Changed computational assumption: none that survives.** Per principle 2, the absence of a
paper implementing this exact oracle intervention is *not* novelty.

**Status: 🔴 closed as a primary direction.** Already retired before compute in
`PHASE_E15_PRIOR_ART_AUDIT.md`. This audit does not reopen it.

### #3 — E54/A96, both unnovel and unconfirmed

Two independent disqualifications:

1. **Not robust.** MM_A96 vs MM (i.e. resolution's *own* contribution with modality held
   constant) = +0.38 pp, bootstrap CI **[−0.00, +0.79]** — includes zero. The 3-seed
   re-test gave 0.8944 vs 0.8942.
2. **Not novel.** Resolution/patch-size scaling is routine.

The apparent early +1pp was **modality gain misattributed to resolution** — MM_A96 vs
FLAIR-only is +2.19 pp, but almost all of it is the modality expansion. This is a
baseline-mixing artifact, and exactly what principle 4 warns against.

---

## Verdict

$$\boxed{\textbf{Case B} — \text{all candidates are known methods or oracle interventions}}$$

**Scope note**: the audit covers **five** candidates, not three — E70 MM, E15, E54/A96, E25
D4-only deep supervision, and E45 D4+D8. Deep supervision was omitted from the first draft of
this document and added at the user's prompt; the omission was a real gap, since the master
report calls it "the only real positive result in the project so far." Auditing it did not
change the verdict (it is +0.33 pp, single-seed, non-significant, and explicitly acknowledged
as standard prior art in the project's own autopsy), but it did surface E36's
gradient-misalignment mechanism, recorded above.

Under the stated decision rule, Case B reopens the research frontier. But two caveats before
acting on that:

**Caveat 1 — the premise needs correcting.** The framing was that the +1pp bar is *already
satisfied* by an existing result, so no new method is needed. That is true only for **#1**, and
#1 is multimodal input — a known technique that cannot carry a paper's contribution. So the
performance bar is met by something unclaimable. The project does **not** currently hold a
positive result that is both robust and novel. There is no winner to freeze.

**Caveat 2 — a fourth candidate exists and is not in the three.** E84 lists a *second* strong
result that is not a Dice gain at all:

> **Bottleneck causal self-knowledge** — the network's bottleneck representation predicts its
> own ablation sensitivity, held-out ρ≈0.87 across two independent validations.

Confirmed in `PHASE_E70_E71_OVERNIGHT_RESULTS.md` § 3: held-out Spearman **+0.701** (n=125,
permutation p<0.001), partial ρ controlling lesion size **+0.904** (p=4.1e-47), auxiliary head
trained on a disjoint 200-subject set.

This is the project's **strongest validated scientific phenomenon** and it is genuinely
non-obvious. Its status is the mirror image of #1: **novel-but-not-performance**, where #1 is
**performance-but-not-novel**. E84 records that every attempt to convert it into an algorithm
(routing, reweighting, spatial gating — E71/E72/E73) has failed.

---

## Recommended next step

Do **not** immediately reopen new-hypothesis generation. Case B's honest consequence is
narrower than "invent something new":

The project holds one robust performance result that is not claimable, and one strongly
validated phenomenon that is not yet performance. A Q1-style contribution most plausibly comes
from **converting the self-knowledge phenomenon into a performance result** — that is the only
asset with unexhausted novelty.

That is, notably, the *same* structural question E147/Gate C was circling (can an internal
signal be read early enough to change computation), but anchored to a phenomenon with
**ρ=0.904 held-out validation and an intact generator**, rather than to a variable whose
generating script does not exist.

Before any compute, the required step is a targeted prior-art audit of the self-knowledge
phenomenon's *mechanism* (Step 3/4 of the workflow), since E71/E72/E73 failed on
**implementation** and it has not been established whether the failure was the principle or the
three specific levers tried.

---

## Open items / honesty notes

- ~~The E84 audit is dated pre-E107; E108–E153 not re-scanned.~~ **SWEEP COMPLETED
  2026-09-15.** Every E108–E153 outcome is a kill, a null, or a diagnostic — no positive Dice
  result was found. The closest approaches were all negative: E134's four ET/TC fixes
  (−2.40 pp, 0.00 pp, −0.14 pp, −0.76 pp), E137 conditional readout (−0.89 pp), E141
  evidence-conditioned supervision (−0.37 pp, shuffled control −0.63 pp), E128 capacity
  (+0.139 pp vs AMP-matched control, below bar), E131 rank-gated pooling (null). E150's readout
  gap was real but worth only ~+0.07 pp overall. **Case B stands.** The table is complete.
- E15's +3.93 pp is on **n=20 subjects**, not the 125-subject validation set — it is not
  comparable to #1's or #3's numbers and must never be tabulated alongside them as if it were.
