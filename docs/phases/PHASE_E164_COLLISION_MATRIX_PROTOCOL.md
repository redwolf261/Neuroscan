# E164 — Collision-Matrix Protocol

**Date**: 2026-09-15
**Status**: Protocol specified. Pre-registered. **Not yet executed.** No compute licensed.
**Follows**: E162 (pivot premise not supported), E163 (Vaswani Track 2.0 charter)

---

## Context

Five consecutive prior-art audits returned OCCUPIED: query-based segmentation (E),
modality informativeness (K), predictive self-knowledge, effective-rank regularization,
causal-quantity-as-training-signal. E162 derived the structural law:

$$\boxed{\text{On a well-studied task, a measurement's informativeness predicts its prior-art density.}}$$

Measurements chosen because they explain a model's failures are salient to everyone working on
that task. That is why five audits died from one cause, not five.

The proposal this document specifies: **extract what 2026 papers themselves admit is unsolved,
and collide that against NeuroScan's findings** — locate a hole the field concedes rather than
one we guess.

---

## Part 0 — The critique that shapes the design

**The naive collision matrix is the old failure loop inverted, not escaped.**

Failed loop (E163): `NeuroScan failure → generalise → search → occupied`
Naive matrix: `field's hole → match to our findings → search → occupied`

The direction of travel is inverted; the terminal filter is not. The matching step anchors on
our findings — precisely the set E162 proved is densely occupied. **A collision with an occupied
finding re-derives the occupancy; it is not a lead.**

All seven themes gathered from the 2026 search (small lesions, domain shift, missing modalities,
boundary ambiguity, benchmark-vs-clinical gap, evaluation heterogeneity, adaptive-mechanism
proliferation) will collide with something we measured. That is seven false positives that will
*feel* productive.

### The one thing that escapes the law

> "the field keeps ADDING adaptive mechanisms (dynamic routing, adaptive resolution, uncertainty
> fusion, test-time adaptation, Mamba/SSM, foundation models) yet the same failures persist."

That is not a limitation extraction. It is a **negative claim about a method family**, and it is
structurally identical to what NeuroScan proved on its own cohort three independent ways:

| Route | Evidence |
|---|---|
| E71 | ρ(ablation-sensitivity, skip-suppression-tolerance) = **−0.290**, p=0.001 — **sign backwards** from what allocation requires |
| E147 | demand↔capacity **inverse**, partial ρ=−0.496; **74/88 subjects served by rank ≤4 of 256** ⇒ no capacity scarcity exists |
| E72/E73 | loss-weighting and spatial gating on that signal, both inert |

The structural law prices **measurements**, because measurements that explain failures get built
on. It does **not** price **evidence that a popular family is inert**, because nobody publishes
"our mechanism does nothing." Occupancy density is driven by publication incentive, and the
incentive on negative results is inverted.

**That asymmetry is the only exploitable surface on the board.** The protocol runs the matrix but
targets the inertness claim as primary.

---

## Part 1 — Verified arithmetic that constrains everything

Recomputed this session from `experiments/exp_e12_eggo_m/e160/E160_L_per_subject.json`
(n=125, full native resolution, 4-modality, 3-region):

| Quantity | Value |
|---|---|
| Cohort 3-region mean Dice | **0.8595** |
| Headroom | **15-subject tail** (ET/TC bottom deciles are the same subjects, Jaccard 0.733) |
| On the 15 | ET 0.227, TC 0.186, **WT 0.824** — tumour IS found; failure is sub-partition |
| Lift ET&TC on the 15 → 0.40 | **+1.94pp** |
| Lift → 0.50 | +2.59pp |
| Lift → 0.60 | +3.29pp |
| **E140's per-subject oracle (+0.0154 ET on the 15)** | **+0.062pp overall** |

That last row **retroactively explains E137 / E140 / E141**: they were *arithmetically incapable*
of clearing 1pp regardless of mechanism.

$$\boxed{\text{Any candidate helping only SOME of the 15, or helping them PARTIALLY, cannot clear the bar.}}$$

This check is free, is pure arithmetic, and **must run before any prior-art audit.**

### Two attractive hooks closed by our own data (this session)

1. **The tail is NOT a small-lesion problem.** ET sizes of the 15 failing subjects:
   `[85, 123, 125, 536, 540, 820, 2093, 2412, 2615, 3003, 6769, 11742, 12841, 19677, 39207]` —
   9/15 above 2000 voxels, 4/15 above 10000. Cohort median ET is 17,529.
2. **Dice-instability-on-tiny-targets does not apply here.** The reported "1 voxel ≈ 6% DSC at
   3 mm" requires targets of ~10 voxels. Only **3/125** subjects have ET<500 voxels.

Both were among the strongest-looking hooks in the 2026 search. Both are dead on our cohort.

---

## Part 2 — Extraction schema (deterministic)

"Stated limitation" is the wrong unit: a Future Work sentence is often the authors' own next
paper. Record per paper, and classify by **rule, not judgement**:

| Field | Definition |
|---|---|
| `paper_id` | arXiv/DOI + venue + month |
| `claimed_contribution` | verbatim novelty sentence (abstract/intro only) |
| `limitation_verbatim` | exact quote, no paraphrase |
| `limitation_locus` | abstract / results / discussion / future_work / appendix |
| `is_quantified` | does the paper give a NUMBER for the residual failure? |
| `author_next_step` | does the paper name the method it would try? |
| `same_group_followup` | does a later/concurrent overlapping-author paper address it? |
| `mechanism_stated` | does it explain WHY the failure occurs, or only THAT it occurs? |
| `failure_cohort_size` | n of the failing subset, where reported |
| `our_evidence_bearing` | which E-number speaks to this (must cite `EXPERIMENT_INDEX.md`) |
| `class` | A/B/C/D, derived below |

### Classification rules

- **(b) Boilerplate → DISCARD**: `locus == future_work` ∧ ¬`is_quantified` ∧ ¬`mechanism_stated`
- **(c) Self-claimed → DISCARD**: `author_next_step` named, **or** `same_group_followup == true`
- **(d) Requires scale → FLAG** for Part 4 (multi-site, federated, foundation-model pretraining,
  >1 GPU-week). Not discarded yet.
- **(a) Genuine hole → KEEP**: `is_quantified` ∧ ¬`mechanism_stated` ∧ `author_next_step == null`

This **inverts the usual reading**: a confident future-work paragraph is evidence of
**occupancy**; a number in a results table with no accompanying story is evidence of a **hole**.
The rule is mechanical, so it applies before you know whether you like the answer.

`failure_cohort_size` is cheap and high-yield: our entire headroom is a 12% tail. Papers
reporting aggregate Dice only cannot speak to tail behaviour at all — and a field-wide inability
to report tail statistics is itself a finding, feeding Part 4 and the fallback deliverable.

---

## Part 3 — Matrix axes and pre-registered decision rule

**Not** `limitation × our-finding`. That construction scores *similarity*, and similarity to our
findings is exactly what inherits their occupancy.

**Axis 1 — Field status**: `unstated` / `stated-boilerplate` / `stated-quantified-unexplained` /
`stated-and-claimed`
**Axis 2 — Our evidence**: `contradicts` / `explains` / `corroborates` / `silent`
**Axis 3 — Reachability**: `reachable` / `borderline` / `unreachable` (Part 4)

| Cell | Definition | Action |
|---|---|---|
| **COLLISION** | `corroborates` × `stated-and-claimed` | **Dead. Log and close** so it is not re-proposed. |
| **HOLE** | `stated-quantified-unexplained` × `explains` | Advance |
| **CONTRADICTION** | anything × `contradicts` | Advance ← the inertness claim lands here |
| UNSTATED | any × `unstated` | Not scored as promising (Part 4) |

**Pre-registered rule: only CONTRADICTION and HOLE advance to prior-art audit.** If the matrix
yields zero of both, Part 6's stop rule fires **immediately** — no second pass with looser
definitions.

---

## Part 4 — Feasibility gate, moved EARLY, and the anti-structural-law filter

### 4a. Feasibility runs BEFORE prior-art audit

This **reverses E163's charter ordering** (its Phase 2 = prior-art, Phase 3 = feasibility). The
audit is the expensive step in the only currency available — attention and session count — and
it has a **5/5 kill rate**. Auditing an unreachable candidate costs the same as a reachable one
and can never pay.

Hard gates, not scores:

1. **Hardware**: must train within 8 GB VRAM at 128³ / 4-modality / 3-region — already near the
   ceiling. Larger context, 3D transformers at full volume, or foundation-model finetuning: out.
2. **Data**: answerable on locally-held BraTS. Multi-site / federated / external-validation /
   new-annotation: out — unreachable this cycle, not uninteresting.
3. **Seeds**: testable in ≤3 seeds × ≤1 training run each within the remaining timeframe.
4. **Measurability** (the one usually skipped): a ≥1pp per-subject gain must be *detectable*,
   computed arithmetically per Part 1. Most tail-targeted candidates die here, in ten minutes,
   for free.

### 4b. Frequency is not the discriminator — the derivative is

Requires a **longitudinal slice** (~10 papers from 2022/23) alongside the 2026 corpus:

| Mentions | Remedies | Reading |
|---|---|---|
| High | proposed, and residual improving | Actively worked → **discard** |
| High | **cycling by fashion, residual flat** | **The interesting cell** |
| High | none, just recurring acknowledgement | Structural (e.g. inter-rater ceiling) → usually discard |
| Low / none | — | Unmeasurable or uninteresting → **discard by default** |

A problem mentioned by 40 papers in 2026 that was mentioned by 40 in 2022, with a different
fashionable remedy each year and no monotone movement in the quantified residual, is **not being
solved**. That is the "field keeps adding mechanisms yet failures persist" observation made
operational.

**Direct answer to "is a hole nobody mentions more promising?" — No. Treat `unstated` as a
negative signal.** In a field with thousands of active groups on a decade-old benchmark, silence
usually means the problem isn't real, isn't measurable, or isn't publishable. The romantic
reading requires thousands of better-equipped groups to have missed something visible from a
laptop; our own 5/5 base rate argues against it.

This filter is also the formal answer to the structural law: the law prices measurements, not
inertness claims.

---

## Part 5 — Audit format

Reuse E161's format exactly: state the **exact novelty claim**, pre-register the
**distinguishing axes** before searching, then score each axis 🔴/🟡/🟢 individually, and record
the pre-audit expectation so the result cannot be read as confirmation.

---

## Part 6 — Stop rule (numbers fixed before any paper is read)

- **Corpus**: 25–30 papers from 2026 + ~10 from 2022/23. **Hard cap** — not "until saturation."
  Saturation arguments are how 20–50-experiment branches happen.
- **Candidates advancing to audit**: maximum **3**. Ranked by Part 3, ties broken by Part 4.
  Excess candidates are logged as closed-unaudited.
- **Audit rounds per candidate**: maximum **2**. Round 1 = direct claim; Round 2 = *one*
  narrowing if Round 1 returns 🟡. **A third narrowing is forbidden** — E67/E67b ran 8 rounds to
  reach "occupied," and narrowing a claim until it survives produces a claim too narrow to
  publish.
- **Total budget**: **4 sessions**, no extensions.

**Stop triggers — any one fires, take the write-up option:**

1. Zero CONTRADICTION and zero HOLE cells after the full matrix (immediate, no relaxation).
2. All 3 advanced candidates return 🔴 or 🟡 after their ≤2 rounds. **This is the expected
   outcome.**
3. Any survivor fails the Part 1 arithmetic. Kill it; **do not soften the bar** — it was already
   corrected once (E56), and softening again invalidates comparison to every prior kill.
4. **The tell-tale**: proposing a *fourth* narrowing, re-defining "collision" mid-protocol, or
   arguing a 🟡 is really a 🟢. Documented branch-extension signature. Stop on sight.

---

## Part 7 — Honest expected value, split by branch

**P(novel ∧ reachable ∧ ≥1pp on 3 seeds) = 3–5%.**

Base rate 0/5, and the audits were not independent draws — E162 identified the common cause,
which lowers it further. Even *granting* novelty, a candidate must deliver ~+8pp mean on 15
subjects where three intervention families already produced −0.89pp, +0.062pp (oracle) and
−0.37pp, and where E140/E142 indicate the information is **absent from the input**.

**P(defensible CONTRADICTION-branch contribution) = 35–50%.**

Needs no Dice improvement. The experiments are **already done** (E71/E147/E72/E73); the matrix
supplies the external half of the argument. Negative-result/analysis contributions are a real
category, and the incentive asymmetry works in our favour here.

### Scope caution for the contradiction branch

Our evidence is **one architecture family, one cohort, n=125, one benchmark**.

- **Not supportable**: "the field's adaptive mechanisms are inert."
- **Supportable**: "predict-demand→allocate-capacity is inert on this failure mode, shown three
  independent ways, and the 2026 literature's persistent residuals are consistent with that."

Overreaching to the first would be the same error in a new place — and it is exactly what the
`mechanism_stated` field is designed to catch in *other people's* papers. Apply it to ours.

---

## Verification — the protocol is correctly specified iff, before any paper is read

1. Every classification rule is mechanically applicable without judgement (Part 2).
2. The decision rule names which cells advance (Part 3).
3. The stop triggers are observable, not subjective (Part 6).
4. The arithmetic check is stated with verified numbers (Part 1).

**No compute is licensed by this document. The first execution step is literature extraction, not
code.**
