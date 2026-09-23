# E196 — Headroom gate applied to the whole remaining space, before the SOTA search

**Date**: 2026-09-20
**Status**: GATE. No compute beyond recomputation from the frozen evaluation.
**Follows**: E195 (recoverability-action closed by arithmetic).

---

## Why this precedes the SOTA search

The specified chain is:

$$\text{SOTA} \to \text{unresolved limitation} \to \text{candidate principle} \to \text{prior-art audit} \to \text{cheap arithmetic gate}$$

E195 demonstrated that the arithmetic gate can fire **before** the audit and save the audit
entirely. This document applies that gate one level higher: **to the whole remaining search
space at once**, by asking where ≥1pp can arithmetically come from — independent of which
principle is proposed.

If a region of the space has no ≥1pp room, no candidate targeting it needs auditing. That
constrains the SOTA search rather than replacing it.

## Baseline (recomputed, n=125, `E131_v5control_seed0`)

ET 0.8191 / TC 0.8481 / WT 0.9119 → **3-region mean 0.8597**; bar = **0.8697**.

## What +1pp actually demands

| route | requirement |
|---|---|
| ET alone | 0.8191 → 0.8491 = eliminate **16.6%** of all remaining ET error |
| TC alone | 0.8481 → 0.8781 = eliminate **19.7%** of all remaining TC error |
| WT alone | 0.9119 → 0.9419 = eliminate **34.1%** of all remaining WT error |

## The gate: interior (non-boundary) headroom is arithmetically insufficient

E167 measured the interior fraction of the residual on the 110-subject good stratum: ET 1.4%,
TC 2.8%, WT 15.0% (boundary-enrichment controlled — the control is load-bearing).

| region | residual | × interior frac | 3-region gain |
|---|---:|---:|---:|
| ET | 0.1809 | 0.014 | +0.084 pp |
| TC | 0.1519 | 0.028 | +0.142 pp |
| WT | 0.0881 | 0.150 | +0.441 pp |
| **total** | | | **+0.667 pp** |

$$
\boxed{\text{Perfecting ALL interior error in ALL three regions yields } +0.667\text{ pp} < 1.0\text{ pp.}}
$$

**Consequence — a whole class of candidates is excluded without audit.** Any method whose
mechanism is better interior/semantic labelling of a correctly-detected region cannot reach the
bar, however novel. Boundary-targeting methods are separately excluded by the annotation
ceiling (model ET 0.900 vs Menze inter-rater median 0.77).

That eliminates most of what a SOTA-limitations search returns: better context modelling,
attention variants, multi-scale fusion, refinement decoders, loss reshaping.

## The one route with ≥1pp room — and it is not new

The 15 catastrophic-tail subjects (ET<0.5 or TC<0.5) are a **different failure mode** from
E167's stratum: whole-region misses, not boundary error. E167's interior fractions were measured
on the other 110 and do not bound the tail.

| intervention | 3-region mean | gain |
|---|---:|---:|
| lift tail ET&TC to 0.30 | 0.8730 | **+1.328 pp** |
| lift tail ET&TC to 0.40 | 0.8791 | **+1.940 pp** |
| lift tail ET&TC to 0.50 | 0.8856 | **+2.588 pp** |
| lift tail ET&TC to 0.75 | 0.9036 | **+4.390 pp** |

Tail: ET 0.2269 / TC 0.1864 / **WT 0.8239**. Rest: ET 0.8999 / TC 0.9383 / WT 0.9239.
(Reproduces E139's target definition exactly.)

$$
\boxed{\text{The 15-subject tail is the ONLY region of the space with} \ge 1\text{pp arithmetic room.}}
$$

## Why this closes the search rather than opening it

The tail is precisely where **E142** applies: zeroing t1c collapses ET 0.8433 → 0.0015, and
these subjects are functionally that ablation (four have negative contrast). WT holds at 0.8239
on the same subjects because WT runs on t2f, which is intact — the tumour is *found* but cannot
be *sub-partitioned*.

So the space resolves as follows. **Stated precisely**: arithmetic closes *mechanism classes*,
it does not prove no conceivable algorithm exists. The correct claim is — *within the
experimentally measured error decomposition and the specified mechanism classes, no additional
≥1pp headroom is identified*. An algorithm that obtains **genuinely new information**
(additional acquisition, external priors, another modality, generative reconstruction) is not
covered by the interior/boundary calculation and is not excluded here.

| target | arithmetic | mechanism |
|---|---|---|
| interior / boundary of the 110 | **< 1pp** (0.667 total) | closed by E196 arithmetic + annotation ceiling |
| the 15-subject tail | **≥ 1pp available** | closed by E142 (information absent, causal) |

A method clearing the bar must extract ET/TC structure from subjects whose contrast-bearing
channel carries no usable enhancement signal. That is not an architecture, objective, or
optimization problem — it is an **acquisition** problem.

## Verdict

$$
\boxed{\text{The SOTA-limitation search cannot succeed on this cohort, and the reason is arithmetic, not exhaustion.}}
$$

This is not "we failed to find a principle." It is: **every region of the outcome space has been
measured, and the one with room is information-limited by causal test.** Running the SOTA search
now would be searching for a principle to apply to a target that does not exist.

**The gate is reusable**: any future candidate must first state which row of the dichotomy it
targets. If interior/boundary → capped at +0.667pp. If tail → must explain how it defeats E142.

## Consequence for the project requirement

The original standard (*differentiated principle + ≥1pp Dice*) is now closed by **measurement of
the outcome space itself**, not only by per-branch kills. The two honest options stated in the
directive stand, and the first is now better supported:

1. **Relax the ≥1pp algorithmic requirement**; publish E143→E193 as the recoverability/
   reliability finding — with E195/E196 as the quantitative reason no algorithm follows.
2. **Change domain** — a dataset that is not saturated. On disk this is not available (PediMS =
   2 files; BraTS is the only usable cohort).

This document does **not** recommend redefining "algorithmic contribution" to mean a referral
flag. E195 and E196 exist to make that redefinition unnecessary: the absence of an algorithm is
now a *measured result*, which is a stronger thing to publish than a weak algorithm.

## Not claimed here

No prior-art audit was run in this document, and no novelty claim is made about anything. The
narrowed Question-A claim (cross-fitted target-specific image observer → pre-segmentation
cross-model recoverability, ET-works/TC-collapses) remains **unaudited** and must be audited
before any submission that rests on it.
