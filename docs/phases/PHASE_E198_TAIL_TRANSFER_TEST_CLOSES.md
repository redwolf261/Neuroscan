# E198 — The tail ET rule does NOT transfer. E197 closed; E195/E196 stand.

**Date**: 2026-09-20
**Status**: DECISIVE. Pre-registered reading fired as written. No method proposed.
**Follows**: E197 (in-subject oracle 0.741 on the tail — leaky by construction).

---

## Pre-registered reading (fixed before running, quoted from E197)

> `loo_tail` ≈ 0.68 → rule TRANSFERS → real headroom → E196 reopens
> `loo_tail` ≈ 0.28 → subject-specific boundary → E195/E196 stand, closure complete
> intermediate → report as intermediate, **do NOT round toward the convenient end**

## Result (n=15 tail subjects, identical architecture/epochs/features to E197)

| arm | training set | tail ET Dice |
|---|---|---:|
| model (3D CNN) | — | 0.227 |
| `in_subject` (E197, **leaky**) | subject $i$ itself | **0.741** |
| **`loo_tail`** (**the test**) | the other 14 tail subjects | **0.193** |
| `global_good` | 30 good-stratum subjects | **0.352** |
| `loo_tail+good` | other tail + good | 0.319 |

$$
\boxed{\texttt{loo\_tail} = 0.193 \;<\; \texttt{model} = 0.227 \;<\; \texttt{global\_good} = 0.352 \;\lll\; \texttt{in\_subject} = 0.741}
$$

**The rule does not transfer.** Trained on the other tail subjects, the observer scores 0.193 —
*below* the 3D CNN it was supposed to beat, and below E195's cross-fitted $O_i$ (~0.279). The
in-subject 0.741 was almost entirely leakage, exactly as E136→E143 was.

## Three findings, the second one important

**1. E197 is fully explained as leakage.** 0.741 → 0.193 once the subject's own labels are
withheld. This is the **third** time this project has caught the same artifact (E136's 0.531
oracle; E188's E15 reproduction; now E197). The pattern is robust enough to be a standing rule.

**2. The E136 "different decision boundary" claim is refuted a second time, and in the opposite
direction.** E136 hypothesised the tail needs its own decision boundary distinct from the
population's. If true, training *on tail subjects* should beat training on good subjects. The
measurement is the reverse:

$$
\texttt{global\_good } (0.352) \;>\; \texttt{loo\_tail+good } (0.319) \;>\; \texttt{loo\_tail } (0.193)
$$

Adding tail data **hurts**. There is no shared "tail rule" to learn — the tail subjects are not
a coherent population with a common alternative boundary. They are individually idiosyncratic.
E143 retracted this claim on calibration grounds; E198 now refutes it on transfer grounds too.

**3. Even the best honest arm cannot clear the bar.**

| lift tail ET to | 3-region mean | gain |
|---|---:|---:|
| `loo_tail` 0.193 | 0.8637 | +0.404 pp |
| `loo_tail+good` 0.319 | 0.8671 | +0.742 pp |
| `global_good` 0.352 | 0.8680 | **+0.827 pp** |

The most favourable honest arm yields **+0.827 pp < 1.0 pp** — and that arm is a *worse* ET
predictor than the CNN already is on the other 110, so it is not an improvement one could
actually deploy. The tail route is closed on its own arithmetic, independent of novelty.

## Verdict

$$
\boxed{\text{E197's reopening is CLOSED. E195 and E196 stand unmodified.}}
$$

E196's dichotomy survives with no third row:

| target | arithmetic | status |
|---|---|---|
| interior/boundary of the 110 | < 1pp (+0.667 total) | closed by measurement + annotation ceiling |
| the 15-subject tail | oracle-honest ≤ +0.827 pp | **closed by E198 transfer failure** |

E142's practical conclusion stands, with E197's refinement to its *interpretation* preserved:
t1c is not an informationally dead channel for these subjects (dropping it costs −0.320 in the
in-subject oracle), but **no transferable rule recovers ET on them**. Whether the residual
information is subject-specific physiology or unlearnable from 4 intensities is not resolved,
and does not need to be — either way no method reaches the bar.

## What this settles about the project requirement

The original standard (*differentiated computational principle + ≥1pp Dice*) is now closed by
measurement across the entire outcome space, with the tail — the last region holding ≥1pp of
arithmetic room — closed by a pre-registered transfer test rather than by assumption.

This is the strongest form of the negative result the project can produce: not "we could not
find a method," but **"we measured where a method could possibly help, and honestly-estimated
performance in that region falls short of the bar."**

## Discipline note

E197 must never be cited as a performance figure. Its sole legitimate role was to motivate E198,
and E198 is the number of record. Both are retained in the archive precisely because the
leaky/honest pair is itself the methodological contribution.

**Artifacts**: `E198_transfer.json`; script `e198_transfer.py` (scratchpad).
**Caveat**: n=15 tail, one subject (`01169-000`) had insufficient ET voxels and is NaN
throughout; means are over 14.
