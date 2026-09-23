# E195 — "What action can exploit $O_i$?" — terminated by arithmetic, before literature

**Date**: 2026-09-20
**Status**: PASS TERMINATED at step 0. No prior-art search was run, because the feasibility
arithmetic closed the question first.
**Follows**: the directive to find an algorithmic operation exploiting pre-segmentation ET
recoverability that is not routing / uncertainty / TTA / error-correction / selective compute.

---

## Why the literature search was not run

The standing rule is *prior-art audit before code*. There is a cheaper gate that precedes even
that, and this project has used it before (E134 killed a global $\alpha/\beta$ shift with two
minutes of arithmetic; E166 failed on arithmetic independently of its occupancy verdict):

> **Before asking whether an action is novel, ask whether it could clear the bar if it worked
> perfectly.**

Running an audit first would have risked finding a clean 🟢 for an operation that is
arithmetically incapable of reaching ≥1pp — and a clean audit is exactly the kind of result that
generates momentum toward compute.

## The oracle computation

Take the most generous possible algorithm: one that, for **every** subject, lifts ET Dice to
that subject's own measured recoverability $O_i$. No method can do better while remaining
consistent with $O_i$ as a recoverability estimate — this is the ceiling of the entire
$O_i \to$ action family, not of one candidate operation.

$$
\text{Dice}^{\text{oracle}}_i = \max(\text{Dice}_i,\ O_i)
$$

| quantity | value |
|---|---|
| ET mean, current (n=90, `E131_v5control_seed0`) | 0.8441 |
| ET mean, **perfect** $O_i$-oracle | 0.8554 |
| ET gain | **+1.132 pp** |
| **3-region mean gain** | **+0.377 pp** |
| subjects improved at all | 16 of 90 |
| pre-registered bar | **≥ 1.0 pp** |

$$
\boxed{+0.377\text{ pp} \ll 1.0\text{ pp — the entire family is capped at roughly one third of the bar.}}
$$

This is the **same structural result as E140**, whose oracle came in at +0.062pp, and the same
reason E137/E140/E141 could not clear 1pp. It is now established for the $O_i$ family too.

## Why — the gap distribution

$O_i$ does not sit *above* the model. It sits slightly **below** it.

| $O_i$ band | n | mean $O_i$ | mean Dice | gap ($O_i -$ Dice) |
|---|---:|---:|---:|---:|
| [0.00, 0.30) | 4 | 0.088 | 0.052 | **+0.036** |
| [0.30, 0.50) | 5 | 0.432 | 0.567 | −0.136 |
| [0.50, 0.70) | 5 | 0.618 | 0.746 | −0.128 |
| [0.70, 0.85) | 22 | 0.794 | 0.832 | −0.038 |
| [0.85, 1.01) | 54 | 0.920 | 0.942 | −0.022 |

- **Mean gap = −0.036**, median −0.027. The model **beats** $O_i$ on average, in every band
  except the lowest.
- Only **5 of 90** subjects have $O_i > \text{Dice} + 0.05$; only **2 of 90** exceed $+0.15$.

This is the quantitative form of E144's empty Regime III, and it is the decisive fact: there is
no reservoir of recoverable-but-unrecovered performance for *any* action to capture. The 4
subjects in the lowest band contribute +0.036 mean gap over 4/90 of the cohort — arithmetically
negligible.

## The low-$O_i$ population was the only live target, and it is too small

$O_i < 0.5$: **n = 9**, mean $O_i$ = 0.279, mean Dice = 0.338.

| hypothetical | ET mean | 3-region gain |
|---|---:|---:|
| lift all 9 to Dice 0.30 | 0.8571 | +0.433 pp |
| lift all 9 to Dice 0.40 | 0.8626 | +0.619 pp |
| lift all 9 to Dice 0.50 | 0.8684 | +0.811 pp |

Even lifting every low-recoverability subject to **0.50 Dice — well above their measured
recoverability of 0.279** — yields **+0.811 pp**, still under the bar. To clear 1pp one must
exceed $O_i$ substantially on precisely the subjects where the evidence is measured to be
weakest, which contradicts the premise that $O_i$ estimates recoverability at all.

## Verdict

$$
\boxed{\text{The } O_i \rightarrow \text{action} \rightarrow \text{better segmentation family is CLOSED by arithmetic, not by prior art.}}
$$

No operation in this family — however novel, however non-routing — can reach ≥1pp on this
cohort. The question "what action exploits pre-segmentation recoverability" has a feasibility
answer that precedes its novelty answer.

**This strengthens rather than weakens E143/E144.** $O_i$ tracking the model so tightly
(mean gap −0.036) is the *evidence* for the frontier claim. A large positive gap would have
meant $O_i$ was measuring something the model was missing; a near-zero/negative gap means the
model is already extracting what the intensities permit. The finding and the absence of an
algorithm are the same fact measured two ways.

## Consequence for the NeuroScan requirement

The original standard — *differentiated computational principle + ≥1pp Dice* — is now closed
by measurement on this cohort through **every** identified route:

| route | closed by |
|---|---|
| input information | E142 (causal: t1c → ET 0.0015) |
| boundary residual | E167 (98.6% within 3 vox; 0.900 vs inter-rater 0.77) |
| allocation / capacity | E144 (Regime III empty, n=0/90) |
| adaptive compute | E178 (≤ Constant baseline) |
| numerical residual | E179 (22% of 118 bins) |
| latent geometry | E188-F (readout-mediated, max 7e-5) |
| **recoverability-driven action** | **E195 (oracle = +0.377 pp)** |

This is not search exhaustion. Each is a measured closure with a stated reason.

## What remains, stated without inflation

The triage/reliability direction (Question A) is **not** affected by this result — it never
claimed a Dice improvement. It remains viable and now has a sharper justification: the reason
to predict failure rather than fix it is that fixing it is arithmetically impossible on this
cohort.

Its prior-art audit **has not been run**. The narrowed claim requiring audit is:

> cross-fitted, target-specific image observer → pre-segmentation, cross-model recoverability,
> with ET-works / TC-collapses asymmetry

That audit is the correct next step **if** Question A is pursued. It was not run here, and this
document makes no novelty claim about it.
