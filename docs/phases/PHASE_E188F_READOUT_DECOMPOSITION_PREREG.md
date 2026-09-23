# E188-F — Readout-vs-representation decomposition of E15 (pre-registration)

**Date**: 2026-09-20
**Status**: PRE-REGISTERED. Inference only. No training. Follows E188-A.

## Why F and not B

E188-A established two things across 40 cells (10 subjects × ε∈{4,8,12,14}, 32 random directions
each):

- **E15 is at the 100th percentile in 38/40 cells** — its direction is not arbitrary.
- **Headroom $R_i < 0$ in 40/40 cells** — the best of 32 random directions *never* beat native
  Dice, at any ε, for any subject.

So the "recoverable-state neighborhood" model is dead: there is no *region*, there is one
direction. The original Experiment B ("what geometry makes a direction good?") is therefore the
wrong next question. The sharper one, which A's own result raises:

$$
\boxed{\text{Does E15 contain information beyond what the frozen readout already explains?}}
$$

## The structural fact that shapes this experiment

`seg_head` is `Conv3d(32, 3, k=1)` — **rank-3, not rank-1**. Measured on the frozen checkpoint:

| quantity | value |
|---|---|
| row norms (ET/TC/WT) | 3.272 / 3.228 / 3.178 |
| cos(ET,TC) / cos(ET,WT) / cos(TC,WT) | +0.848 / +0.671 / +0.774 |
| singular values of $W$ | 5.134, 1.867, 1.176 |
| energy in top direction | **84.4%** |
| energy in top two | 95.6% |

The readout is *approximately* rank-1 but not exactly — 15.6% of its linear action lives outside
the leading direction. **Projecting onto a single $w$ would leak that 15.6% into the "orthogonal"
arm and invalidate the inference.** F therefore projects onto the full 3D row space:

$$
P_W = W^{+}W \in \mathbb{R}^{32\times32}, \qquad
\delta_\parallel = P_W\,\delta_E, \qquad
\delta_\perp = \delta_E - \delta_\parallel
$$

so that $W\delta_\perp = 0$ **exactly**, at every voxel. Any Dice effect from $\delta_\perp$
provably cannot be a first-order logit shift. This is verified numerically at runtime
($\|W\delta_\perp\| \approx$ machine epsilon), not assumed.

## Arms (7 per subject × ε), no random sweep needed

| Arm | Perturbation | Question it answers |
|---|---|---|
| native | $z$ | baseline |
| full | $z + \delta_E$ | the E15 effect |
| par_nat | $z + \delta_\parallel$ | does the readout component alone reproduce E15? |
| perp_nat | $z + \delta_\perp$ | does the readout-invisible component do anything? |
| par_renorm | $z + \epsilon\,\delta_\parallel/\|\delta_\parallel\|$ | is the parallel *direction* useful at matched magnitude? |
| perp_renorm | $z + \epsilon\,\delta_\perp/\|\delta_\perp\|$ | is the orthogonal *direction* useful at matched magnitude? |
| reversed | $z - \delta_E$ | sign control — is the direction signed-specific or an axis? |

Natural-norm arms give the true decomposition ($\delta_\parallel + \delta_\perp = \delta_E$
exactly). Renormalized arms separate "which direction matters" from "how big is the step" —
necessary because A showed magnitude dominates outcomes here.

**Setup frozen and matched to E188-A / E172**: checkpoint `E131_v5control_seed0`, `dec1` site,
single centered 128³ crop, binary fg = union of ET/TC/WT, per-voxel direction construction,
same 10 subjects, ε ∈ {4, 8, 12, 14}. Cost: 10 × 4 × 7 = 280 forward passes (~5 min).

## Also recorded: logit displacement

Per arm, per voxel, the actual readout displacement $\Delta\ell = W\delta \in \mathbb{R}^3$, summarized as
mean $|\Delta\ell|$ per region. This lets us ask the sharper version of the question: **do two
perturbations with the same $W\delta$ produce different Dice?** If yes, the decoder responds to
something beyond its own linear readout.

## Pre-registered decision rule (fixed before running)

| Outcome | Criterion | Verdict |
|---|---|---|
| **H1 — readout steering** | $D(z+\delta_\parallel) \approx D(z+\delta_E)$ **and** $D(z+\delta_\perp) \approx D(z)$ | **KILL E15.** It manipulates the frozen readout; not a computational principle. New inclusion criterion for all future candidates: the benefit must not reduce to movement in the readout row space. |
| **H2 — structure beyond the readout** | $D(z+\delta_E)$ substantially exceeds $D(z+\delta_\parallel)$, **or** $\delta_\perp$ alone moves Dice materially | First genuinely interesting result in this branch — proceed to ask what property $\delta_\perp$ changes |
| **H3 — oracle artifact** | (carried, not tested here) the effect requires GT to construct the direction | Already strongly indicated by E172's label-free null; F does not re-test it |

"Substantially" is judged on the per-subject distribution (sign consistency across 10 subjects),
not a pooled mean — same discipline as E179/E188-A.

## What this does NOT do

No training, no new network, no loss, no optimizer, no random sweep. Does not re-test the
label-free selection question (E172 already answered that negatively at this exact site). GT is
used only to construct $\delta_E$ (inherited from E15's design, acknowledged oracle) and to score
Dice.
