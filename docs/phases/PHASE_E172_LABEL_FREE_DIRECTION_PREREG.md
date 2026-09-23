# E172 — Label-Free Corrective Direction Test (pre-registration)

**Date**: 2026-09-16
**Status**: PRE-REGISTERED. Feasibility verified on hardware (below). Not yet run.
**Blocks**: E171. Dependency graph is
`E172 (d_l) → rank-definition unification → exact E171 prior-art decision → E171`.

---

## The question

$$\exists\; d_l(X,\hat Y) \;\;\text{such that}\;\; z_l + \alpha d_l \;\text{corrects model errors, with } d_l \text{ constructed without GT}$$

**The hard requirement**, which kills ordinary confidence gradients a priori: when the model is
*confidently wrong*, a useful direction must **change** the prediction, not merely sharpen it.
E133 measured the regime that matters — 5/8 ET failures predict **exactly zero voxels at max
probability 0.0000**. Sharpening zero yields zero.

## The inviolable rule

$$\boxed{\text{GT may EVALUATE } d_l; \quad \text{GT may NOT CONSTRUCT } d_l}$$

Every direction is built from $X$ and $\hat Y$ only. GT enters exactly once, at scoring time,
after the direction and the perturbed prediction already exist. Any violation invalidates the
experiment — this is the failure that retired E15.

---

## Feasibility, verified on hardware before writing the protocol

| Check | Result |
|---|---|
| enc1 tensor | `(1, 32, 128, 128, 128)` = **67.1M elements** |
| Full Jacobian / SVD | **impossible** — confirms only VJP/JVP products are viable |
| Single VJP through enc1→output | **9.9 s**, peak **4.16 GB** of 8 GB — feasible |

### The decisive pre-measurement: is enc1→output rank-1 too?

The readout is rank-1 ($\nabla_z D = D(1{-}D)w$), which is why E15's direction needed GT for its
sign. If enc1→output were *also* effectively rank-1, E171 would be dead immediately. Measured
cosines between VJP directions at enc1 for three output functionals:

| pair | cosine |
|---|---:|
| sum_ET vs sum_TC | **+0.9858** |
| sum_ET vs entropy | **+0.1556** |
| sum_TC vs entropy | **+0.2701** |

**Two consequences, both load-bearing:**

1. **The map is NOT globally rank-1.** Entropy is near-orthogonal to the region-mass directions,
   so confidence-sharpening and mass-changing are genuinely distinct operations at enc1. E172 is
   therefore a well-posed experiment rather than a foregone conclusion.
2. **Region selectivity is nearly rank-1.** ET and TC directions are collinear at 0.986 — asking
   "more ET" and "more TC" moves enc1 the same way. This independently corroborates E133's
   ET/TC contagion (bottom deciles are the same 15 subjects, Jaccard 0.733) and predicts that
   *region-targeted* label-free steering cannot separate ET from TC at this stage.

---

## Protocol

Frozen `E131_v5control_seed0`, inference only, no training. Fixed 125-subject validation set.

### Directions under test (all label-free, matched norms)

| # | Name | Construction | Rationale |
|---|---|---|---|
| 1 | $d_{\text{Jac}}$ | VJP of a **label-free output functional** w.r.t. enc1 — specifically total predicted foreground mass $\sum_r \sum_p \hat y_r(p)$ | the serious candidate; not rank-1 at enc1 |
| 2 | $d_{\text{conf}}$ | VJP of prediction entropy | **the diagnostic control** — expected to sharpen, not correct |
| 3 | $d_{\text{rand}}$ | Gaussian, matched norm | pure-noise floor |
| 4 | $d_{\text{GT}}$ *(reference only)* | E15-style GT-selected centroid | **upper bound**, never a candidate; establishes the scale a deployable direction would have to reach |

Sweep $\alpha$ over a matched-norm grid; report the full curve, not a single point.

### Three separate evaluations — reported independently

1. **Prediction movement**: $\|\Delta\hat Y\|$, fraction of voxels flipping, and change in
   predicted foreground mass. *No GT.*
2. **GT-independent viability**: is the change **structured** or merely a confidence rescale?
   Measure the change in predicted mass at fixed threshold vs the change in mean logit — a pure
   sharpening operation moves the latter with little of the former.
3. **Oracle efficacy** *(GT used only here, after the fact)*: per-subject Dice as a function of
   $\alpha$; and critically, Dice change **restricted to voxels the model got wrong**.

---

## Pre-registered decision rule (fixed before running)

| Outcome | Criterion | Verdict |
|---|---|---|
| **A — corrective direction exists** | $d_{\text{Jac}}$ raises Dice significantly above **both** $d_{\text{rand}}$ and $d_{\text{conf}}$ at matched norm, **and** the gain is concentrated in initially-wrong voxels | E171 proceeds to rank unification |
| **B — no corrective direction** | $d_{\text{Jac}}$ is indistinguishable from $d_{\text{rand}}$, or moves predictions without preferentially repairing errors | **E171 DIES CLEANLY** |
| **C — sharpening only** | $d_{\text{conf}}$ raises confidence while confident errors stay errors | confirms the E133 objection *experimentally* rather than theoretically — a real result either way |

**Outcome B is a valuable result, not a failure.** It would establish:

> E15's causal representation steering requires oracle information unavailable at inference or
> training time.

That is worth stating in the record, and far better than spending GPU hours disguising a
confidence-sharpening operation as "task-sensitive transport."

### Pre-recorded expectation

$d_{\text{conf}}$ sharpens and does not correct (outcome C is near-certain, given E133).
$d_{\text{Jac}}$ is genuinely uncertain — the near-orthogonality to entropy says it is *a
different operation*, but "different" does not imply "corrective." Recorded so the result cannot
be read as confirmation.

---

## What E172 does NOT do

- Does not modify architecture, training, loss, or resolution.
- Does not use GT to construct any direction.
- Does not resolve the $R_{\mathrm{eff}}$ definition mismatch (E165 SVD-truncation vs E169
  entropy@20k vs E170 entropy@8k) — that is the **next** step, and only if E172 returns A.
- Does not reopen the E171 novelty question: conditional activation steering (CAST/GAPS/DSAS/GSS)
  remains occupied regardless of this outcome.

## Cost

~125 subjects × (1 forward + 3 VJPs + α-sweep forwards). Inference only, ≈4.2 GB peak. No training.
