# E170b — TDM prior-art audit: the exact chain, not "adaptive rank"

**Date**: 2026-09-16
**Status**: Targeted audit complete. Verdict: **🟡 PARTIALLY DIFFERENTIATED — not a safe novelty claim.**
**Run BEFORE** the 2-hour R_hat precompute, per the user's instruction.

---

## What was audited

Not "adaptive rank" (already known occupied). The exact chain:

> predicted **task-required** representation rank − sample's **actual** effective rank,
> used as a **deficit-only** training signal

---

## Correction to the premise: the LoRA family is a DIFFERENT OBJECT

The cited prior art — ARENA (MICCAI 2025), AdaLoRA-QAT, SeLoRA — allocates rank in
**adapter weight matrices** for parameter-efficient fine-tuning. Ours is the effective rank
of an **activation tensor** at a named stage, measured by causal truncation of that activation.

Different quantity, different space, different purpose. These are **not** near-neighbours and
should not be cited as occupying our claim. The real danger lies elsewhere, and the audit found it.

## The real collisions

| Component | Status | Evidence |
|---|---|---|
| Effective rank as a differentiable regulariser | 🔴 **OCCUPIED** | "the real-valued and differentiable nature of effective rank allows it to be utilized as a regularization objective" — Effective Rank Regularization for 3D Gaussian Splatting (NeurIPS 2024) |
| **One-sided hinge rank penalty** (fires only below a target) | 🔴 **OCCUPIED** | Same NeurIPS 2024 work: keeps effective rank above a target, penalising only those near 1. Squared-hinge "zero when the constraint is satisfied" is a standard construction |
| Rank-collapse prevention by regularisation | 🔴 **OCCUPIED** | WERank (2402.09586); BatchNorm-avoids-rank-collapse (NeurIPS 2020); MPNN graph-splitting (2409.11504) |
| "Rank deficit" terminology | 🟡 adjacent | *Rank Diminishing in Deep Neural Networks* (NeurIPS 2022) defines an "independence deficit" — but it is about **output-class interdependence**, and the paper is **diagnostic only**, not a training signal |
| Per-sample rank target | 🟢 **not found** | Every regulariser found uses a **fixed constant** target. No instance-specific target located |
| Target = *empirically measured minimum rank preserving the model's own output* | 🟢 **not found** | No prior work located that measures R* by causal truncation against the model's **own undegraded prediction** and then uses it as a supervision target |

## Verdict

$$\boxed{\text{🟡 The MECHANISM is occupied. The TARGET may be differentiated.}}$$

The deficit-only hinge on effective rank is **published**. What was not found is:
(a) a **per-sample** rank target, and (b) a target defined as the **causally measured minimum
rank that preserves the network's own downstream output**.

So the defensible claim is narrow and sits entirely in *what the rank is compared against*, not
in the penalty form:

> A segmentation representation can be *instance-specifically* task-deficient when its effective
> rank falls below an empirically measured minimum rank required to preserve its **own** downstream
> output — and training can target that deficit.

**This is not yet established as novel.** Absence of evidence after one targeted pass is weak,
and E162's structural law applies: if per-sample rank targets were obviously useful, they would
likely exist.

## Consequence for E170 — and it is decisive for the experiment design

The audit **sharpens which arm carries the novelty**:

- If the **constant** arm matches **tdm**, then only the occupied part is doing the work (a fixed
  rank floor, = NeurIPS 2024), and the branch is dead as a novelty claim regardless of Dice.
- Only a **tdm > constant** separation would isolate the potentially differentiated component.

This makes the four-arm design load-bearing rather than merely careful, and it means the
2-hour per-subject R_hat precompute is **required** — without real per-subject targets the tdm
arm cannot differ from constant, and the experiment could only ever confirm occupied prior art.

## Recommendation

Proceed to the precompute and run the four arms. But the success criterion is now explicitly
**tdm > constant**, not "TDM beats baseline." Beating baseline alone would replicate NeurIPS 2024.

## Sources

Effective Rank Analysis and Regularization for 3D Gaussian Splatting (NeurIPS 2024) ·
WERank (arXiv 2402.09586) · Rank Diminishing in Deep Neural Networks (NeurIPS 2022, arXiv
2206.06072) · Preventing Representational Rank Collapse in MPNNs (arXiv 2409.11504) ·
BatchNorm Provably Avoids Rank Collapse (NeurIPS 2020) · Q3R (arXiv 2511.04485)
