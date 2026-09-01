# Phase E71 — CDCG (Causally-Distilled Capacity Gating): Design & Pre-Registered Protocol

**Status: DESIGN ONLY. No code written, no training run.** Written before any
implementation, following this project's design-before-code discipline (matching
E68's process).

## 1. Origin and novelty search summary

After 10 architecture-level ideas checked and found occupied this session (skip
correspondence, positional decoupling, adaptive depth, uncertainty routing,
sparsity gating, physics disentanglement, and others), this idea comes from a
different category: not a new architectural component, but a new **training
signal**, derived from this project's own causal-audit infrastructure (E48).

**What was searched** (≈12 queries across several rounds, this session): input-space
counterfactual distillation (CFKD and its family — Clever-Hans fixing, few-shot LLM
counterfactual distillation), standard post-hoc component-ablation studies (found
extensively — InfiltrNet, MedSAM-CA, DMAF-Net, "No Modality Left Behind," all within
the last 12 months, all reporting Dice drops from removing a module as a validation
table, not as a training signal), curriculum learning from model confidence/gradient
norm, and mechanistic-interpretability work predicting unit-level ablation impact
from static properties (post-hoc, for interpretability, not training).

**What was not found**: a network trained with an auxiliary objective to predict its
own **component-ablation-derived, per-sample causal sensitivity** — computed via a
real model-space intervention (not an input perturbation, not a confidence proxy) —
used *during training* to shape *inference-time* behavior, with no intervention
required at test time.

**Honest confidence**: moderately-high, not certain. A literature search of this
scope cannot prove absence. The closest adjacent work (CFKD family) was checked
structurally, not just by keyword, and found to differ on a real, load-bearing axis
(input-space vs. model-space counterfactual — see Section 2). This should be
reported to any reviewer with the same honesty the rest of this project's
novelty checks have been reported.

## 2. Precise novelty statement

| Property | CFKD / counterfactual distillation (closest prior art) | CDCG (this design) |
|---|---|---|
| What is intervened on | The **input** (perturb a spurious feature) | The **model** (ablate the bottleneck), input fixed |
| What the intervention measures | Whether a label flips | How much Dice degrades, continuously, per subject |
| What is distilled | Teacher behavior on counterfactual inputs | A causally-measured scalar sensitivity score |
| Purpose | Debiasing (remove reliance on confounders) | Self-aware capacity allocation (know when coarse context is load-bearing) |
| Test-time cost | None (student only) | None (predicted, not measured, at inference) |

The mathematical object being perturbed differs: $x \to x'$ (input-space) vs.
$f \to f^{\text{ablate}}$ holding $x$ fixed (model-space). This is the same
distinction this project's own causal chain (E43–E65) has used throughout —
CDCG is a direct continuation of that discipline into training-time use, not
a new invention of the intervention style itself.

## 3. Mechanism

### 3.1 Step 1 — the causal label (reuses E48's own verified construction)

For subject $i$, using the frozen encoder-decoder trunk:

$$
d_i = \text{Dice}\big(f_\theta(x_i)\big) - \text{Dice}\big(f_\theta^{\text{ablate}}(x_i)\big)
$$

where $f_\theta^{\text{ablate}}$ zeros the bottleneck tensor before it reaches the
decoder (bit-for-bit the same construction as E48's `forward_with_bottleneck_ablation`,
reused, not reimplemented). $d_i$ is a real, measured quantity — not a proxy.

### 3.2 Step 2 — auxiliary self-prediction head

A small head $g_\phi$ reads the (un-ablated) bottleneck representation $z_i$ and
predicts:

$$
\hat{d}_i = g_\phi(z_i)
$$

Loss:

$$
\mathcal{L}_{\text{aux}} = \frac{1}{N}\sum_i \left(\hat{d}_i - d_i\right)^2
$$

### 3.3 Step 3 — inference-time use (no ablation ever runs at test time)

$\hat{d}_i$ (the network's own predicted sensitivity, computed in the same forward
pass) gates a mixing coefficient between coarse (bottleneck-derived) and fine
(skip-derived) contributions to the decoder at the `enc1` junction:

$$
z_i^{\text{mixed}} = \sigma(\hat{d}_i) \cdot z_i^{\text{coarse}} + \big(1 - \sigma(\hat{d}_i)\big) \cdot z_i^{\text{fine}}
$$

Total training loss:

$$
\mathcal{L} = \mathcal{L}_{\text{seg}} + \mu \cdot \mathcal{L}_{\text{boundary}} + \lambda \cdot \mathcal{L}_{\text{aux}}
$$

added as one new term, matching the project's "one variable at a time" discipline.
$\lambda$ requires its own calibration pass (not guessed), same convention as this
project's established `mu`/`lambda_margin`/`lambda_off` calibration history.

## 4. What this does NOT change (isolation discipline)

- No change to the loss weights already established.
- No change to the checkpoint/training recipe otherwise (optimizer, LR schedule,
  seed convention, batch size).
- $g_\phi$ is a small, separate head — does not touch `seg_head`, `evidential_head`,
  or `boundary_head`.
- The gating mechanism at `enc1` is architecturally simple (a scalar-gated convex
  blend) — deliberately not reusing CAS's deformable-offset machinery, to keep this
  a clean, single-variable test of the *causal self-prediction* idea, not conflated
  with CAS's (falsified) correspondence-correction idea.

## 5. Pre-registered falsifiable predictions (before any training)

1. **Self-prediction check** (cheapest, run first, same discipline as E68's
   Section 7.3 and CAS's gate-fidelity check): on held-out subjects, does
   $\hat{d}_i$ correlate with the independently-measured $d_i$ (Spearman,
   permutation test, subject-level, matching this project's own convention)?
   **If this fails — the network cannot learn to predict its own causal
   sensitivity better than chance — KILL before any further step.** This
   is the load-bearing assumption; nothing downstream matters if it doesn't hold.
2. **Mechanism check**: if prediction 1 holds, does the gate $\sigma(\hat{d}_i)$
   behave sensibly — is it elevated for subjects E48 independently found to be
   highly bottleneck-dependent (small lesions, per E48's own ρ=−0.454 finding),
   and not just a constant or degenerate function? Tested the same way CAS's gate
   was tested against E65 (and found, honestly, to have failed) — this project's
   own falsification discipline applies equally here.
3. **Performance check** (the actual project bar): full 125-subject validation,
   ≥3 seeds, per-subject Dice, compared against the matched baseline (same recipe,
   no CDCG) — needs ≥1pp mean improvement with a CI excluding 0.

Only if 1 AND 2 hold does 3's result mean what it appears to mean. If 1 or 2 fail
but 3 somehow shows +1pp, that must be reported as an unexplained gain, not
attributed to the claimed mechanism — the same rule CAS's design applied (project
constraint: mechanism ≠ performance).

## 6. Open implementation questions (to resolve before writing code)

1. Whether $d_i$ should be recomputed periodically during training (as $\theta$
   changes, the "true" ablation sensitivity for a subject may drift) or computed
   once from a fixed reference checkpoint — a real design choice with a
   correctness/cost tradeoff, not yet decided.
2. Calibration of $\lambda$ (the auxiliary loss weight) — pilot runs required,
   not a guess.
3. Whether the gate should act only at the `enc1` junction (matching E62–E65's
   own established locus) or elsewhere — scoped to `enc1` only for this first
   test, per "one variable at a time."

## Next step (not yet started)

Implement and run ONLY prediction 1 (the self-prediction fidelity check) —
cheapest, first, most falsifiable test, no full training loop needed beyond what's
required to get $\hat d_i$ trained against real $d_i$ labels on a held-out split.
Do not implement the gating mechanism (step 3.3) or run a Dice comparison until
prediction 1 passes.
