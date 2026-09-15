# E162 — Testing the pivot premise before building a phase on it

**Date**: 2026-09-15
**Status**: Premise test. No compute. Result: **premise NOT supported.**
**Follows**: E161 (K occupied)

---

## Why this document exists

The proposed Option-3 pivot rests on one empirical claim:

> "Existing architectural assumptions are crowded. But **the causal measurements we've developed
> are much less crowded.**"

with the new search structure:

$$\text{existing model} \rightarrow \text{causal measurement} \rightarrow \text{previously unoptimized quantity} \rightarrow \text{algorithmic intervention}$$

That is a good reframe *if the premise holds*. If it does not, the pivot rebuilds the same trap
with better vocabulary — and this project has now spent four audits learning that the cost of
not testing a premise first is a branch.

**The premise was tested before designing the phase. It does not hold.**

---

## Test 1 — "optimize a causal/ablation-derived quantity as a training signal"

This is the exact structure the pivot proposes, and it is the structure of CDCG (E71), which
E157 already found occupied by arXiv 2608.14894.

Search returns it as an **established and active pattern**:

- **CausalDisenSeg** (arXiv 2604.13409) — causality-guided disentanglement with counterfactual
  reasoning for **brain tumour segmentation under missing modalities**. Trains on a weighted
  sum of segmentation loss **plus causal intervention losses** (CVAE, HSIC, conflict,
  disentanglement), end-to-end.
- **Spectrum-intervention invariant causal representation learning** for single-domain
  generalizable **medical image segmentation** (Medical Image Analysis) — infuses non-linear
  interventions to derive unobservable causal factors.
- "Beyond Correlation: Causal Intervention for Multi-Label Medical Image Diagnosis";
  Cross-Modal Causal Intervention; Causal Transfer in Medical Image Analysis (2603.24388).

A representative summary of the space: *"causal intervention-derived quantities are being used
as training signals beyond just diagnostic representations."* That is the pivot's proposed
structure, stated as the current state of the field.

**Verdict: 🔴 the structure itself is occupied.**

## Test 2 — effective rank as an optimizable quantity

Our strongest validated causal finding (E124→E126: effective rank of the pool3 window causally
drives $N_b$, dose-response confirmed) would be the natural candidate for
"previously unoptimized quantity."

It is not unoptimized:

- **WERank** (arXiv 2402.09586) — weight regularization added to the primary loss specifically
  to prevent rank degradation.
- **Dynamic Rank Adjustment** (arXiv 2508.08625) — restores effective rank during training.
- Soft orthogonality regularizers; BatchNorm-avoids-rank-collapse (NeurIPS 2020); neural
  collapse vs low-rank bias (NeurIPS 2024).

Effective rank is described in this literature as "a measure commonly used to assess the
quality of the representation learned by a neural network," with an established family of
methods that regularize it.

**Verdict: 🔴 occupied.**

## Test 3 — the remaining measurements

| Our measurement | Status |
|---|---|
| $N_b$ (ablation sensitivity) | 🔴 E157: predictive self-knowledge (2608.14894); ablation studies literature |
| $G_b$ / gradient allocation | 🔴 GradNorm-family MTL, disclosed as adjacent since E44 |
| Effective rank | 🔴 Test 2 |
| Recoverability $O_i$ | 🔴 E138 partial hit; conformal/evidential reliability weighting |
| Modality informativeness | 🔴 E161, on six independent axes |
| Regional dependence | 🔴 E161 (sub-region-aware fusion, $\alpha_{m,r}$) |

**Every measurement in the "causal laboratory" has an occupied optimization counterpart.**

---

## Why the premise failed — the structural reason

The measurements were chosen *because they explain this model's failures*. A model's salient
failure modes are salient to everyone working on the same task. So the measurement being
informative is positively correlated with someone having already built a method on it. This is
the same generator behind E, K, and the self-knowledge result — not four coincidences.

Restated as a rule worth carrying:

$$\boxed{\text{On a well-studied task, a measurement's informativeness predicts its prior-art density.}}$$

The pivot's structure is sound. The asset it proposed to exploit is not scarce.

---

## What this does and does not conclude

**Does not conclude** that the project failed, or that the measurements are worthless. E160
showed the causal laboratory works: one inference run eliminated a major alternative
explanation and *strengthened* the finding ($\rho_{WT}$ −0.454 → **−0.5234**, p=3.8e-10). That
is the process working.

**Does conclude** that searching for an unoccupied *quantity* inside BraTS 3-region
segmentation has the same expected value as searching for an unoccupied *assumption* did. Both
inherit the same dense prior-art surface.

---

## The honest position

Five audits (E, K, self-knowledge, effective rank, causal-training-signal) have returned 🔴.
One (L) returned 🟢 but is orthogonal to the remaining headroom. The base rate is now
informative in itself, and continuing to generate candidate #23 inside this problem is not a
research strategy — it is the roulette the user explicitly asked to stop.

Two defensible endpoints remain. Both are legitimate; the choice is the user's because it is a
scope decision, not a technical one.

**(A) Genuine problem change.** Not a different dataset with the same question — that
reproduces this surface with a thinner search. A different *task family* where the prior-art
density is lower and where the causal-intervention machinery still applies. This requires
accepting that BraTS-specific infrastructure (loaders, checkpoints, N_b machinery) is partly
sunk.

**(B) Option 2 as the rational endpoint.** The project's real, defensible, and genuinely
unusual output is the methodology: ~20 pre-registered kills, a documented habit of catching its
own bugs before they calcified (E62's shared-tensor bug, E25's sign convention, E80's
coordinate frame, E109's gate conditioning, E30/E31's division-by-vanished-footprint), the E56
measurement correction, and E160's cross-regime confirmation. That is a methods contribution
with an evidence base most student projects cannot assemble.

Stated plainly, as promised: **NeuroScan's algorithmic-novelty route inside BraTS 3-region
segmentation is exhausted.** I would rather say that than produce candidate #23.
