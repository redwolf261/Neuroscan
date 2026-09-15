# E155 — Pre-registration: Gate C, Early Predictability of Representational Demand

**Date**: 2026-09-15
**Status**: PRE-REGISTERED. Not yet run. **BLOCKED on E154 (Gate B′).**

---

## The question

$$
E_{\text{early}}(x) \rightarrow \widehat{R_i^*}
$$

Can the representational demand of a subject be predicted from an early representation —
i.e. **before** the expensive representation has been computed — and is that signal something
more specific than ordinary difficulty or recoverability?

## Why this is not just another difficulty predictor

The branch E98–E153 has repeatedly found quantities that correlate with per-subject error and
then turned out to be difficulty in a new coat. Gate C is written so that **passing it requires
beating difficulty and beating $O_i$ simultaneously.** A high raw correlation is explicitly
declared insufficient.

---

## Ledger position

| Gate | Claim | Status |
|---|---|---|
| A | Demand exists (heterogeneous `R*`) | GREEN — *conditional on E154* |
| B | Demand is observable (`rho_partial(O_i, R*) = -0.496`) | GREEN provisional |
| **B′** | **`R*` is checkpoint-invariant** | **NOT TESTED — E154, blocking** |
| **C** | **Demand is predictable before late representation** | **NOT TESTED — this document** |
| D | Predicted demand improves the accuracy/compute frontier | not started |
| E | The allocation rule is genuinely new | not started |
| F | It yields the required Dice improvement | not started |

---

## Three ways Gate C can pass and still be worthless

**C-fail-1 — it's difficulty.** $\widehat{R^*}$ is really tracking lesion volume, baseline Dice,
or a generic "hard subject" axis. Difficulty-conditioned compute is thoroughly published.

**C-fail-2 — it's $O_i$ relabeled.** The early predictor recovers the same subspace the
recoverability probe already found. Then Gate C adds nothing over Gate B: a measurement was
moved earlier in the network without discovering that early features carry anything new.

**C-fail-3 — it's leakage.** This is the one the staged diagram $x \to E_\text{early} \to
E_\text{late} \to y$ hides. In a U-Net, depth-early is **not** information-early: enc1 features
reach the output through a skip that bypasses the bottleneck entirely, and early filters were
*trained through* the late loss. "Predictable from $E_\text{early}$" therefore does not license
"computable before paying the cost" unless the features used are ones a genuinely truncated
forward pass would have.

---

## Gate C — pre-registered tests and bars

$R^*$ is a **6-valued ordinal** (dyadic grid, observed `1..32` — see E154 Part 0). All
correlations are **Spearman**. Pearson on `R*` would be measuring arbitrary dyadic spacing and
is not admissible.

| # | Test | Bar |
|---|---|---|
| **C1** | Predictability: out-of-sample `rho(R̂*, R*)` | **`>= 0.5`** |
| **C2** | Cross-subject validity: cross-fitted, folds split by subject, no within-subject leakage | required, not scored |
| **C3** | Beyond difficulty: partial `rho` controlling lesion volume, baseline Dice, contrast statistic (E138) | **retains `>= 60%` of raw; permutation `p < 0.01`** |
| **C4** | Temporal availability: features come from a genuinely truncated forward pass | binary — pass/fail |
| **C5** | Specificity over $O_i$: `ΔR²` of `R̂*` over $O_i$ alone in predicting `R*` | **`>= 0.05`, permutation `p < 0.01`** |
| **C6** | Monotonicity: predicted-demand bins vs mean `R*` | **monotone across `>= 4` bins**, Spearman on bin means |

### On C3 vs C5

These are deliberately separate and fail independently, so a failure says *which* vacuity bit us.
C3 is the **difficulty** control (volume, baseline Dice, contrast). C5 is the **$O_i$**
control (incrementality over recoverability). C5 is the harder bar and the more informative one.

### On C4 — the operational definition

$E_\text{early}$ is restricted to what a truncated pass actually produces: **enc1/enc2
activations only**, no bottleneck, no decoder, no skip-connection contamination in the feature
set. Cross-fitted across subjects exactly as E143/E144 did for $O_i$ (which achieved pairwise
`rho` 0.786–0.974 and is the methodological template here).

**C4 is checked by re-running the predictor on features from a halted forward pass.** If the
correlation only survives with full-pass features, the "before paying the cost" claim is false
and the architectural opportunity does not exist — regardless of C1.

---

## Pre-registered falsifiers

Written down now, before seeing any number:

1. **If `ΔR² < 0.05` over $O_i$ (C5), Gate C is NOT PASSED even if C1 is high.**
   High raw `rho` with zero incrementality is precisely "recoverability predictor wearing a new
   name." Gate C collapses back into Gate B and contributes nothing.
2. **If C3 loses more than 40% of the raw correlation, Gate C is NOT PASSED.** Demand was
   difficulty.
3. **If C4 fails, Gate C is NOT PASSED** regardless of C1/C3/C5 — the result would be real but
   architecturally unusable.

## The one outcome worth the compute

Gate C passing **with a large `ΔR²` over $O_i$** is the single result that would distinguish this
branch from everything killed in E98–E153. It would be a *positive* claim about representation —
early features carry a demand signal the recoverability probe does not — rather than another
elimination. That, and only that, justifies proceeding to Gate D.

A Gate C pass still does **not** establish novelty. The remaining chain is:

$$
\text{early demand} \rightarrow \text{different computational contract} \rightarrow
\text{better accuracy/compute frontier}
$$

Only the final step is a potential algorithmic contribution.

---

## Sequencing note — audit novelty before Gate D, not after

Gate E currently sits after D. It should be **scoped before D**. "Estimate per-input capacity
demand, then allocate" is adjacent to adaptive-width networks, MoE with learned capacity
factors, and early-exit-with-budget. If E kills the branch, it should do so before D's training
compute is spent. E138 already cost a partial hit by auditing late.

**Recommended order:** E154 (B′, cheap, can kill outright) → E-scoping (cheap, can kill
outright) → E155 (Gate C) → Gate D.

---

## Dependency on E154 — do not run this first

E154 sets the **attenuation ceiling** on C1. If cross-seed reliability of `R*` is `r`, the
maximum achievable `rho(R̂*, R*)` is bounded roughly by `sqrt(r)`. If `r ≈ 0.25`, the ceiling is
`0.5` — exactly the C1 bar — and Gate C would fail mechanically while telling us nothing about
demand.

If E154 returns PARTIAL, **re-derive the C1 and C5 bars against the measured ceiling before
running this experiment.** Do not run E155 against these bars unchanged after a PARTIAL.

---

## Status

PRE-REGISTERED, not run. Blocked on E154.
