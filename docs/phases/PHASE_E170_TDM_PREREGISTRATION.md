# E170 — Task-Demand Matching (TDM): pre-registration

**Date**: 2026-09-16
**Status**: PRE-REGISTERED. Premise verified first (below). Not yet trained.
**The formula under test**:

$$\boxed{L_{\mathrm{TDM}} = \lambda \sum_{l\in\{\text{enc1},\text{enc2},\text{enc3}\}} \left[\max\left(0,\; \widehat R_l - R_{\mathrm{eff},l}\right)\right]^2}$$

**The scientific claim**: *segmentation performance is limited not by low rank per se, but by
rank falling below the task-required rank.*

---

## Premise verification — run BEFORE committing training compute

The project's own history (E141) established the discipline: verify the premise first, because a
penalty term that is identically zero trains the baseline. Four checks, all on existing data at
zero cost.

### 1. Does the deficit region exist? **YES**

$L_{\mathrm{TDM}}$ is zero unless $R_{\mathrm{eff},l} < \widehat R_l$.

| Stage | C | $R_{\mathrm{eff}}$ mean | $R^*$ mean | subjects with deficit | mean deficit |
|---|---:|---:|---:|---:|---:|
| **enc1** | 32 | 10.83 | 19.62 | **84.0%** | **9.25** |
| enc2 | 64 | 12.26 | 10.50 | 29.6% | 3.81 |
| enc3 | 128 | 16.39 | 6.78 | 6.4% | 1.28 |

The penalty is live, and it is **concentrated at enc1** — the opposite of where the E165 rank
*fraction* gradient would suggest. enc3 is nearly saturated (6.4%), so including it costs little
but buys little.

### 2. Is the deficit associated with worse outcomes? **YES, both ways**

| Stage | ρ(deficit, Dice) | ρ(deficit, $N_b$) |
|---|---:|---:|
| enc1 | **−0.539** (p<1e-4) | **+0.425** (p<1e-4) |
| enc2 | **−0.549** (p<1e-4) | +0.351 (p=1e-4) |
| enc3 | −0.359 (p<1e-4) | +0.094 (p=0.30, n.s.) |

Both signs are what TDM requires: more deficit → worse Dice, and more deficit → **higher
bottleneck necessity**. The second is the link E126 established *causally* (rank reduction at
pool3 drives $N_b$), so the mechanism has a causal leg, not just a correlational one.

### 3. Is it lesion size again? **NO — 54–64% retained**

| Stage | raw | partial \| log size (ET/TC/WT) | retained |
|---|---:|---:|---:|
| enc1 | −0.539 | **−0.345** (p=0.0001) | 0.64 |
| enc2 | −0.549 | **−0.296** (p=0.0008) | 0.54 |

### 4. Is the deficit circular? **NO at enc1/enc2**

The obvious objection: $R^*$ and $R_{\mathrm{eff}}$ are both functions of the same activation
spectrum, so their difference could be an algebraic artifact. Measured:

| Stage | ρ($R_{\mathrm{eff}}$, $R^*$) |
|---|---:|
| enc1 | **−0.092** |
| enc2 | **+0.030** |
| enc3 | +0.701 |

At enc1/enc2 the two quantities are **nearly independent** — the deficit is a real gap between
two things that do not track each other. **At enc3 they are strongly coupled (0.70), so enc3's
deficit is partly an algebraic artifact and its contribution must be reported separately.**

### 5. Which cohort does it target? **Both — including route (b)**

| Stage | mean deficit, 15-subject tail | mean deficit, the 110 |
|---|---:|---:|
| enc1 | 17.22 | **8.16** |
| enc2 | 18.73 | 1.77 |

This matters because E167 closed route (a) (information absent from input) and the only route
with headroom is **(b): +2pp on the 110 → +1.14pp overall**. The enc1 deficit is substantial in
the 110, so TDM is aimed at the reachable cohort.

**Premise verdict: VERIFIED on all five checks.** This is the first candidate this session to
clear a premise test before compute.

---

## Design

**Predictor $\widehat R_l$.** A lightweight head on the E169 feature set (activation spectral +
energy statistics from one intact forward pass), trained on E165's $R^*$ measurements.

**Leakage control — non-negotiable.** E169's folds are subject-level and seeded. $\widehat R_l$
must be produced **out-of-fold** for every subject; a predictor fitted on a subject and then used
in that subject's training loss would leak the truncation sweep into the method. Freeze the
predictor before segmentation training begins.

**$R_{\mathrm{eff},l}$** is the entropy-based effective rank of the live stage activation,
computed per batch, differentiable through the singular values.

**What TDM is not.** It does not allocate channels, change architecture, or add capacity. It
penalises only the **deficit** — surplus rank is free. This is what distinguishes it from the
closed allocation family (E71 sign backwards, E147 no scarcity, E72/E73 inert) and from generic
rank regularisation (which pushes rank up unconditionally).

**Sweep**: λ ∈ {0.01, 0.03, 0.1}, one seed each. Deliberately small — mechanistic test, not a
hyperparameter search.

---

## Pre-registered predictions and falsifiers

| # | Prediction | Falsifier |
|---|---|---|
| **P1** | $R_{\mathrm{eff},l}$ rises specifically for subjects with deficit > 0; near-unchanged for the rest | if rank rises uniformly, TDM is acting as a generic rank regulariser — **KILL as non-specific** |
| **P2** | $N_b$ falls (E126's causal link runs the right way) | if $N_b$ is unchanged while rank rises, the E126 pathway does not transfer — **KILL** |
| **P3** | Dice improves, concentrated in the high-deficit subjects | if Dice moves only on low-deficit subjects, the mechanism is not the stated one |
| **P4** | ΔDice ≥ **+1.0pp** on 3 seeds, per-subject, E56 protocol | below bar → report honestly as below bar |

**Mandatory control — the shuffled-deficit arm.** Train an identical run with $\widehat R_l$
**shuffled across subjects** (same marginal distribution, wrong subject assignment). If the
shuffled arm performs as well, TDM is a generic regulariser and the subject-specificity claim is
dead. This is E141's control design, which caught exactly this failure mode there.

**Gate order**: P1 → P2 → P3 → P4. A failure at P1 or P2 kills the branch without spending the
3-seed budget.

---

## Honest position on novelty

**Not claimed novel.** Adaptive rank is occupied (ARENA, GaRA-SAM; MICCAI 2025 low-rank
adaptation for organ segmentation), and generic rank regularisation is occupied (WERank, dynamic
rank adjustment — E162 verdict).

What is *less obviously* occupied is the specific chain:

$$\text{empirically measured task-required rank} \rightarrow \text{single-pass prediction} \rightarrow \text{sample-specific rank-\emph{deficit} penalty}$$

**That is not enough for a novelty claim, and a full audit has not been run.** Per this project's
own discipline the audit should come *before* the training compute — but here the premise
verification was the cheaper gate and it passed, so the ordering is: run the λ=0.03 arm plus its
shuffled control (P1/P2 only, single seed), and if P1 and P2 both hold, **then** run the hostile
prior-art audit before spending the 3-seed budget.

## Expected value, stated plainly

Seven audits returned OCCUPIED; E167 measured both routes to +1pp closed. TDM does not change
the input information (route a) or the label certainty (route b) — so **P4 is unlikely to clear
on this cohort** even if P1–P3 all hold. The realistic outcome is a mechanistic result
(rank deficit is causally consequential) rather than a Dice result.

That is worth the single-seed P1/P2 test. It is not yet worth the 3-seed budget.
