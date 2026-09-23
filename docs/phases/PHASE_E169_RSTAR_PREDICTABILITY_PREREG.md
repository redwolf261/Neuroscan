# E169 — Is $R^*_{i,l}$ predictable from a single forward pass?

**Date**: 2026-09-15
**Status**: PRE-REGISTERED. Not run. Diagnostic only — no architecture change, no new loss.
**Numbering**: filed as E169; **E168 is taken** by the cross-domain search.

---

## Two corrections to the framing, made before writing the protocol

### 1. $R^*$ is already label-free

The proposal states $R^*$ "is measured using the ground truth." **It is not.** E165's confound
guard, inherited verbatim from E147 and recorded in `E165_summary.json`:

> `"confound_guard": "agreement vs subject's OWN undegraded prediction, never ground truth"`

This was deliberate: measuring against ground truth would let a subject at Dice 0.95 fall
further than one at 0.30, manufacturing a correlation with difficulty.

**Consequence**: $X_i \to \hat R_{i,l}$ does not bridge a supervision gap, because there is no
gap. It asks whether $R^*$ can be obtained **without running the truncation sweep** — a
*compute* question, not a *label* question.

This is still worth asking, but it changes what a pass means. It would show $R^*$ is a cheap
statistic, not that a retrospective quantity became prospective.

### 2. The proposed algorithm terminates on a closed rule

The stated target is $X \to \hat R_l \to$ **allocate computation/capacity** $\to \hat Y$. That is
the pre-rejected family, closed three independent ways:

| Route | Evidence |
|---|---|
| E71 | ρ(ablation-sensitivity, skip-suppression-tolerance) = **−0.290**, p=0.001 — **sign backwards** from what allocation requires |
| E147 | demand↔capacity **inverse** (partial ρ=−0.496); **74/88 subjects served by rank ≤4 of 256** ⇒ no scarcity to allocate |
| E72/E73 | loss-weighting and spatial gating on that signal, both inert |

`PHASE_E158` records this as a standing pre-rejection rule. **A successful E169 does not reopen
it.** Predicting a quantity does not create a resource to allocate.

**So E169 is run as a diagnostic that can only KILL or PARK, never as a GO.** Written this way
deliberately: the honest reason to run it is that outcome A closes the branch cheaply.

---

## Exact definition of $R^*$ as measured in E165

Required by the proposal, and stated so the predictand is unambiguous:

> For subject $i$ and stage $l$ with $C_l$ channels: hook the stage's output tensor, flatten to
> $(C_l, N_{\text{voxels}})$, mean-centre across voxels, take the SVD over the channel axis, and
> reconstruct with rank $r$. Run the **rest of the network normally** and threshold at 0.5.
> Agreement is the mean binary Dice over ET/TC/WT against the **intact prediction** of the same
> subject. $R^*_{i,l}$ is the **first** $r$ on the grid $\{1,2,4,8,16,32,64,128,256\}$ (capped at
> $C_l$) reaching agreement $\geq 0.90$; if none does, $R^* = C_l$.

Verified sanity at run time: checkpoint identity asserted; dormant hook an **exact** no-op;
rank=$C$ truncation max abs diff **0.000e+00**.

Measured values (n=125): enc1 R\*/C **0.613**, enc2 0.164, enc3 0.053, bottleneck **0.018**,
dec1 0.154.

---

## Protocol

### Predictors — must come from ONE forward pass

This is the binding constraint. E165's agreement curves are already stored, so any predictor
built from the truncation sweep is **not cheaper than the measurement itself** and would be
circular. Admissible features, all computable from a single intact forward pass:

| Family | Features |
|---|---|
| Spectral | effective rank, participation ratio, stable rank, top-$k$ singular-value mass, spectral entropy of the stage activation |
| Energy | mean/std/max ‖activation‖, per-channel energy dispersion, fraction of near-dead channels |
| Entropy | activation entropy, per-channel occupancy |
| Readout | predicted foreground volume per region, mean/max predicted probability, predicted-boundary voxel fraction |
| Image-only | intensity moments per modality, the E138 contrast statistic |

**Excluded**: anything requiring a second forward pass, and anything derived from `agree_curve`.

### Estimator

Simple and cross-fitted, per the proposal: ridge and gradient-boosted regression on
$\log_2 R^*$ (the grid is dyadic, so log-scale is the natural target), plus Spearman on raw rank.
**No neural network.** 5-fold cross-fitting **by subject**, folds fixed by a seeded split before
any feature is computed.

### Reported per stage

Out-of-sample $R^2$ and Spearman $\rho$; the same **after partialling out lesion volume**
(size_ET / size_TC / size_WT from `E160_L_per_subject.json`); and a permutation null (1,000
shuffles of the target within folds).

### Pre-registered decision rule — fixed now

| Outcome | Criterion | Verdict |
|---|---|---|
| **A — no predictability** | out-of-sample $R^2 \leq 0.10$ at every stage, or permutation $p > 0.05$ | **KILL the adaptive-rank branch.** $R^*$ is obtainable only by running the sweep |
| **B — size proxy** | $R^2 > 0.10$ raw but **loses ≥50%** of $R^2$ after partialling lesion volume | **PARK.** Real but weak — a restatement of lesion size, which E165 already shows correlates with rank demand |
| **C — representation-driven** | $R^2 > 0.10$ **and** retains ≥50% after the volume control **and** permutation $p < 0.01$ | **PARK with a note.** $R^*$ is a cheap statistic. This does **not** authorise an allocation design (see correction 2); any use must be argued against E71/E147/E72/E73 first |

**Note on the bar.** $R^2 > 0.10$ is deliberately low for a *kill* threshold — it must be easy to
clear, so that failing it is decisive. It is not a success bar.

### Anticipated result, recorded before running

**Outcome B or C at enc1, A at the bottleneck.** E165 showed the bottleneck is 90% served by
rank ≤4 — near-constant, so there is little variance to predict. enc1 has real spread
(R\* ∈ {4,8,16,32}, 31% of subjects above half capacity) and is the only stage where prediction
is even well-posed. Recording this so the result cannot be read as confirmation.

---

## Cost

One forward pass per subject over 125 subjects (~15 min GPU), then CPU-only regression. No
training, no truncation sweep — E165's stored `R_star` values are the target.

## What this can and cannot conclude

**Can**: whether $R^*$ is a cheap forward-pass statistic or requires the sweep.
**Cannot**: whether anything can be built on it. The allocation family is closed by direct
evidence, and E167 has separately closed both routes to the ≥1pp bar. E169 is a diagnostic about
a measurement, not a route to a contribution.

If outcome A: the adaptive-rank branch closes and E165 remains what
`PHASE_E165_E167_CLOSURE_EVIDENCE_CHAIN` already says it is — part of the mechanistic
investigation that justified terminating the architecture search.
