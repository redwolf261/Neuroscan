# Measuring the Information Frontier in Brain-Tumour Segmentation

**A model-independent estimator of per-subject recoverability, and what it reveals about
where segmentation error actually comes from**

Rivan Avinash Shetty · 2026-09-20
Dataset: BraTS 2023 GLI (1,251 subjects) · Model: UNet3D_v5, frozen checkpoint
`E131_v5control_seed0`

---

## Summary

Segmentation models are usually improved by changing the model. This report takes the opposite
approach: it measures how much of the remaining error is *attributable to the input rather than
the model*, using a quantity computed **without any segmentation network**.

The central object is a per-subject scalar $O_i$, the **recoverability** of the enhancing-tumour
(ET) label from raw voxel intensities. It is estimated by simple per-voxel classifiers that are
cross-fitted — each subject is scored by an observer trained only on *other* subjects.

Three results follow:

1. $O_i$ raises out-of-sample $R^2$ for predicting ET segmentation error from **0.12 to 0.69**,
   where fifteen standard difficulty covariates reach only 0.12. This holds across **six
   independently trained models**.
2. There is **no subject** where the label is recoverable but the model fails to recover it
   ($n = 0/90$). The model operates *at* the recoverability frontier.
3. Consequently, **no post-hoc method can gain ≥1pp Dice on this cohort.** The oracle that
   perfectly exploits $O_i$ yields +0.377pp. This is proved arithmetically, not inferred from
   failed attempts — though eleven failed attempts independently corroborate it.

**There is no Dice improvement in this work, and that is the result.** The contribution is a
measurement showing why improvement is not available, together with an estimator that predicts,
before segmentation, which subjects will fail.

---

## 1. The problem

BraTS scores three nested regions: enhancing tumour (ET), tumour core (TC), and whole tumour
(WT). The frozen baseline achieves:

| region | Dice |
|---|---:|
| ET | 0.8191 |
| TC | 0.8481 |
| WT | 0.9119 |
| **mean** | **0.8597** |

The benchmark is saturated — *nnU-Net Revisited* (MICCAI 2024) states outright that "scores on
BraTS21 are saturated," and challenge-winning ET scores moved from 0.8245 (2018) to 0.8203
(2020). Roughly twenty-five architectural and training interventions in this project failed to
clear a pre-registered +1pp bar.

The usual interpretation is that the model is imperfect. The alternative, tested here, is that
**the information required is not present in the input for the subjects that fail.**

---

## 2. The model: what $O_i$ is and what it does

### 2.1 Intuition

Ask a deliberately weak question: *ignoring all spatial context, can the ET label be predicted
from voxel intensities alone?*

If yes, the label is *intensity-recoverable* — a competent segmentation network should get it.
If no, no amount of architecture will help, because the evidence is absent from the numbers.

$O_i$ quantifies this per subject. Critically, it is computed **with no segmentation model
involved**, which is what allows it to serve as an external reference rather than a restatement
of the model's own confidence.

### 2.2 Formal definition

For subject $i$, let $\mathcal{V}_i$ be the set of tumour voxels and for $v \in \mathcal{V}_i$
let

$$
x_v = \big(z_{t1c}(v),\; z_{t1n}(v),\; z_{t2f}(v),\; z_{t2w}(v)\big) \in \mathbb{R}^4
$$

be the four z-scored modality intensities (normalised within the brain mask). Let
$y_v \in \{0,1\}$ indicate ET membership.

Partition the $N$ subjects into $K=5$ folds. For fold $k$, train an observer $f^{(-k)}$ on the
subjects **outside** that fold:

$$
f^{(-k)} \;=\; \arg\min_{f \in \mathcal{F}} \;
\sum_{j \notin \mathcal{F}_k} \sum_{v \in \mathcal{V}_j}
\mathcal{L}_{\text{BCE}}\big(f(x_v),\, y_v\big)
$$

Then for each held-out subject $i \in \mathcal{F}_k$:

$$
\boxed{\;
O_i \;=\; \mathrm{Dice}\Big(\mathbb{1}\big[f^{(-k)}(x_v) \ge \tfrac12\big]_{v \in \mathcal{V}_i},\;\; y_i\Big)
\;}
$$

**$O_i$ is the Dice a context-free intensity classifier achieves on subject $i$, having never
seen subject $i$.** It is an empirical, cross-fitted recoverability estimate.

### 2.3 The four observers

To ensure $O_i$ measures a property of the *image* rather than of one estimator, four
deliberately different $\mathcal{F}$ are used:

| observer | form | parameters |
|---|---|---|
| `simple_thresh` | global threshold on $\delta = z_{t1c} - z_{t1n}$ | 1 |
| `mlp_small` | MLP $4 \to 8 \to 1$ | ~50 |
| `mlp_deep` | MLP $4 \to 64 \to 64 \to 1$ | ~4.5k |
| `mlp_spatial` | MLP $6 \to 64 \to 64 \to 1$, adding local mean/std of $\delta$ in a $5^3$ window | ~4.6k |

Span: a one-parameter threshold to a small spatial network — four orders of magnitude in
capacity.

### 2.4 Scope conditions

Three properties define what $O_i$ is and is not, and each is load-bearing:

- **Model-independent** — no segmentation network appears anywhere in its computation. This is
  what makes claims about the model non-circular.
- **Cross-fitted, not label-free** — ground truth trains the observers and restricts voxels to
  the tumour mask. $O_i$ is blind to the *target subject's* labels, not to labels in general.
  It is a research instrument, not a deployable pre-segmentation tool.
- **Empirical, not information-theoretic** — no Shannon quantity is estimated. $O_i$ is a
  measured recoverability boundary under a specific observer class.

---

## 3. Results

### 3.1 $O_i$ measures the image, not the estimator

All six pairwise Spearman correlations between observers' subject rankings ($n=90$):

| | simple | small | deep | spatial |
|---|---:|---:|---:|---:|
| **simple_thresh** | 1.000 | 0.786 | 0.924 | 0.895 |
| **mlp_small** | 0.786 | 1.000 | 0.806 | 0.811 |
| **mlp_deep** | 0.924 | 0.806 | 1.000 | 0.974 |
| **mlp_spatial** | 0.895 | 0.811 | 0.974 | 1.000 |

A one-parameter threshold ranks subjects at $\rho = 0.924$ with a 4,500-parameter network.
Recoverability is carried by the contrast relationship itself, not by estimator capacity.

### 3.2 $O_i$ explains error far beyond standard difficulty

Target $E_i = 1 - \mathrm{Dice}_{ET}(i)$. Baseline $Z$: fifteen pre-specified covariates —
log ET size, log tumour size, distance of centre-of-mass from brain centre, superior–inferior
position, number of connected components, log largest component, ET fraction of tumour, and
per-modality mean and standard deviation inside the tumour. Ridge regression, 5-fold, **out of
sample**:

$$
R^2(Z) = 0.1243 \qquad R^2(Z \cup \{O_i\}) = 0.6874 \qquad
\boxed{\Delta R^2 = +0.5631}
$$

Permutation null (200 shuffles of $O_i$): mean $+0.0152$, sd $0.0395$, observed $+0.5631$ —
approximately **14 standard deviations**, $p = 0.0000$.

### 3.3 It transfers across six independently trained models

$O_i$ is frozen and regressed against each model's ET error:

| model | mean ET | $R^2(Z)$ | $R^2(Z{+}O)$ | $\Delta R^2$ | perm $p$ | $\rho(O_i,\mathrm{Dice})$ |
|---|---:|---:|---:|---:|---:|---:|
| E130_baseline_seed0 | 0.8387 | 0.1470 | 0.6861 | +0.5391 | 0.0000 | 0.837 |
| E131_v14_seed0 | 0.8359 | 0.1081 | 0.5594 | +0.4512 | 0.0000 | 0.842 |
| E131_v5control_seed0 | 0.8441 | 0.1243 | 0.6874 | +0.5631 | 0.0000 | 0.872 |
| E137_v16_seed0 | 0.8329 | 0.1377 | 0.6772 | +0.5395 | 0.0000 | 0.840 |
| E141_evidence_ep30 | 0.8362 | 0.1021 | 0.5539 | +0.4517 | 0.0000 | 0.856 |
| E141_shuffled_ep30 | 0.8365 | 0.0967 | 0.5486 | +0.4519 | 0.0000 | 0.827 |

$O_i$ predicts the **deliberately corrupted** control (`E141_shuffled`) as well as the others.
Subject-level recoverability governs where even a damaged model fails.

*Scope limit*: all six are UNet3D variants (inter-model $\rho$ = 0.936–0.980). This establishes
cross-checkpoint, not cross-architecture, generality.

### 3.4 The frontier: no recoverable-but-unrecovered population

Define **Regime III** = subjects with $O_i \ge 0.75$ (recoverable) but model Dice $< 0.50$
(failed).

$$
\boxed{\text{Regime III} : n = 0 \text{ of } 90}
$$

| stratum | mean $O_i$ | mean model ET |
|---|---:|---:|
| lowest $O_i$ quartile | 0.538 | 0.605 |
| remainder | 0.901 | 0.926 |

The model tracks recoverability. Where information is recoverable, it is already recovered.

### 3.5 The boundary: it fails for tumour core

The identical pipeline, with TC substituted for ET (methodology frozen; only definitionally
region-indexed variables changed):

| | ET | TC |
|---|---|---|
| observer agreement | 0.786 – 0.974 | **0.549 – 0.870** |
| $\Delta R^2$, 6 models | +0.451 … +0.563 | **−0.046 … +0.099** |
| models with negative $\Delta R^2$ | 0 of 6 | **3 of 6** |

A 5–10× collapse, with sign reversal on half the models. **The phenomenon is ET-specific.**

The mechanism is independently established. ET is defined by a *local intensity relationship*:
$\rho(\text{contrast}, \text{Dice}) = +0.595$, and zeroing $t1c$ collapses ET from 0.8433 to
0.0015. A context-free per-voxel observer can evaluate that rule. TC is a *compound anatomical*
region — necrotic core is defined by morphology and context, which such an observer cannot see.

$$
\boxed{\text{The signal exists where the label is defined by a local intensity relationship,
and fails where it is defined by context and morphology.}}
$$

---

## 4. The gain: there is none, and here is the proof

This section answers the question directly. The honest answer is that **no Dice gain is
available on this cohort**, and the reason is arithmetic.

### 4.1 The $O_i$ oracle

Consider the most generous conceivable algorithm: one that lifts every subject to its own
measured recoverability, $\mathrm{Dice}^{\text{oracle}}_i = \max(\mathrm{Dice}_i, O_i)$. No
method consistent with $O_i$ can exceed this.

| quantity | value |
|---|---:|
| ET mean, current | 0.8441 |
| ET mean, perfect oracle | 0.8554 |
| ET gain | +1.132 pp |
| **3-region mean gain** | **+0.377 pp** |
| pre-registered bar | **≥ 1.0 pp** |

The entire family is capped at roughly one third of the bar. The reason is visible in the gap
distribution: mean $O_i - \mathrm{Dice} = -0.036$. **The model beats $O_i$ on average.** Only 5
of 90 subjects have $O_i > \mathrm{Dice} + 0.05$.

### 4.2 The whole outcome space

The same gate applied to every region of the error, not just $O_i$'s:

| source | residual | interior (non-boundary) fraction | max 3-region gain |
|---|---:|---:|---:|
| ET | 0.1809 | 0.014 | +0.084 pp |
| TC | 0.1519 | 0.028 | +0.142 pp |
| WT | 0.0881 | 0.150 | +0.441 pp |
| **total** | | | **+0.667 pp** |

$$
\boxed{\text{Perfecting ALL interior error in ALL three regions} = +0.667\text{ pp} < 1.0\text{ pp}}
$$

Boundary error is separately bounded by the annotation ceiling: the model scores ET **0.900**
against a published inter-rater median of **0.77** — it already agrees with the reference more
than radiologists agree with each other.

### 4.3 The one region with room, and why it is closed

The 15 catastrophic-tail subjects (ET or TC < 0.5) are a different failure mode — whole-region
misses, not boundary error. Lifting them to 0.40 would yield **+1.940 pp**.

A leave-one-out transfer test was pre-registered to decide whether a rule exists that recovers
them:

| training set | tail ET Dice |
|---|---:|
| the subject itself (**leaky**, upper bound) | 0.741 |
| the other 14 tail subjects (**the test**) | **0.193** |
| 30 good-stratum subjects | 0.352 |
| tail + good | 0.319 |

The rule does not transfer. Trained honestly, the observer scores **0.193** — below the CNN it
was meant to beat. Adding tail data *hurts* (0.352 → 0.319 → 0.193), so the tail subjects are
not a coherent population with a shared alternative decision boundary; they are individually
idiosyncratic. The best honest arm yields **+0.827 pp**, still short.

### 4.4 Consolidated closure

| route | status | evidence |
|---|---|---|
| input information | closed | zeroing $t1c$ → ET 0.0015 (causal) |
| boundary residual | closed | 98.6% of error within 3 voxels; 0.900 vs 0.77 inter-rater |
| interior residual | closed | +0.667 pp total, arithmetic |
| capacity / allocation | closed | Regime III empty, $n=0/90$ |
| adaptive computation | closed | routing ≤ constant baseline |
| latent geometry | closed | effect entirely readout-mediated (max $7\times10^{-5}$) |
| recoverability-driven action | closed | oracle +0.377 pp |
| catastrophic tail | closed | transfer 0.193 < model 0.227 |

Within the measured error decomposition and these mechanism classes, no ≥1pp headroom is
identified. *Arithmetic closes mechanism classes, not the space of conceivable algorithms* — a
method obtaining genuinely new information (additional acquisition, external priors) is not
excluded by this analysis.

---

## 5. Novelty

Audited against 2024–2026 prior art. The nearest work is **Fournel et al., MICCAI 2025**,
"Difficulty Estimation for Image-Specific Medical Image Segmentation Quality Control," read in
full:

| | Fournel et al. | $O_i$ |
|---|---|---|
| target | mean **annotator** DSC (expert disagreement) | **model** error |
| inputs | radiomics extracted *from the model's prediction* + uncertainty maps | raw intensities, no network |
| requires segmentation first | **yes** | **no** |
| cross-model validation | no | **6 models** |
| reported failure region | none | **pre-registered TC failure** |

Their finding that input properties govern segmentation difficulty is the correct related-work
anchor. The construction here differs in target, inputs, and validation. Other neighbours —
uncertainty calibration, failure detection, EvanySeg, diffusion-based QC — all take model output
as input and are therefore post-segmentation and circular for the frontier question.

---

## 6. Limitations

- **ET only**, $n=90$ (ET ≥ 200 voxels). Does not hold for TC; WT untested.
- **One dataset, one architecture family.** No external cohort; all six models are UNet3D
  variants.
- **Cross-fitted, not label-free.** GT trains the observers. Not deployable as-is.
- **Empirical, not a bound.** No information-theoretic quantity is estimated.
- $\rho(O_i, \text{Ridge residual}) = -0.147$, $p = 0.17$, not significant — $O_i$'s power is
  largely linearly independent of $Z$ rather than concentrated in its residual. Both statistics
  are reported; the $\Delta R^2$ + permutation test is the sounder one.

---

## 7. Method notes

Every number above was recomputed from a stored artifact rather than cited from notes. Three
measurement artifacts were caught and corrected during this work:

- An earlier recoverability estimate (0.531) was fitted and evaluated on the same subject's
  labels; cross-fitting corrected the associated transfer estimate by **4.6×**.
- A reported readout gap of +0.0388 AP was recomputed as **+0.0316**, with two of three probe
  sites negative.
- The in-subject tail oracle (0.741) collapsed to **0.193** under honest transfer.

All three are the same artifact — in-sample evaluation masquerading as capability. It is the
dominant failure mode in this kind of analysis, and the leaky/honest contrast is retained in the
archive deliberately.

Code: `experiments/exp_recoverability/` (7 scripts, run in order, seeds fixed).
Verification: the reconstructed $\Delta R^2$ script reproduces its original log bit-identically.
