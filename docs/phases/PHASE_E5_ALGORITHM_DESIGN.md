# Phase E5: Algorithm Design Specification — Evidence-Guided Geometry Optimization (EGGO)

**Status**: Draft specification — mathematics only, no code yet

**Date**: 2026-08-04

## 0. Why this document exists

Following the ABO precedent: Phase C did not start with code. It started
with a design brief that every constant traced to a measurement. This
document applies the same discipline to the new algorithm motivated by
Phase E's diagnostic chain (Phase D → E1.1 → E1.2 → E1.3 → E1.4 →
E2 literature review → this loss-math literature search). No
implementation should begin until this specification is reviewed.

## 1. Why this algorithm exists — the diagnostic chain, restated as justification

| Phase | Question | Finding |
|---|---|---|
| D | Does gradient magnitude imbalance limit Dice? | No (null result, 50ep×3seeds) |
| E1.1 | Does image-space boundary distance explain evidential uncertainty? | Weakly (R²=0.150) |
| E1.2 | Does latent feature magnitude explain it? | Partially, class-conditional, not usable globally |
| E1.3 | Does latent decision-boundary distance explain it? | Better (R²=0.219); incorrect voxels concentrate near this boundary (Cohen's d=1.41) |
| E1.4 | Does local manifold density explain the remainder? | Yes, independently (r≈0 vs. E1.3's boundary distance; Cohen's d=−3.0 for correct/incorrect separation; combined R²=0.140) |
| E2 | Is a margin/density loss on latent embeddings already standard? | The general strategy yes; the specific combination (evidential-head-gated, dual independently-weighted terms, local-not-global density centroid) no |

**Conclusion driving this design**: uncertainty in this model is best
explained by two measurably independent latent-geometry properties —
proximity to the network's own decision boundary, and local
representation-manifold sparsity — neither of which is well explained by
image-space geometry or feature magnitude alone. The algorithm should
therefore act on these two properties directly, as two separate
mechanisms, gated by the evidential head's own uncertainty output (which
Phase E1.1–E1.4 established as more informative than either boundary
measure alone at separating correct from incorrect voxels).

## 2. Variable definitions and normalization

All quantities computed from the shared trunk's `dec1` feature vector
$z_i \in \mathbb{R}^{32}$ for voxel $i$, exactly as extracted in
`experiments/exp_e_latent_analysis/extract_features.py`.

### 2.1 Uncertainty gate input: $U_i$

From the evidential head's Beta parameters $\alpha_i, \beta_i \geq 1$.
Total evidence $e_i = \alpha_i + \beta_i - 2 \in [0, \infty)$ (measured
range in the 30-volume extraction: $[0.121, 33.319]$, mean 19.78, heavily
left-skewed with a long low-evidence tail — the tail E1.1 showed contains
the incorrect voxels).

Define normalized **uncertainty** (inverse of evidence, since the
controller should fire on *low*-evidence voxels):

$$\hat{U}_i = 1 - \text{clip}\!\left(\frac{e_i}{e_{p99}}, 0, 1\right)$$

where $e_{p99}$ is the 99th percentile of evidence measured on the
training set (empirically ≈23.25 in the 30-volume sample; must be
re-measured on the full training set before use, not hard-coded from this
subsample). Clipping at the 99th rather than the max avoids a handful of
outlier high-evidence voxels compressing the entire useful range.

### 2.2 Latent boundary proximity: $B_i$

From E1.3's method: fit a linear classifier (logistic regression,
class-balanced) separating tumor/background in $z$-space per training
batch (or periodically, not necessarily every step — see §4 complexity
discussion). Signed distance:

$$d_i = \frac{w \cdot z_i + b}{\lVert w \rVert}$$

Convert to a smooth, differentiable proximity weight:

$$B_i = \exp(-|d_i| / \tau_b)$$

**$\tau_b$ derivation (measured, not assumed)**: E1.3 found mean
$|d_i|=1.456$ for correct voxels and $0.648$ for incorrect voxels. Set
$\tau_b$ at the geometric mean of these two crossover points,
$\tau_b = \sqrt{1.456 \times 0.648} \approx 0.972$, so that $B_i \approx
e^{-1} \approx 0.37$ at the empirical correct/incorrect crossover — this
mirrors exactly how ABO's `r_target` was derived from a measured
crossover rather than picked arbitrarily. **This must be re-derived from
the full training set's own measurement before locking**, exactly as
ABO's $r_{target}=0.294$ was later validated via the C2a sweep — treat
this initial value as a starting point for an analogous sensitivity
sweep, not a final constant.

### 2.3 Local density: $D_i$

From E1.4's method: $\rho_i$ = mean distance to $k{=}20$ nearest
neighbors in $z$-space (measured range: $[0.0195, 9.87]$, mean 0.233;
correct-voxel mean 0.229, incorrect-voxel mean 2.660).

$$D_i = 1 - \exp(-\rho_i / \tau_\rho)$$

**$\tau_\rho$ derivation**: mirroring §2.2, set $\tau_\rho$ near the
measured correct/incorrect crossover. Geometric mean of 0.229 and 2.660
gives $\tau_\rho \approx 0.781$. Same caveat: re-derive from the full
training set, treat as a sweep starting point.

$D_i \to 0$ in dense regions (low $\rho_i$), $D_i \to 1$ in sparse
regions (high $\rho_i$) — same direction convention as $B_i$ (both →1
means "this voxel needs attention").

### 2.4 Summary table

| Symbol | Meaning | Source | Range |
|---|---|---|---|
| $z_i$ | dec1 feature vector | extraction pipeline | $\mathbb{R}^{32}$ |
| $\hat U_i$ | normalized uncertainty (gate input) | evidential head | $[0,1]$ |
| $d_i$ | signed latent boundary distance | E1.3 classifier | measured $\approx[-3,3.4]$ |
| $B_i$ | boundary-proximity weight | derived from $d_i$ | $(0,1]$ |
| $\rho_i$ | k-NN mean distance (raw density) | E1.4 method | measured $\approx[0.02,9.9]$ |
| $D_i$ | sparsity weight | derived from $\rho_i$ | $[0,1)$ |

## 3. Geometry controller

### 3.1 The gate

Per Step 6 of the original design discussion: uncertainty gates whether
optimization happens at all; boundary and density independently
determine *what kind* of correction, when it does.

$$g(\hat U_i) = \hat U_i$$

(Identity gate as the starting form — simplest option satisfying "low
uncertainty → near-zero force." A sharper gate, e.g.
$g(\hat U_i) = \hat U_i^\kappa$ or a sigmoid threshold, is a tunable
variant, not the base design; keep the base case simple per Occam's
razor until a sweep shows a sharper gate is needed.)

### 3.2 Two independently-weighted forces, not one merged scalar

$$w_{sep}(i) = g(\hat U_i) \cdot B_i$$

$$w_{comp}(i) = g(\hat U_i) \cdot D_i$$

Explicitly **not** $G_i = \hat U_i \cdot B_i \cdot D_i$ (rejected in the
original design discussion: a low-density-but-far-from-boundary voxel
would incorrectly get near-zero total weight even though it may need
compactification specifically) and **not** $G_i = \hat U_i + B_i + D_i$
(rejected: a certain, distant, sparse voxel would incorrectly trigger
optimization from density alone). The multiplicative-gate/separate-force
structure directly encodes the four cases (A–D) worked through in the
original design discussion.

## 4. Loss derivation (full attribution)

Two additive terms, applied per training batch over a set of anchor
voxels $B$ (subsampled — see §5 for why dense per-voxel computation is
infeasible).

### 4.1 Separation term

$$\mathcal{L}_{sep} = \frac{1}{|B|}\sum_{i \in B} w_{sep}(i) \cdot \frac{1}{|N(i)|}\sum_{j \in N(i)} \big[\,2\delta_d - \lVert z_i - z_j \rVert\,\big]_+^2$$

where $N(i)$ is a same-minibatch set of opposite-class voxels (in-batch
negative sampling, per the tractability finding in §5).

**Attribution**: the hinge/margin skeleton
$[\,2\delta_d - \lVert \cdot \rVert\,]_+^2$ is taken directly from De
Brabandere, Neven & Van Gool (2017), *Semantic Instance Segmentation with
a Discriminative Loss Function*, Eq. 2 (their $L_{dist}$, originally
defined between cluster centers; extended here to per-voxel pairs against
an in-batch negative set, mirroring how DyCON and Multi-Similarity Loss
structure their negative-pair sums). The per-voxel multiplicative
weight $w_{sep}(i)$ — combining the evidential-uncertainty gate with the
boundary-proximity weight — is structurally the same mechanism as
DyCON's $\mathbf{F}_q^- = (S_{iq})^\gamma$ and $\exp(H_{gs}(p_i^s))$
multiplicative gates (An et al., CVPR 2025, *DyCON*), substituting this
project's evidential/Beta uncertainty for DyCON's softmax entropy.

### 4.2 Compactification term

$$\mathcal{L}_{comp} = \frac{1}{|B|}\sum_{i \in B} w_{comp}(i) \cdot \big[\,\lVert z_i - \mu_{c(i)}^{local} \rVert - \delta_v\,\big]_+^2$$

where $\mu_{c(i)}^{local}$ is the mean of voxel $i$'s $k=20$ nearest
same-class neighbors' feature vectors (the same neighbor set already
computed for $\rho_i$ in §2.3 — no additional k-NN computation needed) —
**a local, not global, per-class centroid**.

**Attribution**: the hinge/margin pull-to-center skeleton is taken from
De Brabandere et al.'s $L_{var}$ (Eq. 1) and is mathematically equivalent
to Wen et al. (2016) *center loss* $L_C = \frac12\sum \lVert x_i -
c_{y_i}\rVert_2^2$ (ECCV 2016, Eq. 2). **The local-centroid choice (as
opposed to the classical single global centroid both source papers use)
is this project's own contribution**, directly motivated by E1.3's
finding that the tumor cluster is ~3× more diffuse than background (a
single global tumor centroid would poorly represent such a diffuse
cluster) and E1.4's finding that density varies substantially *within*
the tumor cluster itself. The density-based per-sample weighting
mechanism ($w_{comp}(i)$ incorporating $D_i$) follows the precedent of
Wang et al. (CVPR 2023, *DGCL — Density-Guided Contrastive Learning*) and
the density-aware medical segmentation paper (arXiv:2412.19871), both of
which use k-NN-derived density as a differentiable per-sample loss
weight rather than only a diagnostic tool.

### 4.3 Combined loss

$$\mathcal{L} = \mathcal{L}_{seg} + \lambda_{sep}\,\mathcal{L}_{sep} + \lambda_{comp}\,\mathcal{L}_{comp}$$

$\mathcal{L}_{seg}$ is the frozen baseline's FocalTversky segmentation
loss (per `PHASE_A5_BASELINE_FROZEN.md`), unchanged. The evidential
branch's own loss (Beta-KL) is also unchanged — this design modifies only
the shared trunk's representation via an auxiliary geometry loss, not the
segmentation or evidential objectives themselves. $\lambda_{sep},
\lambda_{comp}$ are the two top-level scalars requiring their own
sensitivity sweep (C2a/C2b-style), analogous to ABO's $r_{target}$/$\gamma$.

### 4.4 What is explicitly borrowed vs. genuinely novel (honesty check per E2's guidance)

| Component | Status |
|---|---|
| Hinge/margin loss skeleton (both terms) | **Borrowed** — De Brabandere et al. 2017, Wen et al. 2016 |
| Two independently-weighted additive terms | **Borrowed structure** — De Brabandere et al. 2017 already does this |
| Uncertainty-gating a metric-learning term via a separate prediction head | **Borrowed mechanism** — DyCON, An et al. 2025 |
| Density as a differentiable per-sample weight | **Borrowed mechanism** — DGCL (Wang et al. 2023), arXiv:2412.19871 |
| Gating specifically by an evidential/Beta-distribution head (not softmax entropy) | **Novel substitution** — not found combined with the above in the literature search |
| Local (not global) density-weighted centroid for compactification | **Novel, and specifically motivated by this project's own E1.3/E1.4 measurements** — flagged as a genuine gap in the base literature |
| The specific empirical justification (latent boundary + density jointly explain evidence better than either alone, R²=0.140) | **This project's own diagnostic contribution**, independent of what algorithm follows it |

Per E2's explicit guidance: **do not claim** the margin/hinge loss form
or the general "uncertainty-weighted metric learning" strategy as novel —
both are established, and this table exists precisely so no borrowed
mathematics gets presented as invented.

## 5. Complexity analysis

### 5.1 The critical constraint: dense per-voxel computation is infeasible

Training config (`configs/brats.yaml`): batch_size=8, volume shape
64×64×64 = 262,144 voxels/volume → **2,097,152 voxels per training
batch** if every voxel were used as an anchor. E1.4's diagnostic used a
pooled, static set of 60,000 voxels (2,000/volume × 30 volumes) — the
full dense training batch is **~35× larger** than the diagnostic scale,
and unlike the diagnostic (a one-time offline analysis), this must run
**every training step**.

Naive brute-force pairwise k-NN cost at full batch density: $O(n^2 d)$
with $n{=}2{,}097{,}152$, $d{=}32$ → on the order of $10^{14}$
floating-point operations per batch. **Computationally infeasible.**
Full pairwise graph-Laplacian-style terms are ruled out entirely at this
scale (confirmed by the literature search: every source treating this
problem sparsifies to a local kNN graph or uses in-batch sampling — see
`PHASE_E5_LOSS_MATH_LITERATURE.md` §"Tractability findings").

### 5.2 Required mitigations (load-bearing design decisions, not optional optimizations)

1. **Voxel subsampling per volume.** Anchor set $B$ must be a subsample
   of each volume's voxels, not the full volume. E1.4's rate (2,000/volume)
   is a reasonable starting point (already validated not to break the
   diagnostic signal), giving $8 \times 2{,}000 = 16{,}000$ anchors/batch
   — a ~130× reduction from the dense case.
2. **Stratified sampling toward uncertainty**, not uniform random: since
   $\hat U_i$ is cheap to compute for every voxel (a forward pass, no
   pairwise computation), sampling should oversample high-uncertainty
   regions — this both reduces wasted compute on voxels the gate would
   zero out anyway and improves the effective sample size of the
   voxels that matter (recall: only ~0.14% of voxels were incorrect in
   the diagnostic sample — uniform random sampling would rarely include
   them).
3. **In-batch-only positive/negative sets.** $N(i)$ (opposite-class
   negatives for $\mathcal{L}_{sep}$) and the same-class neighbor set
   for $\mu_{c(i)}^{local}$ must be restricted to the current minibatch's
   anchor set, not the full dataset — standard practice per every
   precedent found (SupCon, DyCON, Multi-Similarity, DGCL).
4. **Boundary classifier refresh frequency.** Refitting the logistic
   classifier for $d_i$ every single training step is unnecessary
   overhead; refresh periodically (e.g., every N steps or once per epoch)
   using the current batch's or a recent buffer's features, accepting
   slight staleness — needs empirical validation that staleness doesn't
   materially degrade $B_i$'s quality, not yet tested.

### 5.3 Estimated overhead (rough, pre-implementation)

With the above mitigations: k-NN over ~16,000 anchors × 32 dims is
comparable in cost to ABO's own per-batch gradient-capture overhead
(ABO's verified refactor achieved 5.3GB peak / ~194s per epoch vs.
baseline's 4.4GB / ~140-160s). **Expect a similar order-of-magnitude
overhead category as ABO (roughly 1.2–2× epoch time, not 5–10×)**, but
this is an estimate, not a measurement — must be verified empirically
with a smoke test before any full training run, exactly as ABO's
refactor was verified before the active-mode launch.

## 6. Failure mode analysis

| # | Failure mode | Mechanism | Grounded in |
|---|---|---|---|
| 1 | **Gate degeneracy** | Only ~0.14% of voxels were incorrect in the diagnostic sample, but evidence values are extremely tightly clustered (10th/50th percentile only ~0.5 apart on a 0–33 scale). A poorly-calibrated $\hat U_i$ threshold could fire on almost nothing (if tuned to the true error rate) or on a large, mostly-irrelevant fraction of voxels (if tuned loosely) — very little margin between "correctly selective" and "degenerate." | Direct computation on `per_voxel_stats.csv` (this analysis) |
| 2 | **Tiny/absent tumors per volume** | A subject with very few tumor voxels contributes few/no tumor-class anchors to its own local density/centroid estimate; compactification could then pull that subject's sparse tumor voxels toward *other patients'* tumor manifold (via in-batch same-class neighbors from different subjects), injecting a subtly wrong cross-patient signal. | Direct consequence of §5.2's in-batch same-class neighbor requirement combined with tumor being only ~1% of voxels |
| 3 | **Over-aggressive compactification on genuinely heterogeneous tumors** | E1.3 found the tumor cluster is ~3× more diffuse than background *in the current, presumably reasonably-performing model* — some of that diffuseness may reflect genuine, clinically real tumor heterogeneity, not a representation defect. Excessive $\lambda_{comp}$ risks forcing legitimately different tumor appearances into an artificially tight cluster, which could hurt Dice on atypical tumors — the opposite of the intended calibration benefit. | Direct interpretation risk of E1.3's own finding |
| 4 | **Batch-composition-dependent density noise** | Since $\rho_i$ and $\mu_{c(i)}^{local}$ must be computed in-batch (§5.2), a batch that happens to sample few tumor voxels produces noisy, unstable estimates purely from batch luck — analogous to BatchNorm's small-batch instability, compounding failure mode #2. | Direct consequence of tractability constraints in §5 |
| 5 | **Noise-driven false sparsity** | If MRI acquisition noise independently inflates local feature-space variance in low-SNR image regions, those regions could appear artificially "sparse" in $z$-space regardless of true tumor-boundary proximity, causing the compactification force to fire on noise artifacts rather than genuine representation gaps. **Not yet checked against data** — testable by correlating extracted `feature_norm`/density against independently known noisy regions before committing further, but this has not been done. | Hypothesized risk, not yet empirically checked — flagged as an open question |
| 6 | **Boundary classifier staleness/instability** | If $B_i$'s underlying logistic classifier (§2.2) is refreshed infrequently (§5.2, mitigation 4) while the trunk representation is actively changing under $\mathcal{L}_{comp}$'s own pressure, $B_i$ could become a stale, self-inconsistent target — the classifier and the representation it measures could drift apart during training, a feedback dynamic not present in the static diagnostic setting where the classifier was fit once on a frozen, fully-trained model. | Consequence of applying a diagnostic method (fit *after* training) as a live training-time signal (needed *during* training) — genuinely new risk not present in E1.3 itself |

## 7. Ablation plan

| # | Configuration | Purpose |
|---|---|---|
| 0 | Baseline (frozen, Phase A.5) | Reference — no geometry terms |
| 1 | + $\mathcal{L}_{sep}$ only, **unweighted** (fixed $\lambda_{sep}$, $w_{sep}(i){\equiv}1$) | Tests the literature-standard margin loss alone — the "obvious" baseline flagged by `PHASE_E2_LITERATURE_REVIEW.md` as already well-precedented; must not be oversold as novel |
| 2 | + $\mathcal{L}_{comp}$ only, **unweighted** (fixed $\lambda_{comp}$, $w_{comp}(i){\equiv}1$) | Tests the density/compactness hypothesis alone, unweighted |
| 3 | + Both terms, **unweighted** (no gate, both fixed) | Tests whether the two-term *decomposition* itself matters, independent of adaptive weighting |
| 4 | + Both terms, **boundary/density-weighted, no uncertainty gate** ($g(\hat U_i) \equiv 1$) | Isolates the contribution of $B_i, D_i$ weighting from the uncertainty gate specifically |
| 5 | + **Full controller** ($g(\hat U_i)$-gated, $B_i/D_i$-weighted) | The proposed algorithm |

This structure directly isolates each design decision: rows 0→1/2/3
test whether *any* geometry loss helps at all; row 3→4 tests whether
*adaptive* (boundary/density) weighting beats a flat loss; row 4→5 tests
whether the *uncertainty gate* specifically adds value beyond
geometry-only weighting. Per §4.4's honesty table, rows 1–3 are expected
to replicate (not exceed) existing literature results — the headline
claim rests on row 5 outperforming rows 1–4, not on rows 1–3 alone
showing improvement over row 0.

Each configuration should report the full metric suite used in
Experiment D (Dice, IoU, Precision, Recall, F1, HD95, ECE) plus the
diagnostic quantities from E1.3/E1.4 (latent boundary AUC, mean
$|d_i|$/$\rho_i$ for correct vs. incorrect, Cohen's d) **re-measured
post-training** — to directly verify whether the algorithm actually
changed the geometry it was designed to change, the same verification
discipline used for ABO's effective-ratio tracking in Experiment D.

## 8. Open questions before implementation

1. $\tau_b$, $\tau_\rho$ in §2.2/2.3 are derived from the 30-volume
   diagnostic sample; must be re-measured on the full training set.
2. Failure mode #5 (noise-driven false sparsity) is unchecked — worth a
   cheap diagnostic pass before implementation, not just noted as a risk.
3. §5.2's boundary-classifier refresh frequency and §5.3's overhead
   estimate are both unverified; need a smoke test analogous to ABO's
   `verify_refactor.py` before any full run.
4. $g(\hat U_i)$'s exact form (identity vs. sharper gate) is left as the
   simplest starting choice per §3.1 — not yet justified by a sweep.
5. No decision yet on whether the boundary classifier (§2.2) should be
   refit per-batch, periodically, or trained jointly as part of the
   network (e.g., as a small linear head) rather than fit externally via
   scikit-learn as in the diagnostic scripts — this is an implementation
   detail with real consequences for tractability and gradient flow that
   this specification deliberately leaves open pending the "peer-review
   your own algorithm" pass.

## Files

| File | Purpose |
|---|---|
| `PHASE_E5_LOSS_MATH_LITERATURE.md` | The literature search this derivation is built on |
| `PHASE_E1_3_LATENT_GEOMETRY.md`, `PHASE_E1_4_DENSITY_ANALYSIS.md` | The empirical findings motivating every constant and mechanism above |
| `PHASE_E2_LITERATURE_REVIEW.md` | The novelty framing this design must stay consistent with |

---

**Drafted**: 2026-08-04 — not yet reviewed, not yet implemented
