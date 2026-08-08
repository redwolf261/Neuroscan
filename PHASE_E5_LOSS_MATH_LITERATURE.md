# Phase E5 (pre-derivation): Literature Search — Mathematical Forms for Geometry Loss Terms

**Status**: ✅ Complete — two precedents identified and combined

**Date**: 2026-08-04

## Purpose

Narrower, more technical follow-up to `PHASE_E2_LITERATURE_REVIEW.md`
(which confirmed the general "margin/prototype loss for segmentation"
strategy has precedent but no paper combines all the needed pieces).
This search specifically hunted for the **mathematical form** of loss
functions to build the two-term (separation + compactification),
uncertainty-gated geometry controller from, rather than novelty-checking
the overall idea again. Directly informs `PHASE_E5_ALGORITHM_DESIGN.md`'s
Deliverable 3 (loss derivation).

## Two decisive precedents found

### 1. De Brabandere, Neven, Van Gool — "Semantic Instance Segmentation with a Discriminative Loss Function" (CVPR 2017 workshop, [arXiv:1708.02551](https://arxiv.org/abs/1708.02551))

The closest structural match for "two separate, independently-weighted
corrective forces, not one merged scalar":

$$L_{var}=\frac{1}{C}\sum_{c=1}^{C}\frac{1}{N_c}\sum_{i=1}^{N_c}\big[\lVert\mu_c-x_i\rVert-\delta_v\big]_+^2$$

$$L_{dist}=\frac{1}{C(C-1)}\sum_{c_A\neq c_B}\big[2\delta_d-\lVert\mu_{c_A}-\mu_{c_B}\rVert\big]_+^2$$

$$L=\alpha L_{var}+\beta L_{dist}+\gamma L_{reg}$$

$L_{var}$ (pull toward own cluster center, hinged past margin $\delta_v$)
and $L_{dist}$ (push cluster centers apart, hinged past margin $\delta_d$)
are **literally additive with independent scalar weights** $\alpha,\beta$,
already per-pixel, and already hinge/margin-based — the hinge only
activates for points beyond $\delta_v$ or within $2\delta_d$, which is a
short step from "weighted more when near the boundary/sparse."

### 2. An et al. — "DyCON: Dynamic Uncertainty-aware Consistency and Contrastive Learning for Semi-supervised Medical Image Segmentation" (CVPR 2025, [arXiv:2504.04566](https://arxiv.org/pdf/2504.04566))

The closest match for "uncertainty-gating a metric-learning term by a
signal external to the embedding distance itself" — this is a **3D
medical segmentation paper**, essentially the same setting as this
project:

$$\mathcal{L}_{FeCL}=\frac{1}{|P(i)|}\sum_{k\in P(i)}\mathbf{F}_k^+\cdot\Big[-\log\Big(\frac{\exp(S_{ik})}{D(i)}\Big)\Big]$$

$$\mathbf{F}_k^+=(1-S_{ik})^\gamma\cdot\exp(H_{gs}(p_i^s)), \quad \mathbf{F}_q^-=(S_{iq})^\gamma$$

$$\mathcal{L}_{UnCL}=\frac{1}{N}\sum_i\frac{\mathcal{L}(p_i^s,p_i^t)}{\exp(\beta H_s(p_i^s))+\exp(\beta H_t(p_i^t))}+\frac{\beta}{N}\sum_i\big(H_s(p_i^s)+H_t(p_i^t)\big)$$

Voxel-wise entropy $H(p^s)$ from the segmentation prediction head — a
signal separate from the embedding similarity $S_{ik}$ — multiplicatively
gates the contrastive/consistency terms. Directly analogous to gating by
this project's evidential (Beta) head's uncertainty instead of softmax
entropy.

**No single paper combines both properties** (two-term decomposition AND
external-uncertainty gating) — the proposed design is a genuine,
defensible combination of these two precedents, not a reproduction of
either.

## Tractability findings (critical constraint, feeds Deliverable 4)

- **Full pairwise / graph Laplacian regularization is confirmed
  impractical** at this project's scale (tens of thousands of voxels per
  volume, batch size 8): $O(n^2)$ affinity matrix is intractable.
  Standard tractable approximation across all sources found: **sparsify
  to a local kNN graph** ($k=20$, already computed for the density
  measurement) — turns $O(n^2)$ into $O(nk)$.
- **Center-loss-style pull terms are cheap** — $O(n)$ per voxel against a
  running/mini-batch-local centroid, standard practice since the
  original center loss paper (Wen et al., ECCV 2016).
- **Pairwise triplet/contrastive terms are made tractable via in-batch
  sampling** — every precedent found (SupCon, DyCON, Multi-Similarity,
  N-pair, DGCL) restricts positive/negative sets to same-minibatch
  candidates, often further restricted via top-k hard-mining. DyCON's own
  solution should be mirrored directly.

## Recommended starting forms (full derivation with attribution in `PHASE_E5_ALGORITHM_DESIGN.md`)

Separation term skeleton borrowed from De Brabandere et al.'s $L_{dist}$,
extended to per-voxel pairs (mirroring DyCON's/Multi-Similarity's
in-batch negative-set structure) and boundary-weighted. Compactification
term skeleton borrowed from De Brabandere's $L_{var}$ / Wen et al.'s
center loss, but pulling toward a **local, density-weighted same-class
centroid** (the k=20 nearest same-class neighbors, not a single global
per-class centroid) rather than the classical global prototype — this
local-not-global centroid choice is directly motivated by this project's
own E1.3/E1.4 findings (tumor cluster diffuseness, density independent of
boundary distance) and is flagged in the search as a genuine gap in the
base literature (most center-loss-family methods use one global
prototype per class).

**Full derivation, with per-term attribution, is in
`PHASE_E5_ALGORITHM_DESIGN.md`.**

## Files

| File | Purpose |
|---|---|
| `PHASE_E5_ALGORITHM_DESIGN.md` | Full design specification using this literature as its mathematical basis |
| `PHASE_E2_LITERATURE_REVIEW.md` | The earlier, broader novelty-check this search follows up on |

---

**Completed**: 2026-08-04
