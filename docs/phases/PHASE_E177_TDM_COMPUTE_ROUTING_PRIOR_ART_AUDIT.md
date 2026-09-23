# E177 — Task-Demand-Weighted Compute Routing: prior-art audit

**Date**: 2026-09-19
**Status**: Targeted audit complete, real web search used (not memory/training-data recall).
**Scope**: audits ONLY the compute-ROUTING mechanism proposed this session —

$$
\mathcal{D}_i = \frac{\widehat R_i^*}{\widehat R_i^* + \lambda R_{i,\text{eff}} + \epsilon}, \qquad
B_i = B_{\min} + (B_{\max}-B_{\min})\,\text{clip}(\mathcal{D}_i^\gamma, 0, 1)
$$

used to allocate **inference-time refinement depth/compute** per region — this is a materially
different claim from E170's already-audited training-time loss penalty
($L_{\text{TDM}} = \lambda\sum[\hat R - R_{\text{eff}}]_+^2$) and from E171's already-audited
activation-steering mechanism. Neither prior audit covers this. Building on already-verified
infrastructure: the frozen $\hat R^*_{\text{enc1}}$ predictor (OOF $R^2=0.63$, ρ=0.79) and E170's
five-check premise verification (deficit exists, correlates with worse Dice, not just size, not
circular at enc1/enc2) — reused, not re-derived.

## What was searched

Six independent web queries (live search, session date 2026-09-19), targeting the specific
chain **task-required representational rank/complexity → inference-time per-region compute
allocation** — deliberately distinct from "adaptive rank in weight matrices" (E170b: LoRA/RaNA
family, different object) and "activation steering" (E171: CAST/GAPS family, different mechanism).

## Findings, closest-to-furthest

| Work | What it does | Distance from our claim |
|---|---|---|
| **RaNA / Adaptive Rank Allocation** (ICLR 2025, arXiv 2503.18216) | Router/masker allocates rank in **weight matrices** (linear layers, transformer MLPs/attention) per input, for inference **speed** at matched accuracy | Same shape (input-conditional rank allocation) but different object (weights, not activations) and different goal (speed, not accuracy) — confirms E170b's own "different object" finding, now for the routing use case too |
| **Learning How Hard to Think** (arXiv 2410.04707) | Predicts per-input reward-benefit of more decode-time compute (search/reranking), allocates compute accordingly | Same two-stage shape (predict benefit → allocate compute) but LLM decoding compute, reward-distribution signal — not rank, not segmentation |
| **Vision-MoR** (patch-level Mixture-of-Recursions) | Per-patch router selects recursion depth in a ViT | Closest **routing target** (per-region compute depth) found — but the routing signal is a learned router score, not an explicit rank/complexity estimate |
| **Med-DANet / Med-DANet V2** (ECCV 2022, WACV 2024) | Slice-wise decision network selects model-bank member by learned "segmentation difficulty," supervised by an accuracy/complexity choice metric | Same domain (medical volumetric segmentation), same purpose (efficient adaptive compute) — but routing signal is difficulty/accuracy-tradeoff, not rank, and not causally measured against the model's own undegraded output |
| Early-exit surveys (EENet, AEBNAS, Early-Exit GNN, Confidence-Gated Training) | Per-sample adaptive depth/exit | Signal is uniformly confidence/entropy-based, never rank/intrinsic-dimension-based |
| Intrinsic dimension / effective rank of activations (Ansuini et al. NeurIPS 2019, arXiv 2211.13239) | ID/effective rank as a **diagnostic** correlate of generalization, layer depth, training dynamics | Same underlying quantity (activation rank) but purely observational — never used as a live routing signal |
| Spectral ViT, BiFormer, regional attention | Rank/spectral structure used for **tokenization or attention sparsity**, not compute-depth routing | Different mechanism entirely |

## Verdict

$$
\boxed{\text{🟢 The specific chain — task-required representational rank, causally measured against
the model's own output, used as a per-region COMPUTE-ROUTING signal — was NOT located.}}
$$

Consistent with, and independently confirming, E170b's parallel finding for the training-penalty
version of this idea: the *ingredients* (rank-based allocation, difficulty-based routing,
per-region adaptive compute) are each separately occupied, but the specific combination — using
a **causally-measured, per-sample task-required rank** (not a learned difficulty score, not a
weight-matrix rank, not a confidence signal) as the **routing criterion** for inference-time
compute depth in a segmentation network — was not found across six targeted searches.

**Per E162's own structural law** (informativeness predicts prior-art density on a well-studied
problem), absence after six searches is suggestive, not proof — this audit should be treated as
the *current best evidence*, not a closed case. The single closest structural analogue
(Vision-MoR's per-patch depth routing) uses a fundamentally different signal (learned router,
not measured rank), which is exactly the kind of "one differentiated axis inside an occupied
mechanism family" pattern this project has previously found insufficient on its own (E161, E166,
E168 precedent, cited in E171) — worth remembering before over-claiming novelty from this result
alone.

## What this licenses, and what it does not

Licenses: proceeding to build and run the routing experiment, since the specific mechanism
clears a real (if necessarily imperfect) audit. Does NOT license claiming novelty before the
experiment runs — the audit clears the way to test the idea, it does not establish that the idea
works, and per this project's own standing rule (E170b), the experiment's own success criterion
must be **NeuroScan > Constant-target arm**, not merely **NeuroScan > Baseline** — beating
baseline alone would not distinguish this from adjacent occupied mechanisms even after this
audit.

## Sources

RaNA / Adaptive Rank Allocation (arXiv 2503.18216, ICLR 2025) · Learning How Hard to Think
(arXiv 2410.04707) · Vision-MoR (AAAI, patch-level Mixture-of-Recursions) · Med-DANet (arXiv
2206.06575, ECCV 2022) · Med-DANet V2 (arXiv 2310.18656, WACV 2024) · Confidence-Gated Training
for Early-Exit Networks (arXiv 2509.17885) · Early-Exit Graph Neural Networks (arXiv 2505.18088)
· EENet (WACV 2024) · Intrinsic Dimension of Data Representations in DNNs (Ansuini et al.,
NeurIPS 2019) · Relating Regularization and Generalization through Intrinsic Dimension (arXiv
2211.13239) · BiFormer (arXiv 2303.08810) · Spectral Vision Transformer (arXiv 2605.12026)
