# E30, Gate D: Prior-Art / Novelty Matrix

**Method**: WebSearch across the 12 required query terms (Section 16), each searched against 2025-2026 literature first. Every paper judged "close" was fetched directly (not judged from title/abstract alone, per the instruction) and probed with specific questions about its exact mathematical mechanism, matched against DTC's defining features: (1) a continuous, multi-level degradation family indexed by α, (2) per-component/per-object geometric ground-truth survival measured as a voxel-count ratio, (3) a model-predicted survival counterpart compared against that GT trajectory, (4) use as a training-time loss (not an architectural module or evaluation-time robustness test).

---

## Close-record table

| paper | year | venue | exact mathematical mechanism | training target | degradation operator | GT used? | component-level? | resolution trajectory modeled? | similarity to DTC | difference from DTC |
|---|---|---|---|---|---|---|---|---|---|---|
| Jin & Li, "An enhanced approach for few-shot segmentation via smooth downsampling mask and label smoothing loss" | 2024 | Image and Vision Computing | Smooth Downsampling Mask (SDM): cascaded downsampling with a smooth kernel applied ONCE to produce soft/smoothed labels at feature-map resolution; label smoothing loss trains against these soft labels instead of hard-thresholded downsampled masks | Standard few-shot segmentation loss (cross-entropy-style) against smoothed labels | Single fixed downsampling to feature-map resolution (no continuous α family) | Yes (mask) | No — global/per-pixel smoothing scheme, not per-component | No — single resolution, not a trajectory | Both concerned with information loss from mask downsampling | No continuous degradation family; no per-component survival curve; no predicted-vs-GT trajectory comparison; general natural-image few-shot segmentation, not medical/lesion-specific |
| "Soft labelling for semantic segmentation: Bringing coherence to label down-sampling" | 2023 | arXiv (2302.13961) | Soft-labeling framework for label down-sampling (full mechanism not confirmable beyond abstract — page fetch returned only metadata) | Semantic segmentation with soft-labeled downsampled targets | Single down-sampling step (per abstract framing) | Yes (label) | Unconfirmed from abstract | Unconfirmed, but no indication of a multi-level trajectory | Same general problem (label information loss under downsampling) | General semantic segmentation (not medical); no confirmed continuous α family or per-component survival measurement |
| Global and Regional Compensation Segmentation Framework (GRCSF) | Feb 2025 | arXiv (2502.08675) | Two components: Global Compensation Unit (GCU) and Regional Compensation Unit (RCU), both ARCHITECTURAL modules recovering pixel-wise information via masked-autoencoder-style reconstruction; residual map RM = 1 - cosine_similarity(original, reconstructed); loss = Dice+Focal (ATLAS) or Focal-only (orCaScore), MAE loss for the reconstruction branch | Lesion segmentation (ATLAS stroke, orCaScore coronary calcium) | Two FIXED MAE masking ratios (50%, 75%) — not a continuous α family | Yes (segmentation mask) | No — patch-based importance scoring, not per-connected-component | No — fixed ratios, not a resolution trajectory | Both address information loss from downsampling/masking in lesion segmentation | Architectural module (feature reconstruction), not a loss term; no per-component survival measurement; no comparison of GT geometric survival against model-predicted survival across a resolution family |

---

## Broadly-related but mathematically distinct (searched, no deep-dive needed — clearly out of scope on the search-result summary alone)

| Topic searched | Why excluded from the table above |
|---|---|
| "trajectory consistency loss" (general medical imaging, 2025) | Measures optical-flow/registration transformation consistency between transformed and original images — a spatial-transform consistency loss, not a resolution-degradation survival trajectory |
| Multi-Size Labeling (MSL) | Categorizes lesion voxels into size bins (tiny/small/etc.) to reweight loss by size class — a static size-based reweighting scheme, not a degradation-trajectory measurement |
| Knowledge distillation for small object detection (ScaleKD, SO-DETR, etc.) | Teacher-student feature distillation across fixed scales; no per-component geometric-survival-vs-predicted-survival comparison, no continuous degradation family |
| Task-aware/domain consistency learning (UniTask+, domain consistency for continual TTA) | Semi-supervised/domain-adaptation consistency regularization across domains or augmented views, unrelated to a geometric degradation family |
| "Resolution degradation curriculum" search | No paper found combining curriculum learning with a component-level survival-trajectory formulation |
| Tiny object tracking image degradation (bicubic downsampling simulation) | Uses downsampling to simulate tiny objects for tracking robustness evaluation, not a training-time loss comparing GT vs. predicted survival curves |

---

## Overall Gate D assessment

No paper found across the 12 required search terms, in 2025-2026 (searched first) or the immediately-adjacent 2023-2024 literature encountered along the way, implements: **(a)** a deterministic, continuous, monotonic multi-level degradation family, **(b)** per-connected-component geometric ground-truth voxel-survival measurement as an explicit ratio across that family, **and (c)** a model-predicted survival counterpart compared against the GT trajectory as a supervisory signal. The three closest records (SDM, GRCSF, soft-labeling) share DTC's general *concern* (information loss under downsampling degrades small-object/lesion segmentation) but differ on multiple defining mathematical features simultaneously — none use a continuous α-indexed family, none measure per-component survival as a tracked geometric quantity, and two of the three (SDM, GRCSF) are architectural or single-resolution mechanisms rather than a multi-level trajectory-based loss.

**Novelty classification: N2** — no close mathematical prior art was identified in the searched literature for the specific combination of mechanisms DTC proposes. This is not a claim that DTC is "first" (per the instruction not to use that word) — it reflects the outcome of this specific, documented search, not an exhaustive literature review, and a deeper targeted search (full-text search of MICCAI/TMI/MIDL proceedings specifically, rather than general web search) would be warranted before a stronger claim.
