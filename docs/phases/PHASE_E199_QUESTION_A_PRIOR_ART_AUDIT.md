# E199 — Question-A prior-art audit: the $O_i$ construction

**Date**: 2026-09-20
**Status**: AUDIT. Real web search, 4 queries + 1 targeted paper retrieval. No compute.
**Purpose**: The audit I asserted the answer to earlier without running, and was correctly
called on. This is the required check before any submission resting on $O_i$'s novelty.

---

## The claim under audit (narrowed, as specified)

$$
\boxed{\text{cross-fitted target-specific image observer} \rightarrow \text{pre-segmentation cross-model recoverability}}
$$

with the ET-works / TC-collapses asymmetry as the empirical signature. **Not** the broad claim I
made earlier ("every published method predicts quality after segmenting") — that was too strong
and is withdrawn.

## The nearest collision, examined directly

**Difficulty Estimation for Image-Specific Medical Image Segmentation Quality Control**
(MICCAI 2025, Springer LNCS 978-3-032-05169-1_12). This is the closest located work, and it is
close enough to matter.

| | MICCAI 2025 | $O_i$ (E143/E192/E193) |
|---|---|---|
| Inputs | ensemble regressor over **radiomics + uncertainty maps derived from AI segmentations** | 4 z-scored voxel intensities (+ local mean/std of $\delta$) — **no model output** |
| Model dependence | **Uses the segmentation model's uncertainty maps** | Model-independent by construction; observers never see a segmentation network |
| Target | mean **annotator** DSC (inter-observer agreement) | **model** ET error, $1-\text{Dice}$ |
| Validation | one pipeline | **6 independently trained models**, incl. a corrupted control |
| Estimator-independence test | not reported | 4 observers, pairwise $\rho$ 0.786–0.974 |
| Negative control | — | **pre-registered TC replication fails** (ΔR² 5–10× collapse, 3/6 negative) |

**Verdict on this collision: 🟡 adjacent, not identical.** It shares the goal (predict difficulty
for quality control) but uses model-derived uncertainty as input and predicts annotator
agreement, not model error. The pre-segmentation, model-independent property — the one that
licenses the frontier claim — is absent.

## Other neighbours checked

| Work | Why it does not collide |
|---|---|
| Confidence calibration / predictive uncertainty (arXiv 1911.13273) — AUROC ≥ .95 detecting failed segmentations | Derived **from the model's own confidence**. Cannot answer "is the model at the frontier?" without circularity |
| Failure-detection benchmarking (arXiv 2406.03323) | Confidence aggregation over model outputs — post-segmentation |
| EvanySeg, diffusion-based QC, coherence evaluators (2409.14874, 2511.09588) | Score a **produced mask**; require segmentation first |
| Diet-Seg: dynamic hardness-aware learning (bioRxiv 2025) | Entropy-based **local** hardness during training, model-internal |
| USE-Evaluator (MedIA 2023) | Metric design for uncertain/small/empty references — not a difficulty predictor |
| Radiomics difficulty/expert-disagreement modelling | Predicts **annotator** disagreement, uses engineered radiomic features |

## Verdict

$$
\boxed{\text{🟢 The specific construction is NOT located. Nearest neighbour is 🟡 adjacent (MICCAI 2025).}}
$$

The differentiating properties, in order of how load-bearing they are:

1. **Model-independent** — computed without any segmentation network. Every located method
   takes model output, features, or uncertainty as input. This is the property that makes the
   frontier claim non-circular, and no located work has it.
2. **Cross-model validated** — 6 models, ΔR² +0.451 to +0.563, all perm p = 0.0000, including
   an evidence-shuffled control. No located work validates a difficulty estimator across
   multiple independently trained models.
3. **Has a measured boundary** — the TC failure. Located works report where their method works;
   none reports a pre-registered region where it fails, with a mechanism.

## FULL TEXT READ (2026-09-20) — collision RESOLVED, verdict strengthened

The blocking prerequisite is discharged. Fournel, Bartoli, Marchi, Maurin, Arjomand Bigdeli,
Jacquier, Feragen (DTU Compute + APHM Hôpital de la Timone), MICCAI 2025 open-access version,
read in full (8 pages). Verbatim facts:

| dimension | Fournel et al. 2025 | $O_i$ |
|---|---|---|
| **Prediction target** | **"DSC Lesion"** = mean inter-/intra-**annotator** DSC across 3 radiologists. A property of *expert disagreement* | **model** ET error, $1-	ext{Dice}$ |
| **Inputs** | "Two U-Nets... pixel-wise average of their respective softmax outputs was used as final deep learning segmentation, from which radiomics were extracted, while their standard deviation gave **uncertainty maps**" | 4 z-scored intensities + local mean/std of $\delta$. **No network anywhere** |
| **Requires segmentation first?** | **Yes** — Fig. 4 input is "DL-based prediction + Uncertainty map" | **No** |
| **Uses GT mask for features?** | Yes — geometry group input is "Binary **GT** mask" | GT restricts voxels to `seg>0` and trains observers; blind to target subject via cross-fitting |
| **Modality / task** | 2D low-dose CT, COVID-19 lung lesions, 7,740 slices | 3D multiparametric MRI, BraTS ET |
| **Cross-model validation** | No — one two-U-Net ensemble | **6 independently trained models** incl. corrupted control |
| **Estimator-independence** | Not tested | 4 observers, pairwise $
ho$ 0.786–0.974 |
| **Reported failure region** | None. Limitation stated: *"we demonstrated the utility... for COVID-19 lesions, further research could explore the applicability of this paradigm to other medical imaging tasks"* | **Pre-registered TC replication fails** (ΔR² collapse 5–10×, 3/6 negative), with mechanism |

Their contributions, verbatim: (1) *"significant variations in segmentation difficulty within
the same task"*; (2) *"segmentation difficulty is strongly influenced by specific input
properties"*; (3) *"a novel method with a two-fold increase in precision for dynamically
predicting segmentation difficulty."*

**Assessment.** Contribution (2) is genuinely adjacent to E143's premise — both establish that
input properties govern difficulty. But their operationalisation is **post-segmentation and
model-dependent by construction** (radiomics extracted *from the lesion prediction*, plus
uncertainty maps from the U-Net ensemble's disagreement), and their target is **annotator**
agreement, not model error. They explicitly frame difficulty as aleatoric uncertainty: *"the
'segmentation difficulty' used... is related to aleatoric uncertainty."*

$O_i$ predicts something they do not measure (model error), from inputs they do not use (no
network), validated in a way they do not attempt (across models). **Verdict revised from 🟡
adjacent to 🟢 clearly distinct**, with Fournel et al. becoming the correct primary citation
for "input properties govern segmentation difficulty" — a related-work anchor, not a collision.

## Honest limitations of this audit

- **4 searches, ~30 min.** Not exhaustive. E162's law applies: informativeness predicts prior-art
  density, and this is an informative observation on a saturated benchmark.
- ~~The MICCAI 2025 paper was read via search-result summaries~~ **RESOLVED 2026-09-20** — full
  open-access text read (8 pp). The Candidate-B rule (*"always verify against actual PDF, not
  search snippets"*) was applied and materially changed the verdict: search summaries said
  "radiomics + uncertainty maps," which sounded like a near-match; the full text shows the
  radiomics are extracted *from the model's own lesion prediction* and the target is annotator
  DSC, not model error. The summary understated the distance.
- Radiomics-based difficulty prediction is a large literature; only its most-cited recent
  entries were checked.

## What may and may not be claimed

**MAY**: the construction was audited against 2024–2026 prior art and no direct match was
located; the nearest neighbour uses model-derived uncertainty and predicts annotator agreement
rather than model error.

**MAY NOT**: that it is novel *tout court*. A 4-query audit with one unread full text is
grounds for proceeding, not for a novelty claim in a paper's contributions section. The MICCAI
2025 full text must be read first.

## Consequence

Question A is **clear to proceed to drafting**, with the MICCAI 2025 full-text read as a
blocking prerequisite. The Dice-improvement question remains closed by E195/E196/E198 — this
audit does not reopen it and makes no claim about it.
