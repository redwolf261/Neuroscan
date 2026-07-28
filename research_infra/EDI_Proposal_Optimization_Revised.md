# Final EDI Research Proposal

## Working Title

**Optimization of Multi-Objective Loss Functions for Uncertainty-Aware Brain MRI Lesion Segmentation**

---

# Problem Statement

Recent deep learning frameworks for brain MRI lesion segmentation increasingly combine multiple training objectives, including segmentation loss, evidential uncertainty loss, consistency regularization, and pseudo-label supervision. These objectives are intended to improve segmentation accuracy, uncertainty estimation, and robustness. However, jointly optimizing several objectives introduces a complex optimization process in which multiple gradient signals simultaneously influence the shared backbone network.

Most existing methods focus on designing stronger individual objectives or improving supervision quality, while comparatively less attention has been given to understanding how these objectives interact during optimization.

**The NeuroScan framework used as the experimental testbed for this project implements exactly this situation**: a Hybrid segmentation loss (Dice + Focal Tversky) and an Evidential uncertainty loss are jointly backpropagated through a shared 2.5D encoder-decoder backbone, with two additional objectives — a consistency loss and a pseudo-label loss — implemented in code but disabled by default configuration flags (`USALD_CONSISTENCY_ENABLED = False`, `USALD_FDR_ENABLED = False`). This gives the project a live, two-objective interaction to characterize immediately, plus two dormant, fully-implemented objectives whose activation and interaction effects can be studied as a natural extension once the baseline two-objective case is understood.

---

# Research Gap

A review of recent literature (2024–2026) shows that research has largely concentrated on:

* designing improved segmentation losses,
* improving pseudo-label quality,
* uncertainty-aware supervision,
* calibration methods,
* consistency regularization.

Although modern segmentation models often optimize several objectives simultaneously, these objectives are typically combined using weighted summation with fixed or dynamically adjusted coefficients. Comparatively little work investigates the optimization dynamics among these objectives, including their cooperation, competition, and influence on the shared feature extractor during training.

As a result, the optimization process itself remains insufficiently characterized, potentially limiting convergence efficiency, stability, uncertainty estimation, and segmentation performance.

---

# Research Objective

The objective of this project is to investigate the optimization behavior of multiple loss objectives in an uncertainty-aware brain MRI lesion segmentation framework and to determine whether improved optimization strategies can enhance training effectiveness.

Specifically, the project aims to:

1. Analyze the optimization interactions among Hybrid Segmentation Loss and Evidential Loss (the two objectives active under the framework's default configuration), and characterize the Consistency Loss and Pseudo-label Loss as implemented-but-inactive components whose interaction effects can be activated and studied as a scoped extension.
2. Study how these objectives influence the shared backbone during training, at both the aggregate and per-component (encoder / decoder / attention / uncertainty-head) level.
3. Identify optimization inefficiencies, such as conflicting or imbalanced optimization signals, supported by direct gradient measurement rather than inference from loss curves alone.
4. Develop and evaluate an improved optimization strategy based on the observed optimization behavior.
5. Compare the proposed approach against the original NeuroScan optimization framework under matched, controlled experimental conditions.

---

# Preliminary Work Already Completed

Unlike a project starting from a literature review alone, Stages 2 and 3 below have already been substantially de-risked through direct instrumentation and experimentation on the actual NeuroScan codebase:

- **A full diagnostic framework has been built and empirically verified**: per-batch loss decomposition (Dice / Focal Tversky / Evidential, logged separately), gradient-norm logging (total and per-component: encoder, decoder, CBAM), per-loss gradient attribution via isolated backward passes, and gradient cosine-similarity measurement between loss pairs. The instrumentation was verified to be strictly additive — with all diagnostics enabled versus disabled, trained model parameters are bit-identical after a training step (max parameter difference: 0.0, across all 240 parameter tensors checked).
- **A concrete, quantified optimization imbalance has already been found**: the evidential/causal uncertainty heads (`anatomy_head`, `pathology_head`, `noise_head`) show 2–5x higher per-parameter gradient magnitude than the primary segmentation head, despite the evidential loss term being weighted at only λ=1×10⁻³ in the total loss. This was confirmed after controlling for the confound of differing parameter counts across layers.
- **A first candidate optimization fix has already been implemented and tested**: component-specific gradient scaling (damping the identified heads' gradients by a factor of 1/3, directly motivated by the measured imbalance ratio). Under a 25-epoch controlled comparison against baseline (identical architecture, data split, and optimizer, differing only in this one hook), the intervention did not produce a statistically significant change in segmentation quality (Dice p=0.89, IoU p=0.89, F1 p=0.37) — a genuine, informative null result, not a failure of the pipeline. The most likely explanation, itself a finding worth carrying into the semester project: the targeted parameter groups are too small a fraction of total network gradient norm for a narrow, isolated scaling intervention to move aggregate quality metrics, at least on the data scale and epoch budget tested.

This preliminary work is documented in full (code, logs, and figures) and directly informs the scoped stages below — the semester project is not starting Stage 2 from zero, but extending and correcting course from a working baseline.

---

# Proposed Methodology (Semester Scope)

The project will proceed in four stages:

### Stage 1 — Literature Review

* Review recent research (primarily 2024–2026) on:

  * segmentation loss functions,
  * evidential learning,
  * consistency learning,
  * pseudo-label learning,
  * multi-objective optimization,
  * gradient-conflict detection and resolution methods (e.g. PCGrad, GradNorm, CAGrad), specifically to determine which class of method is actually warranted by the imbalance already observed, rather than defaulting to a fashionable one.

---

### Stage 2 — Optimization Analysis *(baseline established; extension scoped below)*

Analyze the existing NeuroScan framework by studying:

* mathematical formulation of each loss (Dice, Focal Tversky, Evidential Beta/KL, and the dormant Consistency/Pseudo-label formulations),
* computational graph and exactly which parameters each loss term's gradient reaches (already mapped: the evidential/causal heads are largely — but not entirely — a separate parameter path from the segmentation head, branching from a shared upstream decoder trunk),
* dependency between objectives,
* optimization flow through the shared backbone, including the two learning-rate groups already in use (encoder+CBAM at 1×10⁻⁵, decoder at 4×10⁻⁴) and what that implies for how quickly each component can respond to its assigned gradient signal.

**Scoped extension**: repeat this analysis with the Consistency and Pseudo-label objectives activated (`USALD_CONSISTENCY_ENABLED = True`, `USALD_FDR_ENABLED = True`), to characterize the full four-objective interaction the original literature gap describes, building directly on the two-objective baseline already measured.

---

### Stage 3 — Optimization Diagnostics *(instrumentation built; extended runs scoped below)*

Experimentally investigate, using the diagnostic framework already implemented and verified:

* gradient magnitudes (per-loss and per-component; baseline imbalance already quantified),
* optimization interactions — specifically, gradient cosine similarity between loss pairs across training (preliminary measurement: Dice and Focal Tversky gradients are consistently positively aligned across all sampled epochs in a 29-epoch pilot run; Evidential's gradient is consistently near-orthogonal to both, not conflicting — this needs replication across more epochs and, importantly, in a regime where the model actually converges, since the pilot run's segmentation loss stayed nearly flat throughout, limiting how much can be concluded about optimization behavior *during* genuine learning as opposed to during stalled training),
* objective influence on different network components (already broken down by encoder / CBAM / decoder-probability-head / decoder-evidential-heads),
* temporal behavior during training (early/mid/late phase comparison already run once; showed no clear phase transition in the pilot, but the pilot's limited convergence — discussed under Risks — means this should be re-examined once training is demonstrably learning the task).

The goal is to identify optimization inefficiencies supported by empirical evidence rather than assumptions — the preliminary work above already demonstrates this is achievable with the current instrumentation; the semester extends it to a training regime with enough real convergence to draw stronger conclusions (see Risks and Mitigations).

---

### Stage 4 — Optimization Improvement

Based on the findings, design and evaluate an optimization strategy that improves training while preserving accurate segmentation and reliable uncertainty estimation.

The preliminary gradient-scaling experiment (Stage 3) narrows this stage's starting point usefully: a narrow, single-parameter-group scaling intervention was tested and did not show effect at small scale, which argues against pursuing a similarly narrow fix and toward evaluating either (a) a broader dynamic loss-weighting scheme informed by the full gradient-similarity picture from Stage 3, or (b) confirming first whether the null result was a scale/convergence artifact by re-testing the same intervention once a stronger training baseline is established (see Risks).

---

# Risks and Mitigations

**Risk: the available training data (9 patients, ~28 volumes) is substantially smaller than the dataset used to report this framework's published headline performance (45 patients), and the model did not achieve meaningful segmentation convergence in preliminary runs at this data scale (Dice remained under 1% across all preliminary experiments).** This directly limits how confidently optimization-dynamics conclusions can be drawn, since "does this optimization change help" is hard to answer cleanly when the baseline itself is not learning the task well.

**Mitigation, already partially validated**: a preliminary experiment restructuring how training examples are supervised — sampling training targets from across all lesion-containing slices in each volume (measured: ~22 per volume on average, ~613 total across the available data) rather than a single fixed slice per volume — produced the first stable, non-noise improvement observed across all preliminary experiments (a ~5x increase in Dice measured on a representative sample of slice positions, stable from partway through training rather than a single volatile spike). This is being proposed as a data-pipeline prerequisite for Stage 3/4's extended runs, not as this project's primary contribution — it exists to give the optimization-dynamics analysis a training regime that is actually learning, so that conclusions about gradient interaction are measuring real optimization behavior rather than noise around a stalled loss.

---

# Expected Contributions

This work is expected to provide:

* A systematic analysis of optimization dynamics in uncertainty-aware brain MRI lesion segmentation, grounded in direct gradient measurement rather than loss-curve inference alone.
* An experimental characterization of interactions among multiple optimization objectives, including a quantified account of which components each objective actually influences.
* An improved optimization strategy motivated by observed optimization behavior, including a documented account of at least one candidate strategy already tested and found ineffective at small scale — itself a contribution, since negative results in this space are rarely reported.
* A comprehensive comparison with the baseline NeuroScan framework under matched, controlled conditions.

---

# Expected Deliverables

By the end of the semester:

* Comprehensive literature review.
* Mathematical analysis of the NeuroScan optimization pipeline (encoder/decoder parameter paths per loss term, already substantially mapped).
* Optimization diagnostic framework (already implemented; extended to cover activated Consistency/Pseudo-label objectives and larger-scale training runs).
* Proposed optimization strategy, informed by which candidates preliminary evidence has already ruled out.
* Experimental evaluation against the baseline, with statistical comparison methodology already established (paired testing across matched training runs).
* EDI report and demonstration.

---

# Novelty Statement

Rather than proposing another segmentation loss function, this work investigates **how multiple existing objectives interact during optimization** within an uncertainty-aware segmentation framework. The central hypothesis is that understanding and improving the optimization process itself may yield better convergence, stability, uncertainty estimation, and segmentation performance than modifying individual loss functions in isolation.

Preliminary work strengthens this novelty claim in one specific way: the project has already demonstrated that a plausible, literature-motivated optimization fix (component-specific gradient scaling) does *not* trivially work, at least not in isolation — meaning the semester's contribution is not "confirm an obvious fix helps" but "characterize why the obvious fix is insufficient and identify what actually would need to change."

---

## Confidence Assessment

I would classify this proposal as:

* **Alignment with your assigned component (loss functions/optimization):** ★★★★★
* **Compatibility with the existing NeuroScan codebase:** ★★★★★
* **Suitable scope for an EDI semester project:** ★★★★★
* **Potential to evolve into a publishable contribution:** ★★★★☆ (strengthened by the preliminary gradient-imbalance finding and the documented null result on the first candidate fix; still contingent on Stage 3's extended runs revealing a genuine, addressable optimization inefficiency once the convergence-regime risk above is mitigated)

One final recommendation: present this to your professor explicitly as an **investigation into multi-objective optimization**, with the preliminary gradient-imbalance finding and the tested-but-ineffective first candidate as evidence the investigation is already underway and finding real things — rather than a commitment to a specific fix. That gives you the flexibility to let the experimental evidence guide the final method, which is both scientifically stronger and less risky for the project, while demonstrating to your professor that this is not a from-scratch proposal.
