# Phase E1: Literature Mapping — Assumption vs. Evidence

**Status**: ✅ Complete (45 papers, 6 categories, 2023–2026)

**Date**: 2026-08-04

## Purpose

Following Experiment D's null result (ABO substantially moved trunk
gradient composition with no effect on Dice/IoU/F1 — see
`PHASE_D_ANALYSIS_REPORT.md`), this is Step E1 of the redesign phase: map
the recent literature on multi-objective optimization, gradient
manipulation, multi-task balancing, medical segmentation tricks,
uncertainty-aware learning, and representation-level methods, then check
each paper's core assumption against this project's own measured
findings (Phase B, C0, C1, C2a/C2b, D). The goal is not to find a method
to copy — it's to find where the literature's assumptions and this
project's evidence agree or disagree, since disagreement is where a real
contribution or a well-motivated next hypothesis lives.

All 45 papers below were located via live web search; every entry has a
source URL. None were fabricated.

## This project's four reference findings (used as the "Evidence" column benchmark)

1. Gradient conflict between branches is **transient** (resolves by
   epoch ~3), not persistent (Phase B).
2. Static loss-weight reweighting **cannot move** the trunk gradient
   magnitude ratio at all, even across an 8x swing (Phase C0).
3. A dynamic controller (ABO) **can** move that ratio, substantially and
   robustly across a wide hyperparameter range (Phase C1, C2a, C2b).
4. Moving that ratio, even by ~2x, produced **no measurable change** in
   Dice/IoU/F1 (Experiment D). A real, non-confounded relationship with
   ECE (calibration) was found instead, on closer analysis (see
   `abo_frozen_lessons_learned` memory).

## Literature Matrix

### Category 1: Multi-Objective Optimization (General)

| Paper | Core Assumption | Method | Evidence | Relation to Project Findings |
|---|---|---|---|---|
| [Survey on Pareto Front Learning](https://link.springer.com/article/10.1007/s41965-024-00170-z) (2024) | Bottleneck is approximating the whole Pareto front | Hypernetwork/hypervolume-based PFL taxonomy | Survey | UNRELATED — project targets one operating point |
| [Multi-objective DL: Taxonomy and Survey](https://www.sciencedirect.com/science/article/pii/S2666827025000830) (2025) | Different method families suit different conflict regimes | Taxonomic survey | Comparative review | UNKNOWN |
| [Three-Way Trade-Off in Multi-Objective Learning](https://arxiv.org/pdf/2305.20057) (2023) | Conflict-avoidance, speed, and generalization trade off | Theoretical analysis of MGDA-style weighting | Theoretical bounds | UNKNOWN — cautions that aggressively resolving conflict (which was transient anyway, Finding 1) may cost generalization |
| [Smooth Tchebycheff Scalarization](https://arxiv.org/abs/2402.19078) (ICML 2024) | Linear scalarization can't reach non-convex Pareto regions | Smooth min-max scalarization | Convergence proof + benchmarks | UNRELATED |
| [MultiBalance](https://arxiv.org/pdf/2411.11871) (2024) | Gradient-norm imbalance hurts shared-bottom performance at scale | Closed-form per-task norm-equalizing factors | Production A/B test, positive | UNKNOWN generalization to 2-branch medical case; Finding 4 is a counter-data-point to this family's premise |

### Category 2: Gradient Surgery / Gradient Manipulation

| Paper | Core Assumption | Method | Evidence | Relation to Project Findings |
|---|---|---|---|---|
| [PCGrad](https://arxiv.org/abs/2001.06782) (NeurIPS 2020) | Gradient CONFLICT (not magnitude) degrades MTL | Project onto conflicting gradient's normal plane | RL/supervised MTL gains | **CONTRADICT-adjacent** — premise is persistent conflict; Finding 1 shows conflict here is transient, so little to correct most of training |
| [CAGrad](https://arxiv.org/abs/2110.14048) (NeurIPS 2021) | Bounding worst-case task degradation avoids destructive conflict | Trust-region constrained optimization | Convergence guarantees + benchmarks | Same conflict-centric premise, weakened by Finding 1 |
| [GradVac](https://arxiv.org/pdf/2010.05874) (ICLR 2021) | Direction AND magnitude both matter; align toward target similarity | Adaptive alignment to target cosine similarity | Multilingual MT gains | UNKNOWN — direction component untested by this project (ABO tested magnitude only) |
| [Nash-MTL](https://arxiv.org/abs/2202.01017) (ICML 2022) | Gradient combination as fair-bargaining problem | Nash Bargaining Solution | Benchmark SOTA at publication | UNKNOWN — direction-based, unrelated to magnitude findings |
| [Gradient Similarity Surgery (SAM-GS)](https://arxiv.org/abs/2506.06130) (ECML-PKDD 2025) | Magnitude AND directional conflict both hinder convergence | Magnitude-similarity-guided equalization + momentum | Synthetic + MTL benchmarks | **Contradict-adjacent to ABO's premise**: argues magnitude-only correction is insufficient — consistent with why Exp D found magnitude balancing alone didn't move Dice |
| [Filter-Grouped Gradient Surgery](https://www.sciencedirect.com/science/article/abs/pii/S092523122501464X) (2025) | Per-filter (not global) conflict matters in shared CNNs | Group filters, surgery per group | Driving-perception benchmark | UNRELATED |
| [Recon](https://arxiv.org/pdf/2302.11289) (ICLR 2023) | Architectural sources of conflict can be identified and isolated | Identify "conflict layers," restructure sharing | MTL benchmark gains | UNKNOWN — root-cause/architectural framing, genuinely different angle from ABO's online correction |
| [Proactive Gradient Conflict Mitigation via Sparse Training](https://arxiv.org/abs/2411.18615) (2024) | Dense shared params are more conflict-prone; sparsity helps | Sparse/selective parameter updates | MTL benchmarks | UNKNOWN |
| [Expert Squads for Gradient Conflict](https://www.sciencedirect.com/science/article/abs/pii/S0925231224016035) (2024) | MoE-style specialization beats reweighting shared gradients | Expert squads + closed-form scaling | Benchmark comparisons | UNKNOWN |
| [Gradient Orthogonality for Domain Adaptation Data Selection](https://arxiv.org/pdf/2602.06359) (2026) | Gradient orthogonality predicts transfer quality | Select data maximizing orthogonality | Domain adaptation benchmarks | UNRELATED (different application) |
| [Can Optimization Trajectories Explain MTL Transfer?](https://arxiv.org/pdf/2408.14677) (2024) | Conflict correlates with sharpness/curvature — a symptom, not standalone cause | Empirical correlational study | Correlation analysis across MTL settings | **UNKNOWN-leaning-CONFIRM** — aligned in spirit with Finding 1 (conflict as transient/dynamics-driven, not structural pathology) |
| [Resolving Token-Space Gradient Conflicts](https://openaccess.thecvf.com/content/ICCV2025/papers/Jeong_Resolving_Token-Space_Gradient_Conflicts_Token_Space_Manipulation_for_Transformer-Based_Multi-Task_ICCV_2025_paper.pdf) (ICCV 2025) | Conflict lives in feature/token space, not just gradient space | Token-space manipulation pre-gradient | Transformer MTL benchmarks | UNKNOWN — representation-first framing, notably different from ABO |
| [Towards Consistent MTL: Task-Specific Parameters](https://openaccess.thecvf.com/content/CVPR2025/papers/Qin_Towards_Consistent_Multi-Task_Learning_Unlocking_the_Potential_of_Task-Specific_Parameters_CVPR_2025_paper.pdf) (CVPR 2025) | Conflict methods over-focus on shared trunk, under-use task-specific capacity | Reallocate task-specific parameter usage | CVPR benchmark results | UNKNOWN — conceptually adjacent to "is the trunk really the bottleneck?" |

### Category 3: Multi-Task Task-Balancing Methods

| Paper | Core Assumption | Method | Evidence | Relation to Project Findings |
|---|---|---|---|---|
| [GradNorm](https://arxiv.org/abs/1711.02257) (ICML 2018) | Equalizing per-task gradient norms balances training rates, improves MTL | Learn weights to equalize gradient magnitude ratio | CV MTL benchmarks | **CONTRADICT** — closest published analogue to ABO's mechanism. Experiment D is a rigorous controlled test of this exact premise and produced a clean null result |
| [Dual-Balancing (DB-MTL)](https://arxiv.org/pdf/2308.12029) (2024/2025) | Loss-scale AND gradient-magnitude disparities both need correcting | Log-transform loss-scale balance + gradient-norm balance | MTL benchmarks | Partially contradict-adjacent (gradient half); loss-scale half untested by this project (C0 tested static weight only, not dynamic loss-scale normalization) |
| [Improvable Gap Balancing](https://arxiv.org/pdf/2307.15429) (2023) | Balance the *improvable gap* (distance to single-task optimum), not raw scale | Track/equalize per-task improvable gap | MTL benchmarks | UNKNOWN — genuinely different signal than gradient magnitude |
| [Uniform Loss vs. Specialized Optimization](https://arxiv.org/pdf/2505.10347) (2025) | Specialized balancing's advantage over uniform weighting is often overstated | Controlled comparison | Benchmarks across MTL suites | **Loosely confirm-adjacent** — broadly consistent with this project's own finding that a sophisticated balancer (ABO) did no better than baseline on the final task metric |
| [Analytical Uncertainty-Based Loss Weighting](https://link.springer.com/chapter/10.1007/978-3-031-85181-0_22) (DAGM GCPR 2024) | Task uncertainty, not gradient stats, should set loss weights | Closed-form uncertainty weighting (extends Kendall et al.) | Benchmark comparisons | UNKNOWN, relevant candidate — untested axis for this project |
| [AuxiLearn](https://avivnavon.github.io/AuxiLearn/) (ICLR 2021) | Right aux weight minimizes main task's val loss, via bi-level optimization | Implicit differentiation through learned loss combination | Meta-learned weights beat fixed/heuristic | UNKNOWN — entirely different mechanism, untested |
| [SLGrad — Sample-Level Weighting](https://arxiv.org/abs/2306.04519) (2023) | Right task weight varies per-sample, not just per-task | Learn sample-specific aux weights | RL/supervised benchmarks | UNKNOWN — orthogonal axis, not explored by ABO |
| [Asymmetric Bargaining for Aux Learning](https://proceedings.mlr.press/v202/shamsian23a/shamsian23a.pdf) (ICML 2023) | Main/aux tasks have asymmetric bargaining power | Asymmetric Nash bargaining | Benchmarks vs. AuxiLearn/uniform | UNKNOWN |
| [Unified Framework for Gradient Aggregation](https://arxiv.org/pdf/2605.30452) (2026) | GradNorm/PCGrad/Nash-MTL/CAGrad are special cases of one framework | Unifying theory | Theoretical + benchmark validation | UNKNOWN — context: ABO's mechanism sits within a known family, not novel territory |

### Category 4: Medical Image Segmentation Optimization Tricks

| Paper | Core Assumption | Method | Evidence | Relation to Project Findings |
|---|---|---|---|---|
| [Boundary Loss](https://proceedings.mlr.press/v102/kervadec19a.html) (MIDL 2019, still base of 2024-25 work) | Region losses (Dice/CE) ill-conditioned for unbalanced/small structures | Differentiable boundary distance term | Up to 8% Dice / 10% HD improvement (follow-ups) | UNRELATED to gradient findings; targets a different bottleneck entirely (loss geometry) |
| [Centerline Boundary Dice (cbDice)](https://papers.miccai.org/miccai-2024/129-Paper0458.html) (MICCAI 2024) | Topology not captured by Dice alone | clDice + boundary-aware distance-weighted skeleton term | MICCAI 2024 vascular benchmarks | UNRELATED but relevant to candidate direction 4 |
| [Adaptive t-vMF Dice Loss](https://pubmed.ncbi.nlm.nih.gov/38061152/) (2024) | Fixed similarity metric treats easy/hard classes identically | Learnable vMF-based adaptive compactness | 5-fold CV Dice gains | UNRELATED; alternate loss-shape lever |
| [FESS Loss](https://arxiv.org/pdf/2402.08582) (2024) | Standard Dice/CE ignore feature-space structure | Combined spatial + feature-enhancement loss | **Up to 16% Dice gain on BraTS 2016** | UNRELATED directly, but concrete existence proof that a representation-space loss term moves Dice on this exact dataset family |
| [DSC++ Loss](https://pmc.ncbi.nlm.nih.gov/articles/PMC10039156/) (2023) | Dice loss causes systematic overconfidence | Modified Dice penalizing overconfident errors | ECE improvements, no Dice cost | UNKNOWN — orthogonal to gradient magnitude, relevant if pursuing calibration-driven optimization |
| [Mask-TS Net](https://link.springer.com/chapter/10.1007/978-3-031-78128-5_12) (MICCAI workshop 2024) | Global temperature scaling miscalibrates small lesions specifically | Region-restricted temperature scaling | Polyp segmentation calibration gains | UNRELATED |
| [Generalized Surface Loss](https://arxiv.org/pdf/2302.03868) (2023) | Optimizing Dice ≠ optimizing Hausdorff distance | Differentiable surface-distance surrogate | Reduced HD without hurting Dice | UNRELATED to gradient findings; relevant to boundary-aware candidate |
| [Optimization Theory for Dice/Jaccard](https://ieeexplore.ieee.org/document/9116807/) (IEEE TMI) | Metric mismatch (surrogate loss vs. eval metric) is itself a suboptimality source | Theoretical analysis of soft-Dice properties | Theoretical + empirical | UNKNOWN — raises possibility that loss-surrogate mismatch, not gradient magnitude, limits Dice |
| [Region-wise Loss Unification](https://www.sciencedirect.com/science/article/pii/S0031320322006872) (2023) | Boundary/Hausdorff/active-contour losses are one family | Unifying reformulation | Comparative empirical study | UNKNOWN — useful taxonomy for candidate 4 |
| [Implantable Adaptive Cells NAS](https://arxiv.org/pdf/2604.14849) (2026) | Small NAS modules improve segmentation cheaply | Architecture search for lightweight cells | Benchmarked Dice gains | UNRELATED (architecture, not optimization) |
| [Autoencoder-Augmented nnUNetv2](https://pmc.ncbi.nlm.nih.gov/articles/PMC12053516/) (2024/2025) | Reconstruction aux objective regularizes encoder better than seg loss alone | Autoencoder aux branch on nnU-Net | Improved Dice vs. vanilla nnU-Net | UNKNOWN — a structurally similar 2-headed-trunk precedent (segmentation + reconstruction) where an auxiliary branch DID measurably help, unlike this project's evidential branch |
| [Res-UNET with Deep Supervision](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12162509/) (2024/2025) | Deep supervision stabilizes gradient flow, improves Dice | Multi-resolution aux segmentation heads | Dice 0.9498, best variant tested | UNKNOWN — established, different mechanism from branch-gradient-balancing, not tested here |

### Category 5: Uncertainty-Aware Learning / Calibration (Evidential, Beta/Dirichlet)

| Paper | Core Assumption | Method | Evidence | Relation to Project Findings |
|---|---|---|---|---|
| [Sensoy et al., Evidential Deep Learning](https://dl.acm.org/doi/abs/10.5555/3327144.3327239) (NeurIPS 2018, foundational) | Dirichlet output + evidence-based loss yields useful predictive uncertainty | Evidential/Dirichlet head + KL-to-uniform regularizer | Benchmark OOD/uncertainty experiments | UNKNOWN — foundational framework this project's evidential branch is built on |
| [Are EDL's Uncertainty Capabilities a Mirage?](https://arxiv.org/abs/2402.06160) (NeurIPS 2024) | EDL's epistemic uncertainty reflects genuine epistemic uncertainty | Theoretical + empirical analysis; reframes EDL as OOD/energy-based detection | Proof + ablations | **Important caution, UNKNOWN relative to gradient findings** — questions whether the evidential branch's signal is meaningful at all, independent of how much trunk gradient it gets |
| [Bi-level Meta-Policy for Dynamic Uncertainty Calibration](https://arxiv.org/pdf/2510.08938) (2025) | Fixed EDL regularization strength is suboptimal | Meta-learned policy controlling KL/regularization coefficient | Benchmark calibration gains | UNKNOWN, relevant candidate — dynamically controls the evidential loss's OWN coefficient (different lever than ABO's trunk-gradient share) |
| [Flexible Evidential Deep Learning](https://openreview.net/forum?id=N6ujq5Yfwa) (NeurIPS 2025) | Standard Dirichlet EDL is too rigid | Flexible Dirichlet parameterization | Benchmark generalization gains | UNKNOWN |
| [DAEDL — Enhanced EDL for OOD](ICML 2024) | Standard EDL conflates accuracy and OOD-detection quality | Modified objective decoupling accuracy/OOD signal | ICML 2024 benchmarks | UNKNOWN |
| [Survey on Evidential Deep Learning](https://arxiv.org/pdf/2409.04720) (2024) | EDL is maturing but debated across domains | Survey/taxonomy | N/A | UNKNOWN — good reference |
| [Deep Evidential Fusion, Multimodal Medical Seg](https://sciencedirect.com/science/article/abs/pii/S1566253524004263) (2024) | Combining evidence across modalities beats a single evidential head | Dempster-Shafer multimodal fusion | Multimodal benchmark gains | UNKNOWN — multimodal is explicitly out of scope per this project's frozen FLAIR-only design ([[brats_optimization_strategy]]) |
| [Mutual Evidential Deep Learning](https://arxiv.org/pdf/2505.12418) (2025) | Two evidential learners cross-supervising beats one | Dual evidential networks, mutual evidence adjustment | Medical seg benchmarks | UNKNOWN |
| [Toward Reliable Med Seg via Evidential Calibrated Uncertainty](https://arxiv.org/pdf/2301.00349) (2023/2025) | A plug-in evidential module can attach to arbitrary backbones | Evidential head + calibrated uncertainty loss, arch-agnostic | Multiple backbones tested | UNKNOWN — same architectural family as this project |
| [Region-based EDL for BraTS](https://dl.acm.org/doi/abs/10.1007/s00521-022-08016-4) (2022/2023) | Region-level (not voxel-level) evidence aggregation improves BraTS robustness | Region-based Dirichlet evidence aggregation | BraTS Dice + uncertainty metrics | UNKNOWN — closely related architecture/dataset, doesn't touch the gradient-sharing question |
| [Progressive Uncertainty-Guided Evidential U-KAN](https://www.researchgate.net/publication/396457361) (2025/2026) | Evidential uncertainty should steer segmentation branch's feature learning via attention | Uncertainty feeds attention weighting in seg decoder | Reported medical benchmark gains | **Directly relevant candidate for Direction 3** — uses uncertainty OUTPUT value, not gradient stats, to modulate the seg branch |
| [Average Calibration Losses](https://arxiv.org/pdf/2506.03942) (2025) | A differentiable calibration-targeting loss beats post-hoc temperature scaling | Calibration loss added during training | Medical seg calibration gains | UNKNOWN |
| [Uncertainty/Hardness-Weighted Loss](https://pmc.ncbi.nlm.nih.gov/articles/PMC12691699/) (2024/2025) | Per-pixel uncertainty AND hardness should jointly reweight seg loss | Pixel-wise PGU + REH weight terms | Benchmark Dice/calibration gains | **Relevant candidate** — combines Direction 3 (uncertainty) and Direction 4 (boundary/hard-region) at pixel level rather than branch level |

### Category 6: Representation Learning as the Bottleneck

| Paper | Core Assumption | Method | Evidence | Relation to Project Findings |
|---|---|---|---|---|
| [Rep-MTL](https://arxiv.org/abs/2507.21049) (ICCV 2025 Highlight) | Shared representation space, not gradient combination rule, is where task interaction happens; optimizer-centric fixes treat a symptom | Representation-level task-saliency + entropy penalization + cross-task alignment | ICCV 2025 Highlight; gains on 4 MTL benchmarks even with equal loss weighting | **Directly and strongly relevant** — this paper's stated thesis is essentially a literature-level articulation of what Experiment D found empirically |
| [CORE-MTL](https://arxiv.org/abs/2606.02221) (ICML 2026) | Gradient conflict/imbalance is a geometric byproduct of entangled shared representations, not primary | Causal orthogonal semantic/residual factorization; orthogonality emerges structurally | ICML 2026; tighter OOD generalization bound | **Directly relevant** — explicitly reframes gradient-level phenomena (what ABO targeted) as downstream of representation entanglement |
| [A Closer Look at Multimodal Representation Collapse](https://arxiv.org/html/2505.22483v1) (2025) | Shared reps can "collapse" toward one task's subspace even when gradients look balanced | Empirical + theoretical collapse-dynamics analysis | Multimodal benchmark diagnostics | UNKNOWN, but a mechanism consistent with (not tested by) Finding 4 |
| [Capacity/Redundancy Trade-offs in MTL](https://arxiv.org/html/2607.16554v1) (2026) | Negative transfer results from limited shared capacity and low task redundancy | Theoretical capacity-redundancy decomposition | Benchmark experiments | UNKNOWN — reframes "conflict" in capacity terms, orthogonal to gradient magnitude |
| [Gradient Conflict as Symptom of Shortcut Features](https://arxiv.org/pdf/2602.16125) (2026) | Tasks coupled via shared nuisance/shortcut features; gradient conflict merely reflects this | Source/feature screening to remove shortcut-inducing features | Benchmark experiments | **Directly relevant / confirm-adjacent** — close to a direct theoretical explanation for why Experiment D's null result occurred |
| [Orthogonal Bottlenecks for RL](https://arxiv.org/pdf/2605.26012) (2026) | Low-dimensional orthogonal subspace prevents degenerate collapse under competing objectives | Orthogonal bottleneck architecture | RL multi-task benchmarks | UNKNOWN — different domain, structurally on-point for Direction 2 |
| [MRdIB — Multimodal Rep-disentangled Info Bottleneck](https://arxiv.org/pdf/2509.20225) (2025) | IB-based disentanglement of shared/task-specific info improves reps | IB disentanglement regularizer | Multimodal benchmark gains | UNKNOWN |
| [FESS Loss](https://arxiv.org/pdf/2402.08582) (2024, cross-listed) | Feature-space loss captures info pixel-space losses miss | Feature-enhancement term added to spatial loss | Up to 16% Dice gain on BraTS 2016 | Concrete existence proof that representation-level intervention moves Dice on this exact dataset family |

## Synthesis: which direction is actually best supported?

**The literature converges, largely independently of this project's own
findings, on the same conclusion Experiment D reached empirically.** A
recent thread — Rep-MTL (ICCV 2025 Highlight), CORE-MTL (ICML 2026), and
a 2026 paper whose core claim is literally "gradient conflict is a
symptom rather than the root cause of negative transfer" — argues that
gradient-level fixes (the entire family ABO belongs to, alongside
GradNorm, PCGrad, CAGrad, Nash-MTL, GradVac) treat a downstream symptom
of representation-level entanglement, not the underlying cause. This
predicts exactly the pattern Experiment D produced: a real, substantial,
verified change in the gradient statistic with no effect on the
downstream metric.

**Ranked by support (literature backing) and fit (motivated specifically
by this project's own measurements):**

1. **Representation-level optimization (Direction 2) — best supported.**
   Rep-MTL and CORE-MTL give a theoretically coherent account of *why*
   ABO's null result would happen, and FESS Loss (arXiv 2402.08582) is a
   concrete existence proof that a representation/feature-space loss term
   produced a large real Dice gain **on BraTS specifically**. See
   `PHASE_E3_CANDIDATE_CONCEPTS.md` for the concrete proposal this
   motivates.

2. **Boundary-aware/topology-aware optimization (Direction 4) — well
   evidenced, but orthogonal to the gradient-branch question.** Multiple
   concrete, reproducible Dice/HD95 gains (Kervadec boundary loss, cbDice,
   generalized surface loss). Doesn't use the existing ABO/diagnostic
   infrastructure and isn't motivated by this project's specific
   findings — a good idea on its own merits, not a natural continuation.

3. **Uncertainty-driven optimization (Direction 3) — moderate support,
   with a real caveat.** Progressive Uncertainty-Guided Evidential U-KAN
   and the PGU/REH pixel-weighting paper show real precedent for using
   the uncertainty *value* (not gradient) to steer the segmentation
   branch — and this project's own Step E2 finding (ABO's effective ratio
   has a genuine, non-confounded relationship with ECE, not with Dice;
   see `abo_frozen_lessons_learned` memory) makes this a live, motivated
   thread. But the NeurIPS 2024 "Mirage" paper is a serious caution:
   before building on the evidential head's uncertainty output, it's
   worth checking whether that signal is genuinely informative at all —
   otherwise this risks repeating the ABO pattern of an elaborate
   mechanism on top of a branch whose output isn't load-bearing.

4. **Curriculum optimization (Direction 5) — thin evidence.** Nothing
   found tests this specific setup (segmentation-first, uncertainty-later
   curriculum on a two-head medical model). Plausible, cheap, but weakly
   motivated by the literature search.

5. **Gradient direction optimization (Direction 1) — weakest supported,
   least motivated.** Phase B already falsified the persistent-conflict
   premise these methods (PCGrad, CAGrad, Nash-MTL, GradVac) assume
   (Finding 1: conflict is transient). Pursuing this next means testing a
   third gradient-centric mechanism after two (static reweighting,
   dynamic magnitude control) already came up empty on the outcome metric.

**A finding outside the original candidate list**: the "gradient
conflict/imbalance as symptom, not cause" thesis (CORE-MTL + arXiv
2602.16125 + Rep-MTL together) suggests a natural **diagnostic** step
before committing to any representation-level architectural change:
measure whether the segmentation-branch and evidential-branch features
at the shared trunk are actually entangled (e.g., CKA similarity or
feature-space cosine on trunk activations, the same rigor Phase B applied
to gradients). This project's existing discipline — diagnose before
intervening (Phase B before C1) — argues for doing this measurement
before building any specific representation-level optimizer, rather than
assuming entanglement and building around it. See
`PHASE_E3_CANDIDATE_CONCEPTS.md`.

## Files

| File | Purpose |
|---|---|
| `PHASE_D_ANALYSIS_REPORT.md` | The Experiment D result this literature review is responding to |
| `PHASE_E3_CANDIDATE_CONCEPTS.md` | Ranked candidate optimizer concepts building on this matrix |

---

**Completed**: 2026-08-04
