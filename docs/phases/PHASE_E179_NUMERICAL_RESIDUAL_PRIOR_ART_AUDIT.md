# E179 — Numerical error-correction principle: prior-art audit (pre-code, per explicit instruction)

**Date**: 2026-09-19
**Status**: Audit complete, real web search (10 queries, as specified). NO code written — per
explicit instruction, this audit runs before any implementation, diagnostic or otherwise.

## Context — the strategic pivot

After E178's clean kill (task-demand rank routing: NeuroScan ≤ Constant, no rescue, Stage 2
never run — see `PHASE_E178_TASK_DEMAND_ROUTING_PREREG.md`'s frozen closing note), the user
identified a pattern across 12 prior attempts (E44-E178): every branch tried to extract a method
from a property of the internal representation $Z$, and every one either hit prior art or
produced a diagnostic correlation with no causal transfer to Dice. Explicit strategic
redirection: **leave representation space**. Candidate principle proposed: segmentation as
*numerical error correction* — treat the prediction as a coarse approximation to a geometric
object (the lesion boundary) and ask where a numerical-methods-style residual concentrates,
analogous to a posteriori error estimation / defect correction in finite-element/PDE theory.

**Candidate diagnostic quantity (explicitly NOT claimed novel, not yet a loss/attention map)**:

$$
\mathcal{R}(x) = \frac{\|\nabla p(x)\|}{|p(x) - \tau| + \epsilon}
$$

The actual question under audit, stated precisely: *can segmentation be formulated as iterative
numerical error correction using a prediction-derived geometric residual, where the residual
determines an actual correction operator* — not boundary loss, not generic uncertainty, not
generic adaptive refinement (all separately, already known occupied per this project's own
history).

## Ten queries run (real web search, session date 2026-09-19)

1. residual correction segmentation iterative
2. "a posteriori error estimator" segmentation deep learning
3. finite element error estimator deep learning segmentation
4. level-set residual neural network segmentation boundary
5. PDE-inspired residual correction medical image segmentation
6. boundary residual adaptive neural computation segmentation gradient magnitude decision boundary
7. defect correction segmentation numerical method
8. iterative refinement implicit surface segmentation neural network
9. numerical error estimation neural segmentation gradient over probability minus threshold
10. "dual weighted residual" segmentation neural network adaptive mesh medical

## Findings, closest to furthest

| Work | What it does | Distance from the candidate chain |
|---|---|---|
| **Neural functional a posteriori error estimates** (arXiv 2402.05585) / **DWR-guided adjoint computations** (arXiv 2102.12450) | Computable, GT-free error estimates for neural PDE solutions (PINNs), used as accuracy-independent stopping/refinement criteria | **Closest conceptual match** — a posteriori, prediction-derived, GT-free error estimation feeding an adaptive procedure. But domain is PINN/PDE solving, not segmentation; the estimator is a diagnostic/stopping criterion, not a "correction operator" derived directly from the residual the way the candidate formula proposes |
| **PDE-UNet** (BraTS2020) | PDE-inspired *convolutional block* modifies U-Net architecture directly | Same domain (BraTS), "PDE-inspired" framing, but modifies the network's forward computation architecturally — not a post-hoc residual-driven correction step applied to an existing prediction |
| **Certified ML / a posteriori error estimation for PINNs** (arXiv 2203.17055) | Rigorous error bounds for physics-informed networks | Same PDE-theory lineage, same "not segmentation" gap |
| **Defect-correction methods** (classical numerical analysis, Böhmer & Stetter et al., the actual field the candidate formula is analogizing to) | Iteratively improve an approximate solution using the residual/defect of an operator equation, without mesh refinement | Confirms the *field* is real, well-established, and well-documented (textbook-level) — but zero located applications to neural segmentation specifically. The analogy source is real; its transplant to segmentation was not found |
| **Level-set / active-contour + CNN hybrids** (DRLSU-Net, Deep Active Lesion Segmentation) | CNN estimates contour position/energy-functional parameters, level-set evolves the boundary | Structurally adjacent (uses $\nabla p$-like and boundary-proximity information to drive an iterative geometric update) but the mechanism is level-set energy minimization, not a residual-magnitude-over-distance-to-threshold ratio, and not framed as numerical error correction |
| **Boundary loss for highly unbalanced segmentation** (Kervadec et al.) | Distance-transform-based loss term operating on the boundary | Occupied (this project's own prior history already treats "boundary loss" as closed) — confirms the audit's own explicit instruction to exclude this framing was correct to give |
| **Residual-correction self-training for semi-supervised segmentation** | A second network predicts a *residual of the segmentation itself* (label-space correction), trained with labeled data | Different sense of "residual" (segmentation-output correction via self-training) — not a geometric/gradient-based residual, not GT-free at the mechanism level |
| Iterative refinement encoder-decoders, mesh-adaptation CNNs (AMBER, sizing-field prediction) | Learned iterative refinement / mesh density prediction | Generic "iterative refinement" family, already known occupied per this project's own standing rule; not specific to the residual-as-correction-operator chain |

## Verdict

$$
\boxed{\text{🟢 The specific chain — a prediction-derived geometric residual } \mathcal{R}(x)
\text{ defining an actual correction operator, framed as numerical defect correction — was NOT
located.}}
$$

The **field being analogized to (defect correction / a posteriori error estimation) is real,
established, and well-documented** — this is not a fabricated numerical-methods framing. Its
closest neural instantiation (functional a posteriori error estimates for PINNs, DWR-guided
adjoint computation) targets PDE-solving networks, not segmentation, and stops at *diagnostic
estimation*, never proposing the estimator itself as a *correction operator*. PDE-UNet is the
one BraTS-specific, PDE-motivated result, but it is an architectural change, not a residual-
driven post-hoc correction — a different mechanism shape entirely.

**Per E162's own structural law** (informativeness predicts prior-art density on a well-studied
problem): this is a genuinely narrower, more surprising combination than most of this session's
prior candidates (E170b/E171/E177 all found the general shape occupied even when the specific
combination wasn't) — the "correction operator derived from a numerical residual" framing has
essentially no direct neural-segmentation precedent located, which is itself somewhat unusual
and worth treating with proportionate caution (absence after ten searches on an active
numerical-analysis-adjacent subfield is more informative than the same absence would be on an
obscure one, but still not proof).

## What this licenses, and what it does not

Licenses: proceeding to the next step the user's own plan specifies — determining whether
$\mathcal{R}(x)$ (or a considered alternative) exhibits a real mathematical/statistical
phenomenon worth exploiting, **before** any loss term, attention map, or trained operator is
built. Does not license claiming novelty, does not license skipping the "does the phenomenon
exist" gate, and does not exempt this branch from the same falsifiable, ≥1pp, kill-if-it-fails
discipline just applied to E178.

## Sources

Neural functional a posteriori error estimates (arXiv 2402.05585) · Neural network guided
adjoint computations in dual weighted residual error estimation (arXiv 2102.12450) · Certified
machine learning: a posteriori error estimation for PINNs (arXiv 2203.17055) · A priori/a
posteriori error estimates for the Deep Ritz method (arXiv 2107.11035) · PDE-UNet (BraTS2020) ·
Defect Correction Methods: Theory and Applications (Böhmer/Stetter, Springer) · Local Defect
Correction Method and Domain Decomposition Techniques (Springer) · DRLSU-Net level set + U-Net
(ScienceDirect) · Deep Active Lesion Segmentation (bioRxiv) · Boundary loss for highly
unbalanced segmentation (Kervadec et al., MIDL) · A Residual Correction Approach for
Semi-supervised Semantic Segmentation (Springer) · AMBER adaptive mesh generation (arXiv
2505.23663)
