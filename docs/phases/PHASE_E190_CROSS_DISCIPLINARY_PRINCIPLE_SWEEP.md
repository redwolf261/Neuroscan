# E190 — Cross-disciplinary principle sweep (audit only, no code)

**Date**: 2026-09-20
**Status**: AUDIT COMPLETE. No experiment run, none authorized by this document.
Follows the E188 freeze. Real web search, 2024–2026 prior art.

## P3 fixed BEFORE searching (so principles cannot be retrofitted to findings)

The measured NeuroScan failure, from the record — three facts, distinguished by *kind*:

**F1 — the tail is information-absent, not computation-absent.** E142 (causal): zeroing `t1c`
collapses ET 0.8433→0.0015. The 15 failing subjects *are functionally that ablation*; four have
negative contrast. E137/E140/E141 each failed for this reason — no conditioning, recalibration,
or loss reweighting recovers information absent from the input tensor.

**F2 — the non-tail residual is boundary-localised and at the annotation ceiling.** E167:
98.6% of ET error within 3 voxels, interior 1.4%, enrichment 1.13/2.11/1.96 (geometry control
load-bearing). Model at ET 0.900 vs published inter-rater median 0.77 (Menze TMI 2015).

**F3 — region contagion is the one *computation* failure.** E133: TC Dice vs ET Dice,
controlling log ET size **and** log NCR size, ρ=**+0.797**, p=1.1e-28. Subject 01176 has 45,423
NCR voxels (TC should be large and easy) and the model predicts **exactly zero**, while WT=1.0000
on the same volume. 5/8 ET and 5/7 TC catastrophic cases predict exactly zero voxels; lowering
the threshold 0.5→0.2 changes nothing.

**Consequence for admissibility.** A principle passes P3 **only** if it targets F3. Principles
aimed at boundary refinement fail P3 (F2 is at the ceiling). Principles aimed at recovering weak
signal fail P3 (F1 says it is not there). This is the filter that makes the sweep honest.

**Note carried from E133**: the contagion result is *correlational*. The causal test — does
forcing ET output change TC output? — **has never been run**. This matters for every candidate
below.

## The sweep

| Principle | Imported from | NeuroScan failure addressed | Medical-seg prior art | Exact collision | Novelty | Cheap test |
|---|---|---|---|---|---|---|
| **A. Hierarchical/nested output constraint** (ET⊆TC⊆WT enforced structurally) | Structured prediction | F3 directly | **Yes, heavy** | 🔴 **Stick-breaking parameterization guaranteeing $p_{ET}\le p_{TC}\le p_{WT}$ is published for BraTS**; also "soft cascade of three heads where coarser regions generate spatial attention for finer" — that is nearly verbatim the obvious fix | **KILL at P2** | n/a |
| **B. Successive interference cancellation** (decode strongest, subtract, re-decode) | Communications | F3 — sequential region decoding | Partial | 🟡 SIC itself is mature and unrelated to seg; but its segmentation analogue *is* cascade/sequential region decoding = A's territory. Group-SIC ("detect first group, subtract, re-run on refined observation") maps onto published cascade decoders | **KILL at P2** (collapses into A) | n/a |
| **C. Predictive coding / iterative feedback inference** | Comp. neuroscience | F3 | **Yes** | 🔴 "Deep Recurrence for Dynamical Segmentation Models" (2507.10143) — segmentation network revising its own output over time, explicitly framed as predictive coding; PC networks + inference learning survey (ACM CSUR 2026) | **KILL at P2** | n/a |
| **D. Conformal risk control** (distribution-free FNR guarantee) | Statistics | F3's *symptom* (confident zero-prediction), not its cause | **Yes, and on this exact data** | 🔴 Federated CRC study uses **FeTS-2022, 1,251 subjects, BraTS-2021-trained SegResNet**, reports per-institution FNR violations. Also CVPR 2026 spectral CRC, morphological conformal prediction sets for segmentation | **KILL at P2** | n/a |
| **E. Selective prediction / abstention & failure detection** | Decision theory | F3's symptom | **Yes** | 🔴 FSNet failure-detection for semantic segmentation; false-negative reduction via uncertainty; interpretable cascade classifiers with abstention (AISTATS 2019) | **KILL at P2** | n/a |
| **F. Missing/unreliable-modality detection & deferral** | Multimodal robustness | **F1** (correct target for the tail) | **Yes, extremely dense** | 🔴 CLoE expert-consistency for missing-modality seg; MuteBench; modality-agnostic input channels for brain lesions (2509.09290); deferral-rate analyses per modality combination | **KILL at P2** | n/a |
| **G. Residual/defect correction, a posteriori error estimation** | Numerical PDEs | F2 | Audited **separately and already tested** | 🔴 Killed empirically at E179 Gate 2 — the residual carried no information beyond plain boundary proximity (22% of 118 bins) | **ALREADY KILLED** | n/a |
| **H. Task-demand compute routing** | Adaptive computation | (was aimed at F2/F3) | — | 🔴 Killed empirically at E178 (NeuroScan ≤ Constant) | **ALREADY KILLED** | n/a |
| **I. Latent steering / activation geometry** | Mech-interp | (was aimed at F3) | — | 🔴 Killed at E188-F — effect entirely readout-row-space | **ALREADY KILLED** | n/a |
| **J. Optimal transport, operator splitting, multigrid, adjoint, RG, sheaf, influence functions, compressed sensing, Lyapunov** | 10 fields | various | — | 🔴 E168 already swept these: 7 died on prior art, 3 on arithmetic. Sheaf gluing possibly open but worth ~+0.1–0.3pp, not the bar | **ALREADY SWEPT** | n/a |

## Verdict

$$
\boxed{\text{No candidate survives P2.}}
$$

Every principle that passes **P1** (genuine computational principle) and **P3** (addresses a
measured NeuroScan failure) **dies at P2** — the exact mechanism is already published for medical
segmentation, and in the two most on-target cases (A and D) it is published *for BraTS
specifically*.

This is not search exhaustion. It is the same structural law E162 recorded and this project has
now confirmed nine times: **on a saturated benchmark, a well-motivated observation predicts its
own prior-art density.** F3 is a *good* observation — which is exactly why hierarchical
constraint enforcement and conformal FNR control already exist for it.

## The honest reading

The candidate that best targets the one genuine computation failure (F3, region contagion) is
structural hierarchy enforcement — and that is precisely what the stick-breaking BraTS work does.
The project's own unexploited remnant is narrower than a method: **E133's causal test was never
run.** "Does forcing ET output change TC output?" is an unanswered *mechanistic* question, not a
novelty candidate. It would produce a finding, not a method, and per P4 it does not define an
algorithm whose success or failure is testable against the ≥1pp bar.

## Recommendation

Do not manufacture an experiment to avoid this conclusion. Three branches have now been closed by
their own pre-registered gates (E178, E179, E188) plus the earlier E169–E176 closure and this
sweep. The defensible output of this project is the **measured closure** — the residual
decomposed into information-limited (F1, causal) and boundary-localised (F2, at the annotation
ceiling) components, with F3 documented as a real coupled-decision phenomenon — together with the
methodology: ~25 pre-registered kills, ten prior-art audits, and a documented record of catching
its own errors before they calcified (E62 shared-tensor, E153 normalization artifact, E180
tile-overlap inflation, E185 MaxPool tie-breaking, E186 multi-window contamination, E188
BatchNorm zero-init, and the E15-does-not-reproduce correction found this session).

**That is a report, not a ninth search.**

## Sources

Knowledge-guided brain tumor segmentation w/ hierarchical consistency (arXiv 2507.08574; BMC Med
Imaging 2026) · Hierarchical text-guided BraTS sub-region prompts (arXiv 2603.21083) ·
Multi-modal fusion w/ cascaded stick-breaking parameterization for ET⊆TC⊆WT · Deep Recurrence for
Dynamical Segmentation Models (arXiv 2507.10143) · Predictive Coding Networks and Inference
Learning survey (ACM CSUR 2026) · Federated Conformal Risk Control on FeTS-2022/BraTS-2021
(arXiv 2606.20115) · Spectral Conformal Risk Control (CVPR 2026) · Conformal Prediction for Image
Segmentation via Morphological Prediction Sets (arXiv 2503.05618) · FSNet failure detection
(arXiv 2108.08748) · Interpretable Cascade Classifiers with Abstention (AISTATS 2019) · CLoE
missing-modality expert consistency (arXiv 2603.09316) · MuteBench (arXiv 2605.15235) ·
Modality-agnostic input channels for brain lesions (arXiv 2509.09290) · Successive Interference
Cancellation (survey/Wikipedia; arXiv 2102.07704 coded demixing) · Adaptive cascade decoders for
challenging medical regions (Comput Biol Med 2024) · MaskMed decoupled mask/class prediction
(arXiv 2511.15603)
