# Phase E61 — Novelty Gap Search (Search-First, Design-Second)

## Purpose

Per the Q1 novelty filter: search the scientific gap FIRST, design the architecture SECOND. This phase runs literature searches across the seven candidate directions implied by the E43–E60 causal chain, *before* proposing any mechanism, and records which are occupied.

**Candidate must satisfy all seven filters to survive:**
1. Causal reason grounded in E43–E60
2. Changes the model's computation, not just a loss weight / head / resolution setting
3. Clearly different from known attention, deep supervision, feature dropout, uncertainty sampling, cascades, multi-hypothesis methods
4. Falsifiable pre-training prediction
5. Plausible explanation for the small-lesion failure specifically
6. Literature search finds no close mechanism
7. Measurable advantage over E54/E45 and baseline

## Searches run and verdicts

| # | Direction searched | Closest prior art found | Verdict |
|---|---|---|---|
| 1 | Conditional information sufficiency, coarse/fine, small objects | FineRS (2025, MLLM coarse-to-fine RL for tiny objects); C2FNet (coarse net locates, fine net re-segments patches); "zoom in and segment twice" test-time strategies | **OCCUPIED** — same cascade family already killed at E55 |
| 2 | Small-object representation collapse / bottleneck width theory | Slot-based "reconstruction bottleneck" width work (object-centric, not medical seg); model-collapse literature (generative degradation, different phenomenon) | **PARTIALLY OPEN** — neither directly occupies small-lesion encoding in medical seg, but neither is a mechanism either |
| 3 | Cross-scale arbitration / encoder-bottleneck complementarity / conditional feature reconstruction | ReSeg-UNet (MICCAI 2025, reconstruction-guided, three-level encoder/bottleneck/decoder cross-feature alignment); MSFB-Net (2025, bottleneck interactive fusion); MAFormer (2025, encoder-decoder semantic alignment) | **OCCUPIED** — densely |
| 4 | Representation-level causal intervention during training (not post-hoc) | SI²CRL (Med Image Anal 2025, spectrum-intervention invariant causal representation learning for medical seg); Causal unsupervised semantic segmentation (2026, two-step intervention) | **OCCUPIED** |
| 5 | Conditional computation / per-sample gating of the coarse pathway | BRDG (CVPR 2026, boundary-responsive differentiable gating selecting coarse vs refinement path); Granular-Computing SAM (coarse-to-fine compute concentration) | **OCCUPIED** |

## Interim conclusion

Four of the five directions searched are directly occupied by 2025–2026 work, mostly in the exact medical-segmentation setting. This is consistent with the pattern already established across E55/E57/E58/E59-E60 novelty checks: the "coarse-context / small-object / cross-scale" design space around U-Net bottlenecks is one of the most densely worked areas in current medical imaging literature.

The one partially-open area (#2, representation-collapse theory for small objects) is **not currently a mechanism** — it is a theoretical framing without an obvious operator attached. It would need a genuinely new operator derived from it to satisfy filter #2, and that operator would then need its own novelty check.

## Honest assessment against the seven filters

No candidate emerging from these searches currently satisfies filters 3 and 6 simultaneously. The searches did not surface a gap; they surfaced further confirmation that this neighbourhood is saturated.

