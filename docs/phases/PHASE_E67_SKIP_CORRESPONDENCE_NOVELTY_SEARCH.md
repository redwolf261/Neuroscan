# Phase E67 — Skip-Connection Correspondence-Correction Novelty Search

## Purpose

Per the project's search-first, design-second discipline: before designing any operator
around E65's finding (absolute spatial correspondence at the `[upconv1, enc1]`
concatenation is the dominant causal factor, Δ=0.214 vs. Δ=0.027 for local
arrangement, size-specific ρ=−0.553), search the 2024–2026 literature for
whether this exact design space is already occupied.

## Searches run and verdicts

| # | Direction searched | Closest prior art found | Verdict |
|---|---|---|---|
| 1 | Registration-aware / deformable skip-connection alignment for U-Net | **Dynamic U-Net's DCU module** (arXiv:2403.07303, 2024) — learns a deformable offset from concatenated upsampled-decoder + skip-encoder features via a 3×3 conv, applies it via modulated deformable convolution to spatially realign the skip feature before concatenation. Isolated ablation on FLARE 2021 / AMOS 2022 (abdominal CT, 2D). | **OCCUPIED** — this is directly, closely the mechanism E65's evidence would motivate: learned spatial offset correction at the exact skip-concatenation point, addressing exactly the misalignment phenomenon. |
| 2 | ICCV 2025 offset-learning for spatial/class feature alignment | OffSeg (arXiv:2508.08811) — learns feature + class offsets for per-pixel classification alignment, 2D natural-image segmentation (ADE20K, Cityscapes, COCO-Stuff, Pascal). Different locus (classifier-feature alignment, not encoder-decoder skip alignment specifically) but same general "learned offset corrects misalignment" idea. | **OCCUPIED** (adjacent) |
| 3 | Size-conditional / lesion-scale-adaptive deformable skip correction | No direct match found. DMSF-Net (2026) modulates deformable-offset generation via global attention, not lesion size. SYNAPSE-Net (arXiv:2510.26961, brain MRI, WHM/ISLES/BraTS-2020) has "lesion-aware hierarchical gating" but confirmed (fetched, not assumed) to be **semantic attention gating** (`f_gated = f + f⊗σ(gate)`), not spatial/deformable correspondence correction, and does not explicitly condition on lesion size (only indirectly via oversampling small-lesion slices during training). WT Dice reported = 0.906 on BraTS 2020. | **NOT FOUND** — the specific combination of (a) deformable spatial correspondence correction (b) explicitly modulated by a lesion-size signal is not covered by anything found. |
| 4 | Causal-audit-derived (vs. intuition-derived) skip-connection design | No direct match found across all searches — every skip-alignment paper found proposes the fix first and validates with an ablation afterward; none work backward from a pre-registered causal decomposition (translation vs. local arrangement vs. channel identity vs. smoothing, each independently measured and ranked) the way E62→E65 does. | **NOT FOUND** |

## Honest assessment

The **mechanism** (learned deformable/spatial offset correcting encoder-decoder
skip misalignment before concatenation) is occupied — Dynamic U-Net's DCU is a
close, direct precedent with its own isolated ablation. Building this mechanism
for 3D brain MRI would be a domain port (2D→3D, abdominal CT→brain MRI), not a
novel algorithm, by this project's own standard (existing algorithm + different
domain ≠ novelty).

What is **not** occupied: making the correction's strength or scope explicitly
conditional on a lesion-size signal, AND deriving that specific design choice
from E65's own pre-registered causal decomposition (the ~8× magnitude gap
between translation-sensitivity and local-arrangement-sensitivity, and its
size-specificity, ρ=−0.553) rather than from intuition or a generic ablation
afterward.

## Verdict

This does **not** support a "novel algorithm" claim in the sense of a new
architectural mechanism. It supports a narrower, still legitimate claim: a
**causally-derived, size-conditional calibration** of an existing correction
class (deformable/offset-based skip realignment), where the actual contribution
is the design methodology (audit → magnitude-ranked decomposition → targeted,
size-conditional instantiation) plus the specific size-conditioning choice,
not the base mechanism itself. The project's write-up, if this proceeds,
should frame the contribution honestly on these terms — matching constraint
#17 (no forced novelty) and #4 (existing mathematics + new mechanism/application
= potentially novel; existing algorithm + different name ≠ novelty).

## Next step (not yet done)

Design SC-DCU (size-conditional deformable correspondence correction): same
offset-learning/deformable-realignment mechanism as DCU, adapted to 3D, with
the offset magnitude/gate explicitly modulated by a lesion-scale signal derived
from the same causal chain (E48's native_size, or a learned proxy at
inference/training time since ground-truth size is not available at test time
for a real deployed model — this design detail needs resolving before
implementation). Requires its own feasibility check (does a usable size signal
exist without leaking test-time GT?) before any training.
