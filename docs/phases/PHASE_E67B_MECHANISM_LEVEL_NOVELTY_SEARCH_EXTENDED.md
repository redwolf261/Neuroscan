# Phase E67b — Extended Mechanism-Level Novelty Search

## Purpose

Continuation of E67, per explicit user instruction to search further afield for a
structurally new mechanism (not a calibration of an existing one) before accepting
the size-conditional-DCU-calibration framing. Went back to the raw causal facts
(E48 + E65) rather than the skip-connection framing specifically, and searched
progressively broader/more abstract restatements of "why does knowing WHERE
something is matter disproportionately for SMALL objects."

## Searches run and verdicts

| # | Direction searched | Closest prior art found | Verdict |
|---|---|---|---|
| 5 | Position/localization decoupled from semantic content, learned separately | **MPLSeg** — "Decoupling semantic and localization for semantic segmentation via magnitude-aware and phase-sensitive learning." Uses Fourier magnitude (semantics) / phase (position) decomposition explicitly because phase carries positional information; an Adaptive Frequency-aware Module (AFM) addresses "spatial location misalignment during multi-level feature fusion," with a dedicated auxiliary PSL loss for localization optimization independent of semantics. | **OCCUPIED** — a different mathematical route (Fourier) to the same position/content separation idea E65's evidence would motivate. |
| 6 | Object-scale-conditioned correction magnitude / adaptive registration strength | No direct match. Adjacent: DMSF-Net modulates deformable offsets via global attention (not object scale); adaptive Riemannian registration addresses transform-size-depends-on-image-size at the OPTIMIZATION level, not per-object within a single segmentation forward pass. | **NOT FOUND**, but also no strong candidate literature cluster — likely a genuinely thin area, not necessarily because it's novel, but because it may not be a well-posed research question at this granularity. |
| 7 | Position as its own learned/predicted quantity with its own uncertainty, jointly optimized with content, scale-gated | No direct hit. Found general segmentation-uncertainty literature (U-SEG, RKHS uncertainty operators) and multi-task depth+segmentation uncertainty, none scale-gated on the object itself. | **NOT FOUND**, same caveat as #6. |
| 8 | Explicit coordinate-confidence / noise-aware positional embedding for small objects | **HELP / Heatmap-guided Positional Embedding (HPE)**, arXiv:2604.15065 (2026) — selectively preserves positional encoding in foreground-salient regions, suppresses it in background, for DETR-style transformer small-object DETECTION (not U-Net segmentation). Addresses a related but architecturally distinct problem (query-retrieval noise in a transformer decoder, not encoder-decoder skip correspondence in a CNN). | **ADJACENT, not directly occupied** — different architecture class (transformer detection vs. CNN segmentation) and different mechanism (selective suppression vs. correspondence correction), but conceptually close enough that a direct port would face the same "different domain ≠ novelty" problem as DCU. |

## Assessment after 4 search rounds (8 total searches across E67/E67b)

No search round surfaced a mechanism-level idea that is both (a) clearly
unoccupied and (b) directly actionable from this project's own causal evidence
(E48, E65) without requiring a further, separate design leap of its own. The
space around "small objects need positional precision more than semantic
richness" is being actively worked from multiple architectural angles in
2024-2026: CNN skip-connection deformable correction (DCU), frequency-domain
magnitude/phase decoupling (MPLSeg), and transformer positional-embedding
noise-suppression (HELP) all approach visibly the same underlying intuition
from different mathematical directions, all within roughly the same 24-month
window. This pattern — several independent groups converging on the same
underlying idea from different angles in the same short window — is itself
reasonably strong evidence that the idea is "in the air" and not a genuinely
open gap, rather than evidence that a slightly-different fourth angle would be
the exception.

This mirrors the project's own prior novelty-search conclusions at E55/E57/E58/
E59-E60/E61/E67: this neighborhood (small-object / coarse-context /
position-content separation around U-Net-style encoder-decoder architectures)
is one of the most densely worked areas in current segmentation literature.

## Verdict

No genuinely unoccupied mechanism-level idea was found after 4 focused search
rounds (8 total queries) building directly on E48/E65's own causal evidence.
Per the project's own constraint #17 (no forced novelty), this extended search
does not overturn E67's conclusion. The strongest **honest** contribution
available from this evidence remains the E67 finding: the causally-derived,
size-conditional calibration and design methodology, not a new base mechanism.

## Artifacts

- This document + `PHASE_E67_SKIP_CORRESPONDENCE_NOVELTY_SEARCH.md` (the two
  together constitute the full novelty-search record for this branch).
