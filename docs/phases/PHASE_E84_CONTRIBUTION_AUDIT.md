# Phase E84 — Contribution Audit (no experiments; research-state decision)

## Purpose

Per the user's explicit instruction: freeze experimentation at E83 and
answer a research-design question, not run another causal probe. The
question: does the surviving E48-E83 evidence demand a missing
algorithmic operation, or is it a well-characterized phenomenon that
belongs in analysis/writeup rather than a new method?

## The surviving evidence table (verbatim from the user's own synthesis, confirmed accurate against the phase docs)

| Observation | Establishes | Rules out | Open question |
|---|---|---|---|
| E48 | Bottleneck ablation is size-dependent; genuine causal importance | Purely superficial explanation | Why structured this way |
| E65 | Translation of enc1 severely damages segmentation | Pooling explanation | Exact correspondence property used |
| E65 | Translation ≫ local permutation/smoothing | Local-detail-only explanation | Function of displacement causing damage |
| E73 | Spatial causal-sensitivity map predicts errors | Purely scalar difficulty | Exploitability of that structure |
| E74-E76 | Magnitude correlates/causally affects sensitivity at enc1 | "Just correlation" | Magnitude's role vs. direction |
| E77 | High magnitude ~ lesion signal | Global magnitude suppression | Robustness without destroying salience |
| E78 | Direction explains most translation damage | Pure magnitude-control solution | What property of direction is vulnerable |
| E79 | Local smoothing barely helps | Directional smoothing/Lipschitz fix | What positional property matters |
| E80 (corrected by E81) | Coherent joint shift mostly recovers after realignment | "Both streams independently absolute-bound" | Behavior under relative displacement specifically |
| E82 | Local donor displacement gives graded damage | Hard address identity / free local equivalence | Mathematical response law |
| E83 | Apparent axis anisotropy ~48% explained by physical voxel geometry, remainder not significant | Axis-specific mechanism as PRIMARY explanation | Any residual intrinsic anisotropy (unresolved, underpowered) |

## Failed algorithmic branches (confirmed against this session's own phase docs)

| Branch | Verdict | Reason |
|---|---|---|
| DCU / generic deformable correction | Occupied | Prior art (E67/E67b) |
| SC-DCU correspondence supervision | Not justified | Mechanistic prediction failed (E68) |
| CAS | Killed | Real mechanism was magnitude/variance smoothing, not correspondence correction (E70) |
| Fragility routing (CDCG gate) | Killed | Difficulty markers not substitutable resources (E71) |
| Fragility loss weighting (FWL) | Killed | Aggressive weighting harms; gentle weighting loses the signal (E72) |
| SDLR (spatial gate on predicted sensitivity) | Killed | Weak distilled signal became inert once wired in (E73) |
| E54/A96 as a +1pp mechanism | Unconfirmed | 3-seed aggregate evidence insufficient |

## Strong positive results, and their honest status

1. **4-modality multimodal input gain**: +1.75pp, 3-seed confirmed (std
   0.27pp). This is a real, robust **performance result** — but not
   novel (multimodal input is a well-established technique).
2. **Bottleneck causal self-knowledge**: the network's bottleneck
   representation predicts its own ablation sensitivity, held-out
   ρ≈0.87-0.87 across two independent validations. This is a real,
   validated **scientific phenomenon** — but every attempt to turn it
   into an algorithm (routing, reweighting, spatial gating) has failed.
   It is not yet, and per this audit may never become, a performance
   result.

## The conceptual pivot, evaluated

The user's framing: every surviving observation was tried as a
**control variable** (fragility→gate, fragility→loss weight,
magnitude→suppression, direction→smooth, correspondence→deformable
alignment) and every one failed. This audit tests whether that pattern
means the observations are better understood as **descriptors of the
representation** than as levers to pull.

### Question 1 — What does existing literature already do?

Deformable convolution, cross-attention fusion, spatial transformers,
and DCU-family correspondence modules all solve the general shape of
"the decoder may need a feature from somewhere other than the naively
aligned position" via a **learned, per-location, per-input predicted
offset or attention weight**. This is the entire occupied space
identified across E67, E67b, and tonight's later targeted searches
(around E79-E80). No literature search this session found this general
shape to be open.

### Question 2 — What has our evidence actually added?

Not "correspondence matters" — that is already known and already
addressed by the occupied methods above. What tonight's causal chain
adds, that literature search alone would not have produced, is the
**causal shape of the failure**, established by intervention rather than
assumed:

- Not a magnitude problem — magnitude only amplifies (E77, E78).
- Not local angular noise — smoothing barely helps (E79).
- Not independent per-side coordinate binding — the network is
  near-equivariant when both sides move together (E81 correcting E80).
- Not primarily axis-structured — E83 attributes roughly half of the
  apparent anisotropy to ordinary resampling geometry, with the
  remainder not statistically confirmed.
- It IS smoothly graded with distance, even within a single axis
  (E82's radius 1→2→3 result, unaffected by E83's cross-axis confound).

This is a genuine, non-obvious, causally-established characterization.
It rules out several plausible-sounding mechanisms (magnitude control,
smoothing, per-side independent fixes, axis-aware fixes) before any of
them were built into a training run — which is real research value,
independent of whether it yields a new algorithm.

### Question 3 — Is there a missing operation that follows necessarily from the evidence?

This is the decisive question, and the honest answer is **no**.

A deformable/attention-style module with a learned, smoothly-decaying
local search kernel is not distinguishable, after the fact, from what
E82's graded response law describes. Existing occupied methods are
already flexible enough to express "damage grades smoothly with
distance, is not driven by magnitude, is not fixed by simple smoothing."
Nothing in the E48-E83 chain identifies a functional form or an
architectural constraint that a sufficiently well-trained deformable or
attention module could not, in principle, already fit. The evidence
narrows WHAT KIND of correspondence-repair mechanism would be
appropriate (graded, magnitude-independent, not a smoothing operator)
but does not identify an operation outside the space those existing,
occupied mechanisms already span.

## Verdict

**Per the user's own decision rule: the answer to Question 3 is NO.**
The algorithm hunt on this specific line (enc1 skip-connection
positional/directional vulnerability) is stopped here, not because the
evidence is weak, but because it does not license a genuinely new
architectural operation distinct from occupied prior art. Forcing a
method now would repeat the E70-E73 pattern: build something plausible,
spend GPU time, and find it killed or indistinguishable from existing
work.

**This is a legitimate, deliberate research conclusion, not a failure to
find an idea.** The project has:
1. A confirmed, non-novel but real performance gain (multimodal input).
2. A validated, narrow, genuinely interesting scientific phenomenon
   (bottleneck causal self-knowledge; enc1's graded, direction-carried,
   magnitude-amplified positional sensitivity) that is publishable as
   analysis/characterization, backed by an unusually rigorous causal-
   intervention methodology (E43 through E83) that itself caught and
   corrected multiple real errors along the way (E62/E63's shared-tensor
   bug, E80's coordinate-frame confound, E82's Dice-insensitivity bug,
   E83's physical-spacing confound) — a methodological throughline that
   is, in its own right, a defensible contribution.
3. A clean, honest kill-list of every mechanism actually attempted,
   each with a specific diagnosed reason rather than an ambiguous null.

## What this means going forward

No E85 is scheduled. Per the user's own framing: producing one merely
because the phase number feels unfinished is exactly the failure mode
this audit exists to prevent. Future work on this thread should only
resume if either (a) new evidence emerges that identifies a genuine gap
in the occupied mechanism space, or (b) the user decides the
scientific-phenomenon framing (rather than a new algorithm) is the
right basis to write up, in which case the next natural step is
synthesis/writeup, not further experimentation.
