# Phase E81 — Translation-and-Realignment Control: OUTCOME A — E80's Arm 3 was a coordinate-frame artifact

## Origin

The user identified a confound E80 did not rule out: a conventional
convolutional decoder is spatially structured. Coherently shifting both
sides of the skip concatenation by the same offset t could simply
produce a segmentation displaced by ~t voxels — comparing that displaced
prediction against the fixed, unshifted ground truth would produce a
large Dice drop for a trivial reason, fully explaining E80's counter-
intuitive Arm3 > Arm1 result without any "absolute-coordinate-key"
story. This phase is the clean, zero-training discriminator.

## Method

Reused E80's own `forward_dual_shift` exactly (no reimplementation) for
the coherent joint shift (both enc1 direction and upconv1 shifted by the
same t=3 offset — E80's Arm 3 condition, reproduced here for a matched
comparison). Two independent checks:

1. **Realignment**: undo the same shift on the final prediction before
   scoring — D_realigned = Dice(T_{-t}(P_t), Y) — compared against
   D_unrealigned = Dice(P_t, Y) (E80's own Arm 3 quantity).
2. **Feature-level equivariance**: compare P_t directly against T_t(P_0)
   (a pure translation of the CLEAN output) via mean absolute voxel-wise
   error, with a reference scale (|T_t(P_0) - P_0|) for context.

## Results (125 held-out subjects)

| Quantity | Mean |
|---|---|
| D_intact (clean) | 0.9023 |
| D_unrealigned (E80's Arm 3, reproduced) | 0.4637 |
| D_realigned (undo shift on output) | **0.8737** |

| Quantity | Mean | 95% CI |
|---|---|---|
| S_unrealigned | 0.4386 | [0.422, 0.457] |
| S_realigned | **0.0286** | [0.023, 0.035] |

**Recovery fraction: 93.5%** (S_unrealigned 0.439 → S_realigned 0.029),
paired t p=3.0e-84. Undoing the exact same shift on the output recovers
almost all the way back to clean performance.

### Feature-level equivariance check confirms it independently

- Mean |P_t − T_t(P_0)| (actual equivariance error): **0.0010**
- Mean |T_t(P_0) − P_0| (reference: how much translation alone moves the
  clean output): 0.0101
- **Relative equivariance error: 0.096** — the coherently-shifted output
  P_t is, to within ~10% of the reference scale, exactly what a pure
  translation of the clean output would look like. The network behaves
  near-equivariantly under this coherent internal shift.

## Verdict: Outcome A — retract E80's Arm 3 interpretation

Both checks agree cleanly. **E80's large "coherent joint shift" damage
was substantially a coordinate-frame displacement artifact**, not
evidence of independent absolute-coordinate binding on both sides of the
skip connection. The network essentially reproduces a translated version
of its clean output when both its inputs are coherently translated —
exactly the confound the user predicted.

## What this means for the standing causal chain

**Retracted**: E80's "both encoder and decoder features independently,
additively coordinate-bound" conclusion (drawn from Arm 3 and the
mismatch sweep, both confounded by this same effect).

**NOT retracted, and still standing**:
- E65/E78/E79's own results (enc1 direction translated ALONE, decoder
  side untouched) — this intervention is inherently asymmetric. The
  network has no way to produce a compensating global output shift when
  only one input moved and the other (upconv1) is still in its correct
  frame; there is no coordinate-frame escape hatch available. This
  confound cannot apply to those results, and E81 does not touch them.
- E80's Arm 2 (decoder-only shift, S=0.262) — also asymmetric (only
  upconv1 moved), so also not subject to this confound. This remains a
  genuine, still-standing finding: the decoder's own upconv1 pathway is
  independently sensitive to positional disruption, a real result
  distinct from the retracted Arm 3 claim.

The correct standing picture, post-correction: **enc1's direction
component (E65/E78/E79) and the decoder's own upconv1 feature (E80 Arm
2) are each independently sensitive to being moved relative to their
partner** — but this is NOT the same claim as "each independently
encodes an absolute global coordinate," since the coherent-shift test
that would have supported that stronger claim is now shown to be
confounded. The correspondence/relative-alignment question from E80 is
genuinely still open — E81 shows the network is near-equivariant when
inputs move together, which is actually more consistent with a
correspondence-based mechanism than the retracted "independent absolute
binding" reading was.

## Causal chain, corrected

```
E65: translating enc1 ALONE (asymmetric, no coordinate-frame escape) causes large damage -- STANDS
E73: causal vulnerability has spatial structure -- STANDS
E74: feature magnitude correlates with translation sensitivity -- STANDS
E75: magnitude CAUSALLY affects sensitivity at enc1 -- STANDS
E76: the causal relationship holds locally, at the real operating point -- STANDS
E77: high magnitude = lesion salience + vulnerability -- suppression unsafe -- STANDS
E78: direction carries positional structure; magnitude amplifies -- STANDS
E79: direction's vulnerability is absolute-coordinate-RELATIVE binding, not local noise -- STANDS
     (E81 clarifies: "relative to its correct partner/frame", not necessarily "globally absolute")
E80: decoder's own upconv1 pathway is ALSO independently translation-sensitive (Arm 2) -- STANDS
     Arm 3 "independent additive binding on both sides" -- RETRACTED by E81
E81: network is near-EQUIVARIANT under a COHERENT joint shift -- output moves WITH its inputs,
     consistent with a correspondence-preserving mechanism, not independent absolute keys
```

## Honest reading of where this leaves the search

The near-equivariance finding is actually informative in its own right:
the network CAN produce a coherently-shifted output when both its inputs
move together — meaning the underlying computation is not fundamentally
broken by translation per se. The genuine vulnerability (established by
E65 and E80 Arm 2, both asymmetric and confound-free) is specifically
about MISMATCH between the two sides of the concatenation — which is,
after all, closer to a correspondence problem than E80's retracted
reading suggested, even though the earlier novelty search already flagged
generic deformable/alignment fixes for that exact framing as occupied.
This is a genuine tension to sit with rather than resolve by picking
whichever framing is more convenient: the mechanism-level evidence points
toward correspondence, but the most direct architectural response to
correspondence problems is already occupied territory.

## Artifacts

- `experiments/exp_e12_eggo_m/e81/run_e81_translation_realignment_control.py`
- `experiments/exp_e12_eggo_m/e81/E81_realignment_control_table.json`
- `experiments/exp_e12_eggo_m/e81/E81_realignment_control_summary.json`
