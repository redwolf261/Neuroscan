# Phase E80 — Decoder-Readout Causality: ⚠️ Arm 3 interpretation RETRACTED by E81, see below

## RETRACTION NOTICE (added after E81)

E81 (`PHASE_E81_REALIGNMENT_CONTROL_RESULT.md`) found that E80's Arm 3
("coherent joint shift") result below is confounded: coherently shifting
both sides by the same offset causes the network's OUTPUT to also shift
by approximately the same offset (near-perfect translation-equivariance,
verified directly: relative equivariance error 0.096). Undoing that
shift on the output before scoring recovers 93.5% of Arm 3's apparent
damage. **This retracts this document's "both sides independently,
additively position-bound" conclusion** — Arm 3's large damage was
substantially a trivial coordinate-frame mismatch (comparing a shifted
prediction to a fixed, unshifted label), not evidence of a deeper
representational failure.

**What survives**: Arms 1 and 2 (single-side shifts) are NOT subject to
this confound — an asymmetric shift (only one side of the concatenation
moved) does not give the network any way to "compensate" by producing a
globally shifted output, since the other input is still in its correct
frame. Arm 1 (encoder-only, S=0.213) reproduces E65/E78/E79's own
established result and stands. Arm 2 (decoder-only, S=0.262) is a new,
still-standing finding: the decoder's own upconv1 pathway is ALSO
genuinely sensitive to translation, independent of any coordinate-frame
artifact (since only one side moved, there is no compensating global
output shift available) — this part of E80 is NOT retracted and remains
informative. Only the joint-shift (Arm 3) interpretation and the
mismatch-sweep's shape (which was driven by the same confound) are
retracted.

The rest of this document is kept unedited below as a faithful record of
what was concluded before E81's correction.

---


## Origin

E79 confirmed enc1 direction's translation vulnerability is absolute-
coordinate binding. This left open whether that binding is (i) an
encoder representation property (enc1's direction alone) or (ii) an
encoder-decoder correspondence property (relative alignment between
enc1 and the decoder's own upconv1 feature at concatenation) — a
distinction with opposite intervention implications, following a
targeted novelty search that ruled out generic deformable/alignment
mechanisms as the eventual algorithm.

## Method

No training. Split-forward isolation (E_pool always real, matching
E64/E74/E75/E78/E79). Four arms, all shifting by the project's own
reference 3-voxel offset:
- Arm 1: translate enc1's direction only, upconv1 untouched (= E78/E79's S_u).
- Arm 2: translate upconv1 only (the decoder's own pre-concat feature), enc1 direction untouched.
- Arm 3: translate BOTH by the SAME offset (coherent shift, preserving relative alignment).
- Arm 4: mismatch sweep — enc1 direction fixed at offset 3, upconv1 offset swept {0,1,2,3,4,5}.

## Results (125 held-out subjects)

| Arm | Mean S | 95% CI |
|---|---|---|
| Arm 1 (encoder-only) | 0.2134 | [0.199, 0.229] |
| Arm 2 (decoder-only) | **0.2615** | [0.247, 0.277] |
| Arm 3 (coherent joint shift) | **0.4386** | [0.422, 0.457] |

All highly significant (p<1e-54 throughout). **Arm 3 is WORSE than Arm
1, not better** — coherent shift does NOT recover the damage; it nearly
doubles it. Recovery fraction is **-105.5%** (Arm1 vs Arm3 paired t
p=3.3e-62).

### Mismatch sweep: monotonic in upconv1's own displacement, does NOT trough at zero relative mismatch

| upconv1 offset | relative mismatch \|3-offset\| | mean S |
|---|---|---|
| 0 | 3 | 0.213 |
| 1 | 2 | 0.280 |
| 2 | 1 | 0.368 |
| 3 | 0 (coherent) | 0.439 |
| 4 | 1 | 0.501 |
| 5 | 2 | 0.575 |

Damage increases monotonically with upconv1's OWN absolute offset,
regardless of whether that offset brings it closer to or further from
matching enc1's direction. It does NOT trough at offset=3 (zero relative
mismatch) as hypothesis (ii) predicts — that point is worse than offset=0
or offset=1.

## Verdict: neither pre-declared hypothesis fits; correcting an auto-generated misread

**Important note on this phase's own script**: its automatic reading
function mechanically selected the "(i) encoder representation property"
text because Arm 3's recovery fraction failed the "> 0.5" threshold for
hypothesis (ii) — but the code had no branch for a NEGATIVE recovery
fraction (Arm 3 worse than Arm 1), so it fell through to (i) by default.
That auto-generated conclusion is WRONG and is corrected here rather
than reported as-is, per this project's standing discipline of not
letting a script's literal output substitute for actually reading the
numbers.

**What the data actually show**: hypothesis (i) is also contradicted —
under a pure "encoder representation" story, shifting upconv1 ALONE
(Arm 2, encoder untouched) should barely matter, since the positional
"key" would live entirely in enc1. Instead Arm 2's damage (0.262) is
comparable to, even slightly larger than, Arm 1's (0.213).

**The correct reading**: both the encoder-side direction AND the
decoder's own upconv1 feature are INDEPENDENTLY position-bound, and
their damage combines roughly additively (Arm1 + Arm2 = 0.475, close to
Arm 3's 0.439) or worse — not in a correspondence-restoring way. Moving
either tensor away from its trained spatial address hurts, and moving
both away hurts roughly as much as the sum, regardless of whether they
end up newly "matched" to each other. This rules out a pure relative-
correspondence story: **the decoder is not simply comparing "does enc1's
direction agree with upconv1's feature at this address" — each tensor
individually needs to be at ITS OWN specific trained address, not just
consistent with its partner.**

## What this means for intervention design

This is a stronger, more specific, and less convenient finding than
either pre-declared hypothesis. It rules out:
- **Correspondence/alignment mechanisms** (deformable alignment,
  cross-attention re-matching) as a plausible fix — already flagged as
  occupied by the novelty search, and now also shown not to match the
  actual failure mode. Realigning the two sides to agree with each
  other would not help, since disagreement isn't the mechanism — each
  side's own absolute position is independently load-bearing.
- **A pure encoder-side fix** (targeting enc1's direction alone) as
  sufficient — the decoder's own upconv1 pathway carries an
  independent, comparable-magnitude positional dependency that an
  enc1-only intervention would not touch.

This points toward something more structural: **both the encoder
skip path and the decoder's own upsampling path have independently
learned to rely on exact spatial addressing**, likely because the
network was never trained under any positional perturbation and so had
no pressure to develop tolerance on either side. This reframes the
likely intervention away from a single-junction fix and toward either
(a) training-time exposure to small positional perturbations
(consistency/robustness training — noted as a heavily populated area
requiring a specific formulation, not a novel fix on its own), or (b) a
broader architectural change to how both paths encode position, not
just the skip connection.

## Causal chain, complete

```
E65: spatial correspondence at the enc1 skip matters
E73: causal vulnerability has spatial structure
E74: feature magnitude correlates with translation sensitivity
E75: magnitude CAUSALLY affects sensitivity at enc1 (not bottleneck)
E76: the causal relationship holds locally, at the real operating point
E77: high magnitude = lesion salience + vulnerability -- suppression unsafe
E78: direction carries positional structure; magnitude amplifies (super-additive)
E79: direction's vulnerability is absolute-coordinate binding, not local noise
Novelty search: generic deformable/alignment mechanisms are OCCUPIED
E80: BOTH encoder (enc1 direction) and decoder (upconv1) paths are
     INDEPENDENTLY position-bound -- not a correspondence/alignment
     problem at all; coherent joint shift makes it WORSE, not better
```

Eight phases of pure measurement, zero training, have progressively
ruled out every mechanism this project generated by intuition
(gating, reweighting, smoothing, alignment/correspondence) and arrived
at a genuinely surprising structural finding: the vulnerability is not
localized to one junction or one representation, and is not a matching
problem between two sides — it is a general lack of positional tolerance
learned independently on both sides of the skip connection.

## Artifacts

- `experiments/exp_e12_eggo_m/e80/run_e80_decoder_readout_causality.py`
- `experiments/exp_e12_eggo_m/e80/E80_decoder_readout_table.json`
- `experiments/exp_e12_eggo_m/e80/E80_decoder_readout_summary.json`
  (note: this file's own `reading` field states the "(i) encoder
  representation" conclusion — INCORRECT, superseded by this document's
  corrected reading; kept unedited as a faithful record of the script's
  literal, flawed threshold logic)
