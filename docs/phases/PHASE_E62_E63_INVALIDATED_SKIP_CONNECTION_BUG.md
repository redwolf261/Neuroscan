# E62/E63 INVALIDATED — Skip-Connection Confound in `forward_from_enc1`

## Status: both phases' causal attribution is wrong. Numbers were real; the interpretation was not. No new runs launched pending a corrected design.

## The bug

Both `run_e62_maxpool_position_audit.py` and `run_e63_competition_geometry_audit.py`
define:

```python
def forward_from_enc1(model, enc1, device):
    with torch.no_grad():
        pool1 = model.pool1(enc1)
        ...
        skip = enc1              # <-- SAME tensor, reused
        ...
        enc1_gated = enc1 * psi  # <-- SAME tensor, reused
        cat1 = torch.cat([upconv1, enc1_gated], dim=1)
```

`enc1` is a **single argument used in two different roles**: as `pool1`'s input, and
as the decoder's skip connection (via `model.attn_gate1` and the `cat1`
concatenation into `dec1`). Both phases call this function with the *permuted*
tensor for the intervention condition — so the permutation reaches the skip
connection, not just `pool1`.

An inline comment in both files (`"skip = enc1 -- reads the REAL, un-permuted
enc1"`) asserts the opposite of what the code does. This was an unverified claim,
not a checked one — the identity-forward sanity check (E62/E63's "Sanity check
1"/"Unit test 2") only verifies that `ablate=False`/identity-permutation reproduces
`model.forward()`, which is true regardless of this bug (a no-op permutation
changes nothing anywhere), so it could not have caught the error.

## Why this invalidates the causal claim

`nn.MaxPool3d(2)` defaults to stride = kernel size, so pool1 is exactly `k=2, s=2`
— non-overlapping windows. For any within-cell derangement `π` (permuting the 8
values of one 2×2×2 window, preserving the multiset):

```
max(x_1..x_8) = max(π(x_1..x_8))     — provably, by construction
⟹ pool1(enc1) = pool1(π(enc1))        — exactly, every cell, every channel
```

This was independently verified in this session: `pool1`'s output cannot change
under any within-cell derangement. Therefore **100% of E62's measured Dice drop
(mean 0.0226, p≈2×10⁻²⁸) and E63's Test A/B/C effects must have come through the
skip-connection path** (`attn_gate1` + `cat1`), not through any information lost by
`MaxPool3d`. The entire `pool1→pool2→pool3→bottleneck→upconv3→dec3→upconv2→dec2→upconv1`
computation graph is bit-identical between intact and permuted runs in both phases.

## What is and isn't still true

- **The measured Dice effects were numerically real**, not a measurement bug or
  noise — they came from a real, reproducible sensitivity in the trained network.
- **What they measure is different from what was claimed.** E62/E63 tested
  "does the decoder's use of the `enc1` skip connection (via the v5 attention gate)
  depend on the raw subcell spatial arrangement of `enc1`?" — not "does MaxPool3d's
  own information loss matter causally?" Those are different questions with
  different mechanisms (a skip/attention-gate property vs. a pooling property).
- **E62's GO verdict** ("subcell position loss at pool1 is causally task-relevant
  and small-lesion-specific") is not supported by this data — pool1's contribution
  was never actually isolated.
- **E63's KILL verdict** (winner-runner-up geometry doesn't explain E62's effect)
  may still be internally consistent as a statement about the *skip-connection*
  effect (Test A/B/C's relative sizes are still informative about which aspect of
  `enc1`'s arrangement the skip/gate is sensitive to), but its framing throughout
  ("E62-style local derangement," "the local analogue of the E62 intervention")
  inherits E62's mischaracterization of the mechanism as pooling-related.
- **The PMD motivation is unsupported.** The premise "MaxPool discards
  task-relevant subcell position, and the rest of the network can't recover it"
  was never actually tested by either phase. A pooling-operator redesign (PMD or
  otherwise) has no causal audit behind it at this point.

## Correction path (not yet run)

A corrected audit needs `forward_from_enc1` (or an equivalent) to take **two
independent tensors** — one for `pool1`'s input, one for the skip connection — so
each can be permuted in isolation:

1. **Pooling-only test**: permute `enc1` feeding `pool1`; hold the skip's `enc1` at
   the real, un-permuted value. Since `pool1`'s output is provably invariant to any
   within-cell derangement, this test is expected to show **exactly zero effect**
   by construction — it is really a control confirming the fix, not a new
   experiment.
2. **Skip-only test**: hold `pool1`'s input `enc1` at the real value; permute only
   the skip's `enc1`. This isolates the (apparently real) skip-connection
   sensitivity found in E62/E63, as its own distinct finding, correctly attributed
   this time.
3. Re-derive whatever gap-quartile / small-lesion / winner-runner-up analyses are
   still relevant using test 2's isolated skip-connection intervention, since
   that's where the real signal lives.

This has NOT been run. E62 and E63's GO/KILL verdicts should not be cited as
support for any pooling-related design decision until this correction is done.

## Root-cause note for future causal-intervention scripts

This project's causal-audit discipline (E47/E48/E58/E62/E63) relies on a
"sanity check reproduces `model.forward()` exactly" pattern to catch
reimplementation bugs. That check is necessary but *not sufficient* — it only
tests the identity/no-op case, which cannot detect a bug where a single variable
is silently reused for two logically distinct roles (here: an encoder-trunk input
and a skip-connection input) that happen to hold the same value on the identity
path but diverge under intervention. Future dual-role tensor tests should include
an explicit differential check: intervene on ONE role while asserting the other
role's downstream output is unchanged, not just that the fully-intact path
reproduces `forward()`.
