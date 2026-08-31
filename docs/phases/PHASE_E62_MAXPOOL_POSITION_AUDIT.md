# Phase E62 — Max-Pool Subcell Position-Information Causal Audit

> ## ⚠️ INVALIDATED — see [PHASE_E62_E63_INVALIDATED_SKIP_CONNECTION_BUG.md](PHASE_E62_E63_INVALIDATED_SKIP_CONNECTION_BUG.md)
> `forward_from_enc1()` used a single `enc1` tensor for both `pool1`'s input AND the
> decoder skip connection. Since `nn.MaxPool3d(2)` is non-overlapping (k=2,s=2),
> `pool1(enc1) = pool1(π(enc1))` exactly for any within-cell derangement π — proven,
> not assumed. Every bit of this phase's measured Dice effect therefore came through
> the skip/attention-gate path, not through any information MaxPool3d discarded. The
> numbers below are real; the "MaxPool3d discards task-relevant position" causal
> claim and the resulting GO verdict are **not supported**. Do not cite this phase as
> justification for a pooling-operator redesign (PMD or otherwise).

## Purpose

After E61 killed the "small lesions need a higher-dimensional bottleneck"
hypothesis, the project returned to the network's literal mathematics rather than
another architectural guess. The concrete structural fact: `pool1`/`pool2`/`pool3`
are `nn.MaxPool3d(2)` — each collapses 8 values to 1, discarding not just the other
7 values but the **spatial identity of which subcell position held the max**. This
motivated a candidate fix ("Positional-Moment Downsampling", PMD: replace max
pooling with a mass + first-spatial-moment summary of each 2×2×2 cell).

A literature check found PMD's general idea (moment/position-aware pooling)
already occupied by several 2025–2026 papers (higher-order statistical pooling,
Zernike-moment pooling, Spatial Moment Pooling, DSSC-UNet, Subpixel Embedding for
small lesions). It also surfaced a reasoning gap: the network has conv layers and
skip connections *after* pooling that could recover positional information some
other way, so "max pooling loses position" does not by itself prove the **trained
network's predictions** are causally sensitive to that lost position.

E62 tests that causal claim directly, on the real trained network, before any
operator design or training spend — no training, no new architecture, pure
causal-intervention diagnostic in the same discipline as E47/E48/E58.

## Method

- **Checkpoint**: v5/E46 attention-gate checkpoint (same as E48/E61, for direct
  comparability of the size-dependence signature).
- **Intervention site**: `pool1` (64³→32³), the highest-resolution, most
  lesion-relevant pooling stage.
- **Intervention**: for each subject, run the real `enc1` activation as normal.
  Then, independently for every non-overlapping 2×2×2 cell and every channel,
  apply a **random derangement** (permutation with no fixed point) of that cell's
  8 values — verified to preserve the cell's exact multiset (max, mass/sum, mean,
  variance all identical before/after) while guaranteeing every value's subcell
  position changes. Continue the **real** trained forward pass from `pool1`
  onward (pool2, pool3, bottleneck, decoder, skip connections, attention gate —
  all intact and real) with the permuted `enc1` in place of the real one. The
  skip connection to the decoder still reads the real (un-permuted) `enc1`,
  matching the pre-declared scope: only `MaxPool3d`'s own blindness is tested, not
  the skip connection's separate access to `enc1`.
- **8 independent derangement draws per subject** (not one arbitrary draw) to get
  a distribution of ΔDice rather than a single permutation's outcome.
- Sanity checks: (1) the manual forward-from-`enc1` reimplementation reproduces
  `model.forward()` bit-for-bit (max abs diff = 0.0) when given the real
  `enc1`; (2) the derangement is verified to preserve the per-cell multiset and
  guarantee every position changes, on a held-out synthetic check before running
  on real subjects.
- Derangement construction is fully vectorized (no per-cell Python loop): draw a
  random permutation per row via argsort-of-random-keys, then iteratively
  cyclic-roll only the rows with a residual fixed point until none remain
  (converges in practice in ≤2 rounds); final state asserted to be a true
  derangement everywhere before use.

## Pre-declared decision rule (explicit sign-off before running)

GO only if **both**:
1. Mean Dice drop (intact − permuted) significantly > 0 across subjects (paired
   t-test + Wilcoxon + subject-level sign-flip permutation test, 1000 trials).
2. The per-subject drop is significantly correlated with **smaller** native lesion
   size (Spearman ρ < 0, permutation p < 0.05, 1000 trials) — matching E48's own
   significance bar, not just "any" size relationship.

A significant but size-agnostic drop would be reported honestly as
`SIGNIFICANT_BUT_SIZE_AGNOSTIC` and not meet the bar for the small-lesion-focused
PMD chain.

## Results

- n = 125 subjects, 8 derangement draws each.
- Mean Dice intact = 0.8922, mean Dice permuted = 0.8696.
- **Mean drop = 0.0226** (t-test p=2.0×10⁻²⁸, Wilcoxon p=8.3×10⁻²⁰, sign-flip
  permutation p<0.001).
- **Spearman(native_size, drop) = −0.355** (parametric p=4.95×10⁻⁵, permutation
  p<0.001) — smaller lesions lose significantly more Dice from subcell position
  scrambling, same direction and comparable strength to E48's ρ=−0.454.

## Decision: **GO**

Both pre-declared criteria met. The trained network's downstream layers do **not**
fully absorb the subcell positional information `MaxPool3d` discards at the
earliest downsampling stage — the position was doing real causal work, and losing
it disproportionately hurts small lesions specifically.

## Significance for the project

This is the first mechanism in the E43→E62 chain to survive a pre-declared causal
bar that directly addresses *why* small lesions depend more on the coarse pathway
(E48's original finding). Prior candidate explanations were tested and killed or
returned null:
- E47: boundary-routing gate — null.
- E58: fixed spatial octant localization — null (effect not concentrated).
- E59/E60: octant-pair synergy — formulation did not survive reproduction.
- E61: bottleneck effective dimensionality — killed, reversed direction.

E62 instead locates the mechanism at the **earliest** downsampling stage
(pool1, not the bottleneck) and finds it causally real and size-specific. This
motivates continuing to Positional-Moment Downsampling (PMD) design — the next
step is deriving the moment representation carefully, verifying its coordinate
behavior on synthetic inputs, and a dedicated prior-art search on the *specific*
combination (mass + first-order spatial moments → learned projection, inside a 3D
medical-segmentation encoder) before any training spend, per the project's
established discipline.

## Artifacts

- `experiments/exp_e12_eggo_m/e62/run_e62_maxpool_position_audit.py`
- `experiments/exp_e12_eggo_m/e62/E62_position_audit_table.json` (125 subject records, 8 draws each)
- `experiments/exp_e12_eggo_m/e62/E62_summary.json`
