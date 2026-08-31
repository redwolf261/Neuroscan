# Phase E65 — Skip-Connection Causal Property Decomposition

## Purpose

E64 confirmed the small-lesion-specific effect discovered in E62/E63 lives entirely
in the decoder's skip connection (`enc1` → `dec1`), not in `MaxPool3d`
(Δ_pool = exactly 0.0 on both checkpoints). This established **where** the
sensitivity lives but not **why**. E65 decomposes "sensitivity to `enc1`'s spatial
arrangement" into four independent, targeted causal properties, using E64's
validated split-forward design (`E_pool` always held at the real value; only the
skip tensor `E_skip` is intervened on, per arm) — so every measured effect is
provably attributable to the skip path alone.

## Method

Four interventions, each applied to the skip tensor only, on the v3/D4-only
(ungated) checkpoint — chosen per E64's finding that the effect is not
gate-dependent and is larger without the gate:

1. **Translation** (absolute correspondence): `torch.roll`-equivalent shift of the
   whole `(32,64,64,64)` `enc1` tensor by a fixed +3 voxels along each spatial
   axis (wrapping). Preserves every voxel's own local neighborhood and the
   tensor's full value distribution exactly — changes only which decoder location
   each feature now lines up with.
2. **Local permutation** (local relational structure): the exact same 2×2×2
   non-overlapping-cell derangement as E62/E64, applied only to the skip tensor —
   this *is* E64's Δ_skip, re-measured here as this phase's own reference point.
3. **Channel permutation** (semantic channel identity): one shared random
   derangement of the 32 channel indices, applied identically at every spatial
   location. Preserves each voxel's own local spatial structure and each
   channel's global marginal statistics; changes which channel index carries
   which feature at each location.
4. **Smoothing** (frequency scale): deterministic 3×3×3 per-channel box average
   (reflection padding), removing high-frequency spatial detail while preserving
   the coarse/low-frequency layout.

Same statistical discipline as E47/E48/E58/E62/E64: subject-level paired t-test +
Wilcoxon + sign-flip permutation test (1000 trials) + bootstrap 95% CI per arm,
plus Spearman(native_size, drop) with its own permutation test per arm. All four
arms are pre-declared as non-mutually-exclusive — the pattern across all four
determines the interpretation, not a single pass/fail gate.

A differential check (reused from E64) verified `pool1`'s output is exactly
invariant and a pool-only-permuted forward pass produces zero output change,
confirming `E_pool` isolation before any arm was measured. Unit tests verified
each intervention's invariant properties (translation/local-permutation preserve
value multisets; channel permutation preserves each channel's own full spatial
map as a set; smoothing reduces variance without changing shape).

## Results (n=125 subjects)

| Arm | Mean Dice drop | Significance | Spearman(size, drop) |
|---|---|---|---|
| **Translation** (3vx roll) | **+0.2143** | p=1.5×10⁻⁵⁴ | ρ=−0.553, p<0.001 |
| **Channel permutation** | **+0.1009** | p=3.4×10⁻⁴⁷ | ρ=−0.571, p<0.001 |
| Smoothing (3×3×3 box) | +0.0284 | p=1.9×10⁻³⁰ | ρ=−0.382, p<0.001 |
| Local permutation (= E64's Δ_skip) | +0.0270 | p=4.4×10⁻³³ | ρ=−0.404, p<0.001 |

**All four arms are significant and size-specific** — nothing is ruled out. But the
magnitudes are strikingly unequal: translation's effect (0.214) is roughly
**8× larger** than local permutation's or smoothing's (~0.027–0.028), and channel
permutation (0.101) is roughly **4×** larger than either of those two.

## Interpretation

- **Absolute spatial correspondence dominates.** A translation that preserves
  every local neighborhood and the entire value distribution exactly — changing
  only which decoder location each `enc1` feature lines up with — is far more
  destructive than scrambling local arrangement or removing high-frequency
  detail. This directly supports the user's candidate 4: the concatenation
  `[upconv1, enc1]` treats spatially-corresponding positions as semantically
  aligned, and that registration is what the decoder is most reliant on.
- **Channel identity is a strong secondary factor.** The decoder also
  substantially depends on which channel index carries which feature at a given
  location, independent of spatial content — a distinct, non-trivial causal
  property in its own right.
- **Local fine-grained arrangement and frequency content are real but minor** by
  comparison, and close to each other in magnitude — consistent with both being
  more modest draws on the same "fine local detail" budget rather than
  independent large effects.
- **All four are size-specific in the same direction** (smaller lesions lose
  more), consistent with every prior finding in this chain (E48, E62's original
  measurement, E64) — this is not a new size-dependence discovery, but it
  confirms the decomposition's arms are all measuring the same underlying
  small-lesion-sensitive phenomenon, at different intensities.

## Implication for future design work

This result narrows the design space considerably from "the skip is sensitive to
arrangement" (E64) to a specific, ranked hypothesis: **positional/registration
correspondence between the encoder skip and the decoder's upsampled path is the
dominant causal factor**, with channel identity secondary and fine spatial detail
tertiary. A future architectural response should prioritize the
registration/correspondence problem at the `[upconv1, enc1]` interface — e.g. a
learned alignment/registration-aware transform before concatenation, or an
explicit positional encoding that makes the correspondence robust rather than
implicit — over a generic "preserve more local information" fix (which E65 now
shows addresses the smaller of the four effects).

This has NOT yet been translated into an algorithm design. Per the project's
established discipline, the next step (if pursued) is a novelty search on
registration-aware / correspondence-preserving skip-connection designs
specifically, informed by this ranked decomposition, before any operator is
proposed.

## Artifacts

- `experiments/exp_e12_eggo_m/e65/run_e65_skip_property_decomposition.py`
- `experiments/exp_e12_eggo_m/e65/E65_skip_decomposition_table.json` (125 subject records)
- `experiments/exp_e12_eggo_m/e65/E65_summary.json`
