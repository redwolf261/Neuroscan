# Phase E82 — Local Donor Identity Test: anisotropic, distance-graded, NOT fully interchangeable

## Naming note

The user's message called this "E81," but that name was already used by
the realignment control phase completed immediately prior. Numbered E82
here to keep the project's phase ledger consistent.

## Origin

E81 established that relative encoder-decoder displacement (not global
coordinate shift) is the real failure mode. This phase asks: when
correspondence breaks, what does the decoder actually need — the exact
direction u(p), or would a nearby u(q) work about as well? And is
whatever damage exists isotropic (depends only on distance) or
anisotropic (depends on which specific direction)?

## Method, and a bug caught mid-analysis

Two parts, both no-training inference interventions:
- **Part 1** (directional/radial sweep): reused the established
  whole-volume translation machinery, sweeping small offsets in 6
  axis-aligned directions plus the project's own diagonal reference, plus
  a magnitude sweep along one axis.
- **Part 2** (local donor swap probe): at ~10 well-separated
  lesion-boundary/interior voxels per subject, swap ONLY that single
  voxel's direction for a nearby voxel's direction, magnitude held fixed,
  at radii 1/2/3 voxels.

**A real measurement bug was caught before accepting Part 2's first-pass
result.** The initial script measured swap damage via whole-volume Dice,
which returned near-zero at every radius — but only ~10 voxels out of
262,144 per subject (0.004% of the volume) were perturbed, far too small
a fraction for Dice to register even a complete prediction flip at every
swapped voxel. This was diagnosed numerically (confirmed the voxel
count/fraction directly) before it was reported as a finding, and
corrected: Part 2 was rerun (`run_e82b_local_swap_corrected.py`) measuring
the LOCAL probability change at the swapped voxels themselves.

**A second correction, also caught before accepting the auto-generated
text**: the corrected script's own threshold-based classifier labeled
radius=1's result "near-zero" / "local equivalence" against an arbitrary
0.05 cutoff. Checking the actual radius-to-radius comparison directly
(paired t-tests) shows this is wrong — the trend across radii is real
and highly significant, not flat.

## Results (125 held-out subjects)

### Part 1: anisotropic, magnitude-graded

| Direction | mean S (Dice drop, magnitude 3) |
|---|---|
| +x | 0.192 |
| -x | 0.204 |
| +y | 0.155 |
| -y | 0.149 |
| +z | 0.114 |
| -z | 0.118 |
| diagonal (+3,+3,+3) | 0.213 |

Axis-aligned range (0.090) vs mean (0.155) — **meaningfully
anisotropic**: shifting along z hurts substantially less than shifting
along x, at the same magnitude. Not a uniform distance-decay.

Magnitude sweep (+x axis): S grows monotonically with offset magnitude
(0.075 at mag=1 → 0.239 at mag=5) — damage scales smoothly with distance
along a fixed direction, consistent with earlier translation results.

### Part 2 (corrected): a real, significant, but far from flat local gradient

| Radius | mean LOCAL \|prob change\| at swapped voxel |
|---|---|
| 1 | 0.0274 |
| 2 | 0.0412 |
| 3 | 0.0450 |

All pairwise comparisons significant (r1 vs r2: p=7.3e-12; r1 vs r3:
p=5.1e-12, Wilcoxon p=3.8e-14; r2 vs r3: p=0.020). **Ratio r1/r3 = 0.61**
— the closest possible donor causes 61% as much local disruption as a
donor 3 voxels away, not "near-zero." This is a genuine, statistically
robust gradient, not flat interchangeability.

## Corrected verdict: neither clean local equivalence nor sharp address-identity — a real, moderate, distance-graded penalty

The evidence supports neither of the two clean pre-declared readings.
Even the single nearest neighbor (radius=1) already causes non-trivial
local disruption (0.027, clearly nonzero, p<<0.001 relative to baseline
variation), so this is not the "local equivalence" case where nearby
directions are freely interchangeable. But the gradient from radius 1 to
3 (0.027 → 0.045) is real and monotonic, not the flat "equally bad
everywhere nearby" signature of sharp address-identity either.

**The honest reading**: local donor substitution is graded, not binary.
The decoder tolerates a NEARBY donor better than a FAR donor, but even
the nearest donor is already meaningfully wrong — position specificity
is real and starts biting immediately, but continues to worsen smoothly
with distance rather than being an all-or-nothing address lookup.
Combined with Part 1's anisotropy finding, the fuller picture is: **the
positional penalty has real structure (direction-dependent, distance-
graded) rather than being either a uniform smooth field (fully local
equivalence) or a sharp discrete key-lookup (address-identity only).**

## What this means for intervention design

- A search radius of exactly 1 voxel would not be "free" — it still
  incurs real damage relative to the exact match, so a soft local search
  mechanism cannot expect to fully recover clean performance even at the
  narrowest possible radius.
- But the gradient IS exploitable in principle: a narrow, direction-aware
  local correspondence mechanism (not the isotropic generic kind already
  occupied by DCU/cross-attention) could plausibly recover a meaningful
  fraction of the gap, informed specifically by Part 1's anisotropy
  (weighting search along low-damage axes like z differently from
  high-damage axes like x) — a structural detail generic alignment
  methods are not obviously built around.
- This is a genuinely narrower, more specific finding than either clean
  hypothesis would have given, and (per this project's own standing
  discipline) is reported as the mixed, real result it is rather than
  rounded to whichever pre-declared reading is more convenient.

## Causal chain, updated

```
E65->E79: translation vulnerability, direction-carried, absolute-coordinate-relative binding
E80 (corrected by E81): relative encoder-decoder displacement is the real failure mode
E81: network is near-equivariant under coherent joint shift
E82: the positional penalty is ANISOTROPIC and DISTANCE-GRADED, not
     flat/interchangeable locally and not a sharp all-or-nothing lookup --
     both pre-declared clean hypotheses were too strong; the real
     structure is a moderate, direction-dependent gradient
```

## Artifacts

- `experiments/exp_e12_eggo_m/e82/run_e82_local_donor_identity.py` (Part 1 valid; Part 2 superseded)
- `experiments/exp_e12_eggo_m/e82/run_e82b_local_swap_corrected.py` (Part 2, corrected metric)
- `experiments/exp_e12_eggo_m/e82/E82_directional_sweep_table.json` (Part 1, valid)
- `experiments/exp_e12_eggo_m/e82/E82b_local_swap_table_corrected.json` (Part 2, corrected)
- `experiments/exp_e12_eggo_m/e82/E82b_local_donor_summary_corrected.json`
  (note: this file's own `local_reading` field says "LOCAL EQUIVALENCE" —
  INCORRECT per an arbitrary threshold, superseded by this document's
  corrected reading based on direct pairwise significance tests; kept
  unedited as a faithful record of the script's literal, flawed
  threshold logic)
