# Phase E79 — Directional Positional-Dependence Audit: HYPOTHESIS B CONFIRMED (absolute coordinate binding)

## Origin

E78 established direction u(p) = z(p)/||z(p)|| at enc1 carries the large
majority of translation vulnerability (S_u ~9.6x S_r), with magnitude
acting as an amplifier. This left open WHY direction is vulnerable: is
it locally noisy/oversharp (hypothesis A, fixable by smoothing), or
bound to absolute spatial coordinates independent of local structure
(hypothesis B, NOT fixable by smoothing)? These have opposite
intervention implications.

## Method

No training. Magnitude r(p) held exactly fixed throughout every arm
(per E77's finding that magnitude must be preserved — it carries lesion
salience).

1. **Descriptive**: local angular coherence cos(theta(p, p+delta)) for
   6-connected neighbors, by class.
2. **Local direction smoothing**: 3x3x3 box-average direction,
   renormalized to unit length, magnitude untouched. Measures the pure
   smoothing effect (no translation) AND, critically, whether
   translating the ALREADY-SMOOTHED direction still hurts as much as
   translating the raw direction (E78's S_u).
3. **Local direction permutation**: E62/E65's own 2x2x2 non-overlapping-
   cell derangement, applied to direction only — each voxel keeps its
   OWN magnitude at its OWN original location; only direction identity
   is scrambled within its cell. This preserves rough position while
   destroying local arrangement — the mirror image of translation, which
   preserves local arrangement while destroying position.

Decisive comparison: S_dir_local_perm vs E78's S_u (translation).

## Results (125 held-out subjects)

### (1) Direction is already locally smooth

| Class | mean cos(theta) |
|---|---|
| background | 0.9545 |
| boundary | 0.8773 (lowest, still high) |
| interior | 0.9366 |

High coherence everywhere — there isn't much local angular noise for a
smoothing intervention to remove in the first place.

### (2) Smoothing barely recovers translation-robustness

| Quantity | Mean |
|---|---|
| S_smooth (pure smoothing, no translation) | 0.0265 |
| S_u_after_smooth (translate the smoothed direction) | **0.1899** |
| E78's S_u (translate raw direction) | 0.2134 |
| **Recovery fraction** | **11.0%** |

Pre-smoothing direction before translating it recovers only 11% of the
damage — the large majority survives smoothing intact.

### (3) Local permutation is far less damaging than translation — the decisive result

| Quantity | Mean |
|---|---|
| S_dir_local_perm (scramble locally, position preserved) | **0.0372** |
| E78's S_u (translate, position lost) | 0.2134 |
| **Ratio (local-perm / translate)** | **0.174** |

Scrambling direction within tiny 2x2x2 cells — completely destroying
local neighborhood arrangement — causes only 17.4% of the damage that
moving the same directions 3 voxels away causes.

## Verdict: Hypothesis B confirmed

Both independent lines of evidence converge on the same conclusion:

- Direction is not locally noisy to begin with (high coherence).
- Smoothing it barely helps against translation (11% recovery).
- Scrambling local arrangement while preserving rough position is far
  safer than preserving arrangement while losing position (17.4% vs
  100% relative damage).

**The decoder uses enc1's direction component essentially as a
positional lookup key bound to absolute spatial coordinates — not by
reading local directional texture or gradient structure.** Moving a
direction vector to a different address breaks its usefulness almost
regardless of how locally coherent its neighborhood was; scrambling its
neighbors while keeping it roughly in place barely matters.

## What this rules out / motivates

- **Rules out**: any local-smoothing-based intervention (directional
  Lipschitz/smoothness regularization) — E79 shows directly that this
  would not meaningfully fix the actual vulnerability, only remove 11%
  of it, while risking exactly the boundary-blurring damage the user
  flagged as a concern before this test even ran.
- **Motivates**: the intervention needs to address absolute positional
  binding, not local smoothness. This points toward mechanisms in the
  registration/correspondence family — e.g. making the decoder's use of
  direction more tolerant of small coordinate shifts (a learned local
  search/alignment at the skip junction, or an explicit small-offset
  invariance built into how direction is read at concatenation) — closer
  in spirit to E68's original SC-DCU correspondence-correction idea
  (retracted then for an unrelated design flaw) than to any
  gating/reweighting/smoothing mechanism tried in E70-E78.

## Causal chain, complete

```
E65: spatial correspondence at the enc1 skip matters (translation >> local perm/smoothing)
E73: causal vulnerability has spatial structure
E74: feature magnitude correlates with translation sensitivity
E75: magnitude CAUSALLY affects sensitivity at enc1 (not bottleneck)
E76: the causal relationship holds locally, at the real operating point
E77: high magnitude = lesion salience + vulnerability -- suppression unsafe
E78: DIRECTION carries the positional structure; magnitude amplifies (super-additive)
E79: direction's vulnerability is ABSOLUTE-COORDINATE binding, not local noise --
     smoothing/local-regularization approaches are ruled out
```

Seven phases of pure measurement (E65, E73-E79), zero training, have now
converged on a specific, falsifiable, narrowly-scoped target: **the
decoder needs positional tolerance in how it reads enc1's direction
component at the skip junction — not a smoother representation, not a
reweighted one, but one that survives a small coordinate shift.**

Per the user's stated plan: the novelty search comes next, targeted
specifically at this finding, before any training code is written.

## Artifacts

- `experiments/exp_e12_eggo_m/e79/run_e79_directional_positional_audit.py`
- `experiments/exp_e12_eggo_m/e79/E79_directional_audit_table.json`
- `experiments/exp_e12_eggo_m/e79/E79_directional_audit_summary.json`
