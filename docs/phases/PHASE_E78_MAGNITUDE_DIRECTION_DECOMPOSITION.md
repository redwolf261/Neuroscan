# Phase E78 — Magnitude-Direction Causal Decomposition: DIRECTION DOMINATES

## Origin

E77 established that high enc1 magnitude is simultaneously lesion
salience and translation vulnerability, ruling out magnitude suppression
as an intervention. This left open whether the positional information
translation disrupts is actually carried by the magnitude channel r(p),
the direction channel u(p) = z(p)/||z(p)||, or their interaction —
E75/E76 only established that RESCALING magnitude (direction held fixed)
changes sensitivity, not that magnitude itself is where position lives.

## Method

No training. z(p) = r(p)·u(p) decomposed at every enc1 voxel, for both
the intact and translated (E65's own 3-voxel roll) tensors. Two sanity
checks verified before any subject processed: decomposition/recomposition
is exact (reconstruction diff 2.4e-07, float noise) and translation
commutes exactly with the r/u decomposition (0.0 diff both components).

Three controlled hybrid tensors, isolating each component:
- z_full = r_translated · u_translated (E65's own full translation)
- z_r_only = r_translated · u_intact (ONLY magnitude translated)
- z_u_only = r_intact · u_translated (ONLY direction translated)

## Results (125 held-out subjects)

| Quantity | Mean Dice drop | Share of S_full |
|---|---|---|
| S_full (both translated) | 0.3174 (CI [0.300, 0.336]) | 100% |
| **S_r** (magnitude-only) | **0.0223** (CI [0.018, 0.028]) | ~7% |
| **S_u** (direction-only) | **0.2134** (CI [0.199, 0.229]) | ~67% |

All three highly significant (p<1e-14 throughout). **S_u is ~9.6x larger
than S_r** (paired t p=2.2e-55, Wilcoxon p=3.0e-22) — direction-only
translation reproduces the large majority of the full effect; magnitude-
only translation barely moves Dice.

**Additivity check**: S_r + S_u = 0.236, vs S_full = 0.317 — a real,
positive gap of +0.082 (super-additive interaction, not simple
additivity). Translating both together hurts more than the sum of
translating each alone.

## Interpretation

**Direction carries the positional structure that translation disrupts.
Magnitude is not an independent carrier of position — its earlier-
established causal role (E75/E76: rescaling alpha changes sensitivity)
is better read as an AMPLIFIER of direction's vulnerability, not a
separate source of it.** The super-additive interaction is consistent
with this amplification reading: when magnitude and direction both shift
together (as in a real translation), the large magnitude at
lesion/high-salience voxels multiplies the damage from direction's own
positional disruption — more than either component does alone.

This directly resolves the ambiguity E75/E76 left open, and combined
with E77's finding (magnitude suppression is dangerous — it would
destroy lesion salience), points toward a specific, evidence-derived
intervention shape:

> **Preserve magnitude (and the lesion salience it encodes). Regularize
> or make more translation-robust the DIRECTION component specifically
> — the channel actually carrying the positional information that
> breaks under a small spatial shift.**

## What this rules out / motivates

- Rules out: any mechanism that touches r(p) directly (magnitude
  normalization, clipping, tail suppression) as the primary lever —
  already excluded by E77, now doubly justified since direction, not
  magnitude, is where the actual vulnerability lives.
- Motivates: a mechanism operating on u(p) — e.g. encouraging the
  direction component to vary more smoothly across small spatial
  shifts (a form of positional smoothness / local Lipschitz constraint
  specifically on the unit-direction field, not the full feature
  vector), while leaving r(p) free to remain sharp and lesion-tracking.
- The super-additive interaction suggests any fix should be evaluated
  with BOTH components present (not tested on direction alone in
  isolation) since the real-world failure mode involves both shifting
  together.

## Causal chain, complete

```
E65: spatial correspondence at the enc1 skip matters (translation >> local perm/smoothing)
E73: causal vulnerability has spatial structure, correlates with the network's own errors
E74: feature magnitude correlates with translation sensitivity, at multiple depths
E75: magnitude CAUSALLY affects sensitivity at enc1 (not bottleneck -- killed there)
E76: the causal relationship holds locally, at the network's real operating point
E77: high magnitude is simultaneously lesion salience AND vulnerability -- suppression is unsafe
E78: DIRECTION, not magnitude, carries the actual positional structure; magnitude amplifies
```

Six phases of measurement, no training, converging on a specific,
falsifiable, evidence-derived target for the first time tonight: not
"predict fragility and gate/reweight by it" (E70-E73, all killed), but
"the direction component of high-magnitude features is where positional
fragility lives — make it more robust without touching magnitude."

## Artifacts

- `experiments/exp_e12_eggo_m/e78/run_e78_magnitude_direction_decomposition.py`
- `experiments/exp_e12_eggo_m/e78/E78_decomposition_table.json`
- `experiments/exp_e12_eggo_m/e78/E78_decomposition_summary.json`
