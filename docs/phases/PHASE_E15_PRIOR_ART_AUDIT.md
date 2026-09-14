# E15 Targeted Prior-Art Audit

**Date**: 2026-09-13
**Status**: **Retired before compute**; no training or new experiment run.

## Current verdict

The prior-art search did not identify an obvious paper stating the exact closed loop:

```text
class/representation geometry -> decoder response -> select useful latent direction -> optimize representation
```

That provisional gap is no longer actionable because the proposed experiment does not isolate the claimed interaction. The E15 decoder-calibration branch is **closed as a primary algorithm direction**.

## Claims that are already occupied

The following are not sufficient novelty claims by themselves:

- task-relevant or task-aware latent representation learning;
- decoder-aware representation learning;
- decoder Jacobian or decoder sensitivity analysis;
- gradient/Jacobian-based identification of task-relevant latent dimensions;
- prototype, centroid, margin, or metric-learning geometry;
- generic Jacobian-guided representation optimization.

## Narrow candidate principle

> A representation direction is useful only when it is both geometrically discriminative and useful according to the downstream decoder's local response.

Working name: **decoder-calibrated representation geometry**.

This is a hypothesis about the interaction between two geometries, not a novelty claim:

```text
representation geometry intersect decoder-response geometry
```

E15 changed only `dec1`, kept the decoder frozen, and improved Dice from `0.9050` to `0.9443` at the realistic push magnitude of 14 units. Because the opposite centroid is selected using ground-truth class information, the safe claim is: **supervised latent separation can causally improve a frozen decoder's output**. It does not identify a deployable direction-discovery rule.

## Critical caveat

The 3.83x Jacobian ratio in E15 establishes that the decoder is locally more sensitive along its own weight direction than along E15's Euclidean intervention direction. It does **not** establish that Jacobian-guided optimization, decoder-aware learning, or any proposed combination is novel.

Likewise, neither of these is sufficient as a proposed method:

```text
d = grad_z D(z)
d = z - mu_opp
d = grad_z D(z) + (z - mu_opp)
```

The last expression is only an obvious vector combination unless a specific, justified direction-selection rule gives it a distinct computational meaning.

## Retired offline test

The previously proposed geometry/Jacobian comparison is retired before execution.

Compare equal-norm perturbations at matched magnitudes for:

- `d_G`: normalized Euclidean class direction, away from the opposite-class centroid;
- `d_J`: normalized decoder-response direction, derived from the local decoder derivative;
- `d_R`: normalized random direction;
- `d_GJ`: a precisely defined geometry-constrained decoder-response direction.

Measure per-volume and aggregate `Delta Dice` at the same perturbation magnitudes.

The decisive comparison is:

```text
d_GJ > d_G and d_GJ > d_J
```

If `d_GJ` adds no benefit over `d_G`, the interaction principle has no demonstrated leverage and this branch should be killed. If `d_J` alone explains the effect, the result is more naturally framed as a sensitivity method and faces stronger prior-art exposure. Only a distinct, reproducible interaction effect justifies a second novelty audit.

## Decision

No compute is authorized for the retired test. The E15 branch should not be extended into a decoder-Jacobian loss or a geometry-plus-Jacobian method.

## Candidate algorithm triage

The corrected single-seed per-subject rescoring identified three FLAIR-only candidates crossing +1 pp:

| Candidate | Single-seed per-subject result | Statistical status | 3-seed status | Novelty direction |
|---|---:|---|---|---|
| E45 D4+D8 | +1.44 pp | paired t and Wilcoxon significant | 0.8939 mean; not a defensible +1 pp confirmation | Deep supervision is established; mechanism explanation may be the only possible angle |
| E54 A96 | +1.39 pp | paired t and Wilcoxon significant | 0.8944 mean; threshold margin is indistinguishable from noise | Resolution increase is not novel by itself |
| E46 attention gate | +1.08 pp | Wilcoxon only; paired t not significant | no equivalent confirmed 3-seed +1 pp result | Attention/U-Net gating is heavily occupied |

The reproducible multimodal result is separate: MM versus FLAIR is approximately +1.75 pp across three seeds, but it changes the input modality and is not a novel algorithmic mechanism.

The next research task is therefore an existing-result novelty audit, beginning with E45, E54, and E46, rather than another E15 experiment or a new decoder-Jacobian method.

Controls below belong only to the retired test and were not run:

- equal perturbation norm;
- realistic latent-distance range;
- random directions;
- shuffled class identities or centroid assignments;
- shuffled decoder coordinates or a frozen random head;
- decoder response destroyed or randomized;
- off-manifold behavior.

These controls are part of the future test design, not results.

## Research state transition

```text
E15 causal discovery
  -> potential interaction principle
  -> offline geometry/decoder direction test
  -> prior-art recheck if interaction survives
  -> only then define a trainable method
```

**Current rule**: no training, no method claim, and no novelty claim until the offline direction-selection rule is mathematically specified and survives the falsification controls.

## Related records

- [E15 evidence freeze](PHASE_E15_EVIDENCE_FREEZE.md)
- [E15 code](../../experiments/exp_e12_eggo_m/e15_decoder_sensitivity.py)
- [E15 results](../../experiments/exp_e12_eggo_m/e15_decoder_sensitivity_results/summary.json)
- [Full E15 report](PHASE_E15_DECODER_SENSITIVITY.md)
