# Phase E73 — Self-Diagnostic Localized Refinement (SDLR): probe PASSED, predictor PASSED (weak), refinement pilot KILLED

## Origin

User's reframe after E70-E72's mechanism kills: stop trying to *act on* the
scalar fragility signal (routing, reweighting — both killed for the same
reason, correlated difficulty markers, not separable resources). Instead
ask whether the bottleneck can predict **where**, spatially, it is causally
vulnerable — not just how much. Proposed ladder:

```
E48 -> CDCG-1 -> E71/E72 (can't act on the scalar) -> NEXT: localize it
```

## Stage 1 — cheap probe (no training): PASS

`check_spatial_sensitivity_probe.py`, 40 val subjects, real causal
sensitivity map $S_i(p) = |P_{intact}(p) - P_{ablated}(p)|$ (voxel-wise,
same bottleneck-ablation construction as E48/CDCG-1):

- **Not the lesion mask**: 9x enrichment inside lesion tissue, but 90%+ of
  S's mass sits outside it — a distinct spatial pattern.
- **Correlates with the network's own errors far above chance**: 22.6x
  ratio (mean S at error voxels vs correct voxels), p<1e-9 across 40
  subjects.
- **Survives controlling for boundary-distance confound**: partial
  Spearman(S, error | boundary_distance) = +0.88, still highly
  significant — not simply "errors and S both live near edges."

Verdict: PROCEED to training a predictor, per pre-declared rule.

## Stage 2 — novelty check (before writing code)

Search found the MAP construction itself (model-space ablation → voxel
difference) is an occupied post-hoc explanation technique (Discriminative
Attribution from Counterfactuals, Counterfactual-based Saliency Maps,
Contrast-CAM family). **Not found**: training a head to predict that map
from frozen features as a self-supervised **training-time distillation**
signal — substituting a cheap single forward pass for the expensive
second (ablated) forward pass at inference. That distillation step is
where this phase's novelty claim rests, narrowly.

## Stage 3 — spatial predictor training: PASS, but weak downstream

Small conv head $\hat{S} = H(Z_i)$, bottleneck (256ch, 8³) → predicted
sensitivity map (8³), trained on frozen MM-seed0 bottleneck features.

| Run | pooled ρ (held-out) | per-subject mean ρ | per-subject std |
|---|---|---|---|
| 250 labels, 60 epochs | +0.394 (p≈0) | +0.376 | 0.189 |
| 1126 labels, 150 epochs (scaled re-test) | +0.414 (p≈0) | +0.410 | **0.072** (tighter) |

Both pass the pre-declared design-time bar (pooled + per-subject
correlation, 125/125 subjects valid). Scale-up improved consistency but
not magnitude much.

**Downstream check — does the predicted map retain error-localizing power?**

| Quantity | Real map | 250-label predictor | 1126-label predictor |
|---|---|---|---|
| error/correct ratio | 22.6x | 1.07x | 1.19x |
| partial ρ (boundary-controlled) | +0.88 | +0.09 (p=1.2e-8) | +0.11 (p=4.1e-7) |

Both predictor versions pass their own significance bar, but the scale-up
did **not** meaningfully close the gap to the real map — unlike E71
prediction-2, where scaling flipped a wrong sign and doubled the range,
here scaling only nudged 1.07x→1.19x. This looks like a genuine ceiling
on how much of the real map's error-localizing power survives distillation
into a lightweight bottleneck-only predictor, not a data-scale artifact.

## Stage 4 — refinement pilot: KILLED

Per the user's explicit choice to let Dice decide rather than reason
further about a weak-but-real signal. Built `UNet3D_v11` (neuroscan_3d_v11.py):
v3 + a single spatial gate at dec1, driven by the FROZEN 1126-label
predictor. Verified bit-identical to v3 at init (max abs diff 0.0,
zero-initialized residual conv). NOT pathway routing (E71's mistake) and
NOT subject-level reweighting (E72's mistake) — a within-pathway,
per-voxel refinement, deliberately structured to avoid both previously
diagnosed failure modes.

1 seed, 12 epochs, matched exactly against E72's own baseline pilot
(MM_baseline_pilot_seed0, per_subj Dice 0.8934):

| Run | per-subject Dice (ep 12) |
|---|---|
| MM baseline (unweighted) | 0.8934 |
| SDLR (spatial gate) | 0.8939 |

**Mean delta: +0.0005, not significant** (paired t p=0.78, Wilcoxon
p=0.91). 50.4% of subjects improved — indistinguishable from chance.

**Mechanism-consistency check**: does the delta correlate with baseline
difficulty (lower baseline Dice = harder subject), as the gate was meant
to provide? **No** — Spearman(difficulty, delta) = −0.056, not significant
(p=0.53). The gate doesn't even preferentially help hard subjects; it is
essentially inert.

## Decision

**KILLED at one pilot.** Consistent with the pre-registered expectation
stated before running it: a signal this weak (error ratio ~1.2x) was a
low-probability bet, tested cheaply and quickly rather than debated
further, and the result is unambiguous — no effect in either direction,
no mechanism-consistent targeting. Per the no-rescue discipline, no
further weighting/architecture variants of this refinement gate will be
attempted.

## What survives

- The E73 probe's finding stands on its own: the network's causal
  vulnerability has real, non-trivial spatial structure that correlates
  with its own errors — a genuine, if so-far unexploited, fact about this
  network's failure modes.
- The predictor (1126-label version) is a real, validated, if weak,
  distillation of that structure — cheaper at inference than the real
  ablation, but too weak on its own to drive a useful refinement signal
  through the mechanism tried here.
- This is now the **third** distinct mechanistic verb (pathway routing,
  loss reweighting, spatial refinement gating) applied to variants of the
  bottleneck's causal self-knowledge, and the third to fail — each for a
  specific, diagnosed reason rather than an unexplained null.

## Artifacts

- `experiments/exp_e12_eggo_m/e73/check_spatial_sensitivity_probe.py` (stage 1, PASS)
- `experiments/exp_e12_eggo_m/e73/check_confound_boundary.py` (boundary confound check, survives)
- `experiments/exp_e12_eggo_m/e73/train_spatial_sensitivity_predictor.py` (250-label)
- `experiments/exp_e12_eggo_m/e73/train_spatial_sensitivity_predictor_scaled.py` (1126-label)
- `experiments/exp_e12_eggo_m/e73/check_predicted_map_vs_error.py` / `_scaled.py`
- `neuroscan_3d_v11.py` (SDLR architecture, verified init-identity to v3)
- `experiments/exp_e12_eggo_m/e73/train_e73_sdlr_pilot.py`
- `experiments/exp_e12_eggo_m/e73/check_sdlr_vs_baseline.py`
- `runs/SDLR_seed0/` (checkpoints, per-epoch metrics, per-subject result.json)
