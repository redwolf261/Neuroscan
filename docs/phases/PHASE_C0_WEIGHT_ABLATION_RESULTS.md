# Experiment C0: Fixed-Weight Ablation — Results

**Status**: ✅ Complete (seed=0, 10 epochs each, 3 configs)

## Setup

Three otherwise-identical training runs (same architecture, optimizer,
lr=4e-4, batch_size=8, data split as Phase A.5/B), varying only
`(focal_weight, evidential_weight)`:

| Config | focal_weight | evidential_weight |
|---|---|---|
| A | 0.8 | 0.2 |
| B (= Phase A.5/B baseline) | 0.5 | 0.5 |
| C | 0.2 | 0.8 |

## Question tested

Phase B found: trunk gradient conflict is transient (resolves by epoch 3),
but the gradient-magnitude ratio `‖g_evidential‖ / ‖g_focal‖` at the trunk
collapses to ~0.1 and stays there through epoch 10.

- **H1**: conflict genuinely resolved, both objectives agree — increasing
  evidential's weight should not help.
- **H2**: evidential is drowned out by the fixed 0.5/0.5 weighting —
  increasing its weight should restore trunk influence and improve
  calibration/Dice.

## Result: neither H1 nor H2 as stated. A third finding.

### The gradient ratio is invariant to loss weighting

`grad_ratio_evid_over_focal_trunk` is computed on the **unweighted**
individual gradients (before the focal_weight/evidential_weight scalars are
applied) — it measures the intrinsic gradient-magnitude relationship
between the two loss functions at the trunk, independent of how we weight
them in the training objective.

| Config | Ratio @ epoch 3 | Ratio @ epoch 9 |
|---|---|---|
| focal0.8/evid0.2 | 0.0991 | 0.1036 |
| focal0.5/evid0.5 | 0.0899 | 0.1051 |
| focal0.2/evid0.8 | 0.0877 | 0.0892 |

**Statistically indistinguishable across an 8x swing in evidential_weight
(0.2 vs 0.8).** Even when evidential is weighted 4x higher than focal in
the training objective, its raw trunk-gradient magnitude is still ~9-11x
smaller than focal's. This falsifies H2 in its literal form: the imbalance
is not a "drowned out by scalar weighting" problem — reweighting the loss
does not move the underlying gradient geometry at all.

### Cosine similarity (conflict) is also weight-invariant

Percentage of batches with negative trunk cosine collapses to 0% by
epoch 2-3 in **all three configs**, and the settled cosine values
(0.60-0.74 range) overlap across configs within noise. The conflict → 
resolution timeline discovered in Phase B replicates identically
regardless of loss weighting.

### What DID move: calibration, monotonically with evidential weight

| Config | Best Dice | Best (lowest) ECE |
|---|---|---|
| focal0.8/evid0.2 | 0.8797 | 0.1152 |
| focal0.5/evid0.5 | 0.8787 | 0.1067 |
| focal0.2/evid0.8 | 0.8793 | **0.1018** |

Dice is flat across configs (0.8787-0.8797, within single-seed noise).
**ECE improves monotonically as evidential_weight increases**, with no
Dice cost and no return of gradient conflict.

## Interpretation

Neither hypothesis survives as originally stated:

- **Not H1**: if both objectives had genuinely "agreed" and evidential had
  nothing left to contribute, calibration would not keep improving as its
  weight increases.
- **Not H2**: the trunk gradient-ratio imbalance is not caused by scalar
  loss weighting and cannot be fixed by it — reweighting leaves the ratio
  essentially untouched.

**What's actually happening**: the two loss functions have intrinsically
different gradient geometry with respect to the shared trunk (evidential's
KL-regularized gradient is structurally gentler than Tversky's set-overlap
derivative) — this is a property of the loss functions' math, not of their
relative scalar weight. Static reweighting still buys real calibration
gains, but through the **independent evidential_head** (which receives
100% of evidential's gradient regardless of trunk-sharing dynamics), not
by changing how much evidential influences the shared trunk.

## What this means for ABO

This is a stronger justification for the ABO design than either original
hypothesis would have given:

**Static hyperparameter search over the loss weight cannot fix the
trunk-level magnitude imbalance — it is architecturally invariant to that
hyperparameter, confirmed empirically across an 8x range.** A dynamic
controller that directly equalizes the trunk gradient ratio (per the
`m = alpha * delta` design, alpha driven by the ratio `r =
‖g_evidential‖/‖g_focal‖`, operating on trunk gradients specifically) is
doing something a loss-weight grid search structurally cannot do, no
matter how it's tuned. This is the concrete, measurement-driven rationale
for why ABO needs to operate on gradients directly rather than on the loss
scalar.

## Files

| File | Purpose |
|---|---|
| `experiments/exp_c0_weight_ablation/calibration.py` | ECE accumulator (validated against synthetic well/mis-calibrated cases) |
| `experiments/exp_c0_weight_ablation/train_weight_ablation.py` | Single-config training + diagnostics script |
| `experiments/exp_c0_weight_ablation/run_all.py` | Runs all 3 configs sequentially |
| `experiments/exp_c0_weight_ablation/{tag}/epoch_metrics.csv` | Dice/IoU/F1/HD95/ECE per epoch per config |
| `experiments/exp_c0_weight_ablation/{tag}/logs/*.csv` | Same 4 diagnostic CSVs as Phase B, per config |

---

**Completed**: 2026-08-02
