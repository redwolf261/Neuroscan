# E15 Evidence Freeze

**Purpose**: Preserve the E15 discovery as a fixed empirical record before any novelty or algorithm-design work.

**Classification**: **E15 = causal discovery experiment, not proposed method and not a novelty claim.**

## Experimental setup

- **Checkpoint**: E12f calibrated pilot, seed 0, epoch 30.
- **Subjects**: 20 fixed validation subjects.
- **Model state**: Encoder and decoder frozen; evaluation mode; no retraining or optimizer steps.
- **Representation**: `z = dec1`, the 32-channel shared representation immediately before `seg_head`.
- **Decoder**: `D(z) = sigmoid(w^T z + b)`, implemented as the frozen `1x1x1` segmentation head.
- **Intervention location**: Directly modify `dec1`, then pass the modified representation through the frozen `seg_head`.

## Final intervention

For each voxel, let `mu_opp` be the same-volume centroid of the opposite ground-truth class. The intervention is:

```text
z' = z + alpha * (z - mu_opp) / ||z - mu_opp||
```

The ground-truth label selects which centroid is opposite. It does not select a decoder-specific sign or use the segmentation-head weight vector as an oracle direction.

The tested absolute push magnitudes were:

```text
alpha = {0, 1, 2, 4, 8, 14, 20, 28}
```

The headline, realistic range is `0` through `14`, anchored to E12f's observed margin trajectory.

## Frozen results

| Push alpha | Mean Dice | SD | Subjects |
|---:|---:|---:|---:|
| 0 | 0.9050 | 0.0900 | 20 |
| 4 | 0.9209 | 0.0851 | 20 |
| 8 | 0.9317 | 0.0811 | 20 |
| 14 | **0.9443** | 0.0687 | 20 |

- Realistic-range trend: `r = 0.990456`, `p = 1.3619e-4`.
- Paired push-14 versus push-0: mean improvement `+0.039275` Dice, approximately `+3.9275 percentage points`; `t = 7.6342`, `p = 3.3346e-7`.
- Jacobian sensitivity: `|dD/dz|` along the decoder weight direction is approximately `3.83x` the Euclidean-direction sensitivity (`0.0003970 / 0.0001036`). This is local decoder-sensitivity evidence only; it does **not** establish that Jacobian-guided optimization is novel or that it is the correct future method.

## Scope and interpretation

E15 establishes a causal phenomenon: under a frozen decoder, this class-conditional Euclidean representation displacement changes segmentation performance monotonically and positively. It does **not** establish a trainable method, a novel loss, or novelty over prior art.

The fixed interpretation is narrower: E15 shows that supervised latent separation, using ground-truth class information to select the opposite-class centroid, can causally improve the output of a frozen decoder. It does **not** show that an autonomous or deployable algorithm can discover the direction, and it does not establish a decoder-calibration mechanism.

The one-output-channel decoder has the form `D(z) = sigmoid(w^T z + b)`, so `grad_z D(z) = sigmoid'(w^T z + b) * w`. The decoder sees only the single direction `w` in this head. The 3.83x Jacobian ratio therefore does not support a geometry-decoder interaction claim; it is only a local sensitivity observation.

The proposed geometry-plus-Jacobian offline experiment is **retired**. It would reduce to asking how much to move along `w`, while E15's class direction was selected using ground truth. No decoder-calibration method should be derived from E15 without a separate, non-oracle scientific basis.

The result must remain separate from any future direction-construction or training rule. Any proposed algorithm must be tested independently against prototype, margin, Jacobian, gradient-alignment, and decoder-aware representation-learning prior art.

## Known controls and exclusions

- `push = 0` reproduced the checkpoint's baseline Dice (`0.9050`).
- The earlier ground-truth-signed push along `w` was rejected as oracle/label leakage and is not part of E15's final result.
- Pushes above `14` are off-manifold context, not the primary claim.
- The Jacobian measurement is a local, label-free sensitivity check; it is not a training rule.

## Exact source artifacts

- Code: `experiments/exp_e12_eggo_m/e15_decoder_sensitivity.py`
- Results: `experiments/exp_e12_eggo_m/e15_decoder_sensitivity_results/summary.json`
- Plot: `experiments/exp_e12_eggo_m/e15_decoder_sensitivity_results/decoder_sensitivity_plots.png`
- Full report: `docs/phases/PHASE_E15_DECODER_SENSITIVITY.md`
- Experiment index: `docs/RESEARCH_KNOWLEDGE_MAP.md`

**Freeze date**: 2026-09-13
