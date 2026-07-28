# Phase 3 & 4: Diagnostic Framework + Structured Logging

**Status**: Implemented, wired into `01_source_code/models/final_model.py`, and empirically verified to be a no-op when disabled.

---

## What Was Built

```
01_source_code/diagnostics/
├── __init__.py              - package marker, no side effects
├── config.py                 - all ENABLE_* flags (default False) + sampling-frequency knobs
├── loss_logger.py             - writes training_metrics.csv
├── slice_logger.py            - writes slice_metrics.csv
├── gradient_logger.py         - writes gradient_metrics.csv + gradient_metrics_layerwise.csv
└── gradient_similarity.py     - writes gradient_metrics_similarity.csv
```

Plus additive edits to `final_model.py`:
1. Import block + flag-status banner (printed once on import).
2. `data_dicts` construction: added `"patient_id"` and `"modality"` keys alongside the existing `"case"` key (Section 5a of the Phase 1 audit empirically confirmed these survive the MONAI pipeline unchanged).
3. `AdaptiveSliceSelector.forward`: stashes `self.last_top_indices` / `self.last_slice_scores` (detached) when `ENABLE_SLICE_DIAGNOSTICS` — does not change the function's return value.
4. `Conv2D5Stem.forward`: stashes `self.last_fusion_alphas` (detached) under the same flag — does not change the function's return value.
5. `train_mae_epoch` / `train_segmentation_epoch`: flag-gated logging calls at four points (loss computation, slice-selection readout, pre-backward loss-specific gradient attribution, post-backward gradient norms) — see inline comments in the file for exact placement rationale, especially the AMP `scaler.unscale_()` ordering requirement.

**Every flag defaults to `False`.** Each is also readable from an environment variable of the same name (`ENABLE_LOSS_DIAGNOSTICS=1 python resume_training.py`), so diagnostics can be toggled per run without editing source.

---

## Empirical Non-Invasiveness Verification

Rather than argue by code inspection alone, this was tested directly:

1. Built a minimal synthetic dataset compatible with `final_model.py`'s existing (pre-existing, already-documented-as-broken-against-the-real-dataset — see Phase 1 §Data Flow) directory-scanning logic, solely so the real module could be imported.
2. Instantiated two copies of `HybridMiniSwin2D5_CBAM` from the same random seed.
3. Ran the real, patched `train_segmentation_epoch()` once on each copy, using an identical synthetic batch: once with all four `ENABLE_*` flags `False`, once with all four `True`.
4. Compared all 240 named parameter tensors between the two resulting models.

**Result:**
```
Max abs parameter difference across all layers: 0.000e+00
Number of layers with any difference: 0 / 240
Result tuples equal: True
```
Diagnostics-on and diagnostics-off produced bit-identical trained weights and identical returned metrics. The verification script and its synthetic dataset were scratch artifacts, deleted after the run — they are not part of the shipped codebase, only the evidence behind this claim.

As a side effect, this same test independently reconfirmed the Phase 1 finding that `AdaptiveSliceSelector`'s scorer network receives no gradient: the layer-wise gradient log recorded 228 of 240 parameter tensors (only tensors with non-`None` `.grad` are logged); the missing 12 are exactly `scorer`'s 8 tensors + `score_head`'s 4 tensors.

---

## Structured Logging Format

All CSVs share `epoch`, `iteration`, `timestamp` as their first three columns, so they can be joined in pandas via `(epoch, iteration)`. Column choices follow directly from what Phase 1/2 established as actually measurable.

### `training_metrics.csv` (from `loss_logger.py`)

| Column | Meaning |
|---|---|
| `epoch`, `iteration`, `timestamp` | position in training |
| `phase` | `"segmentation"` or `"mae"` |
| `patient_ids` | semicolon-joined per-sample IDs for this batch (blank if not threaded through by the caller) |
| `loss_dice`, `loss_focal_tversky`, `loss_hybrid` | decomposed `HybridLoss` components |
| `loss_evidential` | `EvidentialBetaLoss` value (0 if USALD disabled) |
| `loss_total` | the actual scalar backpropagated |
| `dice_metric` | thresholded Dice (not a loss — the evaluation metric) |
| `learning_rate_encoder`, `learning_rate_decoder` | current LR of each optimizer param group |

### `slice_metrics.csv` (from `slice_logger.py`)

One row per **selected slice per sample per batch** (not one row per batch):

| Column | Meaning |
|---|---|
| `patient_id`, `case`, `timepoint` | sample identity (blank if not present in the batch dict) |
| `sample_in_batch` | index within the batch (0..B-1) |
| `selected_slice_index` | which of the 64 resampled indices was chosen |
| `slice_score` | that slice's raw score from the (currently untrained) scorer network |
| `fusion_attention_weight` | that slice's learned softmax fusion weight from `Conv2D5Stem` — the one differentiable, actually-trained quantity in this path |
| `supervised_center_index` | always 32 (fixed) |
| `center_index_was_selected` | boolean — directly measures the Phase 1 §5c finding across training |

### `gradient_metrics.csv` (from `gradient_logger.py`)

| Column | Meaning |
|---|---|
| `total_grad_norm`, `encoder_grad_norm`, `decoder_grad_norm`, `cbam_grad_norm` | L2 norms after `scaler.unscale_()`, i.e. true magnitudes |
| `grad_norm_loss_dice`, `grad_norm_loss_ft`, `grad_norm_loss_evid` | gradient norm attributable to each loss term in isolation (sampled every `LOSS_SPECIFIC_GRADIENT_EVERY_N_BATCHES` batches — blank otherwise) |
| `amp_scale_factor` | the scaler's scale at this step, logged for auditability even though norms are already unscaled |

### `gradient_metrics_layerwise.csv` (from `gradient_logger.py`)

One row per parameter tensor per logged batch (sampled every `LAYER_WISE_EVERY_N_BATCHES` batches): `layer_name`, `grad_norm`, `param_count`, `grad_norm_per_param`. Note: a layer with `requires_grad=True` but zero-contributing-loss will simply not appear (its `.grad` stays `None`) — this is itself diagnostic information (see the `AdaptiveSliceSelector` case above).

### `gradient_metrics_similarity.csv` (from `gradient_similarity.py`)

Sampled every `GRADIENT_SIMILARITY_EVERY_N_BATCHES` batches (most expensive diagnostic): pairwise cosine similarities between Dice/FocalTversky/Evidential gradient vectors, plus a `conflict_score` (mean of the negative-similarity magnitudes).

---

## Overhead (measured informally on the CPU smoke test, indicative only — no GPU available in this environment)

| Diagnostic | Per-batch cost | Default sampling |
|---|---|---|
| Loss logging | ~1 extra forward (no backward) | every batch |
| Slice logging | attribute reads only | every batch |
| Gradient norms | 1 extra `unscale_()` + norm computation | every batch |
| Loss-specific gradient attribution | 3 extra `backward(retain_graph=True)` passes | every 5th batch |
| Layer-wise gradients | norm computation over all params | every 10th batch |
| Gradient similarity | 3 extra `backward(retain_graph=True)` + cosine ops | every 20th batch |

All sampling intervals are configurable via `diagnostics/config.py` or environment variables.
