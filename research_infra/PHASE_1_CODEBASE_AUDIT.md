# Phase 1: Codebase Audit

**Purpose**: Complete data flow, computational graph, and instrumentation-point audit of `01_source_code/models/final_model.py` (2149 lines), performed to support a decision between two candidate research directions. Nothing in the codebase has been modified as part of this audit; findings below are read-only observations plus two small empirical gradient-flow checks (scripts run in an isolated scratch file and deleted afterward — not part of the shipped codebase).

**Method**: Static reading of `final_model.py` line-by-line (encoder, stem, attention, CBAM, decoder, evidential head, MAE, losses, training loop, validation loop, main execution), cross-checked against two runtime experiments:
1. Verified whether patient/case identity survives the MONAI `Compose` transform pipeline and `DataLoader` collation.
2. Verified whether the `AdaptiveSliceSelector`'s scorer network receives gradients from a downstream loss.

---

## 1. Complete Data Flow

```
Disk (PediMS/{Patient}/{Timepoint}/processed/*.nii.gz)
    │
    │  glob-based file discovery (final_model.py lines 219-241)
    │  builds data_dicts = [{"image":[path], "label":path, "case":filename}, ...]
    ▼
MONAI Dataset (train_ds / val_ds, 80/20 split, lines 279-281)
    │
    │  Compose transforms (lines 255-276):
    │    LoadImaged → EnsureChannelFirstd → Orientationd(RAS) → Spacingd(1mm iso)
    │    → NormalizeIntensityd → BinarizeLabel (custom) → Resized(64,64,64)
    │    → [train only: RandFlipd, RandRotate90d] → EnsureTyped
    ▼
DataLoader (batch_size=3 train / val, num_workers=0, lines 283-284)
    │
    │  batch = {"image": (B,1,64,64,64), "label": (B,1,64,64,64), "case": [str,...]}
    ▼
Model forward: HybridMiniSwin2D5_CBAM(images)   [images: (B,1,D=64,H=64,W=64)]
    │
    ├─ AdaptiveSliceSelector: scores all 64 slices → torch.topk picks k=9 indices
    │                          → gathers those 9 raw slices from `images`
    │                          → (B,1,9,64,64)
    ├─ Conv2D5Stem: per-slice 2D conv (shared weights) + learned softmax
    │               attention-weighted fusion over the 9 slices → (B,32,64,64)
    ├─ 4× ResidualBlock2D stages (with Mini-Swin 4×4 windowed attention)
    │               → feature pyramid [C0@64, C1@32, C2@16, C3@8, C4@4]
    ├─ CBAM_Module on bottleneck (C4@4×4)
    └─ LightweightDecoder: progressive upsample + skip-add
                    → probs (B,1,64,64) [central-slice-shaped 2D map]
                    → alpha (B,2,64,64) [evidential Beta parameters, if USALD_ENABLED]
    │
    ▼
Supervision target: center_slice_label = labels[:, :, 32, :, :]   (ALWAYS index 32,
    the geometric center of the resampled 64-slice volume — independent of which
    9 indices the selector actually chose)
    │
    ▼
Loss: HybridLoss(probs, center_label) + EvidentialBetaLoss(alpha, center_label)
    │
    ▼
Backward → AdamW step (two param groups: encoder+cbam @1e-5, decoder @4e-4)
```

### Two-phase training driver (lines 1573–2000+)
- **Phase 1 (MAE, ≤200 epochs)**: trains `encoder` (including the slice selector's parameters, per `.parameters()` iteration) + a small transformer decoder to reconstruct masked bottleneck tokens (75% mask). Saves `mae_best.pth`.
- **Phase 2 (Segmentation, ≤80 epochs)**: loads the MAE-pretrained encoder, **freezes the adaptive selector** (`freeze_adaptive_selector()`, line 1732), trains decoder + unfrozen encoder parts with the loss above.

---

## 2. Computational Graph (Gradient-Bearing Path)

```
images (leaf, requires_grad=False — normal input tensor)
   │
   ├── slice_scores = score_head(scorer(images))   [full graph, DIFFERENTIABLE]
   │        │
   │        └── top_indices = topk(slice_scores)   [NON-DIFFERENTIABLE — see §5]
   │
   └── selected_volume = images[gather using top_indices]   [DIFFERENTIABLE w.r.t. images
                                                              values, but indices themselves
                                                              carry no gradient back to
                                                              slice_scores/scorer/score_head]
            │
            ▼
       Conv2D5Stem → encoder stages → CBAM → decoder → {probs, alpha}
            │
            ▼
       loss_hybrid = 0.5·Dice(probs, label) + 0.5·FocalTversky(probs, label)
       loss_evid   = EvidentialBetaLoss(alpha, label)     [if USALD_ENABLED]
       loss_total  = loss_hybrid + loss_evid
            │
            ▼
       loss_total.backward()
            │
            ├─→ decoder params            ✓ receives gradient
            ├─→ CBAM params               ✓ receives gradient
            ├─→ encoder stage params      ✓ receives gradient (via selected_volume → stem)
            ├─→ Conv2D5Stem params        ✓ receives gradient (slice_conv + slice_attention)
            └─→ AdaptiveSliceSelector
                 (scorer, score_head)      ✗ receives NO gradient (confirmed empirically, §5)
```

**Consistency loss / pseudo-label loss / causal-decomposition loss** are fully implemented but gated off by config flags (`USALD_CONSISTENCY_ENABLED=False`, `USALD_FDR_ENABLED=False`, `USALD_CAUSAL_ENABLED=False`), so under the current default configuration they contribute no gradient at all. They exist as dead code paths that would activate if flags were flipped — noted here because Project B's scope statement implies auditing "the loss landscape" broadly, and these are load-bearing for that but currently inert.

---

## 3. Where Every Loss Is Computed

| Loss | Function / Line | Formula | Currently Active? |
|---|---|---|---|
| Dice | `HybridLoss.forward`, line 1033 (via MONAI `DiceLoss`) | `2·TP/(2·TP+FP+FN)` | ✅ Yes |
| FocalTversky | `FocalTverskyLoss.forward`, lines 1008-1019 | `(1 − TP/(TP+0.3FP+0.7FN))^0.75` | ✅ Yes |
| HybridLoss (combination) | lines 1032-1035 | `0.5·Dice + 0.5·FocalTversky` | ✅ Yes |
| EvidentialBetaLoss | lines 1046-1060 | `MSE·(1/(S+1)) + 1e-3·KL(Beta(α)‖Beta(1,1))` | ✅ Yes (`LAMBDA_EVIDENTIAL=1e-3`) |
| Consistency | `consistency_loss()`, lines 1191-1217 | `exp(-3u)·MSE(student,teacher)` | ❌ Disabled |
| Pseudo-label | inline, lines 1405-1417 | Masked BCE on high-confidence teacher pixels | ❌ Disabled |
| MAE reconstruction | `train_mae_epoch`, line 1323 | `L1(reconstruction·mask, target·mask)` | ✅ Yes, Phase 1 (MAE) only |
| Causal decomposition | folded into `alpha` before `EvidentialBetaLoss` (decoder lines 820-851) | Weighted sum of 3 causal α's | Present in forward pass (`USALD_ENABLED`), but **`USALD_CAUSAL_ENABLED=False`** means the causal weights are computed and logged (val_logs.csv columns) yet not separately loss-supervised beyond the single combined evidential loss |

---

## 4. Where Every MRI Slice Enters Training

Two distinct, non-interchangeable notions of "slice" exist in this codebase and must not be conflated:

1. **Acquisition-space slices**: The raw NIfTI volumes have native slice counts that vary per scan (observed in Phase 1 baseline testing: 218×240×153 for one FLAIR volume). These are never seen by the model directly.
2. **Resampled-space slices**: `Resized(spatial_size=(64,64,64))` (lines 262, 275) trilinearly interpolates every volume to exactly 64 slices along the D axis before it reaches the model. `AdaptiveSliceSelector(max_slices=64, ...)` (line 609) and `center_slice_label = labels[:, :, labels.shape[2]//2, :, :]` (line 1373, index 32) both operate in this **resampled** 64-index space.

**Consequence**: any "slice index" that instrumentation records is an index into the interpolated volume, not a physical acquisition slice. Mapping back to physical anatomy/slice thickness would require carrying the original affine/spacing metadata through the pipeline (currently discarded after `Spacingd` resamples to 1mm isotropic, and `Resized` further downsamples to 64 along each axis — the two resampling steps compound).

The model receives slices in two places:
- **All 64** resampled slices are passed to `AdaptiveSliceSelector`, which scores every one of them (`slice_scores`, shape `(B,64)`).
- Only the **k=9** top-scored slices are actually gathered into `selected_volume` (shape `(B,1,9,64,64)`) and passed into `Conv2D5Stem`. Slices not selected are fully discarded — no gradient, no forward computation beyond the scorer.

---

## 5. Whether Slice Identity Is Preserved — VERIFIED

**Two sub-questions, both empirically checked:**

### 5a. Does patient/case identity survive the pipeline?
**YES — confirmed by direct test.** A MONAI `Dataset` + `Compose` pipeline with extra dict keys (`case`, `patient_id`, `timepoint`) alongside `image`/`label` was built and run through the exact same transform chain used in `final_model.py`. Result:
```
After DataLoader collation, batch keys: ['image', 'label', 'case', 'patient_id', 'timepoint']
  case: ['n4_brain_FLAIR.nii.gz']   (type: list)
  patient_id: ['P1']                (type: list)
```
MONAI's dict-transforms pass through unrecognized keys untouched, and PyTorch's default collate function turns per-sample strings into a list-of-strings per batch. **This capability already exists in the framework** — it is simply not wired up in `final_model.py`'s `data_dicts` construction (line 234 stores only `"case": os.path.basename(img)`, not a separate `patient_id`/`timepoint` field, though the filename does encode enough to recover both by parsing the path).

### 5b. Is the *selected slice index* (which of the 64 was chosen) currently recorded?
**NO.** `AdaptiveSliceSelector.forward()` computes `top_indices` (line 367) but only returns `slice_scores` (line 379) — the actual chosen indices are discarded on return. `HybridMiniSwin2D5_ResNetEncoder.forward()` (line 649) even unpacks `slice_scores` into a local variable with a comment "*could be saved for visualization/analysis if needed*" (line 650) and then never uses it. **Zero lines in the codebase currently persist `top_indices` or `slice_scores` to disk, a log, or a return value that reaches the training loop.**

### 5c. Does the selected-slice set have any guaranteed relationship to the supervised slice?
**NO — this is the most consequential finding of this audit.** The supervision label is always resampled-index 32 (dead-center), fixed regardless of input. The `AdaptiveSliceSelector` performs an unconstrained global top-k over all 64 indices with no proximity term, no requirement that index 32 (or any neighborhood of it) be included, and (per §6 below) no training signal steering it toward any particular slice at all. It is architecturally possible — and, absent gradient-driven steering, essentially arbitrary — for the 9 selected slices to exclude index 32 and its neighbors entirely, meaning the model would be asked to predict the segmentation of a slice it may never have been shown.

---

## 6. Whether Slice-Level Statistics Can Already Be Collected — VERIFIED (Gradient Check)

A direct empirical test was run against a verbatim re-implementation of `AdaptiveSliceSelector` (identical architecture, isolated in a scratch script, executed once, then deleted): a downstream 1×1 conv "task" was attached to `selected_volume`, a scalar loss taken, and `.backward()` called.

**Result:**
```
scorer.0.weight   grad_is_None=True
scorer.4.bias     grad_is_None=True
score_head.3.weight  grad_is_None=True
... (all 12 parameter tensors in scorer + score_head)  grad_is_None=True for all
x.grad is None: False   (x.grad norm: 0.011720 — sanity check passes, input path is differentiable)
```

**Interpretation**: `torch.topk(slice_scores, k)` returns indices via its first output, which is discarded (`_, top_indices = torch.topk(...)`), and indices are inherently non-differentiable — there is no continuous relaxation (no Gumbel-softmax, no straight-through estimator, no REINFORCE term) connecting the discrete selection decision back to `slice_scores`. Because `selected_volume` is built by *indexing the raw input `images` tensor* (not by using the scorer's own feature activations as a weighted combination), the only gradient path that exists at all bypasses the scorer entirely.

**This means, under the current implementation: the `AdaptiveSliceSelector`'s learnable parameters receive zero gradient in both MAE pretraining and segmentation fine-tuning.** They are initialized once (PyTorch default init) and never updated by backpropagation. (BatchNorm3d running statistics inside `scorer` would still update via forward-pass statistics regardless of backward, but that is unrelated to the network *learning to rank slices by informativeness* — the stated premise of Project A.)

This is stated as a finding, not a value judgement — it is exactly the kind of pre-existing assumption Project A would need to test, and it is currently **untested and, per the mechanism as written, untestable via the existing training loop** without a code change (see Phase 2 §A for the specific fix options and their difficulty).

---

## 7. Where Gradients Are Computed

| Location | Line(s) | What happens |
|---|---|---|
| `train_mae_epoch` | 1310-1328 | `scaler.scale(loss).backward()` → `scaler.step(mae_optimizer)` |
| `train_segmentation_epoch` | 1419-1422 | `optimizer.zero_grad()` → `scaler.scale(loss).backward()` → `scaler.step(optimizer)` → `scaler.update()` |
| EMA teacher update | `ema_update()`, lines 1176-1189 | No gradient — direct in-place weighted parameter copy (`torch.no_grad()` context), only relevant if `USALD_CONSISTENCY_ENABLED=True` |

`optimizer` (segmentation phase) has **two parameter groups** (lines 1748-1755):
```python
encoder_params = seg_model.encoder.parameters() + seg_model.cbam.parameters()   # lr=1e-5
decoder_params = seg_model.decoder.parameters()                                  # lr=4e-4
```
Note: `encoder.parameters()` includes the (gradient-dead) `AdaptiveSliceSelector` parameters, but since `freeze_adaptive_selector()` (line 1732) sets `requires_grad=False` on them before this optimizer is even constructed, they don't receive optimizer updates in Phase 2 regardless. In Phase 1 (MAE), no such freeze exists, but per §6 they still receive no gradient — so across the *entire* training run, the selector's ranking weights are never updated by gradient descent at any point.

---

## 8. Where Instrumentation Can Safely Be Inserted

All of the following are additive insertion points — none require touching architecture, loss definitions, or optimizer logic:

| Insertion point | Line (current) | What can be read out, with zero side effects |
|---|---|---|
| After `LossLogger`-style extraction, before `.backward()` | ~1381-1387 | Individual `loss_dice`, `loss_ft`, `loss_hybrid`, `loss_evid` scalars (already prototyped in Phase 4 modules) |
| Immediately after `scaler.scale(loss).backward()`, before `scaler.step()` | ~1420 | Gradient norms (total / per-parameter-group / per-layer); requires `scaler.unscale_(optimizer)` first if using AMP, to get true (unscaled) gradient magnitudes — **not currently done anywhere**, a required addition for Project B |
| Inside `AdaptiveSliceSelector.forward`, before the `return` | line 379 | `top_indices` (which slices were picked) and `slice_scores` (raw ranking) — currently computed then discarded |
| Inside `Conv2D5Stem.forward`, at the `alphas` softmax fusion weights | lines 450-452 | Per-slice attention weight (how much each of the 9 selected slices contributed to the fused feature map) — this is a genuinely differentiable, currently-trained quantity, unlike the selector's own ranking |
| `data_dicts` construction | line 234 | `patient_id`/`timepoint` as separate dict keys (currently only a combined `case` filename string) — confirmed to survive the pipeline (§5a) |
| Validation loop, per-batch | 1483-1515 | Per-sample Dice/precision/recall if labels/case identity are threaded through (currently aggregated across the whole loader with no per-sample or per-case breakdown) |

**None of these insertion points require modifying forward-pass semantics, loss formulas, or optimizer configuration.** They require: (a) capturing already-computed intermediate tensors that are currently allowed to fall out of scope, and (b) one `scaler.unscale_()` call for true gradient-norm measurement under AMP.

---

## Summary of Load-Bearing Findings

1. **Patient/case identity**: recoverable, pipeline-compatible, not currently logged. *(Low-difficulty fix.)*
2. **Selected-slice indices**: computed internally, discarded, not currently logged. *(Low-difficulty fix — return value change only.)*
3. **Selected-slice-to-supervision correspondence**: **not guaranteed** — the 9 selected indices and the always-fixed supervision index (32) have no architectural link. *(This is a finding about the existing system, not something instrumentation alone fixes — flagged for Phase 2/6.)*
4. **AdaptiveSliceSelector trainability**: **empirically confirmed to receive zero gradient** under the current selection mechanism (hard top-k, raw-input gather). This is the single most important fact for evaluating Project A — the "adaptive" component of "Adaptive Information-Driven MRI Slice Sampling" does not currently adapt via the training loop as written. *(High-difficulty fix — requires a differentiable relaxation of the selection mechanism, not just added logging.)*
5. **Gradient norms / per-loss attribution**: not currently computed anywhere in the codebase (confirmed by full-file read — no `.grad`, `torch.norm`, or `unscale_` calls exist outside the diagnostic modules built in this session). *(Low-to-medium-difficulty fix — additive, does not touch AMP scaler semantics if inserted correctly.)*
6. **Consistency / pseudo-label / causal losses**: fully implemented, fully inert under default config. Relevant to Project B only if the research question extends to "why were these disabled" / "do they help," which would require re-enabling them (a training-behavior change, out of scope for pure instrumentation).
