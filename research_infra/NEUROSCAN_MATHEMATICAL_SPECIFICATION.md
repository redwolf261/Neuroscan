# NeuroScan: Complete Mathematical Specification

**Purpose**: Turn `final_model.py` into a reconstructable mathematical/architectural baseline. Every claim below is cited to an exact line number in `01_source_code/models/final_model.py` as it exists in this repository today (2492 lines total — grown from the original 2149 due to the diagnostic instrumentation and slice-selection modes added during this investigation; all additions are marked `[ADDED]` below and are opt-in/default-preserving, so they do not change the mathematical model itself unless explicitly enabled).

**Ground rule for this document**: anything not directly verifiable in the code is labeled as a **documentation claim** (from README.md/DATASET_INFO.md), not fact. Anything we measured ourselves in earlier phases of this investigation is labeled **our measurement**. Nothing here is invented.

---

# Priority 1 — Overall Architecture

## Module hierarchy

```
HybridMiniSwin2D5_CBAM                         [line 1059]
├── encoder: HybridMiniSwin2D5_ResNetEncoder   [line 784]
│   ├── slice_selector: AdaptiveSliceSelector  [line 405]  (present iff use_adaptive_selection=True)
│   ├── stem: Conv2D5Stem                      [line 572]
│   └── stages: nn.ModuleList of 4 stages       [line 812]
│       └── each stage: nn.Sequential of 4x ResidualBlock2D  [line 725]
│           └── each ResidualBlock2D contains: conv1→bn1→relu→conv2→bn2→(MiniSwinAttention2D)→+identity→relu
├── cbam: CBAM_Module                           [line 869]
└── decoder: LightweightDecoder                 [line 934]
    ├── up_blocks / skip_convs (4x, symmetric to encoder stages)
    ├── prob_head (segmentation output)
    └── (if USALD_ENABLED) causal_shared, anatomy_head, pathology_head, noise_head, causal_weights
```

## Block diagram

```
Input: (B, 1, D=64, H=64, W=64)  3D volume (after preprocessing, see Priority 6)
   │
   ▼
AdaptiveSliceSelector  ──────────────────────────  selects k=9 of the 64 slices
   │  output: (B, 1, k=9, H=64, W=64)
   ▼
Conv2D5Stem  ─────────────────────────────────────  per-slice 2D conv (shared weights)
   │                                                  + learned softmax fusion across k slices
   │  output: (B, C0=32, 64, 64)
   ▼
Encoder Stage 1 (4x ResidualBlock2D, stride 2 on first block)   → (B, 64, 32, 32)
Encoder Stage 2 (4x ResidualBlock2D, stride 2 on first block)   → (B, 128, 16, 16)
Encoder Stage 3 (4x ResidualBlock2D, stride 2 on first block)   → (B, 256, 8, 8)
Encoder Stage 4 (4x ResidualBlock2D, stride 2 on first block)   → (B, 512, 4, 4)
   │  (skip connections from every stage's output are kept for the decoder)
   ▼
CBAM_Module (channel attention → spatial attention)              on the (B,512,4,4) bottleneck
   ▼
LightweightDecoder:
   Up-block 1 (bilinear ×2 + conv) + skip(stage3 output, 256ch)  → (B, 256, 8, 8)
   Up-block 2 (bilinear ×2 + conv) + skip(stage2 output, 128ch)  → (B, 128, 16, 16)
   Up-block 3 (bilinear ×2 + conv) + skip(stage1 output, 64ch)   → (B, 64, 32, 32)
   Up-block 4 (bilinear ×2 + conv) + skip(stem output, 32ch)     → (B, 32, 64, 64)
   ├── prob_head: Conv2d(32→1) + Sigmoid                          → probs (B,1,64,64)
   └── (if USALD_ENABLED) causal_shared → 3× causal evidential heads → alpha (B,2,64,64)
```

## Component details

**Encoder backbone**: ResNet-style (`ResidualBlock2D`, line 725) — NOT a pretrained torchvision ResNet; built from scratch, 2D convolutions applied to the fused 2.5D feature map, 4 stages, 4 residual blocks per stage (`blocks_per_stage=4`, line 799), channel progression `[32, 64, 128, 256, 512]` (`STAGE_CHANNELS`, line 239). Every block optionally carries a `MiniSwinAttention2D` module (line 660), enabled by default (`use_attention=True`).

**Feature dimensions** (spatial size halves each stage via `stride=2` on the first block of each stage): 64×64 → 32×32 → 16×16 → 8×8 → 4×4, channels 32→64→128→256→512.

**Attention modules**:
- **CBAM** (`CBAM_Module`, line 869): channel attention (avg-pool + max-pool → shared 2-layer 1×1-conv MLP → sigmoid → multiply) followed by spatial attention (channel-wise avg + max → 7×7 conv → sigmoid → multiply). Applied once, to the encoder's final bottleneck only (line 1090 in `HybridMiniSwin2D5_CBAM.forward`).
- **Mini-Swin** (`MiniSwinAttention2D`, line 660): windowed multi-head self-attention with **fixed, non-shifted** 4×4 windows (`MINI_SWIN_WINDOW=4`, line 237; note — this is windowed attention *without* the shifted-window alternation that the original Swin Transformer uses between blocks; every window here is the same fixed partition). 4 heads (`MINI_SWIN_HEADS=4`). Applied inside every `ResidualBlock2D`, i.e. at every encoder stage, not just the bottleneck.

**Evidential branch**: `anatomy_head`, `pathology_head`, `noise_head` (line 986-988), each a single `Conv2d(shared_features→2, kernel_size=1)` producing 2-channel raw evidential outputs per pixel.

**Causal branch**: `causal_shared` (line 979, a Conv2d(512→256)+BN+ReLU trunk shared by the three evidential heads) plus `causal_weights` (line 992, a 3-element learnable `nn.Parameter`, softmax-normalized at forward time to combine the three heads' alpha outputs into one). This is the "Causal Uncertainty Decomposition" component.

**Final segmentation head**: `prob_head` (line 963) — `Conv2d(32→1, kernel_size=1)` followed by `Sigmoid`. Output shape `(B, 1, 64, 64)` — a single 2D slice, not a 3D volume (see Priority 4 for what this slice represents).

---

# Priority 2 — Complete Loss Functions

## 2.1 Hybrid Segmentation Loss

**Exact code** (`HybridLoss`, line 1226-1240):

```python
L_Hybrid = λ1 · L_Dice + λ2 · L_FocalTversky,     λ1 = λ2 = 0.5 (line 1230, and again at instantiation line 2097)
```

### Dice term

Uses **MONAI's `DiceLoss(sigmoid=False)`** directly (line 1234) — i.e. NOT hand-rolled in this file; MONAI's implementation is:

$$L_{Dice} = 1 - \frac{2 \sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$$

where $p_i$ = predicted probability (already passed through sigmoid upstream by `prob_head`, hence `sigmoid=False` here to avoid double-applying it), $g_i$ = ground truth, $\epsilon$ = MONAI's default smoothing constant (`smooth_nr=1e-5, smooth_dr=1e-5` in MONAI's `DiceLoss` defaults — not overridden here, so MONAI's library defaults apply exactly).

### Focal Tversky term

**Hand-rolled** (`FocalTverskyLoss`, line 1203-1224):

$$TP = \sum_i p_i g_i, \quad FP = \sum_i (1-g_i) p_i, \quad FN = \sum_i g_i (1-p_i)$$

$$TI = \frac{TP + \epsilon}{TP + \alpha \cdot FP + \beta \cdot FN + \epsilon}, \quad \epsilon = 10^{-7}$$

$$L_{FocalTversky} = (1 - TI)^{\gamma}$$

with **α = 0.3, β = 0.7, γ = 0.75** (line 1207, defaults, not overridden at instantiation). Since β > α, false negatives are weighted **7/3 ≈ 2.33×** more heavily than false positives — this loss is deliberately recall-favoring.

**Note on reduction**: both `pred` and `target` are flattened to 1D (`.reshape(-1)`, lines 1214-1215) before computing TP/FP/FN — this is a **global** Tversky index over the entire batch's flattened tensor, not computed per-sample and averaged. Batch composition can therefore materially affect this loss's value in a way that's not simply the mean of per-sample losses.

## 2.2 Evidential Loss

**Exact code** (`EvidentialBetaLoss`, line 1245-1265). This is a **Beta-distribution** evidential model (not Dirichlet — there are only 2 classes, lesion/background, at the per-pixel level, so Beta is the natural 2-outcome special case of Dirichlet).

Per pixel, the decoder's evidential head(s) output raw values passed through `softplus(x) + 1` to guarantee valid Beta shape parameters $\alpha_0, \alpha_1 > 1$ (line 1032-1034, in `LightweightDecoder.forward`).

$$S = \alpha_0 + \alpha_1 \quad \text{(total evidence)}, \qquad p = \frac{\alpha_1}{S + \epsilon}, \quad \epsilon = 10^{-8}$$

**Data-fit term** — MSE between predicted mean $p$ and the binary target $g$, weighted by inverse certainty:

$$L_{fit} = \mathbb{E}\left[\frac{(p - g)^2}{S + 1}\right]$$

**KL regularizer to Beta(1,1)** (the uniform/maximum-entropy Beta distribution — this is the standard evidential-deep-learning regularizer that discourages the model from being falsely overconfident):

$$KL\big(\text{Beta}(\alpha_0,\alpha_1) \,\|\, \text{Beta}(1,1)\big) = \ln B(\alpha_0,\alpha_1) - (\alpha_0-1)\psi(\alpha_0) - (\alpha_1-1)\psi(\alpha_1) + (\alpha_0+\alpha_1-2)\psi(\alpha_0+\alpha_1)$$

where $\ln B(a,b) = \ln\Gamma(a) + \ln\Gamma(b) - \ln\Gamma(a+b)$ (the log Beta function, `torch.lgamma`, line 1262) and $\psi$ is the digamma function (`torch.digamma`, line 1263).

$$\boxed{L_{Evidential} = L_{fit} + \lambda_{KL} \cdot KL\big(\text{Beta}(\alpha_0,\alpha_1) \,\|\, \text{Beta}(1,1)\big)}, \qquad \lambda_{KL} = \lambda_{Evidential} = 10^{-3} \text{ (line 266)}$$

**Note**: $\alpha_0, \alpha_1$ passed into the KL term are additionally clamped to a minimum of $10^{-3}$ (`.clamp_min(1e-3)`, line 1261) before the lgamma/digamma calls, purely for numerical stability (they should already be $>1$ from the softplus+1 upstream, so this clamp is not normally active).

**Causal combination** (feeds into the alpha used above): $\alpha = w_{anat}\cdot\alpha_{anat} + w_{path}\cdot\alpha_{path} + w_{noise}\cdot\alpha_{noise}$, where $(w_{anat}, w_{path}, w_{noise}) = \text{softmax}(\theta)$ for a learnable $\theta \in \mathbb{R}^3$ initialized to $[0.3, 0.5, 0.2]$ (line 992). This combination happens inside the decoder (Priority 1) before the alpha ever reaches `EvidentialBetaLoss`.

## 2.3 Consistency Loss — **implemented, but disabled by default** (`USALD_CONSISTENCY_ENABLED = False`, line 260)

**Exact code** (`consistency_loss`, line 1396-1422). Mean-teacher style: a `teacher_model` is an EMA copy of the student (updated via `ema_update`, line 1381: $\theta_{teacher} \leftarrow 0.99\,\theta_{teacher} + 0.01\,\theta_{student}$, decay=`EMA_DECAY=0.99`).

$$u = 1 - 2\,|p_{teacher} - 0.5| \qquad \text{(teacher uncertainty; } u{=}0\text{ confident, } u{=}1\text{ at } p{=}0.5\text{)}$$

$$w = e^{-\gamma u}, \qquad \gamma = \text{UNCERTAINTY\_GAMMA} = 3.0 \text{ (line 278)}$$

$$\boxed{L_{Consistency} = \mathbb{E}\big[w \cdot (p_{student} - p_{teacher})^2\big]}$$

Weighted into the total loss as $\lambda_{Consistency}\cdot L_{Consistency}$ with $\lambda_{Consistency}=1.0$ (line 276), and only activated after a warmup period (`epoch > WARMUP_EPOCHS=10`, line 272) — see line 1592: `use_consistency = USALD_CONSISTENCY_ENABLED and teacher_model is not None and epoch > WARMUP_EPOCHS`. Since `USALD_CONSISTENCY_ENABLED` is `False`, `use_consistency` is always `False` and this entire branch never executes under the current configuration.

## 2.4 Pseudo-label Loss — **implemented, but disabled by default** (`USALD_FDR_ENABLED = False`, line 261)

**Exact code**, inline in `train_segmentation_epoch` (lines 1648-1660, not a standalone function). Requires `use_consistency` to also be true (i.e. it is gated *behind* the already-disabled consistency mechanism — with consistency off, this can never activate regardless of `USALD_FDR_ENABLED`).

Given an adaptive threshold $\tau_{pl}$ (see below):

$$\text{mask} = \mathbb{1}\big[p_{teacher} \geq \tau_{pl}\big] \;\lor\; \mathbb{1}\big[p_{teacher} \leq 1-\tau_{pl}\big] \qquad \text{(confident-either-way mask)}$$

$$\hat{g} = \mathbb{1}\big[p_{teacher} \geq \tau_{pl}\big] \qquad \text{(pseudo-label)}$$

$$\boxed{L_{Pseudo} = \frac{\sum \text{BCE}(p_{student}\cdot\text{mask},\ \hat{g}\cdot\text{mask})}{\sum \text{mask} + 10^{-7}}}$$

Weighted as $\lambda_{Pseudo}\cdot L_{Pseudo}$, $\lambda_{Pseudo}=0.5$ (line 277).

**Adaptive threshold** $\tau_{pl}$ comes from `estimate_fdr_threshold` (line 1424-1462): scans 100 candidate thresholds in $[0.1, 0.9]$, picks the smallest $\tau$ such that empirical precision on the validation set at that threshold is $\geq 1-q$, $q = \text{FDR\_Q} = 0.10$ (line 274) — i.e. targets $\geq$90% precision. Recomputed every `FDR_UPDATE_INTERVAL=5` epochs (line 275) once past warmup (main loop, line ~2214-2220 in the segmentation training loop).

## Summary table — what's actually contributing gradient today

| Loss | Formula location | Active under shipped config? |
|---|---|---|
| Dice | MONAI `DiceLoss`, via line 1234 | ✅ Yes |
| Focal Tversky | line 1203-1224 | ✅ Yes |
| Hybrid (0.5·Dice + 0.5·FT) | line 1226-1240 | ✅ Yes |
| Evidential (fit + λ·KL) | line 1245-1265 | ✅ Yes (λ=1e-3) |
| Consistency | line 1396-1422 | ❌ No (`USALD_CONSISTENCY_ENABLED=False`) |
| Pseudo-label | line 1648-1660 | ❌ No (gated behind Consistency, which is off) |

**Actual total loss backpropagated under the shipped configuration** (line 1624-1630):
$$L_{total} = \underbrace{0.5\,L_{Dice} + 0.5\,L_{FocalTversky}}_{L_{Hybrid}} + L_{Evidential}$$

---

# Priority 3 — Optimizer

## Segmentation phase (line 2087-2098)

```python
optimizer = AdamW([
    {'params': encoder.parameters() + cbam.parameters(), 'lr': 1e-5},   # LEARNING_RATE_ENCODER, line 242
    {'params': decoder.parameters(),                      'lr': 4e-4},   # LEARNING_RATE_DECODER, line 243
], weight_decay=0.01)

scheduler = CosineAnnealingLR(optimizer, T_max=SEGMENTATION_EPOCHS)   # T_max=80 by default
```

- **Two parameter groups**, confirmed exactly as: **Encoder+CBAM at LR=1×10⁻⁵** (fine-tuning a pretrained encoder), **Decoder at LR=4×10⁻⁴** (training from scratch) — a 40× ratio.
- **No warmup** — `CosineAnnealingLR` starts decaying from step 1; there is no linear-warmup phase anywhere in the optimizer/scheduler setup.
- **No separate "heads" learning rate** — the evidential/causal heads (`anatomy_head`, `pathology_head`, `noise_head`, `causal_shared`, `causal_weights`) are all part of `decoder.parameters()` and therefore share the decoder's single 4×10⁻⁴ learning rate; there is no third parameter group isolating them, despite them behaving very differently from the segmentation head under gradient inspection (see companion diagnostic reports).

## MAE pretraining phase (line 1929-1930)

```python
mae_optimizer = AdamW(mae_model.parameters(), lr=1e-4, weight_decay=0.05)   # MAE_LEARNING_RATE
mae_scheduler = CosineAnnealingLR(mae_optimizer, T_max=MAE_EPOCHS)          # T_max=200 by default
```

Single parameter group (whole MAE model — encoder + lightweight transformer decoder), higher weight decay (0.05 vs. 0.01 for segmentation).

## Mixed precision

`torch.cuda.amp.GradScaler` / `autocast` used in both phases (`use_amp = torch.cuda.is_available()`, line 222) — AMP is on whenever a CUDA device is available, off on CPU.

## Gradient clipping

**Not present anywhere in the file** (confirmed by direct search — no `clip_grad_norm_` or `clip_grad_value_` call exists in `final_model.py`).

---

# Priority 4 — Training Pipeline

```
Input MRI volume (FLAIR, native resolution, e.g. 218×240×153 observed)
   ↓  LoadImaged → EnsureChannelFirstd → Orientationd(RAS) → Spacingd(1mm iso)
   ↓  NormalizeIntensityd → BinarizeLabel(label>0) → Resized(trilinear/nearest → 64×64×64)
   ↓  [train only] RandFlipd(p=0.5, all 3 axes) → RandRotate90d(p=0.3, max_k=3)
   ↓
(B, 1, D=64, H=64, W=64) volume + (B, 1, 64, 64, 64) binary label
   ↓
Adaptive Slice Selection  (see Priority 5)              →  (B, 1, k=9, 64, 64)
   ↓
2.5D Stem (Conv2D5Stem): per-slice shared 2D conv + learned softmax fusion across 9 slices
   ↓
Encoder: 4× (4 ResBlocks w/ Mini-Swin attention, stride-2 downsample)
   ↓
CBAM (channel + spatial attention on the 4×4×512 bottleneck)
   ↓
Decoder: 4× (bilinear upsample + conv + skip-connection add)
   ↓
Segmentation Head (Conv1×1 + Sigmoid)         →  probs (B,1,64,64)
Evidential/Causal Heads (if USALD_ENABLED)    →  alpha (B,2,64,64)
   ↓
Supervision target = labels[:, :, D//2, :, :]   ← ALWAYS the fixed volume-center slice (idx 32)
   ↓
Loss = 0.5·Dice(probs,target) + 0.5·FocalTversky(probs,target) + EvidentialBetaLoss(alpha,target)
   ↓
Backward (AMP-scaled)
   ↓
Optimizer step (AdamW, two LR groups) → CosineAnnealingLR.step()
```

**Important, empirically-verified detail not obvious from reading the forward pass alone**: the supervision target is *always* the geometric center slice of the 64-slice resampled volume, regardless of which 9 slices the selector actually chose. We measured (see `research_infra/PHASE_1_CODEBASE_AUDIT.md` and the Phase-8 reproduction experiments) that under the shipped `AdaptiveSliceSelector`, this center index is included in the chosen 9-slice window in only **14.3%** of samples — i.e. **for the large majority of training examples, the model is graded on a slice it was not shown, or shown only as one of many unrelated slices.**

---

# Priority 5 — Adaptive Slice Selector

**Exact code**: `AdaptiveSliceSelector`, line 405-570.

## Architecture of the scorer network

```python
scorer = Sequential(
    Conv3d(1→8, k=3, pad=1), BatchNorm3d(8), ReLU,
    Conv3d(8→16, k=3, pad=1), BatchNorm3d(16), ReLU,
    AdaptiveAvgPool3d((max_slices=64, 4, 4)),
    Flatten(start_dim=2),
)   # (B, 16, D*16) intermediate

score_head = Sequential(
    Linear(16*16 → 64), ReLU, Dropout(0.1),
    Linear(64 → 1),      # one scalar score per candidate slice
)
```

## Selection mechanism

For each of the (up to 64) candidate slices, a scalar score is computed via the `score_head` applied to that slice's pooled 3D-conv features. Scores are concatenated to `(B, D)`.

$$\text{top\_indices} = \text{TopK}_k(\text{scores}), \qquad k=9 \text{ (K\_SLICES, line 234)}$$

**This is a hard `torch.topk`** (line 463 in the original numbering, confirmed unchanged) — **no softmax-weighted soft selection, no Gumbel-softmax, no temperature parameter, no straight-through estimator**. The selected slices are then gathered *directly from the raw input tensor* `x` (not from any feature computed by the scorer), i.e. the scorer's output influences only *which indices* get chosen, and the actual selected content bypasses the scorer's computation graph entirely.

$$\text{selected\_volume}[b] = x[b,\, :,\, \text{top\_indices}[b],\, :,\, :] \qquad \in \mathbb{R}^{1\times 9\times 64\times 64}$$

## Differentiability — **empirically confirmed non-differentiable in practice**

`torch.topk`'s returned *indices* carry no gradient by construction (indices are integers; PyTorch's autograd only ever propagates gradient through the topk *values*, which are discarded here — line reads `_, top_indices = torch.topk(...)`, explicitly dropping the values). Combined with the fact that `selected_volume` is built by indexing the raw input (not the scorer's features), **the scorer network (`scorer` + `score_head`) receives zero gradient from the downstream task loss.** This was verified three independent ways in this investigation:
1. A standalone reimplementation with a synthetic downstream loss (`.grad` was `None`/zero for all 12 scorer/score_head parameter tensors).
2. A full end-to-end run of the actual `final_model.py` module on synthetic data — the model's gradient-diagnostics log recorded gradients for 228 of 240 total parameter tensors; the missing 12 are exactly `scorer`'s 8 tensors + `score_head`'s 4 tensors.
3. A real 29-epoch training run on the actual PediMS data — same result, same 12 missing tensors, every epoch.

## `[ADDED]` Alternative selection modes

Three additional, opt-in, default-preserving `selection_mode` options were added during this investigation (constructor argument `selection_mode='adaptive'` default, line 413):
- `'uniform'`: fixed, evenly-spaced 9 indices spanning the full 64-slice range, no scorer involved.
- `'center_window'`: fixed 9 *consecutive* indices centered on index 32 (the always-used supervision target).
- `'dynamic_window'`: like `center_window`, but the window center is a **per-sample, per-forward-call** index passed in as `center_indices` (used together with a modified training loop that samples a random supervision target per batch — see `research_infra/phase8/reproduce_attempt_multislice.py`).

None of these are active unless explicitly constructed with that mode; the shipped default (`'adaptive'`) reproduces the original, non-differentiable behavior exactly.

## Output equation, restated compactly

$$\text{selected\_volume},\ \text{scores} = \text{AdaptiveSliceSelector}(x), \qquad \text{scores} \in \mathbb{R}^{B\times 64},\ \text{selected\_volume}\in\mathbb{R}^{B\times1\times9\times64\times64}$$

---

# Priority 6 — Dataset

| Field | Value | Source |
|---|---|---|
| Dataset name | PediMS (Pediatric Multiple Sclerosis) | README.md |
| Patients (documentation claim) | 45 (36 train / 9 val) | README.md — **not verifiable on this machine; see below** |
| Patients (actually present on this machine) | **9** | direct filesystem check, this investigation |
| Volumes actually loaded | **28** (9 patients × up to 3 longitudinal timepoints each) | `final_model.py` runtime log, this investigation |
| Train / Val split (actual) | **22 / 6** (fixed 80/20 slice of the 28, `split = int(0.8*len(data_dicts))`, no shuffling of the split itself) | line ~382 area, confirmed |
| Modalities available | FLAIR, T1, T2 (+ N4 bias-corrected variants); **only FLAIR is used** as model input | data-loading code, line 314 |
| Label source | `Consensus.nii` (confirmed to be the true lesion annotation — the `mask_FLAIR/T1/T2.nii.gz` files were checked and found to be **brain-extraction/skull-stripping masks**, not lesion masks, based on their voxel-positive fraction: ~17% vs. Consensus's ~0.03%) | this investigation |
| Spatial size fed to the model | 64×64×64 (`SPATIAL_SIZE`, line 235), after `Spacingd` resamples to 1mm isotropic and `Resized` (trilinear for image, nearest for label) to this fixed size | line 358 |
| Normalization | `NormalizeIntensityd(nonzero=True, channel_wise=True)` — per-channel z-score over nonzero voxels only | line 356 |
| Augmentations | `RandFlipd` (all 3 spatial axes, p=0.5), `RandRotate90d` (p=0.3, max_k=3) — train split only | lines 359-360 |
| Batch size | 3 (`BATCH_SIZE`, line 240, comment: "Limited by 4GB GPU") | |
| DataLoader workers | 0 (comment: "Must be 0 on Windows") | line 241 |

**Patient-count discrepancy**: the "45 patients" figure appears exactly once in the entire file — hardcoded as a string literal inside an f-string template that writes a deployment `README.md` (line ~2414-area, `"- Trained on PediMS dataset (45 patients, pediatric MS lesions)"`). It is not computed from `len(data_dicts)` or any runtime count; it is decorative documentation text, disconnected from the actual data-loading path. On this machine, only 9 patients / 28 volumes are present, and no additional PediMS data was found anywhere else on the filesystem (a `.zip` in Downloads was checked and found to contain the identical 9 patients, not additional ones).

---

# Priority 7 — Experimental Results

## Documentation claims (README.md) — **not independently verified; no pretrained checkpoint or full 45-patient dataset was available on this machine to check them**

| Metric | Claimed value |
|---|---|
| Dice | 83.99% |
| Precision | 77.60% |
| Recall | 91.64% |
| F1 | 84.04% |
| Training | 48 epochs, RTX 2050 GPU |
| Cross-dataset (LGG, brain tumor) | 20.01% ± 16.12% Dice |
| Cross-dataset (MS60, adult MS) | 1.10% ± 1.47% Dice |

**IoU, Hausdorff distance, ASSD, Specificity, calibration metrics, and inference speed are not reported anywhere in the README, DATASET_INFO.md, or the code** — these are gaps in the original project's documentation, not omissions in this document. If the semester project needs these, they would need to be computed fresh (IoU is trivial to derive from Dice: $IoU = \frac{Dice}{2-Dice}$ for the same TP/FP/FN definitions; Hausdorff/ASSD require boundary-distance code that does not currently exist anywhere in this repository).

## Our own measurements (this investigation, 9-patient/28-volume dataset, not the claimed 45)

| Configuration | Best Val Dice | Note |
|---|---|---|
| As-shipped (`adaptive` selector), 60-epoch MAE + up to 80-epoch segmentation, real early stopping | 0.42% | highly volatile, repeatedly crashes to exactly 0 |
| `center_window` fix (fixed, always-correct 9-slice window) | 0.17% | stable but low ceiling |
| Multi-slice supervision + `dynamic_window` (random supervision target per batch) | 0.18% (old fixed-slice metric) / **2.32%** (native random-slice metric, stable) | first stable, non-noise improvement found in this investigation |

These numbers are **far below the documentation's 83.99% claim**, attributable primarily to the ~4× smaller training set (22 vs. 36 patients-worth of volumes) available on this machine, not to a code defect introduced during this investigation (verified: diagnostics are bit-identical when disabled; the dataset-path and sklearn-dependency fixes are metric/loading fixes, not training-behavior changes).

---

# Priority 8 — Implementation Details

| Detail | Value | Source |
|---|---|---|
| Batch size | 3 | line 240 |
| MAE epochs (max) | 200 (`MAE_EPOCHS`), overridable via `MAE_EPOCHS_OVERRIDE` env var `[ADDED]` | line 248 |
| Segmentation epochs (max) | 80 (`SEGMENTATION_EPOCHS`), overridable via `SEGMENTATION_EPOCHS_OVERRIDE` env var `[ADDED]` | line 249 |
| Mixed precision | Yes, `torch.cuda.amp` (`GradScaler`/`autocast`), on whenever CUDA is available | line 222, 45 |
| Gradient clipping | **None** | confirmed absent |
| Early stopping (MAE) | patience = 30 epochs without loss improvement | line 1988 |
| Early stopping (segmentation) | patience = 20 epochs without validation Dice improvement | line 2165 |
| Checkpoint criterion | Segmentation: save whenever `val_dice > best_val_dice` (strict improvement) → `best_model.pth`/`.pt`. MAE: save whenever `mae_loss < best_mae_loss` → `mae_best.pth`. Local "resume" checkpoints (`seg_resume.pth`/`mae_resume.pth`) saved every epoch regardless, for crash recovery only | lines 2029-2043, 2256-2270 |
| Determinism | `monai.utils.set_determinism(42)` called once at import (line 97) | |
| cuDNN | `benchmark=True`, `enabled=True` (line 223-224) — trades exact determinism for speed on fixed-shape inputs (all inputs are resized to the same 64³ shape, so this is safe/appropriate here) | |

---

# Priority 9 — Code Structure

**There is no modular `models/`, `losses/`, `encoder.py`, `decoder.py` file layout.** This is important to state plainly: the entire model (encoder, decoder, all four loss classes, the adaptive selector, the MAE pretraining module, both training loops, validation, checkpointing, and the deployment-package writer) lives in a **single 2492-line file**: `01_source_code/models/final_model.py`. There is no `AdaptiveSliceSelector.py` or `loss.py` to hand over separately — the class definitions live inline at the line numbers cited throughout this document.

## Actual repository layout (top-level, relevant parts)

```
01_source_code/
├── models/
│   ├── final_model.py              ← THE file (architecture + losses + selector + training, 2492 lines)
│   ├── model_artifacts/
│   │   ├── final_model.py          (an older/parallel copy — not the one used in this investigation)
│   │   └── inference.py
│   └── training_scripts/
│       └── resume_training.py      (thin launcher: `import final_model`)
├── evaluation/
│   └── test_model_evaluation.py
├── diagnostics/                     [ADDED, this investigation]
│   ├── config.py                   (diagnostic feature flags, all default False)
│   ├── loss_logger.py, gradient_logger.py, gradient_similarity.py, slice_logger.py
│   └── visualize.py
02_documentation/
├── architecture/FINAL_MODEL_ARCHITECTURE.md
└── ...
05_webapp/                          (Flask + React deployment demo, not used in this investigation)
PediMS/                             (9 patients, see Priority 6)
research_infra/                     [ADDED, this investigation — all audit/experiment artifacts]
├── PHASE_1_CODEBASE_AUDIT.md ... PHASE_8_PROJECT_SELECTION.md
├── phase8/                         (experiment runner scripts)
└── reproduction_attempt_*/         (logged training runs)
```

## The three files you asked for, resolved against what actually exists

1. **`final_model.py`** → `01_source_code/models/final_model.py` (this is the one real source file; everything else below is contained inside it).
2. **`loss.py`** → does not exist as a separate file. All four loss classes (`FocalTverskyLoss`, `HybridLoss`, `EvidentialBetaLoss`) plus the two disabled-objective functions (`consistency_loss`, and the inline pseudo-label logic) are at lines 1203-1265 and 1396-1422/1648-1660 of `final_model.py`, extracted in full in Priority 2 above.
3. **`AdaptiveSliceSelector`** → does not exist as a separate file. It is a class at line 405-570 of `final_model.py`, extracted in full in Priority 5 above.

---

# What This Enables (per your closing note)

With the above, the full gradient path for each objective is now traceable:

- **Dice + Focal Tversky** gradients reach: `prob_head` → decoder up-blocks/skip-convs → CBAM → all 4 encoder stages (incl. Mini-Swin attention) → `Conv2D5Stem` (both `slice_conv` and `slice_attention`) → **not** the `AdaptiveSliceSelector`'s scorer (confirmed zero-gradient path).
- **Evidential** gradients reach: `anatomy_head`/`pathology_head`/`noise_head`/`causal_shared`/`causal_weights` → same shared decoder trunk upstream of `causal_shared` → same encoder path as above. This is why the two loss terms are not fully independent — they share the entire encoder and the decoder's upsampling trunk, diverging only at the final two output branches.
- **Consistency/Pseudo-label** gradients, if activated, would additionally flow through an EMA teacher's forward pass (no gradient there, `torch.no_grad()`-wrapped) into the same student parameters as Dice/FocalTversky, gated by the warmup epoch count and, for pseudo-label, an FDR-derived confidence mask.

This is exactly the map needed to identify where a new optimization rule could intervene (e.g. per-branch gradient scaling at the causal/evidential-head boundary, as already prototyped in `research_infra/phase8/common.py`'s `make_gradient_scale_hook`) without touching architecture.
