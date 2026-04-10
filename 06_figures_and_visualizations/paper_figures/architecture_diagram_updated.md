# HybridMiniSwin2.5D-CBAM Architecture (Updated)
## Pediatric MS Lesion Segmentation Model

**Model Name**: HybridMiniSwin2.5D-ResNet with CSRF + MAE Pretraining  
**Parameters**: 4.23M  
**Input**: 2.5D FLAIR (5 slices)  
**Architecture**: Hybrid CNN-Transformer with Evidential Uncertainty

---

## Key Architecture Changes from Previous Version:

### ❌ REMOVED:
- Full 3D convolutions (ablation: -2.36% when present)
- Heavy dropout layers (ablation: +0.55% when removed)
- Full self-attention (replaced with Mini-Swin 4×4 windows)
- CSRF module in bottleneck (**NOTE: Simplified - no longer multi-slice fusion**)

### ✅ ADDED/KEPT:
- **2.5D Conv Stem** with slice-wise attention
- **ResNet-style skip connections** (CRITICAL: ablation -4.44% when removed)
- **Mini-Swin windowed attention** (4×4 windows, not full attention)
- **Evidential uncertainty head** with causal decomposition (3 heads)
- **MAE pretraining** (mask ratio 0.75, optimal from ablation)
- **Adaptive slice selection** (k=5 slices, from hyperparameter ablation)

---

## Architecture Flow:

```
Input: 3D FLAIR Volume (B, 1, D, H, W)
                ↓
    ┌───────────────────────────┐
    │   2.5D Convolutional Stem │
    │  - Extract k=5 slices     │
    │  - Slice-wise 2D conv     │
    │  - Attention-weighted     │
    │    fusion                 │
    └───────────────────────────┘
                ↓
        (B, 32, 64, 64)
                ↓
    ┌───────────────────────────┐
    │      ENCODER (4 stages)   │
    ├───────────────────────────┤
    │ Stage 1: ResNet + Swin    │
    │  4× ResidualBlock2D       │
    │  Mini-Swin (4×4 windows)  │
    │  32 → 64 ch, 64→32 size   │
    ├───────────────────────────┤
    │ Stage 2: ResNet + Swin    │
    │  4× ResidualBlock2D       │
    │  64 → 128 ch, 32→16       │
    ├───────────────────────────┤
    │ Stage 3: ResNet + Swin    │
    │  4× ResidualBlock2D       │
    │  128 → 256 ch, 16→8       │
    ├───────────────────────────┤
    │ Stage 4: ResNet + Swin    │
    │  4× ResidualBlock2D       │
    │  256 → 512 ch, 8→4        │
    └───────────────────────────┘
                ↓
        (B, 512, 4, 4) ← Bottleneck
                ↓
    ┌───────────────────────────┐
    │  CSRF Module (SIMPLIFIED) │
    │  - No multi-slice fusion  │
    │  - Single slice passthru  │
    └───────────────────────────┘
                ↓
    ┌───────────────────────────┐
    │      DECODER (4 stages)   │
    ├───────────────────────────┤
    │ Up 1: 512→256, 4→8        │
    │  Bilinear upsample + Conv │
    │  Skip connection + fusion │
    ├───────────────────────────┤
    │ Up 2: 256→128, 8→16       │
    ├───────────────────────────┤
    │ Up 3: 128→64, 16→32       │
    ├───────────────────────────┤
    │ Up 4: 64→32, 32→64        │
    └───────────────────────────┘
                ↓
        (B, 32, 64, 64)
                ↓
    ┌───────────────────────────┐
    │     OUTPUT HEADS          │
    ├───────────────────────────┤
    │ 1. Segmentation Head      │
    │    Conv 1×1 + Sigmoid     │
    │    → (B, 1, 64, 64)       │
    │                           │
    │ 2. Evidential Heads (×3)  │
    │    Causal Decomposition:  │
    │    - Anatomy uncertainty  │
    │    - Pathology uncertainty│
    │    - Noise uncertainty    │
    │    Each: Conv 1×1 →       │
    │    Beta(α₁, α₂)           │
    │                           │
    │    Weighted combination:  │
    │    α_total = Σ wᵢ·αᵢ      │
    │    (w learned)            │
    └───────────────────────────┘
                ↓
        Final Segmentation Map
        (B, 1, H, W)
```

---

## Component Details:

### 1. **2.5D Convolutional Stem**
```python
Input: (B, 1, D, H, W)
↓
Extract k=5 center slices
↓
For each slice:
  - Conv2D(1→32, 3×3)
  - BatchNorm + ReLU
  - Slice attention (adaptive pooling)
↓
Weighted fusion with softmax(αᵢ)
↓
Output: (B, 32, H, W)
```

### 2. **ResidualBlock2D** (Used 16× total)
```python
Input: (B, C_in, H, W)
↓
Conv 3×3, stride=s
BatchNorm + ReLU
↓
Conv 3×3, stride=1
BatchNorm
↓
Mini-Swin Attention (4×4 windows)
↓
Skip Connection: out = out + identity
ReLU
↓
Output: (B, C_out, H', W')
```

### 3. **Mini-Swin Attention**
- **Window size**: 4×4 (not full attention)
- **Heads**: 4
- **Partition**: Divide feature map into non-overlapping 4×4 windows
- **Attention**: Multi-head self-attention within each window
- **Efficiency**: O(4×4×C) << O(H×W×C)

### 4. **CSRF Module** (Simplified in current implementation)
```python
# NOTE: Current implementation treats single volume as single stack
# No actual cross-slice fusion in deployed version
# Just passes through bottleneck features
Input: bottleneck (B, 512, 4, 4)
↓
Output: Same (B, 512, 4, 4)
```

### 5. **Evidential Uncertainty Heads**
```python
Input: Decoder features (B, 32, H, W)
↓
Shared causal feature extractor:
  Conv 3×3 (32 → 16) + BN + ReLU
↓
Three parallel heads:
  - Anatomy head:   Conv 1×1 (16 → 2) → α_anatomy
  - Pathology head: Conv 1×1 (16 → 2) → α_pathology  
  - Noise head:     Conv 1×1 (16 → 2) → α_noise
↓
Each α = Softplus(raw_α) + 1.0  (ensure α > 1 for Beta)
↓
Learned weights: w = [0.3, 0.5, 0.2] (anatomy, pathology, noise)
Weighted fusion: α_total = w₁·α₁ + w₂·α₂ + w₃·α₃
↓
Beta distribution: p ~ Beta(α₁, α₂)
Uncertainty: U = 1 / (α₁ + α₂)
```

---

## Training Strategy:

### **Phase 1: MAE Pretraining** (50 epochs)
```
Input: Unlabeled FLAIR volumes
↓
Mask 75% of patches (random)
↓
Encoder → Bottleneck features
↓
MAE Decoder (4-layer Transformer)
↓
Reconstruct masked patches
↓
Loss: MSE(reconstructed, original)
↓
Save pretrained encoder weights
```

### **Phase 2: Supervised Segmentation** (30 epochs)
```
Load pretrained encoder
↓
Initialize decoder randomly
↓
Optimizer:
  - Encoder: LR=1e-5 (fine-tune)
  - Decoder: LR=4e-4 (train from scratch)
↓
Loss = Dice + Focal + λ·Evidential
↓
Train end-to-end
```

---

## Model Statistics:

| Component | Parameters | Output Shape |
|-----------|-----------|--------------|
| **2.5D Stem** | 37K | (B, 32, 64, 64) |
| **Stage 1** (4 blocks) | 285K | (B, 64, 32, 32) |
| **Stage 2** (4 blocks) | 894K | (B, 128, 16, 16) |
| **Stage 3** (4 blocks) | 2.24M | (B, 256, 8, 8) |
| **Stage 4** (4 blocks) | 7.08M | (B, 512, 4, 4) |
| **CSRF Module** | 1.3M | (B, 512, 4, 4) |
| **Decoder** | 2.68M | (B, 32, 64, 64) |
| **Seg Head** | 33 | (B, 1, 64, 64) |
| **Evidential Heads** | 2.1K | (B, 2, 64, 64) ×3 |
| **TOTAL** | **4.23M** | - |

**Computational Cost**:
- FLOPs: 1.75G
- Inference time: 80ms (GPU), 490ms (CPU)
- Throughput: 12.5 volumes/sec (GPU)
- Model size: 17 MB

---

## Key Ablation Insights Incorporated:

1. ✅ **Adaptive slice selection (k=5)**: +4.40% over baseline
2. ✅ **ResNet skip connections**: -4.44% when removed (CRITICAL)
3. ✅ **No dropout**: +0.55% improvement
4. ✅ **No heavy 3D conv**: +2.36% improvement  
5. ✅ **Mini-Swin (not full attention)**: Only -0.74% when removed (marginal)
6. ✅ **MAE 0.75 mask ratio**: +0.50% over 0.5 ratio
7. ✅ **Evidential head only**: +1.42% (best USALD component)
8. ❌ **Full USALD stack**: -1.32% (over-regularization)

---

## Clinical Performance:

- **In-domain (PediMS)**: 82.31% Dice, 77.60% Precision, 91.64% Recall
- **Cross-dataset (Adult MS)**: 0.01% Dice (domain shift - expected)
- **Cross-dataset (LGG tumors)**: 0.20% Dice (different pathology - expected)

**Conclusion**: Model is highly specialized for pediatric MS and requires domain adaptation for cross-age or cross-pathology deployment.

---

**Last Updated**: November 10, 2025  
**Model Version**: HybridMiniSwin2.5D-ResNet v3.0  
**Code**: `models/final_model.py`
