# HybridMiniSwin2.5D-CBAM: Comprehensive Pipeline Workflow Description

**Based on**: `final_model.py` (Root directory)  
**Model**: HybridMiniSwin2.5D-CBAM with Evidential Uncertainty  
**Performance**: 82.31% Dice Score (In-domain validation)

---

## Pipeline Overview

The complete training and deployment pipeline consists of five major stages:

1. **Data Acquisition & Preprocessing**
2. **MAE Pretraining**  
3. **Model Architecture (HybridMiniSwin2.5D-CBAM)**
4. **Training**
5. **Evaluation**

---

## Stage 1: Data Acquisition & Preprocessing

### Input Data
- **Dataset**: PediMS (Pediatric Multiple Sclerosis)
- **Size**: 63 patients
- **Modalities**: 
  - T1-weighted MRI
  - T2-weighted MRI
  - FLAIR MRI

### Preprocessing Pipeline
1. **N4 Bias Correction**
   - Removes intensity inhomogeneity artifacts from MRI scans
   - Ensures consistent intensity distributions across volumes

2. **Registration (T1 space)**
   - Aligns all modalities (T1w, T2w, FLAIR) to T1 space
   - Ensures spatial correspondence across sequences

3. **Intensity Normalization**
   - Z-score normalization per volume
   - Standardizes intensity ranges for stable training

**Output**: Preprocessed 3-channel volume (B, 3, D, H, W) where D=64, H=64, W=64

---

## Stage 2: MAE Pretraining (Masked Autoencoder)

### Purpose
Self-supervised pretraining to learn robust feature representations from limited labeled data.

### Architecture Components

#### Random Masking
- **Mask Ratio**: 75% (optimal from ablation study)

#### MAE Encoder
- **Architecture**: 2.5D ResNet + Swin Transformer hybrid
- **Components**:
  - Adaptive slice selector (selects k=9 most informative slices)
  - ResNet blocks with skip connections
  - Mini-Swin attention windows (W=4×4)
- **Output**: Bottleneck features (B, 512, 4, 4) from central slices

#### MAE Decoder
- **Architecture**: Lightweight transformer decoder
- **Parameters**:
  - Decoder embedding dimension: 256
  - Number of blocks: 4
  - Attention heads: 4
- **Function**: Reconstructs original patches from masked bottleneck features

#### Reconstruction
- **Target**: Reconstruct masked 75% of input patches
- **Loss**: MSE (Mean Squared Error) between predicted and original patches
- **Training**: 
  - 30 epochs MAE pretraining
  - Batch size: 3
  - Optimizer: AdamW (LR=1e-3)

**Output**: Pretrained encoder weights transferred to segmentation model

---

## Stage 3: Model Architecture (HybridMiniSwin2.5D-CBAM)

### Novel Components

#### 1. Adaptive Slice Selector (Novel Contribution)
- **Function**: Learns to select k most informative slices from full 3D volume
- **Parameters**: k=9 slices (optimal from ablation: +4.40% improvement)
- **Architecture**:
  - Lightweight 3D CNN scorer
  - Attention-based slice importance scoring
  - Top-k selection with spatial ordering preserved
- **Advantage**: +3% improvement over fixed center slice selection
- **Input**: (B, 1, D, H, W) full volume
- **Output**: (B, 1, k, H, W) selected slices

#### 2. 3-Channel Input
- **Format**: (B, 3, 9, H, W) where 3 = modalities (T1w, T2w, FLAIR)
- **Slices**: 9 adaptively selected slices per volume
- **Spatial size**: H=64, W=64

### Encoder Path

#### Swin Transformer Blocks
- **Window Size**: 4×4 (optimal for small datasets, prevents overfitting)
- **Ablation Result**: W=4 > W=16 > W=8
- **Function**: Captures local-global context through shifted window attention

#### 2.5D Conv + Residual
- **Architecture**: ResNet-style blocks with skip connections
- **Critical Finding**: Removing residual connections → -4.44% performance drop
- **Stages**: 4 stages with progressive downsampling
  - Stage 1: 32 → 64 channels, 64×64 → 32×32
  - Stage 2: 64 → 128 channels, 32×32 → 16×16
  - Stage 3: 128 → 256 channels, 16×16 → 8×8
  - Stage 4: 256 → 512 channels, 8×8 → 4×4

#### Mini-Swin Attention
- **Heads**: 4 attention heads
- **Window**: 4×4 local windows
- **Type**: Multi-head self-attention (MSA)

### Multi-Scale Features
- **F₁ (High)**: 64 channels @ 32×32 resolution
- **F₂ (Mid)**: 128 channels @ 16×16 resolution
- **F₃ (Mid)**: 256 channels @ 8×8 resolution
- **F₄ (Low)**: 512 channels @ 4×4 resolution (bottleneck)

### CBAM Fusion Module (Key Innovation)

#### Function
Convolutional Block Attention Module for feature refinement at bottleneck.


#### Architecture

**Channel Attention**:
- Adaptive average pooling
- Adaptive max pooling
- Shared FC layers with reduction ratio=4
- Sigmoid activation
- Output: Channel-wise attention weights

**Spatial Attention**:
- Channel-wise average and max pooling
- 7×7 convolution
- Sigmoid activation
- Output: Spatial attention map

**Feature Refinement**:
- Sequential channel → spatial attention
- Residual connection with input features
- Adaptive feature recalibration

**Input**: F₄ bottleneck features (B, 512, 4, 4)  
**Output**: Refined bottleneck (B, 512, 4, 4)

### Decoder Path

#### Upsample Block 1
- **Input**: 512 channels @ 4×4
- **Output**: 256 channels @ 8×8
- **Skip**: F₃ from encoder
- **Operation**: Transposed conv + skip connection + Conv block

#### Upsample Block 2
- **Input**: 256 channels @ 8×8
- **Output**: 128 channels @ 16×16
- **Skip**: F₂ from encoder
- **Operation**: Transposed conv + skip connection + Conv block

#### Upsample Block 3
- **Input**: 128 channels @ 16×16
- **Output**: 64 channels @ 32×32
- **Skip**: F₁ from encoder
- **Operation**: Transposed conv + skip connection + Conv block

#### Final Upsampling
- **Input**: 64 channels @ 32×32
- **Output**: 32 channels @ 64×64
- Bilinear upsampling to original resolution

### Output Heads

#### Evidential Head (Uncertainty Estimation)
- **Type**: Evidential Deep Learning (EDL)
- **Distribution**: Beta distribution parameterization
- **Outputs**:
  - β₀: Beta concentration parameter
  - α: Evidence parameter for uncertainty quantification
- **Ablation Result**: Evidential-only → +1.16% improvement
- **Advantage**: Provides uncertainty estimates with minimal overhead

#### Segmentation Map
- **Format**: (B, 1, H, W) where H=64, W=64
- **Type**: Binary mask (lesion vs background)
- **Activation**: Sigmoid (from evidential probabilities)

---

## Stage 4: Training

### Loss Function
**Combined Loss**: Dice + BCE (Binary Cross-Entropy)

- **Dice Loss**: 
  - Optimizes overlap between prediction and ground truth
  - Handles class imbalance (lesions are sparse)
  
- **BCE Loss**: 
  - Pixel-wise binary classification loss
  - Provides stable gradients

**Total Loss**: λ₁ × Dice Loss + λ₂ × BCE Loss

### Optimizer
- **Type**: AdamW (Adam with weight decay)
- **Learning Rate**: 3e-4
- **Weight Decay**: 0.01
- **Betas**: (0.9, 0.999)


- **Best Model Selection**: Highest validation Dice score
- **Batch Size**: 3 (limited by 4GB GPU)
- **Mixed Precision**: AMP (Automatic Mixed Precision) enabled for speed

### Data Augmentation
- Random flips (horizontal/vertical)
- Random 90° rotations
- Applied during training only

---

## Stage 5: Evaluation

### Metrics

#### Primary Metric
- **Dice Score**: 82.31% ✓
  - Measures overlap between predicted and ground truth masks
  - Primary metric for medical segmentation

#### Supporting Metrics
- **Precision**: 77.60%
  - True Positives / (True Positives + False Positives)
  - Measures accuracy of positive predictions

- **Recall (Sensitivity)**: 91.64%
  - True Positives / (True Positives + False Negatives)
  - **Clinical Importance**: High sensitivity crucial for lesion screening

- **F1 Score**: 84.04%
  - Harmonic mean of Precision and Recall
  - Balanced measure of performance

### Validation Strategy
- **5-Fold Cross-Validation** (complete pipeline)
- **Train/Val Split**: 80/20 for each fold
- **Final Model**: Trained on fold with best validation performance (Fold 0, Epoch 12)

### Inference
- **Input**: Single 3D MRI volume (B, 3, D, H, W)
- **Output**: 2D segmentation mask for central slice (B, 1, H, W)
- **Uncertainty**: Evidential uncertainty map (optional)
- **Speed**: ~490ms ± 168ms per volume (CPU inference)

---

## Key Architecture Decisions (From Ablation Studies)

### Hyperparameters
| Parameter | Optimal | Ablation Result | Impact |
|-----------|---------|----------------|--------|
| **k_slices** | 9 | 72.15% vs 69.11% baseline | +4.40% |
| **window_size** | 4 | Best for small datasets | Prevents overfitting |
| **Fusion** | CBAM | 69.21% vs CSRF 68.73% | +0.48% |
| **Uncertainty** | Evidential only | 82.64% vs 81.48% baseline | +1.16% |
| **MAE mask ratio** | 0.75 | 86.5% vs 86.0% for 0.5 | Better pretraining |

### Architecture Ablations
| Component | Result | Conclusion |
|-----------|--------|------------|
| **ResNet skips** | Removal: -4.44% | **CRITICAL** for performance |
| **Dropout** | Removal: +0.55% | Not needed with batch norm |
| **3D Conv** | Removal: +2.36% | 2.5D sufficient, reduces params |
| **USALD (all 5)** | -1.08% | Component interference |
| **Evidential only** | +1.16% | Single component works best |

---

## Model Specifications

### Parameters
- **Total**: 4.23M parameters
- **Encoder**: ~85.6% of total params
- **Breakdown**:
  - ResNet blocks: ~2.8M
  - Swin attention: ~0.6M
  - CBAM module: ~0.1M
  - Decoder: ~0.6M
  - Heads: ~0.13M

### Computational Cost
- **FLOPs**: 1.75 GFLOPs
- **Inference Time**: 490 ± 168 ms (CPU)
- **Throughput**: ~2.0 volumes/second (CPU)
- **GPU Speedup**: ~10× faster on GPU

### Memory Requirements
- **Training**: ~3.8GB GPU memory (batch size=3)
- **Inference**: ~500MB (single volume)

---

## Clinical Significance

### Screening Performance
- **High Sensitivity (91.64%)**: Detects most lesions → suitable for screening
- **Acceptable Precision (77.60%)**: Minimizes false positives
- **Fast Inference**: <500ms enables real-time clinical use

### Uncertainty Quantification
- **Evidential Uncertainty**: Provides confidence estimates for predictions
- **Clinical Value**: Flags uncertain cases for radiologist review
- **Minimal Overhead**: No performance penalty

---

## Novel Contributions

1. **Adaptive Slice Selection**: +3% improvement by learning which slices contain most information

2. **2.5D Architecture**: Balances 3D context and computational efficiency
   - Better than 2D: Captures inter-slice relationships
   - Better than 3D: Lower memory, faster training

3. **CBAM Fusion**: Simple attention mechanism outperforms complex alternatives

4. **Evidential Uncertainty**: Single-component uncertainty with +1.16% performance gain

5. **Small Dataset Optimization**: 
   - Mini-Swin windows (4×4) prevent overfitting
   - MAE pretraining with 75% masking
   - No dropout (batch norm sufficient)

---

## Comparison to Baselines

### vs. Baseline (No optimizations)
- **Improvement**: +9.22% Dice
- **Components**: Adaptive selection + CBAM + Evidential + MAE

### vs. Alternative Fusion Methods
- **CBAM**: 69.21% ✓
- **No Fusion**: 69.06%
- **CSRF**: 68.73%
- **SE**: Lower

### Parameter Efficiency
- **This Model**: 4.23M params, 82.31% Dice
- **Ratio**: 19.4% Dice per million parameters
- **Advantage**: Lightweight, fast, deployable

---

## Training Configuration Summary

```python
# From final_model.py

# Hyperparameters (OPTIMAL)
K_SLICES = 9                    # Adaptive selection
SPATIAL_SIZE = (64, 64, 64)     # Input size
MINI_SWIN_WINDOW = 4            # Window size
STAGE_CHANNELS = [32, 64, 128, 256, 512]
BATCH_SIZE = 3

# MAE Pretraining
MAE_MASK_RATIO = 0.75           # 75% masking
MAE_EPOCHS = 30
MAE_LR = 1e-3

# Segmentation Training
SEG_EPOCHS = 48
SEG_LR = 3e-4
OPTIMIZER = AdamW
LOSS = Dice + BCE

# Uncertainty
USE_EVIDENTIAL = True           # Evidential only
USE_USALD_FULL = False          # Full USALD causes interference

# Architecture
USE_CBAM = True                 # CBAM fusion (optimal)
USE_ADAPTIVE_SELECTION = True   # Adaptive slice selection
```

---

## File References

- **Main Model**: `C:\Users\HP\EDI\final_model.py`
- **Workflow Diagram**: `C:\Users\HP\EDI\paper_figures\workflow_pipeline_comprehensive.png`
- **Results**: `C:\Users\HP\EDI\OptimalModel_Evidential\segmentation\val_logs.csv`
- **Best Checkpoint**: `C:\Users\HP\EDI\OptimalModel_Evidential\checkpoints\best_model_fold0.pth`

---

*Generated from `final_model.py` - HybridMiniSwin2.5D-CBAM architecture (OPTIMAL configuration)*
