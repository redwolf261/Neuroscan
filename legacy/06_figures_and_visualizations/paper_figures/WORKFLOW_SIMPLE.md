# HybridMiniSwin2.5D-CBAM: Pipeline Workflow

**Source Code**: `C:\Users\HP\EDI\final_model.py`  
**Model**: HybridMiniSwin2.5D-ResNet with CBAM and Evidential Uncertainty (OPTIMAL CONFIGURATION)  
**Performance**: 82.31% Dice Score  
**Output Directory**: `C:\Users\HP\EDI\OptimalModel_Evidential`

---

## Workflow Overview

```
Data → Preprocessing → MAE Pretraining → Segmentation Model → Training → Evaluation
```

---

## Stage 1: Data Acquisition & Preprocessing

**Input**: PediMS Dataset (63 patients)
- T1-weighted MRI
- T2-weighted MRI  
- FLAIR MRI

**Preprocessing Steps**:
1. N4 Bias Correction
2. Registration (T1 space)
3. Intensity Normalization

**Output**: 3-channel volume (3, 64, 64, 64)

---

## Stage 2: MAE Pretraining

**Random Masking**: 75% of input patches masked

**MAE Encoder**: 2.5D ResNet + Swin Transformer
- Adaptive slice selector (k=9)
- ResNet blocks with skip connections
- Mini-Swin attention (W=4×4)

**MAE Decoder**: Lightweight transformer
- Reconstructs masked patches
- 30 epochs, AdamW optimizer

**Output**: Pretrained encoder weights

---

## Stage 3: Model Architecture

**Adaptive Slice Selector** → Selects k=9 most informative slices

**3-Channel Input** → (3 modalities, 9 slices, 64×64)

**Encoder**:
- Swin Transformer blocks (W=4×4)
- 2.5D Conv + Residual blocks
- 4 stages: [32→64→128→256→512 channels]

**Multi-Scale Features**: F₁, F₂, F₃, F₄

**CBAM Fusion Module**:
- Channel attention
- Spatial attention  
- Feature refinement

**Decoder**:
- 3 Upsample blocks with skip connections
- Progressive upsampling: 4×4 → 8×8 → 16×16 → 32×32 → 64×64

**Output Heads**:
- Evidential Head (β₀, α)
- Segmentation Map (64×64)

---

## Stage 4: Training

**Loss**: Dice + BCE

**Optimizer**: AdamW (LR=3e-4)

**Configuration**:
- 48 epochs
- Batch size: 3
- Mixed precision (AMP)

---

## Stage 5: Evaluation

**Validated Results** (Epoch 12 - Best Model):
- **Dice**: 82.31% (0.8231)
- **Precision**: 77.16% (0.7716)
- **Recall**: 88.17% (0.8817)
- **F1**: 82.30% (0.8230)



---

## Configuration From final_model.py

```python
# Optimal hyperparameters (empirically validated)
K_SLICES = 9                 # Adaptive selection (k=9)
SPATIAL_SIZE = (64, 64, 64)  # Input size
MINI_SWIN_WINDOW = 4         # Window=4×4
STAGE_CHANNELS = [32, 64, 128, 256, 512]
BATCH_SIZE = 3
MAE_MASK_RATIO = 0.75        # 75% masking
MAE_EPOCHS = 200
SEGMENTATION_EPOCHS = 80
USALD_ENABLED = True         # Evidential only
```

**Ablation Results**:
- k=9: +4.40% vs baseline
- Window=4: Best for small datasets
- CBAM: +0.48% vs CSRF
- Evidential: +1.16% improvement
- MAE 0.75: 86.5% vs 86.0% for 0.5

---

**Diagram**: `paper_figures/workflow_pipeline_comprehensive.png`  
**Trained Model**: `OptimalModel_Evidential/checkpoints/best_model_fold0.pth`
