# Phase 3: Training Pipeline Mapping

**Date**: 2026-07-27  
**Status**: Complete training flow documented  
**Scope**: Loss computation, gradient flow, parameter interactions

---

## High-Level Training Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAIN EXECUTION (lines 1573+)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PHASE 1: MAE PRETRAINING (200 epochs)                           │
│  ├─ Initialize encoder with adaptive slice selection             │
│  ├─ Initialize MAE (encoder + decoder)                           │
│  ├─ For each epoch:                                              │
│  │  ├─ train_mae_epoch()                                         │
│  │  │  ├─ Forward: x → features → mask 75% → reconstruction      │
│  │  │  ├─ Loss: L1 on masked positions only                      │
│  │  │  ├─ Backward: compute gradients                            │
│  │  │  ├─ Optimizer step (AdamW)                                 │
│  │  │  └─ LR scheduler: CosineAnnealing                          │
│  │  ├─ Save: local resume + best model to Drive                  │
│  │  └─ Early stopping: patience=30                               │
│  └─ Result: Pretrained encoder saved to mae_best.pth             │
│                                                                   │
│  PHASE 2: SEGMENTATION FINE-TUNING (80 epochs)                   │
│  ├─ Load pretrained encoder from MAE                             │
│  ├─ Initialize segmentation model (CBAM architecture)            │
│  ├─ Freeze adaptive slice selector (preserves MAE knowledge)      │
│  ├─ Initialize teacher model (EMA copy) if USALD enabled         │
│  ├─ Separate optimizers: encoder (lr=1e-5), decoder (lr=4e-4)    │
│  ├─ For each epoch:                                              │
│  │  ├─ train_segmentation_epoch()                                │
│  │  │  ├─ Forward: x → features → predictions + alpha            │
│  │  │  ├─ Loss computation (see below)                           │
│  │  │  ├─ Backward: compute gradients                            │
│  │  │  ├─ Optimizer step (AdamW)                                 │
│  │  │  ├─ EMA update teacher (if consistency enabled)            │
│  │  │  └─ Compute Dice metric                                    │
│  │  ├─ validate_segmentation()                                   │
│  │  │  ├─ Forward with optional self-correction                  │
│  │  │  ├─ Compute Dice, Precision, Recall, F1                   │
│  │  │  └─ Estimate FDR threshold (if enabled)                    │
│  │  ├─ Save: local resume + best model to Drive                  │
│  │  └─ Early stopping: patience=20                               │
│  └─ Result: Best model saved to best_model.pth                   │
│                                                                   │
│  DEPLOYMENT PACKAGE: Create model-only checkpoint                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## PHASE 1: MAE PRETRAINING

### `train_mae_epoch()` (lines 1302-1333)

**Input**: 
- `model`: MAE_2D5 (encoder + decoder)
- `loader`: DataLoader with images
- `optimizer`: AdamW
- `scaler`: GradScaler for AMP

**Forward Pass**:
```python
reconstruction, mask, bottleneck = model(images)
# images: (B, 1, D, H, W) → Full 3D volume
# encoder(images):
#   ├─ Adaptive selector scores slices
#   ├─ Selects k=9 most informative slices → (B, 1, k, H, W)
#   ├─ 2.5D stem processes selected slices → (B, C_0, H, W)
#   └─ 4 stages: encoder blocks → bottleneck (B, C_4, H', W')
# random_masking(bottleneck, 0.75):
#   └─ Masks 75% of spatial positions
# decoder(masked_tokens):
#   └─ Reconstructs masked positions
```

**Loss Computation** (lines 1314-1323):
```python
target = bottleneck.flatten(2).transpose(1, 2)  # (B, H*W, C)
mask_expanded = mask.unsqueeze(-1)              # (B, H*W, 1)
loss = F.l1_loss(reconstruction * mask_expanded, 
                  target * mask_expanded)
# L1 loss only on MASKED positions
# Unmasked positions are ignored (self-supervised)
```

**Backward Pass**:
```python
optimizer.zero_grad()
scaler.scale(loss).backward()      # Scaled for AMP
scaler.step(optimizer)
scaler.update()
```

**Gradient Flow**:
- Gradients flow through: decoder → encoder → all parameters
- Adaptive selector LEARNS slice scoring (not frozen yet)
- ALL encoder parameters updated during MAE

**Output**:
- Pretrained encoder weights saved to `mae_best.pth`
- Used to initialize segmentation encoder

---

## PHASE 2: SEGMENTATION FINE-TUNING

### `train_segmentation_epoch()` (lines 1335-1445)

**Setup** (lines 1748-1759):
```python
encoder_params = encoder.parameters() + cbam.parameters()
decoder_params = decoder.parameters()

optimizer = AdamW([
    {'params': encoder_params, 'lr': 1e-5},     # Fine-tune
    {'params': decoder_params, 'lr': 4e-4}      # Train from scratch
])
```

**Critical**: Adaptive slice selector **frozen** (line 1732):
```python
seg_model.encoder.freeze_adaptive_selector()
# adapter_selector.requires_grad = False
# Preserved MAE-learned slice selection
```

### Loss Computation (lines 1375-1417)

```
Input: (B, 1, D, H, W)
         ↓
Forward pass → model(images)
         ↓
Extract: center_slice_label = labels[:, :, labels.shape[2]//2, :, :]
         ↓
Compute losses (see below)
         ↓
Combine: total_loss = supervised + evidential + consistency + pseudo
```

#### A. **SUPERVISED LOSS** (lines 1375-1387)

```python
out_dict = model(images)  # Forward pass
probs = out_dict["probs"]  # (B, 1, H, W) - central slice output

# Hybrid Loss: 0.5*Dice + 0.5*FocalTversky
loss = criterion(probs, center_slice_label)  # line 1381

# Criterion initialized (line 1758):
criterion = HybridLoss(lambda1=0.5, lambda2=0.5)

# HybridLoss.__forward__ (lines 1021-1035):
#   ├─ dice_loss = DiceLoss(probs, label)
#   ├─ ft = FocalTverskyLoss(probs, label)
#   └─ return 0.5*dice + 0.5*ft
```

**Dice Loss Behavior**:
- Punishes false negatives AND false positives equally
- Dice = 2*TP / (2*TP + FP + FN)

**FocalTversky Loss Behavior**:
- Focal Tversky = (1 - Tversky)^γ where γ=0.75
- Tversky = TP / (TP + α*FP + β*FN) with α=0.3, β=0.7
- Weights false negatives 2.33× more than false positives
- Clinical: Better recall (detects more lesions)

#### B. **EVIDENTIAL UNCERTAINTY LOSS** (lines 1383-1392)

**Enabled** (config line 193): `USALD_ENABLED = True`

```python
if USALD_ENABLED and ("alpha" in out_dict):
    loss_evid = evid_criterion(out_dict["alpha"], center_slice_label)
    loss = loss + loss_evid
    
    # evid_criterion = EvidentialBetaLoss(lambda_kl=1e-3)
    
    # out_dict["alpha"]: (B, 2, H, W)
    #   ├─ [:, 0]: α_0 (Beta parameter for "not lesion")
    #   └─ [:, 1]: α_1 (Beta parameter for "lesion")
```

**EvidentialBetaLoss.__forward__** (lines 1046-1060):
```python
alpha0, alpha1 = alpha[:, 0:1], alpha[:, 1:2]
S = alpha0 + alpha1                           # Total evidence
p = alpha1 / (S + 1e-8)                       # Predicted probability

# Data fit: MSE weighted by inverse certainty
mse = (p - target).pow(2)
inv_cert = 1.0 / (S + 1.0)
fit = (mse * inv_cert).mean()

# KL divergence regularizer (push toward Beta(1,1) = uniform)
kl = KL_divergence(Beta(alpha0, alpha1), Beta(1, 1))

# Total: fit + 1e-3 * kl
return fit + 0.001 * kl
```

**Effect**: 
- Encourages model to express uncertainty via α parameters
- Calibrates predictions: high S (evidence) → confident predictions
- +1.16% Dice improvement over baseline

#### C. **CONSISTENCY LOSS** (lines 1394-1403)

**Enabled** (config line 194): `USALD_CONSISTENCY_ENABLED = False` ← **DISABLED**

When enabled (after warmup):
```python
if use_consistency:  # epoch > WARMUP_EPOCHS (10)
    with torch.no_grad():
        teacher_dict = teacher_model(images)
        teacher_probs = teacher_dict["probs"]
    
    loss_cons = consistency_loss(probs, teacher_probs, gamma=3.0)
    loss = loss + LAMBDA_CONSISTENCY * loss_cons  # LAMBDA=1.0
```

**consistency_loss()** (lines 1191-1217):
```python
# Teacher uncertainty: u ∈ [0,1], u=1 when teacher uncertain (p≈0.5)
uncertainty = 1.0 - 2.0 * abs(teacher_probs - 0.5)

# Weight by teacher confidence: exp(-3*u)
# → 1.0 when teacher confident, → exp(-3) when uncertain
weight = exp(-3 * uncertainty)

# Weighted MSE
mse = (student_probs - teacher_probs)^2
return (weight * mse).mean()
```

**Effect**: Forces student to match teacher only when teacher is confident

**Teacher Update** (lines 1424-1426):
```python
if use_consistency:
    ema_update(model, teacher_model, decay=0.99)
    # θ_teacher ← 0.99*θ_teacher + 0.01*θ_student
```

#### D. **PSEUDO-LABEL LOSS** (lines 1405-1417)

**Enabled** (config line 195): `USALD_FDR_ENABLED = False` ← **DISABLED**

When enabled (after warmup + FDR update):
```python
if use_consistency and USALD_FDR_ENABLED:
    with torch.no_grad():
        # Create pseudo-labels from high-confidence teacher predictions
        # using adaptive threshold τ_pl
        pseudo_mask = (teacher_probs >= tau_pl) | (teacher_probs <= 1-tau_pl)
        pseudo_labels = (teacher_probs >= tau_pl).float()
    
    if pseudo_mask.sum() > 0:
        # BCE only on pseudo-labeled pixels
        loss_pl = F.binary_cross_entropy(
            probs * pseudo_mask, 
            pseudo_labels * pseudo_mask
        ) / (pseudo_mask.sum() + 1e-7)
        
        loss = loss + LAMBDA_PSEUDO * loss_pl  # LAMBDA=0.5
```

**Threshold Adaptation** (lines 1839-1846):
```python
if epoch > WARMUP_EPOCHS and USALD_FDR_ENABLED and epoch % 5 == 0:
    tau_pl = estimate_fdr_threshold(val_probs, val_labels, q=0.10)
    # Adaptive threshold: precision ≥ 90% (FDR ≤ 10%)
```

---

### **ACTUAL LOSS COMBINATION (BASELINE CONFIGURATION)**

Given the enabled flags:
- ✅ HybridLoss (Dice + FocalTversky)
- ✅ EvidentialBetaLoss
- ❌ ConsistencyLoss (disabled)
- ❌ PseudoLabelLoss (disabled)

**Actual formula during training**:
```
Loss_total = Loss_hybrid + Loss_evidential
           = (0.5*Dice + 0.5*FocalTversky) + (MSE_fit + 0.001*KL)
```

**No loss interference** (since consistency/pseudo-label disabled)

---

### Backward Pass & Gradient Merging (lines 1419-1422)

```python
optimizer.zero_grad()

scaler.scale(loss_total).backward()
# Backprop through:
#   loss_total → loss_hybrid → Dice, FocalTversky
#   loss_total → loss_evidential → alpha parameters
#   All gradients merge at parameter level

scaler.step(optimizer)        # Apply gradient step
scaler.update()               # Update AMP scaler

# Separate LR updates:
# encoder_params: lr = 1e-5 (fine-tune)
# decoder_params: lr = 4e-4 (train from scratch)
```

**Gradient Flow**:
```
Input (B,1,D,H,W)
    ↓
Adaptive Selector (FROZEN - no gradients)
    ↓
2.5D Stem ← gradients from both losses
    ↓
Encoder Blocks (4 stages) ← gradients from both losses
    ↓
CBAM Bottleneck Fusion ← gradients
    ↓
Decoder Blocks ← gradients (mainly from supervision, some from evidential)
    ↓
Probs head ← gradients from Loss_hybrid
Alpha head ← gradients from Loss_evidential
```

---

### Validation (lines 1447-1542)

```python
def validate_segmentation(model, loader, criterion, device):
    model.eval()
    
    for batch in loader:
        images, labels = batch
        center_slice_label = labels[:, :, labels.shape[2]//2, :, :]
        
        with torch.no_grad():
            out_dict = model(images)
            probs = out_dict["probs"]
            
            # Loss (for monitoring, not gradients)
            loss = criterion(probs, center_slice_label)
        
        # Metrics
        pred_binary = (probs > 0.5).float()
        dice = 2*(pred_binary*label).sum() / (pred_binary.sum() + label.sum())
        
        # Accumulate for precision, recall, F1
    
    # FDR threshold estimation (if enabled)
    if epoch > WARMUP_EPOCHS and USALD_FDR_ENABLED:
        return metrics, val_probs, val_labels
    else:
        return metrics
```

---

## Critical Design Decisions

### 1. **Adaptive Selector Freezing** (line 1732)

```python
seg_model.encoder.freeze_adaptive_selector()
```

**Why**: 
- MAE learns which slices are informative
- Freezing preserves this learned selection
- Prevents catastrophic forgetting during fine-tuning
- Enables interpretability: can visualize learned slice importance

**Effect**: 
- Fewer parameters to optimize during fine-tuning
- Stable slice selection throughout training
- Faster convergence

### 2. **Separate Learning Rates** (lines 1752-1755)

```python
optimizer = AdamW([
    {'params': encoder_params, 'lr': 1e-5},    # Fine-tune pretrained
    {'params': decoder_params, 'lr': 4e-4}     # Train from scratch
], weight_decay=0.01)
```

**Why**:
- Encoder already learned from MAE: small LR prevents forgetting
- Decoder is new: needs larger LR to learn lesion patterns
- 40× difference reflects different initialization states

### 3. **Loss Weight Balance** (line 1758)

```python
criterion = HybridLoss(lambda1=0.5, lambda2=0.5)  # 50/50 Dice vs FocalTversky
LAMBDA_EVIDENTIAL = 1e-3                          # Evidential is 1/500th
```

**Why**:
- Dice and FocalTversky are equally important (validated by ablation)
- Evidential is tiny regularizer (adds +1.16% but shouldn't dominate)
- No loss fighting: evidential is additive, not competitive

### 4. **Center Slice Extraction** (line 1373)

```python
center_slice_label = labels[:, :, labels.shape[2]//2, :, :]
```

**Why**:
- Model outputs 2D segmentation (central slice)
- But input is full 3D volume (for 2.5D context)
- 2.5D architecture: uses neighboring slices to inform central prediction
- Standard in volumetric segmentation with 2D outputs

---

## Parameter Influence Matrix

| Loss Component | Influences | Weight | Epoch |
|---|---|---|---|
| **Dice Loss** | decoder.prob_head, all encoder layers | 0.5 | 1-80 |
| **FocalTversky** | decoder.prob_head, all encoder layers | 0.5 | 1-80 |
| **Evidential** | decoder.alpha_heads, all encoder layers | 0.001 | 1-80 |
| **Consistency** | decoder.prob_head, encoder | 1.0 (if enabled) | 11-80 |
| **Pseudo-label** | decoder.prob_head | 0.5 (if enabled) | 11-80 |

**Frozen**: 
- Adaptive slice selector (all epochs)
- Teacher model parameters (updated via EMA, not backprop)

---

## Checkpointing Strategy

**Local (not synced)**:
```
.resume_checkpoints_frozen_*/
├─ mae_resume.pth       (latest MAE checkpoint)
└─ seg_resume.pth       (latest segmentation checkpoint)
```

**Google Drive (synced)**:
```
OptimalModel_FrozenSelector_{timestamp}/
├─ mae_pretraining/
│  ├─ mae_best.pth      (lowest MAE loss)
│  └─ mae_logs.csv
├─ segmentation/
│  ├─ best_model.pth    (highest validation Dice)
│  ├─ train_logs.csv
│  └─ val_logs.csv
└─ deployment/
   └─ model.pth         (deployment-ready)
```

---

## Summary: What's Actually Happening

### ✅ Correct Behavior
1. MAE pretains encoder with 75% masking
2. Encoder learns adaptive slice selection during MAE
3. Segmentation loads pretrained encoder + freezes selector
4. Training uses HybridLoss (Dice + FocalTversky) + small Evidential regularizer
5. No loss interference (consistency/pseudo disabled)
6. Separate learning rates preserve encoder knowledge
7. Teacher model disabled (no consistency loss)

### ⚠️ Disabled but Implemented
1. **Consistency Loss**: Would match student to teacher after warmup
2. **Pseudo-Label Loss**: Would use high-confidence teacher predictions
3. **FDR Threshold Adaptation**: Adaptive confidence thresholding
4. **Self-Correction at Inference**: Iterative refinement
5. **Causal Decomposition**: 3-factor uncertainty decomposition

These are fully implemented but turned off via config flags.

---

**Ready for Phase 4: Instrumentation and Diagnostics**
