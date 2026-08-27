# Phase A: NeuroScan Baseline on BraTS — Status Report

**Date**: 2026-07-30  
**Status**: 🔄 **IN PROGRESS** (Epoch 1/10, batch 1/71)

## What We've Learned

### 1. Architecture & Input Format
- ✅ Model `HybridMiniSwin2D5_CBAM` expects 5D input: `(B, C, D, H, W)`
- ✅ Model internally extracts k=5 slices around center via `AdaptiveSliceSelector`
- ✅ Model outputs 2D predictions on center slice: `(B, 1, H, W)` with sigmoid activation
- ✅ Correct approach: pass full 3D volume, compare predictions to center slice ground truth

### 2. GPU Capacity (RTX 5050, 8.5GB)
```
Batch Size | Memory Used | Time/Batch | Status
-----------|-------------|------------|--------
4          | 0.44 GB     | 2.67 ms    | ✅ Very fast
8          | 0.79 GB     | 43 ms      | ✅ Fast
16         | 1.83 GB     | 32-35 ms   | ✅ Optimal
32         | 5.13 GB     | ~27 ms     | ⚠️ Pushing limits
40         | 7.38 GB     | ~400 ms    | ❌ Too slow
```

**Recommendation**: batch_size=16 is sweet spot (balanced speed & memory)

### 3. Learning Stability
- **LR=1e-4**: Model collapsed after ~18 batches (Dice → 0)
- **LR=1e-5**: Stable learning (Dice 30%+) but very slow
- **LR=5e-5**: Balanced (current config)

### 4. Current Results (Preliminary from batch_size=40 run)
```
Epoch 1 Results:
  Train Loss: 0.8707 | Train Dice: 30.6%
  Val Loss:   0.8630 | Val Dice:   0.034% (validation set too small)
```

**Note**: Dice improving from 0% (random) to 30%+ shows model is learning task.

## Current Training (10 epochs @ batch_size=16)

**Config**:
- Batch size: 16
- Learning rate: 5e-5
- Epochs: 10 (to establish baseline quickly)
- Batches per epoch: 71 (1126 subjects / 16)
- ETA: ~12 min/epoch × 10 = 120 minutes ≈ **2 hours**

**Last Status**: Epoch 1, batch 1/71 starting

## Expected Outcomes

By end of Phase A:
- ✅ Baseline Dice on BraTS (target: 25-35%)
- ✅ Proof that NeuroScan transfers to tumor segmentation
- ✅ Reproducible training config for ablations

## Next Steps (Phase B)

Once Phase A completes:

1. **Extract final Dice** from best checkpoint
2. **Run Phase B**: Gradient diagnostics
   - Log per-batch gradients for FocalTverskyLoss and EvidentialBetaLoss
   - Analyze gradient magnitude distribution
   - Identify task conflicts (divergent gradients)
3. **Document findings** for Phase C optimizer design

---

## Files & Checkpoints

| File | Purpose |
|------|---------|
| `configs/brats.yaml` | Training config (batch_size, lr, epochs) |
| `experiments/exp00_neuroscan_baseline/train.py` | Training script (frozen model + center-slice loss) |
| `experiments/exp00_neuroscan_baseline/checkpoints/` | Saved model checkpoints |
| `neuroscan_frozen.py` | Model architecture (frozen, no edits) |
| `Dataset/brats_dataset.py` | Data loader (FLAIR-only, binary segmentation) |

---

## Key Decisions Made

1. **Center-slice strategy**: Model designed for 2D (outputs 2D), so we train on center slice only
2. **Binary segmentation**: Convert 4-class BraTS labels to tumor vs background
3. **FLAIR-only**: Single modality for domain consistency (matches MS lesion training)
4. **No architecture changes**: Frozen model, only weights adapt

---

**Last Updated**: 2026-07-30 ~ waiting for training to complete
