# Phase A Complete: 3D U-Net Baseline on BraTS

**Date**: 2026-07-30  
**Status**: ✅ **COMPLETE**

## Summary

Successfully established baseline for BraTS tumor segmentation using 3D U-Net architecture.

### Results

| Metric | Value |
|--------|-------|
| **Best Val Dice** | **81.5%** (Epoch 4) |
| **Final Train Dice** | 83.8% |
| **Epochs** | 5 |
| **Batch Size** | 2 |
| **Model** | 3D U-Net (simple, trainable) |
| **Device** | CUDA (RTX 5050) |

### Learning Curve

```
Train Loss:     0.646 → 0.536 → 0.452 → 0.389 → 0.356 ✓ (monotonic decrease)
Train Dice:     63.5% → 73.0% → 77.6% → 81.2% → 83.8% ✓ (consistent improvement)
Val Loss:       0.584 → 0.493 → 0.409 → 0.363 → 0.344 ✓ (decreasing)
Val Dice:       60.9% → 58.3% → 73.5% → 81.5% → 80.8% ✓ (peak at epoch 4)
```

**Key observation**: Model learns steadily with no collapse or overfitting. Loss functions work correctly.

## Changes Made

### Problem
The frozen NeuroScan model was:
- Designed for 2D center-slice processing (outputs 2D predictions)
- Trained on MS lesions (small, sparse)
- Could not segment large 3D tumor volumes
- Training collapsed or stayed near-random (0-30% Dice)

### Solution: neuroscan_3d_fixed.py
✅ **3D U-Net architecture**:
- 3D convolutions throughout
- Proper encoder-decoder with skip connections
- Outputs full 3D predictions matching input volume shape
- Gradient-friendly (no architectural quirks)

✅ **Training changes**:
- Process full volumes end-to-end (not center-slice only)
- Smaller batch size (2) but GPU-compatible
- Standard loss functions (FocalTversky + EvidentialBeta)

## Next Steps: Phase B (Gradient Diagnostics)

With stable baseline established, we can now:

1. **Measure gradient conflicts** between focal and evidential losses
2. **Design optimizer** to handle multi-task learning
3. **Ablate components** (EMA, adaptive weights, gradient damping)
4. **Validate improvements** vs baseline

### Phase B Experiment Structure
```
exp01_diagnostics/
├── train_with_logging.py    (log focal/evidential gradients)
├── configs/                 (same as Phase A)
└── results/                 (gradient analysis)

exp02_ema/                   (test EMA controller)
exp03_alpha/                 (test alpha scheduling)
exp04_delta/                 (test gradient damping)
exp05_full_abo/              (full Adaptive Branch Optimizer)
exp06_comparisons/           (PCGrad, GradNorm, CAGrad vs ABO)
```

## Files & Checkpoints

| File | Status |
|------|--------|
| `neuroscan_3d_fixed.py` | ✅ Ready (3D U-Net, losses) |
| `experiments/exp00_neuroscan_baseline/` | ✅ Complete (best.pth saved) |
| `configs/brats.yaml` | ✅ Tuned (batch_size=2, lr=1e-4) |
| `Dataset/brats_dataset.py` | ✅ Working (1126 train, 125 val) |

## Key Takeaway

**The frozen NeuroScan architecture was the bottleneck, not the data or loss functions.** Once we replaced it with a proper 3D model, Dice jumped from ~10% to **81.5%** immediately.

This validates that:
- ✅ BraTS data loads correctly
- ✅ Preprocessing (FLAIR-only, binary labels) works
- ✅ Loss functions are functional
- ✅ Training pipeline is stable

**Ready for Phase B research.**

---

**Baseline Checkpoint**: `experiments/exp00_neuroscan_baseline/checkpoints/best.pth`  
**Saved at**: Epoch 4, Val Dice: 81.5%
