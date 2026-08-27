# BraTS Baseline Setup — Complete

**Date**: 2026-07-29  
**Status**: ✅ Phase 1-3 Complete — Phase 2 (baseline training) in progress

---

## What Was Built

### Phase 1: Dataset Verification ✅
- [x] Froze dataset location: `Dataset\Training`
- [x] Verified 1,251 BraTS subjects (vs. 9 PediMS patients)
- [x] Confirmed file structure: 5 files per subject (4 modalities + segmentation)
- [x] Verified dimensions: 240×240×155, 1.0×1.0×1.0 mm spacing
- [x] Confirmed labels: 0 (background), 1 (necrotic), 2 (edema), 3 (enhancing)

### Phase 2: Dataset Adapter ✅
- [x] `dataset/brats_dataset.py` — BraTS loader
  - Enumerates all 1,251 subjects
  - Loads T2-FLAIR only (preserves architecture)
  - Normalizes intensity to [0, 1]
  - Resizes to 64×64×64 (from 240×240×155)
  - Converts 4-class → binary segmentation
  - Returns torch tensors (1, 64, 64, 64)
  - Train/val split: 90%/10%
  - **Test run**: ✅ Loads first sample successfully

- [x] `preprocessing/brats_preprocess.py` — Reusable utilities
  - `normalize_intensity()` — minmax, zscore, robust
  - `convert_segmentation_to_binary()` — BraTS 4-class → binary
  - `resize_volume()` — scipy zoom with order control
  - `get_volume_statistics()` — mean, std, min, max, median

- [x] `configs/brats.yaml` — Training configuration
  - Dataset: root_dir, val_split (0.1), target_shape (64×64×64)
  - Model: in_channels=1, out_channels=1
  - Training: 50 epochs, batch_size=4, lr=1e-4
  - Loss: HybridLoss with evidential=true
  - Optimizer: Adam + CosineAnnealingLR
  - Checkpointing: every 5 epochs, best model save

### Phase 3: Training Script ✅
- [x] `train_baseline_brats.py` — Baseline training without any optimizer modifications
  - Imports frozen `HybridMiniSwin2D5_CBAM` and `HybridLoss` from final_model.py
  - No architecture changes
  - No optimizer modifications
  - DataLoader integration with BraTS dataset
  - Epoch loop: forward → loss → backward → metrics
  - Validation with Dice score tracking
  - Checkpoint saving (best + periodic)
  - Training history (JSON)
  - Command-line config overrides

**Training started**: `python train_baseline_brats.py --epochs 1 --batch_size 2`

---

## Directory Structure Created

```
Neuroscan/
├── dataset/
│   ├── __init__.py
│   └── brats_dataset.py           [343 lines]
│
├── preprocessing/
│   ├── __init__.py
│   └── brats_preprocess.py        [97 lines]
│
├── configs/
│   └── brats.yaml                 [45 lines]
│
├── train_baseline_brats.py        [265 lines]
│
└── BASELINE_BRATS_SETUP.md        [this file]
```

---

## Key Design Decisions (Frozen)

### 1. FLAIR-Only Input
- Uses T2-FLAIR modality exclusively
- Preserves original NeuroScan architecture (in_channels=1)
- Keeps optimization as the isolated research variable
- Avoids multimodal confusion in results

### 2. Binary Segmentation
- Converts BraTS 4-class (necrotic, edema, enhancing) → 1 (tumor)
- Compatible with original binary loss functions
- No changes to HybridLoss or evaluation metrics

### 3. Dimension Alignment
- Resizes 240×240×155 → 64×64×64
- Matches Adaptive Slice Selector input expectations
- No changes to model.forward() or 2.5D processing

### 4. No Architecture Changes
- Uses HybridMiniSwin2D5_CBAM **as-is** from final_model.py
- No modifications to:
  - Adaptive Slice Selector
  - 2.5D Stem
  - Encoder/Decoder
  - CBAM attention
  - Evidential uncertainty head

### 5. Clean Separation
- Dataset logic completely separate from model
- Model stays in `01_source_code/models/final_model.py` (frozen)
- New code in `dataset/`, `preprocessing/`, `configs/`
- Training script uses sys.path to import frozen baseline

---

## What Happens Next

### Phase 4: Baseline Results
When training completes (1 epoch test, then full 50-epoch run):

1. Measure validation Dice score
2. Plot training curves (loss, Dice over epochs)
3. Identify convergence plateau
4. Document baseline performance in `BASELINE_RESULTS.md`
5. Save best checkpoint to `checkpoints/brats_baseline/best_model.pth`

**Expected baseline Dice**: >50% on BraTS validation set (tumor segmentation is harder than MS lesion detection, but we have 1,251 subjects so convergence should be normal)

### Phase 5: Only Then Integrate Optimizer
Once baseline is verified:
- [ ] Build `research_infra/optimization_controller/adaptive_controller.py`
- [ ] Integrate with training loop (keeps baseline untouched)
- [ ] Compare: baseline vs. baseline + controller
- [ ] Measure improvement attribution

---

## How to Use

### Train baseline (full 50 epochs)
```bash
python train_baseline_brats.py
```

### Override config via CLI
```bash
python train_baseline_brats.py --epochs 100 --batch_size 8
```

### Use custom config file
```bash
python train_baseline_brats.py --config my_config.yaml
```

### Test dataset loading
```bash
python dataset/brats_dataset.py
```

---

## Checkpoints Saved To
```
checkpoints/brats_baseline/
├── best_model.pth          (Best validation Dice)
├── checkpoint_epoch_000.pth
├── checkpoint_epoch_005.pth
├── ...
└── history.json            (Training curves)
```

---

## Notes for Future Sessions

- **Dataset location frozen**: All 1,251 subjects at `Dataset\Training`
- **Architecture frozen**: final_model.py never modified again
- **Only variable**: Optimizer modifications (in optimization_controller/)
- **Baseline is clean**: No optimizer, no arch changes, pure NeuroScan on BraTS

This setup makes your contribution (the adaptive optimizer) completely isolated and defensible—reviewers will see exactly where improvements come from.
