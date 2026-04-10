# Reproducibility Guide

This document provides complete details to reproduce all experiments in our paper.

---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Configuration](#training-configuration)
4. [Experiment Protocols](#experiment-protocols)
5. [Random Seeds](#random-seeds)
6. [Hardware Specifications](#hardware-specifications)
7. [Code Release](#code-release)

---

## 1. Environment Setup

### Software Versions

```bash
Python: 3.11.7
PyTorch: 2.5.1+cu124
CUDA: 12.4
cuDNN: 9.1.0

# Deep Learning Framework
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1

# Medical Image Processing
MONAI==1.4.0
nibabel==5.3.2
SimpleITK==2.4.0

# Computer Vision
timm==1.0.11
einops==0.8.0

# Scientific Computing
numpy==1.26.4
scipy==1.14.1
scikit-learn==1.6.0
scikit-image==0.24.0

# Visualization
matplotlib==3.9.2
seaborn==0.13.2
plotly==5.24.1

# Utilities
tqdm==4.67.1
tensorboard==2.18.0
```

### Installation

```bash
# Create conda environment
conda create -n ms_lesion python=3.11
conda activate ms_lesion

# Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import monai; print(f'MONAI: {monai.__version__}')"
```

---

## 2. Dataset Preparation

### ISBI 2015 Longitudinal MS Lesion Segmentation Challenge

**Download:**
- Dataset: [ISBI Challenge Website](https://smart-stats-tools.org/lesion-challenge)
- Training set: 5 patients (2 timepoints each) = 10 volumes
- Test set: 4 patients = 4 volumes
- Total: 9 unique patients

**Data Structure:**
```
data/
├── ISBI2015/
│   ├── training/
│   │   ├── patient01_tp1_flair.nii.gz
│   │   ├── patient01_tp1_mask.nii.gz
│   │   ├── patient01_tp2_flair.nii.gz
│   │   ├── patient01_tp2_mask.nii.gz
│   │   └── ...
│   └── testing/
│       ├── patient06_flair.nii.gz
│       └── ...
```

**Preprocessing Steps:**

1. **Skull Stripping** (if not pre-processed):
   ```python
   # Using HD-BET or manual masks provided
   # No additional skull stripping required for ISBI2015
   ```

2. **Intensity Normalization**:
   ```python
   # Z-score normalization per volume
   def normalize_intensity(volume):
       brain_mask = volume > 0  # Non-zero voxels
       mean = volume[brain_mask].mean()
       std = volume[brain_mask].std()
       volume_norm = (volume - mean) / (std + 1e-8)
       volume_norm[~brain_mask] = 0
       return volume_norm
   ```

3. **Resampling** (Optional):
   ```python
   # Original: 181 x 217 x 181 with spacing ~1mm isotropic
   # No resampling required - use as-is
   ```

4. **5-Fold Patient-Wise Split**:
   ```python
   # Ensure no patient appears in both train and val
   # Split: [01, 02, 03, 04, 05] into 5 folds
   # Fold 1: Train=[02,03,04,05], Val=[01]
   # Fold 2: Train=[01,03,04,05], Val=[02]
   # Fold 3: Train=[01,02,04,05], Val=[03]
   # Fold 4: Train=[01,02,03,05], Val=[04]
   # Fold 5: Train=[01,02,03,04], Val=[05]
   ```

---

## 3. Training Configuration

### Optimizer

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

### Learning Rate Schedule

```python
# Cosine annealing with warmup
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # Total epochs
    eta_min=1e-6
)

# Warmup for first 10 epochs
warmup_epochs = 10
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=warmup_epochs
)

# Combined schedule
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, scheduler],
    milestones=[warmup_epochs]
)
```

### Loss Function

```python
# Dice Loss + Binary Cross-Entropy
loss = DiceLoss(sigmoid=True) + BCEWithLogitsLoss()

# Implementation:
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss(sigmoid=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        return self.dice_loss(pred, target) + self.bce_loss(pred, target)
```

### Hyperparameters

```python
# Training
batch_size = 2          # Per GPU (effective batch size = 2)
epochs = 100
gradient_clip = 1.0     # Max gradient norm

# Data augmentation
random_flip_prob = 0.5
random_rotate_range = (-15, 15)  # degrees
random_scale_range = (0.9, 1.1)
random_intensity_shift = 0.1
random_intensity_scale = 0.1
elastic_deform_prob = 0.2

# Model
k_slices = 5            # Number of input slices (2.5D)
base_channels = 96      # Swin feature dimension
window_size = 7         # Swin window size
num_heads = [3, 6, 12, 24]  # Multi-head attention
depths = [2, 2, 6, 2]   # Swin transformer blocks

# MAE Pre-training
mae_mask_ratio = 0.75
mae_epochs = 50
mae_lr = 1.5e-4

# CSRF
csrf_init_alpha = 1.0
csrf_learnable = True
csrf_per_channel = True
```

### Data Augmentation

```python
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAffined,
    RandGaussianNoised,
    RandElasticDeformd,
)

train_transforms = Compose([
    # Spatial augmentation
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
    RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
    
    RandAffined(
        keys=["image", "mask"],
        prob=0.5,
        rotate_range=(np.pi/12, np.pi/12, 0),  # ±15 degrees in-plane
        scale_range=(0.1, 0.1, 0),
        mode=("bilinear", "nearest"),
    ),
    
    # Elastic deformation
    RandElasticDeformd(
        keys=["image", "mask"],
        prob=0.2,
        sigma_range=(5, 8),
        magnitude_range=(50, 150),
        mode=("bilinear", "nearest"),
    ),
    
    # Intensity augmentation (image only)
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
])
```

---

## 4. Experiment Protocols

### Experiment Battery

**1. Baseline Comparisons:**
- **2D U-Net** (34.61M params): Processes central slice only
- **3D U-Net** (33.06M params): Full volume processing

**2. Ablation Studies:**
- **Ablation A**: 2.5D only (no CSRF, no MAE)
- **Ablation B**: 2.5D + CSRF
- **Ablation C**: 2.5D + MAE
- **Ablation D**: Full method (2.5D + CSRF + MAE)

**3. k-Slice Sweep:**
- k ∈ {1, 3, 5, 9}: Analyze effect of multi-slice input

### Training Protocol

1. **MAE Pre-training** (50 epochs):
   ```python
   # Train MAE encoder on reconstruction task
   python research/mae_pretraining.py --epochs 50 --lr 1.5e-4
   ```

2. **Fine-tuning** (100 epochs):
   ```python
   # Load MAE weights and fine-tune on segmentation
   python final_model.py --pretrained mae_checkpoint.pth --epochs 100
   ```

3. **Cross-Validation**:
   ```python
   # 5-fold patient-wise CV
   python research/cross_validation_framework.py --folds 5
   ```

4. **Multi-Seed Experiments**:
   ```python
   # Run with 5 different seeds
   for seed in [42, 123, 456, 789, 1024]:
       python final_model.py --seed $seed
   ```

### Evaluation Protocol

```python
# Metrics computed:
# - Dice Similarity Coefficient (DSC)
# - Precision, Recall, F1-Score
# - Specificity
# - Hausdorff Distance (95th percentile)
# - Lesion-wise F1 Score
# - Lesion TP/FP/FN
# - False Negative Rate (voxel and lesion level)
# - Small lesion sensitivity (<10 voxels)
# - Expected Calibration Error (ECE)

# Post-processing:
# - Connected component analysis (remove <3 voxels)
# - No morphological operations
```

---

## 5. Random Seeds

### Seed Configuration

```python
import torch
import numpy as np
import random

def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch reproducibility
    torch.use_deterministic_algorithms(True)
    
    # Environment variable for CUBLAS
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
```

### Standard Seeds

We use 5 standard seeds for all multi-seed experiments:

```python
STANDARD_SEEDS = [42, 123, 456, 789, 1024]
```

**Rationale:**
- Seed 42: Standard ML community convention
- Seeds 123, 456, 789: Sequential patterns for consistency
- Seed 1024: Power of 2 for computational significance

**Usage:**
```python
for seed in STANDARD_SEEDS:
    set_all_seeds(seed)
    results = train_and_evaluate()
```

**Statistical Analysis:**
- Report mean ± std across 5 seeds
- Paired t-test for significance (p < 0.05)
- Wilcoxon signed-rank test (non-parametric alternative)
- Cohen's d effect size
- 95% confidence intervals

---

## 6. Hardware Specifications

### Development System

```
GPU: NVIDIA GeForce RTX 2050 (4GB VRAM)
CPU: Intel Core i7 (8 cores, 12 threads)
RAM: 16 GB
OS: Windows 10
CUDA: 12.4
cuDNN: 9.1.0
```

### Training Times (Approximate)

```
Single epoch (5-fold CV, batch_size=2):
- 2D U-Net: ~15 minutes
- 3D U-Net: ~25 minutes
- HybridMiniSwin2D5_CSRF (ours): ~20 minutes

Full training (100 epochs):
- 2D U-Net: ~25 hours
- 3D U-Net: ~42 hours
- HybridMiniSwin2D5_CSRF (ours): ~33 hours

MAE Pre-training (50 epochs): ~16 hours

Total experiment battery (10 experiments × 100 epochs): ~2-3 days
```

### Memory Requirements

```
Training:
- Batch size 2: ~3.8 GB VRAM
- Batch size 4: ~7.2 GB VRAM (requires multi-GPU)

Inference:
- Single volume: ~500 MB VRAM
- Batch-8 processing: ~2.5 GB VRAM
```

### Computational Budget

```
Parameters: 34.20M
FLOPs per inference: 3.51G
Throughput (RTX 2050):
- GPU: 28.51 images/sec (35.08ms/image)
- CPU: 5.96 images/sec (167.84ms/image)
- Batch-8 GPU: 219 images/sec (4.57ms/image)
```

---

## 7. Code Release

### Repository Structure

```
ms-lesion-segmentation/
├── README.md                          # Project overview
├── REPRODUCIBILITY.md                 # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── data/                              # Dataset directory
│   ├── ISBI2015/
│   └── preprocessing.py
│
├── models/                            # Model implementations
│   ├── hybrid_mini_swin_2d5.py       # Main model
│   ├── csrf_module.py                # CSRF component
│   ├── mae_pretraining.py            # MAE pre-training
│   └── baseline_models.py            # 2D/3D U-Net baselines
│
├── research/                          # Experimental frameworks
│   ├── compute_analysis_detailed.py  # Parameter/FLOP analysis
│   ├── benchmark_inference.py        # Latency measurements
│   ├── cross_validation_framework.py # 5-fold CV
│   ├── multi_seed_experiments.py     # Multi-seed protocol
│   ├── experiment_battery.py         # Full experiment suite
│   └── clinical_metrics.py           # Clinical evaluation
│
├── training/                          # Training scripts
│   ├── train.py                      # Main training
│   ├── losses.py                     # Loss functions
│   └── transforms.py                 # Data augmentation
│
├── evaluation/                        # Evaluation scripts
│   ├── evaluate.py                   # Metrics computation
│   └── visualization.py              # Result visualization
│
└── checkpoints/                       # Model weights
    ├── mae_pretrained.pth
    ├── best_model.pth
    └── experiment_results.json
```

### Running Experiments

**1. Full Experiment Battery:**
```bash
python research/experiment_battery.py \
    --data_dir data/ISBI2015 \
    --output_dir results/experiment_battery \
    --seeds 42 123 456 789 1024 \
    --epochs 100
```

**2. Single Model Training:**
```bash
python training/train.py \
    --model HybridMiniSwin2D5_CSRF \
    --data_dir data/ISBI2015 \
    --output_dir results/single_model \
    --seed 42 \
    --epochs 100 \
    --batch_size 2 \
    --lr 1e-4
```

**3. Cross-Validation:**
```bash
python research/cross_validation_framework.py \
    --data_dir data/ISBI2015 \
    --output_dir results/cross_validation \
    --folds 5 \
    --seed 42
```

**4. Clinical Evaluation:**
```bash
python research/clinical_metrics.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/ISBI2015 \
    --output_dir results/clinical_evaluation
```

### Pre-trained Weights

Pre-trained model weights will be available at:
- **Hugging Face**: `[username]/ms-lesion-segmentation`
- **GitHub Release**: `[repo]/releases/latest`

**Download:**
```bash
# Download pre-trained weights
wget https://github.com/[username]/ms-lesion-segmentation/releases/download/v1.0/best_model.pth

# Or using Hugging Face
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="[username]/ms-lesion-segmentation", filename="best_model.pth")
```

### Docker Container (Optional)

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY . .

# Set environment variables
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Run training
CMD ["python", "training/train.py"]
```

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory:**
```python
# Reduce batch size
batch_size = 1

# Enable gradient checkpointing
model.enable_gradient_checkpointing()

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

**2. Deterministic Operations Warning:**
```python
# If you see: "deterministic algorithm not available"
# This is expected for some operations (e.g., bilinear interpolation)
# Results will still be reproducible within ~1e-5 tolerance
```

**3. Slow Data Loading:**
```python
# Increase num_workers
train_loader = DataLoader(..., num_workers=4, pin_memory=True)
```

---

## Citation

If you use this code or reproduce our experiments, please cite:

```bibtex
@article{yourname2025hybrid,
  title={Hybrid 2.5D Swin Transformer with Cross-Scale Residual Fusion for MS Lesion Segmentation},
  author={Your Name and Co-authors},
  journal={Medical Image Analysis},
  year={2025}
}
```

---

## Contact

For questions or issues reproducing experiments:
- Email: your.email@institution.edu
- GitHub Issues: [repo]/issues
- Project Page: [your-project-page]

---

## Changelog

**v1.0 (2025-11-04):**
- Initial release
- Complete experiment battery
- Pre-trained weights
- Full reproducibility guide

---

## License

This code is released under the MIT License. See LICENSE file for details.

---

**Last Updated**: November 4, 2025
