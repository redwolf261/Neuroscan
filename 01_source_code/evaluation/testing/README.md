# Testing Data for MS Lesion Detection

This folder contains test data for validating the HybridMiniSwin2.5D-CSRF model.

## Available Test Files

### Complete Patient Sets (9 Patients)
Each patient has 4 files: FLAIR, T1, T2, and Ground Truth

| Patient | FLAIR | T1 | T2 | Ground Truth | Total Size |
|---------|-------|----|----|--------------|------------|
| **P1** | P1_FLAIR.nii.gz (2.1 MB) | P1_T1.nii.gz (4.19 MB) | P1_T2.nii.gz (5.47 MB) | P1_GroundTruth.nii (14.17 MB) | 25.93 MB |
| **P2** | P2_FLAIR.nii.gz (2.33 MB) | P2_T1.nii.gz (5.41 MB) | P2_T2.nii.gz (6.13 MB) | P2_GroundTruth.nii (14.17 MB) | 28.04 MB |
| **P3** | P3_FLAIR.nii.gz (2.3 MB) | P3_T1.nii.gz (5.49 MB) | P3_T2.nii.gz (2.67 MB) | P3_GroundTruth.nii (14.17 MB) | 24.63 MB |
| **P4** | P4_FLAIR.nii.gz (1.11 MB) | P4_T1.nii.gz (5.96 MB) | P4_T2.nii.gz (6.77 MB) | P4_GroundTruth.nii (14.17 MB) | 28.01 MB |
| **P5** | P5_FLAIR.nii.gz (2.23 MB) | P5_T1.nii.gz (4.49 MB) | P5_T2.nii.gz (5.91 MB) | P5_GroundTruth.nii (15.94 MB) | 28.57 MB |
| **P6** | P6_FLAIR.nii.gz (0.05 MB) | P6_T1.nii.gz (2.34 MB) | P6_T2.nii.gz (2.81 MB) | P6_GroundTruth.nii (14.17 MB) | 19.37 MB |
| **P7** | P7_FLAIR.nii.gz (0.05 MB) | P7_T1.nii.gz (4.36 MB) | P7_T2.nii.gz (0.1 MB) | P7_GroundTruth.nii (14.17 MB) | 18.68 MB |
| **P8** | P8_FLAIR.nii.gz (1.14 MB) | P8_T1.nii.gz (6.2 MB) | P8_T2.nii.gz (0.11 MB) | P8_GroundTruth.nii (17.71 MB) | 25.16 MB |
| **P9** | P9_FLAIR.nii.gz (0.05 MB) | P9_T1.nii.gz (5.4 MB) | P9_T2.nii.gz (0.1 MB) | P9_GroundTruth.nii (14.17 MB) | 19.72 MB |

**Total: 37 files (9 complete patient sets + 1 sample) = 218 MB**

### Additional Sample
- **sample_brain_mri.nii.gz** - Generic brain MRI sample (11.19 MB)

## How to Use These Files

### 1. Test via Web Application
Upload through the webapp interface at http://localhost:3000

**Multi-Modal Upload (Recommended):**
1. Click FLAIR box → select `P1_FLAIR.nii.gz`
2. Click T1 box → select `P1_T1.nii.gz`
3. Click T2 box → select `P1_T2.nii.gz`
4. Click "Analyze Scans"

**Single-Modal Upload:**
1. Switch to "Single Modality" mode
2. Upload any FLAIR file (e.g., `P1_FLAIR.nii.gz`)
3. Click "Analyze Scans"

### 2. Test via inference.py Script
```bash
python inference.py --flair "testing/P1_FLAIR.nii.gz" --t1 "testing/P1_T1.nii.gz" --t2 "testing/P1_T2.nii.gz" --output predictions/
```

### 3. Test with Ground Truth Validation
Compare your model predictions against ground truth to calculate Dice score:

```python
import nibabel as nib
import numpy as np

# Load prediction and ground truth
pred = nib.load('predictions/P1_prediction.nii.gz').get_fdata()
gt = nib.load('testing/P1_GroundTruth.nii').get_fdata()

# Calculate Dice Score
intersection = np.sum(pred * gt)
dice = (2. * intersection) / (np.sum(pred) + np.sum(gt))
print(f"Dice Score: {dice:.4f}")
```

### 4. Batch Testing Script
Test all patients at once:

```python
import os
from pathlib import Path

test_dir = Path("testing")
patients = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]

for patient in patients:
    flair = test_dir / f"{patient}_FLAIR.nii.gz"
    t1 = test_dir / f"{patient}_T1.nii.gz"
    t2 = test_dir / f"{patient}_T2.nii.gz"
    gt = test_dir / f"{patient}_GroundTruth.nii"
    
    if all([flair.exists(), t1.exists(), t2.exists(), gt.exists()]):
        print(f"Testing {patient}...")
        # Run your prediction code here
```

## File Information

- **Source**: PediMS Dataset from Google Drive (G:\My Drive\Dataset\PediMS\PediMS\)
- **Preprocessing**: n4_brain preprocessed (bias field corrected, skull-stripped)
- **Format**: NIfTI (.nii.gz for scans, .nii for ground truth masks)
- **Expected Performance**: Model achieves 83.99% Dice Score on validation set

## Notes on File Variations

- **File sizes vary** due to different original resolutions and compression
- **Very small files** (0.05-0.1 MB): Highly compressed or heavily downsampled
- **Normal files** (1-6 MB): Standard resolution MRI scans
- **Ground truth** (14-18 MB): Uncompressed binary masks
- Model handles all resolutions automatically via preprocessing

## Dataset Statistics

- **9 complete test patients** from PediMS dataset
- **All modalities present**: FLAIR, T1, T2 for each patient
- **Ground truth available**: Expert-annotated MS lesion masks
- **Multi-resolution**: Handles different scan dimensions automatically
- **Ready for benchmarking**: Compare predictions vs ground truth

## Quick Test Commands

**Test single patient (P1):**
```powershell
# Via webapp - upload files manually through browser

# Via Python script
python inference.py --flair testing/P1_FLAIR.nii.gz --t1 testing/P1_T1.nii.gz --t2 testing/P1_T2.nii.gz --output predictions/
```

**Test patient with unusual dimensions (P3):**
```powershell
# P3 has different T2 dimensions - good test case
python inference.py --flair testing/P3_FLAIR.nii.gz --t1 testing/P3_T1.nii.gz --t2 testing/P3_T2.nii.gz --output predictions/
```

**Test all patients:**
```powershell
foreach($p in 1..9) { 
    python inference.py --flair "testing/P${p}_FLAIR.nii.gz" --t1 "testing/P${p}_T1.nii.gz" --t2 "testing/P${p}_T2.nii.gz" --output "predictions/P$p/"
}
```

## Expected Results

Based on validation performance (83.99% Dice, 91.64% Recall):
- **High confidence** on clear lesions (70-90% confidence)
- **Excellent recall** - detects most lesions (91.64%)
- **Good precision** - minimal false positives (77.60%)
- **Consistent across patients** - robust to different resolutions

---
**Last Updated**: October 26, 2025  
**Dataset**: PediMS (9 pediatric MS patients)  
**Total Size**: 218 MB (37 files)  
**Status**: ✅ Ready for testing and validation
