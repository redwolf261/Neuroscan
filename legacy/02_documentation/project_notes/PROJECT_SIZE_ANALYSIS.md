# Project Size Analysis Report
**Total Project Size: 34.57 GB**

## 📊 Storage Breakdown by Category

### 1. **MS cross validation (9.71 GB)** 
- Contains cross-validation experiment results
- **Recommendation:** Archive or delete old cross-validation runs if results are documented

### 2. **Checkpoints (7.12 GB)**
- 49 .pth files totaling ~9.69 GB
- Multiple duplicate checkpoint files (seg_resume.pth, mae_resume.pth)
- Many checkpoints are 300-500MB each
- **Recommendation:** Keep only the best/final checkpoints, delete intermediate ones

### 3. **Dataset (6.9 GB)**
- Contains **DUPLICATE compressed datasets:**
  - MSLesSeg Dataset.zip (1,405 MB)
  - MSLesSeg_RAW.zip (1,235 MB)
  - MSLesSeg-2024-main.zip (10.46 MB)
  - Plus thousands of .nii.gz medical imaging files (4-5 MB each)
- **Recommendation:** Extract datasets once, delete zip files after extraction

### 4. **ms_detector_webapp (6 GB)**
- Web application files
- **Recommendation:** Check if this contains unnecessary node_modules, build artifacts, or duplicated model files

### 5. **LGG (1.97 GB)**
- Lower-grade glioma dataset/results
- **Recommendation:** Keep only if actively used

### 6. **Frozen Checkpoint Directories (2.5 GB total)**
- `.resume_checkpoints_frozen_20251218_135038` (730 MB)
- `.resume_checkpoints_frozen_20251218_125440` (374 MB)
- `.resume_checkpoints_frozen_20251217_224623` (374 MB)
- `.resume_checkpoints_frozen_20251217_220503` (374 MB)
- **Recommendation:** These are temporary backup checkpoints - DELETE all frozen checkpoints after confirming current training is stable

### 7. **Other Notable Items:**
- OptimalModel_Evidential (0.57 GB)
- Multiple testing, research, and trial folders

## 🎯 Recommended Actions to Reduce Size

### **IMMEDIATE WINS (Can save ~18-20 GB)**

#### 1. Delete Frozen Checkpoint Directories (~2.5 GB savings)
```powershell
Remove-Item "c:\Users\HP\Projects\EDI_Multiple_Sclerosis_detector\.resume_checkpoints_frozen_*" -Recurse -Force
```

#### 2. Clean Up Duplicate Checkpoint Files (~5-6 GB savings)
Keep only:
- Latest best model checkpoint
- Latest resume checkpoint
- Final trained model

Delete intermediate training checkpoints:
```powershell
# Review and manually delete old checkpoints in the checkpoints folder
# Keep only the most recent and best performing models
```

#### 3. Remove Compressed Dataset Duplicates (~2.6 GB savings)
```powershell
# After confirming datasets are extracted
Remove-Item "c:\Users\HP\Projects\EDI_Multiple_Sclerosis_detector\Dataset\MSLesSeg Dataset.zip" -Force
Remove-Item "c:\Users\HP\Projects\EDI_Multiple_Sclerosis_detector\Dataset\MSLesSeg_RAW.zip" -Force
Remove-Item "c:\Users\HP\Projects\EDI_Multiple_Sclerosis_detector\Dataset\MSLesSeg-2024-main.zip" -Force
```

#### 4. Archive Old Cross-Validation Results (~5-8 GB savings)
```powershell
# Archive to external storage or cloud, then delete locally
# Keep only recent/important cross-validation runs
```

### **MEDIUM PRIORITY (Can save ~5-10 GB)**

#### 5. Clean ms_detector_webapp Directory
- Check for `node_modules` folder - can be regenerated from package.json
- Check for build artifacts
- Check for duplicate model files

#### 6. Review and Archive Testing/Research Folders
- `testing` (0.22 GB)
- `research` (0.15 GB)
- `trials` folder
- `archive` folder - ironic, but it might contain unneeded duplicates

#### 7. Clean Python Cache Files
```powershell
Get-ChildItem -Path "c:\Users\HP\Projects\EDI_Multiple_Sclerosis_detector" -Include "__pycache__","*.pyc","*.pyo" -Recurse -Force | Remove-Item -Recurse -Force
```

### **LOW PRIORITY / Optional**

#### 8. Compress Infrequently Used Data
- Use NTFS compression for folders you rarely access
```powershell
compact /c /s:"c:\Users\HP\Projects\EDI_Multiple_Sclerosis_detector\LGG"
compact /c /s:"c:\Users\HP\Projects\EDI_Multiple_Sclerosis_detector\archive"
```

#### 9. Use Git LFS or External Storage
- Move large model checkpoints to Git LFS
- Use cloud storage (Google Drive, Azure Blob) for datasets

## 📋 Safe Cleanup Checklist

Before deleting anything, ensure:
- [ ] Recent backups exist
- [ ] Git commits are up to date
- [ ] Important experiment results are documented
- [ ] Best model weights are identified and preserved
- [ ] Datasets can be re-downloaded if needed

## 🔍 Detailed File Type Breakdown

| File Type | Count | Total Size | Notes |
|-----------|-------|------------|-------|
| .pth (PyTorch checkpoints) | 49 | 9.69 GB | Many duplicates |
| .nii.gz (Medical images) | 1000+ | ~4-6 GB | Actual dataset files |
| .zip (Archives) | Various | ~2.6 GB | Mostly in Dataset folder |
| .pkl (Python pickles) | 11 | <10 MB | Negligible |

## 🎬 Execution Plan

### **Phase 1: Quick Wins (Save ~10 GB in 5 minutes)**
1. Delete frozen checkpoint directories
2. Delete dataset zip files (after verification)
3. Clean Python cache

### **Phase 2: Checkpoint Cleanup (Save ~5 GB in 15 minutes)**
1. Identify best model checkpoint
2. Delete intermediate training checkpoints
3. Keep only final and best models

### **Phase 3: Archive Old Results (Save ~5-8 GB)**
1. Archive old cross-validation results
2. Clean testing/research folders
3. Review ms_detector_webapp

## ⚠️ Important Notes

1. **Never delete the Dataset if you can't re-download it**
2. **Always keep at least one working checkpoint**
3. **Document which models are which before deleting**
4. **Consider using version control for tracking important checkpoints**
5. **The .nii.gz files are your actual medical imaging data - compress but don't delete**

## 🎯 Expected Final Size

- **Current:** 34.57 GB
- **After Phase 1:** ~24 GB
- **After Phase 2:** ~19 GB
- **After Phase 3:** ~11-14 GB
- **With compression:** ~8-10 GB

This would reduce your project size by **60-70%** without losing any critical data!
