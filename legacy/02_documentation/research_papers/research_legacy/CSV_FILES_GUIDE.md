# 📊 CSV Files Location Guide

**All training logs and results for your research paper**

---

## 🗂️ CSV Files Overview

### 1️⃣ Final Production Model Training Logs

**Location:** `G:\My Drive\NeuroScan_FinalModel_2.5D_MAE\segmentation\`

These are your **83.99% Dice** model training logs:
- `train_logs.csv` - Training metrics per epoch (28 epochs)
- `val_logs.csv` - Validation metrics per epoch (28 epochs)

**Columns:**
- Epoch, Train_Dice, Train_Loss, Val_Dice, Val_Loss, Precision, Recall, F1_Score, Learning_Rate

**Use for:**
- Figure 2: Training curves (validation Dice over epochs)
- Show early stopping at epoch 28
- Convergence analysis

---

### 2️⃣ Ablation Study Summary

**Location:** `C:\Users\HP\EDI\ablation\reports\`

#### `ablation2_summary_100epochs.csv`
**THE MAIN FILE FOR YOUR PAPER** ✅

Contains final comparison of all variants at epoch 100:
```csv
Variant,Val_Dice,Absolute_Diff,Relative_Diff_%,Train_Dice,Overfitting_Gap,Precision,Recall,F1_Score,Train_Loss
Baseline,0.7309,0.0,0.0,0.7907,0.0598,0.6826,0.7996,0.7363,0.1394
NoSwin,0.7346,+0.0037,+0.51%,0.8008,0.0662,0.6897,0.7939,0.7380,0.1309
No3DConv,0.7481,+0.0172,+2.36%,0.8247,0.0766,0.7273,0.7758,0.7508,0.1190
NoDropout,0.7349,+0.0040,+0.55%,0.8016,0.0667,0.6884,0.7964,0.7383,0.1300
NoResidual,0.6984,-0.0325,-4.44%,0.5414,-0.1570,0.5873,0.9145,0.7149,0.3140
NoAttention,0.7363,+0.0054,+0.74%,0.8035,0.0672,0.6928,0.7924,0.7392,0.1297
```

**Use for:**
- Table 2: Component importance
- Figure 1: Dice score comparison
- Figure 3: Component importance bar chart
- Statistical analysis

#### `ablation_summary.csv`
Older version (before resuming to 100 epochs) - less important

---

### 3️⃣ Individual Variant Training Logs

**Location:** `C:\Users\HP\EDI\ablation\HybridMiniSwin3D_<variant>\results\`

Each variant has:
- `train_logs.csv` - Training metrics per epoch (100 epochs)
- `val_logs.csv` - Validation metrics per epoch (100 epochs)

**Variants:**
1. **HybridMiniSwin3D_NoSwin/** - Without Swin Transformer
2. **HybridMiniSwin3D_No3DConv/** - Without 3D convolutions (best performer!)
3. **HybridMiniSwin3D_NoDropout/** - Without dropout
4. **HybridMiniSwin3D_NoResidual/** - Without skip connections (worst)
5. **HybridMiniSwin3D_NoAttention/** - Without attention

**Use for:**
- Supplementary materials
- Epoch-by-epoch analysis
- Convergence comparison plots
- Overfitting analysis over time

---

### 4️⃣ Archive (Old Training Data)

**Location:** `C:\Users\HP\EDI\archive\`

- `training_metrics.csv` - Old training data
- `training_summary.csv` - Old summary

**Status:** ⚠️ Deprecated - Don't use for paper (use ablation data instead)

---

## 📊 Quick Access Commands

### View Ablation Summary (Main Results)
```powershell
# Open in Excel/Notepad
notepad C:\Users\HP\EDI\ablation\reports\ablation2_summary_100epochs.csv

# View in terminal
Get-Content C:\Users\HP\EDI\ablation\reports\ablation2_summary_100epochs.csv
```

### View Individual Variant Training
```powershell
# Example: No3DConv (best variant)
notepad C:\Users\HP\EDI\ablation\HybridMiniSwin3D_No3DConv\results\val_logs.csv
```

### View Final Model Training (Google Drive)
```powershell
# Navigate to Google Drive
cd "G:\My Drive\NeuroScan_FinalModel_2.5D_MAE\segmentation"
notepad val_logs.csv
```

---

## 📈 What Each CSV is Used For

### For Your Research Paper

| CSV File | Location | Use in Paper | Priority |
|----------|----------|--------------|----------|
| `ablation2_summary_100epochs.csv` | `ablation/reports/` | **Main ablation table, all figures** | 🔴 CRITICAL |
| Production `val_logs.csv` | Google Drive (segmentation/) | **Training curves, convergence** | 🔴 CRITICAL |
| Individual variant CSVs | `ablation/HybridMiniSwin3D_*/results/` | Supplementary materials | 🟡 OPTIONAL |
| Archive CSVs | `archive/` | Not used | ⚪ IGNORE |

---

## 🎯 CSV Data You Need for Paper Figures

### Figure 1: Dice Score Comparison
**Source:** `ablation2_summary_100epochs.csv`
- Column: `Val_Dice`
- All 6 variants (Baseline + 5 ablations)

### Figure 2: Training Curves
**Source:** Production model `val_logs.csv` (Google Drive)
- Column: `Epoch`, `Val_Dice`
- Shows 28 epochs with early stopping

### Figure 3: Component Importance
**Source:** `ablation2_summary_100epochs.csv`
- Column: `Relative_Diff_%`
- Shows impact of each component

### Figure 4: Overfitting Analysis
**Source:** `ablation2_summary_100epochs.csv`
- Columns: `Train_Dice`, `Val_Dice`, `Overfitting_Gap`

### Figure 5: Precision-Recall
**Source:** `ablation2_summary_100epochs.csv`
- Columns: `Precision`, `Recall`, `F1_Score`

---

## 📍 Production Model Logs (83.99% Dice)

### Location on Google Drive
```
G:\My Drive\NeuroScan_FinalModel_2.5D_MAE\
├── mae/
│   └── (MAE pre-training logs - 200 epochs)
└── segmentation/
    ├── train_logs.csv     ← Training metrics (28 epochs)
    ├── val_logs.csv       ← Validation metrics (28 epochs)
    ├── best_model.pth     ← Best checkpoint (epoch 28)
    └── training_history.json
```

### To Copy to Local (For Easy Access)
```powershell
# Copy validation logs
Copy-Item "G:\My Drive\NeuroScan_FinalModel_2.5D_MAE\segmentation\val_logs.csv" C:\Users\HP\EDI\research\

# Copy training logs
Copy-Item "G:\My Drive\NeuroScan_FinalModel_2.5D_MAE\segmentation\train_logs.csv" C:\Users\HP\EDI\research\
```

---

## 🔍 Example: Reading CSV in Python

```python
import pandas as pd

# Read ablation summary
ablation_df = pd.read_csv(r'C:\Users\HP\EDI\ablation\reports\ablation2_summary_100epochs.csv')
print(ablation_df)

# Read production model validation logs
val_logs = pd.read_csv(r'G:\My Drive\NeuroScan_FinalModel_2.5D_MAE\segmentation\val_logs.csv')
print(val_logs)

# Get best epoch
best_epoch = val_logs.loc[val_logs['Val_Dice'].idxmax()]
print(f"Best epoch: {best_epoch['Epoch']}")
print(f"Best Dice: {best_epoch['Val_Dice']:.4f}")
```

---

## ✅ Summary

### Essential CSVs for Paper:
1. ✅ **`ablation2_summary_100epochs.csv`** - Main ablation results
2. ✅ **Production `val_logs.csv`** - Training curves (Google Drive)
3. ⏳ Individual variant CSVs - Optional for supplementary

### Your CSV Files Are:
- ✅ Complete (all 100 epochs for ablation)
- ✅ Organized (in ablation/reports/)
- ✅ Ready for analysis
- ✅ Publication-quality

**Status:** 🎊 All data ready for paper writing!

---

**Last Updated:** October 25, 2025  
**Location:** See above paths
