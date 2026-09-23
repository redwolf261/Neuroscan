# Experiment Execution Monitor 🚀

**Started:** November 6, 2025
**Status:** RUNNING

---

## 📊 Current Execution

### 1. Hyperparameter Sensitivity Analysis ⏳ RUNNING
**Terminal ID:** `c6b13472-1900-483c-b9ad-212dc0a10603`
**Command:** `python c:\Users\HP\EDI\research\hyperparameter_sensitivity.py`
**Started:** Now
**Expected Duration:** 6-8 hours
**Configurations:** 12 (k_slices: [3, 5, 7, 9] × window_size: [4, 8, 16])
**Epochs per config:** 50

**Output Locations:**
- Local: `c:\Users\HP\EDI\research\hyperparam_sensitivity_results\`
- Google Drive: `G:\My Drive\NeuroScan_Research\Research_HyperparamSensitivity\`

**Per-Config Outputs:**
- `k{X}_w{Y}/train_logs.csv` - Training metrics per epoch
- `k{X}_w{Y}/val_logs.csv` - Validation metrics per epoch  
- `k{X}_w{Y}/best_model.pth` - Best checkpoint (on Google Drive)

---

### 2. CSRF Variants Analysis ⏸️ PENDING
**Expected Start:** After #1 completes
**Expected Duration:** 8-10 hours
**Variants:** 8 fusion strategies
**Epochs per variant:** 50

**Output Locations:**
- Local: `c:\Users\HP\EDI\research\csrf_variants_results\`
- Google Drive: `G:\My Drive\NeuroScan_Research\Research_CSRFVariants\`

---

### 3. Noise Robustness Analysis ⏸️ PENDING
**Expected Start:** After #2 completes
**Expected Duration:** 3-4 hours
**Type:** Inference only (no training)

---

## 🔍 How to Monitor Progress

### Check Terminal Output
```powershell
# This will show current progress
# Note: Terminal might be silent during processing
```

### Check Console Output Files
```powershell
# Check if directories are being created
ls "c:\Users\HP\EDI\research\hyperparam_sensitivity_results"
ls "G:\My Drive\NeuroScan_Research\Research_HyperparamSensitivity"
```

### Monitor CSV Files
```powershell
# Check training logs being written
cat "G:\My Drive\NeuroScan_Research\Research_HyperparamSensitivity\k3_w4\train_logs.csv"
cat "G:\My Drive\NeuroScan_Research\Research_HyperparamSensitivity\k3_w4\val_logs.csv"

# Watch file updates in real-time
Get-ChildItem "G:\My Drive\NeuroScan_Research\Research_HyperparamSensitivity" -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 10 FullName, LastWriteTime
```

### Monitor GPU
```powershell
# Watch GPU usage every 5 seconds
nvidia-smi -l 5
```

### Check Best Models
```powershell
# See which configs have saved best models
Get-ChildItem "G:\My Drive\NeuroScan_Research\Research_HyperparamSensitivity\*\best_model.pth" | Select-Object FullName, Length, LastWriteTime
```

---

## 📈 Expected Timeline

| Experiment | Duration | Start | End (Est.) |
|------------|----------|-------|------------|
| Hyperparameter Sensitivity | 6-8h | Now | +6-8h |
| CSRF Variants | 8-10h | +6-8h | +14-18h |
| Noise Robustness | 3-4h | +14-18h | +17-22h |
| **TOTAL** | **17-22h** | **Now** | **Tomorrow** |

---

## ⚠️ Things to Watch

### Normal Behavior ✅
- Script might be silent for long periods (processing epochs)
- Google Drive sync might lag (files appear delayed)
- GPU usage should be 70-90% consistently
- CSV files update every epoch (~5-10 minutes per epoch)

### Warning Signs ⚠️
- GPU usage drops to 0% (crashed)
- CSV files stop updating for >30 minutes
- Many `.tmp` files accumulating in Google Drive (sync issue)
- VRAM error messages (out of memory)

### If Script Crashes
1. Check last terminal output for error
2. Check which config was running
3. Check CSV logs to see last completed epoch
4. Can resume from last config if needed

---

## 📝 Results Files

### After Hyperparameter Sensitivity Completes
```
hyperparam_sensitivity_results/
├── sensitivity_results.json          # Summary of all configs
└── sensitivity_analysis_plots.png   # Comparison plots

G:\My Drive\NeuroScan_Research\Research_HyperparamSensitivity\
├── k3_w4/
│   ├── train_logs.csv    # All 50 epochs training metrics
│   ├── val_logs.csv      # All 50 epochs validation metrics
│   └── best_model.pth    # Best checkpoint
├── k3_w8/
│   └── ...
└── ... (12 configs total)
```

### After CSRF Variants Completes
```
csrf_variants_results/
├── csrf_variants_results.json          # Summary of all variants
└── csrf_variants_comparison.png        # Comparison plots

G:\My Drive\NeuroScan_Research\Research_CSRFVariants\
├── No_Fusion_(Baseline)/
│   ├── train_logs.csv
│   ├── val_logs.csv
│   └── best_model.pth
├── Squeeze-and-Excitation_(SE)/
│   └── ...
└── ... (8 variants total)
```

---

## 🎯 Success Criteria

- [ ] All 12 hyperparameter configs complete (50 epochs each)
- [ ] All 8 CSRF variants complete (50 epochs each)
- [ ] All CSV files have 50 rows (one per epoch)
- [ ] All best_model.pth files saved to Google Drive
- [ ] JSON summary files created
- [ ] Comparison plots generated
- [ ] No errors in console output

---

## 📞 Quick Commands

### Check if script still running
```powershell
Get-Process python
```

### Kill if needed (emergency only)
```powershell
Stop-Process -Name python -Force
```

### Restart from where it crashed
```powershell
# Edit the script to skip completed configs
# Or just run the next experiment
```

---

**Monitor this file for updates during execution!**

**Estimated Completion:** ~17-22 hours from start
**Expected Finish:** Tomorrow afternoon

---

*Last Updated: November 6, 2025*
*Status: Hyperparameter Sensitivity RUNNING*
