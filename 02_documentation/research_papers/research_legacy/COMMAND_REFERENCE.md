# Quick Command Reference 🚀

## For Quick Testing (2-3 epochs)

### Step 1: Modify Scripts
Open each script and change line with `EPOCHS = 50` to `EPOCHS = 2`

### Step 2: Run Tests

```powershell
# Navigate to research directory
cd c:\Users\HP\EDI\research

# Test hyperparameter_sensitivity.py (~15-20 minutes)
python hyperparameter_sensitivity.py

# Test csrf_variants_analysis.py (~15-20 minutes)
python csrf_variants_analysis.py
```

### Step 3: Verify Outputs

```powershell
# Check local directories created
ls hyperparam_sensitivity_results
ls csrf_variants_results

# Check Google Drive directories
ls "G:\My Drive\NeuroScan_Research\Research_HyperparamSensitivity"
ls "G:\My Drive\NeuroScan_Research\Research_CSRFVariants"

# Check a sample CSV
cat "G:\My Drive\NeuroScan_Research\Research_HyperparamSensitivity\k3_w4\train_logs.csv"
cat "G:\My Drive\NeuroScan_Research\Research_HyperparamSensitivity\k3_w4\val_logs.csv"
```

---

## For Full Execution (50 epochs)

### Step 1: Restore Full Config
Change `EPOCHS = 2` back to `EPOCHS = 50` in both scripts

### Step 2: Run Full Experiments

```powershell
cd c:\Users\HP\EDI\research

# Run hyperparameter_sensitivity.py (~6-8 hours)
# 12 configs × 50 epochs
python hyperparameter_sensitivity.py

# After completion, run csrf_variants_analysis.py (~8-10 hours)
# 8 variants × 50 epochs
python csrf_variants_analysis.py

# Finally, run noise_robustness_analysis.py (~3-4 hours)
python noise_robustness_analysis.py
```

**Total Time: 17-22 hours**

---

## Monitor Progress

### GPU Monitoring
```powershell
# Watch GPU usage in real-time (every 5 seconds)
nvidia-smi -l 5
```

### Check CSV Updates
```powershell
# Watch CSV file growing
Get-Content "G:\My Drive\NeuroScan_Research\Research_HyperparamSensitivity\k3_w4\val_logs.csv" -Wait
```

### Check Best Model
```powershell
# See when best model was last updated
Get-ChildItem "G:\My Drive\NeuroScan_Research\Research_HyperparamSensitivity\k3_w4\best_model.pth"
```

---

## Expected Outputs

### Per Experiment:
```
research/
├── hyperparam_sensitivity_results/
│   ├── sensitivity_results.json
│   └── sensitivity_analysis_plots.png
└── csrf_variants_results/
    ├── csrf_variants_results.json
    └── csrf_variants_comparison.png

G:\My Drive\NeuroScan_Research\
├── Research_HyperparamSensitivity/
│   ├── k3_w4/
│   │   ├── train_logs.csv
│   │   ├── val_logs.csv
│   │   └── best_model.pth
│   ├── k3_w8/
│   │   ├── train_logs.csv
│   │   ├── val_logs.csv
│   │   └── best_model.pth
│   └── ... (12 configs total)
└── Research_CSRFVariants/
    ├── No_Fusion_(Baseline)/
    │   ├── train_logs.csv
    │   ├── val_logs.csv
    │   └── best_model.pth
    ├── Squeeze-and-Excitation_(SE)/
    │   ├── train_logs.csv
    │   ├── val_logs.csv
    │   └── best_model.pth
    └── ... (8 variants total)
```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'research_utils'"
```powershell
# Make sure you're in the research directory
cd c:\Users\HP\EDI\research
python hyperparameter_sensitivity.py
```

### Error: "CUDA out of memory"
- Reduce BATCH_SIZE from 3 to 2 or 1
- Or reduce K_SLICES values

### Error: Google Drive path not found
```powershell
# Check if Google Drive is mounted
Test-Path "G:\My Drive"
```

### Warning: Google Drive sync slow
- Normal for large model files
- Check `.tmp` files aren't accumulating (indicates sync issues)

---

## Success Indicators ✅

- [ ] Console shows epoch progress
- [ ] CSVs updating every epoch
- [ ] Best model saved messages appear
- [ ] No errors or crashes
- [ ] GPU memory usage < 4GB
- [ ] Google Drive syncing files

---

**Quick Reference Created: November 6, 2025**
