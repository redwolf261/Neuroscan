# Quick Test Guide

## 🧪 Test Before Full Run (Important!)

Before running the full 17-22 hour experiments, test with 2-3 epochs to verify everything works.

---

## Step 1: Modify Scripts for Quick Test

### hyperparameter_sensitivity.py
**Line 34** - Change:
```python
EPOCHS = 50  # Original
```
To:
```python
EPOCHS = 2  # Quick test
```

### csrf_variants_analysis.py  
**Line 32** - Change:
```python
EPOCHS = 50  # Original
```
To:
```python
EPOCHS = 2  # Quick test
```

---

## Step 2: Run Quick Tests

### Test 1: hyperparameter_sensitivity.py
```powershell
cd c:\Users\HP\EDI\research
python hyperparameter_sensitivity.py
```
**Expected time**: ~15-20 minutes (12 configs × 2 epochs)

**Check outputs:**
1. Console shows progress with metrics
2. Local directory created: `research_hyperparameter_sensitivity_local/`
3. Google Drive directory: `G:\My Drive\NeuroScan_Research\Research_HyperparamSensitivity\`
4. Each config subdirectory has:
   - `train_logs.csv` (2 rows)
   - `val_logs.csv` (2 rows)
   - `best_model.pth` (on Google Drive)

**Verify CSV columns:**
- train_logs.csv: epoch, loss, dice, lr, time_seconds
- val_logs.csv: epoch, loss, dice, precision, recall, f1, specificity, time_seconds

### Test 2: csrf_variants_analysis.py
```powershell
cd c:\Users\HP\EDI\research
python csrf_variants_analysis.py
```
**Expected time**: ~15-20 minutes (8 variants × 2 epochs)

**Check outputs:**
1. Console shows progress with metrics
2. Local directory created: `research_csrf_variants_local/`
3. Google Drive directory: `G:\My Drive\NeuroScan_Research\Research_CSRFVariants\`
4. Each variant subdirectory has:
   - `train_logs.csv` (2 rows)
   - `val_logs.csv` (2 rows)
   - `best_model.pth` (on Google Drive)

**Verify CSV columns:** Same as above

---

## Step 3: Verify Everything Works

### ✅ Success Checklist
- [ ] No errors or crashes
- [ ] CSVs created with correct columns
- [ ] All metrics calculated (not NaN or 0)
- [ ] Best models saved to Google Drive
- [ ] Console output shows epoch progress
- [ ] Time tracking working (time_seconds column filled)
- [ ] Google Drive sync working (files appear in Drive)

### ⚠️ Common Issues

**Issue**: Import error for research_utils
```
ModuleNotFoundError: No module named 'research_utils'
```
**Fix**: Make sure `research_utils.py` is in same directory as scripts

**Issue**: CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Fix**: Reduce batch size or k_slices in config

**Issue**: Google Drive path not found
```
FileNotFoundError: [WinError 3] The system cannot find the path specified
```
**Fix**: Check if Google Drive is mounted at `G:\My Drive\`

---

## Step 4: Restore Full Config & Run

After successful test, restore original settings:

### hyperparameter_sensitivity.py
**Line 34** - Change back:
```python
EPOCHS = 50  # Full run
```

### csrf_variants_analysis.py
**Line 32** - Change back:
```python
EPOCHS = 50  # Full run
```

### Run Full Experiments
```powershell
# Run in background (optional)
cd c:\Users\HP\EDI\research

# Start hyperparameter sensitivity (~6-8 hours)
python hyperparameter_sensitivity.py

# After completion, start CSRF variants (~8-10 hours)
python csrf_variants_analysis.py

# Finally, run noise robustness (~3-4 hours)
python noise_robustness_analysis.py
```

**Total estimated time**: 17-22 hours

---

## 📊 Monitoring Progress

### Console Output
Watch for:
- Epoch progress bars
- Metrics printed each epoch
- Best model save messages
- CSV save confirmations

### Check Google Drive
Every few epochs, verify:
- CSV files updating
- best_model.pth being overwritten when improved
- No `.tmp` files left behind (indicates sync issues)

### GPU Monitoring
```powershell
nvidia-smi -l 5
```
Monitor VRAM usage (should stay under 4GB)

---

## 🎯 Expected Results

### hyperparameter_sensitivity.py
- Identify optimal k_slices and window_size
- See which configurations converge fastest
- Compare final Dice scores across configs

### csrf_variants_analysis.py
- Compare CSRF vs other fusion methods
- See if CSRF + enhancements improve further
- Identify best fusion strategy

### noise_robustness_analysis.py
- Test model stability under noise
- Identify which noise types affect performance most
- Validate model generalization

---

**Ready to test!** 🚀
