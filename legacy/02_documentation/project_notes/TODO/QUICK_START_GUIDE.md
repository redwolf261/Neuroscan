# 🚀 QUICK START GUIDE - Test Execution

## ⚡ Run All Quick Tests (4-6 hours total)

```powershell
# Navigate to project root
cd C:\Users\HP\EDI

# Test 1: Hyperparameter Sensitivity (~1-2 hours)
python research\hyperparameter_sensitivity_quick.py

# Test 2: CSRF Variants (~2-3 hours)
python research\csrf_variants_analysis_quick.py

# Test 3: Noise Robustness (~1 hour)
python research\noise_robustness_analysis_quick.py
```

---

## 📊 What You'll Get

### Results Files:
```
research/hyperparam_sensitivity_results_quick/
  ├── sensitivity_results_quick.json
  └── sensitivity_analysis_plots_quick.png

research/csrf_variants_results_quick/
  ├── csrf_variants_results_quick.json
  └── csrf_variants_comparison_quick.png

research/noise_robustness_results_quick/
  ├── noise_robustness_results_quick.json
  └── noise_robustness_plots_quick.png
```

---

## 🔍 Key Findings to Look For

### Test 1: Hyperparameter Sensitivity
- **Look for**: Which k_slices value gives best Dice?
- **Expected**: k=5 should be optimal or near-optimal
- **Check**: Heatmap plot showing performance matrix

### Test 2: CSRF Variants  
- **Look for**: Does CSRF beat SE and CBAM?
- **Expected**: CSRF > CBAM > SE > Baseline
- **Check**: Bar chart showing improvement percentages

### Test 3: Noise Robustness
- **Look for**: Performance degradation at 15% noise
- **Expected**: Dice should drop gracefully, stay >70%
- **Check**: Line plots showing degradation curves

---

## ⚠️ Pre-Flight Checklist

```powershell
# 1. Check GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"

# 2. Check Dataset
dir "G:\My Drive\Dataset\PediMS\PediMS"

# 3. Check Python Environment
python --version  # Should be 3.13.5
```

---

## 📈 Estimated Timeline

| Test | Quick Version | Full Version |
|------|---------------|--------------|
| Hyperparameter Sensitivity | 1-2 hours | 6-8 hours |
| CSRF Variants | 2-3 hours | 8-10 hours |
| Noise Robustness | 1 hour | 3-4 hours |
| **TOTAL** | **4-6 hours** | **17-22 hours** |

---

## 💡 Pro Tips

1. **Start with quick tests first** to validate everything works
2. **Run overnight** for full tests (long runtime)
3. **Check terminal output** for progress bars and ETA
4. **Results auto-save** after each config (safe to interrupt)
5. **GPU temperature**: Monitor if running long tests

---

## 🎯 Success Criteria

After running quick tests, you should have:

- ✅ 3 JSON result files with numerical data
- ✅ 3 PNG plot files with visualizations
- ✅ No Python errors or crashes
- ✅ Dice scores in reasonable range (70-86%)
- ✅ Clear winner in CSRF variants (should be CSRF)
- ✅ Graceful degradation in noise tests

---

## 🚨 If Something Goes Wrong

```powershell
# Check for errors
python research\hyperparameter_sensitivity_quick.py 2>&1 | Tee-Object -FilePath error_log.txt

# Common issues:
# - Dataset not found → Check path in script
# - CUDA out of memory → Reduce BATCH_SIZE in script
# - Import errors → Check dependencies installed
```

---

## 📞 Quick Commands Reference

```powershell
# Check GPU memory
nvidia-smi

# List created scripts
dir research\*sensitivity*.py
dir research\*csrf*.py
dir research\*noise*.py

# View results
dir research\*_results_quick\

# Open result plots
start research\hyperparam_sensitivity_results_quick\sensitivity_analysis_plots_quick.png
start research\csrf_variants_results_quick\csrf_variants_comparison_quick.png
start research\noise_robustness_results_quick\noise_robustness_plots_quick.png
```

---

## 🎉 When Tests Complete

1. Check all 6 output files exist (3 JSON + 3 PNG)
2. Review plots visually - do trends make sense?
3. Read JSON files - are Dice scores reasonable?
4. If all looks good → Run full tests for paper
5. If issues → Debug and rerun specific test

---

**Quick Start Flowchart:**

```
START
  ↓
Run Quick Tests (4-6h)
  ↓
Check Results Look Good? → NO → Debug & Fix
  ↓ YES
Decide: Need Full Tests?
  ↓ YES
Run Full Tests (17-22h)
  ↓
Use Results in Paper
  ↓
DONE ✅
```

---

**READY TO GO! Just run the 3 commands above to start testing! 🚀**
