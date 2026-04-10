# ABLATION STUDY WORKFLOW FOR Q1 PAPER
# ======================================

## Quick Start: Run All 7 Configs

For each config, follow these steps:

### Config 1: Baseline (No USALD)
```python
# In final_model.py, set:
ABLATION_CONFIG = "baseline"
MAE_EPOCHS = 50
SEGMENTATION_EPOCHS = 30
```
Then run: `python final_model.py`

### Config 2: +Evidential Only
```python
ABLATION_CONFIG = "evidential"
```
Run: `python final_model.py`

### Config 3: +Causal Decomposition
```python
ABLATION_CONFIG = "causal"
```
Run: `python final_model.py`

### Config 4: +Self-Correction
```python
ABLATION_CONFIG = "self_correction"
```
Run: `python final_model.py`

### Config 5: +Consistency (Teacher-Student)
```python
ABLATION_CONFIG = "consistency"
```
Run: `python final_model.py`

### Config 6: +FDR Thresholding
```python
ABLATION_CONFIG = "fdr"
```
Run: `python final_model.py`

### Config 7: Full USALD (All 5 Components)
```python
ABLATION_CONFIG = "full"
```
Run: `python final_model.py`

---

## After All Configs Complete

Run the results collector:
```bash
python run_ablation_study.py
```

Choose option 1 ("Collect results") - it will:
1. Load validation CSVs from each config
2. Extract best epoch metrics (Dice, Precision, Recall, F1, etc.)
3. Calculate improvements vs baseline
4. Save to `C:\Users\HP\EDI\ablation_results\`:
   - `ablation_summary.csv` - Master comparison table
   - `ablation_statistics.csv` - Effect sizes
   - `config_[name]_results.csv` - Per-config epoch-by-epoch data

---

## Expected Outputs Per Config

Each config saves to separate Google Drive folder:
- `G:\My Drive\USALD_Ablation_baseline\`
- `G:\My Drive\USALD_Ablation_evidential\`
- `G:\My Drive\USALD_Ablation_causal\`
- etc.

Files per config:
- `mae_pretraining/mae_best.pth` - Pretrained encoder
- `mae_pretraining/mae_logs.csv` - MAE training logs
- `segmentation/best_model.pth` - Best model
- `segmentation/train_logs.csv` - Training metrics per epoch
- `segmentation/val_logs.csv` - Validation metrics per epoch (MAIN SOURCE)

---

## Metrics Collected

### Standard Metrics (all configs):
- **Dice Score** - Primary metric
- **Precision** - Lesion detection precision
- **Recall** - Lesion detection sensitivity
- **F1 Score** - Harmonic mean
- **Loss** - Combined training loss

### USALD-Specific Metrics (when enabled):
- **mean_evidence** - Evidential uncertainty (α₀ + α₁)
- **tau_pl** - FDR-controlled threshold
- **avg_self_correction_iters** - Iterations until convergence
- **causal_weight_anatomy** - Learned weight for anatomical uncertainty
- **causal_weight_pathology** - Learned weight for pathological uncertainty
- **causal_weight_noise** - Learned weight for noise uncertainty

---

## For Q1 Paper

### Table 1: Ablation Study Results
Use `ablation_summary.csv` columns:
- Config Name
- Dice ↑
- Precision ↑
- Recall ↑
- F1 ↑
- Δ Dice (vs Baseline)
- Improvement %

### Table 2: Component Analysis
Show which components are active per config:
- Baseline: ❌❌❌❌❌
- +Evidential: ✅❌❌❌❌
- +Causal: ✅❌❌✅❌
- +Self-Correction: ✅❌❌❌✅
- +Consistency: ✅✅❌❌❌
- +FDR: ✅❌✅❌❌
- Full USALD: ✅✅✅✅✅

### Figure 1: Performance Comparison
Bar chart with Dice scores for all 7 configs

### Figure 2: Causal Weights (Full USALD only)
Pie chart showing learned importance of anatomy/pathology/noise

### Figure 3: Learning Curves
Line plots of Dice vs Epoch for selected configs (Baseline vs Full)

---

## Statistical Significance

For Q1 journals, you need p-values. Options:

### Option 1: Cross-Validation (Preferred)
Run each config 5 times with 5-fold cross-validation.
Then use paired t-test between Full USALD and each ablation.

### Option 2: Wilcoxon Signed-Rank Test
If you have per-patient scores (9 patients), use non-parametric test.

### Option 3: Bootstrap Confidence Intervals
Resample validation results 1000 times, compute 95% CI.

Currently `ablation_statistics.csv` provides Cohen's d effect sizes as placeholder.

---

## Time Estimate

Per config (RTX 2050):
- MAE 50 epochs: ~60-90 min
- Segmentation 30 epochs: ~40-60 min
- Total: ~2 hours per config

All 7 configs: ~14 hours (can run overnight or across 2 days)

---

## Troubleshooting

### If a config fails:
- Check GPU memory (reduce batch size if OOM)
- Check logs in config's folder
- Restart from that specific config

### If results missing:
- Verify `val_logs.csv` exists in `G:\My Drive\USALD_Ablation_[config]\segmentation\`
- Re-run failed config

### If metrics seem wrong:
- Check `USALD_ENABLED` and related flags actually changed
- Verify separate checkpoint folders used (no mixing)
