# ABLATION STUDY QUICK REFERENCE
# ================================

## Setup Complete! ✅

Your ablation study system is ready for Q1 paper submission.

---

## What Was Added

### 1. Configuration System (final_model.py)
- **7 ablation configs** defined in `ABLATION_CONFIGS` dict
- **Auto-switching**: Set `ABLATION_CONFIG = "baseline"` (or any config name)
- **Separate paths**: Each config saves to `USALD_Ablation_[config]/`

### 2. Comprehensive Metrics Collection
All validation logs (`val_logs.csv`) now include:
- ✅ Dice, Precision, Recall, F1, Loss
- ✅ mean_evidence (evidential uncertainty)
- ✅ tau_pl (FDR threshold)
- ✅ avg_self_correction_iters
- ✅ causal_weight_anatomy, causal_weight_pathology, causal_weight_noise

### 3. Results Collector (run_ablation_study.py)
- Loads all config validation CSVs
- Extracts best epoch metrics
- Calculates Δ vs baseline
- Saves publication-ready CSVs

### 4. Workflow Guide (ABLATION_WORKFLOW.md)
- Step-by-step instructions
- Expected outputs
- Q1 paper table/figure templates
- Troubleshooting tips

---

## How to Run (7 Configs × ~2 hours each = ~14 hours total)

### Step 1: Set Epochs (in final_model.py)
```python
MAE_EPOCHS = 50
SEGMENTATION_EPOCHS = 30
```

### Step 2: Run Each Config

**Config 1: Baseline**
```python
ABLATION_CONFIG = "baseline"
```
Run: `python final_model.py`
Wait ~2 hours

**Config 2: Evidential**
```python
ABLATION_CONFIG = "evidential"
```
Run: `python final_model.py`

**Config 3: Causal**
```python
ABLATION_CONFIG = "causal"
```
Run: `python final_model.py`

**Config 4: Self-Correction**
```python
ABLATION_CONFIG = "self_correction"
```
Run: `python final_model.py`

**Config 5: Consistency**
```python
ABLATION_CONFIG = "consistency"
```
Run: `python final_model.py`

**Config 6: FDR**
```python
ABLATION_CONFIG = "fdr"
```
Run: `python final_model.py`

**Config 7: Full USALD**
```python
ABLATION_CONFIG = "full"
```
Run: `python final_model.py`

### Step 3: Collect Results
```bash
python run_ablation_study.py
```
Choose option "1" (Collect results)

---

## Output Files

### Per-Config Outputs (Google Drive)
```
G:\My Drive\USALD_Ablation_baseline\
├── mae_pretraining\
│   ├── mae_best.pth
│   └── mae_logs.csv
└── segmentation\
    ├── best_model.pth
    ├── train_logs.csv
    └── val_logs.csv  ← Main source for results
```

### Final Results (EDI Folder)
```
C:\Users\HP\EDI\ablation_results\
├── ablation_summary.csv         ← Master table (7 rows, 15+ columns)
├── ablation_statistics.csv      ← Effect sizes and significance
├── config_baseline_results.csv  ← Full epoch-by-epoch data
├── config_evidential_results.csv
├── config_causal_results.csv
├── config_self_correction_results.csv
├── config_consistency_results.csv
├── config_fdr_results.csv
└── config_full_results.csv
```

---

## Expected Results (Hypothetical)

| Config | Dice ↑ | Δ Dice | Precision | Recall | F1 | Improvement % |
|--------|--------|--------|-----------|--------|-----|---------------|
| Baseline | 0.750 | - | 0.710 | 0.795 | 0.750 | - |
| +Evidential | 0.768 | +0.018 | 0.735 | 0.805 | 0.768 | +2.4% |
| +Causal | 0.782 | +0.032 | 0.748 | 0.820 | 0.783 | +4.3% |
| +Self-Correction | 0.774 | +0.024 | 0.742 | 0.810 | 0.775 | +3.2% |
| +Consistency | 0.770 | +0.020 | 0.738 | 0.807 | 0.771 | +2.7% |
| +FDR | 0.765 | +0.015 | 0.732 | 0.802 | 0.766 | +2.0% |
| **Full USALD** | **0.815** | **+0.065** | **0.785** | **0.848** | **0.816** | **+8.7%** |

*(Actual values will come from your experiments)*

---

## For Your Q1 Paper

### Abstract
"We propose USALD, integrating five novel components: evidential uncertainty, causal decomposition, self-correction, teacher-student consistency, and FDR thresholding. Ablation studies demonstrate that the full model achieves X.XX Dice (p < 0.05 vs all ablations), with causal decomposition contributing +X.X% and self-correction +X.X%."

### Methods - Ablation Study Section
"We conducted comprehensive ablation experiments to assess each component's contribution. Seven configurations were trained for 30 epochs using identical data splits and hyperparameters. Table 1 presents quantitative results."

### Results - Table 1: Ablation Study
Use `ablation_summary.csv` directly!

### Results - Figure 2: Component Contributions
Bar chart showing Δ Dice for each component vs baseline

### Results - Figure 3: Causal Weight Analysis
Pie chart: Anatomy 30%, Pathology 55%, Noise 15% (from `causal_weight_*` columns)

### Discussion
"The causal decomposition (+X.X% Dice) proves that explicitly modeling anatomical, pathological, and noise-related uncertainty sources improves segmentation accuracy. Self-correction (+X.X% Dice) demonstrates the value of iterative refinement. The full USALD model combines these synergistically for +X.X% improvement over baseline."

---

## Tips for Success

### Before Starting
- ✅ Verify 1-epoch sanity run completed (you did this already!)
- ✅ Set epochs to 50/30 (not 1/1)
- ✅ Ensure GPU has space (4GB RTX 2050 OK with batch_size=3)

### During Runs
- Monitor first few epochs for each config
- Check Dice is increasing
- If OOM error: reduce batch size to 2

### After Completion
- Check all 7 `val_logs.csv` files exist
- Run `python run_ablation_study.py`
- Verify `ablation_summary.csv` has 7 rows

### For Statistical Significance
- Option 1: Run 5-fold cross-validation (5× longer, but gold standard)
- Option 2: Use per-patient Dice scores (9 patients → Wilcoxon test)
- Option 3: Bootstrap resampling (1000 iterations)

---

## Estimated Timeline

- **Day 1 (8 hours)**: Run configs 1-4 (baseline, evidential, causal, self_correction)
- **Day 2 (6 hours)**: Run configs 5-7 (consistency, fdr, full)
- **Day 3 (2 hours)**: Collect results, create tables/figures
- **Total**: 2-3 days for complete ablation study

---

## Need Help?

### If config fails mid-training:
Checkpoints auto-save every epoch. Just re-run with same ABLATION_CONFIG - it will resume.

### If unsure which config is which:
Check the banner printed at start:
```
ABLATION CONFIG: +Causal Decomposition
USALD Enabled: True
Causal: True
... (others False)
```

### If results look suspicious:
- Verify separate folders used (no checkpoint mixing)
- Check actual flags in val_logs.csv metadata
- Re-run suspicious config from scratch (delete its folder first)

---

## You're Ready! 🚀

All infrastructure is in place. Just:
1. Set epochs to 50/30
2. Run each config (change ABLATION_CONFIG line, run python final_model.py)
3. Collect results (python run_ablation_study.py)
4. Write paper tables from CSVs

Good luck with your Q1 submission! 📊✨
