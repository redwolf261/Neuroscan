# EDI Folder Structure
**Last Updated:** November 8, 2025

Complete organization of the EDI project workspace.

---

## 📁 Core Folders

### **models/** (2 files)
Main model architecture and training scripts
- `final_model.py` - Production training script
- `inference.py` - Model inference script

### **scripts/** (19 files)
Python runner and automation scripts
- **Ablation Runners:**
  - `run_ablation_study.py` - Main ablation study runner
  - `run_all_ablations.py` - Run all ablation variants
  - `run_architecture_ablations.py` - Architecture component ablations
  - `run_remaining_ablations.py` - Run pending ablation tests
- **Test Runners:**
  - `run_all_tests.py` - Master test runner (CSRF + Noise)
  - `run_csrf_test.py` - CSRF variants test
  - `run_noise_test.py` - Noise robustness test
- **Data & Migration:**
  - `migrate_dataset.py` - Dataset migration script
  - `migrate_from_google_drive.py` - Google Drive migration
  - `migrate_dataset.ps1` - PowerShell migration script
  - `update_dataset_paths.py` - Update paths to local dataset
- **Utilities:**
  - `check_ablation_ready.py` - Verify ablation prerequisites
  - `collect_hyperparameter_ablation.py` - Collect hyperparameter results
  - `generate_efficiency_comparison.py` - Generate efficiency metrics
  - `organize_edi.py` - Folder organization script

### **research/** (59 files)
Research experiments and analysis scripts
- `noise_robustness_analysis.py` - Noise robustness testing
- `csrf_variants_analysis.py` - CSRF fusion variants
- `hyperparameter_sensitivity.py` - Hyperparameter tuning
- `cross_validation_framework.py` - Cross-validation
- `generate_confusion_matrix.py` - Confusion matrix generation
- **Subfolders:**
  - `validation_scripts/` - Paper data validation
  - `csrf_variants_results_quick/` - CSRF test results
  - `noise_robustness_results_quick/` - Noise test results
  - And more...

### **Dataset/** (1011 files, 6.01 GB)
Training and validation data
- `PediMS/PediMS/` - Main pediatric MS dataset
- 1011 MRI scans and annotations

---

## 📚 Documentation

### **docs/** (17 files)
Comprehensive project documentation
- **Architecture:**
  - `Novel_Model_Architecture.md` - Model architecture description
  - `USALD_Architecture_Description.md` - USALD component details
- **Ablation Studies:**
  - `COMPLETE_ABLATION_INVENTORY.md` - All 25 ablation experiments
  - `ABLATION_COMPLETE_SUMMARY.md` - Ablation results summary
  - `ABLATION_QUICK_START.md` - Quick start guide
  - `ABLATION_WORKFLOW.md` - Ablation workflow
- **Migration & Setup:**
  - `COMPLETE_MIGRATION_GUIDE.md` - Google Drive migration guide
  - `MIGRATION_COMPLETE_README.md` - Migration completion notes
- **Analysis:**
  - `Q1_Journal_Qualification_Analysis.md` - Q1 journal analysis

### **TODO/** (4 files)
Task tracking and progress
- `PENDING_TESTS_TRACKER.md` - Test status (12/15 complete, 80%)
- `QUICK_START_GUIDE.md` - How to run tests
- `TEST_RESULTS_LOG.md` - Results logging template

---

## 🔬 Experiments & Results

### **ablation/** (4 files)
Ablation study checkpoints
- Organized by variant (NoDropout, NoResidual, NoSwin, etc.)
- Each variant has its own folder with checkpoints

### **ablation_results/** (10 files)
Ablation experiment results
- CSV files with metrics for each variant
- Hyperparameter sensitivity results
- Summary files and comparisons

### **csv_data/** (3 files)
CSV data files and results
- Structured experiment results
- Metrics and performance data

---

## 🧪 Testing & Validation

### **tests/** (2 files)
Test and verification scripts
- `test_data_path.py` - Data path validation
- `quick_measure.py` - Quick model performance measurement

### **testing/** 
Additional testing infrastructure
- Test datasets
- Validation scripts

---

## 🛠️ Utilities & Tools

### **utils/**
Utility scripts and tools
- `verify_recovered_files.py` - File recovery verification
- `verify_google_drive_paths.py` - Path verification
- `verify_final_model.py` - Model validation
- And more...

### **utilities/**
Additional utility scripts
- General-purpose tools

### **trials/**
Experimental trial scripts
- `trial.py` - Trial model scripts
- Various experimental runs

### **archive/**
Archived scripts and old files
- `trial_local.py` - Old local trial script
- `verify_trial_paths.py` - Legacy verification
- `verify_deployment.py` - Deployment checks

---

## 📊 Visualization & Output

### **visualization/**
Visualization scripts and outputs
- Chart generation
- Figure creation for papers

### **paper_figures/**
Publication-ready figures
- Charts and graphs for Q1 journal paper

---

## 🔧 Configuration & Backups

### **config/** (2 files)
Configuration files
- Model hyperparameters
- Training settings

### **backups/** (1 file)
Backup files
- `final_model.py.backup` - Model script backup

### **logs/** (5 files)
Log files and execution records
- `dataset_copy_log.txt` - Dataset copy operations
- `dataset_migration_log.txt` - Migration logs
- `migration_log.txt` - General migration logs
- `EDI_FULL_SCAN.json` - Full folder scan
- `GOOGLE_DRIVE_CLEANUP_LIST.txt` - Drive cleanup checklist

### **google_drive_backup/**
Google Drive migration backups
- 72 CSV files organized by category
- Migration summaries

---

## 📦 Other Folders

### **.resume_*** folders
Resume checkpoints for ablation studies
- `.resume_ablation_baseline/`
- `.resume_ablation_causal/`
- `.resume_ablation_consistency/`
- `.resume_ablation_evidential/`
- `.resume_ablation_fdr/`
- `.resume_ablation_self_correction/`
- `.resume_checkpoints/`
- `.resume_checkpoints_usald/`

### **ms_detector_webapp/**
Web application for MS detection
- Flask/Django web interface
- Deployment scripts

### **MS cross validation/**
Cross-validation experiments
- K-fold validation results
- Performance metrics

### **LGG/**
Low-Grade Glioma experiments
- Additional dataset experiments

### **scripts_batch/**
Batch processing scripts
- Bulk operation scripts

### **documentation/**
Legacy documentation folder
- Old documentation files

---

## 📄 Root Files

Only essential files kept in root:
- `README.md` - Main project README

---

## 📈 Statistics

**Total Project Size:** ~12 GB
- Dataset: 6.01 GB (1011 files)
- Results & Checkpoints: ~4 GB
- Code & Documentation: ~50 MB
- Logs & Backups: ~10 MB

**Code Organization:**
- Python scripts: 100+ files
- Documentation: 20+ markdown files
- Configuration: 5+ files
- Test scripts: 10+ files

**Progress Status:**
- Ablation studies: 90% complete (25/~28 experiments)
- Tests remaining: 2 (CSRF variants, Noise robustness)
- Overall completion: 80% (12/15 tasks)

---

## 🎯 Quick Navigation

### Running Experiments
```bash
# Run all remaining tests (3-4 hours)
python scripts/run_all_tests.py

# Run specific test
python scripts/run_csrf_test.py
python scripts/run_noise_test.py

# Run ablation study (14 hours)
python scripts/run_ablation_study.py
```

### Training Model
```bash
# Train final model
python models/final_model.py

# Run inference
python models/inference.py
```

### View Results
- Ablation results: `ablation_results/`
- Research results: `research/*/results/`
- CSV data: `csv_data/`

### Documentation
- Quick start: `TODO/QUICK_START_GUIDE.md`
- Ablation guide: `docs/ABLATION_QUICK_START.md`
- Full inventory: `docs/COMPLETE_ABLATION_INVENTORY.md`

---

**Organization Date:** November 8, 2025  
**Status:** ✅ Fully organized and ready for research
