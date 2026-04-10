# 📁 Workspace Organization Guide

**Last Updated:** January 25, 2026

This document describes the new organized structure of the EDI Multiple Sclerosis Detector project.

---

## 🗂️ Directory Structure

```
EDI_Multiple_Sclerosis_detector/
│
├── 📄 README.md                          # Main project README
├── 📄 WORKSPACE_ORGANIZATION.md          # This file
│
├── 📁 01_source_code/                    # All source code
│   ├── 📁 models/                        # Model definitions
│   │   ├── final_model.py               # Main 2.5D model (HybridMiniSwin)
│   │   └── model_artifacts/             # Saved model files
│   ├── 📁 training_scripts/             # Training & execution scripts
│   │   ├── resume_training.py           # Resume training script
│   │   └── run_training.bat             # Batch file for training
│   └── 📁 evaluation/                   # Testing & evaluation
│       ├── test_model_evaluation.py     # Model evaluation script
│       ├── tests/                       # Unit tests
│       └── testing/                     # Integration tests
│
├── 📁 02_documentation/                  # All documentation
│   ├── 📁 architecture/                 # Architecture documentation
│   │   ├── 2D5_ARCHITECTURE_VERIFICATION.txt
│   │   ├── FINAL_MODEL_ARCHITECTURE.md
│   │   ├── FINAL_MODEL_CONFIGURATION_ANALYSIS.md
│   │   └── FINAL_OPTIMAL_CONFIGURATION_SUMMARY.md
│   ├── 📁 project_notes/                # Project notes & summaries
│   │   ├── FILE_ESSENTIALITY_ANALYSIS.md
│   │   ├── FULL_EXPLANATION.md
│   │   ├── PROJECT_SIZE_ANALYSIS.md
│   │   └── TODO/                        # TODO lists
│   ├── 📁 research_papers/              # Research & paper materials
│   │   └── research_legacy/             # Legacy research files
│   ├── 📁 docs_legacy/                  # Legacy docs folder
│   └── 📁 documentation_legacy/         # Legacy documentation folder
│
├── 📁 03_ablation_studies/               # Ablation study experiments
│   ├── 📁 experiments/                  # Experiment implementations
│   │   ├── ablation_organized/          # Main ablation experiments
│   │   ├── ablation_codes/              # Ablation code scripts
│   │   ├── USALD_experiments/           # USALD-related experiments
│   │   │   ├── USALD_Ablation_causal/
│   │   │   └── USALD_CausalSelfCorrection/
│   │   ├── MS_cross_validation/         # Cross-validation experiments
│   │   ├── OptimalModel_Evidential/     # Optimal model experiments
│   │   └── trials/                      # Trial runs
│   └── 📁 results/                      # All ablation results
│       ├── ablation_results/            # Aggregated results & CSVs
│       ├── csv_data/                    # Result CSV files
│       ├── checkpoints/                 # Model checkpoints
│       └── logs/                        # Training logs
│
├── 📁 04_utilities/                      # Utility scripts & tools
│   ├── 📁 scripts/                      # General scripts
│   │   ├── main_scripts/                # Main utility scripts
│   │   └── batch_scripts/               # Batch processing scripts
│   ├── 📁 cleanup/                      # Cleanup scripts
│   │   ├── ANALYZE_CHECKPOINTS.ps1
│   │   ├── CLEANUP_*.ps1                # Various cleanup scripts
│   │   ├── SAFE_CLEANUP.ps1
│   │   └── CLEANUP_SUMMARY.md
│   ├── 📁 utilities_legacy/             # Legacy utilities
│   └── 📁 utils_legacy/                 # Legacy utils
│
├── 📁 05_webapp/                         # Web application
│   └── ms_detector_webapp/              # MS detector web interface
│
├── 📁 06_figures_and_visualizations/     # Figures & visualizations
│   ├── paper_figures/                   # Paper figures
│   └── visualization_tools/             # Visualization scripts
│
├── 📁 07_archived/                       # Archived & backup files
│   ├── archive/                         # General archive
│   ├── backups/                         # Backup files
│   └── google_drive_backup/             # Google Drive backups
│
├── 📁 08_configuration/                  # Configuration files
│   ├── pyrightconfig.json               # Python type checking config
│   ├── neuroscan_packages.txt           # Package list
│   └── config/                          # Other config files
│
├── 📁 Dataset/                           # Dataset (PediMS)
└── 📁 LGG/                               # LGG dataset
```

---

## 🎯 Folder Purposes

### 01_source_code/
Contains all executable Python source code:
- **models/** - Model architecture definitions and saved models
- **training_scripts/** - Scripts to train and resume training
- **evaluation/** - Testing, validation, and evaluation scripts

### 02_documentation/
All project documentation:
- **architecture/** - Model architecture specifications
- **project_notes/** - Project status, analyses, and notes
- **research_papers/** - Research materials and paper drafts
- **docs_legacy/** & **documentation_legacy/** - Historical documentation

### 03_ablation_studies/
Complete ablation study experiments and results:
- **experiments/** - All experiment implementations organized by type
- **results/** - CSV results, checkpoints, and logs from experiments

### 04_utilities/
Helper scripts and utilities:
- **scripts/** - General utility and batch scripts
- **cleanup/** - Cleanup and maintenance scripts
- **utilities_legacy/** & **utils_legacy/** - Legacy helper code

### 05_webapp/
Web application for MS lesion detection interface

### 06_figures_and_visualizations/
All paper figures, plots, and visualization tools

### 07_archived/
Historical files, backups, and old versions

### 08_configuration/
Configuration files for development tools and environments

---

## 🚀 Quick Access

### Main Entry Points:
- **Train Model:** `01_source_code/training_scripts/run_training.bat`
- **Model Definition:** `01_source_code/models/final_model.py`
- **Evaluate Model:** `01_source_code/evaluation/test_model_evaluation.py`
- **Web App:** `05_webapp/ms_detector_webapp/`

### Key Documentation:
- **Architecture:** `02_documentation/architecture/FINAL_MODEL_ARCHITECTURE.md`
- **Full Explanation:** `02_documentation/project_notes/FULL_EXPLANATION.md`
- **Ablation Results:** `03_ablation_studies/results/ablation_results/ablation_summary.csv`

---

## 📊 Benefits of New Organization

✅ **Logical Grouping:** Files organized by purpose (code, docs, experiments, etc.)  
✅ **Easy Navigation:** Numbered folders (01-08) show priority/workflow order  
✅ **No Duplication:** Consolidated overlapping folders (docs/, documentation/, scripts/, etc.)  
✅ **Clear Separation:** Development code separated from experiments and archives  
✅ **Scalable:** Easy to add new experiments, docs, or utilities in proper locations  

---

## 🔧 Migration Notes

All files have been moved to their new locations. If you have hardcoded paths in scripts:

**Old Path → New Path Examples:**
```
final_model.py → 01_source_code/models/final_model.py
ablation_results/ → 03_ablation_studies/results/ablation_results/
docs/ → 02_documentation/docs_legacy/
scripts/ → 04_utilities/scripts/main_scripts/
```

Update any import statements or file references accordingly.

---

**Organization Complete! ✨**
