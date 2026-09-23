# EDI Folder Organization - Quick Reference

## ✅ Organized Structure (November 2, 2025)

### 📁 Root Directory (Clean!)
```
EDI/
├── final_model.py          # Main model architecture
├── inference.py            # Inference script
├── requirements.txt        # Dependencies (CPU)
├── requirements-gpu.txt    # Dependencies (GPU)
└── README.md              # This file
```

### 📂 Main Folders

#### **Core Project Folders**
- **`research/`** (35 files) - All research scripts, validation, analysis
- **`ms_detector_webapp/`** (9 files) - Web application (backend + frontend)
- **`utils/`** (9 files) - Utility functions and helpers

#### **Data Folders**
- **`csv_data/`** - Validation results and metrics
- **`LGG/`** - Brain tumor dataset (cross-pathology validation)
- **`MS cross validation/`** - MS60 dataset (cross-dataset validation)
- **`.resume_checkpoints/`** - Model checkpoints (NOT synced)

#### **Output Folders**
- **`paper_figures/`** (39 files) - Publication figures (300 DPI)
- **`visualization/`** (3 files) - Visualization scripts and outputs
- **`testing/`** (38 files) - Test results and scripts

#### **Organization Folders**
- **`documentation/`** (12 files) - All .md and .txt docs
- **`scripts_batch/`** (3 files) - Windows .bat scripts
- **`trials/`** (2 files) - Experimental trial scripts

#### **Archive Folders**
- **`archive/`** (33 files) - Old code and experiments
- **`ablation/`** (4 files) - Ablation study results
- **`docs/`** (8 files) - Additional documentation
- **`scripts/`** (3 files) - Legacy scripts

---

## 🎯 Where to Find Things

### Looking for Documentation?
→ **`documentation/`** folder
- All .md files (guides, logs, showcases)
- All .txt files (project documentation parts)

### Need to Run Scripts?
→ **`scripts_batch/`** folder
- `START_BACKEND.bat` - Start Flask server
- `START_FRONTEND.bat` - Start React app
- `restart_validation_gpu.bat` - GPU validation

### Want to See Results?
→ **`paper_figures/`** folder
- Cross-dataset validation figures
- Architecture diagrams
- Results tables (CSV)

### Need to Run Validation?
→ **`research/`** folder
- `cross_dataset_validation_lgg.py`
- `cross_dataset_validation_ms60.py`
- `create_paper_figures.py`
- `CROSS_DATASET_VALIDATION_RESULTS.md`

### Working on Visualizations?
→ **`visualization/`** folder
- `architecture_viz.py` - Generate architecture diagram
- `neuroscan_architecture.png` - Current diagram
- `show_ground_truth.py` - Visualize ground truth

### Testing Something Quickly?
→ **`trials/`** folder
- Use for temporary test scripts
- Won't clutter main directories

---

## 🚀 Quick Commands

### Run Main Tasks
```bash
# Inference
python inference.py --input data.nii --output results/

# Generate architecture diagram
python visualization/architecture_viz.py

# Create paper figures
python research/create_paper_figures.py

# Cross-dataset validation
python research/cross_dataset_validation_ms60.py
```

### Start Web Application
```bash
# Backend
scripts_batch/START_BACKEND.bat

# Frontend
scripts_batch/START_FRONTEND.bat
```

---

## 📊 File Count Summary

| Folder | Files | Purpose |
|--------|-------|---------|
| Root | 5 | Core project files only |
| research/ | 35 | Research and validation |
| paper_figures/ | 39 | Publication outputs |
| testing/ | 38 | Test results |
| archive/ | 33 | Old experiments |
| documentation/ | 12 | All docs consolidated |
| ms_detector_webapp/ | 9 | Web application |
| utils/ | 9 | Utilities |
| docs/ | 8 | Additional notes |
| ablation/ | 4 | Ablation studies |
| scripts/ | 3 | Legacy scripts |
| scripts_batch/ | 3 | Batch scripts |
| .resume_checkpoints/ | 3 | Model checkpoints |
| visualization/ | 3 | Viz scripts/outputs |
| MS cross validation/ | 2 | MS60 dataset |
| trials/ | 2 | Test scripts |

**Total:** Clean and organized! ✨

---

## 🔍 Finding Specific Files

### Model Files
- Architecture: `final_model.py` (root)
- Checkpoints: `.resume_checkpoints/seg_resume.pth`

### Validation Results
- MS60 CSV: `csv_data/cross_dataset_validation_ms60_patients/`
- LGG CSV: `csv_data/cross_dataset_validation/`

### Documentation
- Cross-dataset results: `research/CROSS_DATASET_VALIDATION_RESULTS.md`
- Project overview: `documentation/PROJECT_DOCUMENTATION_PART1_OVERVIEW.txt`

### Figures for Paper
- Architecture: `visualization/neuroscan_architecture.png`
- Validation: `paper_figures/cross_dataset_validation_comprehensive.png`

---

**Benefits of New Organization:**
✅ Clean root directory (only 5 essential files)
✅ All documentation in one place
✅ Scripts organized by type
✅ Easy to find outputs (figures, results)
✅ Research work isolated in `research/`
✅ Professional structure for GitHub/sharing

---

Last Updated: November 2, 2025
