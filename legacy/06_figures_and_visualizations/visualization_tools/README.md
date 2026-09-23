# Visualization Directory

This directory contains all visualization scripts and their outputs.

## 📁 Structure

```
visualization/
├── scripts/          # Python visualization scripts
│   ├── create_segmentation_visualizations.py
│   ├── create_segmentation_visualizations_simple.py
│   ├── create_workflow_diagram.py
│   ├── create_workflow_simple.py
│   ├── architecture_viz.py
│   ├── generate_dataset_examples.py
│   ├── generate_dataset_examples_simple.py
│   └── show_ground_truth.py
│
└── outputs/          # Generated images and documentation
    ├── neuroscan_architecture.png
    ├── workflow_diagram_comprehensive.png
    ├── workflow_diagram_simple.png
    └── WORKFLOW_DIAGRAM_DRAWIO_GUIDE.md
```

## 🎨 Visualization Scripts

### Segmentation Visualizations
- **`create_segmentation_visualizations_simple.py`** - Simplified segmentation overlay generator (working version)
- **`create_segmentation_visualizations.py`** - Full segmentation visualization with model inference

### Workflow Diagrams
- **`create_workflow_diagram.py`** - Comprehensive workflow diagram generator
- **`create_workflow_simple.py`** - Simplified workflow diagram

### Architecture Visualizations
- **`architecture_viz.py`** - Model architecture visualization
- **`show_ground_truth.py`** - Display ground truth masks

### Dataset Examples
- **`generate_dataset_examples.py`** - Generate dataset example figures
- **`generate_dataset_examples_simple.py`** - Simplified dataset examples

## 📊 Output Files

All generated PNG images and documentation are stored in `outputs/`. The segmentation visualizations are stored separately in `paper_figures/segmentation_visualizations/`.

## Usage

To run any visualization script:
```bash
cd C:\Users\HP\EDI
python visualization\scripts\<script_name>.py
```

Example:
```bash
python visualization\scripts\create_segmentation_visualizations_simple.py
```

Outputs will be saved to appropriate directories (usually `outputs/` or `paper_figures/`).
