# MS Detection Research Project

This directory contains the core training script and production web application for Multiple Sclerosis lesion detection using deep learning.

## 📂 Directory Structure

```
EDI/
├── trial.py                    # 🎯 Main training script (HybridMiniSwin3D)
├── requirements.txt            # Python dependencies (standard)
├── requirements-gpu.txt        # Python dependencies (GPU-optimized)
│
├── ms_detector_webapp/         # 🌐 Production web application
│   ├── backend/               # Flask API with trained model
│   ├── frontend/              # React user interface
│   ├── models/                # Trained model files
│   ├── docs/                  # Complete documentation
│   └── scripts/               # Automation scripts
│
├── ablation/                   # 🔬 Ablation study framework
│   ├── ablation_master.py     # Autonomous ablation control
│   ├── launch_ablation.ps1    # Launch script
│   ├── ablation_monitor.py    # Progress monitoring
│   ├── README.md              # Full ablation documentation
│   └── (6 variant directories after execution)
│
└── archive/                    # 📦 Development history & utilities
    ├── Documentation files
    ├── Training graphs & metrics
    ├── Utility scripts
    └── README.md (detailed inventory)
```

## 🚀 Quick Start

### Training the Model
```powershell
# Install dependencies
pip install -r requirements-gpu.txt

# Run training (requires PEDiMS dataset)
python trial.py
```

### Running the Web Application
```powershell
cd ms_detector_webapp
.\scripts\start_app.ps1
```
Then open http://localhost:3000

### Running Ablation Study
```powershell
cd ablation
.\launch_ablation.ps1
```
Trains 6 model variants (200 epochs each, ~30 hours total)

## 📊 Model Performance (PATH 3 BALANCED)

- **Dice Score:** 74.51%
- **Precision:** 71.43%
- **Recall:** 78.34%
- **F1 Score:** 74.72%
- **Training Time:** ~80s/epoch (200 epochs)
- **GPU:** NVIDIA RTX 2050 4GB

## 📖 Documentation

- **Web App Docs:** See `ms_detector_webapp/docs/`
- **Ablation Study:** See `ablation/README.md`
- **Development History:** See `archive/README.md`

---

**Last Updated:** October 2025
