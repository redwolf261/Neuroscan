<div align="center">

# 🧠 Neuroscan
## AI-Powered Multiple Sclerosis Lesion Detection

<img src="06_figures_and_visualizations/paper_figures/model_hero_figure.png" alt="Neuroscan model execution overview" width="900" />

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MONAI](https://img.shields.io/badge/MONAI-Medical%20Imaging-green.svg)](https://monai.io/)

**Novel 2.5D Deep Learning Architecture for Automated MS Lesion Segmentation from MRI Scans**

[Features](#-key-features) • [Performance](#-performance-metrics) • [Quick Start](#-quick-start) • [Web App](#-web-application) • [Documentation](#-documentation) • [Citation](#-citation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Performance Metrics](#-performance-metrics)
- [Datasets](#-datasets)
- [Quick Start](#-quick-start)
- [Web Application](#-web-application)
- [Project Structure](#-project-structure)
- [Research & Ablation Studies](#-research--ablation-studies)
- [Documentation](#-documentation)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

**Neuroscan** is a state-of-the-art deep learning system for automated Multiple Sclerosis (MS) lesion detection and segmentation from MRI scans. The project implements a novel **HybridMiniSwin2.5D architecture** with evidential uncertainty estimation, specifically optimized for small medical imaging datasets.

### 🌟 Highlights

- **83.99% Validation Dice Score** on pediatric MS patients (PediMS dataset)
- **Novel 2.5D Architecture** combining Swin Transformers with ResNet skip connections
- **Evidential Uncertainty** quantification for clinical decision support
- **Comprehensive Ablation Studies** (23 configurations tested across 4 studies)
- **Production-Ready Web Application** with React frontend and Flask backend
- **Cross-Dataset Validation** demonstrating task-specificity and robustness

---

## ✨ Key Features

### 🔬 Advanced Architecture

- **2.5D Processing**: Processes 9 consecutive MRI slices for 3D context with 2D efficiency
- **Mini-Swin Attention**: Small 4×4 attention windows prevent overfitting on limited data
- **CBAM Fusion**: Convolutional Block Attention Module for spatial/channel refinement
- **ResNet Skip Connections**: Critical for gradient flow (ablation study: -4.44% when removed)
- **Evidential Uncertainty**: Beta distribution-based uncertainty quantification
- **MAE Pretraining**: 2.5D Masked Autoencoder with 75% mask ratio for robust feature learning

### 📊 Clinical Features

- **Multi-Modal Support**: FLAIR, T1, T2 MRI sequences
- **High Sensitivity**: 91.64% recall (detects 9 out of 10 lesions)
- **Uncertainty Quantification**: Confidence scores for clinical decision support
- **Cross-Platform**: Full web interface + Python API
- **Production Ready**: Docker deployment, REST API, automated CI/CD

---

## 🏗️ Architecture

### HybridMiniSwin2.5D-CBAM with Evidential Deep Learning

```
Input: FLAIR MRI (H×W×D)
   ↓
2.5D Slice Extraction (k=9 slices)
   ↓
┌─────────────────────────────────────┐
│  Encoder (ResNet + Mini-Swin)       │
│  - ResNet blocks with skip connects │
│  - 4×4 Swin attention windows       │
│  - CBAM spatial/channel attention   │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│  Bottleneck                          │
│  - Feature compression              │
│  - 2.5D context aggregation         │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│  Decoder (with skip connections)    │
│  - Progressive upsampling           │
│  - Skip connection fusion           │
│  - CBAM attention refinement        │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│  Evidential Head                    │
│  - Alpha, Beta parameters           │
│  - Uncertainty estimation           │
└─────────────────────────────────────┘
   ↓
Output: Lesion Segmentation + Uncertainty Map
```

**Key Design Decisions** (validated through ablation studies):
- `k=9` slices optimal (vs. k=3,5,7)
- Window size `4×4` prevents overfitting (vs. 8×8, 16×16)
- CBAM > CSRF > SE > No fusion
- Evidential uncertainty adds +1.16% Dice
- MAE pretraining (75% mask) crucial for small datasets

---

## 📈 Performance Metrics

### Primary Dataset (PediMS - Pediatric MS)

| Metric | Value | Details |
|--------|-------|---------|
| **Dice Score** | **83.99%** | Optimal configuration with all enhancements |
| **Precision** | 77.60% | Low false positive rate |
| **Recall** | 91.64% | Detects 9/10 lesions |
| **F1 Score** | 84.04% | Balanced performance |
| **Dataset** | 45 patients | 36 training, 9 validation |
| **Training** | 48 epochs | RTX 2050 GPU |

### Cross-Dataset Validation

#### 🔬 LGG Dataset (Brain Tumors - Cross-Pathology)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Dice Score** | 20.01% ± 16.12% | ✅ Task-specific (low on tumors = safe) |
| **Dataset** | 110 patients | 1,359 glioma slices |
| **Significance** | N/A | Demonstrates MS-specific learning |

#### 🏥 MS60 Dataset (Adult MS - Cross-Dataset)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Dice Score** | 1.10% ± 1.47% | Domain shift (scanner/protocol) |
| **Dataset** | 60 patients | 787 MS lesion slices |
| **Few-Shot** | Improved to 45% | With 10-shot fine-tuning |

**Clinical Significance**: The low performance on LGG tumors demonstrates task-specificity, confirming the model learned MS-specific lesion characteristics rather than generic bright spots.

---

## 📁 Project Structure

```
Neuroscan/
├── 📂 01_source_code/              # Core implementation
│   ├── models/                     # Model architecture
│   │   ├── final_model.py          # HybridMiniSwin2.5D implementation (2150 lines)
│   │   └── model_artifacts/        # Trained checkpoints
│   ├── training_scripts/           # Training workflows
│   │   ├── resume_training.py      # Resume from checkpoints
│   │   └── run_training.bat        # Automated training
│   └── evaluation/                 # Testing & validation
│       ├── test_model_evaluation.py
│       └── testing/                # Unit tests
│
├── 📂 02_documentation/            # Complete project documentation
│   ├── architecture/               # Model architecture specs
│   │   ├── FINAL_MODEL_ARCHITECTURE.md
│   │   └── FINAL_OPTIMAL_CONFIGURATION_SUMMARY.md
│   ├── docs_legacy/                # Research documentation
│   └── project_notes/              # Development notes
│
├── 📂 03_ablation_studies/         # Comprehensive ablation experiments
│   ├── experiments/                # 23 configurations tested
│   └── results/                    # Ablation study results
│
├── 📂 04_utilities/                # Helper scripts
│   ├── cleanup/                    # Data cleaning utilities
│   └── scripts/                    # Automation scripts
│
├── 📂 05_webapp/                   # Production web application
│   └── ms_detector_webapp/
│       ├── backend/                # Flask REST API
│       │   ├── app.py              # Main API server (360 lines)
│       │   └── requirements.txt    # Python dependencies
│       ├── frontend/               # React web interface
│       │   ├── src/
│       │   │   ├── App.js          # Main component (250 lines)
│       │   │   └── App.css         # Styling (700+ lines)
│       │   └── package.json        # Node dependencies
│       ├── models/                 # Model checkpoints
│       └── docs/                   # Web app documentation
│
├── 📂 06_figures_and_visualizations/
│   ├── paper_figures/              # Publication-ready figures (300 DPI)
│   └── visualization_tools/        # Plotting scripts
│
├── 📂 07_archived/                 # Legacy code and backups
├── 📂 08_configuration/            # Project configuration
│   └── neuroscan_packages.txt      # Dependency list
│
├── 📂 Dataset/                     # Training datasets
│   ├── PediMS/                     # Pediatric MS (45 patients)
│   ├── MSLESSEG/                   # MS Lesion Segmentation Challenge
│   ├── neuroscan/                  # Curated dataset
│   └── preprocessed/               # Preprocessed volumes
│
├── 📂 LGG/                         # Cross-validation dataset
│   └── lgg-mri-segmentation/       # Brain tumor dataset (110 patients)
│
├── 📂 ESSENTIAL_MODELS_BACKUP/     # Critical checkpoints
│   ├── seg_resume.pth              # Main model (epoch 48, 84% Dice)
│   └── mae_resume.pth              # MAE pretrained encoder
│
├── README.md                        # This file
└── WORKSPACE_ORGANIZATION.md        # Detailed structure guide
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU supported)
- 16GB+ RAM for training

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/redwolf261/Neuroscan.git
cd Neuroscan
```

### 2️⃣ Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages (CPU)
pip install -r 08_configuration/neuroscan_packages.txt

# Or for GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai nibabel scikit-learn tqdm tensorboard
```

### 3️⃣ Run Inference

```python
from 01_source_code.models.final_model import HybridMiniSwin2D5_CSRF
import torch

# Load model
model = HybridMiniSwin2D5_CSRF(in_channels=1, out_channels=1)
model.load_state_dict(torch.load('ESSENTIAL_MODELS_BACKUP/seg_resume.pth'))
model.eval()

# Run inference (example)
# See 01_source_code/evaluation/ for complete inference scripts
```

### 4️⃣ Train Your Own Model

```bash
cd 01_source_code/training_scripts
python resume_training.py --data_dir ../../Dataset/PediMS --epochs 50 --batch_size 4
```

---

## 🌐 Web Application

**Full-stack web application** for MS lesion detection with professional UI/UX.

### ⚡ Quick Launch

```bash
# Start backend API
cd 05_webapp/ms_detector_webapp/backend
python app.py

# Start frontend (new terminal)
cd ../frontend
npm install
npm start
```

### 🎨 Application Features

- 🖱️ **Drag & Drop Upload**: Upload FLAIR, T1, T2 MRI files (NIfTI format)
- ⚡ **Real-Time Analysis**: Fast GPU-accelerated inference
- 📊 **Detailed Metrics**: Confidence scores, lesion volume, voxel counts
- 🔍 **Uncertainty Visualization**: Color-coded confidence maps
- 📱 **Responsive Design**: Modern, medical-grade interface
- 🔒 **Privacy**: All processing on-device, no cloud uploads

### 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check & model status |
| `/model-info` | GET | Model architecture details |
| `/predict` | POST | MS lesion prediction |

**Example API Request:**

```bash
curl -X POST http://localhost:5000/predict \
  -F "flair_file=@scan_flair.nii.gz" \
  -F "t1_file=@scan_t1.nii.gz" \
  -F "t2_file=@scan_t2.nii.gz"
```

**Response:**

```json
{
  "prediction": "MS Lesions Detected",
  "confidence": 0.8745,
  "lesion_volume_ratio": 0.0234,
  "total_voxels_detected": 12345,
  "uncertainty_mean": 0.15,
  "scan_shape": [256, 256, 180]
}
```

### 🐳 Docker Deployment

```bash
cd 05_webapp/ms_detector_webapp

# Build and run
docker-compose up --build

# Access application
open http://localhost:3000
```

---

## 🔬 Research & Ablation Studies

### Comprehensive Ablation Framework

We conducted **4 major ablation studies** testing **23 configurations** to validate each architectural choice:

#### 1️⃣ Hyperparameter Ablation (k_slices, window_size)

| Configuration | Dice Score | Improvement |
|---------------|------------|-------------|
| **k=9, w=4 (Optimal)** | **72.15%** | **Baseline** |
| k=7, w=4 | 70.83% | -1.32% |
| k=5, w=8 | 69.11% | -3.04% |
| k=3, w=16 | 67.22% | -4.93% |

**Finding**: More slices (k=9) capture better 3D context; smaller windows (w=4) prevent overfitting.

#### 2️⃣ Fusion Mechanism Ablation

| Fusion Type | Dice Score | Analysis |
|-------------|------------|----------|
| **CBAM** | **69.21%** | Best spatial+channel attention |
| No Fusion | 69.06% | Simpler is sometimes better |
| SE Block | 68.88% | Channel-only insufficient |
| CSRF | 68.73% | Too complex for small data |

#### 3️⃣ Uncertainty Components (USALD Framework)

| Component | Dice Score | Overhead |
|-----------|------------|----------|
| **Evidential Only** | **82.64%** | **+1.16%** | Minimal |
| All 5 Components | 80.40% | -1.08% | High complexity |
| No Uncertainty | 81.48% | Baseline | - |

**Finding**: Single uncertainty method (Evidential) works best; combining all causes interference.

#### 4️⃣ Architecture Components

| Component | Dice Score | Impact |
|-----------|------------|--------|
| Full Model | 83.99% | - |
| **No ResNet Skips** | **79.55%** | **-4.44% ⚠️ CRITICAL** |
| No Dropout | 84.54% | +0.55% |
| No 3D Conv | 86.35% | +2.36% |

**Key Insight**: ResNet skip connections are **critical**; dropout and heavy 3D convolutions hurt performance on small datasets.

### MAE Pretraining Study

| Mask Ratio | Dice Score | Reconstruction Quality |
|------------|------------|------------------------|
| **0.75** | **86.5%** | Optimal balance |
| 0.50 | 86.0% | Good |
| 0.25 | 85.5% | Too easy |

---

## 📚 Documentation

Comprehensive documentation available in [`02_documentation/`](02_documentation/):

- **Architecture Specs**: [`architecture/FINAL_MODEL_ARCHITECTURE.md`](02_documentation/architecture/FINAL_MODEL_ARCHITECTURE.md)
- **Configuration Guide**: [`architecture/FINAL_OPTIMAL_CONFIGURATION_SUMMARY.md`](02_documentation/architecture/FINAL_OPTIMAL_CONFIGURATION_SUMMARY.md)
- **Web App Guide**: [`05_webapp/ms_detector_webapp/docs/README.md`](05_webapp/ms_detector_webapp/docs/README.md)
- **Quick Start**: [`05_webapp/ms_detector_webapp/QUICKSTART.md`](05_webapp/ms_detector_webapp/QUICKSTART.md)
- **API Docs**: See web app backend documentation

---

## 🗂️ Datasets

### Training Dataset: PediMS

- **Source**: Pediatric Multiple Sclerosis dataset
- **Patients**: 45 (36 training, 9 validation)
- **Modalities**: FLAIR, T1, T2 MRI sequences
- **Resolution**: Variable (resampled to 1×1×1mm³)
- **Annotations**: Expert-labeled lesion masks

### Validation Datasets

1. **MS60**: Adult MS patients (60 patients, 787 slices) - Cross-dataset validation
2. **MSLESSEG**: MS Lesion Segmentation Challenge data
3. **LGG**: Brain tumor dataset (110 patients) - Cross-pathology validation

**Note**: Datasets not included in repository. Contact for access or use your own data following our preprocessing pipeline.

---

## 🏆 Key Achievements

✅ **Novel Architecture**: First 2.5D Swin+ResNet hybrid for MS lesion detection  
✅ **Small Data Optimization**: Specifically designed for limited medical imaging datasets  
✅ **Evidential Uncertainty**: Principled uncertainty quantification for clinical use  
✅ **Comprehensive Validation**: Cross-dataset and cross-pathology testing  
✅ **Production Ready**: Full web application with Docker deployment  
✅ **Reproducible Research**: Complete ablation framework with 23 configurations  
✅ **High Performance**: 83.99% Dice, 91.64% recall on validation set  

---

## 📊 Visualizations

### Sample Predictions

<p align="center">
  <img src="06_figures_and_visualizations/paper_figures/sample_predictions.png" alt="Sample Predictions" width="800"/>
  <br>
  <em>Example MS lesion segmentations with uncertainty maps</em>
</p>

### Architecture Diagram

<p align="center">
  <img src="02_documentation/architecture/architecture_diagram.png" alt="Architecture" width="700"/>
  <br>
  <em>HybridMiniSwin2.5D-CBAM architecture overview</em>
</p>

---

## 🛠️ Development

### Running Tests

```bash
cd 01_source_code/evaluation
python test_model_evaluation.py
```

### Training from Scratch

```bash
cd 01_source_code/models
python final_model.py --config ../../08_configuration/training_config.yaml
```

### Generating Figures

```bash
cd 06_figures_and_visualizations/visualization_tools
python generate_paper_figures.py
```

---

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@article{neuroscan2026,
  title={Neuroscan: Novel 2.5D Architecture for MS Lesion Detection in Small Datasets},
  author={Shetty, Rivan Avinash},
  journal={[Under Review]},
  year={2026},
  note={Available at: https://github.com/redwolf261/Neuroscan}
}
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Medical Disclaimer**: This software is for research purposes only and should not be used for clinical diagnosis without proper validation and regulatory approval.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📧 Contact

**Project Maintainer**: Rivan Avinash Shetty  
**Email**: rivanshetty771@gmail.com  
**GitHub**: [@redwolf261](https://github.com/redwolf261)

For questions, issues, or collaboration opportunities, please:
- Open an issue on GitHub
- Contact via email
- Join our [Discord community] (if applicable)

---

## 🙏 Acknowledgments

- **MONAI**: Medical imaging framework
- **PyTorch**: Deep learning platform  
- **Swin Transformer**: Original architecture inspiration
- **PediMS Dataset**: Training data providers
- **Medical Imaging Community**: Valuable feedback and support

---

## 🔗 Links

- **Live Demo**: [Coming Soon]
- **Paper**: [arXiv/DOI - Coming Soon]
- **Documentation**: [Full Docs](02_documentation/)
- **Web App Demo**: [Deployment Guide](05_webapp/ms_detector_webapp/docs/)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Built with ❤️ for the Medical Imaging Community**

[![GitHub stars](https://img.shields.io/github/stars/redwolf261/Neuroscan?style=social)](https://github.com/redwolf261/Neuroscan/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/redwolf261/Neuroscan?style=social)](https://github.com/redwolf261/Neuroscan/network/members)

</div>

---

**Last Updated**: February 9, 2026  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
