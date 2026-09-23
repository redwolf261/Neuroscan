# MS Lesion Detection Web Application

AI-powered Multiple Sclerosis lesion detection from multi-modal MRI scans using state-of-the-art deep learning.

## 🎯 Model Performance

- **Validation Dice Score:** 83.99%
- **Recall:** 91.64% (detects 9/10 lesions)
- **Precision:** 77.60%
- **F1 Score:** 84.04%

## 🚀 Quick Start

### Option 1: Quick Launch (Recommended)

1. **Start the Backend API:**
```powershell
cd backend
python app.py
```

2. **Start the Frontend** (in a new terminal):
```powershell
cd frontend
npm start
```

3. **Access the Application:**
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:5000

### Option 2: Using Scripts

```powershell
# One-time setup
.\scripts\setup.ps1

# Launch application
.\scripts\start_app.ps1
```

## 📁 Project Structure

```
ms_detector_webapp/
├── backend/                    # Flask API Server
│   ├── app.py                 # Main API application
│   ├── inference.py           # Standalone prediction module
│   ├── test_api.py            # API test suite
│   ├── requirements.txt       # Python dependencies
│   ├── API_DOCUMENTATION.md   # Complete API reference
│   ├── DEPLOYMENT.md          # Production deployment guide
│   └── venv/                  # Virtual environment
│
├── frontend/                   # React Web Interface
│   ├── src/                   # React components
│   ├── public/                # Static assets
│   ├── package.json           # Node dependencies
│   └── Dockerfile             # Frontend container
│
├── models/                     # Model checkpoints (optional local copy)
├── test_samples/              # Sample NIfTI files for testing
├── docs/                      # Additional documentation
├── scripts/                   # Setup and utility scripts
└── docker-compose.yml         # Container orchestration

Training Code (Parent Directory):
../final_model.py              # Complete training pipeline
../inference.py                # Standalone inference (copy in backend/)
```

## 🏥 Model Architecture

**HybridMiniSwin2.5D-ResNet with CSRF and 2.5D-MAE**

- **Input:** 3-channel MRI (FLAIR + T1 + T2)
- **Processing:** 2.5D volumetric analysis (k=5 slices)
- **Encoder:** ResNet-style with 5 stages [32→64→128→256→512]
- **Attention:** Swin Transformer bottleneck
- **Decoder:** Lightweight upsampling with skip connections
- **Features:** Cross-Stage Residual Fusion (CSRF)
- **Pretraining:** 2.5D Masked Autoencoder (200 epochs)
- **Fine-tuning:** Segmentation (28 epochs, early stopped)

## 🔧 API Endpoints

### Health Check
```bash
GET http://localhost:5000/health
```

### Model Information
```bash
GET http://localhost:5000/model-info
```

### Predict MS Lesions (Multi-Modal - Recommended)
```bash
POST http://localhost:5000/predict
Files: flair_file, t1_file, t2_file (.nii or .nii.gz)
```

### Predict MS Lesions (Single-Modal - Fallback)
```bash
POST http://localhost:5000/predict
File: file (.nii or .nii.gz)
```

## 📊 Usage Examples

### Python API Client
```python
import requests

# Multi-modal prediction (best accuracy)
files = {
    'flair_file': open('flair.nii.gz', 'rb'),
    't1_file': open('t1.nii.gz', 'rb'),
    't2_file': open('t2.nii.gz', 'rb')
}
response = requests.post('http://localhost:5000/predict', files=files)
result = response.json()

print(f"MS Detected: {result['result']['has_ms']}")
print(f"Severity: {result['result']['severity']}")
print(f"Confidence: {result['result']['confidence']}%")
print(f"Lesions Found: {result['result']['num_lesions']}")
print(f"Lesion Volume: {result['result']['lesion_volume_ml']} mL")
```

### Using Inference Module (Standalone)
```python
from backend.inference import MSLesionPredictor

predictor = MSLesionPredictor(
    model_path='path/to/best_model.pth'
)

result = predictor.predict(
    flair_path='flair.nii.gz',
    t1_path='t1.nii.gz',
    t2_path='t2.nii.gz'
)

print(f"Detected: {result['has_ms']}")
print(f"Severity: {result['severity']}")
print(f"Lesions: {result['num_lesions']}")
```

## 📚 Documentation

- **API Reference:** `backend/API_DOCUMENTATION.md`
- **Deployment Guide:** `backend/DEPLOYMENT.md`
- **Project Overview:** `docs/PROJECT_OVERVIEW.md`
- **Quick Start Guide:** `docs/QUICKSTART.md`
- **Training Details:** `../README_FINAL_MODEL.md`

## ⚙️ System Requirements

### Minimum
- **CPU:** 4+ cores
- **RAM:** 8GB
- **Python:** 3.8+
- **Node.js:** 14+

### Recommended
- **GPU:** CUDA-capable with 2GB+ VRAM
- **RAM:** 16GB
- **Storage:** 1GB for model files

## 🔬 Model Training

The model was trained using:
- **Dataset:** PediMS (45 pediatric MS patients)
- **Pretraining:** 200 epochs MAE (final loss: 0.0012)
- **Fine-tuning:** 28 epochs segmentation (early stopped)
- **Loss Function:** Combined Dice + BCE
- **Optimizer:** AdamW with cosine annealing
- **Data Augmentation:** Random flips, rotations

See `../final_model.py` for complete training code.

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access
Frontend: http://localhost:3000
Backend: http://localhost:5000
```

## 🧪 Testing

### Test Backend API
```bash
cd backend
python test_api.py
```

### Test with Sample Data
```bash
# Generate sample NIfTI files
python scripts/create_sample_nii.py

# Use test_samples/ directory
```

## 🛠️ Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Development
```bash
cd frontend
npm install
npm start
```

## 📝 Model Comparison

| Model | Dice Score | Recall | Precision | Training Time |
|-------|-----------|--------|-----------|---------------|
| **Final (Production)** | **83.99%** | **91.64%** | 77.60% | 28 epochs |
| Trial (Baseline) | 74.51% | 76.34% | 73.05% | 200 epochs |
| **Improvement** | **+12.72%** | **+16.98%** | +6.23% | -86% time |

## 🔒 Security Notes

- File validation (only .nii/.nii.gz allowed)
- Automatic file cleanup after processing
- Secure filename sanitization
- Input size limits (Flask default: 16MB)
- No sensitive data stored

## 🤝 Contributing

For issues or enhancements:
1. Check existing documentation
2. Review `backend/API_DOCUMENTATION.md`
3. Test with `test_api.py`

## 📄 License

Medical AI Research Project

## 🔗 Links

- **Model Checkpoint:** `G:\My Drive\NeuroScan_FinalModel_2.5D_MAE\segmentation\best_model.pth`
- **Training Logs:** `G:\My Drive\NeuroScan_FinalModel_2.5D_MAE\segmentation\`
- **Dataset:** PediMS Pediatric MS Dataset

---

**Version:** 2.0  
**Model:** HybridMiniSwin2.5D-ResNet-CSRF v1.0  
**Last Updated:** October 2025  
**Status:** ✅ Production Ready (83.99% Dice Score)
- **Performance:** 74.51% Dice Score
- **Training:** 200 epochs on PediMS dataset

---

**Status:** ✅ Ready for Production
