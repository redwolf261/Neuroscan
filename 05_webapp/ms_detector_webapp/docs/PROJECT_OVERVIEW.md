# 🎨 MS Detector Web App - Project Overview

## 📁 Project Structure

```
ms_detector_webapp/
│
├── 📂 backend/                      # Flask REST API
│   ├── app.py                       # Main Flask application (HybridMiniSwin3D model)
│   ├── requirements.txt             # Python dependencies
│   ├── Dockerfile                   # Backend container
│   ├── .env.example                 # Environment configuration template
│   └── models/                      # Model files (user must add)
│       └── model_deployable.pth     # Trained model (3.14 MB)
│
├── 📂 frontend/                     # React Application
│   ├── public/
│   │   └── index.html              # HTML template
│   ├── src/
│   │   ├── App.js                  # Main React component
│   │   ├── App.css                 # Styling
│   │   ├── index.js                # React entry point
│   │   └── index.css               # Global styles
│   ├── package.json                # Node.js dependencies
│   ├── Dockerfile                  # Frontend container
│   └── nginx.conf                  # Nginx configuration
│
├── 📂 models/                       # Shared models directory
│   └── model_deployable.pth        # Copy your trained model here
│
├── 📄 docker-compose.yml           # Multi-container orchestration
├── 📄 .gitignore                   # Git ignore rules
│
├── 📘 README.md                    # Complete documentation
├── 📘 QUICKSTART.md                # 5-minute setup guide
│
├── 🚀 setup.ps1                    # Automated setup script
├── 🚀 start_app.ps1                # Start both servers
├── 🚀 start_backend.ps1            # Start backend only
└── 🚀 start_frontend.ps1           # Start frontend only
```

## 🔧 Technology Stack

### Backend (Flask API)
```
┌─────────────────────────────────────┐
│         Flask REST API              │
├─────────────────────────────────────┤
│  • Flask 3.0 (Web Framework)        │
│  • Flask-CORS (Cross-Origin)        │
│  • PyTorch 2.0 (Deep Learning)      │
│  • NiBabel (NIfTI Processing)       │
│  • NumPy (Array Operations)         │
│  • SciPy (Scientific Computing)     │
└─────────────────────────────────────┘
```

### Frontend (React)
```
┌─────────────────────────────────────┐
│         React 18 SPA                │
├─────────────────────────────────────┤
│  • React 18.2 (UI Framework)        │
│  • Axios (HTTP Client)              │
│  • React-Dropzone (File Upload)     │
│  • CSS3 (Modern Styling)            │
│  • Nginx (Production Server)        │
└─────────────────────────────────────┘
```

### Model Architecture
```
┌─────────────────────────────────────┐
│      HybridMiniSwin3D Model         │
├─────────────────────────────────────┤
│  Configuration: PATH 3 BALANCED     │
│  • Embed Dimension: 112             │
│  • Number of Heads: 7               │
│  • Depth: 4 layers                  │
│  • Input Size: 64×64×64             │
│  • Patch Size: 4×4×4                │
│                                     │
│  Performance:                       │
│  • Validation Dice: 74.51%          │
│  • Precision: 71.43%                │
│  • Recall: 78.34%                   │
│  • F1 Score: 74.72%                 │
│  • Model Size: 3.14 MB              │
└─────────────────────────────────────┘
```

## 🔄 Application Flow

```
┌──────────────┐
│    User      │
│   Browser    │
└──────┬───────┘
       │ 1. Upload .nii/.nii.gz file
       ▼
┌──────────────────────────────────────┐
│      React Frontend (Port 3000)      │
│  • Drag & drop interface             │
│  • File validation                   │
│  • Upload progress                   │
└──────┬───────────────────────────────┘
       │ 2. POST /predict with file
       ▼
┌──────────────────────────────────────┐
│      Flask Backend (Port 5000)       │
│  • Receive file                      │
│  • Save temporarily                  │
│  • Validate format                   │
└──────┬───────────────────────────────┘
       │ 3. Preprocess scan
       ▼
┌──────────────────────────────────────┐
│      Image Preprocessing             │
│  • Load NIfTI file (NiBabel)         │
│  • Normalize intensity (0-1)         │
│  • Resize to 64×64×64                │
│  • Convert to tensor                 │
└──────┬───────────────────────────────┘
       │ 4. Run inference
       ▼
┌──────────────────────────────────────┐
│      HybridMiniSwin3D Model          │
│  • Forward pass (GPU/CPU)            │
│  • Sigmoid activation                │
│  • Threshold at 0.5                  │
└──────┬───────────────────────────────┘
       │ 5. Calculate metrics
       ▼
┌──────────────────────────────────────┐
│      Post-processing                 │
│  • Binary mask generation            │
│  • Lesion volume calculation         │
│  • Confidence score                  │
│  • MS detection logic                │
└──────┬───────────────────────────────┘
       │ 6. Return JSON results
       ▼
┌──────────────────────────────────────┐
│      React Frontend                  │
│  • Display results                   │
│  • Show confidence scores            │
│  • Visualize metrics                 │
└──────┬───────────────────────────────┘
       │ 7. User sees results
       ▼
┌──────────────┐
│    User      │
│  (Results)   │
└──────────────┘
```

## 📊 API Endpoints

### 1. Health Check
```http
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### 2. Model Information
```http
GET /model-info

Response:
{
  "model_name": "HybridMiniSwin3D",
  "architecture": "PATH 3 BALANCED",
  "embed_dim": 112,
  "num_heads": 7,
  "depth": 4,
  "input_size": [64, 64, 64],
  "training_dice": 0.7451
}
```

### 3. Predict MS
```http
POST /predict
Content-Type: multipart/form-data
Body: file (NIfTI file)

Response:
{
  "success": true,
  "result": {
    "has_ms": true,
    "confidence": 87.45,
    "lesion_volume_percentage": 2.3456,
    "total_lesion_voxels": 12345,
    "scan_dimensions": [256, 256, 180],
    "processed_dimensions": [64, 64, 64]
  },
  "filename": "patient_scan.nii.gz"
}
```

## 🚀 Deployment Options

### Option 1: Docker (Recommended)
```bash
# One command deployment
docker-compose up --build -d

# Access:
# Frontend: http://localhost
# Backend: http://localhost:5000
```

### Option 2: Manual Setup
```bash
# Backend (Terminal 1)
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py

# Frontend (Terminal 2)
cd frontend
npm install
npm start

# Access:
# Frontend: http://localhost:3000
# Backend: http://localhost:5000
```

### Option 3: PowerShell Scripts
```bash
# Automated setup
.\setup.ps1

# Start everything
.\start_app.ps1
```

## 🎯 Key Features

### Frontend Features
- ✅ Drag-and-drop file upload
- ✅ File validation (.nii, .nii.gz)
- ✅ Upload progress indicator
- ✅ Real-time analysis feedback
- ✅ Beautiful results visualization
- ✅ Responsive design (mobile-friendly)
- ✅ Medical disclaimer
- ✅ Error handling

### Backend Features
- ✅ RESTful API design
- ✅ NIfTI file processing
- ✅ GPU/CPU automatic detection
- ✅ Model caching (loaded once)
- ✅ File cleanup (auto-delete)
- ✅ CORS enabled
- ✅ Error handling
- ✅ Health monitoring

### Security Features
- ✅ File type validation
- ✅ File size limits (500MB)
- ✅ Secure file naming
- ✅ Automatic file deletion
- ✅ No data storage
- ✅ CORS configuration
- ✅ Environment variables

## 📈 Performance Metrics

### Model Performance
- **Validation Dice:** 74.51%
- **Precision:** 71.43%
- **Recall:** 78.34%
- **F1 Score:** 74.72%
- **Training Epochs:** 200
- **Training Time:** 4.62 hours (RTX 2050)
- **Inference Time:** 5-10 seconds per scan

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB
- OS: Windows 10/11, Linux, macOS

**Recommended:**
- CPU: 8 cores
- RAM: 16GB
- GPU: NVIDIA GPU with 4GB+ VRAM
- Storage: 20GB SSD
- OS: Windows 11, Ubuntu 20.04+

## 🔐 Security Considerations

1. **File Upload Security**
   - Only .nii and .nii.gz accepted
   - File size limited to 500MB
   - Files deleted after processing

2. **Data Privacy**
   - No data storage
   - No logging of patient information
   - Processing in memory only

3. **Production Setup**
   - Use HTTPS (SSL/TLS)
   - Implement authentication
   - Rate limiting
   - Input sanitization

## 🐛 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Model not loading | File missing | Copy model to `backend/models/` |
| CORS errors | Backend not running | Start backend on port 5000 |
| Out of memory | Large file + GPU | Use CPU mode or reduce file size |
| Slow inference | CPU mode | Enable GPU in configuration |
| Port in use | Another app using port | Change port in config |

## 📞 Support & Resources

- **Documentation:** README.md
- **Quick Start:** QUICKSTART.md
- **Setup Script:** setup.ps1
- **Issues:** GitHub Issues
- **Email:** your.email@example.com

## 🎓 Learning Resources

- **Flask:** https://flask.palletsprojects.com/
- **React:** https://react.dev/
- **PyTorch:** https://pytorch.org/
- **Docker:** https://docs.docker.com/
- **NiBabel:** https://nipy.org/nibabel/

## 📝 License

MIT License - Free to use for research and educational purposes

## ⚕️ Medical Disclaimer

**IMPORTANT:** This tool is for research and educational purposes only. Not for clinical diagnosis. Always consult qualified healthcare professionals.

---

**Version:** 1.0.0  
**Last Updated:** October 2025  
**Model:** HybridMiniSwin3D (PATH 3 BALANCED)  
**Dice Score:** 74.51%
