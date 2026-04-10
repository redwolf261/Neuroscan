# 🧠 MS Detector - AI-Powered Multiple Sclerosis Detection

A full-stack web application for detecting Multiple Sclerosis lesions from MRI scans using deep learning.

![MS Detector](https://img.shields.io/badge/AI-Medical%20Imaging-blue)
![Python](https://img.shields.io/badge/Python-3.10-green)
![React](https://img.shields.io/badge/React-18.2-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)

## 🎯 Features

- **🚀 Fast Analysis** - Get results in seconds with optimized AI model
- **🎯 High Accuracy** - 74.5% Dice score on validation dataset
- **🔒 Private & Secure** - All processing done locally, no data stored
- **💻 Modern UI** - Beautiful drag-and-drop interface
- **📊 Detailed Metrics** - Comprehensive analysis with confidence scores
- **🐳 Docker Ready** - Easy deployment with Docker Compose

## 📋 Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Information](#model-information)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture

```
ms_detector_webapp/
├── backend/              # Flask REST API
│   ├── app.py           # Main Flask application
│   ├── requirements.txt # Python dependencies
│   └── Dockerfile       # Backend container
├── frontend/            # React application
│   ├── src/            # React components
│   ├── public/         # Static assets
│   ├── package.json    # Node dependencies
│   ├── nginx.conf      # Nginx configuration
│   └── Dockerfile      # Frontend container
├── models/             # Trained model files
│   └── model_deployable.pth
└── docker-compose.yml  # Multi-container setup
```

### Tech Stack

**Backend:**
- Flask (REST API)
- PyTorch (Deep Learning)
- NiBabel (NIfTI file processing)
- NumPy, SciPy (Data processing)

**Frontend:**
- React 18
- Axios (HTTP client)
- React Dropzone (File upload)
- CSS3 (Modern styling)

**Model:**
- HybridMiniSwin3D (Custom Swin Transformer)
- PATH 3 BALANCED Configuration
- Trained on PediMS Dataset

## ✅ Prerequisites

### Option 1: Docker (Recommended)
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose
- 8GB RAM minimum
- GPU (optional, for faster inference)

### Option 2: Manual Setup
- Python 3.10+
- Node.js 18+
- npm or yarn
- CUDA (optional, for GPU support)

## 📦 Installation

### Quick Start with Docker

1. **Clone or navigate to the project:**
   ```powershell
   cd C:\Users\HP\EDI\ms_detector_webapp
   ```

2. **Copy your trained model:**
   ```powershell
   # Create models directory
   mkdir models

   # Copy the model file
   copy "G:\My Drive\NeuroScan_PEDiMS_v2\deployment\model_deployable.pth" .\models\
   ```

3. **Start the application:**
   ```powershell
   docker-compose up --build
   ```

4. **Access the app:**
   - Frontend: http://localhost
   - Backend API: http://localhost:5000
   - Health Check: http://localhost:5000/health

### Manual Setup

#### Backend Setup

1. **Navigate to backend:**
   ```powershell
   cd backend
   ```

2. **Create virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Copy model file:**
   ```powershell
   mkdir models
   copy "G:\My Drive\NeuroScan_PEDiMS_v2\deployment\model_deployable.pth" .\models\
   ```

5. **Start backend:**
   ```powershell
   python app.py
   ```

#### Frontend Setup

1. **Open new terminal and navigate to frontend:**
   ```powershell
   cd frontend
   ```

2. **Install dependencies:**
   ```powershell
   npm install
   ```

3. **Start development server:**
   ```powershell
   npm start
   ```

4. **Access the app:**
   - Frontend: http://localhost:3000
   - Backend: http://localhost:5000

## 🚀 Usage

### Web Interface

1. **Open the application** in your browser (http://localhost or http://localhost:3000)

2. **Upload MRI Scan:**
   - Drag and drop a NIfTI file (.nii or .nii.gz)
   - Or click to browse and select a file

3. **Analyze:**
   - Click "Analyze Scan" button
   - Wait for processing (typically 5-10 seconds)

4. **View Results:**
   - MS detection status (Positive/Negative)
   - Confidence score
   - Lesion volume percentage
   - Detailed metrics

### API Usage

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Predict MS
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@path/to/scan.nii.gz"
```

#### Response Format
```json
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

## 📚 API Documentation

### Endpoints

#### `GET /health`
Check API health status
- **Response:** `200 OK`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

#### `GET /model-info`
Get model architecture details
- **Response:** `200 OK`
```json
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

#### `POST /predict`
Analyze MRI scan for MS lesions
- **Content-Type:** `multipart/form-data`
- **Body:** `file` (NIfTI file)
- **Response:** `200 OK` with prediction results
- **Error Response:** `400/500` with error message

## 🧠 Model Information

### HybridMiniSwin3D - PATH 3 BALANCED

**Architecture:**
- Swin Transformer-based 3D U-Net
- Patch size: 4×4×4
- Embedding dimension: 112
- Number of heads: 7
- Depth: 4 layers
- Input size: 64×64×64 voxels

**Training Details:**
- Dataset: PediMS (45 samples)
- Training samples: 36
- Validation samples: 9
- Loss function: Hybrid (70% Focal Tversky + 30% Dice)
- Optimizer: AdamW with OneCycleLR
- Epochs: 200
- Batch size: 3 (effective: 6 with gradient accumulation)

**Performance:**
- Validation Dice: 74.51%
- Precision: 71.43%
- Recall: 78.34%
- F1 Score: 74.72%
- Training time: ~83s per epoch on RTX 2050

**Model File:**
- Size: 3.14 MB (state dict) / 3.25 MB (TorchScript)
- Format: PyTorch .pth file
- Location: `models/model_deployable.pth`

## 🐳 Deployment

### Docker Deployment (Production)

1. **Build and start containers:**
   ```powershell
   docker-compose up -d --build
   ```

2. **Check container status:**
   ```powershell
   docker-compose ps
   ```

3. **View logs:**
   ```powershell
   docker-compose logs -f
   ```

4. **Stop containers:**
   ```powershell
   docker-compose down
   ```

### Cloud Deployment Options

#### AWS EC2
1. Launch EC2 instance (t3.medium or larger)
2. Install Docker and Docker Compose
3. Clone repository and copy model
4. Run `docker-compose up -d`
5. Configure security groups (ports 80, 5000)

#### Google Cloud Platform
1. Create Compute Engine VM
2. Install Docker
3. Deploy with docker-compose
4. Set up Cloud Load Balancing

#### Azure
1. Create Azure Container Instances
2. Deploy backend and frontend containers
3. Configure Azure Load Balancer

### Environment Variables

Create `.env` file in backend directory:
```env
MODEL_PATH=/app/models/model_deployable.pth
FLASK_ENV=production
MAX_CONTENT_LENGTH=524288000  # 500MB
```

## 🔒 Security Considerations

- **File Upload:** Only .nii and .nii.gz files accepted
- **File Size:** Limited to 500MB
- **CORS:** Configured for specific origins in production
- **Data Privacy:** Files are deleted immediately after processing
- **No Storage:** Scan data is never stored permanently
- **HTTPS:** Use reverse proxy (nginx/Caddy) with SSL in production

## 📊 Performance Optimization

### Backend
- Use GPU for faster inference (4-5x speedup)
- Enable mixed precision (FP16) for memory efficiency
- Implement result caching for repeated scans
- Use gunicorn with multiple workers for production

### Frontend
- Build optimization with production mode
- Code splitting and lazy loading
- CDN for static assets
- Gzip compression enabled

## 🧪 Testing

### Backend Tests
```powershell
cd backend
python -m pytest tests/
```

### Frontend Tests
```powershell
cd frontend
npm test
```

### Load Testing
```powershell
# Install Apache Bench
ab -n 100 -c 10 http://localhost:5000/health
```

## 🐛 Troubleshooting

### Model Not Loading
- Ensure `model_deployable.pth` is in `backend/models/` directory
- Check file permissions
- Verify PyTorch version compatibility

### CUDA Out of Memory
- Reduce batch size in preprocessing
- Use CPU inference: `DEVICE=cpu python app.py`
- Upgrade GPU or use cloud GPU instances

### Port Already in Use
```powershell
# Change ports in docker-compose.yml
# Frontend: 80 -> 3000
# Backend: 5000 -> 5001
```

### CORS Issues
- Check backend CORS configuration
- Update allowed origins in `app.py`
- Use proxy configuration in frontend

## 📈 Future Enhancements

- [ ] Multi-modal MRI support (T1, T2, FLAIR)
- [ ] 3D visualization of lesions
- [ ] Batch processing for multiple scans
- [ ] User authentication and history
- [ ] PDF report generation
- [ ] Integration with DICOM format
- [ ] Mobile app version
- [ ] Real-time collaboration features

## 👥 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚕️ Medical Disclaimer

**IMPORTANT:** This application is designed for research and educational purposes only. It should NOT be used as the sole basis for medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for proper medical evaluation, diagnosis, and treatment planning.

## 🙏 Acknowledgments

- PediMS Dataset contributors
- MONAI Project for medical imaging tools
- Swin Transformer authors
- PyTorch and React communities

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Email: your.email@example.com
- Documentation: [Link to detailed docs]

## 📊 Project Status

- ✅ Backend API: Complete
- ✅ Frontend UI: Complete
- ✅ Model Integration: Complete
- ✅ Docker Deployment: Complete
- 🔄 Testing: In Progress
- 📝 Documentation: Complete

---

**Built with ❤️ using PyTorch, Flask, and React**

**Model:** HybridMiniSwin3D (PATH 3 BALANCED) | **Dice Score:** 74.51% | **Training:** 200 Epochs on RTX 2050
