# 🚀 Quick Start Guide

Get your MS Detector web app running in 5 minutes!

## Prerequisites
- Windows 10/11
- Python 3.10+
- Node.js 18+
- Your trained model file

## Setup Steps

### Step 1: Copy Your Model

Copy your trained model to the models directory:

```powershell
# From the project root (ms_detector_webapp)
mkdir models
copy "G:\My Drive\NeuroScan_PEDiMS_v2\deployment\model_deployable.pth" .\models\
```

### Step 2: Run Setup Script

```powershell
.\setup.ps1
```

This will:
- Create Python virtual environment
- Install all Python dependencies
- Install all Node.js dependencies
- Copy your model file
- Set up the project structure

### Step 3: Start the Application

**Option A: Start Everything (Recommended)**
```powershell
.\start_app.ps1
```
This opens two terminal windows - one for backend, one for frontend.

**Option B: Start Manually**

Terminal 1 (Backend):
```powershell
.\start_backend.ps1
```

Terminal 2 (Frontend):
```powershell
.\start_frontend.ps1
```

### Step 4: Open the App

Open your browser and go to:
```
http://localhost:3000
```

## Using the App

1. **Upload** - Drag and drop your MRI scan (.nii or .nii.gz)
2. **Analyze** - Click "Analyze Scan" button
3. **Results** - View MS detection results with confidence scores

## Troubleshooting

### Backend won't start
- Check if model file exists: `.\backend\models\model_deployable.pth`
- Check if virtual environment activated
- Try: `pip install -r backend\requirements.txt`

### Frontend won't start
- Check if Node.js installed: `node --version`
- Try: `cd frontend; npm install`
- Check if port 3000 is available

### Model not loading
- Verify model file size (should be ~3.14 MB)
- Check file path in backend\app.py
- Ensure PyTorch is installed correctly

### CORS errors
- Make sure backend is running on port 5000
- Check browser console for details
- Verify CORS is enabled in backend\app.py

## Testing the API

### Health Check
```powershell
curl http://localhost:5000/health
```

### Model Info
```powershell
curl http://localhost:5000/model-info
```

### Predict (with file)
```powershell
curl -X POST http://localhost:5000/predict -F "file=@path\to\scan.nii.gz"
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [API Documentation](#) for endpoint details
- See [Deployment Guide](#) for production deployment

## Common Issues

| Issue | Solution |
|-------|----------|
| Port already in use | Change port in backend\app.py or frontend\package.json |
| Model out of memory | Use CPU mode or reduce batch size |
| Slow predictions | Enable GPU in backend configuration |
| File upload fails | Check file size limit (500MB max) |

## Support

For issues or questions:
- Check the README.md troubleshooting section
- Open an issue on GitHub
- Contact: your.email@example.com

---

**Ready to deploy?** See the [Deployment Guide](README.md#deployment) for production setup!
