# 🚀 Quick Start Guide - MS Lesion Detection Web App

## ⚡ Fastest Way to Get Started (5 minutes)

### Step 1: Verify Prerequisites
```powershell
# Check Python version (need 3.8+)
python --version

# Check Node.js version (need 14+)
node --version
npm --version
```

### Step 2: Start Backend API
```powershell
cd ms_detector_webapp\backend

# If venv doesn't exist yet:
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Start server
python app.py
```

**Expected Output:**
```
======================================================================
🏥 Multiple Sclerosis Detection API
======================================================================
Loading model from G:\My Drive\NeuroScan_FinalModel_2.5D_MAE\...
✅ Model loaded successfully on cpu
📊 Validation Dice: 0.8399
📊 Recall: 0.9164

🚀 Starting Flask server...
📍 API will be available at: http://localhost:5000
```

### Step 3: Test Backend (New Terminal)
```powershell
# Open browser and visit:
http://localhost:5000/health

# Should see:
# {"device":"cpu","model_loaded":true,"status":"healthy"}
```

### Step 4: Start Frontend (New Terminal)
```powershell
cd ms_detector_webapp\frontend

# If node_modules doesn't exist:
npm install

# Start React app
npm start
```

### Step 5: Use the Application
- Open automatically at: **http://localhost:3000**
- Upload FLAIR, T1, T2 MRI files
- Click "Analyze"
- View results!

---

## 🎯 Using the Launcher Script (Even Easier!)

### Windows Batch File
```cmd
# Double-click this file:
ms_detector_webapp\START_APP.bat
```

### PowerShell Script
```powershell
cd ms_detector_webapp
.\START_APP.ps1
```

**Both will:**
1. Start backend in one terminal
2. Start frontend in another terminal
3. Open http://localhost:3000 automatically

---

## 🧪 Testing the API Directly

### Using curl
```bash
# Health check
curl http://localhost:5000/health

# Model info
curl http://localhost:5000/model-info

# Predict (multi-modal)
curl -X POST http://localhost:5000/predict \
  -F "flair_file=@path/to/flair.nii.gz" \
  -F "t1_file=@path/to/t1.nii.gz" \
  -F "t2_file=@path/to/t2.nii.gz"
```

### Using Python
```python
import requests

# Health check
response = requests.get('http://localhost:5000/health')
print(response.json())

# Predict
files = {
    'flair_file': open('flair.nii.gz', 'rb'),
    't1_file': open('t1.nii.gz', 'rb'),
    't2_file': open('t2.nii.gz', 'rb')
}
response = requests.post('http://localhost:5000/predict', files=files)
result = response.json()

print(f"MS Detected: {result['result']['has_ms']}")
print(f"Severity: {result['result']['severity']}")
print(f"Lesions: {result['result']['num_lesions']}")
```

---

## ❓ Common Issues

### Issue: "ModuleNotFoundError: No module named 'final_model'"
**Solution:** Make sure you're in the `ms_detector_webapp/backend` directory and `final_model.py` exists in the parent directory (`EDI/`).

### Issue: "Port 5000 already in use"
**Solution:** Kill existing process:
```powershell
# Find process using port 5000
netstat -ano | findstr :5000

# Kill it (replace PID with actual number)
taskkill /PID <PID> /F
```

### Issue: "Cannot connect to API"
**Solution:** Ensure backend is running first before starting frontend.

### Issue: Frontend won't start
**Solution:** Delete `node_modules` and reinstall:
```powershell
cd frontend
rm -r node_modules
npm install
npm start
```

---

## 📊 What You Should See

### Backend Startup
```
================================================================================
🚀 FINAL MODEL: HybridMiniSwin2.5D-ResNet with CSRF and 2.5D-MAE
================================================================================
✅ Successfully imported model architecture from final_model.py
======================================================================
🏥 Multiple Sclerosis Detection API
======================================================================
Loading model from G:\My Drive\NeuroScan_FinalModel_2.5D_MAE\segmentation\best_model.pth...
✅ Model loaded successfully on cpu
📊 Training Epoch: 28
📊 Validation Dice: 0.8399
📊 Recall: 0.9164

🚀 Starting Flask server...
📍 API will be available at: http://localhost:5000
🔍 Device: cpu
======================================================================
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

### Frontend Startup
```
Compiled successfully!

You can now view ms-detector-frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.1.5:3000
```

---

## 🎉 Success Checklist

- [ ] Backend running on http://localhost:5000
- [ ] `/health` endpoint returns `{"status":"healthy"}`
- [ ] `/model-info` shows 83.99% Dice score
- [ ] Frontend running on http://localhost:3000
- [ ] Can upload files and see results

---

## 📖 Next Steps

Once everything is working:

1. **Read Full Documentation:**
   - `README.md` - Complete project overview
   - `backend/API_DOCUMENTATION.md` - API details
   - `backend/DEPLOYMENT.md` - Production deployment

2. **Try Sample Data:**
   - Check `test_samples/` directory for test files
   - Or create sample NIfTI: `python scripts/create_sample_nii.py`

3. **Explore the Code:**
   - Backend: `backend/app.py`
   - Frontend: `frontend/src/App.js`
   - Model training: `../final_model.py`

---

## 🆘 Still Having Issues?

Run the automated test:
```powershell
cd backend
python test_api.py
```

This will check:
- ✅ Health endpoint
- ✅ Model info endpoint
- ✅ Prediction validation

If all tests pass, the API is working correctly!

---

**Enjoy using the MS Lesion Detection Web App! 🎉**

Model Performance: 83.99% Dice | 91.64% Recall | 77.60% Precision
