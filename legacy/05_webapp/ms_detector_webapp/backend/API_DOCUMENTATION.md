# MS Lesion Detection API Documentation

## Updated: Production Model (83.99% Dice Score)

This API uses the **HybridMiniSwin2.5D-ResNet with CSRF and 2.5D-MAE** model trained for 200 MAE epochs + 28 segmentation epochs.

### Model Performance
- **Validation Dice**: 0.8399 (83.99%)
- **Recall**: 0.9164 (91.64% - detects 9/10 lesions)
- **Precision**: 0.7760
- **F1 Score**: 0.8404

---

## Base URL
```
http://localhost:5000
```

---

## Endpoints

### 1. Health Check
**GET** `/health`

Check if the API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0"
}
```

---

### 2. Predict MS Lesions (Multi-Modal - Recommended)
**POST** `/predict`

Upload 3 MRI modalities for optimal prediction accuracy.

**Request:**
- Content-Type: `multipart/form-data`
- Files:
  - `flair_file`: FLAIR MRI scan (.nii or .nii.gz)
  - `t1_file`: T1-weighted MRI scan (.nii or .nii.gz)
  - `t2_file`: T2-weighted MRI scan (.nii or .nii.gz)

**Example (curl):**
```bash
curl -X POST http://localhost:5000/predict \
  -F "flair_file=@path/to/flair.nii.gz" \
  -F "t1_file=@path/to/t1.nii.gz" \
  -F "t2_file=@path/to/t2.nii.gz"
```

**Example (Python):**
```python
import requests

files = {
    'flair_file': open('flair.nii.gz', 'rb'),
    't1_file': open('t1.nii.gz', 'rb'),
    't2_file': open('t2.nii.gz', 'rb')
}

response = requests.post('http://localhost:5000/predict', files=files)
result = response.json()

print(f"Has MS: {result['result']['has_ms']}")
print(f"Severity: {result['result']['severity']}")
print(f"Confidence: {result['result']['confidence']}%")
print(f"Number of lesions: {result['result']['num_lesions']}")
```

**Response:**
```json
{
  "success": true,
  "mode": "multimodal",
  "result": {
    "has_ms": true,
    "severity": "Moderate",
    "confidence": 78.45,
    "max_confidence": 95.23,
    "lesion_volume_ml": 2.34,
    "lesion_volume_percentage": 1.2345,
    "num_lesions": 15,
    "total_lesion_voxels": 2340,
    "scan_dimensions": [256, 256, 180],
    "processed_dimensions": [64, 64, 64],
    "threshold": 0.5
  },
  "filenames": {
    "flair": "flair.nii.gz",
    "t1": "t1.nii.gz",
    "t2": "t2.nii.gz"
  }
}
```

---

### 3. Predict MS Lesions (Single-Modal - Fallback)
**POST** `/predict`

Upload a single MRI scan (any modality). **Note:** Less accurate than multi-modal.

**Request:**
- Content-Type: `multipart/form-data`
- File:
  - `file`: Single MRI scan (.nii or .nii.gz)

**Example (curl):**
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@path/to/scan.nii.gz"
```

**Example (Python):**
```python
import requests

files = {'file': open('flair.nii.gz', 'rb')}
response = requests.post('http://localhost:5000/predict', files=files)
result = response.json()
```

**Response:**
```json
{
  "success": true,
  "mode": "single_modality",
  "result": {
    "has_ms": true,
    "severity": "Mild",
    "confidence": 65.23,
    "max_confidence": 87.45,
    "lesion_volume_ml": 1.12,
    "lesion_volume_percentage": 0.5678,
    "num_lesions": 8,
    "total_lesion_voxels": 1120,
    "scan_dimensions": [256, 256, 180],
    "processed_dimensions": [64, 64, 64],
    "threshold": 0.5,
    "warning": "Single modality used - results may be less accurate"
  },
  "filename": "scan.nii.gz"
}
```

---

### 4. Model Information
**GET** `/model-info`

Get detailed model architecture and performance metrics.

**Response:**
```json
{
  "model_name": "HybridMiniSwin2.5D-ResNet with CSRF and 2.5D-MAE",
  "architecture": "Hybrid 2.5D ResNet + Swin Transformer with CSRF",
  "input_channels": 3,
  "modalities": ["FLAIR", "T1", "T2"],
  "input_size": [64, 64, 64],
  "device": "cuda:0",
  "training": {
    "mae_epochs": 200,
    "segmentation_epochs": 28,
    "early_stopped": true,
    "best_epoch": 28
  },
  "performance": {
    "val_dice": 0.8399,
    "recall": 0.9164,
    "precision": 0.7760,
    "f1_score": 0.8404
  }
}
```

---

## Result Fields Explained

### MS Detection
- **has_ms** (bool): Whether MS lesions are detected
  - `true` if lesion volume > 0.1% OR num_lesions ≥ 2
  - `false` otherwise

### Severity Levels
- **Minimal/None**: < 0.1% of brain volume
- **Mild**: 0.1% - 1.0%
- **Moderate**: 1.0% - 5.0%
- **Severe**: > 5.0%

### Metrics
- **confidence**: Average prediction confidence (0-100%)
- **max_confidence**: Maximum confidence in any voxel (0-100%)
- **lesion_volume_ml**: Total lesion volume in milliliters
- **lesion_volume_percentage**: Lesion volume as % of total brain
- **num_lesions**: Count of distinct lesion regions (connected components)
- **total_lesion_voxels**: Number of voxels classified as lesions

---

## Error Handling

### 400 Bad Request
- No files provided
- Invalid file type (not .nii or .nii.gz)
- Empty filename

**Example:**
```json
{
  "error": "No files provided. Upload either:\n  - 3 files (flair_file, t1_file, t2_file) for best accuracy\n  - 1 file (file) for single-modality prediction"
}
```

### 500 Internal Server Error
- Model not loaded
- Prediction error
- Processing failure

**Example:**
```json
{
  "success": false,
  "error": "Error during prediction: ..."
}
```

---

## File Requirements

### Supported Formats
- `.nii` (NIfTI)
- `.nii.gz` (Compressed NIfTI)

### Preprocessing
- Files are automatically resized to 64×64×64
- Z-score normalization applied per modality
- Multi-modal: FLAIR, T1, T2 stacked as 3-channel input
- Single-modal: Replicated to 3 channels

---

## Running the API

### Start Server
```bash
cd ms_detector_webapp/backend
python app.py
```

### Expected Output
```
======================================================================
🏥 Multiple Sclerosis Detection API
======================================================================

Loading model from: G:\My Drive\NeuroScan_FinalModel_2.5D_MAE\segmentation\best_model.pth
Model loaded successfully!
Validation Dice: 0.8399
Recall: 0.9164
Precision: 0.7760

🚀 Starting Flask server...
📍 API will be available at: http://localhost:5000
🔍 Device: cuda:0
======================================================================
```

---

## Notes

1. **Multi-modal is strongly recommended** for accurate results (83.99% Dice)
2. Single-modal prediction is available but less reliable
3. Model requires ~2GB GPU memory
4. Inference time: ~0.5-2 seconds per scan (GPU) or ~5-10 seconds (CPU)
5. All uploaded files are automatically deleted after processing

---

## Model Comparison

| Model | Dice Score | Recall | Precision | Training Epochs |
|-------|-----------|--------|-----------|-----------------|
| **Current (Final)** | **0.8399** | **0.9164** | 0.7760 | 28 (early stopped) |
| Previous (Trial) | 0.7451 | 0.7634 | 0.7305 | 200 |
| **Improvement** | **+12.72%** | **+16.98%** | +6.23% | -86% training time |

---

## Contact & Support

For issues or questions about the API, please refer to the model training logs or contact the development team.
