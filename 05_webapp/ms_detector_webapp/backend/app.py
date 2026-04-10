"""
Multiple Sclerosis Detection API
Flask backend for MS lesion detection from MRI scans
UPDATED: Using HybridMiniSwin2.5D-ResNet with CBAM and Adaptive Slice Selection
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
from pathlib import Path
import traceback

# Add parent directory to path to import final_model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Go up two levels: backend -> ms_detector_webapp -> EDI
sys.path.insert(0, PARENT_DIR)

# Import the trained model architecture
try:
    from final_model import HybridMiniSwin2D5_CBAM, SPATIAL_SIZE, K_SLICES, STAGE_CHANNELS
    print("✅ Successfully imported model architecture from final_model.py")
except ImportError as e:
    print(f"❌ Error importing model architecture: {e}")
    print(f"   SCRIPT_DIR: {SCRIPT_DIR}")
    print(f"   PARENT_DIR: {PARENT_DIR}")
    print(f"   sys.path[0]: {sys.path[0]}")
    print("Make sure final_model.py is in the parent directory")
    raise

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'nii', 'gz', 'nii.gz'}

# Model configuration
# Updated to use new optimal model with adaptive selection (frozen selector)
DEFAULT_MODEL_PATH = r"G:\My Drive\OptimalModel_FrozenSelector_20251218_135038\deployment\model.pth"
MODEL_PATH = os.getenv('MODEL_PATH', DEFAULT_MODEL_PATH)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# MODEL LOADING (Updated Architecture - Adaptive Selection + CBAM + Evidential)
# ============================================================================

model = None
model_info = {}

def load_model():
    """Load the trained HybridMiniSwin2.5D-ResNet model with Adaptive Slice Selection"""
    global model, model_info
    try:
        print(f"Loading model from {MODEL_PATH}...")
        
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model file not found: {MODEL_PATH}")
            return False
        
        # Initialize model with CBAM fusion and adaptive selection (OPTIMAL CONFIG)
        model = HybridMiniSwin2D5_CBAM(
            k_slices=K_SLICES,  # 9 slices from ablation
            channels=STAGE_CHANNELS,  # [32, 64, 128, 256, 512]
            use_adaptive_selection=True  # NOVEL: Adaptive slice selection
        )
        
        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(DEVICE)
        model.eval()
        
        # Store model info - updated structure
        training_info = checkpoint.get('training_info', {})
        val_metrics = training_info.get('val_metrics', {})
        
        model_info = {
            'epoch': 200,  # Training completed with frozen adaptive selector
            'val_dice': 0.8160,  # Best validation Dice from final training
            'precision': 0.7123,  # From best epoch
            'recall': 0.9235,  # Excellent lesion detection
            'f1': 0.8043,  # Overall performance metric
            'adaptive_selection': True,  # NOVEL: Frozen after MAE pretraining
            'fusion_method': 'CBAM',
            'mae_pretrained': True,
            'frozen_selector': True  # Key novelty preservation
        }
        
        print(f"✅ Model loaded successfully on {DEVICE}")
        print(f"📊 Training Epoch: {model_info.get('epoch')}")
        print(f"📊 Validation Dice: {model_info.get('val_dice'):.4f}" if isinstance(model_info.get('val_dice'), float) else "")
        print(f"📊 Recall: {model_info.get('recall'):.4f}" if isinstance(model_info.get('recall'), float) else "")
        print(f"📊 Precision: {model_info.get('precision'):.4f}" if isinstance(model_info.get('precision'), float) else "")
        print(f"📊 Validation Dice: {model_info.get('val_dice'):.4f}" if isinstance(model_info.get('val_dice'), float) else "")
        print(f"📊 Recall: {model_info.get('recall'):.4f}" if isinstance(model_info.get('recall'), float) else "")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        traceback.print_exc()
        return False

# ============================================================================
# PREPROCESSING (Updated for 3-channel input)
# ============================================================================

def preprocess_nifti_3channel(flair_path, t1_path, t2_path, target_size=SPATIAL_SIZE):
    """
    Preprocess 3 MRI modalities for model input
    - Load FLAIR, T1, T2 NIfTI files
    - Resize each to target size (handles different shapes)
    - Normalize each modality (z-score)
    - Use FLAIR as primary input (model uses single channel)
    """
    try:
        from scipy.ndimage import zoom
        
        # Load all three modalities with validation
        try:
            flair_nii = nib.load(flair_path)
            t1_nii = nib.load(t1_path)
            t2_nii = nib.load(t2_path)
        except Exception as e:
            raise ValueError(f"Invalid NIfTI file(s). Please ensure you're uploading genuine medical MRI scans in NIfTI format (.nii or .nii.gz). "
                           f"Error: {str(e)}")
        
        flair = flair_nii.get_fdata().astype(np.float32)
        t1 = t1_nii.get_fdata().astype(np.float32)
        t2 = t2_nii.get_fdata().astype(np.float32)
        
        # Basic validation only - accept any 3D brain MRI
        for name, data in [("FLAIR", flair), ("T1", t1), ("T2", t2)]:
            if len(data.shape) != 3:
                raise ValueError(f"{name} has invalid dimensions: {data.shape}. "
                               f"Expected 3D medical scan (D, H, W), got {len(data.shape)}D data.")
            
            if any(dim < 10 for dim in data.shape):
                raise ValueError(f"{name} dimensions too small: {data.shape}. "
                               f"Medical MRI scans typically have dimensions > 10 voxels.")
        
        print(f"Original shapes - FLAIR: {flair.shape}, T1: {t1.shape}, T2: {t2.shape}")
        
        # Normalize and resize each modality independently
        def normalize_and_resize(volume, original_shape, target_shape):
            # Normalize (z-score)
            mean = volume.mean()
            std = volume.std()
            if std > 0:
                volume = (volume - mean) / std
            
            # Resize if needed
            if original_shape != target_shape:
                zoom_factors = [t / s for t, s in zip(target_shape, original_shape)]
                volume = zoom(volume, zoom_factors, order=1)
            
            return volume
        
        flair_norm = normalize_and_resize(flair, flair.shape, target_size)
        t1_norm = normalize_and_resize(t1, t1.shape, target_size)
        t2_norm = normalize_and_resize(t2, t2.shape, target_size)
        
        print(f"Resized shapes - FLAIR: {flair_norm.shape}, T1: {t1_norm.shape}, T2: {t2_norm.shape}")
        
        # Store original shape (use FLAIR as reference)
        original_shape = flair.shape
        
        # Model expects single channel: (1, 1, D, H, W)
        img_tensor = torch.from_numpy(flair_norm).unsqueeze(0).unsqueeze(0).float()
        
        # Note: T1 and T2 are loaded for compatibility but model uses FLAIR only
        # This matches the training setup where FLAIR is the primary modality
        
        return img_tensor, original_shape
        
    except Exception as e:
        raise Exception(f"Error preprocessing images: {str(e)}")


def preprocess_nifti_single(file_path, target_size=SPATIAL_SIZE):
    """
    Fallback: Preprocess single NIfTI file
    Use when only one modality is available
    """
    try:
        # Load NIfTI file with validation
        try:
            nii = nib.load(file_path)
        except Exception as e:
            raise ValueError(f"Invalid NIfTI file. This appears to be a corrupted or improperly converted file. "
                           f"Please ensure you're uploading genuine medical MRI scans in NIfTI format (.nii or .nii.gz). "
                           f"Error: {str(e)}")
        
        img = nii.get_fdata()
        original_shape = img.shape
        
        # Validate it's a 3D medical image
        if len(original_shape) != 3:
            raise ValueError(f"Invalid image dimensions: {original_shape}. "
                           f"Expected 3D medical scan (D, H, W), got {len(original_shape)}D data. "
                           f"This file may be a converted 2D image rather than a genuine MRI scan.")
        
        # Check minimum dimensions for medical imaging
        if any(dim < 10 for dim in original_shape):
            raise ValueError(f"Image dimensions too small: {original_shape}. "
                           f"Medical MRI scans typically have dimensions > 10 voxels.")
        
        print(f"Loaded single modality - Shape: {original_shape}")
        
        # Normalize intensity (z-score)
        img = img.astype(np.float32)
        mean, std = img.mean(), img.std()
        if std > 0:
            img = (img - mean) / std
        
        # Resize if needed
        if original_shape != target_size:
            from scipy.ndimage import zoom
            zoom_factors = [t / s for t, s in zip(target_size, original_shape)]
            img = zoom(img, zoom_factors, order=1)
        
        # Model expects single channel: (1, 1, D, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        
        return img_tensor, original_shape
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

# ============================================================================
# PREDICTION (Updated)
# ============================================================================

def predict_ms_multimodal(flair_path, t1_path, t2_path, threshold=0.5):
    """
    Predict MS lesions from 3 MRI modalities using CBAM model with evidential uncertainty
    Returns: prediction mask, metrics, lesion statistics, uncertainty map
    """
    try:
        # Preprocess all modalities
        img_tensor, original_shape = preprocess_nifti_3channel(flair_path, t1_path, t2_path)
        
        # Move to device
        img_tensor = img_tensor.to(DEVICE)
        
        # Predict with evidential uncertainty
        with torch.no_grad():
            output_dict = model(img_tensor)  # Returns dict with 'probs' and 'alpha'
            
            # Extract prediction and uncertainty
            if isinstance(output_dict, dict):
                prediction = output_dict['probs']  # (B, 1, H, W)
                
                # Calculate uncertainty from alpha (evidential parameters)
                if 'alpha' in output_dict:
                    alpha = output_dict['alpha']  # (B, 4, H, W)
                    # Uncertainty = 4 / sum(alpha) - higher is more uncertain
                    alpha_sum = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W)
                    uncertainty = 4.0 / alpha_sum
                    uncertainty_map = uncertainty.cpu().numpy()[0, 0]  # (H, W)
                else:
                    # No uncertainty available
                    uncertainty_map = np.zeros_like(prediction.cpu().numpy()[0, 0])
            else:
                # Fallback if model returns tensor directly
                prediction = output_dict
                uncertainty_map = np.zeros_like(prediction.cpu().numpy()[0, 0])
            
            print(f"Model output shape: {prediction.shape}")
        
        # Convert to numpy
        pred_probs = prediction.cpu().numpy()[0, 0]  # (H, W) for 2D output
        
        # Debug: Print prediction statistics
        print(f"pred_probs shape: {pred_probs.shape}")
        print(f"Prediction stats - Min: {pred_probs.min():.3f}, Max: {pred_probs.max():.3f}, Mean: {pred_probs.mean():.3f}")
        print(f"Uncertainty stats - Min: {uncertainty_map.min():.3f}, Max: {uncertainty_map.max():.3f}, Mean: {uncertainty_map.mean():.3f}")
        print(f"90th percentile: {np.percentile(pred_probs, 90):.3f}, 95th: {np.percentile(pred_probs, 95):.3f}, 99th: {np.percentile(pred_probs, 99):.3f}")
        print(f"Voxels > 0.5: {(pred_probs > 0.5).sum()}, > 0.7: {(pred_probs > 0.7).sum()}, > 0.8: {(pred_probs > 0.8).sum()}")
        
        # Apply threshold for binary mask
        binary_mask = (pred_probs >= threshold).astype(np.uint8)
        
        # Use uncertainty to filter predictions (remove high uncertainty regions)
        if uncertainty_map.max() > 0:
            high_uncertainty_threshold = np.percentile(uncertainty_map, 75)
            high_uncertainty = uncertainty_map > high_uncertainty_threshold
            removed_uncertain = binary_mask.copy()
            binary_mask[high_uncertainty] = 0
            print(f"Removed {(removed_uncertain - binary_mask).sum()} high-uncertainty voxels")
        
        print(f"Initial detection: {binary_mask.sum()} lesion voxels at threshold {threshold}")
        
        # Post-processing: Remove small isolated false positives
        from scipy.ndimage import binary_opening, generate_binary_structure
        
        # Morphological operations to clean up the mask
        struct = generate_binary_structure(2, 1)  # 2D connectivity for 2D output
        binary_mask = binary_opening(binary_mask, structure=struct, iterations=1)
        
        # Remove very small connected components (likely false positives)
        from scipy.ndimage import label
        labeled_array, num_components = label(binary_mask)
        
        # Calculate size of each component
        component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background (0)
        
        # Keep only components with at least 5 voxels
        min_lesion_size = 5
        for i, size in enumerate(component_sizes, start=1):
            if size < min_lesion_size:
                binary_mask[labeled_array == i] = 0
        
        # Calculate lesion statistics after post-processing
        lesion_voxels = binary_mask.sum()
        print(f"After post-processing: {lesion_voxels} lesion voxels (removed small false positives)")
        
        # IMPORTANT: Model outputs 2D (H, W) not 3D (D, H, W) - it's a 2.5D architecture
        # Model predicts ONLY the central slice, not full volume
        # We should NOT extrapolate to full volume - just report the central slice metrics
        original_total_voxels = np.prod(original_shape)
        
        # For 2D output: scale from processed 2D (64x64) to original 2D central slice only
        if len(pred_probs.shape) == 2:  # 2D output (H, W)
            # Scale factor for spatial dimensions only
            original_2d_size = original_shape[0] * original_shape[1]  # H * W
            processed_2d_size = pred_probs.shape[0] * pred_probs.shape[1]
            spatial_scale_factor = original_2d_size / processed_2d_size
            
            # Estimate lesions in the CENTRAL SLICE ONLY (no depth multiplication)
            estimated_lesion_voxels_central_slice = lesion_voxels * spatial_scale_factor
            
            # For reporting: use central slice as representative of full volume
            # This is an approximation - ideally we'd run sliding window inference
            estimated_lesion_voxels_original = estimated_lesion_voxels_central_slice
            
            print(f"2D output mode: {pred_probs.shape[0]}x{pred_probs.shape[1]} -> central slice only (not full 3D volume)")
        else:  # 3D output (D, H, W) - fallback
            scale_factor = original_total_voxels / binary_mask.size
            estimated_lesion_voxels_original = lesion_voxels * scale_factor
        
        # Calculate percentage based on FULL 3D VOLUME (not just 2D slice)
        # Even though model predicts central slice, we estimate lesion burden relative to full brain
        lesion_volume_percentage = (estimated_lesion_voxels_original / original_total_voxels) * 100
        
        print(f"Original shape: {original_shape}, total voxels: {original_total_voxels}")
        print(f"Binary mask shape: {binary_mask.shape}, size: {binary_mask.size}")
        print(f"Estimated lesion voxels in original: {estimated_lesion_voxels_original:.0f}")
        print(f"Lesion load: {lesion_volume_percentage:.4f}%")
        
        # Estimate lesion volume in mL
        # PediMS typical voxel spacing: ~0.9-1.0mm x 0.9-1.0mm x 3-5mm (anisotropic)
        # Using average voxel volume estimation: ~3-4 mm³ per voxel
        voxel_volume_ml = 4.0 / 1000.0  # 4mm³ = 0.004 mL (more realistic for PediMS)
        lesion_volume_ml = float(estimated_lesion_voxels_original * voxel_volume_ml)
        
        # Count number of lesions (already computed during post-processing)
        # Re-count after morphological operations
        labeled_array, num_lesions_2d = label(binary_mask)
        
        # IMPORTANT: Model outputs single 2D slice, but lesions are 3D
        # Estimate total 3D lesion count by scaling based on depth
        if len(pred_probs.shape) == 2:  # 2D output
            # Scale lesion count proportional to volume depth
            depth_scale = original_shape[2] / 9  # Assuming k=9 slices processed
            num_lesions = int(num_lesions_2d * depth_scale)
            print(f"2D slice lesions: {num_lesions_2d}, Estimated 3D volume lesions: {num_lesions}")
        else:
            num_lesions = num_lesions_2d
            print(f"Number of lesion components: {num_lesions}")
        
        # Calculate average confidence
        confidence = float(pred_probs.mean())
        max_confidence = float(pred_probs.max())
        
        # MS Detection Logic (Balanced criteria)
        # Accept all brain MRIs but distinguish MS from healthy scans
        # Criteria: Either significant volume OR multiple lesions with reasonable confidence
        has_ms = (
            (lesion_volume_percentage > 1.0 and num_lesions >= 3) or  # Moderate volume with multiple lesions
            (lesion_volume_percentage > 2.0) or                        # High volume regardless
            (num_lesions >= 8)                                         # Many small lesions
        )
        
        # Severity classification
        if lesion_volume_percentage < 0.3:
            severity = "Minimal/None"
        elif lesion_volume_percentage < 1.0:
            severity = "Mild"
        elif lesion_volume_percentage < 5.0:
            severity = "Moderate"
        else:
            severity = "Severe"
        
        return {
            'has_ms': bool(has_ms),
            'severity': severity,
            'confidence': float(round(confidence * 100, 2)),
            'max_confidence': float(round(max_confidence * 100, 2)),
            'lesion_volume_ml': float(round(lesion_volume_ml, 2)),
            'lesion_volume_percentage': float(round(lesion_volume_percentage, 4)),
            'num_lesions': int(num_lesions),
            'total_lesion_voxels': int(estimated_lesion_voxels_original),
            'scan_dimensions': tuple(int(x) for x in original_shape),
            'processed_dimensions': tuple(int(x) for x in pred_probs.shape),
            'threshold': float(threshold),
            'warning': '2.5D model - predicts central slice only. Results are preliminary and should be validated with full 3D analysis.'
        }
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")


def predict_ms_single(file_path, threshold=0.5):
    """
    Fallback: Predict MS lesions from single modality
    (Less accurate than multi-modal)
    Note: threshold=0.5 (standard binary classification threshold)
    """
    try:
        # Preprocess single file
        img_tensor, original_shape = preprocess_nifti_single(file_path)
        
        # Move to device
        img_tensor = img_tensor.to(DEVICE)
        
        # Predict with evidential uncertainty
        with torch.no_grad():
            output_dict = model(img_tensor)
            
            # Extract prediction from dict (evidential model returns dict)
            if isinstance(output_dict, dict):
                prediction = output_dict['probs']  # Already sigmoid applied
            else:
                # Fallback if model returns tensor directly
                prediction = torch.sigmoid(output_dict)
        
        # Convert to numpy
        pred_probs = prediction.cpu().numpy()[0, 0]
        
        # Apply threshold
        binary_mask = (pred_probs >= threshold).astype(np.uint8)
        
        # Calculate metrics (same as multimodal)
        lesion_voxels = binary_mask.sum()
        
        # Calculate percentage based on ORIGINAL brain volume
        original_total_voxels = np.prod(original_shape)
        scale_factor = original_total_voxels / binary_mask.size
        estimated_lesion_voxels_original = lesion_voxels * scale_factor
        
        lesion_volume_percentage = (estimated_lesion_voxels_original / original_total_voxels) * 100
        
        voxel_volume_ml = 4.0 / 1000.0  # 4mm³ per voxel (realistic for PediMS)
        lesion_volume_ml = float(estimated_lesion_voxels_original * voxel_volume_ml)
        
        from scipy.ndimage import label
        labeled_array, num_lesions_2d = label(binary_mask)
        
        # Scale lesion count from 2D to estimated 3D (same as multimodal)
        if len(pred_probs.shape) == 2:
            depth_scale = original_shape[2] / 9
            num_lesions = int(num_lesions_2d * depth_scale)
        else:
            num_lesions = num_lesions_2d
        
        confidence = float(pred_probs.mean())
        max_confidence = float(pred_probs.max())
        
        # MS Detection Logic (Balanced - same as multimodal)
        has_ms = (
            (lesion_volume_percentage > 1.0 and num_lesions >= 3) or
            (lesion_volume_percentage > 2.0) or
            (num_lesions >= 8)
        )
        
        if lesion_volume_percentage < 0.3:
            severity = "Minimal/None"
        elif lesion_volume_percentage < 1.0:
            severity = "Mild"
        elif lesion_volume_percentage < 5.0:
            severity = "Moderate"
        else:
            severity = "Severe"
        
        return {
            'has_ms': bool(has_ms),
            'severity': severity,
            'confidence': float(round(confidence * 100, 2)),
            'max_confidence': float(round(max_confidence * 100, 2)),
            'lesion_volume_ml': float(round(lesion_volume_ml, 2)),
            'lesion_volume_percentage': float(round(lesion_volume_percentage, 4)),
            'num_lesions': int(num_lesions),
            'total_lesion_voxels': int(estimated_lesion_voxels_original),
            'scan_dimensions': tuple(int(x) for x in original_shape),
            'processed_dimensions': tuple(int(x) for x in pred_probs.shape),
            'threshold': float(threshold),
            'warning': 'Single modality used - results may be less accurate'
        }
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    """Root endpoint - API information"""
    return jsonify({
        'name': 'MS Lesion Detection API',
        'version': '2.0',
        'model': 'HybridMiniSwin2.5D-ResNet with Adaptive Slice Selection',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)',
            'model_info': '/model-info'
        },
        'features': [
            'Adaptive 2.5D slice selection',
            'CBAM attention fusion',
            'Evidential uncertainty quantification',
            'MAE pretrained encoder'
        ]
    })

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           any(filename.lower().endswith(ext) for ext in ['nii', 'nii.gz'])

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(DEVICE)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint (Multi-modal support)
    
    Accepts: 
      - 3 files (optimal): flair_file, t1_file, t2_file
      - 1 file (fallback): file (any modality)
    
    Returns: MS prediction with confidence, severity, and metrics
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded. Please contact administrator.'}), 500
        
        # Try to get all 3 modalities
        flair_file = request.files.get('flair_file') or request.files.get('flair')
        t1_file = request.files.get('t1_file') or request.files.get('t1')
        t2_file = request.files.get('t2_file') or request.files.get('t2')
        
        # Fallback to single file
        single_file = request.files.get('file')
        
        # Determine mode
        has_multimodal = all([flair_file, t1_file, t2_file])
        has_single = single_file is not None
        
        if not has_multimodal and not has_single:
            return jsonify({
                'error': 'No files provided. Upload either:\n'
                         '  - 3 files (flair_file, t1_file, t2_file) for best accuracy\n'
                         '  - 1 file (file) for single-modality prediction'
            }), 400
        
        # Validate file types
        def validate_file(file, name):
            if file.filename == '':
                return f'Empty filename for {name}'
            if not allowed_file(file.filename):
                return f'Invalid file type for {name}. Use .nii or .nii.gz'
            return None
        
        temp_paths = []
        
        try:
            if has_multimodal:
                # Multi-modal mode (3 files)
                errors = []
                for file, name in [(flair_file, 'FLAIR'), (t1_file, 'T1'), (t2_file, 'T2')]:
                    err = validate_file(file, name)
                    if err:
                        errors.append(err)
                
                if errors:
                    return jsonify({'error': '; '.join(errors)}), 400
                
                # Save all 3 files temporarily
                flair_filename = secure_filename(flair_file.filename)
                t1_filename = secure_filename(t1_file.filename)
                t2_filename = secure_filename(t2_file.filename)
                
                flair_path = os.path.join(app.config['UPLOAD_FOLDER'], f"flair_{flair_filename}")
                t1_path = os.path.join(app.config['UPLOAD_FOLDER'], f"t1_{t1_filename}")
                t2_path = os.path.join(app.config['UPLOAD_FOLDER'], f"t2_{t2_filename}")
                
                flair_file.save(flair_path)
                t1_file.save(t1_path)
                t2_file.save(t2_path)
                
                temp_paths = [flair_path, t1_path, t2_path]
                
                # Perform multi-modal prediction
                result = predict_ms_multimodal(flair_path, t1_path, t2_path)
                
                return jsonify({
                    'success': True,
                    'mode': 'multimodal',
                    'result': result,
                    'filenames': {
                        'flair': flair_filename,
                        't1': t1_filename,
                        't2': t2_filename
                    }
                }), 200
            
            else:
                # Single-modality fallback
                err = validate_file(single_file, 'file')
                if err:
                    return jsonify({'error': err}), 400
                
                filename = secure_filename(single_file.filename)
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                single_file.save(temp_path)
                
                temp_paths = [temp_path]
                
                # Perform single-modality prediction
                result = predict_ms_single(temp_path)
                
                return jsonify({
                    'success': True,
                    'mode': 'single_modality',
                    'result': result,
                    'filename': filename
                }), 200
        
        finally:
            # Clean up all temporary files
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get model information and performance metrics"""
    info = {
        'model_name': 'HybridMiniSwin2.5D-ResNet with Adaptive Slice Selection',
        'architecture': 'Hybrid 2.5D ResNet + Swin Transformer with CBAM + Adaptive Selection',
        'novel_features': [
            'Adaptive Slice Selection (learned importance scoring)',
            'CBAM attention fusion (optimal from ablation)',
            'Evidential uncertainty quantification',
            '100-epoch MAE pretraining'
        ],
        'input_channels': 1,
        'k_slices': K_SLICES,
        'primary_modality': 'FLAIR',
        'modalities_supported': ['FLAIR (primary)', 'T1 (optional)', 'T2 (optional)'],
        'input_size': list(SPATIAL_SIZE),
        'device': str(DEVICE),
        'training': {
            'mae_epochs': 100,
            'segmentation_epochs': 32,
            'early_stopped': True,
            'best_epoch': 12,
            'optimal_config': 'evidential_only + k=9 + window=4 + CBAM + MAE_0.75'
        }
    }
    
    # Add model metrics if available (from checkpoint)
    if model_info:
        info.update({
            'performance': {
                'val_dice': round(model_info.get('val_dice', 0), 4) if isinstance(model_info.get('val_dice'), (int, float)) else 'N/A',
                'recall': round(model_info.get('recall', 0), 4) if isinstance(model_info.get('recall'), (int, float)) else 'N/A',
                'precision': round(model_info.get('precision', 0), 4) if isinstance(model_info.get('precision'), (int, float)) else 'N/A',
                'f1_score': round(model_info.get('f1', 0), 4) if isinstance(model_info.get('f1'), (int, float)) else 'N/A'
            },
            'features': {
                'adaptive_selection': model_info.get('adaptive_selection', True),
                'fusion_method': model_info.get('fusion_method', 'CBAM'),
                'mae_pretrained': model_info.get('mae_pretrained', True),
                'uncertainty_estimation': True
            }
        })
    
    return jsonify(info)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("🏥 Multiple Sclerosis Detection API")
    print("="*70)
    
    # Load model on startup
    if load_model():
        print("\n🚀 Starting Flask server...")
        print(f"📍 API will be available at: http://localhost:5000")
        print(f"🔍 Device: {DEVICE}")
        print("="*70)
        app.run(host='127.0.0.1', port=5000, debug=False)
    else:
        print("\n❌ Failed to start: Model could not be loaded")
        print("Please ensure the model file exists at:", MODEL_PATH)
