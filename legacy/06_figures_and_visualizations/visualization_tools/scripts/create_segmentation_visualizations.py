"""
Generate Segmentation Visualization Overlays
Compares model predictions vs ground truth with color-coded overlays
"""

import os
# Fix encoding for Windows
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import model architecture from final_model.py
from final_model import HybridMiniSwin2D5_ResNetEncoder, CSRF_Module, LightweightDecoder, HybridMiniSwin2D5_CSRF

# Configuration
TESTING_DIR = Path(r"C:\Users\HP\EDI\testing")
MODEL_PATH = Path(r"C:\Users\HP\EDI\.resume_checkpoints\seg_resume.pth")
OUTPUT_DIR = Path(r"C:\Users\HP\EDI\paper_figures\segmentation_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Patients to visualize
PATIENTS = ["P1", "P2", "P3", "P4", "P5"]
SLICES_PER_PATIENT = 3  # Show 3 representative slices per patient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model():
    """Load trained model from checkpoint."""
    print("Loading model...")
    
    # Initialize model
    model = HybridMiniSwin2D5_CSRF(k_slices=5, channels=[32, 64, 128, 256, 512]).to(device)
    
    # Load checkpoint
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Loaded model from {MODEL_PATH}")
    else:
        print(f"⚠ Model not found at {MODEL_PATH}")
        return None
    
    model.eval()
    return model

def load_patient_data(patient_id):
    """Load T1, T2, FLAIR, and ground truth for a patient."""
    t1_path = TESTING_DIR / f"{patient_id}_T1.nii.gz"
    t2_path = TESTING_DIR / f"{patient_id}_T2.nii.gz"
    flair_path = TESTING_DIR / f"{patient_id}_FLAIR.nii.gz"
    gt_path = TESTING_DIR / f"{patient_id}_GroundTruth.nii"
    
    # Check if all files exist
    if not all([t1_path.exists(), t2_path.exists(), flair_path.exists(), gt_path.exists()]):
        print(f"⚠ Missing data for {patient_id}")
        return None
    
    # Load volumes
    t1 = nib.load(str(t1_path)).get_fdata()
    t2 = nib.load(str(t2_path)).get_fdata()
    flair = nib.load(str(flair_path)).get_fdata()
    gt = nib.load(str(gt_path)).get_fdata()
    
    return {
        'T1': t1,
        'T2': t2,
        'FLAIR': flair,
        'GT': gt
    }

def preprocess_for_inference(t1, t2, flair):
    """
    Preprocess volumes for model inference.
    Model expects single-channel input (B, 1, D, H, W).
    We use FLAIR as it's most informative for MS lesions.
    """
    # Use FLAIR only (model is single-modality)
    volume = flair.astype(np.float32)
    
    # Normalize (z-score normalization)
    volume_mean = volume[volume > 0].mean() if (volume > 0).any() else volume.mean()
    volume_std = volume[volume > 0].std() if (volume > 0).any() else volume.std()
    
    if volume_std > 0:
        volume = (volume - volume_mean) / volume_std
    
    # Resize to 64x64x64
    from scipy.ndimage import zoom
    target_size = (64, 64, 64)
    zoom_factors = [target_size[i] / volume.shape[i] for i in range(3)]
    volume_resized = zoom(volume, zoom_factors, order=1)
    
    # Add channel and batch dimensions: (1, 1, D, H, W)
    volume_tensor = torch.from_numpy(volume_resized).float().unsqueeze(0).unsqueeze(0)
    
    return volume_tensor

def predict_volume(model, volume):
    """
    Run inference on volume.
    Model outputs 2D mask for central slice: (B, 1, H, W)
    We need to create a full 3D volume by processing each slice.
    """
    with torch.no_grad():
        volume = volume.to(device)
        B, C, D, H, W = volume.shape
        
        # Create output volume
        pred_volume = np.zeros((D, H, W), dtype=np.float32)
        
        # Process volume - model outputs 2D mask for central slice
        # For full 3D prediction, we need to use sliding window or just use center slice
        prediction = model(volume)  # (B, 1, H, W)
        prediction = torch.sigmoid(prediction)
        
        # Place prediction at center slice
        center_slice = D // 2
        pred_volume[center_slice] = prediction.cpu().numpy()[0, 0]
        
        # For visualization, replicate to nearby slices for better visualization
        # (In production, you'd use sliding window)
        for offset in range(-2, 3):
            idx = center_slice + offset
            if 0 <= idx < D:
                pred_volume[idx] = prediction.cpu().numpy()[0, 0]
    
    return pred_volume

def compute_overlay_colors(pred, gt):
    """
    Compute color-coded overlay:
    - Green: True Positive (TP)
    - Red: False Positive (FP)
    - Blue: False Negative (FN)
    - Gray: True Negative (TN) - not shown
    """
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)
    
    tp = np.logical_and(pred == 1, gt == 1)
    fp = np.logical_and(pred == 1, gt == 0)
    fn = np.logical_and(pred == 0, gt == 1)
    
    # Create RGB overlay
    overlay = np.zeros((*pred.shape, 3), dtype=np.float32)
    overlay[tp, 0] = 0    # Green
    overlay[tp, 1] = 1
    overlay[tp, 2] = 0
    
    overlay[fp, 0] = 1    # Red
    overlay[fp, 1] = 0
    overlay[fp, 2] = 0
    
    overlay[fn, 0] = 0    # Blue
    overlay[fn, 1] = 0
    overlay[fn, 2] = 1
    
    return overlay, tp, fp, fn

def select_representative_slices(gt_volume):
    """
    Select representative slices:
    - High lesion load slice
    - Medium lesion load slice
    - Low/challenging slice
    """
    lesion_counts = np.sum(gt_volume > 0, axis=(1, 2))
    
    # Sort by lesion count
    sorted_indices = np.argsort(lesion_counts)[::-1]
    
    # Filter out slices with zero lesions
    valid_indices = sorted_indices[lesion_counts[sorted_indices] > 0]
    
    if len(valid_indices) < 3:
        # Not enough slices with lesions, use all valid
        return valid_indices[:min(3, len(valid_indices))]
    
    # Select high, medium, low
    high_idx = valid_indices[0]
    mid_idx = valid_indices[len(valid_indices) // 2]
    low_idx = valid_indices[-1]
    
    return [high_idx, mid_idx, low_idx]

def visualize_patient(patient_id, data, model):
    """Create visualization for one patient."""
    print(f"Processing {patient_id}...")
    
    # Preprocess and predict
    volume = preprocess_for_inference(data['T1'], data['T2'], data['FLAIR'])
    pred_volume = predict_volume(model, volume)
    
    # Resize ground truth to match prediction
    from scipy.ndimage import zoom
    gt_resized = zoom(data['GT'], [64/data['GT'].shape[i] for i in range(3)], order=0)
    
    # Select representative slices
    slice_indices = select_representative_slices(gt_resized)
    
    if len(slice_indices) == 0:
        print(f"  ⚠ No lesions found in {patient_id}")
        return
    
    # Create figure
    fig, axes = plt.subplots(len(slice_indices), 4, figsize=(16, 4 * len(slice_indices)))
    
    if len(slice_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for row, slice_idx in enumerate(slice_indices):
        # Get slices
        flair_slice = data['FLAIR'][:, :, int(slice_idx * data['FLAIR'].shape[2] / 64)]
        gt_slice = gt_resized[slice_idx]
        pred_slice = pred_volume[slice_idx]
        
        # Compute overlay
        overlay, tp, fp, fn = compute_overlay_colors(pred_slice, gt_slice)
        
        # Compute metrics for this slice
        tp_count = np.sum(tp)
        fp_count = np.sum(fp)
        fn_count = np.sum(fn)
        
        dice = 2 * tp_count / (2 * tp_count + fp_count + fn_count + 1e-7)
        precision = tp_count / (tp_count + fp_count + 1e-7) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count + 1e-7) if (tp_count + fn_count) > 0 else 0
        
        # Plot FLAIR image
        axes[row, 0].imshow(flair_slice.T, cmap='gray', origin='lower')
        axes[row, 0].set_title(f"FLAIR (Slice {slice_idx})")
        axes[row, 0].axis('off')
        
        # Plot ground truth
        axes[row, 1].imshow(flair_slice.T, cmap='gray', origin='lower')
        gt_mask = np.ma.masked_where(gt_slice.T < 0.5, gt_slice.T)
        axes[row, 1].imshow(gt_mask, cmap='Greens', alpha=0.6, origin='lower')
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis('off')
        
        # Plot prediction
        axes[row, 2].imshow(flair_slice.T, cmap='gray', origin='lower')
        pred_mask = np.ma.masked_where(pred_slice.T < 0.5, pred_slice.T)
        axes[row, 2].imshow(pred_mask, cmap='Reds', alpha=0.6, origin='lower')
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis('off')
        
        # Plot overlay
        axes[row, 3].imshow(flair_slice.T, cmap='gray', origin='lower')
        # Transpose overlay to match image orientation
        overlay_T = np.transpose(overlay, (1, 0, 2))  # (W, H, 3)
        # Create mask from sum of channels
        overlay_mask = overlay_T.sum(axis=2) < 0.1  # (W, H)
        overlay_masked = np.ma.masked_where(overlay_mask[:, :, np.newaxis], overlay_T)
        axes[row, 3].imshow(overlay_masked, alpha=0.7, origin='lower')
        axes[row, 3].set_title(f"Overlay | Dice: {dice:.3f}")
        axes[row, 3].axis('off')
    
    # Add legend
    green_patch = mpatches.Patch(color='green', label='True Positive')
    red_patch = mpatches.Patch(color='red', label='False Positive')
    blue_patch = mpatches.Patch(color='blue', label='False Negative')
    fig.legend(handles=[green_patch, red_patch, blue_patch], 
               loc='upper center', ncol=3, fontsize=12, frameon=True)
    
    plt.suptitle(f"{patient_id} - Segmentation Results", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_path = OUTPUT_DIR / f"{patient_id}_segmentation_overlay.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {output_path}")
    plt.close()

def create_summary_figure(patients_data, model):
    """Create a summary figure with one slice from each patient."""
    print("\nCreating summary figure...")
    
    fig, axes = plt.subplots(len(PATIENTS), 4, figsize=(16, 4 * len(PATIENTS)))
    
    for row, patient_id in enumerate(PATIENTS):
        if patient_id not in patients_data:
            continue
        
        data = patients_data[patient_id]
        
        # Preprocess and predict
        volume = preprocess_for_inference(data['T1'], data['T2'], data['FLAIR'])
        pred_volume = predict_volume(model, volume)
        
        # Resize ground truth
        from scipy.ndimage import zoom
        gt_resized = zoom(data['GT'], [64/data['GT'].shape[i] for i in range(3)], order=0)
        
        # Select best slice (highest lesion load)
        slice_indices = select_representative_slices(gt_resized)
        if len(slice_indices) == 0:
            continue
        slice_idx = slice_indices[0]
        
        # Get slices
        flair_slice = data['FLAIR'][:, :, int(slice_idx * data['FLAIR'].shape[2] / 64)]
        gt_slice = gt_resized[slice_idx]
        pred_slice = pred_volume[slice_idx]
        
        # Compute overlay
        overlay, tp, fp, fn = compute_overlay_colors(pred_slice, gt_slice)
        
        # Compute metrics
        tp_count = np.sum(tp)
        fp_count = np.sum(fp)
        fn_count = np.sum(fn)
        dice = 2 * tp_count / (2 * tp_count + fp_count + fn_count + 1e-7)
        
        # Plot
        axes[row, 0].imshow(flair_slice.T, cmap='gray', origin='lower')
        axes[row, 0].set_ylabel(patient_id, fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(flair_slice.T, cmap='gray', origin='lower')
        gt_mask = np.ma.masked_where(gt_slice.T < 0.5, gt_slice.T)
        axes[row, 1].imshow(gt_mask, cmap='Greens', alpha=0.6, origin='lower')
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(flair_slice.T, cmap='gray', origin='lower')
        pred_mask = np.ma.masked_where(pred_slice.T < 0.5, pred_slice.T)
        axes[row, 2].imshow(pred_mask, cmap='Reds', alpha=0.6, origin='lower')
        axes[row, 2].axis('off')
        
        axes[row, 3].imshow(flair_slice.T, cmap='gray', origin='lower')
        # Transpose overlay to match image orientation
        overlay_T = np.transpose(overlay, (1, 0, 2))  # (W, H, 3)
        # Create mask from sum of channels
        overlay_mask = overlay_T.sum(axis=2) < 0.1  # (W, H)
        overlay_masked = np.ma.masked_where(overlay_mask[:, :, np.newaxis], overlay_T)
        axes[row, 3].imshow(overlay_masked, alpha=0.7, origin='lower')
        axes[row, 3].set_title(f"Dice: {dice:.3f}", fontsize=10)
        axes[row, 3].axis('off')
    
    # Column titles
    axes[0, 0].set_title("FLAIR Image", fontsize=12, fontweight='bold')
    axes[0, 1].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axes[0, 2].set_title("Prediction", fontsize=12, fontweight='bold')
    axes[0, 3].set_title("Overlay", fontsize=12, fontweight='bold')
    
    # Legend
    green_patch = mpatches.Patch(color='green', label='True Positive')
    red_patch = mpatches.Patch(color='red', label='False Positive')
    blue_patch = mpatches.Patch(color='blue', label='False Negative')
    fig.legend(handles=[green_patch, red_patch, blue_patch], 
               loc='upper center', ncol=3, fontsize=12, frameon=True)
    
    plt.suptitle("Segmentation Results - All Patients", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = OUTPUT_DIR / "all_patients_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary to {output_path}")
    plt.close()

def main():
    print("="*80)
    print("SEGMENTATION VISUALIZATION GENERATION")
    print("="*80)
    
    # Load model
    model = load_model()
    if model is None:
        print("✗ Failed to load model")
        return
    
    # Load all patient data
    patients_data = {}
    for patient_id in PATIENTS:
        data = load_patient_data(patient_id)
        if data is not None:
            patients_data[patient_id] = data
    
    print(f"\n✓ Loaded data for {len(patients_data)} patients")
    
    # Generate individual visualizations
    print("\nGenerating individual patient visualizations...")
    for patient_id, data in patients_data.items():
        visualize_patient(patient_id, data, model)
    
    # Generate summary figure
    create_summary_figure(patients_data, model)
    
    print("\n" + "="*80)
    print(f"✓ ALL VISUALIZATIONS SAVED TO: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
