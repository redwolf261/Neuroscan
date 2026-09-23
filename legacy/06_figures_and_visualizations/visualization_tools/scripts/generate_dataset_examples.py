"""
Generate Dataset Visualization Examples for Research Paper
===========================================================
Creates publication-quality figures showing:
- FLAIR images from PediMS training dataset
- Ground truth lesion masks
- Model predictions
- Side-by-side comparisons
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from final_model import HybridMiniSwin2D5_CSRF

# Paths
DATASET_PATH = Path(r"G:\My Drive\Dataset\PediMS\PediMS")
CHECKPOINT_PATH = Path(r"C:\Users\HP\EDI\.resume_checkpoints\seg_resume.pth")
OUTPUT_DIR = Path(r"C:\Users\HP\EDI\paper_figures\dataset_examples")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PediMS patient list (training set - 36 patients)
TRAINING_PATIENTS = [
    'P03', 'P05', 'P06', 'P07', 'P08', 'P09', 'P11', 'P12', 'P13', 'P14',
    'P16', 'P17', 'P18', 'P20', 'P21', 'P22', 'P23', 'P24', 'P26', 'P27',
    'P28', 'P29', 'P30', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'P38',
    'P39', 'P40', 'P42', 'P43', 'P44', 'P45'
]


def load_model():
    """Load trained model"""
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HybridMiniSwin2D5_CSRF(
        in_channels=1,
        out_channels=1,
        channels=[32, 64, 128, 256, 512],
        num_res_units=4
    ).to(device)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    return model, device


def load_patient_data(patient_id):
    """Load FLAIR and ground truth for a patient"""
    patient_dir = DATASET_PATH / patient_id
    
    # Load FLAIR
    flair_path = patient_dir / f"{patient_id}_FLAIR.nii.gz"
    flair_nii = nib.load(str(flair_path))
    flair = flair_nii.get_fdata()
    
    # Load ground truth
    gt_path = patient_dir / f"{patient_id}_GroundTruth.nii"
    gt_nii = nib.load(str(gt_path))
    gt = gt_nii.get_fdata()
    
    return flair, gt


def normalize_flair(flair_volume):
    """Normalize FLAIR volume"""
    p1, p99 = np.percentile(flair_volume[flair_volume > 0], [1, 99])
    flair_norm = np.clip(flair_volume, p1, p99)
    flair_norm = (flair_norm - p1) / (p99 - p1)
    return flair_norm


def predict_slice(model, device, flair_volume, slice_idx, depth=5):
    """Generate prediction for a slice"""
    H, W, D = flair_volume.shape
    
    # Get 2.5D context
    half_depth = depth // 2
    start_idx = max(0, slice_idx - half_depth)
    end_idx = min(D, slice_idx + half_depth + 1)
    
    slices = []
    for i in range(start_idx, end_idx):
        slices.append(flair_volume[:, :, i])
    
    # Pad if needed
    while len(slices) < depth:
        if len(slices) == 0:
            slices.append(np.zeros((H, W)))
        else:
            slices.append(slices[-1])
    
    # Stack and prepare input
    volume = np.stack(slices, axis=0)  # (depth, H, W)
    volume = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)  # (1, 1, depth, H, W)
    
    # Resize to 240x240 if needed
    if H != 240 or W != 240:
        volume = torch.nn.functional.interpolate(
            volume.view(1, depth, H, W), 
            size=(240, 240), 
            mode='bilinear', 
            align_corners=False
        ).view(1, 1, depth, 240, 240)
    
    # Predict
    with torch.no_grad():
        volume = volume.to(device)
        pred = model(volume)
        pred = torch.sigmoid(pred).cpu().numpy()[0, 0]
    
    # Resize back if needed
    if H != 240 or W != 240:
        pred = torch.nn.functional.interpolate(
            torch.from_numpy(pred).unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).numpy()[0, 0]
    
    return pred


def find_best_slices(gt_volume, num_slices=5):
    """Find slices with good lesion visibility"""
    lesion_counts = []
    for i in range(gt_volume.shape[2]):
        lesion_area = np.sum(gt_volume[:, :, i] > 0)
        lesion_counts.append((i, lesion_area))
    
    # Sort by lesion area
    lesion_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Get top slices with lesions
    best_slices = [idx for idx, count in lesion_counts if count > 100][:num_slices]
    best_slices.sort()
    
    return best_slices


def create_single_patient_figure(patient_id, model, device, num_slices=3):
    """Create figure for a single patient showing multiple slices"""
    print(f"\nProcessing {patient_id}...")
    
    # Load data
    flair, gt = load_patient_data(patient_id)
    flair_norm = normalize_flair(flair)
    
    # Find best slices
    best_slices = find_best_slices(gt, num_slices)
    
    if len(best_slices) == 0:
        print(f"  ⚠️ No slices with lesions found for {patient_id}")
        return
    
    # Create figure
    fig, axes = plt.subplots(num_slices, 4, figsize=(16, 4*num_slices))
    if num_slices == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, slice_idx in enumerate(best_slices):
        print(f"  Processing slice {slice_idx}...")
        
        # Get slices
        flair_slice = flair_norm[:, :, slice_idx]
        gt_slice = gt[:, :, slice_idx]
        
        # Generate prediction
        pred_slice = predict_slice(model, device, flair_norm, slice_idx)
        pred_binary = (pred_slice > 0.5).astype(np.float32)
        
        # Calculate metrics
        intersection = np.sum(pred_binary * gt_slice)
        union = np.sum(pred_binary) + np.sum(gt_slice)
        dice = (2 * intersection / union) if union > 0 else 0
        
        # Plot FLAIR
        axes[row_idx, 0].imshow(flair_slice.T, cmap='gray', origin='lower')
        axes[row_idx, 0].set_title(f'FLAIR (Slice {slice_idx})', fontsize=12, fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        # Plot Ground Truth
        axes[row_idx, 1].imshow(flair_slice.T, cmap='gray', origin='lower')
        axes[row_idx, 1].imshow(gt_slice.T, cmap='Reds', alpha=0.5 * (gt_slice.T > 0), origin='lower')
        axes[row_idx, 1].set_title('Ground Truth Lesions', fontsize=12, fontweight='bold')
        axes[row_idx, 1].axis('off')
        
        # Plot Prediction
        axes[row_idx, 2].imshow(flair_slice.T, cmap='gray', origin='lower')
        axes[row_idx, 2].imshow(pred_binary.T, cmap='Blues', alpha=0.5 * (pred_binary.T > 0), origin='lower')
        axes[row_idx, 2].set_title(f'Model Prediction\nDice: {dice:.1%}', fontsize=12, fontweight='bold')
        axes[row_idx, 2].axis('off')
        
        # Plot Overlay Comparison
        axes[row_idx, 3].imshow(flair_slice.T, cmap='gray', origin='lower')
        axes[row_idx, 3].imshow(gt_slice.T, cmap='Reds', alpha=0.3 * (gt_slice.T > 0), origin='lower', label='Ground Truth')
        axes[row_idx, 3].imshow(pred_binary.T, cmap='Blues', alpha=0.3 * (pred_binary.T > 0), origin='lower', label='Prediction')
        axes[row_idx, 3].set_title('Overlay Comparison', fontsize=12, fontweight='bold')
        axes[row_idx, 3].axis('off')
        
        # Add legend only on first row
        if row_idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.5, label='Ground Truth'),
                Patch(facecolor='blue', alpha=0.5, label='Prediction')
            ]
            axes[row_idx, 3].legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.suptitle(f'PediMS Patient {patient_id} - Model Predictions', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / f'patient_{patient_id}_examples.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def create_multi_patient_overview(model, device, num_patients=6):
    """Create overview figure with multiple patients (one slice each)"""
    print(f"\nCreating multi-patient overview...")
    
    # Select diverse patients from training set
    selected_patients = TRAINING_PATIENTS[::len(TRAINING_PATIENTS)//num_patients][:num_patients]
    
    fig, axes = plt.subplots(num_patients, 4, figsize=(16, 4*num_patients))
    
    for row_idx, patient_id in enumerate(selected_patients):
        print(f"  Processing {patient_id}...")
        
        # Load data
        flair, gt = load_patient_data(patient_id)
        flair_norm = normalize_flair(flair)
        
        # Find best slice
        best_slices = find_best_slices(gt, 1)
        if len(best_slices) == 0:
            continue
        
        slice_idx = best_slices[0]
        
        # Get slices
        flair_slice = flair_norm[:, :, slice_idx]
        gt_slice = gt[:, :, slice_idx]
        
        # Generate prediction
        pred_slice = predict_slice(model, device, flair_norm, slice_idx)
        pred_binary = (pred_slice > 0.5).astype(np.float32)
        
        # Calculate Dice
        intersection = np.sum(pred_binary * gt_slice)
        union = np.sum(pred_binary) + np.sum(gt_slice)
        dice = (2 * intersection / union) if union > 0 else 0
        
        # Plot
        axes[row_idx, 0].imshow(flair_slice.T, cmap='gray', origin='lower')
        axes[row_idx, 0].set_ylabel(patient_id, fontsize=12, fontweight='bold', rotation=0, labelpad=40, va='center')
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(flair_slice.T, cmap='gray', origin='lower')
        axes[row_idx, 1].imshow(gt_slice.T, cmap='Reds', alpha=0.5 * (gt_slice.T > 0), origin='lower')
        axes[row_idx, 1].axis('off')
        
        axes[row_idx, 2].imshow(flair_slice.T, cmap='gray', origin='lower')
        axes[row_idx, 2].imshow(pred_binary.T, cmap='Blues', alpha=0.5 * (pred_binary.T > 0), origin='lower')
        axes[row_idx, 2].axis('off')
        
        axes[row_idx, 3].imshow(flair_slice.T, cmap='gray', origin='lower')
        axes[row_idx, 3].imshow(gt_slice.T, cmap='Reds', alpha=0.3 * (gt_slice.T > 0), origin='lower')
        axes[row_idx, 3].imshow(pred_binary.T, cmap='Blues', alpha=0.3 * (pred_binary.T > 0), origin='lower')
        axes[row_idx, 3].text(0.95, 0.05, f'Dice: {dice:.1%}', 
                             transform=axes[row_idx, 3].transAxes,
                             fontsize=10, fontweight='bold', 
                             ha='right', va='bottom',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[row_idx, 3].axis('off')
    
    # Column titles
    axes[0, 0].set_title('FLAIR Image', fontsize=14, fontweight='bold', pad=10)
    axes[0, 1].set_title('Ground Truth', fontsize=14, fontweight='bold', pad=10)
    axes[0, 2].set_title('Prediction', fontsize=14, fontweight='bold', pad=10)
    axes[0, 3].set_title('Overlay', fontsize=14, fontweight='bold', pad=10)
    
    plt.suptitle('PediMS Training Dataset - Model Performance Examples', 
                fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / 'multi_patient_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def main():
    print("="*60)
    print("GENERATING DATASET VISUALIZATION EXAMPLES")
    print("="*60)
    
    # Load model
    model, device = load_model()
    
    # Create multi-patient overview (main figure)
    create_multi_patient_overview(model, device, num_patients=6)
    
    # Create detailed examples for 3 selected patients
    print("\nGenerating detailed patient examples...")
    selected_patients = ['P03', 'P14', 'P28']  # Diverse examples
    
    for patient_id in selected_patients:
        if patient_id in TRAINING_PATIENTS:
            try:
                create_single_patient_figure(patient_id, model, device, num_slices=3)
            except Exception as e:
                print(f"  ✗ Error processing {patient_id}: {e}")
    
    print("\n" + "="*60)
    print("✓ ALL DATASET EXAMPLES GENERATED!")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nFiles created:")
    print("  • multi_patient_overview.png - Main figure (6 patients)")
    print("  • patient_*_examples.png - Detailed examples (3 slices each)")
    print("\nReady for your research paper! 🎉")
    print("="*60)


if __name__ == "__main__":
    main()
