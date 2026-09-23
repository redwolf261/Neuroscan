"""
Generate Dataset Visualization Examples (Dataset Only - No Model Predictions)
==============================================================================
Creates publication-quality figures showing PediMS training dataset:
- FLAIR images
- Ground truth lesion masks
- Side-by-side comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path

# Paths
DATASET_PATH = Path(r"G:\My Drive\Dataset\PediMS\PediMS")
OUTPUT_DIR = Path(r"C:\Users\HP\EDI\paper_figures\dataset_examples")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PediMS patients (9 patients total - without leading zeros)
TRAINING_PATIENTS = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']


def load_patient_data(patient_id):
    """Load FLAIR and ground truth for a patient"""
    patient_dir = DATASET_PATH / patient_id / "T2" / "processed"
    
    # Find FLAIR file (n4_brain_FLAIR.nii.gz)
    flair_files = list(patient_dir.glob("*_n4_brain_FLAIR.nii.gz"))
    if not flair_files:
        flair_files = list(patient_dir.glob("*_brain_FLAIR.nii.gz"))
    if not flair_files:
        raise FileNotFoundError(f"No FLAIR file found for {patient_id}")
    flair_path = flair_files[0]
    flair_nii = nib.load(str(flair_path))
    flair = flair_nii.get_fdata()
    
    # Find ground truth (Consensus.nii)
    gt_files = list(patient_dir.glob("*_Consensus.nii"))
    if not gt_files:
        raise FileNotFoundError(f"No Consensus file found for {patient_id}")
    gt_path = gt_files[0]
    gt_nii = nib.load(str(gt_path))
    gt = gt_nii.get_fdata()
    
    return flair, gt


def normalize_flair(flair_volume):
    """Normalize FLAIR volume for visualization"""
    p1, p99 = np.percentile(flair_volume[flair_volume > 0], [1, 99])
    flair_norm = np.clip(flair_volume, p1, p99)
    flair_norm = (flair_norm - p1) / (p99 - p1)
    return flair_norm


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


def create_single_patient_figure(patient_id, num_slices=3):
    """Create figure for a single patient showing multiple slices"""
    print(f"Processing {patient_id}...")
    
    try:
        # Load data
        flair, gt = load_patient_data(patient_id)
        flair_norm = normalize_flair(flair)
        
        # Find best slices
        best_slices = find_best_slices(gt, num_slices)
        
        if len(best_slices) == 0:
            print(f"  ⚠️ No slices with lesions found")
            return False
        
        # Create figure
        fig, axes = plt.subplots(num_slices, 3, figsize=(12, 4*num_slices))
        if num_slices == 1:
            axes = axes.reshape(1, -1)
        
        for row_idx, slice_idx in enumerate(best_slices):
            # Get slices
            flair_slice = flair_norm[:, :, slice_idx]
            gt_slice = gt[:, :, slice_idx]
            
            # Calculate lesion stats
            lesion_pixels = np.sum(gt_slice > 0)
            total_pixels = gt_slice.size
            lesion_percent = (lesion_pixels / total_pixels) * 100
            
            # Plot FLAIR
            axes[row_idx, 0].imshow(flair_slice.T, cmap='gray', origin='lower')
            axes[row_idx, 0].set_title(f'FLAIR (Slice {slice_idx})', fontsize=12, fontweight='bold')
            axes[row_idx, 0].axis('off')
            
            # Plot Ground Truth mask only
            axes[row_idx, 1].imshow(gt_slice.T, cmap='Reds', origin='lower')
            axes[row_idx, 1].set_title(f'Lesion Mask\n{lesion_pixels} pixels ({lesion_percent:.2f}%)', 
                                      fontsize=12, fontweight='bold')
            axes[row_idx, 1].axis('off')
            
            # Plot Overlay
            axes[row_idx, 2].imshow(flair_slice.T, cmap='gray', origin='lower')
            axes[row_idx, 2].imshow(gt_slice.T, cmap='Reds', alpha=0.5 * (gt_slice.T > 0), origin='lower')
            axes[row_idx, 2].set_title('FLAIR + Lesion Overlay', fontsize=12, fontweight='bold')
            axes[row_idx, 2].axis('off')
        
        plt.suptitle(f'PediMS Training Dataset - Patient {patient_id}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save
        output_path = OUTPUT_DIR / f'patient_{patient_id}_dataset.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Saved: {output_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def create_multi_patient_overview(num_patients=6):
    """Create overview figure with multiple patients"""
    print(f"\nCreating multi-patient overview ({num_patients} patients)...")
    
    # Select diverse patients
    selected_patients = TRAINING_PATIENTS[::len(TRAINING_PATIENTS)//num_patients][:num_patients]
    
    fig, axes = plt.subplots(num_patients, 3, figsize=(12, 4*num_patients))
    
    successful_patients = []
    for row_idx, patient_id in enumerate(selected_patients):
        print(f"  Processing {patient_id}...")
        
        try:
            # Load data
            flair, gt = load_patient_data(patient_id)
            flair_norm = normalize_flair(flair)
            
            # Find best slice
            best_slices = find_best_slices(gt, 1)
            if len(best_slices) == 0:
                print(f"    ⚠️ No lesions found, skipping")
                continue
            
            slice_idx = best_slices[0]
            
            # Get slices
            flair_slice = flair_norm[:, :, slice_idx]
            gt_slice = gt[:, :, slice_idx]
            
            # Calculate stats
            lesion_pixels = np.sum(gt_slice > 0)
            lesion_percent = (lesion_pixels / gt_slice.size) * 100
            
            # Plot FLAIR
            axes[row_idx, 0].imshow(flair_slice.T, cmap='gray', origin='lower')
            axes[row_idx, 0].set_ylabel(patient_id, fontsize=12, fontweight='bold', 
                                       rotation=0, labelpad=40, va='center')
            axes[row_idx, 0].axis('off')
            
            # Plot Lesion Mask
            axes[row_idx, 1].imshow(gt_slice.T, cmap='Reds', origin='lower')
            axes[row_idx, 1].axis('off')
            
            # Plot Overlay
            axes[row_idx, 2].imshow(flair_slice.T, cmap='gray', origin='lower')
            axes[row_idx, 2].imshow(gt_slice.T, cmap='Reds', alpha=0.5 * (gt_slice.T > 0), origin='lower')
            axes[row_idx, 2].text(0.95, 0.05, f'{lesion_percent:.2f}%', 
                                 transform=axes[row_idx, 2].transAxes,
                                 fontsize=10, fontweight='bold',
                                 ha='right', va='bottom',
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            axes[row_idx, 2].axis('off')
            
            successful_patients.append(patient_id)
            print(f"    ✓ Done")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            # Hide axes for failed patients
            for ax in axes[row_idx]:
                ax.axis('off')
    
    # Column titles
    axes[0, 0].set_title('FLAIR Image', fontsize=14, fontweight='bold', pad=10)
    axes[0, 1].set_title('Lesion Mask', fontsize=14, fontweight='bold', pad=10)
    axes[0, 2].set_title('Overlay', fontsize=14, fontweight='bold', pad=10)
    
    plt.suptitle(f'PediMS Training Dataset Examples ({len(successful_patients)} patients)', 
                fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / 'pedims_dataset_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")
    return successful_patients


def main():
    print("="*60)
    print("GENERATING PEDIMS DATASET VISUALIZATION EXAMPLES")
    print("="*60)
    print(f"\nDataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Create multi-patient overview (main figure for paper)
    successful_patients = create_multi_patient_overview(num_patients=6)
    
    # Create detailed examples for 3 selected patients
    print("\nGenerating detailed patient examples...")
    selected_patients = ['P3', 'P5', 'P8']
    
    detail_count = 0
    for patient_id in selected_patients:
        if patient_id in TRAINING_PATIENTS:
            if create_single_patient_figure(patient_id, num_slices=3):
                detail_count += 1
    
    print("\n" + "="*60)
    print("✓ DATASET EXAMPLES GENERATED!")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nFiles created:")
    print(f"  • pedims_dataset_overview.png - Main figure ({len(successful_patients)} patients)")
    print(f"  • patient_*_dataset.png - Detailed examples ({detail_count} patients, 3 slices each)")
    print("\nThese show:")
    print("  ✓ FLAIR images from PediMS training set")
    print("  ✓ Manual lesion annotations (ground truth)")
    print("  ✓ Lesion overlays on FLAIR")
    print("\nPerfect for your research paper! 🎉")
    print("="*60)


if __name__ == "__main__":
    main()
