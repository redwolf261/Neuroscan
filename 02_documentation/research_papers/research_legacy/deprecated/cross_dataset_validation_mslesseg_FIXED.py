"""
Cross-Dataset Validation on MSLESSEG Dataset - FIXED VERSION
=============================================================
CRITICAL FIXES:
1. Use TRUE 3D volumes (64×64×64) not fake 2.5D (1×5×64×64)
2. Use MONAI normalization (same as training)
3. Proper volume extraction and preprocessing

This matches the actual training pipeline!
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
from tqdm import tqdm
import csv
import warnings
warnings.filterwarnings('ignore')

# MONAI transforms (SAME AS TRAINING!)
from monai.transforms import (
    Compose, EnsureChannelFirst, NormalizeIntensity, Resize, EnsureType
)
import monai.transforms as mt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from final_model import HybridMiniSwin2D5_CSRF


class MSLESSEGDataset3D(Dataset):
    """
    MSLESSEG Dataset with PROPER 3D volumes
    Matches training pipeline exactly!
    """
    
    def __init__(self, root_dir, target_size=(64, 64, 64)):
        """
        Args:
            root_dir: Path to MSLESSEG processed folder
            target_size: 3D volume size (MUST match training!)
        """
        self.root_dir = root_dir
        self.target_size = target_size
        
        # Get FLAIR images and labels
        self.images_dir = os.path.join(root_dir, 'imagesTr')
        self.labels_dir = os.path.join(root_dir, 'labelsTr')
        
        # Get all FLAIR cases
        flair_files = sorted([f for f in os.listdir(self.images_dir) 
                             if f.endswith('_flair.nii.gz')])
        
        # Build sample list - extract 3D sub-volumes
        self.samples = []
        print(f"Loading MSLESSEG dataset from: {root_dir}")
        print(f"Found {len(flair_files)} FLAIR volumes")
        print(f"Extracting 3D sub-volumes of size {target_size}...")
        
        for flair_file in tqdm(flair_files, desc="Processing volumes"):
            case_id = flair_file.replace('_flair.nii.gz', '')
            
            flair_path = os.path.join(self.images_dir, flair_file)
            label_path = os.path.join(self.labels_dir, f"{case_id}_mask.nii.gz")
            
            if not os.path.exists(label_path):
                print(f"Warning: No label found for {case_id}")
                continue
            
            # Load volumes
            flair_img = nib.load(flair_path)
            label_img = nib.load(label_path)
            
            flair_data = flair_img.get_fdata()
            label_data = label_img.get_fdata()
            
            H, W, D = flair_data.shape
            
            # Extract overlapping 3D patches (stride = target_size // 2)
            stride_z = max(1, target_size[2] // 2)
            
            for z_start in range(0, D - target_size[2] + 1, stride_z):
                z_end = z_start + target_size[2]
                
                # Check if this volume has lesions
                label_volume = label_data[:, :, z_start:z_end]
                lesion_pixels = np.sum(label_volume > 0)
                
                # Only include volumes with lesions
                if lesion_pixels > 50:  # At least 50 lesion pixels in 3D volume
                    self.samples.append({
                        'flair_path': flair_path,
                        'label_path': label_path,
                        'z_start': z_start,
                        'z_end': z_end,
                        'case_id': case_id,
                        'lesion_pixels': int(lesion_pixels)
                    })
        
        # MONAI transforms (EXACTLY SAME AS TRAINING!)
        self.transform = Compose([
            EnsureChannelFirst(channel_dim='no_channel'),  # Add channel dimension
            NormalizeIntensity(nonzero=True, channel_wise=True),  # SAME AS TRAINING!
            Resize(spatial_size=target_size, mode='trilinear'),
            EnsureType(dtype=torch.float32)
        ])
        
        print(f"\nDataset Summary:")
        print(f"  Total cases: {len(flair_files)}")
        print(f"  3D volumes with lesions: {len(self.samples)}")
        print(f"  Volume size: {target_size}")
        print(f"  Using MONAI normalization (matches training!)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (1, D, H, W) tensor - TRUE 3D!
            mask: (D, H, W) tensor
            info: dict with metadata
        """
        sample = self.samples[idx]
        
        # Load 3D volumes
        flair_img = nib.load(sample['flair_path'])
        label_img = nib.load(sample['label_path'])
        
        flair_data = flair_img.get_fdata()
        label_data = label_img.get_fdata()
        
        # Extract 3D sub-volume
        z_start = sample['z_start']
        z_end = sample['z_end']
        
        flair_volume = flair_data[:, :, z_start:z_end]
        label_volume = label_data[:, :, z_start:z_end]
        
        # Convert to tensors and apply MONAI transforms
        flair_volume = flair_volume.astype(np.float32)
        label_volume = label_volume.astype(np.float32)
        
        # Apply transforms (adds channel, normalizes, resizes)
        image_tensor = self.transform(flair_volume)  # -> (1, 64, 64, 64)
        
        # Process label separately
        label_tensor = torch.from_numpy(label_volume).float()
        label_tensor = label_tensor.unsqueeze(0)  # Add channel
        label_resized = torch.nn.functional.interpolate(
            label_tensor.unsqueeze(0),  # (1, 1, H, W, D)
            size=self.target_size,
            mode='nearest'
        ).squeeze(0).squeeze(0)  # Remove batch and channel
        
        # Binarize
        label_resized = (label_resized > 0).float()
        
        # Metadata
        info = {
            'case_id': sample['case_id'],
            'z_range': f"{z_start}-{z_end}",
            'lesion_pixels': sample['lesion_pixels']
        }
        
        return image_tensor, label_resized, info


def dice_score(pred, target, smooth=1e-5):
    """Calculate Dice coefficient"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()


def precision_score(pred, target, smooth=1e-5):
    """Calculate Precision"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    
    return (tp + smooth) / (tp + fp + smooth)


def recall_score(pred, target, smooth=1e-5):
    """Calculate Recall"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    
    return (tp + smooth) / (tp + fn + smooth)


def specificity_score(pred, target, smooth=1e-5):
    """Calculate Specificity"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    
    return (tn + smooth) / (tn + fp + smooth)


def iou_score(pred, target, smooth=1e-5):
    """Calculate IoU"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)


def validate():
    """Run validation on MSLESSEG dataset"""
    
    # Configuration
    MODEL_PATH = r"C:\Users\HP\EDI\.resume_checkpoints\seg_resume.pth"
    DATASET_ROOT = r"G:\My Drive\Dataset\MSLESSEG\processed"
    OUTPUT_DIR = r"C:\Users\HP\EDI\csv_data\cross_dataset_validation_mslesseg_FIXED"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if DEVICE == 'cpu':
        print("ERROR: GPU not available. This will be too slow.")
        print("Please ensure CUDA is available before running.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("CROSS-DATASET VALIDATION: MSLESSEG MS Lesion Dataset (FIXED)")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*80)
    print("\nCRITICAL FIXES APPLIED:")
    print("  ✓ Using TRUE 3D volumes (64×64×64) not fake 2.5D")
    print("  ✓ Using MONAI normalization (matches training)")
    print("  ✓ Proper volume extraction and preprocessing")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model = HybridMiniSwin2D5_CSRF(
        k_slices=5,
        channels=[32, 64, 128, 256, 512]
    ).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded checkpoint from epoch {epoch}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded! Parameters: {total_params:,}")
    
    model.eval()
    
    # Load dataset
    dataset = MSLESSEGDataset3D(
        root_dir=DATASET_ROOT,
        target_size=(64, 64, 64)  # SAME AS TRAINING!
    )
    
    if len(dataset) == 0:
        print("ERROR: No valid samples found in dataset!")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    # Validation
    print("\n" + "="*80)
    print("RUNNING VALIDATION")
    print("="*80)
    print("NOTE: This is ZERO-SHOT evaluation (NO retraining on MSLESSEG)")
    print("Training: PediMS (Pediatric MS)")
    print("Validation: MSLESSEG (Adult MS - SAME PATHOLOGY!)")
    print("="*80)
    
    # Progressive CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(OUTPUT_DIR, f'progressive_validation_FIXED_{timestamp}.csv')
    
    csv_columns = [
        'sample_idx', 'case_id', 'z_range', 'lesion_pixels',
        'dice', 'precision', 'recall', 'specificity', 'iou',
        'running_avg_dice', 'running_avg_precision', 'running_avg_recall',
        'running_avg_specificity', 'running_avg_iou',
        'running_std_dice', 'running_std_precision', 'running_std_recall',
        'running_std_specificity', 'running_std_iou',
        'samples_processed', 'progress_percent'
    ]
    
    # Metrics storage
    all_dice = []
    all_precision = []
    all_recall = []
    all_specificity = []
    all_iou = []
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        
        with torch.no_grad():
            for idx, (images, masks, infos) in enumerate(tqdm(dataloader, desc="Validating")):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                # Forward pass
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                
                # Calculate metrics
                dice = dice_score(preds, masks)
                precision = precision_score(preds, masks).item()
                recall = recall_score(preds, masks).item()
                specificity = specificity_score(preds, masks).item()
                iou = iou_score(preds, masks).item()
                
                # Store
                all_dice.append(dice)
                all_precision.append(precision)
                all_recall.append(recall)
                all_specificity.append(specificity)
                all_iou.append(iou)
                
                # Write to CSV
                row = {
                    'sample_idx': idx,
                    'case_id': infos['case_id'][0],
                    'z_range': infos['z_range'][0],
                    'lesion_pixels': infos['lesion_pixels'][0].item(),
                    'dice': f"{dice:.6f}",
                    'precision': f"{precision:.6f}",
                    'recall': f"{recall:.6f}",
                    'specificity': f"{specificity:.6f}",
                    'iou': f"{iou:.6f}",
                    'running_avg_dice': f"{np.mean(all_dice):.6f}",
                    'running_avg_precision': f"{np.mean(all_precision):.6f}",
                    'running_avg_recall': f"{np.mean(all_recall):.6f}",
                    'running_avg_specificity': f"{np.mean(all_specificity):.6f}",
                    'running_avg_iou': f"{np.mean(all_iou):.6f}",
                    'running_std_dice': f"{np.std(all_dice):.6f}",
                    'running_std_precision': f"{np.std(all_precision):.6f}",
                    'running_std_recall': f"{np.std(all_recall):.6f}",
                    'running_std_specificity': f"{np.std(all_specificity):.6f}",
                    'running_std_iou': f"{np.std(all_iou):.6f}",
                    'samples_processed': idx + 1,
                    'progress_percent': f"{((idx + 1) / len(dataloader) * 100):.2f}"
                }
                writer.writerow(row)
                
                # Periodic updates
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(dataloader)} | Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    
    # Final results
    print("\n" + "="*80)
    print("MSLESSEG VALIDATION RESULTS (FIXED)")
    print("="*80)
    print(f"Dice Score:   {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"Precision:    {np.mean(all_precision):.4f} ± {np.std(all_precision):.4f}")
    print(f"Recall:       {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")
    print(f"Specificity:  {np.mean(all_specificity):.4f} ± {np.std(all_specificity):.4f}")
    print(f"IoU:          {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
    print("="*80)
    
    # Save summary
    summary = {
        'dataset': 'MSLESSEG (MS Lesions)',
        'model_path': MODEL_PATH,
        'epoch': str(epoch),
        'total_samples': len(dataloader),
        'metrics': {
            'dice': {'mean': float(np.mean(all_dice)), 'std': float(np.std(all_dice))},
            'precision': {'mean': float(np.mean(all_precision)), 'std': float(np.std(all_precision))},
            'recall': {'mean': float(np.mean(all_recall)), 'std': float(np.std(all_recall))},
            'specificity': {'mean': float(np.mean(all_specificity)), 'std': float(np.std(all_specificity))},
            'iou': {'mean': float(np.mean(all_iou)), 'std': float(np.std(all_iou))}
        },
        'fixes_applied': [
            'TRUE 3D volumes (64×64×64) not fake 2.5D',
            'MONAI normalization (matches training)',
            'Proper volume extraction'
        ]
    }
    
    json_path = os.path.join(OUTPUT_DIR, f'validation_summary_FIXED_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Progressive CSV: {csv_path}")
    print("="*80)


if __name__ == "__main__":
    validate()
