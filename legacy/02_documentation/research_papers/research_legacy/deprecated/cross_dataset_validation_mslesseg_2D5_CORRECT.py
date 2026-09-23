"""
CORRECT 2.5D Cross-Dataset Validation for MSLESSEG
===================================================
The model is 2.5D (predicts central slice), not 3D!

Proper evaluation:
1. Extract k neighboring slices around each target slice
2. Model predicts central slice
3. Compare prediction against ground truth central slice
4. Aggregate metrics across all slices

This is the CORRECT way to evaluate a 2.5D model!
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
from tqdm import tqdm
import csv
import warnings
warnings.filterwarnings('ignore')

# MONAI imports
from monai.transforms import (
    Compose, NormalizeIntensity, Resize, EnsureType
)
import torch.nn.functional as F

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from final_model import HybridMiniSwin2D5_CSRF


class MSLESSEG_2D5_Dataset(Dataset):
    """
    2.5D Dataset for slice-by-slice evaluation
    Each sample is k neighboring slices predicting the central slice
    """
    
    def __init__(self, root_dir, k_slices=5, target_size=(64, 64)):
        self.root_dir = root_dir
        self.k_slices = k_slices
        self.target_size = target_size
        self.half_k = k_slices // 2
        
        # Paths
        self.images_dir = os.path.join(root_dir, 'imagesTr')
        self.labels_dir = os.path.join(root_dir, 'labelsTr')
        
        # Build samples list
        self.samples = []
        
        print(f"Loading MSLESSEG dataset (2.5D slice-by-slice)...")
        print(f"Dataset root: {root_dir}")
        print(f"K-slices: {k_slices}, Target size: {target_size}")
        
        flair_files = sorted([f for f in os.listdir(self.images_dir) 
                             if f.endswith('_flair.nii.gz')])
        
        for flair_file in tqdm(flair_files, desc="Processing cases"):
            case_id = flair_file.replace('_flair.nii.gz', '')
            
            flair_path = os.path.join(self.images_dir, flair_file)
            label_path = os.path.join(self.labels_dir, f"{case_id}_mask.nii.gz")
            
            if not os.path.exists(label_path):
                continue
            
            # Load volumes
            flair_img = nib.load(flair_path)
            label_img = nib.load(label_path)
            
            flair_data = flair_img.get_fdata()
            label_data = label_img.get_fdata()
            
            H, W, D = flair_data.shape
            
            # Create samples for each slice with lesions
            for slice_idx in range(self.half_k, D - self.half_k):
                # Check if central slice has lesions
                label_slice = label_data[:, :, slice_idx]
                if np.sum(label_slice > 0) > 10:  # At least 10 lesion pixels
                    self.samples.append({
                        'case_id': case_id,
                        'flair_path': flair_path,
                        'label_path': label_path,
                        'slice_idx': slice_idx,
                        'num_slices': D
                    })
        
        print(f"\nDataset loaded:")
        print(f"  Cases: {len(flair_files)}")
        print(f"  Slices with lesions: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def extract_k_slices(self, volume_3d, slice_idx):
        """Extract k neighboring slices around central slice"""
        slices = []
        for offset in range(-self.half_k, self.half_k + 1):
            idx = slice_idx + offset
            idx = max(0, min(volume_3d.shape[2] - 1, idx))  # Clamp
            slices.append(volume_3d[:, :, idx])
        
        return np.stack(slices, axis=0)  # (k, H, W)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load volumes
        flair_data = nib.load(sample['flair_path']).get_fdata()
        label_data = nib.load(sample['label_path']).get_fdata()
        
        slice_idx = sample['slice_idx']
        
        # Extract k neighboring slices
        flair_k_slices = self.extract_k_slices(flair_data, slice_idx)  # (k, H, W)
        label_central = label_data[:, :, slice_idx]  # (H, W)
        
        # Normalize using MONAI-style normalization
        flair_k_slices_normalized = []
        for i in range(self.k_slices):
            slice_i = flair_k_slices[i].astype(np.float32)
            # Nonzero normalization
            nonzero_mask = slice_i > 0
            if nonzero_mask.sum() > 0:
                mean = slice_i[nonzero_mask].mean()
                std = slice_i[nonzero_mask].std()
                if std > 0:
                    slice_i = (slice_i - mean) / std
            flair_k_slices_normalized.append(slice_i)
        
        flair_k_slices_normalized = np.stack(flair_k_slices_normalized, axis=0)
        
        # Resize slices
        import torch.nn.functional as F
        flair_tensor = torch.from_numpy(flair_k_slices_normalized).float()  # (k, H, W)
        flair_resized = F.interpolate(
            flair_tensor.unsqueeze(0),  # (1, k, H, W)
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # (k, H, W)
        
        # Resize label
        label_tensor = torch.from_numpy(label_central).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        label_resized = F.interpolate(
            label_tensor,
            size=self.target_size,
            mode='nearest'
        ).squeeze()  # (H, W)
        
        # Binarize
        label_binary = (label_resized > 0).float()
        
        # Format as 3D volume (B, 1, D, H, W) where D=k
        # Stack slices along depth dimension
        volume = flair_resized.unsqueeze(0).permute(0, 2, 1, 3)  # (1, k, H, W) -> (1, H, k, W) -> (1, k, H, W)
        volume = flair_resized.unsqueeze(0).unsqueeze(1)  # (1, 1, k, H, W)
        # Actually, model expects (B, 1, D, H, W) where D is depth
        volume = flair_resized.unsqueeze(0)  # (1, k, H, W)
        volume = torch.cat([volume[i:i+1] for i in range(self.k_slices)], dim=0)  # (k, 1, H, W)
        volume = volume.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 1, k, H, W)
        
        # Simpler: just format correctly
        # Model expects: (B, 1, D, H, W)
        # We have: flair_resized = (k, H, W)
        # Convert to: (1, k, H, W) then add batch dim
        volume = flair_resized.unsqueeze(1)  # (k, 1, H, W)
        volume = volume.permute(1, 0, 2, 3)  # (1, k, H, W)
        
        # Wait, let me check what model actually expects...
        # From stem: x: (B, 1, D, H, W) - 3D volume
        # So we need shape: (1, D, H, W) where D=k
        volume = flair_resized  # (k, H, W) - this is actually correct!
        # Model input should be (B=1, C=1, D=k, H=64, W=64)
        # But we have (k, H, W)
        # Need to unsqueeze: (1, 1, k, H, W) NO
        # Actually from code: "Extract k consecutive slices from center"
        # Model creates volume internally from (B, 1, D, H, W)
        
        # Let's just create fake 3D volume with k slices
        fake_volume = torch.zeros(1, 1, self.k_slices, self.target_size[0], self.target_size[1])
        for i in range(self.k_slices):
            fake_volume[0, 0, i, :, :] = flair_resized[i]
        
        info = {
            'case_id': sample['case_id'],
            'slice_idx': slice_idx,
            'lesion_pixels': int(torch.sum(label_binary).item())
        }
        
        return fake_volume.squeeze(0), label_binary, info  # (1, k, H, W), (H, W), dict


def dice_score(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    return (2. * intersection + smooth) / (union + smooth)


def precision_score(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    
    return (tp + smooth) / (tp + fp + smooth)


def recall_score(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    
    return (tp + smooth) / (tp + fn + smooth)


def specificity_score(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    
    return (tn + smooth) / (tn + fp + smooth)


def iou_score(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)


def validate():
    """Run 2.5D validation on MSLESSEG"""
    
    MODEL_PATH = r"C:\Users\HP\EDI\.resume_checkpoints\seg_resume.pth"
    DATASET_ROOT = r"G:\My Drive\Dataset\MSLESSEG\processed"
    OUTPUT_DIR = r"C:\Users\HP\EDI\csv_data\cross_dataset_validation_mslesseg_2D5_CORRECT"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if DEVICE == 'cpu':
        print("ERROR: GPU required")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("CORRECT 2.5D CROSS-DATASET VALIDATION: MSLESSEG")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*80)
    print("\nEVALUATION APPROACH:")
    print("  ✓ Model is 2.5D (predicts central slice from k neighbors)")
    print("  ✓ Slice-by-slice evaluation (NOT volume-by-volume)")
    print("  ✓ Compare 2D prediction vs 2D ground truth")
    print("  ✓ Aggregate metrics across all slices")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model = HybridMiniSwin2D5_CSRF(
        k_slices=5,
        channels=[32, 64, 128, 256, 512]
    ).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch')}")
    print(f"Training best dice: {checkpoint.get('best_val_dice'):.4f}")
    
    model.eval()
    
    # Create dataset
    dataset = MSLESSEG_2D5_Dataset(
        root_dir=DATASET_ROOT,
        k_slices=5,
        target_size=(64, 64)
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Validation
    print("\n" + "="*80)
    print("RUNNING VALIDATION")
    print("="*80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(OUTPUT_DIR, f'progressive_validation_2D5_{timestamp}.csv')
    
    csv_columns = [
        'sample_idx', 'case_id', 'slice_idx', 'lesion_pixels',
        'dice', 'precision', 'recall', 'specificity', 'iou',
        'running_avg_dice', 'running_avg_precision', 'running_avg_recall',
        'running_avg_specificity', 'running_avg_iou',
        'samples_processed', 'progress_percent'
    ]
    
    all_dice = []
    all_precision = []
    all_recall = []
    all_specificity = []
    all_iou = []
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        
        with torch.no_grad():
            for idx, (volumes, masks, infos) in enumerate(tqdm(dataloader, desc="Validating")):
                volumes = volumes.to(DEVICE)
                masks = masks.to(DEVICE)
                
                # Forward pass - model outputs 2D prediction
                outputs = model(volumes)  # (B, 1, H, W)
                preds = torch.sigmoid(outputs).squeeze()  # (H, W)
                masks = masks.squeeze()  # (H, W)
                
                # Calculate metrics
                dice = dice_score(preds, masks).item()
                precision = precision_score(preds, masks).item()
                recall = recall_score(preds, masks).item()
                specificity = specificity_score(preds, masks).item()
                iou = iou_score(preds, masks).item()
                
                all_dice.append(dice)
                all_precision.append(precision)
                all_recall.append(recall)
                all_specificity.append(specificity)
                all_iou.append(iou)
                
                # Write to CSV
                row = {
                    'sample_idx': idx,
                    'case_id': infos['case_id'][0],
                    'slice_idx': infos['slice_idx'][0].item(),
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
                    'samples_processed': idx + 1,
                    'progress_percent': f"{((idx + 1) / len(dataloader) * 100):.2f}"
                }
                writer.writerow(row)
                
                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx+1}/{len(dataloader)} | Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    
    # Final results
    print("\n" + "="*80)
    print("MSLESSEG 2.5D VALIDATION RESULTS (CORRECT)")
    print("="*80)
    print(f"Dice Score:   {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"Precision:    {np.mean(all_precision):.4f} ± {np.std(all_precision):.4f}")
    print(f"Recall:       {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")
    print(f"Specificity:  {np.mean(all_specificity):.4f} ± {np.std(all_specificity):.4f}")
    print(f"IoU:          {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
    print("="*80)
    
    # Interpretation
    dice_mean = np.mean(all_dice)
    if dice_mean >= 0.70:
        print(f"\n✓ EXCELLENT cross-dataset performance ({dice_mean:.1%})")
    elif dice_mean >= 0.55:
        print(f"\n✓ GOOD cross-dataset performance ({dice_mean:.1%})")
    elif dice_mean >= 0.40:
        print(f"\n○ MODERATE cross-dataset performance ({dice_mean:.1%})")
    else:
        print(f"\n○ LIMITED cross-dataset performance ({dice_mean:.1%})")
    
    print(f"Performance drop from training: {(0.8399 - dice_mean)*100:.1f}%")
    
    # Save summary
    summary = {
        'dataset': 'MSLESSEG (MS Lesions)',
        'evaluation': '2.5D slice-by-slice (CORRECT)',
        'model_path': MODEL_PATH,
        'total_slices': len(dataloader),
        'metrics': {
            'dice': {'mean': float(np.mean(all_dice)), 'std': float(np.std(all_dice))},
            'precision': {'mean': float(np.mean(all_precision)), 'std': float(np.std(all_precision))},
            'recall': {'mean': float(np.mean(all_recall)), 'std': float(np.std(all_recall))},
            'specificity': {'mean': float(np.mean(all_specificity)), 'std': float(np.std(all_specificity))},
            'iou': {'mean': float(np.mean(all_iou)), 'std': float(np.std(all_iou))}
        }
    }
    
    json_path = os.path.join(OUTPUT_DIR, f'validation_summary_2D5_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    validate()
