"""
Cross-Dataset Validation on MSLESSEG Dataset
============================================
Evaluates the trained HybridMiniSwin2.5D-CSRF model on MSLESSEG dataset
(MS lesion segmentation - SAME PATHOLOGY, different dataset)

This is MORE relevant than LGG (tumors) for MS lesion segmentation validation!
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from final_model import HybridMiniSwin2D5_CSRF


class MSLESSEGDataset(Dataset):
    """
    MSLESSEG MS Lesion Dataset Loader
    Same pathology as PediMS but different patients/scanners
    """
    
    def __init__(self, root_dir, target_size=(64, 64), k_slices=5):
        """
        Args:
            root_dir: Path to MSLESSEG processed folder
            target_size: Resize to this size
            k_slices: Number of slices for 2.5D processing
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.k_slices = k_slices
        
        # Get FLAIR images and labels
        self.images_dir = os.path.join(root_dir, 'imagesTr')
        self.labels_dir = os.path.join(root_dir, 'labelsTr')
        
        # Get all FLAIR cases
        flair_files = sorted([f for f in os.listdir(self.images_dir) 
                             if f.endswith('_flair.nii.gz')])
        
        # Build sample list (extract slices from 3D volumes)
        self.samples = []
        print(f"Loading MSLESSEG dataset from: {root_dir}")
        print(f"Found {len(flair_files)} FLAIR volumes")
        
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
            
            # Get number of slices
            num_slices = flair_data.shape[2]
            
            # Extract slices with lesions (skip empty slices)
            for slice_idx in range(num_slices):
                label_slice = label_data[:, :, slice_idx]
                
                # Only include slices with lesions
                if np.sum(label_slice > 0) > 10:  # At least 10 lesion pixels
                    self.samples.append({
                        'flair_path': flair_path,
                        'label_path': label_path,
                        'slice_idx': slice_idx,
                        'case_id': case_id
                    })
        
        print(f"\nDataset Summary:")
        print(f"  Total cases: {len(flair_files)}")
        print(f"  Slices with lesions: {len(self.samples)}")
        print(f"  Target size: {target_size}")
        print(f"  K-slices for 2.5D: {k_slices}")
    
    def __len__(self):
        return len(self.samples)
    
    def normalize_image(self, img):
        """Normalize image to [0, 1] range"""
        img = img.astype(np.float32)
        # Z-score normalization (better for MRI)
        mean = np.mean(img[img > 0])  # Exclude background
        std = np.std(img[img > 0])
        if std > 0:
            img = (img - mean) / std
            # Clip to reasonable range
            img = np.clip(img, -5, 5)
            # Scale to [0, 1]
            img = (img + 5) / 10
        return img
    
    def extract_2d5_volume(self, volume_3d, slice_idx, k_slices):
        """Extract k neighboring slices around central slice"""
        num_slices = volume_3d.shape[2]
        half_k = k_slices // 2
        
        # Calculate slice indices
        start_idx = max(0, slice_idx - half_k)
        end_idx = min(num_slices, slice_idx + half_k + 1)
        
        # Extract slices
        slices_2d5 = []
        for i in range(start_idx, end_idx):
            slices_2d5.append(volume_3d[:, :, i])
        
        # Pad if necessary
        while len(slices_2d5) < k_slices:
            if slice_idx < half_k:
                slices_2d5.insert(0, volume_3d[:, :, 0])
            else:
                slices_2d5.append(volume_3d[:, :, -1])
        
        return np.stack(slices_2d5, axis=0)  # (k_slices, H, W)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (1, k_slices, H, W) tensor
            mask: (H, W) tensor
            info: dict with metadata
        """
        sample = self.samples[idx]
        
        # Load 3D volumes
        flair_img = nib.load(sample['flair_path'])
        label_img = nib.load(sample['label_path'])
        
        flair_data = flair_img.get_fdata()
        label_data = label_img.get_fdata()
        
        slice_idx = sample['slice_idx']
        
        # Extract 2.5D volume (k neighboring slices)
        flair_2d5 = self.extract_2d5_volume(flair_data, slice_idx, self.k_slices)
        # Shape: (k_slices, H_orig, W_orig)
        
        # Get mask for central slice
        mask_2d = label_data[:, :, slice_idx]
        
        # Resize slices
        import cv2
        flair_resized = np.zeros((self.k_slices, self.target_size[0], self.target_size[1]), dtype=np.float32)
        for k in range(self.k_slices):
            slice_k = flair_2d5[k]
            slice_k = cv2.resize(slice_k, self.target_size, interpolation=cv2.INTER_LINEAR)
            flair_resized[k] = self.normalize_image(slice_k)
        
        # Resize mask
        mask_resized = cv2.resize(mask_2d, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 0).astype(np.float32)
        
        # Format for model: (1, k_slices, H, W)
        image_tensor = torch.from_numpy(flair_resized[np.newaxis, :, :, :]).float()
        mask_tensor = torch.from_numpy(mask_binary).float()
        
        # Metadata
        info = {
            'case_id': sample['case_id'],
            'slice_idx': slice_idx,
            'lesion_pixels': int(np.sum(mask_binary > 0))
        }
        
        return image_tensor, mask_tensor, info


# Import validation functions from LGG script
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
    
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision.item()


def recall_score(pred, target, smooth=1e-5):
    """Calculate Recall"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall.item()


def specificity_score(pred, target, smooth=1e-5):
    """Calculate Specificity"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    
    specificity = (tn + smooth) / (tn + fp + smooth)
    return specificity.item()


def iou_score(pred, target, smooth=1e-5):
    """Calculate IoU"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def validate_mslesseg():
    """Run validation on MSLESSEG dataset"""
    
    print("="*80)
    print("CROSS-DATASET VALIDATION: MSLESSEG MS Lesion Dataset")
    print("="*80)
    
    # Configuration
    MODEL_PATH = r"C:\Users\HP\EDI\.resume_checkpoints\seg_resume.pth"
    DATASET_ROOT = r"G:\My Drive\Dataset\MSLESSEG\processed"
    OUTPUT_DIR = r"C:\Users\HP\EDI\csv_data\cross_dataset_validation_mslesseg"
    
    # Force GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    DEVICE = 'cuda'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    # Load model
    print("Loading model...")
    model = HybridMiniSwin2D5_CSRF(k_slices=5, channels=[32, 64, 128, 256, 512])
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Load dataset
    dataset = MSLESSEGDataset(
        root_dir=DATASET_ROOT,
        target_size=(64, 64),
        k_slices=5
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Progressive CSV
    progressive_csv = os.path.join(OUTPUT_DIR, f'progressive_validation_{timestamp}.csv')
    
    # Metrics storage
    all_dice, all_precision, all_recall, all_specificity, all_iou = [], [], [], [], []
    per_sample_results = []
    
    # Initialize CSV
    with open(progressive_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_idx', 'case_id', 'slice_idx', 'lesion_pixels',
            'dice', 'precision', 'recall', 'specificity', 'iou',
            'running_avg_dice', 'running_avg_precision', 'running_avg_recall',
            'running_avg_specificity', 'running_avg_iou',
            'running_std_dice', 'running_std_precision', 'running_std_recall',
            'running_std_specificity', 'running_std_iou',
            'samples_processed', 'progress_percent'
        ])
    
    print(f"\n{'='*80}")
    print("RUNNING VALIDATION")
    print("="*80)
    print("NOTE: This is ZERO-SHOT evaluation (NO retraining on MSLESSEG)")
    print("Training: PediMS (Pediatric MS)")
    print("Validation: MSLESSEG (MS Lesions - SAME PATHOLOGY!)")
    print("="*80 + "\n")
    
    with torch.no_grad():
        for idx, (images, masks, info) in enumerate(tqdm(dataloader, desc="Validating")):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            # Forward pass
            predictions = model(images)
            
            # Calculate metrics
            dice = dice_score(predictions, masks)
            precision = precision_score(predictions, masks)
            recall = recall_score(predictions, masks)
            specificity = specificity_score(predictions, masks)
            iou = iou_score(predictions, masks)
            
            # Store
            all_dice.append(dice)
            all_precision.append(precision)
            all_recall.append(recall)
            all_specificity.append(specificity)
            all_iou.append(iou)
            
            # Running stats
            running_avg_dice = np.mean(all_dice)
            running_std_dice = np.std(all_dice) if len(all_dice) > 1 else 0.0
            
            # Write to progressive CSV
            with open(progressive_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx, info['case_id'][0], info['slice_idx'][0].item(),
                    info['lesion_pixels'][0].item(),
                    f"{dice:.6f}", f"{precision:.6f}", f"{recall:.6f}",
                    f"{specificity:.6f}", f"{iou:.6f}",
                    f"{running_avg_dice:.6f}", f"{np.mean(all_precision):.6f}",
                    f"{np.mean(all_recall):.6f}", f"{np.mean(all_specificity):.6f}",
                    f"{np.mean(all_iou):.6f}",
                    f"{running_std_dice:.6f}", f"{np.std(all_precision) if len(all_precision)>1 else 0:.6f}",
                    f"{np.std(all_recall) if len(all_recall)>1 else 0:.6f}",
                    f"{np.std(all_specificity) if len(all_specificity)>1 else 0:.6f}",
                    f"{np.std(all_iou) if len(all_iou)>1 else 0:.6f}",
                    idx + 1, f"{((idx+1)/len(dataloader))*100:.2f}"
                ])
            
            # Log every 50 samples
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx+1}/{len(dataloader)} | Dice: {running_avg_dice:.4f} ± {running_std_dice:.4f}")
    
    # Final results
    print("\n" + "="*80)
    print("MSLESSEG VALIDATION RESULTS")
    print("="*80)
    print(f"Dice Score:   {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"Precision:    {np.mean(all_precision):.4f} ± {np.std(all_precision):.4f}")
    print(f"Recall:       {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")
    print(f"Specificity:  {np.mean(all_specificity):.4f} ± {np.std(all_specificity):.4f}")
    print(f"IoU:          {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Progressive CSV: {progressive_csv}")
    print("="*80)


if __name__ == "__main__":
    validate_mslesseg()
