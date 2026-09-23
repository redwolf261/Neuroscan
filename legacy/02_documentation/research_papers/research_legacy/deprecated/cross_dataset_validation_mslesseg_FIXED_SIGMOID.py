"""
MSLESSEG Validation - FIXED (No Double Sigmoid!)
=================================================
BUG FOUND: Decoder already has Sigmoid, don't apply it again!
This was causing all predictions to be in 0.5-0.75 range.
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
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from final_model import HybridMiniSwin2D5_CSRF


class MSLESSEG_Dataset(Dataset):
    def __init__(self, root_dir, k_slices=5, target_size=(64, 64)):
        self.root_dir = root_dir
        self.k_slices = k_slices
        self.target_size = target_size
        self.half_k = k_slices // 2
        
        self.images_dir = os.path.join(root_dir, 'imagesTr')
        self.labels_dir = os.path.join(root_dir, 'labelsTr')
        
        self.samples = []
        
        print(f"Loading MSLESSEG dataset...")
        
        flair_files = sorted([f for f in os.listdir(self.images_dir) 
                             if f.endswith('_flair.nii.gz')])
        
        for flair_file in tqdm(flair_files, desc="Processing cases"):
            case_id = flair_file.replace('_flair.nii.gz', '')
            
            flair_path = os.path.join(self.images_dir, flair_file)
            label_path = os.path.join(self.labels_dir, f"{case_id}_mask.nii.gz")
            
            if not os.path.exists(label_path):
                continue
            
            flair_img = nib.load(flair_path)
            label_img = nib.load(label_path)
            
            flair_data = flair_img.get_fdata()
            label_data = label_img.get_fdata()
            
            H, W, D = flair_data.shape
            
            for slice_idx in range(self.half_k, D - self.half_k):
                label_slice = label_data[:, :, slice_idx]
                if np.sum(label_slice > 0) > 10:
                    self.samples.append({
                        'case_id': case_id,
                        'flair_path': flair_path,
                        'label_path': label_path,
                        'slice_idx': slice_idx,
                        'num_slices': D
                    })
        
        print(f"Dataset loaded: {len(self.samples)} slices with lesions")
    
    def __len__(self):
        return len(self.samples)
    
    def extract_k_slices(self, volume_3d, slice_idx):
        slices = []
        for offset in range(-self.half_k, self.half_k + 1):
            idx = slice_idx + offset
            idx = max(0, min(volume_3d.shape[2] - 1, idx))
            slices.append(volume_3d[:, :, idx])
        return np.stack(slices, axis=0)
    
    def normalize_slice(self, slice_data):
        slice_data = slice_data.astype(np.float32)
        nonzero_mask = slice_data > 0
        if nonzero_mask.sum() > 0:
            mean = slice_data[nonzero_mask].mean()
            std = slice_data[nonzero_mask].std()
            if std > 0:
                slice_data = (slice_data - mean) / std
        return slice_data
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        flair_data = nib.load(sample['flair_path']).get_fdata()
        label_data = nib.load(sample['label_path']).get_fdata()
        
        slice_idx = sample['slice_idx']
        
        flair_k_slices = self.extract_k_slices(flair_data, slice_idx)
        label_central = label_data[:, :, slice_idx]
        
        # Normalize
        flair_normalized = np.stack([self.normalize_slice(flair_k_slices[i]) 
                                     for i in range(self.k_slices)], axis=0)
        
        # Resize
        flair_tensor = torch.from_numpy(flair_normalized).float()
        flair_resized = F.interpolate(
            flair_tensor.unsqueeze(0),
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        label_tensor = torch.from_numpy(label_central).float().unsqueeze(0).unsqueeze(0)
        label_resized = F.interpolate(
            label_tensor,
            size=self.target_size,
            mode='nearest'
        ).squeeze()
        
        label_binary = (label_resized > 0).float()
        
        # Create 3D volume (1, 1, k, H, W)
        volume = torch.zeros(1, self.k_slices, self.target_size[0], self.target_size[1])
        for i in range(self.k_slices):
            volume[0, i, :, :] = flair_resized[i]
        
        info = {
            'case_id': sample['case_id'],
            'slice_idx': slice_idx,
            'lesion_pixels': int(torch.sum(label_binary).item())
        }
        
        return volume, label_binary, info


def dice_score(pred, target, smooth=1e-5):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)


def precision_score(pred, target, smooth=1e-5):
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + smooth) / (tp + fp + smooth)


def recall_score(pred, target, smooth=1e-5):
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + smooth) / (tp + fn + smooth)


def specificity_score(pred, target, smooth=1e-5):
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    return (tn + smooth) / (tn + fp + smooth)


def iou_score(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def validate():
    """Run validation - NO DOUBLE SIGMOID!"""
    
    MODEL_PATH = r"C:\Users\HP\EDI\.resume_checkpoints\seg_resume.pth"
    DATASET_ROOT = r"G:\My Drive\Dataset\MSLESSEG\processed"
    OUTPUT_DIR = r"C:\Users\HP\EDI\csv_data\cross_dataset_validation_mslesseg_FIXED_NO_DOUBLE_SIGMOID"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if DEVICE == 'cpu':
        print("ERROR: GPU required")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("MSLESSEG VALIDATION - FIXED (No Double Sigmoid!)")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*80)
    print("\n🔧 BUG FIX:")
    print("  ✓ Decoder already has Sigmoid - don't apply it again!")
    print("  ✓ Use model output directly (already in [0, 1])")
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
    
    # Dataset
    dataset = MSLESSEG_Dataset(
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
    csv_path = os.path.join(OUTPUT_DIR, f'progressive_validation_FIXED_{timestamp}.csv')
    
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
                
                # Forward pass - NO SIGMOID! Decoder already has it!
                outputs = model(volumes)  # Already in [0, 1]
                preds = outputs.squeeze()  # (H, W)
                masks = masks.squeeze()  # (H, W)
                
                # Threshold at 0.5
                preds_binary = (preds > 0.5).float()
                
                # Calculate metrics
                dice = dice_score(preds_binary, masks).item()
                precision = precision_score(preds_binary, masks).item()
                recall = recall_score(preds_binary, masks).item()
                specificity = specificity_score(preds_binary, masks).item()
                iou = iou_score(preds_binary, masks).item()
                
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
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx+1}/{len(dataloader)} | Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    
    # Final results
    print("\n" + "="*80)
    print("MSLESSEG VALIDATION RESULTS (FIXED - No Double Sigmoid)")
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
        print(f"\n🎉 EXCELLENT cross-dataset performance ({dice_mean:.1%})")
        print("   Model generalizes very well to unseen MS data!")
    elif dice_mean >= 0.55:
        print(f"\n✓ GOOD cross-dataset performance ({dice_mean:.1%})")
        print("   Strong generalization despite domain shift")
    elif dice_mean >= 0.40:
        print(f"\n○ MODERATE cross-dataset performance ({dice_mean:.1%})")
        print("   Acceptable generalization with domain shift")
    elif dice_mean >= 0.25:
        print(f"\n○ LIMITED cross-dataset performance ({dice_mean:.1%})")
        print("   Significant domain shift but shows some generalization")
    else:
        print(f"\n✗ POOR cross-dataset performance ({dice_mean:.1%})")
        print("   Severe domain shift - further investigation needed")
    
    print(f"\nPerformance vs training: {dice_mean:.1%} (vs 84% in-domain)")
    print(f"Performance drop: {(0.8399 - dice_mean)*100:.1f} percentage points")
    
    # Save summary
    summary = {
        'dataset': 'MSLESSEG (MS Lesions)',
        'fix_applied': 'Removed double sigmoid (decoder already has sigmoid)',
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
    
    json_path = os.path.join(OUTPUT_DIR, f'validation_summary_FIXED_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    validate()
