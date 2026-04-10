"""
Cross-Dataset Validation on MSLESSEG - FINAL CORRECTED VERSION
===============================================================
Uses EXACT SAME MONAI pipeline as training:
1. LoadImaged
2. EnsureChannelFirstd
3. Orientationd (RAS)
4. Spacingd (1mm isotropic)
5. NormalizeIntensityd (nonzero, channel_wise)
6. Resized (64×64×64)
7. EnsureTyped

This should finally give correct results!
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
from tqdm import tqdm
import csv
import warnings
warnings.filterwarnings('ignore')

# MONAI imports - EXACT SAME AS TRAINING!
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, Resized, EnsureTyped
)
from monai.data import Dataset as MonaiDataset

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from final_model import HybridMiniSwin2D5_CSRF


def prepare_mslesseg_data_dicts(root_dir):
    """Prepare data dictionaries in MONAI format"""
    images_dir = os.path.join(root_dir, 'imagesTr')
    labels_dir = os.path.join(root_dir, 'labelsTr')
    
    # Get all FLAIR cases
    flair_files = sorted([f for f in os.listdir(images_dir) 
                         if f.endswith('_flair.nii.gz')])
    
    data_dicts = []
    for flair_file in flair_files:
        case_id = flair_file.replace('_flair.nii.gz', '')
        
        flair_path = os.path.join(images_dir, flair_file)
        label_path = os.path.join(labels_dir, f"{case_id}_mask.nii.gz")
        
        if os.path.exists(label_path):
            data_dicts.append({
                'image': [flair_path],  # List format like training
                'label': label_path,
                'case': case_id
            })
    
    return data_dicts


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
    OUTPUT_DIR = r"C:\Users\HP\EDI\csv_data\cross_dataset_validation_mslesseg_FINAL"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SPATIAL_SIZE = (64, 64, 64)  # SAME AS TRAINING!
    
    if DEVICE == 'cpu':
        print("ERROR: GPU not available. This will be too slow.")
        print("Please ensure CUDA is available before running.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("CROSS-DATASET VALIDATION: MSLESSEG (FINAL CORRECTED)")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*80)
    print("\nUSING EXACT SAME MONAI PIPELINE AS TRAINING:")
    print("  ✓ LoadImaged")
    print("  ✓ EnsureChannelFirstd")
    print("  ✓ Orientationd (RAS)")
    print("  ✓ Spacingd (1mm isotropic)")
    print("  ✓ NormalizeIntensityd (nonzero, channel_wise)")
    print("  ✓ Resized (64×64×64)")
    print("  ✓ EnsureTyped")
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
    best_val_dice = checkpoint.get('best_val_dice', 'unknown')
    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Training best val dice: {best_val_dice}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    model.eval()
    
    # Prepare data
    print("\nPreparing MSLESSEG data...")
    data_dicts = prepare_mslesseg_data_dicts(DATASET_ROOT)
    print(f"Found {len(data_dicts)} cases")
    
    if len(data_dicts) == 0:
        print("ERROR: No valid samples found in dataset!")
        return
    
    # Create EXACT SAME transforms as training validation
    val_transforms = Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        Spacingd(keys=["image","label"], pixdim=(1.0,1.0,1.0), mode=("bilinear","nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        Resized(keys=["image","label"], spatial_size=SPATIAL_SIZE, mode=("trilinear","nearest")),
        EnsureTyped(keys=["image","label"])
    ])
    
    # Create dataset and dataloader
    val_dataset = MonaiDataset(data=data_dicts, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Dataset ready: {len(val_dataset)} volumes")
    print(f"Spatial size: {SPATIAL_SIZE}")
    
    # Validation
    print("\n" + "="*80)
    print("RUNNING VALIDATION")
    print("="*80)
    print("NOTE: This is ZERO-SHOT evaluation (NO retraining on MSLESSEG)")
    print("Training: PediMS (Pediatric MS) - 84% Dice")
    print("Validation: MSLESSEG (Adult MS - SAME PATHOLOGY!)")
    print("="*80)
    
    # Progressive CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(OUTPUT_DIR, f'progressive_validation_FINAL_{timestamp}.csv')
    
    csv_columns = [
        'sample_idx', 'case_id', 'lesion_pixels',
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
            for idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
                images = batch['image'].to(DEVICE)
                masks = batch['label'].to(DEVICE)
                case_id = batch['case'][0]
                
                # Forward pass
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                
                # Binarize masks
                masks_binary = (masks > 0).float()
                
                # Calculate metrics
                dice = dice_score(preds, masks_binary)
                precision = precision_score(preds, masks_binary).item()
                recall = recall_score(preds, masks_binary).item()
                specificity = specificity_score(preds, masks_binary).item()
                iou = iou_score(preds, masks_binary).item()
                
                # Store
                all_dice.append(dice)
                all_precision.append(precision)
                all_recall.append(recall)
                all_specificity.append(specificity)
                all_iou.append(iou)
                
                # Write to CSV
                lesion_pixels = int(torch.sum(masks_binary > 0).item())
                
                row = {
                    'sample_idx': idx,
                    'case_id': case_id,
                    'lesion_pixels': lesion_pixels,
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
                    'progress_percent': f"{((idx + 1) / len(val_loader) * 100):.2f}"
                }
                writer.writerow(row)
                
                # Periodic updates
                if (idx + 1) % 5 == 0:
                    print(f"Processed {idx + 1}/{len(val_loader)} | Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    
    # Final results
    print("\n" + "="*80)
    print("MSLESSEG VALIDATION RESULTS (FINAL)")
    print("="*80)
    print(f"Dice Score:   {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"Precision:    {np.mean(all_precision):.4f} ± {np.std(all_precision):.4f}")
    print(f"Recall:       {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")
    print(f"Specificity:  {np.mean(all_specificity):.4f} ± {np.std(all_specificity):.4f}")
    print(f"IoU:          {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
    print("="*80)
    
    # Interpretation
    dice_mean = np.mean(all_dice)
    print("\nInterpretation:")
    if dice_mean >= 0.70:
        print(f"  ✓ EXCELLENT cross-dataset generalization ({dice_mean:.1%})")
        print("    Model generalizes very well to unseen MS data!")
    elif dice_mean >= 0.55:
        print(f"  ✓ GOOD cross-dataset generalization ({dice_mean:.1%})")
        print("    Model shows strong generalization despite domain shift")
    elif dice_mean >= 0.40:
        print(f"  ○ MODERATE cross-dataset performance ({dice_mean:.1%})")
        print("    Significant domain shift but still clinically useful")
    else:
        print(f"  ✗ POOR cross-dataset performance ({dice_mean:.1%})")
        print("    Severe domain shift - model may need fine-tuning")
    
    print(f"\nPerformance drop from training: {(0.8399 - dice_mean)*100:.1f}%")
    
    # Save summary
    summary = {
        'dataset': 'MSLESSEG (MS Lesions)',
        'model_path': MODEL_PATH,
        'epoch': str(epoch),
        'training_best_dice': float(best_val_dice) if isinstance(best_val_dice, (int, float)) else str(best_val_dice),
        'total_samples': len(val_loader),
        'metrics': {
            'dice': {'mean': float(np.mean(all_dice)), 'std': float(np.std(all_dice))},
            'precision': {'mean': float(np.mean(all_precision)), 'std': float(np.std(all_precision))},
            'recall': {'mean': float(np.mean(all_recall)), 'std': float(np.std(all_recall))},
            'specificity': {'mean': float(np.mean(all_specificity)), 'std': float(np.std(all_specificity))},
            'iou': {'mean': float(np.mean(all_iou)), 'std': float(np.std(all_iou))}
        },
        'pipeline': 'EXACT SAME as training (MONAI transforms)',
        'interpretation': 'See console output above'
    }
    
    json_path = os.path.join(OUTPUT_DIR, f'validation_summary_FINAL_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Progressive CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    print("="*80)


if __name__ == "__main__":
    validate()
