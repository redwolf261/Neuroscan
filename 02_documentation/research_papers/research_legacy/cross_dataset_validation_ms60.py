"""
Cross-Dataset Validation on MS Cross Validation Dataset (60 MS Patients)
=========================================================================
Validates HybridMiniSwin2.5D-CSRF on external MS dataset with 60 patients.
Same pathology as training (MS lesions) but different scanner/protocol.

CRITICAL FIX: Decoder already has nn.Sigmoid() - do NOT apply sigmoid again!
"""

import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))
from final_model import HybridMiniSwin2D5_CSRF

# Paths
DATA_DIR = Path(r"C:\Users\HP\EDI\MS cross validation")
MODEL_PATH = Path(r"C:\Users\HP\EDI\.resume_checkpoints\seg_resume.pth")
OUTPUT_DIR = Path(r"C:\Users\HP\EDI\csv_data\cross_dataset_validation_ms60_patients")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("MS CROSS VALIDATION DATASET (60 MS Patients)")
print("="*80)
print(f"Device: cuda")
print(f"Model: {MODEL_PATH}")
print(f"Dataset: {DATA_DIR}")
print(f"Output: {OUTPUT_DIR}")
print("="*80)
print()
print("🔧 CRITICAL FIX:")
print("  ✓ Decoder already has Sigmoid - don't apply it again!")
print("  ✓ Use model output directly (already in [0, 1])")
print("="*80)
print()


class MS60Dataset(torch.utils.data.Dataset):
    """Dataset for MS Cross Validation (60 patients)"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.samples = []
        
        # Find all patients
        patient_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('Patient-')])
        
        for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
            patient_id = patient_dir.name
            patient_num = patient_id.split('-')[1]
            
            flair_path = patient_dir / f"{patient_num}-Flair.nii"
            mask_path = patient_dir / f"{patient_num}-LesionSeg-Flair.nii"
            
            if not flair_path.exists() or not mask_path.exists():
                print(f"  Skipping {patient_id}: missing files")
                continue
            
            # Load to check for lesions
            try:
                flair_data = nib.load(flair_path).get_fdata()
                mask_data = nib.load(mask_path).get_fdata()
                
                # Find slices with lesions
                for slice_idx in range(mask_data.shape[2]):
                    lesion_count = mask_data[:, :, slice_idx].sum()
                    if lesion_count > 0:
                        self.samples.append({
                            'patient_id': patient_id,
                            'flair_path': flair_path,
                            'mask_path': mask_path,
                            'slice_idx': slice_idx,
                            'lesion_pixels': lesion_count
                        })
            except Exception as e:
                print(f"  Error loading {patient_id}: {e}")
                continue
        
        print(f"Dataset loaded: {len(self.samples)} slices with lesions from {len(patient_dirs)} patients")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load full volumes
        flair_data = nib.load(sample['flair_path']).get_fdata()
        mask_data = nib.load(sample['mask_path']).get_fdata()
        
        slice_idx = sample['slice_idx']
        
        # Extract 2.5D volume (5 slices centered on target slice)
        k = 5
        start_slice = max(0, slice_idx - k//2)
        end_slice = min(flair_data.shape[2], slice_idx + k//2 + 1)
        
        volume = flair_data[:, :, start_slice:end_slice]
        mask = mask_data[:, :, slice_idx]
        
        # Pad if needed
        if volume.shape[2] < k:
            pad_before = (k - volume.shape[2]) // 2
            pad_after = k - volume.shape[2] - pad_before
            volume = np.pad(volume, ((0,0), (0,0), (pad_before, pad_after)), mode='edge')
        
        # Normalize volume
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        
        # Convert to tensors
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0)  # (1, H, W, k)
        volume_tensor = volume_tensor.permute(0, 3, 1, 2)  # (1, k, H, W)
        
        # Resize to match training size (model expects divisible by 16)
        # Resize to 240x240 (divisible by 16)
        target_size = (240, 240)
        volume_tensor = F.interpolate(volume_tensor, size=target_size, mode='bilinear', align_corners=False)
        
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Resize mask to match volume
        mask_tensor = F.interpolate(mask_tensor, size=target_size, mode='nearest')
        mask_tensor = mask_tensor.squeeze()  # Remove batch and channel dims
        
        info = {
            'patient': sample['patient_id'],
            'slice_idx': slice_idx,
            'lesion_pixels': sample['lesion_pixels']
        }
        
        return volume_tensor, mask_tensor, info


def compute_metrics(pred, target):
    """Compute segmentation metrics"""
    pred = pred.flatten()
    target = target.flatten()
    
    # Binary masks
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    # True/False Positives/Negatives
    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()
    tn = ((1 - pred_binary) * (1 - target_binary)).sum().item()
    
    # Metrics
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def validate():
    """Run validation"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model - USING CORRECT CHANNELS [32, 64, 128, 256, 512] (4 stages)
    print("Loading model...")
    model = HybridMiniSwin2D5_CSRF(channels=[32, 64, 128, 256, 512]).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    train_dice = checkpoint.get('best_dice', 'unknown')
    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Training best dice: {train_dice}")
    
    # Load dataset
    print("Loading MS60 dataset...")
    dataset = MS60Dataset(DATA_DIR)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print()
    print("="*80)
    print("RUNNING VALIDATION")
    print("="*80)
    
    # Progressive CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"progressive_validation_{timestamp}.csv"
    
    results = []
    
    with torch.no_grad():
        for idx, (volumes, masks, infos) in enumerate(tqdm(dataloader, desc="Validating")):
            volumes = volumes.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(volumes)
            
            # ✓ CRITICAL: Model output is already in [0, 1] from decoder's sigmoid
            # DO NOT apply sigmoid again!
            preds = outputs.squeeze()
            
            # Binary prediction at threshold 0.5
            preds_binary = (preds > 0.5).float()
            
            # Compute metrics
            metrics = compute_metrics(preds, masks.squeeze())
            
            # Store results
            result = {
                'patient': infos['patient'][0],
                'slice_idx': infos['slice_idx'][0].item(),
                'lesion_pixels': infos['lesion_pixels'][0].item(),
                'dice': metrics['dice'],
                'iou': metrics['iou'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'specificity': metrics['specificity']
            }
            results.append(result)
            
            # Save progressively
            if (idx + 1) % 50 == 0:
                df_temp = pd.DataFrame(results)
                df_temp.to_csv(csv_path, index=False)
                print(f"Processed {idx+1}/{len(dataset)} | Dice: {df_temp['dice'].mean():.4f} ± {df_temp['dice'].std():.4f}")
    
    # Final save
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    
    # Summary statistics
    print()
    print("="*80)
    print("MS60 VALIDATION RESULTS")
    print("="*80)
    print(f"Total slices: {len(df)}")
    print(f"Unique patients: {df['patient'].nunique()}")
    print()
    print(f"Dice Score:   {df['dice'].mean():.4f} ± {df['dice'].std():.4f}")
    print(f"IoU:          {df['iou'].mean():.4f} ± {df['iou'].std():.4f}")
    print(f"Precision:    {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
    print(f"Recall:       {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")
    print(f"Specificity:  {df['specificity'].mean():.4f} ± {df['specificity'].std():.4f}")
    print()
    
    # Per-patient statistics
    patient_stats = df.groupby('patient').agg({
        'dice': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'slice_idx': 'count'
    }).rename(columns={'slice_idx': 'num_slices'})
    
    patient_stats = patient_stats.sort_values('dice', ascending=False)
    
    print("Per-Patient Results (Top 10):")
    print(patient_stats.head(10).to_string())
    print()
    print("Per-Patient Results (Bottom 10):")
    print(patient_stats.tail(10).to_string())
    print()
    
    # Distribution
    print("Dice Score Distribution:")
    print(f"  Excellent (>70%): {(df['dice'] > 0.70).sum()}/{len(df)} ({(df['dice'] > 0.70).mean():.1%})")
    print(f"  Good (50-70%):    {((df['dice'] >= 0.50) & (df['dice'] <= 0.70)).sum()}/{len(df)}")
    print(f"  Moderate (30-50%):{((df['dice'] >= 0.30) & (df['dice'] < 0.50)).sum()}/{len(df)}")
    print(f"  Low (10-30%):     {((df['dice'] >= 0.10) & (df['dice'] < 0.30)).sum()}/{len(df)}")
    print(f"  Very low (<10%):  {(df['dice'] < 0.10).sum()}/{len(df)} ({(df['dice'] < 0.10).mean():.1%})")
    print()
    
    print("="*80)
    print(f"✓ Results saved: {csv_path}")
    print("="*80)
    
    # Interpretation
    mean_dice = df['dice'].mean()
    print()
    print("INTERPRETATION:")
    if mean_dice >= 0.60:
        print("  🎉 EXCELLENT cross-dataset generalization!")
        print("  Model successfully learned MS-specific features.")
    elif mean_dice >= 0.40:
        print("  ✓ GOOD cross-dataset performance.")
        print("  Model generalizes well despite domain shift.")
    elif mean_dice >= 0.25:
        print("  ○ MODERATE cross-dataset performance.")
        print("  Moderate domain shift observed.")
    else:
        print("  ⚠ SIGNIFICANT domain shift.")
        print("  May need few-shot fine-tuning.")
    
    print()
    print("Ready for research paper! 🎉")
    print("="*80)


if __name__ == "__main__":
    validate()
