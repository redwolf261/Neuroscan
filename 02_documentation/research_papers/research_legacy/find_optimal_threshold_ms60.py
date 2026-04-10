"""
Find Optimal Threshold for MS60 Dataset
========================================
Tests different thresholds to find the best Dice score.
"""

import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from final_model import HybridMiniSwin2D5_CSRF

# Paths
DATA_DIR = Path(r"C:\Users\HP\EDI\MS cross validation")
MODEL_PATH = Path(r"C:\Users\HP\EDI\.resume_checkpoints\seg_resume.pth")

print("="*80)
print("FINDING OPTIMAL THRESHOLD FOR MS60")
print("="*80)


class MS60Dataset(torch.utils.data.Dataset):
    """Simplified dataset for threshold testing"""
    
    def __init__(self, data_dir, max_slices=100):
        self.data_dir = Path(data_dir)
        self.samples = []
        
        patient_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('Patient-')])
        
        slice_count = 0
        for patient_dir in patient_dirs:
            if slice_count >= max_slices:
                break
                
            patient_id = patient_dir.name
            patient_num = patient_id.split('-')[1]
            
            flair_path = patient_dir / f"{patient_num}-Flair.nii"
            mask_path = patient_dir / f"{patient_num}-LesionSeg-Flair.nii"
            
            if not flair_path.exists() or not mask_path.exists():
                continue
            
            try:
                flair_data = nib.load(flair_path).get_fdata()
                mask_data = nib.load(mask_path).get_fdata()
                
                for slice_idx in range(mask_data.shape[2]):
                    if slice_count >= max_slices:
                        break
                    lesion_count = mask_data[:, :, slice_idx].sum()
                    if lesion_count > 0:
                        self.samples.append({
                            'patient_id': patient_id,
                            'flair_path': flair_path,
                            'mask_path': mask_path,
                            'slice_idx': slice_idx,
                        })
                        slice_count += 1
            except Exception as e:
                print(f"Error loading {patient_id}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} slices for threshold testing")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        flair_data = nib.load(sample['flair_path']).get_fdata()
        mask_data = nib.load(sample['mask_path']).get_fdata()
        
        slice_idx = sample['slice_idx']
        
        # Extract 2.5D volume
        k = 5
        start_slice = max(0, slice_idx - k//2)
        end_slice = min(flair_data.shape[2], slice_idx + k//2 + 1)
        
        volume = flair_data[:, :, start_slice:end_slice]
        mask = mask_data[:, :, slice_idx]
        
        if volume.shape[2] < k:
            pad_before = (k - volume.shape[2]) // 2
            pad_after = k - volume.shape[2] - pad_before
            volume = np.pad(volume, ((0,0), (0,0), (pad_before, pad_after)), mode='edge')
        
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0)
        volume_tensor = volume_tensor.permute(0, 3, 1, 2)
        
        # Resize to 240x240
        target_size = (240, 240)
        volume_tensor = F.interpolate(volume_tensor, size=target_size, mode='bilinear', align_corners=False)
        
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        mask_tensor = F.interpolate(mask_tensor, size=target_size, mode='nearest')
        mask_tensor = mask_tensor.squeeze()
        
        return volume_tensor, mask_tensor


def compute_dice(pred, target, threshold):
    """Compute Dice score at specific threshold"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    intersection = (pred_binary * target_binary).sum()
    dice = 2 * intersection / (pred_binary.sum() + target_binary.sum() + 1e-8)
    
    return dice.item()


def find_optimal_threshold():
    """Find optimal threshold"""
    
    device = torch.device('cuda')
    
    # Load model
    print("Loading model...")
    model = HybridMiniSwin2D5_CSRF(channels=[32, 64, 128, 256, 512]).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load small subset for testing
    print("Loading dataset subset...")
    dataset = MS60Dataset(DATA_DIR, max_slices=100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Collect all predictions
    print("\nCollecting predictions...")
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for volumes, masks in tqdm(dataloader, desc="Processing"):
            volumes = volumes.to(device)
            outputs = model(volumes)
            
            preds = outputs.squeeze().cpu()
            masks = masks.cpu()
            
            all_preds.append(preds)
            all_masks.append(masks)
    
    # Stack all predictions
    all_preds = torch.stack(all_preds)
    all_masks = torch.stack(all_masks)
    
    print(f"\nPrediction statistics:")
    print(f"  Min: {all_preds.min():.6f}")
    print(f"  Max: {all_preds.max():.6f}")
    print(f"  Mean: {all_preds.mean():.6f}")
    print(f"  Median: {all_preds.median():.6f}")
    print(f"  Std: {all_preds.std():.6f}")
    
    # Test thresholds
    print("\nTesting thresholds...")
    thresholds = np.arange(0.01, 1.0, 0.01)
    results = []
    
    for thresh in tqdm(thresholds, desc="Thresholds"):
        dice_scores = []
        for pred, mask in zip(all_preds, all_masks):
            dice = compute_dice(pred, mask, thresh)
            dice_scores.append(dice)
        
        mean_dice = np.mean(dice_scores)
        results.append({
            'threshold': thresh,
            'dice': mean_dice
        })
    
    df = pd.DataFrame(results)
    
    # Find best threshold
    best_idx = df['dice'].idxmax()
    best_thresh = df.loc[best_idx, 'threshold']
    best_dice = df.loc[best_idx, 'dice']
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Best threshold: {best_thresh:.3f}")
    print(f"Best Dice score: {best_dice:.4f} ({best_dice*100:.2f}%)")
    print()
    
    print("Top 10 thresholds:")
    print(df.nlargest(10, 'dice').to_string(index=False))
    print()
    
    print("Comparison:")
    print(f"  At 0.50: {df[df['threshold'] == 0.50]['dice'].values[0]:.4f}")
    print(f"  At 0.70: {df[df['threshold'] == 0.70]['dice'].values[0]:.4f}")
    print(f"  At 0.90: {df[df['threshold'] == 0.90]['dice'].values[0]:.4f}")
    print(f"  At {best_thresh:.2f}: {best_dice:.4f} ⭐ BEST")
    print()
    
    # Analyze prediction distribution
    print("Prediction distribution:")
    for percentile in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(all_preds.numpy(), percentile)
        print(f"  {percentile}th percentile: {val:.6f}")
    
    print("\n" + "="*80)
    print(f"Improvement: {best_dice / df[df['threshold'] == 0.50]['dice'].values[0]:.2f}x better than 0.5 threshold")
    print("="*80)
    
    return best_thresh, best_dice, df


if __name__ == "__main__":
    best_thresh, best_dice, df = find_optimal_threshold()
    
    print("\n💡 RECOMMENDATION:")
    if best_dice > 0.30:
        print(f"  Use threshold={best_thresh:.3f} for much better performance!")
        print("  This shows the model learned useful features, just needs calibration.")
    elif best_dice > 0.15:
        print(f"  Use threshold={best_thresh:.3f} for moderate improvement.")
        print("  Consider few-shot fine-tuning for better results.")
    else:
        print("  Optimal threshold helps but domain shift is significant.")
        print("  Recommend few-shot fine-tuning on MS60 samples.")
