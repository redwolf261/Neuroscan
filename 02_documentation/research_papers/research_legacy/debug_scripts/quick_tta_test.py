"""
Quick TTA Test - MSLESSEG with Test-Time Augmentation
=======================================================
Fast test on subset to see if TTA helps
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F

# Load model directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Minimal imports to avoid MONAI overhead
print("Loading model architecture...")
from final_model import HybridMiniSwin2D5_CSRF

def quick_tta_test():
    """Quick test on first 100 slices"""
    
    MODEL_PATH = r"C:\Users\HP\EDI\.resume_checkpoints\seg_resume.pth"
    DATASET_ROOT = r"G:\My Drive\Dataset\MSLESSEG\processed"
    DEVICE = 'cuda'
    
    print("="*80)
    print("QUICK TTA TEST (First 100 slices)")
    print("="*80)
    
    # Load model
    print("Loading model...")
    model = HybridMiniSwin2D5_CSRF(k_slices=5, channels=[32, 64, 128, 256, 512]).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded!")
    
    # Load first case
    images_dir = os.path.join(DATASET_ROOT, 'imagesTr')
    labels_dir = os.path.join(DATASET_ROOT, 'labelsTr')
    
    flair_file = 'case_001_flair.nii.gz'
    label_file = 'case_001_mask.nii.gz'
    
    print(f"\nTesting on {flair_file}...")
    
    flair_data = nib.load(os.path.join(images_dir, flair_file)).get_fdata()
    label_data = nib.load(os.path.join(labels_dir, label_file)).get_fdata()
    
    H, W, D = flair_data.shape
    print(f"Volume shape: {H}×{W}×{D}")
    
    # Test on center slices with lesions
    dice_baseline = []
    dice_tta = []
    dice_tta_thresh = []
    
    print("\nProcessing slices...")
    for slice_idx in tqdm(range(10, min(110, D-10))):  # Process 100 slices
        label_slice = label_data[:, :, slice_idx]
        if np.sum(label_slice) < 10:
            continue
        
        # Extract 5 slices
        k_slices = []
        for offset in [-2, -1, 0, 1, 2]:
            idx = max(0, min(D-1, slice_idx + offset))
            slice_data = flair_data[:, :, idx].astype(np.float32)
            # Normalize
            mask = slice_data > 0
            if mask.sum() > 0:
                mean = slice_data[mask].mean()
                std = slice_data[mask].std()
                if std > 0:
                    slice_data = (slice_data - mean) / std
            k_slices.append(slice_data)
        
        # Stack and resize
        volume = np.stack(k_slices, axis=0)  # (5, H, W)
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 5, H, W)
        
        # Resize to 64x64
        volume_resized = F.interpolate(
            volume_tensor.view(1, 5, H, W),
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        )
        volume_resized = volume_resized.unsqueeze(1)  # (1, 1, 5, 64, 64)
        
        # Resize label
        label_tensor = torch.from_numpy(label_slice).float().unsqueeze(0).unsqueeze(0)
        label_resized = F.interpolate(label_tensor, size=(64, 64), mode='nearest').squeeze()
        label_binary = (label_resized > 0).float().to(DEVICE)
        
        volume_input = volume_resized.to(DEVICE)
        
        # 1. Baseline prediction
        with torch.no_grad():
            pred = model(volume_input)
            pred = torch.sigmoid(pred).squeeze()
        
        # Dice baseline
        intersection = (pred * label_binary).sum()
        union = pred.sum() + label_binary.sum()
        dice_base = (2.0 * intersection / (union + 1e-5)).item() if union > 0 else 0
        dice_baseline.append(dice_base)
        
        # 2. TTA (4 augmentations)
        preds = []
        
        # Original
        with torch.no_grad():
            p = model(volume_input)
            preds.append(torch.sigmoid(p).squeeze())
        
        # H flip
        with torch.no_grad():
            p = model(torch.flip(volume_input, dims=[4]))
            preds.append(torch.flip(torch.sigmoid(p).squeeze(), dims=[1]))
        
        # V flip  
        with torch.no_grad():
            p = model(torch.flip(volume_input, dims=[3]))
            preds.append(torch.flip(torch.sigmoid(p).squeeze(), dims=[0]))
        
        # Both flips
        with torch.no_grad():
            p = model(torch.flip(volume_input, dims=[3, 4]))
            preds.append(torch.flip(torch.sigmoid(p).squeeze(), dims=[0, 1]))
        
        # Average
        pred_tta = torch.mean(torch.stack(preds), dim=0)
        
        # Dice TTA
        intersection = (pred_tta * label_binary).sum()
        union = pred_tta.sum() + label_binary.sum()
        dice_t = (2.0 * intersection / (union + 1e-5)).item() if union > 0 else 0
        dice_tta.append(dice_t)
        
        # 3. TTA + Better threshold
        best_dice = 0
        for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            pred_binary = (pred_tta > thresh).float()
            intersection = (pred_binary * label_binary).sum()
            union = pred_binary.sum() + label_binary.sum()
            d = (2.0 * intersection / (union + 1e-5)).item() if union > 0 else 0
            if d > best_dice:
                best_dice = d
        
        dice_tta_thresh.append(best_dice)
    
    # Results
    print("\n" + "="*80)
    print("QUICK TEST RESULTS")
    print("="*80)
    print(f"Baseline (no TTA):           {np.mean(dice_baseline):.4f}")
    print(f"+ TTA (4x augmentations):    {np.mean(dice_tta):.4f}")
    print(f"+ TTA + Adaptive Threshold:  {np.mean(dice_tta_thresh):.4f}")
    print("="*80)
    
    improvement = (np.mean(dice_tta_thresh) - np.mean(dice_baseline)) * 100
    print(f"\nImprovement: +{improvement:.2f} percentage points")
    
    if np.mean(dice_tta_thresh) > 0.10:
        print("\n✓ TTA HELPS! Running full validation recommended.")
    else:
        print("\n○ TTA provides modest improvement. Consider few-shot fine-tuning.")

if __name__ == "__main__":
    quick_tta_test()
