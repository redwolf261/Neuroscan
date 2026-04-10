"""
Debug: Check actual model predictions on MSLESSEG
"""

import torch
import nibabel as nib
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from final_model import HybridMiniSwin2D5_CSRF

# Load model
device = torch.device('cuda')
model = HybridMiniSwin2D5_CSRF(channels=[32, 64, 128, 256]).to(device)
checkpoint = torch.load(r"C:\Users\HP\EDI\.resume_checkpoints\seg_resume.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("="*80)
print("CHECKING MODEL PREDICTIONS ON MSLESSEG")
print("="*80)

# Load one MSLESSEG case
data_dir = Path(r"G:\My Drive\Dataset\MSLESSEG\processed")
case = "case_001"
flair_path = data_dir / case / "FLAIR.nii.gz"
mask_path = data_dir / case / f"{case}_mask.nii.gz"

print(f"\nLoading: {case}")
flair_data = nib.load(flair_path).get_fdata()
mask_data = nib.load(mask_path).get_fdata()

print(f"FLAIR shape: {flair_data.shape}")
print(f"Mask shape: {mask_data.shape}")
print(f"FLAIR range: {flair_data.min():.2f} to {flair_data.max():.2f}")
print(f"Mask unique: {np.unique(mask_data)}")

# Process one slice with lesions
lesion_slices = np.where(mask_data.sum(axis=(0,1)) > 0)[0]
print(f"\nSlices with lesions: {len(lesion_slices)}")

if len(lesion_slices) > 0:
    slice_idx = lesion_slices[len(lesion_slices)//2]  # Middle slice with lesions
    print(f"Testing slice {slice_idx}")
    
    # Extract 2.5D volume (5 slices)
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
    
    print(f"\nVolume for model: {volume.shape}")
    print(f"Mask: {mask.shape}, lesion pixels: {mask.sum()}")
    
    # Normalize
    volume = (volume - volume.mean()) / (volume.std() + 1e-8)
    
    # Prepare for model
    volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, k)
    volume_tensor = volume_tensor.permute(0, 1, 4, 2, 3)  # (1, 1, k, H, W)
    volume_tensor = volume_tensor.to(device)
    
    print(f"\nInput tensor: {volume_tensor.shape}")
    print(f"Input range: {volume_tensor.min():.3f} to {volume_tensor.max():.3f}")
    
    # Forward pass
    with torch.no_grad():
        output = model(volume_tensor)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: {output.min():.6f} to {output.max():.6f}")
    print(f"Output mean: {output.mean():.6f}")
    print(f"Output std: {output.std():.6f}")
    
    # Get prediction
    pred = output.squeeze().cpu().numpy()
    
    print(f"\nPrediction statistics:")
    print(f"  Min: {pred.min():.6f}")
    print(f"  Max: {pred.max():.6f}")
    print(f"  Mean: {pred.mean():.6f}")
    print(f"  Std: {pred.std():.6f}")
    print(f"  Median: {np.median(pred):.6f}")
    
    # Check distribution
    print(f"\nPrediction distribution:")
    print(f"  < 0.1:  {(pred < 0.1).sum()}/{pred.size} ({(pred < 0.1).mean():.1%})")
    print(f"  < 0.3:  {(pred < 0.3).sum()}/{pred.size} ({(pred < 0.3).mean():.1%})")
    print(f"  < 0.5:  {(pred < 0.5).sum()}/{pred.size} ({(pred < 0.5).mean():.1%})")
    print(f"  >= 0.5: {(pred >= 0.5).sum()}/{pred.size} ({(pred >= 0.5).mean():.1%})")
    print(f"  >= 0.7: {(pred >= 0.7).sum()}/{pred.size} ({(pred >= 0.7).mean():.1%})")
    print(f"  >= 0.9: {(pred >= 0.9).sum()}/{pred.size} ({(pred >= 0.9).mean():.1%})")
    
    # Binary prediction at 0.5
    pred_binary = (pred > 0.5).astype(np.float32)
    
    print(f"\nBinary prediction (threshold=0.5):")
    print(f"  Predicted lesion pixels: {pred_binary.sum()}")
    print(f"  Ground truth lesion pixels: {mask.sum()}")
    print(f"  Ratio: {pred_binary.sum() / max(mask.sum(), 1):.2f}x")
    
    # Compute metrics
    intersection = (pred_binary * mask).sum()
    dice = 2 * intersection / (pred_binary.sum() + mask.sum() + 1e-8)
    precision = intersection / (pred_binary.sum() + 1e-8)
    recall = intersection / (mask.sum() + 1e-8)
    
    print(f"\nMetrics:")
    print(f"  Dice: {dice:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    # Try different thresholds
    print(f"\nTrying different thresholds:")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred_thresh = (pred > thresh).astype(np.float32)
        intersection = (pred_thresh * mask).sum()
        dice_thresh = 2 * intersection / (pred_thresh.sum() + mask.sum() + 1e-8)
        prec = intersection / (pred_thresh.sum() + 1e-8)
        rec = intersection / (mask.sum() + 1e-8)
        print(f"  {thresh:.1f}: Dice={dice_thresh:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, Pred={pred_thresh.sum():.0f}")

print("\n" + "="*80)
