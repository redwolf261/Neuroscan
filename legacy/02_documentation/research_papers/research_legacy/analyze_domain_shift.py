"""
Analyze MS60 Dataset vs PediMS to understand domain shift
"""

import nibabel as nib
import numpy as np
from pathlib import Path

# Check MS60
ms60_dir = Path(r"C:\Users\HP\EDI\MS cross validation\Patient-1")
flair = nib.load(ms60_dir / "1-Flair.nii").get_fdata()
mask = nib.load(ms60_dir / "1-LesionSeg-Flair.nii").get_fdata()

print("="*80)
print("MS60 Dataset Analysis")
print("="*80)
print(f"FLAIR shape: {flair.shape}")
print(f"FLAIR dtype: {flair.dtype}")
print(f"FLAIR range: {flair.min():.2f} to {flair.max():.2f}")
print(f"FLAIR mean: {flair.mean():.2f}, std: {flair.std():.2f}")
print(f"FLAIR non-zero mean: {flair[flair > 0].mean():.2f}")
print()
print(f"Mask shape: {mask.shape}")
print(f"Mask dtype: {mask.dtype}")
print(f"Mask unique values: {np.unique(mask)}")
print(f"Mask lesion pixels: {(mask > 0).sum()}")
print(f"Mask lesion percentage: {(mask > 0).sum() / mask.size * 100:.3f}%")
print()

# Check a slice with lesions
lesion_slice = np.where(mask.sum(axis=(0,1)) > 0)[0][0]
slice_flair = flair[:, :, lesion_slice]
slice_mask = mask[:, :, lesion_slice]

print(f"Slice {lesion_slice} with lesions:")
print(f"  FLAIR in lesion region: {slice_flair[slice_mask > 0].mean():.2f} ± {slice_flair[slice_mask > 0].std():.2f}")
print(f"  FLAIR in non-lesion region: {slice_flair[slice_mask == 0].mean():.2f} ± {slice_flair[slice_mask == 0].std():.2f}")
print(f"  Contrast: {(slice_flair[slice_mask > 0].mean() / (slice_flair[slice_mask == 0].mean() + 1e-8)):.2f}x")
print()

# Check PediMS for comparison
print("="*80)
print("PediMS Dataset Analysis (for comparison)")
print("="*80)
pedims_dir = Path(r"G:\My Drive\Dataset\PediMS\PediMS")
if pedims_dir.exists():
    # Find a case
    cases = list(pedims_dir.glob("*/"))
    if cases:
        case = cases[0]
        flair_files = list(case.glob("*FLAIR*.nii.gz"))
        mask_files = list(case.glob("*Consensus*.nii.gz"))
        
        if flair_files and mask_files:
            pedims_flair = nib.load(flair_files[0]).get_fdata()
            pedims_mask = nib.load(mask_files[0]).get_fdata()
            
            print(f"FLAIR shape: {pedims_flair.shape}")
            print(f"FLAIR range: {pedims_flair.min():.2f} to {pedims_flair.max():.2f}")
            print(f"FLAIR mean: {pedims_flair.mean():.2f}, std: {pedims_flair.std():.2f}")
            print(f"FLAIR non-zero mean: {pedims_flair[pedims_flair > 0].mean():.2f}")
            print()
            print(f"Mask unique values: {np.unique(pedims_mask)}")
            print(f"Mask lesion percentage: {(pedims_mask > 0).sum() / pedims_mask.size * 100:.3f}%")
else:
    print("PediMS not accessible from local drive")

print()
print("="*80)
print("DIAGNOSIS:")
print("="*80)
print("Compare the intensity ranges and lesion characteristics above.")
print("Key factors for domain shift:")
print("  1. Different intensity ranges")
print("  2. Different lesion-to-background contrast")
print("  3. Different spatial resolution")
print("  4. Different scanner characteristics")
print("="*80)
