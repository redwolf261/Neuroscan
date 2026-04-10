"""
Quick script to check what type of ISBI dataset this is
"""
import numpy as np
from PIL import Image
import os

dataset_path = r"C:\Users\HP\EDI\ISBI"

print("="*80)
print("CHECKING ISBI DATASET")
print("="*80)

# Check train-volume
train_vol_path = os.path.join(dataset_path, "train-volume.tif")
train_label_path = os.path.join(dataset_path, "train-labels.tif")
test_vol_path = os.path.join(dataset_path, "test-volume.tif")

print(f"\n1. Loading train-volume.tif...")
try:
    img = Image.open(train_vol_path)
    
    # Get info
    num_frames = 0
    try:
        while True:
            img.seek(num_frames)
            num_frames += 1
    except EOFError:
        pass
    
    img.seek(0)
    width, height = img.size
    
    print(f"   ✓ Loaded successfully")
    print(f"   - Number of slices: {num_frames}")
    print(f"   - Image size: {width} x {height}")
    print(f"   - Mode: {img.mode}")
    
    # Load first slice to check data
    first_slice = np.array(img)
    print(f"   - Data type: {first_slice.dtype}")
    print(f"   - Value range: [{first_slice.min()}, {first_slice.max()}]")
    print(f"   - Shape: {first_slice.shape}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")

print(f"\n2. Loading train-labels.tif...")
try:
    img = Image.open(train_label_path)
    
    # Get info
    num_frames = 0
    try:
        while True:
            img.seek(num_frames)
            num_frames += 1
    except EOFError:
        pass
    
    img.seek(0)
    width, height = img.size
    
    print(f"   ✓ Loaded successfully")
    print(f"   - Number of slices: {num_frames}")
    print(f"   - Image size: {width} x {height}")
    print(f"   - Mode: {img.mode}")
    
    # Load first slice
    first_slice = np.array(img)
    print(f"   - Data type: {first_slice.dtype}")
    print(f"   - Value range: [{first_slice.min()}, {first_slice.max()}]")
    print(f"   - Unique values: {len(np.unique(first_slice))}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)

# Typical ISBI 2012 EM: 512x512 pixels, 30 slices
# Typical MS MRI: 256x256 or 512x512, many more slices (>100)

if width == 512 and height == 512 and num_frames == 30:
    print("\n⚠️  THIS IS LIKELY ISBI 2012 EM SEGMENTATION CHALLENGE")
    print("    (Electron Microscopy - Neuronal Structures)")
    print("\n❌ NOT the ISBI 2015 MS Lesion Segmentation Challenge")
    print("\nWhat you need:")
    print("  - ISBI 2015 Longitudinal MS Lesion Segmentation")
    print("  - Files: FLAIR/T1/T2 NIfTI format (.nii or .nii.gz)")
    print("  - MS lesion ground truth masks")
    print("\nDataset info: https://smart-stats-tools.org/lesion-challenge-2015")
elif num_frames > 50:
    print("\n✓ This might be an MRI dataset (many slices)")
    print("  Further inspection needed...")
else:
    print("\n? Unknown dataset type")
    print(f"  Dimensions: {width}x{height}, {num_frames} slices")

print("\n" + "="*80)
