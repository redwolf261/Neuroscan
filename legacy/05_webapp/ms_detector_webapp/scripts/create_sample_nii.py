"""
Create a sample NIfTI file for testing the MS Detector web app
This creates a synthetic brain-like 3D volume
"""

import numpy as np
import nibabel as nib
import os

print("Creating sample NIfTI file for testing...")

# Create a 3D volume (simulating a brain MRI)
# Size: 256 x 256 x 180 (typical brain MRI dimensions)
img_data = np.random.randint(0, 255, (256, 256, 180), dtype=np.uint8)

# Add some structure to make it look more like a brain scan
# Create a brain-like ellipsoid
x, y, z = np.ogrid[:256, :256, :180]
cx, cy, cz = 128, 128, 90  # Center
rx, ry, rz = 80, 80, 60    # Radii

# Create ellipsoid mask
mask = ((x - cx)**2 / rx**2 + (y - cy)**2 / ry**2 + (z - cz)**2 / rz**2) <= 1
img_data[mask] = np.random.randint(100, 200, np.sum(mask), dtype=np.uint8)

# Add some "lesion-like" bright spots (for testing detection)
num_lesions = 5
for i in range(num_lesions):
    lx = np.random.randint(80, 176)
    ly = np.random.randint(80, 176)
    lz = np.random.randint(40, 140)
    # Small bright sphere
    lesion_mask = ((x - lx)**2 + (y - ly)**2 + (z - lz)**2) <= 25
    img_data[lesion_mask] = 255

# Create NIfTI image
affine = np.eye(4)
nii_img = nib.Nifti1Image(img_data, affine)

# Save the file
output_dir = "test_samples"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "sample_brain_mri.nii.gz")

nib.save(nii_img, output_file)

print(f"✅ Sample NIfTI file created: {output_file}")
print(f"   Size: {img_data.shape}")
print(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
print(f"\n🎯 You can now upload this file to test your MS Detector app!")
print(f"   Upload it at: http://localhost:3000")
