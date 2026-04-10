import os
import nibabel as nib
import numpy as np
from scipy.ndimage import label

testing_dir = 'testing'
patients = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']

print('='*80)
print('GROUND TRUTH ANALYSIS - Expected Results')
print('='*80)
print(f"{'Patient':<10} {'Lesions':<10} {'Volume %':<15} {'Volume mL':<15} {'Severity':<15}")
print('-'*80)

for p in patients:
    gt_path = os.path.join(testing_dir, f'{p}_GroundTruth.nii')
    if not os.path.exists(gt_path):
        continue
    
    gt_mask = nib.load(gt_path).get_fdata()
    lesion_voxels = np.sum(gt_mask > 0)
    vol_pct = (lesion_voxels / gt_mask.size) * 100
    vol_ml = lesion_voxels * 0.001
    labeled_array, num_lesions = label(gt_mask > 0)
    
    if vol_pct < 0.3:
        severity = 'Minimal'
    elif vol_pct < 1.0:
        severity = 'Mild'
    elif vol_pct < 5.0:
        severity = 'Moderate'
    else:
        severity = 'Severe'
    
    print(f'{p:<10} {num_lesions:<10} {vol_pct:<15.4f} {vol_ml:<15.2f} {severity:<15}')

print('='*80)
print('\nAll 9 patients should show "MS DETECTED" (all have lesions)')
print('P2 should show the highest lesion load (most severe case)')
