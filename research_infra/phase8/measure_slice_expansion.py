"""
Measures, for every one of the 28 real PediMS samples, how many of the 64
resampled D-axis slices actually contain a positive (lesion) voxel after
running through the EXACT same preprocessing pipeline final_model.py uses
(val_transforms: no random augmentation, so counts are deterministic).

This answers "how big could the dataset get" if training examples were
generated per-informative-slice instead of one fixed center slice per volume.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "01_source_code", "models"))
import final_model as fm  # noqa: E402

import torch
import numpy as np

print(f"\nTotal samples in data_dicts: {len(fm.data_dicts)}")
print(f"Train split: {len(fm.train_ds)}  Val split: {len(fm.val_ds)}\n")

results = []
for ds_name, ds in [("train", fm.train_ds), ("val", fm.val_ds)]:
    for i in range(len(ds)):
        sample = ds[i]
        label = sample["label"]  # (1, 64, 64, 64) after Resized, binarized
        if torch.is_tensor(label):
            label_np = label.numpy()
        else:
            label_np = np.asarray(label)
        label_np = label_np[0]  # drop channel -> (64,64,64)

        # D-axis (slice axis) is axis 0 of the (D,H,W) volume, matching how
        # AdaptiveSliceSelector/center_slice_label index into it.
        per_slice_positive = label_np.sum(axis=(1, 2)) > 0  # (64,) bool
        n_positive_slices = int(per_slice_positive.sum())
        total_voxels = int(label_np.sum())
        results.append({
            'split': ds_name, 'idx': i,
            'case': fm.data_dicts[(0 if ds_name == "train" else len(fm.train_ds)) + i].get('case', '?'),
            'n_positive_slices': n_positive_slices,
            'total_lesion_voxels': total_voxels,
            'center_slice_positive': bool(per_slice_positive[32]),
        })

print(f"{'split':6s} {'case':35s} {'pos_slices/64':>13s} {'lesion_voxels':>13s} {'center_pos':>10s}")
for r in results:
    print(f"{r['split']:6s} {r['case']:35s} {r['n_positive_slices']:>13d} {r['total_lesion_voxels']:>13d} {str(r['center_slice_positive']):>10s}")

total_pos_slices = sum(r['n_positive_slices'] for r in results)
volumes_with_any_lesion = sum(1 for r in results if r['n_positive_slices'] > 0)
volumes_with_center_lesion = sum(1 for r in results if r['center_slice_positive'])

print(f"\n=== SUMMARY ===")
print(f"Total volumes: {len(results)}")
print(f"Volumes with >=1 lesion-containing slice (out of 64): {volumes_with_any_lesion}")
print(f"Volumes where the FIXED center slice (idx 32) happens to contain lesion: {volumes_with_center_lesion}")
print(f"Total lesion-positive slices across all volumes (sum): {total_pos_slices}")
print(f"Mean lesion-positive slices per volume: {total_pos_slices / len(results):.1f}")
print(f"\nIf training used 1 example per lesion-positive slice instead of 1 fixed center slice per volume:")
print(f"  Effective positive-example count: {total_pos_slices} (vs {len(results)} today)")
print(f"  Multiplier: {total_pos_slices / len(results):.1f}x")
print(f"\nIf training also included, say, 2x as many negative (no-lesion) slices per volume for balance:")
print(f"  Effective total examples: ~{total_pos_slices + 2*total_pos_slices} "
      f"(~{(total_pos_slices + 2*total_pos_slices) / len(results):.1f}x today's count)")
