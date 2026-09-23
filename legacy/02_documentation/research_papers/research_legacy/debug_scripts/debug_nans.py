"""
Debug: Find where NaN is coming from
====================================
Check model layer by layer to find the source of NaN
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from final_model import HybridMiniSwin2D5_CSRF

def debug_model_nans():
    """Find which layer produces NaN"""
    
    MODEL_PATH = r"C:\Users\HP\EDI\.resume_checkpoints\seg_resume.pth"
    DATASET_ROOT = r"G:\My Drive\Dataset\MSLESSEG\processed"
    DEVICE = 'cuda'
    
    print("="*80)
    print("DEBUGGING NaN OUTPUTS")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model = HybridMiniSwin2D5_CSRF(k_slices=5, channels=[32, 64, 128, 256, 512]).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load sample
    print("Loading MSLESSEG sample...")
    images_dir = os.path.join(DATASET_ROOT, 'imagesTr')
    flair_data = nib.load(os.path.join(images_dir, 'case_001_flair.nii.gz')).get_fdata()
    
    H, W, D = flair_data.shape
    slice_idx = D // 2
    
    # Extract and normalize
    k_slices = []
    for offset in [-2, -1, 0, 1, 2]:
        idx = max(0, min(D-1, slice_idx + offset))
        slice_data = flair_data[:, :, idx].astype(np.float32)
        
        # Check input
        print(f"\nSlice {offset}:")
        print(f"  Shape: {slice_data.shape}")
        print(f"  Min: {slice_data.min():.2f}, Max: {slice_data.max():.2f}")
        print(f"  Mean: {slice_data.mean():.2f}, Std: {slice_data.std():.2f}")
        print(f"  Has NaN: {np.isnan(slice_data).any()}")
        print(f"  Has Inf: {np.isinf(slice_data).any()}")
        
        # Normalize
        mask = slice_data > 0
        if mask.sum() > 0:
            mean = slice_data[mask].mean()
            std = slice_data[mask].std()
            print(f"  Nonzero mean: {mean:.2f}, std: {std:.2f}")
            if std > 0:
                slice_data = (slice_data - mean) / std
            print(f"  After norm - Min: {slice_data.min():.2f}, Max: {slice_data.max():.2f}")
        
        k_slices.append(slice_data)
    
    # Create volume
    volume = np.stack(k_slices, axis=0)
    print(f"\nVolume shape: {volume.shape}")
    print(f"Volume - Min: {volume.min():.2f}, Max: {volume.max():.2f}")
    print(f"Has NaN: {np.isnan(volume).any()}")
    
    volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    
    # Resize
    volume_resized = F.interpolate(
        volume_tensor.view(1, 5, H, W),
        size=(64, 64),
        mode='bilinear',
        align_corners=False
    ).unsqueeze(1).to(DEVICE)
    
    print(f"\nResized volume shape: {volume_resized.shape}")
    print(f"Resized - Min: {volume_resized.min().item():.2f}, Max: {volume_resized.max().item():.2f}")
    print(f"Has NaN: {torch.isnan(volume_resized).any().item()}")
    print(f"Has Inf: {torch.isinf(volume_resized).any().item()}")
    
    # Test PediMS sample for comparison
    print("\n" + "="*80)
    print("TESTING ON PEDIMS SAMPLE (should work)")
    print("="*80)
    
    pedims_path = r"G:\My Drive\Dataset\PediMS\PediMS\train\sub-001\ses-01\anat\sub-001_ses-01_FLAIR.nii.gz"
    if os.path.exists(pedims_path):
        pedims_data = nib.load(pedims_path).get_fdata()
        print(f"PediMS shape: {pedims_data.shape}")
        
        H_p, W_p, D_p = pedims_data.shape
        slice_idx_p = D_p // 2
        
        k_slices_p = []
        for offset in [-2, -1, 0, 1, 2]:
            idx = max(0, min(D_p-1, slice_idx_p + offset))
            slice_data = pedims_data[:, :, idx].astype(np.float32)
            mask = slice_data > 0
            if mask.sum() > 0:
                mean = slice_data[mask].mean()
                std = slice_data[mask].std()
                if std > 0:
                    slice_data = (slice_data - mean) / std
            k_slices_p.append(slice_data)
        
        volume_p = np.stack(k_slices_p, axis=0)
        volume_tensor_p = torch.from_numpy(volume_p).float().unsqueeze(0).unsqueeze(0)
        
        volume_resized_p = F.interpolate(
            volume_tensor_p.view(1, 5, H_p, W_p),
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        ).unsqueeze(1).to(DEVICE)
        
        print(f"PediMS resized - Min: {volume_resized_p.min().item():.2f}, Max: {volume_resized_p.max().item():.2f}")
        
        with torch.no_grad():
            output_p = model(volume_resized_p)
            pred_p = torch.sigmoid(output_p)
        
        print(f"PediMS output - Min: {output_p.min().item():.2f}, Max: {output_p.max().item():.2f}")
        print(f"PediMS pred - Min: {pred_p.min().item():.2f}, Max: {pred_p.max().item():.2f}")
        print(f"PediMS Has NaN: {torch.isnan(pred_p).any().item()}")
    
    # Now test MSLESSEG with hooks
    print("\n" + "="*80)
    print("TESTING MSLESSEG WITH LAYER-BY-LAYER INSPECTION")
    print("="*80)
    
    activations = {}
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output
        return hook
    
    # Register hooks
    model.encoder.stem.register_forward_hook(get_activation('stem'))
    for i, stage in enumerate(model.encoder.stages):
        stage.register_forward_hook(get_activation(f'stage_{i}'))
    model.decoder.register_forward_hook(get_activation('decoder'))
    
    # Forward pass
    print("\nRunning forward pass on MSLESSEG...")
    with torch.no_grad():
        try:
            output = model(volume_resized)
            pred = torch.sigmoid(output)
            
            print("\nLayer outputs:")
            for name, act in activations.items():
                if torch.isnan(act).any():
                    print(f"  {name}: ❌ HAS NaN! Shape: {act.shape}")
                else:
                    print(f"  {name}: ✓ OK - Min: {act.min().item():.2f}, Max: {act.max().item():.2f}, Shape: {act.shape}")
            
            print(f"\nFinal output shape: {output.shape}")
            print(f"Output - Min: {output.min().item():.2f}, Max: {output.max().item():.2f}")
            print(f"Output Has NaN: {torch.isnan(output).any().item()}")
            print(f"Pred - Min: {pred.min().item():.2f}, Max: {pred.max().item():.2f}")
            print(f"Pred Has NaN: {torch.isnan(pred).any().item()}")
            
        except Exception as e:
            print(f"\nERROR during forward pass: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    debug_model_nans()
