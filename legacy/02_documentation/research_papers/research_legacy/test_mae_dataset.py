"""
Quick test script to verify PediMS dataset loading for MAE ablation
"""

import sys
sys.path.insert(0, r"C:\Users\HP\EDI\research")

from mae_ablation import MAEConfig, create_data_loaders

def test_dataset_loading():
    """Test that the dataset loads correctly"""
    print("\n" + "="*80)
    print("TESTING PEDIMS DATASET LOADING")
    print("="*80 + "\n")
    
    config = MAEConfig()
    
    # Test MAE data loaders (images only)
    print("Testing MAE data loaders (images only)...")
    try:
        mae_train_loader, mae_val_loader = create_data_loaders(config, for_mae=True)
        
        # Test first batch
        batch = next(iter(mae_train_loader))
        images = batch["image"]
        
        print(f"✓ MAE batch loaded successfully")
        print(f"  Shape: {images.shape}")
        print(f"  Dtype: {images.dtype}")
        print(f"  Device: {images.device}")
        print(f"  Min/Max: {images.min():.4f} / {images.max():.4f}")
        print(f"  Expected: (B, k, H, W) = (B, 5, 181, 217)")
        
        assert images.shape[1] == 5, f"Expected 5 slices, got {images.shape[1]}"
        assert images.shape[2] == 181, f"Expected H=181, got {images.shape[2]}"
        assert images.shape[3] == 217, f"Expected W=217, got {images.shape[3]}"
        assert images.dtype == torch.float32, f"Expected float32, got {images.dtype}"
        
        print("\n✓ MAE data loader test PASSED!\n")
        
    except Exception as e:
        print(f"\n❌ MAE data loader test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    # Test segmentation data loaders (images + labels)
    print("Testing segmentation data loaders (images + labels)...")
    try:
        seg_train_loader, seg_val_loader = create_data_loaders(config, for_mae=False)
        
        # Test first batch
        batch = next(iter(seg_train_loader))
        images = batch["image"]
        labels = batch["label"]
        
        print(f"✓ Segmentation batch loaded successfully")
        print(f"  Image shape: {images.shape}")
        print(f"  Label shape: {labels.shape}")
        print(f"  Image dtype: {images.dtype}")
        print(f"  Label dtype: {labels.dtype}")
        print(f"  Image min/max: {images.min():.4f} / {images.max():.4f}")
        print(f"  Label min/max: {labels.min():.4f} / {labels.max():.4f}")
        
        assert images.shape == labels.shape, f"Shape mismatch: {images.shape} vs {labels.shape}"
        assert labels.dtype == torch.float32, f"Expected float32 labels, got {labels.dtype}"
        
        print("\n✓ Segmentation data loader test PASSED!\n")
        
    except Exception as e:
        print(f"\n❌ Segmentation data loader test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    print("="*80)
    print("✓ ALL DATASET TESTS PASSED!")
    print("="*80 + "\n")
    return True


if __name__ == "__main__":
    import torch
    success = test_dataset_loading()
    sys.exit(0 if success else 1)
