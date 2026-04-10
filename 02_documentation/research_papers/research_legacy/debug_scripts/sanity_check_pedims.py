"""
Sanity Check: Test model on PediMS validation set
=================================================
If this doesn't give ~84% Dice, the model is broken!
"""

import os
import sys
import torch
import numpy as np

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from final_model import HybridMiniSwin2D5_CSRF, val_loader

def dice_score(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    return (2. * intersection + smooth) / (union + smooth)

def test_pedims_validation():
    """Test if model still works on PediMS validation"""
    
    DEVICE = 'cuda'
    MODEL_PATH = r"C:\Users\HP\EDI\.resume_checkpoints\seg_resume.pth"
    
    print("="*80)
    print("SANITY CHECK: Testing model on PediMS validation set")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"Expected Dice: ~84%")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model = HybridMiniSwin2D5_CSRF(
        k_slices=5,
        channels=[32, 64, 128, 256, 512]
    ).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch')}")
    print(f"Checkpoint best dice: {checkpoint.get('best_val_dice')}")
    
    model.eval()
    
    # Test on PediMS validation
    print("\nTesting on PediMS validation set...")
    all_dice = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch['image'].to(DEVICE)
            masks = batch['label'].to(DEVICE)
            
            # Forward
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            
            # Dice
            for i in range(images.shape[0]):
                dice = dice_score(preds[i], masks[i])
                all_dice.append(dice.item())
                
                # Check if prediction looks reasonable
                pred_binary = (preds[i] > 0.5).float()
                pred_sum = pred_binary.sum().item()
                mask_sum = masks[i].sum().item()
                
                print(f"Sample {batch_idx * 3 + i + 1}:")
                print(f"  Dice: {dice.item():.4f}")
                print(f"  Predicted pixels: {pred_sum:.0f}")
                print(f"  True lesion pixels: {mask_sum:.0f}")
                print(f"  Prediction ratio: {pred_sum / mask_sum:.2f}x" if mask_sum > 0 else "  No lesions in mask")
    
    print("\n" + "="*80)
    print("PEDIMS VALIDATION RESULTS")
    print("="*80)
    print(f"Mean Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"Expected: ~0.8400")
    print("="*80)
    
    if np.mean(all_dice) > 0.70:
        print("\n✓ Model is working correctly on PediMS!")
        print("  The cross-dataset failure is due to domain shift, not model corruption.")
    else:
        print("\n✗ MODEL IS BROKEN!")
        print("  Model doesn't even work on PediMS validation.")
        print("  The checkpoint may be corrupted or from early training.")

if __name__ == "__main__":
    test_pedims_validation()
