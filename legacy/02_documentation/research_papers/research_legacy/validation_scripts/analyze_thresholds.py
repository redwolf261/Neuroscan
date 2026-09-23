"""
Threshold Analysis - Find optimal threshold for MSLESSEG
=========================================================
Test a wide range of thresholds to see if we're just using wrong threshold
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from final_model import HybridMiniSwin2D5_CSRF

def analyze_thresholds():
    """Test thresholds from 0.01 to 0.99"""
    
    MODEL_PATH = r"C:\Users\HP\EDI\.resume_checkpoints\seg_resume.pth"
    DATASET_ROOT = r"G:\My Drive\Dataset\MSLESSEG\processed"
    DEVICE = 'cuda'
    
    print("="*80)
    print("THRESHOLD ANALYSIS - Finding Optimal Threshold")
    print("="*80)
    
    # Load model
    model = HybridMiniSwin2D5_CSRF(k_slices=5, channels=[32, 64, 128, 256, 512]).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load first case  
    images_dir = os.path.join(DATASET_ROOT, 'imagesTr')
    labels_dir = os.path.join(DATASET_ROOT, 'labelsTr')
    
    flair_data = nib.load(os.path.join(images_dir, 'case_001_flair.nii.gz')).get_fdata()
    label_data = nib.load(os.path.join(labels_dir, 'case_001_mask.nii.gz')).get_fdata()
    
    H, W, D = flair_data.shape
    
    # Collect predictions
    print("\nCollecting predictions...")
    all_preds = []
    all_labels = []
    
    for slice_idx in tqdm(range(10, min(50, D-10))):  # 40 slices
        label_slice = label_data[:, :, slice_idx]
        if np.sum(label_slice) < 10:
            continue
        
        # Extract 5 slices
        k_slices = []
        for offset in [-2, -1, 0, 1, 2]:
            idx = max(0, min(D-1, slice_idx + offset))
            slice_data = flair_data[:, :, idx].astype(np.float32)
            mask = slice_data > 0
            if mask.sum() > 0:
                mean = slice_data[mask].mean()
                std = slice_data[mask].std()
                if std > 0:
                    slice_data = (slice_data - mean) / std
            k_slices.append(slice_data)
        
        volume = np.stack(k_slices, axis=0)
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        
        volume_resized = F.interpolate(
            volume_tensor.view(1, 5, H, W),
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        ).unsqueeze(1).to(DEVICE)
        
        label_tensor = torch.from_numpy(label_slice).float().unsqueeze(0).unsqueeze(0)
        label_resized = F.interpolate(label_tensor, size=(64, 64), mode='nearest').squeeze()
        label_binary = (label_resized > 0).float()
        
        # Predict
        with torch.no_grad():
            pred = model(volume_resized)
            pred = torch.sigmoid(pred).squeeze()
        
        all_preds.append(pred.cpu())
        all_labels.append(label_binary)
    
    # Test different thresholds
    print("\nTesting thresholds...")
    thresholds = np.arange(0.01, 1.0, 0.01)
    results = []
    
    for thresh in tqdm(thresholds):
        dice_scores = []
        precision_scores = []
        recall_scores = []
        
        for pred, label in zip(all_preds, all_labels):
            pred_binary = (pred > thresh).float()
            
            tp = (pred_binary * label).sum()
            fp = (pred_binary * (1 - label)).sum()
            fn = ((1 - pred_binary) * label).sum()
            
            intersection = (pred_binary * label).sum()
            union = pred_binary.sum() + label.sum()
            
            dice = (2.0 * intersection / (union + 1e-5)).item() if union > 0 else 0
            precision = (tp / (tp + fp + 1e-5)).item()
            recall = (tp / (tp + fn + 1e-5)).item()
            
            dice_scores.append(dice)
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        results.append({
            'threshold': thresh,
            'dice': np.mean(dice_scores),
            'precision': np.mean(precision_scores),
            'recall': np.mean(recall_scores)
        })
    
    # Find best
    best_result = max(results, key=lambda x: x['dice'])
    
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS RESULTS")
    print("="*80)
    print(f"\nDefault threshold (0.5):")
    default = [r for r in results if abs(r['threshold'] - 0.5) < 0.01][0]
    print(f"  Dice:      {default['dice']:.4f}")
    print(f"  Precision: {default['precision']:.4f}")
    print(f"  Recall:    {default['recall']:.4f}")
    
    print(f"\nBEST threshold ({best_result['threshold']:.2f}):")
    print(f"  Dice:      {best_result['dice']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall:    {best_result['recall']:.4f}")
    
    improvement = (best_result['dice'] - default['dice']) * 100
    print(f"\nImprovement: +{improvement:.2f} percentage points")
    
    # Show top 10 thresholds
    print("\nTop 10 thresholds:")
    sorted_results = sorted(results, key=lambda x: x['dice'], reverse=True)[:10]
    for i, r in enumerate(sorted_results, 1):
        print(f"  {i}. Thresh={r['threshold']:.2f}: Dice={r['dice']:.4f}, Prec={r['precision']:.4f}, Recall={r['recall']:.4f}")
    
    print("="*80)
    
    # Check prediction distribution
    print("\nPrediction value distribution:")
    all_pred_values = torch.cat([p.flatten() for p in all_preds])
    print(f"  Min:    {all_pred_values.min():.6f}")
    print(f"  Max:    {all_pred_values.max():.6f}")
    print(f"  Mean:   {all_pred_values.mean():.6f}")
    print(f"  Median: {all_pred_values.median():.6f}")
    print(f"  Std:    {all_pred_values.std():.6f}")
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\n  Percentiles:")
    for p in percentiles:
        val = torch.quantile(all_pred_values, p/100.0)
        print(f"    {p:2d}%: {val:.6f}")
    
    print("="*80)

if __name__ == "__main__":
    analyze_thresholds()
