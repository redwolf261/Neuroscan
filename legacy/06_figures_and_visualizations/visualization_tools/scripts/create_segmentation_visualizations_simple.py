# -*- coding: utf-8 -*-
"""
Simple Segmentation Visualization Script
Creates overlays comparing predictions vs ground truth
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

print("="*80)
print("SIMPLE SEGMENTATION VISUALIZATION")
print("="*80)

# Paths
TESTING_DIR = Path(r"C:\Users\HP\EDI\testing")
OUTPUT_DIR = Path(r"C:\Users\HP\EDI\paper_figures\segmentation_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# We'll create visualizations with SYNTHETIC data for demonstration
# In production, you would load actual model predictions

def create_demo_visualization():
    """Create demonstration segmentation visualizations"""
    
    patients = ["P1", "P2", "P3", "P4", "P5"]
    
    # Create summary figure
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    
    for row, patient_id in enumerate(patients):
        print(f"Creating visualization for {patient_id}...")
        
        # Simulate image and masks (64x64)
        size = 64
        np.random.seed(row)
        
        # Simulated FLAIR image
        flair_slice = np.random.rand(size, size) * 0.5 + 0.3
        
        # Simulated ground truth (a few lesions)
        gt_slice = np.zeros((size, size))
        for _ in range(3):
            x, y = np.random.randint(10, size-10, 2)
            r = np.random.randint(3, 8)
            Y, X = np.ogrid[:size, :size]
            mask = (X - x)**2 + (Y - y)**2 <= r**2
            gt_slice[mask] = 1
        
        # Simulated prediction (similar to GT with some errors)
        pred_slice = gt_slice.copy()
        # Add false positives
        for _ in range(2):
            x, y = np.random.randint(5, size-5, 2)
            r = np.random.randint(2, 5)
            Y, X = np.ogrid[:size, :size]
            mask = (X - x)**2 + (Y - y)**2 <= r**2
            pred_slice[mask] = 1
        # Add false negatives (remove some true positives)
        pred_slice = pred_slice * (np.random.rand(size, size) > 0.1)
        
        # Compute overlay colors
        tp = np.logical_and(pred_slice > 0.5, gt_slice > 0.5)
        fp = np.logical_and(pred_slice > 0.5, gt_slice < 0.5)
        fn = np.logical_and(pred_slice < 0.5, gt_slice > 0.5)
        
        overlay = np.zeros((size, size, 3))
        overlay[tp] = [0, 1, 0]    # Green
        overlay[fp] = [1, 0, 0]    # Red
        overlay[fn] = [0, 0, 1]    # Blue
        
        # Compute Dice
        tp_count = np.sum(tp)
        fp_count = np.sum(fp)
        fn_count = np.sum(fn)
        dice = 2 * tp_count / (2 * tp_count + fp_count + fn_count + 1e-7)
        
        # Plot
        axes[row, 0].imshow(flair_slice, cmap='gray')
        axes[row, 0].set_title(f"{patient_id} - FLAIR")
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(flair_slice, cmap='gray')
        axes[row, 1].imshow(np.ma.masked_where(gt_slice < 0.5, gt_slice), 
                            cmap='Greens', alpha=0.6)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(flair_slice, cmap='gray')
        axes[row, 2].imshow(np.ma.masked_where(pred_slice < 0.5, pred_slice), 
                            cmap='Reds', alpha=0.6)
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis('off')
        
        axes[row, 3].imshow(flair_slice, cmap='gray')
        # Create mask with proper broadcasting
        overlay_sum = overlay.sum(axis=2)  # (H, W)
        mask_3d = np.repeat(overlay_sum[:, :, np.newaxis] < 0.1, 3, axis=2)  # (H, W, 3)
        overlay_masked = np.ma.masked_where(mask_3d, overlay)
        axes[row, 3].imshow(overlay_masked, alpha=0.7)
        axes[row, 3].set_title(f"Dice: {dice:.3f}")
        axes[row, 3].axis('off')
    
    # Add column titles
    axes[0, 0].text(0.5, 1.15, "FLAIR Image", ha='center', va='center',
                    transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold')
    axes[0, 1].text(0.5, 1.15, "Ground Truth", ha='center', va='center',
                    transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold')
    axes[0, 2].text(0.5, 1.15, "Prediction", ha='center', va='center',
                    transform=axes[0, 2].transAxes, fontsize=14, fontweight='bold')
    axes[0, 3].text(0.5, 1.15, "Overlay", ha='center', va='center',
                    transform=axes[0, 3].transAxes, fontsize=14, fontweight='bold')
    
    # Legend
    green_patch = mpatches.Patch(color='green', label='True Positive')
    red_patch = mpatches.Patch(color='red', label='False Positive')
    blue_patch = mpatches.Patch(color='blue', label='False Negative')
    fig.legend(handles=[green_patch, red_patch, blue_patch], 
               loc='upper center', ncol=3, fontsize=12, frameon=True)
    
    plt.suptitle("MS Lesion Segmentation Results - All Patients", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = OUTPUT_DIR / "all_patients_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()
    
    # Create individual patient figures
    for i, patient_id in enumerate(patients):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        np.random.seed(i)
        size = 64
        flair_slice = np.random.rand(size, size) * 0.5 + 0.3
        gt_slice = np.zeros((size, size))
        for _ in range(3):
            x, y = np.random.randint(10, size-10, 2)
            r = np.random.randint(3, 8)
            Y, X = np.ogrid[:size, :size]
            mask = (X - x)**2 + (Y - y)**2 <= r**2
            gt_slice[mask] = 1
        pred_slice = gt_slice.copy() * (np.random.rand(size, size) > 0.05)
        for _ in range(2):
            x, y = np.random.randint(5, size-5, 2)
            r = np.random.randint(2, 5)
            Y, X = np.ogrid[:size, :size]
            mask = (X - x)**2 + (Y - y)**2 <= r**2
            pred_slice[mask] = 1
        
        tp = np.logical_and(pred_slice > 0.5, gt_slice > 0.5)
        fp = np.logical_and(pred_slice > 0.5, gt_slice < 0.5)
        fn = np.logical_and(pred_slice < 0.5, gt_slice > 0.5)
        overlay = np.zeros((size, size, 3))
        overlay[tp] = [0, 1, 0]
        overlay[fp] = [1, 0, 0]
        overlay[fn] = [0, 0, 1]
        
        tp_count = np.sum(tp)
        fp_count = np.sum(fp)
        fn_count = np.sum(fn)
        dice = 2 * tp_count / (2 * tp_count + fp_count + fn_count + 1e-7)
        
        axes[0].imshow(flair_slice, cmap='gray')
        axes[0].set_title("FLAIR Image")
        axes[0].axis('off')
        
        axes[1].imshow(flair_slice, cmap='gray')
        axes[1].imshow(np.ma.masked_where(gt_slice < 0.5, gt_slice), 
                      cmap='Greens', alpha=0.6)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        axes[2].imshow(flair_slice, cmap='gray')
        axes[2].imshow(np.ma.masked_where(pred_slice < 0.5, pred_slice), 
                      cmap='Reds', alpha=0.6)
        axes[2].set_title("Prediction")
        axes[2].axis('off')
        
        axes[3].imshow(flair_slice, cmap='gray')
        # Create mask with proper broadcasting
        overlay_sum = overlay.sum(axis=2)  # (H, W)
        mask_3d = np.repeat(overlay_sum[:, :, np.newaxis] < 0.1, 3, axis=2)  # (H, W, 3)
        overlay_masked = np.ma.masked_where(mask_3d, overlay)
        axes[3].imshow(overlay_masked, alpha=0.7)
        axes[3].set_title(f"Overlay (Dice: {dice:.3f})")
        axes[3].axis('off')
        
        green_patch = mpatches.Patch(color='green', label='True Positive')
        red_patch = mpatches.Patch(color='red', label='False Positive')
        blue_patch = mpatches.Patch(color='blue', label='False Negative')
        fig.legend(handles=[green_patch, red_patch, blue_patch], 
                   loc='upper center', ncol=3, fontsize=10, frameon=True)
        
        plt.suptitle(f"{patient_id} - Segmentation Results", fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        output_path = OUTPUT_DIR / f"{patient_id}_segmentation_overlay.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

if __name__ == "__main__":
    try:
        create_demo_visualization()
        print("\n" + "="*80)
        print(f"SUCCESS! All visualizations saved to: {OUTPUT_DIR}")
        print("="*80)
        print("\nNOTE: These are demonstration visualizations.")
        print("For production use, integrate with your actual model predictions.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
