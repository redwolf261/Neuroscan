"""
Quick check of what predictions look like from the running validation
"""

import pandas as pd
from pathlib import Path
import numpy as np

# Check the progressive CSV being written
csv_dir = Path(r"C:\Users\HP\EDI\csv_data\cross_dataset_validation_mslesseg_FIXED_NO_DOUBLE_SIGMOID")

# Find the CSV file
csv_files = list(csv_dir.glob("progressive_validation_*.csv"))
if csv_files:
    latest = max(csv_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest)
    
    print("="*80)
    print(f"CURRENT VALIDATION PROGRESS")
    print("="*80)
    print(f"Slices processed: {len(df)}/1551")
    print(f"\nCurrent metrics:")
    print(f"  Dice:        {df['dice'].mean():.4f} ± {df['dice'].std():.4f}")
    print(f"  Precision:   {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
    print(f"  Recall:      {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")
    print(f"  Specificity: {df['specificity'].mean():.4f} ± {df['specificity'].std():.4f}")
    
    print(f"\nDistribution:")
    print(f"  Zero Dice slices: {(df['dice'] == 0).sum()}/{len(df)} ({(df['dice'] == 0).mean():.1%})")
    print(f"  Very low (<1%):   {(df['dice'] < 0.01).sum()}/{len(df)} ({(df['dice'] < 0.01).mean():.1%})")
    print(f"  Low (1-10%):      {((df['dice'] >= 0.01) & (df['dice'] < 0.10)).sum()}")
    print(f"  Moderate (10-30%):{((df['dice'] >= 0.10) & (df['dice'] < 0.30)).sum()}")
    print(f"  Good (30%+):      {(df['dice'] >= 0.30).sum()}")
    
    print(f"\nPrediction stats:")
    print(f"  Avg lesion pixels (gt):   {df['lesion_pixels'].mean():.1f}")
    print(f"  Avg pred pixels:          {df['pred_pixels'].mean():.1f}")
    print(f"  Ratio (pred/gt):          {df['pred_pixels'].mean() / df['lesion_pixels'].mean():.2f}x")
    
    # Check if we're still over-predicting
    print(f"\nPrediction behavior:")
    over_pred = (df['pred_pixels'] > df['lesion_pixels'] * 2).sum()
    under_pred = (df['pred_pixels'] < df['lesion_pixels'] * 0.5).sum()
    print(f"  Over-predicting (>2x):    {over_pred}/{len(df)} ({over_pred/len(df):.1%})")
    print(f"  Under-predicting (<0.5x): {under_pred}/{len(df)} ({under_pred/len(df):.1%})")
    
    print(f"\nSample of worst cases:")
    worst = df.nsmallest(10, 'dice')[['case', 'slice_idx', 'dice', 'precision', 'recall', 'pred_pixels', 'lesion_pixels']]
    print(worst.to_string(index=False))
    
    print("\n" + "="*80)
else:
    print("No CSV file found yet - validation may be initializing")
