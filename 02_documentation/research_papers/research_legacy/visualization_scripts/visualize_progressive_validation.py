"""
Visualize Progressive Cross-Dataset Validation Metrics
======================================================
Shows how metrics evolve as more samples are evaluated
Similar to training curves but for validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Configuration
RESULTS_DIR = r"C:\Users\HP\EDI\csv_data\cross_dataset_validation"
OUTPUT_DIR = r"C:\Users\HP\EDI\paper_figures"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def load_progressive_data():
    """Load the progressive validation CSV"""
    
    # Find latest progressive CSV
    csv_files = glob.glob(str(Path(RESULTS_DIR) / "progressive_validation_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No progressive validation CSV found. Run validation first!")
    
    latest_csv = max(csv_files, key=lambda x: Path(x).stat().st_mtime)
    
    # Load data
    df = pd.read_csv(latest_csv)
    
    print(f"Loaded progressive data from: {latest_csv}")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df, latest_csv


def create_progressive_metrics_figure(df):
    """
    Create figure showing how metrics evolve during validation
    Similar to training curves
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Progressive Cross-Dataset Validation Metrics\n(LGG Brain Tumor Dataset)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Running Average Dice Score with Confidence Interval
    ax = axes[0, 0]
    x = df['samples_processed']
    dice_avg = df['running_avg_dice']
    dice_std = df['running_std_dice']
    
    ax.plot(x, dice_avg, linewidth=2, color='#2E86AB', label='Running Average')
    ax.fill_between(x, dice_avg - dice_std, dice_avg + dice_std, 
                     alpha=0.3, color='#2E86AB', label='±1 Std Dev')
    
    # Add final value annotation
    final_dice = dice_avg.iloc[-1]
    final_std = dice_std.iloc[-1]
    ax.axhline(y=final_dice, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(0.02, 0.98, f'Final: {final_dice:.4f} ± {final_std:.4f}', 
           transform=ax.transAxes, fontsize=11, va='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('Samples Processed', fontsize=11, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=11, fontweight='bold')
    ax.set_title('(A) Progressive Dice Score', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    
    # 2. Running Average Precision & Recall
    ax = axes[0, 1]
    ax.plot(x, df['running_avg_precision'], linewidth=2, color='#FF6B6B', label='Precision')
    ax.plot(x, df['running_avg_recall'], linewidth=2, color='#4ECDC4', label='Recall')
    
    # Shade regions
    ax.fill_between(x, df['running_avg_precision'] - df['running_std_precision'],
                     df['running_avg_precision'] + df['running_std_precision'],
                     alpha=0.2, color='#FF6B6B')
    ax.fill_between(x, df['running_avg_recall'] - df['running_std_recall'],
                     df['running_avg_recall'] + df['running_std_recall'],
                     alpha=0.2, color='#4ECDC4')
    
    final_prec = df['running_avg_precision'].iloc[-1]
    final_rec = df['running_avg_recall'].iloc[-1]
    ax.text(0.02, 0.98, f'Precision: {final_prec:.4f}\nRecall: {final_rec:.4f}', 
           transform=ax.transAxes, fontsize=10, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax.set_xlabel('Samples Processed', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('(B) Progressive Precision & Recall', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    
    # 3. Running Average Specificity & IoU
    ax = axes[1, 0]
    ax.plot(x, df['running_avg_specificity'], linewidth=2, color='#9B59B6', label='Specificity')
    ax.plot(x, df['running_avg_iou'], linewidth=2, color='#F39C12', label='IoU (Jaccard)')
    
    ax.fill_between(x, df['running_avg_specificity'] - df['running_std_specificity'],
                     df['running_avg_specificity'] + df['running_std_specificity'],
                     alpha=0.2, color='#9B59B6')
    ax.fill_between(x, df['running_avg_iou'] - df['running_std_iou'],
                     df['running_avg_iou'] + df['running_std_iou'],
                     alpha=0.2, color='#F39C12')
    
    final_spec = df['running_avg_specificity'].iloc[-1]
    final_iou = df['running_avg_iou'].iloc[-1]
    ax.text(0.02, 0.98, f'Specificity: {final_spec:.4f}\nIoU: {final_iou:.4f}', 
           transform=ax.transAxes, fontsize=10, va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax.set_xlabel('Samples Processed', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('(C) Progressive Specificity & IoU', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    
    # 4. Standard Deviation Evolution (Stability)
    ax = axes[1, 1]
    ax.plot(x, df['running_std_dice'], linewidth=2, color='#2E86AB', label='Dice Std')
    ax.plot(x, df['running_std_precision'], linewidth=2, color='#FF6B6B', label='Precision Std')
    ax.plot(x, df['running_std_recall'], linewidth=2, color='#4ECDC4', label='Recall Std')
    
    ax.set_xlabel('Samples Processed', fontsize=11, fontweight='bold')
    ax.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
    ax.set_title('(D) Metric Stability (Lower is More Consistent)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    
    # 5. Per-Sample Dice Score (Raw Values)
    ax = axes[2, 0]
    
    # Scatter plot of individual samples
    ax.scatter(df['sample_idx'], df['dice'], alpha=0.3, s=10, color='gray', label='Individual Samples')
    
    # Overlay running average
    ax.plot(x, df['running_avg_dice'], linewidth=2, color='red', label='Running Average')
    
    # Calculate rolling window average (smoothed)
    window = min(50, len(df) // 10)
    if window > 1:
        rolling_avg = df['dice'].rolling(window=window, center=True).mean()
        ax.plot(df['sample_idx'], rolling_avg, linewidth=2, color='blue', 
               linestyle='--', label=f'Smoothed (window={window})')
    
    ax.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=11, fontweight='bold')
    ax.set_title('(E) Per-Sample Dice Scores', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    
    # 6. Convergence Analysis
    ax = axes[2, 1]
    
    # Calculate change in running average (convergence indicator)
    dice_change = df['running_avg_dice'].diff().abs()
    
    ax.plot(x[1:], dice_change[1:], linewidth=1.5, color='#E74C3C', label='Absolute Change')
    
    # Add rolling average of change
    if len(dice_change) > 10:
        change_smooth = dice_change.rolling(window=50, center=True).mean()
        ax.plot(x, change_smooth, linewidth=2, color='#3498DB', label='Smoothed Change')
    
    ax.axhline(y=0.001, color='green', linestyle='--', linewidth=1, 
              label='Convergence Threshold (0.001)')
    
    # Check when converged
    if len(dice_change) > 100:
        converged_idx = None
        for i in range(100, len(dice_change)):
            if dice_change[i:i+50].mean() < 0.001:
                converged_idx = i
                break
        
        if converged_idx:
            ax.axvline(x=converged_idx, color='orange', linestyle=':', linewidth=2, alpha=0.5)
            ax.text(converged_idx, ax.get_ylim()[1] * 0.9, 
                   f'Converged\n~{converged_idx} samples',
                   ha='left', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    
    ax.set_xlabel('Samples Processed', fontsize=11, fontweight='bold')
    ax.set_ylabel('Change in Running Avg Dice', fontsize=11, fontweight='bold')
    ax.set_title('(F) Convergence Analysis', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(OUTPUT_DIR) / 'progressive_validation_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    plt.show()


def create_metrics_heatmap(df):
    """Create heatmap showing correlation between metrics over time"""
    
    # Sample data for heatmap (every 50th sample to reduce size)
    step = max(1, len(df) // 100)
    df_sampled = df.iloc[::step].copy()
    
    # Prepare correlation data
    metrics_cols = ['dice', 'precision', 'recall', 'specificity', 'iou']
    corr_matrix = df[metrics_cols].corr()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Correlation Heatmap
    ax = axes[0]
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0.5, vmin=0, vmax=1, ax=ax, 
                cbar_kws={'label': 'Correlation'})
    ax.set_title('(A) Metric Correlation Matrix', fontsize=12, fontweight='bold')
    
    # 2. Metrics Evolution Heatmap
    ax = axes[1]
    
    # Normalize metrics to 0-1 for heatmap
    metrics_normalized = df_sampled[metrics_cols].values.T
    
    im = ax.imshow(metrics_normalized, aspect='auto', cmap='RdYlGn', 
                   interpolation='bilinear', vmin=0, vmax=1)
    
    ax.set_yticks(range(len(metrics_cols)))
    ax.set_yticklabels(metrics_cols)
    ax.set_xlabel('Sample Progress', fontsize=11, fontweight='bold')
    ax.set_title('(B) Metrics Evolution Over Validation', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Score')
    
    plt.suptitle('Cross-Dataset Validation: Metric Analysis', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = Path(OUTPUT_DIR) / 'validation_metrics_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.show()


def create_summary_statistics_table(df):
    """Create summary statistics table"""
    
    metrics = ['dice', 'precision', 'recall', 'specificity', 'iou']
    
    summary_stats = []
    for metric in metrics:
        stats = {
            'Metric': metric.capitalize(),
            'Mean': f"{df[metric].mean():.4f}",
            'Std': f"{df[metric].std():.4f}",
            'Min': f"{df[metric].min():.4f}",
            'Q25': f"{df[metric].quantile(0.25):.4f}",
            'Median': f"{df[metric].median():.4f}",
            'Q75': f"{df[metric].quantile(0.75):.4f}",
            'Max': f"{df[metric].max():.4f}",
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save to CSV
    output_path = Path(RESULTS_DIR) / 'summary_statistics_table.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\nSummary statistics saved to: {output_path}")
    
    # Print table
    print("\n" + "="*80)
    print("SUMMARY STATISTICS TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    return summary_df


def analyze_patient_progression(df):
    """Analyze how performance varies across patients"""
    
    # Group by patient and calculate stats
    patient_groups = df.groupby('patient_id').agg({
        'dice': ['mean', 'std', 'count', 'min', 'max'],
        'tumor_pixels': 'mean'
    }).reset_index()
    
    patient_groups.columns = ['patient_id', 'dice_mean', 'dice_std', 'n_slices', 
                              'dice_min', 'dice_max', 'avg_tumor_size']
    
    # Sort by dice mean
    patient_groups = patient_groups.sort_values('dice_mean', ascending=False)
    
    print("\n" + "="*80)
    print("PER-PATIENT PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Total unique patients: {len(patient_groups)}")
    print(f"\nTop 5 Patients:")
    print(patient_groups.head(5).to_string(index=False))
    print(f"\nBottom 5 Patients:")
    print(patient_groups.tail(5).to_string(index=False))
    print("="*80)
    
    # Save patient summary
    output_path = Path(RESULTS_DIR) / 'per_patient_summary.csv'
    patient_groups.to_csv(output_path, index=False)
    print(f"\nPer-patient summary saved to: {output_path}")
    
    return patient_groups


def main():
    """Main execution"""
    print("="*80)
    print("VISUALIZING PROGRESSIVE CROSS-DATASET VALIDATION")
    print("="*80)
    
    try:
        # Load progressive data
        df, csv_path = load_progressive_data()
        
        print(f"\n{'='*80}")
        print("FINAL METRICS")
        print("="*80)
        print(f"Dice Score:   {df['running_avg_dice'].iloc[-1]:.4f} ± {df['running_std_dice'].iloc[-1]:.4f}")
        print(f"Precision:    {df['running_avg_precision'].iloc[-1]:.4f} ± {df['running_std_precision'].iloc[-1]:.4f}")
        print(f"Recall:       {df['running_avg_recall'].iloc[-1]:.4f} ± {df['running_std_recall'].iloc[-1]:.4f}")
        print(f"Specificity:  {df['running_avg_specificity'].iloc[-1]:.4f} ± {df['running_std_specificity'].iloc[-1]:.4f}")
        print(f"IoU:          {df['running_avg_iou'].iloc[-1]:.4f} ± {df['running_std_iou'].iloc[-1]:.4f}")
        print(f"Total Samples: {len(df)}")
        print("="*80)
        
        # Create visualizations
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)
        
        print("\n1. Creating progressive metrics figure...")
        create_progressive_metrics_figure(df)
        
        print("\n2. Creating metrics heatmap...")
        create_metrics_heatmap(df)
        
        print("\n3. Creating summary statistics table...")
        summary_df = create_summary_statistics_table(df)
        
        print("\n4. Analyzing per-patient progression...")
        patient_summary = analyze_patient_progression(df)
        
        print("\n" + "="*80)
        print("ALL VISUALIZATIONS COMPLETED!")
        print("="*80)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nGenerated files:")
        print("  - progressive_validation_metrics.png/pdf (6-panel progressive analysis)")
        print("  - validation_metrics_heatmap.png/pdf (correlation & evolution)")
        print("  - summary_statistics_table.csv (detailed stats)")
        print("  - per_patient_summary.csv (patient-level analysis)")
        print("\nThese files track metrics exactly like training logs!")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please run cross_dataset_validation_lgg.py first!")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
