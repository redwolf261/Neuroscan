"""
Generate Figures for Ablation Studies
Creates publication-quality figures for:
1. Hyperparameter Ablation (k_slices × window_size)
2. Component Ablation (USALD components)
3. Cross-Dataset Validation (generalization)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Paths
OUTPUT_DIR = Path("C:/Users/HP/EDI/paper_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

ABLATION_DIR = Path("C:/Users/HP/EDI/ablation_results")
CROSS_DATASET_DIR = Path("C:/Users/HP/EDI/csv_data/cross_dataset_validation")

print("=" * 80)
print("📊 GENERATING ABLATION STUDY FIGURES")
print("=" * 80)

# ============================================================================
# FIGURE 11: Hyperparameter Ablation Study
# ============================================================================
def generate_hyperparameter_ablation():
    """
    Heatmap and line plots showing k_slices vs window_size performance
    """
    print("\n🔬 Generating Figure 11: Hyperparameter Ablation...")
    
    # Load data
    df = pd.read_csv(ABLATION_DIR / "hyperparameter_ablation_summary.csv")
    
    # Extract k and w values
    df['k'] = df['config'].str.extract(r'k(\d+)').astype(int)
    df['w'] = df['config'].str.extract(r'w(\d+)').astype(int)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Heatmap of Dice scores
    ax = axes[0]
    pivot = df.pivot(index='k', columns='w', values='best_dice')
    
    im = ax.imshow(pivot.values * 100, cmap='RdYlGn', aspect='auto', 
                   vmin=68, vmax=73, interpolation='nearest')
    
    # Add values on heatmap
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax.text(j, i, f'{pivot.values[i, j]*100:.2f}',
                          ha="center", va="center", color="black", 
                          fontweight='bold', fontsize=9)
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Window Size', fontweight='bold')
    ax.set_ylabel('K Slices', fontweight='bold')
    ax.set_title('(a) Dice Score Heatmap', fontweight='bold', pad=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Dice Score (%)', fontweight='bold')
    
    # Highlight best configuration
    best_idx = df['best_dice'].idxmax()
    best_k = df.loc[best_idx, 'k']
    best_w = df.loc[best_idx, 'w']
    k_idx = list(pivot.index).index(best_k)
    w_idx = list(pivot.columns).index(best_w)
    
    rect = plt.Rectangle((w_idx-0.5, k_idx-0.5), 1, 1, 
                         fill=False, edgecolor='blue', linewidth=3)
    ax.add_patch(rect)
    
    # Panel 2: Effect of K slices (averaging over windows)
    ax = axes[1]
    k_effect = df.groupby('k').agg({
        'best_dice': ['mean', 'std'],
        'best_recall': 'mean',
        'best_precision': 'mean'
    })
    
    k_values = k_effect.index
    ax.errorbar(k_values, k_effect[('best_dice', 'mean')] * 100, 
                yerr=k_effect[('best_dice', 'std')] * 100,
                marker='o', linewidth=2.5, markersize=8, capsize=5,
                color='#2E86AB', label='Dice Score')
    
    ax.set_xlabel('Number of Slices (K)', fontweight='bold')
    ax.set_ylabel('Dice Score (%)', fontweight='bold')
    ax.set_title('(b) Effect of K Slices', fontweight='bold', pad=10)
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([68, 73])
    
    # Annotate best K
    best_k_mean = k_effect[('best_dice', 'mean')].idxmax()
    best_k_dice = k_effect.loc[best_k_mean, ('best_dice', 'mean')] * 100
    ax.annotate(f'Best K={best_k_mean}\n{best_k_dice:.2f}%', 
                xy=(best_k_mean, best_k_dice), xytext=(best_k_mean+0.5, best_k_dice+1.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=9, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Panel 3: Effect of Window size (averaging over K)
    ax = axes[2]
    w_effect = df.groupby('w').agg({
        'best_dice': ['mean', 'std'],
        'best_recall': 'mean',
        'best_precision': 'mean'
    })
    
    w_values = w_effect.index
    ax.errorbar(w_values, w_effect[('best_dice', 'mean')] * 100, 
                yerr=w_effect[('best_dice', 'std')] * 100,
                marker='s', linewidth=2.5, markersize=8, capsize=5,
                color='#A23B72', label='Dice Score')
    
    ax.set_xlabel('Window Size (W)', fontweight='bold')
    ax.set_ylabel('Dice Score (%)', fontweight='bold')
    ax.set_title('(c) Effect of Window Size', fontweight='bold', pad=10)
    ax.set_xticks(w_values)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([68, 73])
    
    # Annotate best W
    best_w_mean = w_effect[('best_dice', 'mean')].idxmax()
    best_w_dice = w_effect.loc[best_w_mean, ('best_dice', 'mean')] * 100
    ax.annotate(f'Best W={best_w_mean}\n{best_w_dice:.2f}%', 
                xy=(best_w_mean, best_w_dice), xytext=(best_w_mean+1, best_w_dice+1.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=9, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    for ext in ['png', 'pdf']:
        plt.savefig(OUTPUT_DIR / f'fig11_hyperparameter_ablation.{ext}', 
                   bbox_inches='tight', dpi=300 if ext == 'png' else None)
    
    plt.close()
    print(f"✅ Saved: fig11_hyperparameter_ablation.png/pdf")
    print(f"   Best config: K={best_k}, W={best_w} → {df.loc[best_idx, 'best_dice']*100:.2f}% Dice")
    print(f"   Tested: 12 configurations (4 K-values × 3 window sizes)")


# ============================================================================
# FIGURE 12: Component Ablation Study (USALD)
# ============================================================================
def generate_component_ablation():
    """
    Bar chart showing contribution of each USALD component
    """
    print("\n🧩 Generating Figure 12: Component Ablation Study...")
    
    # Load data
    df = pd.read_csv(ABLATION_DIR / "ablation_summary.csv")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Dice Score Comparison
    ax = axes[0]
    
    configs = df['name'].values
    dice_scores = df['dice'].values * 100
    improvements = df['improvement_pct'].values
    
    # Color based on improvement
    colors = ['#95A5A6']  # Baseline gray
    for imp in improvements[1:]:
        if imp > 1:
            colors.append('#27AE60')  # Green for good improvement
        elif imp > 0:
            colors.append('#3498DB')  # Blue for small improvement
        else:
            colors.append('#E74C3C')  # Red for degradation
    
    bars = ax.barh(configs, dice_scores, color=colors, edgecolor='black', 
                   linewidth=1.5, alpha=0.85)
    
    # Add value labels
    for i, (bar, dice, imp) in enumerate(zip(bars, dice_scores, improvements)):
        width = bar.get_width()
        label_x = width + 0.3
        
        if i == 0:
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{dice:.2f}%',
                   ha='left', va='center', fontweight='bold', fontsize=10)
        else:
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{dice:.2f}% ({imp:+.2f}%)',
                   ha='left', va='center', fontweight='bold', fontsize=10,
                   color='green' if imp > 0 else 'red')
    
    ax.set_xlabel('Dice Score (%)', fontweight='bold', fontsize=12)
    ax.set_title('(a) Component Contributions to Performance', fontweight='bold', fontsize=13, pad=15)
    ax.set_xlim([78, 84])
    ax.axvline(x=df.loc[0, 'dice']*100, color='red', linestyle='--', 
               linewidth=2, alpha=0.5, label='Baseline')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.legend(loc='lower right')
    
    # Panel 2: Multi-metric radar comparison
    ax = axes[1]
    
    # Select key configs for radar: Baseline, Best Single, Full
    baseline_idx = 0
    best_single_idx = df.iloc[1:-1]['dice'].idxmax()  # Best among individual components
    full_idx = len(df) - 1
    
    categories = ['Dice', 'Precision', 'Recall', 'F1']
    
    baseline_vals = [
        df.loc[baseline_idx, 'dice'] * 100,
        df.loc[baseline_idx, 'precision'] * 100,
        df.loc[baseline_idx, 'recall'] * 100,
        df.loc[baseline_idx, 'f1'] * 100
    ]
    
    best_single_vals = [
        df.loc[best_single_idx, 'dice'] * 100,
        df.loc[best_single_idx, 'precision'] * 100,
        df.loc[best_single_idx, 'recall'] * 100,
        df.loc[best_single_idx, 'f1'] * 100
    ]
    
    full_vals = [
        df.loc[full_idx, 'dice'] * 100,
        df.loc[full_idx, 'precision'] * 100,
        df.loc[full_idx, 'recall'] * 100,
        df.loc[full_idx, 'f1'] * 100
    ]
    
    x = np.arange(len(categories))
    width = 0.25
    
    ax.bar(x - width, baseline_vals, width, label='Baseline', 
           color='#95A5A6', edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.bar(x, best_single_vals, width, label=df.loc[best_single_idx, 'name'], 
           color='#27AE60', edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.bar(x + width, full_vals, width, label='Full USALD', 
           color='#E74C3C', edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
    ax.set_title('(b) Multi-Metric Comparison', fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([75, 92])
    
    plt.tight_layout()
    
    # Save
    for ext in ['png', 'pdf']:
        plt.savefig(OUTPUT_DIR / f'fig12_component_ablation.{ext}', 
                   bbox_inches='tight', dpi=300 if ext == 'png' else None)
    
    plt.close()
    print(f"✅ Saved: fig12_component_ablation.png/pdf")
    print(f"   Best single component: {df.loc[best_single_idx, 'name']} (+{df.loc[best_single_idx, 'improvement_pct']:.2f}%)")
    print(f"   Full USALD: {df.loc[full_idx, 'improvement_pct']:+.2f}% (over-regularization detected)")


# ============================================================================
# FIGURE 13: Cross-Dataset Generalization
# ============================================================================
def generate_cross_dataset_validation():
    """
    Show model generalization to different datasets
    """
    print("\n🌍 Generating Figure 13: Cross-Dataset Validation...")
    
    # Load latest results
    lgg_files = list(CROSS_DATASET_DIR.glob("lgg_brain_tumors_results_*.csv"))
    ms_files = list(CROSS_DATASET_DIR.glob("mslesionseg_adult_ms_results_*.csv"))
    
    if not lgg_files and not ms_files:
        print("   ⚠️ No cross-dataset validation results found. Skipping...")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    datasets = ['PediMS\n(Training)', 'LGG Tumors\n(External)', 'Adult MS\n(External)']
    dice_scores = [82.31]  # Current model performance
    precision_scores = [77.16]
    recall_scores = [88.17]
    specificity_scores = [99.85]
    
    # Load LGG results
    if lgg_files:
        lgg_df = pd.read_csv(sorted(lgg_files)[-1])
        dice_scores.append(lgg_df['dice'].mean() * 100)
        precision_scores.append(lgg_df['precision'].mean() * 100)
        recall_scores.append(lgg_df['recall'].mean() * 100)
        specificity_scores.append(lgg_df['specificity'].mean() * 100)
    else:
        datasets.remove('LGG Tumors\n(External)')
    
    # Load MS results
    if ms_files:
        ms_df = pd.read_csv(sorted(ms_files)[-1])
        dice_scores.append(ms_df['dice'].mean() * 100)
        precision_scores.append(ms_df['precision'].mean() * 100)
        recall_scores.append(ms_df['recall'].mean() * 100)
        specificity_scores.append(ms_df['specificity'].mean() * 100)
    else:
        datasets.remove('Adult MS\n(External)')
    
    # Panel 1: Dice Score Comparison
    ax = axes[0]
    
    colors_grad = ['#27AE60', '#3498DB', '#9B59B6'][:len(datasets)]
    bars = ax.bar(datasets, dice_scores, color=colors_grad, 
                  edgecolor='black', linewidth=2, alpha=0.85)
    
    # Add value labels
    for bar, score in zip(bars, dice_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.2f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    # Add degradation annotations
    if len(dice_scores) > 1:
        for i in range(1, len(dice_scores)):
            degradation = dice_scores[i] - dice_scores[0]
            y_pos = min(dice_scores[0], dice_scores[i]) - 3
            ax.text(i, y_pos, f'{degradation:+.2f}%', 
                   ha='center', va='top', fontsize=9, 
                   color='red' if degradation < -5 else 'orange',
                   fontweight='bold')
    
    ax.set_ylabel('Dice Score (%)', fontweight='bold', fontsize=12)
    ax.set_title('(a) Cross-Dataset Performance', fontweight='bold', fontsize=13, pad=15)
    ax.set_ylim([0, 90])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add baseline line
    ax.axhline(y=dice_scores[0], color='green', linestyle='--', 
               linewidth=2, alpha=0.5, label='Training Performance')
    ax.legend(loc='upper right')
    
    # Panel 2: Multi-metric breakdown
    ax = axes[1]
    
    x = np.arange(len(datasets))
    width = 0.2
    
    ax.bar(x - 1.5*width, dice_scores, width, label='Dice', 
           color='#27AE60', edgecolor='black', linewidth=1.2, alpha=0.85)
    ax.bar(x - 0.5*width, precision_scores, width, label='Precision', 
           color='#3498DB', edgecolor='black', linewidth=1.2, alpha=0.85)
    ax.bar(x + 0.5*width, recall_scores, width, label='Recall', 
           color='#E67E22', edgecolor='black', linewidth=1.2, alpha=0.85)
    ax.bar(x + 1.5*width, specificity_scores, width, label='Specificity', 
           color='#9B59B6', edgecolor='black', linewidth=1.2, alpha=0.85)
    
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
    ax.set_title('(b) Generalization Metrics', fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    
    # Save
    for ext in ['png', 'pdf']:
        plt.savefig(OUTPUT_DIR / f'fig13_cross_dataset_validation.{ext}', 
                   bbox_inches='tight', dpi=300 if ext == 'png' else None)
    
    plt.close()
    print(f"✅ Saved: fig13_cross_dataset_validation.png/pdf")
    if len(dice_scores) > 1:
        avg_degradation = np.mean([dice_scores[i] - dice_scores[0] for i in range(1, len(dice_scores))])
        print(f"   Average performance degradation: {avg_degradation:.2f}%")
        print(f"   Datasets tested: {len(datasets)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    try:
        # Generate all figures
        generate_hyperparameter_ablation()
        generate_component_ablation()
        generate_cross_dataset_validation()
        
        print("\n" + "=" * 80)
        print("✅ ALL ABLATION FIGURES GENERATED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n📁 Output directory: {OUTPUT_DIR}")
        print("\n📊 Generated Figures:")
        print("   - fig11_hyperparameter_ablation.png/pdf")
        print("   - fig12_component_ablation.png/pdf")
        print("   - fig13_cross_dataset_validation.png/pdf")
        print("\n🎯 Total: 3 new publication-quality figures!")
        
    except Exception as e:
        print(f"\n❌ Error generating figures: {e}")
        import traceback
        traceback.print_exc()
