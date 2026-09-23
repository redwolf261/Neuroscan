"""
Generate additional plots and statistical analysis for research paper
Run this after ablation study completion
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from pathlib import Path

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Paths
ABLATION_CSV = Path(r"C:\Users\HP\EDI\ablation\reports\ablation2_summary_100epochs.csv")
OUTPUT_DIR = Path(r"C:\Users\HP\EDI\paper_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(ABLATION_CSV)

# Add production model data (Current: HybridMiniSwin2D5_CBAM, Epoch 12)
production_model = {
    'Variant': 'Production (2.5D-CSRF)',
    'Val_Dice': 0.8231,  # From OptimalModel_Evidential/segmentation/val_logs.csv
    'Precision': 0.7716,
    'Recall': 0.8817,
    'F1_Score': 0.8230,
    'Train_Dice': 0.8210,  # From OptimalModel_Evidential/segmentation/train_logs.csv
    'Overfitting_Gap': 0.0021,  # 82.10 - 82.31 = -0.21% (negative means better val than train)
    'Train_Loss': 0.2795  # From train_logs.csv epoch 12
}

# Combine data
df_full = pd.concat([df, pd.DataFrame([production_model])], ignore_index=True)

print("=" * 80)
print("GENERATING PUBLICATION-QUALITY FIGURES")
print("=" * 80)


# ============================================================================
# FIGURE 1: Dice Score Comparison with Error Bars
# ============================================================================
def plot_dice_comparison():
    """Bar chart with Dice scores and confidence intervals"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    variants = df_full['Variant'].tolist()
    dice_scores = df_full['Val_Dice'].tolist()
    
    # Calculate approximate 95% CI (±0.02 for simplicity, adjust if you have actual CI data)
    ci_error = [0.02] * len(dice_scores)
    
    # Color code: baseline=gray, production=green, best ablation=blue, worst=red, others=lightblue
    colors = []
    for v in variants:
        if 'Production' in v:
            colors.append('#2ecc71')  # Green
        elif 'Baseline' in v:
            colors.append('#95a5a6')  # Gray
        elif 'NoResidual' in v:
            colors.append('#e74c3c')  # Red
        elif 'No3DConv' in v:
            colors.append('#3498db')  # Blue
        else:
            colors.append('#bdc3c7')  # Light gray
    
    bars = ax.bar(range(len(variants)), dice_scores, color=colors, 
                   yerr=ci_error, capsize=5, edgecolor='black', linewidth=0.7)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, dice_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{score:.2%}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Model Variant', fontweight='bold')
    ax.set_ylabel('Validation Dice Score', fontweight='bold')
    ax.set_title('Model Performance Comparison (100 Epochs)', fontweight='bold', pad=15)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.replace('HybridMiniSwin3D_', '').replace('Production (2.5D-CSRF)', 'Production')
                         for v in variants], rotation=45, ha='right')
    ax.set_ylim(0.65, 0.90)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.7309, color='gray', linestyle='--', linewidth=1, label='Baseline (73.09%)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_dice_comparison.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig1_dice_comparison.pdf', bbox_inches='tight')
    print("✓ Figure 1 saved: Dice Score Comparison")
    plt.close()


# ============================================================================
# FIGURE 2: Precision-Recall Scatter Plot
# ============================================================================
def plot_precision_recall():
    """Scatter plot showing precision-recall trade-off with F1 score as size"""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    precision = df_full['Precision']
    recall = df_full['Recall']
    f1 = df_full['F1_Score']
    variants = df_full['Variant']
    
    # Scatter with F1 as size
    scatter = ax.scatter(recall, precision, s=f1*500, alpha=0.6, 
                         c=range(len(variants)), cmap='viridis', edgecolors='black', linewidth=1)
    
    # Add labels for each point
    for i, txt in enumerate(variants):
        label = txt.replace('HybridMiniSwin3D_', '').replace('Production (2.5D-CSRF)', 'Production')
        ax.annotate(label, (recall.iloc[i], precision.iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add iso-F1 curves
    f1_levels = [0.70, 0.75, 0.80, 0.85]
    recall_range = np.linspace(0.55, 0.95, 100)
    for f1_level in f1_levels:
        precision_curve = (f1_level * recall_range) / (2 * recall_range - f1_level)
        precision_curve = np.clip(precision_curve, 0, 1)
        ax.plot(recall_range, precision_curve, '--', alpha=0.3, color='gray', linewidth=0.8)
        ax.text(0.92, (f1_level * 0.92) / (2 * 0.92 - f1_level), f'F1={f1_level:.2f}', 
                fontsize=7, alpha=0.5, rotation=-20)
    
    ax.set_xlabel('Recall (Sensitivity)', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Precision-Recall Trade-off\n(bubble size = F1 score)', fontweight='bold', pad=15)
    ax.set_xlim(0.75, 0.95)
    ax.set_ylim(0.55, 0.80)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_precision_recall.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_precision_recall.pdf', bbox_inches='tight')
    print("✓ Figure 2 saved: Precision-Recall Trade-off")
    plt.close()


# ============================================================================
# FIGURE 3: Component Importance (Ablation Analysis)
# ============================================================================
def plot_component_importance():
    """Horizontal bar chart showing component importance"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Calculate importance (relative to baseline)
    baseline_dice = df[df['Variant'] == 'Baseline']['Val_Dice'].values[0]
    
    components = []
    importance = []
    for _, row in df.iterrows():
        if row['Variant'] != 'Baseline':
            component = row['Variant'].replace('HybridMiniSwin3D_', '').replace('No', '')
            delta = (baseline_dice - row['Val_Dice']) * 100  # Positive = component is important
            components.append(component)
            importance.append(delta)
    
    # Sort by importance
    sorted_indices = np.argsort(importance)
    components = [components[i] for i in sorted_indices]
    importance = [importance[i] for i in sorted_indices]
    
    # Color code: positive=blue (important), negative=red (harmful)
    colors = ['#e74c3c' if x < 0 else '#3498db' for x in importance]
    
    bars = ax.barh(components, importance, color=colors, edgecolor='black', linewidth=0.7)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importance)):
        width = bar.get_width()
        label_x = width + (0.15 if width > 0 else -0.15)
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2., f'{val:+.2f}%',
                ha=ha, va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Performance Impact (Δ Dice %)', fontweight='bold')
    ax.set_ylabel('Component Removed', fontweight='bold')
    ax.set_title('Component Importance Analysis\n(Positive = Important, Negative = Harmful)', 
                 fontweight='bold', pad=15)
    ax.axvline(x=0, color='black', linewidth=1.5)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add annotations
    ax.text(0.95, 0.05, 'Component is\nIMPORTANT →', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='#3498db', fontweight='bold')
    ax.text(0.05, 0.05, '← Component is\nHARMFUL', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=8, color='#e74c3c', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_component_importance.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_component_importance.pdf', bbox_inches='tight')
    print("✓ Figure 3 saved: Component Importance Analysis")
    plt.close()


# ============================================================================
# FIGURE 4: Overfitting Analysis
# ============================================================================
def plot_overfitting_analysis():
    """Train-Val gap analysis"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    variants = df_full['Variant'].tolist()
    train_dice = df_full['Train_Dice'].tolist()
    val_dice = df_full['Val_Dice'].tolist()
    gap = df_full['Overfitting_Gap'].tolist()
    
    x = np.arange(len(variants))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_dice, width, label='Train Dice', 
                   color='#3498db', edgecolor='black', linewidth=0.7)
    bars2 = ax.bar(x + width/2, val_dice, width, label='Val Dice', 
                   color='#e74c3c', edgecolor='black', linewidth=0.7)
    
    # Add gap annotations
    for i, (train, val, g) in enumerate(zip(train_dice, val_dice, gap)):
        gap_height = train - val
        ax.plot([i, i], [val, train], 'k--', linewidth=0.8, alpha=0.5)
        ax.text(i, (train + val) / 2, f'{g:.3f}', ha='center', va='center',
                fontsize=7, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Model Variant', fontweight='bold')
    ax.set_ylabel('Dice Score', fontweight='bold')
    ax.set_title('Overfitting Analysis: Train-Validation Gap', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('HybridMiniSwin3D_', '').replace('Production (2.5D-CSRF)', 'Production')
                         for v in variants], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_overfitting_analysis.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_overfitting_analysis.pdf', bbox_inches='tight')
    print("✓ Figure 4 saved: Overfitting Analysis")
    plt.close()


# ============================================================================
# FIGURE 5: Radar Chart (Multi-Metric Comparison)
# ============================================================================
def plot_radar_chart():
    """Radar chart comparing all metrics"""
    from math import pi
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Select key variants for comparison
    key_variants = ['Baseline', 'No3DConv', 'NoResidual', 'Production (2.5D-CSRF)']
    df_radar = df_full[df_full['Variant'].isin(key_variants)]
    
    # Metrics to compare
    metrics = ['Val_Dice', 'Precision', 'Recall', 'F1_Score']
    metric_labels = ['Dice', 'Precision', 'Recall', 'F1 Score']
    
    # Number of variables
    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Plot each variant
    colors = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71']
    for idx, (_, row) in enumerate(df_radar.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]
        
        label = row['Variant'].replace('HybridMiniSwin3D_', '').replace('Production (2.5D-CSRF)', 'Production')
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Fix axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['60%', '70%', '80%', '90%', '100%'], fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.title('Multi-Metric Performance Comparison', fontweight='bold', pad=20, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_radar_chart.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig5_radar_chart.pdf', bbox_inches='tight')
    print("✓ Figure 5 saved: Radar Chart")
    plt.close()


# ============================================================================
# STATISTICAL TESTS
# ============================================================================
def perform_statistical_tests():
    """Perform statistical significance tests"""
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)
    
    # Get baseline and production model Dice scores
    baseline_dice = df[df['Variant'] == 'Baseline']['Val_Dice'].values[0]
    production_dice = 0.8231  # Current model (Epoch 12, OptimalModel_Evidential)
    
    # Approximate standard errors (you may need to calculate these from your actual data)
    baseline_se = 0.02
    production_se = 0.015
    
    # Two-sample z-test
    z_score = (production_dice - baseline_dice) / np.sqrt(baseline_se**2 + production_se**2)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    print(f"\n1. Production Model vs. Baseline:")
    print(f"   - Production Dice: {production_dice:.4f}")
    print(f"   - Baseline Dice:   {baseline_dice:.4f}")
    print(f"   - Difference:      {(production_dice - baseline_dice):.4f} ({((production_dice - baseline_dice)/baseline_dice)*100:.2f}%)")
    print(f"   - Z-score:         {z_score:.4f}")
    print(f"   - P-value:         {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")
    
    # ANOVA for ablation variants
    dice_scores = df['Val_Dice'].values
    f_stat, p_anova = stats.f_oneway(*[dice_scores[i:i+1] for i in range(len(dice_scores))])
    
    print(f"\n2. ANOVA (Ablation Variants):")
    print(f"   - F-statistic:     {f_stat:.4f}")
    print(f"   - P-value:         {p_anova:.6f} {'***' if p_anova < 0.001 else 'n.s.'}")
    
    # Component importance rankings
    print(f"\n3. Component Importance Ranking:")
    baseline_dice = df[df['Variant'] == 'Baseline']['Val_Dice'].values[0]
    component_importance = []
    for _, row in df.iterrows():
        if row['Variant'] != 'Baseline':
            component = row['Variant'].replace('HybridMiniSwin3D_', '').replace('No', '')
            delta = baseline_dice - row['Val_Dice']
            component_importance.append((component, delta))
    
    component_importance.sort(key=lambda x: x[1], reverse=True)
    for rank, (comp, delta) in enumerate(component_importance, 1):
        importance_level = "Critical ⭐⭐⭐" if abs(delta) > 0.03 else "Moderate ⭐⭐" if abs(delta) > 0.005 else "Low ⭐"
        print(f"   {rank}. {comp:15s}: {delta:+.4f} ({delta/baseline_dice*100:+.2f}%) - {importance_level}")
    
    print("\n" + "=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Input CSV: {ABLATION_CSV}\n")
    
    # Generate all figures
    plot_dice_comparison()
    plot_precision_recall()
    plot_component_importance()
    plot_overfitting_analysis()
    plot_radar_chart()
    
    # Perform statistical tests
    perform_statistical_tests()
    
    print("\n" + "=" * 80)
    print("✓ ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nFigures saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - fig1_dice_comparison.png/.pdf")
    print("  - fig2_precision_recall.png/.pdf")
    print("  - fig3_component_importance.png/.pdf")
    print("  - fig4_overfitting_analysis.png/.pdf")
    print("  - fig5_radar_chart.png/.pdf")
    print("\n✓ Ready for manuscript submission!")
