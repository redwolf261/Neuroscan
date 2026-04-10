"""
Generate Figures for Noise Robustness and CSRF Variants Studies
Creates publication-quality figures for:
1. Noise Robustness Analysis (5 noise types × 4 levels)
2. CSRF Fusion Variants Comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
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

NOISE_DIR = Path("C:/Users/HP/EDI/research/noise_robustness_results_quick")
CSRF_DIR = Path("C:/Users/HP/EDI/research/csrf_variants_results_quick")

print("=" * 80)
print("📊 GENERATING ROBUSTNESS & ABLATION FIGURES")
print("=" * 80)

# ============================================================================
# FIGURE 14: Noise Robustness Analysis
# ============================================================================
def generate_noise_robustness():
    """
    Performance degradation under different noise types and levels
    """
    print("\n🔊 Generating Figure 14: Noise Robustness Analysis...")
    
    # Load data
    df = pd.read_csv(NOISE_DIR / "noise_robustness_results.csv")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Line plot - Dice score vs noise level for each noise type
    ax = axes[0]
    
    noise_types = df['Noise_Type'].unique()
    noise_levels = df['Noise_Level'].unique()
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#E63946']
    markers = ['o', 's', '^', 'd', 'v']
    
    for i, noise_type in enumerate(noise_types):
        noise_data = df[df['Noise_Type'] == noise_type]
        levels = [float(l.strip('%')) for l in noise_data['Noise_Level']]
        dice_scores = noise_data['Dice'].values * 100
        
        ax.plot(levels, dice_scores, marker=markers[i], linewidth=2.5, 
                markersize=8, label=noise_type.replace('_', ' ').title(),
                color=colors[i], alpha=0.85)
    
    ax.set_xlabel('Noise Level (%)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Dice Score (%)', fontweight='bold', fontsize=12)
    ax.set_title('(a) Performance Under Different Noise Types', 
                 fontweight='bold', fontsize=13, pad=15)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks([0, 5, 10, 15])
    ax.set_ylim([55, 82])
    
    # Add baseline reference
    baseline_dice = df[df['Noise_Level'] == '0.00%']['Dice'].iloc[0] * 100
    ax.axhline(y=baseline_dice, color='red', linestyle='--', 
               linewidth=2, alpha=0.5, label=f'Baseline ({baseline_dice:.1f}%)')
    
    # Panel 2: Heatmap - Degradation percentage
    ax = axes[1]
    
    # Calculate degradation
    degradation_data = []
    for noise_type in noise_types:
        noise_data = df[df['Noise_Type'] == noise_type]
        baseline = noise_data[noise_data['Noise_Level'] == '0.00%']['Dice'].values[0] * 100
        
        row_degradation = []
        for level in ['5.00%', '10.00%', '15.00%']:
            level_dice = noise_data[noise_data['Noise_Level'] == level]['Dice'].values[0] * 100
            degradation = level_dice - baseline
            row_degradation.append(degradation)
        
        degradation_data.append(row_degradation)
    
    degradation_df = pd.DataFrame(
        degradation_data,
        index=[n.replace('_', ' ').title() for n in noise_types],
        columns=['5%', '10%', '15%']
    )
    
    # Create heatmap
    im = ax.imshow(degradation_df.values, cmap='RdYlGn', aspect='auto', 
                   vmin=-10, vmax=25, interpolation='nearest')
    
    # Add values on heatmap
    for i in range(len(degradation_df.index)):
        for j in range(len(degradation_df.columns)):
            value = degradation_df.values[i, j]
            color = 'white' if abs(value) > 10 else 'black'
            text = ax.text(j, i, f'{value:+.1f}%',
                          ha="center", va="center", color=color, 
                          fontweight='bold', fontsize=9)
    
    ax.set_xticks(range(len(degradation_df.columns)))
    ax.set_yticks(range(len(degradation_df.index)))
    ax.set_xticklabels(degradation_df.columns)
    ax.set_yticklabels(degradation_df.index)
    ax.set_xlabel('Noise Level', fontweight='bold', fontsize=12)
    ax.set_title('(b) Performance Change from Baseline', 
                 fontweight='bold', fontsize=13, pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Dice Change (%)', fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    for ext in ['png', 'pdf']:
        plt.savefig(OUTPUT_DIR / f'fig14_noise_robustness.{ext}', 
                   bbox_inches='tight', dpi=300 if ext == 'png' else None)
    
    plt.close()
    
    # Calculate statistics
    best_robust = degradation_df.mean(axis=1).idxmax()
    worst_robust = degradation_df.mean(axis=1).idxmin()
    
    print(f"✅ Saved: fig14_noise_robustness.png/pdf")
    print(f"   Most robust to: {best_robust} (avg {degradation_df.mean(axis=1).max():+.2f}%)")
    print(f"   Least robust to: {worst_robust} (avg {degradation_df.mean(axis=1).min():+.2f}%)")


# ============================================================================
# FIGURE 15: CSRF Fusion Variants Comparison (Bar Graphs Only)
# ============================================================================
def generate_csrf_variants():
    """
    Compare different channel-spatial fusion strategies - BAR CHARTS ONLY
    """
    print("\n🔀 Generating Figure 15: CSRF Fusion Variants...")
    
    # Load data
    with open(CSRF_DIR / "csrf_variants_results_quick.json", 'r') as f:
        data = json.load(f)
    
    # Create figure with 2 subplots (validation dice + training curves)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract data
    variants = []
    variant_names = []
    best_dices = []
    final_dices = []
    params = []
    
    for key, value in data.items():
        variants.append(key)
        # Replace "CBAM" with "CBAM (Proposed)"
        name = value['variant_name']
        if 'CBAM' in name:
            name = 'CBAM (Proposed)'
        variant_names.append(name)
        best_dices.append(value['best_val_dice'] * 100)
        final_dices.append(value['final_val_dice'] * 100)
        params.append(value['total_params'] / 1e6)
    
    # Panel 1: Best Validation Dice Bar Chart
    ax = axes[0]
    
    colors_grad = ['#95A5A6', '#3498DB', '#27AE60', '#E74C3C']  # Green for CBAM (proposed)
    x = np.arange(len(variant_names))
    
    bars = ax.bar(x, best_dices, color=colors_grad, 
                  edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add value labels on bars
    for i, (bar, dice) in enumerate(zip(bars, best_dices)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{dice:.2f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, rotation=0, ha='center', fontsize=11)
    ax.set_ylabel('Best Validation Dice (%)', fontweight='bold', fontsize=13)
    ax.set_title('(a) Validation Performance Comparison', fontweight='bold', fontsize=14, pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([62, 71])
    
    # Panel 2: Training Curves
    ax = axes[1]
    
    for i, (key, value, color, name) in enumerate(zip(variants, data.values(), colors_grad, variant_names)):
        epochs = range(1, len(value['val_dices']) + 1)
        dices = [d * 100 for d in value['val_dices']]
        
        # Plot line
        ax.plot(epochs, dices, linewidth=2.5, 
                color=color, alpha=0.9, label=name)
        
        # Add final value annotation
        final_dice = dices[-1]
        ax.text(len(epochs), final_dice, f'{final_dice:.1f}%', 
               fontsize=9, fontweight='bold', color=color,
               ha='left', va='center')
    
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=13)
    ax.set_ylabel('Validation Dice (%)', fontweight='bold', fontsize=13)
    ax.set_title('(b) Training Convergence', fontweight='bold', fontsize=14, pad=15)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([58, 72])
    ax.set_xlim([1, 15])
    
    plt.tight_layout()
    
    # Save
    for ext in ['png', 'pdf']:
        plt.savefig(OUTPUT_DIR / f'fig15_csrf_variants.{ext}', 
                   bbox_inches='tight', dpi=300 if ext == 'png' else None)
    
    plt.close()
    
    best_idx = np.argmax(best_dices)
    print(f"✅ Saved: fig15_csrf_variants.png/pdf")
    print(f"   Best variant: {variant_names[best_idx]} ({best_dices[best_idx]:.2f}% Dice)")
    print(f"   Variants tested: {len(variant_names)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    try:
        # Generate all figures
        generate_noise_robustness()
        generate_csrf_variants()
        
        print("\n" + "=" * 80)
        print("✅ ALL ROBUSTNESS FIGURES GENERATED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n📁 Output directory: {OUTPUT_DIR}")
        print("\n📊 Generated Figures:")
        print("   - fig14_noise_robustness.png/pdf")
        print("   - fig15_csrf_variants.png/pdf")
        print("\n🎯 Total: 2 new publication-quality figures!")
        
    except Exception as e:
        print(f"\n❌ Error generating figures: {e}")
        import traceback
        traceback.print_exc()
