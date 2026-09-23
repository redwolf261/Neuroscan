"""
Regenerate csrf_variants_comparison_quick.png from existing results
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = r"C:\Users\HP\EDI\research\csrf_variants_results_quick"

def generate_plots(results):
    print("\nGenerating plots...")
    
    # Only 2 panels: validation dice + training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CSRF Variants Analysis (Quick)', fontsize=16, fontweight='bold')
    
    variants = list(results.keys())
    # Keep original variant names and remove "(Proposed)" suffix
    variant_names = [results[v]['variant_name'].replace(' (Proposed)', '') for v in variants]
    dice_scores = [results[v]['best_val_dice'] for v in variants]
    
    # 1. Validation Dice comparison with numbers
    colors = 'steelblue'  # Same color for all variants
    bars = axes[0].barh(range(len(variants)), dice_scores, color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(variants)))
    axes[0].set_yticklabels(variant_names)
    axes[0].set_xlabel('Best Validation Dice', fontsize=11)
    axes[0].set_title('Performance Comparison', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add numbers on bars
    for i, (bar, score) in enumerate(zip(bars, dice_scores)):
        axes[0].text(score + 0.003, i, f'{score:.4f}', 
                    va='center', fontsize=9, fontweight='bold')
    
    # 2. Training curves with numbers at endpoints
    for i, v in enumerate(variants):
        epochs = range(1, len(results[v]['val_dices']) + 1)
        label_name = results[v]['variant_name'].replace(' (Proposed)', '')
        line, = axes[1].plot(epochs, results[v]['val_dices'], label=label_name, 
                            linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        # Add number at the end of each curve
        final_dice = results[v]['val_dices'][-1]
        final_epoch = len(results[v]['val_dices'])
        axes[1].annotate(f'{final_dice:.3f}', 
                        xy=(final_epoch, final_dice),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        color=line.get_color())
    
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Validation Dice', fontsize=11)
    axes[1].set_title('Training Curves', fontsize=12, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, "csrf_variants_comparison_quick.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to {plot_file}")
    plt.close()

if __name__ == "__main__":
    # Load existing results
    results_file = os.path.join(OUTPUT_DIR, "csrf_variants_results_quick.json")
    print(f"Loading results from {results_file}...")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Found {len(results)} variants")
    
    # Generate the plot
    generate_plots(results)
    
    print("\n✓ Figure regeneration complete!")
