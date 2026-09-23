"""
Create Computational Efficiency Comparison Figure
Based on actual measured checkpoint data
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Real measured data from checkpoints
trial_data = {
    'Parameters (M)': 0.82,           # Measured from checkpoint
    'Model Size (MB)': 3.14,          # Measured from file size
    'Dice Score (%)': 74.51,          # From best_model_meta.json
    'Epochs to Best': 200,            # From metadata
}

final_data = {
    'Parameters (M)': 34.24,          # Measured: 34,235,231 parameters
    'Model Size (MB)': 366.91,        # Measured from file size
    'Dice Score (%)': 82.31,          # From best epoch 12: val_dice: 0.8231
    'Epochs to Best': 28,             # From checkpoint epoch
}

# Colors
TRIAL_COLOR = '#e74c3c'
FINAL_COLOR = '#27ae60'

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Computational Efficiency: Trial vs Final Model', 
             fontsize=16, fontweight='bold', y=0.98)

# 1. Parameters comparison
ax1 = axes[0, 0]
x = ['Trial Model', 'Final Model']
params = [trial_data['Parameters (M)'], final_data['Parameters (M)']]
bars = ax1.bar(x, params, color=[TRIAL_COLOR, FINAL_COLOR], alpha=0.8, 
              edgecolor='black', linewidth=2)
for bar, val in zip(bars, params):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.2f}M', ha='center', fontweight='bold', fontsize=11)
ax1.set_ylabel('Parameters (Millions)', fontweight='bold', fontsize=12)
ax1.set_title('Model Parameters', fontweight='bold', fontsize=13, pad=15)
ax1.text(0.5, max(params) * 0.5, f'41.94× more\nparameters', 
        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax1.grid(axis='y', alpha=0.3)

# 2. Model size comparison
ax2 = axes[0, 1]
sizes = [trial_data['Model Size (MB)'], final_data['Model Size (MB)']]
bars = ax2.bar(x, sizes, color=[TRIAL_COLOR, FINAL_COLOR], alpha=0.8,
              edgecolor='black', linewidth=2)
for bar, val in zip(bars, sizes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val:.2f} MB', ha='center', fontweight='bold', fontsize=11)
ax2.set_ylabel('Checkpoint Size (MB)', fontweight='bold', fontsize=12)
ax2.set_title('Model Size on Disk', fontweight='bold', fontsize=13, pad=15)
ax2.text(0.5, max(sizes) * 0.5, f'116.85× larger\ncheckpoint', 
        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax2.grid(axis='y', alpha=0.3)

# 3. Performance vs Complexity
ax3 = axes[1, 0]
models = ['Trial\n(0.82M params)', 'Final\n(34.24M params)']
dice_scores = [trial_data['Dice Score (%)'], final_data['Dice Score (%)']]
bars = ax3.bar(models, dice_scores, color=[TRIAL_COLOR, FINAL_COLOR], alpha=0.8,
              edgecolor='black', linewidth=2)
for bar, val in zip(bars, dice_scores):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.2f}%', ha='center', fontweight='bold', fontsize=11)
ax3.set_ylabel('Dice Score (%)', fontweight='bold', fontsize=12)
ax3.set_title('Performance vs Model Complexity', fontweight='bold', fontsize=13, pad=15)
ax3.set_ylim([0, 100])
improvement = final_data['Dice Score (%)'] - trial_data['Dice Score (%)']
ax3.text(0.5, 50, f'+{improvement:.2f}%\nimprovement', 
        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax3.grid(axis='y', alpha=0.3)

# 4. Training efficiency
ax4 = axes[1, 1]
epochs = [trial_data['Epochs to Best'], final_data['Epochs to Best']]
bars = ax4.bar(x, epochs, color=[TRIAL_COLOR, FINAL_COLOR], alpha=0.8,
              edgecolor='black', linewidth=2)
for bar, val in zip(bars, epochs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val} epochs', ha='center', fontweight='bold', fontsize=11)
ax4.set_ylabel('Epochs to Best Model', fontweight='bold', fontsize=12)
ax4.set_title('Training Efficiency', fontweight='bold', fontsize=13, pad=15)
speedup = trial_data['Epochs to Best'] / final_data['Epochs to Best']
ax4.text(0.5, max(epochs) * 0.5, f'{speedup:.2f}× faster\nconvergence', 
        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Save
output_dir = Path("paper_figures/model_comparison")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "computational_efficiency_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "computational_efficiency_comparison.pdf", bbox_inches='tight')

print("✅ Saved computational efficiency comparison figures!")
print(f"   PNG: {output_dir / 'computational_efficiency_comparison.png'}")
print(f"   PDF: {output_dir / 'computational_efficiency_comparison.pdf'}")

# Summary table
print("\n" + "=" * 80)
print("COMPUTATIONAL EFFICIENCY SUMMARY")
print("=" * 80)
print(f"\n{'Metric':<25} {'Trial Model':<15} {'Final Model':<15} {'Ratio':<10}")
print("-" * 80)
print(f"{'Parameters':<25} {trial_data['Parameters (M)']:.2f}M{'':<10} {final_data['Parameters (M)']:.2f}M{'':<10} {final_data['Parameters (M)']/trial_data['Parameters (M)']:.2f}×")
print(f"{'Checkpoint Size':<25} {trial_data['Model Size (MB)']:.2f} MB{'':<8} {final_data['Model Size (MB)']:.2f} MB{'':<6} {final_data['Model Size (MB)']/trial_data['Model Size (MB)']:.2f}×")
print(f"{'Dice Score':<25} {trial_data['Dice Score (%)']:.2f}%{'':<10} {final_data['Dice Score (%)']:.2f}%{'':<10} +{final_data['Dice Score (%)'] - trial_data['Dice Score (%)']:.2f}%")
print(f"{'Epochs to Best':<25} {trial_data['Epochs to Best']}{'':<13} {final_data['Epochs to Best']}{'':<13} {trial_data['Epochs to Best']/final_data['Epochs to Best']:.2f}× faster")
print("=" * 80)
