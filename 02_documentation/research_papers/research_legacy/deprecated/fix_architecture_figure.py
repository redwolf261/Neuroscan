"""
Regenerate only the Architecture & Efficiency figure with fixed layout
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Colors
PRIMARY_COLOR = '#2E86AB'
SECONDARY_COLOR = '#A23B72'
SUCCESS_COLOR = '#06A77D'
ACCENT_COLOR = '#F18F01'

OUTPUT_DIR = Path("C:/Users/HP/EDI/paper_figures")
val_logs = pd.read_csv("C:/Users/HP/EDI/csv_data/production_model/production_val_logs.csv")
metrics_data = json.load(open("C:/Users/HP/EDI/csv_data/model_metrics.json"))

print("Regenerating Architecture & Efficiency figure with fixed layout...")

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35, top=0.92, bottom=0.08, left=0.08, right=0.95)

fig.suptitle('Model Architecture & Computational Efficiency', 
            fontsize=16, fontweight='bold', y=0.97)

# 4.1 Parameter Distribution
ax1 = fig.add_subplot(gs[0, 0])

components = ['Encoder', 'CSRF', 'Decoder']
params = [29.28, 3.28, 1.66]  # Millions
colors_comp = [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR]

wedges, texts, autotexts = ax1.pie(params, labels=components, autopct='%1.1f%%',
                                    colors=colors_comp, startangle=90,
                                    textprops={'fontsize': 10, 'fontweight': 'bold'})

# Add parameter counts with better positioning
for i, (component, param) in enumerate(zip(components, params)):
    angle = (wedges[i].theta2 + wedges[i].theta1) / 2
    x = 1.35 * np.cos(np.radians(angle))
    y = 1.35 * np.sin(np.radians(angle))
    ax1.text(x, y, f'{param:.2f}M', ha='center', va='center',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
            edgecolor=colors_comp[i], linewidth=1.5))

ax1.set_title('Parameter Distribution\n(Total: 34.2M)', fontweight='bold', pad=20)

# 4.2 Computational Cost Breakdown
ax2 = fig.add_subplot(gs[0, 1])

operations = ['Conv\nOperations', 'Linear\nLayers', 'Batch\nNorm', 
             'Attention\n(MatMul)', 'Other']
flops_percent = [84.0, 15.4, 0.3, 0.2, 0.1]
colors_ops = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#95a5a6']

bars = ax2.barh(operations, flops_percent, color=colors_ops, 
               edgecolor='black', linewidth=1.5)

for bar, value in zip(bars, flops_percent):
    width = bar.get_width()
    ax2.text(width + 1.5, bar.get_y() + bar.get_height()/2,
            f'{value:.1f}%', va='center', fontweight='bold', fontsize=9)

ax2.set_xlabel('Percentage of Total FLOPs', fontweight='bold', fontsize=11)
ax2.set_title('Computational Cost Breakdown\n(Total: 1.75 GFLOPs)', 
             fontweight='bold', pad=20, fontsize=12)
ax2.set_xlim([0, 100])
ax2.grid(True, axis='x', alpha=0.3)
ax2.tick_params(labelsize=9)

# 4.3 Efficiency Metrics
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')

efficiency_data = [
    ('Model Size', '~135 MB', 'Checkpoint file size'),
    ('Parameters', '34.2M', 'Trainable parameters'),
    ('FLOPs', '1.75 G', 'Floating point operations'),
    ('Inference Time', '490 ms', 'Average on CPU'),
    ('Throughput', '2.0 vol/sec', 'Processing speed'),
    ('Training Time', '28 epochs', 'To best performance'),
]

y_pos = 0.92
for i, (metric, value, description) in enumerate(efficiency_data):
    color = [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, 
            SUCCESS_COLOR, '#9B59B6', '#E91E63'][i]
    
    # Background
    rect = plt.Rectangle((0.03, y_pos-0.135), 0.94, 0.12,
                         facecolor=color, alpha=0.1,
                         transform=ax3.transAxes)
    ax3.add_patch(rect)
    
    # Metric name and value with better spacing
    ax3.text(0.06, y_pos-0.045, metric, ha='left', va='top',
            fontsize=9, fontweight='bold', transform=ax3.transAxes)
    ax3.text(0.93, y_pos-0.045, value, ha='right', va='top',
            fontsize=10, fontweight='bold', color=color,
            transform=ax3.transAxes)
    
    # Description with smaller font
    ax3.text(0.06, y_pos-0.105, description, ha='left', va='top',
            fontsize=7.5, style='italic', color='gray',
            transform=ax3.transAxes)
    
    y_pos -= 0.15

ax3.set_title('Efficiency Summary', fontweight='bold', pad=15, fontsize=12)

# 4.4 Training Efficiency
ax4 = fig.add_subplot(gs[1, 1])

epochs_to_threshold = []
thresholds = []

for threshold in range(70, 85):
    epoch = val_logs[val_logs['dice'] * 100 >= threshold]['epoch'].min()
    if not np.isnan(epoch):
        thresholds.append(threshold)
        epochs_to_threshold.append(epoch)

ax4.plot(thresholds, epochs_to_threshold, linewidth=2.5, 
        color=PRIMARY_COLOR, marker='o', markersize=8)
ax4.axhline(y=27, color=SUCCESS_COLOR, linestyle='--', linewidth=2,
           label='Best Performance (Epoch 28)', alpha=0.7)

ax4.fill_between(thresholds, 0, epochs_to_threshold, 
                alpha=0.2, color=PRIMARY_COLOR)

ax4.set_xlabel('Dice Score Threshold (%)', fontweight='bold', fontsize=11)
ax4.set_ylabel('Epochs Required', fontweight='bold', fontsize=11)
ax4.set_title('Training Efficiency\n(Epochs to Reach Performance)', 
             fontweight='bold', pad=20, fontsize=12)
ax4.legend(loc='upper left', framealpha=0.9, fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 50])
ax4.tick_params(labelsize=9)

plt.savefig(OUTPUT_DIR / "architecture_efficiency.png", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "architecture_efficiency.pdf", bbox_inches='tight')
plt.close()

print("✅ Architecture & Efficiency figure regenerated successfully!")
print(f"   Saved to: {OUTPUT_DIR / 'architecture_efficiency.png'}")
