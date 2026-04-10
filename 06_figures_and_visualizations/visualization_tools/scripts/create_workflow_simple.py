"""
Create SIMPLIFIED Workflow Diagram for HybridMiniSwin2.5D-CSRF
Inspired by the clean style of the provided paper examples
LESS IS MORE - Focus on key flow, not overwhelming details
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set up the figure - simpler, more horizontal, less vertical space
fig = plt.figure(figsize=(18, 5))
ax = fig.add_subplot(111)
ax.set_xlim(0, 18)
ax.set_ylim(0, 5)
ax.axis('off')

# Simple color scheme
color_input = '#E8F4F8'
color_process = '#F9E4D4'
color_highlight = '#FFD700'  # Gold for CSRF
color_output = '#D4E9D7'
color_arrow = '#2C5F2D'

def draw_box(ax, x, y, width, height, text, color, fontsize=10, bold=False, linewidth=2):
    """Draw a simple rounded box"""
    box = FancyBboxPatch((x, y), width, height, 
                          boxstyle="round,pad=0.05", 
                          facecolor=color, 
                          edgecolor='black', 
                          linewidth=linewidth)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', 
            fontsize=fontsize, weight=weight)

def draw_arrow(ax, x1, y1, x2, y2, label='', linewidth=2.5):
    """Draw a simple arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', 
                           color=color_arrow, 
                           linewidth=linewidth,
                           mutation_scale=25)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.3, label, 
                ha='center', fontsize=9, style='italic', color=color_arrow)

# ==============================================================================
# MAIN TITLE
# ==============================================================================
ax.text(9, 4.6, 'HybridMiniSwin2.5D-CSRF: Pediatric MS Lesion Segmentation', 
        ha='center', fontsize=14, weight='bold')

# ==============================================================================
# SIMPLIFIED WORKFLOW (HORIZONTAL FLOW ONLY)
# ==============================================================================

# 1. DATA INPUT
draw_box(ax, 0.5, 1.8, 2.0, 1.8, 
         'PediMS Dataset\n\n3-Channel MRI\n(T1w, T2w, FLAIR)\n64³ volumes', 
         color_input, fontsize=9, bold=True)

# 2. PREPROCESSING
draw_box(ax, 3.0, 1.8, 1.8, 1.8, 
         'Preprocessing\n\nN4 Bias\nRegistration\nNormalization', 
         color_process, fontsize=9)

# 3. MAE PRETRAINING
draw_box(ax, 5.3, 1.8, 1.8, 1.8, 
         'MAE Pretraining\n\n75% Masking', 
         color_process, fontsize=9, bold=True)

# 4. ENCODER
draw_box(ax, 7.6, 1.8, 1.8, 1.8, 
         'Hybrid Encoder\n\nSwin + 3D Conv', 
         color_process, fontsize=9)

# 5. CSRF MODULE (HIGHLIGHT)
draw_box(ax, 9.9, 1.5, 2.2, 2.4, 
         'CSRF Fusion\nModule\n\n⭐ Channel-wise\nScaling (α)\n⭐ Multi-scale\nAttention', 
         color_highlight, fontsize=9, bold=True, linewidth=3)

# 6. DECODER
draw_box(ax, 12.6, 1.8, 1.8, 1.8, 
         'Decoder\n\nUpsampling\nSkip Connections', 
         color_process, fontsize=9)

# 7. OUTPUT
draw_box(ax, 14.9, 1.8, 2.0, 1.8, 
         'Segmentation\n(64³)\n\nDice: 82.31%\nRecall: 88.17%', 
         color_output, fontsize=9, bold=True)

# ==============================================================================
# ARROWS (SIMPLE FLOW)
# ==============================================================================
draw_arrow(ax, 2.5, 2.7, 3.0, 2.7)
draw_arrow(ax, 4.8, 2.7, 5.3, 2.7)
draw_arrow(ax, 7.1, 2.7, 7.6, 2.7)
draw_arrow(ax, 9.4, 2.7, 9.9, 2.7)
draw_arrow(ax, 12.1, 2.7, 12.6, 2.7)
draw_arrow(ax, 14.4, 2.7, 14.9, 2.7)

# ==============================================================================
# BOTTOM NOTE (MINIMAL - OPTIONAL)
# ==============================================================================
ax.text(9, 1.0, 
        'End-to-end pipeline for pediatric MS lesion segmentation with MAE pretraining and CSRF fusion',
        ha='center', fontsize=10, style='italic', color='#555555')

# Adjust and save
plt.tight_layout()
plt.savefig('C:/Users/HP/EDI/visualization/workflow_diagram_simple.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('C:/Users/HP/EDI/paper_figures/main_figures/workflow_diagram_simple.pdf', 
            bbox_inches='tight', facecolor='white')
print("✅ SIMPLIFIED workflow diagram saved!")
print("   - PNG: visualization/workflow_diagram_simple.png")
print("   - PDF: paper_figures/main_figures/workflow_diagram_simple.pdf")
print("\n📊 This version is MUCH cleaner - focuses on the key workflow!")

plt.show()
