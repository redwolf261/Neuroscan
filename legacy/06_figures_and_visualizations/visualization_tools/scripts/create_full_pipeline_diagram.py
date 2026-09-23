"""
Create Comprehensive Horizontal Pipeline Diagram for HybridMiniSwin2.5D-CBAM
Matches the style of the reference workflow diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# Set up the figure - very wide for horizontal layout
fig = plt.figure(figsize=(32, 9))
ax = fig.add_subplot(111)
ax.set_xlim(0, 32)
ax.set_ylim(0, 9)
ax.axis('off')

# Color scheme
color_header = '#5D7B6F'
color_data = '#E8F4F8'
color_preprocess = '#D4E9D7'
color_mae = '#F9E4D4'
color_model = '#FFE4B5'
color_fusion = '#FFD700'
color_training = '#FFF9E6'
color_eval = '#E8F8F4'

def draw_box(ax, x, y, width, height, text, color, fontsize=9, bold=False, edgecolor='black', linewidth=1.5):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x, y), width, height, 
                          boxstyle="round,pad=0.05", 
                          facecolor=color, 
                          edgecolor=edgecolor, 
                          linewidth=linewidth)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', 
            fontsize=fontsize, weight=weight)

def draw_arrow(ax, x1, y1, x2, y2, color='black', linewidth=2):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', 
                           color=color, 
                           linewidth=linewidth,
                           mutation_scale=15)
    ax.add_patch(arrow)

# ==============================================================================
# HEADER WITH STAGE TITLES
# ==============================================================================
header_y = 8.2
header_height = 0.6

# Main title banner
title_box = Rectangle((0.3, 8.4), 31.4, 0.5, 
                      facecolor=color_header, 
                      edgecolor='black', 
                      linewidth=2)
ax.add_patch(title_box)
ax.text(16, 8.65, 'HybridMiniSwin2.5D-CBAM: Pediatric MS Lesion Segmentation Pipeline', 
        ha='center', va='center', 
        fontsize=15, weight='bold', color='white')

# Stage headers
stages = [
    (0.3, 7.5, 4.0, 0.5, 'Data Acquisition &\nPreprocessing'),
    (4.8, 7.5, 4.2, 0.5, 'MAE Pretraining'),
    (9.5, 7.5, 9.5, 0.5, 'Model Architecture (HybridMiniSwin2.5D-CBAM)'),
    (19.5, 7.5, 4.5, 0.5, 'Training'),
    (24.5, 7.5, 3.5, 0.5, 'Evaluation')
]

for x, y, w, h, text in stages:
    box = Rectangle((x, y), w, h, 
                    facecolor='#E0E0E0', 
                    edgecolor='black', 
                    linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, 
            ha='center', va='center', 
            fontsize=10, weight='bold')

# ==============================================================================
# STAGE 1: DATA ACQUISITION & PREPROCESSING
# ==============================================================================
x_data = 0.4

# Input data
draw_box(ax, x_data, 5.0, 1.5, 1.6, 'PediMS\nDataset\n(63 patients)', 
         color_data, fontsize=10, bold=True)

# MRI sequences
draw_box(ax, x_data, 2.8, 1.5, 1.5, 'T1w\nT2w\nFLAIR', 
         color_data, fontsize=10)

# Preprocessing steps
draw_box(ax, x_data + 2.0, 5.8, 1.6, 0.8, 'N4 Bias\nCorrection', 
         color_preprocess, fontsize=9)
draw_box(ax, x_data + 2.0, 4.7, 1.6, 0.8, 'Registration\n(T1 space)', 
         color_preprocess, fontsize=9)
draw_box(ax, x_data + 2.0, 3.6, 1.6, 0.8, 'Intensity\nNormalization', 
         color_preprocess, fontsize=9)

# Arrows
draw_arrow(ax, x_data + 1.5, 5.8, x_data + 2.0, 6.2)
draw_arrow(ax, x_data + 1.5, 5.0, x_data + 2.0, 5.1)
draw_arrow(ax, x_data + 1.5, 3.5, x_data + 2.0, 4.0)
draw_arrow(ax, x_data + 3.6, 5.0, 4.8, 5.0)

# ==============================================================================
# STAGE 2: MAE PRETRAINING
# ==============================================================================
x_mae = 4.9

# Random masking
draw_box(ax, x_mae + 0.9, 6.5, 1.3, 0.6, 'Random\nMasking (75%)', 
         '#FFE4E1', fontsize=9)

# MAE Encoder
draw_box(ax, x_mae, 4.8, 1.6, 1.2, 'MAE\nEncoder\n(2.5D ResNet+Swin)', 
         color_mae, fontsize=9, bold=True)

# Pretrained weights indicator
draw_box(ax, x_mae + 1.9, 5.0, 1.3, 0.6, 'Pretrained\nWeights', 
         '#FFF9E6', fontsize=8)

# MAE Decoder
draw_box(ax, x_mae + 0.9, 3.8, 1.3, 0.7, 'MAE\nDecoder', 
         color_mae, fontsize=9)

# Reconstruction
draw_box(ax, x_mae + 0.9, 2.6, 1.3, 0.9, 'Reconstruct\nMasked\nPatches', 
         '#FFE4E1', fontsize=9)

# Arrows
draw_arrow(ax, x_mae + 1.5, 5.9, x_mae + 0.8, 6.2)
draw_arrow(ax, x_mae + 0.8, 6.0, x_mae + 0.8, 4.8)
draw_arrow(ax, x_mae + 0.8, 4.8, x_mae + 1.5, 4.5)
draw_arrow(ax, x_mae + 1.5, 3.8, x_mae + 1.5, 3.5)

# ==============================================================================
# STAGE 3: MODEL ARCHITECTURE
# ==============================================================================
x_model = 9.6

# Arrow to model architecture (pretrained weights)
draw_arrow(ax, x_mae + 3.2, 5.2, x_model, 6.8)
ax.text(7.5, 6.2, 'Pretrained\nWeights', ha='center', fontsize=8, style='italic')

# Adaptive Slice Selector (Novel Component)
draw_box(ax, x_model, 6.4, 1.3, 1.0, 'Adaptive\nSlice\nSelector\n(k=9)', 
         '#FFF9E6', fontsize=8, bold=True, edgecolor='#FF6B6B', linewidth=2)

# Arrow from adaptive selector to input
draw_arrow(ax, x_model + 0.65, 6.4, x_model + 0.65, 5.9)

# Input
draw_box(ax, x_model, 4.6, 1.3, 1.2, '3-Channel\nInput\n(9×H×W)', 
         color_data, fontsize=9, bold=True)

# Encoder components
draw_box(ax, x_model + 1.5, 6.2, 1.4, 0.7, 'Swin\nTransformer\nW=4×4', 
         color_model, fontsize=8)
draw_box(ax, x_model + 1.5, 5.3, 1.4, 0.7, '2.5D Conv +\nResidual', 
         color_model, fontsize=8)
draw_box(ax, x_model + 1.5, 4.4, 1.4, 0.7, 'Mini-Swin\nAttention', 
         color_model, fontsize=8)

# Multi-scale features
draw_box(ax, x_model + 3.1, 6.4, 0.8, 0.5, 'F₁\n(High)', 
         '#FFE4B5', fontsize=8)
draw_box(ax, x_model + 3.1, 5.8, 0.8, 0.5, 'F₂\n(Mid)', 
         '#FFE4B5', fontsize=8)
draw_box(ax, x_model + 3.1, 5.2, 0.8, 0.5, 'F₃\n(Mid)', 
         '#FFE4B5', fontsize=8)
draw_box(ax, x_model + 3.1, 4.6, 0.8, 0.5, 'F₄\n(Low)', 
         '#FFE4B5', fontsize=8)

# CBAM Fusion Module (HIGHLIGHTED)
draw_box(ax, x_model + 4.1, 3.8, 2.0, 3.0, 
         'CBAM Fusion\nModule\n\n• Channel\n  Attention\n• Spatial\n  Attention\n• Feature\n  Refinement', 
         color_fusion, fontsize=9, bold=True, linewidth=2.5)

# Arrows to CBAM
draw_arrow(ax, x_model + 3.9, 6.65, x_model + 4.1, 6.2)
draw_arrow(ax, x_model + 3.9, 6.05, x_model + 4.1, 5.8)
draw_arrow(ax, x_model + 3.9, 5.45, x_model + 4.1, 5.4)
draw_arrow(ax, x_model + 3.9, 4.85, x_model + 4.1, 5.0)

# Decoder blocks
draw_box(ax, x_model + 6.3, 6.2, 1.3, 0.6, 'Upsample\nBlock 1', 
         color_model, fontsize=8)
draw_box(ax, x_model + 6.3, 5.3, 1.3, 0.6, 'Upsample\nBlock 2', 
         color_model, fontsize=8)
draw_box(ax, x_model + 6.3, 4.4, 1.3, 0.6, 'Upsample\nBlock 3', 
         color_model, fontsize=8)

# Arrows from CBAM to decoder
draw_arrow(ax, x_model + 6.1, 5.3, x_model + 6.3, 6.5)
draw_arrow(ax, x_model + 6.1, 5.3, x_model + 6.3, 5.6)
draw_arrow(ax, x_model + 6.1, 5.3, x_model + 6.3, 4.7)

# Output segmentation
draw_box(ax, x_model + 7.8, 5.6, 1.4, 0.9, 'Evidential\nHead\n(β₀, α)', 
         '#F4E4F8', fontsize=8, bold=True)

draw_box(ax, x_model + 7.8, 4.3, 1.4, 1.0, 'Segmentation\nMap\n(H×W)', 
         color_data, fontsize=9, bold=True)

# Arrows through encoder
draw_arrow(ax, x_model + 1.3, 5.2, x_model + 1.5, 6.6)
draw_arrow(ax, x_model + 1.3, 5.2, x_model + 1.5, 5.7)
draw_arrow(ax, x_model + 1.3, 5.2, x_model + 1.5, 4.8)

# Arrow to decoder output
draw_arrow(ax, x_model + 7.6, 6.5, x_model + 7.8, 6.05)
draw_arrow(ax, x_model + 8.5, 5.6, x_model + 8.5, 5.3)

# ==============================================================================
# STAGE 4: TRAINING
# ==============================================================================
x_train = 14.9

# Loss function
draw_box(ax, x_train, 5.5, 1.4, 0.7, 'Dice + BCE\nLoss', 
         color_training, fontsize=8, bold=True)

# Optimizer
draw_box(ax, x_train, 4.6, 1.4, 0.7, 'AdamW\nLR=3e-4', 
         color_training, fontsize=8)

# Training details
draw_box(ax, x_train, 3.7, 1.4, 0.7, '5-Fold CV\n48 epochs', 
         color_training, fontsize=8)

# Arrow from segmentation to training
draw_arrow(ax, x_model + 9.2, 4.8, 19.6, 6.5)

# ==============================================================================
# STAGE 4: TRAINING
# ==============================================================================
x_train = 19.7

# Loss function
draw_box(ax, x_train, 6.2, 1.8, 0.9, 'Dice + BCE\nLoss', 
         color_training, fontsize=10, bold=True)

# Optimizer
draw_box(ax, x_train, 5.0, 1.8, 0.9, 'AdamW\nLR=3e-4', 
         color_training, fontsize=10)

# Training epochs
draw_box(ax, x_train, 3.8, 1.8, 0.9, '48 epochs', 
         color_training, fontsize=10)

# ==============================================================================
# STAGE 5: EVALUATION
# ==============================================================================
x_eval = 24.6

# Metrics
draw_box(ax, x_eval, 6.4, 1.6, 0.6, 'Dice: 82.31%', 
         color_eval, fontsize=10, bold=True)
draw_box(ax, x_eval, 5.6, 1.6, 0.6, 'Precision: 77.60%', 
         color_eval, fontsize=10)
draw_box(ax, x_eval, 4.8, 1.6, 0.6, 'Recall: 91.64%', 
         color_eval, fontsize=10)
draw_box(ax, x_eval, 4.0, 1.6, 0.6, 'F1: 84.04%', 
         color_eval, fontsize=10)

# Arrow to evaluation
draw_arrow(ax, x_train + 1.8, 5.3, x_eval, 5.9)

# ==============================================================================
# BOTTOM SECTION: FOOTER TEXT
# ==============================================================================
ax.text(16, 1.2, 
        'HybridMiniSwin2.5D-CBAM: 2.5D Deep Learning Architecture for Pediatric MS Lesion Segmentation',
        ha='center', fontsize=11, style='italic', color='#555555')

ax.text(16, 0.5, 
        'Adaptive Slice Selection (k=9) • MAE Pretraining (75% mask) • CBAM Fusion • Evidential Uncertainty',
        ha='center', fontsize=10, color='#777777', style='italic')

# Adjust layout and save
plt.tight_layout()
output_path_png = 'C:/Users/HP/EDI/paper_figures/workflow_pipeline_comprehensive.png'
output_path_pdf = 'C:/Users/HP/EDI/paper_figures/workflow_pipeline_comprehensive.pdf'

plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')

print("✅ Comprehensive pipeline workflow diagram saved!")
print(f"   - PNG: {output_path_png}")
print(f"   - PDF: {output_path_pdf}")

plt.show()
