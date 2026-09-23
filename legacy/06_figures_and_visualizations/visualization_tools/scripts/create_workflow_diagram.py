"""
Create Professional Workflow Diagram for HybridMiniSwin2.5D-CBAM
Similar style to the provided paper examples
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set up the figure with more space
fig = plt.figure(figsize=(24, 16))
ax = fig.add_subplot(111)
ax.set_xlim(0, 24)
ax.set_ylim(0, 16)
ax.axis('off')

# Color scheme (professional medical AI colors)
color_data = '#E8F4F8'  # Light blue
color_preprocess = '#D4E9D7'  # Light green
color_model = '#F9E4D4'  # Light orange
color_postprocess = '#F4E4F8'  # Light purple
color_eval = '#E8F8F4'  # Light teal
color_main = '#5D7B6F'  # Professional green
color_arrow = '#2C5F2D'

def draw_box(ax, x, y, width, height, text, color, fontsize=10, bold=False):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x, y), width, height, 
                          boxstyle="round,pad=0.1", 
                          facecolor=color, 
                          edgecolor='black', 
                          linewidth=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', 
            fontsize=fontsize, weight=weight, wrap=True)
    return box

def draw_arrow(ax, x1, y1, x2, y2, style='->'):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, 
                           color=color_arrow, 
                           linewidth=2.5,
                           mutation_scale=20)
    ax.add_patch(arrow)
    return arrow

def draw_stage_header(ax, x, y, width, height, text):
    """Draw stage header box"""
    box = Rectangle((x, y), width, height, 
                    facecolor='#E8E8E8', 
                    edgecolor='black', 
                    linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', 
            fontsize=11, weight='bold')

# ==============================================================================
# TITLE AND MAIN WORKFLOW BANNER
# ==============================================================================

# Main title banner with more space
title_box = Rectangle((0.8, 14.5), 22.4, 1.0, 
                      facecolor=color_main, 
                      edgecolor='black', 
                      linewidth=2.5)
ax.add_patch(title_box)
ax.text(12, 15.0, 'HybridMiniSwin2.5D-CBAM: Pediatric MS Lesion Segmentation Pipeline', 
        ha='center', va='center', 
        fontsize=16, weight='bold', color='white')

# Stage headers - more spaced
stages = [
    (0.8, 13.5, 4.2, 0.6, 'Data Acquisition\n& Preprocessing'),
    (5.5, 13.5, 4.0, 0.6, 'MAE Pretraining'),
    (10.0, 13.5, 7.5, 0.6, 'Model Architecture\n(HybridMiniSwin2.5D-CBAM)'),
    (18.0, 13.5, 3.0, 0.6, 'Training'),
    (21.5, 13.5, 2.0, 0.6, 'Evaluation')
]

for x, y, w, h, text in stages:
    draw_stage_header(ax, x, y, w, h, text)

# ==============================================================================
# STAGE 1: DATA ACQUISITION & PREPROCESSING
# ==============================================================================

# Data input - more space
draw_box(ax, 0.8, 10.5, 1.8, 1.8, 'PediMS\nDataset\n(63 patients)', 
         color_data, fontsize=10, bold=True)

# MRI sequences
draw_box(ax, 0.8, 8.0, 1.8, 1.5, 'T1w\nT2w\nFLAIR', 
         color_data, fontsize=10)

# Preprocessing steps - more vertical space
draw_box(ax, 3.0, 11.0, 1.8, 1.0, 'N4 Bias\nCorrection', 
         color_preprocess, fontsize=9)
draw_box(ax, 3.0, 9.5, 1.8, 1.0, 'Registration\n(T1 space)', 
         color_preprocess, fontsize=9)
draw_box(ax, 3.0, 8.0, 1.8, 1.0, 'Intensity\nNormalization', 
         color_preprocess, fontsize=9)

# Arrows from data to preprocessing
draw_arrow(ax, 2.6, 11.3, 3.0, 11.5)
draw_arrow(ax, 2.6, 10.0, 3.0, 10.0)
draw_arrow(ax, 2.6, 8.7, 3.0, 8.5)

# Output to next stage
draw_arrow(ax, 4.8, 10.0, 5.5, 10.0)

# ==============================================================================
# STAGE 2: MAE PRETRAINING
# ==============================================================================

# MAE encoder - more space
draw_box(ax, 5.5, 10.5, 1.8, 1.8, 'MAE\nEncoder\n(3D ViT)', 
         color_model, fontsize=10, bold=True)

# Masking
draw_box(ax, 7.7, 11.6, 1.6, 0.8, 'Random\nMasking (75%)', 
         '#FFE4E1', fontsize=9)

# Decoder
draw_box(ax, 7.7, 10.5, 1.6, 0.8, 'MAE\nDecoder', 
         color_model, fontsize=9)

# Reconstruction
draw_box(ax, 7.7, 9.4, 1.6, 0.8, 'Reconstruct\nMasked Patches', 
         '#FFE4E1', fontsize=9)

# Arrows
draw_arrow(ax, 7.3, 11.4, 7.7, 12.0)
draw_arrow(ax, 7.3, 11.4, 7.7, 10.9)
draw_arrow(ax, 8.5, 10.5, 8.5, 10.2)

# Pretrained encoder to main model
draw_arrow(ax, 6.5, 10.0, 10.0, 10.0)
ax.text(8.3, 10.4, 'Pretrained\nWeights', ha='center', fontsize=9, 
        style='italic', color=color_arrow)

# ==============================================================================
# STAGE 3: MODEL ARCHITECTURE
# ==============================================================================

# Input - larger and more spaced
draw_box(ax, 10.0, 10.3, 1.6, 1.6, '3-Channel\nInput\n(64³)', 
         color_data, fontsize=10, bold=True)

# Encoder branch - more vertical spacing
draw_box(ax, 12.0, 11.5, 1.5, 1.0, 'Swin\nTransformer\nBlocks', 
         color_model, fontsize=9)
draw_box(ax, 12.0, 10.2, 1.5, 1.0, '3D Conv\n+ Residual', 
         color_model, fontsize=9)
draw_box(ax, 12.0, 8.9, 1.5, 1.0, 'Attention\nMechanism', 
         color_model, fontsize=9)

# Multi-scale features - more spacing
draw_box(ax, 13.8, 11.8, 0.9, 0.6, 'F₁\n(High)', 
         '#FFE4B5', fontsize=8)
draw_box(ax, 13.8, 11.0, 0.9, 0.6, 'F₂\n(Mid)', 
         '#FFE4B5', fontsize=8)
draw_box(ax, 13.8, 10.2, 0.9, 0.6, 'F₃\n(Mid)', 
         '#FFE4B5', fontsize=8)
draw_box(ax, 13.8, 9.4, 0.9, 0.6, 'F₄\n(Low)', 
         '#FFE4B5', fontsize=8)

# CBAM Module (central - highlighted) - LARGER
draw_box(ax, 15.0, 9.0, 2.2, 3.0, 'CBAM Fusion\nModule\n\n• Channel-wise\n  Attention\n• Spatial\n  Attention\n• Feature\n  Refinement', 
         '#FFD700', fontsize=9, bold=True)  # Gold color for emphasis

# Arrows to CBAM
draw_arrow(ax, 14.7, 12.1, 15.0, 11.5)
draw_arrow(ax, 14.7, 11.3, 15.0, 11.0)
draw_arrow(ax, 14.7, 10.5, 15.0, 10.5)
draw_arrow(ax, 14.7, 9.7, 15.0, 10.0)

# Decoder branch - more spacing
draw_box(ax, 17.5, 11.4, 1.3, 0.8, 'Upsample\nBlock 1', 
         color_model, fontsize=9)
draw_box(ax, 17.5, 10.3, 1.3, 0.8, 'Upsample\nBlock 2', 
         color_model, fontsize=9)
draw_box(ax, 17.5, 9.2, 1.3, 0.8, 'Upsample\nBlock 3', 
         color_model, fontsize=9)

# Arrows from CBAM to decoder
draw_arrow(ax, 17.2, 10.5, 17.5, 11.8)
draw_arrow(ax, 17.2, 10.5, 17.5, 10.7)
draw_arrow(ax, 17.2, 10.5, 17.5, 9.6)

# Output segmentation
draw_box(ax, 19.2, 10.2, 1.5, 1.5, 'Segmentation\nMap\n(64³)', 
         color_data, fontsize=10, bold=True)

draw_arrow(ax, 18.8, 10.7, 19.2, 10.9)

# Arrows through architecture
draw_arrow(ax, 11.6, 11.0, 12.0, 12.0)
draw_arrow(ax, 11.6, 11.0, 12.0, 10.7)
draw_arrow(ax, 11.6, 11.0, 12.0, 9.4)

# ==============================================================================
# STAGE 4: TRAINING
# ==============================================================================

# Loss function - more spacing
draw_box(ax, 18.2, 11.5, 1.6, 1.0, 'Dice + BCE\nLoss', 
         color_postprocess, fontsize=9, bold=True)

# Optimizer
draw_box(ax, 18.2, 10.2, 1.6, 1.0, 'AdamW\nLR=3e-4', 
         color_postprocess, fontsize=9)

# Training strategy
draw_box(ax, 18.2, 8.9, 1.6, 1.0, '5-Fold CV\n48 epochs', 
         color_postprocess, fontsize=9)

# Arrow from output to training
draw_arrow(ax, 20.7, 10.9, 18.2, 12.0)

# ==============================================================================
# STAGE 5: EVALUATION
# ==============================================================================

# Metrics - more vertical spacing
draw_box(ax, 21.6, 11.8, 1.5, 0.7, 'Dice: 82.31%', 
         color_eval, fontsize=9, bold=True)
draw_box(ax, 21.6, 10.9, 1.5, 0.7, 'Precision: 77.60%', 
         color_eval, fontsize=9)
draw_box(ax, 21.6, 10.0, 1.5, 0.7, 'Recall: 91.64%', 
         color_eval, fontsize=9)
draw_box(ax, 21.6, 9.1, 1.5, 0.7, 'F1: 84.04%', 
         color_eval, fontsize=9)

# Arrow to metrics
draw_arrow(ax, 19.8, 10.5, 21.6, 11.2)

# ==============================================================================
# BOTTOM SECTION: KEY INNOVATIONS & RESULTS
# ==============================================================================

# Innovation boxes - more spacing
innovations_y = 6.5
draw_box(ax, 0.8, innovations_y, 4.2, 1.5, 
         '✓ MAE Pretraining\n+3.5% Dice improvement\n3.6× faster convergence', 
         '#D4E9D7', fontsize=9, bold=True)

draw_box(ax, 5.3, innovations_y, 4.2, 1.5, 
         '✓ CBAM Fusion Module\n+0.48% vs No Fusion\nChannel + Spatial Attention', 
         '#FFE4B5', fontsize=9, bold=True)

draw_box(ax, 9.8, innovations_y, 4.2, 1.5, 
         '✓ Hybrid Architecture\nSwin Transformer + 3D Conv\nMulti-scale feature fusion', 
         '#F4E4F8', fontsize=9, bold=True)

draw_box(ax, 14.3, innovations_y, 4.2, 1.5, 
         '✓ Clinical Performance\n91.64% Sensitivity\nSuitable for screening', 
         '#E8F8F4', fontsize=9, bold=True)

draw_box(ax, 18.8, innovations_y, 4.4, 1.5, 
         '✓ State-of-the-Art Results\n+9.22% over baseline\n4.23M params, 1.75 GFLOPs', 
         '#FFE4E1', fontsize=9, bold=True)

# Model specifications - more spacing
specs_y = 4.3
ax.text(12, specs_y + 1.2, 'Model Specifications', 
        ha='center', fontsize=12, weight='bold')

draw_box(ax, 1.5, specs_y, 3.2, 0.8, 
         'Parameters: 4.23M\nEncoder: 85.6%', 
         '#F0F0F0', fontsize=8)

draw_box(ax, 5.0, specs_y, 3.2, 0.8, 
         'FLOPs: 1.75 GFLOPs\nConv: 78.4%', 
         '#F0F0F0', fontsize=8)

draw_box(ax, 8.5, specs_y, 3.2, 0.8, 
         'Inference: 490 ± 168 ms\nCPU throughput: 2.0 vol/s', 
         '#F0F0F0', fontsize=8)

draw_box(ax, 12.0, specs_y, 3.2, 0.8, 
         'Training: 28 epochs\nConvergence: Fast', 
         '#F0F0F0', fontsize=8)

draw_box(ax, 15.5, specs_y, 3.2, 0.8, 
         'Dataset: PediMS (63 patients)\nValidation: 5-Fold CV', 
         '#F0F0F0', fontsize=8)

draw_box(ax, 19.0, specs_y, 3.2, 0.8, 
         'Task: MS Lesion Segmentation\nPopulation: Pediatric', 
         '#F0F0F0', fontsize=8)

# Validation results - more spacing
validation_y = 2.3
ax.text(12, validation_y + 1.0, 'Cross-Dataset Validation', 
        ha='center', fontsize=12, weight='bold')

draw_box(ax, 1.5, validation_y, 3.8, 0.7, 
         'PediMS (In-domain): 82.31% ✓', 
         '#D4E9D7', fontsize=9, bold=True)

draw_box(ax, 5.6, validation_y, 3.8, 0.7, 
         'LGG (Cross-pathology): 20.01%', 
         '#FFE4B5', fontsize=9)

draw_box(ax, 9.7, validation_y, 3.8, 0.7, 
         'MS60 (Pediatric→Adult): 1.10%', 
         '#FFE4E1', fontsize=9)

draw_box(ax, 13.8, validation_y, 3.8, 0.7, 
         'Task-specific learning ✓', 
         '#E8F8F4', fontsize=9)

draw_box(ax, 17.9, validation_y, 3.8, 0.7, 
         'Population-specific model', 
         '#F4E4F8', fontsize=9)

# Footer - more spacing
ax.text(12, 1.0, 
        'HybridMiniSwin2.5D-CBAM: Deep Learning Framework for Pediatric MS Lesion Segmentation with MAE Pretraining and Channel-Spatial Attention',
        ha='center', fontsize=10, style='italic', color='#555555')

ax.text(12, 0.4, 
        'Comprehensive workflow from data preprocessing to clinical evaluation | Publication-ready visualization',
        ha='center', fontsize=9, color='#777777')

# Adjust layout and save
plt.tight_layout()
plt.savefig('C:/Users/HP/EDI/visualization/workflow_diagram_comprehensive.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('C:/Users/HP/EDI/paper_figures/main_figures/workflow_diagram_comprehensive.pdf', 
            bbox_inches='tight', facecolor='white')
print("✅ Workflow diagram saved!")
print("   - PNG: visualization/workflow_diagram_comprehensive.png")
print("   - PDF: paper_figures/main_figures/workflow_diagram_comprehensive.pdf")

plt.show()
