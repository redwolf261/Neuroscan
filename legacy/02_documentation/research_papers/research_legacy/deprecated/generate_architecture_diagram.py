"""
Generate Architecture Diagram for HybridMiniSwin2.5D-CSRF Model
Clean flowchart-style architecture diagram inspired by standard deep learning papers
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
import matplotlib.lines as mlines
import numpy as np

# Style settings - cleaner, more professional
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'normal'

# Color scheme - cleaner, more professional
COLOR_INPUT = '#FFB6C1'      # Light pink for input
COLOR_ENCODER = '#FFD4A3'    # Light orange for encoder
COLOR_BOTTLENECK = '#D4A5FF' # Light purple for bottleneck/CSRF
COLOR_DECODER = '#A3D4FF'    # Light blue for decoder
COLOR_OUTPUT = '#B4E7B4'     # Light green for output
COLOR_SKIP = '#FFE5A3'       # Light yellow for skip
COLOR_TEXT = '#000000'       # Black text

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(8, 9.5, 'HybridMiniSwin2.5D-CSRF Architecture', 
        ha='center', va='top', fontsize=18, fontweight='bold')
ax.text(8, 9.1, 'Novel 2.5D Volumetric Segmentation with Cross-Slice Residual Fusion',
        ha='center', va='top', fontsize=11, style='italic', color='gray')

# ============================================================================
# INPUT
# ============================================================================
y_start = 8.0
x_input = 0.5

# Input volume
input_box = FancyBboxPatch((x_input, y_start), 1.2, 0.6, 
                           boxstyle="round,pad=0.05", 
                           facecolor=COLOR_INPUT, edgecolor='black', linewidth=2)
ax.add_patch(input_box)
ax.text(x_input + 0.6, y_start + 0.45, '3D MRI Volume', 
        ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(x_input + 0.6, y_start + 0.15, '(B, 1, D, H, W)', 
        ha='center', va='center', fontsize=7, family='monospace')

# ============================================================================
# 2.5D STEM
# ============================================================================
x_stem = x_input + 2.0

# Arrow from input to stem
arrow1 = FancyArrowPatch((x_input + 1.2, y_start + 0.3), (x_stem, y_start + 0.3),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow1)

# Stem box
stem_box = FancyBboxPatch((x_stem, y_start), 1.2, 0.6,
                          boxstyle="round,pad=0.05",
                          facecolor=COLOR_STEM, edgecolor='black', linewidth=2)
ax.add_patch(stem_box)
ax.text(x_stem + 0.6, y_start + 0.45, '2.5D Stem', 
        ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(x_stem + 0.6, y_start + 0.25, 'Conv2D(k×1, 32)', 
        ha='center', va='center', fontsize=7)
ax.text(x_stem + 0.6, y_start + 0.05, '64×64', 
        ha='center', va='center', fontsize=7, style='italic')

# ============================================================================
# ENCODER STAGES
# ============================================================================
x_encoder_start = x_stem + 1.5
y_encoder = y_start - 1.0

encoder_stages = [
    {'name': 'Stage 1', 'channels': 64, 'size': '32×32', 'blocks': 4},
    {'name': 'Stage 2', 'channels': 128, 'size': '16×16', 'blocks': 4},
    {'name': 'Stage 3', 'channels': 256, 'size': '8×8', 'blocks': 4},
    {'name': 'Stage 4', 'channels': 512, 'size': '4×4', 'blocks': 4},
]

# Arrow from stem to encoder
arrow2 = FancyArrowPatch((x_stem + 0.6, y_start), (x_stem + 0.6, y_encoder + 0.6),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow2)

encoder_positions = []
for i, stage in enumerate(encoder_stages):
    x_pos = x_encoder_start + i * 1.8
    
    # Stage box
    stage_box = FancyBboxPatch((x_pos, y_encoder), 1.4, 0.6,
                               boxstyle="round,pad=0.05",
                               facecolor=COLOR_ENCODER, edgecolor='black', linewidth=2)
    ax.add_patch(stage_box)
    
    ax.text(x_pos + 0.7, y_encoder + 0.48, stage['name'], 
            ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(x_pos + 0.7, y_encoder + 0.30, f"{stage['blocks']}× ResBlock", 
            ha='center', va='center', fontsize=7)
    ax.text(x_pos + 0.7, y_encoder + 0.15, f"C={stage['channels']}", 
            ha='center', va='center', fontsize=7)
    ax.text(x_pos + 0.7, y_encoder + 0.00, stage['size'], 
            ha='center', va='center', fontsize=7, style='italic')
    
    # Add attention indicator
    if True:  # All stages have Mini-Swin attention
        attn_circle = Circle((x_pos + 1.3, y_encoder + 0.5), 0.08, 
                             facecolor=COLOR_ATTENTION, edgecolor='purple', linewidth=1.5)
        ax.add_patch(attn_circle)
        ax.text(x_pos + 1.3, y_encoder + 0.5, '⊕', 
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    encoder_positions.append((x_pos + 0.7, y_encoder))
    
    # Arrows between stages
    if i < len(encoder_stages) - 1:
        arrow = FancyArrowPatch((x_pos + 1.4, y_encoder + 0.3), 
                               (x_pos + 1.8, y_encoder + 0.3),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='black')
        ax.add_patch(arrow)

# ============================================================================
# CSRF MODULE
# ============================================================================
x_csrf = x_encoder_start + 3.5 * 1.8
y_csrf = y_encoder - 1.2

# Arrow to CSRF
arrow_csrf = FancyArrowPatch((x_encoder_start + 6.5, y_encoder), 
                            (x_csrf + 0.8, y_csrf + 0.8),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
ax.add_patch(arrow_csrf)

# CSRF box
csrf_box = FancyBboxPatch((x_csrf, y_csrf), 1.6, 0.8,
                          boxstyle="round,pad=0.05",
                          facecolor=COLOR_CSRF, edgecolor='darkred', linewidth=2.5)
ax.add_patch(csrf_box)
ax.text(x_csrf + 0.8, y_csrf + 0.62, 'CSRF Module', 
        ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(x_csrf + 0.8, y_csrf + 0.43, 'Cross-Slice', 
        ha='center', va='center', fontsize=7)
ax.text(x_csrf + 0.8, y_csrf + 0.28, 'Residual Fusion', 
        ha='center', va='center', fontsize=7)
ax.text(x_csrf + 0.8, y_csrf + 0.08, 'C=512, k=5', 
        ha='center', va='center', fontsize=7, style='italic')

# ============================================================================
# DECODER STAGES
# ============================================================================
y_decoder = y_csrf - 1.4
decoder_stages = [
    {'name': 'Up 1', 'channels': 256, 'size': '8×8'},
    {'name': 'Up 2', 'channels': 128, 'size': '16×16'},
    {'name': 'Up 3', 'channels': 64, 'size': '32×32'},
    {'name': 'Up 4', 'channels': 32, 'size': '64×64'},
]

# Arrow from CSRF to decoder
arrow_to_decoder = FancyArrowPatch((x_csrf + 0.8, y_csrf), 
                                   (x_csrf + 0.8, y_decoder + 0.6),
                                   arrowstyle='->', mutation_scale=20, 
                                   linewidth=2, color='black')
ax.add_patch(arrow_to_decoder)

decoder_positions = []
for i, stage in enumerate(decoder_stages):
    x_pos = x_csrf + 0.1 - i * 1.8  # Move leftward
    
    # Decoder stage box
    dec_box = FancyBboxPatch((x_pos, y_decoder), 1.4, 0.6,
                            boxstyle="round,pad=0.05",
                            facecolor=COLOR_DECODER, edgecolor='black', linewidth=2)
    ax.add_patch(dec_box)
    
    ax.text(x_pos + 0.7, y_decoder + 0.45, stage['name'], 
            ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(x_pos + 0.7, y_decoder + 0.25, f"C={stage['channels']}", 
            ha='center', va='center', fontsize=7)
    ax.text(x_pos + 0.7, y_decoder + 0.05, stage['size'], 
            ha='center', va='center', fontsize=7, style='italic')
    
    decoder_positions.append((x_pos + 0.7, y_decoder))
    
    # Arrows between decoder stages
    if i < len(decoder_stages) - 1:
        arrow = FancyArrowPatch((x_pos, y_decoder + 0.3), 
                               (x_pos - 0.4, y_decoder + 0.3),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='black')
        ax.add_patch(arrow)

# ============================================================================
# SKIP CONNECTIONS
# ============================================================================
# Draw skip connections from encoder to decoder
skip_pairs = [
    (2, 1),  # Stage 3 -> Up 1
    (1, 2),  # Stage 2 -> Up 2
    (0, 3),  # Stage 1 -> Up 3
]

for enc_idx, dec_idx in skip_pairs:
    enc_x, enc_y = encoder_positions[enc_idx]
    dec_x, dec_y = decoder_positions[dec_idx]
    
    # Curved skip connection
    x_mid = (enc_x + dec_x) / 2
    y_mid = (enc_y + dec_y) / 2 + 0.5
    
    # Create curved path
    from matplotlib.path import Path
    import matplotlib.patches as patches
    
    verts = [
        (enc_x, enc_y + 0.6),
        (enc_x, enc_y + 1.0),
        (x_mid, y_mid),
        (dec_x, dec_y + 1.0),
        (dec_x, dec_y + 0.6),
    ]
    
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor=COLOR_SKIP, 
                              linewidth=2, linestyle='--', alpha=0.7)
    ax.add_patch(patch)
    
    # Add arrow at the end
    arrow_skip = FancyArrowPatch((dec_x, dec_y + 0.65), (dec_x, dec_y + 0.6),
                                arrowstyle='->', mutation_scale=15, 
                                linewidth=2, color=COLOR_SKIP, alpha=0.7)
    ax.add_patch(arrow_skip)

# ============================================================================
# OUTPUT
# ============================================================================
x_output = decoder_positions[-1][0] - 1.8
y_output = y_decoder

# Arrow to output
arrow_out = FancyArrowPatch((decoder_positions[-1][0] - 0.7, y_decoder + 0.3), 
                           (x_output + 1.2, y_output + 0.3),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='black')
ax.add_patch(arrow_out)

# Output box
output_box = FancyBboxPatch((x_output, y_output), 1.2, 0.6,
                           boxstyle="round,pad=0.05",
                           facecolor=COLOR_OUTPUT, edgecolor='black', linewidth=2)
ax.add_patch(output_box)
ax.text(x_output + 0.6, y_output + 0.45, 'Segmentation', 
        ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(x_output + 0.6, y_output + 0.25, 'Conv 1×1 + σ', 
        ha='center', va='center', fontsize=7)
ax.text(x_output + 0.6, y_output + 0.05, '(B, 1, H, W)', 
        ha='center', va='center', fontsize=7, family='monospace')

# ============================================================================
# DETAILED COMPONENT BOXES
# ============================================================================
# ResBlock Detail (bottom left)
y_detail = 1.5
x_detail = 0.5

detail_box = FancyBboxPatch((x_detail, y_detail), 3.5, 1.0,
                           boxstyle="round,pad=0.08",
                           facecolor='#F5F5F5', edgecolor='gray', linewidth=1.5)
ax.add_patch(detail_box)
ax.text(x_detail + 1.75, y_detail + 0.85, 'ResBlock (with Mini-Swin Attention)', 
        ha='center', va='center', fontsize=8, fontweight='bold')

detail_text = [
    'Conv 3×3 + BN + ReLU',
    'Conv 3×3 + BN',
    'Window Attention (7×7)',
    'Residual Add + ReLU'
]
for i, text in enumerate(detail_text):
    ax.text(x_detail + 0.15, y_detail + 0.6 - i*0.18, f'• {text}', 
            ha='left', va='center', fontsize=7)

# CSRF Detail (bottom center)
x_csrf_detail = x_detail + 4.0
csrf_detail_box = FancyBboxPatch((x_csrf_detail, y_detail), 3.5, 1.0,
                                boxstyle="round,pad=0.08",
                                facecolor='#FFF0F5', edgecolor='darkred', linewidth=1.5)
ax.add_patch(csrf_detail_box)
ax.text(x_csrf_detail + 1.75, y_detail + 0.85, 'CSRF Module Detail', 
        ha='center', va='center', fontsize=8, fontweight='bold')

csrf_text = [
    'Compute residuals: R_i = F_i - 0.5(F_{i-1}+F_{i+1})',
    'Fusion: F\'_i = F_i + α·R_i',
    'SE attention across k slices',
    'Output central slice features'
]
for i, text in enumerate(csrf_text):
    ax.text(x_csrf_detail + 0.15, y_detail + 0.6 - i*0.18, f'• {text}', 
            ha='left', va='center', fontsize=7)

# Model Stats (bottom right)
x_stats = x_csrf_detail + 4.0
stats_box = FancyBboxPatch((x_stats, y_detail), 3.5, 1.0,
                          boxstyle="round,pad=0.08",
                          facecolor='#F0F8FF', edgecolor='blue', linewidth=1.5)
ax.add_patch(stats_box)
ax.text(x_stats + 1.75, y_detail + 0.85, 'Model Statistics', 
        ha='center', va='center', fontsize=8, fontweight='bold')

stats_text = [
    'Parameters: 34.22M',
    'FLOPs: 1.75 GFLOPs',
    'Input: 64×64×64 volume',
    'Output: 64×64 segmentation'
]
for i, text in enumerate(stats_text):
    ax.text(x_stats + 0.15, y_detail + 0.6 - i*0.18, f'• {text}', 
            ha='left', va='center', fontsize=7, fontweight='bold' if i < 2 else 'normal')

# ============================================================================
# LEGEND
# ============================================================================
y_legend = 0.3
x_legend = 0.5

legend_items = [
    ('Input/Output', COLOR_INPUT),
    ('2.5D Processing', COLOR_STEM),
    ('Encoder Stages', COLOR_ENCODER),
    ('CSRF Module', COLOR_CSRF),
    ('Decoder Stages', COLOR_DECODER),
    ('Skip Connections', COLOR_SKIP),
    ('Attention', COLOR_ATTENTION),
]

ax.text(x_legend, y_legend, 'Legend:', ha='left', va='center', 
        fontsize=8, fontweight='bold')

for i, (label, color) in enumerate(legend_items):
    x_pos = x_legend + 1.5 + (i % 4) * 2.5
    y_pos = y_legend - 0.3 if i >= 4 else y_legend
    
    rect = Rectangle((x_pos, y_pos - 0.08), 0.25, 0.16, 
                     facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(x_pos + 0.35, y_pos, label, ha='left', va='center', fontsize=7)

# ============================================================================
# KEY FEATURES ANNOTATION
# ============================================================================
features_y = 0.8
ax.text(12.0, features_y + 0.3, 'Key Design Features:', 
        ha='left', va='top', fontsize=9, fontweight='bold')

features = [
    '✓ 2.5D: k=5 consecutive slices',
    '✓ ResNet: Critical for performance (-4.44% when removed)',
    '✓ Mini-Swin: Efficient window attention',
    '✓ CSRF: Cross-slice coherence',
    '✓ No 3D Conv: +2.36% improvement',
    '✓ MAE Pretrained: 3.6× faster convergence'
]

for i, feat in enumerate(features):
    ax.text(12.0, features_y - i*0.15, feat, ha='left', va='top', 
            fontsize=7, color='darkgreen' if '✓' in feat else 'black')

plt.tight_layout()
plt.savefig("C:/Users/HP/EDI/paper_figures/model_architecture_diagram.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig("C:/Users/HP/EDI/paper_figures/model_architecture_diagram.pdf", 
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Architecture diagram generated successfully!")
print("   Saved to: paper_figures/model_architecture_diagram.png/.pdf")
print("\nDiagram includes:")
print("  • Complete model architecture flow")
print("  • All encoder/decoder stages with dimensions")
print("  • CSRF module placement and function")
print("  • Skip connections visualization")
print("  • ResBlock and CSRF details")
print("  • Model statistics (34.22M params, 1.75 GFLOPs)")
print("  • Key design features from ablation study")
