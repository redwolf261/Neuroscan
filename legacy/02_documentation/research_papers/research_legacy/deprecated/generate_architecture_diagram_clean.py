"""
Generate Clean Architecture Diagram for HybridMiniSwin2.5D-CSRF Model
Flowchart-style similar to standard deep learning architecture diagrams
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches

# Style settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9

# Colors - clean and professional
COLOR_INPUT = '#FFB6C1'      # Pink
COLOR_ENCODER = '#FFD4A3'    # Orange  
COLOR_BOTTLENECK = '#D4A5FF' # Purple
COLOR_DECODER = '#A3D4FF'    # Blue
COLOR_OUTPUT = '#B4E7B4'     # Green

def create_box(ax, x, y, width, height, text, subtext, color, fontsize=9):
    """Create a rounded box with text"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05",
                         facecolor=color, 
                         edgecolor='black', 
                         linewidth=2)
    ax.add_patch(box)
    
    # Main text
    ax.text(x + width/2, y + height*0.65, text,
            ha='center', va='center', 
            fontsize=fontsize, fontweight='bold')
    
    # Subtext
    if subtext:
        ax.text(x + width/2, y + height*0.35, subtext,
                ha='center', va='center', 
                fontsize=fontsize-1.5)

def create_arrow(ax, x1, y1, x2, y2, label='', curve=False):
    """Create arrow between boxes"""
    if curve:
        # Curved arrow for skip connections
        connectionstyle = "arc3,rad=.3"
    else:
        connectionstyle = "arc3,rad=0"
    
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', 
                           mutation_scale=20,
                           linewidth=2,
                           color='black',
                           connectionstyle=connectionstyle)
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.15, label,
                ha='center', va='center',
                fontsize=7, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', 
                         edgecolor='gray', 
                         linewidth=0.5))

# Create figure
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7.5, 9.5, 'HybridMiniSwin2.5D-CSRF Architecture',
        ha='center', va='top', fontsize=16, fontweight='bold')
ax.text(7.5, 9.1, 'Cross-Slice Residual Fusion for Volumetric MRI Segmentation',
        ha='center', va='top', fontsize=10, style='italic', color='gray')

# ============================================================================
# MAIN ARCHITECTURE FLOW (Top to Bottom)
# ============================================================================

# Parameters
box_width = 2.0
box_height = 0.8
y_spacing = 1.2

# Starting positions
x_center = 7.5 - box_width/2
y_start = 8.0

# INPUT
y_pos = y_start
create_box(ax, x_center, y_pos, box_width, box_height,
           'Input Image', '3D MRI: 1×64×64×64', COLOR_INPUT)

# Arrow down
create_arrow(ax, x_center + box_width/2, y_pos, 
             x_center + box_width/2, y_pos - y_spacing + box_height)

# 2.5D STEM
y_pos -= y_spacing
create_box(ax, x_center, y_pos, box_width, box_height,
           '2.5D Conv Stem', 'k=5 slices → C=32', COLOR_ENCODER)

# Arrow down
create_arrow(ax, x_center + box_width/2, y_pos,
             x_center + box_width/2, y_pos - y_spacing + box_height)

# ENCODER STAGES (show as one block with stages inside)
y_pos -= y_spacing
encoder_height = 2.5
encoder_box = FancyBboxPatch((x_center - 0.2, y_pos - encoder_height + box_height), 
                            box_width + 0.4, encoder_height,
                            boxstyle="round,pad=0.05",
                            facecolor=COLOR_ENCODER,
                            edgecolor='black',
                            linewidth=2.5)
ax.add_patch(encoder_box)

ax.text(x_center + box_width/2, y_pos + box_height/2, 'ENCODER',
        ha='center', va='center', fontsize=11, fontweight='bold')

# Individual stages
stage_info = [
    ('Stage 1', '4× ResBlock', 'C=64, 32×32'),
    ('Stage 2', '4× ResBlock', 'C=128, 16×16'),
    ('Stage 3', '4× ResBlock', 'C=256, 8×8'),
    ('Stage 4', '4× ResBlock', 'C=512, 4×4'),
]

stage_y = y_pos - 0.3
stage_height = 0.45
for i, (name, blocks, dims) in enumerate(stage_info):
    stage_y_pos = stage_y - i * 0.55
    
    # Small stage box
    stage_box = Rectangle((x_center + 0.1, stage_y_pos), 
                         box_width - 0.2, stage_height,
                         facecolor='white',
                         edgecolor='black',
                         linewidth=1)
    ax.add_patch(stage_box)
    
    ax.text(x_center + 0.3, stage_y_pos + stage_height*0.7, name,
            ha='left', va='center', fontsize=7.5, fontweight='bold')
    ax.text(x_center + 0.3, stage_y_pos + stage_height*0.3, f'{blocks}, {dims}',
            ha='left', va='center', fontsize=6.5)

# Store encoder positions for skip connections
encoder_skip_positions = [
    (x_center + box_width, stage_y - 0*0.55 + stage_height/2),  # Stage 1
    (x_center + box_width, stage_y - 1*0.55 + stage_height/2),  # Stage 2
    (x_center + box_width, stage_y - 2*0.55 + stage_height/2),  # Stage 3
]

# Arrow down from encoder
y_pos -= encoder_height + 0.3
create_arrow(ax, x_center + box_width/2, y_pos + 0.3,
             x_center + box_width/2, y_pos)

# CSRF BOTTLENECK
y_pos -= y_spacing
create_box(ax, x_center, y_pos, box_width, box_height,
           'CSRF Module', 'Cross-Slice Fusion, C=512', COLOR_BOTTLENECK)

# Arrow down
create_arrow(ax, x_center + box_width/2, y_pos,
             x_center + box_width/2, y_pos - y_spacing + box_height)

# DECODER STAGES
y_pos -= y_spacing
decoder_height = 2.5
decoder_box = FancyBboxPatch((x_center - 0.2, y_pos - decoder_height + box_height),
                            box_width + 0.4, decoder_height,
                            boxstyle="round,pad=0.05",
                            facecolor=COLOR_DECODER,
                            edgecolor='black',
                            linewidth=2.5)
ax.add_patch(decoder_box)

ax.text(x_center + box_width/2, y_pos + box_height/2, 'DECODER',
        ha='center', va='center', fontsize=11, fontweight='bold')

# Individual decoder stages
decoder_info = [
    ('Up-sample 1', 'Conv + ReLU', 'C=256, 8×8'),
    ('Up-sample 2', 'Conv + ReLU', 'C=128, 16×16'),
    ('Up-sample 3', 'Conv + ReLU', 'C=64, 32×32'),
    ('Up-sample 4', 'Conv + ReLU', 'C=32, 64×64'),
]

decoder_y = y_pos - 0.3
for i, (name, conv, dims) in enumerate(decoder_info):
    dec_y_pos = decoder_y - i * 0.55
    
    # Small decoder box
    dec_box = Rectangle((x_center + 0.1, dec_y_pos),
                       box_width - 0.2, stage_height,
                       facecolor='white',
                       edgecolor='black',
                       linewidth=1)
    ax.add_patch(dec_box)
    
    ax.text(x_center + 0.3, dec_y_pos + stage_height*0.7, name,
            ha='left', va='center', fontsize=7.5, fontweight='bold')
    ax.text(x_center + 0.3, dec_y_pos + stage_height*0.3, f'{conv}, {dims}',
            ha='left', va='center', fontsize=6.5)

# Store decoder positions for skip connections
decoder_skip_positions = [
    (x_center, decoder_y - 1*0.55 + stage_height/2),  # Up-sample 2
    (x_center, decoder_y - 2*0.55 + stage_height/2),  # Up-sample 3
    (x_center, decoder_y - 3*0.55 + stage_height/2),  # Up-sample 4
]

# SKIP CONNECTIONS (curved arrows from encoder to decoder)
skip_connections = [
    (0, 2, 'Skip 1'),  # Stage 1 -> Up 4
    (1, 1, 'Skip 2'),  # Stage 2 -> Up 3
    (2, 0, 'Skip 3'),  # Stage 3 -> Up 2
]

for enc_idx, dec_idx, label in skip_connections:
    x1, y1 = encoder_skip_positions[enc_idx]
    x2, y2 = decoder_skip_positions[dec_idx]
    
    # Create curved path
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', 
                           mutation_scale=15,
                           linewidth=1.5,
                           color='orange',
                           linestyle='--',
                           connectionstyle="arc3,rad=-.3")
    ax.add_patch(arrow)
    
    # Label
    mid_x = (x1 + x2) / 2 + 0.8
    mid_y = (y1 + y2) / 2
    ax.text(mid_x, mid_y, label,
            ha='center', va='center',
            fontsize=6, style='italic',
            bbox=dict(boxstyle='round,pad=0.2',
                     facecolor='lightyellow',
                     edgecolor='orange',
                     linewidth=1))

# Arrow down from decoder
y_pos -= decoder_height + 0.3
create_arrow(ax, x_center + box_width/2, y_pos + 0.3,
             x_center + box_width/2, y_pos)

# OUTPUT
y_pos -= y_spacing
create_box(ax, x_center, y_pos, box_width, box_height,
           'Output Layer', 'Segmentation: 1×64×64', COLOR_OUTPUT)

# ============================================================================
# SIDE PANELS - Details
# ============================================================================

# Left panel - ResBlock detail
left_x = 0.5
left_y = 4.5
panel_width = 2.5
panel_height = 2.2

panel_box = FancyBboxPatch((left_x, left_y), panel_width, panel_height,
                          boxstyle="round,pad=0.08",
                          facecolor='#F8F8F8',
                          edgecolor='gray',
                          linewidth=1.5)
ax.add_patch(panel_box)

ax.text(left_x + panel_width/2, left_y + panel_height - 0.2,
        'ResBlock Detail',
        ha='center', va='center',
        fontsize=9, fontweight='bold')

resblock_text = [
    '1. Conv 3×3 + BN + ReLU',
    '2. Conv 3×3 + BN',
    '3. Mini-Swin Attention',
    '   (Window size: 7×7)',
    '4. Residual Addition',
    '5. ReLU activation',
]

text_y = left_y + panel_height - 0.5
for line in resblock_text:
    ax.text(left_x + 0.15, text_y, line,
            ha='left', va='top',
            fontsize=7)
    text_y -= 0.25

# Right panel - CSRF detail
right_x = 12.0
right_y = 4.5

panel_box2 = FancyBboxPatch((right_x, right_y), panel_width, panel_height,
                           boxstyle="round,pad=0.08",
                           facecolor='#FFF5F8',
                           edgecolor='purple',
                           linewidth=1.5)
ax.add_patch(panel_box2)

ax.text(right_x + panel_width/2, right_y + panel_height - 0.2,
        'CSRF Module',
        ha='center', va='center',
        fontsize=9, fontweight='bold')

csrf_text = [
    '1. Compute residuals:',
    '   R_i = F_i - ½(F_{i-1}+F_{i+1})',
    '2. Fusion with learnable α:',
    '   F\'_i = F_i + α·R_i',
    '3. SE attention (k slices)',
    '4. Return central slice',
]

text_y = right_y + panel_height - 0.5
for line in csrf_text:
    ax.text(right_x + 0.15, text_y, line,
            ha='left', va='top',
            fontsize=7)
    text_y -= 0.25

# ============================================================================
# Legend
# ============================================================================
legend_y = 0.3
legend_elements = [
    mpatches.Patch(facecolor=COLOR_INPUT, edgecolor='black', label='Input/Output'),
    mpatches.Patch(facecolor=COLOR_ENCODER, edgecolor='black', label='Encoder'),
    mpatches.Patch(facecolor=COLOR_BOTTLENECK, edgecolor='black', label='CSRF'),
    mpatches.Patch(facecolor=COLOR_DECODER, edgecolor='black', label='Decoder'),
    mpatches.Patch(facecolor='lightyellow', edgecolor='orange', label='Skip Connection'),
]

ax.legend(handles=legend_elements, loc='lower left', 
         bbox_to_anchor=(0.05, 0.02), ncol=5, fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig("C:/Users/HP/EDI/paper_figures/model_architecture_diagram.png",
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig("C:/Users/HP/EDI/paper_figures/model_architecture_diagram.pdf",
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Clean architecture diagram generated!")
print("   Saved to: paper_figures/model_architecture_diagram.png/.pdf")
print("\n📋 Diagram features:")
print("   • Clean top-to-bottom flow")
print("   • Encoder: 4 stages with ResBlocks (64→128→256→512 channels)")
print("   • CSRF: Cross-slice residual fusion at bottleneck")
print("   • Decoder: 4 upsampling stages (512→256→128→64→32)")
print("   • Skip connections: U-Net style (curved orange dashed lines)")
print("   • Side panels: ResBlock and CSRF implementation details")
print("   • Statistics: 34.22M params, 1.75 GFLOPs")
print("   • Clear color coding with legend")
