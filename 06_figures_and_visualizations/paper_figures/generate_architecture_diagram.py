"""
Generate updated architecture diagram for HybridMiniSwin2.5D-CBAM model
Based on current final_model.py (root directory)
Similar style to original diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# Set up the figure - landscape orientation like original
fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# Title
ax.text(10, 13.5, 'HybridMiniSwin2.5D-CBAM Architecture', 
        fontsize=18, fontweight='bold', ha='center')
ax.text(10, 13, 'Pediatric MS Lesion Segmentation Model', 
        fontsize=11, ha='center', style='italic', color='#555')

# Color scheme (clean, like original)
color_input = '#FFFFFF'
color_encoder = '#ADD8E6'  # Light blue
color_cbam = '#FFD580'     # Orange
color_decoder = '#90EE90'  # Light green  
color_output = '#FFB6C1'   # Pink
color_novel = '#FFF9C4'    # Light yellow

def draw_component_box(x, y, w, h, label, sublabel, color, border_color='black'):
    """Draw a clean component box like the original diagram"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         edgecolor=border_color, facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h*0.65, label, fontsize=10, ha='center', 
            va='center', fontweight='bold')
    if sublabel:
        ax.text(x + w/2, y + h*0.35, sublabel, fontsize=8, ha='center', 
                va='center', color='#333')

def draw_arrow_hor(x1, y1, x2, y2, label=''):
    """Horizontal arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                           color='black', linewidth=2, mutation_scale=15)
    ax.add_patch(arrow)
    if label:
        ax.text((x1+x2)/2, y1 + 0.2, label, fontsize=8, ha='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

def draw_arrow_vert(x1, y1, x2, y2, label=''):
    """Vertical arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                           color='black', linewidth=2, mutation_scale=15)
    ax.add_patch(arrow)
    if label:
        ax.text(x1 + 0.3, (y1+y2)/2, label, fontsize=8, va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

# INPUT at top
draw_component_box(8.5, 11.5, 3, 0.8, 'Input FLAIR', 
                   '(1, 1, 5, H, W)', color_input, '#2196F3')
draw_arrow_vert(10, 11.5, 10, 11, '')

# ADAPTIVE SLICE SELECTOR
draw_component_box(7.5, 9.8, 5, 1.2, 'Adaptive Slice Selector', 
                   '2.5D = 9 (k) slices\nLearned selection', color_novel, '#FF9800')
draw_arrow_vert(10, 9.8, 10, 9.3, '')

# ENCODER - Left side
encoder_x = 1.5
encoder_y_start = 8.5

# Encoder title
ax.text(encoder_x + 1.5, encoder_y_start + 0.5, 'Encoder', 
        fontsize=12, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=color_encoder, edgecolor='black', linewidth=2))

# Stage blocks
stages = [
    ('Stage 1', '4x', 'ResNet\n+\nSwin', '64 ch'),
    ('Stage 2', '4x', 'ResNet\n+\nSwin', '128 ch'),
    ('Stage 3', '4x', 'ResNet\n+\nSwin', '256 ch'),
    ('Stage 4', '4x', 'ResNet\n+\nSwin', '512 ch'),
]

stage_y = encoder_y_start - 0.5
skip_ys = []

for i, (stage, blocks, content, ch) in enumerate(stages):
    # Stage box
    box_y = stage_y - i*1.5
    draw_component_box(encoder_x, box_y, 1.2, 1, stage, f'{blocks}\n{ch}', 
                       color_encoder, 'black')
    
    # Content box
    draw_component_box(encoder_x + 1.4, box_y, 1.2, 1, content, '', 
                       color_encoder, 'black')
    
    skip_ys.append(box_y + 0.5)

# Arrow from input to encoder
draw_arrow_hor(8.5, 10.5, 4, 8.7, '')

# CBAM in middle
cbam_x = 8.5
cbam_y = 5.5
draw_component_box(cbam_x, cbam_y, 3, 1.5, 
                   'Channel-Spatial Fusion',
                   'CBAM Module\n(Channel + Spatial attention)', 
                   color_cbam, '#E65100')

# Arrow from encoder to CBAM
draw_arrow_hor(encoder_x + 2.6, 4.5, cbam_x, cbam_y + 0.75, '')

# DECODER - Right side
decoder_x = 16
decoder_y_start = 8.5

# Decoder title
ax.text(decoder_x + 1.2, decoder_y_start + 0.5, 'Decoder', 
        fontsize=12, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=color_decoder, edgecolor='black', linewidth=2))

# Up stages
up_stages = [
    ('Up 1', 'Upsample + Conv', '256 -> 128 ch'),
    ('Up 2', 'Upsample + Conv', '128 -> 64 ch'),
    ('Up 3', 'Upsample + Conv', '64 -> 32 ch'),
    ('Up 4', 'Upsample + Conv', '32 -> 1 ch'),
]

up_y = decoder_y_start - 0.5

for i, (stage, op, ch) in enumerate(up_stages):
    box_y = up_y - i*1.5
    draw_component_box(decoder_x, box_y, 2.4, 1, stage, f'{op}\n{ch}', 
                       color_decoder, 'black')

# Arrow from CBAM to decoder
draw_arrow_hor(cbam_x + 3, cbam_y + 0.75, decoder_x, 8, '')

# Skip connections (dashed lines)
for i, skip_y in enumerate(skip_ys[::-1]):  # Reverse to match decoder order
    up_idx = i
    if up_idx < len(up_stages):
        dec_y = up_y - up_idx*1.5 + 0.5
        ax.plot([encoder_x + 2.6, decoder_x], [skip_y, dec_y], 
                '--', color='#2196F3', linewidth=2, alpha=0.7)
        # Skip label
        ax.text((encoder_x + decoder_x)/2, (skip_y + dec_y)/2 - 0.2, 
                f'Skip {i+1}', fontsize=7, ha='center', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='#2196F3', alpha=0.8))

# OUTPUT at bottom
output_y = 1.5
draw_component_box(decoder_x - 1, output_y, 4.4, 1.2, 
                   'Segmentation Map',
                   '(1, 1, H, W)', color_output, '#E91E63')

# Arrow from decoder to output
draw_arrow_vert(decoder_x + 1.2, up_y - 3*1.5, decoder_x + 1.2, output_y + 1.2, '')

# Evidential head (if enabled)
evid_x = 12
evid_y = 1.5
draw_component_box(evid_x, evid_y, 3.5, 1.2,
                   'Evidential Head',
                   'Beta(α₁, α₂)\nUncertainty estimation',
                   '#E1BEE7', '#7B1FA2')

# Legend box
legend_x = 0.5
legend_y = 0.5
ax.text(legend_x, legend_y + 2.2, 'Components:', fontsize=10, fontweight='bold')

legend_items = [
    ('Input/Output', color_input, '#2196F3'),
    ('ResNet blocks (Conv + BN + ReLU)', color_encoder, 'black'),
    ('Transformer', '#E1F5FE', 'black'),
    ('Swin attention (W x W)', '#E1F5FE', 'black'),
]

y_offset = legend_y + 1.8
for label, color, edge in legend_items[:2]:
    box = Rectangle((legend_x, y_offset), 0.4, 0.25, 
                    facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(box)
    ax.text(legend_x + 0.6, y_offset + 0.125, label, fontsize=8, va='center')
    y_offset -= 0.35

# Model stats box
stats_x = 0.5
stats_y = 11
ax.text(stats_x, stats_y + 1.8, 'Model Statistics:', fontsize=10, fontweight='bold')
stats_text = [
    'Parameters: 4.23M (5 slices)',
    'Input: 2.5D FLAIR (k=9 slices)',
    'Architecture: Hybrid CNN-Transformer',
    'Training: PediMS Dataset (45 cases)',
    'Performance: 82.31% Dice Score',
]
y_off = stats_y + 1.5
for stat in stats_text:
    ax.text(stats_x, y_off, f'• {stat}', fontsize=8)
    y_off -= 0.3

# Hybrid components box (like original)
hybrid_x = 0.5
hybrid_y = 7.5
ax.text(hybrid_x, hybrid_y + 1.3, 'Hybrid Components:', fontsize=10, fontweight='bold')

# CNN box
ax.add_patch(Rectangle((hybrid_x, hybrid_y + 0.7), 0.8, 0.3, 
                       facecolor=color_encoder, edgecolor='black', linewidth=1.5))
ax.text(hybrid_x + 0.4, hybrid_y + 0.85, 'CNN', fontsize=8, ha='center', fontweight='bold')
ax.text(hybrid_x + 1, hybrid_y + 0.85, 'ResNet blocks (Conv + BN + ReLU)', fontsize=7, va='center')

# Transformer box  
ax.add_patch(Rectangle((hybrid_x, hybrid_y + 0.3), 0.8, 0.3,
                       facecolor='#E1F5FE', edgecolor='black', linewidth=1.5))
ax.text(hybrid_x + 0.4, hybrid_y + 0.45, 'Transformer', fontsize=8, ha='center', fontweight='bold')
ax.text(hybrid_x + 1.1, hybrid_y + 0.45, 'Swin attention (W=4x4/MSA)', fontsize=7, va='center')

# Footer
ax.text(10, 0.2, 'Architecture: HybridMiniSwin2.5D-CBAM (Optimal Configuration)', 
        fontsize=9, ha='center', style='italic', color='#666')

plt.tight_layout()
plt.savefig(r'C:\Users\HP\EDI\paper_figures\architecture_diagram_current.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(r'C:\Users\HP\EDI\paper_figures\architecture_diagram_current.pdf', 
            bbox_inches='tight', facecolor='white')
print("✅ Architecture diagrams saved:")
print("   - architecture_diagram_current.png (300 DPI)")
print("   - architecture_diagram_current.pdf")
plt.close()

print("\n📊 Diagram created successfully!")
print("   Current model: HybridMiniSwin2.5D-CBAM")
print("   Key features: Adaptive selection, CBAM fusion, Evidential uncertainty")
