"""
Architecture Visualization for HybridMiniSwin2D5_CSRF Model
============================================================
Generates a clean block-level diagram showing the model architecture.

Usage:
    python architecture_viz.py

Output:
    neuroscan_architecture.png - High-level block diagram
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent))
from final_model import HybridMiniSwin2D5_CSRF

# Try to import visualization libraries
try:
    from torchview import draw_graph
    TORCHVIEW_AVAILABLE = True
except ImportError:
    TORCHVIEW_AVAILABLE = False
    print("⚠️  torchview not available. Install with: pip install torchview")

try:
    import hiddenlayer as hl
    HIDDENLAYER_AVAILABLE = True
except ImportError:
    HIDDENLAYER_AVAILABLE = False
    print("⚠️  hiddenlayer not available. Install with: pip install hiddenlayer")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Install with: pip install matplotlib")

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("⚠️  graphviz not available (optional)")


def create_custom_architecture_diagram():
    """
    Create a custom high-level architecture diagram using matplotlib
    This doesn't require external graph libraries
    """
    
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib is required for custom diagram")
        return False
    
    fig, ax = plt.subplots(figsize=(18, 14), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'HybridMiniSwin2.5D-CSRF Architecture', 
           ha='center', fontsize=26, fontweight='bold')
    ax.text(5, 11, 'Pediatric MS Lesion Segmentation Model',
            ha='center', fontsize=14, style='italic', color='gray')
    
    # Colors
    input_color = '#E8F4F8'
    encoder_color = '#B3D9E6'
    csrf_color = '#FFE5B4'
    decoder_color = '#D4E6B5'
    output_color = '#FFB3BA'
    
    # Input
    input_box = FancyBboxPatch((4, 9.5), 2, 0.6, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=input_color, linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 9.8, 'Input FLAIR', ha='center', va='center', fontweight='bold', fontsize=14)
    ax.text(5, 9.55, '(1, 1, 5, H, W)', ha='center', va='center', fontsize=12)
    
    # Arrow down
    ax.annotate('', xy=(5, 9.5), xytext=(5, 9.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # ============ ENCODER ============
    encoder_y = 5.5
    
    # Encoder container
    encoder_container = FancyBboxPatch((0.5, encoder_y-0.3), 4, 3.4,
                                      boxstyle="round,pad=0.1",
                                      edgecolor='#2E86AB', facecolor='white', 
                                      linewidth=3, linestyle='--', alpha=0.3)
    ax.add_patch(encoder_container)
    ax.text(2.5, 8.9, 'Encoder', ha='center', fontsize=16, 
            fontweight='bold', color='#2E86AB')
    
    # Stem
    stem_box = FancyBboxPatch((1.5, 8.2), 2, 0.5,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor=encoder_color, linewidth=1.5)
    ax.add_patch(stem_box)
    ax.text(2.5, 8.45, 'Slice Attention Stem', ha='center', va='center', fontsize=12)
    ax.text(2.5, 8.25, '2.5D → 2D (32 ch)', ha='center', va='center', fontsize=10)
    
    ax.annotate('', xy=(2.5, 8.2), xytext=(2.5, 7.95),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Encoder stages with CNN + Transformer details
    stages = [
        ('Stage 1', '4× ResCSRF', '64 ch', 7.5),
        ('Stage 2', '4× ResCSRF', '128 ch', 6.8),
        ('Stage 3', '4× ResCSRF', '256 ch', 6.1),
        ('Stage 4', '4× ResCSRF', '512 ch', 5.4)
    ]
    
    for i, (name, blocks, channels, y_pos) in enumerate(stages):
        # Main stage box
        box = FancyBboxPatch((1.2, y_pos), 2.6, 0.5,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor=encoder_color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(1.5, y_pos+0.35, name, ha='left', va='center', fontsize=11, fontweight='bold')
        ax.text(3.5, y_pos+0.35, channels, ha='right', va='center', fontsize=10, style='italic')
        
        # Show CNN + Transformer components inside
        ax.text(1.5, y_pos+0.15, '4× [', ha='left', va='center', fontsize=9, color='darkblue')
        
        # CNN component
        cnn_box = FancyBboxPatch((1.65, y_pos+0.06), 0.55, 0.18,
                                boxstyle="round,pad=0.02",
                                edgecolor='#1E88E5', facecolor='#90CAF9', linewidth=1, alpha=0.8)
        ax.add_patch(cnn_box)
        ax.text(1.925, y_pos+0.15, 'ResNet', ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Plus sign
        ax.text(2.25, y_pos+0.15, '+', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Transformer component
        trans_box = FancyBboxPatch((2.35, y_pos+0.06), 0.55, 0.18,
                                  boxstyle="round,pad=0.02",
                                  edgecolor='#7B1FA2', facecolor='#CE93D8', linewidth=1, alpha=0.8)
        ax.add_patch(trans_box)
        ax.text(2.625, y_pos+0.15, 'Swin', ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax.text(2.95, y_pos+0.15, ']', ha='left', va='center', fontsize=9, color='darkblue')
        
        if y_pos > 5.4:
            ax.annotate('', xy=(2.5, y_pos), xytext=(2.5, y_pos+0.5),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Skip connections
    skip_x = [4.6, 4.8, 5.0, 5.2]
    skip_y = [7.75, 7.05, 6.35, 5.65]
    for i, (sx, sy) in enumerate(zip(skip_x, skip_y)):
        ax.annotate('', xy=(sx+1.5, sy+0.25), xytext=(sx-0.4, sy+0.25),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='#2E86AB', 
                                 linestyle='--', alpha=0.7))
        ax.text(sx+0.5, sy+0.4, f'Skip {i+1}', ha='center', fontsize=9, 
               color='#2E86AB', style='italic')
    
    # ============ CSRF ============
    csrf_box = FancyBboxPatch((1.2, 4.5), 2.6, 0.6,
                             boxstyle="round,pad=0.1",
                             edgecolor='#FF8C00', facecolor=csrf_color, linewidth=2)
    ax.add_patch(csrf_box)
    ax.text(2.5, 4.95, 'Channel-Spatial RF Fusion', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(2.5, 4.75, 'Multi-scale feature refinement', ha='center', va='center', fontsize=10)
    ax.text(2.5, 4.6, '(Channel SE + Spatial attention)', ha='center', va='center', 
           fontsize=9, style='italic')
    
    ax.annotate('', xy=(2.5, 5.4), xytext=(2.5, 5.1),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(2.5, 4.5), xytext=(2.5, 4.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # ============ DECODER ============
    decoder_y = 0.5
    
    # Decoder container
    decoder_container = FancyBboxPatch((5.5, decoder_y-0.3), 4, 3.4,
                                      boxstyle="round,pad=0.1",
                                      edgecolor='#4CAF50', facecolor='white',
                                      linewidth=3, linestyle='--', alpha=0.3)
    ax.add_patch(decoder_container)
    ax.text(7.5, 3.9, 'Decoder', ha='center', fontsize=16,
           fontweight='bold', color='#4CAF50')
    
    # Decoder stages
    decoder_stages = [
        ('Up 1', '256 → 256 ch', 3.4),
        ('Up 2', '128 → 128 ch', 2.7),
        ('Up 3', '64 → 64 ch', 2.0),
        ('Up 4', '32 → 32 ch', 1.3)
    ]
    
    # Connect CSRF to decoder
    ax.annotate('', xy=(6.2, 3.65), xytext=(3.8, 4.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black',
                              connectionstyle="arc3,rad=0.3"))
    
    for name, channels, y_pos in decoder_stages:
        box = FancyBboxPatch((6.2, y_pos), 2.6, 0.5,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor=decoder_color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(6.5, y_pos+0.35, name, ha='left', va='center', fontsize=11, fontweight='bold')
        ax.text(6.5, y_pos+0.15, 'Upsample + Conv', ha='left', va='center', fontsize=10)
        ax.text(8.5, y_pos+0.25, channels, ha='right', va='center', fontsize=10, style='italic')
        
        if y_pos > 1.3:
            ax.annotate('', xy=(7.5, y_pos), xytext=(7.5, y_pos+0.5),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Final conv
    final_box = FancyBboxPatch((6.7, 0.5), 1.6, 0.5,
                              boxstyle="round,pad=0.05",
                              edgecolor='black', facecolor=decoder_color, linewidth=1.5)
    ax.add_patch(final_box)
    ax.text(7.5, 0.8, 'Final Conv', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7.5, 0.6, '32 → 1 ch', ha='center', va='center', fontsize=10)
    
    ax.annotate('', xy=(7.5, 1.3), xytext=(7.5, 1.0),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(7.5, 0.5), xytext=(7.5, 0.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Output
    output_box = FancyBboxPatch((6.5, -0.3), 2, 0.6,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor=output_color, linewidth=2)
    ax.add_patch(output_box)
    ax.text(7.5, 0, 'Segmentation Map', ha='center', va='center', fontweight='bold', fontsize=14)
    ax.text(7.5, -0.2, '(1, 1, H, W)', ha='center', va='center', fontsize=12)
    
    # Legend
    legend_y = 10.3
    legend_elements = [
        ('Input/Output', input_color),
        ('Encoder', encoder_color),
        ('CSRF Module', csrf_color),
        ('Decoder', decoder_color)
    ]
    
    legend_x = 0.5
    for i, (label, color) in enumerate(legend_elements):
        box = FancyBboxPatch((legend_x + i*2.2, legend_y), 0.3, 0.2,
                            boxstyle="round,pad=0.02",
                            edgecolor='black', facecolor=color, linewidth=1)
        ax.add_patch(box)
        ax.text(legend_x + i*2.2 + 0.5, legend_y+0.1, label, 
               ha='left', va='center', fontsize=11)
    
    # Hybrid Architecture Legend (pushed further left)
    legend_y2 = 9.3
    ax.text(0.2, legend_y2, 'Hybrid Components:', ha='left', fontsize=12, fontweight='bold')
    
    # CNN component
    cnn_box = FancyBboxPatch((0.2, legend_y2-0.3), 0.5, 0.2,
                            boxstyle="round,pad=0.02",
                            edgecolor='#1E88E5', facecolor='#90CAF9', linewidth=1)
    ax.add_patch(cnn_box)
    ax.text(0.45, legend_y2-0.2, 'CNN', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(0.8, legend_y2-0.2, 'ResNet blocks (Conv + BN + ReLU)', ha='left', va='center', fontsize=9)
    
    # Transformer component
    trans_box = FancyBboxPatch((0.2, legend_y2-0.6), 0.5, 0.2,
                              boxstyle="round,pad=0.02",
                              edgecolor='#7B1FA2', facecolor='#CE93D8', linewidth=1)
    ax.add_patch(trans_box)
    ax.text(0.45, legend_y2-0.5, 'Transformer', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(0.8, legend_y2-0.5, 'Swin attention (W-MSA + SW-MSA)', ha='left', va='center', fontsize=9)
    
    # Detailed ResCSRF Block Inset (moved down to avoid overlap with input)
    inset_x, inset_y = 9.2, 4.5
    inset_w, inset_h = 0.7, 2.3
    
    # Inset background
    inset_bg = FancyBboxPatch((inset_x-0.12, inset_y-0.1), inset_w+0.24, inset_h+0.25,
                             boxstyle="round,pad=0.05",
                             edgecolor='darkblue', facecolor='white', linewidth=2.5, alpha=0.95)
    ax.add_patch(inset_bg)
    ax.text(inset_x+inset_w/2, inset_y+inset_h+0.1, 'ResCSRF Block', 
           ha='center', fontsize=10, fontweight='bold', color='darkblue')
    
    # Components in the block
    components = [
        ('Input', 2.2, 'white'),
        ('Conv 3×3', 2.0, '#90CAF9'),
        ('BN + ReLU', 1.8, '#90CAF9'),
        ('Conv 3×3', 1.6, '#90CAF9'),
        ('BN', 1.4, '#90CAF9'),
        ('Swin Attn', 1.1, '#CE93D8'),
        ('Add + ReLU', 0.8, 'white'),
        ('Output', 0.6, 'white')
    ]
    
    prev_y = None
    for label, rel_y, color in components:
        y = inset_y + rel_y
        box = FancyBboxPatch((inset_x+0.05, y-0.08), inset_w-0.1, 0.16,
                            boxstyle="round,pad=0.02",
                            edgecolor='black', facecolor=color, linewidth=1)
        ax.add_patch(box)
        ax.text(inset_x+inset_w/2, y, label, ha='center', va='center', fontsize=8, fontweight='bold')
        
        if prev_y is not None and label not in ['Input', 'Output']:
            ax.annotate('', xy=(inset_x+inset_w/2, y+0.08), xytext=(inset_x+inset_w/2, prev_y-0.08),
                       arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))
        prev_y = y
    
    # Skip connection
    ax.annotate('', xy=(inset_x-0.08, inset_y+0.7), xytext=(inset_x-0.08, inset_y+2.1),
               arrowprops=dict(arrowstyle='->', lw=1.8, color='red', linestyle='--'))
    ax.text(inset_x-0.08, inset_y+1.4, 'skip', ha='right', fontsize=8, color='red', rotation=90, fontweight='bold')
    
    # Model stats (bottom left corner)
    stats_text = """Model Statistics:
• Parameters: 34.2M
• Input: 2.5D FLAIR (5 slices)
• Output: 2D Segmentation
• Architecture: Hybrid CNN-Transformer
• Training: PediMS Dataset (45 cases)
• Performance: 84% Dice Score"""
    
    ax.text(0.2, 0.3, stats_text, ha='left', va='bottom', fontsize=10,
           family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', 
                                        alpha=0.3, edgecolor='gray', linewidth=1.5))
    
    plt.tight_layout()
    
    # Save
    output_path = Path('neuroscan_architecture.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved custom architecture diagram: {output_path}")
    
    plt.show()
    
    return True


def create_torchview_diagram(model, input_tensor):
    """Create diagram using torchview"""
    
    if not TORCHVIEW_AVAILABLE:
        return False
    
    print("\n🎨 Creating torchview diagram...")
    
    try:
        model_graph = draw_graph(
            model, 
            input_data=input_tensor,
            expand_nested=True,
            depth=3,  # Show 3 levels of hierarchy
            device='cpu',
            save_graph=True,
            filename='neuroscan_architecture_torchview',
            directory='.'
        )
        
        print("✓ Saved torchview diagram: neuroscan_architecture_torchview.png")
        return True
        
    except Exception as e:
        print(f"❌ torchview failed: {e}")
        return False


def create_hiddenlayer_diagram(model, input_tensor):
    """Create diagram using hiddenlayer"""
    
    if not HIDDENLAYER_AVAILABLE:
        return False
    
    print("\n🎨 Creating hiddenlayer diagram...")
    
    try:
        # Create transforms to clean up names
        transforms = [
            hl.transforms.Prune('Constant'),
            hl.transforms.Fold("Conv > BatchNorm > ReLU", "ConvBlock"),
            hl.transforms.Fold("Linear > ReLU", "LinearBlock"),
            hl.transforms.Rename(op=lambda op: {
                'encoder': 'Encoder',
                'decoder': 'Decoder', 
                'csrf': 'CSRF'
            }.get(op.split('.')[0], op))
        ]
        
        graph = hl.build_graph(model, input_tensor, transforms=transforms)
        graph.save('neuroscan_architecture_hiddenlayer', format='png')
        
        print("✓ Saved hiddenlayer diagram: neuroscan_architecture_hiddenlayer.png")
        return True
        
    except Exception as e:
        print(f"❌ hiddenlayer failed: {e}")
        return False


def print_model_summary(model):
    """Print a text-based model summary"""
    
    print("\n" + "="*80)
    print("MODEL SUMMARY: HybridMiniSwin2D5_CSRF")
    print("="*80)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size:           {total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    print("\nArchitecture Hierarchy:")
    print("-" * 80)
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name:20s} | {type(module).__name__:30s} | {params:>15,} params")
        
        # Show sub-modules for major components
        if name in ['encoder', 'csrf', 'decoder']:
            for sub_name, sub_module in module.named_children():
                sub_params = sum(p.numel() for p in sub_module.parameters())
                print(f"  ├─ {sub_name:16s} | {type(sub_module).__name__:28s} | {sub_params:>13,} params")
    
    print("="*80)


def main():
    """Main execution"""
    
    print("="*80)
    print("NEUROSCAN ARCHITECTURE VISUALIZATION")
    print("="*80)
    print("Model: HybridMiniSwin2D5_CSRF")
    print("Purpose: Pediatric MS Lesion Segmentation")
    print("="*80)
    
    # Initialize model
    print("\n📦 Loading model...")
    try:
        # Use the correct channels configuration [32, 64, 128, 256, 512]
        model = HybridMiniSwin2D5_CSRF(channels=[32, 64, 128, 256, 512])
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create dummy input
    print("\n🔧 Creating dummy input...")
    # Input shape: (batch, channels, depth, height, width)
    # For 2.5D: (1, 1, 5, 64, 64) - 5 neighboring slices
    input_tensor = torch.randn(1, 1, 5, 64, 64)
    print(f"✓ Input shape: {tuple(input_tensor.shape)}")
    
    # Print model summary
    print_model_summary(model)
    
    # Test forward pass
    print("\n🧪 Testing forward pass...")
    try:
        with torch.no_grad():
            output = model(input_tensor)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {tuple(output.shape)}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    success = False
    
    # Try custom matplotlib diagram first (most reliable)
    print("\n1. Custom Architecture Diagram")
    print("-" * 80)
    if create_custom_architecture_diagram():
        success = True
    
    # Try torchview
    if TORCHVIEW_AVAILABLE:
        print("\n2. TorchView Diagram")
        print("-" * 80)
        create_torchview_diagram(model, input_tensor)
    
    # Try hiddenlayer
    if HIDDENLAYER_AVAILABLE:
        print("\n3. HiddenLayer Diagram")
        print("-" * 80)
        create_hiddenlayer_diagram(model, input_tensor)
    
    print("\n" + "="*80)
    if success:
        print("✅ VISUALIZATION COMPLETE!")
        print("="*80)
        print("\n📁 Output files:")
        print("  • neuroscan_architecture.png (main diagram)")
        if TORCHVIEW_AVAILABLE:
            print("  • neuroscan_architecture_torchview.png")
        if HIDDENLAYER_AVAILABLE:
            print("  • neuroscan_architecture_hiddenlayer.png")
    else:
        print("⚠️  No visualizations could be created")
        print("="*80)
        print("\nTo install required packages:")
        print("  pip install matplotlib torchview hiddenlayer graphviz")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
