"""
Generate Figure 1: Architecture Flow diagram.
Strictly follows Q1 journal standards (IEEE TMI / Medical Image Analysis):
- Single Unified End-to-End Inference Flow:
  [1. 4-Modal MRI] -> [2. Preprocessing] -> [3. UNet3D-v5] -> [4. Shared Trunk D1] ->
  [5. Calibrated Core] -> [6. MRD Residuals] -> [7. Spatial Gating] -> [8. A2 & Final Mask]
- Crisp rectangular boxes with snug padding and zero excess whitespace.
- Clean headline: "Architecture Flow" (no brackets, no Phase A).
- Direct bottom runner carrying M_core from Step 5 to Step 8.
- Outputs at 300 DPI to figures/fig1_architecture_flow.png and figures/fig5_roadmap.png.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Curated publication color palette
C_BASE      = '#334155'  # Slate-700 (Preprocessing, Backbone)
C_HUB       = '#1E3A8A'  # Deep Navy / Blue-900 (Shared D1 Trunk)
C_CORE      = '#2563EB'  # Blue-600 (Production Core Path)
C_MRD       = '#D97706'  # Amber-600 (MRD Heads)
C_GATE      = '#DC2626'  # Red-600 (Dual-Zone Gating)
C_A2        = '#0D9488'  # Teal-600 (A2 Classifier & Consensus)
C_FINAL     = '#15803D'  # Emerald-700 (Final Composite Mask)

fig, ax = plt.subplots(figsize=(16.8, 4.6), facecolor='white')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# ==============================================================================
# Helper functions for drawing sharp rectangular cards and connection arrows
# ==============================================================================
def draw_rect_card(x, y, w, h, title, subtitle, header_color, border_color=None,
                   bg_color='white', title_color='white', sub_color='#1E293B',
                   title_fs=7.8, sub_fs=6.9, header_ratio=0.22):
    if border_color is None:
        border_color = header_color
        
    # Main container rectangle
    box = patches.Rectangle(
        (x, y), w, h,
        ec=border_color, fc=bg_color, lw=1.3, zorder=2
    )
    ax.add_patch(box)
    
    # Header rectangle
    hh = h * header_ratio
    header = patches.Rectangle(
        (x, y + h - hh), w, hh,
        ec=border_color, fc=header_color, lw=1.3, zorder=3
    )
    ax.add_patch(header)
    
    # Header text
    ax.text(x + w/2, y + h - hh/2, title, ha='center', va='center',
            color=title_color, weight='bold', fontsize=title_fs, zorder=4)
            
    # Body text (centered vertically in remaining body space)
    ax.text(x + w/2, y + (h - hh)/2, subtitle, ha='center', va='center',
            color=sub_color, fontsize=sub_fs, zorder=4, multialignment='center',
            linespacing=1.24)

def draw_arrow(x1, y1, x2, y2, color='#64748B', lw=1.6, style='->', rad=0.0, mutation_scale=11):
    connectionstyle = f"arc3,rad={rad}" if rad != 0 else "arc3"
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                mutation_scale=mutation_scale, connectionstyle=connectionstyle),
                zorder=5)

def draw_pill_badge(x, y, text, color, bg_color='#F1F5F9', fs=7.1):
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, weight='bold', color=color,
            bbox=dict(boxstyle='square,pad=0.28', fc=bg_color, ec=color, lw=1.0),
            zorder=6)

# ==============================================================================
# Title: Clean "Architecture Flow"
# ==============================================================================
ax.text(2.5, 93.5, "Architecture Flow", fontsize=11.5, weight='bold', color='#0F172A', zorder=3)

# ------------------------------------------------------------------------------
# 8 Sequential Stages in a Single Continuous Left-to-Right Chain
# Card width: 10.45, Card gap: 1.6, Y: 18.0 to 86.0 (height = 68.0)
# ------------------------------------------------------------------------------
card_w = 10.45
card_h = 68.0
card_y = 18.5
gap = 1.6
arrow_y = card_y + card_h * 0.44

# Card 1: 4-Modal MRI Input
x1 = 2.5
draw_rect_card(x1, card_y, card_w, card_h,
               "1. Multimodal MRI",
               "Input Scans:\n\n"
               "• T1 (native)\n\n"
               "• T1c (+contrast)\n\n"
               "• T2-weighted\n\n"
               "• T2-FLAIR\n\n"
               "4 coregistered\nanatomical channels",
               C_BASE, title_fs=7.6, sub_fs=6.9)

# Card 2: Preprocessing
x2 = x1 + card_w + gap
draw_rect_card(x2, card_y, card_w, card_h,
               "2. Preprocessing",
               "Standardization:\n\n"
               "• SRI24 atlas space\n\n"
               "• 1.0 mm³ isotropic\n\n"
               "• Skull stripping\n\n"
               "• Intensity z-norm\n\n"
               "• 128³ patch crop",
               C_BASE, title_fs=7.6, sub_fs=6.9)
draw_arrow(x1 + card_w, arrow_y, x2, arrow_y, color='#475569', lw=1.8)

# Card 3: UNet3D-v5 Backbone
x3 = x2 + card_w + gap
draw_rect_card(x3, card_y, card_w, card_h,
               "3. UNet3D-v5",
               "Frozen Backbone:\n\n"
               "• 3D Encoder\n\n"
               "• Residual blocks\n\n"
               "• Attention gate\n\n"
               "• Full decoder dec1\n\n"
               "5.61M parameters",
               C_BASE, title_fs=7.6, sub_fs=6.9)
draw_arrow(x2 + card_w, arrow_y, x3, arrow_y, color='#475569', lw=1.8)

# Card 4: Shared Trunk D1
x4 = x3 + card_w + gap
draw_rect_card(x4, card_y, card_w, card_h,
               "4. Shared Trunk D1",
               "Latent Representation:\n\n"
               "• dec1.relu2_out\n\n"
               "• Shape: 32 × 128³\n\n"
               "• Direct substrate for\n"
               "  both production\n"
               "  and residual readouts\n\n"
               "No extra features",
               C_HUB, title_fs=7.6, sub_fs=6.9,
               bg_color='#EFF6FF')
draw_arrow(x3 + card_w, arrow_y, x4, arrow_y, color=C_HUB, lw=2.0)

# Card 5: Calibrated Core (M_core)
x5 = x4 + card_w + gap
draw_rect_card(x5, card_y, card_w, card_h,
               "5. Calibrated Core",
               "Primary Segmentation:\n\n"
               "• Linear head: w_P\n"
               "  P = σ(w_P · D1 + b_P)\n\n"
               "• 8-fold reflection TTA\n\n"
               "• Core threshold: τ* = 0.20\n\n"
               "• M_core = { x : P ≥ τ* }\n\n"
               "Bulk tumor preserved",
               C_CORE, title_fs=7.6, sub_fs=6.9,
               bg_color='#F0F9FF')
draw_arrow(x4 + card_w, arrow_y, x5, arrow_y, color=C_CORE, lw=2.0)

# Card 6: MRD Residual Readouts
x6 = x5 + card_w + gap
draw_rect_card(x6, card_y, card_w, card_h,
               "6. MRD Residuals",
               "Residual Discovery:\n\n"
               "• 5 Frozen Linear Heads\n"
               "  R_s = σ(w_R,s · D1 + b_R,s)\n\n"
               "• Operating point: τ_R* = 0.9508\n\n"
               "• Residual Candidate Map:\n"
               "  M_cand,s = (R_s ≥ τ_R*) \\ M_core\n\n"
               "Excludes core tumor",
               C_MRD, title_fs=7.6, sub_fs=6.9,
               bg_color='#FFFBEB')
draw_arrow(x5 + card_w, arrow_y, x6, arrow_y, color=C_MRD, lw=2.0)

# Card 7: Spatial Gating (6-CC & 2 mm)
x7 = x6 + card_w + gap
draw_rect_card(x7, card_y, card_w, card_h,
               "7. Spatial Gating",
               "Dual-Zone Partitioning:\n\n"
               "• 6-Connected Components C_i\n\n"
               "• Distance d to M_core boundary\n\n"
               "• Zone 1 (d ≤ 2.0 mm): REJECT\n"
               "  (prevents dilation bleed)\n\n"
               "• Zone 2 (d > 2.0 mm): PASS\n"
               "  (adjudicate as satellite)",
               C_GATE, title_fs=7.6, sub_fs=6.9,
               bg_color='#FEF2F2')
draw_arrow(x6 + card_w, arrow_y, x7, arrow_y, color=C_GATE, lw=2.0)

# Card 8: A2 Consensus & Final Mask
x8 = x7 + card_w + gap
draw_rect_card(x8, card_y, card_w, card_h,
               "8. A2 & Final Mask",
               "Consensus Synthesis:\n\n"
               "• 27 Handcrafted Features\n"
               "  (morphology + intensity)\n\n"
               "• A2 Classifier: P_acc ≥ 0.80\n\n"
               "• 5-Seed Consensus (≥ 3/5)\n"
               "  Yields M_accepted\n\n"
               "• M_final = M_core ∪ M_acc",
               C_FINAL, title_fs=7.6, sub_fs=6.9,
               bg_color='#F0FDF4')
draw_arrow(x7 + card_w, arrow_y, x8, arrow_y, color=C_FINAL, lw=2.0)

# ------------------------------------------------------------------------------
# Bottom Core Preservation Track: Clean direct bypass under Cards 6 & 7
# ------------------------------------------------------------------------------
track_y = 7.5
# Horizontal runner bar
ax.plot([x5 + card_w/2, x8 + card_w/2], [track_y, track_y], color=C_CORE, lw=2.0, zorder=4)
# Downward connector from Card 5 to runner
ax.plot([x5 + card_w/2, x5 + card_w/2], [card_y, track_y], color=C_CORE, lw=2.0, zorder=4)
# Upward arrow from runner into Card 8
draw_arrow(x8 + card_w/2, track_y, x8 + card_w/2, card_y, color=C_CORE, lw=2.0)

# Badge on the runner bar
draw_pill_badge((x5 + x8 + card_w)/2, track_y,
                "Direct Core Preservation Track: M_core carried forward and unioned with accepted satellites M_accepted",
                C_CORE, bg_color='#EFF6FF', fs=7.1)

# Footnote caption
fig.text(0.5, 0.008,
         "Figure 1. NeuroScan architecture flow: preprocessed 4-modal MRI is encoded into the full-resolution latent representation D1, "
         "which sequentially drives primary core segmentation (M_core at τ* = 0.20), marginal recovery decomposition outside the core, "
         "2-mm spatial gating against dilation bleed, and 27-feature A2 consensus, yielding the composite mask M_final = M_core ∪ M_accepted.",
         ha='center', va='bottom', fontsize=7.6, style='italic', color='#334155')

plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04)

# Save to both locations as 350 DPI 4K+ PNG and Vector SVG
out1 = FIG_DIR / 'fig1_architecture_flow.png'
out1_svg = FIG_DIR / 'fig1_architecture_flow.svg'
out2 = FIG_DIR / 'fig5_roadmap.png'
out2_svg = FIG_DIR / 'fig5_roadmap.svg'
plt.savefig(out1, dpi=350, bbox_inches='tight')
plt.savefig(out1_svg, bbox_inches='tight')
plt.savefig(out2, dpi=350, bbox_inches='tight')
plt.savefig(out2_svg, bbox_inches='tight')
plt.close()

print(f"Generated Figure 1 successfully at:\n  {out1} & {out1_svg}\n  {out2} & {out2_svg}")

