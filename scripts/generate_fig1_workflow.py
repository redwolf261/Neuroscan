"""
Generate Figure 1: Architecture Flow diagram (Publication-Grade 2-Row Layout).
Optimized for Word Document embedding & Q1 Journal printing (IEEE TMI / MedIA):
- 2-Row Balanced Layout (Row 1: Stages 1-4; Row 2: Stages 5-8).
- Clear, readable typography (10.5pt headers, 9.2pt body text, no micro-text).
- Aspect ratio ~1.44 (11.8 x 8.2 inches) fits standard 6.5-inch page margins with high legibility.
- Elegant inter-row connector carrying D1 latent features from Row 1 to Row 2.
- Direct bottom bypass carrying M_core from Stage 5 to Stage 8.
- High-resolution 350 DPI PNG and infinite-resolution Vector SVG.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Publication typography & styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9.2,
    'figure.dpi': 300,
    'savefig.dpi': 350,
    'savefig.bbox': 'tight'
})

# Curated palette
C_BASE      = '#334155'  # Slate-700
C_HUB       = '#1E3A8A'  # Deep Navy / Blue-900 (D1 Trunk)
C_CORE      = '#2563EB'  # Clinical Blue (Calibrated Core)
C_MRD       = '#D97706'  # Amber-600 (MRD Heads)
C_GATE      = '#DC2626'  # Red-600 (Dual-Zone Shell)
C_FINAL     = '#15803D'  # Emerald-700 (Final Composite Mask)

# Canvas: 11.8 x 8.2 inches (Aspect ratio ~1.44, ideal for portrait & landscape Word docs)
fig, ax = plt.subplots(figsize=(11.8, 8.2), facecolor='white')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')


def draw_rect_card(x, y, w, h, title, subtitle, header_color, border_color=None,
                   bg_color='white', title_color='white', sub_color='#0F172A',
                   title_fs=9.8, sub_fs=8.5, header_ratio=0.20):
    if border_color is None:
        border_color = header_color
        
    # Main container box
    box = patches.Rectangle(
        (x, y), w, h,
        ec=border_color, fc=bg_color, lw=1.4, zorder=2
    )
    ax.add_patch(box)
    
    # Header bar
    hh = h * header_ratio
    header = patches.Rectangle(
        (x, y + h - hh), w, hh,
        ec=border_color, fc=header_color, lw=1.4, zorder=3
    )
    ax.add_patch(header)
    
    # Header title text
    ax.text(x + w/2, y + h - hh/2, title, ha='center', va='center',
            color=title_color, weight='bold', fontsize=title_fs, zorder=4)
            
    # Subtitle / content lines
    ax.text(x + w/2, y + (h - hh)/2, subtitle, ha='center', va='center',
            color=sub_color, fontsize=sub_fs, zorder=4, multialignment='center',
            linespacing=1.28)


def draw_arrow(x1, y1, x2, y2, color='#475569', lw=2.0, style='->', rad=0.0, mutation_scale=13):
    connectionstyle = f"arc3,rad={rad}" if rad != 0 else "arc3"
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                mutation_scale=mutation_scale, connectionstyle=connectionstyle),
                zorder=5)


def draw_pill_badge(x, y, text, color, bg_color='#F8FAFC', fs=8.2, weight='bold'):
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, weight=weight, color=color,
            bbox=dict(boxstyle='square,pad=0.35', fc=bg_color, ec=color, lw=1.1),
            zorder=6)


# Main Figure Title
ax.text(3.0, 96.5, "Architecture Flow: End-to-End Decision Calibration & Residual Rescue Pipeline",
        fontsize=13.0, weight='bold', color='#0F172A', zorder=3)

# ------------------------------------------------------------------------------
# Geometry Parameters for 2-Row Grid
# ------------------------------------------------------------------------------
card_w = 21.2
card_h = 34.0
gap_x  = 4.2

# Row 1 (Top): Stages 1 -> 2 -> 3 -> 4
row1_y = 58.0
arrow1_y = row1_y + card_h * 0.50

x1 = 3.0
x2 = x1 + card_w + gap_x
x3 = x2 + card_w + gap_x
x4 = x3 + card_w + gap_x

# Card 1: Multimodal MRI
draw_rect_card(x1, row1_y, card_w, card_h,
               "1. Multimodal MRI",
               "Coregistered 3D Volumes:\n\n"
               "• T1 (Native Anatomy)\n\n"
               "• T1c (Post-Gadolinium)\n\n"
               "• T2 (T2-Weighted Fluid)\n\n"
               "• T2-FLAIR (Edema Zone)\n\n"
               "Input Grid: 4 × 240 × 240 × 155",
               C_BASE, title_fs=9.8, sub_fs=8.5, bg_color='#F8FAFC')

# Card 2: Preprocessing
draw_rect_card(x2, row1_y, card_w, card_h,
               "2. Preprocessing",
               "Standardization Pipeline:\n\n"
               "• SRI24 Atlas Alignment\n\n"
               "• 1.0 mm³ Isotropic Resampling\n\n"
               "• Skull Stripping & Brain BBox\n\n"
               "• Non-Zero Voxel Z-Score Norm\n\n"
               "• 128³ Subvolume Extraction",
               C_BASE, title_fs=9.8, sub_fs=8.5, bg_color='#F8FAFC')
draw_arrow(x1 + card_w, arrow1_y, x2, arrow1_y, color='#475569', lw=2.0)

# Card 3: UNet3D-v5 Backbone
draw_rect_card(x3, row1_y, card_w, card_h,
               "3. UNet3D-v5 Backbone",
               "Frozen Neural Encoder-Decoder:\n\n"
               "• 3D Residual Encoder Blocks\n\n"
               "• Bottleneck Attention Gate\n\n"
               "• Decoder Stages (dec3, dec2)\n\n"
               "• Full-Resolution dec1 Stage\n\n"
               "5.61M Parameters (Strictly Frozen)",
               C_BASE, title_fs=9.8, sub_fs=8.5, bg_color='#F8FAFC')
draw_arrow(x2 + card_w, arrow1_y, x3, arrow1_y, color='#475569', lw=2.0)

# Card 4: Shared Trunk D1
draw_rect_card(x4, row1_y, card_w, card_h,
               "4. Shared Trunk D1",
               "Latent Substrate Representation:\n\n"
               "• Extracted from dec1.relu2_out\n\n"
               "• Latent Tensor: 32 × 128³ Voxels\n\n"
               "• Preserves Fine-Grained Features\n"
               "  Lost by Production Readout\n\n"
               "Common Substrate for Both Heads",
               C_HUB, title_fs=9.8, sub_fs=8.5, bg_color='#EFF6FF')
draw_arrow(x3 + card_w, arrow1_y, x4, arrow1_y, color=C_HUB, lw=2.2)

# ------------------------------------------------------------------------------
# Inter-Row Transition Connector (Row 1 -> Row 2)
# Routes D1 from Card 4 (Top Right) down and across to Card 5 (Bottom Left)
# ------------------------------------------------------------------------------
row2_y = 10.0
arrow2_y = row2_y + card_h * 0.50
inter_y = (row1_y + (row2_y + card_h)) / 2.0  # Exactly 51.0

# Drop line from Card 4
ax.plot([x4 + card_w/2, x4 + card_w/2], [row1_y, inter_y], color=C_HUB, lw=2.2, zorder=4)
# Horizontal connecting rail across to Card 5
ax.plot([x4 + card_w/2, x1 + card_w/2], [inter_y, inter_y], color=C_HUB, lw=2.2, zorder=4)
# Down arrow into Card 5
draw_arrow(x1 + card_w/2, inter_y, x1 + card_w/2, row2_y + card_h, color=C_HUB, lw=2.2)

# Inter-row descriptive badge
draw_pill_badge(50.0, inter_y,
                "Shared Latent Features D1 (32 × 128³) passed to Primary Production & Residual Readouts",
                C_HUB, bg_color='#DBEAFE', fs=8.4)

# ------------------------------------------------------------------------------
# Row 2 (Bottom): Stages 5 -> 6 -> 7 -> 8
# ------------------------------------------------------------------------------

# Card 5: Calibrated Core (M_core)
draw_rect_card(x1, row2_y, card_w, card_h,
               "5. Calibrated Core",
               "Primary Boundary Calibration:\n\n"
               "• Linear Head:\n"
               "  P = σ(w_P · D1 + b_P)\n\n"
               "• 8-Fold Dihedral Symmetry TTA\n\n"
               "• Calibrated Threshold: τ* = 0.20\n\n"
               "• Hierarchical Gate: P_WT ≥ 0.50\n\n"
               "Yields Robust Core Mask M_core",
               C_CORE, title_fs=9.8, sub_fs=8.3, bg_color='#F0F9FF')

# Card 6: MRD Residual Readouts
draw_rect_card(x2, row2_y, card_w, card_h,
               "6. MRD Residuals",
               "Targeted Lesion Discovery:\n\n"
               "• 5 Frozen Linear Heads: w_R,s\n\n"
               "• Residual Map:\n"
               "  R_s = σ(w_R,s · D1 + b_R,s)\n\n"
               "• High-Specificity Cut: τ_R* = 0.9508\n\n"
               "• Residual Candidate Mask:\n"
               "  M_cand,s = (R_s ≥ τ_R*) \\ M_core",
               C_MRD, title_fs=9.8, sub_fs=8.3, bg_color='#FFFBEB')
draw_arrow(x1 + card_w, arrow2_y, x2, arrow2_y, color=C_MRD, lw=2.0)

# Card 7: Spatial Gating (2 mm Boundary Shell)
draw_rect_card(x3, row2_y, card_w, card_h,
               "7. Spatial Gating",
               "Dual-Zone Spatial Partitioning:\n\n"
               "• 6-Connected Components C_i\n\n"
               "• Distance d to M_core Surface\n\n"
               "• Zone 1 (d ≤ 2.0 mm): REJECT\n"
               "  (Eliminates 92% boundary jitter)\n\n"
               "• Zone 2 (d > 2.0 mm): ADMIT\n"
               "  (Adjudicates distant satellites)",
               C_GATE, title_fs=9.8, sub_fs=8.5, bg_color='#FEF2F2')
draw_arrow(x2 + card_w, arrow2_y, x3, arrow2_y, color=C_GATE, lw=2.0)

# Card 8: A2 Consensus & Final Mask
draw_rect_card(x4, row2_y, card_w, card_h,
               "8. A2 & Final Mask",
               "Multimodal Consensus Synthesis:\n\n"
               "• 27 Morpho-Evidence Features\n\n"
               "• A2 Classifier: P_acc ≥ 0.80\n\n"
               "• 5-Seed Majority Vote (≥ 3/5)\n\n"
               "• Composite Segmentation:\n"
               "  M_final = M_core ∪ M_accepted",
               C_FINAL, title_fs=9.8, sub_fs=8.5, bg_color='#F0FDF4')
draw_arrow(x3 + card_w, arrow2_y, x4, arrow2_y, color=C_FINAL, lw=2.0)

# ------------------------------------------------------------------------------
# Direct Core Preservation Track: Underneath Row 2 (Carries M_core 5 -> 8)
# ------------------------------------------------------------------------------
track_y = 3.6
ax.plot([x1 + card_w/2, x4 + card_w/2], [track_y, track_y], color=C_CORE, lw=2.0, zorder=4)
ax.plot([x1 + card_w/2, x1 + card_w/2], [row2_y, track_y], color=C_CORE, lw=2.0, zorder=4)
draw_arrow(x4 + card_w/2, track_y, x4 + card_w/2, row2_y, color=C_CORE, lw=2.0)

draw_pill_badge((x1 + x4 + card_w)/2, track_y,
                "Direct Core Preservation Track: M_core strictly preserved and unioned with accepted satellites M_accepted",
                C_CORE, bg_color='#EFF6FF', fs=7.8)

# Figure Footnote / Caption
fig.text(0.5, 0.004,
         "Figure 1. NeuroScan end-to-end architecture flow: Row 1 standardizes multimodal MRI and extracts shared full-resolution latent representations D1 (32 × 128³);\n"
         "Row 2 executes primary core calibration (M_core at τ* = 0.20), residual lesion discovery (τ_R* = 0.9508), 2.0 mm boundary shell exclusion, and 27-feature A2 consensus fusion.",
         ha='center', va='bottom', fontsize=8.0, style='italic', color='#334155')

plt.subplots_adjust(left=0.015, right=0.985, top=0.95, bottom=0.045)

# Save high-res PNG and vector SVG
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
