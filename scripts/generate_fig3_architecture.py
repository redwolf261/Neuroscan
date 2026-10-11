"""
Generate Figure 3: Detailed Architecture of NeuroScan.
Strictly follows Q1 journal standards (IEEE TMI / Medical Image Analysis):
- PURE SCIENTIFIC DIAGRAMS: Zero explanatory text boxes, zero paragraphs, zero marketing slogans.
- Panel (a): UNet3D-v5 Trunk Architecture (Input MRI -> Encoders -> Bottleneck -> Attention Gate -> Decoders -> D1).
- Panel (b): Dual Readouts & Candidate Adjudication Pipeline:
  - Shared D1 trunk splits into Production Head (w_P) and MRD Residual Head (w_R).
  - Production Head produces calibrated core M_core (left stream).
  - MRD Head feeds into Candidate Adjudication steps (right stream: 6-CC -> 2-mm Gate -> 27-Feat -> A2 -> Consensus -> M_rescued).
  - Streams converge into Final ET Mask: M_final = M_core U M_rescued.
- Completely removed the 2D latent space scatter plot block as requested.
Outputs at 300 DPI to figures/fig3_neuroscan_architecture.png.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Color palette
C_NAVY     = '#0F172A'       # Slate 900
C_SLATE    = '#1E293B'       # Slate 800
C_MUTED    = '#64748B'       # Slate 500
C_BORDER   = '#CBD5E1'       # Slate 300
C_CARD_BG  = '#F8FAFC'       # Slate 50

C_ENC      = '#2563EB'       # Blue 600
C_BOT      = '#7C3AED'       # Purple 600
C_DEC      = '#0D9488'       # Teal 600
C_D1       = '#047857'       # Emerald 700
C_PROD     = '#1D4ED8'       # Deep Blue (Production)
C_MRD      = '#D97706'       # Warm Amber (MRD)
C_REJECT   = '#DC2626'       # Red 600
C_ACCEPT   = '#16A34A'       # Green 600

def draw_rounded_box(ax, x, y, w, h, title, subtitle="", fc='#FFFFFF', ec=C_BORDER, lw=1.1,
                     radius=0.012, title_color=C_SLATE, sub_color=C_MUTED, title_size=8.4, sub_size=7.0,
                     title_weight='bold'):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(box)
    
    if subtitle:
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center',
                fontsize=title_size, weight=title_weight, color=title_color, zorder=3)
        ax.text(x + w/2, y + h*0.30, subtitle, ha='center', va='center',
                fontsize=sub_size, color=sub_color, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                fontsize=title_size, weight=title_weight, color=title_color, zorder=3)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color=C_MUTED, lw=1.1, style='-|>', rad=0.0):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, mutation_scale=8.5,
                            connectionstyle=f"arc3,rad={rad}",
                            color=color, linewidth=lw, zorder=4)
    ax.add_patch(arrow)
    return arrow

def generate_fig3():
    fig_w = 15.2
    fig_h = 8.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='#FFFFFF')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # --------------------------------------------------------------------------
    # Panel Layout: Two Master Cards
    # Left:  Panel (a) UNet3D-v5 Trunk Architecture
    # Right: Panel (b) Dual Readouts & Candidate Adjudication Pipeline
    # --------------------------------------------------------------------------
    panel_y = 0.030
    panel_h = 0.940
    
    pA_x, pA_w = 0.016, 0.430
    pB_x, pB_w = 0.464, 0.520
    
    # Panel (a) Card
    card_a = FancyBboxPatch((pA_x, panel_y), pA_w, panel_h,
                            boxstyle="round,pad=0,rounding_size=0.012",
                            facecolor=C_CARD_BG, edgecolor='#E2E8F0', linewidth=1.1, zorder=1)
    ax.add_patch(card_a)
    ax.text(pA_x + 0.014, panel_y + panel_h - 0.028, "(a) UNet3D-v5 Trunk Architecture",
            ha='left', va='center', fontsize=9.8, weight='bold', color=C_NAVY, zorder=3)
    ax.plot([pA_x + 0.010, pA_x + pA_w - 0.010], [panel_y + panel_h - 0.046, panel_y + panel_h - 0.046],
            color='#E2E8F0', lw=0.9, zorder=2)
    
    # Panel (b) Card
    card_b = FancyBboxPatch((pB_x, panel_y), pB_w, panel_h,
                            boxstyle="round,pad=0,rounding_size=0.012",
                            facecolor=C_CARD_BG, edgecolor='#E2E8F0', linewidth=1.1, zorder=1)
    ax.add_patch(card_b)
    ax.text(pB_x + 0.014, panel_y + panel_h - 0.028,
            r"(b) Shared Representation $D_1$ Readouts & Candidate Adjudication",
            ha='left', va='center', fontsize=9.8, weight='bold', color=C_NAVY, zorder=3)
    ax.plot([pB_x + 0.010, pB_x + pB_w - 0.010], [panel_y + panel_h - 0.046, panel_y + panel_h - 0.046],
            color='#E2E8F0', lw=0.9, zorder=2)

    # ==========================================================================
    # PANEL A: UNet3D-v5 Architecture
    # ==========================================================================
    enc_x = pA_x + 0.024
    dec_x = pA_x + pA_w - 0.164
    box_w = 0.140
    box_h = 0.062
    
    # 1. Input Box
    in_box_y = panel_y + panel_h - 0.100
    draw_rounded_box(ax, enc_x, in_box_y, box_w, box_h,
                     "Input MRI", "4 × 128³ (T1, T1c, T2, FLAIR)",
                     fc='#EFF6FF', ec=C_ENC, lw=1.2, title_color=C_ENC, title_size=8.5, sub_size=6.8)
    
    # 2. Encoder Stages
    enc1_y = in_box_y - 0.135
    enc2_y = enc1_y - 0.185
    enc3_y = enc2_y - 0.185
    
    # Input -> Enc1
    draw_arrow(ax, enc_x + box_w/2, in_box_y, enc_x + box_w/2, enc1_y + box_h, color=C_ENC)
    ax.text(enc_x + box_w/2 + 0.006, (in_box_y + enc1_y + box_h)/2, "Conv3D 3×3×3",
            fontsize=6.8, color='#3B82F6', va='center')
    
    draw_rounded_box(ax, enc_x, enc1_y, box_w, box_h,
                     "Encoder 1", "32 ch × 128³",
                     fc='#FFFFFF', ec=C_ENC, lw=1.2)
    
    # Enc1 -> Enc2
    draw_arrow(ax, enc_x + box_w/2, enc1_y, enc_x + box_w/2, enc2_y + box_h, color=C_ENC)
    ax.text(enc_x + box_w/2 + 0.006, (enc1_y + enc2_y + box_h)/2, "Downsample 2×",
            fontsize=6.6, color=C_MUTED, va='center')
    
    draw_rounded_box(ax, enc_x, enc2_y, box_w, box_h,
                     "Encoder 2", "64 ch × 64³",
                     fc='#FFFFFF', ec=C_ENC, lw=1.2)
    
    # Enc2 -> Enc3
    draw_arrow(ax, enc_x + box_w/2, enc2_y, enc_x + box_w/2, enc3_y + box_h, color=C_ENC)
    ax.text(enc_x + box_w/2 + 0.006, (enc2_y + enc3_y + box_h)/2, "Downsample 2×",
            fontsize=6.6, color=C_MUTED, va='center')
    
    draw_rounded_box(ax, enc_x, enc3_y, box_w, box_h,
                     "Encoder 3", "128 ch × 32³",
                     fc='#FFFFFF', ec=C_ENC, lw=1.2)
    
    # 3. Bottleneck (bottom center)
    bot_w = box_w * 1.20
    bot_x = pA_x + (pA_w - bot_w) / 2
    bot_y = enc3_y - 0.170
    
    # Enc3 -> Bottleneck
    draw_arrow(ax, enc_x + box_w*0.5, enc3_y, bot_x + 0.015, bot_y + box_h,
               color=C_BOT, lw=1.2, rad=0.12)
    
    draw_rounded_box(ax, bot_x, bot_y, bot_w, box_h,
                     "Bottleneck", "256 ch × 16³",
                     fc='#FAF5FF', ec=C_BOT, lw=1.4, title_color=C_BOT, title_size=8.8)
    
    # 4. Decoder Stages
    dec3_y = enc3_y
    dec2_y = enc2_y
    dec1_y = enc1_y
    
    # Bottleneck -> Dec3
    draw_arrow(ax, bot_x + bot_w - 0.015, bot_y + box_h, dec_x + box_w*0.5, dec3_y,
               color=C_DEC, lw=1.2, rad=0.12)
    
    draw_rounded_box(ax, dec_x, dec3_y, box_w, box_h,
                     "Decoder 3", "128 ch × 32³",
                     fc='#FFFFFF', ec=C_DEC, lw=1.2)
    
    # Dec3 -> Dec2
    draw_arrow(ax, dec_x + box_w/2, dec3_y + box_h, dec_x + box_w/2, dec2_y, color=C_DEC)
    ax.text(dec_x + box_w/2 + 0.006, (dec3_y + box_h + dec2_y)/2, "Upsample 2×",
            fontsize=6.6, color=C_MUTED, va='center')
    
    draw_rounded_box(ax, dec_x, dec2_y, box_w, box_h,
                     "Decoder 2", "64 ch × 64³",
                     fc='#FFFFFF', ec=C_DEC, lw=1.2)
    
    # Dec2 -> Dec1
    draw_arrow(ax, dec_x + box_w/2, dec2_y + box_h, dec_x + box_w/2, dec1_y, color=C_DEC)
    ax.text(dec_x + box_w/2 + 0.006, (dec2_y + box_h + dec1_y)/2, "Upsample 2×",
            fontsize=6.6, color=C_MUTED, va='center')
    
    # Dec1 Box: The shared representation D1
    draw_rounded_box(ax, dec_x, dec1_y, box_w, box_h,
                     "Decoder 1 (D₁)", "32 ch × 128³",
                     fc='#ECFDF5', ec=C_D1, lw=1.6, title_color=C_D1, title_size=8.6)
    
    # 5. Skip Connections
    draw_arrow(ax, enc_x + box_w, enc3_y + box_h/2, dec_x, dec3_y + box_h/2, color='#94A3B8', lw=1.0, style='->')
    ax.text((enc_x + box_w + dec_x)/2, enc3_y + box_h/2 + 0.008, "Skip", fontsize=6.6, color='#64748B', ha='center')
    
    draw_arrow(ax, enc_x + box_w, enc2_y + box_h/2, dec_x, dec2_y + box_h/2, color='#94A3B8', lw=1.0, style='->')
    ax.text((enc_x + box_w + dec_x)/2, enc2_y + box_h/2 + 0.008, "Skip", fontsize=6.6, color='#64748B', ha='center')
    
    # Skip 1 with Attention Gate: Enc1 -> AG -> Dec1
    ag_x = (enc_x + box_w + dec_x)/2 - 0.016
    ag_y = enc1_y + box_h/2 - 0.016
    ag_w, ag_h = 0.032, 0.032
    draw_rounded_box(ax, ag_x, ag_y, ag_w, ag_h, "AG", "", fc='#F3E8FF', ec=C_BOT, lw=1.1,
                     title_color=C_BOT, title_size=7.4)
    
    draw_arrow(ax, enc_x + box_w, enc1_y + box_h/2, ag_x, ag_y + ag_h/2, color='#94A3B8', lw=1.0, style='->')
    draw_arrow(ax, ag_x + ag_w, ag_y + ag_h/2, dec_x, dec1_y + box_h/2, color='#94A3B8', lw=1.0, style='->')
    
    # Conditioning arrow from Bottleneck up to Attention Gate
    draw_arrow(ax, bot_x + bot_w*0.5, bot_y + box_h, ag_x + ag_w/2, ag_y,
               color=C_BOT, lw=1.0, style='->', rad=-0.16)
    ax.text(ag_x + 0.016, (bot_y + ag_y)/2 + 0.045, "Gate ψ",
            fontsize=6.5, color=C_BOT, ha='left', style='italic')

    # ==========================================================================
    # PANEL B: Dual Readouts & Candidate Adjudication
    # ==========================================================================
    d1_box_x = pB_x + 0.024
    d1_box_y = panel_y + panel_h - 0.126
    d1_box_w = pB_w - 0.048
    d1_box_h = 0.058
    
    # Connector arrow from Panel A (Dec 1) to Panel B (Shared D1)
    draw_arrow(ax, dec_x + box_w, dec1_y + box_h*0.5, d1_box_x, d1_box_y + d1_box_h*0.5,
               color=C_D1, lw=2.2, style='-|>', rad=0.08)
    
    # Shared Representation Box
    draw_rounded_box(ax, d1_box_x, d1_box_y, d1_box_w, d1_box_h,
                     r"Shared Representation Trunk: $D_1 \in \mathbb{R}^{32 \times 128 \times 128 \times 128}$",
                     "Identical 32-D voxel embedding trunk evaluated by frozen linear readouts",
                     fc='#ECFDF5', ec=C_D1, lw=1.6, title_color=C_D1, title_size=8.8, sub_size=7.2)
    
    # Fork directly to the two Readout Heads
    branch_y = d1_box_y - 0.106
    head_w = (d1_box_w - 0.024) / 2
    head_h = 0.076
    
    head_p_x = d1_box_x
    head_r_x = d1_box_x + head_w + 0.024
    
    draw_arrow(ax, d1_box_x + d1_box_w*0.25, d1_box_y, head_p_x + head_w/2, branch_y + head_h,
               color=C_PROD, lw=1.6, style='-|>', rad=-0.06)
    draw_arrow(ax, d1_box_x + d1_box_w*0.75, d1_box_y, head_r_x + head_w/2, branch_y + head_h,
               color=C_MRD, lw=1.6, style='-|>', rad=0.06)
    
    # Production Head Box (Left Stream: Calibrated Core)
    draw_rounded_box(ax, head_p_x, branch_y, head_w, head_h,
                     r"Production Head ($\mathbf{w}_P$)",
                     r"$\mathbf{w}_P^T D_1 + b_P \to P_{\mathrm{ET}}$" + "\n" +
                     r"$\tau^* = 0.20 \to M_{\mathrm{core}}$ (Calibrated Core)",
                     fc='#EFF6FF', ec=C_PROD, lw=1.4, title_color=C_PROD, title_size=8.4, sub_size=7.2)
    
    # MRD Head Box (Right Stream: Residual Candidates)
    draw_rounded_box(ax, head_r_x, branch_y, head_w, head_h,
                     r"MRD Residual Head ($\mathbf{w}_R$)",
                     r"$\mathbf{w}_R^T D_1 + b_R \to R_{\mathrm{MRD}}$" + "\n" +
                     r"$\tau_R^* = 0.9508 \to M_{\mathrm{cand}}$ (Residual Lesions)",
                     fc='#FFFBEB', ec=C_MRD, lw=1.4, title_color=C_MRD, title_size=8.4, sub_size=7.2)

    # --------------------------------------------------------------------------
    # Candidate Adjudication Steps (flowing down from MRD Head)
    # --------------------------------------------------------------------------
    flow_x = head_r_x
    flow_w = head_w
    step_h = 0.052
    drop = 0.074
    
    # Step 1: 3D Connected Components Grouping
    s1_y = branch_y - 0.096
    draw_arrow(ax, head_r_x + head_w/2, branch_y, flow_x + flow_w/2, s1_y + step_h,
               color=C_MRD, lw=1.8, style='-|>')
    
    draw_rounded_box(ax, flow_x, s1_y, flow_w, step_h,
                     "1. 3D Connected Components",
                     r"6-connectivity clustering $\to \{C_k\}$",
                     fc='#FFFFFF', ec='#94A3B8', lw=1.2, title_size=8.4, sub_size=7.2)
    
    # Step 2: Spatial Distance Gating (2-mm Rule)
    s2_y = s1_y - drop
    draw_arrow(ax, flow_x + flow_w/2, s1_y, flow_x + flow_w/2, s2_y + step_h, color=C_MUTED)
    
    gate_h = step_h + 0.046
    gate_card = FancyBboxPatch((flow_x, s2_y - 0.046), flow_w, gate_h,
                               boxstyle="round,pad=0,rounding_size=0.010",
                               facecolor='#F8FAFC', edgecolor=C_BORDER, lw=1.1, zorder=2)
    ax.add_patch(gate_card)
    ax.text(flow_x + flow_w/2, s2_y + step_h*0.72,
            "2. Spatial Distance Gating",
            ha='center', va='center', fontsize=8.4, weight='bold', color=C_NAVY, zorder=3)
    ax.text(flow_x + flow_w/2, s2_y + step_h*0.32,
            r"$d_{\mathrm{core}}(C_k) = \min \|u - v\|_2$",
            ha='center', va='center', fontsize=7.2, color=C_MUTED, zorder=3)
    
    # Split sub-boxes: Zone 1 (Reject) vs Zone 2 (Retain)
    split_w = (flow_w - 0.012) / 2
    split_h = 0.035
    split_y = s2_y - 0.040
    
    draw_rounded_box(ax, flow_x, split_y, split_w, split_h,
                     r"$d \leq 2\text{ mm}$ (Zone 1)", "Reject (Boundary)",
                     fc='#FEF2F2', ec=C_REJECT, lw=1.1, title_color=C_REJECT, sub_color='#991B1B',
                     title_size=7.4, sub_size=6.6)
    
    draw_rounded_box(ax, flow_x + split_w + 0.012, split_y, split_w, split_h,
                     r"$d > 2\text{ mm}$ (Zone 2)", "Retain (Distant)",
                     fc='#F0FDF4', ec=C_ACCEPT, lw=1.1, title_color=C_ACCEPT, sub_color='#166534',
                     title_size=7.4, sub_size=6.6)
    
    # Step 3: Feature Extraction (27 Descriptors)
    s3_y = split_y - drop
    draw_arrow(ax, flow_x + split_w + 0.012 + split_w/2, split_y, flow_x + flow_w/2, s3_y + step_h,
               color=C_ACCEPT, style='-|>', rad=0.0)
    draw_rounded_box(ax, flow_x, s3_y, flow_w, step_h,
                     "3. Radiomic Feature Extraction",
                     "27 Features (Morph, Int, Dist & Context)",
                     fc='#FFFFFF', ec='#94A3B8', lw=1.2, title_size=8.4, sub_size=7.0)
    
    # Step 4: A2 Meta-Classifier
    s4_y = s3_y - drop
    draw_arrow(ax, flow_x + flow_w/2, s3_y, flow_x + flow_w/2, s4_y + step_h, color=C_MUTED)
    draw_rounded_box(ax, flow_x, s4_y, flow_w, step_h,
                     "4. A2 Meta-Classifier Adjudication",
                     r"Logistic regression: $P_{\mathrm{A2}} \geq 0.80$",
                     fc='#F0FDF4', ec=C_ACCEPT, lw=1.3, title_color=C_ACCEPT, title_size=8.4, sub_size=7.2)
    
    # Step 5: Multi-Seed Ensemble Consensus
    s5_y = s4_y - drop
    draw_arrow(ax, flow_x + flow_w/2, s4_y, flow_x + flow_w/2, s5_y + step_h, color=C_MUTED)
    draw_rounded_box(ax, flow_x, s5_y, flow_w, step_h,
                     "5. Multi-Seed Consensus Voting",
                     r"$\geq 3 / 5$ Seeds $\to M_{\mathrm{rescued}}$",
                     fc='#FAF5FF', ec=C_BOT, lw=1.3, title_color=C_BOT, title_size=8.4, sub_size=7.2)
    
    # --------------------------------------------------------------------------
    # Step 6: Final Mask Fusion (spanning full width of Panel B)
    # --------------------------------------------------------------------------
    s6_y = s5_y - drop - 0.015
    s6_h = step_h * 1.30
    s6_w = d1_box_w
    s6_x = d1_box_x
    
    draw_rounded_box(ax, s6_x, s6_y, s6_w, s6_h,
                     "Final ET Segmentation Mask",
                     r"$M_{\mathrm{final}} = M_{\mathrm{core}} \cup M_{\mathrm{rescued}}$ (Calibrated Core + Rescued Satellites)",
                     fc='#EFF6FF', ec=C_NAVY, lw=1.6, title_color=C_NAVY, sub_color=C_SLATE,
                     title_size=8.8, sub_size=7.4)
    
    # Rescued stream entering Final Mask straight from Step 5
    draw_arrow(ax, flow_x + flow_w/2, s5_y, flow_x + flow_w/2, s6_y + s6_h,
               color=C_ACCEPT, lw=1.8, style='-|>')
    
    # Badge in left lane
    core_badge_y = (branch_y + s6_y + s6_h) / 2
    draw_rounded_box(ax, head_p_x + head_w*0.15, core_badge_y - 0.024, head_w*0.70, 0.048,
                     r"Protected Core Stream", r"$M_{\mathrm{core}}$ (Unchanged)",
                     fc='#F0F9FF', ec=C_PROD, lw=1.1, title_color=C_PROD, sub_color='#1E40AF',
                     title_size=7.6, sub_size=6.8)

    # Core stream traveling straight down from Production Head (Left lane) into Final Mask!
    draw_arrow(ax, head_p_x + head_w/2, branch_y, head_p_x + head_w/2, core_badge_y + 0.024,
               color=C_PROD, lw=1.8, style='-|>')
    draw_arrow(ax, head_p_x + head_w/2, core_badge_y - 0.024, head_p_x + head_w/2, s6_y + s6_h,
               color=C_PROD, lw=1.8, style='-|>')

    # Save both 4K PNG (350 DPI) and infinite-resolution Vector SVG
    out_png = FIG_DIR / 'fig3_neuroscan_architecture.png'
    out_svg = FIG_DIR / 'fig3_neuroscan_architecture.svg'
    plt.savefig(out_png, dpi=350, facecolor='#FFFFFF', edgecolor='none', bbox_inches='tight')
    plt.savefig(out_svg, facecolor='#FFFFFF', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated 4K PNG ({out_png}) and Vector SVG ({out_svg})")

if __name__ == '__main__':
    generate_fig3()
