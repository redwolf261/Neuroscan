"""
Generate Figure 4: Computational Efficiency of NeuroScan.
Optimized for Word Document embedding & Q1 Journal printing (IEEE TMI / MedIA):
- 2x2 Balanced Grid with high-contrast, large typography (12pt titles, 11pt axis labels, 9.6pt ticks).
- Explicit X and Y axis labels on ALL 4 panels with clear physical units.
- Non-overlapping data labels with smart inside-bar positioning and staggered novelty callouts.
- Saves to both fig4_computational_efficiency and fig_4_computational_efficiency (PNG & SVG).
- Panel (a): Parameter breakdown by architectural submodule.
- Panel (b): Peak GPU VRAM footprint across training and inference regimes.
- Panel (c): Measured execution latency across operational scales (log scale).
- Panel (d): Incremental stage latency waterfall (Trunk -> TTA -> MRD -> A2) with residual overhead callout.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Publication styling matching IEEE TMI / MedIA, optimized for Word sharpness
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9.6,
    'axes.labelsize': 11.0,
    'axes.titlesize': 12.0,
    'xtick.labelsize': 9.6,
    'ytick.labelsize': 9.6,
    'figure.dpi': 350,
    'savefig.dpi': 350,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Curated Q1 color palette
C_NAVY     = '#0F172A'       # Slate 900
C_SLATE    = '#1E293B'       # Slate 800
C_MUTED    = '#64748B'       # Slate 500
C_BORDER   = '#CBD5E1'       # Slate 300
C_GRID     = '#E2E8F0'       # Slate 200

# Module colors
C_BLUE     = '#2563EB'       # Blue 600 (Trunk/Backbone)
C_PURPLE   = '#7C3AED'       # Purple 600 (Bottleneck)
C_TEAL     = '#0D9488'       # Teal 600 (Decoder)
C_AMBER    = '#D97706'       # Amber 600 (MRD)
C_GREEN    = '#16A34A'       # Green 600 (A2)
C_CORAL    = '#E11D48'       # Rose 600 (Training / Peak)
C_INDIGO   = '#4F46E5'       # Indigo 600


def create_figure():
    fig = plt.figure(figsize=(12.2, 9.2), facecolor='#FFFFFF')
    
    # 2x2 grid with ample padding for axis titles and callouts
    gs = fig.add_gridspec(2, 2, left=0.11, right=0.96, bottom=0.09, top=0.94,
                          wspace=0.38, hspace=0.38)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # =========================================================================
    # PANEL A: Parameter Allocation
    # =========================================================================
    ax_a.set_facecolor('#FFFFFF')
    ax_a.set_title(r"$\mathbf{a}$   Parameter Distribution by Module", loc='left', fontsize=12.0, weight='bold', color=C_NAVY, pad=10)

    modules = [
        'Bottleneck (16³)',
        'Decoder (D3, D2, D1)',
        'Encoder (E1, E2, E3)',
        'Heads & Aux',
        'MRD + A2 Readout'
    ]
    params_k = [2655.744, 2087.456, 861.888, 5.603, 0.193]
    pcts = [47.34, 37.20, 15.36, 0.10, 0.003]
    colors_a = [C_PURPLE, C_TEAL, C_BLUE, '#94A3B8', C_GREEN]

    y_pos = np.arange(len(modules))[::-1]
    bars_a = ax_a.barh(y_pos, params_k, height=0.56, color=colors_a, edgecolor='#0F172A', linewidth=1.1, zorder=3)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(modules, fontsize=9.6, weight='bold', color=C_SLATE)
    ax_a.set_ylabel("Architectural Submodule", fontsize=11.0, weight='bold', color=C_NAVY, labelpad=8)
    ax_a.set_xlabel("Parameter Count (× 1,000)", fontsize=11.0, weight='bold', color=C_NAVY, labelpad=8)
    ax_a.set_xlim(0, 3100)
    ax_a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax_a.grid(axis='x', linestyle='--', alpha=0.55, color=C_GRID, zorder=0)

    for bar, k, p in zip(bars_a, params_k, pcts):
        w = bar.get_width()
        y = bar.get_y() + bar.get_height()/2
        if k >= 1500.0:
            # Place inside bar in crisp white text for wide bars
            lbl = f"{k:,.1f}k ({p:.1f}%)"
            ax_a.text(w - 70, y, lbl, va='center', ha='right', fontsize=9.2, weight='bold', color='#FFFFFF')
        elif k >= 1.0:
            lbl = f"{k:,.1f}k ({p:.1f}%)"
            ax_a.text(w + 45, y, lbl, va='center', ha='left', fontsize=9.0, weight='bold', color=C_SLATE)
        else:
            lbl = f"193 params (<0.01%)"
            ax_a.text(w + 45, y, lbl, va='center', ha='left', fontsize=9.0, weight='bold', color='#15803D')

    # =========================================================================
    # PANEL B: Memory Footprint (VRAM)
    # =========================================================================
    ax_b.set_facecolor('#FFFFFF')
    ax_b.set_title(r"$\mathbf{b}$   Peak GPU Memory Allocation (VRAM)", loc='left', fontsize=12.0, weight='bold', color=C_NAVY, pad=10)

    mem_labels = [
        'Model Weights (FP32)',
        'Single Forward Pass',
        '8-Fold TTA Inference',
        'Training Peak (Adam)'
    ]
    mem_vals_gb = [0.021, 3.18, 3.69, 5.84]
    mem_colors = ['#94A3B8', C_BLUE, C_INDIGO, C_CORAL]

    y_pos_b = np.arange(len(mem_labels))[::-1]
    bars_b = ax_b.barh(y_pos_b, mem_vals_gb, height=0.56, color=mem_colors, edgecolor='#0F172A', linewidth=1.1, zorder=3)
    ax_b.set_yticks(y_pos_b)
    ax_b.set_yticklabels(mem_labels, fontsize=9.6, weight='bold', color=C_SLATE)
    ax_b.set_ylabel("Operational Regime", fontsize=11.0, weight='bold', color=C_NAVY, labelpad=8)
    ax_b.set_xlabel("Peak GPU Memory (GB)", fontsize=11.0, weight='bold', color=C_NAVY, labelpad=8)
    ax_b.set_xlim(0, 9.4)
    ax_b.grid(axis='x', linestyle='--', alpha=0.55, color=C_GRID, zorder=0)

    # Dashed threshold line at 8.0 GB hardware limit
    ax_b.axvline(7.93, color='#DC2626', linestyle='--', linewidth=1.4, zorder=2)
    ax_b.text(7.75, 2.8, "8 GB Hardware Limit\n(RTX 4070 / 3070)", ha='right', va='center',
              fontsize=8.5, weight='bold', color='#DC2626',
              bbox=dict(boxstyle='square,pad=0.2', facecolor='#FEF2F2', edgecolor='#FECACA', lw=0.8))

    for bar, val in zip(bars_b, mem_vals_gb):
        w = bar.get_width()
        y = bar.get_y() + bar.get_height()/2
        if val >= 2.5:
            lbl = f"{val:.2f} GB"
            ax_b.text(w - 0.18, y, lbl, va='center', ha='right', fontsize=9.2, weight='bold', color='#FFFFFF')
        else:
            lbl = f"{val*1024:.1f} MB (0.02 GB)"
            ax_b.text(w + 0.18, y, lbl, va='center', ha='left', fontsize=9.0, weight='bold', color=C_SLATE)

    # =========================================================================
    # PANEL C: Inference Latency across Scales
    # =========================================================================
    ax_c.set_facecolor('#FFFFFF')
    ax_c.set_title(r"$\mathbf{c}$   Execution Latency Across Operational Scales", loc='left', fontsize=12.0, weight='bold', color=C_NAVY, pad=10)

    scales = [
        'Single Forward\n(128³ Patch)',
        '8-Fold TTA\n(128³ Patch)',
        'Full Pipeline\n(Subject End-to-End)'
    ]
    lat_vals_s = [0.170, 1.335, 10.91]
    lat_errs_s = [0.011, 0.008, 0.45]
    lat_colors = [C_BLUE, C_INDIGO, C_SLATE]

    x_pos_c = np.arange(len(scales))
    bars_c = ax_c.bar(x_pos_c, lat_vals_s, yerr=lat_errs_s, width=0.48,
                      color=lat_colors, edgecolor='#0F172A', linewidth=1.1,
                      capsize=5.0, error_kw=dict(lw=1.4, capthick=1.4, ecolor='#1E293B'), zorder=3)
    ax_c.set_xticks(x_pos_c)
    ax_c.set_xticklabels(scales, fontsize=9.6, weight='bold', color=C_SLATE)
    ax_c.set_xlabel("Inference Operational Scope", fontsize=11.0, weight='bold', color=C_NAVY, labelpad=8)
    ax_c.set_ylabel("Execution Latency (seconds) [Log Scale]", fontsize=11.0, weight='bold', color=C_NAVY, labelpad=8)
    ax_c.set_yscale('log')
    ax_c.set_ylim(0.06, 40.0)
    ax_c.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g} s"))
    ax_c.grid(axis='y', linestyle='--', alpha=0.55, color=C_GRID, zorder=0)

    labels_c = ["0.17 s (170 ms)", "1.34 s", "10.91 s"]
    for bar, lbl in zip(bars_c, labels_c):
        x = bar.get_x() + bar.get_width()/2
        y = bar.get_height()
        ax_c.text(x, y * 1.38, lbl, ha='center', va='bottom', fontsize=9.2, weight='bold', color=C_NAVY)

    # =========================================================================
    # PANEL D: Incremental Computational Overhead (Waterfall)
    # =========================================================================
    ax_d.set_facecolor('#FFFFFF')
    ax_d.set_title(r"$\mathbf{d}$   Incremental Stage Latency Overhead", loc='left', fontsize=12.0, weight='bold', color=C_NAVY, pad=10)

    stages = [
        'Trunk Pass',
        '+ 8-Fold\nTTA',
        '+ Residual\nMRD',
        '+ Consensus\nA2'
    ]
    inc_ms = [170.03, 1164.93, 6.28, 0.52]
    cum_ms = [170.03, 1334.96, 1341.24, 1341.76]
    stage_colors = [C_BLUE, C_INDIGO, C_AMBER, C_GREEN]

    x_pos_d = np.arange(len(stages))
    bottoms = [0.0, 170.03, 1334.96, 1341.24]
    bars_d = ax_d.bar(x_pos_d, inc_ms, bottom=bottoms, width=0.48,
                      color=stage_colors, edgecolor='#0F172A', linewidth=1.1, zorder=3)
    ax_d.set_xticks(x_pos_d)
    ax_d.set_xticklabels(stages, fontsize=9.6, weight='bold', color=C_SLATE)
    ax_d.set_xlabel("Sequential Computational Stage", fontsize=11.0, weight='bold', color=C_NAVY, labelpad=8)
    ax_d.set_ylabel("Cumulative Stage Latency (ms)", fontsize=11.0, weight='bold', color=C_NAVY, labelpad=8)
    ax_d.set_ylim(0, 1850)
    ax_d.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,} ms"))
    ax_d.grid(axis='y', linestyle='--', alpha=0.55, color=C_GRID, zorder=0)

    # Step connector lines
    for i in range(len(stages)-1):
        ax_d.plot([x_pos_d[i], x_pos_d[i+1]], [cum_ms[i], cum_ms[i]],
                  color=C_MUTED, linestyle=':', linewidth=1.2, zorder=2)

    # Bar 0 label
    ax_d.text(x_pos_d[0], cum_ms[0] + 50, "170.0 ms", ha='center', va='bottom', fontsize=9.2, weight='bold', color=C_NAVY)
    
    # Bar 1 label (+ TTA)
    ax_d.text(x_pos_d[1], cum_ms[1] + 50, "+1,164.9 ms\n(8× TTA)", ha='center', va='bottom', fontsize=9.0, weight='bold', color=C_NAVY)

    # Bracket spanning Bar 2 (+ MRD) and Bar 3 (+ A2)
    bracket_y = 1460
    ax_d.plot([x_pos_d[2], x_pos_d[2], x_pos_d[3], x_pos_d[3]],
              [bracket_y - 25, bracket_y, bracket_y, bracket_y - 25],
              color='#B45309', linewidth=1.4, zorder=4)
    
    bracket_mid = (x_pos_d[2] + x_pos_d[3]) / 2
    novelty_label = (
        "+6.8 ms Total (+0.51%)\n"
        "[MRD: +6.3 ms | A2: +0.5 ms]"
    )
    ax_d.text(bracket_mid, bracket_y + 35, novelty_label, ha='center', va='bottom',
              fontsize=8.8, weight='bold', color='#B45309',
              bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFFBEB', edgecolor='#FDE68A', lw=1.0))

    # Connection from bars to bracket
    ax_d.plot([x_pos_d[2], x_pos_d[2]], [cum_ms[2] + 8, bracket_y - 25], color='#B45309', linestyle=':', linewidth=1.0)
    ax_d.plot([x_pos_d[3], x_pos_d[3]], [cum_ms[3] + 8, bracket_y - 25], color='#B45309', linestyle=':', linewidth=1.0)

    # Clean spine styling
    for ax in [ax_a, ax_b, ax_c, ax_d]:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(C_BORDER)

    # Save both canonical naming conventions
    targets = [
        (FIG_DIR / 'fig4_computational_efficiency.png', FIG_DIR / 'fig4_computational_efficiency.svg'),
        (FIG_DIR / 'fig_4_computational_efficiency.png', FIG_DIR / 'fig_4_computational_efficiency.svg')
    ]
    for out_png, out_svg in targets:
        fig.savefig(out_png, dpi=350, facecolor='#FFFFFF', bbox_inches='tight')
        fig.savefig(out_svg, facecolor='#FFFFFF', bbox_inches='tight')
        print(f"Generated: {out_png.name} & {out_svg.name}")
        
    plt.close(fig)


if __name__ == '__main__':
    create_figure()
