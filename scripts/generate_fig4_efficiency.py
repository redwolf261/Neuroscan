"""
Generate Figure 4: Computational Efficiency of NeuroScan.
Strictly follows Q1 journal standards (IEEE TMI / Medical Image Analysis):
- PURE SCIENTIFIC GRAPHS: Zero explanatory text boxes, zero paragraphs, zero marketing callouts.
- Panel (a): Parameter breakdown by architectural module (UNet3D-v5 trunk vs. frozen linear readouts).
- Panel (b): Peak GPU VRAM footprint across training and inference regimes.
- Panel (c): Measured execution latency across operational scales.
- Panel (d): Incremental latency waterfall (Backbone -> TTA -> MRD -> A2).
Outputs at 300 DPI to figures/fig4_computational_efficiency.png.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Publication styling matching IEEE TMI / Nature Machine Intelligence
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8.5,
    'axes.labelsize': 8.5,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8.0,
    'ytick.labelsize': 8.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
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
    fig = plt.figure(figsize=(11.5, 7.5), facecolor='#FFFFFF')
    
    # Clean 2x2 grid without top figure titles (captions belong in manuscript)
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.96, bottom=0.08, top=0.94,
                          wspace=0.32, hspace=0.32)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # =========================================================================
    # PANEL A: Parameter Allocation
    # =========================================================================
    ax_a.set_facecolor('#FFFFFF')
    ax_a.set_title("(a) Parameter Distribution", loc='left', fontsize=9.5, weight='bold', color=C_NAVY, pad=8)

    modules = [
        'Bottleneck (16³)',
        'Decoder (D3, D2, D1)',
        'Encoder (E1, E2, E3)',
        'Heads & Aux',
        'MRD + A2 Readouts'
    ]
    # In thousands of parameters
    params_k = [2655.744, 2087.456, 861.888, 5.603, 0.193]
    pcts = [47.34, 37.20, 15.36, 0.10, 0.003]
    colors_a = [C_PURPLE, C_TEAL, C_BLUE, '#94A3B8', C_GREEN]

    y_pos = np.arange(len(modules))[::-1]
    bars_a = ax_a.barh(y_pos, params_k, height=0.52, color=colors_a, edgecolor='none', zorder=3)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(modules, fontsize=8.0, color=C_SLATE)
    ax_a.set_xlabel("Parameters (× 1,000)", fontsize=8.2, color=C_SLATE, labelpad=4)
    ax_a.set_xlim(0, 3450)
    ax_a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax_a.grid(axis='x', linestyle='--', alpha=0.5, color=C_GRID, zorder=0)

    for bar, k, p in zip(bars_a, params_k, pcts):
        w = bar.get_width()
        y = bar.get_y() + bar.get_height()/2
        if k >= 1.0:
            lbl = f"{k:,.1f}k ({p:.1f}%)"
        else:
            lbl = f"193 params (<0.01%)"
        ax_a.text(w + 40, y, lbl, va='center', ha='left', fontsize=7.6, weight='bold', color=C_SLATE)

    # =========================================================================
    # PANEL B: Memory Footprint (VRAM)
    # =========================================================================
    ax_b.set_facecolor('#FFFFFF')
    ax_b.set_title("(b) Peak GPU Memory (VRAM)", loc='left', fontsize=9.5, weight='bold', color=C_NAVY, pad=8)

    mem_labels = [
        'Weights (FP32)',
        'Single Forward Pass',
        '8-Fold TTA Inference',
        'Training Peak (Adam)'
    ]
    mem_vals_gb = [0.021, 3.18, 3.69, 5.84]
    mem_colors = ['#94A3B8', C_BLUE, C_INDIGO, C_CORAL]

    y_pos_b = np.arange(len(mem_labels))[::-1]
    bars_b = ax_b.barh(y_pos_b, mem_vals_gb, height=0.52, color=mem_colors, edgecolor='none', zorder=3)
    ax_b.set_yticks(y_pos_b)
    ax_b.set_yticklabels(mem_labels, fontsize=8.0, color=C_SLATE)
    ax_b.set_xlabel("GPU Memory (GB)", fontsize=8.2, color=C_SLATE, labelpad=4)
    ax_b.set_xlim(0, 9.0)
    ax_b.grid(axis='x', linestyle='--', alpha=0.5, color=C_GRID, zorder=0)

    # Dashed threshold line at 8.0 GB
    ax_b.axvline(7.93, color='#DC2626', linestyle='--', linewidth=1.1, zorder=2)
    ax_b.text(7.80, 3.3, "8 GB Hardware Limit", ha='right', va='center', fontsize=7.2, weight='bold', color='#DC2626')

    for bar, val in zip(bars_b, mem_vals_gb):
        w = bar.get_width()
        y = bar.get_y() + bar.get_height()/2
        lbl = f"{val*1024:.1f} MB" if val < 0.1 else f"{val:.2f} GB"
        ax_b.text(w + 0.15, y, lbl, va='center', ha='left', fontsize=7.6, weight='bold', color=C_SLATE)

    # =========================================================================
    # PANEL C: Inference Latency across Scales
    # =========================================================================
    ax_c.set_facecolor('#FFFFFF')
    ax_c.set_title("(c) Inference Execution Latency", loc='left', fontsize=9.5, weight='bold', color=C_NAVY, pad=8)

    scales = [
        'Single Forward\n(128³ Volume)',
        '8-Fold TTA\n(128³ Volume)',
        'Full Pipeline\n(Subject End-to-End)'
    ]
    lat_vals_s = [0.170, 1.335, 10.91]
    lat_errs_s = [0.011, 0.008, 0.45]
    lat_colors = [C_BLUE, C_INDIGO, C_SLATE]

    x_pos_c = np.arange(len(scales))
    bars_c = ax_c.bar(x_pos_c, lat_vals_s, yerr=lat_errs_s, width=0.48,
                      color=lat_colors, edgecolor='none', capsize=3.5, ecolor=C_MUTED, zorder=3)
    ax_c.set_xticks(x_pos_c)
    ax_c.set_xticklabels(scales, fontsize=8.0, color=C_SLATE)
    ax_c.set_ylabel("Execution Time (seconds)", fontsize=8.2, color=C_SLATE, labelpad=4)
    ax_c.set_yscale('log')
    ax_c.set_ylim(0.06, 30.0)
    ax_c.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g} s"))
    ax_c.grid(axis='y', linestyle='--', alpha=0.5, color=C_GRID, zorder=0)

    labels_c = ["0.17 s (170 ms)", "1.34 s", "10.91 s"]
    for bar, lbl in zip(bars_c, labels_c):
        x = bar.get_x() + bar.get_width()/2
        y = bar.get_height()
        ax_c.text(x, y * 1.30, lbl, ha='center', va='bottom', fontsize=7.8, weight='bold', color=C_SLATE)

    # =========================================================================
    # PANEL D: Incremental Computational Overhead
    # =========================================================================
    ax_d.set_facecolor('#FFFFFF')
    ax_d.set_title("(d) Incremental Stage Latency", loc='left', fontsize=9.5, weight='bold', color=C_NAVY, pad=8)

    stages = [
        'Trunk Pass',
        '+ 8-Fold TTA',
        '+ MRD Readout',
        '+ A2 Consensus'
    ]
    inc_ms = [170.03, 1164.93, 6.28, 0.52]
    cum_ms = [170.03, 1334.96, 1341.24, 1341.76]
    stage_colors = [C_BLUE, C_INDIGO, C_AMBER, C_GREEN]

    x_pos_d = np.arange(len(stages))
    bottoms = [0.0, 170.03, 1334.96, 1341.24]
    bars_d = ax_d.bar(x_pos_d, inc_ms, bottom=bottoms, width=0.48,
                      color=stage_colors, edgecolor='none', zorder=3)
    ax_d.set_xticks(x_pos_d)
    ax_d.set_xticklabels(stages, fontsize=8.0, color=C_SLATE)
    ax_d.set_ylabel("Cumulative Latency (ms)", fontsize=8.2, color=C_SLATE, labelpad=4)
    ax_d.set_ylim(0, 1600)
    ax_d.grid(axis='y', linestyle='--', alpha=0.5, color=C_GRID, zorder=0)

    # Connector lines
    for i in range(len(stages)-1):
        ax_d.plot([x_pos_d[i], x_pos_d[i+1]], [cum_ms[i], cum_ms[i]],
                  color=C_MUTED, linestyle=':', linewidth=0.9, zorder=2)

    inc_labels = [
        "170.0 ms",
        "+1,164.9 ms",
        "+6.3 ms (+0.47%)",
        "+0.5 ms (+0.04%)"
    ]
    for bar, lbl, b_top in zip(bars_d, inc_labels, cum_ms):
        x = bar.get_x() + bar.get_width()/2
        ax_d.text(x, b_top + 30, lbl, ha='center', va='bottom', fontsize=7.4, weight='bold', color=C_SLATE)

    # Clean spine styling
    for ax in [ax_a, ax_b, ax_c, ax_d]:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(C_BORDER)

    # Save both 4K PNG (350 DPI) and infinite-resolution Vector SVG
    out_png = FIG_DIR / 'fig4_computational_efficiency.png'
    out_svg = FIG_DIR / 'fig4_computational_efficiency.svg'
    fig.savefig(out_png, dpi=350, facecolor='#FFFFFF', bbox_inches='tight')
    fig.savefig(out_svg, facecolor='#FFFFFF', bbox_inches='tight')
    plt.close(fig)
    print(f"Generated 4K PNG ({out_png}) and Vector SVG ({out_svg})")

if __name__ == '__main__':
    create_figure()
