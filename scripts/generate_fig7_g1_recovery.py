"""
Generate Figure 7: G1 Missed Satellite Lesion Recovery and Volume Characterization.
Strictly follows Q1 journal standards (IEEE TMI / Medical Image Analysis):
- PURE SCIENTIFIC GRAPHICS: Zero marketing cards, zero text essays, zero AI slop.
- Panel (a): G1 Missed Satellite Lesion Recovery Rate across pre-registered configurations:
  - Production Baseline (tau=0.50): 0 / 162 (0.0%)
  - Decision-Calibrated Core (tau*=0.20): 18 / 162 (11.1%)
  - Hierarchically Gated Core (tau*=0.15): 20 / 162 (12.3%)
  - NeuroScan (Proposed Integrated): 26 / 162 (16.0%)  [Visually emphasized, +44% over calib core]
- Panel (b): Empirical Cumulative Distribution Function (ECDF) of G1 Lesion Volumes:
  - Exact data from E292_G1_lesion_volumes.csv (N=162 lesions).
  - Highlights extreme right-skew:
    - Median = 10.5 mm³ (50% of lesions)
    - 84.6% of lesions < 50 mm³ (microscopic satellites)
    - Mean = 109.0 mm³ (pulled by extreme right tail)
  - Log-scale x-axis with rug plot of individual lesions.
Outputs at 300 DPI to figures/fig7_g1_recovery.png.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8.5,
    'axes.labelsize': 9.2,
    'axes.titlesize': 10.0,
    'xtick.labelsize': 8.2,
    'ytick.labelsize': 8.2,
    'legend.fontsize': 8.0,
    'figure.titlesize': 11.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette (Scientific, high-contrast publication grade)
C_NAVY     = '#0F172A'
C_SLATE    = '#334155'
C_MUTED    = '#64748B'
C_BORDER   = '#CBD5E1'

C_BASE     = '#64748B'   # Slate (Baseline)
C_CALIB    = '#0284C7'   # Cyan/Sky (Calibrated Core)
C_HIER     = '#0D9488'   # Teal (Hierarchical Core)
C_PROP     = '#2563EB'   # Royal Blue (Proposed NeuroScan)

C_ECDF     = '#2563EB'   # Primary curve
C_FILL     = '#EFF6FF'   # Under-curve fill
C_MED      = '#E11D48'   # Rose for Median
C_85       = '#059669'   # Emerald for 84.6% threshold
C_MEAN     = '#D97706'   # Amber for Mean

def generate_fig7():
    # Load G1 lesion volume data
    vol_path = ROOT / 'novelty' / 'pairwise_evidence_geometry' / 'E292_G1_lesion_volumes.csv'
    df_vol = pd.read_csv(vol_path)
    volumes = np.sort(df_vol['volume_voxels'].values)  # isotropic 1mm3 -> mm3
    n_lesions = len(volumes)
    
    fig = plt.figure(figsize=(13.6, 5.0), facecolor='#FFFFFF')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.30,
                           left=0.08, right=0.96, top=0.86, bottom=0.16)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    
    # ==========================================================================
    # PANEL A: G1 Missed Satellite Lesion Recovery Rate
    # ==========================================================================
    methods = [
        "Production Baseline\n(τ = 0.50)",
        "Decision-Calibrated\nCore (τ* = 0.20)",
        "Hierarchically Gated\nCore (τ* = 0.15)",
        "★ NeuroScan\n(Proposed Integrated)"
    ]
    
    recovered = [0, 18, 20, 26]
    total_g1 = 162
    pcts = [r / total_g1 * 100 for r in recovered]
    
    colors = [C_BASE, C_CALIB, C_HIER, C_PROP]
    edge_colors = ['#475569', '#0369A1', '#0F766E', '#1D4ED8']
    
    x_pos = np.arange(len(methods))
    bar_width = 0.54
    
    ax_a.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1', zorder=0)
    
    for i in range(len(methods)):
        is_prop = (i == 3)
        lw = 2.2 if is_prop else 1.1
        
        ax_a.bar(x_pos[i], recovered[i], width=bar_width,
                 color=colors[i], edgecolor=edge_colors[i], linewidth=lw,
                 zorder=3, alpha=0.92)
        
        # Count & percentage text above bar
        val_str = f"{recovered[i]} / {total_g1}\n({pcts[i]:.1f}%)"
        text_weight = 'bold' if is_prop else 'normal'
        text_color = C_PROP if is_prop else C_NAVY
        ax_a.text(x_pos[i], recovered[i] + 0.9, val_str,
                  ha='center', va='bottom', fontsize=7.6, weight=text_weight, color=text_color)
        
    # Relative gain callout on Proposed vs Calibrated Core
    rel_gain = ((26 - 18) / 18) * 100
    ax_a.annotate(f"+{rel_gain:.1f}% vs.\nCalib. Core",
                  xy=(x_pos[3], 26), xytext=(x_pos[3] - 0.55, 23.5),
                  arrowprops=dict(arrowstyle="->", color=C_PROP, lw=1.2),
                  fontsize=7.2, weight='bold', color=C_PROP, ha='center',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='#EFF6FF', edgecolor='#BFDBFE', lw=0.8))
    
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(methods, fontsize=7.8)
    ax_a.set_ylabel(f"Recovered G1 Lesions (out of {total_g1})", weight='bold', labelpad=6)
    ax_a.set_ylim(0, 32)
    ax_a.set_yticks(np.arange(0, 31, 5))
    ax_a.set_title("(a) Missed Satellite Lesion Recovery\n(G1 Candidates Rescued on Locked Test Set, N=162)",
                   loc='left', weight='bold', pad=10, fontsize=9.2, color=C_NAVY)

    # ==========================================================================
    # PANEL B: G1 Component Volume Distribution (ECDF)
    # ==========================================================================
    # ECDF coordinates
    ecdf_y = np.arange(1, n_lesions + 1) / n_lesions
    
    median_vol = np.median(volumes)       # 10.5 mm3
    mean_vol   = np.mean(volumes)         # 109.0 mm3
    pct_50mm3  = (volumes < 50).mean()    # 0.8457 (84.6%)
    
    ax_b.set_xscale('log')
    ax_b.grid(True, which='both', linestyle='--', alpha=0.4, color='#CBD5E1', zorder=0)
    
    # Fill area under ECDF
    ax_b.fill_between(volumes, ecdf_y, color=C_FILL, alpha=0.9, zorder=1)
    
    # ECDF step curve
    ax_b.step(volumes, ecdf_y, where='post', color=C_ECDF, lw=2.2, zorder=3,
              label='Empirical CDF (N=162 lesions)')
    
    # Rug plot: Individual lesion ticks at bottom
    ax_b.plot(volumes, np.zeros_like(volumes) + 0.015, '|', color='#3B82F6',
              markersize=7, markeredgewidth=0.8, alpha=0.55, zorder=2, label='Individual Lesion')
    
    # 1. Median Guideline (50% of lesions <= 10.5 mm3)
    ax_b.plot([volumes[0]*0.8, median_vol, median_vol], [0.50, 0.50, 0.0],
              color=C_MED, lw=1.3, linestyle='--', zorder=4)
    ax_b.plot(median_vol, 0.50, marker='o', color=C_MED, markersize=5.5, zorder=5)
    ax_b.text(median_vol * 1.15, 0.46,
              f"Median: {median_vol:.1f} mm³\n(50% of lesions)",
              fontsize=7.4, weight='bold', color=C_MED, va='top',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF1F2', edgecolor='#FECDD3', lw=0.8))
    
    # 2. 50 mm3 Threshold Guideline (84.6% of lesions < 50 mm3)
    ax_b.plot([volumes[0]*0.8, 50, 50], [pct_50mm3, pct_50mm3, 0.0],
              color=C_85, lw=1.3, linestyle='--', zorder=4)
    ax_b.plot(50, pct_50mm3, marker='s', color=C_85, markersize=5.5, zorder=5)
    ax_b.text(50 * 1.15, pct_50mm3 - 0.02,
              f"84.6% < 50 mm³\n(Microscopic Satellites)",
              fontsize=7.4, weight='bold', color=C_85, va='top',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#F0FDF4', edgecolor='#BBF7D0', lw=0.8))
    
    # 3. Mean Marker (109.0 mm3)
    mean_y = (volumes <= mean_vol).mean()
    ax_b.axvline(mean_vol, color=C_MEAN, lw=1.2, linestyle=':', zorder=4)
    ax_b.plot(mean_vol, mean_y, marker='^', color=C_MEAN, markersize=5.5, zorder=5)
    ax_b.text(mean_vol * 1.15, 0.25,
              f"Mean: {mean_vol:.1f} mm³\n(Skewed by right tail)",
              fontsize=7.2, color=C_MEAN, weight='bold', va='center',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFBEB', edgecolor='#FDE68A', lw=0.8))
    
    # Formatting
    ax_b.set_xlim(3.5, 9000)
    ax_b.set_ylim(-0.02, 1.05)
    ax_b.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_b.set_yticklabels([f"{p:.0f}%" for p in np.linspace(0, 100, 6)])
    
    ax_b.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax_b.set_xlabel("G1 Lesion Volume (mm³, Log Scale)", weight='bold', labelpad=6)
    ax_b.set_ylabel("Cumulative Proportion P(Volume ≤ V)", weight='bold', labelpad=6)
    ax_b.set_title("(b) G1 Missed Lesion Volume Distribution (ECDF)\n(Demonstrating Extreme Right-Skew & Sub-50 mm³ Satellites)",
                   loc='left', weight='bold', pad=10, fontsize=9.2, color=C_NAVY)

    # Save both 4K PNG (350 DPI) and infinite-resolution Vector SVG
    out_png = FIG_DIR / 'fig7_g1_recovery.png'
    out_svg = FIG_DIR / 'fig7_g1_recovery.svg'
    plt.savefig(out_png, dpi=350, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.savefig(out_svg, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated 4K PNG ({out_png}) and Vector SVG ({out_svg})")

if __name__ == '__main__':
    generate_fig7()
