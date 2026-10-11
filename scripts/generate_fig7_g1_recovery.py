"""
Generate Figure 7: G1 Missed Satellite Lesion Recovery and Volume Characterization.
Optimized for Word Document embedding & Q1 Journal printing (IEEE TMI / MedIA):
- 2-Row Stacked Layout (Row 1: Panel a - Recovery Rates; Row 2: Panel b - Volume ECDF).
- High Legibility in Word: Aspect ratio ~1.05 (9.6 x 9.2 inches), fits 6.5-inch page margin with crisp 10-12pt text.
- Explicit X and Y axis labels on both panels with physical units.
- Dual Y-axes on Panel a (Absolute Count out of N=162 on left, Recovery Rate % on right).
- Precision callout badges for Median (10.5 mm³), Sub-50 mm³ (84.6%), and Mean (109.0 mm³) on Panel b.
- Dual naming output: saves to both fig7_g1_recovery and fig_7_g1_recovery (PNG & SVG).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Publication styling matching IEEE TMI / MedIA, optimized for Word sharpness
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10.0,
    'axes.labelsize': 11.2,
    'axes.titlesize': 12.2,
    'xtick.labelsize': 10.0,
    'ytick.labelsize': 10.0,
    'legend.fontsize': 9.8,
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

# Method colors
C_BASE     = '#64748B'       # Slate (Baseline)
C_CALIB    = '#0284C7'       # Cyan/Sky (Calibrated Core)
C_HIER     = '#0D9488'       # Teal (Hierarchical Core)
C_PROP     = '#2563EB'       # Royal Blue (Proposed NeuroScan)

# ECDF curve colors
C_ECDF     = '#2563EB'       # Primary curve
C_FILL     = '#EFF6FF'       # Under-curve fill
C_MED      = '#E11D48'       # Rose for Median
C_85       = '#059669'       # Emerald for 84.6% threshold
C_MEAN     = '#D97706'       # Amber for Mean


def generate_fig7():
    # 1. Load G1 lesion volume data
    vol_path = ROOT / 'novelty' / 'pairwise_evidence_geometry' / 'E292_G1_lesion_volumes.csv'
    df_vol = pd.read_csv(vol_path)
    volumes = np.sort(df_vol['volume_voxels'].values)  # isotropic 1mm3 -> mm3
    n_lesions = len(volumes)
    
    # 2-Row Stacked Canvas (9.6 x 9.2 inches)
    fig, (ax_a, ax_b) = plt.subplots(nrows=2, ncols=1, figsize=(9.6, 9.2), facecolor='#FFFFFF')
    plt.subplots_adjust(hspace=0.42, left=0.10, right=0.90, top=0.94, bottom=0.08)

    # =========================================================================
    # ROW 1 (PANEL A): Missed Satellite Lesion Recovery Rates
    # =========================================================================
    ax_a.set_facecolor('#FFFFFF')
    ax_a.set_title(r"$\mathbf{a}$   Missed Satellite Lesion Recovery Rate (G1 Rescued Candidates, $N=162$)",
                   loc='left', fontsize=12.2, weight='bold', color=C_NAVY, pad=12)

    methods = [
        "Production Baseline\n(τ = 0.50)",
        "Decision-Calibrated\nCore (τ* = 0.20)",
        "Hierarchically Gated\nCore (τ* = 0.15)",
        "★ NeuroScan Pipeline\n(Proposed Integrated)"
    ]
    recovered = [0, 18, 20, 26]
    total_g1 = 162
    pcts = [r / total_g1 * 100 for r in recovered]

    colors = [C_BASE, C_CALIB, C_HIER, C_PROP]
    edge_colors = ['#334155', '#0369A1', '#0F766E', '#1D4ED8']

    x_pos = np.arange(len(methods))
    bar_width = 0.52

    ax_a.grid(axis='y', linestyle='--', alpha=0.55, color=C_GRID, zorder=0)

    for i in range(len(methods)):
        is_prop = (i == 3)
        lw = 2.0 if is_prop else 1.1
        
        ax_a.bar(x_pos[i], recovered[i], width=bar_width,
                 color=colors[i], edgecolor=edge_colors[i], linewidth=lw,
                 zorder=3, alpha=0.92)

        # Count & percentage text above bar
        val_str = f"{recovered[i]} / {total_g1}\n({pcts[i]:.1f}%)"
        text_weight = 'bold'
        text_color = C_PROP if is_prop else C_NAVY
        ax_a.text(x_pos[i], recovered[i] + 1.1, val_str,
                  ha='center', va='bottom', fontsize=9.4, weight=text_weight, color=text_color)

    # Relative gain callout on Proposed vs Calibrated Core
    rel_gain = ((26 - 18) / 18) * 100
    ax_a.annotate(f"+{rel_gain:.1f}% Relative Gain\n(+8 satellites vs. calibrated core)",
                  xy=(x_pos[3], 26.2), xytext=(2.15, 30.5),
                  arrowprops=dict(arrowstyle="->", color=C_PROP, lw=1.3, shrinkA=3, shrinkB=4),
                  fontsize=9.0, weight='bold', color=C_PROP, ha='center',
                  bbox=dict(boxstyle='round,pad=0.35', facecolor='#EFF6FF', edgecolor='#BFDBFE', lw=1.1))

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(methods, fontsize=9.8, weight='bold', color=C_SLATE)
    ax_a.set_xlabel("Operating Threshold & Model Architecture Configuration",
                    fontsize=11.2, weight='bold', color=C_NAVY, labelpad=8)
    ax_a.set_ylabel(f"Recovered G1 Lesions (Count out of {total_g1})",
                    fontsize=11.2, weight='bold', color=C_NAVY, labelpad=8)
    ax_a.set_ylim(0, 36)
    ax_a.set_yticks(np.arange(0, 36, 5))

    # Right secondary Y-axis showing exact percentage scale
    ax_a_r = ax_a.twinx()
    ax_a_r.set_ylim(0, 36 / total_g1 * 100)
    ax_a_r.set_ylabel("Lesion Recovery Rate (%)", fontsize=11.0, weight='bold', color=C_NAVY, labelpad=8)
    ax_a_r.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax_a_r.tick_params(axis='y', labelsize=9.8, colors=C_SLATE)
    ax_a_r.spines['top'].set_visible(False)
    ax_a_r.spines['left'].set_visible(False)
    ax_a_r.spines['right'].set_color(C_BORDER)

    # Clean primary spine styling
    for spine in ['top', 'right']:
        ax_a.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax_a.spines[spine].set_color(C_BORDER)

    # =========================================================================
    # ROW 2 (PANEL B): G1 Component Volume Distribution (ECDF)
    # =========================================================================
    ax_b.set_facecolor('#FFFFFF')
    ax_b.set_title(r"$\mathbf{b}$   Empirical Cumulative Distribution Function (ECDF) of Missed Lesion Volumes",
                   loc='left', fontsize=12.2, weight='bold', color=C_NAVY, pad=12)

    ecdf_y = np.arange(1, n_lesions + 1) / n_lesions
    median_vol = np.median(volumes)       # 10.5 mm3
    mean_vol   = np.mean(volumes)         # 109.0 mm3
    pct_50mm3  = (volumes < 50).mean()    # 0.8457 (84.6%)

    ax_b.set_xscale('log')
    ax_b.grid(True, which='both', linestyle='--', alpha=0.45, color=C_GRID, zorder=0)

    # Fill area under ECDF
    ax_b.fill_between(volumes, ecdf_y, color=C_FILL, alpha=0.90, zorder=1)

    # Step curve
    ax_b.step(volumes, ecdf_y, where='post', color=C_ECDF, lw=2.4, zorder=3,
              label='Empirical CDF (N = 162 lesions)')

    # Rug plot of individual lesions
    ax_b.plot(volumes, np.zeros_like(volumes) + 0.015, '|', color='#3B82F6',
              markersize=9, markeredgewidth=1.0, alpha=0.65, zorder=2, label='Individual Lesion')

    # 1. Median Guideline (50% of lesions <= 10.5 mm3)
    ax_b.plot([3.5, median_vol, median_vol], [0.50, 0.50, 0.0],
              color=C_MED, lw=1.5, linestyle='--', zorder=4)
    ax_b.plot(median_vol, 0.50, marker='o', color=C_MED, markersize=7.0, zorder=5)
    ax_b.text(median_vol * 1.22, 0.44,
              f"Median: {median_vol:.1f} mm³\n(50% of missed satellites)",
              fontsize=9.0, weight='bold', color=C_MED, va='top',
              bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF1F2', edgecolor='#FECDD3', lw=1.1))

    # 2. 50 mm3 Threshold Guideline (84.6% of lesions < 50 mm3) - placed to the left of x=50
    ax_b.plot([3.5, 50, 50], [pct_50mm3, pct_50mm3, 0.0],
              color=C_85, lw=1.5, linestyle='--', zorder=4)
    ax_b.plot(50, pct_50mm3, marker='s', color=C_85, markersize=7.0, zorder=5)
    ax_b.text(45, pct_50mm3 + 0.02,
              f"84.6% < 50 mm³\n(Microscopic Satellites)",
              fontsize=9.0, weight='bold', color=C_85, va='bottom', ha='right',
              bbox=dict(boxstyle='round,pad=0.35', facecolor='#F0FDF4', edgecolor='#BBF7D0', lw=1.1))

    # 3. Mean Marker (109.0 mm3)
    mean_y = (volumes <= mean_vol).mean()
    ax_b.axvline(mean_vol, color=C_MEAN, lw=1.4, linestyle=':', zorder=4)
    ax_b.plot(mean_vol, mean_y, marker='^', color=C_MEAN, markersize=7.5, zorder=5)
    ax_b.text(mean_vol * 1.30, 0.30,
              f"Mean: {mean_vol:.1f} mm³\n(Skewed by rare tail)",
              fontsize=9.0, color=C_MEAN, weight='bold', va='center', ha='left',
              bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFFBEB', edgecolor='#FDE68A', lw=1.1))

    # Formatting axes
    ax_b.set_xlim(3.5, 9500)
    ax_b.set_ylim(-0.02, 1.05)
    ax_b.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_b.set_yticklabels([f"{p:.0f}%" for p in np.linspace(0, 100, 6)], fontsize=9.8, weight='bold', color=C_SLATE)

    ax_b.set_xticks([5, 10, 50, 100, 500, 1000, 5000])
    ax_b.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax_b.tick_params(axis='x', labelsize=9.8, colors=C_SLATE)
    
    ax_b.set_xlabel("G1 Satellite Lesion Volume (mm³, Log Scale)",
                    fontsize=11.2, weight='bold', color=C_NAVY, labelpad=8)
    ax_b.set_ylabel("Cumulative Proportion P(Volume ≤ V)",
                    fontsize=11.2, weight='bold', color=C_NAVY, labelpad=8)

    # Clean spines
    for spine in ['top', 'right']:
        ax_b.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax_b.spines[spine].set_color(C_BORDER)

    # Save both canonical naming conventions
    targets = [
        (FIG_DIR / 'fig7_g1_recovery.png', FIG_DIR / 'fig7_g1_recovery.svg'),
        (FIG_DIR / 'fig_7_g1_recovery.png', FIG_DIR / 'fig_7_g1_recovery.svg')
    ]
    for out_png, out_svg in targets:
        fig.savefig(out_png, dpi=350, facecolor='#FFFFFF', bbox_inches='tight')
        fig.savefig(out_svg, facecolor='#FFFFFF', bbox_inches='tight')
        print(f"Generated: {out_png.name} & {out_svg.name}")

    plt.close(fig)


if __name__ == '__main__':
    generate_fig7()
