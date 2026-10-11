"""
Generate publication-grade Figure 11 for IEEE TMI / MedIA:
Free-Response ROC (FROC) Lesion Detection Sensitivity & Size-Stratified Recovery Breakdown.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------------------
# Publication styling parameters
# ------------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 8.8,
    'axes.labelsize': 9.2,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8.4,
    'ytick.labelsize': 8.4,
    'legend.fontsize': 8.2,
    'figure.titlesize': 10.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

C_NAVY = '#0F172A'
C_SLATE = '#475569'
C_BASE = '#334155'
C_MRD = '#0D9488'
C_MRD_LIGHT = '#99F6E4'
C_OPT = '#059669'
C_BAR_BASE = '#64748B'
C_BAR_500 = '#0EA5E9'
C_BAR_1000 = '#0D9488'

def generate_fig11():
    froc_path = ROOT / 'novelty/pairwise_evidence_geometry/E286_froc.csv'
    size_path = ROOT / 'novelty/pairwise_evidence_geometry/E286_size_stratified.csv'
    
    df_froc = pd.read_csv(froc_path)
    df_size = pd.read_csv(size_path)
    
    fig = plt.figure(figsize=(14.8, 4.6), facecolor='#FFFFFF')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28,
                           left=0.06, right=0.97, top=0.86, bottom=0.15)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    
    # ==========================================================================
    # PANEL A: Free-Response ROC (FROC) Curve for Missed-Lesion Recovery
    # ==========================================================================
    ax_a.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1', zorder=0)
    
    # 1. Baseline curve
    prod = df_froc[df_froc['source'] == 'prod'].sort_values('mean_fp')
    prod_valid = prod[(prod['mean_fp'] >= 100) & (prod['mean_fp'] <= 10000)]
    ax_a.plot(prod_valid['mean_fp'], prod_valid['sens_G1_confidently_missed'] * 100,
              color=C_BASE, linestyle='--', lw=2.0, label='Baseline UNet3D ($B_0$)', zorder=2)
    
    # 2. Individual MRD seeds
    mrd_seeds = [s for s in df_froc['source'].unique() if 'h3_mrd_t2' in s]
    all_fps = np.logspace(np.log10(120), np.log10(9500), 100)
    interp_sens = []
    
    for s in mrd_seeds:
        sub = df_froc[df_froc['source'] == s].sort_values('mean_fp')
        sub_valid = sub[(sub['mean_fp'] >= 100) & (sub['mean_fp'] <= 10000)]
        ax_a.plot(sub_valid['mean_fp'], sub_valid['sens_G1_confidently_missed'] * 100,
                  color=C_MRD, alpha=0.25, lw=1.0, zorder=1)
        # Interpolate for mean
        ys = np.interp(all_fps, sub['mean_fp'], sub['sens_G1_confidently_missed'] * 100)
        interp_sens.append(ys)
        
    # Mean MRD curve
    mean_sens = np.mean(interp_sens, axis=0)
    std_sens = np.std(interp_sens, axis=0)
    ax_a.plot(all_fps, mean_sens, color=C_MRD, lw=2.4,
              label=r'MRD Evidence $R(x)$ (5-Seed Mean)', zorder=3)
    ax_a.fill_between(all_fps, mean_sens - std_sens, mean_sens + std_sens,
                      color=C_MRD_LIGHT, alpha=0.35, zorder=1)
    
    # 3. Proposed Operating Point (NeuroScan B3)
    opt_fp = 1828.6
    opt_rec = 16.0
    ax_a.scatter([opt_fp], [opt_rec], color=C_OPT, s=130, marker='*',
                 edgecolors='#0F172A', linewidths=1.2, zorder=5,
                 label=r'Proposed NeuroScan ($B_3$, Net $+0.0047$ Dice)')
    
    ax_a.annotate(f"Proposed Operating Point\n(FP={opt_fp:.0f}, Recovery={opt_rec:.1f}%)",
                  xy=(opt_fp, opt_rec), xytext=(3200, 11),
                  arrowprops=dict(arrowstyle="->", color=C_OPT, lw=1.2),
                  fontsize=7.8, weight='bold', color='#065F46', ha='left')
    
    # Formatting Panel A
    ax_a.set_xscale('log')
    ax_a.set_xlim(100, 10000)
    ax_a.set_ylim(-3, 72)
    ax_a.set_xlabel(r"False-Positive Burden: Mean FP Voxels / Scan (Log Scale)", weight='bold', labelpad=6)
    ax_a.set_ylabel(r"Missed Lesion ($G_1$) Detection Sensitivity (%)", weight='bold', labelpad=6)
    ax_a.legend(loc='upper left', frameon=True, framealpha=0.92, facecolor='#FFFFFF', edgecolor='#E2E8F0')
    ax_a.set_title(r"$\mathbf{a}$   Free-Response ROC (FROC) Curve for Missed Satellite Detection",
                   loc='left', weight='bold', pad=9, fontsize=9.2, color=C_NAVY)
    
    # ==========================================================================
    # PANEL B: Lesion Size-Stratified Sensitivity Breakdown
    # ==========================================================================
    ax_b.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1', zorder=0, axis='y')
    
    # Aggregate size-stratified data
    mrd_size = df_size[df_size['source'].str.contains('h3_mrd_t2')]
    stats_500 = mrd_size[mrd_size['fp_target'] == 500].groupby('size_bucket')['sensitivity'].agg(['mean', 'std'])
    stats_1000 = mrd_size[mrd_size['fp_target'] == 1000].groupby('size_bucket')['sensitivity'].agg(['mean', 'std'])
    
    # Order: small, medium, large
    buckets = ['small', 'medium', 'large']
    bucket_labels = [
        "Micro-Lesions\n($<50\\text{ mm}^3$, $N=66$)",
        "Intermediate\n($50\\text{--}200\\text{ mm}^3$, $N=39$)",
        "Macro-Lesions\n($>200\\text{ mm}^3$, $N=57$)"
    ]
    
    y_base = [0.0, 0.0, 0.0]
    y_500 = [stats_500.loc[b, 'mean'] * 100 for b in buckets]
    err_500 = [stats_500.loc[b, 'std'] * 100 for b in buckets]
    y_1000 = [stats_1000.loc[b, 'mean'] * 100 for b in buckets]
    err_1000 = [stats_1000.loc[b, 'std'] * 100 for b in buckets]
    
    x = np.arange(len(buckets))
    width = 0.24
    
    rects1 = ax_b.bar(x - width, y_base, width, label='Baseline ($B_0$, Any FP)',
                      color=C_BAR_BASE, edgecolor='#0F172A', linewidth=0.9, zorder=3)
    rects2 = ax_b.bar(x, y_500, width, yerr=err_500, capsize=3,
                      label=r'MRD Evidence (FP $\leq 500$)',
                      color=C_BAR_500, edgecolor='#0F172A', linewidth=0.9, zorder=3)
    rects3 = ax_b.bar(x + width, y_1000, width, yerr=err_1000, capsize=3,
                      label=r'MRD Evidence (FP $\leq 1000$)',
                      color=C_BAR_1000, edgecolor='#0F172A', linewidth=0.9, zorder=3)
    
    # Annotate bar values cleanly above error bars
    for r in rects1:
        ax_b.text(r.get_x() + r.get_width() / 2, 1.2, "0.0%",
                  ha='center', va='bottom', fontsize=7.2, color='#334155', weight='bold')
    for r, val, err in zip(rects2, y_500, err_500):
        ax_b.text(r.get_x() + r.get_width() / 2, val + err + 1.8, f"{val:.1f}%",
                  ha='center', va='bottom', fontsize=7.2, color='#0284C7', weight='bold')
    for r, val, err in zip(rects3, y_1000, err_1000):
        ax_b.text(r.get_x() + r.get_width() / 2, val + err + 1.8, f"{val:.1f}%",
                  ha='center', va='bottom', fontsize=7.2, color='#0F766E', weight='bold')
        
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(bucket_labels, weight='bold')
    ax_b.set_ylabel(r"Lesion Detection Sensitivity (%)", weight='bold', labelpad=6)
    ax_b.set_ylim(0, 56)
    ax_b.legend(loc='upper left', frameon=True, framealpha=0.92, facecolor='#FFFFFF', edgecolor='#E2E8F0')
    ax_b.set_title(r"$\mathbf{b}$   Lesion Size-Stratified Sensitivity Breakdown",
                   loc='left', weight='bold', pad=9, fontsize=9.2, color=C_NAVY)
    
    # Clean spines
    for ax in [ax_a, ax_b]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#94A3B8')
        ax.spines['bottom'].set_color('#94A3B8')
        
    # Save both 4K PNG (350 DPI) and infinite-resolution Vector SVG
    out_png = FIG_DIR / 'fig11_froc_sensitivity.png'
    out_svg = FIG_DIR / 'fig11_froc_sensitivity.svg'
    plt.savefig(out_png, dpi=350, bbox_inches='tight')
    plt.savefig(out_svg, bbox_inches='tight')
    plt.close()
    print(f"Generated 4K PNG ({out_png}) and Vector SVG ({out_svg})")

if __name__ == '__main__':
    generate_fig11()
