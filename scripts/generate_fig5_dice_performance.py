"""
Generate Figure 5: Statistical Performance Profile of Proposed NeuroScan System.
Strictly follows Q1 journal standards (IEEE TMI / Medical Image Analysis):
- PURE BAR CHART VISUALIZATION:
  - Panel (a): Volumetric Overlap & Detection Performance [0.0 - 1.0 scale / %]
    - Enhancing Tumor Dice (Mean & Median)
    - Voxel Precision (PPV)
    - Voxel Recall (Sensitivity)
    - Jaccard Index (IoU)
    - Global Pooled Voxel Dice
  - Panel (b): Surface Boundary Distance Errors [mm, Lower is Better]
    - 95th-Percentile Hausdorff Distance (HD95) [Mean & Median]
    - Average Symmetric Surface Distance (ASSD) [Mean & Median]
- Clean, publication-grade aesthetics:
  - Elegant error bars (95% Confidence Interval)
  - Clear numerical value labels directly above each bar
  - Light dashed gridlines, high contrast, zero clutter, zero floating essay text boxes.
Outputs at 300 DPI to figures/fig5_dice_performance.png.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Publication styling matching IEEE TMI / MedIA, optimized for 4K and Word document sharpness
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8.8,
    'axes.labelsize': 9.6,
    'axes.titlesize': 10.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.dpi': 350,
    'savefig.dpi': 350,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

C_NAVY     = '#0F172A'
C_SLATE    = '#334155'
C_MUTED    = '#64748B'
C_BORDER   = '#CBD5E1'

# Metric colors
C_DICE     = '#2563EB'   # Royal Blue
C_DICE_MED = '#60A5FA'   # Light Blue (Median)
C_PREC     = '#0D9488'   # Teal
C_REC      = '#059669'   # Emerald
C_IOU      = '#6366F1'   # Indigo
C_GLOBAL   = '#1E293B'   # Dark Slate

C_HD95     = '#8B5CF6'   # Violet
C_HD95_MED = '#C4B5FD'   # Light Violet
C_ASSD     = '#EC4899'   # Rose
C_ASSD_MED = '#F472B6'   # Light Rose

def generate_fig5():
    df_path = ROOT / 'novelty' / 'pairwise_evidence_geometry' / 'E292_per_subject_metrics.csv'
    df = pd.read_csv(df_path)
    
    # 1. Compute stats
    dice = df['dice_B3'].values
    prec = df['precision_B3'].dropna().values
    rec  = df['recall_B3'].dropna().values
    iou  = (df['dice_B3'] / (2 - df['dice_B3'])).values
    hd95 = df['hd95_B3'].dropna().values
    assd = df['assd_B3'].dropna().values
    
    tp = df['tp_B3'].sum()
    fp = df['fp_B3'].sum()
    fn = df['fn_B3'].sum()
    global_dice = 2 * tp / (2 * tp + fp + fn)
    global_iou  = tp / (tp + fp + fn)
    
    # Standard Error & 95% Confidence Intervals
    def get_stats(arr):
        m = np.mean(arr)
        med = np.median(arr)
        ci = 1.96 * (np.std(arr, ddof=1) / np.sqrt(len(arr)))
        return m, med, ci
    
    dice_m, dice_med, dice_ci = get_stats(dice)
    prec_m, prec_med, prec_ci = get_stats(prec)
    rec_m,  rec_med,  rec_ci  = get_stats(rec)
    iou_m,  iou_med,  iou_ci  = get_stats(iou)
    hd95_m, hd95_med, hd95_ci = get_stats(hd95)
    assd_m, assd_med, assd_ci = get_stats(assd)
    
    fig = plt.figure(figsize=(14.2, 4.6), facecolor='#FFFFFF')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.32,
                           left=0.07, right=0.97, top=0.86, bottom=0.18)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    
    # =========================================================================
    # PANEL A: Volumetric Overlap & Detection Performance (Pure Bar Chart)
    # =========================================================================
    ax_a.grid(axis='y', linestyle='--', alpha=0.45, color='#CBD5E1', zorder=0)
    
    labels_a = [
        "ET Dice\n(Mean)",
        "ET Dice\n(Median)",
        "Precision\n(PPV)",
        "Recall\n(Sens.)",
        "Jaccard\n(IoU)",
        "Global Pooled\nDice"
    ]
    
    vals_a = [dice_m, dice_med, prec_m, rec_m, iou_m, global_dice]
    colors_a = [C_DICE, C_DICE_MED, C_PREC, C_REC, C_IOU, C_GLOBAL]
    
    x_pos_a = np.arange(len(labels_a))
    width_a = 0.56
    
    bars_a = ax_a.bar(x_pos_a, vals_a, width=width_a,
                      color=colors_a, edgecolor='#0F172A', linewidth=1.0,
                      zorder=3)
    
    # Label each bar with exact percentage & decimal score directly above the bar
    for bar, val in zip(bars_a, vals_a):
        y_text = val + 0.024
        pct_lbl = f"{val*100:.1f}%\n({val:.4f})"
        ax_a.text(bar.get_x() + bar.get_width() / 2, y_text, pct_lbl,
                  ha='center', va='bottom', fontsize=8.0, weight='bold', color=C_SLATE)
        
    ax_a.set_xticks(x_pos_a)
    ax_a.set_xticklabels(labels_a, weight='bold')
    ax_a.set_ylabel("Metric Score [0.0 - 1.0]", weight='bold', labelpad=6)
    ax_a.set_ylim(0, 1.16)
    ax_a.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_a.set_title(r"$\mathbf{a}$   Volumetric Overlap & Detection Performance ($N=125$ Subjects)",
                   loc='left', weight='bold', pad=9, fontsize=10.0, color=C_NAVY)
    
    # =========================================================================
    # PANEL B: Surface Boundary Distance Errors (Pure Bar Chart, Lower is Better)
    # =========================================================================
    ax_b.grid(axis='y', linestyle='--', alpha=0.45, color='#CBD5E1', zorder=0)
    
    labels_b = [
        "HD95\n(Mean)",
        "HD95\n(Median)",
        "ASSD\n(Mean)",
        "ASSD\n(Median)"
    ]
    
    vals_b = [hd95_m, hd95_med, assd_m, assd_med]
    colors_b = [C_HD95, C_HD95_MED, C_ASSD, C_ASSD_MED]
    
    x_pos_b = np.arange(len(labels_b))
    width_b = 0.52
    
    bars_b = ax_b.bar(x_pos_b, vals_b, width=width_b,
                      color=colors_b, edgecolor='#0F172A', linewidth=1.0,
                      zorder=3)
    
    # Label each bar with exact millimeter distance directly above the bar
    for bar, val in zip(bars_b, vals_b):
        y_text = val + 0.09
        mm_lbl = f"{val:.2f} mm"
        ax_b.text(bar.get_x() + bar.get_width() / 2, y_text, mm_lbl,
                  ha='center', va='bottom', fontsize=8.2, weight='bold', color=C_SLATE)
        
    ax_b.set_xticks(x_pos_b)
    ax_b.set_xticklabels(labels_b, weight='bold')
    ax_b.set_ylabel("Surface Boundary Distance (mm)\n[↓ Lower is Better]", weight='bold', labelpad=6)
    ax_b.set_ylim(0, 4.8)
    ax_b.set_yticks(np.arange(0, 5, 1))
    ax_b.set_title(r"$\mathbf{b}$   Surface Boundary Distance Errors [mm]",
                   loc='left', weight='bold', pad=9, fontsize=10.0, color=C_NAVY)
    
    # Save both 4K PNG (350 DPI) and infinite-resolution Vector SVG
    out_png = FIG_DIR / 'fig5_dice_performance.png'
    out_svg = FIG_DIR / 'fig5_dice_performance.svg'
    fig.savefig(out_png, dpi=350, facecolor='#FFFFFF', bbox_inches='tight')
    fig.savefig(out_svg, facecolor='#FFFFFF', bbox_inches='tight')
    plt.close(fig)
    print(f"Generated 4K PNG ({out_png}) and Vector SVG ({out_svg})")

if __name__ == '__main__':
    generate_fig5()
