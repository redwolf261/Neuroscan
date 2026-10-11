"""
Generate Figure 5: Statistical Performance Profile of Proposed NeuroScan System.
Optimized for Word Document embedding & Q1 Journal printing (IEEE TMI / MedIA):
- 2-Row Stacked Layout (Row 1: Panel a - Volumetric Overlap; Row 2: Panel b - Boundary Errors).
- High Legibility in Word: Aspect ratio ~1.15 (9.4 x 8.2 inches), fits 6.5-inch page margin with crisp 10-11pt text.
- Elegant 95% Confidence Interval error bars on mean statistics.
- Exact percentage and millimeter values annotated above each bar with ample clearance.
- Dual output: saves to both fig5_dice_performance and fig_5_dice_performance (PNG & SVG).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Publication styling matching IEEE TMI / MedIA, optimized for Word document legibility
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9.6,
    'axes.labelsize': 10.8,
    'axes.titlesize': 11.8,
    'xtick.labelsize': 9.8,
    'ytick.labelsize': 9.6,
    'legend.fontsize': 9.5,
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

# Metric colors (curated high-contrast palette)
C_DICE     = '#2563EB'   # Royal Blue (Mean)
C_DICE_MED = '#60A5FA'   # Sky Blue (Median)
C_PREC     = '#0D9488'   # Teal (Precision)
C_REC      = '#059669'   # Emerald (Recall)
C_IOU      = '#6366F1'   # Indigo (Jaccard)
C_GLOBAL   = '#1E293B'   # Dark Slate (Global Pooled)

C_HD95     = '#7C3AED'   # Deep Violet (HD95 Mean)
C_HD95_MED = '#A78BFA'   # Soft Violet (HD95 Median)
C_ASSD     = '#DB2777'   # Deep Rose (ASSD Mean)
C_ASSD_MED = '#F472B6'   # Soft Rose (ASSD Median)


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
    
    # 2-Row Stacked Canvas (9.6 x 8.8 inches)
    fig, (ax_a, ax_b) = plt.subplots(nrows=2, ncols=1, figsize=(9.6, 8.8), facecolor='#FFFFFF')
    plt.subplots_adjust(hspace=0.46, left=0.10, right=0.96, top=0.94, bottom=0.08)
    
    # =========================================================================
    # ROW 1 (PANEL A): Volumetric Overlap & Detection Performance
    # =========================================================================
    ax_a.grid(axis='y', linestyle='--', alpha=0.50, color='#CBD5E1', zorder=0)
    
    labels_a = [
        "ET Dice\n(Mean)",
        "ET Dice\n(Median)",
        "Precision\n(PPV)",
        "Recall\n(Sensitivity)",
        "Jaccard\n(IoU)",
        "Global Pooled\nVoxel Dice"
    ]
    
    vals_a = [dice_m, dice_med, prec_m, rec_m, iou_m, global_dice]
    colors_a = [C_DICE, C_DICE_MED, C_PREC, C_REC, C_IOU, C_GLOBAL]
    # 95% CI error bars for sample means, 0 for medians/global
    errs_a = [dice_ci, 0.0, prec_ci, rec_ci, iou_ci, 0.0]
    
    x_pos_a = np.arange(len(labels_a))
    width_a = 0.54
    
    bars_a = ax_a.bar(x_pos_a, vals_a, width=width_a, yerr=errs_a,
                      capsize=4.5, error_kw=dict(lw=1.3, capthick=1.3, ecolor='#1E293B'),
                      color=colors_a, edgecolor='#0F172A', linewidth=1.1,
                      zorder=3)
    
    # Label each bar with percentage & decimal score directly above bar / error cap
    for bar, val, err in zip(bars_a, vals_a, errs_a):
        y_text = val + err + 0.032
        pct_lbl = f"{val*100:.1f}%\n({val:.4f})"
        ax_a.text(bar.get_x() + bar.get_width() / 2, y_text, pct_lbl,
                  ha='center', va='bottom', fontsize=8.8, weight='bold', color=C_SLATE,
                  linespacing=1.15)
        
    ax_a.set_xticks(x_pos_a)
    ax_a.set_xticklabels(labels_a, weight='bold', fontsize=9.6)
    ax_a.set_xlabel("Volumetric Overlap & Detection Metrics", weight='bold', fontsize=10.5, labelpad=8)
    ax_a.set_ylabel("Metric Score [0.0 - 1.0]", weight='bold', fontsize=10.5, labelpad=8)
    ax_a.set_ylim(0, 1.22)
    ax_a.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_a.set_yticklabels([f"{y:.1f}" for y in np.linspace(0.0, 1.0, 6)], fontsize=9.4)
    ax_a.set_title(r"$\mathbf{a}$   Volumetric Overlap & Detection Performance ($N=125$ Locked Test Subjects)",
                   loc='left', weight='bold', pad=12, fontsize=11.5, color=C_NAVY)
    
    # =========================================================================
    # ROW 2 (PANEL B): Surface Boundary Distance Errors [mm]
    # =========================================================================
    ax_b.grid(axis='y', linestyle='--', alpha=0.50, color='#CBD5E1', zorder=0)
    
    labels_b = [
        "HD95 (Mean)\n95% Hausdorff",
        "HD95 (Median)\n95% Hausdorff",
        "ASSD (Mean)\nSymmetric Surface",
        "ASSD (Median)\nSymmetric Surface"
    ]
    
    vals_b = [hd95_m, hd95_med, assd_m, assd_med]
    colors_b = [C_HD95, C_HD95_MED, C_ASSD, C_ASSD_MED]
    errs_b = [hd95_ci, 0.0, assd_ci, 0.0]
    
    x_pos_b = np.arange(len(labels_b))
    width_b = 0.48
    
    bars_b = ax_b.bar(x_pos_b, vals_b, width=width_b, yerr=errs_b,
                      capsize=4.5, error_kw=dict(lw=1.3, capthick=1.3, ecolor='#1E293B'),
                      color=colors_b, edgecolor='#0F172A', linewidth=1.1,
                      zorder=3)
    
    # Label each bar with millimeter distance directly above bar / error cap
    for bar, val, err in zip(bars_b, vals_b, errs_b):
        y_text = val + err + 0.12
        mm_lbl = f"{val:.2f} mm"
        ax_b.text(bar.get_x() + bar.get_width() / 2, y_text, mm_lbl,
                  ha='center', va='bottom', fontsize=9.2, weight='bold', color=C_SLATE)
        
    ax_b.set_xticks(x_pos_b)
    ax_b.set_xticklabels(labels_b, weight='bold', fontsize=9.6)
    ax_b.set_xlabel("Surface Distance Error Metrics", weight='bold', fontsize=10.5, labelpad=8)
    ax_b.set_ylabel("Boundary Distance (mm)\n[↓ Lower is Better]", weight='bold', fontsize=10.5, labelpad=8)
    ax_b.set_ylim(0, 4.6)
    ax_b.set_yticks(np.arange(0, 5, 1))
    ax_b.set_yticklabels([f"{y} mm" for y in np.arange(0, 5, 1)], fontsize=9.4)
    ax_b.set_title(r"$\mathbf{b}$   Surface Boundary Distance Errors [mm, Lower is Better]",
                   loc='left', weight='bold', pad=12, fontsize=11.5, color=C_NAVY)
    
    # Save both canonical naming conventions
    targets = [
        (FIG_DIR / 'fig5_dice_performance.png', FIG_DIR / 'fig5_dice_performance.svg'),
        (FIG_DIR / 'fig_5_dice_performance.png', FIG_DIR / 'fig_5_dice_performance.svg')
    ]
    for out_png, out_svg in targets:
        fig.savefig(out_png, dpi=350, facecolor='#FFFFFF', bbox_inches='tight')
        fig.savefig(out_svg, facecolor='#FFFFFF', bbox_inches='tight')
        print(f"Generated: {out_png.name} & {out_svg.name}")
        
    plt.close(fig)


if __name__ == '__main__':
    generate_fig5()
