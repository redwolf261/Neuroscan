"""
Generate publication-quality horizontal bar chart comparing NeuroScan E292 B3
with selected brain tumor segmentation methods from the literature.
Strictly adheres to Q1 journal presentation standards (IEEE TMI / MedIA):
- Top-to-bottom numerical progression (0.8004 -> 0.8221)
- Truncated x-axis (0.79 - 0.83) clearly resolved
- Neutral slate-gray for published methods, restrained deep blue for NeuroScan
- Exact 4-decimal values annotated at bar ends
- Contextual comparability footnote at the bottom
- Zero 3D, zero gradients, zero superlative/SOTA claims
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Publication typography & styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Methods in top-to-bottom order (lowest to highest ET Dice)
methods_data = [
    {"label": "dResU-Net (2022)",         "dice": 0.8004, "is_proposed": False},
    {"label": "DeepGlioSeg (2025)",       "dice": 0.8080, "is_proposed": False},
    {"label": "CAGs-Net (2025)",          "dice": 0.8100, "is_proposed": False},
    {"label": "MS-SegNet (2024)",         "dice": 0.8100, "is_proposed": False},
    {"label": "BTS U-Net (2025)",         "dice": 0.8110, "is_proposed": False},
    {"label": "DKA-U-Net (2026)",         "dice": 0.8160, "is_proposed": False},
    {"label": "Proposed Model (2026)",    "dice": 0.8221, "is_proposed": True},
]

labels = [m["label"] for m in methods_data]
dices = [m["dice"] for m in methods_data]
is_proposed = [m["is_proposed"] for m in methods_data]

# Color palette: restrained slate-gray for prior work, deep accent blue for NeuroScan
C_PRIOR = "#4A5568"      # Slate gray
C_PROPOSED = "#2B6CB0"   # Restrained deep blue

colors = [C_PROPOSED if p else C_PRIOR for p in is_proposed]
edgecolors = ["#1A365D" if p else "#2D3748" for p in is_proposed]

# Canvas: 8.5 x 5.5 inches
fig, ax = plt.subplots(figsize=(8.8, 5.5), facecolor="white")

y_pos = np.arange(len(methods_data))
bar_height = 0.58

bars = ax.barh(y_pos, dices, height=bar_height, color=colors, edgecolor=edgecolors, linewidth=1.1, zorder=3)

# Display top-to-bottom progression (index 0 at top, index 6 at bottom)
ax.invert_yaxis()

# Annotate exact 4-decimal values at the end of each bar
for bar, d, p in zip(bars, dices, is_proposed):
    w = bar.get_width()
    fontweight = 'bold' if p else 'normal'
    textcolor = '#1A365D' if p else '#2D3748'
    ax.text(w + 0.0008, bar.get_y() + bar.get_height() / 2, f"{d:.4f}",
            va='center', ha='left', fontsize=10, weight=fontweight, color=textcolor)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10.5)

# X-axis configuration
ax.set_xlabel("ET Dice", fontsize=11.5, labelpad=8)
ax.set_xlim(0.79, 0.832)
ax.set_xticks([0.79, 0.80, 0.81, 0.82, 0.83])
ax.set_xticklabels(["0.79", "0.80", "0.81", "0.82", "0.83"])

# Vertical gridlines only (thin, light, behind bars)
ax.xaxis.grid(True, linestyle=":", alpha=0.6, color="#CBD5E0", zorder=0)

# Y-axis styling
ax.set_ylim(len(methods_data) - 0.4, -0.6)  # Padding above top bar and below bottom bar

# Figure title
ax.set_title("Contextual comparison of reported ET Dice across selected brain tumor segmentation methods",
             fontsize=12.2, weight='bold', pad=16)

# Methodological comparability footnote
fig.text(0.5, 0.015,
         "Published results use different datasets, splits, and evaluation protocols; comparison is contextual rather than a direct head-to-head benchmark.",
         ha="center", va="bottom", fontsize=8.8, style="italic", color="#4A5568")

plt.subplots_adjust(bottom=0.12, top=0.90, left=0.28, right=0.94)

# Save both 4K PNG (350 DPI) and infinite-resolution Vector SVG
out_png = FIG_DIR / 'fig_literature_comparison.png'
out_svg = FIG_DIR / 'fig_literature_comparison.svg'
plt.savefig(out_png, dpi=350, bbox_inches='tight')
plt.savefig(out_svg, bbox_inches='tight')
plt.close()

print(f"Figure successfully generated at:\n  {out_png}\n  {out_svg}")
