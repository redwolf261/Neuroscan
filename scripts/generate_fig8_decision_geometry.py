"""
Generate Figure 8: Complementary Decision Geometry in Shared Representation Space.
Strictly follows Q1 journal standards (IEEE TMI / Medical Image Analysis):
- PURE SCIENTIFIC GRAPHICS: Zero marketing cards, zero text essays, zero AI slop.
- Panel (a): Real component-level P_ET,max vs R_MRD,max from the actual 21,718 training
  component observations in E290_train_records.pt (146 ground-truth rescues, 21,572 candidates).
  - Highlights operating thresholds: tau=0.50 (production), tau*=0.20 (calibrated core),
    tau_R*=0.9508 (MRD operating point).
  - Demonstrates that 96.6% of real rescued lesions have P_ET,max < 0.20.
- Panel (b): Shared-D1 hyperplane geometry across 5 independent seeds:
  - 77.58°, 75.96°, 78.19°, 77.05°, 76.40° (Mean: 77.04° ± 0.89°).
  - Statement: "Distinct decision directions in the shared decoder representation — not orthogonal".
- Panel (c): Conceptual 2D projection schematic:
  - Explicitly labeled as: "[Conceptual 2D Schematic — Not Actual 32-D Coordinates]".
  - Illustrates production decision direction w_P vs MRD decision direction w_R.
Outputs at 300 DPI to figures/fig8_decision_geometry.png.
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8.5,
    'axes.labelsize': 9.2,
    'axes.titlesize': 9.8,
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

C_PROD     = '#0284C7'   # Sky/Cyan for Production
C_MRD      = '#2563EB'   # Royal Blue for MRD
C_TP       = '#059669'   # Emerald for True Positive Rescues
C_FP       = '#94A3B8'   # Muted slate for candidates
C_ALERT    = '#DC2626'   # Red for threshold lines

def generate_fig8():
    # --------------------------------------------------------------------------
    # Load Real 21,718 Training Component Observations
    # --------------------------------------------------------------------------
    rec_path = ROOT / 'novelty' / 'pairwise_evidence_geometry' / 'E290_train_records.pt'
    train_recs = torch.load(rec_path, map_location='cpu', weights_only=False)
    
    feats = np.stack([r['feat'] for r in train_recs])
    labels = np.array([r['label'] for r in train_recs])
    
    # Feature 8: max_p (P_ET,max), Feature 2: max_r (R_MRD,max)
    p_max = feats[:, 8]
    r_max = feats[:, 2]
    
    tp_mask = (labels == 1)
    fp_mask = (labels == 0)
    
    fig = plt.figure(figsize=(15.6, 4.8), facecolor='#FFFFFF')
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.95, 1.15], wspace=0.28,
                           left=0.06, right=0.97, top=0.86, bottom=0.14)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    
    # ==========================================================================
    # PANEL A: Real Component-Level P_ET,max vs R_MRD,max (N = 21,718 Observations)
    # ==========================================================================
    ax_a.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1', zorder=0)
    
    # 1. Background Candidates (N = 21,572, subsampled scatter for clarity)
    np.random.seed(42)
    sub_idx = np.random.choice(np.where(fp_mask)[0], size=5000, replace=False)
    ax_a.scatter(p_max[sub_idx], r_max[sub_idx], color='#94A3B8', alpha=0.18, s=14,
                 edgecolors='none', zorder=1, label=r'Candidate Components ($N = 21,572$)')
    
    # 2. Real Ground-Truth Rescued Lesions (N = 146)
    ax_a.scatter(p_max[tp_mask], r_max[tp_mask], color=C_TP, alpha=0.90, s=36,
                 edgecolors='#064E3B', linewidths=0.7, zorder=3,
                 label=r'True Rescued Lesions ($N = 146$)')
    
    # 3. Operating Threshold Guidelines
    tau_mrd = 0.95082877
    tau_star = 0.20
    tau_prod = 0.50
    
    # Subtle shaded recovery zone (P_ET < 0.20 and R_MRD >= tau_R*)
    ax_a.axvspan(-0.01, tau_star, ymin=(tau_mrd - 0.949)/(1.001 - 0.949), ymax=1.0,
                 color='#10B981', alpha=0.07, zorder=0)
    
    ax_a.axhline(tau_mrd, color='#D97706', lw=1.2, linestyle='--', zorder=4)
    ax_a.axvline(tau_star, color=C_PROD, lw=1.2, linestyle='--', zorder=4)
    ax_a.axvline(tau_prod, color='#475569', lw=1.1, linestyle=':', zorder=4)
    
    # Threshold Labels - positioned in clean unobstructed open areas
    ax_a.text(0.32, tau_mrd + 0.0015, r"$\tau_R^* = 0.9508$ (MRD threshold)",
              fontsize=7.3, weight='bold', color='#B45309')
    ax_a.text(tau_star + 0.008, 0.985, r"$\tau^* = 0.20$" + "\n(Core thresh.)",
              fontsize=7.3, weight='bold', color=C_PROD, va='top')
    ax_a.text(tau_prod - 0.008, 0.965, r"$\tau_P = 0.50$" + "\n(Prod. thresh.)",
              fontsize=7.1, color='#475569', ha='right', va='center')
    
    # Clean non-boxed label for key finding
    pct_under_20 = (p_max[tp_mask] < 0.20).mean() * 100
    ax_a.text(0.015, 0.966, rf"Rescued Lesions: $\mathbf{{{pct_under_20:.1f}\%}}$ have $P_{{\mathrm{{ET,max}}}} < 0.20$",
              fontsize=7.5, color='#065F46', weight='bold', zorder=5)
    
    ax_a.set_xlabel(r"Production Confidence $P_{\mathrm{ET,max}}$", weight='bold', labelpad=5)
    ax_a.set_ylabel(r"MRD Readout Confidence $R_{\mathrm{MRD,max}}$", weight='bold', labelpad=5)
    ax_a.set_xlim(-0.01, 0.52)
    ax_a.set_ylim(0.949, 1.001)
    ax_a.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1',
                framealpha=0.95, fontsize=7.5)
    ax_a.set_title(r"$\mathbf{a}$   Real Component Evidence Distribution ($N = 21,718$)",
                   loc='left', weight='bold', pad=9, fontsize=9.2, color=C_NAVY)

    # ==========================================================================
    # PANEL B: Shared-D1 Hyperplane Geometry (Angular Separation across 5 Seeds)
    # ==========================================================================
    seeds = ["Seed 999", "Seed 4242", "Seed 7", "Seed 123", "Seed 2024"]
    angles = [77.58, 75.96, 78.19, 77.05, 76.40]
    mean_angle = 77.04
    std_angle = 0.89
    
    ax_b.set_aspect('equal')
    ax_b.set_xlim(-0.15, 1.30)
    ax_b.set_ylim(-0.25, 1.20)
    ax_b.axis('off')
    
    # Origin
    ox, oy = 0.12, 0.08
    vec_len = 0.82
    
    # Vector w_P (Production Head, 0 deg)
    ax_b.annotate('', xy=(ox + vec_len, oy), xytext=(ox, oy),
                  arrowprops=dict(arrowstyle="-|>", color=C_PROD, lw=2.2, mutation_scale=13), zorder=3)
    ax_b.text(ox + vec_len + 0.04, oy, r"$\mathbf{w}_P$" + "\n(Production)",
              fontsize=7.8, weight='bold', color=C_PROD, va='center')
    
    # Vector w_R (MRD Readout, 77.04 deg)
    theta_rad = np.radians(mean_angle)
    wx = ox + vec_len * np.cos(theta_rad)
    wy = oy + vec_len * np.sin(theta_rad)
    ax_b.annotate('', xy=(wx, wy), xytext=(ox, oy),
                  arrowprops=dict(arrowstyle="-|>", color=C_MRD, lw=2.2, mutation_scale=13), zorder=3)
    ax_b.text(wx, wy + 0.05, r"$\mathbf{w}_R$" + "\n(MRD)",
              fontsize=7.8, weight='bold', color=C_MRD, ha='center', va='bottom')
    
    # 90-degree Orthogonal Reference (Dotted Line)
    orth_x = ox
    orth_y = oy + vec_len
    ax_b.plot([ox, orth_x], [oy, orth_y], color='#94A3B8', lw=1.1, linestyle=':', zorder=2)
    ax_b.text(orth_x - 0.03, orth_y, r"$90^\circ$ (Orthogonal)",
              fontsize=6.8, color='#64748B', ha='right', va='center')
    
    # Arc showing theta = 77.04 deg
    arc = Arc((ox, oy), width=0.40, height=0.40, angle=0, theta1=0, theta2=mean_angle,
              color='#E11D48', lw=1.6, zorder=4)
    ax_b.add_patch(arc)
    ax_b.text(ox + 0.25 * np.cos(theta_rad / 2), oy + 0.25 * np.sin(theta_rad / 2),
              rf"$\theta = {mean_angle:.2f}^\circ$",
              fontsize=8.2, weight='bold', color='#E11D48', ha='center', va='center')
    
    # 5-Seed Angular Replications mini-table
    y_seed_start = 0.72
    ax_b.text(0.68, y_seed_start, "5-Seed Replications:", fontsize=7.4, weight='bold', color=C_NAVY)
    for s_i, (s_name, s_ang) in enumerate(zip(seeds, angles)):
        y_pos_s = y_seed_start - 0.075 * (s_i + 1)
        ax_b.plot(0.70, y_pos_s, marker='o', markersize=4, color=C_MRD)
        ax_b.text(0.75, y_pos_s, f"{s_name}: {s_ang:.2f}°", fontsize=7.0, color=C_SLATE, va='center')
        
    # User-specified required text:
    # "Distinct decision directions in the shared decoder representation — not orthogonal"
    ax_b.text(0.55, -0.16,
              "Distinct decision directions in the shared decoder representation\n" +
              rf"($\theta = {mean_angle:.2f}^\circ \pm {std_angle:.2f}^\circ$, not orthogonal)",
              fontsize=7.4, weight='bold', color=C_NAVY, ha='center', va='center')
    
    ax_b.set_title(r"$\mathbf{b}$   Shared-$D_1$ Hyperplane Geometry",
                   loc='left', weight='bold', pad=9, fontsize=9.2, color=C_NAVY)

    # ==========================================================================
    # PANEL C: Conceptual 2D Projection (Schematic of Latent Feature Space)
    # ==========================================================================
    ax_c.set_xlim(-3.2, 3.2)
    ax_c.set_ylim(-2.6, 2.7)
    ax_c.axis('off')
    
    # Header Badge: Explicitly Conceptual
    ax_c.text(0.0, 2.50, "[Conceptual 2D Schematic — Not Actual 32-D Coordinates]",
              ha='center', va='center', fontsize=7.0, weight='bold', color='#64748B')
    
    # Hyperplane 1: Production Decision Boundary (w_P . D_1 + b_P = 0)
    # Positioned at x = 0.0
    hp1_x = 0.0
    ax_c.plot([hp1_x, hp1_x], [-2.1, 2.05], color=C_PROD, lw=1.8, linestyle='-', zorder=2)
    ax_c.annotate('', xy=(hp1_x + 1.25, -0.1), xytext=(hp1_x, -0.1),
                  arrowprops=dict(arrowstyle="-|>", color=C_PROD, lw=1.8, mutation_scale=12), zorder=3)
    ax_c.text(hp1_x + 0.65, 0.05, r"$\mathbf{w}_P$ direction",
              fontsize=7.2, weight='bold', color=C_PROD, ha='center')
    ax_c.text(hp1_x, 2.18, "Production Hyperplane: " + r"$\mathbf{w}_P^T D_1 + b_P = 0$",
              fontsize=7.0, color=C_PROD, ha='center', weight='bold')
    
    # Hyperplane 2: MRD Residual Decision Boundary (at 77 deg to w_P)
    ang_hp = np.radians(-13)
    x_hp = np.array([-2.8, 2.8])
    y_hp = x_hp * np.tan(ang_hp) - 0.45
    ax_c.plot(x_hp, y_hp, color=C_MRD, lw=1.8, linestyle='-', zorder=2)
    
    # Normal vector w_R pointing at ~77 deg
    mrd_center_x = 0.3
    mrd_center_y = -0.52
    mrd_vec_x = mrd_center_x + 1.15 * np.cos(np.radians(77))
    mrd_vec_y = mrd_center_y + 1.15 * np.sin(np.radians(77))
    ax_c.annotate('', xy=(mrd_vec_x, mrd_vec_y), xytext=(mrd_center_x, mrd_center_y),
                  arrowprops=dict(arrowstyle="-|>", color=C_MRD, lw=1.8, mutation_scale=12), zorder=3)
    ax_c.text(mrd_vec_x + 0.10, mrd_vec_y, r"$\mathbf{w}_R$ direction" + "\n" + r"($77.04^\circ$ separation)",
              fontsize=7.2, weight='bold', color=C_MRD, va='bottom')
    ax_c.text(-2.6, 0.35, "MRD Hyperplane\n" + r"$\mathbf{w}_R^T D_1 + b_R = 0$",
              fontsize=7.0, color=C_MRD, ha='left', weight='bold')
    
    # Clusters of voxels:
    # 1. Main Enhancing Tumor Core (High w_P, High w_R)
    np.random.seed(99)
    core_x = np.random.normal(1.6, 0.28, 40)
    core_y = np.random.normal(1.1, 0.28, 40)
    ax_c.scatter(core_x, core_y, color='#0284C7', alpha=0.55, s=22, edgecolors='none', zorder=4)
    ax_c.text(1.6, 1.65, "Main Tumor Core\n" + r"($P_{\mathrm{ET}} \geq \tau^*$)",
              fontsize=7.0, weight='bold', color='#0369A1', ha='center')
    
    # 2. Rescued Satellite Lesions (LOW w_P, HIGH w_R)
    sat_x = np.random.normal(-1.4, 0.25, 25)
    sat_y = np.random.normal(1.1, 0.25, 25)
    ax_c.scatter(sat_x, sat_y, color=C_TP, alpha=0.85, s=34, edgecolors='#064E3B', lw=0.8, zorder=4)
    ax_c.text(-1.4, 1.65, "Rescued Satellites\n" + r"($P_{\mathrm{ET}} < \tau^*,\ R_{\mathrm{MRD}} \geq \tau_R^*$)",
              fontsize=7.0, weight='bold', color='#065F46', ha='center')
    
    # 3. Distant Background / False Positives (LOW w_P, LOW w_R)
    bg_x = np.random.normal(-1.6, 0.35, 45)
    bg_y = np.random.normal(-1.4, 0.35, 45)
    ax_c.scatter(bg_x, bg_y, color='#94A3B8', alpha=0.40, s=15, edgecolors='none', zorder=4)
    ax_c.text(-1.6, -2.0, "Background / Healthy Tissue\n" + r"($P_{\mathrm{ET}} < \tau^*,\ R_{\mathrm{MRD}} < \tau_R^*$)",
              fontsize=6.8, color='#64748B', ha='center')
    
    ax_c.set_title(r"$\mathbf{c}$   Conceptual Representation Geometry",
                   loc='left', weight='bold', pad=9, fontsize=9.2, color=C_NAVY)

    # Save both 4K PNG (350 DPI) and infinite-resolution Vector SVG
    out_png = FIG_DIR / 'fig8_decision_geometry.png'
    out_svg = FIG_DIR / 'fig8_decision_geometry.svg'
    plt.savefig(out_png, dpi=350, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.savefig(out_svg, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated 4K PNG ({out_png}) and Vector SVG ({out_svg})")

if __name__ == '__main__':
    generate_fig8()
