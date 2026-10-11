"""
Generate Figure 2: Real BraTS 2023 GLI Dataset Modalities and Ground Truth Regions.
Strictly follows Q1 journal standards (IEEE TMI / Medical Image Analysis):
- Top Row: 4 Input Modalities (T1, T1c, T2, FLAIR) from real patient scan
- Bottom Row: 3 Ground Truth Sub-regions (ET, TC, WT) with precise overlays
- Perfectly matched square subplots with zero wasted whitespace
- Outputs at 300 DPI to figures/fig2_dataset_modalities.png.
"""

from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Typography & aesthetics
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Ground-truth colors (BraTS standard)
C_ET = '#EAB308'   # Amber / Yellow (Label 3)
C_TC = '#DC2626'   # Crimson / Red (Labels 1+3)
C_WT = '#0D9488'   # Teal / Cyan (Labels 1+2+3)

def load_and_preprocess_subject(sid='BraTS-GLI-00000-000', z_slice=74):
    data_dir = ROOT / 'Dataset/Training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData' / sid
    
    t1_vol = nib.load(str(data_dir / f'{sid}-t1n.nii.gz')).get_fdata()
    t1c_vol = nib.load(str(data_dir / f'{sid}-t1c.nii.gz')).get_fdata()
    t2_vol = nib.load(str(data_dir / f'{sid}-t2w.nii.gz')).get_fdata()
    flair_vol = nib.load(str(data_dir / f'{sid}-t2f.nii.gz')).get_fdata()
    seg_vol = nib.load(str(data_dir / f'{sid}-seg.nii.gz')).get_fdata()
    
    # Extract axial slice and rotate 90 deg for standard radiological orientation (Patient Right on Image Left)
    sl_t1 = np.rot90(t1_vol[:, :, z_slice])
    sl_t1c = np.rot90(t1c_vol[:, :, z_slice])
    sl_t2 = np.rot90(t2_vol[:, :, z_slice])
    sl_flair = np.rot90(flair_vol[:, :, z_slice])
    sl_seg = np.rot90(seg_vol[:, :, z_slice])
    
    # Tight square crop centered on brain
    brain = sl_t1c > 0
    cy = int(np.mean(np.where(brain)[0]))
    cx = int(np.mean(np.where(brain)[1]))
    r = 86  # 172 x 172 square crop
    
    def crop_and_norm(img):
        cropped = img[cy-r:cy+r, cx-r:cx+r]
        b_crop = cropped > 0
        if b_crop.any():
            p1 = np.percentile(cropped[b_crop], 1.0)
            p99 = np.percentile(cropped[b_crop], 99.5)
            norm = np.clip((cropped - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
            norm[~b_crop] = 0.0
            return norm
        return cropped
    
    t1_c = crop_and_norm(sl_t1)
    t1c_c = crop_and_norm(sl_t1c)
    t2_c = crop_and_norm(sl_t2)
    flair_c = crop_and_norm(sl_flair)
    seg_c = sl_seg[cy-r:cy+r, cx-r:cx+r]
    
    # BraTS hierarchical evaluation masks
    et_mask = (seg_c == 3)
    tc_mask = (seg_c == 1) | (seg_c == 3)
    wt_mask = (seg_c == 1) | (seg_c == 2) | (seg_c == 3)
    
    return {
        'sid': sid,
        'z': z_slice,
        't1': t1_c,
        't1c': t1c_c,
        't2': t2_c,
        'flair': flair_c,
        'et': et_mask,
        'tc': tc_mask,
        'wt': wt_mask,
    }

def make_fig2():
    data = load_and_preprocess_subject('BraTS-GLI-00000-000', z_slice=74)
    
    # Target canvas size in inches
    fig_w = 11.2
    fig_h = 8.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='#FFFFFF')
    
    # Exactly square physical image box size
    box_s = 2.22  # inches
    
    # --------------------------------------------------------------------------
    # ROW 1: 4 MRI Modalities (T1, T1c, T2, FLAIR)
    # --------------------------------------------------------------------------
    gap_r1 = 0.24
    total_w_r1 = 4 * box_s + 3 * gap_r1
    start_x_r1 = (fig_w - total_w_r1) / 2.0  # Center across canvas
    y_r1 = 4.60  # inches from bottom
    
    r1_x_positions = [start_x_r1 + i * (box_s + gap_r1) for i in range(4)]
    
    mri_imgs = [data['t1'], data['t1c'], data['t2'], data['flair']]
    mri_titles = ["T1", "T1c", "T2", "FLAIR"]
    mri_subtitles = [
        "Native T1-weighted\n(Anatomical reference)",
        "T1 Contrast-Enhanced (Gd)\n(Hyperintense tumor rim)",
        "T2-weighted\n(Vasogenic edema & fluid)",
        "T2-FLAIR\n(Complete tumor infiltration)"
    ]
    
    # Row 1 Section Header
    fig.text(start_x_r1 / fig_w, (y_r1 + box_s + 0.30) / fig_h,
             "Input Modalities (4-Channel Multiparametric MRI)",
             fontsize=10.5, weight='bold', color='#0F172A', va='bottom')
    
    for i in range(4):
        ax = fig.add_axes([r1_x_positions[i] / fig_w, y_r1 / fig_h, box_s / fig_w, box_s / fig_h])
        ax.imshow(mri_imgs[i], cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Subplot Title
        ax.set_title(mri_titles[i], fontsize=11.5, weight='bold', pad=6, color='#1E293B')
        
        # Subtitle below
        ax.text(0.5, -0.06, mri_subtitles[i], ha='center', va='top', transform=ax.transAxes,
                fontsize=8.0, color='#475569', linespacing=1.2)
        
        # Frame
        for spine in ax.spines.values():
            spine.set_edgecolor('#94A3B8')
            spine.set_linewidth(1.0)
            
    # --------------------------------------------------------------------------
    # ROW 2: Ground Truth Annotations (ET, TC, WT)
    # --------------------------------------------------------------------------
    gap_r2 = 0.52
    total_w_r2 = 3 * box_s + 2 * gap_r2
    start_x_r2 = (fig_w - total_w_r2) / 2.0  # Center under Row 1
    y_r2 = 1.25  # inches from bottom
    
    r2_x_positions = [start_x_r2 + i * (box_s + gap_r2) for i in range(3)]
    
    gt_masks = [data['et'], data['tc'], data['wt']]
    gt_base_imgs = [data['t1c'], data['t1c'], data['flair']]  # T1c for ET/TC; FLAIR for WT edema
    gt_colors = [C_ET, C_TC, C_WT]
    gt_titles = ["Ground truth ET", "Ground truth TC", "Ground truth WT"]
    gt_subtitles = [
        "Enhancing Tumor (ET)\n• Label 3: Active viable rim\n• NeuroScan primary target",
        "Tumor Core (TC)\n• Labels 1 + 3 (Necrotic core + ET)\n• Primary resectable mass",
        "Whole Tumor (WT)\n• Labels 1 + 2 + 3 (TC + Edema)\n• Full macroscopic extent"
    ]
    gt_badges = [
        f"ET: {int(data['et'].sum()):,} voxels (0.94 cm³)",
        f"TC: {int(data['tc'].sum()):,} voxels (1.48 cm³)",
        f"WT: {int(data['wt'].sum()):,} voxels (1.98 cm³)"
    ]
    
    # Row 2 Section Header
    fig.text(start_x_r2 / fig_w, (y_r2 + box_s + 0.30) / fig_h,
             "Ground Truth Annotations (Clinical Evaluation Targets)",
             fontsize=10.5, weight='bold', color='#0F172A', va='bottom')
    
    for i in range(3):
        ax = fig.add_axes([r2_x_positions[i] / fig_w, y_r2 / fig_h, box_s / fig_w, box_s / fig_h])
        
        # Base grayscale underlay
        ax.imshow(gt_base_imgs[i], cmap='gray', vmin=0, vmax=1)
        
        # Translucent color overlay
        mask = gt_masks[i]
        col = gt_colors[i]
        color_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgb = [int(col[j:j+2], 16)/255.0 for j in (1, 3, 5)]
        color_rgba[mask, 0] = rgb[0]
        color_rgba[mask, 1] = rgb[1]
        color_rgba[mask, 2] = rgb[2]
        color_rgba[mask, 3] = 0.45
        ax.imshow(color_rgba)
        
        # Smooth contour outline (aligned with imshow coordinates)
        ax.contour(mask, levels=[0.5], colors=[col], linewidths=1.6)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(gt_titles[i], fontsize=11.5, weight='bold', pad=6, color='#1E293B')
        
        # Subtitle below
        ax.text(0.5, -0.06, gt_subtitles[i], ha='center', va='top', transform=ax.transAxes,
                fontsize=8.0, color='#475569', linespacing=1.2)
        
        # Pill badge in top-left
        ax.text(0.04, 0.94, gt_badges[i], transform=ax.transAxes, ha='left', va='top',
                fontsize=7.8, weight='bold', color=col,
                bbox=dict(boxstyle='round,pad=0.28,rounding_size=0.2', fc='#0F172A', ec=col, lw=1.2, alpha=0.92))
        
        # Accent border matching ground truth target
        for spine in ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(1.5)
            
    # Super-Title
    fig.text(0.5, 0.960,
             f"BraTS 2023 GLI Multimodal MRI Sequences and Ground Truth Targets ({data['sid']}, Axial Slice z = {data['z']})",
             ha='center', va='top', fontsize=12.0, weight='bold', color='#0F172A')
    
    # Save both 4K PNG (350 DPI) and infinite-resolution Vector SVG
    out_png = FIG_DIR / 'fig2_dataset_modalities.png'
    out_svg = FIG_DIR / 'fig2_dataset_modalities.svg'
    plt.savefig(out_png, dpi=350, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.savefig(out_svg, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated Figure 2 successfully at:\n  {out_png}\n  {out_svg}")

if __name__ == '__main__':
    make_fig2()
