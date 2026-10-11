"""
Generate publication-grade Figure 6 for IEEE TMI / MedIA:
Visual Evidence of Peripheral Satellite Lesion Recovery by NeuroScan with High-Resolution Zoom Insets.
"""

import os, sys
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'novelty' / 'pairwise_evidence_geometry'))

from e257_common import load_model, ROOT as REPO_ROOT, PATCH, MIN_VOX
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch
from e291_dual_zone_precision_integration import compute_subject_maps_with_tta, extract_27_features, FEATS_A2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 9.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

print("Initializing model and dataset for Fig. 6...")
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(dev)

conv_w = model.seg_head[0].weight.detach().cpu().numpy()
conv_b = model.seg_head[0].bias.detach().cpu().numpy()
w_p, b_p = conv_w[0].reshape(32).astype(np.float64), float(conv_b[0])
w_tc, b_tc = conv_w[1].reshape(32).astype(np.float64), float(conv_b[1])
w_wt, b_wt = conv_w[2].reshape(32).astype(np.float64), float(conv_b[2])

cached_mrd = torch.load(ROOT / 'novelty/pairwise_evidence_geometry/E290_mrd_cache.pt', map_location='cpu', weights_only=False)
fitted_mrd = {s: cached_mrd[s] for s in [999, 4242, 7, 123, 2024]}

train_records = torch.load(ROOT / 'novelty/pairwise_evidence_geometry/E290_train_records.pt', map_location='cpu', weights_only=False)
X_train = np.stack([r['feat'][FEATS_A2] for r in train_records])
y_train = np.array([r['label'] for r in train_records], dtype=int)
acceptance_model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=290))
])
acceptance_model.fit(X_train, y_train)

ds_full = BraTSMultimodalDataset(str(REPO_ROOT / 'Dataset' / 'Training'), 'train', val_split=0.0, patch_size=PATCH)
sids_all = [os.path.basename(d) for d in ds_full.subject_dirs]
sid_to_idx = {s: i for i, s in enumerate(sids_all)}
STRUCT = ndimage.generate_binary_structure(3, 1)

def run_case_inference(sid):
    img_c, tgt_c = load_patch(ds_full, sid_to_idx, sid)
    raw_maps, calib_maps, _ = compute_subject_maps_with_tta(
        model, ds_full, sid_to_idx, sid, w_p, b_p, w_tc, b_tc, w_wt, b_wt, fitted_mrd, dev
    )
    p_et_raw, p_tc_raw, p_wt_raw, r_maps_raw = raw_maps
    p_et_cal, p_tc_cal, p_wt_cal, r_maps_cal = calib_maps
    
    mask_b0 = p_et_raw >= 0.50
    core_b2 = (p_et_cal >= 0.15) & (p_wt_cal >= 0.50)
    
    edt_et = ndimage.distance_transform_edt(~core_b2) if core_b2.any() else None
    edt_wt = ndimage.distance_transform_edt(~(p_wt_cal >= 0.50)) if (p_wt_cal >= 0.50).any() else None
    
    seed_preds = []
    r_mean = np.mean(list(r_maps_cal.values()), axis=0)
    for s in fitted_mrd.keys():
        r_seed = r_maps_cal[s]
        mrd_pos = (r_seed > 0.95082877089122) & (~core_b2)
        cc_l, cc_n = ndimage.label(mrd_pos, structure=STRUCT)
        ad_s = np.zeros_like(core_b2)
        for cid in range(1, cc_n + 1):
            cm = cc_l == cid
            feat = extract_27_features(cm, p_et_cal, p_tc_cal, p_wt_cal, r_seed, edt_et, edt_wt)
            prob = acceptance_model.predict_proba(feat[FEATS_A2].reshape(1, -1))[0, 1]
            if prob >= 0.80 and feat[21] > 2.0:
                ad_s |= cm
        seed_preds.append(core_b2 | ad_s)
    mask_b3 = (np.stack(seed_preds).sum(axis=0) >= (len(fitted_mrd) / 2.0))
    return img_c, tgt_c, mask_b0, r_mean, mask_b3

def generate_fig6():
    cases_meta = [
        {
            'sid': 'BraTS-GLI-00299-000',
            'z': 28,
            'crop_y': (82, 122),
            'crop_x': (46, 86),
            'label': 'Subject A (GLI-00299, z=28)'
        },
        {
            'sid': 'BraTS-GLI-00404-000',
            'z': 85,
            'crop_y': (8, 52),
            'crop_x': (64, 108),
            'label': 'Subject B (GLI-00404, z=85)'
        }
    ]
    
    fig, axes = plt.subplots(2, 6, figsize=(15.5, 5.4), facecolor='white')
    
    col_titles = [
        "T2-FLAIR\n(Full Field)",
        "T1-CE with ROI Box\n(Full Field)",
        "Ground Truth ET\n(Magnified ROI)",
        "Baseline $B_0$ ($\\tau=0.50$)\n[Missed Satellite]",
        "Residual Evidence $R(x)$\n[Complementary Signal]",
        "Proposed NeuroScan\n[Satellite Rescued]"
    ]
    
    for row, meta in enumerate(cases_meta):
        sid = meta['sid']
        z = meta['z']
        cy0, cy1 = meta['crop_y']
        cx0, cx1 = meta['crop_x']
        
        img_c, tgt_c, mask_b0, r_map, mask_b3 = run_case_inference(sid)
        
        flair_full = img_c[2, z]
        t1c_full = img_c[0, z]
        gt_full = tgt_c[0, z] > 0.5
        b0_full = mask_b0[z]
        r_full = r_map[z]
        b3_full = mask_b3[z]
        
        # Crops
        t1c_crop = t1c_full[cy0:cy1, cx0:cx1]
        gt_crop = gt_full[cy0:cy1, cx0:cx1]
        b0_crop = b0_full[cy0:cy1, cx0:cx1]
        r_crop = r_full[cy0:cy1, cx0:cx1]
        b3_crop = b3_full[cy0:cy1, cx0:cx1]
        
        # Col 0: FLAIR full
        axes[row, 0].imshow(flair_full, cmap='gray', origin='lower')
        axes[row, 0].set_ylabel(meta['label'], fontsize=9.2, weight='bold')
        
        # Col 1: T1-CE full with ROI box
        axes[row, 1].imshow(t1c_full, cmap='gray', origin='lower')
        rect = patches.Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0,
                                 linewidth=1.8, edgecolor='#F59E0B', facecolor='none', linestyle='--')
        axes[row, 1].add_patch(rect)
        
        # Col 2: Magnified GT
        axes[row, 2].imshow(t1c_crop, cmap='gray', origin='lower')
        if gt_crop.any():
            axes[row, 2].imshow(np.ma.masked_where(~gt_crop, gt_crop), cmap='autumn', alpha=0.85, origin='lower')
            axes[row, 2].contour(gt_crop, colors=['#EF4444'], linewidths=1.2, origin='lower')
            
        # Col 3: Magnified Baseline B0
        axes[row, 3].imshow(t1c_crop, cmap='gray', origin='lower')
        if b0_crop.any():
            axes[row, 3].imshow(np.ma.masked_where(~b0_crop, b0_crop), cmap='cool', alpha=0.85, origin='lower')
            axes[row, 3].contour(b0_crop, colors=['#06B6D4'], linewidths=1.2, origin='lower')
        # Draw GT contour in faint white to show what was missed
        if gt_crop.any():
            axes[row, 3].contour(gt_crop, colors=['#F59E0B'], linewidths=0.9, linestyles=':', origin='lower')
            
        # Col 4: Magnified Latent Residual R(x)
        axes[row, 4].imshow(r_crop, cmap='magma', vmin=0.0, vmax=1.0, origin='lower')
        if gt_crop.any():
            axes[row, 4].contour(gt_crop, colors=['#FFFFFF'], linewidths=0.9, linestyles=':', origin='lower')
            
        # Col 5: Magnified Proposed NeuroScan B3
        axes[row, 5].imshow(t1c_crop, cmap='gray', origin='lower')
        if b3_crop.any():
            axes[row, 5].imshow(np.ma.masked_where(~b3_crop, b3_crop), cmap='summer', alpha=0.85, origin='lower')
            axes[row, 5].contour(b3_crop, colors=['#10B981'], linewidths=1.2, origin='lower')
        if gt_crop.any():
            axes[row, 5].contour(gt_crop, colors=['#EF4444'], linewidths=0.9, linestyles=':', origin='lower')
            
        for col in range(6):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=9.2, weight='bold', pad=8)
                
    plt.tight_layout()
    # Save both 4K PNG (350 DPI) and infinite-resolution Vector SVG
    out_png = FIG_DIR / 'fig6_qualitative_rescues.png'
    out_svg = FIG_DIR / 'fig6_qualitative_rescues.svg'
    plt.savefig(out_png, dpi=350, bbox_inches='tight')
    plt.savefig(out_svg, bbox_inches='tight')
    plt.close()
    print(f"Generated 4K PNG ({out_png}) and Vector SVG ({out_svg})")

if __name__ == '__main__':
    generate_fig6()
