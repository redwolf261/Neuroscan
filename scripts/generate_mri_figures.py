"""
Generate qualitative MRI slice figures for Section 5 (Fig 6 and Fig 11).
Loads real scan data from BraTS 2023 GLI and applies the frozen UNet3D_v5,
TTA, MRD readout, and dual-zone / hierarchical gating.
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

# Plotting parameters
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 9.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

print("Initializing model and dataset for MRI qualitative figure generation...")
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
    
    # Baseline mask
    mask_b0 = p_et_raw >= 0.50
    
    # E293 Hierarchical Core
    core_b2 = (p_et_cal >= 0.15) & (p_wt_cal >= 0.50)
    
    # E293 B3 Integrated (Core + Zone 2 MRD)
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

# ==============================================================================
# Fig. 6 — Central Qualitative Result: Visual Proof of Satellite Recovery
# ==============================================================================
def make_fig6():
    print("Generating Figure 6 (Qualitative Rescues)...")
    cases = ['BraTS-GLI-00299-000', 'BraTS-GLI-00404-000']
    
    fig, axes = plt.subplots(len(cases), 6, figsize=(15, 5.2), facecolor='white')
    col_titles = [
        "T2-FLAIR", "T1-CE (t1c)", "Ground Truth ET", 
        "Baseline (B0, τ=0.50)\n[Misses Satellite]", 
        "Latent Residual R(x)\n[Exposes Evidence]", 
        "Proposed NeuroScan\n[Satellite Rescued]"
    ]
    
    for row, sid in enumerate(cases):
        img_c, tgt_c, mask_b0, r_map, mask_b3 = run_case_inference(sid)
        
        # Find slice with satellite lesion (true ET that B0 missed)
        diff_et = (tgt_c[0] > 0.5) & (~mask_b0)
        lbl_diff, n_diff = ndimage.label(diff_et, structure=STRUCT)
        if n_diff > 0:
            # Pick center slice of the largest missed component
            comp_sizes = [np.sum(lbl_diff == k) for k in range(1, n_diff + 1)]
            best_comp = np.argmax(comp_sizes) + 1
            z_indices = np.where(lbl_diff == best_comp)[0]
            z_slice = int(np.median(z_indices))
        else:
            z_slice = 64
            
        flair_s = img_c[2, z_slice]
        t1c_s = img_c[0, z_slice]
        gt_s = tgt_c[0, z_slice] > 0.5
        b0_s = mask_b0[z_slice]
        r_s = r_map[z_slice]
        b3_s = mask_b3[z_slice]
        
        # Col 0: FLAIR
        axes[row, 0].imshow(flair_s, cmap='gray', origin='lower')
        axes[row, 0].set_ylabel(f"{sid.replace('-000','')}\n(z={z_slice})", fontsize=9.5, weight='bold')
        
        # Col 1: T1c
        axes[row, 1].imshow(t1c_s, cmap='gray', origin='lower')
        
        # Col 2: Ground Truth overlay (Yellow)
        axes[row, 2].imshow(t1c_s, cmap='gray', origin='lower')
        axes[row, 2].imshow(np.ma.masked_where(~gt_s, gt_s), cmap='autumn', alpha=0.85, origin='lower')
        
        # Col 3: Baseline (Blue)
        axes[row, 3].imshow(t1c_s, cmap='gray', origin='lower')
        if b0_s.any():
            axes[row, 3].imshow(np.ma.masked_where(~b0_s, b0_s), cmap='cool', alpha=0.85, origin='lower')
            
        # Col 4: Latent Residual Map R(x) (Hot colormap)
        im_r = axes[row, 4].imshow(r_s, cmap='magma', vmin=0.0, vmax=1.0, origin='lower')
        
        # Col 5: Proposed NeuroScan (Green)
        axes[row, 5].imshow(t1c_s, cmap='gray', origin='lower')
        axes[row, 5].imshow(np.ma.masked_where(~b3_s, b3_s), cmap='summer', alpha=0.85, origin='lower')
        
        for col in range(6):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=9.5, weight='bold', pad=8)
                
    plt.suptitle("Figure 6: Visual Evidence of Peripheral Satellite Lesion Recovery by NeuroScan", fontsize=12.5, weight='bold', y=1.03)
    plt.tight_layout()
    out_path = FIG_DIR / 'fig6_qualitative_rescues.png'
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Fig 6 -> {out_path}")

# ==============================================================================
# Fig. 11 — Controlled Failures & Boundary Selectivity
# ==============================================================================
def make_fig11():
    print("Generating Figure 11 (Controlled Failure Analysis)...")
    cases = ['BraTS-GLI-00186-000', 'BraTS-GLI-00021-000']
    
    fig, axes = plt.subplots(len(cases), 5, figsize=(13.5, 5.2), facecolor='white')
    col_titles = [
        "T1-CE (t1c)", "Ground Truth ET", 
        "Latent Residual R(x)\n[Noise vs Signal]", 
        "Candidate Gating\n[Accepted / Rejected]", 
        "Final Segmentation\n[Controlled Boundary]"
    ]
    
    for row, sid in enumerate(cases):
        img_c, tgt_c, mask_b0, r_map, mask_b3 = run_case_inference(sid)
        
        # Slice selection
        if (tgt_c[0] > 0.5).any():
            z_slice = int(np.median(np.where(tgt_c[0] > 0.5)[0]))
        else:
            z_slice = 64
            
        t1c_s = img_c[0, z_slice]
        gt_s = tgt_c[0, z_slice] > 0.5
        r_s = r_map[z_slice]
        b3_s = mask_b3[z_slice]
        cand_s = (r_s >= 0.9508)
        
        axes[row, 0].imshow(t1c_s, cmap='gray', origin='lower')
        axes[row, 0].set_ylabel(f"{sid.replace('-000','')}\n(z={z_slice})", fontsize=9.5, weight='bold')
        
        axes[row, 1].imshow(t1c_s, cmap='gray', origin='lower')
        if gt_s.any():
            axes[row, 1].imshow(np.ma.masked_where(~gt_s, gt_s), cmap='autumn', alpha=0.85, origin='lower')
            
        axes[row, 2].imshow(r_s, cmap='magma', vmin=0.0, vmax=1.0, origin='lower')
        
        # Gating visualization (Candidate candidates vs filtered)
        axes[row, 3].imshow(t1c_s, cmap='gray', origin='lower')
        if cand_s.any():
            axes[row, 3].imshow(np.ma.masked_where(~cand_s, cand_s), cmap='autumn', alpha=0.7, origin='lower')
            
        axes[row, 4].imshow(t1c_s, cmap='gray', origin='lower')
        if b3_s.any():
            axes[row, 4].imshow(np.ma.masked_where(~b3_s, b3_s), cmap='summer', alpha=0.85, origin='lower')
            
        for col in range(5):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=9.5, weight='bold', pad=8)
                
    plt.suptitle("Figure 11: Controlled Failure Analysis & Boundary Gating Selectivity", fontsize=12.5, weight='bold', y=1.03)
    plt.tight_layout()
    out_path = FIG_DIR / 'fig11_controlled_failures.png'
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Fig 11 -> {out_path}")

if __name__ == '__main__':
    make_fig6()
    make_fig11()
    print("Qualitative MRI figures generated successfully!")
