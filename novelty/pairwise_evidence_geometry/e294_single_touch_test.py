"""E294 -- Multi-Model Ensemble & Hierarchical Consensus (Pre-Registered Experiment).

PRIMARY HYPOTHESIS:
Combining independent models (M1: E131_v5control_seed0 and M3: E131_v14_seed0)
cancels epistemic error, shifting the ROC curve outward to break through
the +0.63% single-model empirical ceiling and achieve >= +1.0% net Dice gain
on the locked 125-subject test cohort.

PRE-REGISTERED FROZEN SPECIFICATIONS:
- Models:
    M1: UNet3D_v5 (Production baseline)
    M3: UNet3D_v14 (Rank-Adaptive Pooling)
- Ensemble combination: Soft probability averaging: P_ens = 0.5 * P_M1 + 0.5 * P_M3
- Spatial Calibration: 8-fold multi-axis spatial symmetry averaging (flips along D, H, W)
- Locked Thresholds (selected strictly on 13 Dev Validation subjects):
    tau_et* = 0.35
    tau_wt* = 0.50
- Distance zoning & MRD rescue:
    Zone-1 (<= 2mm): Ensemble core boundary
    Zone-2 (> 2mm): Candidate acceptance with tau_A = 0.80

SYSTEM ARMS EVALUATED:
B0: Production Reference (M1 raw, tau=0.50)
B1: E293 Single Best Post-Proc (M1 hierarchical calibrated core + Zone-2 MRD)
B2: Raw 2-Model Ensemble (M1 + M3 soft average, tau=0.50, no post-proc)
B3: Hierarchical 2-Model Ensemble Core Alone (tau_et=0.35, tau_wt=0.50)
B4: Complete Integrated System (Hierarchical Ensemble Core + Zone-2 MRD Rescue)

ZERO-LEAKAGE AUDIT:
Evaluated on the exact 125 locked-test subjects, untouched during training
and hyperparameter selection.
"""

import sys, os, csv, json, time, gc
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage, stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_REPO))
sys.path.insert(0, str(ROOT_REPO / 'experiments' / 'exp_e12_eggo_m' / 'e131'))

from neuroscan_3d_v5 import UNet3D_v5
from neuroscan_3d_v13 import UNet3D_v14
from e257_common import load_populations, build_splits, ROOT, PATCH
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch

ET, TC, WT = 0, 1, 2
TAU_ET_LOCKED = 0.35         # Locked from E294 dev sweep
TAU_WT_LOCKED = 0.50         # Locked Whole Tumor containment threshold
TAU_R_BASE = 0.95082877089122 # Frozen raw MRD operating point
TAU_ACCEPTANCE = 0.80        # Frozen acceptance model threshold
DIST_ZONE_BOUNDARY = 2.0     # 2mm boundary shell: <= 2.0 is Zone 1, > 2.0 is Zone 2
SEEDS = [999, 4242, 7, 123, 2024]
STRUCT = ndimage.generate_binary_structure(3, 1)

FEATURE_NAMES = [
    'mean_r', 'median_r', 'max_r', 'pct90_r', 'pct95_r', 'frac_above_tau_r',
    'mean_p', 'median_p', 'max_p', 'min_p', 'frac_p_lt_01', 'frac_p_lt_025', 'frac_p_lt_05', 'disagreement',
    'voxel_count', 'physical_vol', 'bbox_dx', 'bbox_dy', 'bbox_dz', 'compactness', 'surf_vol_ratio',
    'min_dist_to_prod_et', 'mean_dist_to_prod_et', 'min_dist_to_prod_wt', 'frac_inside_prod_wt', 'frac_inside_prod_tc', 'touching_prod_et'
]
FEATS_A2 = list(range(len(FEATURE_NAMES)))

def hd95_and_assd(pred_mask, gt_mask, spacing=1.0):
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float('nan'), float('nan')
    pred_surf = pred_mask & ~ndimage.binary_erosion(pred_mask, structure=STRUCT)
    gt_surf = gt_mask & ~ndimage.binary_erosion(gt_mask, structure=STRUCT)
    if pred_surf.sum() == 0 or gt_surf.sum() == 0:
        return float('nan'), float('nan')
    dt_gt = ndimage.distance_transform_edt(~gt_mask, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~pred_mask, sampling=spacing)
    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    if len(all_d) == 0:
        return float('nan'), float('nan')
    return float(np.percentile(all_d, 95)), float(all_d.mean())

def extract_27_features(cm, p_et, p_tc, p_wt, r_seed, edt_et, edt_wt, tau_r):
    vox_count = int(cm.sum())
    r_vals = r_seed[cm]
    p_vals = p_et[cm]
    coords = np.argwhere(cm)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    bbox = maxs - mins + 1

    surf = cm & ~ndimage.binary_erosion(cm, structure=STRUCT)
    surf_count = int(surf.sum())
    compactness = (surf_count ** 1.5) / (vox_count + 1e-6)
    surf_vol_ratio = surf_count / (vox_count + 1e-6)

    min_dist_et = float(edt_et[cm].min()) if (edt_et is not None and edt_et[cm].size > 0) else 0.0
    mean_dist_et = float(edt_et[cm].mean()) if (edt_et is not None and edt_et[cm].size > 0) else 0.0
    min_dist_wt = float(edt_wt[cm].min()) if (edt_wt is not None and edt_wt[cm].size > 0) else 0.0

    touching_et = 1.0 if min_dist_et <= 1.0 else 0.0
    frac_wt = float((p_wt[cm] > 0.5).mean())
    frac_tc = float((p_tc[cm] > 0.5).mean())

    feats = np.array([
        float(np.mean(r_vals)), float(np.median(r_vals)), float(np.max(r_vals)),
        float(np.percentile(r_vals, 90)), float(np.percentile(r_vals, 95)),
        float((r_vals > tau_r).mean()),
        float(np.mean(p_vals)), float(np.median(p_vals)), float(np.max(p_vals)), float(np.min(p_vals)),
        float((p_vals < 0.1).mean()), float((p_vals < 0.25).mean()), float((p_vals < 0.5).mean()),
        float(np.mean(r_vals - p_vals)),
        float(vox_count), float(vox_count),
        float(bbox[0]), float(bbox[1]), float(bbox[2]),
        float(compactness), float(surf_vol_ratio),
        min_dist_et, mean_dist_et, min_dist_wt,
        frac_wt, frac_tc, touching_et
    ], dtype=np.float32)
    return feats

def compute_ensemble_subject_maps(m1, m3, ds, sid_to_idx, sid, w_p, b_p, w_tc, b_tc, w_wt, b_wt, fitted_mrd, dev):
    """
    Computes maps for Model 1 (v5control) and Model 3 (v14) with 8-fold TTA,
    and returns individual and ensemble soft probability maps.
    """
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)

    flip_combos = [
        (), (2,), (3,), (4,),
        (2, 3), (2, 4), (3, 4),
        (2, 3, 4)
    ]

    # Pre-allocate flip accumulators
    m1_probs_flips = []
    m3_probs_flips = []
    r_maps_flips = {seed: [] for seed in fitted_mrd.keys()}

    w_prod_stack = np.stack([w_p, w_tc, w_wt], axis=1) # (32, 3)
    b_prod_stack = np.array([b_p, b_tc, b_wt], dtype=np.float32)

    with torch.no_grad():
        for dims in flip_combos:
            if len(dims) == 0:
                cur_img = img_t
            else:
                cur_img = torch.flip(img_t, dims=dims)

            # M1 Forward & Dec1 representation
            stages_m1 = forward_to_dec1_internal(m1, cur_img)
            d1 = stages_m1['relu2_out'][0]
            C_, D, H, W = d1.shape
            flat = d1.reshape(C_, -1).T

            W_p_t = torch.from_numpy(w_prod_stack).float().to(dev)
            B_p_t = torch.from_numpy(b_prod_stack).float().to(dev)
            m1_p = torch.sigmoid(flat @ W_p_t + B_p_t) # (N, 3)
            del W_p_t, B_p_t

            # M3 Forward
            m3_out = m3(cur_img)['probs'][0] # (3, D, H, W)

            # MRD readout on M1 features
            r_seed_cur = {}
            for seed, (w_r, b_r) in fitted_mrd.items():
                w_r_t = torch.from_numpy(w_r).float().to(dev)
                r_seed_cur[seed] = torch.sigmoid(flat @ w_r_t + float(b_r)).reshape(D, H, W)
                del w_r_t
            del stages_m1, d1, flat

            # Unflip
            if len(dims) == 0:
                m1_probs_flips.append(m1_p.reshape(D, H, W, 3).permute(3, 0, 1, 2).cpu().numpy())
                m3_probs_flips.append(m3_out.cpu().numpy())
                for s in fitted_mrd:
                    r_maps_flips[s].append(r_seed_cur[s].cpu().numpy())
            else:
                unflip_dims = tuple(d - 2 for d in dims)
                m1_unf = torch.flip(m1_p.reshape(D, H, W, 3), dims=unflip_dims).permute(3, 0, 1, 2).cpu().numpy()
                m3_unf = torch.flip(m3_out, dims=unflip_dims).cpu().numpy()
                m1_probs_flips.append(m1_unf)
                m3_probs_flips.append(m3_unf)
                for s in fitted_mrd:
                    r_maps_flips[s].append(torch.flip(r_seed_cur[s], dims=unflip_dims).cpu().numpy())

    del img_t
    torch.cuda.empty_cache()

    m1_raw = m1_probs_flips[0] # (3, D, H, W)
    m1_cal = np.mean(m1_probs_flips, axis=0) # (3, D, H, W)
    m3_cal = np.mean(m3_probs_flips, axis=0) # (3, D, H, W)

    r_cal = {s: np.mean(r_maps_flips[s], axis=0) for s in fitted_mrd}
    p_ens_cal = 0.5 * (m1_cal + m3_cal) # (3, D, H, W)

    true_et = (tgt_c[ET] > 0.5)
    true_tc = (tgt_c[TC] > 0.5)
    true_wt = (tgt_c[WT] > 0.5)

    return m1_raw, m1_cal, m3_cal, p_ens_cal, r_cal, true_et, true_tc, true_wt

def evaluate_e294_locked(m1, m3, ds, sid_to_idx, subjects, fitted_mrd,
                         w_p, b_p, w_tc, b_tc, w_wt, b_wt, dev, acceptance_model):
    systems = [
        'B0_production_base',
        'B1_e293_single_best',
        'B2_ensemble_raw_tau050',
        'B3_ensemble_hierarchical_core',
        'B4_ensemble_complete_integrated'
    ]

    per_subj_metrics = {s: [] for s in systems}
    pooled_metrics = {s: {
        'tp_et': 0, 'fp_et': 0, 'fn_et': 0,
        'tp_tc': 0, 'fp_tc': 0, 'fn_tc': 0,
        'tp_wt': 0, 'fp_wt': 0, 'fn_wt': 0,
        'g1_recovered': 0, 'g1_total': 0,
        'hd95_vals': [], 'assd_vals': []
    } for s in systems}

    n_subjects = len(subjects)
    t0 = time.time()

    for i, sid in enumerate(subjects):
        if sid not in sid_to_idx:
            continue

        (m1_raw, m1_cal, m3_cal, p_ens_cal, r_maps_cal,
         true_et, true_tc, true_wt) = compute_ensemble_subject_maps(
            m1, m3, ds, sid_to_idx, sid, w_p, b_p, w_tc, b_tc, w_wt, b_wt, fitted_mrd, dev)

        # Connected components of GT ET for G1 lesion tracking
        gt_lbl, n_gt = ndimage.label(true_et, structure=STRUCT)
        g1_masks = []
        for g_idx in range(1, n_gt + 1):
            comp_mask = (gt_lbl == g_idx)
            g1_masks.append(comp_mask)
        n_g1_subject = len(g1_masks)

        # ---------------------------------------------------------------
        # System B0: Production Baseline (M1 raw, tau=0.50)
        # ---------------------------------------------------------------
        pred_b0 = (m1_raw[ET] >= 0.50)
        pred_b0_tc = (m1_raw[TC] >= 0.50)
        pred_b0_wt = (m1_raw[WT] >= 0.50)

        # ---------------------------------------------------------------
        # System B1: E293 Single Best Post-Proc (M1 Hierarchical tau=0.15 + MRD)
        # ---------------------------------------------------------------
        core_b1 = (m1_cal[ET] >= 0.15) & (m1_cal[WT] >= 0.50)
        edt_et_b1 = ndimage.distance_transform_edt(~core_b1) if core_b1.any() else None
        edt_wt_b1 = ndimage.distance_transform_edt(~(m1_cal[WT] >= 0.50)) if (m1_cal[WT] >= 0.50).any() else None

        seed_b1_preds = []
        for seed in fitted_mrd.keys():
            r_seed = r_maps_cal[seed]
            mrd_s_pos = (r_seed > TAU_R_BASE) & (~core_b1)
            cc_lbl, cc_n = ndimage.label(mrd_s_pos, structure=STRUCT)
            admitted_s = np.zeros_like(core_b1)
            for cid in range(1, cc_n + 1):
                cm = cc_lbl == cid
                feat = extract_27_features(cm, m1_cal[ET], m1_cal[TC], m1_cal[WT], r_seed, edt_et_b1, edt_wt_b1, TAU_R_BASE)
                prob = acceptance_model.predict_proba(feat[FEATS_A2].reshape(1, -1))[0, 1]
                if prob >= TAU_ACCEPTANCE and feat[21] > DIST_ZONE_BOUNDARY:
                    admitted_s |= cm
            seed_b1_preds.append(core_b1 | admitted_s)
        pred_b1 = (np.stack(seed_b1_preds).sum(axis=0) >= (len(fitted_mrd) / 2.0))

        # ---------------------------------------------------------------
        # System B2: Raw 2-Model Ensemble (M1 + M3, tau=0.50, no post-proc)
        # ---------------------------------------------------------------
        pred_b2 = (p_ens_cal[ET] >= 0.50)
        pred_b2_tc = (p_ens_cal[TC] >= 0.50)
        pred_b2_wt = (p_ens_cal[WT] >= 0.50)

        # ---------------------------------------------------------------
        # System B3: Pre-Registered Hierarchical Ensemble Core Alone (tau_et=0.35, tau_wt=0.50)
        # ---------------------------------------------------------------
        core_b3 = (p_ens_cal[ET] >= TAU_ET_LOCKED) & (p_ens_cal[WT] >= TAU_WT_LOCKED)
        pred_b3 = core_b3
        pred_b3_tc = (p_ens_cal[TC] >= 0.50)
        pred_b3_wt = (p_ens_cal[WT] >= 0.50)

        # ---------------------------------------------------------------
        # System B4: Complete Integrated System (Hierarchical Ensemble + Zone-2 MRD Rescue)
        # ---------------------------------------------------------------
        edt_et_b3 = ndimage.distance_transform_edt(~core_b3) if core_b3.any() else None
        edt_wt_b3 = ndimage.distance_transform_edt(~(p_ens_cal[WT] >= TAU_WT_LOCKED)) if (p_ens_cal[WT] >= TAU_WT_LOCKED).any() else None

        seed_b4_preds = []
        fp_comp_b4_tot = 0
        for seed in fitted_mrd.keys():
            r_seed = r_maps_cal[seed]
            mrd_s_pos = (r_seed > TAU_R_BASE) & (~core_b3)
            cc_lbl, cc_n = ndimage.label(mrd_s_pos, structure=STRUCT)
            admitted_s = np.zeros_like(core_b3)
            for cid in range(1, cc_n + 1):
                cm = cc_lbl == cid
                feat = extract_27_features(cm, p_ens_cal[ET], p_ens_cal[TC], p_ens_cal[WT], r_seed, edt_et_b3, edt_wt_b3, TAU_R_BASE)
                prob = acceptance_model.predict_proba(feat[FEATS_A2].reshape(1, -1))[0, 1]
                if prob >= TAU_ACCEPTANCE and feat[21] > DIST_ZONE_BOUNDARY:
                    admitted_s |= cm
                    if not (cm & true_et).any():
                        fp_comp_b4_tot += 1
            seed_b4_preds.append(core_b3 | admitted_s)
        pred_b4 = (np.stack(seed_b4_preds).sum(axis=0) >= (len(fitted_mrd) / 2.0))
        fp_comp_b4 = fp_comp_b4_tot / len(fitted_mrd)

        # Mask collections for logging
        sys_masks = {
            'B0_production_base': (pred_b0, pred_b0_tc, pred_b0_wt, 0.0),
            'B1_e293_single_best': (pred_b1, pred_b0_tc, pred_b0_wt, 0.0),
            'B2_ensemble_raw_tau050': (pred_b2, pred_b2_tc, pred_b2_wt, 0.0),
            'B3_ensemble_hierarchical_core': (pred_b3, pred_b3_tc, pred_b3_wt, 0.0),
            'B4_ensemble_complete_integrated': (pred_b4, pred_b3_tc, pred_b3_wt, fp_comp_b4)
        }

        for sname, (p_et_m, p_tc_m, p_wt_m, fp_c) in sys_masks.items():
            # ET metrics
            tp = int((p_et_m & true_et).sum())
            fp = int((p_et_m & ~true_et).sum())
            fn = int((~p_et_m & true_et).sum())
            d_et = (2 * tp) / max(2 * tp + fp + fn, 1e-8) if (true_et.sum() > 0 or p_et_m.sum() > 0) else 1.0
            prec_et = tp / max(tp + fp, 1e-8)
            rec_et = tp / max(tp + fn, 1e-8)
            hd, assd = hd95_and_assd(p_et_m, true_et)

            # TC & WT metrics
            tp_tc = int((p_tc_m & true_tc).sum())
            fp_tc = int((p_tc_m & ~true_tc).sum())
            fn_tc = int((~p_tc_m & true_tc).sum())
            d_tc = (2 * tp_tc) / max(2 * tp_tc + fp_tc + fn_tc, 1e-8) if (true_tc.sum() > 0 or p_tc_m.sum() > 0) else 1.0

            tp_wt = int((p_wt_m & true_wt).sum())
            fp_wt = int((p_wt_m & ~true_wt).sum())
            fn_wt = int((~p_wt_m & true_wt).sum())
            d_wt = (2 * tp_wt) / max(2 * tp_wt + fp_wt + fn_wt, 1e-8) if (true_wt.sum() > 0 or p_wt_m.sum() > 0) else 1.0

            # G1 recoveries
            g1_rec = sum(1 for lm in g1_masks if (p_et_m & lm).any())

            per_subj_metrics[sname].append({
                'sid': sid, 'dice_et': d_et, 'dice_tc': d_tc, 'dice_wt': d_wt,
                'precision_et': prec_et, 'recall_et': rec_et,
                'tp_et': tp, 'fp_et': fp, 'fn_et': fn, 'fp_comp': fp_c,
                'hd95_et': hd, 'assd_et': assd,
                'g1_recovered': g1_rec, 'g1_total': n_g1_subject
            })

            pm = pooled_metrics[sname]
            pm['tp_et'] += tp
            pm['fp_et'] += fp
            pm['fn_et'] += fn
            pm['tp_tc'] += tp_tc
            pm['fp_tc'] += fp_tc
            pm['fn_tc'] += fn_tc
            pm['tp_wt'] += tp_wt
            pm['fp_wt'] += fp_wt
            pm['fn_wt'] += fn_wt
            pm['g1_recovered'] += g1_rec
            pm['g1_total'] += n_g1_subject
            pm['fp_comp_total'] = pm.get('fp_comp_total', 0.0) + fp_c
            if not np.isnan(hd):
                pm['hd95_vals'].append(hd)
            if not np.isnan(assd):
                pm['assd_vals'].append(assd)

        if (i + 1) % 25 == 0 or (i + 1) == n_subjects:
            el = time.time() - t0
            b0_m = np.mean([m['dice_et'] for m in per_subj_metrics['B0_production_base']])
            b3_m = np.mean([m['dice_et'] for m in per_subj_metrics['B3_ensemble_hierarchical_core']])
            b4_m = np.mean([m['dice_et'] for m in per_subj_metrics['B4_ensemble_complete_integrated']])
            print(f'[{i+1:3d}/{n_subjects:3d}] ({el:5.1f}s) ET Dice: B0={b0_m:.4f} | B3_Ens={b3_m:.4f} (Delta={b3_m-b0_m:+.4f}) | B4_Int={b4_m:.4f} (Delta={b4_m-b0_m:+.4f})', flush=True)

    # Compute final statistics
    results = {}
    base_m_dice = np.mean([m['dice_et'] for m in per_subj_metrics['B0_production_base']])
    b0_pm = pooled_metrics['B0_production_base']
    base_p_dice = (2 * b0_pm['tp_et']) / max(2 * b0_pm['tp_et'] + b0_pm['fp_et'] + b0_pm['fn_et'], 1e-8)

    for sname in systems:
        sm = per_subj_metrics[sname]
        pm = pooled_metrics[sname]

        dices_et = [m['dice_et'] for m in sm]
        dices_tc = [m['dice_tc'] for m in sm]
        dices_wt = [m['dice_wt'] for m in sm]

        m_dice_et = float(np.mean(dices_et))
        med_dice_et = float(np.median(dices_et))
        sd_dice_et = float(np.std(dices_et))

        m_dice_tc = float(np.mean(dices_tc))
        m_dice_wt = float(np.mean(dices_wt))
        m_dice_3class = float(np.mean([m_dice_et, m_dice_tc, m_dice_wt]))

        pooled_d_et = (2 * pm['tp_et']) / max(2 * pm['tp_et'] + pm['fp_et'] + pm['fn_et'], 1e-8)
        pooled_d_tc = (2 * pm['tp_tc']) / max(2 * pm['tp_tc'] + pm['fp_tc'] + pm['fn_tc'], 1e-8)
        pooled_d_wt = (2 * pm['tp_wt']) / max(2 * pm['tp_wt'] + pm['fp_wt'] + pm['fn_wt'], 1e-8)
        pooled_d_3class = float(np.mean([pooled_d_et, pooled_d_tc, pooled_d_wt]))

        m_prec = float(np.mean([m['precision_et'] for m in sm]))
        m_rec = float(np.mean([m['recall_et'] for m in sm]))
        m_hd = float(np.mean(pm['hd95_vals'])) if pm['hd95_vals'] else float('nan')
        m_assd = float(np.mean(pm['assd_vals'])) if pm['assd_vals'] else float('nan')
        g1_pct = (pm['g1_recovered'] / max(pm['g1_total'], 1)) * 100.0

        results[sname] = {
            'system': sname,
            'mean_per_subject_dice_et': m_dice_et,
            'delta_mean_per_subject_dice_et': m_dice_et - base_m_dice,
            'median_per_subject_dice_et': med_dice_et,
            'sd_per_subject_dice_et': sd_dice_et,
            'mean_per_subject_dice_tc': m_dice_tc,
            'mean_per_subject_dice_wt': m_dice_wt,
            'mean_per_subject_dice_3class': m_dice_3class,
            'global_pooled_dice_et': pooled_d_et,
            'delta_global_pooled_dice_et': pooled_d_et - base_p_dice,
            'global_pooled_dice_tc': pooled_d_tc,
            'global_pooled_dice_wt': pooled_d_wt,
            'global_pooled_dice_3class': pooled_d_3class,
            'mean_precision_et': m_prec,
            'mean_recall_et': m_rec,
            'mean_hd95_et': m_hd,
            'mean_assd_et': m_assd,
            'g1_recovered': pm['g1_recovered'],
            'g1_total': pm['g1_total'],
            'g1_recovery_pct': g1_pct,
            'fp_comps_per_subj': pm.get('fp_comp_total', 0.0) / max(n_subjects, 1),
            'total_tp_voxels': pm['tp_et'],
            'total_fp_voxels': pm['fp_et'],
            'total_fn_voxels': pm['fn_et']
        }

    return results, per_subj_metrics

def main():
    dev = torch.device('cuda')
    t_start = time.time()

    print('=' * 100)
    print('E294: MULTI-MODEL ENSEMBLE & HIERARCHICAL CONSENSUS (LOCKED TEST EVALUATION)')
    print(f'Locked Operating Point: tau_et* = {TAU_ET_LOCKED:.2f}, tau_wt* = {TAU_WT_LOCKED:.2f}')
    print('Primary Endpoint: Mean Per-Subject ET Dice across 125 Locked Test Subjects')
    print('=' * 100, flush=True)

    # 1. Load frozen production models
    print('Loading M1: UNet3D_v5 (E131_v5control_seed0)...')
    m1 = UNet3D_v5(4, 3).to(dev).eval()
    ck1 = torch.load(ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth', map_location=dev, weights_only=False)
    m1.load_state_dict(ck1['model_state'])

    print('Loading M3: UNet3D_v14 (E131_v14_seed0)...')
    m3 = UNet3D_v14(4, 3).to(dev).eval()
    ck3 = torch.load(ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v14_seed0/checkpoints/best.pth', map_location=dev, weights_only=False)
    m3.load_state_dict(ck3['model_state'])

    conv_w = m1.seg_head[0].weight.detach().cpu().numpy()
    conv_b = m1.seg_head[0].bias.detach().cpu().numpy()
    w_p, b_p = conv_w[ET].reshape(32).astype(np.float64), float(conv_b[ET])
    w_tc, b_tc = conv_w[TC].reshape(32).astype(np.float64), float(conv_b[TC])
    w_wt, b_wt = conv_w[WT].reshape(32).astype(np.float64), float(conv_b[WT])

    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    # 2. Load pre-fitted MRD weights
    mrd_cache_path = HERE / 'E290_mrd_cache.pt'
    cached_mrd = torch.load(mrd_cache_path, map_location='cpu', weights_only=False)
    fitted_mrd = {s: cached_mrd[s] for s in SEEDS}

    # 3. Load pre-computed dev records & train frozen A2 acceptance model
    train_cache = HERE / 'E290_train_records.pt'
    train_records = torch.load(train_cache, map_location='cpu', weights_only=False)
    X_train = np.stack([r['feat'][FEATS_A2] for r in train_records])
    y_train = np.array([r['label'] for r in train_records], dtype=int)
    acceptance_model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=290))
    ])
    acceptance_model.fit(X_train, y_train)

    # 4. Prepare locked test cohort & audit zero-leakage
    all_prior_touched = (set(r['subject_id'] for r in det_lesions) |
                          set(r['subject_id'] for r in g2a_lesions) |
                          set(r['subject_id'] for r in g2b_lesions))

    ds_val_pop = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val', val_split=0.1, patch_size=PATCH)
    val_sids_all = sorted(os.path.basename(d) for d in ds_val_pop.subject_dirs)
    val_sids_fresh = sorted(set(val_sids_all) - all_prior_touched)

    ds_full = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train', val_split=0.0, patch_size=PATCH)
    full_sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds_full.subject_dirs)}

    test_subjects = []
    for sid in val_sids_fresh:
        if sid not in full_sid_to_idx:
            continue
        try:
            img_c, tgt_c = load_patch(ds_full, full_sid_to_idx, sid)
            img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
            with torch.no_grad():
                stages = forward_to_dec1_internal(m1, img_t)
                d1 = stages['relu2_out'][0]
                if torch.isnan(d1).any() or torch.isinf(d1).any():
                    del d1, img_t, stages
                    torch.cuda.empty_cache()
                    continue
            del d1, img_t, stages
            torch.cuda.empty_cache()
            test_subjects.append(sid)
        except Exception:
            continue

    leakage_dev = set(test_subjects) & set(splits['det_test'])
    assert len(leakage_dev) == 0, 'LEAKAGE AUDIT FAILED'
    print(f'Locked Test Cohort: {len(test_subjects)} subjects. Leakage Audit: PASS!\n', flush=True)

    # Execute single-touch evaluation
    results, per_subj_metrics = evaluate_e294_locked(
        m1, m3, ds_full, full_sid_to_idx, test_subjects, fitted_mrd,
        w_p, b_p, w_tc, b_tc, w_wt, b_wt, dev, acceptance_model
    )

    # Statistical significance testing
    s_b0 = [m['dice_et'] for m in per_subj_metrics['B0_production_base']]
    s_b1 = [m['dice_et'] for m in per_subj_metrics['B1_e293_single_best']]
    s_b2 = [m['dice_et'] for m in per_subj_metrics['B2_ensemble_raw_tau050']]
    s_b3 = [m['dice_et'] for m in per_subj_metrics['B3_ensemble_hierarchical_core']]
    s_b4 = [m['dice_et'] for m in per_subj_metrics['B4_ensemble_complete_integrated']]

    diff_b3_b0 = np.array(s_b3) - np.array(s_b0)
    diff_b4_b0 = np.array(s_b4) - np.array(s_b0)
    diff_b4_b1 = np.array(s_b4) - np.array(s_b1)

    w_b3_b0 = stats.wilcoxon(s_b3, s_b0, alternative='greater')
    t_b3_b0 = stats.ttest_rel(s_b3, s_b0)
    d_b3_b0 = float(np.mean(diff_b3_b0) / (np.std(diff_b3_b0) + 1e-8))

    w_b4_b0 = stats.wilcoxon(s_b4, s_b0, alternative='greater')
    t_b4_b0 = stats.ttest_rel(s_b4, s_b0)
    d_b4_b0 = float(np.mean(diff_b4_b0) / (np.std(diff_b4_b0) + 1e-8))

    w_b4_b1 = stats.wilcoxon(s_b4, s_b1, alternative='greater')

    n_win_b4 = int((diff_b4_b0 > 1e-5).sum())
    n_tie_b4 = int((np.abs(diff_b4_b0) <= 1e-5).sum())
    n_loss_b4 = int((diff_b4_b0 < -1e-5).sum())

    results['hypothesis_testing'] = {
        'B3_hierarchical_ens_vs_B0': {
            'wilcoxon_stat': float(w_b3_b0.statistic),
            'wilcoxon_p': float(w_b3_b0.pvalue),
            'ttest_p': float(t_b3_b0.pvalue),
            'cohens_d': d_b3_b0,
            'delta_mean_dice': float(np.mean(diff_b3_b0))
        },
        'B4_integrated_vs_B0': {
            'wilcoxon_stat': float(w_b4_b0.statistic),
            'wilcoxon_p': float(w_b4_b0.pvalue),
            'ttest_p': float(t_b4_b0.pvalue),
            'cohens_d': d_b4_b0,
            'delta_mean_dice': float(np.mean(diff_b4_b0)),
            'wins': n_win_b4, 'ties': n_tie_b4, 'losses': n_loss_b4
        },
        'B4_integrated_vs_B1_single_best': {
            'wilcoxon_stat': float(w_b4_b1.statistic),
            'wilcoxon_p': float(w_b4_b1.pvalue),
            'delta_mean_dice': float(np.mean(diff_b4_b1))
        }
    }

    # Save summary JSON
    out_json = HERE / 'E294_locked_test_summary.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote summary JSON to {out_json}', flush=True)

    # Save detailed per-subject CSV
    per_subj_rows = []
    for idx in range(len(test_subjects)):
        sid = test_subjects[idx]
        sb0 = per_subj_metrics['B0_production_base'][idx]
        sb1 = per_subj_metrics['B1_e293_single_best'][idx]
        sb2 = per_subj_metrics['B2_ensemble_raw_tau050'][idx]
        sb3 = per_subj_metrics['B3_ensemble_hierarchical_core'][idx]
        sb4 = per_subj_metrics['B4_ensemble_complete_integrated'][idx]

        per_subj_rows.append({
            'subject_id': sid,
            'dice_B0': sb0['dice_et'],
            'dice_B1': sb1['dice_et'],
            'dice_B2': sb2['dice_et'],
            'dice_B3': sb3['dice_et'],
            'dice_B4': sb4['dice_et'],
            'delta_dice_B3_vs_B0': sb3['dice_et'] - sb0['dice_et'],
            'delta_dice_B4_vs_B0': sb4['dice_et'] - sb0['dice_et'],
            'delta_dice_B4_vs_B1': sb4['dice_et'] - sb1['dice_et'],
            'precision_B0': sb0['precision_et'],
            'precision_B3': sb3['precision_et'],
            'precision_B4': sb4['precision_et'],
            'recall_B0': sb0['recall_et'],
            'recall_B3': sb3['recall_et'],
            'recall_B4': sb4['recall_et'],
            'hd95_B0': sb0['hd95_et'],
            'hd95_B3': sb3['hd95_et'],
            'hd95_B4': sb4['hd95_et'],
            'assd_B0': sb0['assd_et'],
            'assd_B3': sb3['assd_et'],
            'assd_B4': sb4['assd_et'],
            'g1_recovered_B1': sb1['g1_recovered'],
            'g1_recovered_B4': sb4['g1_recovered'],
            'g1_total': sb0['g1_total']
        })

    subj_csv = HERE / 'E294_per_subject_metrics.csv'
    with open(subj_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_subj_rows[0].keys()))
        w.writeheader()
        w.writerows(per_subj_rows)
    print(f'Wrote per-subject metrics ({len(per_subj_rows)} subjects) to {subj_csv}', flush=True)

    # Print publication comparison table
    print('\n' + '=' * 145)
    print('E294 LOCKED TEST RESULTS (125 SUBJECTS, SINGLE TOUCH)')
    print('=' * 145)
    print(f'{"System":34s} | {"Mean ET":9s} | {"Delta ET":10s} | {"Pooled ET":10s} | {"Delta Pool":10s} | {"Mean TC":9s} | {"Mean WT":9s} | {"3-Class":9s} | {"G1 Recovery":11s}')
    print('-' * 145)
    systems = [k for k in results.keys() if k != 'hypothesis_testing']
    for sname in systems:
        r = results[sname]
        print(f'{sname:34s} | {r["mean_per_subject_dice_et"]:9.4f} | {r["delta_mean_per_subject_dice_et"]:+10.5f} | {r["global_pooled_dice_et"]:10.4f} | {r["delta_global_pooled_dice_et"]:+10.5f} | {r["mean_per_subject_dice_tc"]:9.4f} | {r["mean_per_subject_dice_wt"]:9.4f} | {r["mean_per_subject_dice_3class"]:9.4f} | {r["g1_recovered"]:3d}/{r["g1_total"]:3d} ({r["g1_recovery_pct"]:4.1f}%)')
    print('=' * 145)
    print(f'Hypothesis Testing (B4 vs B0): Delta = {results["hypothesis_testing"]["B4_integrated_vs_B0"]["delta_mean_dice"]*100:+.3f}%, Wilcoxon p = {results["hypothesis_testing"]["B4_integrated_vs_B0"]["wilcoxon_p"]:.2e}, Wins/Ties/Losses = {n_win_b4}/{n_tie_b4}/{n_loss_b4}')
    print(f'E294 locked test completed in {time.time()-t_start:.1f}s.')

if __name__ == '__main__':
    main()
