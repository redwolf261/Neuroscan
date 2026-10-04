"""E291 -- Dual-Zone Precision Integration (Boundary Calibration + Distant MRD Rescue).

PRIMARY HYPOTHESIS:
Separating the spatial domain into two distinct zones:
  Zone 1 (Near Boundary <= 2mm of primary tumor): Boundary Calibration via multi-axis
         spatial averaging to eliminate the 1-voxel jitter responsible for 92% of
         production false positives.
  Zone 2 (Distant > 2mm from primary tumor): E290 Learned Candidate Acceptance to
         rescue genuine missed lesions (G1/G2-A satellites) without boundary pollution.
will simultaneously:
  1. PRESERVE MRD's missed-lesion recovery benefit (>8-10% G1 recovery).
  2. ACHIEVE a net positive ET Dice improvement (+0.7% to +1.0% or more) over raw production
     on the 125-subject locked test cohort.

PRE-REGISTERED CONDITIONS (Locked Test Cohort, 125 subjects):
  B0: Production Baseline (raw single-pass production)
  B1: Production + Raw MRD (E287 raw union baseline)
  B2: Production + E290 Learned Acceptance (pure candidate addition baseline)
  B3: Calibrated Production Alone (Zone-1 boundary calibration only)
  B4: Dual-Zone E291 (Zone-1 Calibrated Production + Zone-2 Distant MRD Acceptance)
  B5: Dual-Zone E291 Majority-Voted across seeds
"""

import sys, os, csv, json, time, gc
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_REPO / 'experiments' / 'exp_e12_eggo_m' / 'e131'))

from e257_common import (load_model, get_w_prod, load_populations, build_splits,
                          ROOT, PATCH, MIN_VOX, TAU)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch

ET, TC, WT = 0, 1, 2
EPS_MARG = 1e-6
TAU_LOW = 0.5
TAU_P = 0.5
TAU_R_BASE = 0.95082877089122
DIST_ZONE_BOUNDARY = 2.0  # Voxels: <= 2.0 is Zone 1 (Boundary), > 2.0 is Zone 2 (Distant)
DELTA_RESCUE = 0.05
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


def extract_27_features(comp_mask, p_et, p_tc, p_wt, r_map, edt_et, edt_wt, tau_r=TAU_R_BASE):
    voxels = np.argwhere(comp_mask)
    v_count = len(voxels)
    if v_count == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    r_vals = r_map[comp_mask]
    p_vals = p_et[comp_mask]

    mean_r = float(r_vals.mean())
    median_r = float(np.median(r_vals))
    max_r = float(r_vals.max())
    pct90_r = float(np.percentile(r_vals, 90))
    pct95_r = float(np.percentile(r_vals, 95))
    frac_above_tau_r = float((r_vals >= tau_r).mean())

    mean_p = float(p_vals.mean())
    median_p = float(np.median(p_vals))
    max_p = float(p_vals.max())
    min_p = float(p_vals.min())
    frac_p_lt_01 = float((p_vals < 0.1).mean())
    frac_p_lt_025 = float((p_vals < 0.25).mean())
    frac_p_lt_05 = float((p_vals < 0.5).mean())
    disagreement = float((r_vals - p_vals).mean())

    phys_vol = float(v_count)
    min_coords = voxels.min(axis=0)
    max_coords = voxels.max(axis=0)
    bbox_dx = float(max_coords[0] - min_coords[0] + 1)
    bbox_dy = float(max_coords[1] - min_coords[1] + 1)
    bbox_dz = float(max_coords[2] - min_coords[2] + 1)
    bbox_vol = max(bbox_dx * bbox_dy * bbox_dz, 1.0)
    compactness = float(v_count / bbox_vol)
    
    eroded = ndimage.binary_erosion(comp_mask, structure=STRUCT)
    surf_count = float((comp_mask & ~eroded).sum())
    surf_vol_ratio = float(surf_count / v_count)

    if edt_et is not None:
        min_dist_et = float(edt_et[comp_mask].min())
        mean_dist_et = float(edt_et[comp_mask].mean())
    else:
        min_dist_et = 128.0
        mean_dist_et = 128.0

    if edt_wt is not None:
        min_dist_wt = float(edt_wt[comp_mask].min())
    else:
        min_dist_wt = 128.0

    frac_in_wt = float((p_wt[comp_mask] >= 0.5).mean())
    frac_in_tc = float((p_tc[comp_mask] >= 0.5).mean())
    touching_et = 1.0 if min_dist_et <= 1.5 else 0.0

    feat = np.array([
        mean_r, median_r, max_r, pct90_r, pct95_r, frac_above_tau_r,
        mean_p, median_p, max_p, min_p, frac_p_lt_01, frac_p_lt_025, frac_p_lt_05, disagreement,
        float(v_count), phys_vol, bbox_dx, bbox_dy, bbox_dz, compactness, surf_vol_ratio,
        min_dist_et, mean_dist_et, min_dist_wt, frac_in_wt, frac_in_tc, touching_et
    ], dtype=np.float64)
    return feat


def compute_subject_maps_with_tta(model, ds, sid_to_idx, sid, w_p, b_p, w_tc, b_tc, w_wt, b_wt, fitted_mrd, dev):
    """
    Computes standard raw maps and 8-fold multi-axis spatial symmetry averaged (TTA) maps.
    Reclaims memory immediately after computing.
    """
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)

    # 1. Raw forward pass
    with torch.no_grad():
        d1 = forward_to_dec1_internal(model, img_t)['relu2_out'][0]
        C_, D, H, W = d1.shape
        flat = d1.reshape(C_, -1).T
        
        w_prod_stack = np.stack([w_p, w_tc, w_wt], axis=1) # (32, 3)
        b_prod_stack = np.array([b_p, b_tc, b_wt], dtype=np.float32)
        W_p_t = torch.from_numpy(w_prod_stack).float().to(dev)
        B_p_t = torch.from_numpy(b_prod_stack).float().to(dev)
        prod_probs = torch.sigmoid(flat @ W_p_t + B_p_t).cpu().numpy()
        del W_p_t, B_p_t

        p_et_raw = prod_probs[:, 0].reshape(D, H, W)
        p_tc_raw = prod_probs[:, 1].reshape(D, H, W)
        p_wt_raw = prod_probs[:, 2].reshape(D, H, W)
        del prod_probs

        r_maps_raw = {}
        for seed, (w_r, b_r) in fitted_mrd.items():
            w_r_t = torch.from_numpy(w_r).float().to(dev)
            r_maps_raw[seed] = torch.sigmoid(flat @ w_r_t + float(b_r)).reshape(D, H, W).cpu().numpy()
            del w_r_t

    del d1, flat

    # 2. 8-fold spatial symmetry averaging (flips along D, H, W)
    flip_combos = [
        (2,), (3,), (4,),
        (2, 3), (2, 4), (3, 4),
        (2, 3, 4)
    ]
    p_et_flips = [p_et_raw]
    p_tc_flips = [p_tc_raw]
    p_wt_flips = [p_wt_raw]
    r_maps_flips = {seed: [r_maps_raw[seed]] for seed in fitted_mrd.keys()}

    for dims in flip_combos:
        flipped_img = torch.flip(img_t, dims=dims)
        with torch.no_grad():
            d1_f = forward_to_dec1_internal(model, flipped_img)['relu2_out'][0]
            flat_f = d1_f.reshape(C_, -1).T
            
            W_p_t = torch.from_numpy(w_prod_stack).float().to(dev)
            B_p_t = torch.from_numpy(b_prod_stack).float().to(dev)
            f_prod = torch.sigmoid(flat_f @ W_p_t + B_p_t)
            del W_p_t, B_p_t
            
            unflip_dims = tuple(d - 2 for d in dims)
            p_et_f = torch.flip(f_prod[:, 0].reshape(D, H, W), dims=unflip_dims).cpu().numpy()
            p_tc_f = torch.flip(f_prod[:, 1].reshape(D, H, W), dims=unflip_dims).cpu().numpy()
            p_wt_f = torch.flip(f_prod[:, 2].reshape(D, H, W), dims=unflip_dims).cpu().numpy()
            del f_prod
            
            p_et_flips.append(p_et_f)
            p_tc_flips.append(p_tc_f)
            p_wt_flips.append(p_wt_f)

            for seed, (w_r, b_r) in fitted_mrd.items():
                w_r_t = torch.from_numpy(w_r).float().to(dev)
                r_f = torch.sigmoid(flat_f @ w_r_t + float(b_r)).reshape(D, H, W)
                r_maps_flips[seed].append(torch.flip(r_f, dims=unflip_dims).cpu().numpy())
                del w_r_t, r_f

        del d1_f, flat_f, flipped_img

    del img_t
    torch.cuda.empty_cache()

    p_et_calib = np.mean(p_et_flips, axis=0)
    p_tc_calib = np.mean(p_tc_flips, axis=0)
    p_wt_calib = np.mean(p_wt_flips, axis=0)
    r_maps_calib = {seed: np.mean(r_maps_flips[seed], axis=0) for seed in fitted_mrd.keys()}

    return (p_et_raw, p_tc_raw, p_wt_raw, r_maps_raw), (p_et_calib, p_tc_calib, p_wt_calib, r_maps_calib), tgt_c


# ---------------------------------------------------------------------------
# Streaming Locked-Test Evaluator for E291
# ---------------------------------------------------------------------------
def evaluate_e291_locked(model, ds, sid_to_idx, subjects, fitted_mrd,
                         w_p, b_p, w_tc, b_tc, w_wt, b_wt, dev,
                         acceptance_model, tau_star, tau_r=TAU_R_BASE, desc=''):
    """
    Evaluates B0 to B5 on locked test subjects in a single streaming pass.
    B0: Raw Production
    B1: Raw Production + Raw MRD (E287)
    B2: Raw Production + E290 Acceptance
    B3: Calibrated Production Alone (Zone-1 boundary calibration)
    B4: Dual-Zone E291 (Zone-1 Calibrated Prod + Zone-2 Distant Accepted MRD)
    B5: Dual-Zone E291 with Multi-Seed Majority Voting
    """
    conditions = [
        'B0_production_raw',
        'B1_production_plus_raw_mrd',
        'B2_production_plus_e290_acceptance',
        'B3_calibrated_production_alone',
        'B4_dual_zone_e291',
        'B5_dual_zone_e291_majority_voted'
    ]

    stats = {c: {
        'tp_et': 0, 'fp_et': 0, 'fn_et': 0, 'tn_et': 0,
        'tp_tc': 0, 'fp_tc': 0, 'fn_tc': 0,
        'tp_wt': 0, 'fp_wt': 0, 'fn_wt': 0,
        'g1_recovered': [], 'g1_total': 0,
        'fp_vox_total': 0, 'fp_comp_total': 0.0,
        'hd95_vals': [], 'assd_vals': []
    } for c in conditions}

    n_subjects = len(subjects)
    t0 = time.time()

    for i, sid in enumerate(subjects):
        if sid not in sid_to_idx:
            continue
        
        raw_maps, calib_maps, tgt_c = compute_subject_maps_with_tta(
            model, ds, sid_to_idx, sid, w_p, b_p, w_tc, b_tc, w_wt, b_wt, fitted_mrd, dev)
        
        p_et_raw, p_tc_raw, p_wt_raw, r_maps_raw = raw_maps
        p_et_cal, p_tc_cal, p_wt_cal, r_maps_cal = calib_maps

        true_et = tgt_c[ET] > 0.5
        true_tc = tgt_c[TC] > 0.5
        true_wt = tgt_c[WT] > 0.5

        # Reference G1 lesions (production confidently missed)
        et_lbl_gt, n_et_gt = ndimage.label(true_et, structure=STRUCT)
        g1_lesion_masks = []
        for lid in range(1, n_et_gt + 1):
            lmask = et_lbl_gt == lid
            if lmask.sum() < MIN_VOX:
                continue
            if p_et_raw[lmask].max() < TAU_LOW:
                g1_lesion_masks.append(lmask)

        # Base masks
        prod_et_raw = p_et_raw >= TAU_P
        prod_tc_raw = p_tc_raw >= TAU_P
        prod_wt_raw = p_wt_raw >= TAU_P

        prod_et_cal = p_et_cal >= TAU_P
        prod_tc_cal = p_tc_cal >= TAU_P
        prod_wt_cal = p_wt_cal >= TAU_P

        edt_et_raw = ndimage.distance_transform_edt(~prod_et_raw) if prod_et_raw.any() else None
        edt_wt_raw = ndimage.distance_transform_edt(~prod_wt_raw) if prod_wt_raw.any() else None
        edt_et_cal = ndimage.distance_transform_edt(~prod_et_cal) if prod_et_cal.any() else None
        edt_wt_cal = ndimage.distance_transform_edt(~prod_wt_cal) if prod_wt_cal.any() else None

        # Build predictions for this subject:
        # B0: Raw Production
        pred_b0_et = prod_et_raw
        pred_b0_tc = prod_tc_raw
        pred_b0_wt = prod_wt_raw

        # B3: Calibrated Production Alone
        pred_b3_et = prod_et_cal
        pred_b3_tc = prod_tc_cal
        pred_b3_wt = prod_wt_cal

        # B1: Production + Raw MRD (seed 999)
        ref_seed = SEEDS[0]
        r_raw_ref = r_maps_raw[ref_seed]
        mrd_raw_pos = (r_raw_ref > tau_r) & (~prod_et_raw)
        pred_b1_et = prod_et_raw | mrd_raw_pos
        pred_b1_tc = prod_tc_raw | pred_b1_et
        pred_b1_wt = prod_wt_raw | pred_b1_et

        # B2: Production + E290 Learned Acceptance (Multi-Seed Majority Voted, matching E290 B5)
        seed_b2_preds = []
        fp_comp_b2_tot = 0
        for seed in fitted_mrd.keys():
            r_seed_raw = r_maps_raw[seed]
            mrd_s_raw = (r_seed_raw > tau_r) & (~prod_et_raw)
            cc_lbl_s, cc_n_s = ndimage.label(mrd_s_raw, structure=STRUCT)
            admitted_s = np.zeros_like(prod_et_raw)
            for cid in range(1, cc_n_s + 1):
                cmask = cc_lbl_s == cid
                feat = extract_27_features(cmask, p_et_raw, p_tc_raw, p_wt_raw, r_seed_raw, edt_et_raw, edt_wt_raw, tau_r)
                prob = acceptance_model.predict_proba(feat[FEATS_A2].reshape(1, -1))[0, 1]
                if prob >= tau_star:
                    admitted_s |= cmask
                    if not (cmask & true_et).any():
                        fp_comp_b2_tot += 1
            seed_b2_preds.append(prod_et_raw | admitted_s)
        pred_b2_et = (np.stack(seed_b2_preds).sum(axis=0) >= (len(fitted_mrd) / 2.0))
        pred_b2_tc = prod_tc_raw | pred_b2_et
        pred_b2_wt = prod_wt_raw | pred_b2_et
        fp_comp_b2 = fp_comp_b2_tot / len(fitted_mrd)

        # B4: Dual-Zone E291 (Zone-1 Calibrated Prod + Zone-2 Distant Accepted MRD, single seed)
        r_cal_ref = r_maps_cal[ref_seed]
        mrd_cal_pos = (r_cal_ref > tau_r) & (~prod_et_cal)
        cc_lbl_cal, cc_n_cal = ndimage.label(mrd_cal_pos, structure=STRUCT)
        admitted_b4 = np.zeros_like(prod_et_cal)
        fp_comp_b4 = 0
        for cid in range(1, cc_n_cal + 1):
            cmask = cc_lbl_cal == cid
            feat = extract_27_features(cmask, p_et_cal, p_tc_cal, p_wt_cal, r_cal_ref, edt_et_cal, edt_wt_cal, tau_r)
            prob = acceptance_model.predict_proba(feat[FEATS_A2].reshape(1, -1))[0, 1]
            min_dist_to_prod = feat[21] # min_dist_to_prod_et
            
            # Dual-Zone Rule: only admit if accepted AND in Zone 2 (Distant > 2mm from primary core)
            if prob >= tau_star and min_dist_to_prod > DIST_ZONE_BOUNDARY:
                admitted_b4 |= cmask
                if not (cmask & true_et).any():
                    fp_comp_b4 += 1

        pred_b4_et = prod_et_cal | admitted_b4
        pred_b4_tc = prod_tc_cal | pred_b4_et
        pred_b4_wt = prod_wt_cal | pred_b4_et

        # B5: Dual-Zone E291 Multi-Seed Majority Voted
        seed_b5_et = []
        fp_comp_b5_total = 0
        for seed in fitted_mrd.keys():
            r_seed = r_maps_cal[seed]
            mrd_s_pos = (r_seed > tau_r) & (~prod_et_cal)
            cc_lbl_s, cc_n_s = ndimage.label(mrd_s_pos, structure=STRUCT)
            admitted_s = np.zeros_like(prod_et_cal)
            for cid in range(1, cc_n_s + 1):
                cmask = cc_lbl_s == cid
                feat = extract_27_features(cmask, p_et_cal, p_tc_cal, p_wt_cal, r_seed, edt_et_cal, edt_wt_cal, tau_r)
                prob = acceptance_model.predict_proba(feat[FEATS_A2].reshape(1, -1))[0, 1]
                if prob >= tau_star and feat[21] > DIST_ZONE_BOUNDARY:
                    admitted_s |= cmask
                    if not (cmask & true_et).any():
                        fp_comp_b5_total += 1
            seed_b5_et.append(prod_et_cal | admitted_s)

        pred_b5_et = (np.stack(seed_b5_et).sum(axis=0) >= (len(fitted_mrd) / 2.0))
        pred_b5_tc = prod_tc_cal | pred_b5_et
        pred_b5_wt = prod_wt_cal | pred_b5_et
        fp_comp_b5 = fp_comp_b5_total / len(fitted_mrd)

        subj_preds = {
            'B0_production_raw': (pred_b0_et, pred_b0_tc, pred_b0_wt, 0),
            'B1_production_plus_raw_mrd': (pred_b1_et, pred_b1_tc, pred_b1_wt, float((mrd_raw_pos & ~true_et).sum() > 0)),
            'B2_production_plus_e290_acceptance': (pred_b2_et, pred_b2_tc, pred_b2_wt, fp_comp_b2),
            'B3_calibrated_production_alone': (pred_b3_et, pred_b3_tc, pred_b3_wt, 0),
            'B4_dual_zone_e291': (pred_b4_et, pred_b4_tc, pred_b4_wt, fp_comp_b4),
            'B5_dual_zone_e291_majority_voted': (pred_b5_et, pred_b5_tc, pred_b5_wt, fp_comp_b5),
        }

        for cname in conditions:
            st = stats[cname]
            st['g1_total'] += len(g1_lesion_masks)
            p_et_m, p_tc_m, p_wt_m, fp_c = subj_preds[cname]

            st['tp_et'] += int((p_et_m & true_et).sum())
            st['fp_et'] += int((p_et_m & ~true_et).sum())
            st['fn_et'] += int((~p_et_m & true_et).sum())
            st['tn_et'] += int((~p_et_m & ~true_et).sum())

            st['tp_tc'] += int((p_tc_m & true_tc).sum())
            st['fp_tc'] += int((p_tc_m & ~true_tc).sum())
            st['fn_tc'] += int((~p_tc_m & true_tc).sum())

            st['tp_wt'] += int((p_wt_m & true_wt).sum())
            st['fp_wt'] += int((p_wt_m & ~true_wt).sum())
            st['fn_wt'] += int((~p_wt_m & true_wt).sum())

            st['fp_vox_total'] += int((p_et_m & ~true_et).sum())
            st['fp_comp_total'] += fp_c

            for lmask in g1_lesion_masks:
                st['g1_recovered'].append(int((p_et_m & lmask).any()))

            hd, assd = hd95_and_assd(p_et_m, true_et)
            if not np.isnan(hd):
                st['hd95_vals'].append(hd)
            if not np.isnan(assd):
                st['assd_vals'].append(assd)

        del raw_maps, calib_maps, tgt_c, true_et, true_tc, true_wt, subj_preds
        gc.collect()

        if (i + 1) % 10 == 0 or (i + 1) == n_subjects:
            print(f'  {desc} {i+1}/{n_subjects} subjects processed ({time.time()-t0:.0f}s)', flush=True)

    results = {}
    for cname in conditions:
        st = stats[cname]
        tp_et, fp_et, fn_et, tn_et = st['tp_et'], st['fp_et'], st['fn_et'], st['tn_et']
        dice_et = (2 * tp_et) / max(2 * tp_et + fp_et + fn_et, 1e-8)
        dice_tc = (2 * st['tp_tc']) / max(2 * st['tp_tc'] + st['fp_tc'] + st['fn_tc'], 1e-8)
        dice_wt = (2 * st['tp_wt']) / max(2 * st['tp_wt'] + st['fp_wt'] + st['fn_wt'], 1e-8)
        mean_dice = (dice_et + dice_tc + dice_wt) / 3.0

        iou = tp_et / max(tp_et + fp_et + fn_et, 1e-8)
        precision = tp_et / max(tp_et + fp_et, 1e-8)
        recall = tp_et / max(tp_et + fn_et, 1e-8)
        specificity = tn_et / max(tn_et + fp_et, 1e-8)

        g1_rate = float(np.mean(st['g1_recovered'])) if st['g1_recovered'] else 0.0
        hd95_mean = float(np.mean(st['hd95_vals'])) if st['hd95_vals'] else float('nan')
        assd_mean = float(np.mean(st['assd_vals'])) if st['assd_vals'] else float('nan')

        results[cname] = {
            'dice_et': dice_et, 'dice_tc': dice_tc, 'dice_wt': dice_wt, 'mean_dice': mean_dice,
            'iou': iou, 'precision': precision, 'recall': recall, 'specificity': specificity,
            'hd95': hd95_mean, 'assd': assd_mean,
            'g1_recovered': sum(st['g1_recovered']), 'g1_total': st['g1_total'], 'g1_recovery_pct': g1_rate * 100.0,
            'fp_voxels_per_subj': st['fp_vox_total'] / max(n_subjects, 1),
            'fp_comps_per_subj': st['fp_comp_total'] / max(n_subjects, 1),
            'total_fp_voxels': st['fp_vox_total'], 'total_fp_comps': st['fp_comp_total']
        }
    return results


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')
    t_start = time.time()
    sfx = '_smoke' if smoke else ''

    print('=' * 80)
    print('E291: DUAL-ZONE PRECISION INTEGRATION (PRE-REGISTERED EXPERIMENT)')
    print('Zone 1 (Near Boundary <= 2mm): Calibration | Zone 2 (Distant > 2mm): MRD Rescue')
    print('=' * 80, flush=True)

    # 1. Load frozen model & weights
    model = load_model(dev)
    conv_w = model.seg_head[0].weight.detach().cpu().numpy()
    conv_b = model.seg_head[0].bias.detach().cpu().numpy()
    w_p, b_p = conv_w[ET].reshape(32).astype(np.float64), float(conv_b[ET])
    w_tc, b_tc = conv_w[TC].reshape(32).astype(np.float64), float(conv_b[TC])
    w_wt, b_wt = conv_w[WT].reshape(32).astype(np.float64), float(conv_b[WT])

    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    # 2. Load pre-fitted MRD weights from E290 cache
    mrd_cache_path = HERE / 'E290_mrd_cache.pt'
    cached_mrd = torch.load(mrd_cache_path, map_location='cpu', weights_only=False)
    seeds = SEEDS[:2] if smoke else SEEDS
    fitted_mrd = {s: cached_mrd[s] for s in seeds}
    print(f'Loaded fitted MRD weights for seeds {seeds} from {mrd_cache_path.name}', flush=True)

    # 3. Load pre-computed dev records & train frozen A2 acceptance model
    train_cache = HERE / f'E290_train_records{sfx}.pt'
    train_records = torch.load(train_cache, map_location='cpu', weights_only=False)
    print(f'Loaded {len(train_records)} Dev-Train components from {train_cache.name}', flush=True)

    X_train = np.stack([r['feat'][FEATS_A2] for r in train_records])
    y_train = np.array([r['label'] for r in train_records], dtype=int)
    acceptance_model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=290))
    ])
    acceptance_model.fit(X_train, y_train)
    tau_star = 0.80 # Frozen pre-registered threshold from E290
    print(f'Fit frozen A2 acceptance model (tau_star = {tau_star:.2f})', flush=True)

    # 4. Prepare locked test cohort
    all_prior_touched = (set(r['subject_id'] for r in det_lesions) |
                          set(r['subject_id'] for r in g2a_lesions) |
                          set(r['subject_id'] for r in g2b_lesions))
    
    ds_val_pop = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val', val_split=0.1, patch_size=PATCH)
    val_sids_all = sorted(os.path.basename(d) for d in ds_val_pop.subject_dirs)
    val_sids_fresh = sorted(set(val_sids_all) - all_prior_touched)

    ds_full = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train', val_split=0.0, patch_size=PATCH)
    full_sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds_full.subject_dirs)}

    test_pool = val_sids_fresh[:10] if smoke else val_sids_fresh
    test_subjects = []
    for sid in test_pool:
        if sid not in full_sid_to_idx:
            continue
        try:
            img_c, tgt_c = load_patch(ds_full, full_sid_to_idx, sid)
            img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
            with torch.no_grad():
                stages = forward_to_dec1_internal(model, img_t)
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

    leakage_train = set(test_subjects) & (splits['det_train'] | splits['g2a_train'])
    assert len(leakage_train) == 0, 'LEAKAGE AUDIT FAILED'
    print(f'Locked Test Cohort: {len(test_subjects)} subjects. Leakage Audit: PASS!\n', flush=True)

    # 5. Execute Single-Touch Locked Test Evaluation
    print('Running Single-Touch Streaming Evaluation of B0..B5 on Locked Test Cohort...', flush=True)
    res_dict = evaluate_e291_locked(
        model, ds_full, full_sid_to_idx, test_subjects, fitted_mrd,
        w_p, b_p, w_tc, b_tc, w_wt, b_wt, dev,
        acceptance_model, tau_star=tau_star, tau_r=TAU_R_BASE, desc='[Locked-Test]')

    prod_res = res_dict['B0_production_raw']
    test_rows = []

    for cname in res_dict.keys():
        res = res_dict[cname]
        delta_dice = res['dice_et'] - prod_res['dice_et']
        delta_rec = res['recall'] - prod_res['recall']
        delta_prec = res['precision'] - prod_res['precision']
        delta_fp_vox = res['total_fp_voxels'] - prod_res['total_fp_voxels']
        delta_g1 = res['g1_recovery_pct'] - prod_res['g1_recovery_pct']

        row = {
            'condition': cname,
            'dice_et': res['dice_et'], 'dice_tc': res['dice_tc'], 'dice_wt': res['dice_wt'],
            'mean_dice': res['mean_dice'], 'iou': res['iou'],
            'precision': res['precision'], 'recall': res['recall'], 'specificity': res['specificity'],
            'hd95': res['hd95'], 'assd': res['assd'],
            'g1_recovered': res['g1_recovered'], 'g1_total': res['g1_total'], 'g1_recovery_pct': res['g1_recovery_pct'],
            'fp_voxels_per_subj': res['fp_voxels_per_subj'], 'fp_comps_per_subj': res['fp_comps_per_subj'],
            'delta_dice': delta_dice, 'delta_recall': delta_rec, 'delta_precision': delta_prec,
            'delta_fp_voxels': delta_fp_vox, 'delta_g1_recovery': delta_g1
        }
        test_rows.append(row)
        print(f'  {cname:38s}: ET Dice={res["dice_et"]:.4f} (dDice={delta_dice:+.5f}) | G1={res["g1_recovery_pct"]:5.1f}% | HD95={res["hd95"]:.2f}', flush=True)

    test_path = HERE / f'E291_locked_test{sfx}.csv'
    with open(test_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
        w.writeheader()
        w.writerows(test_rows)
    print(f'\nWrote locked test results to {test_path}', flush=True)

    # 6. Generate Figures
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for r in test_rows:
        c = r['condition']
        marker = 'o'
        color = 'blue'
        if 'B0' in c:
            marker, color = '*', 'black'
        elif 'B1' in c:
            marker, color = 'x', 'red'
        elif 'B2' in c:
            marker, color = 's', 'orange'
        elif 'B3' in c:
            marker, color = 'D', 'cyan'
        elif 'B4' in c:
            marker, color = '^', 'green'
        elif 'B5' in c:
            marker, color = 'P', 'magenta'

        ax.scatter(r['g1_recovery_pct'], r['dice_et'], s=140, color=color, marker=marker, label=c, zorder=5)

    ax.axhline(prod_res['dice_et'], color='gray', linestyle='--', alpha=0.5, label='Production Baseline')
    ax.set_xlabel('G1 Missed-Lesion Recovery (%)', fontsize=12)
    ax.set_ylabel('ET Dice Score', fontsize=12)
    ax.set_title('E291: Dual-Zone Precision Integration (Dice vs G1 Recovery)', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.tight_layout()
    fig1_path = HERE / f'E291_fig1_dice_vs_g1{sfx}.png'
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f'Saved Figure 1 to {fig1_path}', flush=True)

    # 7. Summary Verdict
    print('\n' + '=' * 80)
    print('E291 EXPERIMENT SUMMARY & VERDICT')
    print('=' * 80)
    b0 = next(r for r in test_rows if 'B0' in r['condition'])
    b1 = next(r for r in test_rows if 'B1' in r['condition'])
    b2 = next(r for r in test_rows if 'B2' in r['condition'])
    b3 = next(r for r in test_rows if 'B3' in r['condition'])
    b4 = next(r for r in test_rows if 'B4' in r['condition'])
    b5 = next(r for r in test_rows if 'B5' in r['condition'])

    print(f'B0: Raw Production:       Dice={b0["dice_et"]:.4f}  G1={b0["g1_recovery_pct"]:.1f}%  HD95={b0["hd95"]:.2f}')
    print(f'B1: Raw MRD (E287):       Dice={b1["dice_et"]:.4f}  (dDice={b1["delta_dice"]:+.5f})  G1={b1["g1_recovery_pct"]:.1f}%')
    print(f'B2: E290 Acceptance:      Dice={b2["dice_et"]:.4f}  (dDice={b2["delta_dice"]:+.5f})  G1={b2["g1_recovery_pct"]:.1f}%')
    print(f'B3: Calibrated Prod Only: Dice={b3["dice_et"]:.4f}  (dDice={b3["delta_dice"]:+.5f})  G1={b3["g1_recovery_pct"]:.1f}%')
    print(f'B4: Dual-Zone E291:       Dice={b4["dice_et"]:.4f}  (dDice={b4["delta_dice"]:+.5f})  G1={b4["g1_recovery_pct"]:.1f}%')
    print(f'B5: Dual-Zone E291 Maj:   Dice={b5["dice_et"]:.4f}  (dDice={b5["delta_dice"]:+.5f})  G1={b5["g1_recovery_pct"]:.1f}%')

    summary_json = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'smoke': smoke,
        'results': test_rows
    }
    summary_path = HERE / f'E291_summary{sfx}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_json, f, indent=2)
    print(f'Wrote summary to {summary_path}')
    print(f'E291 completed in {time.time()-t_start:.1f}s.')


if __name__ == '__main__':
    main()
