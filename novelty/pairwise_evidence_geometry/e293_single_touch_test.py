"""E293 -- Hierarchical Anatomical Consensus & Precision Gating (Single-Touch Locked Test).

PRIMARY HYPOTHESIS:
Enforcing the fundamental neuro-oncology anatomical constraint:
  ET subset WT  (Enhancing Tumor must lie within the Whole Tumor envelope)
via hierarchical containment:
  Core_ET = (P_ET >= tau_et*) & (P_WT >= 0.50)
will eliminate the localized false-positive leakage in outlier cases,
pushing the cohort Mean Per-Subject ET Dice gain past the +1.0% threshold
while preserving MRD's 16% G1 missed-lesion recovery.

FROZEN OPERATING SPECIFICATION (Locked strictly from dev validation):
  tau_et* = 0.15 (from dev plateau: +1.04% gain on dev validation)
  tau_wt* = 0.50 (whole tumor containment threshold)
  tau_acceptance = 0.80 (frozen A2 candidate acceptance model)
  tau_r_base = 0.95082877089122 (frozen MRD readout operating point)
  dist_zone_boundary = 2.0 voxels (Zone 1 <= 2mm, Zone 2 > 2mm)

PRE-REGISTERED SYSTEMS (Touch locked test set ONCE):
  B0: Production Baseline (raw single-pass, tau=0.50)
  B1: E292 Unconstrained Best (calib tau=0.20 + Zone-2 MRD, previous +0.60% baseline)
  B2: E293 Hierarchically Calibrated Core Alone (tau_et=0.15 & WT>=0.50)
  B3: E293 Complete Integrated System (Hierarchical Core + Zone-2 Distant MRD Rescue)
"""

import sys, os, csv, json, time, gc
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_REPO / 'experiments' / 'exp_e12_eggo_m' / 'e131'))

from e257_common import (load_model, load_populations, build_splits,
                          ROOT, PATCH, MIN_VOX, TAU)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch
from e291_dual_zone_precision_integration import extract_27_features, FEATS_A2, compute_subject_maps_with_tta

ET, TC, WT = 0, 1, 2
TAU_LOW = 0.50
TAU_P_BASE = 0.50
TAU_E292 = 0.20              # E292 baseline threshold
TAU_ET_LOCKED = 0.15         # Locked from E293 dev sweep (+1.04% dev gain)
TAU_WT_LOCKED = 0.50         # Locked Whole Tumor containment threshold
TAU_R_BASE = 0.95082877089122 # Frozen raw MRD operating point
TAU_ACCEPTANCE = 0.80        # Frozen acceptance model threshold
DIST_ZONE_BOUNDARY = 2.0     # 2mm boundary shell: <= 2.0 is Zone 1, > 2.0 is Zone 2
SEEDS = [999, 4242, 7, 123, 2024]
STRUCT = ndimage.generate_binary_structure(3, 1)


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


def evaluate_e293_locked(model, ds, sid_to_idx, subjects, fitted_mrd,
                         w_p, b_p, w_tc, b_tc, w_wt, b_wt, dev, acceptance_model):
    systems = [
        'B0_production_base',
        'B1_e292_unconstrained_best',
        'B2_e293_hierarchical_core_alone',
        'B3_e293_complete_integrated'
    ]

    per_subj_metrics = {s: [] for s in systems}
    pooled_metrics = {s: {
        'tp_et': 0, 'fp_et': 0, 'fn_et': 0,
        'g1_recovered': 0, 'g1_total': 0,
        'hd95_vals': [], 'assd_vals': []
    } for s in systems}

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

        # Reference G1 lesions (confidently missed by raw production)
        et_lbl_gt, n_et_gt = ndimage.label(true_et, structure=STRUCT)
        g1_masks = []
        for lid in range(1, n_et_gt + 1):
            lm = et_lbl_gt == lid
            if lm.sum() >= MIN_VOX and p_et_raw[lm].max() < TAU_LOW:
                g1_masks.append(lm)

        # ---------------------------------------------------------------
        # System B0: Raw Production (tau = 0.50)
        # ---------------------------------------------------------------
        pred_b0 = p_et_raw >= TAU_P_BASE

        # ---------------------------------------------------------------
        # System B1: E292 Unconstrained Best (tau = 0.20 + Zone-2 MRD)
        # ---------------------------------------------------------------
        core_b1 = p_et_cal >= TAU_E292
        edt_et_b1 = ndimage.distance_transform_edt(~core_b1) if core_b1.any() else None
        edt_wt_b1 = ndimage.distance_transform_edt(~(p_wt_cal >= 0.50)) if (p_wt_cal >= 0.50).any() else None

        seed_b1_preds = []
        fp_comp_b1_tot = 0
        for seed in fitted_mrd.keys():
            r_seed = r_maps_cal[seed]
            mrd_s_pos = (r_seed > TAU_R_BASE) & (~core_b1)
            cc_lbl, cc_n = ndimage.label(mrd_s_pos, structure=STRUCT)
            admitted_s = np.zeros_like(core_b1)
            for cid in range(1, cc_n + 1):
                cm = cc_lbl == cid
                feat = extract_27_features(cm, p_et_cal, p_tc_cal, p_wt_cal, r_seed, edt_et_b1, edt_wt_b1, TAU_R_BASE)
                prob = acceptance_model.predict_proba(feat[FEATS_A2].reshape(1, -1))[0, 1]
                if prob >= TAU_ACCEPTANCE and feat[21] > DIST_ZONE_BOUNDARY:
                    admitted_s |= cm
                    if not (cm & true_et).any():
                        fp_comp_b1_tot += 1
            seed_b1_preds.append(core_b1 | admitted_s)
        pred_b1 = (np.stack(seed_b1_preds).sum(axis=0) >= (len(fitted_mrd) / 2.0))
        fp_comp_b1 = fp_comp_b1_tot / len(fitted_mrd)

        # ---------------------------------------------------------------
        # System B2: E293 Hierarchically Calibrated Core Alone (tau_et=0.15 & WT>=0.50)
        # ---------------------------------------------------------------
        core_b2 = (p_et_cal >= TAU_ET_LOCKED) & (p_wt_cal >= TAU_WT_LOCKED)
        pred_b2 = core_b2

        # ---------------------------------------------------------------
        # System B3: E293 Complete Integrated System (Hierarchical Core + Zone-2 MRD Rescue)
        # ---------------------------------------------------------------
        edt_et_b3 = ndimage.distance_transform_edt(~core_b2) if core_b2.any() else None
        edt_wt_b3 = edt_wt_b1

        seed_b3_preds = []
        fp_comp_b3_tot = 0
        for seed in fitted_mrd.keys():
            r_seed = r_maps_cal[seed]
            mrd_s_pos = (r_seed > TAU_R_BASE) & (~core_b2)
            cc_lbl, cc_n = ndimage.label(mrd_s_pos, structure=STRUCT)
            admitted_s = np.zeros_like(core_b2)
            for cid in range(1, cc_n + 1):
                cm = cc_lbl == cid
                feat = extract_27_features(cm, p_et_cal, p_tc_cal, p_wt_cal, r_seed, edt_et_b3, edt_wt_b3, TAU_R_BASE)
                prob = acceptance_model.predict_proba(feat[FEATS_A2].reshape(1, -1))[0, 1]
                if prob >= TAU_ACCEPTANCE and feat[21] > DIST_ZONE_BOUNDARY:
                    admitted_s |= cm
                    if not (cm & true_et).any():
                        fp_comp_b3_tot += 1
            seed_b3_preds.append(core_b2 | admitted_s)
        pred_b3 = (np.stack(seed_b3_preds).sum(axis=0) >= (len(fitted_mrd) / 2.0))
        fp_comp_b3 = fp_comp_b3_tot / len(fitted_mrd)

        # Record metrics for all 4 systems
        sys_masks = {
            'B0_production_base': (pred_b0, 0.0),
            'B1_e292_unconstrained_best': (pred_b1, fp_comp_b1),
            'B2_e293_hierarchical_core_alone': (pred_b2, 0.0),
            'B3_e293_complete_integrated': (pred_b3, fp_comp_b3)
        }

        for sname, (pred, fp_c) in sys_masks.items():
            tp = int((pred & true_et).sum())
            fp = int((pred & ~true_et).sum())
            fn = int((~pred & true_et).sum())

            d = (2 * tp) / max(2 * tp + fp + fn, 1e-8) if (true_et.sum() > 0 or pred.sum() > 0) else 1.0
            prec = tp / max(tp + fp, 1e-8)
            rec = tp / max(tp + fn, 1e-8)
            hd, assd = hd95_and_assd(pred, true_et)

            # G1 recoveries
            g1_rec = sum(1 for lm in g1_masks if (pred & lm).any())

            per_subj_metrics[sname].append({
                'sid': sid, 'dice': d, 'precision': prec, 'recall': rec,
                'tp': tp, 'fp': fp, 'fn': fn, 'fp_comp': fp_c,
                'hd95': hd, 'assd': assd,
                'g1_recovered': g1_rec, 'g1_total': len(g1_masks)
            })

            # Pooled accumulation
            pm = pooled_metrics[sname]
            pm['tp_et'] += tp
            pm['fp_et'] += fp
            pm['fn_et'] += fn
            pm['g1_recovered'] += g1_rec
            pm['g1_total'] += len(g1_masks)
            pm['fp_comp_total'] = pm.get('fp_comp_total', 0.0) + fp_c
            if not np.isnan(hd): pm['hd95_vals'].append(hd)
            if not np.isnan(assd): pm['assd_vals'].append(assd)

        del raw_maps, calib_maps, tgt_c, true_et, sys_masks, pred_b0, pred_b1, pred_b2, pred_b3
        gc.collect()

        if (i + 1) % 10 == 0 or (i + 1) == n_subjects:
            print(f'  [E293 Locked-Test Single Touch] {i+1}/{n_subjects} subjects processed ({time.time()-t0:.0f}s)', flush=True)

    # Compute final consolidated results
    results = {}
    base_m_dice = float(np.mean([x['dice'] for x in per_subj_metrics['B0_production_base']]))
    base_p_dice = (2 * pooled_metrics['B0_production_base']['tp_et']) / max(
        2 * pooled_metrics['B0_production_base']['tp_et'] + pooled_metrics['B0_production_base']['fp_et'] + pooled_metrics['B0_production_base']['fn_et'], 1e-8)

    for sname in systems:
        subj_list = per_subj_metrics[sname]
        d_vals = [x['dice'] for x in subj_list]
        m_dice = float(np.mean(d_vals))
        med_dice = float(np.median(d_vals))
        sd_dice = float(np.std(d_vals))

        pm = pooled_metrics[sname]
        pooled_d = (2 * pm['tp_et']) / max(2 * pm['tp_et'] + pm['fp_et'] + pm['fn_et'], 1e-8)

        m_prec = float(np.mean([x['precision'] for x in subj_list]))
        m_rec = float(np.mean([x['recall'] for x in subj_list]))
        m_hd = float(np.mean(pm['hd95_vals'])) if pm['hd95_vals'] else float('nan')
        m_assd = float(np.mean(pm['assd_vals'])) if pm['assd_vals'] else float('nan')

        g1_pct = (pm['g1_recovered'] / max(pm['g1_total'], 1)) * 100.0

        results[sname] = {
            'system': sname,
            'mean_per_subject_dice': m_dice,
            'median_per_subject_dice': med_dice,
            'sd_per_subject_dice': sd_dice,
            'delta_mean_per_subject_dice': m_dice - base_m_dice,
            'global_pooled_dice': pooled_d,
            'delta_global_pooled_dice': pooled_d - base_p_dice,
            'mean_precision': m_prec,
            'mean_recall': m_rec,
            'mean_hd95': m_hd,
            'mean_assd': m_assd,
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

    print('=' * 95)
    print('E293: HIERARCHICAL ANATOMICAL CONSENSUS & PRECISION GATING (LOCKED TEST)')
    print(f'Locked Operating Point: tau_et* = {TAU_ET_LOCKED:.2f}, tau_wt* = {TAU_WT_LOCKED:.2f}')
    print('Primary Endpoint: Mean Per-Subject ET Dice')
    print('=' * 95, flush=True)

    # 1. Load frozen production model
    model = load_model(dev)
    conv_w = model.seg_head[0].weight.detach().cpu().numpy()
    conv_b = model.seg_head[0].bias.detach().cpu().numpy()
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

    leakage_dev = set(test_subjects) & set(splits['det_test'])
    assert len(leakage_dev) == 0, 'LEAKAGE AUDIT FAILED'
    print(f'Locked Test Cohort: {len(test_subjects)} subjects. Leakage Audit: PASS!\n', flush=True)

    # Serialize pre-test frozen specification (E293)
    frozen_spec = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'locked_tau_et_star': TAU_ET_LOCKED,
        'locked_tau_wt_star': TAU_WT_LOCKED,
        'primary_endpoint': 'mean_per_subject_dice',
        'n_locked_test_subjects': len(test_subjects),
        'seeds': SEEDS,
        'tau_r_base': TAU_R_BASE,
        'tau_acceptance': TAU_ACCEPTANCE,
        'dist_zone_boundary_voxels': DIST_ZONE_BOUNDARY
    }
    with open(HERE / 'E293_frozen_spec.json', 'w') as f:
        json.dump(frozen_spec, f, indent=2)

    # Execute single-touch evaluation
    results, per_subj_metrics = evaluate_e293_locked(
        model, ds_full, full_sid_to_idx, test_subjects, fitted_mrd,
        w_p, b_p, w_tc, b_tc, w_wt, b_wt, dev, acceptance_model
    )

    # Save summary tables
    rows = list(results.values())
    out_csv = HERE / 'E293_locked_test_results.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nWrote locked test results to {out_csv}', flush=True)

    out_json = HERE / 'E293_locked_test_summary.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote summary JSON to {out_json}', flush=True)

    # Save detailed per-subject metrics for paired analysis
    per_subj_rows = []
    n_subjs = len(test_subjects)
    for idx in range(n_subjs):
        s_b0 = per_subj_metrics['B0_production_base'][idx]
        s_b1 = per_subj_metrics['B1_e292_unconstrained_best'][idx]
        s_b2 = per_subj_metrics['B2_e293_hierarchical_core_alone'][idx]
        s_b3 = per_subj_metrics['B3_e293_complete_integrated'][idx]
        sid = s_b0['sid']

        per_subj_rows.append({
            'subject_id': sid,
            'dice_B0': s_b0['dice'],
            'dice_B1': s_b1['dice'],
            'dice_B2': s_b2['dice'],
            'dice_B3': s_b3['dice'],
            'delta_dice_B2_vs_B0': s_b2['dice'] - s_b0['dice'],
            'delta_dice_B3_vs_B0': s_b3['dice'] - s_b0['dice'],
            'delta_dice_B3_vs_B1': s_b3['dice'] - s_b1['dice'],
            'precision_B0': s_b0['precision'],
            'precision_B1': s_b1['precision'],
            'precision_B2': s_b2['precision'],
            'precision_B3': s_b3['precision'],
            'recall_B0': s_b0['recall'],
            'recall_B1': s_b1['recall'],
            'recall_B2': s_b2['recall'],
            'recall_B3': s_b3['recall'],
            'hd95_B0': s_b0['hd95'],
            'hd95_B1': s_b1['hd95'],
            'hd95_B2': s_b2['hd95'],
            'hd95_B3': s_b3['hd95'],
            'assd_B0': s_b0['assd'],
            'assd_B1': s_b1['assd'],
            'assd_B2': s_b2['assd'],
            'assd_B3': s_b3['assd'],
            'g1_recovered_B1': s_b1['g1_recovered'],
            'g1_recovered_B3': s_b3['g1_recovered'],
            'g1_total': s_b0['g1_total'],
            'tp_B0': s_b0['tp'], 'fp_B0': s_b0['fp'], 'fn_B0': s_b0['fn'],
            'tp_B2': s_b2['tp'], 'fp_B2': s_b2['fp'], 'fn_B2': s_b2['fn'],
            'tp_B3': s_b3['tp'], 'fp_B3': s_b3['fp'], 'fn_B3': s_b3['fn'],
        })

    subj_csv = HERE / 'E293_per_subject_metrics.csv'
    with open(subj_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_subj_rows[0].keys()))
        w.writeheader()
        w.writerows(per_subj_rows)
    print(f'Wrote per-subject metrics ({len(per_subj_rows)} subjects) to {subj_csv}', flush=True)

    # Print publication comparison table
    print('\n' + '=' * 135)
    print('E293 LOCKED TEST RESULTS (125 SUBJECTS, SINGLE TOUCH)')
    print('=' * 135)
    print(f'{"System":34s} | {"Mean Dice":9s} | {"Delta Mean":10s} | {"Pooled Dice":11s} | {"Delta Pool":10s} | {"Precision":9s} | {"Recall":9s} | {"HD95":7s} | {"G1 Recovery":11s}')
    print('-' * 135)
    for r in rows:
        print(f'{r["system"]:34s} | {r["mean_per_subject_dice"]:9.4f} | {r["delta_mean_per_subject_dice"]:+10.5f} | {r["global_pooled_dice"]:11.4f} | {r["delta_global_pooled_dice"]:+10.5f} | {r["mean_precision"]:9.4f} | {r["mean_recall"]:9.4f} | {r["mean_hd95"]:7.2f} | {r["g1_recovered"]:3d}/{r["g1_total"]:3d} ({r["g1_recovery_pct"]:4.1f}%)')
    print('=' * 135)
    print(f'E293 locked test completed in {time.time()-t_start:.1f}s.')


if __name__ == '__main__':
    main()
