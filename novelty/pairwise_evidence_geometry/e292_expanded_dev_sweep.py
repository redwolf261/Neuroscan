"""E292 -- Expanded Development Threshold Sweep (Option B, Development Only).

PROTOCOL:
1. Frozen production weights: w_p, b_p
2. Frozen Zone-1 calibration: 8-fold multi-axis spatial symmetry averaging (flips along D, H, W).
3. Evaluated strictly on the 13-subject development validation cohort (zero leakage to test).
4. Expanded grid: tau_p in {0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50}
5. For every tau record:
   - Mean per-subject Dice
   - Median Dice
   - SD
   - Pooled Dice
   - Precision
   - Recall
   - TP, FP, FN
   - HD95, ASSD
6. Select tau* = argmax_tau MeanDice_dev(tau)
7. Save full curve to E292_dev_expanded_curve.csv and summary to E292_dev_selected_tau.json
"""

import sys, os, csv, json, time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_REPO / 'experiments' / 'exp_e12_eggo_m' / 'e131'))

from e257_common import load_model, load_populations, build_splits, ROOT, PATCH
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch

ET = 0
STRUCT = ndimage.generate_binary_structure(3, 1)
THRESHOLDS = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50]


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


def compute_calibrated_map(model, ds, sid_to_idx, sid, w_p, b_p, dev):
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)

    with torch.no_grad():
        d1 = forward_to_dec1_internal(model, img_t)['relu2_out'][0]
        C_, D, H, W = d1.shape
        flat = d1.reshape(C_, -1).T
        W_p_t = torch.from_numpy(w_p.astype(np.float32)).to(dev)
        p_et_raw = torch.sigmoid(flat @ W_p_t + float(b_p)).reshape(D, H, W).cpu().numpy()
        del d1, flat, W_p_t

    flip_combos = [
        (2,), (3,), (4,),
        (2, 3), (2, 4), (3, 4),
        (2, 3, 4)
    ]
    p_et_flips = [p_et_raw]

    for dims in flip_combos:
        flipped_img = torch.flip(img_t, dims=dims)
        with torch.no_grad():
            d1_f = forward_to_dec1_internal(model, flipped_img)['relu2_out'][0]
            flat_f = d1_f.reshape(C_, -1).T
            W_p_t = torch.from_numpy(w_p.astype(np.float32)).to(dev)
            f_prod = torch.sigmoid(flat_f @ W_p_t + float(b_p))
            del W_p_t
            unflip_dims = tuple(d - 2 for d in dims)
            p_et_f = torch.flip(f_prod.reshape(D, H, W), dims=unflip_dims).cpu().numpy()
            del f_prod
            p_et_flips.append(p_et_f)
        del d1_f, flat_f, flipped_img

    del img_t
    torch.cuda.empty_cache()
    p_et_calib = np.mean(p_et_flips, axis=0)
    return p_et_raw, p_et_calib, tgt_c[ET] > 0.5


def run_expanded_dev_sweep(dev_subjects, model, ds, sid_to_idx, w_p, b_p, dev):
    print(f'Running Expanded Development Threshold Sweep on {len(dev_subjects)} subjects...')
    t0 = time.time()
    
    per_subj_data = {t: [] for t in THRESHOLDS}
    raw_subj_data = []

    for i, sid in enumerate(dev_subjects):
        if sid not in sid_to_idx:
            continue
        try:
            p_et_raw, p_et_cal, true_et = compute_calibrated_map(model, ds, sid_to_idx, sid, w_p, b_p, dev)
        except Exception as e:
            print(f'Error processing {sid}: {e}')
            continue
            
        # Baseline raw production (tau = 0.50)
        pred_raw = p_et_raw >= 0.50
        tp_r = int((pred_raw & true_et).sum())
        fp_r = int((pred_raw & ~true_et).sum())
        fn_r = int((~pred_raw & true_et).sum())
        d_r = (2 * tp_r) / max(2 * tp_r + fp_r + fn_r, 1e-8) if (true_et.sum() > 0 or pred_raw.sum() > 0) else 1.0
        prec_r = tp_r / max(tp_r + fp_r, 1e-8)
        rec_r = tp_r / max(tp_r + fn_r, 1e-8)
        hd_r, assd_r = hd95_and_assd(pred_raw, true_et)
        raw_subj_data.append({
            'sid': sid, 'tp': tp_r, 'fp': fp_r, 'fn': fn_r,
            'dice': d_r, 'precision': prec_r, 'recall': rec_r,
            'hd95': hd_r, 'assd': assd_r
        })
        
        # Sweep thresholds on calibrated map
        for t in THRESHOLDS:
            pred_t = p_et_cal >= t
            tp = int((pred_t & true_et).sum())
            fp = int((pred_t & ~true_et).sum())
            fn = int((~pred_t & true_et).sum())
            d = (2 * tp) / max(2 * tp + fp + fn, 1e-8) if (true_et.sum() > 0 or pred_t.sum() > 0) else 1.0
            prec = tp / max(tp + fp, 1e-8)
            rec = tp / max(tp + fn, 1e-8)
            hd, assd = hd95_and_assd(pred_t, true_et)
            per_subj_data[t].append({
                'sid': sid, 'tp': tp, 'fp': fp, 'fn': fn,
                'dice': d, 'precision': prec, 'recall': rec,
                'hd95': hd, 'assd': assd
            })
            
        del p_et_raw, p_et_cal, true_et
        if (i + 1) % 5 == 0 or (i + 1) == len(dev_subjects):
            print(f'  {i+1}/{len(dev_subjects)} subjects processed ({time.time()-t0:.0f}s)', flush=True)

    # Base raw production stats
    raw_mean_dice = float(np.mean([x['dice'] for x in raw_subj_data]))
    raw_med_dice = float(np.median([x['dice'] for x in raw_subj_data]))
    raw_sd_dice = float(np.std([x['dice'] for x in raw_subj_data]))
    raw_tot_tp = sum(x['tp'] for x in raw_subj_data)
    raw_tot_fp = sum(x['fp'] for x in raw_subj_data)
    raw_tot_fn = sum(x['fn'] for x in raw_subj_data)
    raw_pooled_dice = (2 * raw_tot_tp) / max(2 * raw_tot_tp + raw_tot_fp + raw_tot_fn, 1e-8)
    raw_mean_prec = float(np.mean([x['precision'] for x in raw_subj_data]))
    raw_mean_rec = float(np.mean([x['recall'] for x in raw_subj_data]))
    raw_hd_vals = [x['hd95'] for x in raw_subj_data if not np.isnan(x['hd95'])]
    raw_assd_vals = [x['assd'] for x in raw_subj_data if not np.isnan(x['assd'])]
    raw_mean_hd = float(np.mean(raw_hd_vals)) if raw_hd_vals else float('nan')
    raw_mean_assd = float(np.mean(raw_assd_vals)) if raw_assd_vals else float('nan')

    print('\n' + '=' * 115)
    print(f'RAW PRODUCTION BASELINE (tau=0.50):')
    print(f'  Mean Per-Subject Dice = {raw_mean_dice:.4f} +/- {raw_sd_dice:.4f} (Median: {raw_med_dice:.4f})')
    print(f'  Global Pooled Dice    = {raw_pooled_dice:.4f}')
    print(f'  Mean Precision        = {raw_mean_prec:.4f} | Mean Recall = {raw_mean_rec:.4f}')
    print(f'  HD95                  = {raw_mean_hd:.2f} mm | ASSD = {raw_mean_assd:.2f} mm')
    print(f'  Total FP Voxels       = {raw_tot_fp} | Total FN Voxels = {raw_tot_fn}')
    print('=' * 115)

    sweep_results = []
    best_tau = None
    best_mean_dice = -1.0

    print(f'{"Threshold":9s} | {"Mean Dice":9s} | {"Median":9s} | {"SD":6s} | {"Pooled":9s} | {"Delta Mean":10s} | {"Precision":9s} | {"Recall":9s} | {"HD95":7s} | {"Total FP":9s} | {"Total FN":9s}')
    print('-' * 118)

    for t in THRESHOLDS:
        data = per_subj_data[t]
        d_vals = [x['dice'] for x in data]
        m_dice = float(np.mean(d_vals))
        med_dice = float(np.median(d_vals))
        sd_dice = float(np.std(d_vals))
        tot_tp = sum(x['tp'] for x in data)
        tot_fp = sum(x['fp'] for x in data)
        tot_fn = sum(x['fn'] for x in data)
        p_dice = (2 * tot_tp) / max(2 * tot_tp + tot_fp + tot_fn, 1e-8)
        m_prec = float(np.mean([x['precision'] for x in data]))
        m_rec = float(np.mean([x['recall'] for x in data]))
        hd_vals = [x['hd95'] for x in data if not np.isnan(x['hd95'])]
        assd_vals = [x['assd'] for x in data if not np.isnan(x['assd'])]
        m_hd = float(np.mean(hd_vals)) if hd_vals else float('nan')
        m_assd = float(np.mean(assd_vals)) if assd_vals else float('nan')
        delta_m = m_dice - raw_mean_dice

        row = {
            'threshold': t,
            'mean_per_subject_dice': m_dice,
            'median_per_subject_dice': med_dice,
            'sd_per_subject_dice': sd_dice,
            'global_pooled_dice': p_dice,
            'delta_mean_dice_vs_raw_prod': delta_m,
            'mean_precision': m_prec,
            'mean_recall': m_rec,
            'mean_hd95': m_hd,
            'mean_assd': m_assd,
            'total_tp_voxels': tot_tp,
            'total_fp_voxels': tot_fp,
            'total_fn_voxels': tot_fn
        }
        sweep_results.append(row)
        print(f'{t:9.2f} | {m_dice:9.4f} | {med_dice:9.4f} | {sd_dice:6.4f} | {p_dice:9.4f} | {delta_m:+10.5f} | {m_prec:9.4f} | {m_rec:9.4f} | {m_hd:7.2f} | {tot_fp:9d} | {tot_fn:9d}')

        if m_dice > best_mean_dice:
            best_mean_dice = m_dice
            best_tau = t

    print('=' * 118)
    print(f'OPTIMAL THRESHOLD ON DEVELOPMENT COHORT: tau* = {best_tau:.2f}')
    print(f'  Mean Per-Subject Dice = {best_mean_dice:.4f} (Delta vs raw prod = {best_mean_dice - raw_mean_dice:+.5f})')
    print('=' * 118)

    # Save CSV
    out_csv = HERE / 'E292_dev_expanded_curve.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_results[0].keys()))
        w.writeheader()
        w.writerows(sweep_results)
    print(f'Wrote development curve to {out_csv}')

    # Save JSON summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_dev_subjects': len(dev_subjects),
        'primary_metric': 'mean_per_subject_dice',
        'raw_production_baseline': {
            'mean_per_subject_dice': raw_mean_dice,
            'median_per_subject_dice': raw_med_dice,
            'sd_per_subject_dice': raw_sd_dice,
            'global_pooled_dice': raw_pooled_dice,
            'mean_precision': raw_mean_prec,
            'mean_recall': raw_mean_rec,
            'mean_hd95': raw_mean_hd,
            'mean_assd': raw_mean_assd,
            'total_fp_voxels': raw_tot_fp,
            'total_fn_voxels': raw_tot_fn
        },
        'selected_tau_star': best_tau,
        'selected_mean_dice': best_mean_dice,
        'delta_vs_raw_prod': best_mean_dice - raw_mean_dice,
        'curve': sweep_results
    }
    out_json = HERE / 'E292_dev_selected_tau.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote selection summary to {out_json}')
    return best_tau, summary


def main():
    dev = torch.device('cuda')
    model = load_model(dev)
    conv_w = model.seg_head[0].weight.detach().cpu().numpy()
    conv_b = model.seg_head[0].bias.detach().cpu().numpy()
    w_p, b_p = conv_w[ET].reshape(32).astype(np.float64), float(conv_b[ET])

    det_lesions, g2a_lesions, _ = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    all_dev_sids = sorted(splits['det_test'])
    rng = np.random.default_rng(290)
    shuffled_dev = rng.permutation(all_dev_sids)
    n_dev = len(shuffled_dev)
    n_train = int(n_dev * 0.6)
    n_val = int(n_dev * 0.2)
    dev_val_sids = list(shuffled_dev[n_train:n_train + n_val])

    ds_full = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train', val_split=0.0, patch_size=PATCH)
    full_sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds_full.subject_dirs)}

    valid_sids = [sid for sid in dev_val_sids if sid in full_sid_to_idx]
    print(f'Development Validation Cohort (for tau selection): {len(valid_sids)} subjects')

    best_tau, summary = run_expanded_dev_sweep(valid_sids, model, ds_full, full_sid_to_idx, w_p, b_p, dev)


if __name__ == '__main__':
    main()
