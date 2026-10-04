"""E293 -- Development Sweep for Hierarchical Anatomical Consensus.

PROTOCOL (Strictly on 13 Development Validation Subjects):
1. Frozen production weights: w_p, w_tc, w_wt, b_p, b_tc, b_wt
2. Frozen spatial calibration (8-fold TTA)
3. Enforce anatomical hierarchy: ET is bounded by TC (Tumor Core)
   pred_ET = (p_et_cal >= tau_et) & (p_tc_cal >= tau_tc)
4. Sweep grid on development validation cohort:
   tau_et in [0.18, 0.20, 0.22, 0.25]
   tau_tc in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, None]
   min_comp_vox in [1, 2, 3]
5. Primary selection metric: Mean Per-Subject ET Dice.
6. Select and lock optimal parameters (tau_et*, tau_tc*, min_vox*).
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
from e291_dual_zone_precision_integration import compute_subject_maps_with_tta

ET, TC, WT = 0, 1, 2
STRUCT = ndimage.generate_binary_structure(3, 1)

TAU_ET_GRID = [0.18, 0.20, 0.22, 0.25]
TAU_TC_GRID = [None, 0.30, 0.35, 0.40, 0.45, 0.50]
MIN_VOX_GRID = [1, 2, 3]


def run_dev_sweep(valid_sids, model, ds_full, full_sid_to_idx, w_p, b_p, w_tc, b_tc, w_wt, b_wt, dev):
    print(f'Running Hierarchical Sweep on {len(valid_sids)} Development Subjects...')
    t0 = time.time()

    mrd_cache = torch.load(HERE / 'E290_mrd_cache.pt', map_location='cpu', weights_only=False)
    fitted_mrd = {999: mrd_cache[999]}

    # Pre-compute calibrated maps for all 13 dev subjects
    subj_data = []
    for sid in valid_sids:
        raw_maps, calib_maps, tgt_c = compute_subject_maps_with_tta(
            model, ds_full, full_sid_to_idx, sid, w_p, b_p, w_tc, b_tc, w_wt, b_wt, fitted_mrd, dev)
        subj_data.append({
            'sid': sid,
            'p_et': calib_maps[0],
            'p_tc': calib_maps[1],
            'p_wt': calib_maps[2],
            'true_et': tgt_c[0] > 0.5,
            'p_et_raw': raw_maps[0]
        })
    print(f'Computed calibrated maps for all dev subjects in {time.time()-t0:.1f}s')

    # Baseline raw production (tau=0.50)
    raw_dices = []
    for s in subj_data:
        p = s['p_et_raw'] >= 0.50
        g = s['true_et']
        d = (2 * (p & g).sum()) / max(2 * (p & g).sum() + (p & ~g).sum() + (~p & g).sum(), 1e-8) if (g.sum() > 0 or p.sum() > 0) else 1.0
        raw_dices.append(d)
    raw_mean_dice = float(np.mean(raw_dices))
    print(f'Development Baseline Raw Production Dice: {raw_mean_dice:.4f}')

    # Grid search
    sweep_rows = []
    best_mean_dice = -1.0
    best_config = None

    for tau_et in TAU_ET_GRID:
        for tau_tc in TAU_TC_GRID:
            for min_vox in MIN_VOX_GRID:
                dices = []
                for s in subj_data:
                    p_et = s['p_et']
                    p_tc = s['p_tc']
                    g = s['true_et']

                    mask = p_et >= tau_et
                    if tau_tc is not None:
                        mask &= (p_tc >= tau_tc)

                    if min_vox > 1:
                        lbl, n_c = ndimage.label(mask, structure=STRUCT)
                        if n_c > 0:
                            sizes = np.bincount(lbl.ravel())
                            too_small = sizes < min_vox
                            too_small[0] = False
                            mask[too_small[lbl]] = False

                    d = (2 * (mask & g).sum()) / max(2 * (mask & g).sum() + (mask & ~g).sum() + (~mask & g).sum(), 1e-8) if (g.sum() > 0 or mask.sum() > 0) else 1.0
                    dices.append(d)

                m_dice = float(np.mean(dices))
                med_dice = float(np.median(dices))
                delta = m_dice - raw_mean_dice

                tc_str = f'{tau_tc:.2f}' if tau_tc is not None else 'None'
                row = {
                    'tau_et': tau_et,
                    'tau_tc': tc_str,
                    'min_vox': min_vox,
                    'mean_per_subject_dice': m_dice,
                    'median_dice': med_dice,
                    'delta_mean_dice': delta
                }
                sweep_rows.append(row)

                if m_dice > best_mean_dice:
                    best_mean_dice = m_dice
                    best_config = (tau_et, tau_tc, min_vox)

    print('=' * 80)
    print(f'BEST CONFIGURATION ON DEVELOPMENT VALIDATION:')
    print(f'  tau_et = {best_config[0]:.2f}, tau_tc = {best_config[1]}, min_vox = {best_config[2]}')
    print(f'  Mean Per-Subject Dice = {best_mean_dice:.4f} (Delta vs raw prod = {best_mean_dice - raw_mean_dice:+.5f})')
    print('=' * 80)

    # Sort and display top 10 configurations
    sweep_rows.sort(key=lambda r: r['mean_per_subject_dice'], reverse=True)
    print('\nTop 10 Configurations on Development:')
    print(f'{"tau_et":7s} | {"tau_tc":7s} | {"min_vox":7s} | {"Mean Dice":9s} | {"Median":9s} | {"Delta":10s}')
    print('-' * 60)
    for r in sweep_rows[:10]:
        print(f'{r["tau_et"]:7.2f} | {r["tau_tc"]:7s} | {r["min_vox"]:7d} | {r["mean_per_subject_dice"]:9.4f} | {r["median_dice"]:9.4f} | {r["delta_mean_dice"]:+10.5f}')

    out_csv = HERE / 'E293_dev_hierarchical_curve.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)
    print(f'\nWrote sweep results to {out_csv}')

    spec = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'best_tau_et': best_config[0],
        'best_tau_tc': best_config[1],
        'best_min_vox': best_config[2],
        'best_mean_dice': best_mean_dice,
        'delta_vs_raw_prod': best_mean_dice - raw_mean_dice
    }
    out_json = HERE / 'E293_dev_best_config.json'
    with open(out_json, 'w') as f:
        json.dump(spec, f, indent=2)
    print(f'Wrote best config to {out_json}')


def main():
    dev = torch.device('cuda')
    model = load_model(dev)
    conv_w = model.seg_head[0].weight.detach().cpu().numpy()
    conv_b = model.seg_head[0].bias.detach().cpu().numpy()
    w_p, b_p = conv_w[ET].reshape(32).astype(np.float64), float(conv_b[ET])
    w_tc, b_tc = conv_w[TC].reshape(32).astype(np.float64), float(conv_b[TC])
    w_wt, b_wt = conv_w[WT].reshape(32).astype(np.float64), float(conv_b[WT])

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
    run_dev_sweep(valid_sids, model, ds_full, full_sid_to_idx, w_p, b_p, w_tc, b_tc, w_wt, b_wt, dev)


if __name__ == '__main__':
    main()
