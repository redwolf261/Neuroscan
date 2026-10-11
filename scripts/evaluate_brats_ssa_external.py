"""
External Cross-Dataset Evaluation on BraTS 2023 Sub-Saharan Africa (BraTS-Africa / SSA).
Cohort: N = 60 subjects (Adewole et al., 2026, DOI: 10.7937/v8h6-8x67).

This script performs a strictly frozen, single-touch external evaluation:
- Frozen Model: UNet3D_v5 (E131, seed 0)
- Frozen Thresholds:
    * B0 (Production Baseline): tau = 0.50
    * B2 (Calibrated Core): tau* = 0.20 (hierarchically gated by P_WT >= 0.50)
    * B3 (Proposed Integrated): B2 + Zone 2 Distant MRD Rescue (tau_R = 0.9508, A2 accept prob >= 0.80, dist > 2.0mm)
- Zero retraining, zero re-fitting, zero threshold adaptation on the SSA cohort.
"""

import sys
import os
import csv
import json
import time
import gc
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from scipy import ndimage
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "novelty" / "pairwise_evidence_geometry"))
sys.path.insert(0, str(ROOT / "experiments" / "exp_e12_eggo_m" / "e131"))

from neuroscan_3d_v5 import UNet3D_v5
from Dataset.brats_multimodal_dataset import MODALITIES, znorm_brain, labels_to_regions
from e245_dec1_internal_decomposition import forward_to_dec1_internal
from e291_dual_zone_precision_integration import extract_27_features, FEATS_A2

CKPT = ROOT / "experiments" / "exp_e12_eggo_m" / "e131" / "runs" / "E131_v5control_seed0" / "checkpoints" / "best.pth"
DATA_DIR = ROOT / "Dataset" / "BraTS2023_SSA"
MRD_CACHE = ROOT / "novelty" / "pairwise_evidence_geometry" / "E290_mrd_cache.pt"
TRAIN_CACHE = ROOT / "novelty" / "pairwise_evidence_geometry" / "E290_train_records.pt"
OUT_CSV = ROOT / "novelty" / "pairwise_evidence_geometry" / "BraTS_SSA_external_metrics.csv"
OUT_JSON = ROOT / "novelty" / "pairwise_evidence_geometry" / "BraTS_SSA_external_summary.json"

ET, TC, WT = 0, 1, 2
PATCH = (128, 128, 128)
MIN_VOX = 5
TAU_LOW = 0.50
TAU_P_BASE = 0.50
TAU_STAR_LOCKED = 0.20
TAU_R_BASE = 0.95082877089122
TAU_ACCEPTANCE = 0.80
DIST_ZONE_BOUNDARY = 2.0
SEEDS = [999, 4242, 7, 123, 2024]
STRUCT = ndimage.generate_binary_structure(3, 1)


def load_model(dev):
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)
    return model


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


def load_ssa_subject_patch(subject_dir):
    sid = subject_dir.name
    chans = []
    for m in MODALITIES:
        p = subject_dir / f"{sid}-{m}.nii.gz"
        if not p.exists():
            raise FileNotFoundError(f"Missing modality file: {p}")
        vol = nib.load(str(p)).get_fdata().astype(np.float32)
        chans.append(znorm_brain(vol))
    img = np.stack(chans, axis=0)

    seg_p = subject_dir / f"{sid}-seg.nii.gz"
    if not seg_p.exists():
        raise FileNotFoundError(f"Missing seg file: {seg_p}")
    seg = nib.load(str(seg_p)).get_fdata().astype(np.uint8)
    tgt = labels_to_regions(seg)

    # Compute brain bounding box across all 4 channels
    brain = np.zeros(img.shape[1:], dtype=bool)
    for c in range(img.shape[0]):
        brain |= (img[c] != 0)
    if not brain.any():
        lo = np.array([0, 0, 0])
        hi = np.array(img.shape[1:])
    else:
        idx = np.argwhere(brain)
        lo, hi = idx.min(0), idx.max(0) + 1

    img_cropped = img[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    tgt_cropped = tgt[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    # Center crop / pad to PATCH (128, 128, 128)
    D, H, W = img_cropped.shape[1:]
    pd_, ph_, pw_ = PATCH
    cd, ch, cw = D // 2, H // 2, W // 2
    sd_, ed_ = max(0, cd - pd_ // 2), min(D, cd + pd_ // 2)
    sh_, eh_ = max(0, ch - ph_ // 2), min(H, ch + ph_ // 2)
    sw_, ew_ = max(0, cw - pw_ // 2), min(W, cw + pw_ // 2)

    img_c = img_cropped[:, sd_:ed_, sh_:eh_, sw_:ew_]
    tgt_c = tgt_cropped[:, sd_:ed_, sh_:eh_, sw_:ew_]

    if img_c.shape[1:] != PATCH:
        ip = np.zeros((4, pd_, ph_, pw_), dtype=np.float32)
        tp_ = np.zeros((3, pd_, ph_, pw_), dtype=np.float32)
        ip[:, :img_c.shape[1], :img_c.shape[2], :img_c.shape[3]] = img_c
        tp_[:, :tgt_c.shape[1], :tgt_c.shape[2], :tgt_c.shape[3]] = tgt_c
        img_c, tgt_c = ip, tp_

    return img_c, tgt_c


def compute_subject_maps_with_tta(model, img_c, tgt_c, w_p, b_p, w_tc, b_tc, w_wt, b_wt, fitted_mrd, dev):
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)

    # 1. Raw forward pass
    with torch.no_grad():
        d1 = forward_to_dec1_internal(model, img_t)['relu2_out'][0]
        C_, D, H, W = d1.shape
        flat = d1.reshape(C_, -1).T

        w_prod_stack = np.stack([w_p, w_tc, w_wt], axis=1)
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

    # 2. 8-fold spatial symmetry averaging
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


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 85)
    print('EXTERNAL CROSS-DATASET EVALUATION: BraTS-Africa (Sub-Saharan Africa / BraTS-SSA)')
    print('Cohort: N = 60 subjects (Adewole et al., 2026, DOI: 10.7937/v8h6-8x67)')
    print(f'Device: {dev}')
    print(f'Primary Endpoint: Per-Subject ET Dice (B0 vs B2 vs B3)')
    print('=' * 85, flush=True)

    # 1. Discover subjects
    subject_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("BraTS-SSA")])
    if len(subject_dirs) == 0:
        # Check if inside nested folder
        sub_nested = DATA_DIR / "ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
        if sub_nested.exists():
            subject_dirs = sorted([d for d in sub_nested.iterdir() if d.is_dir() and d.name.startswith("BraTS-SSA")])
    print(f"Found {len(subject_dirs)} BraTS-SSA subjects for external evaluation.")
    if len(subject_dirs) == 0:
        print("ERROR: No BraTS-SSA subjects found. Ensure download is finished.")
        return

    # 2. Load frozen model & production weights
    model = load_model(dev)
    conv_w = model.seg_head[0].weight.detach().cpu().numpy()
    conv_b = model.seg_head[0].bias.detach().cpu().numpy()
    w_p, b_p = conv_w[ET].reshape(32).astype(np.float64), float(conv_b[ET])
    w_tc, b_tc = conv_w[TC].reshape(32).astype(np.float64), float(conv_b[TC])
    w_wt, b_wt = conv_w[WT].reshape(32).astype(np.float64), float(conv_b[WT])

    # 3. Load pre-fitted MRD weights
    cached_mrd = torch.load(MRD_CACHE, map_location='cpu', weights_only=False)
    fitted_mrd = {s: cached_mrd[s] for s in SEEDS}

    # 4. Load pre-computed dev records & train frozen A2 acceptance model
    train_records = torch.load(TRAIN_CACHE, map_location='cpu', weights_only=False)
    X_train = np.stack([r['feat'][FEATS_A2] for r in train_records])
    y_train = np.array([r['label'] for r in train_records], dtype=int)
    acceptance_model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=290))
    ])
    acceptance_model.fit(X_train, y_train)

    systems = ['B0_production_base', 'B1_e291_dual_zone_tau0.50', 'B2_calib_prod_tau_star', 'B3_e292_calib_tau_star_plus_mrd']
    per_subj_metrics = {s: [] for s in systems}
    pooled_metrics = {s: {
        'tp_et': 0, 'fp_et': 0, 'fn_et': 0,
        'g1_recovered': 0, 'g1_total': 0,
        'hd95_vals': [], 'assd_vals': []
    } for s in systems}

    t_start = time.time()
    t0 = t_start
    n_subjects = len(subject_dirs)

    for i, sdir in enumerate(subject_dirs):
        sid = sdir.name
        img_c, tgt_c = load_ssa_subject_patch(sdir)

        raw_maps, calib_maps, tgt_c = compute_subject_maps_with_tta(
            model, img_c, tgt_c, w_p, b_p, w_tc, b_tc, w_wt, b_wt, fitted_mrd, dev
        )

        p_et_raw, p_tc_raw, p_wt_raw, r_maps_raw = raw_maps
        p_et_cal, p_tc_cal, p_wt_cal, r_maps_cal = calib_maps
        true_et = tgt_c[ET] > 0.5

        # Reference G1 lesions (missed by raw baseline, voxels >= MIN_VOX)
        et_lbl_gt, n_et_gt = ndimage.label(true_et, structure=STRUCT)
        g1_masks = []
        for lid in range(1, n_et_gt + 1):
            lm = et_lbl_gt == lid
            if lm.sum() >= MIN_VOX and p_et_raw[lm].max() < TAU_LOW:
                g1_masks.append(lm)

        # B0: Raw Production
        pred_b0 = p_et_raw >= TAU_P_BASE

        # B2: Calibrated Production alone (tau* = 0.20)
        pred_b2 = p_et_cal >= TAU_STAR_LOCKED

        # B1: Calibrated tau = 0.50 + MRD Zone 2
        core_b1 = p_et_cal >= TAU_P_BASE
        wt_cal_50 = p_wt_cal >= 0.50
        edt_et_b1 = ndimage.distance_transform_edt(~core_b1) if core_b1.any() else None
        edt_wt_b1 = ndimage.distance_transform_edt(~wt_cal_50) if wt_cal_50.any() else None

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

        # B3: Calibrated tau* = 0.20 + MRD Zone 2
        core_b3 = p_et_cal >= TAU_STAR_LOCKED
        edt_et_b3 = ndimage.distance_transform_edt(~core_b3) if core_b3.any() else None
        edt_wt_b3 = edt_wt_b1

        seed_b3_preds = []
        fp_comp_b3_tot = 0
        for seed in fitted_mrd.keys():
            r_seed = r_maps_cal[seed]
            mrd_s_pos = (r_seed > TAU_R_BASE) & (~core_b3)
            cc_lbl, cc_n = ndimage.label(mrd_s_pos, structure=STRUCT)
            admitted_s = np.zeros_like(core_b3)
            for cid in range(1, cc_n + 1):
                cm = cc_lbl == cid
                feat = extract_27_features(cm, p_et_cal, p_tc_cal, p_wt_cal, r_seed, edt_et_b3, edt_wt_b3, TAU_R_BASE)
                prob = acceptance_model.predict_proba(feat[FEATS_A2].reshape(1, -1))[0, 1]
                if prob >= TAU_ACCEPTANCE and feat[21] > DIST_ZONE_BOUNDARY:
                    admitted_s |= cm
                    if not (cm & true_et).any():
                        fp_comp_b3_tot += 1
            seed_b3_preds.append(core_b3 | admitted_s)
        pred_b3 = (np.stack(seed_b3_preds).sum(axis=0) >= (len(fitted_mrd) / 2.0))
        fp_comp_b3 = fp_comp_b3_tot / len(fitted_mrd)

        sys_masks = {
            'B0_production_base': (pred_b0, 0.0),
            'B1_e291_dual_zone_tau0.50': (pred_b1, fp_comp_b1),
            'B2_calib_prod_tau_star': (pred_b2, 0.0),
            'B3_e292_calib_tau_star_plus_mrd': (pred_b3, fp_comp_b3)
        }

        for sname, (pred, fp_c) in sys_masks.items():
            tp = int((pred & true_et).sum())
            fp = int((pred & ~true_et).sum())
            fn = int((~pred & true_et).sum())

            d = (2 * tp) / max(2 * tp + fp + fn, 1e-8) if (true_et.sum() > 0 or pred.sum() > 0) else 1.0
            prec = tp / max(tp + fp, 1e-8)
            rec = tp / max(tp + fn, 1e-8)
            hd, assd = hd95_and_assd(pred, true_et)

            g1_rec = sum(1 for lm in g1_masks if (pred & lm).any())

            per_subj_metrics[sname].append({
                'sid': sid, 'dice': d, 'precision': prec, 'recall': rec,
                'tp': tp, 'fp': fp, 'fn': fn, 'fp_comp': fp_c,
                'hd95': hd, 'assd': assd,
                'g1_recovered': g1_rec, 'g1_total': len(g1_masks)
            })

            pm = pooled_metrics[sname]
            pm['tp_et'] += tp
            pm['fp_et'] += fp
            pm['fn_et'] += fn
            pm['g1_recovered'] += g1_rec
            pm['g1_total'] += len(g1_masks)
            pm['fp_comp_total'] = pm.get('fp_comp_total', 0.0) + fp_c
            if not np.isnan(hd):
                pm['hd95_vals'].append(hd)
            if not np.isnan(assd):
                pm['assd_vals'].append(assd)

        del raw_maps, calib_maps, tgt_c, true_et, sys_masks, pred_b0, pred_b1, pred_b2, pred_b3
        gc.collect()

        if (i + 1) % 5 == 0 or (i + 1) == n_subjects:
            print(f"  [BraTS-SSA External] {i+1}/{n_subjects} subjects processed ({time.time()-t0:.0f}s)", flush=True)

    # Statistical Testing & Summary
    summary = {}
    base_m_dice = float(np.mean([x['dice'] for x in per_subj_metrics['B0_production_base']]))
    base_p_dice = (2 * pooled_metrics['B0_production_base']['tp_et']) / max(
        2 * pooled_metrics['B0_production_base']['tp_et'] + pooled_metrics['B0_production_base']['fp_et'] + pooled_metrics['B0_production_base']['fn_et'], 1e-8
    )

    b0_dices = [x['dice'] for x in per_subj_metrics['B0_production_base']]
    b2_dices = [x['dice'] for x in per_subj_metrics['B2_calib_prod_tau_star']]
    b3_dices = [x['dice'] for x in per_subj_metrics['B3_e292_calib_tau_star_plus_mrd']]

    # Wilcoxon paired tests
    def paired_stats(a, b):
        diff = np.array(b) - np.array(a)
        med_diff = float(np.median(diff))
        q25, q75 = float(np.percentile(diff, 25)), float(np.percentile(diff, 75))
        nz = diff[diff != 0]
        if len(nz) > 0:
            res = wilcoxon(b, a, alternative='two-sided')
            p_val = float(res.pvalue)
            n_pos = np.sum(diff > 0)
            n_neg = np.sum(diff < 0)
            r_rb = float((n_pos - n_neg) / (n_pos + n_neg)) if (n_pos + n_neg) > 0 else 0.0
        else:
            p_val, r_rb = 1.0, 0.0
        return med_diff, (q75 - q25), p_val, r_rb

    b0_b2_med, b0_b2_iqr, b0_b2_p, b0_b2_r = paired_stats(b0_dices, b2_dices)
    b0_b3_med, b0_b3_iqr, b0_b3_p, b0_b3_r = paired_stats(b0_dices, b3_dices)
    b2_b3_med, b2_b3_iqr, b2_b3_p, b2_b3_r = paired_stats(b2_dices, b3_dices)

    for sname in systems:
        subj_list = per_subj_metrics[sname]
        d_vals = [x['dice'] for x in subj_list]
        m_dice = float(np.mean(d_vals))
        med_dice = float(np.median(d_vals))
        sd_dice = float(np.std(d_vals))
        iqr_dice = float(np.percentile(d_vals, 75) - np.percentile(d_vals, 25))

        pm = pooled_metrics[sname]
        pooled_d = (2 * pm['tp_et']) / max(2 * pm['tp_et'] + pm['fp_et'] + pm['fn_et'], 1e-8)
        m_prec = float(np.mean([x['precision'] for x in subj_list]))
        m_rec = float(np.mean([x['recall'] for x in subj_list]))
        m_hd = float(np.nanmean(pm['hd95_vals'])) if pm['hd95_vals'] else float('nan')
        m_assd = float(np.nanmean(pm['assd_vals'])) if pm['assd_vals'] else float('nan')
        g1_pct = (pm['g1_recovered'] / max(pm['g1_total'], 1)) * 100.0

        summary[sname] = {
            'mean_dice': m_dice,
            'median_dice': med_dice,
            'iqr_dice': iqr_dice,
            'std_dice': sd_dice,
            'delta_mean_dice': m_dice - base_m_dice,
            'global_pooled_dice': pooled_d,
            'delta_pooled_dice': pooled_d - base_p_dice,
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

    summary['comparative_statistics'] = {
        'B0_to_B2': {'median_diff': b0_b2_med, 'iqr_diff': b0_b2_iqr, 'p_value': b0_b2_p, 'rank_biserial_r': b0_b2_r},
        'B0_to_B3': {'median_diff': b0_b3_med, 'iqr_diff': b0_b3_iqr, 'p_value': b0_b3_p, 'rank_biserial_r': b0_b3_r},
        'B2_to_B3': {'median_diff': b2_b3_med, 'iqr_diff': b2_b3_iqr, 'p_value': b2_b3_p, 'rank_biserial_r': b2_b3_r}
    }

    # Write per-subject CSV
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'subject_id',
            'b0_dice', 'b0_prec', 'b0_rec', 'b0_hd95', 'b0_assd', 'b0_g1_rec', 'b0_g1_tot',
            'b2_dice', 'b2_prec', 'b2_rec', 'b2_hd95', 'b2_assd', 'b2_g1_rec', 'b2_g1_tot',
            'b3_dice', 'b3_prec', 'b3_rec', 'b3_hd95', 'b3_assd', 'b3_g1_rec', 'b3_g1_tot',
            'delta_b3_minus_b0'
        ])
        for idx in range(n_subjects):
            s_b0 = per_subj_metrics['B0_production_base'][idx]
            s_b2 = per_subj_metrics['B2_calib_prod_tau_star'][idx]
            s_b3 = per_subj_metrics['B3_e292_calib_tau_star_plus_mrd'][idx]
            sid = s_b0['sid']
            writer.writerow([
                sid,
                s_b0['dice'], s_b0['precision'], s_b0['recall'], s_b0['hd95'], s_b0['assd'], s_b0['g1_recovered'], s_b0['g1_total'],
                s_b2['dice'], s_b2['precision'], s_b2['recall'], s_b2['hd95'], s_b2['assd'], s_b2['g1_recovered'], s_b2['g1_total'],
                s_b3['dice'], s_b3['precision'], s_b3['recall'], s_b3['hd95'], s_b3['assd'], s_b3['g1_recovered'], s_b3['g1_total'],
                s_b3['dice'] - s_b0['dice']
            ])

    # Write summary JSON
    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 85)
    print("CONSOLIDATED EXTERNAL EVALUATION RESULTS (BraTS-SSA, N = 60)")
    print("=" * 85)
    print(f"{'System':<32} | {'Mean Dice':<10} | {'Median':<8} | {'HD95':<8} | {'ASSD':<8} | {'G1 Recov':<12}")
    print("-" * 85)
    for sname in systems:
        res = summary[sname]
        g1_str = f"{res['g1_recovered']}/{res['g1_total']} ({res['g1_recovery_pct']:.1f}%)"
        print(f"{sname:<32} | {res['mean_dice']*100:.2f}%     | {res['median_dice']*100:.2f}%  | {res['mean_hd95']:.2f}   | {res['mean_assd']:.2f}   | {g1_str}")
    print("=" * 85)
    print(f"B0 -> B3 Wilcoxon p-value: {b0_b3_p:.4e}, Rank-biserial r: {b0_b3_r:.3f}")
    print(f"Per-subject metrics written to: {OUT_CSV}")
    print(f"Summary statistics written to: {OUT_JSON}")
    print(f"Total time elapsed: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
