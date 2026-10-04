"""E290 -- Learned MRD Candidate Acceptance (Pre-Registered Experiment).

PRIMARY HYPOTHESIS:
A learned acceptance function using inference-time properties of an MRD candidate
can preserve the G1-recovery benefit of MRD while reducing its Dice/FP cost
relative to raw MRD and fixed gating.

The experiment does NOT modify the frozen MRD formula:
  t(x) = clip((y - p) / (1 - p + 1e-6), 0, 1)
Production and MRD weights remain frozen.

EXPERIMENT PROTOCOL:
Stage A: Development Cohort Split (60% train, 20% validation, 20% holdout)
  - Extract connected components C_k from M_R \\ M_P (tau_R = 0.9508)
  - Extract 27 inference-time features (MRD, Production, Geometry, Spatial)
  - Train 3 pre-registered logistic acceptance models:
      A0: Size-only baseline
      A1: MRD/Production confidence
      A2: Full candidate model
  - Sweep acceptance threshold tau_A on dev-validation set:
      Select tau_A* = argmax G1Recovery(tau) s.t. Dice >= Dice_prod - 0.001
  - Freeze models and operating thresholds before Stage B.
  - Development ablation: randomly shuffled labels test.

Stage B: Locked Evaluation Cohort (125 subjects, evaluated ONCE)
  - Evaluate 6 pre-registered conditions in a streaming unified pass:
      B0: Production alone
      B1: Production + raw MRD
      B2: Production + E289 fixed gate
      B3: Production + learned acceptance A0
      B4: Production + learned acceptance A1
      B5: Production + learned acceptance A2
  - Report segmentation metrics: ET/TC/WT Dice, mean Dice, IoU, Precision, Recall, Specificity, HD95, ASSD
  - Report detection metrics: G1 recovered, G1 %, FP voxels/subject, FP components/subject
  - Report changes from production (Delta Dice, Delta Recall, Delta Precision, Delta FP, Delta G1)
  - Descriptive feature analysis: true rescues vs false candidates
  - Generate figures: ET Dice vs G1 recovery, G1 recovery vs FP components
"""

import sys, os, csv, json, time, gc
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_REPO / 'experiments' / 'exp_e12_eggo_m' / 'e131'))

from e257_common import (load_model, get_w_prod, load_populations, build_splits,
                          extract_lesion_shell, extract_distant_background,
                          ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch

ET, TC, WT = 0, 1, 2
EPS_MARG = 1e-6
TAU_LOW = 0.5
TAU_P = 0.5
TAU_R_BASE = 0.95082877089122       # Frozen raw MRD operating point (E283 FP~500)
E289A_GATE_TAU_R = 0.987932080391847 # Frozen E289A fixed gate operating point
DELTA_RESCUE = 0.05                  # Minimum 5% overlap with missed lesion
SEEDS = [999, 4242, 7, 123, 2024]
STRUCT = ndimage.generate_binary_structure(3, 1)

FEATURE_NAMES = [
    # Group A: MRD scores
    'mean_r', 'median_r', 'max_r', 'pct90_r', 'pct95_r', 'frac_above_tau_r',
    # Group B: Production scores & Disagreement
    'mean_p', 'median_p', 'max_p', 'min_p', 'frac_p_lt_01', 'frac_p_lt_025', 'frac_p_lt_05', 'disagreement',
    # Group C: Geometry
    'voxel_count', 'physical_vol', 'bbox_dx', 'bbox_dy', 'bbox_dz', 'compactness', 'surf_vol_ratio',
    # Group D: Spatial relationships to production
    'min_dist_to_prod_et', 'mean_dist_to_prod_et', 'min_dist_to_prod_wt', 'frac_inside_prod_wt', 'frac_inside_prod_tc', 'touching_prod_et'
]

FEATS_A0 = [14]                      # Size only
FEATS_A1 = [0, 2, 6, 8, 13]          # Confidence & disagreement
FEATS_A2 = list(range(len(FEATURE_NAMES))) # Full 27 features

MODELS_CONFIG = [
    ('A0_size_only', FEATS_A0),
    ('A1_confidence', FEATS_A1),
    ('A2_full', FEATS_A2),
]


# ---------------------------------------------------------------------------
# Boundary metrics (HD95 & ASSD)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# MRD readout fit (Identical to E286-E289)
# ---------------------------------------------------------------------------
def fit_soft_target(X, y_soft, max_iter=500, C=1.0):
    X2 = np.concatenate([X, X], axis=0)
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w2 = np.concatenate([y_soft, 1.0 - y_soft])
    keep = w2 > 1e-12
    X2, y2, w2 = X2[keep], y2[keep], w2[keep]
    clf = LogisticRegression(max_iter=max_iter, C=C, class_weight='balanced')
    clf.fit(X2, y2, sample_weight=w2)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])


def fit_mrd_readout(model, ds, sid_to_idx, det_lesions, bg_sids_set, seeds, dev, t0):
    conv_w = model.seg_head[0].weight.detach().cpu().numpy()
    conv_b = model.seg_head[0].bias.detach().cpu().numpy()
    w_p = conv_w[ET].reshape(32).astype(np.float64)
    b_p = float(conv_b[ET])

    cache_path = HERE / 'E290_mrd_cache.pt'
    fitted = {}
    if cache_path.exists():
        try:
            cached_data = torch.load(cache_path, map_location='cpu', weights_only=False)
            if all(s in cached_data for s in seeds):
                print(f'  Loaded fitted MRD weights for seeds {seeds} from {cache_path.name}', flush=True)
                return {s: cached_data[s] for s in seeds}, w_p, b_p
            else:
                fitted.update(cached_data)
        except Exception:
            pass

    det_lesions_fit = [r for r in det_lesions if r['subject_id'] in bg_sids_set]
    for seed in seeds:
        if seed in fitted:
            continue
        rng = np.random.default_rng(seed)
        lesion_pool, neg_pool = [], []
        for r in det_lesions_fit:
            res = extract_lesion_shell(
                model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
            if res is None:
                continue
            lf, sf, _ = res
            if len(lf) > MAX_VOX_PER_LESION:
                lf = lf[rng.choice(len(lf), MAX_VOX_PER_LESION, replace=False)]
            if len(sf) > MAX_VOX_PER_LESION:
                sf = sf[rng.choice(len(sf), MAX_VOX_PER_LESION, replace=False)]
            lesion_pool.append(lf)
            neg_pool.append(sf)
        for sid in sorted(bg_sids_set):
            feats = extract_distant_background(model, ds, sid_to_idx, sid, dev, rng)
            if feats is None:
                continue
            neg_pool.append(feats)
        lesion_pool = np.concatenate(lesion_pool).astype(np.float64)
        neg_pool = np.concatenate(neg_pool).astype(np.float64)
        X = np.concatenate([lesion_pool, neg_pool])
        y_h = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
        p_i = 1.0 / (1.0 + np.exp(-(X @ w_p + b_p)))
        t2 = np.clip((y_h - p_i) / (1.0 - p_i + EPS_MARG), 0.0, 1.0)
        w_r, b_r = fit_soft_target(X, t2)
        fitted[seed] = (w_r, b_r)
        print(f'  seed={seed} fit ({time.time()-t0:.0f}s)', flush=True)

    try:
        torch.save(fitted, cache_path)
    except Exception:
        pass
    return {s: fitted[s] for s in seeds}, w_p, b_p


# ---------------------------------------------------------------------------
# Streaming Feature Extraction Helpers
# ---------------------------------------------------------------------------
def extract_27_features(comp_mask, p_et, p_tc, p_wt, r_map, edt_et, edt_wt, tau_r=TAU_R_BASE):
    """Extracts all 27 Group A, B, C, D features for a single connected component."""
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


def compute_subject_maps(model, ds, sid_to_idx, sid, w_p, b_p, w_tc, b_tc, w_wt, b_wt, fitted_mrd, dev):
    """Computes production and MRD score maps for one subject with immediate memory reclamation."""
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C_, D, H, W = d1.shape
        flat = d1.reshape(C_, -1).T
        
        # Production ET, TC, WT heads
        w_prod_stack = np.stack([w_p, w_tc, w_wt], axis=1) # (32, 3)
        b_prod_stack = np.array([b_p, b_tc, b_wt], dtype=np.float32)
        W_p_t = torch.from_numpy(w_prod_stack).float().to(dev)
        B_p_t = torch.from_numpy(b_prod_stack).float().to(dev)
        prod_probs = torch.sigmoid(flat @ W_p_t + B_p_t).cpu().numpy()
        del W_p_t, B_p_t

        p_et = prod_probs[:, 0].reshape(D, H, W)
        p_tc = prod_probs[:, 1].reshape(D, H, W)
        p_wt = prod_probs[:, 2].reshape(D, H, W)
        del prod_probs

        r_maps = {}
        for seed, (w_r, b_r) in fitted_mrd.items():
            w_r_t = torch.from_numpy(w_r).float().to(dev)
            r_map = torch.sigmoid(flat @ w_r_t + float(b_r)).reshape(D, H, W).cpu().numpy()
            del w_r_t
            r_maps[seed] = r_map

    del d1, flat, img_t, stages
    torch.cuda.empty_cache()
    return p_et, p_tc, p_wt, r_maps, tgt_c


# ---------------------------------------------------------------------------
# Candidate Component Record Extraction with Frozen Missed-Lesion Target
# ---------------------------------------------------------------------------
def collect_cohort_components(model, ds, sid_to_idx, subjects, fitted_mrd,
                              w_p, b_p, w_tc, b_tc, w_wt, b_wt, dev,
                              tau_r=TAU_R_BASE, delta=DELTA_RESCUE, desc='', cache_file=None):
    if cache_file and Path(cache_file).exists():
        try:
            records = torch.load(cache_file, map_location='cpu', weights_only=False)
            print(f'  Loaded {desc} components ({len(records)} records) from {Path(cache_file).name}', flush=True)
            return records
        except Exception:
            pass

    t0 = time.time()
    records = []
    n = len(subjects)
    for i, sid in enumerate(subjects):
        if sid not in sid_to_idx:
            continue
        p_et, p_tc, p_wt, r_maps, tgt_c = compute_subject_maps(
            model, ds, sid_to_idx, sid, w_p, b_p, w_tc, b_tc, w_wt, b_wt, fitted_mrd, dev)
        
        true_et = tgt_c[ET] > 0.5
        prod_et = p_et >= TAU_P
        prod_wt = p_wt >= TAU_P
        edt_et = ndimage.distance_transform_edt(~prod_et) if prod_et.any() else None
        edt_wt = ndimage.distance_transform_edt(~prod_wt) if prod_wt.any() else None

        # Ground truth ET missed lesions
        et_lbl_gt, n_et_gt = ndimage.label(true_et, structure=STRUCT)
        missed_lesions = []
        for lid in range(1, n_et_gt + 1):
            lmask = et_lbl_gt == lid
            if lmask.sum() < MIN_VOX:
                continue
            if not (lmask & prod_et).any():
                missed_lesions.append(lmask)

        for seed, r_map in r_maps.items():
            uncertain = ~prod_et
            mrd_pos = (r_map > tau_r) & uncertain
            cc_lbl, cc_n = ndimage.label(mrd_pos, structure=STRUCT)

            for cid in range(1, cc_n + 1):
                comp_mask = cc_lbl == cid
                feat = extract_27_features(comp_mask, p_et, p_tc, p_wt, r_map, edt_et, edt_wt, tau_r)
                
                label = 0
                for ml in missed_lesions:
                    ml_vol = ml.sum()
                    overlap = (comp_mask & ml).sum()
                    if ml_vol > 0 and (overlap / ml_vol) >= delta:
                        label = 1
                        break
                
                any_gt_overlap = int((comp_mask & true_et).any())
                records.append({
                    'sid': sid, 'seed': seed, 'cid': cid,
                    'feat': feat, 'label': label,
                    'any_gt_overlap': any_gt_overlap,
                    'vox_count': int(feat[14])
                })

        del p_et, p_tc, p_wt, r_maps, tgt_c, true_et, prod_et, prod_wt, edt_et, edt_wt
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f'  {desc} {i+1}/{n} subjects processed ({time.time()-t0:.0f}s)', flush=True)

    if cache_file:
        try:
            torch.save(records, cache_file)
        except Exception:
            pass
    return records


# ---------------------------------------------------------------------------
# Model Training Helpers
# ---------------------------------------------------------------------------
def fit_logistic_acceptance(records, findices, seed=0):
    X = np.stack([r['feat'][findices] for r in records])
    y = np.array([r['label'] for r in records], dtype=int)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=seed))
    ])
    pipe.fit(X, y)
    return pipe


# ---------------------------------------------------------------------------
# Unified Streaming Multi-Condition Evaluator (Memory-Safe O(1))
# ---------------------------------------------------------------------------
def evaluate_conditions_unified(model, ds, sid_to_idx, subjects, fitted_mrd,
                                w_p, b_p, w_tc, b_tc, w_wt, b_wt, dev,
                                conditions, tau_r=TAU_R_BASE, compute_boundary=False, desc=''):
    """
    Evaluates multiple conditions in a SINGLE streaming pass over subjects.
    Memory footprint: strictly O(1) subjects in memory at any time.
    conditions: list of (condition_name, accept_fn)
    """
    n_subjects = len(subjects)
    cond_names = [c[0] for c in conditions]
    
    stats = {cname: {
        'tp_et': 0, 'fp_et': 0, 'fn_et': 0, 'tn_et': 0,
        'tp_tc': 0, 'fp_tc': 0, 'fn_tc': 0,
        'tp_wt': 0, 'fp_wt': 0, 'fn_wt': 0,
        'g1_recovered': [], 'g1_total': 0,
        'fp_vox_total': 0, 'fp_comp_total': 0.0,
        'hd95_vals': [], 'assd_vals': []
    } for cname in cond_names}

    t0 = time.time()
    for i, sid in enumerate(subjects):
        if sid not in sid_to_idx:
            continue
        p_et, p_tc, p_wt, r_maps, tgt_c = compute_subject_maps(
            model, ds, sid_to_idx, sid, w_p, b_p, w_tc, b_tc, w_wt, b_wt, fitted_mrd, dev)
        
        true_et = tgt_c[ET] > 0.5
        true_tc = tgt_c[TC] > 0.5
        true_wt = tgt_c[WT] > 0.5

        # Reference G1 lesions using seed 999
        p_et_ref = p_et
        et_lbl_gt, n_et_gt = ndimage.label(true_et, structure=STRUCT)
        g1_lesion_masks = []
        for lid in range(1, n_et_gt + 1):
            lmask = et_lbl_gt == lid
            if lmask.sum() < MIN_VOX:
                continue
            if p_et_ref[lmask].max() < TAU_LOW:
                g1_lesion_masks.append(lmask)

        prod_et = p_et >= TAU_P
        prod_tc = p_tc >= TAU_P
        prod_wt = p_wt >= TAU_P

        edt_et = ndimage.distance_transform_edt(~prod_et) if prod_et.any() else None
        edt_wt = ndimage.distance_transform_edt(~prod_wt) if prod_wt.any() else None

        # Extract components once per seed for this subject
        seed_comp_data = {}
        for seed, r_map in r_maps.items():
            uncertain = ~prod_et
            mrd_pos = (r_map > tau_r) & uncertain
            cc_lbl, cc_n = ndimage.label(mrd_pos, structure=STRUCT)
            feats = []
            for cid in range(1, cc_n + 1):
                cmask = cc_lbl == cid
                feat = extract_27_features(cmask, p_et, p_tc, p_wt, r_map, edt_et, edt_wt, tau_r)
                feats.append(feat)
            seed_comp_data[seed] = (cc_lbl.astype(np.int16), feats)

        # Evaluate all conditions for this subject
        for cname, accept_fn in conditions:
            st = stats[cname]
            st['g1_total'] += len(g1_lesion_masks)
            
            seed_et_preds = []
            seed_tc_preds = []
            seed_wt_preds = []
            subj_fp_comps = 0

            for seed in fitted_mrd.keys():
                cc_lbl, feats = seed_comp_data[seed]
                if feats:
                    accepted_cids = [cid for cid, feat in enumerate(feats, 1) if accept_fn(feat)]
                    if accepted_cids:
                        admitted = np.isin(cc_lbl, accepted_cids)
                        for cid in accepted_cids:
                            if not (true_et[cc_lbl == cid].any()):
                                subj_fp_comps += 1
                    else:
                        admitted = np.zeros_like(prod_et)
                else:
                    admitted = np.zeros_like(prod_et)

                pred_et = prod_et | admitted
                pred_tc = prod_tc | pred_et
                pred_wt = prod_wt | pred_et

                seed_et_preds.append(pred_et)
                seed_tc_preds.append(pred_tc)
                seed_wt_preds.append(pred_wt)

            maj_threshold = len(fitted_mrd) / 2.0
            final_et = (np.stack(seed_et_preds).sum(axis=0) >= maj_threshold)
            final_tc = (np.stack(seed_tc_preds).sum(axis=0) >= maj_threshold)
            final_wt = (np.stack(seed_wt_preds).sum(axis=0) >= maj_threshold)

            st['tp_et'] += int((final_et & true_et).sum())
            st['fp_et'] += int((final_et & ~true_et).sum())
            st['fn_et'] += int((~final_et & true_et).sum())
            st['tn_et'] += int((~final_et & ~true_et).sum())

            st['tp_tc'] += int((final_tc & true_tc).sum())
            st['fp_tc'] += int((final_tc & ~true_tc).sum())
            st['fn_tc'] += int((~final_tc & true_tc).sum())

            st['tp_wt'] += int((final_wt & true_wt).sum())
            st['fp_wt'] += int((final_wt & ~true_wt).sum())
            st['fn_wt'] += int((~final_wt & true_wt).sum())

            st['fp_vox_total'] += int((final_et & ~true_et).sum())
            st['fp_comp_total'] += (subj_fp_comps / len(fitted_mrd))

            for lmask in g1_lesion_masks:
                st['g1_recovered'].append(int((final_et & lmask).any()))

            if compute_boundary:
                hd, assd = hd95_and_assd(final_et, true_et)
                if not np.isnan(hd):
                    st['hd95_vals'].append(hd)
                if not np.isnan(assd):
                    st['assd_vals'].append(assd)

        del p_et, p_tc, p_wt, r_maps, tgt_c, true_et, true_tc, true_wt, seed_comp_data
        gc.collect()

        if (i + 1) % 10 == 0 or (i + 1) == n_subjects:
            print(f'  {desc} {i+1}/{n_subjects} subjects processed ({time.time()-t0:.0f}s)', flush=True)

    results = {}
    for cname in cond_names:
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
    print('E290: LEARNED MRD CANDIDATE ACCEPTANCE (PRE-REGISTERED)')
    print('=' * 80, flush=True)

    # Load frozen model & weights
    model = load_model(dev)
    conv_w = model.seg_head[0].weight.detach().cpu().numpy()
    conv_b = model.seg_head[0].bias.detach().cpu().numpy()
    w_p, b_p = conv_w[ET].reshape(32).astype(np.float64), float(conv_b[ET])
    w_tc, b_tc = conv_w[TC].reshape(32).astype(np.float64), float(conv_b[TC])
    w_wt, b_wt = conv_w[WT].reshape(32).astype(np.float64), float(conv_b[WT])

    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    ds_dev = BraTSMultimodalDataset(
        str(ROOT / 'Dataset' / 'Training'), 'train',
        val_split=0.1, patch_size=PATCH)
    dev_sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds_dev.subject_dirs)}

    bg_sids_set = splits['det_train'] | splits['g2a_train']
    seeds = SEEDS[:2] if smoke else SEEDS

    # -----------------------------------------------------------------------
    # Step 1: Fit MRD-T2 (Frozen protocol)
    # -----------------------------------------------------------------------
    print('\n[STEP 1] Fitting MRD-T2 readout (frozen protocol)...', flush=True)
    fitted_mrd, _, _ = fit_mrd_readout(
        model, ds_dev, dev_sid_to_idx, det_lesions, bg_sids_set, seeds, dev, t_start)

    # -----------------------------------------------------------------------
    # Step 2: Split Development Cohort (Section 7)
    # -----------------------------------------------------------------------
    print('\n[STEP 2] Splitting Development Cohort (Section 7)...', flush=True)
    all_dev_sids = sorted(splits['det_test'])
    if smoke:
        all_dev_sids = all_dev_sids[:15]
    
    rng = np.random.default_rng(290)
    shuffled_dev = rng.permutation(all_dev_sids)
    n_dev = len(shuffled_dev)
    n_train = int(n_dev * 0.6)
    n_val = int(n_dev * 0.2)

    dev_train_sids = list(shuffled_dev[:n_train])
    dev_val_sids = list(shuffled_dev[n_train:n_train + n_val])
    dev_holdout_sids = list(shuffled_dev[n_train + n_val:])

    print(f'  Development cohort size: {n_dev} subjects')
    print(f'  Acceptance-Train:        {len(dev_train_sids)} subjects')
    print(f'  Acceptance-Validation:   {len(dev_val_sids)} subjects (threshold tuning)')
    print(f'  Development-Holdout:     {len(dev_holdout_sids)} subjects (pre-locked check)')

    # -----------------------------------------------------------------------
    # Step 3: Extract Candidate Components & Inference-Time Features
    # -----------------------------------------------------------------------
    print('\n[STEP 3] Extracting candidate components on Acceptance-Train...', flush=True)
    train_cache = HERE / f'E290_train_records{sfx}.pt'
    train_records = collect_cohort_components(
        model, ds_dev, dev_sid_to_idx, dev_train_sids, fitted_mrd,
        w_p, b_p, w_tc, b_tc, w_wt, b_wt, dev, tau_r=TAU_R_BASE, desc='[Dev-Train]', cache_file=train_cache)
    
    n_tp_train = sum(r['label'] for r in train_records)
    n_fp_train = len(train_records) - n_tp_train
    print(f'  Dev-Train components: {len(train_records)} total ({n_tp_train} True Rescues, {n_fp_train} False Positives)', flush=True)

    print('\nExtracting candidate components on Acceptance-Validation...', flush=True)
    val_cache = HERE / f'E290_val_records{sfx}.pt'
    val_records = collect_cohort_components(
        model, ds_dev, dev_sid_to_idx, dev_val_sids, fitted_mrd,
        w_p, b_p, w_tc, b_tc, w_wt, b_wt, dev, tau_r=TAU_R_BASE, desc='[Dev-Val]', cache_file=val_cache)
    
    n_tp_val = sum(r['label'] for r in val_records)
    n_fp_val = len(val_records) - n_tp_val
    print(f'  Dev-Val components: {len(val_records)} total ({n_tp_val} True Rescues, {n_fp_val} False Positives)', flush=True)

    # -----------------------------------------------------------------------
    # Step 4: Fit Pre-Registered Logistic Acceptance Models (Section 8 & 9)
    # -----------------------------------------------------------------------
    print('\n[STEP 4] Fitting Pre-Registered Logistic Acceptance Models (Section 8 & 9)...', flush=True)
    models = {}
    X_val_all = np.stack([r['feat'] for r in val_records])
    y_val_all = np.array([r['label'] for r in val_records], dtype=int)

    for mname, findices in MODELS_CONFIG:
        pipe = fit_logistic_acceptance(train_records, findices, seed=290)
        models[mname] = (pipe, findices)
        
        if y_val_all.sum() > 0 and (1 - y_val_all).sum() > 0:
            val_probs = pipe.predict_proba(X_val_all[:, findices])[:, 1]
            auc = roc_auc_score(y_val_all, val_probs)
        else:
            auc = float('nan')
        print(f'  Model {mname:15s}: Dev-Val Component AUC = {auc:.4f}')

    # -----------------------------------------------------------------------
    # Step 5: Section 16 Ablation -- Shuffled Labels Test on Dev
    # -----------------------------------------------------------------------
    print('\n[STEP 5] Label-Shuffle Ablation (Section 16)...', flush=True)
    shuffled_train_records = [dict(r) for r in train_records]
    y_shuffled = rng.permutation([r['label'] for r in shuffled_train_records])
    for r, ys in zip(shuffled_train_records, y_shuffled):
        r['label'] = ys

    shuffled_pipe = fit_logistic_acceptance(shuffled_train_records, FEATS_A2, seed=290)
    if y_val_all.sum() > 0 and (1 - y_val_all).sum() > 0:
        val_probs_shuf = shuffled_pipe.predict_proba(X_val_all[:, FEATS_A2])[:, 1]
        auc_shuf = roc_auc_score(y_val_all, val_probs_shuf)
    else:
        auc_shuf = float('nan')
    print(f'  A2 with Shuffled Labels: Dev-Val Component AUC = {auc_shuf:.4f} (expected ~0.50)')

    # -----------------------------------------------------------------------
    # Step 6: Dev-Validation Sweep & Selection Rule (Section 11)
    # -----------------------------------------------------------------------
    print('\n[STEP 6] Sweeping Acceptance Thresholds on Dev-Validation (Section 11)...', flush=True)
    
    def accept_none(f): return False
    def accept_all(f): return True
    def accept_gate_e289(f): return bool(f[2] > E289A_GATE_TAU_R)

    threshold_grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    dev_conditions = [
        ('prod_dev_val', accept_none),
        ('raw_dev_val', accept_all),
        ('gate_dev_val', accept_gate_e289),
    ]

    for mname, (pipe, findices) in models.items():
        for tau in threshold_grid:
            def make_fn(p=pipe, fi=findices, t=tau):
                return lambda feat: bool(p.predict_proba(feat[fi].reshape(1, -1))[0, 1] >= t)
            dev_conditions.append((f'{mname}_tau{tau:.2f}', make_fn()))

    dev_val_res_all = evaluate_conditions_unified(
        model, ds_dev, dev_sid_to_idx, dev_val_sids, fitted_mrd,
        w_p, b_p, w_tc, b_tc, w_wt, b_wt, dev, dev_conditions, desc='[Dev-Val-Sweep]')

    dice_p_val = dev_val_res_all['prod_dev_val']['dice_et']
    print(f'  Dev-Val Production ET Dice: {dice_p_val:.4f}')
    print(f'  Dev-Val Raw MRD ET Dice:    {dev_val_res_all["raw_dev_val"]["dice_et"]:.4f}  (G1={dev_val_res_all["raw_dev_val"]["g1_recovery_pct"]:.1f}%)')
    print(f'  Dev-Val Fixed Gate ET Dice: {dev_val_res_all["gate_dev_val"]["dice_et"]:.4f} (G1={dev_val_res_all["gate_dev_val"]["g1_recovery_pct"]:.1f}%)')

    dev_sweep_rows = []
    selected_thresholds = {}

    for mname in ['A0_size_only', 'A1_confidence', 'A2_full']:
        best_tau = None
        best_g1 = -1.0
        best_dice = -1.0

        for tau in threshold_grid:
            cname = f'{mname}_tau{tau:.2f}'
            res = dev_val_res_all[cname]
            d_et = res['dice_et']
            g1_pct = res['g1_recovery_pct']
            qualifies = (d_et >= (dice_p_val - 0.001))

            dev_sweep_rows.append({
                'model': mname, 'threshold': tau,
                'dice_et': d_et, 'delta_dice': d_et - dice_p_val,
                'g1_recovery_pct': g1_pct, 'qualifies': qualifies,
                'fp_comps_per_subj': res['fp_comps_per_subj']
            })

            if qualifies:
                if (g1_pct > best_g1) or (g1_pct == best_g1 and d_et > best_dice):
                    best_g1 = g1_pct
                    best_dice = d_et
                    best_tau = tau

        if best_tau is None:
            m_rows = [r for r in dev_sweep_rows if r['model'] == mname]
            best_r = max(m_rows, key=lambda r: (r['delta_dice'], r['g1_recovery_pct']))
            best_tau = best_r['threshold']
            selected_thresholds[mname] = (best_tau, False, best_r['dice_et'], best_r['g1_recovery_pct'])
            print(f'  {mname:15s}: Fallback tau={best_tau:.2f} (Dice={best_r["dice_et"]:.4f}, G1={best_r["g1_recovery_pct"]:.1f}%)')
        else:
            selected_thresholds[mname] = (best_tau, True, best_dice, best_g1)
            print(f'  {mname:15s}: SELECTED tau={best_tau:.2f} (Dice={best_dice:.4f}, G1={best_g1:.1f}%)')

    sweep_path = HERE / f'E290A_dev_sweep{sfx}.csv'
    with open(sweep_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dev_sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(dev_sweep_rows)
    print(f'  Wrote {sweep_path}', flush=True)

    # -----------------------------------------------------------------------
    # Step 7: Descriptive Feature Analysis (Section 15)
    # -----------------------------------------------------------------------
    print('\n[STEP 7] Descriptive Feature Analysis (Section 15)...', flush=True)
    all_dev_records = train_records + val_records
    tp_recs = [r for r in all_dev_records if r['label'] == 1]
    fp_recs = [r for r in all_dev_records if r['label'] == 0]

    feature_analysis_rows = []
    key_features = [
        ('mean_r', 0), ('max_r', 2), ('mean_p', 6), ('disagreement', 13),
        ('voxel_count', 14), ('compactness', 19), ('min_dist_to_prod_et', 21),
        ('frac_inside_prod_wt', 24)
    ]
    print(f'  Comparing {len(tp_recs)} True Missed Rescues vs {len(fp_recs)} False Candidates:')
    for fname, idx in key_features:
        tp_vals = np.array([r['feat'][idx] for r in tp_recs]) if tp_recs else np.array([0.0])
        fp_vals = np.array([r['feat'][idx] for r in fp_recs]) if fp_recs else np.array([0.0])
        
        m_tp, s_tp = tp_vals.mean(), tp_vals.std()
        m_fp, s_fp = fp_vals.mean(), fp_vals.std()
        pooled_std = np.sqrt(0.5 * (s_tp**2 + s_fp**2)) + 1e-8
        cohen_d = (m_tp - m_fp) / pooled_std
        
        print(f'    {fname:20s}: True Rescue = {m_tp:8.4f} +/- {s_tp:6.4f} | False Cand = {m_fp:8.4f} +/- {s_fp:6.4f} | Cohen d = {cohen_d:+6.2f}')
        feature_analysis_rows.append({
            'feature': fname, 'mean_true_rescue': m_tp, 'std_true_rescue': s_tp,
            'mean_false_candidate': m_fp, 'std_false_candidate': s_fp, 'cohens_d': cohen_d
        })

    feat_path = HERE / f'E290_feature_analysis{sfx}.csv'
    with open(feat_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(feature_analysis_rows[0].keys()))
        w.writeheader()
        w.writerows(feature_analysis_rows)
    print(f'  Wrote {feat_path}', flush=True)

    # -----------------------------------------------------------------------
    # Step 8: Locked Test Cohort Setup & Leakage Audit (Section 7 & 12)
    # -----------------------------------------------------------------------
    print('\n[STEP 8] Preparing Locked Test Cohort (Stage B)...', flush=True)
    all_prior_touched = (set(r['subject_id'] for r in det_lesions) |
                          set(r['subject_id'] for r in g2a_lesions) |
                          set(r['subject_id'] for r in g2b_lesions))
    
    ds_val_pop = BraTSMultimodalDataset(
        str(ROOT / 'Dataset' / 'Training'), 'val',
        val_split=0.1, patch_size=PATCH)
    val_sids_all = sorted(os.path.basename(d) for d in ds_val_pop.subject_dirs)
    val_sids_fresh = sorted(set(val_sids_all) - all_prior_touched)

    ds_full = BraTSMultimodalDataset(
        str(ROOT / 'Dataset' / 'Training'), 'train',
        val_split=0.0, patch_size=PATCH)
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

    leakage_dev = set(test_subjects) & set(all_dev_sids)
    leakage_train = set(test_subjects) & bg_sids_set
    assert len(leakage_dev) == 0 and len(leakage_train) == 0, 'LEAKAGE AUDIT FAILED'
    print(f'  Locked Test Cohort: {len(test_subjects)} subjects. Leakage Audit: PASS!', flush=True)

    # -----------------------------------------------------------------------
    # Step 9: Single-Touch Locked Test Evaluation (Section 10 & 12)
    # -----------------------------------------------------------------------
    print('\n[STEP 9] Running Single-Touch Locked Test Evaluation (Section 10 & 12)...', flush=True)
    
    conditions = [
        ('B0_production_only', accept_none),
        ('B1_raw_mrd', accept_all),
        ('B2_fixed_gate_E289', accept_gate_e289),
    ]

    for mname in ['A0_size_only', 'A1_confidence', 'A2_full']:
        pipe, findices = models[mname]
        tau_star = selected_thresholds[mname][0]
        def make_locked_fn(p=pipe, fi=findices, t=tau_star):
            return lambda feat: bool(p.predict_proba(feat[fi].reshape(1, -1))[0, 1] >= t)
        
        b_label = f'B{len(conditions)}_{mname}_tau{tau_star:.2f}'
        conditions.append((b_label, make_locked_fn()))

    # Run single-touch streaming evaluation
    locked_results = evaluate_conditions_unified(
        model, ds_full, full_sid_to_idx, test_subjects, fitted_mrd,
        w_p, b_p, w_tc, b_tc, w_wt, b_wt, dev, conditions, compute_boundary=True, desc='[Locked-Test]')

    prod_test_res = locked_results['B0_production_only']
    test_rows = []

    for cname, _ in conditions:
        res = locked_results[cname]
        if cname == 'B0_production_only':
            delta_dice = 0.0
            delta_rec = 0.0
            delta_prec = 0.0
            delta_fp_vox = 0.0
            delta_g1 = 0.0
        else:
            delta_dice = res['dice_et'] - prod_test_res['dice_et']
            delta_rec = res['recall'] - prod_test_res['recall']
            delta_prec = res['precision'] - prod_test_res['precision']
            delta_fp_vox = res['total_fp_voxels'] - prod_test_res['total_fp_voxels']
            delta_g1 = res['g1_recovery_pct'] - prod_test_res['g1_recovery_pct']

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
        print(f'  {cname:28s}: ET Dice={res["dice_et"]:.4f} (dDice={delta_dice:+.5f}) | G1={res["g1_recovery_pct"]:5.1f}% | HD95={res["hd95"]:.2f}', flush=True)

    test_path = HERE / f'E290B_locked_test{sfx}.csv'
    with open(test_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
        w.writeheader()
        w.writerows(test_rows)
    print(f'  Wrote {test_path}', flush=True)

    # -----------------------------------------------------------------------
    # Step 10: Generate Figures (Sections 13 & 14)
    # -----------------------------------------------------------------------
    print('\n[STEP 10] Generating Figures (Sections 13 & 14)...', flush=True)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = {'A0_size_only': 'blue', 'A1_confidence': 'green', 'A2_full': 'purple'}

    for mname in ['A0_size_only', 'A1_confidence', 'A2_full']:
        m_rows = sorted([r for r in dev_sweep_rows if r['model'] == mname], key=lambda r: r['threshold'])
        dices = [r['dice_et'] for r in m_rows]
        g1s = [r['g1_recovery_pct'] for r in m_rows]
        ax.plot(g1s, dices, label=f'Dev Curve: {mname}', color=colors[mname], linestyle='--', alpha=0.5)

    for r in test_rows:
        c = r['condition']
        if 'production_only' in c:
            ax.scatter(r['g1_recovery_pct'], r['dice_et'], color='black', s=120, zorder=5, label='B0: Production Only', marker='*')
        elif 'raw_mrd' in c:
            ax.scatter(r['g1_recovery_pct'], r['dice_et'], color='red', s=100, zorder=5, label='B1: Raw MRD', marker='x')
        elif 'fixed_gate' in c:
            ax.scatter(r['g1_recovery_pct'], r['dice_et'], color='orange', s=100, zorder=5, label='B2: E289 Fixed Gate', marker='D')
        elif 'A0_size_only' in c:
            ax.scatter(r['g1_recovery_pct'], r['dice_et'], color='blue', s=110, zorder=5, label='B3: Learned A0 (Size)', marker='o')
        elif 'A1_confidence' in c:
            ax.scatter(r['g1_recovery_pct'], r['dice_et'], color='green', s=110, zorder=5, label='B4: Learned A1 (Conf)', marker='s')
        elif 'A2_full' in c:
            ax.scatter(r['g1_recovery_pct'], r['dice_et'], color='purple', s=120, zorder=5, label='B5: Learned A2 (Full)', marker='^')

    ax.set_xlabel('G1 Missed-Lesion Recovery (%)', fontsize=12)
    ax.set_ylabel('ET Dice Score', fontsize=12)
    ax.set_title('E290: ET Dice vs G1 Recovery Operating Frontier', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.tight_layout()
    fig1_path = HERE / f'E290_fig1_dice_vs_g1{sfx}.png'
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f'  Saved Figure 1 to {fig1_path}')

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for mname in ['A0_size_only', 'A1_confidence', 'A2_full']:
        m_rows = sorted([r for r in dev_sweep_rows if r['model'] == mname], key=lambda r: r['threshold'])
        fps = [r['fp_comps_per_subj'] for r in m_rows]
        g1s = [r['g1_recovery_pct'] for r in m_rows]
        ax.plot(fps, g1s, label=f'Dev Curve: {mname}', color=colors[mname], linestyle='--', alpha=0.5)

    for r in test_rows:
        c = r['condition']
        if 'production_only' in c:
            ax.scatter(r['fp_comps_per_subj'], r['g1_recovery_pct'], color='black', s=120, zorder=5, label='B0: Production Only', marker='*')
        elif 'raw_mrd' in c:
            ax.scatter(r['fp_comps_per_subj'], r['g1_recovery_pct'], color='red', s=100, zorder=5, label='B1: Raw MRD', marker='x')
        elif 'fixed_gate' in c:
            ax.scatter(r['fp_comps_per_subj'], r['g1_recovery_pct'], color='orange', s=100, zorder=5, label='B2: E289 Fixed Gate', marker='D')
        elif 'A0_size_only' in c:
            ax.scatter(r['fp_comps_per_subj'], r['g1_recovery_pct'], color='blue', s=110, zorder=5, label='B3: Learned A0 (Size)', marker='o')
        elif 'A1_confidence' in c:
            ax.scatter(r['fp_comps_per_subj'], r['g1_recovery_pct'], color='green', s=110, zorder=5, label='B4: Learned A1 (Conf)', marker='s')
        elif 'A2_full' in c:
            ax.scatter(r['fp_comps_per_subj'], r['g1_recovery_pct'], color='purple', s=120, zorder=5, label='B5: Learned A2 (Full)', marker='^')

    ax.set_xlabel('False-Positive Components / Subject', fontsize=12)
    ax.set_ylabel('G1 Missed-Lesion Recovery (%)', fontsize=12)
    ax.set_title('E290: Candidate Quality -- G1 Recovery vs FP Components', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.tight_layout()
    fig2_path = HERE / f'E290_fig2_g1_vs_fp_components{sfx}.png'
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f'  Saved Figure 2 to {fig2_path}')

    # -----------------------------------------------------------------------
    # Final Verdict & Summary
    # -----------------------------------------------------------------------
    print('\n' + '=' * 80)
    print('E290 PRE-REGISTERED EXPERIMENT SUMMARY & VERDICT')
    print('=' * 80)
    prod_row = next(r for r in test_rows if 'production_only' in r['condition'])
    raw_row = next(r for r in test_rows if 'raw_mrd' in r['condition'])
    gate_row = next(r for r in test_rows if 'fixed_gate' in r['condition'])
    b3_row = next(r for r in test_rows if 'A0_size_only' in r['condition'])
    b4_row = next(r for r in test_rows if 'A1_confidence' in r['condition'])
    b5_row = next(r for r in test_rows if 'A2_full' in r['condition'])

    print(f'Production Alone (B0):  Dice={prod_row["dice_et"]:.4f}  Recall={prod_row["recall"]:.4f}  HD95={prod_row["hd95"]:.2f}')
    print(f'Raw MRD (B1):          Dice={raw_row["dice_et"]:.4f}  (delta={raw_row["delta_dice"]:+.5f})  G1={raw_row["g1_recovery_pct"]:.1f}%  FP_comp/subj={raw_row["fp_comps_per_subj"]:.1f}')
    print(f'Fixed Gate E289 (B2):  Dice={gate_row["dice_et"]:.4f} (delta={gate_row["delta_dice"]:+.5f})  G1={gate_row["g1_recovery_pct"]:.1f}%  FP_comp/subj={gate_row["fp_comps_per_subj"]:.1f}')
    print(f'Learned A0 Size (B3):  Dice={b3_row["dice_et"]:.4f}   (delta={b3_row["delta_dice"]:+.5f})  G1={b3_row["g1_recovery_pct"]:.1f}%  FP_comp/subj={b3_row["fp_comps_per_subj"]:.1f}')
    print(f'Learned A1 Conf (B4):  Dice={b4_row["dice_et"]:.4f}   (delta={b4_row["delta_dice"]:+.5f})  G1={b4_row["g1_recovery_pct"]:.1f}%  FP_comp/subj={b4_row["fp_comps_per_subj"]:.1f}')
    print(f'Learned A2 Full (B5):  Dice={b5_row["dice_et"]:.4f}   (delta={b5_row["delta_dice"]:+.5f})  G1={b5_row["g1_recovery_pct"]:.1f}%  FP_comp/subj={b5_row["fp_comps_per_subj"]:.1f}')

    summary_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'smoke': smoke,
        'selected_thresholds': {k: float(v[0]) for k, v in selected_thresholds.items()},
        'dev_auc': {
            'A0_size_only': float(roc_auc_score(y_val_all, models['A0_size_only'][0].predict_proba(X_val_all[:, FEATS_A0])[:, 1])) if y_val_all.sum() > 0 else None,
            'A1_confidence': float(roc_auc_score(y_val_all, models['A1_confidence'][0].predict_proba(X_val_all[:, FEATS_A1])[:, 1])) if y_val_all.sum() > 0 else None,
            'A2_full': float(roc_auc_score(y_val_all, models['A2_full'][0].predict_proba(X_val_all[:, FEATS_A2])[:, 1])) if y_val_all.sum() > 0 else None,
            'A2_shuffled_ablation': float(auc_shuf)
        },
        'locked_test_results': test_rows
    }
    summary_json_path = HERE / f'E290_summary{sfx}.json'
    with open(summary_json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f'\nWrote experiment summary to {summary_json_path}')
    print(f'E290 finished in {time.time()-t_start:.1f}s.')


if __name__ == '__main__':
    main()
