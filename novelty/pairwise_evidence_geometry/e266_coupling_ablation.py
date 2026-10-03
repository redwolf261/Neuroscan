"""E266 -- coupling ablation, per the user's exact spec. Tests whether
the GAIN found in E264/E265 comes from the SPECIFIC INTERACTION of
(R1-B's recovery) and (Delta_C's evidence-consistency filter), rather
than from either half alone or from Delta_C being a generic filter that
helps any readout.

FOUR BRANCHES (same held-out TEST population, same Delta_C mechanics
throughout -- only the input probability field changes):
  A: P_prod  -> Delta_C      (does local validation help PRODUCTION alone?)
  B: D1 -> R1-B              (baseline recovery readout, no filter -- == E257-B/E264's no_filter)
  C: D1 -> R1-B -> Delta_C   (the proposed coupling -- == E264/E265's winning result)
  D: D1 -> R1-B-seed2 -> Delta_C   (does Delta_C help ANY similarly-trained recovery
                                     readout, or specifically R1-B's fit? R1-B-seed2 is
                                     refit IDENTICALLY to R1-B -- same features, same
                                     negatives, same hyperparameters -- but with a
                                     DIFFERENT rng seed for negative sampling, per
                                     explicit user choice this session)

R_alt DESIGN NOTE: the original spec's R_alt could not be production's
own w_P, because that would make Branch D identical to Branch A
(P_prod->Delta_C twice), collapsing the intended comparison. Flagged to
and resolved with the user: R_alt = R1-B refit with an independent rng
seed (999 -> 4242) for negative sampling, keeping every other choice
(features, MAX_VOX_PER_LESION cap, class_weight='balanced', C=1.0)
identical to R1-B. This tests genericity (does Delta_C help "a similar
recovery readout" broadly) while keeping Branch D meaningfully distinct
from both A and B/C.

Delta_C threshold: for branches C and D, selected on VAL using the SAME
lesion-level zero-cost criterion as E264/E265 (independently for each
readout, since R1-B-seed2's probability field differs from R1-B's).
Branch A's Delta_C threshold is ALSO selected independently on VAL
using production's own probability field, for a fair per-field
comparison (not reusing R1-B's threshold on a different field).

METRICS (per explicit user list, all on the SAME 158 held-out test
subjects): G2-A lesion recovery, whole-volume ET Dice, ET HD95, total
FP voxels, WT-not-TC FP voxels, distant-background (outside-all-tumor)
FP voxels, lesion/component retention (== recovery, restated), and
recovery on detected_test + G2-B populations.

FINAL ANALYSIS (readout-disagreement comparison, per explicit user
spec): for voxels in 4 populations (recovered G2-A, detected ET,
WT-not-TC FP, distant-background FP), compute
Delta_readout(x) = z_R1B(x) - z_prod(x) (raw LOGIT difference, not
probability, to avoid the exact sigmoid-saturation confound E263 hit --
production logits are often deep in saturation where probability
differences vanish numerically even when logit differences are large
and real) and compare the distribution across the 4 populations. This
tests whether R1-B is exposing a SUPPRESSED evidence population
(large positive Delta_readout specifically at recovered-G2A voxels,
distinct from ordinary FP) rather than just being "a better classifier"
everywhere uniformly.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e257_common import (load_model, get_w_prod, load_populations, build_splits,  # noqa: E402
                         extract_lesion_shell, extract_distant_background,
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION, NEAR_DILATION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402

ET, TC, WT = 0, 1, 2
NEIGHBORHOOD_DILATION = 3
EPS = 1e-6
STRUCT = ndimage.generate_binary_structure(3, 1)


def dice_score(pred_mask, gt_mask):
    ps, gs = pred_mask.sum(), gt_mask.sum()
    if gs == 0:
        return 1.0 if ps == 0 else float('nan')
    return float(2 * (pred_mask & gt_mask).sum() / (ps + gs))


def hd95(pred_mask, gt_mask, spacing=1.0):
    """95th-percentile symmetric Hausdorff distance via EDT-based surface
    distances. Returns nan if either mask is empty (no surface to
    measure), matching standard BraTS convention for degenerate cases."""
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float('nan')
    pred_surf = pred_mask & ~ndimage.binary_erosion(pred_mask, structure=STRUCT)
    gt_surf = gt_mask & ~ndimage.binary_erosion(gt_mask, structure=STRUCT)
    dt_gt = ndimage.distance_transform_edt(~gt_mask, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~pred_mask, sampling=spacing)
    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    if len(all_d) == 0:
        return float('nan')
    return float(np.percentile(all_d, 95))


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    model = load_model(dev)
    w_et, b_et = get_w_prod(model)
    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    if smoke:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']][:30]
        g2a_lesions_val = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_val']][:15]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']][:15]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']][:15]
        g2b_lesions_eval = g2b_lesions[:15]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        g2a_lesions_val = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_val']]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']]
        g2b_lesions_eval = g2b_lesions
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    t0 = time.time()

    def fit_r1b(seed):
        rng = np.random.default_rng(seed)
        lesion_pool, neg_pool = [], []
        for r in det_lesions_fit:
            res = extract_lesion_shell(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
            if res is None:
                continue
            lf, sf, sz = res
            if len(lf) > MAX_VOX_PER_LESION:
                lf = lf[rng.choice(len(lf), MAX_VOX_PER_LESION, replace=False)]
            if len(sf) > MAX_VOX_PER_LESION:
                sf = sf[rng.choice(len(sf), MAX_VOX_PER_LESION, replace=False)]
            lesion_pool.append(lf); neg_pool.append(sf)
        for sid in bg_subjects:
            feats = extract_distant_background(model, ds, sid_to_idx, sid, dev, rng)
            if feats is None:
                continue
            neg_pool.append(feats)
        lesion_pool = np.concatenate(lesion_pool); neg_pool = np.concatenate(neg_pool)
        X = np.concatenate([lesion_pool, neg_pool])
        y = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
        clf = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X, y)
        w = torch.from_numpy(clf.coef_[0].astype(np.float64)).float().to(dev)
        b = float(clf.intercept_[0])
        return w, b

    print('Fitting R1-B (seed=999, identical to E257-B)...', flush=True)
    w_r1b, b_r1b = fit_r1b(999)
    print(f'R1-B fit ({time.time()-t0:.0f}s)', flush=True)
    print('Fitting R1-B-seed2 (seed=4242, same features/hyperparameters)...', flush=True)
    w_r1b2, b_r1b2 = fit_r1b(4242)
    print(f'R1-B-seed2 fit ({time.time()-t0:.0f}s)', flush=True)

    w_p_t = torch.from_numpy(w_et).float().to(dev)
    b_p = float(b_et)

    dense_cache = {}

    def dense_logits(sid):
        """Returns (l_prod, l_r1b, l_r1b2) logit maps + tgt_c + img_c, cached."""
        if sid in dense_cache:
            return dense_cache[sid]
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T
        with torch.no_grad():
            l_prod = (flat @ w_p_t + b_p).reshape(D, H, W).cpu().numpy()
            l_r1b = (flat @ w_r1b + b_r1b).reshape(D, H, W).cpu().numpy()
            l_r1b2 = (flat @ w_r1b2 + b_r1b2).reshape(D, H, W).cpu().numpy()
        result = (l_prod, l_r1b, l_r1b2, tgt_c, img_c)
        dense_cache[sid] = result
        return result

    def prob_map(field):
        return 1.0 / (1.0 + np.exp(-field))

    def component_table(sid, field_key):
        """field_key in {'prod','r1b','r1b2'}. Labels ALL positive
        components (p>TAU) of that field and computes Delta_C (mu_C-mu_N)
        against the SAME field's own neighborhood."""
        l_prod, l_r1b, l_r1b2, tgt_c, img_c = dense_logits(sid)
        field = {'prod': l_prod, 'r1b': l_r1b, 'r1b2': l_r1b2}[field_key]
        p_map = prob_map(field)
        brain_mask = img_c[0] != 0
        true_et = tgt_c[ET] > 0.5
        pos_mask = (p_map > TAU) & brain_mask
        lbl, n = ndimage.label(pos_mask, structure=STRUCT)
        rows = []
        for cid in range(1, n + 1):
            comp_mask = lbl == cid
            A_C = int(comp_mask.sum())
            if A_C < 1:
                continue
            is_tp = bool((comp_mask & true_et).any())
            mu_C = float(p_map[comp_mask].mean())
            dilated = ndimage.binary_dilation(comp_mask, structure=STRUCT,
                                              iterations=NEIGHBORHOOD_DILATION)
            shell = dilated & (~comp_mask) & (~true_et) & brain_mask
            n_shell = int(shell.sum())
            mu_N = float(p_map[shell].mean()) if n_shell > 0 else 0.0
            Delta_C = mu_C - mu_N
            rows.append({'subject_id': sid, 'component_id': cid, 'is_tp': int(is_tp),
                        'A_C': A_C, 'Delta_C': Delta_C})
        return rows, lbl, p_map, tgt_c, img_c

    comp_cache = {}

    def get_components(sid, field_key):
        key = (sid, field_key)
        if key not in comp_cache:
            comp_cache[key] = component_table(sid, field_key)
        return comp_cache[key]

    # ---- select Delta_C threshold on VAL, independently per field ----
    def per_lesion_max_delta(lesion_list, field_key):
        maxima = []
        for r in lesion_list:
            sid, cid = r['subject_id'], int(r['comp_id'])
            if sid not in sid_to_idx:
                continue
            rows, lbl, p_map, tgt_c, img_c = get_components(sid, field_key)
            et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
            if cid < 1 or cid > et_n:
                continue
            lesion_mask = et_lbl == cid
            if int(lesion_mask.sum()) < MIN_VOX:
                continue
            overlap_ids = set(np.unique(lbl[lesion_mask])) - {0}
            vals = [row['Delta_C'] for row in rows if row['component_id'] in overlap_ids]
            if vals:
                maxima.append(max(vals))
        return maxima

    def select_threshold(lesion_list, field_key):
        maxima = per_lesion_max_delta(lesion_list, field_key)
        return min(maxima) if maxima else -np.inf

    val_subjects_prod = sorted(set(r['subject_id'] for r in g2a_lesions_val))
    print(f'\nSelecting Delta_C thresholds on {len(val_subjects_prod)} VAL subjects '
         f'(per field)...', flush=True)
    for sid in val_subjects_prod:
        if sid in sid_to_idx:
            get_components(sid, 'prod'); get_components(sid, 'r1b'); get_components(sid, 'r1b2')
    thresh_prod = select_threshold(g2a_lesions_val, 'prod')
    thresh_r1b = select_threshold(g2a_lesions_val, 'r1b')
    thresh_r1b2 = select_threshold(g2a_lesions_val, 'r1b2')
    print(f'VAL thresholds: prod={thresh_prod:.4f} r1b={thresh_r1b:.4f} r1b2={thresh_r1b2:.4f} '
         f'({time.time()-t0:.0f}s)', flush=True)

    # ---- evaluate all 4 branches on TEST ----
    test_subjects = sorted(set(r['subject_id'] for r in det_lesions_test) |
                          set(r['subject_id'] for r in g2a_lesions_test) |
                          set(r['subject_id'] for r in g2b_lesions_eval))
    if smoke:
        test_subjects = test_subjects[:15]
    print(f'\nEvaluating 4 branches on {len(test_subjects)} TEST subjects...', flush=True)

    branches = {
        'A_prod_delta': ('prod', thresh_prod),
        'B_r1b_nofilter': ('r1b', None),
        'C_r1b_delta': ('r1b', thresh_r1b),
        'D_r1b2_delta': ('r1b2', thresh_r1b2),
    }

    def branch_mask(sid, field_key, threshold):
        rows, lbl, p_map, tgt_c, img_c = get_components(sid, field_key)
        if threshold is None:
            return (p_map > TAU) & (img_c[0] != 0), tgt_c, img_c
        keep_ids = {row['component_id'] for row in rows if row['Delta_C'] >= threshold}
        mask = np.isin(lbl, list(keep_ids)) if keep_ids else np.zeros_like(lbl, dtype=bool)
        return mask, tgt_c, img_c

    per_subject_dice = {b: [] for b in branches}
    per_subject_hd95 = {b: [] for b in branches}
    fp_total = {b: 0 for b in branches}
    fp_wt_not_tc = {b: 0 for b in branches}
    fp_distant = {b: 0 for b in branches}

    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        for bname, (field_key, thresh) in branches.items():
            mask, tgt_c, img_c = branch_mask(sid, field_key, thresh)
            true_et = tgt_c[ET] > 0.5
            true_tc = tgt_c[TC] > 0.5
            true_wt = tgt_c[WT] > 0.5
            brain_mask = img_c[0] != 0
            per_subject_dice[bname].append(dice_score(mask, true_et))
            per_subject_hd95[bname].append(hd95(mask, true_et))
            fp_mask = mask & brain_mask & (~true_et)
            fp_total[bname] += int(fp_mask.sum())
            fp_wt_not_tc[bname] += int((fp_mask & true_wt & (~true_tc)).sum())
            dilated = ndimage.binary_dilation(true_et, structure=STRUCT,
                                              iterations=NEAR_DILATION) if true_et.any() else np.zeros_like(true_et)
            local_shell = dilated & (~true_et) & brain_mask
            distant_bg = brain_mask & (~true_et) & (~local_shell)
            fp_distant[bname] += int((fp_mask & distant_bg).sum())
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    def lesion_recovery(lesion_list, field_key, threshold):
        recs = []
        for r in lesion_list:
            sid, cid = r['subject_id'], int(r['comp_id'])
            if sid not in sid_to_idx:
                continue
            _, _, _, tgt_c, img_c = get_components(sid, field_key)
            et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
            if cid < 1 or cid > et_n:
                continue
            lesion_mask = et_lbl == cid
            if int(lesion_mask.sum()) < MIN_VOX:
                continue
            mask, _, _ = branch_mask(sid, field_key, threshold)
            recs.append(int((mask & lesion_mask).any()))
        return float(np.mean(recs)) if recs else float('nan')

    print('\nComputing lesion-level recovery per branch/population...', flush=True)
    results = {}
    for bname, (field_key, thresh) in branches.items():
        results[bname] = {
            'g2a_recovery': lesion_recovery(g2a_lesions_test, field_key, thresh),
            'det_recovery': lesion_recovery(det_lesions_test, field_key, thresh),
            'g2b_recovery': lesion_recovery(g2b_lesions_eval, field_key, thresh),
            'et_dice_mean': float(np.nanmean(per_subject_dice[bname])),
            'et_hd95_median': float(np.nanmedian(per_subject_hd95[bname])),
            'fp_total': fp_total[bname],
            'fp_wt_not_tc': fp_wt_not_tc[bname],
            'fp_distant_bg': fp_distant[bname],
        }

    print(f'\n=== E266 TEST results ===')
    for k, v in results.items():
        print(f'{k}: {v}')

    out = HERE / ('E266_smoke.csv' if smoke else 'E266_results.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['branch', 'g2a_recovery', 'det_recovery', 'g2b_recovery',
                                           'et_dice_mean', 'et_hd95_median', 'fp_total',
                                           'fp_wt_not_tc', 'fp_distant_bg'])
        w.writeheader()
        for bname, v in results.items():
            w.writerow({'branch': bname, **v})

    # ---- FINAL ANALYSIS: Delta_readout(x) = z_R1B(x) - z_prod(x) across 4 voxel populations ----
    print('\nDelta_readout analysis across 4 populations (logit space, not probability,'
         ' to avoid E263-style saturation confound)...', flush=True)
    pop_deltas = {'recovered_G2A': [], 'detected_ET': [], 'WT_not_TC_FP': [], 'distant_bg_FP': []}

    for r in g2a_lesions_test:
        sid, cid = r['subject_id'], int(r['comp_id'])
        if sid not in sid_to_idx:
            continue
        l_prod, l_r1b, l_r1b2, tgt_c, img_c = dense_logits(sid)
        et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
        if cid < 1 or cid > et_n:
            continue
        lesion_mask = et_lbl == cid
        if int(lesion_mask.sum()) < MIN_VOX:
            continue
        r1b_pos = (prob_map(l_r1b) > TAU) & lesion_mask
        if r1b_pos.any():
            pop_deltas['recovered_G2A'].extend((l_r1b[r1b_pos] - l_prod[r1b_pos]).tolist())

    for r in det_lesions_test:
        sid, cid = r['subject_id'], int(r['comp_id'])
        if sid not in sid_to_idx:
            continue
        l_prod, l_r1b, l_r1b2, tgt_c, img_c = dense_logits(sid)
        et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
        if cid < 1 or cid > et_n:
            continue
        lesion_mask = et_lbl == cid
        if int(lesion_mask.sum()) < MIN_VOX:
            continue
        pop_deltas['detected_ET'].extend((l_r1b[lesion_mask] - l_prod[lesion_mask]).tolist())

    for sid in test_subjects:
        if sid not in sid_to_idx:
            continue
        rows, lbl, p_map_r1b, tgt_c, img_c = get_components(sid, 'r1b')
        l_prod, l_r1b, l_r1b2, _, _ = dense_logits(sid)
        true_wt = tgt_c[WT] > 0.5
        true_tc = tgt_c[TC] > 0.5
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        fp_mask_all = (p_map_r1b > TAU) & brain_mask & (~true_et)
        wt_not_tc_fp = fp_mask_all & true_wt & (~true_tc)
        dilated = ndimage.binary_dilation(true_et, structure=STRUCT,
                                          iterations=NEAR_DILATION) if true_et.any() else np.zeros_like(true_et)
        local_shell = dilated & (~true_et) & brain_mask
        distant_bg_mask = brain_mask & (~true_et) & (~local_shell)
        distant_bg_fp = fp_mask_all & distant_bg_mask
        if wt_not_tc_fp.any():
            pop_deltas['WT_not_TC_FP'].extend((l_r1b[wt_not_tc_fp] - l_prod[wt_not_tc_fp]).tolist())
        if distant_bg_fp.any():
            pop_deltas['distant_bg_FP'].extend((l_r1b[distant_bg_fp] - l_prod[distant_bg_fp]).tolist())

    delta_summary = {}
    for pop, vals in pop_deltas.items():
        arr = np.array(vals)
        delta_summary[pop] = {
            'n': len(arr),
            'mean': float(arr.mean()) if len(arr) else float('nan'),
            'median': float(np.median(arr)) if len(arr) else float('nan'),
            'std': float(arr.std()) if len(arr) else float('nan'),
        }
    print(f'\n=== Delta_readout summary ===')
    for pop, s in delta_summary.items():
        print(f'{pop}: n={s["n"]} mean={s["mean"]:.4f} median={s["median"]:.4f} std={s["std"]:.4f}')

    delta_out = HERE / ('E266_delta_readout_smoke.csv' if smoke else 'E266_delta_readout.csv')
    with open(delta_out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['population', 'n', 'mean', 'median', 'std'])
        for pop, s in delta_summary.items():
            w.writerow([pop, s['n'], s['mean'], s['median'], s['std']])

    print(f'\nE266 complete ({time.time()-t0:.0f}s). wrote {out.name}, {delta_out.name}', flush=True)


if __name__ == '__main__':
    main()
