"""E268 -- readout-subspace decomposition, per the user's exact spec.
A DIAGNOSTIC experiment, not a new model: tests whether G2-A recovery
information in D1 lies preferentially in a subspace the production
readout w_P does not use, by decomposing D1 several ways and training
SIMPLE linear readouts (same LogisticRegression machinery as R1-B,
same training pool, same hyperparameters) on each decomposition.

Explicit guardrails per the user: no Delta_C, no filtering, no
prototype, no architectural modification -- this establishes geometry
only. w_P and w_R (R1-B) are NOT retrained; only new linear readouts on
PROJECTED D1 features are fit.

FOUR DECOMPOSITIONS (all computed with NO leakage: w_P is the frozen
production readout; w_R is R1-B refit identically to E257-B; the SVD
basis V is fit on TRAIN voxels ONLY, then frozen and applied to test):

  1. Production-direction split: x = x_||P + x_perpP
     R-P    = readout on x_||P  (the single scalar w_P^T x, replicated
              since a 1D projection makes a 2-class logistic equivalent
              to a monotonic transform of the production logit itself)
     R-perpP = readout on x_perpP (D1's 31-dim orthogonal complement to w_P)
     R-full  = readout on x (== R1-B itself, reused unchanged)

  2. SVD per-direction usage: a_k=|w_P . v_k|, b_k=|w_R . v_k| for each
     of D1's 32 principal directions v_k (from TRAIN-only SVD), reported
     as a table, no readout fit for this part (diagnostic only).

  3. Cumulative-k subspace sweep: x^(k) = V_k V_k^T x for k in
     {1,2,4,8,16,24,32} (coarse grid, per explicit user choice), SVD
     directions ranked by |b_k - a_k| (the R1-B-vs-production USAGE
     DIFFERENCE, per the user's own "sort by relevance to the R1-B vs
     production difference, not simply variance" instruction) -- a
     readout fit on each x^(k), G2-A recovery plotted vs k.

  4. Two-dimensional span(w_P, w_R) decomposition: x = x_S + x_Sperp
     where S = span(w_P, w_R) (2D). R-S on x_S, R-Sperp on x_Sperp.

ALL readouts fit on the SAME lesion+shell+distant-background voxel
pool R1-B itself was trained on (per explicit user choice), same
LogisticRegression(C=1.0, class_weight='balanced') configuration, same
train/val/test split. All evaluated on the SAME locked test population
used throughout E254/E264/E267 (det_test + g2a_test + g2b), reporting
G2-A recovery (primary), ET Dice, voxel AUC, voxel PR-AUC, WT-not-TC
FP, distant-background FP, total FP -- broken out across the 4
populations (G2-A, detected ET, WT-not-TC FP, distant background FP)
per explicit user instruction.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e257_common import (load_model, get_w_prod, load_populations, build_splits,  # noqa: E402
                         extract_lesion_shell, extract_distant_background,
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION, NEAR_DILATION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402

ET, TC, WT = 0, 1, 2
STRUCT = ndimage.generate_binary_structure(3, 1)
D1_DIM = 32
K_GRID = [1, 2, 4, 8, 16, 24, 32]


def dice_score(pred_mask, gt_mask):
    ps, gs = pred_mask.sum(), gt_mask.sum()
    if gs == 0:
        return 1.0 if ps == 0 else float('nan')
    return float(2 * (pred_mask & gt_mask).sum() / (ps + gs))


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    model = load_model(dev)
    w_et, b_et = get_w_prod(model)
    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    if smoke:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']][:30]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']][:15]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']][:15]
        g2b_lesions_eval = g2b_lesions[:15]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']]
        g2b_lesions_eval = g2b_lesions
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng = np.random.default_rng(999)
    t0 = time.time()

    # ---- build the SAME training voxel pool R1-B was trained on ----
    print('Building training voxel pool (TRAIN subjects only)...', flush=True)
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
    X_train = np.concatenate([lesion_pool, neg_pool]).astype(np.float64)
    y_train = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
    print(f'  pool: lesion={len(lesion_pool)} neg={len(neg_pool)} ({time.time()-t0:.0f}s)', flush=True)

    # ---- R1-B (R-full), IDENTICAL to E257-B ----
    r1b = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_train, y_train)
    w_r1b = r1b.coef_[0].astype(np.float64)
    b_r1b = float(r1b.intercept_[0])
    print(f'R-full (R1-B) fit ({time.time()-t0:.0f}s)', flush=True)

    w_p = w_et.astype(np.float64)

    # ---- decomposition 1: production-direction split ----
    w_p_unit = w_p / (np.linalg.norm(w_p) + 1e-12)
    X_par_p = np.outer(X_train @ w_p_unit, w_p_unit)      # x_||P
    X_perp_p = X_train - X_par_p                            # x_perpP

    def fit_lr(X, y):
        clf = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X, y)
        return clf

    r_p = fit_lr(X_par_p, y_train)
    r_perp_p = fit_lr(X_perp_p, y_train)
    print(f'R-P, R-perpP fit ({time.time()-t0:.0f}s)', flush=True)

    # ---- decomposition 2 (diagnostic table): SVD per-direction usage ----
    X_centered = X_train - X_train.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    V = Vt.T  # (32, 32), columns = principal directions, frozen from TRAIN only
    a_k = np.abs(V.T @ w_p)         # |w_P . v_k| for each direction
    b_k = np.abs(V.T @ w_r1b)       # |w_R . v_k| for each direction
    usage_diff = np.abs(b_k - a_k)
    print(f'\nSVD per-direction usage (|w_P.v_k|, |w_R.v_k|):', flush=True)
    for k in range(D1_DIM):
        print(f'  v{k}: a_k={a_k[k]:.4f} b_k={b_k[k]:.4f} diff={usage_diff[k]:.4f} '
             f'singular_value={S[k]:.4f}', flush=True)

    # rank directions by |b_k - a_k| (R1-B-vs-production usage difference)
    rank_order = np.argsort(-usage_diff)  # descending

    # ---- decomposition 3: cumulative-k subspace sweep ----
    k_readouts = {}
    for k in K_GRID:
        V_k = V[:, rank_order[:k]]  # (32, k)
        X_k = (X_train @ V_k) @ V_k.T  # project onto top-k ranked directions, back to 32-dim
        k_readouts[k] = fit_lr(X_k, y_train)
    print(f'K-sweep readouts fit ({time.time()-t0:.0f}s)', flush=True)

    # ---- decomposition 4: span(w_P, w_R) 2D subspace ----
    # orthonormal basis for S = span(w_P, w_R) via QR
    basis_raw = np.stack([w_p, w_r1b], axis=1)  # (32, 2)
    Q, _ = np.linalg.qr(basis_raw)  # Q: (32,2) orthonormal basis for S
    X_S = (X_train @ Q) @ Q.T
    X_Sperp = X_train - X_S
    r_s = fit_lr(X_S, y_train)
    r_sperp = fit_lr(X_Sperp, y_train)
    cos_theta = float(np.dot(w_p_unit, w_r1b / (np.linalg.norm(w_r1b) + 1e-12)))
    theta_deg = float(np.degrees(np.arccos(np.clip(cos_theta, -1, 1))))
    print(f'span(w_P,w_R): cos(theta)={cos_theta:.4f} theta={theta_deg:.2f}deg '
         f'({time.time()-t0:.0f}s)', flush=True)

    # ======== EVALUATION ========
    w_p_t = torch.from_numpy(w_p).float().to(dev)
    b_p_t = float(b_et)

    def project_apply(readout, proj_fn):
        """Returns a (w,b) pair usable as a linear map on RAW D1, by
        composing the readout's own (coef,intercept) with the upstream
        linear projection proj_fn (a (32,32) numpy matrix or None for identity)."""
        w = readout.coef_[0].astype(np.float64)
        b = float(readout.intercept_[0])
        if proj_fn is not None:
            w = proj_fn.T @ w  # since x_proj = x @ P (P symmetric for our projections), w_eff = P @ w
        return w, b

    variants = {}
    variants['R_full'] = (w_r1b, b_r1b)
    # x_||P = X @ (w_p_unit outer w_p_unit) -> effective projection matrix P_par = outer(w_p_unit,w_p_unit)
    P_par = np.outer(w_p_unit, w_p_unit)
    P_perp = np.eye(D1_DIM) - P_par
    variants['R_P'] = project_apply(r_p, P_par)
    variants['R_perpP'] = project_apply(r_perp_p, P_perp)
    for k in K_GRID:
        V_k = V[:, rank_order[:k]]
        P_k = V_k @ V_k.T
        variants[f'R_k{k}'] = project_apply(k_readouts[k], P_k)
    P_S = Q @ Q.T
    P_Sperp = np.eye(D1_DIM) - P_S
    variants['R_S'] = project_apply(r_s, P_S)
    variants['R_Sperp'] = project_apply(r_sperp, P_Sperp)

    variant_tensors = {name: (torch.from_numpy(w).float().to(dev), b) for name, (w, b) in variants.items()}

    dense_cache = {}

    def dense_eval(sid):
        if sid in dense_cache:
            return dense_cache[sid]
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T
        maps = {}
        with torch.no_grad():
            p_prod = torch.sigmoid(flat @ w_p_t + b_p_t).reshape(D, H, W).cpu().numpy()
            maps['prod'] = p_prod
            for name, (w_t, b_v) in variant_tensors.items():
                p = torch.sigmoid(flat @ w_t + b_v).reshape(D, H, W).cpu().numpy()
                maps[name] = p
        result = (maps, tgt_c, img_c)
        dense_cache[sid] = result
        return result

    test_subjects = sorted(set(r['subject_id'] for r in det_lesions_test) |
                          set(r['subject_id'] for r in g2a_lesions_test) |
                          set(r['subject_id'] for r in g2b_lesions_eval))
    if smoke:
        test_subjects = test_subjects[:15]
    print(f'\nEvaluating {len(variants)} variants on {len(test_subjects)} TEST subjects...', flush=True)

    variant_names = list(variants.keys())
    per_subject_dice = {v: [] for v in variant_names}
    fp_total = {v: 0 for v in variant_names}
    fp_wt_not_tc = {v: 0 for v in variant_names}
    fp_distant = {v: 0 for v in variant_names}
    auc_y = {v: [] for v in variant_names}
    auc_scores = {v: [] for v in variant_names}

    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        maps, tgt_c, img_c = dense_eval(sid)
        true_et = tgt_c[ET] > 0.5
        true_tc = tgt_c[TC] > 0.5
        true_wt = tgt_c[WT] > 0.5
        brain_mask = img_c[0] != 0
        y_true_flat = true_et[brain_mask].astype(int)
        for vname in variant_names:
            p_map = maps[vname]
            mask = (p_map > TAU) & brain_mask
            per_subject_dice[vname].append(dice_score(mask, true_et))
            fp_mask = mask & brain_mask & (~true_et)
            fp_total[vname] += int(fp_mask.sum())
            fp_wt_not_tc[vname] += int((fp_mask & true_wt & (~true_tc)).sum())
            dilated = ndimage.binary_dilation(true_et, structure=STRUCT,
                                              iterations=NEAR_DILATION) if true_et.any() else np.zeros_like(true_et)
            local_shell = dilated & (~true_et) & brain_mask
            distant_bg = brain_mask & (~true_et) & (~local_shell)
            fp_distant[vname] += int((fp_mask & distant_bg).sum())
            # subsample for AUC/PR-AUC to keep memory bounded (every 4th voxel)
            p_flat = p_map[brain_mask][::4]
            y_flat = y_true_flat[::4]
            auc_y[vname].append(y_flat); auc_scores[vname].append(p_flat)
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    def lesion_recovery(lesion_list, vname):
        recs = []
        for r in lesion_list:
            sid, cid = r['subject_id'], int(r['comp_id'])
            if sid not in sid_to_idx:
                continue
            maps, tgt_c, img_c = dense_eval(sid)
            et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
            if cid < 1 or cid > et_n:
                continue
            lesion_mask = et_lbl == cid
            if int(lesion_mask.sum()) < MIN_VOX:
                continue
            p_map = maps[vname]
            recs.append(int(((p_map > TAU) & lesion_mask).any()))
        return float(np.mean(recs)) if recs else float('nan')

    print('\nComputing recovery + AUC/PR-AUC per variant...', flush=True)
    results = {}
    for vname in variant_names:
        y_cat = np.concatenate(auc_y[vname]) if auc_y[vname] else np.array([])
        s_cat = np.concatenate(auc_scores[vname]) if auc_scores[vname] else np.array([])
        try:
            auc = float(roc_auc_score(y_cat, s_cat)) if len(np.unique(y_cat)) > 1 else float('nan')
            pr_auc = float(average_precision_score(y_cat, s_cat)) if len(np.unique(y_cat)) > 1 else float('nan')
        except Exception:
            auc, pr_auc = float('nan'), float('nan')
        results[vname] = {
            'g2a_recovery': lesion_recovery(g2a_lesions_test, vname),
            'det_recovery': lesion_recovery(det_lesions_test, vname),
            'g2b_recovery': lesion_recovery(g2b_lesions_eval, vname),
            'et_dice_mean': float(np.nanmean(per_subject_dice[vname])),
            'voxel_auc': auc,
            'voxel_pr_auc': pr_auc,
            'fp_total': fp_total[vname],
            'fp_wt_not_tc': fp_wt_not_tc[vname],
            'fp_distant_bg': fp_distant[vname],
        }

    print(f'\n=== E268 TEST results ===')
    for k, v in results.items():
        print(f'{k}: {v}')

    out = HERE / ('E268_results_smoke.csv' if smoke else 'E268_results.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['variant', 'g2a_recovery', 'det_recovery', 'g2b_recovery',
                                           'et_dice_mean', 'voxel_auc', 'voxel_pr_auc', 'fp_total',
                                           'fp_wt_not_tc', 'fp_distant_bg'])
        w.writeheader()
        for vname, v in results.items():
            w.writerow({'variant': vname, **v})

    svd_out = HERE / ('E268_svd_usage_smoke.csv' if smoke else 'E268_svd_usage.csv')
    with open(svd_out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['direction', 'a_k_prod_usage', 'b_k_r1b_usage', 'usage_diff', 'singular_value'])
        for k in range(D1_DIM):
            w.writerow([k, a_k[k], b_k[k], usage_diff[k], S[k]])

    print(f'\ncos(w_P,w_R)={cos_theta:.4f}  theta={theta_deg:.2f}deg')
    print(f'E268 complete ({time.time()-t0:.0f}s). wrote {out.name}, {svd_out.name}', flush=True)


if __name__ == '__main__':
    main()
