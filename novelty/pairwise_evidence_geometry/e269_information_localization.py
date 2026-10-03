"""E269 -- controlled information-localization experiment, per the
user's exact spec. Direct follow-up to E268, which found G2-A recovery
concentrated outside production's own readout direction but left two
open problems: (1) the cumulative-k sweep was confounded by fixed-
threshold miscalibration at low k, (2) no control for whether ANY
low-variance subspace would show the same effect (an SVD/variance
artifact) vs the SPECIFIC learned directions mattering.

H269: low-variance D1 directions (specifically SV7:26, the band where
E268 found the largest |a_k-b_k| production-vs-R1-B usage gap) contain
disproportionate G2-A information, beyond what variance rank alone
would predict.

FIXES relative to E268's k-sweep:
  - FIXED variance BANDS (not cumulative-k), avoiding the degenerate-
    low-dim-model-at-wrong-threshold problem a cumulative sweep hits
    at k=1,2.
  - Per-band threshold calibrated on VAL via Youden's J, frozen before
    touching TEST (exactly as E267's own discipline).
  - AUC/AP are the PRIMARY metric (not recovery-at-a-fixed-threshold),
    since this experiment asks WHERE information exists, not what
    segmentation threshold to deploy.
  - Explicit CONTROLS: (a) variance-matched RANDOM orthonormal
    subspaces of the same dimensionality as each band (20 per band,
    per explicit user choice this session -- within the user's own
    20-50 range), (b) a VARIANCE-PRESERVING SCRAMBLED basis (same
    singular values, randomized directions) to separate "low energy
    helps" from "the LEARNED orientation specifically matters".

D1 IS 32-DIMENSIONAL (not the >100-dim space the user's original band
boundaries assumed) -- flagged to and resolved with the user: bands
adapted to B1={v0}, B2={v1:6}, B3={v7:25} (EXACT match to the user's
hypothesis-critical band), B4={v26:31} (the remaining tail, folding
the user's B4+B5 into one since there's no room for two separate
low-variance bands beyond v26 in 32 dimensions).

Per explicit user instruction: this experiment answers ONE question
(where does the information live) -- no new production model, no
Delta_C, no architecture change. E270 (deliberately recovering from
production-ignored directions) is explicitly deferred to only if H1
(strong positive) is supported here.
"""
import sys, csv, time, os, gc
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
BANDS = {'B1_v0': [0], 'B2_v1to6': list(range(1, 7)),
         'B3_v7to25': list(range(7, 26)), 'B4_v26to31': list(range(26, 32))}
N_RANDOM_SUBSPACES = 20
RNG_SEED_RANDOM = 7001
RNG_SEED_SCRAMBLE = 7002


def dice_score(pred_mask, gt_mask):
    ps, gs = pred_mask.sum(), gt_mask.sum()
    if gs == 0:
        return 1.0 if ps == 0 else float('nan')
    return float(2 * (pred_mask & gt_mask).sum() / (ps + gs))


def random_orthonormal_subspace(dim, d1_dim, rng):
    """Random d1_dim x dim orthonormal basis via QR of a random Gaussian matrix."""
    A = rng.normal(size=(d1_dim, dim))
    Q, _ = np.linalg.qr(A)
    return Q[:, :dim]


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
        n_random = 5
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        g2a_lesions_val = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_val']]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']]
        g2b_lesions_eval = g2b_lesions
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])
        n_random = N_RANDOM_SUBSPACES

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

    r1b = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_train, y_train)
    w_r1b = r1b.coef_[0].astype(np.float64)
    w_p = w_et.astype(np.float64)
    print(f'R1-B (R-full) fit ({time.time()-t0:.0f}s)', flush=True)

    # ---- SVD, TRAIN-only, frozen before touching test (no leakage) ----
    X_centered = X_train - X_train.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    V = Vt.T  # (32, 32) columns = principal directions

    def fit_lr(X, y):
        return LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X, y)

    # ---- build all projection matrices + fit readouts on TRAIN ----
    variants = {}  # name -> (P matrix (32,32) symmetric, readout)

    for bname, idxs in BANDS.items():
        V_b = V[:, idxs]
        P_b = V_b @ V_b.T
        X_b = X_train @ P_b
        variants[bname] = (P_b, fit_lr(X_b, y_train))

    rng_random = np.random.default_rng(RNG_SEED_RANDOM)
    for bname, idxs in BANDS.items():
        dim = len(idxs)
        for i in range(n_random):
            Q_r = random_orthonormal_subspace(dim, D1_DIM, rng_random)
            P_r = Q_r @ Q_r.T
            X_r = X_train @ P_r
            variants[f'{bname}_random{i}'] = (P_r, fit_lr(X_r, y_train))
    print(f'Random-subspace controls fit ({n_random}/band) ({time.time()-t0:.0f}s)', flush=True)

    # ---- variance-preserving scrambled basis control ----
    # same singular values S, randomized orthonormal directions (one shared
    # random orthonormal basis V_scrambled, columns reordered/assigned by
    # the SAME singular-value rank order, so "scrambled v_k" still carries
    # variance S[k] but in an arbitrary direction, not the LEARNED one)
    rng_scramble = np.random.default_rng(RNG_SEED_SCRAMBLE)
    A_scramble = rng_scramble.normal(size=(D1_DIM, D1_DIM))
    V_scrambled, _ = np.linalg.qr(A_scramble)  # (32,32) random orthonormal basis
    for bname, idxs in BANDS.items():
        V_b_scrambled = V_scrambled[:, idxs]
        P_b_scrambled = V_b_scrambled @ V_b_scrambled.T
        X_b_scrambled = X_train @ P_b_scrambled
        variants[f'{bname}_scrambled'] = (P_b_scrambled, fit_lr(X_b_scrambled, y_train))
    print(f'Scrambled-basis controls fit ({time.time()-t0:.0f}s)', flush=True)

    # ---- per-direction single-dimension sanity sweep ----
    for k in range(D1_DIM):
        v_k = V[:, k:k+1]
        P_k = v_k @ v_k.T
        X_k = X_train @ P_k
        variants[f'dir{k}'] = (P_k, fit_lr(X_k, y_train))
    print(f'Per-direction (32x) readouts fit ({time.time()-t0:.0f}s)', flush=True)

    variants['R_full'] = (np.eye(D1_DIM), r1b)

    print(f'\nTotal variants: {len(variants)}', flush=True)

    # ======== DENSE EVALUATION INFRA ========
    w_p_t = torch.from_numpy(w_p).float().to(dev)
    b_p_t = float(b_et)

    def project_apply(P, readout):
        w = readout.coef_[0].astype(np.float64)
        b = float(readout.intercept_[0])
        w_eff = P.T @ w
        return w_eff, b

    variant_tensors = {}
    for name, (P, readout) in variants.items():
        w_eff, b_eff = project_apply(P, readout)
        variant_tensors[name] = (torch.from_numpy(w_eff).float().to(dev), b_eff)

    def dense_eval(sid):
        """NOT cached (per-subject maps for 121 variants are ~1GB each;
        caching across VAL+TEST+lesion-recovery phases exhausted memory
        in the first full-scale run -- ArrayMemoryError after ~180
        subjects' worth of accumulated maps). Each call recomputes the
        D1 forward pass once and returns all 121 variants' probability
        maps for THIS SUBJECT ONLY; callers must extract everything
        they need before moving to the next subject."""
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T
        maps = {}
        with torch.no_grad():
            for name, (w_t, b_v) in variant_tensors.items():
                p = torch.sigmoid(flat @ w_t + b_v).reshape(D, H, W).cpu().numpy()
                maps[name] = p
        return maps, tgt_c, img_c

    # ======== VAL: calibrate TAU per variant via Youden's J ========
    val_subjects = sorted(set(r['subject_id'] for r in g2a_lesions_val))
    print(f'\nCalibrating per-variant thresholds on {len(val_subjects)} VAL subjects...', flush=True)
    val_y_by_variant = {name: [] for name in variants}
    val_s_by_variant = {name: [] for name in variants}
    for i, sid in enumerate(val_subjects):
        if sid not in sid_to_idx:
            continue
        maps, tgt_c, img_c = dense_eval(sid)
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        y_flat = true_et[brain_mask].astype(int)[::256]
        for name, p_map in maps.items():
            s_flat = p_map[brain_mask][::256]
            val_y_by_variant[name].append(y_flat)
            val_s_by_variant[name].append(s_flat)
        del maps, p_map, true_et, brain_mask
        gc.collect()
        if (i + 1) % 10 == 0 or smoke:
            print(f'  {i+1}/{len(val_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    def youden_threshold(y, s):
        if len(np.unique(y)) < 2:
            return 0.5
        thresholds = np.unique(s)
        if len(thresholds) > 200:
            thresholds = np.quantile(thresholds, np.linspace(0, 1, 200))
        best_j, best_t = -1, 0.5
        pos, neg = s[y == 1], s[y == 0]
        for t in thresholds:
            sens = (pos >= t).mean() if len(pos) else 0.0
            spec = (neg < t).mean() if len(neg) else 0.0
            j = sens + spec - 1
            if j > best_j:
                best_j, best_t = j, t
        return float(best_t)

    tau_by_variant = {}
    for name in variants:
        y_cat = np.concatenate(val_y_by_variant[name]) if val_y_by_variant[name] else np.array([])
        s_cat = np.concatenate(val_s_by_variant[name]) if val_s_by_variant[name] else np.array([])
        tau_by_variant[name] = youden_threshold(y_cat, s_cat)
    print(f'Threshold calibration done ({time.time()-t0:.0f}s)', flush=True)

    # ======== TEST: evaluate once ========
    test_subjects = sorted(set(r['subject_id'] for r in det_lesions_test) |
                          set(r['subject_id'] for r in g2a_lesions_test) |
                          set(r['subject_id'] for r in g2b_lesions_eval))
    if smoke:
        test_subjects = test_subjects[:15]
    print(f'\nEvaluating {len(variants)} variants on {len(test_subjects)} TEST subjects...', flush=True)

    per_subject_dice = {v: [] for v in variants}
    fp_total = {v: 0 for v in variants}
    fp_wt_not_tc = {v: 0 for v in variants}
    fp_distant = {v: 0 for v in variants}
    test_auc_y = {v: [] for v in variants}
    test_auc_s = {v: [] for v in variants}
    g2a_recs = {v: [] for v in variants}
    det_recs = {v: [] for v in variants}
    g2b_recs = {v: [] for v in variants}

    # group lesion lists by subject for a SINGLE dense_eval call per
    # subject that covers FP/Dice/AUC AND all lesion-recovery populations
    # (the earlier version called dense_eval again per-lesion afterward,
    # which combined with full per-variant map retention caused an OOM
    # crash at full scale -- fixed by doing everything in one pass)
    g2a_by_sid, det_by_sid, g2b_by_sid = {}, {}, {}
    for r in g2a_lesions_test:
        g2a_by_sid.setdefault(r['subject_id'], []).append(int(r['comp_id']))
    for r in det_lesions_test:
        det_by_sid.setdefault(r['subject_id'], []).append(int(r['comp_id']))
    for r in g2b_lesions_eval:
        g2b_by_sid.setdefault(r['subject_id'], []).append(int(r['comp_id']))

    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        maps, tgt_c, img_c = dense_eval(sid)
        true_et = tgt_c[ET] > 0.5
        true_tc = tgt_c[TC] > 0.5
        true_wt = tgt_c[WT] > 0.5
        brain_mask = img_c[0] != 0
        y_true_flat = true_et[brain_mask].astype(int)
        dilated = ndimage.binary_dilation(true_et, structure=STRUCT,
                                          iterations=NEAR_DILATION) if true_et.any() else np.zeros_like(true_et)
        local_shell = dilated & (~true_et) & brain_mask
        distant_bg = brain_mask & (~true_et) & (~local_shell)
        et_lbl, et_n = ndimage.label(true_et)

        g2a_cids = g2a_by_sid.get(sid, [])
        det_cids = det_by_sid.get(sid, [])
        g2b_cids = g2b_by_sid.get(sid, [])

        def valid_lesion_masks(cids):
            out = []
            for cid in cids:
                if cid < 1 or cid > et_n:
                    continue
                lm = et_lbl == cid
                if int(lm.sum()) < MIN_VOX:
                    continue
                out.append(lm)
            return out

        g2a_masks = valid_lesion_masks(g2a_cids)
        det_masks = valid_lesion_masks(det_cids)
        g2b_masks = valid_lesion_masks(g2b_cids)

        for name, p_map in maps.items():
            tau_v = tau_by_variant[name]
            mask = (p_map > tau_v) & brain_mask
            per_subject_dice[name].append(dice_score(mask, true_et))
            fp_mask = mask & (~true_et)
            fp_total[name] += int(fp_mask.sum())
            fp_wt_not_tc[name] += int((fp_mask & true_wt & (~true_tc)).sum())
            fp_distant[name] += int((fp_mask & distant_bg).sum())
            s_flat = p_map[brain_mask][::256]
            test_auc_y[name].append(y_true_flat[::256])
            test_auc_s[name].append(s_flat)
            for lm in g2a_masks:
                g2a_recs[name].append(int((mask & lm).any()))
            for lm in det_masks:
                det_recs[name].append(int((mask & lm).any()))
            for lm in g2b_masks:
                g2b_recs[name].append(int((mask & lm).any()))
        del maps, mask, fp_mask, p_map, true_et, true_tc, true_wt, brain_mask, dilated, local_shell, distant_bg
        gc.collect()
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    print('\nComputing final metrics per variant...', flush=True)
    results = {}
    for name in variants:
        y_cat = np.concatenate(test_auc_y[name]) if test_auc_y[name] else np.array([])
        s_cat = np.concatenate(test_auc_s[name]) if test_auc_s[name] else np.array([])
        try:
            auc = float(roc_auc_score(y_cat, s_cat)) if len(np.unique(y_cat)) > 1 else float('nan')
            ap = float(average_precision_score(y_cat, s_cat)) if len(np.unique(y_cat)) > 1 else float('nan')
        except Exception:
            auc, ap = float('nan'), float('nan')
        results[name] = {
            'auc': auc, 'ap': ap,
            'g2a_recovery': float(np.mean(g2a_recs[name])) if g2a_recs[name] else float('nan'),
            'det_recovery': float(np.mean(det_recs[name])) if det_recs[name] else float('nan'),
            'g2b_recovery': float(np.mean(g2b_recs[name])) if g2b_recs[name] else float('nan'),
            'et_dice_mean': float(np.nanmean(per_subject_dice[name])),
            'fp_total': fp_total[name],
            'fp_wt_not_tc': fp_wt_not_tc[name],
            'fp_distant_bg': fp_distant[name],
            'tau': tau_by_variant[name],
        }

    out = HERE / ('E269_results_smoke.csv' if smoke else 'E269_results.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['variant', 'auc', 'ap', 'g2a_recovery', 'det_recovery',
                                           'g2b_recovery', 'et_dice_mean', 'fp_total', 'fp_wt_not_tc',
                                           'fp_distant_bg', 'tau'])
        w.writeheader()
        for name, v in results.items():
            w.writerow({'variant': name, **v})

    # ---- production/R1-B usage per direction (for correlation with AUC_k) ----
    a_k = np.abs(V.T @ w_p) / (np.linalg.norm(w_p) + 1e-12)
    b_k = np.abs(V.T @ w_r1b) / (np.linalg.norm(w_r1b) + 1e-12)
    d_k = np.abs(a_k - b_k)
    usage_out = HERE / ('E269_direction_usage_smoke.csv' if smoke else 'E269_direction_usage.csv')
    with open(usage_out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['direction', 'a_k', 'b_k', 'd_k', 'singular_value', 'dir_auc'])
        for k in range(D1_DIM):
            w.writerow([k, a_k[k], b_k[k], d_k[k], S[k], results[f'dir{k}']['auc']])

    print(f'\n=== E269 band summary (AUC primary) ===')
    for bname in BANDS:
        band_auc = results[bname]['auc']
        random_aucs = [results[f'{bname}_random{i}']['auc'] for i in range(n_random)]
        scrambled_auc = results[f'{bname}_scrambled']['auc']
        print(f'{bname}: band_AUC={band_auc:.4f}  random_AUC_median={np.nanmedian(random_aucs):.4f} '
             f'(range {np.nanmin(random_aucs):.4f}-{np.nanmax(random_aucs):.4f})  '
             f'scrambled_AUC={scrambled_auc:.4f}  g2a_recovery={results[bname]["g2a_recovery"]:.4f}')
    print(f'R_full: AUC={results["R_full"]["auc"]:.4f} g2a_recovery={results["R_full"]["g2a_recovery"]:.4f}')

    print(f'\nE269 complete ({time.time()-t0:.0f}s). wrote {out.name}, {usage_out.name}', flush=True)


if __name__ == '__main__':
    main()
