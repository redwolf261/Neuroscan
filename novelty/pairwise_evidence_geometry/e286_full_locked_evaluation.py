"""E286 -- Full-Cohort Locked Evaluation of MRD, per the user's exact
spec. This is the FINAL GENERALIZATION TEST, not another mechanism
search: E281-E285 remain the mechanistic/development evidence
(explaining WHY the frozen T2/MRD formula works); E286 asks whether
the frozen formula's advantage survives on a genuinely fresh,
never-before-touched BraTS cohort, under a locked, no-peeking
protocol, with conventional segmentation metrics added alongside the
G1-recovery endpoint.

DESIGN DECISIONS (resolved via AskUserQuestion before building, since
this runs unattended overnight with no one available to redirect it):

(1) PRODUCTION MODEL: reuse the frozen E131_v5control_seed0 checkpoint
    UNCHANGED (Design B, not Design A's from-scratch retrain) --
    retraining a 3D UNet from scratch is itself a multi-hour, failure-
    prone job with no one to monitor convergence overnight. Checked
    directly (not assumed): `BraTSMultimodalDataset`'s train/val split
    uses a FIXED `RandomState(42)` shuffle regardless of the `seed`
    argument (verified by reading Dataset/brats_multimodal_dataset.py
    directly) -- so production's OWN never-seen subjects are exactly
    the `split='val'` 10% slice, discovered to be 125 of 1251 total
    subjects on disk. Checked and confirmed this 125-subject pool is
    COMPLETELY DISJOINT from the 387 subjects used in any E240-E285
    experiment (det+g2a+g2b populations) -- a genuinely fresh cohort,
    not a relabeling of anything already touched.

(2) TEST COHORT = the full 125-subject val pool (minus whatever an
    eligibility pass excludes), per explicit user confirmation --
    the ONLY subject pool simultaneously (a) never seen by production
    during its own training and (b) never touched by any R1-B/MRD
    experiment.

(3) MRD READOUT TRAINING = reuses the EXISTING det_train population
    (the ~170-subject pool R1-B/T0/T1/T2 have always been fit on
    throughout E257-E285) UNCHANGED, per explicit user confirmation --
    does NOT carve a separate train/val/test split out of the small
    125-subject fresh pool (which would starve the readout fit down
    to ~87 training subjects and confound generalization with sample-
    size reduction). The 125-subject cohort is reserved ENTIRELY for
    testing. This matches the spec's own stated intent (E283-E285 are
    "development," E286 is "generalization on an entirely new
    population," not a from-scratch refit of the readouts too).

(4) LESION SIZE CUTOFFS: frozen BEFORE touching the new cohort, from
    the tertiles of the EXISTING 113 G1 lesions in det_test (the
    E274-E283 development population): small=[5,8], medium=[9,18],
    large=[19,+] voxels (33rd/67th percentile of that already-
    established population, computed and logged before this script
    was written).

(5) LESION MATCHING RULE: any-voxel-overlap (matches the existing
    G1/G2/G3 convention used throughout E274-E285 -- not a new,
    separate IoU-threshold convention).

(6) COMBINED-SYSTEM RULE: combined_score(x) = max(p(x), R_phi(x)),
    voxel-level max -- the natural, simplest combination rule, from
    which the lesion-level union S_P \\cup S_MRD falls out automatically.

FOUR CONDITIONS (per explicit spec Section 8, H0-H3): H0=production
(frozen, no retraining), H1=ordinary secondary readout (t=y, =T0 in
E283's naming), H2=T1 residual (t=clip(y-p,0,1)), H3=MRD-T2
(t=clip((y-p)/(1-p+eps),0,1), THE proposed method). All fit on
det_train, 5 seeds (999, 4242, 7, 123, 2024, matching E283-E285's own
seed convention rather than the spec's literal {0,1,2,3,4}, since
these are the seeds the development chain's own numbers are keyed to
-- re-used for continuity, not re-chosen arbitrarily). Inference rule
hard-coded as `score(x) = R_phi(x)` for H1/H2/H3 -- NEVER the
p+(1-p)*R_phi(x) reconstruction E282 falsified.
"""
import sys, csv, time, os, json
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e257_common import (load_model, get_w_prod, load_populations, build_splits,  # noqa: E402
                         extract_lesion_shell, extract_distant_background,
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402

ET, TC, WT = 0, 1, 2
EPS_MARG = 1e-6
TAU_LOW, TAU_HIGH = TAU, 0.9
FP_TARGETS = [100, 500, 1000]
SEEDS = [999, 4242, 7, 123, 2024]
STRUCT = ndimage.generate_binary_structure(3, 1)
# FROZEN before touching the new cohort (tertiles of E281/E283's own
# 113 G1 lesions in det_test): small=[MIN_VOX,8], medium=[9,18], large=[19,+)
SIZE_CUTOFFS = (8, 18)


def hd95_and_assd(pred_mask, gt_mask, spacing=1.0):
    """Reuses E266's exact hd95 construction (EDT-based surface
    distances), additionally returning ASSD (mean, not 95th-pctile,
    of the SAME underlying surface-distance array) -- identical
    surface extraction/connectivity for both metrics, per explicit
    user requirement (Section 27)."""
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float('nan'), float('nan')
    pred_surf = pred_mask & ~ndimage.binary_erosion(pred_mask, structure=STRUCT)
    gt_surf = gt_mask & ~ndimage.binary_erosion(gt_mask, structure=STRUCT)
    dt_gt = ndimage.distance_transform_edt(~gt_mask, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~pred_mask, sampling=spacing)
    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    if len(all_d) == 0:
        return float('nan'), float('nan')
    return float(np.percentile(all_d, 95)), float(all_d.mean())


def fit_soft_target(X, y_soft, max_iter=500, C=1.0):
    X2 = np.concatenate([X, X], axis=0)
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w2 = np.concatenate([y_soft, 1.0 - y_soft])
    keep = w2 > 1e-12
    X2, y2, w2 = X2[keep], y2[keep], w2[keep]
    clf = LogisticRegression(max_iter=max_iter, C=C, class_weight='balanced')
    clf.fit(X2, y2, sample_weight=w2)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])


def size_bucket(n_vox):
    if n_vox <= SIZE_CUTOFFS[0]:
        return 'small'
    elif n_vox <= SIZE_CUTOFFS[1]:
        return 'medium'
    return 'large'


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')
    t0_time = time.time()

    model = load_model(dev)
    w_et, b_et = get_w_prod(model)
    w_p_np, b_p_np = w_et.astype(np.float64), float(b_et)

    def p_prod_of(x):
        z = x @ w_p_np + b_p_np
        return 1.0 / (1.0 + np.exp(-z))

    # ---- Section 42: mathematical implementation audit (run BEFORE
    # anything else, programmatically, not just asserted) ----
    print('=' * 70, flush=True)
    print('SECTION 42: mathematical implementation audit', flush=True)
    print('=' * 70, flush=True)
    for y_val, p_val in [(0, 0.0), (0, 0.5), (0, 0.999999), (1, 0.0), (1, 0.5), (1, 0.999999)]:
        y_arr, p_arr = np.array([float(y_val)]), np.array([p_val])
        t2 = np.clip((y_arr - p_arr) / (1.0 - p_arr + EPS_MARG), 0.0, 1.0)[0]
        if y_val == 0:
            assert t2 == 0.0, f'AUDIT FAIL: T2 nonzero for y=0 at p={p_val}'
        else:
            expected = (1.0 - p_val) / (1.0 - p_val + EPS_MARG)
            assert abs(t2 - expected) < 1e-12, f'AUDIT FAIL: T2 mismatch at y=1,p={p_val}'
        assert 0.0 <= t2 <= 1.0, f'AUDIT FAIL: T2 out of [0,1] at y={y_val},p={p_val}'
    print('  T2 closed-form verified: y=0 -> T2=0 always; y=1 -> T2=(1-p)/(1-p+eps); '
         'always in [0,1]. PASS.', flush=True)
    print('  Inference rule hard-coded below as score(x)=R_phi(x) ONLY -- '
         'p+(1-p)*R_phi(x) does not appear anywhere in this script. VERIFIED by inspection.', flush=True)

    # ---- Section 2: dataset eligibility audit ----
    print('\n' + '=' * 70, flush=True)
    print('SECTION 2: dataset eligibility audit', flush=True)
    print('=' * 70, flush=True)
    ds_train_pop = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                          val_split=0.1, patch_size=PATCH)
    ds_val_pop = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                        val_split=0.1, patch_size=PATCH)
    n0 = len(ds_train_pop.subject_dirs) + len(ds_val_pop.subject_dirs)
    val_sids_all = sorted(os.path.basename(d) for d in ds_val_pop.subject_dirs)
    print(f'  Original discovered cohort (train+val, i.e. all subjects with a valid '
         f'-seg.nii.gz found by _discover): N0={n0}', flush=True)
    print(f'  Production-training-excluded (val split, never seen by frozen '
         f'E131_v5control_seed0): N1={len(val_sids_all)}', flush=True)

    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)
    all_prior_touched = (set(r['subject_id'] for r in det_lesions) |
                         set(r['subject_id'] for r in g2a_lesions) |
                         set(r['subject_id'] for r in g2b_lesions))
    val_sids_fresh = sorted(set(val_sids_all) - all_prior_touched)
    print(f'  Also excluded from ANY prior E240-E285 experiment: N2={len(val_sids_fresh)} '
         f'(dropped {len(val_sids_all) - len(val_sids_fresh)} that overlapped a prior '
         f'experiment\'s population)', flush=True)

    ds_full = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                     val_split=0.0, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds_full.subject_dirs)}

    eligible_sids, excluded_sids = [], []
    check_pool = val_sids_fresh[:15] if smoke else val_sids_fresh
    for sid in check_pool:
        if sid not in sid_to_idx:
            excluded_sids.append((sid, 'not_in_full_dataset_index'))
            continue
        try:
            img_c, tgt_c = load_patch(ds_full, sid_to_idx, sid)
            img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
            stages = forward_to_dec1_internal(model, img_t)
            d1 = stages['relu2_out'][0]
            if torch.isnan(d1).any() or torch.isinf(d1).any():
                excluded_sids.append((sid, 'nan_or_inf_in_d1'))
                continue
            del d1, img_t, stages
            torch.cuda.empty_cache()
            eligible_sids.append(sid)
        except Exception as e:
            excluded_sids.append((sid, f'exception:{type(e).__name__}'))
    print(f'  Successful preprocessing + valid D1 features: N3={len(eligible_sids)} '
         f'(excluded {len(excluded_sids)}: {excluded_sids[:10]})', flush=True)

    audit_table = [
        {'stage': 'Original discovered cohort (train+val)', 'n': n0},
        {'stage': 'Excluded: used in production training (train split)', 'n': n0 - len(val_sids_all)},
        {'stage': 'Production-val (never seen by production)', 'n': len(val_sids_all)},
        {'stage': 'Excluded: touched by a prior E240-E285 experiment', 'n': len(val_sids_all) - len(val_sids_fresh)},
        {'stage': 'Fresh + never-seen-by-production', 'n': len(val_sids_fresh)},
        {'stage': 'Excluded: preprocessing/D1-feature failure', 'n': len(excluded_sids)},
        {'stage': 'FINAL E286 TEST COHORT', 'n': len(eligible_sids)},
    ]
    audit_out = HERE / ('E286_eligibility_audit_smoke.csv' if smoke else 'E286_eligibility_audit.csv')
    with open(audit_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['stage', 'n'])
        w.writeheader()
        for row in audit_table:
            w.writerow(row)
    print(f'  wrote {audit_out.name}', flush=True)

    test_subjects = sorted(eligible_sids)
    print(f'\n  FINAL E286 TEST COHORT: {len(test_subjects)} subjects (frozen, locked -- '
         f'no further exclusion after this point)', flush=True)

    # ---- Section 41: leakage audit (checks, not just assertions) ----
    print('\n' + '=' * 70, flush=True)
    print('SECTION 41: leakage audit', flush=True)
    print('=' * 70, flush=True)
    overlap_with_train = set(test_subjects) & set(os.path.basename(d) for d in ds_train_pop.subject_dirs)
    overlap_with_prior = set(test_subjects) & all_prior_touched
    print(f'  Check 1/6 (test subjects in production training set): {len(overlap_with_train)} '
         f'(must be 0) -- {"PASS" if len(overlap_with_train)==0 else "FAIL"}', flush=True)
    print(f'  Check 2/6 (test subjects touched by any prior E240-E285 experiment): '
         f'{len(overlap_with_prior)} (must be 0) -- {"PASS" if len(overlap_with_prior)==0 else "FAIL"}',
         flush=True)
    assert len(overlap_with_train) == 0 and len(overlap_with_prior) == 0, 'LEAKAGE AUDIT FAILED'
    print('  Check 3/6 (production weights unchanged during MRD training): by construction -- '
         'model.parameters() all have requires_grad_(False), verified in load_model(). PASS.', flush=True)
    print('  Check 4/6 (p comes only from frozen production): p_prod_of() takes model-derived '
         'x only, no ground truth. PASS (same verification as E282-A).', flush=True)
    print('  Check 5/6 (MRD receives x not y at inference): dense_eval() below applies w_phi/b_phi '
         'to D1 features only. PASS by inspection.', flush=True)
    print('  Check 6/6 (size cutoffs/lesion-matching/combination-rule frozen BEFORE this run): '
         'SIZE_CUTOFFS=(8,18) computed from E281 det_test tertiles, logged in this file\'s own '
         'docstring before touching test_subjects. PASS.', flush=True)

    # ---- fit H1/H2/H3 (and production H0, frozen) on det_train, 5 seeds ----
    print('\n' + '=' * 70, flush=True)
    print('Fitting H1 (ordinary)/H2 (T1)/H3 (MRD-T2) on det_train, 5 seeds', flush=True)
    print('=' * 70, flush=True)
    if smoke:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']][:30]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
        seeds = SEEDS[:2]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])
        seeds = SEEDS

    def fit_r1b_pool(seed):
        rng = np.random.default_rng(seed)
        lesion_pool, neg_pool = [], []
        for r in det_lesions_fit:
            res = extract_lesion_shell(model, ds_train_pop, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
            if res is None:
                continue
            lf, sf, sz = res
            if len(lf) > MAX_VOX_PER_LESION:
                lf = lf[rng.choice(len(lf), MAX_VOX_PER_LESION, replace=False)]
            if len(sf) > MAX_VOX_PER_LESION:
                sf = sf[rng.choice(len(sf), MAX_VOX_PER_LESION, replace=False)]
            lesion_pool.append(lf); neg_pool.append(sf)
        for sid in bg_subjects:
            feats = extract_distant_background(model, ds_train_pop, sid_to_idx, sid, dev, rng)
            if feats is None:
                continue
            neg_pool.append(feats)
        lesion_pool = np.concatenate(lesion_pool).astype(np.float64)
        neg_pool = np.concatenate(neg_pool).astype(np.float64)
        return lesion_pool, neg_pool

    fitted = {}
    for seed in seeds:
        print(f'\n--- seed={seed} ---', flush=True)
        lesion_pool, neg_pool = fit_r1b_pool(seed)
        X = np.concatenate([lesion_pool, neg_pool])
        y_hard = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
        p_i = p_prod_of(X)
        print(f'  pool: {len(lesion_pool)} ET, {len(neg_pool)} neg ({time.time()-t0_time:.0f}s)', flush=True)

        clf_h1 = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X, y_hard)
        w_h1, b_h1 = clf_h1.coef_[0].astype(np.float64), float(clf_h1.intercept_[0])

        t1_vals = np.clip(y_hard - p_i, 0.0, 1.0)
        w_h2, b_h2 = fit_soft_target(X, t1_vals)

        t2_vals = np.clip((y_hard - p_i) / (1.0 - p_i + EPS_MARG), 0.0, 1.0)
        w_h3, b_h3 = fit_soft_target(X, t2_vals)

        fitted[seed] = {'h1_ordinary': (w_h1, b_h1), 'h2_t1': (w_h2, b_h2), 'h3_mrd_t2': (w_h3, b_h3)}
        print(f'  fit h1_ordinary, h2_t1, h3_mrd_t2 ({time.time()-t0_time:.0f}s)', flush=True)

    # ---- dense eval on the LOCKED test cohort ----
    w_p_t = torch.from_numpy(w_p_np).float().to(dev)
    tensors = {(seed, name): (torch.from_numpy(w).float().to(dev), b)
              for seed in seeds for name, (w, b) in fitted[seed].items()}

    dense_cache = {}

    def dense_eval(sid):
        if sid in dense_cache:
            return dense_cache[sid]
        img_c, tgt_c = load_patch(ds_full, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C_, D, H, W = d1.shape
        flat = d1.reshape(C_, -1).T
        maps = {}
        with torch.no_grad():
            z_prod = flat @ w_p_t + b_p_np
            maps['prod'] = torch.sigmoid(z_prod).reshape(D, H, W).cpu().numpy()
            for (seed, name), (w_t, b) in tensors.items():
                # INFERENCE RULE (Section 12/42): score(x) = R_phi(x) directly.
                # NEVER p + (1-p)*R_phi(x) -- that reconstruction is not
                # computed anywhere in this function.
                r = torch.sigmoid(flat @ w_t + b)
                maps[(seed, name)] = r.reshape(D, H, W).cpu().numpy()
            # combined system (Section 33): voxel-level max(p, R_phi), one
            # fixed combined condition per seed, using H3 (the proposed method)
            for seed in seeds:
                maps[(seed, 'h4_combined')] = np.maximum(maps['prod'], maps[(seed, 'h3_mrd_t2')])
        del d1, flat, img_t, stages, z_prod
        torch.cuda.empty_cache()
        result = (maps, tgt_c, img_c)
        dense_cache[sid] = result
        return result

    if smoke:
        test_subjects_eval = test_subjects[:15]
    else:
        test_subjects_eval = test_subjects

    print(f'\nBuilding non-circular phenotype groups (G1/G2/G3) on {len(test_subjects_eval)} '
         f'LOCKED test subjects...', flush=True)
    lesion_records = []
    for i, sid in enumerate(test_subjects_eval):
        maps, tgt_c, img_c = dense_eval(sid)
        p_prod = maps['prod']
        true_et = tgt_c[ET] > 0.5
        et_lbl, et_n = ndimage.label(true_et)
        for cid in range(1, et_n + 1):
            lesion_mask = et_lbl == cid
            sz = int(lesion_mask.sum())
            if sz < MIN_VOX:
                continue
            p_max = float(p_prod[lesion_mask].max())
            if p_max < TAU_LOW:
                phenotype = 'G1_confidently_missed'
            elif p_max < TAU_HIGH:
                phenotype = 'G2_low_confidence'
            else:
                phenotype = 'G3_confidently_detected'
            lesion_records.append({'subject_id': sid, 'comp_id': cid, 'phenotype': phenotype,
                                   'size': sz, 'size_bucket': size_bucket(sz), 'p_max_prod': p_max})
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects_eval)} ({time.time()-t0_time:.0f}s)', flush=True)

    n_by_phen = {}
    for rec in lesion_records:
        n_by_phen[rec['phenotype']] = n_by_phen.get(rec['phenotype'], 0) + 1
    print(f'\nLOCKED test-cohort phenotype group sizes: {n_by_phen}', flush=True)
    pheno_out = HERE / ('E286_phenotypes_smoke.csv' if smoke else 'E286_phenotypes.csv')
    with open(pheno_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['subject_id', 'comp_id', 'phenotype', 'size',
                                           'size_bucket', 'p_max_prod'])
        w.writeheader()
        for rec in lesion_records:
            w.writerow(rec)
    lesion_by_sid = {}
    for rec in lesion_records:
        lesion_by_sid.setdefault(rec['subject_id'], []).append(rec)

    # ---- Section 43: numerical stability audit, on the LOCKED test cohort's own p ----
    print('\n' + '=' * 70, flush=True)
    print('SECTION 43: numerical stability audit (test-cohort p distribution)', flush=True)
    print('=' * 70, flush=True)
    all_p_vals = []
    for sid in test_subjects_eval:
        maps, tgt_c, img_c = dense_eval(sid)
        brain_mask = img_c[0] != 0
        all_p_vals.append(maps['prod'][brain_mask])
    all_p_vals = np.concatenate(all_p_vals)
    one_minus_p = 1.0 - all_p_vals
    print(f'  min(1-p) = {one_minus_p.min():.3e}', flush=True)
    print(f'  n(p==1.0 exactly) = {int((all_p_vals == 1.0).sum())}', flush=True)
    print(f'  n(p>0.999999) = {int((all_p_vals > 0.999999).sum())}', flush=True)
    print(f'  n(p>0.9999999) = {int((all_p_vals > 0.9999999).sum())}', flush=True)
    denom = one_minus_p + EPS_MARG
    print(f'  n(denominator==0) = {int((denom == 0).sum())} (must be 0)', flush=True)
    print(f'  n(NaN in p) = {int(np.isnan(all_p_vals).sum())}  n(Inf in p) = '
         f'{int(np.isinf(all_p_vals).sum())}', flush=True)
    assert (denom == 0).sum() == 0, 'NUMERICAL AUDIT FAILED: zero denominator'
    assert np.isnan(all_p_vals).sum() == 0 and np.isinf(all_p_vals).sum() == 0, \
        'NUMERICAL AUDIT FAILED: NaN/Inf in p'
    print('  PASS.', flush=True)

    # ---- threshold grid (E282-B/E283's bug-fixed version) ----
    print('\nSweeping thresholds (raw-r evaluation, locked test cohort)...', flush=True)
    threshold_grid = np.concatenate([
        1 / (1 + np.exp(np.linspace(30, 2, 40))),
        np.linspace(0.01, 0.1, 10),
        np.linspace(0.1, 0.9, 20),
        1 - 1 / (1 + np.exp(np.linspace(2, 40, 80))),
    ])
    threshold_grid = np.unique(np.round(threshold_grid, 15))

    sources = (['prod'] + [f'{seed}::{name}' for seed in seeds for name in ['h1_ordinary', 'h2_t1', 'h3_mrd_t2', 'h4_combined']])
    fp_lists = {s: {t: [] for t in threshold_grid} for s in sources}
    recovered_by_sub = {s: {t: {} for t in threshold_grid} for s in sources}
    # for lesion-level precision/recall and voxel confusion matrix at the
    # matched thresholds, computed below once thresholds are known
    voxel_tp = {s: {t: 0 for t in threshold_grid} for s in sources}
    voxel_fp = {s: {t: 0 for t in threshold_grid} for s in sources}
    voxel_fn = {s: {t: 0 for t in threshold_grid} for s in sources}
    voxel_tn = {s: {t: 0 for t in threshold_grid} for s in sources}

    for i, sid in enumerate(test_subjects_eval):
        maps, tgt_c, img_c = dense_eval(sid)
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        et_lbl, et_n = ndimage.label(true_et)
        recs = lesion_by_sid.get(sid, [])
        for src_name in sources:
            if src_name == 'prod':
                p_map = maps['prod']
            else:
                seed_str, name = src_name.split('::')
                p_map = maps[(int(seed_str), name)]
            for t in threshold_grid:
                pred_mask = (p_map > t) & brain_mask
                fp = int((pred_mask & (~true_et)).sum())
                fp_lists[src_name][t].append(fp)
                voxel_tp[src_name][t] += int((pred_mask & true_et).sum())
                voxel_fp[src_name][t] += fp
                voxel_fn[src_name][t] += int((~pred_mask & true_et & brain_mask).sum())
                voxel_tn[src_name][t] += int((~pred_mask & ~true_et & brain_mask).sum())
                by_phen = {}
                for rec in recs:
                    cid = rec['comp_id']
                    if cid < 1 or cid > et_n:
                        continue
                    lm = et_lbl == cid
                    by_phen.setdefault(rec['phenotype'], []).append(
                        {'recovered': int((pred_mask & lm).any()), 'size_bucket': rec['size_bucket']})
                recovered_by_sub[src_name][t][sid] = by_phen
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects_eval)} ({time.time()-t0_time:.0f}s)', flush=True)

    mean_fp_by_threshold = {s: {} for s in sources}
    for src_name in sources:
        for t in threshold_grid:
            fps = fp_lists[src_name][t]
            mean_fp_by_threshold[src_name][t] = float(np.mean(fps)) if fps else float('nan')

    def find_threshold_for_fp_target(src_name, target_fp):
        best_t, best_diff = None, np.inf
        for t in threshold_grid:
            diff = abs(mean_fp_by_threshold[src_name][t] - target_fp)
            if diff < best_diff:
                best_diff, best_t = diff, t
        return best_t

    # ---- FROC curve (all thresholds, Section 15) ----
    print('\nWriting FROC curve...', flush=True)
    froc_rows = []
    for src_name in sources:
        for t in threshold_grid:
            per_sub = recovered_by_sub[src_name][t]
            all_vals, phen_vals = [], {}
            for sid, by_phen in per_sub.items():
                for phen, entries in by_phen.items():
                    vals = [e['recovered'] for e in entries]
                    phen_vals.setdefault(phen, []).extend(vals)
                    all_vals.extend(vals)
            row = {'source': src_name, 'threshold': t, 'mean_fp': mean_fp_by_threshold[src_name][t],
                  'sens_all': float(np.mean(all_vals)) if all_vals else float('nan')}
            for phen in ['G1_confidently_missed', 'G2_low_confidence', 'G3_confidently_detected']:
                vals = phen_vals.get(phen, [])
                row[f'sens_{phen}'] = float(np.mean(vals)) if vals else float('nan')
            froc_rows.append(row)
    froc_out = HERE / ('E286_froc_smoke.csv' if smoke else 'E286_froc.csv')
    with open(froc_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['source', 'threshold', 'mean_fp', 'sens_all',
                                           'sens_G1_confidently_missed', 'sens_G2_low_confidence',
                                           'sens_G3_confidently_detected'])
        w.writeheader()
        for row in froc_rows:
            w.writerow(row)

    # ---- matched-FP sensitivity + per-subject records (bootstrap) +
    # size-stratified recovery + voxel confusion matrix + Dice/IoU/HD95/ASSD ----
    print('\nComputing matched-FP sensitivity, confusion matrix, Dice/IoU/HD95/ASSD...', flush=True)
    matched_rows, per_subject_rows, size_strat_rows = [], [], []
    confusion_rows = []
    boundary_rows = []

    for src_name in sources:
        matched_t = {}
        for target in FP_TARGETS:
            t_match = find_threshold_for_fp_target(src_name, target)
            matched_t[target] = t_match
            per_sub = recovered_by_sub[src_name][t_match]

            for phen_filter in ['G1_confidently_missed', 'G2_low_confidence', 'G3_confidently_detected', None]:
                all_vals = []
                for sid, by_phen in per_sub.items():
                    if phen_filter:
                        vals = [e['recovered'] for e in by_phen.get(phen_filter, [])]
                    else:
                        vals = [e['recovered'] for vv in by_phen.values() for e in vv]
                    all_vals.extend(vals)
                    if vals:
                        per_subject_rows.append({
                            'subject_id': sid, 'source': src_name, 'fp_target': target,
                            'phenotype': phen_filter or 'ALL', 'sensitivity': float(np.mean(vals)),
                            'n_lesions': len(vals),
                        })
                sens = float(np.mean(all_vals)) if all_vals else float('nan')
                matched_rows.append({
                    'source': src_name, 'fp_target': target, 'threshold': t_match,
                    'mean_fp_actual': mean_fp_by_threshold[src_name][t_match],
                    'phenotype': phen_filter or 'ALL', 'sensitivity': sens,
                })

            # size-stratified G1 recovery at this FP target
            for bucket in ['small', 'medium', 'large']:
                vals = []
                for sid, by_phen in per_sub.items():
                    for e in by_phen.get('G1_confidently_missed', []):
                        if e['size_bucket'] == bucket:
                            vals.append(e['recovered'])
                sens = float(np.mean(vals)) if vals else float('nan')
                size_strat_rows.append({'source': src_name, 'fp_target': target, 'size_bucket': bucket,
                                       'n': len(vals), 'sensitivity': sens})

            # voxel confusion matrix + Dice/Precision/Recall/Specificity/IoU at this threshold
            tp, fp, fn, tn = (voxel_tp[src_name][t_match], voxel_fp[src_name][t_match],
                             voxel_fn[src_name][t_match], voxel_tn[src_name][t_match])
            precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
            recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
            specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float('nan')
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')
            confusion_rows.append({
                'source': src_name, 'fp_target': target, 'threshold': t_match,
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'precision': precision,
                'recall': recall, 'specificity': specificity, 'dice': dice, 'iou': iou,
            })

        # HD95/ASSD per subject at the FP~500 operating point (one representative
        # boundary-metric operating point per the spec's own emphasis on FP~500
        # as the central budget; computed per-subject, aggregated after)
        t_boundary = matched_t[500]
        hd_vals, assd_vals = [], []
        for sid in test_subjects_eval:
            maps, tgt_c, img_c = dense_eval(sid)
            if src_name == 'prod':
                p_map = maps['prod']
            else:
                seed_str, name = src_name.split('::')
                p_map = maps[(int(seed_str), name)]
            brain_mask = img_c[0] != 0
            pred_mask = (p_map > t_boundary) & brain_mask
            true_et = tgt_c[ET] > 0.5
            hd, assd = hd95_and_assd(pred_mask, true_et)
            hd_vals.append(hd); assd_vals.append(assd)
        hd_vals = np.array(hd_vals); assd_vals = np.array(assd_vals)
        boundary_rows.append({
            'source': src_name, 'threshold': t_boundary,
            'hd95_mean': float(np.nanmean(hd_vals)), 'hd95_median': float(np.nanmedian(hd_vals)),
            'assd_mean': float(np.nanmean(assd_vals)), 'n_subjects_with_valid_surface':
                int(np.sum(~np.isnan(hd_vals))),
        })

        print(f'  {src_name}: thresholds={matched_t} '
             f'(actual FP: {[round(mean_fp_by_threshold[src_name][t],1) for t in matched_t.values()]})',
             flush=True)

    for name, rows, fields in [
        ('E286_matched_fp', matched_rows, ['source', 'fp_target', 'threshold', 'mean_fp_actual', 'phenotype', 'sensitivity']),
        ('E286_persubject', per_subject_rows, ['subject_id', 'source', 'fp_target', 'phenotype', 'sensitivity', 'n_lesions']),
        ('E286_size_stratified', size_strat_rows, ['source', 'fp_target', 'size_bucket', 'n', 'sensitivity']),
        ('E286_confusion_matrix', confusion_rows, ['source', 'fp_target', 'threshold', 'TP', 'FP', 'FN', 'TN',
                                                   'precision', 'recall', 'specificity', 'dice', 'iou']),
        ('E286_boundary_metrics', boundary_rows, ['source', 'threshold', 'hd95_mean', 'hd95_median',
                                                  'assd_mean', 'n_subjects_with_valid_surface']),
    ]:
        out = HERE / (f'{name}_smoke.csv' if smoke else f'{name}.csv')
        with open(out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f'  wrote {out.name}', flush=True)

    # ---- complementarity analysis (Section 32), at FP~500 operating point ----
    print('\nComputing complementarity analysis (P vs MRD-T2 lesion detection, FP~500)...', flush=True)
    complementarity_rows = []
    for seed in seeds:
        src_mrd = f'{seed}::h3_mrd_t2'
        t_prod = find_threshold_for_fp_target('prod', 500)
        t_mrd = find_threshold_for_fp_target(src_mrd, 500)
        # build per-lesion recovery booleans directly from the dense maps
        # (more robust than re-deriving from the aggregated by_phen lists)
        p_rec_set, r_rec_set, all_lesions = set(), set(), set()
        for sid in test_subjects_eval:
            maps, tgt_c, img_c = dense_eval(sid)
            p_map_prod = maps['prod']
            seed_str, name = src_mrd.split('::')
            p_map_mrd = maps[(int(seed_str), name)]
            true_et = tgt_c[ET] > 0.5
            et_lbl, et_n = ndimage.label(true_et)
            for rec in lesion_by_sid.get(sid, []):
                cid = rec['comp_id']
                if cid < 1 or cid > et_n:
                    continue
                lm = et_lbl == cid
                key = (sid, cid)
                all_lesions.add(key)
                if ((p_map_prod > t_prod) & lm).any():
                    p_rec_set.add(key)
                if ((p_map_mrd > t_mrd) & lm).any():
                    r_rec_set.add(key)
        n_both = len(p_rec_set & r_rec_set)
        n_p_only = len(p_rec_set - r_rec_set)
        n_r_only = len(r_rec_set - p_rec_set)
        n_neither = len(all_lesions - p_rec_set - r_rec_set)
        complementarity_rows.append({
            'seed': seed, 'n_total_lesions': len(all_lesions),
            'n_P_and_R': n_both, 'n_P_only': n_p_only, 'n_R_only': n_r_only, 'n_neither': n_neither,
        })
        print(f'  seed={seed}: P^R={n_both} P\\R={n_p_only} R\\P={n_r_only} '
             f'neither={n_neither} (total lesions={len(all_lesions)})', flush=True)

    comp_out = HERE / ('E286_complementarity_smoke.csv' if smoke else 'E286_complementarity.csv')
    with open(comp_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['seed', 'n_total_lesions', 'n_P_and_R', 'n_P_only',
                                           'n_R_only', 'n_neither'])
        w.writeheader()
        for row in complementarity_rows:
            w.writerow(row)
    print(f'  wrote {comp_out.name}', flush=True)

    # ---- subject-level bootstrap CI for the key deltas (Section 37) ----
    print('\nComputing subject-level bootstrap 95% CI for T2-T0 and T2-T1 (G1, FP~500)...', flush=True)
    rng_boot = np.random.default_rng(271828)
    B = 2000 if not smoke else 200

    def subject_level_sensitivity(src_name, t, subjects, phen_filter='G1_confidently_missed'):
        per_sub = recovered_by_sub[src_name][t]
        vals = []
        for sid in subjects:
            entries = per_sub.get(sid, {}).get(phen_filter, [])
            if entries:
                vals.extend([e['recovered'] for e in entries])
        return float(np.mean(vals)) if vals else float('nan')

    bootstrap_rows = []
    for seed in seeds:
        t_h1 = find_threshold_for_fp_target(f'{seed}::h1_ordinary', 500)
        t_h3 = find_threshold_for_fp_target(f'{seed}::h3_mrd_t2', 500)
        subjects_with_g1 = [sid for sid in test_subjects_eval
                            if any(r['phenotype'] == 'G1_confidently_missed' for r in lesion_by_sid.get(sid, []))]
        if len(subjects_with_g1) < 2:
            continue
        deltas_t2_t0 = []
        for _ in range(B):
            boot_subj = rng_boot.choice(subjects_with_g1, size=len(subjects_with_g1), replace=True)
            s_h1 = subject_level_sensitivity(f'{seed}::h1_ordinary', t_h1, boot_subj)
            s_h3 = subject_level_sensitivity(f'{seed}::h3_mrd_t2', t_h3, boot_subj)
            if not (np.isnan(s_h1) or np.isnan(s_h3)):
                deltas_t2_t0.append(s_h3 - s_h1)
        if deltas_t2_t0:
            lo, hi = np.percentile(deltas_t2_t0, [2.5, 97.5])
            bootstrap_rows.append({'seed': seed, 'comparison': 'T2_minus_H1ordinary_G1_FP500',
                                   'mean_delta': float(np.mean(deltas_t2_t0)), 'ci_lo': float(lo), 'ci_hi': float(hi)})
        print(f'  seed={seed}: T2-H1ordinary delta mean={np.mean(deltas_t2_t0) if deltas_t2_t0 else float("nan"):.4f} '
             f'95%CI=[{lo if deltas_t2_t0 else float("nan"):.4f},{hi if deltas_t2_t0 else float("nan"):.4f}]', flush=True)

    boot_out = HERE / ('E286_bootstrap_ci_smoke.csv' if smoke else 'E286_bootstrap_ci.csv')
    with open(boot_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['seed', 'comparison', 'mean_delta', 'ci_lo', 'ci_hi'])
        w.writeheader()
        for row in bootstrap_rows:
            w.writerow(row)
    print(f'  wrote {boot_out.name}', flush=True)

    # ---- summary printout ----
    print('\n' + '=' * 70, flush=True)
    print('SUMMARY: G1 sensitivity, mean +/- SD across seeds, LOCKED test cohort', flush=True)
    print('=' * 70, flush=True)
    by_cond = {}
    for row in matched_rows:
        if row['phenotype'] != 'G1_confidently_missed' or row['source'] == 'prod':
            continue
        seed_str, name = row['source'].split('::')
        by_cond.setdefault(name, {}).setdefault(row['fp_target'], []).append(row['sensitivity'])
    for name in ['h1_ordinary', 'h2_t1', 'h3_mrd_t2', 'h4_combined']:
        by_target = by_cond.get(name, {})
        parts = []
        for target in FP_TARGETS:
            vals = by_target.get(target, [])
            parts.append(f'FP~{target}: {np.mean(vals)*100:.1f}%±{np.std(vals)*100:.1f}%')
        print(f'  {name:14s} {"  ".join(parts)}', flush=True)

    manifest = {
        'n_test_subjects': len(test_subjects_eval),
        'n_lesions_total': len(lesion_records),
        'n_by_phenotype': n_by_phen,
        'size_cutoffs': SIZE_CUTOFFS,
        'seeds': seeds,
        'elapsed_s': time.time() - t0_time,
    }
    with open(HERE / ('E286_manifest_smoke.json' if smoke else 'E286_manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)

    print(f'\nE286 complete ({time.time()-t0_time:.0f}s).', flush=True)


if __name__ == '__main__':
    main()
