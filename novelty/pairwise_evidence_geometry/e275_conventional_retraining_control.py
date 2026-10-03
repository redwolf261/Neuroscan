"""E275 -- Conventional-Retraining Control, per the user's exact spec.
Direct follow-up to E274, which found R1-B recovers a real, FP-budget-
scaling fraction of non-circularly-defined "confidently missed" (G1)
and "low-confidence" (G2) ET lesions that production recovers 0% of,
reproducible across 2 independent R1-B seeds (Jaccard=0.955). The
user's explicit next question is the obvious reviewer objection this
result invites: "is R1-B discovering something special about D1, or
did you simply retrain a DIFFERENT classifier and get a better
operating point?" -- i.e. is R1-B's specific ingredient (curated
lesion+shell+distant-background negative sampling, established back
in E257-B) actually doing something beyond "any retrained head on the
same frozen D1 would do."

THREE READOUTS on the SAME frozen D1 (architecture is IDENTICAL for
B and C -- both are a 32-dim linear map + sigmoid, mathematically a
1x1 convolution; the only real difference is TRAINING DATA
DISTRIBUTION, isolating exactly the variable this control is meant to
test, per explicit user scoping this session):
  A: production            -- frozen, no retraining (w_P, b_P)
  B: R1-B                  -- E257-B's own curated negative sampling
                               (lesion + local shell + distant
                               background), reused unchanged
  C: conventional head      -- NEW. Same architecture, same det_train
                               subjects, same LogisticRegression/
                               class_weight='balanced' config as B, but
                               trained on ALL ET voxels (positives) +
                               a UNIFORM RANDOM subsample of non-ET
                               voxels across the whole brain (negatives,
                               sized to roughly match B's total pool),
                               i.e. the sampling a conventional
                               retraining would use, with NO deliberate
                               near/far negative curation.

EVALUATION: EXACTLY E274's own evaluation, unchanged, per explicit
user instruction ("do not change the evaluation now"): the same
non-circular G1/G2/G3 phenotype definition (production probability +
ground truth only), the same logit-spaced threshold grid, the same
matched-mean-FP-per-subject operating points (100/500/1000), the same
test subjects (det_test), the same lesion-level sensitivity primary
endpoint.

THREE-WAY DECISION (per explicit user framing):
  Outcome 1: R1-B ~= conventional head -> the phenomenon is "the
             original production head is poorly optimized," useful
             but less distinctive.
  Outcome 2: R1-B >> conventional head -> R1-B's specific negative-
             sampling choice is doing something a naive retraining
             does not -- a real methodological contribution.
  Outcome 3: BOTH replacement heads dramatically beat production ->
             the decoder representation was never the bottleneck, the
             deployed readout itself was -- still valuable, but does
             not single out R1-B's particular construction.
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
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402

ET, TC, WT = 0, 1, 2
STRUCT = ndimage.generate_binary_structure(3, 1)
EPS = 1e-12
TAU_LOW, TAU_HIGH = TAU, 0.9
FP_TARGETS = [100, 500, 1000]


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    model = load_model(dev)
    w_et, b_et = get_w_prod(model)
    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    if smoke:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']][:30]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
        conv_subjects = sorted(splits['det_train'])[:30]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])
        conv_subjects = sorted(splits['det_train'])

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng_global = np.random.default_rng(999)
    t0 = time.time()

    # ---- fit R1-B (B), identical to E257-B/E274 ----
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
        return lesion_pool, neg_pool

    print('Fitting R1-B (B, curated negatives)...', flush=True)
    r1b_lesion_pool, r1b_neg_pool = fit_r1b(999)
    r1b_pool_size = len(r1b_lesion_pool) + len(r1b_neg_pool)
    X_r1b = np.concatenate([r1b_lesion_pool, r1b_neg_pool]).astype(np.float64)
    y_r1b = np.concatenate([np.ones(len(r1b_lesion_pool)), np.zeros(len(r1b_neg_pool))])
    clf_r1b = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_r1b, y_r1b)
    w_r1b = clf_r1b.coef_[0].astype(np.float64)
    b_r1b = float(clf_r1b.intercept_[0])
    print(f'  R1-B pool size={r1b_pool_size} ({len(r1b_lesion_pool)} ET, {len(r1b_neg_pool)} curated neg) '
         f'({time.time()-t0:.0f}s)', flush=True)

    # ---- fit conventional head (C): ALL ET voxels + uniform-random
    # non-ET subsample, same TRAIN subjects, budget matched to R1-B's
    # total pool size ----
    print('\nFitting conventional head (C, uniform-random negatives)...', flush=True)
    conv_et_pool, conv_neg_candidates = [], []
    for sid in conv_subjects:
        if sid not in sid_to_idx:
            continue
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0].cpu().numpy()
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        et_idx = np.where(true_et)
        if len(et_idx[0]) > 0:
            conv_et_pool.append(d1[:, et_idx[0], et_idx[1], et_idx[2]].T)
        non_et_mask = brain_mask & (~true_et)
        non_et_idx = np.where(non_et_mask)
        if len(non_et_idx[0]) > 0:
            # subsample a modest number per subject up front, final
            # match to R1-B's neg pool size happens after pooling
            n_sample = min(500, len(non_et_idx[0]))
            sel = rng_global.choice(len(non_et_idx[0]), size=n_sample, replace=False)
            coords = (non_et_idx[0][sel], non_et_idx[1][sel], non_et_idx[2][sel])
            conv_neg_candidates.append(d1[:, coords[0], coords[1], coords[2]].T)
        del d1, img_t, stages
        torch.cuda.empty_cache()
    conv_et_pool = np.concatenate(conv_et_pool).astype(np.float64)
    conv_neg_candidates = np.concatenate(conv_neg_candidates).astype(np.float64)
    target_neg_size = len(r1b_neg_pool)
    if len(conv_neg_candidates) > target_neg_size:
        sel = rng_global.choice(len(conv_neg_candidates), size=target_neg_size, replace=False)
        conv_neg_pool = conv_neg_candidates[sel]
    else:
        conv_neg_pool = conv_neg_candidates
    print(f'  conventional pool: {len(conv_et_pool)} ET voxels (ALL, dense), '
         f'{len(conv_neg_pool)} uniform-random non-ET voxels (target matched to R1-B\'s {len(r1b_neg_pool)}) '
         f'({time.time()-t0:.0f}s)', flush=True)

    X_conv = np.concatenate([conv_et_pool, conv_neg_pool])
    y_conv = np.concatenate([np.ones(len(conv_et_pool)), np.zeros(len(conv_neg_pool))])
    clf_conv = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_conv, y_conv)
    w_conv = clf_conv.coef_[0].astype(np.float64)
    b_conv = float(clf_conv.intercept_[0])
    print(f'Conventional head fit ({time.time()-t0:.0f}s)', flush=True)

    w_p_t = torch.from_numpy(w_et.astype(np.float64)).float().to(dev)
    w_r1b_t = torch.from_numpy(w_r1b).float().to(dev)
    w_conv_t = torch.from_numpy(w_conv).float().to(dev)
    b_p = float(b_et)

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
        with torch.no_grad():
            p_prod = torch.sigmoid(flat @ w_p_t + b_p).reshape(D, H, W).cpu().numpy()
            p_r1b = torch.sigmoid(flat @ w_r1b_t + b_r1b).reshape(D, H, W).cpu().numpy()
            p_conv = torch.sigmoid(flat @ w_conv_t + b_conv).reshape(D, H, W).cpu().numpy()
        del d1, flat, img_t, stages
        torch.cuda.empty_cache()
        result = (p_prod, p_r1b, p_conv, tgt_c, img_c)
        dense_cache[sid] = result
        return result

    test_subjects = sorted(splits['det_test'])
    if smoke:
        test_subjects = test_subjects[:15]

    print(f'\nBuilding non-circular phenotype groups on {len(test_subjects)} TEST subjects '
         f'(IDENTICAL to E274, production + ground truth ONLY)...', flush=True)
    lesion_records = []
    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        p_prod, p_r1b, p_conv, tgt_c, img_c = dense_eval(sid)
        true_et = tgt_c[ET] > 0.5
        et_lbl, et_n = ndimage.label(true_et)
        for cid in range(1, et_n + 1):
            lesion_mask = et_lbl == cid
            if int(lesion_mask.sum()) < MIN_VOX:
                continue
            p_max = float(p_prod[lesion_mask].max())
            if p_max < TAU_LOW:
                phenotype = 'G1_confidently_missed'
            elif p_max < TAU_HIGH:
                phenotype = 'G2_low_confidence'
            else:
                phenotype = 'G3_confidently_detected'
            lesion_records.append({'subject_id': sid, 'comp_id': cid, 'phenotype': phenotype,
                                   'size': int(lesion_mask.sum()), 'p_max_prod': p_max})
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    n_by_phen = {}
    for rec in lesion_records:
        n_by_phen[rec['phenotype']] = n_by_phen.get(rec['phenotype'], 0) + 1
    print(f'\nPhenotype group sizes: {n_by_phen}', flush=True)

    # ---- threshold sweep: production, R1-B, conventional ----
    print('\nSweeping thresholds (IDENTICAL grid to E274, logit-spaced near 1.0)...', flush=True)
    threshold_grid = np.concatenate([
        np.linspace(0.01, 0.1, 10),
        np.linspace(0.1, 0.9, 20),
        1 - 1 / (1 + np.exp(np.linspace(2, 30, 60))),
    ])
    threshold_grid = np.unique(np.round(threshold_grid, 8))

    sources = ['prod', 'r1b', 'conv']
    fp_lists = {s: {t: [] for t in threshold_grid} for s in sources}

    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        p_prod, p_r1b, p_conv, tgt_c, img_c = dense_eval(sid)
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        for src_name, p_map in [('prod', p_prod), ('r1b', p_r1b), ('conv', p_conv)]:
            for t in threshold_grid:
                mask = (p_map > t) & brain_mask
                fp = int((mask & (~true_et)).sum())
                fp_lists[src_name][t].append(fp)
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)

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

    matched_thresholds = {}
    for src_name in sources:
        matched_thresholds[src_name] = {}
        for target in FP_TARGETS:
            matched_thresholds[src_name][target] = find_threshold_for_fp_target(src_name, target)
        print(f'{src_name}: matched thresholds = {matched_thresholds[src_name]} '
             f'(actual mean FP: {[round(mean_fp_by_threshold[src_name][t],1) for t in matched_thresholds[src_name].values()]})',
             flush=True)

    lesion_by_sid = {}
    for rec in lesion_records:
        lesion_by_sid.setdefault(rec['subject_id'], []).append(rec)

    def sensitivity_at_threshold(src_name, t, phenotype_filter=None):
        recs = []
        for sid in test_subjects:
            if sid not in sid_to_idx:
                continue
            p_prod, p_r1b, p_conv, tgt_c, img_c = dense_eval(sid)
            p_map = {'prod': p_prod, 'r1b': p_r1b, 'conv': p_conv}[src_name]
            et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
            for rec in lesion_by_sid.get(sid, []):
                if phenotype_filter and rec['phenotype'] != phenotype_filter:
                    continue
                cid = rec['comp_id']
                if cid < 1 or cid > et_n:
                    continue
                lm = et_lbl == cid
                recs.append(int(((p_map > t) & lm).any()))
        return float(np.mean(recs)) if recs else float('nan')

    print('\nComputing matched-FP sensitivity by phenotype...', flush=True)
    matched_results = []
    for src_name in sources:
        for target in FP_TARGETS:
            t_match = matched_thresholds[src_name][target]
            for phen in ['G1_confidently_missed', 'G2_low_confidence', 'G3_confidently_detected', None]:
                sens = sensitivity_at_threshold(src_name, t_match, phen)
                matched_results.append({
                    'readout': src_name, 'fp_target': target, 'threshold': t_match,
                    'mean_fp_actual': mean_fp_by_threshold[src_name][t_match],
                    'phenotype': phen or 'ALL', 'sensitivity': sens,
                })
                print(f'  {src_name} FP~{target} (t={t_match:.4f}): {phen or "ALL":28s} '
                     f'sensitivity={sens:.4f}', flush=True)

    out = HERE / ('E275_matched_fp_smoke.csv' if smoke else 'E275_matched_fp.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['readout', 'fp_target', 'threshold', 'mean_fp_actual',
                                           'phenotype', 'sensitivity'])
        w.writeheader()
        for row in matched_results:
            w.writerow(row)

    print(f'\nE275 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
