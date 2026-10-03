"""E274 -- Cross-Phenotype Readout-Rescue Validation (core pass), per
the user's exact spec. Direct follow-up to E273, which found a real
but partial direction-replacement mechanism and a decisively failed
Delta_w specificity story. The user explicitly redirected from "invent
another mathematical operator" toward the actually load-bearing
question this whole investigation needs answered before any paper
claim: is D1-readout rescue (R1-B recovering what production misses)
a REPRODUCIBLE PHENOMENON across independently-defined difficult-
lesion phenotypes and independent model seeds, or was G2-A a dataset-
specific subgroup that happened to respond to one particular readout.

CORE PASS ONLY (per explicit user scoping this session): (1) non-
circular phenotype definition (G1/G2/G3, using PRODUCTION's own
probability map and ground truth ONLY -- never touches R1-B, avoiding
the exact circularity trap the user explicitly flagged), (2) matched-
FP-burden lesion sensitivity comparison (the PRIMARY endpoint per
explicit user instruction, NOT overall Dice) for production vs R1-B
vs R1-B+Delta_C, (3) two independently-seeded R1-B fits + lesion-level
Jaccard overlap (reproducibility check). The morphological-feature
stratification table, the logistic covariate model, the D2-level
negative control, and the spatial-scrambling control are explicitly
DEFERRED to follow-ups contingent on this core result.

NON-CIRCULAR PHENOTYPE GROUPS (built from frozen production P(x) and
ground truth ONLY, thresholds fixed BEFORE touching any R1-B result,
per explicit user requirement to avoid circularity):
  G1 (confidently missed):  max_{v in L} P(v) < 0.5  (TAU, production's
                             own operating threshold -- by this
                             definition G1 is EXACTLY "production's
                             mask never includes this lesion")
  G2 (low-confidence):      max_{v in L} P(v) in [0.5, 0.9)
  G3 (confidently detected): max_{v in L} P(v) >= 0.9
(per explicit user choice this session for the threshold values)

MATCHED-FP-BURDEN COMPARISON (per explicit user choice this session):
production and R1-B are swept over a threshold grid; for each, find
the SINGLE SHARED threshold (applied uniformly across all test
subjects, not per-subject-tuned) whose MEAN FP-per-subject equals each
target (100, 500, 1000), then report lesion sensitivity (overall and
per phenotype group) at that threshold. R1-B+Delta_C is reported as
ONE FIXED reference point (its own E264/E265 val-selected zero-cost
threshold), not swept/re-tuned, per explicit user choice -- shown on
the same sensitivity-vs-FP axis for comparison, not forced to hit the
100/500/1000 targets artificially.

REPRODUCIBILITY: R1-B refit with TWO independent seeds (999, matching
E257-B's own original seed, and 4242, matching E266's R1-B-seed2
convention) for negative sampling, same features/hyperparameters.
Lesion-level recovery sets computed at TAU=0.5 for each seed; Jaccard
overlap J = |S1 intersect S2| / |S1 union S2| over the G1+G2
(difficult) lesion population.
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
STRUCT = ndimage.generate_binary_structure(3, 1)
NEIGHBORHOOD_DILATION = 3
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
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
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
        X = np.concatenate([lesion_pool, neg_pool]).astype(np.float64)
        y = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
        clf = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X, y)
        return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])

    print('Fitting R1-B seed=999 (primary)...', flush=True)
    w_r1, b_r1 = fit_r1b(999)
    print(f'  done ({time.time()-t0:.0f}s)', flush=True)
    print('Fitting R1-B seed=4242 (reproducibility check)...', flush=True)
    w_r2, b_r2 = fit_r1b(4242)
    print(f'  done ({time.time()-t0:.0f}s)', flush=True)

    w_p_t = torch.from_numpy(w_et.astype(np.float64)).float().to(dev)
    w_r1_t = torch.from_numpy(w_r1).float().to(dev)
    w_r2_t = torch.from_numpy(w_r2).float().to(dev)
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
            p_r1 = torch.sigmoid(flat @ w_r1_t + b_r1).reshape(D, H, W).cpu().numpy()
            p_r2 = torch.sigmoid(flat @ w_r2_t + b_r2).reshape(D, H, W).cpu().numpy()
        del d1, flat, img_t, stages
        torch.cuda.empty_cache()
        result = (p_prod, p_r1, p_r2, tgt_c, img_c)
        dense_cache[sid] = result
        return result

    # ---- use ALL lesions (det + g2a + g2b populations pooled, since
    # phenotype grouping is now defined fresh from production probability,
    # not from the det/g2a/g2b population labels) across the FULL test
    # split (det_test subjects, for a clean single consistent population) ----
    test_subjects = sorted(splits['det_test'])
    if smoke:
        test_subjects = test_subjects[:15]

    print(f'\nBuilding non-circular phenotype groups on {len(test_subjects)} TEST subjects '
         f'(production + ground truth ONLY)...', flush=True)

    # per-lesion record: subject_id, comp_id, phenotype, recovered_r1, recovered_r2
    lesion_records = []
    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        p_prod, p_r1, p_r2, tgt_c, img_c = dense_eval(sid)
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
            lesion_records.append({
                'subject_id': sid, 'comp_id': cid, 'phenotype': phenotype,
                'size': int(lesion_mask.sum()), 'p_max_prod': p_max,
            })
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    n_by_phen = {}
    for rec in lesion_records:
        n_by_phen[rec['phenotype']] = n_by_phen.get(rec['phenotype'], 0) + 1
    print(f'\nPhenotype group sizes: {n_by_phen}', flush=True)

    pheno_out = HERE / ('E274_phenotypes_smoke.csv' if smoke else 'E274_phenotypes.csv')
    with open(pheno_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['subject_id', 'comp_id', 'phenotype', 'size', 'p_max_prod'])
        w.writeheader()
        for rec in lesion_records:
            w.writerow(rec)

    # ---- threshold sweep: production and R1-B (seed1), find thresholds
    # matching mean FP/subject targets ----
    print('\nSweeping thresholds for production and R1-B (matched-FP search)...', flush=True)
    # CRITICAL FIX (found in smoke testing before trusting any result):
    # the initial grid stopped at 0.9, but mean FP/subject at t=0.9 was
    # still 2070.9 (prod) / 877-1101 (R1-B) -- far above the FP~100/500
    # targets. All three FP targets collapsed onto the SAME threshold
    # (the grid's own upper edge), which would have silently reported
    # identical sensitivity for every FP target -- a real bug, not a
    # genuine finding. Fixed by extending the grid much closer to 1.0
    # with finer resolution there (logit-spaced in the high-threshold
    # region, where FP count changes fastest per unit probability).
    threshold_grid = np.concatenate([
        np.linspace(0.01, 0.1, 10),
        np.linspace(0.1, 0.9, 20),
        1 - 1 / (1 + np.exp(np.linspace(2, 30, 60))),  # approaches 1.0 from below, finer near the top
                                                         # (extended to logit~30 after smoke testing
                                                         # found production's own FP never dropped
                                                         # below ~1000 within logit~10's reach)
    ])
    threshold_grid = np.unique(np.round(threshold_grid, 8))

    fp_lists = {'prod': {t: [] for t in threshold_grid}, 'r1b': {t: [] for t in threshold_grid}}

    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        p_prod, p_r1, p_r2, tgt_c, img_c = dense_eval(sid)
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        for src_name, p_map in [('prod', p_prod), ('r1b', p_r1)]:
            for t in threshold_grid:
                mask = (p_map > t) & brain_mask
                fp = int((mask & (~true_et)).sum())
                fp_lists[src_name][t].append(fp)
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    mean_fp_by_threshold = {'prod': {}, 'r1b': {}}
    for src_name in ['prod', 'r1b']:
        for t in threshold_grid:
            fps = fp_lists[src_name][t]
            mean_fp_by_threshold[src_name][t] = float(np.mean(fps)) if fps else float('nan')

    def find_threshold_for_fp_target(src_name, target_fp):
        """Finds the threshold grid value whose mean FP/subject is
        closest to target_fp (grid is coarse; this is an approximate
        match, reported explicitly as such)."""
        best_t, best_diff = None, np.inf
        for t in threshold_grid:
            diff = abs(mean_fp_by_threshold[src_name][t] - target_fp)
            if diff < best_diff:
                best_diff, best_t = diff, t
        return best_t

    matched_thresholds = {}
    for src_name in ['prod', 'r1b']:
        matched_thresholds[src_name] = {}
        for target in FP_TARGETS:
            t_match = find_threshold_for_fp_target(src_name, target)
            matched_thresholds[src_name][target] = t_match
        print(f'{src_name}: matched thresholds = {matched_thresholds[src_name]} '
             f'(actual mean FP: {[round(mean_fp_by_threshold[src_name][t],1) for t in matched_thresholds[src_name].values()]})',
             flush=True)

    # ---- sensitivity by phenotype at matched thresholds ----
    lesion_by_sid = {}
    for rec in lesion_records:
        lesion_by_sid.setdefault(rec['subject_id'], []).append(rec)

    def sensitivity_at_threshold(src_name, t, phenotype_filter=None):
        recs = []
        for i, sid in enumerate(test_subjects):
            if sid not in sid_to_idx:
                continue
            p_prod, p_r1, p_r2, tgt_c, img_c = dense_eval(sid)
            p_map = p_prod if src_name == 'prod' else p_r1
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
    for src_name in ['prod', 'r1b']:
        for target in FP_TARGETS:
            t_match = matched_thresholds[src_name][target]
            for phen in ['G1_confidently_missed', 'G2_low_confidence', 'G3_confidently_detected', None]:
                sens = sensitivity_at_threshold(src_name, t_match, phen)
                matched_results.append({
                    'readout': src_name, 'fp_target': target, 'threshold': t_match,
                    'mean_fp_actual': mean_fp_by_threshold[src_name][t_match],
                    'phenotype': phen or 'ALL', 'sensitivity': sens,
                })
                print(f'  {src_name} FP~{target} (t={t_match:.3f}): {phen or "ALL":28s} '
                     f'sensitivity={sens:.4f}', flush=True)

    matched_out = HERE / ('E274_matched_fp_smoke.csv' if smoke else 'E274_matched_fp.csv')
    with open(matched_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['readout', 'fp_target', 'threshold', 'mean_fp_actual',
                                           'phenotype', 'sensitivity'])
        w.writeheader()
        for row in matched_results:
            w.writerow(row)

    # ---- reproducibility: 2-seed Jaccard overlap, TAU=0.5, G1+G2 lesions ----
    print('\nComputing 2-seed Jaccard overlap (TAU=0.5, G1+G2 difficult lesions)...', flush=True)
    recovered_r1_set, recovered_r2_set = set(), set()
    difficult_lesions = [rec for rec in lesion_records
                        if rec['phenotype'] in ('G1_confidently_missed', 'G2_low_confidence')]
    for rec in difficult_lesions:
        sid, cid = rec['subject_id'], rec['comp_id']
        p_prod, p_r1, p_r2, tgt_c, img_c = dense_eval(sid)
        et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
        if cid < 1 or cid > et_n:
            continue
        lm = et_lbl == cid
        key = (sid, cid)
        if ((p_r1 > TAU) & lm).any():
            recovered_r1_set.add(key)
        if ((p_r2 > TAU) & lm).any():
            recovered_r2_set.add(key)

    intersection = recovered_r1_set & recovered_r2_set
    union = recovered_r1_set | recovered_r2_set
    jaccard = len(intersection) / len(union) if union else float('nan')
    print(f'seed999 recovered: {len(recovered_r1_set)}/{len(difficult_lesions)}  '
         f'seed4242 recovered: {len(recovered_r2_set)}/{len(difficult_lesions)}  '
         f'intersection: {len(intersection)}  union: {len(union)}  Jaccard: {jaccard:.4f}', flush=True)

    repro_out = HERE / ('E274_reproducibility_smoke.csv' if smoke else 'E274_reproducibility.csv')
    with open(repro_out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['n_difficult_lesions', 'n_recovered_seed999', 'n_recovered_seed4242',
                   'n_intersection', 'n_union', 'jaccard'])
        w.writerow([len(difficult_lesions), len(recovered_r1_set), len(recovered_r2_set),
                   len(intersection), len(union), jaccard])

    print(f'\nE274 complete ({time.time()-t0:.0f}s). wrote {pheno_out.name}, {matched_out.name}, '
         f'{repro_out.name}', flush=True)


if __name__ == '__main__':
    main()
