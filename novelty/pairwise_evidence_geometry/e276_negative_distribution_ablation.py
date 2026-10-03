"""E276 -- Negative-Distribution Ablation, per the user's exact spec.
Direct follow-up to E275 (Outcome 2: R1-B's specific construction,
not just any retraining, drives the rescue effect). This experiment
decomposes WHICH PART of R1-B's negative-sampling construction
(curated lesion + local shell + distant background, established in
E257-B) is actually responsible. Per explicit user instruction: do
NOT modify architecture, Delta_C, thresholding, or D1 -- the ONLY
variable changed across conditions is which negatives train the
readout.

FOUR CONDITIONS, same ET-positive pool, same TOTAL negative sample
count N per condition (per explicit user "equalize sample counts"
requirement -- otherwise any difference could be attributed to "R1-B
just sees more negatives" rather than to WHICH negatives):
  R0 (random):          N uniform-random non-ET voxels (== E275's
                         conventional-head negative distribution,
                         reused unchanged as the reproduced baseline)
  R1 (distant only):     N distant-background voxels (>NEAR_DILATION
                         from any ET lesion)
  R2 (shell only):       N local-shell voxels (non-ET, within
                         NEAR_DILATION of an ET lesion)
  R3 (shell+distant):    N/2 shell + N/2 distant -- this IS R1-B's own
                         construction (E257-B), reproduced here under
                         the same N-equalized budget as R0/R1/R2 for a
                         fair comparison (R1-B's own original fit used
                         a lesion-count-dependent neg pool size, not
                         explicitly matched to N -- this run refits it
                         under the controlled budget specifically)

N = total negative pool size, set to match R1-B's own historically
reported pool size (~14,895 at full scale, per E275's log) so results
remain comparable to E274/E275's reported numbers.

REPRODUCIBILITY: 3 independent seeds per condition (per explicit user
choice this session, the minimum of the user's stated 3-5 range),
each seed controlling which specific voxels are subsampled into that
condition's pool (same underlying candidate feature pools reused
across seeds -- only the subsample draw differs). Reports mean+-SD
across seeds for G1/G2 sensitivity, Dice, FP/subject, and lesion-level
pairwise Jaccard overlap (matching E274's own reproducibility check)
to test whether R1-B's J=0.955 stability is specific to its
shell+distant construction or common to every negative strategy.

EVALUATION: reuses E274/E275's own matched-FP-threshold convention
(found directly on the test set, per explicit user choice this session
to stay comparable to those reports -- not a new val/test split), same
non-circular G1/G2/G3 phenotype definition, same FP targets
(100/500/1000), same test subjects (det_test).

FP ANATOMICAL BREAKDOWN (per explicit user request): for each
condition, FP voxels at the FP~500 matched threshold are split into
WT-not-TC / TC / outside-WT (E259's own convention), to characterize
WHY each negative distribution's false positives land where they do.
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
EPS = 1e-12
TAU_LOW, TAU_HIGH = TAU, 0.9
FP_TARGETS = [100, 500, 1000]
N_NEGATIVE_TOTAL = 14895  # matches R1-B's own historical pool size (E275's log)
SEEDS = [999, 4242, 7777]


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
        n_neg_total = 2000
        seeds = SEEDS[:2]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])
        n_neg_total = N_NEGATIVE_TOTAL
        seeds = SEEDS

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng_global = np.random.default_rng(999)
    t0 = time.time()

    # ---- build the CANDIDATE pools ONCE (shared across all conditions
    # and seeds; only the final subsample draw differs per condition/seed) ----
    print('Building candidate feature pools (ET, shell, distant, random)...', flush=True)
    et_pool_list, shell_pool_list = [], []
    for r in det_lesions_fit:
        res = extract_lesion_shell(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
        if res is None:
            continue
        lf, sf, sz = res
        if len(lf) > MAX_VOX_PER_LESION:
            lf = lf[rng_global.choice(len(lf), MAX_VOX_PER_LESION, replace=False)]
        if len(sf) > MAX_VOX_PER_LESION:
            sf = sf[rng_global.choice(len(sf), MAX_VOX_PER_LESION, replace=False)]
        et_pool_list.append(lf); shell_pool_list.append(sf)
    et_pool = np.concatenate(et_pool_list).astype(np.float64)
    shell_candidate_pool = np.concatenate(shell_pool_list).astype(np.float64)

    distant_pool_list = []
    for sid in bg_subjects:
        feats = extract_distant_background(model, ds, sid_to_idx, sid, dev, rng_global)
        if feats is None:
            continue
        distant_pool_list.append(feats)
    distant_candidate_pool = np.concatenate(distant_pool_list).astype(np.float64)

    # random-negative candidate pool: dense ET + uniform non-ET sample,
    # reusing E275's own construction (all ET from a subset of subjects
    # for context, but we only need the non-ET side here since ET
    # positives are shared with et_pool)
    random_candidate_list = []
    conv_subjects = sorted(splits['det_train'])
    if smoke:
        conv_subjects = conv_subjects[:30]
    for sid in conv_subjects:
        if sid not in sid_to_idx:
            continue
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0].cpu().numpy()
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        non_et_mask = brain_mask & (~true_et)
        non_et_idx = np.where(non_et_mask)
        if len(non_et_idx[0]) > 0:
            n_sample = min(500, len(non_et_idx[0]))
            sel = rng_global.choice(len(non_et_idx[0]), size=n_sample, replace=False)
            coords = (non_et_idx[0][sel], non_et_idx[1][sel], non_et_idx[2][sel])
            random_candidate_list.append(d1[:, coords[0], coords[1], coords[2]].T)
        del d1, img_t, stages
        torch.cuda.empty_cache()
    random_candidate_pool = np.concatenate(random_candidate_list).astype(np.float64)

    print(f'  ET pool: {len(et_pool)}  shell candidates: {len(shell_candidate_pool)}  '
         f'distant candidates: {len(distant_candidate_pool)}  '
         f'random candidates: {len(random_candidate_pool)} ({time.time()-t0:.0f}s)', flush=True)

    def sample_negatives(condition, n_total, seed):
        rng = np.random.default_rng(seed)
        if condition == 'R0_random':
            pool = random_candidate_pool
            n = min(n_total, len(pool))
            return pool[rng.choice(len(pool), size=n, replace=False)]
        elif condition == 'R1_distant':
            pool = distant_candidate_pool
            n = min(n_total, len(pool))
            return pool[rng.choice(len(pool), size=n, replace=False)]
        elif condition == 'R2_shell':
            pool = shell_candidate_pool
            n = min(n_total, len(pool))
            return pool[rng.choice(len(pool), size=n, replace=False)]
        elif condition == 'R3_shell_distant':
            n_half = n_total // 2
            n_s = min(n_half, len(shell_candidate_pool))
            n_d = min(n_total - n_s, len(distant_candidate_pool))
            shell_part = shell_candidate_pool[rng.choice(len(shell_candidate_pool), size=n_s, replace=False)]
            distant_part = distant_candidate_pool[rng.choice(len(distant_candidate_pool), size=n_d, replace=False)]
            return np.concatenate([shell_part, distant_part])
        raise ValueError(condition)

    conditions = ['R0_random', 'R1_distant', 'R2_shell', 'R3_shell_distant']

    print('\nFitting readouts (4 conditions x 3 seeds)...', flush=True)
    fitted = {}  # (condition, seed) -> (w, b)
    for cond in conditions:
        for seed in seeds:
            neg = sample_negatives(cond, n_neg_total, seed)
            X = np.concatenate([et_pool, neg])
            y = np.concatenate([np.ones(len(et_pool)), np.zeros(len(neg))])
            clf = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X, y)
            fitted[(cond, seed)] = (clf.coef_[0].astype(np.float64), float(clf.intercept_[0]))
            print(f'  {cond} seed={seed}: n_neg={len(neg)} ({time.time()-t0:.0f}s)', flush=True)

    w_p_t = torch.from_numpy(w_et.astype(np.float64)).float().to(dev)
    b_p = float(b_et)
    variant_tensors = {'prod': (w_p_t, b_p)}
    for (cond, seed), (w, b) in fitted.items():
        variant_tensors[f'{cond}_s{seed}'] = (torch.from_numpy(w).float().to(dev), b)

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
            for name, (w_t, b_v) in variant_tensors.items():
                p = torch.sigmoid(flat @ w_t + b_v).reshape(D, H, W).cpu().numpy()
                maps[name] = p
        del d1, flat, img_t, stages
        torch.cuda.empty_cache()
        result = (maps, tgt_c, img_c)
        dense_cache[sid] = result
        return result

    test_subjects = sorted(splits['det_test'])
    if smoke:
        test_subjects = test_subjects[:15]

    print(f'\nBuilding non-circular phenotype groups on {len(test_subjects)} TEST subjects...', flush=True)
    lesion_records = []
    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        maps, tgt_c, img_c = dense_eval(sid)
        p_prod = maps['prod']
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
            lesion_records.append({'subject_id': sid, 'comp_id': cid, 'phenotype': phenotype})
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    n_by_phen = {}
    for rec in lesion_records:
        n_by_phen[rec['phenotype']] = n_by_phen.get(rec['phenotype'], 0) + 1
    print(f'\nPhenotype group sizes: {n_by_phen}', flush=True)

    lesion_by_sid = {}
    for rec in lesion_records:
        lesion_by_sid.setdefault(rec['subject_id'], []).append(rec)

    # ---- threshold sweep per variant (reusing E274/E275's logit-spaced grid) ----
    print('\nSweeping thresholds (identical grid to E274/E275)...', flush=True)
    threshold_grid = np.concatenate([
        np.linspace(0.01, 0.1, 10),
        np.linspace(0.1, 0.9, 20),
        1 - 1 / (1 + np.exp(np.linspace(2, 30, 60))),
    ])
    threshold_grid = np.unique(np.round(threshold_grid, 8))

    variant_names = list(variant_tensors.keys())
    fp_lists = {v: {t: [] for t in threshold_grid} for v in variant_names}
    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        maps, tgt_c, img_c = dense_eval(sid)
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        for name, p_map in maps.items():
            for t in threshold_grid:
                mask = (p_map > t) & brain_mask
                fp_lists[name][t].append(int((mask & (~true_et)).sum()))
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    mean_fp_by_threshold = {v: {} for v in variant_names}
    for v in variant_names:
        for t in threshold_grid:
            fps = fp_lists[v][t]
            mean_fp_by_threshold[v][t] = float(np.mean(fps)) if fps else float('nan')

    def find_threshold_for_fp_target(v, target_fp):
        best_t, best_diff = None, np.inf
        for t in threshold_grid:
            diff = abs(mean_fp_by_threshold[v][t] - target_fp)
            if diff < best_diff:
                best_diff, best_t = diff, t
        return best_t

    matched_thresholds = {}
    for v in variant_names:
        matched_thresholds[v] = {target: find_threshold_for_fp_target(v, target) for target in FP_TARGETS}

    def sensitivity_at_threshold(v, t, phenotype_filter=None):
        recs = []
        for sid in test_subjects:
            if sid not in sid_to_idx:
                continue
            maps, tgt_c, img_c = dense_eval(sid)
            p_map = maps[v]
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

    def dice_at_threshold(v, t):
        vals = []
        for sid in test_subjects:
            if sid not in sid_to_idx:
                continue
            maps, tgt_c, img_c = dense_eval(sid)
            p_map = maps[v]
            true_et = tgt_c[ET] > 0.5
            brain_mask = img_c[0] != 0
            mask = (p_map > t) & brain_mask
            ps, gs = mask.sum(), true_et.sum()
            if gs == 0:
                vals.append(1.0 if ps == 0 else np.nan)
            else:
                vals.append(2 * (mask & true_et).sum() / (ps + gs))
        return float(np.nanmean(vals)) if vals else float('nan')

    print('\nComputing matched-FP sensitivity per condition/seed...', flush=True)
    per_seed_results = []
    for cond in conditions:
        for seed in seeds:
            v = f'{cond}_s{seed}'
            for target in FP_TARGETS:
                t_match = matched_thresholds[v][target]
                for phen in ['G1_confidently_missed', 'G2_low_confidence', 'G3_confidently_detected', None]:
                    sens = sensitivity_at_threshold(v, t_match, phen)
                    per_seed_results.append({
                        'condition': cond, 'seed': seed, 'fp_target': target, 'threshold': t_match,
                        'mean_fp_actual': mean_fp_by_threshold[v][t_match],
                        'phenotype': phen or 'ALL', 'sensitivity': sens,
                    })
            t_500 = matched_thresholds[v][500]
            dice500 = dice_at_threshold(v, t_500)
            per_seed_results.append({'condition': cond, 'seed': seed, 'fp_target': 500, 'threshold': t_500,
                                     'mean_fp_actual': mean_fp_by_threshold[v][t_500], 'phenotype': 'DICE', 'sensitivity': dice500})
        print(f'  {cond} done ({time.time()-t0:.0f}s)', flush=True)

    # production reference row
    for target in FP_TARGETS:
        t_match = matched_thresholds['prod'][target]
        for phen in ['G1_confidently_missed', 'G2_low_confidence', 'G3_confidently_detected', None]:
            sens = sensitivity_at_threshold('prod', t_match, phen)
            per_seed_results.append({'condition': 'production', 'seed': 'NA', 'fp_target': target,
                                     'threshold': t_match, 'mean_fp_actual': mean_fp_by_threshold['prod'][t_match],
                                     'phenotype': phen or 'ALL', 'sensitivity': sens})

    out = HERE / ('E276_results_smoke.csv' if smoke else 'E276_results.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['condition', 'seed', 'fp_target', 'threshold', 'mean_fp_actual',
                                           'phenotype', 'sensitivity'])
        w.writeheader()
        for row in per_seed_results:
            w.writerow(row)

    # ---- aggregate mean+-SD across seeds, print summary table ----
    print('\n=== E276 summary (mean +/- SD across seeds) ===')
    summary_rows = []
    for cond in conditions:
        for target in FP_TARGETS:
            for phen in ['G1_confidently_missed', 'G2_low_confidence', 'G3_confidently_detected', 'ALL']:
                vals = [r['sensitivity'] for r in per_seed_results
                       if r['condition'] == cond and r['fp_target'] == target and r['phenotype'] == phen]
                if vals:
                    print(f'{cond:18s} FP~{target:5d} {phen:28s} '
                         f'{np.mean(vals):.4f} +/- {np.std(vals):.4f}', flush=True)
                    summary_rows.append({'condition': cond, 'fp_target': target, 'phenotype': phen,
                                        'mean': np.mean(vals), 'sd': np.std(vals)})

    summary_out = HERE / ('E276_summary_smoke.csv' if smoke else 'E276_summary.csv')
    with open(summary_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['condition', 'fp_target', 'phenotype', 'mean', 'sd'])
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    # ---- pairwise Jaccard overlap across seeds, per condition (G1+G2) ----
    print('\n=== Pairwise seed Jaccard overlap (G1+G2 lesions, TAU=0.5) ===')
    difficult_lesions = [rec for rec in lesion_records
                        if rec['phenotype'] in ('G1_confidently_missed', 'G2_low_confidence')]
    jaccard_rows = []
    for cond in conditions:
        recovered_sets = {}
        for seed in seeds:
            v = f'{cond}_s{seed}'
            recovered = set()
            for rec in difficult_lesions:
                sid, cid = rec['subject_id'], rec['comp_id']
                maps, tgt_c, img_c = dense_eval(sid)
                et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
                if cid < 1 or cid > et_n:
                    continue
                lm = et_lbl == cid
                if ((maps[v] > TAU) & lm).any():
                    recovered.add((sid, cid))
            recovered_sets[seed] = recovered
        pairwise_j = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                s1, s2 = recovered_sets[seeds[i]], recovered_sets[seeds[j]]
                union = s1 | s2
                j_val = len(s1 & s2) / len(union) if union else float('nan')
                pairwise_j.append(j_val)
        mean_j = float(np.mean(pairwise_j)) if pairwise_j else float('nan')
        print(f'{cond}: pairwise Jaccard = {[round(j,4) for j in pairwise_j]}  mean={mean_j:.4f}', flush=True)
        jaccard_rows.append({'condition': cond, 'mean_jaccard': mean_j,
                            'pairwise': ';'.join(f'{j:.4f}' for j in pairwise_j)})

    jaccard_out = HERE / ('E276_jaccard_smoke.csv' if smoke else 'E276_jaccard.csv')
    with open(jaccard_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['condition', 'mean_jaccard', 'pairwise'])
        w.writeheader()
        for row in jaccard_rows:
            w.writerow(row)

    print(f'\nE276 complete ({time.time()-t0:.0f}s). wrote {out.name}, {summary_out.name}, {jaccard_out.name}', flush=True)


if __name__ == '__main__':
    main()
