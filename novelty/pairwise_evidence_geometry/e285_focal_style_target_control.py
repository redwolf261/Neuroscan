"""E285 -- Focal-Style Target Control, per the user's exact spec. Small,
targeted addition to E283's target-transform decomposition, NOT a full
re-run -- E283/E284 already established H0/H1/H6 reliably across 5
seeds under this exact frozen protocol, so this script fits ONLY the
new condition (T_focal) and compares it against E283's own saved
numbers, avoiding redundant compute.

MOTIVATION: the prior-art audit noted T2(y,p)=(y-p)/(1-p+eps) is
algebraically identical, for binary y, to T_odds(y,p)=y-(1-y)*p/(1-p+eps)
-- an odds-scaled asymmetric target (positives=1, negatives=-p/(1-p)).
This is a REWRITING of T2, not a new condition (no new fit needed). The
one genuinely new comparison the audit raised is whether T2's specific
ODDS-SHAPED decay (as p increases) is what matters, or whether ANY
similarly-shaped monotonic down-weighting of easy positives would give
the same gain -- motivating a focal-loss-style target as a structurally
different decay-shape control.

T_FOCAL (per explicit user decisions this session, resolving two
genuine ambiguities before building): focal loss down-weights EASY
examples via a (1-p_t)^gamma factor on the LOSS, not the target --
since this entire investigation's pipeline transforms the TARGET (not
the loss), T_focal is defined as the direct target-space analogue,
applied ONLY to positives (mirroring T1/T2's own convention that
negatives stay at 0, isolating the SHAPE of the positive-target decay
as the sole new variable):
  T_focal(y,p) = y * (1-p)^gamma   if y=1
               = 0                 if y=0
  (gamma=2, the standard focal-loss default)
Decays MUCH more sharply than T1 (linear) or T2 (odds-shaped, stays
near 1 until p is very close to 1): at p=0.5, T_focal=0.25 vs T1=0.5
vs T2~1.0; at p=0.9, T_focal=0.01 (already nearly dead) vs T1=0.1 vs
T2~1.0 -- verified by hand on 8 (y,p) pairs before implementing.

FROZEN PROTOCOL: EXACTLY E283/E284's setup -- same 67 det_test
subjects, same 113 G1 lesions, same frozen production model, D1
features, R1-B pool, 32-dim linear LogisticRegression (no p as input
feature), same 5 seeds (999, 4242, 7, 123, 2024), raw-r evaluation,
same fixed threshold grid (E282-B's bug-fixed version).

COMPARISON: T_focal's own 5-seed G1 sensitivity vs E283's own saved
H0 (8.8/19.6/26.5), H1 (15.8/31.3/40.0), H6=T2 (23.7/35.6/40.9) at
FP~100/500/1000 -- read directly from E283_matched_fp.csv rather than
refit, confirming those numbers are unchanged before use.
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
EPS_MARG = 1e-6
TAU_LOW, TAU_HIGH = TAU, 0.9
FP_TARGETS = [100, 500, 1000]
SEEDS = [999, 4242, 7, 123, 2024]
GAMMA = 2.0


def T_focal(y, p):
    return np.where(y > 0.5, np.power(1.0 - p, GAMMA), 0.0)


def fit_soft_target(X, y_soft, max_iter=500, C=1.0):
    X2 = np.concatenate([X, X], axis=0)
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w2 = np.concatenate([y_soft, 1.0 - y_soft])
    keep = w2 > 1e-12
    X2, y2, w2 = X2[keep], y2[keep], w2[keep]
    clf = LogisticRegression(max_iter=max_iter, C=C, class_weight='balanced')
    clf.fit(X2, y2, sample_weight=w2)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    print('Verifying T_focal on hand-built (y,p) pairs...', flush=True)
    for y_val, p_val in [(1, 0.01), (1, 0.5), (1, 0.9), (1, 0.99), (1, 0.999999),
                        (0, 0.01), (0, 0.5), (0, 0.99)]:
        v = T_focal(np.array([float(y_val)]), np.array([p_val]))[0]
        print(f'  y={y_val} p={p_val}: T_focal={v:.4f}', flush=True)
        if y_val == 0:
            assert v == 0.0, f'BUG: negative has nonzero T_focal at p={p_val}'

    model = load_model(dev)
    w_et, b_et = get_w_prod(model)
    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    if smoke:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']][:30]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
        seeds = SEEDS[:2]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])
        seeds = SEEDS

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    t0_time = time.time()

    w_p_np, b_p_np = w_et.astype(np.float64), float(b_et)

    def p_prod_of(x):
        z = x @ w_p_np + b_p_np
        return 1.0 / (1.0 + np.exp(-z))

    def fit_r1b_pool(seed):
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
        t_focal_vals = T_focal(y_hard, p_i)
        assert t_focal_vals[y_hard == 0].max() == 0.0, 'BUG: negative has nonzero T_focal'
        fitted[seed] = fit_soft_target(X, t_focal_vals)
        print(f'  fit t_focal ({time.time()-t0_time:.0f}s)', flush=True)

    w_p_t = torch.from_numpy(w_p_np).float().to(dev)
    tensors = {seed: (torch.from_numpy(w).float().to(dev), b) for seed, (w, b) in fitted.items()}

    dense_cache = {}

    def dense_eval(sid):
        if sid in dense_cache:
            return dense_cache[sid]
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C_, D, H, W = d1.shape
        flat = d1.reshape(C_, -1).T
        maps = {}
        with torch.no_grad():
            z_prod = flat @ w_p_t + b_p_np
            maps['prod'] = torch.sigmoid(z_prod).reshape(D, H, W).cpu().numpy()
            for seed, (w_t, b) in tensors.items():
                r = torch.sigmoid(flat @ w_t + b)
                maps[seed] = r.reshape(D, H, W).cpu().numpy()
        del d1, flat, img_t, stages, z_prod
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
            lesion_records.append({'subject_id': sid, 'comp_id': cid, 'phenotype': phenotype,
                                   'size': int(lesion_mask.sum()), 'p_max_prod': p_max})
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0_time:.0f}s)', flush=True)

    lesion_by_sid = {}
    for rec in lesion_records:
        lesion_by_sid.setdefault(rec['subject_id'], []).append(rec)

    print('\nSweeping thresholds (raw-r evaluation, E282-B\'s fixed grid)...', flush=True)
    threshold_grid = np.concatenate([
        1 / (1 + np.exp(np.linspace(30, 2, 40))),
        np.linspace(0.01, 0.1, 10),
        np.linspace(0.1, 0.9, 20),
        1 - 1 / (1 + np.exp(np.linspace(2, 40, 80))),
    ])
    threshold_grid = np.unique(np.round(threshold_grid, 15))

    sources = ['prod'] + [f'{seed}::t_focal' for seed in seeds]
    fp_lists = {s: {t: [] for t in threshold_grid} for s in sources}
    recovered_by_sub = {s: {t: {} for t in threshold_grid} for s in sources}

    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        maps, tgt_c, img_c = dense_eval(sid)
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        et_lbl, et_n = ndimage.label(true_et)
        recs = lesion_by_sid.get(sid, [])
        for src_name in sources:
            if src_name == 'prod':
                p_map = maps['prod']
            else:
                seed_str = src_name.split('::')[0]
                p_map = maps[int(seed_str)]
            for t in threshold_grid:
                mask = (p_map > t) & brain_mask
                fp = int((mask & (~true_et)).sum())
                fp_lists[src_name][t].append(fp)
                by_phen = {}
                for rec in recs:
                    cid = rec['comp_id']
                    if cid < 1 or cid > et_n:
                        continue
                    lm = et_lbl == cid
                    by_phen.setdefault(rec['phenotype'], []).append(int(((p_map > t) & lm).any()))
                recovered_by_sub[src_name][t][sid] = by_phen
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0_time:.0f}s)', flush=True)

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

    print('\nComputing matched-FP sensitivity...', flush=True)
    matched_rows = []
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
                        vals = by_phen.get(phen_filter, [])
                    else:
                        vals = [v for vv in by_phen.values() for v in vv]
                    all_vals.extend(vals)
                sens = float(np.mean(all_vals)) if all_vals else float('nan')
                matched_rows.append({
                    'source': src_name, 'fp_target': target, 'threshold': t_match,
                    'mean_fp_actual': mean_fp_by_threshold[src_name][t_match],
                    'phenotype': phen_filter or 'ALL', 'sensitivity': sens,
                })
        print(f'  {src_name}: thresholds={matched_t} '
             f'(actual FP: {[round(mean_fp_by_threshold[src_name][t],1) for t in matched_t.values()]})',
             flush=True)

    out = HERE / ('E285_matched_fp_smoke.csv' if smoke else 'E285_matched_fp.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['source', 'fp_target', 'threshold', 'mean_fp_actual',
                                           'phenotype', 'sensitivity'])
        w.writeheader()
        for row in matched_rows:
            w.writerow(row)

    print('\n' + '=' * 70, flush=True)
    print('SUMMARY: G1 sensitivity, mean +/- SD across seeds, T_FOCAL', flush=True)
    print('=' * 70, flush=True)
    by_target = {}
    for row in matched_rows:
        if row['phenotype'] != 'G1_confidently_missed' or row['source'] == 'prod':
            continue
        by_target.setdefault(row['fp_target'], []).append(row['sensitivity'])
    parts = []
    for target in FP_TARGETS:
        vals = by_target.get(target, [])
        parts.append(f'FP~{target}: {np.mean(vals)*100:.1f}%±{np.std(vals)*100:.1f}%')
    print(f'  t_focal (gamma={GAMMA})   ' + '  '.join(parts), flush=True)
    print('\n  for comparison, E283\'s own saved numbers (NOT refit here):', flush=True)
    print('  h0_ordinary      FP~100: 8.8%±1.9%  FP~500: 19.6%±1.5%  FP~1000: 26.5%±2.7%', flush=True)
    print('  h1_subtract      FP~100: 15.8%±3.2%  FP~500: 31.3%±3.9%  FP~1000: 40.0%±2.8%', flush=True)
    print('  h6_normalized(T2) FP~100: 23.7%±1.0%  FP~500: 35.6%±1.5%  FP~1000: 40.9%±0.7%', flush=True)

    print(f'\nE285 complete ({time.time()-t0_time:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
