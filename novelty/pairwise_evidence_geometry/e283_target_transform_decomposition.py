"""E283 -- Target-Transform Decomposition, per the user's exact spec.
Direct follow-up to E282-A/B, which found the algebraic reconstruction
hat_y=p+(1-p)*r COLLAPSES MRD's G1 recovery to exactly 0% at every FP
budget -- a decisive negative for the RECONSTRUCTION scoring rule, but
NOT for MRD itself (E281's raw-r result is untouched by that finding,
since it used a different scoring convention throughout). The user's
explicit framing: "E282 is therefore not a setback to MRD. It is a
useful falsification: we now know the elegant probability-
reconstruction story is wrong and should not appear in the paper."

E283 isolates WHICH target transform is actually responsible for
E281's gain, using the evaluation convention we ALREADY KNOW WORKS
(raw sigmoid output of R, swept directly over the threshold grid --
E281's exact convention, explicitly NOT the E282 reconstruction).

FOUR TARGET TRANSFORMS (frozen before running, per explicit user
instruction and this session's established discipline), all computed
from the SAME (y,p) pair per training voxel:
  T0(y,p) = y                                    (ordinary target --
                                                    IDENTICAL to R1-B/
                                                    H1 throughout this
                                                    entire investigation)
  T1(y,p) = clip(y - p, 0, 1)                     (simple subtraction,
                                                    no normalization --
                                                    decays LINEARLY to 0
                                                    as p->1, unlike T2's
                                                    steeper/bounded decay)
  T2(y,p) = clip((y-p)/(1-p+eps), 0, 1)           (E281's winning
                                                    transform, the
                                                    "marginal recovery"
                                                    target -- normalizes
                                                    by remaining
                                                    probability mass)
  T3(y,p) = clip(y/(1-p+eps), 0, 1)               (normalizes by
                                                    remaining mass but
                                                    WITHOUT subtracting
                                                    p first -- verified
                                                    by hand before
                                                    implementing: for
                                                    y=1, T3=min(1/(1-p),1)
                                                    which clips to
                                                    EXACTLY 1.0 for any
                                                    p<1 except
                                                    numerically at the
                                                    very top, i.e. T3 is
                                                    nearly DEGENERATE
                                                    with T0 for
                                                    positives -- included
                                                    as a control that
                                                    should NOT show
                                                    T2-like gains despite
                                                    sharing T2's
                                                    denominator, isolating
                                                    whether the
                                                    SUBTRACTION of p (not
                                                    just the division) is
                                                    what matters)
  All negatives (y=0) have T_k=0 for every transform k, confirmed by
  hand-check and by assertion in the fitting code.

ARCHITECTURE/POOL (per explicit user requirement "identical
architecture, seeds, negative sampling"): EXACTLY R1-B's own
curated shell+distant pool, EXACTLY the same 5 seeds as E282-B (999,
4242, 7, 123, 2024), EXACTLY the same 32-dim linear LogisticRegression
(fit_soft_target for T1/T2/T3's continuous targets, ordinary
LogisticRegression.fit for T0). NO p_i as an input feature for ANY
condition (matching E281's H3, the best-performing MRD variant, and
removing the p-as-feature question entirely since E281/E282-B already
showed it doesn't matter) -- per explicit user framing this session,
all 4 conditions take ONLY x_i (32-dim D1 features) as input.

EVALUATION: raw sigmoid output swept over E274's exact threshold grid
(logit-spaced, extended both low and high as fixed in E282-B), matched-
FP-budget sensitivity (100/500/1000) on the SAME non-circular G1/G2/G3
phenotype definition, SAME det_test subjects.

DECISION RULE (per explicit user framing): if T2 uniquely survives this
decomposition (beats T0 decisively, with T1/T3 NOT reproducing the
gain), the paper's candidate contribution becomes a mathematically
specific, falsifiable operation -- "production-conditioned target
reparameterization for complementary segmentation learning" -- which
is then subject to a dedicated prior-art audit before any further
algorithm invention, per explicit user plan.
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


def T0(y, p):
    return y.copy()


def T1(y, p):
    return np.clip(y - p, 0.0, 1.0)


def T2(y, p):
    return np.clip((y - p) / (1.0 - p + EPS_MARG), 0.0, 1.0)


def T3(y, p):
    return np.clip(y / (1.0 - p + EPS_MARG), 0.0, 1.0)


TRANSFORMS = {'t0_ordinary': T0, 't1_subtract': T1, 't2_marginal': T2, 't3_divide_only': T3}


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

    # ---- hand-verify the 4 transforms before trusting anything (per
    # this session's established discipline) ----
    print('Verifying target transforms on hand-built (y,p) pairs...', flush=True)
    for y_val, p_val in [(1, 0.01), (1, 0.5), (1, 0.99), (1, 0.999999), (0, 0.01), (0, 0.5), (0, 0.99)]:
        y_arr = np.array([float(y_val)])
        p_arr = np.array([p_val])
        vals = {name: fn(y_arr, p_arr)[0] for name, fn in TRANSFORMS.items()}
        print(f'  y={y_val} p={p_val}: ' + '  '.join(f'{k}={v:.4f}' for k, v in vals.items()), flush=True)
        if y_val == 0:
            assert all(v == 0.0 for v in vals.values()), f'BUG: negative has nonzero transform at p={p_val}'

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

    w_p_np = w_et.astype(np.float64)
    b_p_np = float(b_et)

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

        fitted[seed] = {}
        for name, fn in TRANSFORMS.items():
            target = fn(y_hard, p_i)
            assert target[y_hard == 0].max() == 0.0, f'BUG: {name} nonzero on a negative'
            if name == 't0_ordinary':
                clf = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X, target)
                w, b = clf.coef_[0].astype(np.float64), float(clf.intercept_[0])
            else:
                w, b = fit_soft_target(X, target)
            fitted[seed][name] = (w, b)
            print(f'  fit {name} ({time.time()-t0_time:.0f}s)', flush=True)

    # ---- dense eval ----
    w_p_t = torch.from_numpy(w_p_np).float().to(dev)
    tensors = {(seed, name): (torch.from_numpy(w).float().to(dev), b)
              for seed in seeds for name, (w, b) in fitted[seed].items()}

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
            for (seed, name), (w_t, b) in tensors.items():
                r = torch.sigmoid(flat @ w_t + b)
                maps[(seed, name)] = r.reshape(D, H, W).cpu().numpy()
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

    n_by_phen = {}
    for rec in lesion_records:
        n_by_phen[rec['phenotype']] = n_by_phen.get(rec['phenotype'], 0) + 1
    print(f'\nPhenotype group sizes: {n_by_phen}', flush=True)
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

    sources = ['prod'] + [f'{seed}::{name}' for seed in seeds for name in TRANSFORMS]
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
                seed_str, name = src_name.split('::')
                p_map = maps[(int(seed_str), name)]
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

    out = HERE / ('E283_matched_fp_smoke.csv' if smoke else 'E283_matched_fp.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['source', 'fp_target', 'threshold', 'mean_fp_actual',
                                           'phenotype', 'sensitivity'])
        w.writeheader()
        for row in matched_rows:
            w.writerow(row)

    print('\n' + '=' * 70, flush=True)
    print('SUMMARY: G1 (confidently-missed) sensitivity, mean +/- SD across seeds', flush=True)
    print('=' * 70, flush=True)
    by_cond = {}
    for row in matched_rows:
        if row['phenotype'] != 'G1_confidently_missed' or row['source'] == 'prod':
            continue
        seed_str, name = row['source'].split('::')
        by_cond.setdefault(name, {}).setdefault(row['fp_target'], []).append(row['sensitivity'])
    for name in TRANSFORMS:
        by_target = by_cond.get(name, {})
        parts = []
        for target in FP_TARGETS:
            vals = by_target.get(target, [])
            parts.append(f'FP~{target}: {np.mean(vals)*100:.1f}%±{np.std(vals)*100:.1f}%')
        print(f'  {name:16s} {"  ".join(parts)}', flush=True)

    print(f'\nE283 complete ({time.time()-t0_time:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
