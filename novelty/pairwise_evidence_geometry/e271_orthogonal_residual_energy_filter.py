"""E271 -- Orthogonal Residual Energy Filter, per the user's exact
spec. Direct follow-up to E270, which found A_C (component residual
MAGNITUDE under M_P = I - w_P w_P^T/||w_P||^2) is specific to
production's own direction (beats M_R, M_I, and a 100-random-direction
null) AND correctly directed (TP > FP), while Q_C (internal COHERENCE)
is also specific but REVERSED (FP > TP) -- interpreted per the user's
own framing as "coherence != correctness": a false-positive component
can be internally self-consistent while representing the wrong tissue.
Per the user's explicit redirect, this experiment does NOT try to
rescue Q_C. It adds ONE new quantity (E_C, mean per-voxel residual
energy, to separate "large coherent residual" from "large residual
that cancels in the mean") and then moves to the DECISIVE practical
test the whole E268-E270 geometry thread has been building toward:
does adding A_C to the validated R1-B+Delta_C pipeline measurably
improve it, on genuinely held-out data?

PART 1 -- confirm the A_C/E_C distinction (diagnostic, reusing E270's
exact component extraction and 100-direction random null):
  A_C = ||mean_v r_v||          (E270's original -- magnitude of the MEAN direction)
  E_C = mean_v ||r_v||          (NEW -- mean magnitude regardless of direction)
  A_C/E_C in [0,1]-ish           (directional concentration ratio -- NOT claimed
                                   novel on its own, per explicit user caution)
Tests A_C(TP)>A_C(FP) [replicates E270], E_C(TP) vs E_C(FP), and
A_C/E_C(TP) vs A_C/E_C(FP), each against the SAME M_P/M_R/M_I/100-random
comparison as E270.

PART 2 -- the decisive practical test:
  R1-B + Delta_C                              (current best, E264/E265: 58.3%
                                                recovery, Dice=0.777, FP=501,290)
  vs.
  R1-B + Delta_C + A_C   (per explicit user choice: A_C as an ADDITIONAL
                           AND-gate -- a component must pass BOTH Delta_C's
                           existing val-selected threshold AND a new
                           val-selected tau_A to survive; components with
                           size<MIN_COMPONENT_SIZE=5, where A_C was never
                           validated [E270's own confound-avoidance floor],
                           automatically PASS the A_C gate -- A_C only adds
                           filtering opportunity for size>=5 components, it
                           never removes coverage for smaller genuine
                           recoveries)
tau_A selected via the SAME zero-G2A-val-lesion-cost criterion as
Delta_C (E264's own discipline), per explicit user choice -- kept
apples-to-apples with the existing validated filter, directly answering
"without sacrificing recovery."
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from scipy.stats import mannwhitneyu
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
D1_DIM = 32
N_RANDOM = 100
RNG_SEED_RANDOM_POOL = 9100
EPS = 1e-12
MIN_COMPONENT_SIZE = 5
NEIGHBORHOOD_DILATION = 3


def projection_matrix(w):
    w = w.astype(np.float64)
    return np.eye(D1_DIM) - np.outer(w, w) / (np.dot(w, w) + EPS)


def A_E_Q(voxel_feats, M):
    """voxel_feats: (n,32) raw D1 features. Returns (A_C, E_C, Q_C).
    A_C/Q_C verified in E270 against a hand-computed synthetic case;
    E_C is new here (mean per-voxel residual norm) -- verified below
    against a direct loop before use."""
    r = voxel_feats @ M.T
    r_bar = r.mean(axis=0)
    A_C = float(np.linalg.norm(r_bar))
    norms = np.linalg.norm(r, axis=1)
    E_C = float(norms.mean())
    num = r @ r_bar
    den = norms * np.linalg.norm(r_bar) + EPS
    Q_C = float(np.mean(num / den))
    return A_C, E_C, Q_C


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
        g2a_lesions_val = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_val']][:15]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']][:15]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']][:15]
        g2b_lesions_eval = g2b_lesions[:15]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
        n_random = 10
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        g2a_lesions_val = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_val']]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']]
        g2b_lesions_eval = g2b_lesions
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])
        n_random = N_RANDOM

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng = np.random.default_rng(999)
    t0 = time.time()

    print('Fitting R1-B (identical to E257-B/E269/E270)...', flush=True)
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
    r1b = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_train, y_train)
    w_r1b_np = r1b.coef_[0].astype(np.float64)
    b_r1b = float(r1b.intercept_[0])
    print(f'R1-B fit ({time.time()-t0:.0f}s)', flush=True)

    w_p = w_et.astype(np.float64)
    M_P = projection_matrix(w_p)
    M_R = projection_matrix(w_r1b_np)
    M_I = np.eye(D1_DIM)
    rng_rand_pool = np.random.default_rng(RNG_SEED_RANDOM_POOL)
    random_dirs = [rng_rand_pool.normal(size=D1_DIM) for _ in range(n_random)]
    M_rand_list = [projection_matrix(w) for w in random_dirs]

    # ---- verify E_C against a direct loop before trusting any real data ----
    _r_test = np.array([[3.0, 4.0], [0.0, 1.0], [5.0, 0.0]])
    _, _E_vec, _ = A_E_Q(_r_test, np.eye(2))
    _E_loop = np.mean([np.linalg.norm(v) for v in _r_test])
    assert np.isclose(_E_vec, _E_loop), 'E_C verification failed'
    print('A_C/E_C/Q_C computation verified exact.', flush=True)

    w_r1b_t = torch.from_numpy(w_r1b_np).float().to(dev)
    w_p_t = torch.from_numpy(w_p).float().to(dev)

    def dense_probs_and_components(sid, g2a_cids=None, det_cids=None):
        """ONE forward pass. Returns p_r1b_map, raw d1_np (32,D,H,W),
        tgt_c, img_c, pos_lbl, pos_n -- everything needed for both the
        Part-1 discrimination extraction and Part-2 filter evaluation,
        computed once per subject."""
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T
        with torch.no_grad():
            p_r1b_flat = torch.sigmoid(flat @ w_r1b_t + b_r1b)
        p_r1b_map = p_r1b_flat.reshape(D, H, W).cpu().numpy()
        d1_np = d1.cpu().numpy()
        del d1, flat, img_t, stages, p_r1b_flat
        torch.cuda.empty_cache()
        brain_mask = img_c[0] != 0
        pos_mask = (p_r1b_map > TAU) & brain_mask
        pos_lbl, pos_n = ndimage.label(pos_mask, structure=STRUCT)
        return p_r1b_map, d1_np, tgt_c, img_c, pos_lbl, pos_n

    # ================= PART 1: discrimination analysis =================
    g2a_by_sid_test, det_by_sid_test = {}, {}
    for r in g2a_lesions_test:
        g2a_by_sid_test.setdefault(r['subject_id'], []).append(int(r['comp_id']))
    for r in det_lesions_test:
        det_by_sid_test.setdefault(r['subject_id'], []).append(int(r['comp_id']))

    test_subjects_p1 = sorted(set(r['subject_id'] for r in det_lesions_test) |
                              set(r['subject_id'] for r in g2a_lesions_test))
    if smoke:
        test_subjects_p1 = test_subjects_p1[:15]

    print(f'\nPART 1: extracting component D1 features from {len(test_subjects_p1)} TEST subjects...', flush=True)
    all_components = []
    for i, sid in enumerate(test_subjects_p1):
        if sid not in sid_to_idx:
            continue
        p_r1b_map, d1_np, tgt_c, img_c, pos_lbl, pos_n = dense_probs_and_components(sid)
        brain_mask = img_c[0] != 0
        true_et = tgt_c[ET] > 0.5
        true_tc = tgt_c[TC] > 0.5
        true_wt = tgt_c[WT] > 0.5
        dilated = ndimage.binary_dilation(true_et, structure=STRUCT,
                                          iterations=NEAR_DILATION) if true_et.any() else np.zeros_like(true_et)
        local_shell = dilated & (~true_et) & brain_mask
        distant_bg_mask = brain_mask & (~true_et) & (~local_shell)
        for cid in range(1, pos_n + 1):
            comp_mask = pos_lbl == cid
            size = int(comp_mask.sum())
            if size < MIN_COMPONENT_SIZE:
                continue
            is_tp = bool((comp_mask & true_et).any())
            if is_tp:
                pop = 'TP'
            else:
                frac_tc = (comp_mask & true_tc).sum() / size
                frac_wt_not_tc = (comp_mask & true_wt & (~true_tc)).sum() / size
                frac_distant = (comp_mask & distant_bg_mask).sum() / size
                if frac_tc > 0.5 or frac_wt_not_tc > 0.5:
                    pop = 'WT_not_TC_FP'
                elif frac_distant > 0.5:
                    pop = 'distant_bg_FP'
                else:
                    continue
            idx = np.where(comp_mask)
            voxel_feats = d1_np[:, idx[0], idx[1], idx[2]].T.astype(np.float64)
            all_components.append({'population': pop, 'voxel_feats': voxel_feats})
        del d1_np, p_r1b_map, pos_lbl
        if (i + 1) % 10 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects_p1)} ({time.time()-t0:.0f}s), '
                 f'{len(all_components)} components so far', flush=True)

    n_tp = sum(1 for c in all_components if c['population'] == 'TP')
    n_fp = len(all_components) - n_tp
    print(f'Total: {len(all_components)} ({n_tp} TP, {n_fp} FP)', flush=True)

    projections = {'M_P': M_P, 'M_R': M_R, 'M_I': M_I}
    for j, M_r in enumerate(M_rand_list):
        projections[f'M_rand{j}'] = M_r

    print(f'Computing A_C/E_C/Q_C/ratio under {len(projections)} projections...', flush=True)
    results_AEQ = {name: [] for name in projections}
    for comp in all_components:
        vf = comp['voxel_feats']
        for name, M in projections.items():
            A_C, E_C, Q_C = A_E_Q(vf, M)
            ratio = A_C / (E_C + EPS)
            results_AEQ[name].append((comp['population'], A_C, E_C, Q_C, ratio))
        comp.pop('voxel_feats')
    print(f'Done ({time.time()-t0:.0f}s)', flush=True)

    def separation(name, metric_idx):
        tp_vals = [r[metric_idx] for r in results_AEQ[name] if r[0] == 'TP']
        fp_vals = [r[metric_idx] for r in results_AEQ[name] if r[0] != 'TP']
        if not tp_vals or not fp_vals:
            return float('nan')
        stat, _ = mannwhitneyu(tp_vals, fp_vals, alternative='two-sided')
        return float(stat / (len(tp_vals) * len(fp_vals)))

    part1_out = HERE / ('E271_part1_discrimination_smoke.csv' if smoke else 'E271_part1_discrimination.csv')
    with open(part1_out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['metric', 'sep_M_P', 'sep_M_R', 'sep_M_I', 'random_median', 'random_min',
                   'random_max', 'perm_p', 'n_random', 'n_tp', 'n_fp'])
        for metric_name, idx in [('A_C', 1), ('E_C', 2), ('Q_C', 3), ('A_C_over_E_C', 4)]:
            sep_P = separation('M_P', idx)
            sep_R = separation('M_R', idx)
            sep_I = separation('M_I', idx)
            sep_rand = [separation(f'M_rand{j}', idx) for j in range(n_random)]
            perm_p = float(np.mean([s >= sep_P for s in sep_rand])) if sep_rand else float('nan')
            w.writerow([metric_name, sep_P, sep_R, sep_I, np.median(sep_rand), np.min(sep_rand),
                       np.max(sep_rand), perm_p, n_random, n_tp, n_fp])
            print(f'{metric_name:15s} sep(M_P)={sep_P:.4f} sep(M_R)={sep_R:.4f} sep(M_I)={sep_I:.4f} '
                 f'random=[{np.min(sep_rand):.4f},{np.max(sep_rand):.4f}] perm_p={perm_p:.4f}', flush=True)
    print(f'Part 1 complete, wrote {part1_out.name}', flush=True)

    # ================= PART 2: practical pipeline test =================
    print('\nPART 2: R1-B+Delta_C vs R1-B+Delta_C+A_C on TEST...', flush=True)

    def component_table_full(sid):
        """Full component table (ALL sizes, for Delta_C's own
        zero-cost threshold which was validated across all sizes in
        E264), plus A_C (only meaningful for size>=MIN_COMPONENT_SIZE,
        else None -> auto-pass)."""
        p_r1b_map, d1_np, tgt_c, img_c, pos_lbl, pos_n = dense_probs_and_components(sid)
        brain_mask = img_c[0] != 0
        true_et = tgt_c[ET] > 0.5
        rows = []
        for cid in range(1, pos_n + 1):
            comp_mask = pos_lbl == cid
            size = int(comp_mask.sum())
            if size < 1:
                continue
            is_tp = bool((comp_mask & true_et).any())
            M_C = float(p_r1b_map[comp_mask].mean())
            dilated = ndimage.binary_dilation(comp_mask, structure=STRUCT,
                                              iterations=NEIGHBORHOOD_DILATION)
            shell = dilated & (~comp_mask) & (~true_et) & brain_mask
            n_shell = int(shell.sum())
            mu_N = float(p_r1b_map[shell].mean()) if n_shell > 0 else 0.0
            Delta_C = M_C - mu_N
            if size >= MIN_COMPONENT_SIZE:
                idx = np.where(comp_mask)
                vf = d1_np[:, idx[0], idx[1], idx[2]].T.astype(np.float64)
                A_C, _, _ = A_E_Q(vf, M_P)
            else:
                A_C = None
            rows.append({'component_id': cid, 'is_tp': int(is_tp), 'size': size,
                        'Delta_C': Delta_C, 'A_C': A_C})
        return rows, pos_lbl, p_r1b_map, tgt_c, img_c

    comp_cache = {}

    def get_components_full(sid):
        if sid not in comp_cache:
            comp_cache[sid] = component_table_full(sid)
        return comp_cache[sid]

    # ---- select Delta_C threshold (E264's own zero-cost rule, lesion-level) ----
    def per_lesion_max_stat(lesion_list, stat_key, default_pass_if_none=True):
        maxima = []
        for r in lesion_list:
            sid, cid = r['subject_id'], int(r['comp_id'])
            if sid not in sid_to_idx:
                continue
            rows, lbl, p_map, tgt_c, img_c = get_components_full(sid)
            et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
            if cid < 1 or cid > et_n:
                continue
            lesion_mask = et_lbl == cid
            if int(lesion_mask.sum()) < MIN_VOX:
                continue
            overlap_ids = set(np.unique(lbl[lesion_mask])) - {0}
            vals = []
            for row in rows:
                if row['component_id'] not in overlap_ids:
                    continue
                v = row[stat_key]
                if v is None:
                    v = np.inf if default_pass_if_none else -np.inf
                vals.append(v)
            if vals:
                maxima.append(max(vals))
        return maxima

    def select_threshold(lesion_list, stat_key, default_pass_if_none=True):
        maxima = per_lesion_max_stat(lesion_list, stat_key, default_pass_if_none)
        finite = [m for m in maxima if np.isfinite(m)]
        return min(finite) if finite else -np.inf

    thresh_delta = select_threshold(g2a_lesions_val, 'Delta_C', default_pass_if_none=False)
    thresh_A = select_threshold(g2a_lesions_val, 'A_C', default_pass_if_none=True)
    print(f'VAL-selected thresholds: Delta_C>={thresh_delta:.4f}  A_C>={thresh_A:.4f} '
         f'({time.time()-t0:.0f}s)', flush=True)

    def passes_filter(row, use_A_gate):
        if row['Delta_C'] < thresh_delta:
            return False
        if use_A_gate:
            if row['A_C'] is not None and row['A_C'] < thresh_A:
                return False
        return True

    test_subjects_p2 = sorted(set(r['subject_id'] for r in det_lesions_test) |
                              set(r['subject_id'] for r in g2a_lesions_test) |
                              set(r['subject_id'] for r in g2b_lesions_eval))
    if smoke:
        test_subjects_p2 = test_subjects_p2[:15]

    def lesion_recovery(lesion_list, use_A_gate):
        recs = []
        for r in lesion_list:
            sid, cid = r['subject_id'], int(r['comp_id'])
            if sid not in sid_to_idx:
                continue
            rows, lbl, p_map, tgt_c, img_c = get_components_full(sid)
            et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
            if cid < 1 or cid > et_n:
                continue
            lesion_mask = et_lbl == cid
            if int(lesion_mask.sum()) < MIN_VOX:
                continue
            overlap_ids = set(np.unique(lbl[lesion_mask])) - {0}
            survived = any(row['component_id'] in overlap_ids and passes_filter(row, use_A_gate)
                          for row in rows)
            recs.append(int(survived))
        return float(np.mean(recs)) if recs else float('nan')

    print(f'\nEvaluating on {len(test_subjects_p2)} TEST subjects...', flush=True)
    fp_total = {'delta_only': 0, 'delta_plus_A': 0}
    dice_vals = {'delta_only': [], 'delta_plus_A': []}
    for i, sid in enumerate(test_subjects_p2):
        if sid not in sid_to_idx:
            continue
        rows, lbl, p_map, tgt_c, img_c = get_components_full(sid)
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        for variant, use_A in [('delta_only', False), ('delta_plus_A', True)]:
            keep_ids = {row['component_id'] for row in rows if passes_filter(row, use_A)}
            mask = np.isin(lbl, list(keep_ids)) if keep_ids else np.zeros_like(lbl, dtype=bool)
            fp_mask = mask & (~true_et) & brain_mask
            fp_total[variant] += int(fp_mask.sum())
            dice_vals[variant].append(dice_score(mask, true_et))
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects_p2)} ({time.time()-t0:.0f}s)', flush=True)

    results_p2 = {}
    for variant, use_A in [('delta_only', False), ('delta_plus_A', True)]:
        results_p2[variant] = {
            'g2a_recovery': lesion_recovery(g2a_lesions_test, use_A),
            'det_recovery': lesion_recovery(det_lesions_test, use_A),
            'g2b_recovery': lesion_recovery(g2b_lesions_eval, use_A),
            'et_dice_mean': float(np.nanmean(dice_vals[variant])),
            'fp_total': fp_total[variant],
        }

    print(f'\n=== E271 PART 2 RESULTS ===')
    for k, v in results_p2.items():
        print(f'{k}: {v}')

    part2_out = HERE / ('E271_part2_pipeline_smoke.csv' if smoke else 'E271_part2_pipeline.csv')
    with open(part2_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['variant', 'g2a_recovery', 'det_recovery', 'g2b_recovery',
                                           'et_dice_mean', 'fp_total'])
        w.writeheader()
        for variant, v in results_p2.items():
            w.writerow({'variant': variant, **v})

    print(f'\nE271 complete ({time.time()-t0:.0f}s). wrote {part1_out.name}, {part2_out.name}', flush=True)


if __name__ == '__main__':
    main()
