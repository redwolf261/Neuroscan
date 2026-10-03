"""E270 -- Hierarchical Residual Evidence, per the user's exact spec.
Direct follow-up to E269, which found a MIXED result: production-
orthogonal residual coherence (C_mu/C_sigma, 26-neighbor spatial
cosine similarity) separates recovered-G2A from R1-B's FP strongly at
the COMPONENT level (p=2.1e-13) but NOT at the voxel level (where
E269's own pre-registered Control-2 criterion -- random projection
should not perform equally well -- actually FAILED: controls matched
or beat prod_orth). Per the user's own reframing: this is now two
distinct hypotheses, not one. H1 (voxel-level) is NOT supported.
H2 (component-level) is supported by the ONE random direction tested
in E269, but that is not yet a real statistical test (a single lucky
random vector proves nothing). E270 is the harder, decisive version.

NEW OBJECT per component C (NOT spatial-neighbor coherence like E269 --
this is INTERNAL coherence of a candidate component's own voxels
relative to their own mean, under a given D1 projection M):

  r_v  = M @ x_v                          (projected D1 feature, per voxel in C)
  rbar_C = mean_{v in C} r_v               (component mean residual)
  A_C  = ||rbar_C||_2                      (component residual magnitude)
  Q_C  = mean_{v in C} cos(r_v, rbar_C)    (internal coherence -- does the
                                             WHOLE component point consistently
                                             in one direction?)

FOUR PROJECTIONS compared on the SAME components: M_P (production-
orthogonal, hypothesis), M_R (R1-B-orthogonal, Control 3), M_I (raw D1,
identity, Control 1), and -- per the user's explicit fix for "one lucky
random vector proves nothing" -- N_RANDOM=100 independently seeded
random-direction-orthogonal projections M_rand^(1..100) (Control 2,
proper null distribution).

STATISTICAL TEST (the actual pre-registered decision procedure): for
each projection M, compute the Mann-Whitney U separation statistic
between TP-component Q_C and FP-component Q_C (AUC-equivalent, bounded
0-1, 0.5=no separation). This gives ONE separation score per
projection: sep(M_P), sep(M_R), sep(M_I), and {sep(M_rand^(j))}_{j=1..100}.
Then:
  - permutation p-value: fraction of the 100 random-direction
    separations that equal or exceed sep(M_P) (this is a genuine
    null-distribution test, not a single A/B comparison)
  - direct comparisons: sep(M_P) vs sep(M_I) (is the projection doing
    anything beyond raw-D1 component aggregation?) and sep(M_P) vs
    sep(M_R) (is it specific to PRODUCTION's direction, not just any
    learned direction?)

COMPUTATIONAL DESIGN (deliberately avoids both prior E269 scaling
failure modes): per test subject, ONE dense D1 forward pass extracts
RAW (unprojected) D1 feature vectors for every voxel in every R1-B
positive component (typically a few to a few hundred voxels per
component, not the full 128^3 volume) -- small arrays. ALL 103
projections (M_P, M_R, M_I, 100x M_rand) are then cheap matrix
multiplies on these small per-component voxel-feature arrays, not
dense-volume operations -- no repeat of E269's GPU dense-field
approach (unnecessary here since this experiment needs no spatial
neighbor information, only within-component aggregation).
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


def projection_matrix(w):
    w = w.astype(np.float64)
    return np.eye(D1_DIM) - np.outer(w, w) / (np.dot(w, w) + EPS)


def A_and_Q(voxel_feats, M):
    """voxel_feats: (n,32) raw D1 features for one component's voxels.
    M: (32,32) projection matrix. Returns (A_C, Q_C) -- verified exact
    against a hand-computed synthetic case (both loop and vectorized
    forms matched) before use in the real pipeline."""
    r = voxel_feats @ M.T  # M symmetric, so this == (M @ v) per row
    r_bar = r.mean(axis=0)
    A_C = float(np.linalg.norm(r_bar))
    num = r @ r_bar
    den = np.linalg.norm(r, axis=1) * np.linalg.norm(r_bar) + EPS
    Q_C = float(np.mean(num / den))
    return A_C, Q_C


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
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
        n_random = 10
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])
        n_random = N_RANDOM

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng = np.random.default_rng(999)
    t0 = time.time()

    print('Fitting R1-B (identical to E257-B/E269)...', flush=True)
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
    w_r1b = r1b.coef_[0].astype(np.float64)
    b_r1b = float(r1b.intercept_[0])
    print(f'R1-B fit ({time.time()-t0:.0f}s)', flush=True)

    w_p = w_et.astype(np.float64)
    M_P = projection_matrix(w_p)
    M_R = projection_matrix(w_r1b)
    M_I = np.eye(D1_DIM)

    rng_rand_pool = np.random.default_rng(RNG_SEED_RANDOM_POOL)
    random_dirs = [rng_rand_pool.normal(size=D1_DIM) for _ in range(n_random)]
    M_rand_list = [projection_matrix(w) for w in random_dirs]

    # ---- verify A_and_Q on a hand-computed synthetic case before use ----
    _r_test = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    _M_test = np.eye(2)
    _A, _Q = A_and_Q(_r_test, _M_test)
    _rbar = _r_test.mean(axis=0)
    _A_direct = np.linalg.norm(_rbar)
    _cos_direct = np.mean([np.dot(v, _rbar) / (np.linalg.norm(v) * np.linalg.norm(_rbar)) for v in _r_test])
    assert np.isclose(_A, _A_direct) and np.isclose(_Q, _cos_direct), 'A_and_Q verification failed'
    print('A_C/Q_C computation verified exact.', flush=True)

    w_r1b_t = torch.from_numpy(w_r1b).float().to(dev)

    def extract_component_features(sid, g2a_cids, det_cids):
        """ONE dense forward pass per subject. Returns a list of dict
        rows: {'population': ..., 'voxel_feats': (n,32) RAW D1 array}
        per R1-B positive component -- NOT dense volumes, small arrays
        only. Population in {'TP', 'WT_not_TC_FP', 'distant_bg_FP'}."""
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]  # (32,D,H,W)
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T  # (N,32)
        with torch.no_grad():
            p_r1b_flat = torch.sigmoid(flat @ w_r1b_t + b_r1b)
        p_r1b_map = p_r1b_flat.reshape(D, H, W).cpu().numpy()
        d1_np = d1.cpu().numpy()  # (32,D,H,W) -- small enough to keep (8MB), freed at function end
        del d1, flat, img_t, stages, p_r1b_flat
        torch.cuda.empty_cache()

        brain_mask = img_c[0] != 0
        true_et = tgt_c[ET] > 0.5
        true_tc = tgt_c[TC] > 0.5
        true_wt = tgt_c[WT] > 0.5
        dilated = ndimage.binary_dilation(true_et, structure=STRUCT,
                                          iterations=NEAR_DILATION) if true_et.any() else np.zeros_like(true_et)
        local_shell = dilated & (~true_et) & brain_mask
        distant_bg_mask = brain_mask & (~true_et) & (~local_shell)

        pos_mask = (p_r1b_map > TAU) & brain_mask
        pos_lbl, pos_n = ndimage.label(pos_mask, structure=STRUCT)

        rows = []
        for cid in range(1, pos_n + 1):
            comp_mask = pos_lbl == cid
            size = int(comp_mask.sum())
            if size < MIN_COMPONENT_SIZE:
                # CRITICAL FIX (found before trusting any result): Q_C is
                # TRIVIALLY 1.0 for a single-voxel component (cosine of a
                # vector with its own mean is always exactly 1) -- verified
                # analytically. Since ~59% of FP components are single-voxel
                # (known from E262/E264/E269) vs TP's much larger median
                # size (~8 voxels), including tiny components would make
                # ANY TP-vs-FP Q_C comparison an artifact of component size,
                # not of the projection -- exactly the confound the user's
                # own spec (point 4 of the success criteria) warned about.
                # Excluding components below MIN_COMPONENT_SIZE from BOTH
                # populations removes this artifact at its source.
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
                    continue  # ambiguous local-shell component, skip
            idx = np.where(comp_mask)
            voxel_feats = d1_np[:, idx[0], idx[1], idx[2]].T.astype(np.float64)  # (size,32)
            rows.append({'subject_id': sid, 'population': pop, 'size': size, 'voxel_feats': voxel_feats})
        del d1_np, pos_mask, pos_lbl, p_r1b_map
        return rows

    g2a_by_sid, det_by_sid = {}, {}
    for r in g2a_lesions_test:
        g2a_by_sid.setdefault(r['subject_id'], []).append(int(r['comp_id']))
    for r in det_lesions_test:
        det_by_sid.setdefault(r['subject_id'], []).append(int(r['comp_id']))

    test_subjects = sorted(set(r['subject_id'] for r in det_lesions_test) |
                          set(r['subject_id'] for r in g2a_lesions_test))
    if smoke:
        test_subjects = test_subjects[:15]

    print(f'\nExtracting component D1 features from {len(test_subjects)} TEST subjects...', flush=True)
    all_components = []
    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        g2a_cids = g2a_by_sid.get(sid, [])
        det_cids = det_by_sid.get(sid, [])
        rows = extract_component_features(sid, g2a_cids, det_cids)
        all_components.extend(rows)
        if (i + 1) % 10 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s), '
                 f'{len(all_components)} components so far', flush=True)

    n_tp = sum(1 for c in all_components if c['population'] == 'TP')
    n_fp = len(all_components) - n_tp
    print(f'\nTotal components: {len(all_components)} ({n_tp} TP, {n_fp} FP)', flush=True)

    # ---- compute A_C, Q_C for every component under every projection ----
    print('\nComputing A_C, Q_C under M_P, M_R, M_I, and '
         f'{n_random} random-direction projections...', flush=True)
    projections = {'M_P': M_P, 'M_R': M_R, 'M_I': M_I}
    for j, M_r in enumerate(M_rand_list):
        projections[f'M_rand{j}'] = M_r

    results_AQ = {name: [] for name in projections}  # list of (population, A_C, Q_C)
    for comp in all_components:
        vf = comp['voxel_feats']
        for name, M in projections.items():
            A_C, Q_C = A_and_Q(vf, M)
            results_AQ[name].append((comp['population'], A_C, Q_C))
        comp.pop('voxel_feats')  # free memory once consumed
    print(f'A_C/Q_C computed for all components x {len(projections)} projections '
         f'({time.time()-t0:.0f}s)', flush=True)

    def separation(name, metric_idx):
        """Mann-Whitney U-based AUC-equivalent separation (0.5=none,
        1.0=perfect TP>FP) for Q_C (metric_idx=2) or A_C (metric_idx=1)
        under projection `name`."""
        tp_vals = [r[metric_idx] for r in results_AQ[name] if r[0] == 'TP']
        fp_vals = [r[metric_idx] for r in results_AQ[name] if r[0] != 'TP']
        if not tp_vals or not fp_vals:
            return float('nan')
        stat, _ = mannwhitneyu(tp_vals, fp_vals, alternative='two-sided')
        auc_equiv = stat / (len(tp_vals) * len(fp_vals))
        return float(auc_equiv)

    sep_Q_P = separation('M_P', 2)
    sep_Q_R = separation('M_R', 2)
    sep_Q_I = separation('M_I', 2)
    sep_Q_rand = [separation(f'M_rand{j}', 2) for j in range(n_random)]

    sep_A_P = separation('M_P', 1)
    sep_A_R = separation('M_R', 1)
    sep_A_I = separation('M_I', 1)
    sep_A_rand = [separation(f'M_rand{j}', 1) for j in range(n_random)]

    perm_p_Q = float(np.mean([s >= sep_Q_P for s in sep_Q_rand])) if sep_Q_rand else float('nan')
    perm_p_A = float(np.mean([s >= sep_A_P for s in sep_A_rand])) if sep_A_rand else float('nan')

    print(f'\n=== E270 RESULTS (Q_C separation, Mann-Whitney AUC-equivalent, TP vs FP) ===')
    print(f'sep(Q_C | M_P)    = {sep_Q_P:.4f}   (hypothesis)')
    print(f'sep(Q_C | M_R)    = {sep_Q_R:.4f}   (Control 3: R1-B-orthogonal)')
    print(f'sep(Q_C | M_I)    = {sep_Q_I:.4f}   (Control 1: raw D1, no projection)')
    print(f'sep(Q_C | random) = median {np.median(sep_Q_rand):.4f}, '
         f'range [{np.min(sep_Q_rand):.4f}, {np.max(sep_Q_rand):.4f}] (n={n_random}, Control 2)')
    print(f'Permutation p-value (M_P >= random): {perm_p_Q:.4f}')
    print()
    print(f'=== A_C separation (magnitude only, secondary) ===')
    print(f'sep(A_C | M_P)    = {sep_A_P:.4f}')
    print(f'sep(A_C | M_R)    = {sep_A_R:.4f}')
    print(f'sep(A_C | M_I)    = {sep_A_I:.4f}')
    print(f'sep(A_C | random) = median {np.median(sep_A_rand):.4f}, '
         f'range [{np.min(sep_A_rand):.4f}, {np.max(sep_A_rand):.4f}]')
    print(f'Permutation p-value (M_P >= random): {perm_p_A:.4f}')

    out = HERE / ('E270_summary_smoke.csv' if smoke else 'E270_summary.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['metric', 'sep_M_P', 'sep_M_R', 'sep_M_I', 'random_median', 'random_min',
                   'random_max', 'perm_p_value', 'n_random', 'n_tp', 'n_fp'])
        w.writerow(['Q_C', sep_Q_P, sep_Q_R, sep_Q_I, np.median(sep_Q_rand), np.min(sep_Q_rand),
                   np.max(sep_Q_rand), perm_p_Q, n_random, n_tp, n_fp])
        w.writerow(['A_C', sep_A_P, sep_A_R, sep_A_I, np.median(sep_A_rand), np.min(sep_A_rand),
                   np.max(sep_A_rand), perm_p_A, n_random, n_tp, n_fp])

    comp_detail_out = HERE / ('E270_components_smoke.csv' if smoke else 'E270_components.csv')
    with open(comp_detail_out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['population', 'A_C_MP', 'Q_C_MP', 'A_C_MR', 'Q_C_MR', 'A_C_MI', 'Q_C_MI'])
        for i in range(len(results_AQ['M_P'])):
            pop = results_AQ['M_P'][i][0]
            w.writerow([pop, results_AQ['M_P'][i][1], results_AQ['M_P'][i][2],
                       results_AQ['M_R'][i][1], results_AQ['M_R'][i][2],
                       results_AQ['M_I'][i][1], results_AQ['M_I'][i][2]])

    print(f'\nE270 complete ({time.time()-t0:.0f}s). wrote {out.name}, {comp_detail_out.name}', flush=True)


if __name__ == '__main__':
    main()
