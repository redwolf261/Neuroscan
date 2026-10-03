"""E284 -- Production-Conditioned Residual Specificity, per the user's
exact spec. Direct follow-up to E283, which decomposed the marginal
target into its two operations (subtract p, normalize by 1-p) and
found subtraction is the primary active ingredient, normalization a
smaller stabilizing refinement. E284 asks the harder causal question
E283 left open: WHY does subtracting p work -- because p is the ACTUAL
frozen production model's own readout, or because subtracting ANY
reasonable probability field produces the same effect?

FROZEN PROTOCOL (per explicit user requirement "freeze everything...
only the target construction changes"): EXACTLY E283's setup -- same
67 det_test subjects, same 113 G1 lesions, same frozen production
model (w_P/b_P), same D1 features, same R1-B hard-negative
distribution, same 32-dim linear LogisticRegression architecture (no
p_i as an input feature, matching E283's decomposition), same
threshold/FROC procedure (E282-B's fixed grid), same 5 seeds (999,
4242, 7, 123, 2024), raw recovery score evaluated directly (NO
p+(1-p)*r reconstruction, per E282's falsification).

SEVEN CONDITIONS:
  H0 ordinary          T0 = y                               (=R1-B)
  H1 true-prod-resid    T_P   = clip(y - p, 0, 1)             (=E283's T1)
  H2 shuffled-prod      T_pi  = clip(y - p_pi, 0, 1)          (p_pi = a
                                                                DIFFERENT
                                                                det_train
                                                                subject's
                                                                production
                                                                probability
                                                                AT THE SAME
                                                                PATCH
                                                                COORDINATE,
                                                                per
                                                                explicit
                                                                user
                                                                decision:
                                                                permutation
                                                                happens at
                                                                TRAINING
                                                                time, over
                                                                det_train
                                                                subjects --
                                                                a single
                                                                fixed
                                                                derangement,
                                                                no subject
                                                                mapped to
                                                                itself,
                                                                built once
                                                                and reused
                                                                across all
                                                                5 R1-B
                                                                seeds, kept
                                                                as a
                                                                SEPARATE
                                                                randomness
                                                                source
                                                                (seed=31415)
                                                                from R1-B's
                                                                own negative-
                                                                sampling
                                                                seeds)
  H3 population-base    T_pbar = clip(y - pbar, 0, 1)         (pbar = a
                                                                single
                                                                frozen
                                                                SCALAR,
                                                                mean
                                                                production
                                                                probability
                                                                over ALL
                                                                det_train
                                                                brain
                                                                voxels --
                                                                not per-
                                                                voxel,
                                                                since a
                                                                per-voxel
                                                                spatial
                                                                average
                                                                would need
                                                                an
                                                                arbitrary
                                                                shared
                                                                coordinate
                                                                frame
                                                                across
                                                                subjects
                                                                with
                                                                different
                                                                lesion
                                                                locations;
                                                                computed
                                                                ONLY from
                                                                det_train,
                                                                never
                                                                touching
                                                                test
                                                                subjects)
  H4 alt-model-resid     T_Q = clip(y - q, 0, 1)               (q =
                                                                E130_
                                                                baseline_
                                                                seed0's
                                                                own
                                                                sigmoid
                                                                probability
                                                                -- an
                                                                INDEPEND-
                                                                ENTLY
                                                                TRAINED
                                                                UNet3D_v5
                                                                (4-in/3-out),
                                                                verified
                                                                identical
                                                                architecture
                                                                class to
                                                                production
                                                                via direct
                                                                checkpoint
                                                                inspection
                                                                (seg_head
                                                                shape
                                                                [3,32,1,1,1]
                                                                matches;
                                                                ruled OUT
                                                                several
                                                                other
                                                                candidates,
                                                                e.g.
                                                                Control_
                                                                v5amp_seed0,
                                                                found via
                                                                inspection
                                                                to be
                                                                single-head
                                                                [1,32,...]
                                                                variants,
                                                                architec-
                                                                turally
                                                                incompatible)
                                                                -- a
                                                                genuinely
                                                                different
                                                                training
                                                                run
                                                                (experiment
                                                                line E130),
                                                                NOT a
                                                                sibling-
                                                                seed of
                                                                the SAME
                                                                training
                                                                run since
                                                                no such
                                                                checkpoint
                                                                exists in
                                                                this repo)
  H5 sign-reversed        T_neg = clip(p - y, 0, 1)            (directional
                                                                control --
                                                                NOT
                                                                expected to
                                                                work; a
                                                                meaningful
                                                                recovery
                                                                here would
                                                                be
                                                                SUSPICIOUS)
  H6 normalized-prod      T_MRD = clip((y-p)/(1-p+eps), 0, 1)  (=E283's
                                                                T2, the
                                                                final MRD
                                                                candidate)

ARCHITECTURE: all 7 conditions take ONLY x_i (32-dim D1 features) as
input, matching E283 and E281/E282-B's finding that p-as-feature never
mattered.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from e257_common import (load_model, get_w_prod, load_populations, build_splits,  # noqa: E402
                         extract_lesion_shell, extract_distant_background,
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch, get_masks  # noqa: E402
from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402

ET, TC, WT = 0, 1, 2
EPS_MARG = 1e-6
TAU_LOW, TAU_HIGH = TAU, 0.9
FP_TARGETS = [100, 500, 1000]
SEEDS = [999, 4242, 7, 123, 2024]
Q_CKPT = ROOT_REPO / 'experiments/exp_e12_eggo_m/e130/runs/E130_baseline_seed0/checkpoints/best.pth'
CONDS = ['h0_ordinary', 'h1_true_prod', 'h2_shuffled_prod', 'h3_population_baseline',
        'h4_alt_model', 'h5_sign_reversed', 'h6_normalized_prod']


def load_q_model(dev):
    model = UNet3D_v5(4, 3).to(dev).eval()
    ck = torch.load(str(Q_CKPT), map_location=dev, weights_only=False)
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def fit_soft_target(X, y_soft, max_iter=500, C=1.0):
    X2 = np.concatenate([X, X], axis=0)
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w2 = np.concatenate([y_soft, 1.0 - y_soft])
    keep = w2 > 1e-12
    X2, y2, w2 = X2[keep], y2[keep], w2[keep]
    clf = LogisticRegression(max_iter=max_iter, C=C, class_weight='balanced')
    clf.fit(X2, y2, sample_weight=w2)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])


def dense_p_map(model, w_np, b_np, ds, sid_to_idx, sid, dev):
    """Returns (p_map (D,H,W) float64, d1 (32,D,H,W) float64, img_c, tgt_c)
    for a subject, using the given frozen model's w/b on its OWN D1
    features (so this works for both production and the alt model Q,
    and for computing a different subject's p-map for H2's permutation)."""
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    stages = forward_to_dec1_internal(model, img_t)
    d1 = stages['relu2_out'][0].cpu().numpy().astype(np.float64)
    C_, D, H, W = d1.shape
    flat = d1.reshape(C_, -1).T
    z = flat @ w_np + b_np
    p_map = (1.0 / (1.0 + np.exp(-z))).reshape(D, H, W)
    return p_map, d1, img_c, tgt_c


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    model = load_model(dev)
    w_et, b_et = get_w_prod(model)
    q_model = load_q_model(dev)
    w_q, b_q = get_w_prod(q_model)
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
    w_q_np, b_q_np = w_q.astype(np.float64), float(b_q)

    def p_prod_of(x):
        z = x @ w_p_np + b_p_np
        return 1.0 / (1.0 + np.exp(-z))

    def q_alt_of(x):
        z = x @ w_q_np + b_q_np
        return 1.0 / (1.0 + np.exp(-z))

    # ---- H3's population-baseline pbar: scalar mean production
    # probability over det_train brain voxels ONLY (leakage-safe) ----
    print('Computing population-baseline pbar (scalar, det_train voxels only)...', flush=True)
    pbar_accum, pbar_n = 0.0, 0
    pbar_subjects = sorted(splits['det_train'])[:20] if smoke else sorted(splits['det_train'])
    for sid in pbar_subjects:
        if sid not in sid_to_idx:
            continue
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0].cpu().numpy().astype(np.float64)
        C_, D, H, W = d1.shape
        flat = d1.reshape(C_, -1).T
        brain_mask = (img_c[0] != 0).reshape(-1)
        p_vals = p_prod_of(flat[brain_mask])
        pbar_accum += p_vals.sum()
        pbar_n += len(p_vals)
        del d1, img_t, stages
        torch.cuda.empty_cache()
    pbar = pbar_accum / pbar_n
    print(f'  pbar = {pbar:.6f} (over {pbar_n} det_train brain voxels, {time.time()-t0_time:.0f}s)', flush=True)

    # ---- H2's subject-level derangement over det_train subjects (the
    # ones actually contributing to the training pool), single fixed
    # seed, SEPARATE randomness source from R1-B's own seeds ----
    train_subjects_for_perm = sorted(set(r['subject_id'] for r in det_lesions_fit))
    rng_perm = np.random.default_rng(31415)
    n_subj = len(train_subjects_for_perm)
    idx = np.arange(n_subj)
    while True:
        shuffled = rng_perm.permutation(idx)
        if not np.any(shuffled == idx):
            break
    perm_map = {train_subjects_for_perm[i]: train_subjects_for_perm[shuffled[i]] for i in range(n_subj)}
    print(f'Subject-level derangement over {n_subj} det_train subjects (no self-maps), '
         f'e.g. {list(perm_map.items())[:3]}', flush=True)

    # precompute each det_train subject's own production p-map ONCE
    # (needed both for its own residual AND as a lookup target for
    # OTHER subjects' H2 permutation)
    print('Precomputing production p-maps for all det_train subjects (for H2 lookup)...', flush=True)
    train_p_cache = {}
    for sid in train_subjects_for_perm:
        if sid not in sid_to_idx:
            continue
        p_map, _, _, _ = dense_p_map(model, w_p_np, b_p_np, ds, sid_to_idx, sid, dev)
        train_p_cache[sid] = p_map
    print(f'  cached {len(train_p_cache)} p-maps ({time.time()-t0_time:.0f}s)', flush=True)

    def fit_r1b_pool_with_coords(seed):
        """Builds R1-B's curated pool, ALSO retaining each lesion/shell
        voxel's (subject_id, coord) so H2 can look up the permuted
        partner subject's p AT THE SAME COORDINATE."""
        rng = np.random.default_rng(seed)
        lesion_pool, neg_pool = [], []
        lesion_p_true, neg_p_true = [], []
        lesion_p_perm, neg_p_perm = [], []
        for r in det_lesions_fit:
            sid, cid = r['subject_id'], int(r['comp_id'])
            if sid not in sid_to_idx:
                continue
            img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
            brain_mask = img_c[0] != 0
            cm, shell = get_masks(tgt_c, cid, brain_mask)
            if cm is None:
                continue
            img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
            stages = forward_to_dec1_internal(model, img_t)
            d1 = stages['relu2_out'][0].cpu().numpy().astype(np.float64)
            lf = d1[:, cm].T
            sf = d1[:, shell].T
            p_true_full = train_p_cache[sid]
            p_perm_full = train_p_cache.get(perm_map[sid], train_p_cache[sid])
            lp_true, sp_true = p_true_full[cm], p_true_full[shell]
            lp_perm, sp_perm = p_perm_full[cm], p_perm_full[shell]
            if len(lf) > MAX_VOX_PER_LESION:
                sel = rng.choice(len(lf), MAX_VOX_PER_LESION, replace=False)
                lf, lp_true, lp_perm = lf[sel], lp_true[sel], lp_perm[sel]
            if len(sf) > MAX_VOX_PER_LESION:
                sel = rng.choice(len(sf), MAX_VOX_PER_LESION, replace=False)
                sf, sp_true, sp_perm = sf[sel], sp_true[sel], sp_perm[sel]
            lesion_pool.append(lf); neg_pool.append(sf)
            lesion_p_true.append(lp_true); neg_p_true.append(sp_true)
            lesion_p_perm.append(lp_perm); neg_p_perm.append(sp_perm)
            del d1, img_t, stages
            torch.cuda.empty_cache()
        for sid in bg_subjects:
            feats = extract_distant_background(model, ds, sid_to_idx, sid, dev, rng)
            if feats is None:
                continue
            neg_pool.append(feats)
            # distant background's own true p: recompute via p_prod_of
            # (cheap, feats already extracted); permuted p for distant
            # background uses the SAME permutation partner's SCALAR
            # pbar-like fallback is wrong -- instead, since distant
            # background coordinates are not tracked by extract_
            # distant_background, and H2's spec is specifically about
            # LESION/SHELL residual construction (the region E283
            # showed the signal concentrates in), distant-background
            # rows use their OWN true p for ALL conditions including H2
            # -- H2 only permutes the LESION+SHELL portion, which is
            # where the subject-identity question is meaningful. This
            # is noted explicitly in the results writeup, not hidden.
            p_true_bg = p_prod_of(feats)
            neg_p_true.append(p_true_bg)
            neg_p_perm.append(p_true_bg)  # unchanged for background rows
        lesion_pool = np.concatenate(lesion_pool).astype(np.float64)
        neg_pool = np.concatenate(neg_pool).astype(np.float64)
        lesion_p_true = np.concatenate(lesion_p_true).astype(np.float64)
        neg_p_true = np.concatenate(neg_p_true).astype(np.float64)
        lesion_p_perm = np.concatenate(lesion_p_perm).astype(np.float64)
        neg_p_perm = np.concatenate(neg_p_perm).astype(np.float64)
        return lesion_pool, neg_pool, lesion_p_true, neg_p_true, lesion_p_perm, neg_p_perm

    fitted = {}
    for seed in seeds:
        print(f'\n--- seed={seed} ---', flush=True)
        lesion_pool, neg_pool, lp_true, np_true, lp_perm, np_perm = fit_r1b_pool_with_coords(seed)
        X = np.concatenate([lesion_pool, neg_pool])
        y_hard = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
        p_true = np.concatenate([lp_true, np_true])
        p_perm = np.concatenate([lp_perm, np_perm])
        q_i = q_alt_of(X)
        print(f'  pool: {len(lesion_pool)} ET, {len(neg_pool)} neg ({time.time()-t0_time:.0f}s)', flush=True)

        fitted[seed] = {}
        clf0 = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X, y_hard)
        fitted[seed]['h0_ordinary'] = (clf0.coef_[0].astype(np.float64), float(clf0.intercept_[0]))

        t_p = np.clip(y_hard - p_true, 0.0, 1.0)
        fitted[seed]['h1_true_prod'] = fit_soft_target(X, t_p)

        t_pi = np.clip(y_hard - p_perm, 0.0, 1.0)
        fitted[seed]['h2_shuffled_prod'] = fit_soft_target(X, t_pi)

        t_pbar = np.clip(y_hard - pbar, 0.0, 1.0)
        fitted[seed]['h3_population_baseline'] = fit_soft_target(X, t_pbar)

        t_q = np.clip(y_hard - q_i, 0.0, 1.0)
        fitted[seed]['h4_alt_model'] = fit_soft_target(X, t_q)

        t_neg = np.clip(p_true - y_hard, 0.0, 1.0)
        fitted[seed]['h5_sign_reversed'] = fit_soft_target(X, t_neg)

        t_mrd = np.clip((y_hard - p_true) / (1.0 - p_true + EPS_MARG), 0.0, 1.0)
        fitted[seed]['h6_normalized_prod'] = fit_soft_target(X, t_mrd)

        for name in CONDS:
            print(f'  fit {name} ({time.time()-t0_time:.0f}s)', flush=True)

    # ---- dense eval (test subjects always use their OWN true p for
    # scoring every condition's FITTED readout -- the permutation only
    # ever affected H2's TRAINING target, never evaluation) ----
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

    sources = ['prod'] + [f'{seed}::{name}' for seed in seeds for name in CONDS]
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

    out = HERE / ('E284_matched_fp_smoke.csv' if smoke else 'E284_matched_fp.csv')
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
    for name in CONDS:
        by_target = by_cond.get(name, {})
        parts = []
        for target in FP_TARGETS:
            vals = by_target.get(target, [])
            parts.append(f'FP~{target}: {np.mean(vals)*100:.1f}%±{np.std(vals)*100:.1f}%')
        print(f'  {name:24s} {"  ".join(parts)}', flush=True)

    print(f'\nE284 complete ({time.time()-t0_time:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
