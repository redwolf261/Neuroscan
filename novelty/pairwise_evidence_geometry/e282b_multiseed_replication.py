"""E282-B -- Multi-Seed MRD Replication + Matched-FP/FROC Evaluation,
per the user's exact spec. Gating stage #2 of the E282 validation
campaign (per explicit user sequencing: A->B->C->D first, before the
later ablation/permutation/mechanistic stages). Folds in E282-C
(matched-FP + FROC-style full-curve evaluation) since both reuse the
identical dense_eval infrastructure -- running them as one script
avoids rebuilding the (expensive) dense D1 cache twice.

FROZEN METHOD DEFINITIONS (per explicit user spec + two
AskUserQuestion resolutions this session):
  H0 Production       -- frozen w_P, b_P. hat_y = p.
  H1 R1-B              -- linear on x_i (32-dim), target=y (ordinary),
                           R1-B's curated shell+distant negative pool.
                           hat_y = r (raw sigmoid output, UNCHANGED
                           from every R1-B fit in this investigation).
  H2 MRD                -- linear on [x_i,p_i] (33-dim), target=y_marg,
                           R1-B's SAME curated pool. Per explicit user
                           decision this session, hat_y uses the
                           RECONSTRUCTION FORMULA:
                               hat_y = p + (1-p) * r
                           (a real, deliberate change from E281, which
                           evaluated raw r directly -- E282 re-tests
                           the underlying finding under this more
                           principled, mathematically-motivated scoring
                           rule, per the user's own E282 formalization:
                           "production contribution + remaining
                           recoverable contribution").
  H3 MRD-no-p           -- linear on x_i ONLY (32-dim), target=y_marg,
                           R1-B's SAME curated pool. hat_y = p + (1-p)*r
                           (same reconstruction, r has no p_i input).
  H4 MRD-ordinary-Y      -- linear on [x_i,p_i] (33-dim), target=y
                           (ordinary), R1-B's SAME curated pool.
                           hat_y = r (raw; this ablation's PURPOSE is
                           to show that just adding p_i as an input
                           feature, with the ORDINARY target, does not
                           reproduce R1-B's behavior via reconstruction
                           -- so it is evaluated the same way H1 always
                           has been, as a direct classifier, matching
                           its own training target).
  H5 MRD-conventional-neg -- linear on x_i ONLY (32-dim, same as H3),
                           target=y_marg, but trained on E275's
                           CONVENTIONAL negative pool (ALL ET voxels +
                           uniform-random non-ET subsample, matched in
                           size) instead of R1-B's curated shell+distant
                           pool -- isolating the negative-DISTRIBUTION
                           variable from the target-transform variable,
                           per explicit user requirement ("we need to
                           separate target transformation from
                           negative-distribution effect"). hat_y =
                           p + (1-p)*r (same reconstruction as H2/H3).

5 SEEDS (999, 4242, 7, 123, 2024) for H1-H5, per explicit user
requirement ("mandatory... at least 5 seeds"). Each seed independently
re-samples the shell/distant/conventional negative pools (same RNG
convention as E274's seed999/seed4242 reproducibility check) and refits
every classifier from scratch.

EVALUATION: EXACTLY E274's matched-FP-budget protocol (same logit-
spaced threshold grid, same det_test subjects, same non-circular
G1/G2/G3 phenotype definition from production probability + ground
truth only) -- reported per-seed (not averaged away) so E282-D's
subject-level bootstrap/paired statistics can be computed from the
per-seed per-subject records written here. ADDITIONALLY (E282-C
folded in): a full FROC-style curve (lesion sensitivity vs mean FP/
subject) is written at EVERY threshold-grid point, not just the 3
matched targets, to support precision/threshold-independent analysis
without re-running the expensive dense evaluation.
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


def y_marg_fn(y, p):
    return np.clip((y - p) / (1.0 - p + EPS_MARG), 0.0, 1.0)


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

    model = load_model(dev)
    w_et, b_et = get_w_prod(model)
    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    if smoke:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']][:30]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
        conv_subjects = sorted(splits['det_train'])[:30]
        seeds = SEEDS[:2]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])
        conv_subjects = sorted(splits['det_train'])
        seeds = SEEDS

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    t0 = time.time()

    w_p_np = w_et.astype(np.float64)
    b_p_np = float(b_et)

    def p_prod_of(x):
        z = x @ w_p_np + b_p_np
        return 1.0 / (1.0 + np.exp(-z))

    # ---- precompute conventional-head candidate pools ONCE (E275-style,
    # subject-level, independent of seed structure beyond the final
    # subsample draw) for H5 ----
    print('Precomputing conventional-head (H5 negative source) candidate pools '
         '(E275-style, dense ET + per-subject non-ET subsample)...', flush=True)
    conv_et_pool_list, conv_neg_candidates_list = [], []
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
            conv_et_pool_list.append(d1[:, et_idx[0], et_idx[1], et_idx[2]].T)
        non_et_mask = brain_mask & (~true_et)
        non_et_idx = np.where(non_et_mask)
        if len(non_et_idx[0]) > 0:
            n_sample = min(500, len(non_et_idx[0]))
            rng0 = np.random.default_rng(0)  # fixed, pool-construction only (not a fitted seed)
            sel = rng0.choice(len(non_et_idx[0]), size=n_sample, replace=False)
            coords = (non_et_idx[0][sel], non_et_idx[1][sel], non_et_idx[2][sel])
            conv_neg_candidates_list.append(d1[:, coords[0], coords[1], coords[2]].T)
        del d1, img_t, stages
        torch.cuda.empty_cache()
    conv_et_pool_all = np.concatenate(conv_et_pool_list).astype(np.float64)
    conv_neg_candidates_all = np.concatenate(conv_neg_candidates_list).astype(np.float64)
    print(f'  conventional candidates: {len(conv_et_pool_all)} ET (dense), '
         f'{len(conv_neg_candidates_all)} non-ET candidates ({time.time()-t0:.0f}s)', flush=True)

    def fit_r1b_pool(seed):
        """Builds R1-B's curated shell+distant pool for this seed. Returns
        (lesion_pool, neg_pool), both (n,32) float64."""
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

    def fit_conv_pool(seed, target_neg_size):
        """Subsamples the conventional candidate pools to this seed's draw,
        sized to match target_neg_size (R1-B's own neg pool size at this seed)."""
        rng = np.random.default_rng(seed + 500000)  # offset namespace, avoids seed collision with R1-B's own rng stream
        if len(conv_neg_candidates_all) > target_neg_size:
            sel = rng.choice(len(conv_neg_candidates_all), size=target_neg_size, replace=False)
            neg_pool = conv_neg_candidates_all[sel]
        else:
            neg_pool = conv_neg_candidates_all
        return conv_et_pool_all, neg_pool

    # ---- per-seed fits for all 5 conditions ----
    fitted = {}  # seed -> dict of {condition: (w, b, is_33dim)}
    for seed in seeds:
        print(f'\n--- seed={seed} ---', flush=True)
        lesion_pool, neg_pool = fit_r1b_pool(seed)
        print(f'  R1-B pool: {len(lesion_pool)} ET, {len(neg_pool)} shell+distant neg '
             f'({time.time()-t0:.0f}s)', flush=True)

        X_r1b = np.concatenate([lesion_pool, neg_pool])
        y_r1b = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
        p_r1b = p_prod_of(X_r1b)
        y_marg_r1b = y_marg_fn(y_r1b, p_r1b)
        X_r1b_aug = np.concatenate([X_r1b, p_r1b.reshape(-1, 1)], axis=1)

        # H1: R1-B, x, target=y
        clf_h1 = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_r1b, y_r1b)
        w_h1, b_h1 = clf_h1.coef_[0].astype(np.float64), float(clf_h1.intercept_[0])

        # H2: MRD, [x,p], target=y_marg
        w_h2, b_h2 = fit_soft_target(X_r1b_aug, y_marg_r1b)

        # H3: MRD-no-p, x, target=y_marg
        w_h3, b_h3 = fit_soft_target(X_r1b, y_marg_r1b)

        # H4: MRD-ordinary-Y, [x,p], target=y
        clf_h4 = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_r1b_aug, y_r1b)
        w_h4, b_h4 = clf_h4.coef_[0].astype(np.float64), float(clf_h4.intercept_[0])

        # H5: MRD-conventional-neg, x, target=y_marg, CONVENTIONAL negatives
        conv_et, conv_neg = fit_conv_pool(seed, len(neg_pool))
        X_h5 = np.concatenate([conv_et, conv_neg])
        y_h5_hard = np.concatenate([np.ones(len(conv_et)), np.zeros(len(conv_neg))])
        p_h5 = p_prod_of(X_h5)
        y_marg_h5 = y_marg_fn(y_h5_hard, p_h5)
        w_h5, b_h5 = fit_soft_target(X_h5, y_marg_h5)
        print(f'  H5 conventional pool: {len(conv_et)} ET (dense), {len(conv_neg)} uniform-random neg '
             f'({time.time()-t0:.0f}s)', flush=True)

        fitted[seed] = {
            'h1_r1b': (w_h1, b_h1, False, False),            # (w, b, is_33dim, use_reconstruction)
            'h2_mrd': (w_h2, b_h2, True, True),
            'h3_mrd_no_p': (w_h3, b_h3, False, True),
            'h4_mrd_ordinary_y': (w_h4, b_h4, True, False),
            'h5_mrd_conv_neg': (w_h5, b_h5, False, True),
        }
        print(f'  all 5 conditions fit for seed={seed} ({time.time()-t0:.0f}s)', flush=True)

    # ---- dense eval, all seeds x all conditions in one cache pass ----
    w_p_t = torch.from_numpy(w_p_np).float().to(dev)
    tensors = {}
    for seed in seeds:
        for cond, (w, b, is_33, use_recon) in fitted[seed].items():
            tensors[(seed, cond)] = (torch.from_numpy(w).float().to(dev), float(b), is_33, use_recon)

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
            p_prod_flat = torch.sigmoid(z_prod)
            maps['prod'] = p_prod_flat.reshape(D, H, W).cpu().numpy()
            flat_aug = torch.cat([flat, p_prod_flat.reshape(-1, 1)], dim=1)
            for seed in seeds:
                for cond, (w_t, b, is_33, use_recon) in [(c, tensors[(seed, c)]) for c in fitted[seed]]:
                    inp = flat_aug if is_33 else flat
                    r = torch.sigmoid(inp @ w_t + b)
                    if use_recon:
                        out = p_prod_flat + (1.0 - p_prod_flat) * r
                    else:
                        out = r
                    maps[(seed, cond)] = out.reshape(D, H, W).cpu().numpy()
        del d1, flat, flat_aug, img_t, stages, z_prod, p_prod_flat
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
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    n_by_phen = {}
    for rec in lesion_records:
        n_by_phen[rec['phenotype']] = n_by_phen.get(rec['phenotype'], 0) + 1
    print(f'\nPhenotype group sizes: {n_by_phen}', flush=True)
    lesion_by_sid = {}
    for rec in lesion_records:
        lesion_by_sid.setdefault(rec['subject_id'], []).append(rec)

    pheno_out = HERE / ('E282B_phenotypes_smoke.csv' if smoke else 'E282B_phenotypes.csv')
    with open(pheno_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['subject_id', 'comp_id', 'phenotype', 'size', 'p_max_prod'])
        w.writeheader()
        for rec in lesion_records:
            w.writerow(rec)

    print('\nSweeping thresholds (full grid, per seed per condition, for FROC curves)...', flush=True)
    # EXTENDED DOWNWARD (found necessary in smoke testing before trusting
    # any result): the reconstruction-based conditions (H2/H3/H5, all
    # using hat_y = p + (1-p)*r with r fit via the dual-row soft-target
    # trick) produce a systematically higher background score floor than
    # R1-B's ordinary fit -- smoke testing found r's mean on TRUE DISTANT
    # BACKGROUND (not the curated training pool, the full dense test-
    # subject background) is ~2.4x higher for H3 than H1 (0.041 vs 0.017),
    # and the frac of background exceeding 0.5 is ~14x higher (0.39% vs
    # 0.027%) -- small per-voxel differences that compound across ~1.2M
    # background voxels per subject into mean FP ~840,000-970,000 at the
    # grid's OLD lowest threshold (0.01), meaning FP~100/500/1000 targets
    # were simply UNREACHABLE within the old grid's range for these three
    # conditions (all 3 saturated at threshold=1.0, giving the spurious
    # 0% sensitivity seen in the first smoke run). Extended down to 1e-6
    # with fine log-spacing to actually reach the FP~100-1000 regime for
    # H2/H3/H5, while keeping the original grid (which worked correctly
    # for prod/H1/H4) unchanged in its upper range.
    # SECOND BUG found in smoke testing: np.round(..., 10) collapsed the
    # finest near-1.0 grid points (needed to resolve FP in the 300-1500
    # range for H2/H3/H5, whose scores pile up extremely close to 1.0)
    # to the single float value 1.0 -- since sigmoid output is always
    # strictly <1.0, the mask (p_map > 1.0) is always False, silently
    # producing FP=0/sensitivity=0 at that collapsed threshold and
    # making it look like the "closest match" to every FP target even
    # though genuinely distinct, finer thresholds existed before
    # rounding. Fixed by rounding to 15 decimals (enough to keep the
    # near-1.0 logit-spaced points distinct in float64) instead of 10.
    threshold_grid = np.concatenate([
        1 / (1 + np.exp(np.linspace(30, 2, 40))),  # approaches 0 from above, covers the H2/H3/H5 low-threshold regime
        np.linspace(0.01, 0.1, 10),
        np.linspace(0.1, 0.9, 20),
        1 - 1 / (1 + np.exp(np.linspace(2, 40, 80))),  # extended to logit~40 (from ~30) for finer near-1.0 resolution
    ])
    threshold_grid = np.unique(np.round(threshold_grid, 15))

    sources = ['prod'] + [f'{seed}::{cond}' for seed in seeds for cond in fitted[seed]]
    fp_lists = {s: {t: [] for t in threshold_grid} for s in sources}
    # per-subject-per-threshold lesion recovery booleans, for FROC +
    # subject-level bootstrap in E282-D: store once per (source, subject,
    # threshold) whether EACH phenotype group was fully/any-recovered
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
                seed_str, cond = src_name.split('::')
                p_map = maps[(int(seed_str), cond)]
            for t in threshold_grid:
                mask = (p_map > t) & brain_mask
                fp = int((mask & (~true_et)).sum())
                fp_lists[src_name][t].append(fp)
                # lesion recovery for this subject at this threshold, by phenotype
                by_phen = {}
                for rec in recs:
                    cid = rec['comp_id']
                    if cid < 1 or cid > et_n:
                        continue
                    lm = et_lbl == cid
                    by_phen.setdefault(rec['phenotype'], []).append(int(((p_map > t) & lm).any()))
                recovered_by_sub[src_name][t][sid] = by_phen
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

    # ---- FROC curve: write mean FP + sensitivity (ALL + per-phenotype) at EVERY grid threshold ----
    print('\nWriting FROC curve (all thresholds)...', flush=True)
    froc_rows = []
    for src_name in sources:
        for t in threshold_grid:
            per_sub = recovered_by_sub[src_name][t]
            all_vals, phen_vals = [], {'G1_confidently_missed': [], 'G2_low_confidence': [],
                                      'G3_confidently_detected': []}
            for sid, by_phen in per_sub.items():
                for phen, vals in by_phen.items():
                    phen_vals.setdefault(phen, []).extend(vals)
                    all_vals.extend(vals)
            row = {'source': src_name, 'threshold': t, 'mean_fp': mean_fp_by_threshold[src_name][t],
                  'sens_all': float(np.mean(all_vals)) if all_vals else float('nan')}
            for phen in ['G1_confidently_missed', 'G2_low_confidence', 'G3_confidently_detected']:
                vals = phen_vals.get(phen, [])
                row[f'sens_{phen}'] = float(np.mean(vals)) if vals else float('nan')
            froc_rows.append(row)

    froc_out = HERE / ('E282B_froc_smoke.csv' if smoke else 'E282B_froc.csv')
    with open(froc_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['source', 'threshold', 'mean_fp', 'sens_all',
                                           'sens_G1_confidently_missed', 'sens_G2_low_confidence',
                                           'sens_G3_confidently_detected'])
        w.writeheader()
        for row in froc_rows:
            w.writerow(row)

    # ---- matched-FP sensitivity, PER-SUBJECT records (for E282-D bootstrap) ----
    print('\nComputing matched-FP sensitivity (per-subject records for bootstrap)...', flush=True)
    matched_rows = []
    per_subject_rows = []  # subject_id, source, fp_target, phenotype, recovered (for paired bootstrap)
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
                    # per-subject mean (for bootstrap: one scalar per subject per phenotype)
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
        print(f'  {src_name}: thresholds={matched_t} '
             f'(actual FP: {[round(mean_fp_by_threshold[src_name][t],1) for t in matched_t.values()]})',
             flush=True)

    matched_out = HERE / ('E282B_matched_fp_smoke.csv' if smoke else 'E282B_matched_fp.csv')
    with open(matched_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['source', 'fp_target', 'threshold', 'mean_fp_actual',
                                           'phenotype', 'sensitivity'])
        w.writeheader()
        for row in matched_rows:
            w.writerow(row)

    persub_out = HERE / ('E282B_persubject_smoke.csv' if smoke else 'E282B_persubject.csv')
    with open(persub_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['subject_id', 'source', 'fp_target', 'phenotype',
                                           'sensitivity', 'n_lesions'])
        w.writeheader()
        for row in per_subject_rows:
            w.writerow(row)

    # ---- summary table: mean +/- SD across seeds, per condition, G1 phenotype ----
    print('\n' + '=' * 70, flush=True)
    print('SUMMARY: G1 (confidently-missed) sensitivity, mean +/- SD across seeds', flush=True)
    print('=' * 70, flush=True)
    by_cond = {}
    for row in matched_rows:
        if row['phenotype'] != 'G1_confidently_missed':
            continue
        if row['source'] == 'prod':
            continue
        seed_str, cond = row['source'].split('::')
        by_cond.setdefault(cond, {}).setdefault(row['fp_target'], []).append(row['sensitivity'])
    for cond, by_target in by_cond.items():
        parts = []
        for target in FP_TARGETS:
            vals = by_target.get(target, [])
            parts.append(f'FP~{target}: {np.mean(vals)*100:.1f}%±{np.std(vals)*100:.1f}%')
        print(f'  {cond:22s} {"  ".join(parts)}', flush=True)

    print(f'\nE282-B complete ({time.time()-t0:.0f}s). wrote {pheno_out.name}, {froc_out.name}, '
         f'{matched_out.name}, {persub_out.name}', flush=True)


if __name__ == '__main__':
    main()
