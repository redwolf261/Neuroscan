"""E281 -- Marginal Recovery Decomposition (MRD), per the user's exact
spec, FROZEN before running (per explicit user instruction: "before
running it, we should freeze the exact equation and protocol, so we
don't subsequently modify the method around the test results").

IDEA: instead of training a readout to predict y (is this voxel ET)
directly, train it to predict the MARGINAL signal production itself
failed to capture -- how much of the positive label is "left over"
after accounting for what production's own probability p_i already
explains. For a true positive voxel (y=1) where production is already
confident (p_i near 1), there is little left to learn; where production
is confidently wrong (p_i near 0), nearly the full positive signal
remains to be recovered. For true negatives (y=0), there is never any
marginal signal regardless of p_i.

  y_marg_i = clip( (y_i - p_i) / (1 - p_i + eps), 0, 1 ),  eps=1e-6

Verified by hand before implementing (y=1,p=0.01 -> 1.0; y=1,p=0.5 ->
1.0; y=1,p=0.99 -> 0.9999; y=0, any p -> 0.0) -- the clip only ever
binds from above for positives near p=1, and from below for all
negatives (since y-p<0 whenever y=0), confirming the formula does what
the spec intends: full positive weight unless production already
explains it, and zero weight for all negatives.

FIVE CONDITIONS, all using R1-B's EXACT unchanged negative/positive
sampling pool (E257-B's curated lesion + local shell + distant
background, seed=999) -- per explicit user choice this session, this
isolates the TARGET/ARCHITECTURE as the only real variable, uncon-
founded by a different training distribution:
  H0 Production        -- frozen w_P, b_P. Unchanged reference.
  H1 R1-B               -- linear on x_i (32-dim D1 features), target=y
                            (ordinary binary label). IDENTICAL to E274/
                            E275/.../E280's R1-B, refit here for a
                            self-contained script.
  H2 R_MRD (main test)  -- linear on [x_i, p_i] (33-dim, p_i appended
                            as an extra feature), target=y_marg.
  H3 MRD-no-p           -- linear on x_i ONLY (32-dim, same as R1-B's
                            input), target=y_marg. Ablation: does
                            removing p_i as an input feature (keeping
                            the marginal target) change anything --
                            isolates whether the TARGET transform alone
                            (independent of also seeing p_i at
                            inference) is what matters.
  H4 MRD-ordinary-Y     -- linear on [x_i, p_i] (33-dim, same input as
                            H2), target=y (ordinary). Ablation: does
                            adding p_i as an input feature alone help,
                            even with the ordinary target -- isolates
                            the target-transform's specific contribution
                            from the extra-feature's contribution.

ARCHITECTURE (per explicit user decision this session): ALL of H1-H4
are plain LogisticRegression (max_iter=500, C=1.0,
class_weight='balanced'), identical hyperparameters to R1-B throughout
this entire investigation. H2/H4's 33-dim input is built by appending
p_i (production's own sigmoid probability, computed from the frozen
w_P/b_P on that same voxel's x_i) as an extra feature. For H2/H3, the
regression TARGET replaces the usual {0,1} label with continuous
y_marg in [0,1] -- scikit-learn's LogisticRegression requires discrete
classes, so following the one standard approach for a continuous
[0,1] target that preserves a probabilistic interpretation, each
training sample is given classification weight y_marg_i for the
positive class and (1 - y_marg_i) for the negative class, i.e. SOFT
LABELS via sample duplication is avoided; instead a weighted log-loss
formulation is used by fitting on both labels per sample, each
weighted by sample_weight -- see the `fit_soft_target` helper below
for the exact mechanism, verified against a hand-computed small example
before trusting on real data.

EVALUATION (per explicit user decision this session, "do not change
the evaluation now"): EXACTLY E274's unchanged protocol -- raw sigmoid
output of each readout swept over the SAME logit-spaced threshold
grid, matched-mean-FP-per-subject targets (100/500/1000) found on
det_test, lesion-level sensitivity reported by the SAME non-circular
G1/G2/G3 phenotype groups (built from production probability + ground
truth only, never touching any of H1-H4).

DECISION RULE (per explicit user pre-registration, frozen before
running): compare H2 (R_MRD) against H1 (R1-B) on G1+G2 (difficult
lesion) sensitivity at matched FP budgets.
  MRD > R1-B  -> the marginal-recovery reformulation is a genuine
                 improvement over predicting y directly from the same
                 negative pool -- a real, novel algorithmic candidate.
  MRD ~= R1-B -> the reformulation does not help beyond R1-B's existing
                 construction; H3/H4 then help explain WHY (is p_i-as-
                 feature or the target-transform responsible for
                 whatever small effect exists).
  MRD < R1-B  -> the marginal-target reformulation is actively worse,
                 e.g. because duplicating/weighting samples by y_marg
                 effectively just downweights already-confident
                 positives, discarding real training signal without
                 compensating benefit.
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


def y_marg(y, p):
    """Marginal recovery target, per the frozen spec. y, p: arrays in [0,1]."""
    return np.clip((y - p) / (1.0 - p + EPS_MARG), 0.0, 1.0)


def fit_soft_target(X, y_soft, max_iter=500, C=1.0):
    """Fits a LogisticRegression on a continuous [0,1] soft target by
    the standard weighted-dual-label trick: each sample i contributes
    BOTH a positive-class row (weight=y_soft_i) and a negative-class
    row (weight=1-y_soft_i), using the SAME feature vector x_i for
    both rows. This is equivalent (up to the loss being a convex
    combination per-sample rather than drawing one hard label) to
    minimizing the weighted cross-entropy against the continuous
    target, which is the correct generalization of binary log-loss to
    soft labels. Verified below on a small hand-built example before
    running on this experiment's real data.
    """
    X2 = np.concatenate([X, X], axis=0)
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w2 = np.concatenate([y_soft, 1.0 - y_soft])
    # drop zero-weight rows (keeps fit numerically clean, no effect on result)
    keep = w2 > 1e-12
    X2, y2, w2 = X2[keep], y2[keep], w2[keep]
    clf = LogisticRegression(max_iter=max_iter, C=C, class_weight='balanced')
    clf.fit(X2, y2, sample_weight=w2)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])


def _verify_fit_soft_target():
    """Hand-built sanity check: with y_soft all 0 or 1, fit_soft_target
    must reduce EXACTLY to ordinary LogisticRegression (since one of
    the two weight columns is always 1e-12-dropped)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y_hard = (X[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(np.float64)
    w_soft, b_soft = fit_soft_target(X, y_hard)
    clf_hard = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X, y_hard)
    w_hard, b_hard = clf_hard.coef_[0], float(clf_hard.intercept_[0])
    assert np.allclose(w_soft, w_hard, atol=1e-6), (w_soft, w_hard)
    assert abs(b_soft - b_hard) < 1e-6, (b_soft, b_hard)
    print(f'  [verify] fit_soft_target matches ordinary LogisticRegression on hard labels '
         f'(max coef diff={np.max(np.abs(w_soft - w_hard)):.2e}, '
         f'intercept diff={abs(b_soft - b_hard):.2e})', flush=True)


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    print('Verifying fit_soft_target helper on a hand-built hard-label example...', flush=True)
    _verify_fit_soft_target()

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

    w_p_np = w_et.astype(np.float64)
    b_p_np = float(b_et)

    def p_prod_of(x):
        """Production's own sigmoid probability for raw D1 feature rows x (n,32)."""
        z = x @ w_p_np + b_p_np
        return 1.0 / (1.0 + np.exp(-z))

    # ---- build R1-B's EXACT shared pool ONCE (seed=999), reused for
    # ALL of H1-H4 (per explicit user decision: isolate target/
    # architecture as the only variable) ----
    print('\nBuilding R1-B\'s shared lesion+shell+distant-background pool (seed=999)...', flush=True)
    rng = np.random.default_rng(999)
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
    print(f'  pool: {len(lesion_pool)} ET positives, {len(neg_pool)} negatives (shell+distant) '
         f'({time.time()-t0:.0f}s)', flush=True)

    X = np.concatenate([lesion_pool, neg_pool])
    y_hard = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
    p_i = p_prod_of(X)
    y_marg_vals = y_marg(y_hard, p_i)
    print(f'  y_marg stats: positives mean={y_marg_vals[y_hard==1].mean():.4f} '
         f'(min={y_marg_vals[y_hard==1].min():.4f}, max={y_marg_vals[y_hard==1].max():.4f}), '
         f'negatives mean={y_marg_vals[y_hard==0].mean():.6f} (should be ~0)', flush=True)
    assert y_marg_vals[y_hard == 0].max() < 1e-4, 'BUG: negatives should have y_marg ~ 0'

    X_aug = np.concatenate([X, p_i.reshape(-1, 1)], axis=1)  # 33-dim, for H2/H4

    # ---- H1: R1-B, linear on x_i, target=y (ordinary) ----
    print('\nFitting H1 R1-B (x_i, target=y)...', flush=True)
    clf_h1 = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X, y_hard)
    w_h1, b_h1 = clf_h1.coef_[0].astype(np.float64), float(clf_h1.intercept_[0])

    # ---- H2: R_MRD (main test), linear on [x_i, p_i], target=y_marg ----
    print('Fitting H2 R_MRD ([x_i,p_i], target=y_marg)...', flush=True)
    w_h2, b_h2 = fit_soft_target(X_aug, y_marg_vals)

    # ---- H3: MRD-no-p, linear on x_i ONLY, target=y_marg ----
    print('Fitting H3 MRD-no-p (x_i, target=y_marg)...', flush=True)
    w_h3, b_h3 = fit_soft_target(X, y_marg_vals)

    # ---- H4: MRD-ordinary-Y, linear on [x_i, p_i], target=y (ordinary) ----
    print('Fitting H4 MRD-ordinary-Y ([x_i,p_i], target=y)...', flush=True)
    clf_h4 = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_aug, y_hard)
    w_h4, b_h4 = clf_h4.coef_[0].astype(np.float64), float(clf_h4.intercept_[0])

    print(f'All 4 readouts fit ({time.time()-t0:.0f}s)', flush=True)

    w_p_t = torch.from_numpy(w_p_np).float().to(dev)
    w_h1_t = torch.from_numpy(w_h1).float().to(dev)
    w_h2_t = torch.from_numpy(w_h2).float().to(dev)  # 33-dim
    w_h3_t = torch.from_numpy(w_h3).float().to(dev)
    w_h4_t = torch.from_numpy(w_h4).float().to(dev)  # 33-dim

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
        with torch.no_grad():
            z_prod = flat @ w_p_t + float(b_p_np)
            p_prod_flat = torch.sigmoid(z_prod)
            p_prod = p_prod_flat.reshape(D, H, W).cpu().numpy()
            flat_aug = torch.cat([flat, p_prod_flat.reshape(-1, 1)], dim=1)  # (N,33)

            p_h1 = torch.sigmoid(flat @ w_h1_t + b_h1).reshape(D, H, W).cpu().numpy()
            p_h2 = torch.sigmoid(flat_aug @ w_h2_t + b_h2).reshape(D, H, W).cpu().numpy()
            p_h3 = torch.sigmoid(flat @ w_h3_t + b_h3).reshape(D, H, W).cpu().numpy()
            p_h4 = torch.sigmoid(flat_aug @ w_h4_t + b_h4).reshape(D, H, W).cpu().numpy()
        del d1, flat, flat_aug, img_t, stages, z_prod, p_prod_flat
        torch.cuda.empty_cache()
        result = (p_prod, p_h1, p_h2, p_h3, p_h4, tgt_c, img_c)
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
        p_prod, p_h1, p_h2, p_h3, p_h4, tgt_c, img_c = dense_eval(sid)
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

    pheno_out = HERE / ('E281_phenotypes_smoke.csv' if smoke else 'E281_phenotypes.csv')
    with open(pheno_out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['subject_id', 'comp_id', 'phenotype', 'size', 'p_max_prod'])
        w.writeheader()
        for rec in lesion_records:
            w.writerow(rec)

    print('\nSweeping thresholds (IDENTICAL grid to E274, logit-spaced near 1.0)...', flush=True)
    threshold_grid = np.concatenate([
        np.linspace(0.01, 0.1, 10),
        np.linspace(0.1, 0.9, 20),
        1 - 1 / (1 + np.exp(np.linspace(2, 30, 60))),
    ])
    threshold_grid = np.unique(np.round(threshold_grid, 8))

    sources = ['prod', 'h1_r1b', 'h2_mrd', 'h3_mrd_no_p', 'h4_mrd_ordinary_y']
    fp_lists = {s: {t: [] for t in threshold_grid} for s in sources}

    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        p_prod, p_h1, p_h2, p_h3, p_h4, tgt_c, img_c = dense_eval(sid)
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        maps = {'prod': p_prod, 'h1_r1b': p_h1, 'h2_mrd': p_h2,
               'h3_mrd_no_p': p_h3, 'h4_mrd_ordinary_y': p_h4}
        for src_name, p_map in maps.items():
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
            p_prod, p_h1, p_h2, p_h3, p_h4, tgt_c, img_c = dense_eval(sid)
            p_map = {'prod': p_prod, 'h1_r1b': p_h1, 'h2_mrd': p_h2,
                     'h3_mrd_no_p': p_h3, 'h4_mrd_ordinary_y': p_h4}[src_name]
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

    out = HERE / ('E281_matched_fp_smoke.csv' if smoke else 'E281_matched_fp.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['readout', 'fp_target', 'threshold', 'mean_fp_actual',
                                           'phenotype', 'sensitivity'])
        w.writeheader()
        for row in matched_results:
            w.writerow(row)

    print(f'\nE281 complete ({time.time()-t0:.0f}s). wrote {pheno_out.name}, {out.name}', flush=True)


if __name__ == '__main__':
    main()
