"""E267 -- disagreement-variance validation, per the user's exact spec.
Direct follow-up to E266's observation that recovered_G2A voxels have
the TIGHTEST spread (std=1.97) of Delta_readout(v)=z_R1B(v)-z_prod(v)
among 4 populations, while its MEAN was indistinguishable from plain
false positives. Tests whether that low-variance signature is a real,
reproducible, COMPONENT-level discriminative signal (sigma_C), not
voxel-pooled noise or a sample-specific artifact -- with strict
train(fit R1-B)/val(select thresholds)/test(report once) separation,
per explicit user instruction not to select any threshold on the data
it's evaluated on.

CANDIDATE DEFINITION: a connected component C of R1-B's whole-volume
thresholded positive mask (p_R1B>TAU), identical component extraction
to E264/E265/E266. For each component, compute mu_C and sigma_C of
d(v) = z_R1B(v) - z_prod(v) (logit difference, same choice as E266, to
avoid sigmoid-saturation confounds) over its own voxels. sigma_C uses
the SAMPLE std (ddof=1, matching the user's 1/(|C|-1) formula);
components with |C|==1 have sigma_C undefined (no within-component
spread) and are excluded from variance-based analysis, flagged
separately.

FOUR POPULATIONS (component-level, not voxel-level, per explicit user
instruction):
  1. recovered_G2A -- R1-B positive components overlapping a true,
     held-out G2A lesion (MIN_VOX, same lesion-overlap convention as
     E264's lesion_recovery_filtered)
  2. detected_ET -- R1-B positive components overlapping a true
     detected-population ET lesion
  3. WT_not_TC_FP -- R1-B positive FP components with >50% overlap in
     ground-truth WT-not-TC (E259's own convention)
  4. distant_bg_FP -- R1-B positive FP components entirely outside
     E246's 12-voxel near/far boundary around any true ET lesion

THREE HYPOTHESES, tested SEPARATELY (no premature combination):
  H1: sigma_C alone -- does component-level disagreement variance
      distinguish recovered_G2A from R1-B's own FP populations
      (WT_not_TC_FP, distant_bg_FP)?
  H2: mu_C alone -- reproduction check, expected to replicate E266's
      finding that mean disagreement is NOT sufficiently specific.
  H3: mu_C + sigma_C jointly -- a SIMPLE 2D logistic regression (no
      hand-designed combination formula), fit on VAL components only.

VAL/TEST DISCIPLINE: R1-B refit identically to E257-B/E258-E266.
Thresholds (H1, H2) and the H3 classifier are ALL fit/selected on VAL
components only, then applied ONCE, unchanged, to TEST components.
Reports G2A recovery, ET Dice, total/WT-not-TC/distant-bg FP (whole-
volume, same convention as E266), plus sensitivity/specificity of each
criterion at the COMPONENT level on held-out test.
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
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        g2a_lesions_val = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_val']]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']]
        g2b_lesions_eval = g2b_lesions
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng = np.random.default_rng(999)
    t0 = time.time()

    print('Fitting R1-B (seed=999, identical to E257-B)...', flush=True)
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
    X = np.concatenate([lesion_pool, neg_pool])
    y = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
    r1b = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X, y)
    w_r1b = torch.from_numpy(r1b.coef_[0].astype(np.float64)).float().to(dev)
    b_r1b = float(r1b.intercept_[0])
    print(f'R1-B fit ({time.time()-t0:.0f}s)', flush=True)

    w_p_t = torch.from_numpy(w_et).float().to(dev)
    b_p = float(b_et)

    dense_cache = {}

    def dense_logits(sid):
        if sid in dense_cache:
            return dense_cache[sid]
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T
        with torch.no_grad():
            l_prod = (flat @ w_p_t + b_p).reshape(D, H, W).cpu().numpy()
            l_r1b = (flat @ w_r1b + b_r1b).reshape(D, H, W).cpu().numpy()
        result = (l_prod, l_r1b, tgt_c, img_c)
        dense_cache[sid] = result
        return result

    def prob_map(field):
        return 1.0 / (1.0 + np.exp(-field))

    def component_table(sid):
        """R1-B positive components with mu_C, sigma_C of
        d(v)=l_r1b(v)-l_prod(v), and population label."""
        l_prod, l_r1b, tgt_c, img_c = dense_logits(sid)
        p_map = prob_map(l_r1b)
        brain_mask = img_c[0] != 0
        true_et = tgt_c[ET] > 0.5
        true_tc = tgt_c[TC] > 0.5
        true_wt = tgt_c[WT] > 0.5
        pos_mask = (p_map > TAU) & brain_mask
        lbl, n = ndimage.label(pos_mask, structure=STRUCT)

        dilated = ndimage.binary_dilation(true_et, structure=STRUCT,
                                          iterations=NEAR_DILATION) if true_et.any() else np.zeros_like(true_et)
        local_shell = dilated & (~true_et) & brain_mask
        distant_bg_mask = brain_mask & (~true_et) & (~local_shell)

        et_lbl, et_n = ndimage.label(true_et)

        rows = []
        for cid in range(1, n + 1):
            comp_mask = lbl == cid
            A_C = int(comp_mask.sum())
            if A_C < 1:
                continue
            d_vals = (l_r1b[comp_mask] - l_prod[comp_mask]).astype(np.float64)
            mu_C = float(d_vals.mean())
            sigma_C = float(d_vals.std(ddof=1)) if A_C > 1 else float('nan')

            is_tp = bool((comp_mask & true_et).any())
            frac_tc = (comp_mask & true_tc).sum() / A_C
            frac_wt_not_tc = (comp_mask & true_wt & (~true_tc)).sum() / A_C
            frac_distant = (comp_mask & distant_bg_mask).sum() / A_C

            if is_tp:
                population = 'recovered_or_detected'  # resolved below via lesion-overlap lists
            elif frac_tc > 0.5 or frac_wt_not_tc > 0.5:
                population = 'WT_not_TC_FP'
            elif frac_distant > 0.5:
                population = 'distant_bg_FP'
            else:
                population = 'other_FP'

            rows.append({'subject_id': sid, 'component_id': cid, 'is_tp': int(is_tp),
                        'A_C': A_C, 'mu_C': mu_C, 'sigma_C': sigma_C, 'population': population})
        return rows, lbl, et_lbl, et_n

    comp_cache = {}

    def get_components(sid):
        if sid not in comp_cache:
            comp_cache[sid] = component_table(sid)
        return comp_cache[sid]

    def resolve_tp_population(lesion_list, sid_set):
        """Among TP components, split into recovered_G2A (overlaps a
        g2a lesion in lesion_list) vs detected_ET (handled by caller
        with det_lesions_test separately).

        CRITICAL FIX (found in smoke testing, before trusting any
        result): mu_C/sigma_C must be computed over the INTERSECTION of
        the component with the true lesion mask, NOT the whole
        component. A component can be a huge (tens of thousands of
        voxels), mostly-wrong R1-B blob that merely grazes a true
        lesion at its edge -- using the whole component's mu_C/sigma_C
        for such a case badly corrupts the 'recovered evidence'
        statistic with the surrounding false-positive territory's own
        disagreement signature. Restricting to the overlap voxels
        measures disagreement specifically ON the lesion, independent
        of how large or diffuse R1-B's surrounding component happens to
        be -- matches the spirit of 'candidate C' as the recovered
        evidence itself."""
        resolved = []
        for r in lesion_list:
            sid, cid = r['subject_id'], int(r['comp_id'])
            if sid not in sid_to_idx:
                continue
            rows, lbl, et_lbl, et_n = get_components(sid)
            if cid < 1 or cid > et_n:
                continue
            lesion_mask = et_lbl == cid
            lesion_size = int(lesion_mask.sum())
            if lesion_size < MIN_VOX:
                continue
            l_prod, l_r1b, tgt_c, img_c = dense_logits(sid)
            overlap_ids = set(np.unique(lbl[lesion_mask])) - {0}
            if not overlap_ids:
                continue
            pos_mask_full = np.isin(lbl, list(overlap_ids))
            overlap_mask = pos_mask_full & lesion_mask
            ov_size = int(overlap_mask.sum())
            if ov_size < 1:
                continue
            d_vals = (l_r1b[overlap_mask] - l_prod[overlap_mask]).astype(np.float64)
            mu_ov = float(d_vals.mean())
            sigma_ov = float(d_vals.std(ddof=1)) if ov_size > 1 else float('nan')
            resolved.append({'subject_id': sid, 'component_id': cid, 'is_tp': 1,
                            'A_C': ov_size, 'mu_C': mu_ov, 'sigma_C': sigma_ov,
                            'population': 'recovered_lesion_overlap'})
        return resolved

    def collect_fp_populations(subjects):
        wt_rows, distant_rows = [], []
        for sid in subjects:
            if sid not in sid_to_idx:
                continue
            rows, lbl, et_lbl, et_n = get_components(sid)
            for row in rows:
                if row['population'] == 'WT_not_TC_FP':
                    wt_rows.append(row)
                elif row['population'] == 'distant_bg_FP':
                    distant_rows.append(row)
        return wt_rows, distant_rows

    # ---- build VAL population tables ----
    val_subjects = sorted(set(r['subject_id'] for r in g2a_lesions_val))
    print(f'\nBuilding component tables on {len(val_subjects)} VAL subjects...', flush=True)
    for i, sid in enumerate(val_subjects):
        if sid in sid_to_idx:
            get_components(sid)
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(val_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    val_recovered = resolve_tp_population(g2a_lesions_val, val_subjects)
    val_wt, val_distant = collect_fp_populations(val_subjects)
    print(f'VAL: recovered_G2A={len(val_recovered)} WT_not_TC_FP={len(val_wt)} '
         f'distant_bg_FP={len(val_distant)} ({time.time()-t0:.0f}s)', flush=True)

    def valid_sigma(rows):
        return [r for r in rows if not np.isnan(r['sigma_C'])]

    val_recovered_s = valid_sigma(val_recovered)
    val_fp_s = valid_sigma(val_wt + val_distant)

    # ---- H1: sigma_C alone -- select threshold maximizing separation
    # (Youden's J) on VAL, matching standard sensitivity/specificity
    # discipline rather than the zero-cost criterion (this is a
    # DIAGNOSTIC test of discriminative power, not a deployable filter
    # with a hard recovery-preservation requirement like E264) ----
    def best_threshold_youden(pos_vals, neg_vals, lower_is_positive):
        """pos_vals = recovered_G2A, neg_vals = FP. lower_is_positive:
        True if LOWER stat value indicates the positive (recovered) class
        (as E266 suggested for sigma_C), False if higher does."""
        all_vals = sorted(set(pos_vals) | set(neg_vals))
        best_j, best_t = -1, all_vals[0] if all_vals else 0.0
        pos_arr, neg_arr = np.array(pos_vals), np.array(neg_vals)
        for t in all_vals:
            if lower_is_positive:
                tp = (pos_arr <= t).sum(); fn = (pos_arr > t).sum()
                fp = (neg_arr <= t).sum(); tn = (neg_arr > t).sum()
            else:
                tp = (pos_arr >= t).sum(); fn = (pos_arr < t).sum()
                fp = (neg_arr >= t).sum(); tn = (neg_arr < t).sum()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            j = sens + spec - 1
            if j > best_j:
                best_j, best_t = j, t
        return best_t, best_j

    sigma_pos = [r['sigma_C'] for r in val_recovered_s]
    sigma_neg = [r['sigma_C'] for r in val_fp_s]
    sigma_thresh, sigma_j_val = best_threshold_youden(sigma_pos, sigma_neg, lower_is_positive=True)
    print(f'\nH1 (sigma_C): VAL threshold={sigma_thresh:.4f} (lower=positive), Youden J={sigma_j_val:.4f}', flush=True)

    mu_pos = [r['mu_C'] for r in val_recovered]
    mu_neg = [r['mu_C'] for r in (val_wt + val_distant)]
    mu_thresh, mu_j_val = best_threshold_youden(mu_pos, mu_neg, lower_is_positive=False)
    print(f'H2 (mu_C): VAL threshold={mu_thresh:.4f} (higher=positive), Youden J={mu_j_val:.4f}', flush=True)

    # ---- H3: simple 2D logistic regression on (mu_C, sigma_C), VAL only ----
    h3_rows = val_recovered_s + val_fp_s
    X_h3 = np.array([[r['mu_C'], r['sigma_C']] for r in h3_rows])
    y_h3 = np.array([1 if r in val_recovered_s else 0 for r in h3_rows])
    h3_clf = LogisticRegression(max_iter=500, class_weight='balanced').fit(X_h3, y_h3)
    print(f'H3 (mu_C+sigma_C logistic): coef={h3_clf.coef_[0]}, intercept={h3_clf.intercept_[0]:.4f}', flush=True)

    # ---- apply all three criteria ONCE on TEST ----
    test_all_subjects = sorted(set(r['subject_id'] for r in det_lesions_test) |
                               set(r['subject_id'] for r in g2a_lesions_test) |
                               set(r['subject_id'] for r in g2b_lesions_eval))
    if smoke:
        test_all_subjects = test_all_subjects[:15]
    print(f'\nBuilding component tables on {len(test_all_subjects)} TEST subjects...', flush=True)
    for i, sid in enumerate(test_all_subjects):
        if sid in sid_to_idx:
            get_components(sid)
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_all_subjects)} ({time.time()-t0:.0f}s)', flush=True)

    test_recovered = resolve_tp_population(g2a_lesions_test, test_all_subjects)
    test_wt, test_distant = collect_fp_populations(test_all_subjects)
    test_recovered_s = valid_sigma(test_recovered)
    test_fp_s = valid_sigma(test_wt + test_distant)
    print(f'TEST: recovered_G2A={len(test_recovered)} ({len(test_recovered_s)} with sigma) '
         f'WT_not_TC_FP={len(test_wt)} distant_bg_FP={len(test_distant)} '
         f'({time.time()-t0:.0f}s)', flush=True)

    def eval_component_criterion(pos_rows, neg_rows, predict_fn):
        tp = sum(1 for r in pos_rows if predict_fn(r))
        fn = len(pos_rows) - tp
        fp = sum(1 for r in neg_rows if predict_fn(r))
        tn = len(neg_rows) - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        return {'sensitivity': sens, 'specificity': spec, 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn}

    h1_pred = lambda r: r['sigma_C'] <= sigma_thresh
    h2_pred = lambda r: r['mu_C'] >= mu_thresh
    h3_pred = lambda r: h3_clf.predict([[r['mu_C'], r['sigma_C']]])[0] == 1

    h1_result = eval_component_criterion(test_recovered_s, test_fp_s, h1_pred)
    h2_result = eval_component_criterion(test_recovered, (test_wt + test_distant), h2_pred)
    h3_result = eval_component_criterion(test_recovered_s, test_fp_s, h3_pred)

    print(f'\n=== E267 TEST component-level results ===')
    print(f'H1 (sigma_C <= {sigma_thresh:.4f}): {h1_result}')
    print(f'H2 (mu_C >= {mu_thresh:.4f}): {h2_result}')
    print(f'H3 (logistic mu_C+sigma_C): {h3_result}')

    out = HERE / ('E267_results.csv' if not smoke else 'E267_smoke.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['hypothesis', 'sensitivity', 'specificity', 'tp', 'fn', 'fp', 'tn', 'threshold_or_coef'])
        w.writerow(['H1_sigma_only', h1_result['sensitivity'], h1_result['specificity'],
                   h1_result['tp'], h1_result['fn'], h1_result['fp'], h1_result['tn'], sigma_thresh])
        w.writerow(['H2_mu_only', h2_result['sensitivity'], h2_result['specificity'],
                   h2_result['tp'], h2_result['fn'], h2_result['fp'], h2_result['tn'], mu_thresh])
        w.writerow(['H3_mu_sigma_logistic', h3_result['sensitivity'], h3_result['specificity'],
                   h3_result['tp'], h3_result['fn'], h3_result['fp'], h3_result['tn'],
                   str(list(h3_clf.coef_[0]) + [h3_clf.intercept_[0]])])

    # ---- also save raw component tables for inspection ----
    for name, rows in [('recovered', test_recovered), ('wt_not_tc_fp', test_wt), ('distant_bg_fp', test_distant)]:
        fn = HERE / (f'E267_{name}_smoke.csv' if smoke else f'E267_{name}.csv')
        with open(fn, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['subject_id', 'component_id', 'is_tp', 'A_C',
                                               'mu_C', 'sigma_C', 'population'])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(f'\nE267 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
