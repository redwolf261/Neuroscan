"""E265 -- three mathematically distinct spatial evidence operators,
compared head-to-head, per the user's exact spec following the E264
prior-art audit finding: Delta_C alone is NOT novel (established local-
contrast / connected-component post-processing prior art exists), so
before any novelty claim we test which operator actually carries the
separation, with NO operator combination and NO premature "this is the
novel one" framing -- that determination happens only after this
experiment, followed by its own prior-art audit on the winner.

Stage A (R1-B) kept FROZEN, identical to E257-B/E258-E264.

THREE OPERATORS (all computed from the SAME component C and the SAME
3-voxel-dilation neighborhood shell N(C), so this is a fair, apples-to-
apples comparison, not three different neighborhood definitions):

  Delta_C = mu_C - mu_N                                    (E264's original, raw contrast)
  R_C     = (mu_C - mu_N) / (mu_C + mu_N + eps)             (normalized contrast, bounded in (-1,1))
  Psi_C   = log(1+|C|) * log((mu_C+eps)/(mu_N+eps))         (scale-compensated log-ratio)

where mu_C = mean R1-B probability inside C, mu_N = mean R1-B
probability in the shell N(C) (dilation=3, excluding C itself and any
true-ET voxels -- identical neighborhood definition to E264's Delta_C,
so mu_N is recorded directly this time rather than back-derived).

VAL/TEST DISCIPLINE: identical to E264 -- each operator's threshold is
selected on VAL only, using the LESION-level zero-cost criterion E264
established (a threshold survives if no G2A_val lesion loses ALL its
overlapping components), then applied ONCE to TEST.
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
NEIGHBORHOOD_DILATION = 3
EPS = 1e-6
STRUCT = ndimage.generate_binary_structure(3, 1)


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
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        g2a_lesions_val = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_val']]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng = np.random.default_rng(999)
    t0 = time.time()

    # ---- refit R1-B, IDENTICAL to E257-B/E258-E264 (Stage A frozen) ----
    print('Refitting R1-B (frozen Stage A, identical to E257-B)...', flush=True)
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
    print(f'R1-B refit ({time.time()-t0:.0f}s)', flush=True)

    dense_cache = {}

    def dense_probs(sid):
        if sid in dense_cache:
            return dense_cache[sid]
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T
        with torch.no_grad():
            p_r1b_flat = torch.sigmoid(flat @ w_r1b + b_r1b)
        p_r1b_map = p_r1b_flat.reshape(D, H, W).cpu().numpy()
        result = (p_r1b_map, tgt_c, img_c)
        dense_cache[sid] = result
        return result

    def component_table(sid):
        """Labels ALL R1-B positive components (p>TAU, brain) in this
        subject's volume and computes mu_C, mu_N, |C| plus the three
        ladder operators computed from the SAME neighborhood shell."""
        p_map, tgt_c, img_c = dense_probs(sid)
        brain_mask = img_c[0] != 0
        true_et = tgt_c[ET] > 0.5
        pos_mask = (p_map > TAU) & brain_mask
        lbl, n = ndimage.label(pos_mask, structure=STRUCT)
        rows = []
        for cid in range(1, n + 1):
            comp_mask = lbl == cid
            A_C = int(comp_mask.sum())
            if A_C < 1:
                continue
            is_tp = bool((comp_mask & true_et).any())
            mu_C = float(p_map[comp_mask].mean())

            dilated = ndimage.binary_dilation(comp_mask, structure=STRUCT,
                                              iterations=NEIGHBORHOOD_DILATION)
            shell = dilated & (~comp_mask) & (~true_et) & brain_mask
            n_shell = int(shell.sum())
            mu_N = float(p_map[shell].mean()) if n_shell > 0 else 0.0

            Delta_C = mu_C - mu_N
            R_C = (mu_C - mu_N) / (mu_C + mu_N + EPS)
            Psi_C = np.log1p(A_C) * np.log((mu_C + EPS) / (mu_N + EPS))

            rows.append({
                'subject_id': sid, 'component_id': cid, 'is_tp': int(is_tp),
                'A_C': A_C, 'mu_C': mu_C, 'mu_N': mu_N, 'n_shell': n_shell,
                'Delta_C': float(Delta_C), 'R_C': float(R_C), 'Psi_C': float(Psi_C),
            })
        return rows, lbl

    comp_cache = {}

    def get_components(sid):
        if sid not in comp_cache:
            comp_cache[sid] = component_table(sid)
        return comp_cache[sid]

    def collect_components(subjects):
        all_rows = []
        for i, sid in enumerate(subjects):
            if sid not in sid_to_idx:
                continue
            rows, _ = get_components(sid)
            all_rows.extend(rows)
            if (i + 1) % 20 == 0 or smoke:
                print(f'  components: {i+1}/{len(subjects)} subjects ({time.time()-t0:.0f}s)', flush=True)
        return all_rows

    def lesion_recovery_baseline(lesion_list):
        recs = []
        for r in lesion_list:
            sid, cid = r['subject_id'], int(r['comp_id'])
            if sid not in sid_to_idx:
                continue
            p_map, tgt_c, img_c = dense_probs(sid)
            et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
            if cid < 1 or cid > et_n:
                continue
            lesion_mask = et_lbl == cid
            if int(lesion_mask.sum()) < MIN_VOX:
                continue
            recs.append(int(((p_map > TAU) & lesion_mask).any()))
        return recs

    def lesion_recovery_filtered(lesion_list, stat_key, threshold):
        recs = []
        for r in lesion_list:
            sid, cid = r['subject_id'], int(r['comp_id'])
            if sid not in sid_to_idx:
                continue
            p_map, tgt_c, img_c = dense_probs(sid)
            et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
            if cid < 1 or cid > et_n:
                continue
            lesion_mask = et_lbl == cid
            if int(lesion_mask.sum()) < MIN_VOX:
                continue
            rows, lbl = get_components(sid)
            overlap_ids = set(np.unique(lbl[lesion_mask])) - {0}
            survived = any(row['component_id'] in overlap_ids and row[stat_key] >= threshold
                          for row in rows)
            recs.append(int(survived))
        return recs

    def fp_voxels_baseline(subjects):
        total = 0
        for sid in subjects:
            if sid not in sid_to_idx:
                continue
            p_map, tgt_c, img_c = dense_probs(sid)
            brain_mask = img_c[0] != 0
            true_et = tgt_c[ET] > 0.5
            total += int(((p_map > TAU) & brain_mask & (~true_et)).sum())
        return total

    def fp_voxels_filtered(subjects, stat_key, threshold):
        total = 0
        for sid in subjects:
            if sid not in sid_to_idx:
                continue
            rows, _ = get_components(sid)
            for row in rows:
                if row['is_tp'] == 1:
                    continue
                if row[stat_key] >= threshold:
                    total += row['A_C']
        return total

    # ---- collect component table on VAL subjects ----
    val_subjects = sorted(set(r['subject_id'] for r in g2a_lesions_val))
    print(f'\nCollecting component table on {len(val_subjects)} VAL subjects...', flush=True)
    val_rows = collect_components(val_subjects)
    n_tp_val = sum(r['is_tp'] for r in val_rows)
    print(f'val component table: {len(val_rows)} components '
         f'({n_tp_val} TP, {len(val_rows)-n_tp_val} FP) ({time.time()-t0:.0f}s)', flush=True)

    val_csv = HERE / ('E265_val_components_smoke.csv' if smoke else 'E265_val_components.csv')
    with open(val_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['subject_id', 'component_id', 'is_tp',
                                           'A_C', 'mu_C', 'mu_N', 'n_shell',
                                           'Delta_C', 'R_C', 'Psi_C'])
        w.writeheader()
        for row in val_rows:
            w.writerow(row)

    # ---- LESION-level zero-cost threshold selection (E264's criterion) ----
    def per_lesion_max_stat(lesion_list, stat_key):
        maxima = []
        for r in lesion_list:
            sid, cid = r['subject_id'], int(r['comp_id'])
            if sid not in sid_to_idx:
                continue
            p_map, tgt_c, img_c = dense_probs(sid)
            et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
            if cid < 1 or cid > et_n:
                continue
            lesion_mask = et_lbl == cid
            if int(lesion_mask.sum()) < MIN_VOX:
                continue
            rows, lbl = get_components(sid)
            overlap_ids = set(np.unique(lbl[lesion_mask])) - {0}
            vals = [row[stat_key] for row in rows if row['component_id'] in overlap_ids]
            if vals:
                maxima.append(max(vals))
        return maxima

    def select_lesion_zero_cost_threshold(lesion_list, stat_key):
        maxima = per_lesion_max_stat(lesion_list, stat_key)
        if not maxima:
            return -np.inf
        return min(maxima)

    def val_fp_removed_frac(rows, stat_key, threshold):
        fp_rows = [r for r in rows if r['is_tp'] == 0]
        total_fp_vox = sum(r['A_C'] for r in fp_rows)
        if total_fp_vox == 0:
            return 0.0
        removed_vox = sum(r['A_C'] for r in fp_rows if r[stat_key] < threshold)
        return removed_vox / total_fp_vox

    thresholds = {}
    val_removed = {}
    for stat_key in ['Delta_C', 'R_C', 'Psi_C']:
        thresholds[stat_key] = select_lesion_zero_cost_threshold(g2a_lesions_val, stat_key)
        val_removed[stat_key] = val_fp_removed_frac(val_rows, stat_key, thresholds[stat_key])
    print(f'\nVAL-selected lesion-level zero-cost thresholds: {thresholds}', flush=True)
    print(f'VAL FP-voxel-burden removed at zero lesion-recovery cost: {val_removed}', flush=True)
    best_stat = max(val_removed, key=val_removed.get)
    print(f'E265 winner on VAL: {best_stat}', flush=True)

    # ---- apply EACH fixed (val-selected) threshold ONCE on TEST ----
    test_subjects_fp = sorted(set(r['subject_id'] for r in det_lesions if r['subject_id'] in splits['det_test']) |
                              set(r['subject_id'] for r in g2a_lesions_test) |
                              set(r['subject_id'] for r in g2b_lesions))
    if smoke:
        test_subjects_fp = test_subjects_fp[:15]

    print(f'\nApplying val-selected thresholds ONCE on {len(test_subjects_fp)} TEST subjects...', flush=True)

    results = {}
    recov_a = lesion_recovery_baseline(g2a_lesions_test)
    fp_a = fp_voxels_baseline(test_subjects_fp)
    results['no_filter'] = {'recovery': float(np.mean(recov_a)) if recov_a else float('nan'),
                            'fp_voxels': fp_a, 'threshold': None}

    labels = {'Delta_C': 'Delta_C_raw_contrast', 'R_C': 'R_C_normalized', 'Psi_C': 'Psi_C_scale_compensated'}
    for stat_key, label in labels.items():
        thresh = thresholds[stat_key]
        recov = lesion_recovery_filtered(g2a_lesions_test, stat_key, thresh)
        fp = fp_voxels_filtered(test_subjects_fp, stat_key, thresh)
        results[label] = {'recovery': float(np.mean(recov)) if recov else float('nan'),
                          'fp_voxels': fp, 'threshold': thresh}
        print(f'  {label} ({time.time()-t0:.0f}s)', flush=True)

    results['winner'] = {'stat': best_stat, **results[labels[best_stat]]}

    print(f'\n=== E265 TEST results ===')
    for k, v in results.items():
        print(f'{k}: {v}')

    out = HERE / ('E265_smoke.csv' if smoke else 'E265_results.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['variant', 'recovery', 'fp_voxels', 'threshold'])
        for k, v in results.items():
            w.writerow([k, v.get('recovery'), v.get('fp_voxels'), v.get('threshold')])

    print(f'\nE265 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
