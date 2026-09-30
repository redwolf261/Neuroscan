"""E257-B -- retrain R1 with a REPRESENTATIVE negative class, per the
user's exact spec: the second candidate mechanism, testing whether the
problem is that R1 needs SELECTIVE activation (E257-A's gate) or that
R1 was simply trained with the WRONG negative distribution in the first
place (this script). Same frozen backbone, same frozen production
seg_head, same train/val/test split as E257-A (import from e257_common,
identical subject assignment -- directly comparable).

R1-B TRAINING SET: lesion (positive) vs {local_shell, distant_background}
(BOTH negative), pooled -- unlike every prior R1 (E251-E257A), which
used ONLY local shell as the negative class. This directly tests E256's
own diagnosis: R1's catastrophic FP was caused by never seeing distant
tissue as a negative example during training.

NO GATE HERE -- R1-B is used DIRECTLY as the final readout (p_final =
p_R1B), exactly like E251-E254's original R1, just with a better-
specified negative class. This isolates: is a smarter TRAINING SET
enough on its own, without any inference-time selectivity mechanism.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e257_common import (load_model, get_w_prod, load_populations, build_splits,  # noqa: E402
                         extract_lesion_shell, extract_distant_background,
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    model = load_model(dev)
    w_prod, b_prod = get_w_prod(model)
    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    if smoke:
        det_lesions = [r for r in det_lesions if r['subject_id'] in splits['det_train']][:30] + \
                     [r for r in det_lesions if r['subject_id'] in splits['det_test']][:15]
        g2a_lesions = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_train']][:20] + \
                     [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']][:15]
        g2b_lesions = g2b_lesions[:15]

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng = np.random.default_rng(999)
    t0 = time.time()

    # ---- fit R1-B on TRAIN-split detected lesions: lesion vs {shell + distant} ----
    print('Fitting R1-B (lesion vs shell+distant) on train-split detected lesions...', flush=True)
    X, y = [], []
    n_fit = 0
    for r in det_lesions:
        if r['subject_id'] not in splits['det_train']:
            continue
        res = extract_lesion_shell(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
        if res is None:
            continue
        lf, sf, sz = res
        if len(lf) > MAX_VOX_PER_LESION:
            lf = lf[rng.choice(len(lf), MAX_VOX_PER_LESION, replace=False)]
        if len(sf) > MAX_VOX_PER_LESION:
            sf = sf[rng.choice(len(sf), MAX_VOX_PER_LESION, replace=False)]
        X.append(lf); y.append(np.ones(len(lf)))
        X.append(sf); y.append(np.zeros(len(sf)))
        n_fit += 1
    print(f'  {n_fit} detected train lesions (lesion+shell)', flush=True)

    train_subjects_union = sorted(splits['det_train'] | splits['g2a_train'])
    if smoke:
        train_subjects_union = train_subjects_union[:20]
    n_bg = 0
    for sid in train_subjects_union:
        feats = extract_distant_background(model, ds, sid_to_idx, sid, dev, rng)
        if feats is None:
            continue
        X.append(feats); y.append(np.zeros(len(feats)))
        n_bg += 1
    print(f'  {n_bg} distant-background subjects added as negatives', flush=True)

    X = np.concatenate(X); y = np.concatenate(y)
    print(f'  training set: n={len(y)}  positive_rate={y.mean():.4f}', flush=True)
    r1b = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced')
    r1b.fit(X, y)
    w_r1b = r1b.coef_[0].astype(np.float64)
    b_r1b = float(r1b.intercept_[0])
    print(f'R1-B fit ({time.time()-t0:.0f}s)', flush=True)

    # ---- dense whole-volume evaluation on TEST-split subjects ----
    print('\nDense whole-volume evaluation on held-out test subjects...', flush=True)
    w_prod_t = torch.from_numpy(w_prod).float().to(dev)
    w_r1b_t = torch.from_numpy(w_r1b).float().to(dev)

    def dense_eval(sid):
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T
        with torch.no_grad():
            p_prod_flat = torch.sigmoid(flat @ w_prod_t + b_prod)
            p_r1b_flat = torch.sigmoid(flat @ w_r1b_t + b_r1b)
            p_prod_map = p_prod_flat.reshape(D, H, W).cpu().numpy()
            p_r1b_map = p_r1b_flat.reshape(D, H, W).cpu().numpy()
        return p_prod_map, p_r1b_map, tgt_c, img_c

    test_subjects = sorted(splits['det_test']) + \
                    sorted(set(r['subject_id'] for r in g2a_lesions) & splits['g2a_test']) + \
                    sorted(set(r['subject_id'] for r in g2b_lesions))
    test_subjects = sorted(set(test_subjects))
    if smoke:
        test_subjects = test_subjects[:15]

    out = HERE / ('E257b_smoke.csv' if smoke else 'E257b_r1b_eval.csv')
    fh = open(out, 'w', newline='')
    w_csv = csv.DictWriter(fh, fieldnames=['subject_id', 'fp_voxels_prod', 'fp_components_prod',
                                           'fp_voxels_r1b', 'fp_components_r1b'])
    w_csv.writeheader()
    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        p_prod_map, p_r1b_map, tgt_c, img_c = dense_eval(sid)
        brain_mask = img_c[0] != 0
        true_et = tgt_c[0] > 0.5
        fp_prod_mask = (p_prod_map > TAU) & brain_mask & (~true_et)
        fp_r1b_mask = (p_r1b_map > TAU) & brain_mask & (~true_et)
        _, fp_prod_n = ndimage.label(fp_prod_mask)
        _, fp_r1b_n = ndimage.label(fp_r1b_mask)
        w_csv.writerow({'subject_id': sid, 'fp_voxels_prod': int(fp_prod_mask.sum()),
                        'fp_components_prod': fp_prod_n, 'fp_voxels_r1b': int(fp_r1b_mask.sum()),
                        'fp_components_r1b': fp_r1b_n})
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)
    fh.close()

    def lesion_recovery(lesion_list, subj_filter):
        recs_prod, recs_r1b = [], []
        for r in lesion_list:
            sid = r['subject_id']
            if subj_filter is not None and sid not in subj_filter:
                continue
            res = extract_lesion_shell(model, ds, sid_to_idx, sid, int(r['comp_id']), dev)
            if res is None:
                continue
            lf, sf, sz = res
            p_prod_l = expit(lf @ w_prod + b_prod)
            p_r1b_l = expit(lf @ w_r1b + b_r1b)
            recs_prod.append(int(p_prod_l.max() > TAU))
            recs_r1b.append(int(p_r1b_l.max() > TAU))
        return (np.mean(recs_prod) if recs_prod else float('nan'),
               np.mean(recs_r1b) if recs_r1b else float('nan'))

    det_test_lesions = [r for r in det_lesions if r['subject_id'] in splits['det_test']]
    g2a_test_lesions = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']]

    det_prod, det_r1b = lesion_recovery(det_test_lesions, None)
    g2a_prod, g2a_r1b = lesion_recovery(g2a_test_lesions, None)
    g2b_prod, g2b_r1b = lesion_recovery(g2b_lesions, None)
    print(f'\ndetected_test: production={det_prod:.4f}  R1-B={det_r1b:.4f}')
    print(f'G2A_test: production={g2a_prod:.4f}  R1-B={g2a_r1b:.4f}')
    print(f'G2B: production={g2b_prod:.4f}  R1-B={g2b_r1b:.4f}')

    print(f'\nE257-B complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
