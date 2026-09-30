"""E257-A -- learned feature-space gate, per the user's exact spec.
Frozen backbone, frozen R1 (fit on TRAIN-split detected lesions, same
convention as E251-E256), frozen production seg_head. Trains a SMALL
gate g(z) in [0,1] on D1 features, target label:
  y_gate = 1 if R1 gives the correct lesion decision AND production
           does not (i.e. p_R1>tau AND p_prod<=tau, matching ground
           truth at THAT voxel)
  y_gate = 0 otherwise
This directly targets "is switching from production to R1 beneficial
HERE", not "reproduce R1's own output" (a different, weaker target).

TRAINING VOXELS (per explicit user spec): detected-train lesion+shell,
G2A-train lesion+shell, distant-background (sampled from TRAIN-split
subjects of EITHER population, since distant background isn't lesion-
specific). NEVER uses G2A-val or G2A-test voxels for gate fitting.

Final blend: p_final = (1-g(z))*p_prod + g(z)*p_R1.
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
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION, N_DISTANT_PER_SUBJECT)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch, get_masks  # noqa: E402


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

    # ---- STEP 1: fit R1 on TRAIN-split detected lesions (same as E251-E256) ----
    print('Fitting R1 on train-split detected lesions...', flush=True)
    r1_X, r1_y = [], []
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
        r1_X.append(lf); r1_y.append(np.ones(len(lf)))
        r1_X.append(sf); r1_y.append(np.zeros(len(sf)))
        n_fit += 1
    X_r1 = np.concatenate(r1_X); y_r1 = np.concatenate(r1_y)
    r1_model = LogisticRegression(max_iter=500, C=1.0).fit(X_r1, y_r1)
    w_r1 = r1_model.coef_[0].astype(np.float64)
    b_r1 = float(r1_model.intercept_[0])
    print(f'R1 fit on {n_fit} detected lesions ({time.time()-t0:.0f}s)', flush=True)

    # ---- STEP 2: build gate training set with y_gate labels ----
    print('\nBuilding gate training set...', flush=True)

    def gate_examples_from_lesion(sid, cid, is_train_subj):
        res = extract_lesion_shell(model, ds, sid_to_idx, sid, cid, dev)
        if res is None:
            return None
        lf, sf, sz = res
        # CRITICAL FIX (found and corrected before trusting any full-run
        # result): this function was NOT capping lf/sf at
        # MAX_VOX_PER_LESION, unlike the R1-fitting loop above it --
        # large lesions (especially big G2-A ones) contributed thousands
        # of UNCAPPED voxels each, producing a gate training set ~19x
        # larger than intended (635,614 actual vs ~33,555 expected) and
        # almost certainly a distorted training distribution dominated
        # by a handful of large lesions rather than a balanced sample.
        # This directly correlates with the full-run gate producing a
        # catastrophic whole-volume FP result matching E255's original
        # failure (median 640->207,854 FP voxels, ratio never <=2x for
        # ANY of 158 test subjects) despite looking fine on a 15-subject
        # smoke sample -- caught via this size-accounting check, not
        # assumed to be a real negative result, before writing it up.
        if len(lf) > MAX_VOX_PER_LESION:
            lf = lf[rng.choice(len(lf), MAX_VOX_PER_LESION, replace=False)]
        if len(sf) > MAX_VOX_PER_LESION:
            sf = sf[rng.choice(len(sf), MAX_VOX_PER_LESION, replace=False)]
        p_prod_l = expit(lf @ w_prod + b_prod)
        p_r1_l = expit(lf @ w_r1 + b_r1)
        # ground truth: lesion voxels are ALL positive (gt=1)
        y_gate_lesion = ((p_r1_l > TAU) & (p_prod_l <= TAU)).astype(float)
        p_prod_s = expit(sf @ w_prod + b_prod)
        p_r1_s = expit(sf @ w_r1 + b_r1)
        # shell voxels are ALL negative (gt=0): R1 "correct" means p_r1<=tau
        y_gate_shell = ((p_r1_s <= TAU) & (p_prod_s > TAU)).astype(float)
        return lf, y_gate_lesion, sf, y_gate_shell

    gate_X, gate_y = [], []
    n_g2a_train = 0
    for r in g2a_lesions:
        if r['subject_id'] not in splits['g2a_train']:
            continue
        res = gate_examples_from_lesion(r['subject_id'], int(r['comp_id']), True)
        if res is None:
            continue
        lf, yl, sf, ys = res
        gate_X.append(lf); gate_y.append(yl)
        gate_X.append(sf); gate_y.append(ys)
        n_g2a_train += 1
    print(f'  G2A train lesions used: {n_g2a_train}', flush=True)

    n_det_train = 0
    for r in det_lesions:
        if r['subject_id'] not in splits['det_train']:
            continue
        res = gate_examples_from_lesion(r['subject_id'], int(r['comp_id']), True)
        if res is None:
            continue
        lf, yl, sf, ys = res
        gate_X.append(lf); gate_y.append(yl)
        gate_X.append(sf); gate_y.append(ys)
        n_det_train += 1
        if smoke and n_det_train >= 30:
            break
    print(f'  detected train lesions used: {n_det_train}', flush=True)

    # distant background: y_gate=0 by construction (production is already
    # near-perfect there; R1 should NEVER be preferred)
    train_subjects_union = sorted(splits['det_train'] | splits['g2a_train'])
    if smoke:
        train_subjects_union = train_subjects_union[:20]
    n_bg = 0
    for sid in train_subjects_union:
        feats = extract_distant_background(model, ds, sid_to_idx, sid, dev, rng)
        if feats is None:
            continue
        gate_X.append(feats); gate_y.append(np.zeros(len(feats)))
        n_bg += 1
    print(f'  distant-background subjects used: {n_bg}', flush=True)

    X_gate = np.concatenate(gate_X); y_gate = np.concatenate(gate_y)
    print(f'\nGate training set: n={len(y_gate)}  positive_rate={y_gate.mean():.4f}', flush=True)

    gate_model = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced')
    gate_model.fit(X_gate, y_gate)
    print(f'Gate fit ({time.time()-t0:.0f}s)', flush=True)

    # ---- STEP 3: final dense evaluation on TEST-split subjects ----
    print('\nDense whole-volume evaluation on held-out test subjects...', flush=True)
    w_prod_t = torch.from_numpy(w_prod).float().to(dev)
    w_r1_t = torch.from_numpy(w_r1).float().to(dev)
    gate_w = torch.from_numpy(gate_model.coef_[0].astype(np.float64)).float().to(dev)
    gate_b = float(gate_model.intercept_[0])

    def dense_eval(sid):
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T
        with torch.no_grad():
            p_prod_flat = torch.sigmoid(flat @ w_prod_t + b_prod)
            p_r1_flat = torch.sigmoid(flat @ w_r1_t + b_r1)
            g_flat = torch.sigmoid(flat @ gate_w + gate_b)
            p_final_flat = (1 - g_flat) * p_prod_flat + g_flat * p_r1_flat
            p_prod_map = p_prod_flat.reshape(D, H, W).cpu().numpy()
            p_final_map = p_final_flat.reshape(D, H, W).cpu().numpy()
        return p_prod_map, p_final_map, tgt_c, img_c

    test_subjects = sorted(splits['det_test']) + \
                    sorted(set(r['subject_id'] for r in g2a_lesions) & splits['g2a_test']) + \
                    sorted(set(r['subject_id'] for r in g2b_lesions))
    test_subjects = sorted(set(test_subjects))
    if smoke:
        test_subjects = test_subjects[:15]

    out = HERE / ('E257a_smoke.csv' if smoke else 'E257a_gate_eval.csv')
    fh = open(out, 'w', newline='')
    w_csv = csv.DictWriter(fh, fieldnames=['subject_id', 'fp_voxels_prod', 'fp_components_prod',
                                           'fp_voxels_final', 'fp_components_final'])
    w_csv.writeheader()
    struct = ndimage.generate_binary_structure(3, 1)
    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        p_prod_map, p_final_map, tgt_c, img_c = dense_eval(sid)
        brain_mask = img_c[0] != 0
        true_et = tgt_c[0] > 0.5
        fp_prod_mask = (p_prod_map > TAU) & brain_mask & (~true_et)
        fp_final_mask = (p_final_map > TAU) & brain_mask & (~true_et)
        _, fp_prod_n = ndimage.label(fp_prod_mask)
        _, fp_final_n = ndimage.label(fp_final_mask)
        w_csv.writerow({'subject_id': sid, 'fp_voxels_prod': int(fp_prod_mask.sum()),
                        'fp_components_prod': fp_prod_n, 'fp_voxels_final': int(fp_final_mask.sum()),
                        'fp_components_final': fp_final_n})
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)
    fh.close()

    # ---- lesion-level recovery, held-out only ----
    def lesion_recovery(lesion_list, subj_filter):
        recs_prod, recs_final = [], []
        for r in lesion_list:
            sid = r['subject_id']
            if subj_filter is not None and sid not in subj_filter:
                continue
            res = extract_lesion_shell(model, ds, sid_to_idx, sid, int(r['comp_id']), dev)
            if res is None:
                continue
            lf, sf, sz = res
            p_prod_l = expit(lf @ w_prod + b_prod)
            p_r1_l = expit(lf @ w_r1 + b_r1)
            g_l = expit(lf @ gate_model.coef_[0] + gate_model.intercept_[0])
            p_final_l = (1 - g_l) * p_prod_l + g_l * p_r1_l
            recs_prod.append(int(p_prod_l.max() > TAU))
            recs_final.append(int(p_final_l.max() > TAU))
        return (np.mean(recs_prod) if recs_prod else float('nan'),
               np.mean(recs_final) if recs_final else float('nan'))

    det_test_lesions = [r for r in det_lesions if r['subject_id'] in splits['det_test']]
    g2a_test_lesions = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']]

    det_prod, det_final = lesion_recovery(det_test_lesions, None)
    g2a_prod, g2a_final = lesion_recovery(g2a_test_lesions, None)
    g2b_prod, g2b_final = lesion_recovery(g2b_lesions, None)
    print(f'\ndetected_test: production={det_prod:.4f}  gated={det_final:.4f}')
    print(f'G2A_test: production={g2a_prod:.4f}  gated={g2a_final:.4f}')
    print(f'G2B: production={g2b_prod:.4f}  gated={g2b_final:.4f}')

    print(f'\nE257-A complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
