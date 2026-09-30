"""E260 -- tumor-context-conditioned readout, per the user's exact spec.
Tests whether "D1 + existing WT/TC prediction" can distinguish
recoverable ET from tumor tissue that should remain non-ET, rather than
simply building a more powerful classifier on the SAME D1-only features.

THREE READOUTS COMPARED (per explicit user spec):
  R1-B: D1 only (32-dim) -- CURRENT BASELINE, refit identically to
        E257-B/E258/E259 for a fair, apples-to-apples comparison.
  R1-C: D1 (32-dim) + production's OWN TC/WT probabilities (2 more dims,
        34 total). CRITICAL: p_TC_prod and p_WT_prod are NOT a new
        computation -- they are seg_head's OWN existing output channels
        1 and 2 (verified: conv_w shape (3,32,1,1,1), TC/WT are
        literally different ROWS of the SAME weight matrix already used
        for w_prod's ET row throughout E251-E259). We are not inventing
        a new information source, exactly as the user's own framing
        requires -- reusing information the production network already
        computes.

R1-D (a spatial-neighborhood feature) is NOT built in this script --
per the user's own conditional phrasing ("+ a small local spatial
feature, IF NEEDED"), this is deferred unless R1-C's own result
motivates it specifically.

CRITICAL EVALUATION REQUIREMENT (per explicit user instruction, to
avoid the "accidentally became a broader tumor detector" failure mode):
tracks WT-not-TC FP and outside-all-tumor FP SEPARATELY, not just
aggregate FP -- reusing E259's own region-overlap computation. If R1-C
improves aggregate FP mainly by reducing WT-not-TC false positives while
outside-tumor FP stays flat (or vice versa), that distinction matters
for interpretation and must not be collapsed into one number.

Per the user's own explicit caution: E259's WT/TC-overlap finding
establishes WHERE the FP burden lies, not that every such voxel is
"legitimately" tumor tissue -- this script does not re-litigate that
interpretation, it tests whether conditioning on WT/TC probability
information measurably changes the READOUT's behavior, which is an
independent, directly-answerable question regardless of how one
interprets E259's own finding.
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

ET, TC, WT = 0, 1, 2


def get_full_w_prod(model):
    """Returns (w_prod_et, b_et, w_prod_tc, b_tc, w_prod_wt, b_wt) -- ALL
    THREE rows of seg_head's own conv weight matrix, verified same
    mechanism as get_w_prod's own ET-only extraction."""
    conv_w = model.seg_head[0].weight.detach().cpu().numpy()
    conv_b = model.seg_head[0].bias.detach().cpu().numpy()
    return (conv_w[ET].reshape(32).astype(np.float64), float(conv_b[ET]),
           conv_w[TC].reshape(32).astype(np.float64), float(conv_b[TC]),
           conv_w[WT].reshape(32).astype(np.float64), float(conv_b[WT]))


def augment_with_tumor_context(feats_32, w_tc, b_tc, w_wt, b_wt):
    """feats_32: (n,32) D1 features. Returns (n,34) = [D1, p_TC, p_WT]."""
    p_tc = expit(feats_32 @ w_tc + b_tc).reshape(-1, 1)
    p_wt = expit(feats_32 @ w_wt + b_wt).reshape(-1, 1)
    return np.concatenate([feats_32, p_tc, p_wt], axis=1)


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    model = load_model(dev)
    w_et, b_et, w_tc, b_tc, w_wt, b_wt = get_full_w_prod(model)
    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    if smoke:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']][:30]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']][:15]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']][:15]
        g2b_lesions_eval = g2b_lesions[:15]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']]
        g2b_lesions_eval = g2b_lesions
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng = np.random.default_rng(999)
    t0 = time.time()

    # ---- build shared training voxel pools (raw D1, 32-dim) ----
    print('Building training voxel pools...', flush=True)
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
    lesion_pool = np.concatenate(lesion_pool)
    neg_pool = np.concatenate(neg_pool)
    print(f'  lesion_pool n={len(lesion_pool)}  neg_pool n={len(neg_pool)} ({time.time()-t0:.0f}s)', flush=True)

    # ---- fit R1-B (D1 only, 32-dim) ----
    X_b = np.concatenate([lesion_pool, neg_pool])
    y_b = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
    r1b = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_b, y_b)
    w_r1b = torch.from_numpy(r1b.coef_[0].astype(np.float64)).float().to(dev)
    b_r1b = float(r1b.intercept_[0])
    print(f'R1-B fit ({time.time()-t0:.0f}s)', flush=True)

    # ---- fit R1-C (D1 + p_TC + p_WT, 34-dim) ----
    lesion_pool_c = augment_with_tumor_context(lesion_pool, w_tc, b_tc, w_wt, b_wt)
    neg_pool_c = augment_with_tumor_context(neg_pool, w_tc, b_tc, w_wt, b_wt)
    X_c = np.concatenate([lesion_pool_c, neg_pool_c])
    y_c = y_b
    r1c = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_c, y_c)
    w_r1c = r1c.coef_[0].astype(np.float64)  # (34,)
    b_r1c = float(r1c.intercept_[0])
    w_r1c_d1 = torch.from_numpy(w_r1c[:32]).float().to(dev)
    w_r1c_tc_coef = float(w_r1c[32])
    w_r1c_wt_coef = float(w_r1c[33])
    print(f'R1-C fit ({time.time()-t0:.0f}s). '
         f'coef on p_TC={w_r1c_tc_coef:+.4f}  coef on p_WT={w_r1c_wt_coef:+.4f}', flush=True)

    w_tc_t = torch.from_numpy(w_tc).float().to(dev)
    w_wt_t = torch.from_numpy(w_wt).float().to(dev)
    w_et_t = torch.from_numpy(w_et).float().to(dev)

    def dense_eval(sid):
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T
        with torch.no_grad():
            p_prod_flat = torch.sigmoid(flat @ w_et_t + b_et)
            p_r1b_flat = torch.sigmoid(flat @ w_r1b + b_r1b)
            p_tc_flat = torch.sigmoid(flat @ w_tc_t + b_tc)
            p_wt_flat = torch.sigmoid(flat @ w_wt_t + b_wt)
            logit_r1c_flat = (flat @ w_r1c_d1) + w_r1c_tc_coef * p_tc_flat + w_r1c_wt_coef * p_wt_flat + b_r1c
            p_r1c_flat = torch.sigmoid(logit_r1c_flat)
            p_prod_map = p_prod_flat.reshape(D, H, W).cpu().numpy()
            p_r1b_map = p_r1b_flat.reshape(D, H, W).cpu().numpy()
            p_r1c_map = p_r1c_flat.reshape(D, H, W).cpu().numpy()
        return p_prod_map, p_r1b_map, p_r1c_map, tgt_c, img_c

    test_subjects = sorted(set(r['subject_id'] for r in det_lesions_test) |
                          set(r['subject_id'] for r in g2a_lesions_test) |
                          set(r['subject_id'] for r in g2b_lesions_eval))

    print(f'\nDense whole-volume evaluation on {len(test_subjects)} held-out subjects...', flush=True)
    struct = ndimage.generate_binary_structure(3, 1)

    out = HERE / ('E260_smoke.csv' if smoke else 'E260_tumor_context.csv')
    fh = open(out, 'w', newline='')
    w_csv = csv.DictWriter(fh, fieldnames=[
        'subject_id',
        'fp_voxels_prod', 'fp_wt_not_tc_prod', 'fp_outside_prod',
        'fp_voxels_r1b', 'fp_wt_not_tc_r1b', 'fp_outside_r1b',
        'fp_voxels_r1c', 'fp_wt_not_tc_r1c', 'fp_outside_r1c'])
    w_csv.writeheader()

    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        p_prod_map, p_r1b_map, p_r1c_map, tgt_c, img_c = dense_eval(sid)
        brain_mask = img_c[0] != 0
        true_et = tgt_c[ET] > 0.5
        true_tc = tgt_c[TC] > 0.5
        true_wt = tgt_c[WT] > 0.5

        def fp_breakdown(p_map):
            fp_mask = (p_map > TAU) & brain_mask & (~true_et)
            n_total = int(fp_mask.sum())
            n_wt_not_tc = int((fp_mask & true_wt & (~true_tc)).sum())
            n_outside = int((fp_mask & (~true_wt)).sum())
            return n_total, n_wt_not_tc, n_outside

        prod_total, prod_wt, prod_out = fp_breakdown(p_prod_map)
        r1b_total, r1b_wt, r1b_out = fp_breakdown(p_r1b_map)
        r1c_total, r1c_wt, r1c_out = fp_breakdown(p_r1c_map)

        w_csv.writerow({'subject_id': sid,
                        'fp_voxels_prod': prod_total, 'fp_wt_not_tc_prod': prod_wt, 'fp_outside_prod': prod_out,
                        'fp_voxels_r1b': r1b_total, 'fp_wt_not_tc_r1b': r1b_wt, 'fp_outside_r1b': r1b_out,
                        'fp_voxels_r1c': r1c_total, 'fp_wt_not_tc_r1c': r1c_wt, 'fp_outside_r1c': r1c_out})
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)
    fh.close()

    # ---- lesion-level recovery ----
    def lesion_recovery(lesion_list):
        recs = {'prod': [], 'r1b': [], 'r1c': []}
        for r in lesion_list:
            sid = r['subject_id']
            res = extract_lesion_shell(model, ds, sid_to_idx, sid, int(r['comp_id']), dev)
            if res is None:
                continue
            lf, sf, sz = res
            p_prod_l = expit(lf @ w_et + b_et)
            p_r1b_l = expit(lf @ r1b.coef_[0] + r1b.intercept_[0])
            lf_c = augment_with_tumor_context(lf, w_tc, b_tc, w_wt, b_wt)
            p_r1c_l = expit(lf_c @ w_r1c + b_r1c)
            recs['prod'].append(int(p_prod_l.max() > TAU))
            recs['r1b'].append(int(p_r1b_l.max() > TAU))
            recs['r1c'].append(int(p_r1c_l.max() > TAU))
        return {k: (np.mean(v) if v else float('nan')) for k, v in recs.items()}

    det_rec = lesion_recovery(det_lesions_test)
    g2a_rec = lesion_recovery(g2a_lesions_test)
    g2b_rec = lesion_recovery(g2b_lesions_eval)
    print(f'\ndetected_test: {det_rec}')
    print(f'G2A_test: {g2a_rec}')
    print(f'G2B: {g2b_rec}')

    print(f'\nE260 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
