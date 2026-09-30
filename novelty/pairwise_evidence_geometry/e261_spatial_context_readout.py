"""E261 -- spatial-context readout, per the user's exact spec. Tests
whether the missing information for the FP tail is SPATIAL (local
neighborhood appearance) rather than purely per-voxel, as a distinct
axis from E260's (negative) per-voxel tumor-context-conditioning result.

DESCRIPTOR (per explicit user decision this turn): for each voxel,
concatenate its own 32-channel D1 vector with the MEAN D1 vector over a
3-voxel-radius neighborhood (7x7x7 cube, matching E233/E234's own
established local-shell scale) -- 64-dim total. Computed via exact 3D
average pooling over the FULL D1 volume (cheap, GPU, exact box-filter
mean), not a sampled approximation.

TWO READOUTS COMPARED (same train/val/test split, same voxel
populations as E257-B/E258/E259/E260, for direct comparability):
  R1-B: D1 only (32-dim) -- baseline, refit identically.
  R1-E: D1 + local-neighborhood-mean-D1 (64-dim).

SPECIFIC, FALSIFIABLE PREDICTION (per explicit user framing, NOT just
"higher recovery"): R1-E should SPECIFICALLY reduce WT-not-TC FP burden
while PRESERVING the 58.3% G2-A recovery. A result that raises recovery
without reducing WT-not-TC FP, or reduces FP by also gutting recovery,
does NOT confirm the spatial-context hypothesis -- report exactly what
happened, do not force the predicted pattern onto an ambiguous result.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e257_common import (load_model, get_w_prod, load_populations, build_splits,
                         extract_lesion_shell, extract_distant_background,
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION, NEAR_DILATION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch

ET, TC, WT = 0, 1, 2
NEIGHBORHOOD_RADIUS = 3  # matches E233/E234's own local-shell scale


def get_d1_with_neighborhood_mean(model, img_t):
    """Returns (d1, d1_nbhd_mean): both (32,D,H,W) torch tensors on GPU.
    d1_nbhd_mean = exact 3D box-filter average of d1 over a
    (2*radius+1)^3 cube, computed via avg_pool3d with matching kernel/
    stride=1/padding -- an EXACT local mean at every voxel, not a
    sampled approximation."""
    stages = forward_to_dec1_internal(model, img_t)
    d1 = stages['relu2_out']  # (1,32,D,H,W)
    k = 2 * NEIGHBORHOOD_RADIUS + 1
    d1_nbhd_mean = F.avg_pool3d(d1, kernel_size=k, stride=1, padding=NEIGHBORHOOD_RADIUS,
                                count_include_pad=False)
    return d1[0], d1_nbhd_mean[0]


def extract_lesion_shell_with_context(model, ds, sid_to_idx, sid, cid, dev):
    """Same masks as extract_lesion_shell, but ALSO returns the
    neighborhood-mean-augmented (n,64) feature arrays."""
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    from e245_dec1_internal_decomposition import get_masks
    brain_mask = img_c[0] != 0
    cm, shell = get_masks(tgt_c, cid, brain_mask)
    if cm is None:
        return None
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    d1, d1_nbhd = get_d1_with_neighborhood_mean(model, img_t)
    d1_np = d1.cpu().numpy()
    nbhd_np = d1_nbhd.cpu().numpy()
    lesion_feats = np.concatenate([d1_np[:, cm].T, nbhd_np[:, cm].T], axis=1)  # (n,64)
    shell_feats = np.concatenate([d1_np[:, shell].T, nbhd_np[:, shell].T], axis=1)
    return lesion_feats, shell_feats, cm.sum()


def extract_distant_background_with_context(model, ds, sid_to_idx, sid, dev, rng, n_sample=35):
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    brain_mask = img_c[0] != 0
    true_et = tgt_c[ET] > 0.5
    struct = ndimage.generate_binary_structure(3, 1)
    if true_et.any():
        dilated = ndimage.binary_dilation(true_et, structure=struct, iterations=NEAR_DILATION)
        local_shell = dilated & (~true_et) & brain_mask
    else:
        local_shell = np.zeros_like(true_et)
    distant_bg = brain_mask & (~true_et) & (~local_shell)
    idx = np.where(distant_bg)
    if len(idx[0]) == 0:
        return None
    n_avail = len(idx[0])
    sel = rng.choice(n_avail, size=min(n_sample, n_avail), replace=False)
    sel_coords = (idx[0][sel], idx[1][sel], idx[2][sel])
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    d1, d1_nbhd = get_d1_with_neighborhood_mean(model, img_t)
    d1_np = d1.cpu().numpy(); nbhd_np = d1_nbhd.cpu().numpy()
    feats = np.concatenate([d1_np[:, sel_coords[0], sel_coords[1], sel_coords[2]].T,
                            nbhd_np[:, sel_coords[0], sel_coords[1], sel_coords[2]].T], axis=1)
    return feats


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

    print('Building training voxel pools (64-dim: D1 + neighborhood-mean-D1)...', flush=True)
    lesion_pool, neg_pool = [], []
    for r in det_lesions_fit:
        res = extract_lesion_shell_with_context(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
        if res is None:
            continue
        lf, sf, sz = res
        if len(lf) > MAX_VOX_PER_LESION:
            lf = lf[rng.choice(len(lf), MAX_VOX_PER_LESION, replace=False)]
        if len(sf) > MAX_VOX_PER_LESION:
            sf = sf[rng.choice(len(sf), MAX_VOX_PER_LESION, replace=False)]
        lesion_pool.append(lf); neg_pool.append(sf)
    for sid in bg_subjects:
        feats = extract_distant_background_with_context(model, ds, sid_to_idx, sid, dev, rng)
        if feats is None:
            continue
        neg_pool.append(feats)
    lesion_pool = np.concatenate(lesion_pool)  # (n,64)
    neg_pool = np.concatenate(neg_pool)
    print(f'  lesion_pool n={len(lesion_pool)}  neg_pool n={len(neg_pool)} ({time.time()-t0:.0f}s)', flush=True)

    X_full = np.concatenate([lesion_pool, neg_pool])
    y_full = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])

    # R1-B: D1 only (first 32 cols)
    r1b = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_full[:, :32], y_full)
    w_r1b = torch.from_numpy(r1b.coef_[0].astype(np.float64)).float().to(dev)
    b_r1b = float(r1b.intercept_[0])
    print(f'R1-B fit ({time.time()-t0:.0f}s)', flush=True)

    # R1-E: D1 + neighborhood-mean-D1 (all 64 cols)
    r1e = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_full, y_full)
    w_r1e = torch.from_numpy(r1e.coef_[0].astype(np.float64)).float().to(dev)
    b_r1e = float(r1e.intercept_[0])
    w_r1e_own = r1e.coef_[0][:32]; w_r1e_nbhd = r1e.coef_[0][32:]
    print(f'R1-E fit ({time.time()-t0:.0f}s). '
         f'||coef on own D1||={np.linalg.norm(w_r1e_own):.4f}  ||coef on neighborhood mean||={np.linalg.norm(w_r1e_nbhd):.4f}', flush=True)

    w_et_t = torch.from_numpy(w_et).float().to(dev)
    k = 2 * NEIGHBORHOOD_RADIUS + 1

    def dense_eval(sid):
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        d1, d1_nbhd = get_d1_with_neighborhood_mean(model, img_t)
        C, D, H, W = d1.shape
        flat_own = d1.reshape(C, -1).T
        flat_nbhd = d1_nbhd.reshape(C, -1).T
        flat_64 = torch.cat([flat_own, flat_nbhd], dim=1)
        with torch.no_grad():
            p_prod_flat = torch.sigmoid(flat_own @ w_et_t + b_et)
            p_r1b_flat = torch.sigmoid(flat_own @ w_r1b + b_r1b)
            p_r1e_flat = torch.sigmoid(flat_64 @ w_r1e + b_r1e)
            p_prod_map = p_prod_flat.reshape(D, H, W).cpu().numpy()
            p_r1b_map = p_r1b_flat.reshape(D, H, W).cpu().numpy()
            p_r1e_map = p_r1e_flat.reshape(D, H, W).cpu().numpy()
        return p_prod_map, p_r1b_map, p_r1e_map, tgt_c, img_c

    test_subjects = sorted(set(r['subject_id'] for r in det_lesions_test) |
                          set(r['subject_id'] for r in g2a_lesions_test) |
                          set(r['subject_id'] for r in g2b_lesions_eval))

    print(f'\nDense whole-volume evaluation on {len(test_subjects)} held-out subjects...', flush=True)
    out = HERE / ('E261_smoke.csv' if smoke else 'E261_spatial_context.csv')
    fh = open(out, 'w', newline='')
    w_csv = csv.DictWriter(fh, fieldnames=[
        'subject_id',
        'fp_voxels_prod', 'fp_wt_not_tc_prod', 'fp_outside_prod',
        'fp_voxels_r1b', 'fp_wt_not_tc_r1b', 'fp_outside_r1b',
        'fp_voxels_r1e', 'fp_wt_not_tc_r1e', 'fp_outside_r1e'])
    w_csv.writeheader()

    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        p_prod_map, p_r1b_map, p_r1e_map, tgt_c, img_c = dense_eval(sid)
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
        r1e_total, r1e_wt, r1e_out = fp_breakdown(p_r1e_map)

        w_csv.writerow({'subject_id': sid,
                        'fp_voxels_prod': prod_total, 'fp_wt_not_tc_prod': prod_wt, 'fp_outside_prod': prod_out,
                        'fp_voxels_r1b': r1b_total, 'fp_wt_not_tc_r1b': r1b_wt, 'fp_outside_r1b': r1b_out,
                        'fp_voxels_r1e': r1e_total, 'fp_wt_not_tc_r1e': r1e_wt, 'fp_outside_r1e': r1e_out})
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)
    fh.close()

    def lesion_recovery(lesion_list):
        recs = {'prod': [], 'r1b': [], 'r1e': []}
        for r in lesion_list:
            sid = r['subject_id']
            res = extract_lesion_shell_with_context(model, ds, sid_to_idx, sid, int(r['comp_id']), dev)
            if res is None:
                continue
            lf64, sf64, sz = res
            lf32 = lf64[:, :32]
            p_prod_l = expit(lf32 @ w_et + b_et)
            p_r1b_l = expit(lf32 @ r1b.coef_[0] + r1b.intercept_[0])
            p_r1e_l = expit(lf64 @ r1e.coef_[0] + r1e.intercept_[0])
            recs['prod'].append(int(p_prod_l.max() > TAU))
            recs['r1b'].append(int(p_r1b_l.max() > TAU))
            recs['r1e'].append(int(p_r1e_l.max() > TAU))
        return {k: (np.mean(v) if v else float('nan')) for k, v in recs.items()}

    det_rec = lesion_recovery(det_lesions_test)
    g2a_rec = lesion_recovery(g2a_lesions_test)
    g2b_rec = lesion_recovery(g2b_lesions_eval)
    print(f'\ndetected_test: {det_rec}')
    print(f'G2A_test: {g2a_rec}')
    print(f'G2B: {g2b_rec}')

    print(f'\nE261 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
