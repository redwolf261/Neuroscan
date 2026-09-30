"""E262 -- component-structure audit of R1-B, per the user's exact spec.
Compares the SPATIAL COHERENCE of true G2-A recoveries against R1-B's
false positives, across the FULL 158 held-out test subjects (not just
E259's worst-14 FP-tail subset) -- the decisive question: does R1-B's
positive-voxel STRUCTURE differ systematically between genuine
recoveries and false positives, in a way a component-aware post-
processing filter could exploit?

MEASUREMENTS (per explicit user list):
  For each G2-A TEST lesion recovered by R1-B (max_prob>tau within the
  lesion's own true-ET mask): connected-component structure of R1-B's
  OWN positive prediction mask, RESTRICTED to that lesion's local
  patch region -- number of components, largest component size, whether
  the true recovery is one coherent blob vs scattered voxels.

  For R1-B's FALSE POSITIVES (distant_background, wrong label) across
  ALL 158 test subjects (not just worst-14): connected-component size
  distribution, split into WT-not-TC vs outside-all-tumor sub-
  populations (reusing E259's own region-overlap convention), plus
  distance to nearest TRUE ET lesion centroid.

Reuses R1-B refit IDENTICALLL to E257-B/E258/E259/E260/E261 (same
split, same training population) for a fair, apples-to-apples
comparison -- this is a re-analysis/re-characterization of the SAME
readout already validated, not a new model.
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
from e257_common import (load_model, get_w_prod, load_populations, build_splits,
                         extract_lesion_shell, extract_distant_background,
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION, NEAR_DILATION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch

ET, TC, WT = 0, 1, 2


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
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng = np.random.default_rng(999)
    t0 = time.time()

    print('Refitting R1-B (identical to E257-B/E258/E259/E260/E261)...', flush=True)
    X, y = [], []
    for r in det_lesions_fit:
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
    for sid in bg_subjects:
        feats = extract_distant_background(model, ds, sid_to_idx, sid, dev, rng)
        if feats is None:
            continue
        X.append(feats); y.append(np.zeros(len(feats)))
    X = np.concatenate(X); y = np.concatenate(y)
    r1b = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X, y)
    w_r1b = torch.from_numpy(r1b.coef_[0].astype(np.float64)).float().to(dev)
    b_r1b = float(r1b.intercept_[0])
    print(f'R1-B refit ({time.time()-t0:.0f}s)', flush=True)

    struct = ndimage.generate_binary_structure(3, 1)

    def dense_probs(sid):
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T
        with torch.no_grad():
            p_r1b_flat = torch.sigmoid(flat @ w_r1b + b_r1b)
        p_r1b_map = p_r1b_flat.reshape(D, H, W).cpu().numpy()
        return p_r1b_map, tgt_c, img_c

    # ---- PART 1: recovered-lesion component structure ----
    print('\nPart 1: component structure of RECOVERED G2-A lesions...', flush=True)
    recov_out = HERE / ('E262_recovered_smoke.csv' if smoke else 'E262_recovered_components.csv')
    fh1 = open(recov_out, 'w', newline='')
    w1 = csv.DictWriter(fh1, fieldnames=['subject_id', 'comp_id', 'lesion_size', 'recovered',
                                         'n_components_in_lesion', 'largest_component_size',
                                         'largest_component_frac'])
    w1.writeheader()

    subj_cache = {}
    n_done = 0
    for r in g2a_lesions_test:
        sid, cid = r['subject_id'], int(r['comp_id'])
        if sid not in sid_to_idx:
            continue
        if sid not in subj_cache:
            subj_cache[sid] = dense_probs(sid)
        p_r1b_map, tgt_c, img_c = subj_cache[sid]
        et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
        if cid < 1 or cid > et_n:
            continue
        lesion_mask = et_lbl == cid
        lesion_size = int(lesion_mask.sum())
        if lesion_size < MIN_VOX:
            continue
        pos_in_lesion = (p_r1b_map > TAU) & lesion_mask
        recovered = int(pos_in_lesion.any())
        if recovered:
            pos_lbl, pos_n = ndimage.label(pos_in_lesion, structure=struct)
            comp_sizes = [int((pos_lbl == c).sum()) for c in range(1, pos_n + 1)]
            largest = max(comp_sizes) if comp_sizes else 0
            largest_frac = largest / max(1, int(pos_in_lesion.sum()))
        else:
            pos_n, largest, largest_frac = 0, 0, 0.0
        w1.writerow({'subject_id': sid, 'comp_id': cid, 'lesion_size': lesion_size,
                    'recovered': recovered, 'n_components_in_lesion': pos_n,
                    'largest_component_size': largest, 'largest_component_frac': largest_frac})
        n_done += 1
        if n_done % 20 == 0 or smoke:
            print(f'  {n_done} G2A lesions ({time.time()-t0:.0f}s)', flush=True)
    fh1.close()
    print(f'Part 1 complete: wrote {recov_out.name}', flush=True)

    # ---- PART 2: FP component structure across ALL 158 test subjects ----
    print('\nPart 2: FP component structure across ALL held-out subjects...', flush=True)
    # test subjects = det_test + g2a_test + all g2b (same convention as E257-B)
    det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']]
    test_subjects = sorted(set(r['subject_id'] for r in det_lesions_test) |
                          set(r['subject_id'] for r in g2a_lesions_test) |
                          set(r['subject_id'] for r in g2b_lesions))
    if smoke:
        test_subjects = test_subjects[:15]

    fp_out = HERE / ('E262_fp_smoke.csv' if smoke else 'E262_fp_components.csv')
    fh2 = open(fp_out, 'w', newline='')
    w2 = csv.DictWriter(fh2, fieldnames=['subject_id', 'component_id', 'component_size',
                                         'category', 'dist_to_nearest_ET_lesion'])
    w2.writeheader()

    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        if sid not in subj_cache:
            subj_cache[sid] = dense_probs(sid)
        p_r1b_map, tgt_c, img_c = subj_cache[sid]
        brain_mask = img_c[0] != 0
        true_et = tgt_c[ET] > 0.5
        true_tc = tgt_c[TC] > 0.5
        true_wt = tgt_c[WT] > 0.5
        dilated = ndimage.binary_dilation(true_et, structure=struct, iterations=NEAR_DILATION) if true_et.any() else np.zeros_like(true_et)
        local_shell = dilated & (~true_et) & brain_mask
        distant_bg = brain_mask & (~true_et) & (~local_shell)

        fp_mask = (p_r1b_map > TAU) & distant_bg
        fp_lbl, fp_n = ndimage.label(fp_mask, structure=struct)

        et_lbl, et_n = ndimage.label(true_et)
        et_centroids = [np.array(ndimage.center_of_mass(true_et, et_lbl, c)) for c in range(1, et_n + 1)] if et_n > 0 else []

        for comp_id in range(1, fp_n + 1):
            comp_mask = fp_lbl == comp_id
            comp_size = int(comp_mask.sum())
            if comp_size < 1:
                continue
            frac_tc = (comp_mask & true_tc).sum() / comp_size
            frac_wt_not_tc = (comp_mask & true_wt & (~true_tc)).sum() / comp_size
            if frac_tc > 0.5 or frac_wt_not_tc > 0.5:
                category = 'WT_or_TC'
            else:
                category = 'outside_tumor'
            if et_centroids:
                comp_centroid = np.array(ndimage.center_of_mass(comp_mask))
                dist_nearest = float(min(np.linalg.norm(comp_centroid - c) for c in et_centroids))
            else:
                dist_nearest = float('nan')
            w2.writerow({'subject_id': sid, 'component_id': comp_id, 'component_size': comp_size,
                        'category': category, 'dist_to_nearest_ET_lesion': dist_nearest})
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} subjects ({time.time()-t0:.0f}s)', flush=True)
    fh2.close()
    print(f'Part 2 complete: wrote {fp_out.name}', flush=True)

    print(f'\nE262 complete ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
