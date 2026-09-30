"""E259 -- anatomical/semantic audit of E257-B's FP tail, per the user's
exact spec: BEFORE designing a nonlinear readout, determine whether the
worst-14 subjects' false-positive voxels are (A) legitimate hard
negatives (truly non-lesion tissue a linear boundary just can't
separate) or (B) anatomically meaningful structures (real WT/TC
pathology, just outside the ET sub-label) that D1 genuinely cannot
distinguish from ET given the information available.

PRIMARY SIGNAL (per explicit user decision this turn): WT/TC label
overlap. Channel convention CONFIRMED against brats_multimodal_dataset.py
directly (not assumed): target channels are FIXED order (ET=0, TC=1,
WT=2), TC={1,3} superset relationship, WT={1,2,3} superset of TC. For
each FP voxel (from E258's own worst-14 population, same extraction),
checks whether that voxel falls inside TC (real tumor core, just not
labeled ET) or WT-but-not-TC (edema/broader tumor) or fully outside all
tumor regions (genuinely normal-looking tissue by the ground truth).

SECONDARY SIGNALS (per user's 5-question list):
  - spatial coherence: connected-component SIZE distribution of FP
    regions (large coherent blobs vs scattered isolated voxels)
  - distance to nearest TRUE lesion (ET) component in the same subject
    (near existing pathology vs genuinely distant/isolated)
  - raw t1c/t2f intensity signature of FP voxels vs true-ET's own
    signature vs ordinary background's own signature (E233's own
    contrast convention, reused)

Reuses E258's own worst-14 FP voxel EXTRACTION exactly (same R1-B
refit, same TAU, same distant_background/local_shell/lesion
partition) -- this is a re-analysis of the SAME false-positive
locations, adding label/intensity context that E258's D1-feature-space
distance measurement didn't include.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e257_common import (load_model, get_w_prod, load_populations, build_splits,  # noqa: E402
                         extract_lesion_shell, extract_distant_background,
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402
from sklearn.linear_model import LogisticRegression

ET, TC, WT = 0, 1, 2  # CONFIRMED fixed channel order, brats_multimodal_dataset.py

WORST_14 = ['BraTS-GLI-00033-000', 'BraTS-GLI-01496-000', 'BraTS-GLI-00085-000',
           'BraTS-GLI-01508-000', 'BraTS-GLI-00043-000', 'BraTS-GLI-00113-000',
           'BraTS-GLI-00597-000', 'BraTS-GLI-00025-000', 'BraTS-GLI-00543-000',
           'BraTS-GLI-00061-001', 'BraTS-GLI-00014-000', 'BraTS-GLI-00656-000',
           'BraTS-GLI-00498-000', 'BraTS-GLI-00293-000']


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    model = load_model(dev)
    w_prod, b_prod = get_w_prod(model)
    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng = np.random.default_rng(999)
    t0 = time.time()

    # ---- refit R1-B IDENTICALLY to E257-B/E258 ----
    print('Refitting R1-B (identical to E257-B/E258)...', flush=True)
    X, y = [], []
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
        if smoke and len(X) >= 60:
            break
    train_subjects_union = sorted(splits['det_train'] | splits['g2a_train'])
    if smoke:
        train_subjects_union = train_subjects_union[:20]
    for sid in train_subjects_union:
        feats = extract_distant_background(model, ds, sid_to_idx, sid, dev, rng)
        if feats is None:
            continue
        X.append(feats); y.append(np.zeros(len(feats)))
    X = np.concatenate(X); y = np.concatenate(y)
    r1b = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X, y)
    w_r1b = torch.from_numpy(r1b.coef_[0].astype(np.float64)).float().to(dev)
    b_r1b = float(r1b.intercept_[0])
    print(f'R1-B refit ({time.time()-t0:.0f}s)', flush=True)

    subjects = WORST_14[:6] if smoke else WORST_14
    struct = ndimage.generate_binary_structure(3, 1)

    out = HERE / ('E259_smoke.csv' if smoke else 'E259_anatomical_audit.csv')
    fh = open(out, 'w', newline='')
    w_csv = csv.DictWriter(fh, fieldnames=[
        'subject_id', 'component_id', 'component_size',
        'frac_in_TC', 'frac_in_WT_not_TC', 'frac_outside_all_tumor',
        'dist_to_nearest_ET_lesion', 'mean_t1c', 'mean_t2f'])
    w_csv.writeheader()

    print('\nAnalyzing FP components in worst-14 subjects...', flush=True)
    for i, sid in enumerate(subjects):
        if sid not in sid_to_idx:
            continue
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T
        with torch.no_grad():
            p_r1b_flat = torch.sigmoid(flat @ w_r1b + b_r1b)
        p_r1b_map = p_r1b_flat.reshape(D, H, W).cpu().numpy()

        brain_mask = img_c[0] != 0
        true_et = tgt_c[ET] > 0.5
        true_tc = tgt_c[TC] > 0.5
        true_wt = tgt_c[WT] > 0.5
        dilated = ndimage.binary_dilation(true_et, structure=struct, iterations=12) if true_et.any() else np.zeros_like(true_et)
        local_shell = dilated & (~true_et) & brain_mask
        distant_bg = brain_mask & (~true_et) & (~local_shell)

        fp_mask = (p_r1b_map > TAU) & distant_bg
        fp_lbl, fp_n = ndimage.label(fp_mask)

        et_lbl, et_n = ndimage.label(true_et)
        et_centroids = [np.array(ndimage.center_of_mass(true_et, et_lbl, c)) for c in range(1, et_n + 1)] if et_n > 0 else []

        for comp_id in range(1, fp_n + 1):
            comp_mask = fp_lbl == comp_id
            comp_size = int(comp_mask.sum())
            if comp_size < 1:
                continue
            frac_tc = float((comp_mask & true_tc).sum() / comp_size)
            frac_wt_not_tc = float((comp_mask & true_wt & (~true_tc)).sum() / comp_size)
            frac_outside = float((comp_mask & (~true_wt)).sum() / comp_size)

            if et_centroids:
                comp_centroid = np.array(ndimage.center_of_mass(comp_mask))
                dists = [np.linalg.norm(comp_centroid - c) for c in et_centroids]
                dist_nearest = float(min(dists))
            else:
                dist_nearest = float('nan')

            # channel order CONFIRMED against brats_multimodal_dataset.py:
            # MODALITIES = ("t1c", "t1n", "t2f", "t2w") -- t1c=0, t2f=2
            mean_t1c = float(img_c[0][comp_mask].mean())
            mean_t2f = float(img_c[2][comp_mask].mean())

            w_csv.writerow({'subject_id': sid, 'component_id': comp_id, 'component_size': comp_size,
                           'frac_in_TC': frac_tc, 'frac_in_WT_not_TC': frac_wt_not_tc,
                           'frac_outside_all_tumor': frac_outside, 'dist_to_nearest_ET_lesion': dist_nearest,
                           'mean_t1c': mean_t1c, 'mean_t2f': mean_t2f})
        print(f'  {sid}: {fp_n} FP components ({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'\nE259 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
