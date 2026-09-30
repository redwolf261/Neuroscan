"""E258 -- diagnose E257-B's FP tail (14/158 subjects with >100x FP
increase), per the user's exact spec. Does NOT change the algorithm --
pure measurement, distinguishing three competing explanations:
  1. Under-sampled subtype: FP voxels are far from BOTH the training
     negative sample AND real lesion voxels -- a distinct background
     subtype the training sample just didn't cover -> stratified
     sampling would help.
  2. Feature-space overlap with lesions: FP voxels are CLOSE to real
     lesion voxels in D1 feature space -> R1-B genuinely cannot
     distinguish this tissue from lesion with a linear boundary -> more
     negatives won't fix it, need a more constrained readout.
  3. Pure sampling density: FP voxels look like ordinary background,
     just under-sampled -> more negatives (same distribution) would help.

METHOD (per explicit user choice on this turn's disambiguation):
distance-in-feature-space, reusing e234's own separability() z-scoring
convention DIRECTLY (not reimplemented) -- for each FP voxel (from the
14 worst subjects), compute the z-scored distance to (a) the TRAINING
NEGATIVE distribution's own mean/std (same distribution E257-B's R1-B
was actually trained on), (b) the TRAINING LESION distribution's own
mean/std. Compare against the SAME two distances computed for (c)
ordinary sampled negatives from the 44 NORMAL subjects (ratio<=2x, per
E257-B's own results) -- the reference/control population showing what
"typical" distant-background voxels look like under this same metric.
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
from scipy.special import expit

WORST_14 = ['BraTS-GLI-00033-000', 'BraTS-GLI-01496-000', 'BraTS-GLI-00085-000',
           'BraTS-GLI-01508-000', 'BraTS-GLI-00043-000', 'BraTS-GLI-00113-000',
           'BraTS-GLI-00597-000', 'BraTS-GLI-00025-000', 'BraTS-GLI-00543-000',
           'BraTS-GLI-00061-001', 'BraTS-GLI-00014-000', 'BraTS-GLI-00656-000',
           'BraTS-GLI-00498-000', 'BraTS-GLI-00293-000']


def zscore_distance_robust(voxels, ref_mean, ref_std, floor_std):
    """voxels: (n,32). Returns (n,) -- per-voxel z-scored Euclidean
    distance to ref_mean, normalized by ref_std per-channel, EXCLUDING
    channels where ref_std < floor_std (per e234_layerwise_separability.py's
    own separability() relative-floor convention -- 1% of that channel's
    overall activation scale across ALL pooled training data, NOT a flat
    epsilon). floor_std: (32,) array, computed ONCE by the caller from a
    stable, large reference sample (all pooled train data combined), so
    every distance computation in this script uses the SAME floor
    consistently.

    BUG FOUND AND FIXED before trusting any result: an earlier version
    used a flat ref_std+1e-6 divisor -- a smoke-test run showed mean
    distances of 982,982 and 352,688 for two reference comparisons (vs
    sane medians of ~13-15 for the SAME data), immediately recognizable
    as a few near-zero-variance channels dominating the aggregate via
    division blowup, exactly the failure mode e234's own separability()
    was built to avoid. Fixed by reusing that exact established
    convention instead of inventing a different one."""
    valid = ref_std >= floor_std
    if valid.sum() == 0:
        return np.full(len(voxels), np.nan)
    z = (voxels[:, valid] - ref_mean[valid]) / ref_std[valid]
    return np.sqrt((z ** 2).sum(axis=1)) / np.sqrt(valid.sum())


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

    # ---- refit R1-B IDENTICALLY to E257-B ----
    print('Refitting R1-B (identical to E257-B)...', flush=True)
    X, y = [], []
    lesion_feats_all = []  # keep raw lesion feats for the "training lesion distribution" reference
    neg_feats_all = []     # keep raw negative feats (shell+distant) for the "training negative distribution" reference
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
        X.append(lf); y.append(np.ones(len(lf))); lesion_feats_all.append(lf)
        X.append(sf); y.append(np.zeros(len(sf))); neg_feats_all.append(sf)
        if smoke and len(lesion_feats_all) >= 30:
            break

    train_subjects_union = sorted(splits['det_train'] | splits['g2a_train'])
    if smoke:
        train_subjects_union = train_subjects_union[:20]
    for sid in train_subjects_union:
        feats = extract_distant_background(model, ds, sid_to_idx, sid, dev, rng)
        if feats is None:
            continue
        X.append(feats); y.append(np.zeros(len(feats))); neg_feats_all.append(feats)

    X = np.concatenate(X); y = np.concatenate(y)
    lesion_feats_all = np.concatenate(lesion_feats_all)
    neg_feats_all = np.concatenate(neg_feats_all)
    r1b = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X, y)
    w_r1b = r1b.coef_[0].astype(np.float64)
    b_r1b = float(r1b.intercept_[0])
    print(f'R1-B refit ({time.time()-t0:.0f}s). train_neg n={len(neg_feats_all)}  train_lesion n={len(lesion_feats_all)}', flush=True)

    # ---- reference distributions ----
    neg_mean = neg_feats_all.mean(axis=0); neg_std = neg_feats_all.std(axis=0)
    lesion_mean = lesion_feats_all.mean(axis=0); lesion_std = lesion_feats_all.std(axis=0)
    # SAME relative-floor convention as e234's separability(): 1% of each
    # channel's own overall activation std, computed from ALL pooled
    # training data (neg+lesion combined) for a single stable reference.
    all_pooled = np.concatenate([neg_feats_all, lesion_feats_all], axis=0)
    overall_std = all_pooled.std(axis=0)
    floor_std = np.maximum(overall_std * 0.01, 1e-6)

    w_prod_t = torch.from_numpy(w_prod).float().to(dev)
    w_r1b_t = torch.from_numpy(w_r1b).float().to(dev)

    def get_fp_and_normal_bg_voxels(sid, n_normal_sample=200):
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T
        with torch.no_grad():
            p_r1b_flat = torch.sigmoid(flat @ w_r1b_t + b_r1b)
        p_r1b_map = p_r1b_flat.reshape(D, H, W).cpu().numpy()
        d1_np = d1.cpu().numpy()

        brain_mask = img_c[0] != 0
        true_et = tgt_c[0] > 0.5
        struct = ndimage.generate_binary_structure(3, 1)
        if true_et.any():
            dilated = ndimage.binary_dilation(true_et, structure=struct, iterations=12)
            local_shell = dilated & (~true_et) & brain_mask
        else:
            local_shell = np.zeros_like(true_et)
        distant_bg = brain_mask & (~true_et) & (~local_shell)

        fp_mask = (p_r1b_map > TAU) & distant_bg
        fp_coords = np.where(fp_mask)
        fp_feats = d1_np[:, fp_coords[0], fp_coords[1], fp_coords[2]].T if len(fp_coords[0]) > 0 else np.zeros((0, 32))

        normal_bg_coords = np.where(distant_bg & (~fp_mask))
        if len(normal_bg_coords[0]) > 0:
            idx = rng.choice(len(normal_bg_coords[0]), size=min(n_normal_sample, len(normal_bg_coords[0])), replace=False)
            sel = (normal_bg_coords[0][idx], normal_bg_coords[1][idx], normal_bg_coords[2][idx])
            normal_bg_feats = d1_np[:, sel[0], sel[1], sel[2]].T
        else:
            normal_bg_feats = np.zeros((0, 32))
        return fp_feats, normal_bg_feats

    print('\nExtracting FP voxels from WORST 14 subjects...', flush=True)
    worst_fp_all = []
    for i, sid in enumerate(WORST_14):
        if sid not in sid_to_idx:
            continue
        fp_feats, _ = get_fp_and_normal_bg_voxels(sid)
        if len(fp_feats) > 500:
            fp_feats = fp_feats[rng.choice(len(fp_feats), 500, replace=False)]
        worst_fp_all.append(fp_feats)
        print(f'  {sid}: {len(fp_feats)} FP voxels ({time.time()-t0:.0f}s)', flush=True)
    worst_fp_all = np.concatenate(worst_fp_all) if worst_fp_all else np.zeros((0, 32))

    # ---- normal-subject background sample (control) ----
    print('\nExtracting ordinary background from NORMAL subjects (ratio<=2x)...', flush=True)
    e257b_rows = list(csv.DictReader(open(HERE / 'E257b_r1b_eval.csv')))
    prod_arr = np.array([int(r['fp_voxels_prod']) for r in e257b_rows])
    final_arr = np.array([int(r['fp_voxels_r1b']) for r in e257b_rows])
    ratio_arr = final_arr / np.maximum(prod_arr, 1)
    normal_subjects = [e257b_rows[i]['subject_id'] for i in np.where(ratio_arr <= 2)[0]]
    if smoke:
        normal_subjects = normal_subjects[:10]

    normal_bg_all = []
    for i, sid in enumerate(normal_subjects):
        if sid not in sid_to_idx:
            continue
        _, normal_feats = get_fp_and_normal_bg_voxels(sid, n_normal_sample=50)
        normal_bg_all.append(normal_feats)
        if (i + 1) % 10 == 0 or smoke:
            print(f'  {i+1}/{len(normal_subjects)} ({time.time()-t0:.0f}s)', flush=True)
    normal_bg_all = np.concatenate(normal_bg_all) if normal_bg_all else np.zeros((0, 32))

    # ---- compute z-scored distances ----
    print('\nComputing z-scored distances (e234 separability-statistic convention)...', flush=True)

    def summarize(name, voxels):
        if len(voxels) == 0:
            print(f'  {name}: n=0, skipped')
            return
        d_to_neg = zscore_distance_robust(voxels, neg_mean, neg_std, floor_std)
        d_to_lesion = zscore_distance_robust(voxels, lesion_mean, lesion_std, floor_std)
        print(f'  {name} (n={len(voxels)}):')
        print(f'    distance to TRAIN NEGATIVE dist: mean={d_to_neg.mean():.3f}  median={np.median(d_to_neg):.3f}')
        print(f'    distance to TRAIN LESION dist:   mean={d_to_lesion.mean():.3f}  median={np.median(d_to_lesion):.3f}')

    summarize('Worst-14 FP voxels', worst_fp_all)
    summarize('Normal-subject ordinary background', normal_bg_all)
    summarize('Training negative sample itself (self-distance sanity check)', neg_feats_all[:2000])
    summarize('Training lesion sample itself (self-distance sanity check)', lesion_feats_all[:2000])

    np.savez(HERE / ('E258_smoke.npz' if smoke else 'E258_fp_diagnosis.npz'),
             worst_fp=worst_fp_all, normal_bg=normal_bg_all,
             neg_mean=neg_mean, neg_std=neg_std, lesion_mean=lesion_mean, lesion_std=lesion_std)
    print(f'\nE258 complete ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
