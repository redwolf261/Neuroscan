"""E288 -- Sliding-Window Dice Comparison, per explicit user request:
"same stats for our best model" -- a clean, apples-to-apples ET Dice
comparison between production alone and production+MRD, using the
EXACT SAME measurement protocol as the production checkpoint's own
reported baseline (0.8399 ET / 0.8929 mean Dice, epoch 33,
E131_v5control_seed0/epoch_metrics.csv), not E286/E287's own center-
crop + pooled-voxel convention (a genuinely different, non-comparable
measurement, confirmed by reading train_e131_rap.py directly).

PROTOCOL (reused verbatim from train_e131_rap.py, NOT reimplemented --
imported directly so this is guaranteed bit-identical to how the
checkpoint's own 0.8399 ET Dice was produced):
  - sliding_window_predict(): Gaussian-weighted overlap-tile inference
    over the FULL volume (not a single fixed-center crop), PATCH=
    (128,128,128), SW_OVERLAP=0.5 -- both constants reused from the
    original script, not re-chosen.
  - dice_per_region(): threshold=0.5, per-subject Dice, empty-target
    convention Dice=1.0 if both empty else 0.0 -- reused verbatim.
  - Per-subject Dice values AVERAGED across subjects (not voxels
    pooled across the whole cohort first) -- matches the original
    validate() method's own convention exactly.

EXTENSION (the only new code): since sliding_window_predict() only
returns the production model's own softmax probs, this script adds a
PARALLEL sliding-window pass that also runs forward_to_dec1_internal()
on each tile and applies the frozen MRD-T2 readout (w_phi/b_phi, fit
identically to E286-E287: R1-B pool, det_train, 5 seeds), accumulated
with the SAME Gaussian tile-weighting as production's own probs --
guarantees the MRD score map is computed with the SAME sliding-window
discipline as production's comparison point, not a different
(center-crop) one.

TEST COHORT: the EXACT SAME 125-subject locked cohort as E286/E287
(rebuilt identically, re-verified 0 leakage).

CONDITIONS: A=production alone (hat_Y_P = P(x)>0.5, the SAME threshold
as the checkpoint's own dice_per_region convention), C=production
UNION raw MRD candidates (same tau_P=0.5, tau_R=E283's own FP~500
threshold, same construction as E287's winning condition C) -- the
two conditions needed to answer "production vs our best model," not
the full E287 ablation (D/E are not re-run here since E287 already
established their relationship to C).
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT_REPO / 'experiments' / 'exp_e12_eggo_m' / 'e131'))
from e257_common import (load_model, get_w_prod, load_populations, build_splits,  # noqa: E402
                         extract_lesion_shell, extract_distant_background,
                         ROOT, PATCH, MAX_VOX_PER_LESION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402
from train_e131_rap import dice_per_region, _gaussian_weight, SW_OVERLAP  # noqa: E402

ET, TC, WT = 0, 1, 2
EPS_MARG = 1e-6
SEEDS = [999, 4242, 7, 123, 2024]
TAU_P = 0.5
TAU_R = 0.95082877089122  # T2's own FP~500 threshold, E283 development cohort (frozen, unchanged from E287)


def fit_soft_target(X, y_soft, max_iter=500, C=1.0):
    X2 = np.concatenate([X, X], axis=0)
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w2 = np.concatenate([y_soft, 1.0 - y_soft])
    keep = w2 > 1e-12
    X2, y2, w2 = X2[keep], y2[keep], w2[keep]
    clf = LogisticRegression(max_iter=max_iter, C=C, class_weight='balanced')
    clf.fit(X2, y2, sample_weight=w2)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])


def sliding_window_predict_with_mrd(model, image, patch, overlap, device, amp,
                                    w_mrd_t, b_mrd):
    """Variant of train_e131_rap.py's sliding_window_predict() that ALSO
    accumulates the MRD-T2 score map via forward_to_dec1_internal(), using
    the IDENTICAL tiling/Gaussian-weighting loop as the original (copied,
    not reimplemented from scratch, to guarantee the same tile geometry).
    Returns (probs (3,D,H,W), mrd_score (D,H,W))."""
    _, _, D, H, W = image.shape
    pd, ph, pw = patch
    stride = [max(1, int(p * (1 - overlap))) for p in patch]

    def starts(full, p, st):
        if full <= p:
            return [0]
        s = list(range(0, full - p + 1, st))
        if s[-1] != full - p:
            s.append(full - p)
        return s

    zs, ys, xs = (starts(D, pd, stride[0]), starts(H, ph, stride[1]), starts(W, pw, stride[2]))
    acc_p = torch.zeros((3, D, H, W), device=device, dtype=torch.float32)
    acc_r = torch.zeros((1, D, H, W), device=device, dtype=torch.float32)
    wsum = torch.zeros((1, D, H, W), device=device, dtype=torch.float32)
    gw = torch.from_numpy(_gaussian_weight((min(pd, D), min(ph, H), min(pw, W)))).to(device)

    with torch.no_grad():
        for z in zs:
            for y in ys:
                for x in xs:
                    zc, yc, xc = min(pd, D), min(ph, H), min(pw, W)
                    tile = image[:, :, z:z + zc, y:y + yc, x:x + xc]
                    if tile.shape[2:] != (pd, ph, pw):
                        tile = F.pad(tile, (0, pw - tile.shape[4], 0, ph - tile.shape[3], 0, pd - tile.shape[2]))
                    with torch.amp.autocast("cuda", enabled=amp):
                        out = model(tile)
                    pr = out["probs"].float()[:, :, :zc, :yc, :xc].squeeze(0)
                    acc_p[:, z:z + zc, y:y + yc, x:x + xc] += pr * gw
                    # MRD: replay dec1 on the SAME tile, apply the frozen readout
                    stages = forward_to_dec1_internal(model, tile)
                    d1 = stages['relu2_out']  # (1,32,pd,ph,pw)
                    d1_flat = d1[0].reshape(32, -1).T  # (pd*ph*pw, 32)
                    r_score = torch.sigmoid(d1_flat.float() @ w_mrd_t + b_mrd).reshape(1, pd, ph, pw)
                    r_score = r_score[:, :zc, :yc, :xc]
                    acc_r[:, z:z + zc, y:y + yc, x:x + xc] += r_score * gw
                    wsum[:, z:z + zc, y:y + yc, x:x + xc] += gw
    probs = (acc_p / wsum.clamp(min=1e-6)).cpu().numpy()
    mrd_score = (acc_r / wsum.clamp(min=1e-6))[0].cpu().numpy()
    return probs, mrd_score


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')
    t0_time = time.time()

    model = load_model(dev)
    w_et, b_et = get_w_prod(model)
    w_p_np, b_p_np = w_et.astype(np.float64), float(b_et)

    def p_prod_of(x):
        z = x @ w_p_np + b_p_np
        return 1.0 / (1.0 + np.exp(-z))

    # ---- rebuild E286's exact locked test cohort ----
    print('Rebuilding E286\'s exact locked test cohort...', flush=True)
    ds_train_pop = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                          val_split=0.1, patch_size=PATCH)
    ds_val_pop = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                        val_split=0.1, patch_size=PATCH)
    val_sids_all = sorted(os.path.basename(d) for d in ds_val_pop.subject_dirs)
    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)
    all_prior_touched = (set(r['subject_id'] for r in det_lesions) |
                         set(r['subject_id'] for r in g2a_lesions) |
                         set(r['subject_id'] for r in g2b_lesions))
    val_sids_fresh = sorted(set(val_sids_all) - all_prior_touched)

    ds_full = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                     val_split=0.0, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds_full.subject_dirs)}

    eligible_sids = []
    check_pool = val_sids_fresh[:10] if smoke else val_sids_fresh
    for sid in check_pool:
        if sid not in sid_to_idx:
            continue
        try:
            img_c, tgt_c = load_patch(ds_full, sid_to_idx, sid)
            img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
            stages = forward_to_dec1_internal(model, img_t)
            d1 = stages['relu2_out'][0]
            if torch.isnan(d1).any() or torch.isinf(d1).any():
                continue
            del d1, img_t, stages
            torch.cuda.empty_cache()
            eligible_sids.append(sid)
        except Exception:
            continue
    test_subjects = sorted(eligible_sids)
    overlap_with_train = set(test_subjects) & set(os.path.basename(d) for d in ds_train_pop.subject_dirs)
    overlap_with_prior = set(test_subjects) & all_prior_touched
    assert len(overlap_with_train) == 0 and len(overlap_with_prior) == 0, 'LEAKAGE AUDIT FAILED'
    print(f'  test cohort: {len(test_subjects)} subjects, leakage audit PASS', flush=True)

    # ---- fit MRD-T2 on det_train, 5 seeds (identical to E286-E287) ----
    print('\nFitting MRD-T2 on det_train, 5 seeds...', flush=True)
    if smoke:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']][:30]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
        seeds = SEEDS[:2]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])
        seeds = SEEDS

    def fit_r1b_pool(seed):
        rng = np.random.default_rng(seed)
        lesion_pool, neg_pool = [], []
        for r in det_lesions_fit:
            res = extract_lesion_shell(model, ds_train_pop, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
            if res is None:
                continue
            lf, sf, sz = res
            if len(lf) > MAX_VOX_PER_LESION:
                lf = lf[rng.choice(len(lf), MAX_VOX_PER_LESION, replace=False)]
            if len(sf) > MAX_VOX_PER_LESION:
                sf = sf[rng.choice(len(sf), MAX_VOX_PER_LESION, replace=False)]
            lesion_pool.append(lf); neg_pool.append(sf)
        for sid in bg_subjects:
            feats = extract_distant_background(model, ds_train_pop, sid_to_idx, sid, dev, rng)
            if feats is None:
                continue
            neg_pool.append(feats)
        lesion_pool = np.concatenate(lesion_pool).astype(np.float64)
        neg_pool = np.concatenate(neg_pool).astype(np.float64)
        return lesion_pool, neg_pool

    fitted = {}
    for seed in seeds:
        lesion_pool, neg_pool = fit_r1b_pool(seed)
        X = np.concatenate([lesion_pool, neg_pool])
        y_hard = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
        p_i = p_prod_of(X)
        t2_vals = np.clip((y_hard - p_i) / (1.0 - p_i + EPS_MARG), 0.0, 1.0)
        w_h3, b_h3 = fit_soft_target(X, t2_vals)
        fitted[seed] = (w_h3, b_h3)
        print(f'  seed={seed} fit ({time.time()-t0_time:.0f}s)', flush=True)

    # ---- sliding-window Dice, per subject, per seed ----
    print('\nRunning sliding-window inference (production + MRD-T2) per subject...', flush=True)
    test_subjects_eval = test_subjects[:10] if smoke else test_subjects
    amp = True

    rows = []
    for seed in seeds:
        w_mrd_t = torch.from_numpy(fitted[seed][0]).float().to(dev)
        b_mrd = fitted[seed][1]
        dice_A, dice_C = [], []
        for i, sid in enumerate(test_subjects_eval):
            idx = sid_to_idx[sid]
            image, target, _ = ds_full._load_subject(ds_full.subject_dirs[idx])
            img_t = torch.from_numpy(image).unsqueeze(0).float().to(dev)
            probs, mrd_score = sliding_window_predict_with_mrd(model, img_t, PATCH, SW_OVERLAP,
                                                                dev, amp, w_mrd_t, b_mrd)
            tgt_np = target  # (3,D,H,W)

            # condition A: production alone, threshold 0.5 (SAME convention as dice_per_region)
            pred_A = (probs >= 0.5).astype(np.float32)
            dA = dice_per_region(pred_A, tgt_np)
            dice_A.append(dA)

            # condition C: production UNION raw MRD candidates, in the region
            # production's ET channel is uncertain (P_ET<tau_P) AND MRD exceeds tau_R
            # -- ET channel only (index 0), matching E287's own construction
            p_et = probs[ET]
            uncertain = p_et < TAU_P
            mrd_candidates = mrd_score > TAU_R
            c_region = uncertain & mrd_candidates
            pred_C = pred_A.copy()
            pred_C[ET] = np.logical_or(pred_A[ET] > 0.5, c_region).astype(np.float32)
            dC = dice_per_region(pred_C, tgt_np)
            dice_C.append(dC)

            if (i + 1) % 10 == 0 or smoke:
                print(f'  seed={seed} {i+1}/{len(test_subjects_eval)} ({time.time()-t0_time:.0f}s)', flush=True)

        dice_A = np.array(dice_A)  # (n_subj, 3)
        dice_C = np.array(dice_C)
        rows.append({
            'seed': seed, 'condition': 'A_production_alone',
            'dice_ET': float(dice_A[:, ET].mean()), 'dice_TC': float(dice_A[:, TC].mean()),
            'dice_WT': float(dice_A[:, WT].mean()), 'dice_mean': float(dice_A.mean()),
            'n_subjects': len(test_subjects_eval),
        })
        rows.append({
            'seed': seed, 'condition': 'C_production_plus_mrd',
            'dice_ET': float(dice_C[:, ET].mean()), 'dice_TC': float(dice_C[:, TC].mean()),
            'dice_WT': float(dice_C[:, WT].mean()), 'dice_mean': float(dice_C.mean()),
            'n_subjects': len(test_subjects_eval),
        })
        print(f'  seed={seed}: A ET={dice_A[:,ET].mean():.4f} mean={dice_A.mean():.4f}  |  '
             f'C ET={dice_C[:,ET].mean():.4f} mean={dice_C.mean():.4f}', flush=True)

    out = HERE / ('E288_sliding_window_dice_smoke.csv' if smoke else 'E288_sliding_window_dice.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['seed', 'condition', 'dice_ET', 'dice_TC', 'dice_WT',
                                           'dice_mean', 'n_subjects'])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f'\nwrote {out.name}', flush=True)

    print('\n' + '=' * 70, flush=True)
    print('SUMMARY (mean +/- SD across seeds) -- compare against checkpoint\'s own', flush=True)
    print('epoch-33 reference: ET=0.8399 TC=0.9137 WT=0.9253 mean=0.8929', flush=True)
    print('=' * 70, flush=True)
    for cond in ['A_production_alone', 'C_production_plus_mrd']:
        cond_rows = [r for r in rows if r['condition'] == cond]
        for k in ['dice_ET', 'dice_TC', 'dice_WT', 'dice_mean']:
            vals = [r[k] for r in cond_rows]
            print(f'  {cond:24s} {k}: {np.mean(vals):.4f}±{np.std(vals):.4f}', flush=True)

    print(f'\nE288 complete ({time.time()-t0_time:.0f}s).', flush=True)


if __name__ == '__main__':
    main()
