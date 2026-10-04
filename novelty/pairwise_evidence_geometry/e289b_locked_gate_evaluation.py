"""E289-B -- Locked Test Evaluation of the SELECTED Gate, per the
user's exact spec. Applies the SINGLE frozen gate selected in E289-A
(read directly from E289A_selected_gate.json, never re-tuned here) to
E286-E288's exact 125-subject LOCKED test cohort -- the one and only
touch of that cohort for this experiment, per explicit user protocol
("No threshold tuning... after seeing locked-test results").

Evaluated under BOTH protocols for completeness (per explicit user
scoping this session): (1) E286/E287's fast center-crop + pooled-voxel
protocol, directly comparable to E287's own production/raw-MRD/
filtered-component numbers; (2) E288's sliding-window + per-subject-
averaged protocol, directly comparable to the production checkpoint's
own original training-time Dice convention.
"""
import sys, csv, json, time, os
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
                         ROOT, PATCH, MIN_VOX, TAU, MAX_VOX_PER_LESION)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402
from train_e131_rap import dice_per_region, _gaussian_weight, SW_OVERLAP  # noqa: E402

ET, TC, WT = 0, 1, 2
EPS_MARG = 1e-6
TAU_LOW, TAU_HIGH = TAU, 0.9
SEEDS = [999, 4242, 7, 123, 2024]
STRUCT = ndimage.generate_binary_structure(3, 1)
TAU_P = 0.5


def fit_soft_target(X, y_soft, max_iter=500, C=1.0):
    X2 = np.concatenate([X, X], axis=0)
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w2 = np.concatenate([y_soft, 1.0 - y_soft])
    keep = w2 > 1e-12
    X2, y2, w2 = X2[keep], y2[keep], w2[keep]
    clf = LogisticRegression(max_iter=max_iter, C=C, class_weight='balanced')
    clf.fit(X2, y2, sample_weight=w2)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])


def apply_gate(r_map, uncertain, cfg):
    mrd_pos = (r_map > cfg['tau_R']) & uncertain
    if cfg['gate'] in ('G0_G1_raw',):
        return mrd_pos
    cc_lbl, cc_n = ndimage.label(mrd_pos, structure=STRUCT)
    if cc_n == 0:
        return mrd_pos
    admit = np.zeros(cc_n + 1, dtype=bool)
    sizes = ndimage.sum(mrd_pos, cc_lbl, index=np.arange(1, cc_n + 1))
    if cfg['gate'] == 'G2_component_size':
        admit[1:] = sizes >= cfg['k']
    elif cfg['gate'] == 'G4_confidence':
        means = ndimage.mean(r_map, cc_lbl, index=np.arange(1, cc_n + 1))
        admit[1:] = means >= cfg['gamma']
    elif cfg['gate'] == 'G5_combined':
        means = ndimage.mean(r_map, cc_lbl, index=np.arange(1, cc_n + 1))
        admit[1:] = (sizes >= cfg['k']) & (means >= cfg['gamma'])
    return admit[cc_lbl]


def sliding_window_predict_with_mrd(model, image, patch, overlap, device, amp, w_mrd_t, b_mrd):
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
                    stages = forward_to_dec1_internal(model, tile)
                    d1 = stages['relu2_out']
                    d1_flat = d1[0].reshape(32, -1).T
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

    sel_path = HERE / ('E289A_selected_gate_smoke.json' if smoke else 'E289A_selected_gate.json')
    with open(sel_path) as fh:
        selection = json.load(fh)
    cfg = selection['selected_config']
    print('FROZEN SELECTED GATE (from E289-A, never re-tuned here):', flush=True)
    print(f'  {cfg}', flush=True)
    print(f'  dev-cohort reference: production_dice={selection["production_dev_dice"]:.4f}, '
         f'R_min={selection["R_min"]}', flush=True)

    model = load_model(dev)
    w_et, b_et = get_w_prod(model)
    w_p_np, b_p_np = w_et.astype(np.float64), float(b_et)

    def p_prod_of(x):
        z = x @ w_p_np + b_p_np
        return 1.0 / (1.0 + np.exp(-z))

    print('\nRebuilding E286/E288\'s exact locked test cohort...', flush=True)
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

    print('\nFitting MRD-T2 on det_train, 5 seeds (identical to E286-E289A)...', flush=True)
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

    w_p_t = torch.from_numpy(w_p_np).float().to(dev)
    tensors = {seed: (torch.from_numpy(w).float().to(dev), b) for seed, (w, b) in fitted.items()}

    dense_cache = {}

    def dense_eval(sid):
        if sid in dense_cache:
            return dense_cache[sid]
        img_c, tgt_c = load_patch(ds_full, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]
        C_, D, H, W = d1.shape
        flat = d1.reshape(C_, -1).T
        maps = {}
        with torch.no_grad():
            z_prod = flat @ w_p_t + b_p_np
            maps['prod'] = torch.sigmoid(z_prod).reshape(D, H, W).cpu().numpy()
            for seed, (w_t, b) in tensors.items():
                r = torch.sigmoid(flat @ w_t + b)
                maps[seed] = r.reshape(D, H, W).cpu().numpy()
        del d1, flat, img_t, stages, z_prod
        torch.cuda.empty_cache()
        result = (maps, tgt_c, img_c)
        dense_cache[sid] = result
        return result

    test_subjects_eval = test_subjects[:10] if smoke else test_subjects

    print(f'\nBuilding G1/G2/G3 phenotype groups on {len(test_subjects_eval)} LOCKED test subjects...',
         flush=True)
    lesion_records = []
    for i, sid in enumerate(test_subjects_eval):
        maps, tgt_c, img_c = dense_eval(sid)
        p_prod = maps['prod']
        true_et = tgt_c[ET] > 0.5
        et_lbl, et_n = ndimage.label(true_et)
        for cid in range(1, et_n + 1):
            lesion_mask = et_lbl == cid
            sz = int(lesion_mask.sum())
            if sz < MIN_VOX:
                continue
            p_max = float(p_prod[lesion_mask].max())
            if p_max < TAU_LOW:
                phenotype = 'G1_confidently_missed'
            elif p_max < TAU_HIGH:
                phenotype = 'G2_low_confidence'
            else:
                phenotype = 'G3_confidently_detected'
            lesion_records.append({'subject_id': sid, 'comp_id': cid, 'phenotype': phenotype, 'size': sz})
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects_eval)} ({time.time()-t0_time:.0f}s)', flush=True)
    lesion_by_sid = {}
    for rec in lesion_records:
        lesion_by_sid.setdefault(rec['subject_id'], []).append(rec)
    n_by_phen = {}
    for rec in lesion_records:
        n_by_phen[rec['phenotype']] = n_by_phen.get(rec['phenotype'], 0) + 1
    print(f'  phenotype group sizes: {n_by_phen}', flush=True)

    # ---- PROTOCOL 1: center-crop + pooled voxel (E286/E287's convention) ----
    print('\n--- PROTOCOL 1: center-crop + pooled voxels (E286/E287 convention) ---', flush=True)
    results_p1 = []
    for seed in seeds:
        tp = fp = fn = tn = 0
        g1_vals = []
        for sid in test_subjects_eval:
            maps, tgt_c, img_c = dense_eval(sid)
            p_map = maps['prod']
            r_map = maps[seed]
            true_et = tgt_c[ET] > 0.5
            brain_mask = img_c[0] != 0
            et_lbl, et_n = ndimage.label(true_et)

            pred_A = (p_map > 0.5) & brain_mask
            uncertain = (p_map < TAU_P) & brain_mask
            admitted = apply_gate(r_map, uncertain, cfg)
            pred_final = pred_A | admitted

            tp += int((pred_final & true_et).sum())
            fp += int((pred_final & ~true_et).sum())
            fn += int((~pred_final & true_et & brain_mask).sum())
            tn += int((~pred_final & ~true_et & brain_mask).sum())

            for rec in lesion_by_sid.get(sid, []):
                if rec['phenotype'] != 'G1_confidently_missed':
                    continue
                cid = rec['comp_id']
                if cid < 1 or cid > et_n:
                    continue
                lm = et_lbl == cid
                g1_vals.append(int((pred_final & lm).any()))

        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float('nan')
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')
        g1_sens = float(np.mean(g1_vals)) if g1_vals else float('nan')
        results_p1.append({'seed': seed, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                           'precision': precision, 'recall': recall, 'dice': dice, 'iou': iou,
                           'g1_sensitivity': g1_sens})
        print(f'  seed={seed}: Dice={dice:.4f} Precision={precision:.4f} Recall={recall:.4f} '
             f'IoU={iou:.4f} G1={g1_sens*100:.1f}% ({time.time()-t0_time:.0f}s)', flush=True)

    # also compute production-alone on the SAME protocol for direct reference
    tp = fp = fn = 0
    for sid in test_subjects_eval:
        maps, tgt_c, img_c = dense_eval(sid)
        p_map = maps['prod']
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        pred_A = (p_map > 0.5) & brain_mask
        tp += int((pred_A & true_et).sum())
        fp += int((pred_A & ~true_et).sum())
        fn += int((~pred_A & true_et & brain_mask).sum())
    dice_prod_p1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float('nan')
    print(f'  production alone (protocol 1): Dice={dice_prod_p1:.4f}', flush=True)

    out1 = HERE / ('E289B_protocol1_smoke.csv' if smoke else 'E289B_protocol1.csv')
    with open(out1, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['seed', 'TP', 'FP', 'FN', 'TN', 'precision', 'recall',
                                           'dice', 'iou', 'g1_sensitivity'])
        w.writeheader()
        for row in results_p1:
            w.writerow(row)
    print(f'  wrote {out1.name}', flush=True)

    # ---- PROTOCOL 2: sliding-window + per-subject (E288's convention) ----
    print('\n--- PROTOCOL 2: sliding-window + per-subject averaging (E288 convention) ---', flush=True)
    amp = True
    results_p2 = []
    for seed in seeds:
        w_mrd_t = torch.from_numpy(fitted[seed][0]).float().to(dev)
        b_mrd = fitted[seed][1]
        dice_A, dice_gated = [], []
        g1_vals_sw = []
        for i, sid in enumerate(test_subjects_eval):
            idx = sid_to_idx[sid]
            image, target, _ = ds_full._load_subject(ds_full.subject_dirs[idx])
            img_t = torch.from_numpy(image).unsqueeze(0).float().to(dev)
            probs, mrd_score = sliding_window_predict_with_mrd(model, img_t, PATCH, SW_OVERLAP,
                                                                dev, amp, w_mrd_t, b_mrd)
            pred_A = (probs >= 0.5).astype(np.float32)
            dA = dice_per_region(pred_A, target)
            dice_A.append(dA)

            p_et = probs[ET]
            uncertain = p_et < TAU_P
            admitted = apply_gate(mrd_score, uncertain, cfg)
            pred_G = pred_A.copy()
            pred_G[ET] = np.logical_or(pred_A[ET] > 0.5, admitted).astype(np.float32)
            dG = dice_per_region(pred_G, target)
            dice_gated.append(dG)

            true_et = target[ET] > 0.5
            et_lbl, et_n = ndimage.label(true_et)
            for rec in lesion_by_sid.get(sid, []):
                if rec['phenotype'] != 'G1_confidently_missed':
                    continue
                cid = rec['comp_id']
                if cid < 1 or cid > et_n:
                    continue
                lm = et_lbl == cid
                g1_vals_sw.append(int((pred_G[ET].astype(bool) & lm).any()))

            if (i + 1) % 20 == 0 or smoke:
                print(f'  seed={seed} {i+1}/{len(test_subjects_eval)} ({time.time()-t0_time:.0f}s)',
                     flush=True)

        dice_A = np.array(dice_A); dice_gated = np.array(dice_gated)
        results_p2.append({
            'seed': seed, 'condition': 'A_production_alone', 'dice_ET': float(dice_A[:, ET].mean()),
            'dice_TC': float(dice_A[:, TC].mean()), 'dice_WT': float(dice_A[:, WT].mean()),
            'dice_mean': float(dice_A.mean()), 'g1_sensitivity': float('nan'),
        })
        results_p2.append({
            'seed': seed, 'condition': 'gated_mrd', 'dice_ET': float(dice_gated[:, ET].mean()),
            'dice_TC': float(dice_gated[:, TC].mean()), 'dice_WT': float(dice_gated[:, WT].mean()),
            'dice_mean': float(dice_gated.mean()),
            'g1_sensitivity': float(np.mean(g1_vals_sw)) if g1_vals_sw else float('nan'),
        })
        print(f'  seed={seed}: A ET={dice_A[:,ET].mean():.4f}  |  gated ET={dice_gated[:,ET].mean():.4f} '
             f'G1={np.mean(g1_vals_sw)*100 if g1_vals_sw else float("nan"):.1f}%', flush=True)

    out2 = HERE / ('E289B_protocol2_smoke.csv' if smoke else 'E289B_protocol2.csv')
    with open(out2, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['seed', 'condition', 'dice_ET', 'dice_TC', 'dice_WT',
                                           'dice_mean', 'g1_sensitivity'])
        w.writeheader()
        for row in results_p2:
            w.writerow(row)
    print(f'  wrote {out2.name}', flush=True)

    print(f'\nE289-B complete ({time.time()-t0_time:.0f}s).', flush=True)


if __name__ == '__main__':
    main()
