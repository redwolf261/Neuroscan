"""E289-A -- MRD Precision-Gated Integration, DEVELOPMENT SWEEP, per
the user's exact spec. Tests whether SELECTING which MRD voxels/
components are admitted into the production mask (rather than taking
the raw union, E287's own result) can convert MRD's lesion-recovery
signal into a net ET-Dice improvement. Does NOT touch the frozen MRD
formula (T2=(y-p)/(1-p+eps), raw r inference) -- only the ADMISSION
RULE for which MRD-positive voxels get unioned with production.

SCOPE (per explicit user framing): "test increasingly informative
gates rather than jumping directly to a complicated learned model."
This script runs the GATE LADDER (G0-G5) on the DEVELOPMENT cohort
(det_test, E274-E285's own 67-subject population -- explicitly NOT
E286-E288's 125-subject locked test cohort, per explicit user
requirement "no threshold tuning... after seeing locked-test
results"), using E286/E287's FAST center-crop + pooled-voxel protocol
(not E288's slow sliding-window protocol -- per explicit user scoping
this session, since gates are compared to EACH OTHER within the same
protocol, so the protocol choice does not bias which gate wins; the
final selected gate is re-evaluated on the LOCKED cohort under BOTH
protocols in e289b).

GATE LADDER:
  G0 raw MRD (tau_R only)          = E287's own condition C, the reference
  G1 tau_R threshold sweep          = multiple tau_R values (NOT a new
                                      gate type, a parameter sweep of G0)
  G2 component-size gate            |C| >= k, for k in a sweep
  G4 component-confidence gate      mean(R_C) >= gamma, for gamma in a sweep
  G5 combined size+confidence gate  |C|>=k AND mean(R_C)>=gamma, for
                                     (k,gamma) pairs built from G2/G4's
                                     own individually-reasonable values
  (G3, the production-proximity gate, is EXPLICITLY NOT run as a primary
  candidate per the user's own text: "If the goal is genuinely to
  recover completely missed lesions, a strict proximity requirement
  could eliminate the very phenomenon MRD was designed to recover... I
  would not make this the primary proposed gate." Skipped per that
  explicit reasoning, not an oversight.)

PRE-REGISTERED SELECTION RULE (frozen BEFORE running, per explicit
user requirement and AskUserQuestion resolution this session):
  R_min = 15% (G1 recovery floor on the DEVELOPMENT cohort, matching
  the user's own worked example in their proposal). Among all gate
  configurations achieving G1 recovery >= R_min on det_test, select
  the ONE with the HIGHEST Dice. That gate's EXACT parameters (tau_R,
  k, gamma as applicable) are then frozen and carried into e289b's
  locked-cohort evaluation, unchanged.
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
EPS_MARG = 1e-6
TAU_LOW, TAU_HIGH = TAU, 0.9
SEEDS = [999, 4242, 7, 123, 2024]
STRUCT = ndimage.generate_binary_structure(3, 1)
TAU_P = 0.5  # production-uncertain threshold, = TAU, unchanged from E287
R_MIN = 0.15  # FROZEN pre-registered G1-recovery floor on the dev cohort

# Gate parameter grids (dev-cohort sweep only)
TAU_R_GRID = [0.85, 0.90, 0.922799970717367, 0.95082877089122, 0.969022838532718,
             0.980622037447513, 0.987932080391847, 0.995353985151741]
K_GRID = [1, 2, 4, 8, 16, 32, 64]
GAMMA_GRID = [0.90, 0.95, 0.97, 0.99]


def fit_soft_target(X, y_soft, max_iter=500, C=1.0):
    X2 = np.concatenate([X, X], axis=0)
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w2 = np.concatenate([y_soft, 1.0 - y_soft])
    keep = w2 > 1e-12
    X2, y2, w2 = X2[keep], y2[keep], w2[keep]
    clf = LogisticRegression(max_iter=max_iter, C=C, class_weight='balanced')
    clf.fit(X2, y2, sample_weight=w2)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])


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

    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

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
        lesion_pool = np.concatenate(lesion_pool).astype(np.float64)
        neg_pool = np.concatenate(neg_pool).astype(np.float64)
        return lesion_pool, neg_pool

    print('Fitting MRD-T2 on det_train, 5 seeds...', flush=True)
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
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
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

    test_subjects = sorted(splits['det_test'])
    if smoke:
        test_subjects = test_subjects[:15]

    print(f'\nBuilding G1/G2/G3 phenotype groups on {len(test_subjects)} DEV (det_test) subjects...',
         flush=True)
    lesion_records = []
    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
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
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0_time:.0f}s)', flush=True)
    lesion_by_sid = {}
    for rec in lesion_records:
        lesion_by_sid.setdefault(rec['subject_id'], []).append(rec)
    n_by_phen = {}
    for rec in lesion_records:
        n_by_phen[rec['phenotype']] = n_by_phen.get(rec['phenotype'], 0) + 1
    print(f'  phenotype group sizes: {n_by_phen}', flush=True)

    # ---- build the full gate configuration list ----
    configs = []
    for tau_r in TAU_R_GRID:
        configs.append({'gate': 'G0_G1_raw', 'tau_R': tau_r, 'k': None, 'gamma': None})
    for tau_r in [0.95082877089122]:  # hold tau_R at E287's own value for the size/confidence sweeps
        for k in K_GRID:
            configs.append({'gate': 'G2_component_size', 'tau_R': tau_r, 'k': k, 'gamma': None})
        for gamma in GAMMA_GRID:
            configs.append({'gate': 'G4_confidence', 'tau_R': tau_r, 'k': None, 'gamma': gamma})
        for k in [4, 8, 16]:
            for gamma in [0.95, 0.97]:
                configs.append({'gate': 'G5_combined', 'tau_R': tau_r, 'k': k, 'gamma': gamma})
    if smoke:
        configs = configs[:6]
    print(f'\nEvaluating {len(configs)} gate configurations x {len(seeds)} seeds on dev cohort...',
         flush=True)

    def apply_gate(r_map, uncertain, cfg):
        """Returns boolean mask of ADMITTED MRD voxels (already restricted
        to the uncertain region, matching Section 3's M_G definition)."""
        mrd_pos = (r_map > cfg['tau_R']) & uncertain
        if cfg['gate'] in ('G0_G1_raw',):
            return mrd_pos
        cc_lbl, cc_n = ndimage.label(mrd_pos, structure=STRUCT)
        if cc_n == 0:
            return mrd_pos
        admit = np.zeros(cc_n + 1, dtype=bool)
        sizes = ndimage.sum(mrd_pos, cc_lbl, index=np.arange(1, cc_n + 1))
        if cfg['gate'] == 'G2_component_size':
            ok = sizes >= cfg['k']
            admit[1:] = ok
        elif cfg['gate'] == 'G4_confidence':
            means = ndimage.mean(r_map, cc_lbl, index=np.arange(1, cc_n + 1))
            admit[1:] = means >= cfg['gamma']
        elif cfg['gate'] == 'G5_combined':
            means = ndimage.mean(r_map, cc_lbl, index=np.arange(1, cc_n + 1))
            admit[1:] = (sizes >= cfg['k']) & (means >= cfg['gamma'])
        return admit[cc_lbl]

    results = []
    for ci, cfg in enumerate(configs):
        dice_per_seed, g1_per_seed = [], []
        for seed in seeds:
            tp = fp = fn = tn = 0
            g1_vals = []
            for sid in test_subjects:
                if sid not in sid_to_idx:
                    continue
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

            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float('nan')
            dice_per_seed.append(dice)
            g1_per_seed.append(float(np.mean(g1_vals)) if g1_vals else float('nan'))

        results.append({
            'config_id': ci, 'gate': cfg['gate'], 'tau_R': cfg['tau_R'], 'k': cfg['k'],
            'gamma': cfg['gamma'], 'dice_mean': float(np.mean(dice_per_seed)),
            'dice_sd': float(np.std(dice_per_seed)), 'g1_recovery_mean': float(np.mean(g1_per_seed)),
            'g1_recovery_sd': float(np.std(g1_per_seed)),
        })
        print(f'  [{ci+1}/{len(configs)}] {cfg["gate"]:20s} tau_R={cfg["tau_R"]:.4f} k={cfg["k"]} '
             f'gamma={cfg["gamma"]}: Dice={np.mean(dice_per_seed):.4f} '
             f'G1={np.mean(g1_per_seed)*100:.1f}% ({time.time()-t0_time:.0f}s)', flush=True)

    out = HERE / ('E289A_dev_sweep_smoke.csv' if smoke else 'E289A_dev_sweep.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['config_id', 'gate', 'tau_R', 'k', 'gamma', 'dice_mean',
                                           'dice_sd', 'g1_recovery_mean', 'g1_recovery_sd'])
        w.writeheader()
        for row in results:
            w.writerow(row)
    print(f'\nwrote {out.name}', flush=True)

    # ---- also compute production-alone Dice/G1 on dev cohort for reference ----
    tp = fp = fn = 0
    for sid in test_subjects:
        if sid not in sid_to_idx:
            continue
        maps, tgt_c, img_c = dense_eval(sid)
        p_map = maps['prod']
        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        pred_A = (p_map > 0.5) & brain_mask
        tp += int((pred_A & true_et).sum())
        fp += int((pred_A & ~true_et).sum())
        fn += int((~pred_A & true_et & brain_mask).sum())
    dice_prod = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float('nan')
    print(f'\nProduction alone on dev cohort: Dice={dice_prod:.4f} G1=0.0% (by definition)', flush=True)

    # ---- PRE-REGISTERED SELECTION: among configs with G1 >= R_MIN, pick highest Dice ----
    print('\n' + '=' * 70, flush=True)
    print(f'PRE-REGISTERED SELECTION: R_min={R_MIN*100:.0f}%, among qualifying configs pick max Dice',
         flush=True)
    print('=' * 70, flush=True)
    qualifying = [r for r in results if r['g1_recovery_mean'] >= R_MIN]
    print(f'  {len(qualifying)}/{len(results)} configs meet G1 recovery >= {R_MIN*100:.0f}%', flush=True)
    if qualifying:
        best = max(qualifying, key=lambda r: r['dice_mean'])
        print(f'  SELECTED: {best}', flush=True)
        sel_out = HERE / ('E289A_selected_gate_smoke.json' if smoke else 'E289A_selected_gate.json')
        import json
        with open(sel_out, 'w') as fh:
            json.dump({'production_dev_dice': dice_prod, 'R_min': R_MIN, 'selected_config': best}, fh, indent=2)
        print(f'  wrote {sel_out.name}', flush=True)
    else:
        print('  NO CONFIG MEETS R_min -- selection FAILED, report honestly, do not lower R_min post-hoc',
             flush=True)

    print(f'\nE289-A complete ({time.time()-t0_time:.0f}s).', flush=True)


if __name__ == '__main__':
    main()
