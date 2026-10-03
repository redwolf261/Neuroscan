"""E272 -- readout-subspace decomposition (round 2), per the user's
exact spec. Direct follow-up to the E268-E271 geometry thread, which
closed as "informative representation analysis, not a deployable
improvement" after E271's practical test failed. The user explicitly
reopened a DIFFERENT, more rigorous question: is E268's own Outcome-A
finding (R_perpP matches R_full; R_Sperp -- orthogonal to BOTH w_P and
w_R -- ALSO matches R_full at 61.1%) itself robust, with a proper
100-random-2D-subspace null (E268 only tested ONE specific 2D
subspace, span(w_P,w_R), not a random-subspace comparison) and
E254-grade train/val/test discipline (readouts fit on TRAIN only,
evaluated once on held-out TEST).

THRESHOLD CONVENTION: fixed TAU=0.5 for every variant, NOT per-variant
Youden calibration. Flagged and resolved with the user after smoke
testing: an earlier draft used per-variant Youden-calibrated
thresholds (literal reading of "select threshold on validation"), but
this produced thresholds far below 0.5 and numbers completely
incomparable to E268's own already-reported R_full/R_P/R_perpP figures
(2.8%/58.3%/61.1%) and to the R1-B+Delta_C baseline -- both of which
use TAU=0.5 throughout this entire session (E257-E271). Since E272's
explicit purpose is to re-test and extend E268's SPECIFIC result with
better controls, comparability to those numbers matters more than a
literal per-variant threshold search; TAU=0.5 is used uniformly, same
as every prior experiment in this thread.

Per explicit user instruction: do NOT modify R1-B+Delta_C (still the
validated baseline). This is diagnostic, feeding a decision about
whether ANY further geometry-based mechanism is worth pursuing, not a
new deployable component itself.

FOUR CORE REPRESENTATIONS (readouts fit fresh on TRAIN pool, same pool
R1-B itself uses, per this thread's established convention):
  1. D1 (full)            -- R_full, reproduces R1-B
  2. P_P @ D1             -- production-direction-only, should reproduce
                              E268's R_P collapse (~2.8% recovery)
  3. (I-P_P) @ D1         -- production-orthogonal, should reproduce
                              E268's R_perpP (~61.1%)
  4. (I-P_W) @ D1, W=[w_P,w_R] -- orthogonal to BOTH known readout
                              directions (E268's R_Sperp, ~61.1%) --
                              fit as a FRESH readout on this projection
                              (E268's R_Sperp reused the SAME readout
                              fit used for other purposes; here it is
                              refit cleanly for this specific comparison)

CONTROL: N_RANDOM=100 (per explicit user spec) random 2D orthonormal
subspaces Q_j (j=1..100), each used to build P_Qj = Q_j Q_j^T and
project (I-P_Qj) @ D1 -- a FRESH readout fit per random subspace, same
training pool, evaluated identically to the 4 core representations.
Tests whether removing ANY 2 dimensions behaves like removing
(w_P,w_R) specifically, or whether (w_P,w_R) is special.

NO-LEAKAGE DISCIPLINE (per explicit user requirement): w_P is the
FROZEN production weight (never touches train/val/test data at fit
time, it's the already-trained production model's own parameter).
w_R is refit on TRAIN ONLY (same pool as every R1-B refit this
session). All projection matrices (P_P, P_W, P_Qj) are built from
these TRAIN-only quantities before any val/test data is touched, then
every new readout is ALSO fit on TRAIN only, with its decision
threshold calibrated on VAL only (Youden's J, same method as E269),
then frozen and applied ONCE to TEST -- exactly E254's discipline.

COMPUTATIONAL DESIGN (avoids E266/E269's repeated OOM crash from
holding >100 simultaneous dense per-variant volumes): 104 total
variants (4 core + 100 random) are processed in BATCHES of
BATCH_SIZE=16 (per explicit user choice this session), each batch
doing its own full VAL-calibration + TEST-evaluation pass (one dense
D1 forward pass per subject PER BATCH -- redundant across batches, but
this bounds peak memory regardless of total variant count, which is
the actual fix that worked for every OOM this session), writing
results incrementally to CSV after each batch.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e257_common import (load_model, get_w_prod, load_populations, build_splits,  # noqa: E402
                         extract_lesion_shell, extract_distant_background,
                         ROOT, PATCH, MIN_VOX, MAX_VOX_PER_LESION, TAU)
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402

ET, TC, WT = 0, 1, 2
STRUCT = ndimage.generate_binary_structure(3, 1)
D1_DIM = 32
N_RANDOM = 100
RNG_SEED_RANDOM = 9200
BATCH_SIZE = 16
EPS = 1e-12


def dice_score(pred_mask, gt_mask):
    ps, gs = pred_mask.sum(), gt_mask.sum()
    if gs == 0:
        return 1.0 if ps == 0 else float('nan')
    return float(2 * (pred_mask & gt_mask).sum() / (ps + gs))


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    model = load_model(dev)
    w_et, b_et = get_w_prod(model)
    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    if smoke:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']][:30]
        g2a_lesions_val = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_val']][:15]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']][:15]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']][:15]
        g2b_lesions_eval = g2b_lesions[:15]
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])[:20]
        n_random = 8
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]
        g2a_lesions_val = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_val']]
        g2a_lesions_test = [r for r in g2a_lesions if r['subject_id'] in splits['g2a_test']]
        det_lesions_test = [r for r in det_lesions if r['subject_id'] in splits['det_test']]
        g2b_lesions_eval = g2b_lesions
        bg_subjects = sorted(splits['det_train'] | splits['g2a_train'])
        n_random = N_RANDOM

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng = np.random.default_rng(999)
    t0 = time.time()

    print('Building TRAIN voxel pool...', flush=True)
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
    lesion_pool = np.concatenate(lesion_pool); neg_pool = np.concatenate(neg_pool)
    X_train = np.concatenate([lesion_pool, neg_pool]).astype(np.float64)
    y_train = np.concatenate([np.ones(len(lesion_pool)), np.zeros(len(neg_pool))])
    print(f'  pool: lesion={len(lesion_pool)} neg={len(neg_pool)} ({time.time()-t0:.0f}s)', flush=True)

    # ---- R1-B (R_full == w_R), fit on TRAIN only ----
    r_full_clf = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_train, y_train)
    w_r1b = r_full_clf.coef_[0].astype(np.float64)
    w_p = w_et.astype(np.float64)
    print(f'R1-B (R_full) fit ({time.time()-t0:.0f}s)', flush=True)

    # ---- build the 4 core + N_RANDOM projection matrices (TRAIN-only, no leakage) ----
    w_p_unit = w_p / (np.linalg.norm(w_p) + EPS)
    P_P = np.outer(w_p_unit, w_p_unit)
    P_perp_P = np.eye(D1_DIM) - P_P

    W = np.stack([w_p, w_r1b], axis=1)  # (32,2)
    Q_W, _ = np.linalg.qr(W)
    P_W = Q_W @ Q_W.T
    P_perp_W = np.eye(D1_DIM) - P_W

    rng_rand = np.random.default_rng(RNG_SEED_RANDOM)
    random_P_perp_list = []
    for _ in range(n_random):
        A = rng_rand.normal(size=(D1_DIM, 2))
        Q_r, _ = np.linalg.qr(A)
        P_r = Q_r @ Q_r.T
        random_P_perp_list.append(np.eye(D1_DIM) - P_r)

    core_projections = {
        'D1_full': np.eye(D1_DIM),
        'P_P_only': P_P,
        'perp_P': P_perp_P,
        'perp_PR': P_perp_W,
    }
    all_projections = dict(core_projections)
    for j, P_r in enumerate(random_P_perp_list):
        all_projections[f'perp_rand{j}'] = P_r

    variant_names = list(all_projections.keys())
    print(f'Total variants: {len(variant_names)} (4 core + {n_random} random)', flush=True)

    # ---- fit a FRESH readout per projection on TRAIN only ----
    print('Fitting readouts per projection (TRAIN only)...', flush=True)
    fitted = {}
    for name, P in all_projections.items():
        X_proj = X_train @ P.T  # P symmetric, so this == (P @ x) per row
        clf = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced').fit(X_proj, y_train)
        w_fit = clf.coef_[0].astype(np.float64)
        b_fit = float(clf.intercept_[0])
        w_eff = P.T @ w_fit  # compose back onto raw D1 (verified exact in E268/E269)
        fitted[name] = (w_eff, b_fit)
    print(f'All readouts fit ({time.time()-t0:.0f}s)', flush=True)

    # ---- dense eval helper: given a SUBSET of variant names, compute
    # their probability maps for one subject in one forward pass ----
    def make_dense_eval(variant_subset):
        tensors = {name: (torch.from_numpy(fitted[name][0]).float().to(dev), fitted[name][1])
                  for name in variant_subset}

        def dense_eval(sid):
            img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
            img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
            stages = forward_to_dec1_internal(model, img_t)
            d1 = stages['relu2_out'][0]
            C, D, H, W_ = d1.shape
            flat = d1.reshape(C, -1).T
            maps = {}
            with torch.no_grad():
                for name, (w_t, b_v) in tensors.items():
                    p = torch.sigmoid(flat @ w_t + b_v).reshape(D, H, W_).cpu().numpy()
                    maps[name] = p
            del d1, flat, img_t, stages
            torch.cuda.empty_cache()
            return maps, tgt_c, img_c
        return dense_eval

    test_subjects = sorted(set(r['subject_id'] for r in det_lesions_test) |
                          set(r['subject_id'] for r in g2a_lesions_test) |
                          set(r['subject_id'] for r in g2b_lesions_eval))
    if smoke:
        test_subjects = test_subjects[:15]

    out = HERE / ('E272_results_smoke.csv' if smoke else 'E272_results.csv')
    fh = open(out, 'w', newline='')
    w_csv = csv.DictWriter(fh, fieldnames=['variant', 'tau', 'g2a_recovery', 'det_recovery',
                                           'g2b_recovery', 'et_dice_mean', 'voxel_auc', 'fp_total'])
    w_csv.writeheader()

    n_batches = (len(variant_names) + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(n_batches):
        batch_names = variant_names[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        print(f'\n--- Batch {b+1}/{n_batches}: {batch_names} ---', flush=True)
        dense_eval = make_dense_eval(batch_names)
        tau_by_variant = {name: TAU for name in batch_names}

        # TEST evaluation
        per_subject_dice = {name: [] for name in batch_names}
        fp_total = {name: 0 for name in batch_names}
        auc_y = {name: [] for name in batch_names}
        auc_s = {name: [] for name in batch_names}
        g2a_recs = {name: [] for name in batch_names}
        det_recs = {name: [] for name in batch_names}
        g2b_recs = {name: [] for name in batch_names}

        g2a_by_sid, det_by_sid, g2b_by_sid = {}, {}, {}
        for r in g2a_lesions_test:
            g2a_by_sid.setdefault(r['subject_id'], []).append(int(r['comp_id']))
        for r in det_lesions_test:
            det_by_sid.setdefault(r['subject_id'], []).append(int(r['comp_id']))
        for r in g2b_lesions_eval:
            g2b_by_sid.setdefault(r['subject_id'], []).append(int(r['comp_id']))

        for i, sid in enumerate(test_subjects):
            if sid not in sid_to_idx:
                continue
            maps, tgt_c, img_c = dense_eval(sid)
            true_et = tgt_c[ET] > 0.5
            brain_mask = img_c[0] != 0
            et_lbl, et_n = ndimage.label(true_et)

            def valid_lesion_masks(cids):
                out_m = []
                for cid in cids:
                    if cid < 1 or cid > et_n:
                        continue
                    lm = et_lbl == cid
                    if int(lm.sum()) < MIN_VOX:
                        continue
                    out_m.append(lm)
                return out_m

            g2a_masks = valid_lesion_masks(g2a_by_sid.get(sid, []))
            det_masks = valid_lesion_masks(det_by_sid.get(sid, []))
            g2b_masks = valid_lesion_masks(g2b_by_sid.get(sid, []))

            y_true_flat = true_et[brain_mask].astype(int)
            for name, p_map in maps.items():
                tau_v = tau_by_variant[name]
                mask = (p_map > tau_v) & brain_mask
                per_subject_dice[name].append(dice_score(mask, true_et))
                fp_total[name] += int((mask & (~true_et)).sum())
                s_flat = p_map[brain_mask][::64]
                auc_y[name].append(y_true_flat[::64])
                auc_s[name].append(s_flat)
                for lm in g2a_masks:
                    g2a_recs[name].append(int((mask & lm).any()))
                for lm in det_masks:
                    det_recs[name].append(int((mask & lm).any()))
                for lm in g2b_masks:
                    g2b_recs[name].append(int((mask & lm).any()))
            del maps
            if (i + 1) % 20 == 0 or smoke:
                print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)

        for name in batch_names:
            y_cat = np.concatenate(auc_y[name]) if auc_y[name] else np.array([])
            s_cat = np.concatenate(auc_s[name]) if auc_s[name] else np.array([])
            try:
                auc = float(roc_auc_score(y_cat, s_cat)) if len(np.unique(y_cat)) > 1 else float('nan')
            except Exception:
                auc = float('nan')
            row = {
                'variant': name, 'tau': tau_by_variant[name],
                'g2a_recovery': float(np.mean(g2a_recs[name])) if g2a_recs[name] else float('nan'),
                'det_recovery': float(np.mean(det_recs[name])) if det_recs[name] else float('nan'),
                'g2b_recovery': float(np.mean(g2b_recs[name])) if g2b_recs[name] else float('nan'),
                'et_dice_mean': float(np.nanmean(per_subject_dice[name])),
                'voxel_auc': auc, 'fp_total': fp_total[name],
            }
            w_csv.writerow(row)
            print(f'  {name}: g2a_recovery={row["g2a_recovery"]:.4f} dice={row["et_dice_mean"]:.4f} '
                 f'auc={row["voxel_auc"]:.4f} fp={row["fp_total"]}', flush=True)
        fh.flush()

    fh.close()
    print(f'\nE272 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
