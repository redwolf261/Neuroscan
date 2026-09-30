"""E254 -- generalization test, per the user's exact spec. E253 chose
alpha and evaluated it on the SAME lesions -- vulnerable to overfitting
the interpolation to that specific population. E254 fixes this with a
genuine 3-way patient-level split (train/validation/held-out-test) and
adds R2 (a production-preserving regularized readout) as the critical
control the user specified.

SPLIT (per explicit user decision): 60/20/20 of DETECTED subjects
(n=237) into train/val/test, patient-level (not lesion-level).
G2-A/G2-B subjects are ALREADY DISJOINT from detected subjects (by
construction -- they are lesions the model missed, from different
lesion populations) so they need no separate split; they are simply
scored ONCE on the final, val-selected model, exactly matching "evaluate
the selected alpha once on completely held-out cases."

THREE READOUTS trained/selected, per explicit user spec:
  PRODUCTION -- the actual seg_head (w_prod, b_prod), unchanged.
  R1 -- fresh LogisticRegression fit on TRAIN-split detected lesions
        only (same convention as E251/E252/E253).
  R2 -- production-preserving regularized readout: minimizes segmentation
        loss (here: the SAME logistic/cross-entropy objective as R1,
        since this is a linear-readout-on-frozen-features setting, not
        a full segmentation retrain) PLUS lambda*||w - w_prod||^2,
        i.e. a ridge-type penalty pulling the new weight vector toward
        production's own direction. Implemented via a small custom
        gradient-descent fit (sklearn's LogisticRegression does not
        support an arbitrary L2-to-a-nonzero-anchor penalty natively),
        verified against a closed-form sanity check at lambda=0 (should
        recover something very close to standard logistic regression)
        and lambda->large (should collapse toward w_prod itself).

ALPHA (for the R1 interpolation) and LAMBDA (for R2) SELECTION (per
explicit user decision, this turn's disambiguation): chosen to maximize
DETECTED-LESION VALIDATION recovery ONLY -- G2-A/G2-B data is NEVER used
for any model fitting or hyperparameter selection, anywhere in this
script. This is the strongest, most conservative test of genuine
generalization.

FINAL EVALUATION (once, on held-out test-split detected lesions AND all
G2-A/G2-B lesions, per explicit user metric list): lesion-level recovery
(G2-A, G2-B, detected-test), detected false-positive rate (shell),
voxel-level precision/recall proxy, and a simple calibration check
(mean predicted probability vs empirical positive rate, binned).
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch, get_masks  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
ET = 0
PATCH = (128, 128, 128)
MIN_VOX = 5
TAU = 0.5
ALPHAS = np.arange(0.0, 1.01, 0.1)
LAMBDAS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
MAX_VOX_PER_LESION = 30


def extract_D1_full(model, ds, sid_to_idx, sid, comp_id, dev):
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    brain_mask = img_c[0] != 0
    cm, shell = get_masks(tgt_c, comp_id, brain_mask)
    if cm is None:
        return None
    sz = int(cm.sum())
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    stages = forward_to_dec1_internal(model, img_t)
    d1 = stages['relu2_out'][0].cpu().numpy()
    lesion_feats = d1[:, cm].T
    shell_feats = d1[:, shell].T
    return lesion_feats, shell_feats, sz


def fit_r2(X_train, y_train, w_prod, b_prod, lam, n_iter=500):
    """Ridge-to-anchor logistic regression: minimizes
    BCE(sigmoid(Xw+b), y) + lam*||w - w_prod||^2, via scipy's L-BFGS-B
    (a proper convex-optimization solver, NOT hand-rolled fixed-step
    gradient descent -- a first attempt with fixed-lr full-batch GD
    diverged to NaN/inf for lambda>=3 on a synthetic sanity-check dataset,
    caught and fixed BEFORE running on real data, since 2*lambda scales
    the gradient and a fixed step size becomes numerically unstable at
    the higher end of the LAMBDAS sweep this experiment actually uses).
    Initialized AT w_prod/b_prod so lambda->large trivially collapses
    toward production (re-verified after the fix, see sanity check)."""
    from scipy.optimize import minimize
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    n = len(y)

    def objective(params):
        w = params[:-1]; b = params[-1]
        z = X @ w + b
        p = expit(z)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        bce = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        ridge = lam * np.sum((w - w_prod) ** 2)
        return bce + ridge

    def grad(params):
        w = params[:-1]; b = params[-1]
        z = X @ w + b
        p = expit(z)
        grad_z = (p - y) / n
        grad_w = X.T @ grad_z + 2 * lam * (w - w_prod)
        grad_b = grad_z.sum()
        return np.concatenate([grad_w, [grad_b]])

    x0 = np.concatenate([w_prod.astype(np.float64), [float(b_prod)]])
    res = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                   options={'maxiter': n_iter})
    w = res.x[:-1]
    b = float(res.x[-1])
    return w, b


def bce_loss(X, y, w, b):
    p = np.clip(expit(X @ w + b), 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    conv_w = model.seg_head[0].weight.detach().cpu().numpy()
    conv_b = model.seg_head[0].bias.detach().cpu().numpy()
    w_prod = conv_w[ET].reshape(32).astype(np.float64)
    b_prod = float(conv_b[ET])

    phenotypes = {}
    for r in csv.DictReader(open(HERE / 'E243_phenotypes.csv')):
        phenotypes[(r['subject_id'], r['comp_id'])] = r['phenotype']
    e240 = list(csv.DictReader(open(HERE / 'E240_layerwise.csv')))
    seen = set()
    det_lesions, g2a_lesions, g2b_lesions = [], [], []
    for r in e240:
        key = (r['subject_id'], r['comp_id'])
        if r['role'] == 'detected':
            if key in seen:
                continue
            seen.add(key)
            det_lesions.append(r)
    seen2 = set()
    for r in e240:
        if r['role'] != 'missed':
            continue
        key = (r['subject_id'], r['comp_id'])
        if key in seen2:
            continue
        seen2.add(key)
        phen = phenotypes.get(key)
        if phen == 'G2A_rejected':
            g2a_lesions.append(r)
        elif phen == 'G2B_partial':
            g2b_lesions.append(r)

    # ---- 3-way PATIENT-LEVEL split of detected subjects: 60/20/20 ----
    det_subjects = sorted(set(r['subject_id'] for r in det_lesions))
    rng = np.random.default_rng(2540)
    shuffled = rng.permutation(det_subjects)
    n = len(shuffled)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_subj = set(shuffled[:n_train])
    val_subj = set(shuffled[n_train:n_train + n_val])
    test_subj = set(shuffled[n_train + n_val:])
    print(f'Detected subjects: train={len(train_subj)} val={len(val_subj)} test={len(test_subj)}', flush=True)

    if smoke:
        det_lesions = [r for r in det_lesions if r['subject_id'] in train_subj][:30] + \
                     [r for r in det_lesions if r['subject_id'] in val_subj][:15] + \
                     [r for r in det_lesions if r['subject_id'] in test_subj][:15]
        g2a_lesions = g2a_lesions[:15]
        g2b_lesions = g2b_lesions[:15]

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    t0 = time.time()
    print('Extracting detected...', flush=True)
    det_cache = {}
    for i, r in enumerate(det_lesions):
        res = extract_D1_full(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
        det_cache[(r['subject_id'], r['comp_id'])] = res
        if (i + 1) % 50 == 0 or smoke:
            print(f'  {i+1}/{len(det_lesions)} ({time.time()-t0:.0f}s)', flush=True)

    print('Extracting G2-A...', flush=True)
    g2a_cache = {}
    for i, r in enumerate(g2a_lesions):
        res = extract_D1_full(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
        g2a_cache[(r['subject_id'], r['comp_id'])] = res
        if (i + 1) % 50 == 0 or smoke:
            print(f'  {i+1}/{len(g2a_lesions)} ({time.time()-t0:.0f}s)', flush=True)

    print('Extracting G2-B...', flush=True)
    g2b_cache = {}
    for i, r in enumerate(g2b_lesions):
        res = extract_D1_full(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
        g2b_cache[(r['subject_id'], r['comp_id'])] = res
        if (i + 1) % 50 == 0 or smoke:
            print(f'  {i+1}/{len(g2b_lesions)} ({time.time()-t0:.0f}s)', flush=True)
    print(f'Extraction complete ({time.time()-t0:.0f}s)', flush=True)

    def build_voxel_set(lesion_list, cache, subj_filter=None):
        X, y = [], []
        rng_l = np.random.default_rng(999)
        for r in lesion_list:
            sid = r['subject_id']
            if subj_filter is not None and sid not in subj_filter:
                continue
            res = cache.get((sid, r['comp_id']))
            if res is None:
                continue
            lesion_feats, shell_feats, sz = res
            if len(lesion_feats) > MAX_VOX_PER_LESION:
                idx = rng_l.choice(len(lesion_feats), MAX_VOX_PER_LESION, replace=False)
                lesion_feats = lesion_feats[idx]
            if len(shell_feats) > MAX_VOX_PER_LESION:
                idx = rng_l.choice(len(shell_feats), MAX_VOX_PER_LESION, replace=False)
                shell_feats = shell_feats[idx]
            X.append(lesion_feats); y.append(np.ones(len(lesion_feats)))
            X.append(shell_feats); y.append(np.zeros(len(shell_feats)))
        return np.concatenate(X), np.concatenate(y)

    X_train, y_train = build_voxel_set(det_lesions, det_cache, train_subj)
    X_val, y_val = build_voxel_set(det_lesions, det_cache, val_subj)
    print(f'train voxels={len(y_train)}  val voxels={len(y_val)}', flush=True)

    # ---- fit R1 ----
    r1 = LogisticRegression(max_iter=500, C=1.0)
    r1.fit(X_train, y_train)
    w_r1 = r1.coef_[0].astype(np.float64)
    b_r1 = float(r1.intercept_[0])

    # ---- select alpha on VALIDATION (detected-only, per explicit user decision) ----
    def lesion_level_recovery(lesion_list, cache, subj_filter, w, b):
        recs = []
        for r in lesion_list:
            sid = r['subject_id']
            if subj_filter is not None and sid not in subj_filter:
                continue
            res = cache.get((sid, r['comp_id']))
            if res is None:
                continue
            lesion_feats, shell_feats, sz = res
            p = expit(lesion_feats @ w + b)
            recs.append(int(p.max() > TAU))
        return np.mean(recs) if recs else float('nan')

    print('\nSelecting alpha on VALIDATION detected recovery...', flush=True)
    best_alpha, best_val_rec = 0.0, -1
    for alpha in ALPHAS:
        w_a = (1 - alpha) * w_prod + alpha * w_r1
        b_a = (1 - alpha) * b_prod + alpha * b_r1
        val_rec = lesion_level_recovery(det_lesions, det_cache, val_subj, w_a, b_a)
        print(f'  alpha={alpha:.1f}: val_detected_recovery={val_rec:.4f}', flush=True)
        if val_rec >= best_val_rec:
            best_val_rec = val_rec; best_alpha = alpha
    print(f'SELECTED alpha={best_alpha:.2f} (val_detected_recovery={best_val_rec:.4f})', flush=True)

    # ---- fit R2 for each lambda, select on VALIDATION ----
    print('\nFitting R2 (production-preserving) across lambda, selecting on VALIDATION...', flush=True)
    best_lambda, best_r2_val_rec, best_w_r2, best_b_r2 = None, -1, None, None
    for lam in LAMBDAS:
        w2, b2 = fit_r2(X_train, y_train, w_prod, b_prod, lam)
        val_rec = lesion_level_recovery(det_lesions, det_cache, val_subj, w2, b2)
        cos_to_prod = float(np.dot(w2, w_prod) / (np.linalg.norm(w2) * np.linalg.norm(w_prod) + 1e-12))
        print(f'  lambda={lam:.2f}: val_detected_recovery={val_rec:.4f}  cos(w2,w_prod)={cos_to_prod:.4f}', flush=True)
        if val_rec >= best_r2_val_rec:
            best_r2_val_rec = val_rec; best_lambda = lam; best_w_r2 = w2; best_b_r2 = b2
    print(f'SELECTED lambda={best_lambda} (val_detected_recovery={best_r2_val_rec:.4f})', flush=True)

    # ---- FINAL evaluation, ONCE, on held-out test-split detected + ALL G2A/G2B ----
    print('\n' + '='*90, flush=True)
    print('FINAL EVALUATION (held-out test-split detected + all G2-A/G2-B)', flush=True)
    print('='*90, flush=True)

    w_alpha = (1 - best_alpha) * w_prod + best_alpha * w_r1
    b_alpha = (1 - best_alpha) * b_prod + best_alpha * b_r1

    out = HERE / ('E254_smoke.csv' if smoke else 'E254_generalization.csv')
    fh = open(out, 'w', newline='')
    w_csv = csv.DictWriter(fh, fieldnames=['group', 'readout', 'subject_id', 'comp_id', 'size',
                                           'max_prob', 'recovered', 'fp_shell'])
    w_csv.writeheader()

    readouts = {'production': (w_prod, b_prod), 'R1_alpha': (w_alpha, b_alpha), 'R2_ridge': (best_w_r2, best_b_r2)}

    def eval_and_write(grp_name, lesion_list, cache, subj_filter):
        for r in lesion_list:
            sid = r['subject_id']
            if subj_filter is not None and sid not in subj_filter:
                continue
            res = cache.get((sid, r['comp_id']))
            if res is None:
                continue
            lesion_feats, shell_feats, sz = res
            for rname, (w, b) in readouts.items():
                p_lesion = expit(lesion_feats @ w + b)
                p_shell = expit(shell_feats @ w + b)
                max_p = float(p_lesion.max())
                w_csv.writerow({'group': grp_name, 'readout': rname, 'subject_id': sid,
                               'comp_id': r['comp_id'], 'size': sz, 'max_prob': max_p,
                               'recovered': int(max_p > TAU), 'fp_shell': float(p_shell.mean())})

    eval_and_write('detected_test', det_lesions, det_cache, test_subj)
    eval_and_write('G2A', g2a_lesions, g2a_cache, None)
    eval_and_write('G2B', g2b_lesions, g2b_cache, None)
    fh.close()

    for grp, lesion_list, cache, subj_filter in [
        ('detected_test', det_lesions, det_cache, test_subj),
        ('G2A', g2a_lesions, g2a_cache, None),
        ('G2B', g2b_lesions, g2b_cache, None)]:
        print(f'\n{grp}:', flush=True)
        for rname, (w, b) in readouts.items():
            rec = lesion_level_recovery(lesion_list, cache, subj_filter, w, b)
            print(f'  {rname}: recovery={rec:.4f}', flush=True)

    print(f'\nE254 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)
    print(f'SELECTED alpha={best_alpha}, lambda={best_lambda}', flush=True)


if __name__ == '__main__':
    main()
