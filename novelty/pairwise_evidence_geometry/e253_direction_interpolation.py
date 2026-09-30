"""E253 -- feature-direction hypothesis test, per the user's exact spec.
Two parts:

PART A -- per-channel contribution decomposition: c_j = w_j * z_j for
w_prod and w_R1, comparing G2-A vs detected, to identify WHICH D1
channels drive the production head's rejection of G2-A and which
channels R1 relies on instead. Includes the explicit E248 connection
test: does w_prod disproportionately weight E248's fixed protected-
channel set {30,27,20,25,15,12,4,16}?

PART B -- counterfactual interpolation sweep (the decisive causal test):
  w(alpha) = (1-alpha)*w_prod + alpha*w_R1
  b(alpha) = (1-alpha)*b_prod + alpha*b_R1
  alpha in {0, 0.1, 0.2, ..., 1.0}
Measures G2-A recovery, detected recovery, G2-B recovery, and a false-
positive proxy (mean probability in background shell) at EVERY alpha,
using the model's REAL, UNCHANGED forward pass up through D1 (=relu2_out)
-- only the FINAL linear+sigmoid readout is replaced with w(alpha)/b(alpha),
everything upstream is the frozen production model, exactly as E251/E252
computed D1. This is NOT a retrained model -- pure post-hoc interpolation
of two ALREADY-FIT linear readouts, applied to the SAME frozen D1
features.

Per explicit user framing: if recovery rises smoothly toward R1's alpha
WITHOUT comparable degradation on detected, that is strong evidence for
"wrong learned direction, fixable by moving along the prod-to-R1 axis"
(possibility A). If only alpha=1 (full R1) works, or if detected/G2-B
degrade just as fast as G2-A improves, that points to interactions the
interpolation can't capture (possibility B).
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
N_FOLDS = 5
TAU = 0.5
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TARGET_CHANNELS = [30, 27, 20, 25, 15, 12, 4, 16]  # E248's protected set


def extract_D1_full(model, ds, sid_to_idx, sid, comp_id, dev):
    """Returns (d1_lesion_voxels (n,32), d1_shell_voxels (n,32), size) --
    FULL voxel set this time (not capped), needed for accurate alpha-sweep
    recovery/FP measurement, not just probe training."""
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
    d1 = stages['relu2_out'][0].cpu().numpy()  # (32,D,H,W)
    lesion_feats = d1[:, cm].T
    shell_feats = d1[:, shell].T
    return lesion_feats, shell_feats, sz


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
    w_prod = conv_w[ET].reshape(32)
    b_prod = float(conv_b[ET])

    # ---- PART A: E248 connection -- does w_prod over-weight the protected set? ----
    print("="*90)
    print("PART A: w_prod magnitude vs E248's protected-channel set")
    print("="*90)
    d248 = np.load(HERE / 'E248_protection.npz')
    det_pos = d248['detected'].mean(axis=0)  # (32,) E248's own protection score
    abs_w_prod = np.abs(w_prod)
    is_target = np.array([1 if c in TARGET_CHANNELS else 0 for c in range(32)])
    target_mean_w = abs_w_prod[is_target == 1].mean()
    nontarget_mean_w = abs_w_prod[is_target == 0].mean()
    print(f"  mean |w_prod| on E248's 8 protected channels: {target_mean_w:.4f}")
    print(f"  mean |w_prod| on remaining 24 channels: {nontarget_mean_w:.4f}")
    from scipy.stats import pearsonr, spearmanr
    r_p, p_p = pearsonr(det_pos, abs_w_prod)
    r_s, p_s = spearmanr(det_pos, abs_w_prod)
    print(f"  corr(E248 protection score, |w_prod|): Pearson r={r_p:.4f} p={p_p:.4e}  "
          f"Spearman r={r_s:.4f} p={p_s:.4e}")
    print(f"  (per-channel table)")
    order = np.argsort(-abs_w_prod)
    for c in order:
        flag = ' <- E248 TARGET' if c in TARGET_CHANNELS else ''
        print(f"    ch{c:2d}: |w_prod|={abs_w_prod[c]:.4f}  E248_protection={det_pos[c]:.4f}{flag}")

    # ---- PART B: interpolation sweep ----
    print("\n" + "="*90)
    print("PART B: extracting D1 features for interpolation sweep")
    print("="*90)

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

    print(f'detected={len(det_lesions)} G2A={len(g2a_lesions)} G2B={len(g2b_lesions)}', flush=True)
    if smoke:
        det_lesions = det_lesions[:40]
        g2a_lesions = g2a_lesions[:15]
        g2b_lesions = g2b_lesions[:15]

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    det_subjects = sorted(set(r['subject_id'] for r in det_lesions))
    rng = np.random.default_rng(2510)  # SAME seed as E251/E252
    shuffled = rng.permutation(det_subjects)
    fold_of = {sid: i % N_FOLDS for i, sid in enumerate(shuffled)}

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

    print(f'\nExtraction complete ({time.time()-t0:.0f}s)', flush=True)

    out = HERE / ('E253_smoke.csv' if smoke else 'E253_interpolation.csv')
    fh = open(out, 'w', newline='')
    w_csv = csv.DictWriter(fh, fieldnames=['group', 'subject_id', 'comp_id', 'fold', 'alpha',
                                           'max_prob', 'recovered', 'fp_shell'])
    w_csv.writeheader()

    for fold in range(N_FOLDS):
        train_X, train_y = [], []
        for r in det_lesions:
            sid = r['subject_id']
            if fold_of.get(sid) == fold:
                continue
            res = det_cache.get((sid, r['comp_id']))
            if res is None:
                continue
            lesion_feats, shell_feats, sz = res
            rng_l = np.random.default_rng(hash((r['subject_id'], r['comp_id'])) % (2**32))
            lf = lesion_feats if len(lesion_feats) <= 30 else lesion_feats[rng_l.choice(len(lesion_feats), 30, replace=False)]
            sf = shell_feats if len(shell_feats) <= 30 else shell_feats[rng_l.choice(len(shell_feats), 30, replace=False)]
            train_X.append(lf); train_y.append(np.ones(len(lf)))
            train_X.append(sf); train_y.append(np.zeros(len(sf)))
        if not train_X:
            continue
        X_train = np.concatenate(train_X); y_train = np.concatenate(train_y)
        r1 = LogisticRegression(max_iter=500, C=1.0)
        r1.fit(X_train, y_train)
        w_r1 = r1.coef_[0]
        b_r1 = float(r1.intercept_[0])

        def eval_group(grp_name, cache, lesions, subj_fold_lookup):
            for r in lesions:
                sid = r['subject_id']
                sid_fold = subj_fold_lookup.get(sid)
                if sid_fold is None:
                    sid_fold = abs(hash(sid)) % N_FOLDS
                if sid_fold != fold:
                    continue
                res = cache.get((sid, r['comp_id']))
                if res is None:
                    continue
                lesion_feats, shell_feats, sz = res
                for alpha in ALPHAS:
                    w_a = (1 - alpha) * w_prod + alpha * w_r1
                    b_a = (1 - alpha) * b_prod + alpha * b_r1
                    p_lesion = expit(lesion_feats @ w_a + b_a)
                    p_shell = expit(shell_feats @ w_a + b_a)
                    max_p = float(p_lesion.max())
                    w_csv.writerow({'group': grp_name, 'subject_id': sid, 'comp_id': r['comp_id'],
                                   'fold': fold, 'alpha': alpha, 'max_prob': max_p,
                                   'recovered': int(max_p > TAU), 'fp_shell': float(p_shell.mean())})

        eval_group('detected', det_cache, det_lesions, fold_of)
        eval_group('G2A', g2a_cache, g2a_lesions, fold_of)
        eval_group('G2B', g2b_cache, g2b_lesions, fold_of)
        fh.flush()
        print(f'  fold {fold} done ({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'\nE253 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
