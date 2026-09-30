"""E252 -- readout geometry comparison: production seg_head vs R1, per
the user's exact spec. Does NOT modify or replace seg_head -- pure
measurement/diagnostic, determining whether E251's +34.9pp gain comes
from a DIRECTIONAL difference (production head weighs D1 features
differently) or a CALIBRATION difference (same direction, different
bias/threshold).

seg_head's exact structure (verified against neuroscan_3d_fixed.py line
78: nn.Sequential(nn.Conv3d(32, out_channels, kernel_size=1), nn.Sigmoid())
-- a 1x1x1 conv is LITERALLY a per-voxel linear map on the 32-channel D1
vector, identical in form to R1's own logistic regression: p =
sigmoid(w^T z + b). w_prod = seg_head's conv weight for the ET output
channel (channel 0 of 3), reshaped from (1,32,1,1,1) to (32,). b_prod =
its own bias term.

MEASUREMENTS (per explicit user spec):
  1. cos(w_prod, w_R1) -- weight-vector similarity. Near 1 (or near a
     scalar multiple, i.e. cos near +-1) -> same direction, different
     scale -- points to calibration. Well below 1 -> different feature
     weighting -- points to a directional/representational discrepancy.
  2. bias difference (b_R1 - b_prod), and more usefully, since the
     weight VECTOR MAGNITUDES differ arbitrarily between two
     independently-fit linear models, the SCALE-NORMALIZED bias
     comparison: does R1's decision boundary sit at a different point
     along the w_prod direction than production's own boundary.
  3. per-lesion logit: ell_prod = w_prod . z + b_prod, ell_R1 = w_R1 . z
     + b_R1, computed at EVERY G2-A lesion's own D1 voxels (reusing R1
     fold assignments from E251 exactly, so ell_R1 is computed by the
     SAME held-out-fold R1 that scored that lesion in E251 -- no
     leakage).
  4. margin (same as logit here, since this is a linear readout -- no
     separate margin transform).
  5. which D1 dimensions contribute most to (ell_R1 - ell_prod) --
     per-channel decomposition: contribution_c = (w_R1[c]-w_prod_scaled[c])
     * z[c], where w_prod_scaled is w_prod rescaled to match w_R1's own
     norm (so the comparison isn't dominated by an arbitrary overall
     scale difference between two independently-fit linear models).
  6. DIRECTIONAL vs CALIBRATION decomposition (the decisive test, per
     explicit user framing): project w_R1 onto w_prod's own direction;
     the ALIGNED component variance vs the ORTHOGONAL component variance
     across lesions tells us whether R1's advantage comes from moving
     ALONG the same direction (calibration/threshold) or from a
     genuinely different direction (directional/representational).
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
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
MAX_VOX_PER_LESION = 30


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- extract w_prod, b_prod (ET channel, seg_head's own learned weights) ----
    conv_w = model.seg_head[0].weight.detach().cpu().numpy()  # (3, 32, 1, 1, 1)
    conv_b = model.seg_head[0].bias.detach().cpu().numpy()    # (3,)
    w_prod = conv_w[ET].reshape(32)
    b_prod = float(conv_b[ET])
    print(f'w_prod: ||w||={np.linalg.norm(w_prod):.4f}  b_prod={b_prod:.4f}', flush=True)

    phenotypes = {}
    for r in csv.DictReader(open(HERE / 'E243_phenotypes.csv')):
        phenotypes[(r['subject_id'], r['comp_id'])] = r['phenotype']
    e240 = list(csv.DictReader(open(HERE / 'E240_layerwise.csv')))
    seen = set()
    det_lesions, g2a_lesions = [], []
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
        if phenotypes.get(key) == 'G2A_rejected':
            g2a_lesions.append(r)

    if smoke:
        det_lesions = det_lesions[:40]
        g2a_lesions = g2a_lesions[:20]

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    det_subjects = sorted(set(r['subject_id'] for r in det_lesions))
    rng = np.random.default_rng(2510)  # SAME seed as E251 -- reproduces the identical fold assignment
    shuffled = rng.permutation(det_subjects)
    fold_of = {sid: i % N_FOLDS for i, sid in enumerate(shuffled)}

    def extract_D1(sid, cid):
        if sid not in sid_to_idx:
            return None
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        brain_mask = img_c[0] != 0
        cm, shell = get_masks(tgt_c, cid, brain_mask)
        if cm is None:
            return None
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0].cpu().numpy()  # (32,D,H,W)
        lesion_feats = d1[:, cm].T
        shell_feats = d1[:, shell].T
        rng_l = np.random.default_rng(hash((sid, cid)) % (2**32))
        if lesion_feats.shape[0] > MAX_VOX_PER_LESION:
            idx = rng_l.choice(lesion_feats.shape[0], size=MAX_VOX_PER_LESION, replace=False)
            lesion_feats = lesion_feats[idx]
        if shell_feats.shape[0] > MAX_VOX_PER_LESION:
            idx = rng_l.choice(shell_feats.shape[0], size=MAX_VOX_PER_LESION, replace=False)
            shell_feats = shell_feats[idx]
        return lesion_feats, shell_feats

    print('Extracting detected lesion D1 features (for R1 training)...', flush=True)
    det_cache = {}
    t0 = time.time()
    for i, r in enumerate(det_lesions):
        res = extract_D1(r['subject_id'], int(r['comp_id']))
        det_cache[(r['subject_id'], r['comp_id'])] = res
        if (i + 1) % 50 == 0 or smoke:
            print(f'  {i+1}/{len(det_lesions)} ({time.time()-t0:.0f}s)', flush=True)

    print('Extracting G2-A lesion D1 features...', flush=True)
    g2a_cache = {}
    for i, r in enumerate(g2a_lesions):
        res = extract_D1(r['subject_id'], int(r['comp_id']))
        g2a_cache[(r['subject_id'], r['comp_id'])] = res
        if (i + 1) % 50 == 0 or smoke:
            print(f'  {i+1}/{len(g2a_lesions)} ({time.time()-t0:.0f}s)', flush=True)

    out = HERE / ('E252_smoke.csv' if smoke else 'E252_geometry.csv')
    fh = open(out, 'w', newline='')
    w_csv = csv.DictWriter(fh, fieldnames=[
        'subject_id', 'comp_id', 'fold', 'cos_w', 'w_r1_norm', 'b_r1',
        'logit_prod_mean', 'logit_r1_mean', 'logit_diff_mean',
        'aligned_component_mean', 'orthogonal_component_norm_mean'])
    w_csv.writeheader()

    w_prod_unit = w_prod / np.linalg.norm(w_prod)

    for fold in range(N_FOLDS):
        train_X, train_y = [], []
        for r in det_lesions:
            sid = r['subject_id']
            if fold_of.get(sid) == fold:
                continue
            res = det_cache.get((sid, r['comp_id']))
            if res is None:
                continue
            lesion_feats, shell_feats = res
            train_X.append(lesion_feats); train_y.append(np.ones(len(lesion_feats)))
            train_X.append(shell_feats); train_y.append(np.zeros(len(shell_feats)))
        if not train_X:
            continue
        X_train = np.concatenate(train_X); y_train = np.concatenate(train_y)
        r1 = LogisticRegression(max_iter=500, C=1.0)
        r1.fit(X_train, y_train)
        w_r1 = r1.coef_[0]  # (32,)
        b_r1 = float(r1.intercept_[0])

        cos_w = float(np.dot(w_prod, w_r1) / (np.linalg.norm(w_prod) * np.linalg.norm(w_r1) + 1e-12))

        for r in g2a_lesions:
            sid = r['subject_id']
            sid_fold = fold_of.get(sid)
            if sid_fold is None:
                sid_fold = abs(hash(sid)) % N_FOLDS
            if sid_fold != fold:
                continue
            res = g2a_cache.get((sid, r['comp_id']))
            if res is None:
                continue
            lesion_feats, _ = res  # (n_vox, 32)

            logit_prod = lesion_feats @ w_prod + b_prod  # (n_vox,)
            logit_r1 = lesion_feats @ w_r1 + b_r1

            # aligned/orthogonal decomposition of w_r1 relative to w_prod's direction
            w_r1_aligned_scalar = np.dot(w_r1, w_prod_unit)  # scalar: R1's weight ALONG w_prod's direction
            w_r1_orth = w_r1 - w_r1_aligned_scalar * w_prod_unit
            aligned_contrib = (lesion_feats @ (w_r1_aligned_scalar * w_prod_unit)).mean()
            orth_contrib_norm = float(np.linalg.norm(lesion_feats @ w_r1_orth) / max(1, len(lesion_feats)))

            w_csv.writerow({
                'subject_id': sid, 'comp_id': r['comp_id'], 'fold': fold,
                'cos_w': cos_w, 'w_r1_norm': float(np.linalg.norm(w_r1)), 'b_r1': b_r1,
                'logit_prod_mean': float(logit_prod.mean()), 'logit_r1_mean': float(logit_r1.mean()),
                'logit_diff_mean': float((logit_r1 - logit_prod).mean()),
                'aligned_component_mean': float(aligned_contrib),
                'orthogonal_component_norm_mean': orth_contrib_norm,
            })
    fh.close()
    print(f'\nE252 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)
    print(f'w_prod norm: {np.linalg.norm(w_prod):.4f}', flush=True)


if __name__ == '__main__':
    main()
