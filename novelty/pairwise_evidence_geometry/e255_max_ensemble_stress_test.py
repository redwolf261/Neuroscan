"""E255 -- stress test of the simplest possible readout-adaptation
mechanism, per the user's exact sequencing and explicit choice on this
turn's disambiguation: max(p_prod, p_R1), ZERO new trained parameters,
applied DENSELY over the WHOLE 128^3 patch (not just lesion+shell
voxels) -- because the critical missing measurement from E251-E254 is
WHOLE-VOLUME false positive cost. Every prior readout experiment only
measured FP in the immediate local shell around a known lesion; a
max() ensemble could plausibly fire anywhere R1 is confident, including
regions with no real lesion at all. This must be checked before any
mechanism is taken seriously, per this session's own E237-informed
discipline.

USES E254's OWN held-out-selected R1 (alpha=1.0 -- i.e. the actual fresh
R1 fit on TRAIN-split detected lesions, re-fit here identically, same
seed/split, so this is the SAME R1 already validated in E254, not a new
model). w_prod/b_prod: production seg_head, unchanged.

DENSE APPLICATION: for a sampled set of subjects, compute p_prod and
p_R1 at EVERY voxel of the 128^3 patch (not sampled lesion/shell voxels)
via the SAME frozen D1=relu2_out tensor, then p_max = max(p_prod, p_R1)
elementwise. Measures:
  - G2-A/G2-B/detected-test lesion-level recovery (same tau=0.5
    convention, for direct comparability with E251/E253/E254)
  - WHOLE-VOLUME false positive VOXEL COUNT and CONNECTED-COMPONENT
    COUNT outside the true ET label, for p_max vs p_prod alone (the
    critical new measurement)
  - Per-subject FP burden distribution (not just a mean -- a few bad
    subjects could hide behind a low average)
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
MAX_VOX_PER_LESION = 30


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

    # ---- REPRODUCE E254's exact train/val/test split (same seed) ----
    det_subjects = sorted(set(r['subject_id'] for r in det_lesions))
    rng = np.random.default_rng(2540)
    shuffled = rng.permutation(det_subjects)
    n = len(shuffled)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_subj = set(shuffled[:n_train])
    test_subj = set(shuffled[n_train + n_val:])

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    # ---- refit R1 on TRAIN-split detected lesions (identical to E254's R1) ----
    def extract_D1_full(sid, cid):
        if sid not in sid_to_idx:
            return None
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        brain_mask = img_c[0] != 0
        cm, shell = get_masks(tgt_c, cid, brain_mask)
        if cm is None:
            return None
        sz = int(cm.sum())
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0].cpu().numpy()
        lesion_feats = d1[:, cm].T
        shell_feats = d1[:, shell].T
        return lesion_feats, shell_feats, sz

    print('Refitting R1 on E254 train split (identical procedure)...', flush=True)
    train_X, train_y = [], []
    rng_l = np.random.default_rng(999)
    n_fit = 0
    for r in det_lesions:
        if r['subject_id'] not in train_subj:
            continue
        res = extract_D1_full(r['subject_id'], int(r['comp_id']))
        if res is None:
            continue
        lesion_feats, shell_feats, sz = res
        if len(lesion_feats) > MAX_VOX_PER_LESION:
            idx = rng_l.choice(len(lesion_feats), MAX_VOX_PER_LESION, replace=False)
            lesion_feats = lesion_feats[idx]
        if len(shell_feats) > MAX_VOX_PER_LESION:
            idx = rng_l.choice(len(shell_feats), MAX_VOX_PER_LESION, replace=False)
            shell_feats = shell_feats[idx]
        train_X.append(lesion_feats); train_y.append(np.ones(len(lesion_feats)))
        train_X.append(shell_feats); train_y.append(np.zeros(len(shell_feats)))
        n_fit += 1
        if smoke and n_fit >= 40:
            break
    X_train = np.concatenate(train_X); y_train = np.concatenate(train_y)
    r1 = LogisticRegression(max_iter=500, C=1.0)
    r1.fit(X_train, y_train)
    w_r1 = r1.coef_[0].astype(np.float64)
    b_r1 = float(r1.intercept_[0])
    print(f'R1 refit on {n_fit} detected lesions', flush=True)

    # ---- dense whole-volume evaluation ----
    # keep the dense per-voxel linear-readout computation ON GPU (torch) --
    # the pure-numpy CPU version was profiled at ~7.7s PER MATMUL on a
    # (2M, 32) array (128^3 voxels), making the original implementation
    # ~29s/subject and projected to take ~2+ hours for the full run.
    # caught via direct profiling before launching the full run, not
    # discovered mid-run.
    w_prod_t = torch.from_numpy(w_prod).float().to(dev)
    w_r1_t = torch.from_numpy(w_r1).float().to(dev)

    def dense_probs(sid):
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]  # (32,D,H,W) torch, ON GPU
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T  # (D*H*W, 32), GPU
        with torch.no_grad():
            logit_prod = flat @ w_prod_t + b_prod
            logit_r1 = flat @ w_r1_t + b_r1
            p_prod = torch.sigmoid(logit_prod).reshape(D, H, W).cpu().numpy()
            p_r1 = torch.sigmoid(logit_r1).reshape(D, H, W).cpu().numpy()
        p_max = np.maximum(p_prod, p_r1)
        return p_prod, p_r1, p_max, tgt_c

    def fp_stats(p_map, tgt_c, brain_mask):
        pred_mask = (p_map > TAU) & brain_mask
        true_et = tgt_c[ET] > 0.5
        fp_mask = pred_mask & (~true_et)
        fp_voxels = int(fp_mask.sum())
        fp_lbl, fp_n = ndimage.label(fp_mask)
        return fp_voxels, fp_n

    print('\nDense whole-volume evaluation...', flush=True)
    subjects_to_eval = sorted(test_subj) + sorted(set(r['subject_id'] for r in g2a_lesions)) + \
                       sorted(set(r['subject_id'] for r in g2b_lesions))
    subjects_to_eval = sorted(set(subjects_to_eval))
    if smoke:
        subjects_to_eval = subjects_to_eval[:15]

    out = HERE / ('E255_smoke.csv' if smoke else 'E255_max_ensemble.csv')
    fh = open(out, 'w', newline='')
    w_csv = csv.DictWriter(fh, fieldnames=['subject_id', 'in_test_split', 'fp_voxels_prod',
                                           'fp_components_prod', 'fp_voxels_max', 'fp_components_max'])
    w_csv.writeheader()

    t0 = time.time()
    for i, sid in enumerate(subjects_to_eval):
        if sid not in sid_to_idx:
            continue
        p_prod, p_r1, p_max, tgt_c = dense_probs(sid)
        image, _ = load_patch(ds, sid_to_idx, sid)
        brain_mask = image[0] != 0
        fp_v_prod, fp_c_prod = fp_stats(p_prod, tgt_c, brain_mask)
        fp_v_max, fp_c_max = fp_stats(p_max, tgt_c, brain_mask)
        w_csv.writerow({'subject_id': sid, 'in_test_split': int(sid in test_subj),
                        'fp_voxels_prod': fp_v_prod, 'fp_components_prod': fp_c_prod,
                        'fp_voxels_max': fp_v_max, 'fp_components_max': fp_c_max})
        if (i + 1) % 10 == 0 or smoke:
            print(f'  {i+1}/{len(subjects_to_eval)} ({time.time()-t0:.0f}s)', flush=True)
    fh.close()

    # ---- lesion-level recovery, same convention as E251/E253/E254 ----
    print('\nLesion-level recovery (max ensemble vs production)...', flush=True)

    def lesion_recovery(lesion_list, subj_filter):
        recs_prod, recs_max = [], []
        for r in lesion_list:
            sid = r['subject_id']
            if subj_filter is not None and sid not in subj_filter:
                continue
            if sid not in sid_to_idx:
                continue
            res = extract_D1_full(sid, int(r['comp_id']))
            if res is None:
                continue
            lesion_feats, shell_feats, sz = res
            p_prod_l = expit(lesion_feats @ w_prod + b_prod)
            p_r1_l = expit(lesion_feats @ w_r1 + b_r1)
            p_max_l = np.maximum(p_prod_l, p_r1_l)
            recs_prod.append(int(p_prod_l.max() > TAU))
            recs_max.append(int(p_max_l.max() > TAU))
        return np.mean(recs_prod) if recs_prod else float('nan'), np.mean(recs_max) if recs_max else float('nan')

    det_lesions_eval = det_lesions[:60] if smoke else det_lesions
    g2a_eval = g2a_lesions[:30] if smoke else g2a_lesions
    g2b_eval = g2b_lesions[:20] if smoke else g2b_lesions

    det_prod, det_max = lesion_recovery(det_lesions_eval, test_subj)
    g2a_prod, g2a_max = lesion_recovery(g2a_eval, None)
    g2b_prod, g2b_max = lesion_recovery(g2b_eval, None)
    print(f'  detected_test: production={det_prod:.4f}  max_ensemble={det_max:.4f}')
    print(f'  G2A: production={g2a_prod:.4f}  max_ensemble={g2a_max:.4f}')
    print(f'  G2B: production={g2b_prod:.4f}  max_ensemble={g2b_max:.4f}')

    print(f'\nE255 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
