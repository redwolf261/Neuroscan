"""E235 -- layerwise LINEAR-PROBE information-retention test, per the
user's exact spec (numbered E235 to avoid collision with the already-
completed E233 intensity test and E234 frozen-Mahalanobis-probe test --
the user's own message called this "E233" again but that name is taken;
flagged explicitly, as with E234).

REPLACES E234's frozen Mahalanobis-distance probe (found underpowered,
n=78 after tight matching) with a TRAINED linear probe per stage,
evaluated by HELD-OUT AUC on unseen subjects -- much higher statistical
power because the probe can learn which channel combinations actually
separate lesion from background, and doesn't require the aggressive
lesion-pair-matching that starved E234 of sample size.

THREE GROUPS (per explicit user design):
  1. detected            -- successfully segmented ET lesions
  2. high_contrast_missed -- MISSED lesions whose raw t1c contrast is
                              >= the DETECTED population's own median t1c
                              contrast (i.e. "as separable in raw
                              intensity as a typical successfully-detected
                              lesion, yet still missed" -- the decisive
                              population per the user's own framing)
  3. matched_missed        -- ordinary missed lesions (t1c contrast BELOW
                              the detected median), for contrast against
                              group 2

PROBE PROTOCOL (per explicit user decision after being asked): ONE linear
probe per stage, TRAINED ONCE on POOLED lesion-voxel vs shell-voxel
examples from ALL lesions (detected + missed pooled) in a TRAINING
SUBJECT FOLD, then its FIXED weights are used to score held-out lesions
in EACH of the 3 groups separately (subject-level held-out split, per
explicit user warning against voxel-level leakage from spatial
autocorrelation). Same probe capacity (single linear layer, logistic
regression) and same training protocol at every stage -- no stage gets
an advantage from more/less training.

STAGES: raw, E1, E2, E3, E4_BN (bottleneck), D3, D2, D1 -- reusing
e234_layerwise_separability.py's VERIFIED bit-exact resumed_forward()
unchanged.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e234_layerwise_separability import resumed_forward, downsample_mask  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
MIN_VOX = 5
ET = 0
SHELL_DILATION = 3
STAGES = ('raw', 'E1', 'E2', 'E3', 'E4_BN', 'D3', 'D2', 'D1')
STAGE_DOWNSAMPLE = {'raw': 1, 'E1': 1, 'E2': 2, 'E3': 4, 'E4_BN': 8, 'D3': 4, 'D2': 2, 'D1': 1}
N_FOLDS = 5
MAX_VOX_PER_LESION = 30  # cap per-lesion voxel contribution so large lesions
                          # don't dominate the probe training set


def build_groups():
    e223 = list(csv.DictReader(open(HERE / 'E223_exposure.csv')))
    e233 = list(csv.DictReader(open(HERE / 'E233_intensity.csv')))
    e233_by_key = {(r['subject_id'], r['comp_id']): r for r in e233}
    joined = []
    for r in e223:
        key = (r['subject_id'], r['comp_id'])
        r233 = e233_by_key.get(key)
        if r233 is None:
            continue
        joined.append({'subject_id': r['subject_id'], 'comp_id': r['comp_id'],
                       'size': float(r['size']), 'detected': int(r['detected']),
                       'contrast_t1c': float(r233['contrast_t1c'])})
    det = [r for r in joined if r['detected'] == 1]
    missed = [r for r in joined if r['detected'] == 0]
    det_t1c = np.array([r['contrast_t1c'] for r in det])
    med = float(np.median(det_t1c))
    high_contrast_missed = [r for r in missed if r['contrast_t1c'] >= med]
    matched_missed = [r for r in missed if r['contrast_t1c'] < med]
    return det, high_contrast_missed, matched_missed, med


def extract_lesion_voxels(model, ds, sid_to_idx, sid, comp_id, dev):
    """Returns {stage: (lesion_feats (n_lesion_capped, C), shell_feats
    (n_shell_capped, C))} for one lesion, or None if unusable. Reuses
    resumed_forward (VERIFIED bit-exact in E234) and the same shell
    construction convention as E233/E234."""
    image, target, _ = ds._load_subject(ds.subject_dirs[sid_to_idx[sid]])
    D, H, W = image.shape[1:]
    pd, ph, pw = 128, 128, 128
    cd, ch, cw = D // 2, H // 2, W // 2
    sd_, ed_ = max(0, cd - pd//2), min(D, cd + pd//2)
    sh_, eh_ = max(0, ch - ph//2), min(H, ch + ph//2)
    sw_, ew_ = max(0, cw - pw//2), min(W, cw + pw//2)
    img_c = image[:, sd_:ed_, sh_:eh_, sw_:ew_]
    tgt_c = target[:, sd_:ed_, sh_:eh_, sw_:ew_]
    if img_c.shape[1:] != (pd, ph, pw):
        ip = np.zeros((4, pd, ph, pw), dtype=np.float32)
        tp_ = np.zeros((3, pd, ph, pw), dtype=np.float32)
        ip[:, :img_c.shape[1], :img_c.shape[2], :img_c.shape[3]] = img_c
        tp_[:, :tgt_c.shape[1], :tgt_c.shape[2], :tgt_c.shape[3]] = tgt_c
        img_c, tgt_c = ip, tp_
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        stages = resumed_forward(model, img_t)
    stages_np = {k: v[0].cpu().numpy() for k, v in stages.items() if k != 'probs'}

    et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
    if comp_id < 1 or comp_id > et_n:
        return None
    cm = et_lbl == comp_id
    sz = int(cm.sum())
    if sz < MIN_VOX:
        return None
    struct = ndimage.generate_binary_structure(3, 1)
    dilated = ndimage.binary_dilation(cm, structure=struct, iterations=SHELL_DILATION)
    brain_mask = img_c[0] != 0
    shell = dilated & (~cm) & brain_mask & (~(tgt_c[ET] > 0.5))
    if shell.sum() < MIN_VOX:
        return None

    rng = np.random.default_rng(hash((sid, comp_id)) % (2**32))
    result = {}
    for s in STAGES:
        factor = STAGE_DOWNSAMPLE[s]
        lesion_s = downsample_mask(cm, factor)
        shell_s = downsample_mask(shell, factor)
        tensor = stages_np[s]
        C = tensor.shape[0]
        flat = tensor.reshape(C, -1)
        lesion_idx = np.where(lesion_s.reshape(-1))[0]
        shell_idx = np.where(shell_s.reshape(-1))[0]
        if len(lesion_idx) == 0 or len(shell_idx) == 0:
            result[s] = None
            continue
        if len(lesion_idx) > MAX_VOX_PER_LESION:
            lesion_idx = rng.choice(lesion_idx, MAX_VOX_PER_LESION, replace=False)
        if len(shell_idx) > MAX_VOX_PER_LESION:
            shell_idx = rng.choice(shell_idx, MAX_VOX_PER_LESION, replace=False)
        lesion_feats = flat[:, lesion_idx].T  # (n, C)
        shell_feats = flat[:, shell_idx].T
        result[s] = (lesion_feats, shell_feats)
    return result


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    det, high_contrast_missed, matched_missed, t1c_median = build_groups()
    print(f'E235 linear-probe layerwise test: detected={len(det)} '
          f'high_contrast_missed={len(high_contrast_missed)} '
          f'matched_missed={len(matched_missed)} (t1c split at {t1c_median:.4f})', flush=True)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=(128, 128, 128))
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    if smoke:
        det = det[:8]; high_contrast_missed = high_contrast_missed[:5]; matched_missed = matched_missed[:5]

    all_lesions = ([(r, 'detected') for r in det] +
                  [(r, 'high_contrast_missed') for r in high_contrast_missed] +
                  [(r, 'matched_missed') for r in matched_missed])

    # subject-level fold assignment: ALL lesions from the same subject go
    # in the SAME fold, preventing spatial-autocorrelation leakage across
    # train/test (per explicit user warning)
    unique_subjects = sorted(set(r['subject_id'] for r, _ in all_lesions))
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(unique_subjects)
    fold_of = {sid: i % N_FOLDS for i, sid in enumerate(shuffled)}

    t0 = time.time()
    print('Extracting features for all lesions (this is the expensive step)...', flush=True)
    lesion_data = {}  # (sid, comp_id) -> {stage: (lesion_feats, shell_feats)} or None
    n_done = 0
    for r, group in all_lesions:
        key = (r['subject_id'], r['comp_id'])
        if key in lesion_data:
            continue
        if r['subject_id'] not in sid_to_idx:
            lesion_data[key] = None
            continue
        res = extract_lesion_voxels(model, ds, sid_to_idx, r['subject_id'], int(r['comp_id']), dev)
        lesion_data[key] = res
        n_done += 1
        if n_done % 50 == 0 or smoke:
            print(f'  extracted {n_done}/{len(set((r["subject_id"], r["comp_id"]) for r,_ in all_lesions))} '
                  f'({time.time()-t0:.0f}s)', flush=True)
    print(f'Extraction complete: {n_done} unique lesions ({time.time()-t0:.0f}s)', flush=True)

    # ---- per-stage: train probe on train folds, score held-out per group ----
    out = HERE / ('E235_smoke.csv' if smoke else 'E235_probe_auc.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['stage', 'fold', 'group', 'auc', 'n_pos', 'n_neg'])
    w.writeheader()

    for s in STAGES:
        for fold in range(N_FOLDS):
            train_X, train_y = [], []
            for r, group in all_lesions:
                sid = r['subject_id']
                if fold_of.get(sid) == fold:
                    continue  # held out
                key = (sid, r['comp_id'])
                res = lesion_data.get(key)
                if res is None or res.get(s) is None:
                    continue
                lesion_feats, shell_feats = res[s]
                train_X.append(lesion_feats); train_y.append(np.ones(len(lesion_feats)))
                train_X.append(shell_feats); train_y.append(np.zeros(len(shell_feats)))
            if not train_X:
                continue
            X_train = np.concatenate(train_X, axis=0)
            y_train = np.concatenate(train_y, axis=0)
            if len(np.unique(y_train)) < 2:
                continue
            probe = LogisticRegression(max_iter=500, C=1.0)
            probe.fit(X_train, y_train)

            for group_name, group_rows in [('detected', det), ('high_contrast_missed', high_contrast_missed),
                                           ('matched_missed', matched_missed)]:
                test_X, test_y = [], []
                for r in group_rows:
                    sid = r['subject_id']
                    if fold_of.get(sid) != fold:
                        continue  # only score subjects IN this held-out fold
                    key = (sid, r['comp_id'])
                    res = lesion_data.get(key)
                    if res is None or res.get(s) is None:
                        continue
                    lesion_feats, shell_feats = res[s]
                    test_X.append(lesion_feats); test_y.append(np.ones(len(lesion_feats)))
                    test_X.append(shell_feats); test_y.append(np.zeros(len(shell_feats)))
                if not test_X:
                    continue
                X_test = np.concatenate(test_X, axis=0)
                y_test = np.concatenate(test_y, axis=0)
                if len(np.unique(y_test)) < 2:
                    continue
                probs = probe.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, probs)
                w.writerow({'stage': s, 'fold': fold, 'group': group_name, 'auc': auc,
                           'n_pos': int(y_test.sum()), 'n_neg': int((1-y_test).sum())})
        fh.flush()
        print(f'  stage {s} done ({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'wrote {out.name} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
