"""E256 -- oracle-style feasibility bound for a selective/gated readout,
per the user's exact spec. Does NOT train any gate yet -- establishes
whether selective R1 utilization is even geometrically feasible before
investing in learning g(z).

VOXEL PARTITION (dense, whole 128^3 patch, per subject), per explicit
user decision (reusing E246's own 12-voxel near/far convention):
  lesion             -- true ET voxels
  local_shell        -- within 12-voxel dilation of ANY ET lesion in this
                        subject, excluding lesion voxels themselves
  distant_background -- everything else (not lesion, not local_shell)
  (production_positive flagged separately, cross-cutting: p_prod>0.5)

ORACLE FEASIBILITY QUESTIONS (per explicit user list):
  1. Where p_R1 > p_prod: how often is R1 actually CORRECT (using ground
     truth, oracle-only -- a real gate could never see this at inference,
     this establishes the CEILING), broken down by spatial category.
  2. Where p_prod > p_R1: how often is production actually correct,
     same breakdown.
  3. How much of total POSSIBLE G2-A recovery (lesion voxels where R1
     alone would recover, p_R1>0.5) lives in each spatial category --
     tells us whether a PURELY SPATIAL gate (no ground truth, no
     feature-space reasoning, just "trust R1 within some radius of
     existing production activity") could plausibly work at all.
  4. How many distant-background voxels would a naive "trust R1
     wherever p_R1>p_prod" gate expose -- the SAME failure mode E255
     found, quantified per spatial category this time (not aggregated),
     to see whether it's specifically a distant-background problem (as
     hypothesized) or spread more broadly.

TRAIN/VAL/TEST DISCIPLINE (per explicit user requirement): reuses
E254's EXACT same 60/20/20 patient-level split (same seed) for detected
subjects. R1 refit identically (train-split only). This entire analysis
is run on the TEST split for detected subjects, plus ALL G2-A and G2-B
subjects (already disjoint populations) -- i.e. this IS the held-out
evaluation, not a separate train/eval step (there is nothing to fit in
this script; it is pure oracle/feasibility measurement).
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
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
ET = 0
PATCH = (128, 128, 128)
MIN_VOX = 5
TAU = 0.5
NEAR_DILATION = 12  # matches E246's own near/far convention
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
    det_lesions = []
    for r in e240:
        key = (r['subject_id'], r['comp_id'])
        if r['role'] == 'detected':
            if key in seen:
                continue
            seen.add(key)
            det_lesions.append(r)

    # ---- REPRODUCE E254's exact split ----
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

    def extract_D1_voxels(sid, cid):
        if sid not in sid_to_idx:
            return None
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
        if cid < 1 or cid > et_n:
            return None
        cm = et_lbl == cid
        if cm.sum() < MIN_VOX:
            return None
        brain_mask = img_c[0] != 0
        struct = ndimage.generate_binary_structure(3, 1)
        dilated = ndimage.binary_dilation(cm, structure=struct, iterations=3)  # 3-vox shell, E233 convention
        shell = dilated & (~cm) & brain_mask & (~(tgt_c[ET] > 0.5))
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0].cpu().numpy()
        lesion_feats = d1[:, cm].T
        shell_feats = d1[:, shell].T
        return lesion_feats, shell_feats

    print('Refitting R1 on E254 train split...', flush=True)
    train_X, train_y = [], []
    rng_l = np.random.default_rng(999)
    n_fit = 0
    for r in det_lesions:
        if r['subject_id'] not in train_subj:
            continue
        res = extract_D1_voxels(r['subject_id'], int(r['comp_id']))
        if res is None:
            continue
        lesion_feats, shell_feats = res
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
    w_r1 = torch.from_numpy(r1.coef_[0].astype(np.float64)).float().to(dev)
    b_r1 = float(r1.intercept_[0])
    w_prod_t = torch.from_numpy(w_prod).float().to(dev)
    print(f'R1 refit on {n_fit} detected lesions', flush=True)

    # ---- test subjects: detected-test + ALL G2A + ALL G2B ----
    seen2 = set()
    g2a_lesions, g2b_lesions = [], []
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

    test_subjects = sorted(test_subj) + sorted(set(r['subject_id'] for r in g2a_lesions)) + \
                    sorted(set(r['subject_id'] for r in g2b_lesions))
    test_subjects = sorted(set(test_subjects))
    if smoke:
        test_subjects = test_subjects[:15]

    print(f'\nEvaluating {len(test_subjects)} held-out subjects...', flush=True)

    out = HERE / ('E256_smoke.csv' if smoke else 'E256_feasibility.csv')
    fh = open(out, 'w', newline='')
    w_csv = csv.DictWriter(fh, fieldnames=[
        'subject_id', 'category',
        'n_voxels', 'n_r1_gt_prod', 'n_r1_gt_prod_correct',
        'n_prod_gt_r1', 'n_prod_gt_r1_correct',
        'n_r1_recovers_lesion',  # lesion voxels where p_R1>0.5 (only meaningful for category='lesion')
    ])
    w_csv.writeheader()

    t0 = time.time()
    struct = ndimage.generate_binary_structure(3, 1)
    for i, sid in enumerate(test_subjects):
        if sid not in sid_to_idx:
            continue
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        d1 = stages['relu2_out'][0]  # (32,D,H,W) GPU
        C, D, H, W = d1.shape
        flat = d1.reshape(C, -1).T
        with torch.no_grad():
            p_prod = torch.sigmoid(flat @ w_prod_t + b_prod).reshape(D, H, W).cpu().numpy()
            p_r1 = torch.sigmoid(flat @ w_r1 + b_r1).reshape(D, H, W).cpu().numpy()

        true_et = tgt_c[ET] > 0.5
        brain_mask = img_c[0] != 0
        if true_et.any():
            dilated = ndimage.binary_dilation(true_et, structure=struct, iterations=NEAR_DILATION)
            local_shell = dilated & (~true_et) & brain_mask
        else:
            local_shell = np.zeros_like(true_et)
        distant_bg = brain_mask & (~true_et) & (~local_shell)

        categories = {'lesion': true_et, 'local_shell': local_shell, 'distant_background': distant_bg}
        for cat_name, cat_mask in categories.items():
            if cat_mask.sum() == 0:
                continue
            p_prod_c = p_prod[cat_mask]
            p_r1_c = p_r1[cat_mask]
            gt_c = true_et[cat_mask].astype(int)

            r1_gt_prod = p_r1_c > p_prod_c
            prod_gt_r1 = p_prod_c > p_r1_c

            r1_gt_prod_pred = (p_r1_c[r1_gt_prod] > TAU).astype(int)
            r1_gt_prod_correct = int((r1_gt_prod_pred == gt_c[r1_gt_prod]).sum())

            prod_gt_r1_pred = (p_prod_c[prod_gt_r1] > TAU).astype(int)
            prod_gt_r1_correct = int((prod_gt_r1_pred == gt_c[prod_gt_r1]).sum())

            n_r1_recovers = int(((p_r1_c > TAU) & (gt_c == 1)).sum()) if cat_name == 'lesion' else 0

            w_csv.writerow({
                'subject_id': sid, 'category': cat_name,
                'n_voxels': int(cat_mask.sum()),
                'n_r1_gt_prod': int(r1_gt_prod.sum()), 'n_r1_gt_prod_correct': r1_gt_prod_correct,
                'n_prod_gt_r1': int(prod_gt_r1.sum()), 'n_prod_gt_r1_correct': prod_gt_r1_correct,
                'n_r1_recovers_lesion': n_r1_recovers,
            })
        if (i + 1) % 20 == 0 or smoke:
            print(f'  {i+1}/{len(test_subjects)} ({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'\nE256 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
