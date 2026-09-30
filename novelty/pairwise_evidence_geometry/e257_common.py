"""E257 common infrastructure -- shared by e257a (learned gate) and
e257b (retrained R1 with representative negatives). Genuine algorithm
design, per the user's explicit framing: prior experiments established
WHAT the algorithm must solve (D1 has recoverable info, production
readout fails to use it, naive exposure causes catastrophic FP, no
spatial/rank gate is feasible); E257 tests the first actual candidate
mechanisms.

SPLIT DISCIPLINE (per explicit user decisions this turn):
  - detected subjects (n=237): SAME 60/20/20 split as E254 (identical
    seed=2540), reused unchanged.
  - G2-A subjects (n=150): NEW independent 60/20/20 split (own seed),
    per explicit user choice -- the gate needs G2-A training examples
    to learn y_gate=1 cases from, but a genuine held-out G2-A test
    portion is never touched during training or hyperparameter
    selection.
  - G2-B subjects (n=97): entirely held out, NEVER used in training --
    pure evaluation population, same convention as E251-E256.
  - distant_background voxels: sampled ~30-40/subject (per explicit
    user choice, matching MAX_VOX_PER_LESION's existing scale) from
    OUTSIDE E246's own 12-voxel near/far boundary, for TRAIN-split
    subjects only (detected-train + G2A-train).
"""
import sys, csv, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

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
NEAR_DILATION = 12
MAX_VOX_PER_LESION = 30
N_DISTANT_PER_SUBJECT = 35


def load_model(dev):
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_w_prod(model):
    conv_w = model.seg_head[0].weight.detach().cpu().numpy()
    conv_b = model.seg_head[0].bias.detach().cpu().numpy()
    return conv_w[ET].reshape(32).astype(np.float64), float(conv_b[ET])


def load_populations():
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
    return det_lesions, g2a_lesions, g2b_lesions


def build_splits(det_lesions, g2a_lesions):
    """Returns dict: {'det_train','det_val','det_test','g2a_train',
    'g2a_val','g2a_test'} -> set of subject_ids.

    CRITICAL FIX (found and corrected before running anything on real
    data): a subject can carry BOTH a detected lesion and a separate
    G2-A missed lesion (27/237 detected subjects also appear in the 150
    G2-A subjects). Splitting det_subjects and g2a_subjects with two
    INDEPENDENT random assignments produced real cross-population
    leakage -- e.g. 3 subjects nominally in det_test (held out for
    detected evaluation) were simultaneously in g2a_train (used for
    training via their G2-A lesion), meaning their D1 features WERE
    seen during training despite being labeled 'held out'. Caught via a
    direct overlap check before trusting any downstream result.

    FIX: build ONE unified subject-level train/val/test assignment
    (60/20/20) over the UNION of all subjects appearing in EITHER
    det_lesions or g2a_lesions, so every subject gets exactly ONE role
    regardless of which population(s) its lesions belong to. This
    guarantees a subject in 'test' contributes NO voxels to training,
    full stop, regardless of lesion type."""
    all_subjects = sorted(set(r['subject_id'] for r in det_lesions) |
                          set(r['subject_id'] for r in g2a_lesions))
    rng = np.random.default_rng(2540)
    shuffled = rng.permutation(all_subjects)
    n = len(shuffled)
    n_train = int(n * 0.6); n_val = int(n * 0.2)
    train_subj = set(shuffled[:n_train])
    val_subj = set(shuffled[n_train:n_train + n_val])
    test_subj = set(shuffled[n_train + n_val:])

    return {'det_train': train_subj, 'det_val': val_subj, 'det_test': test_subj,
           'g2a_train': train_subj, 'g2a_val': val_subj, 'g2a_test': test_subj}


def extract_lesion_shell(model, ds, sid_to_idx, sid, cid, dev):
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    brain_mask = img_c[0] != 0
    cm, shell = get_masks(tgt_c, cid, brain_mask)
    if cm is None:
        return None
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    stages = forward_to_dec1_internal(model, img_t)
    d1 = stages['relu2_out'][0].cpu().numpy()
    lesion_feats = d1[:, cm].T
    shell_feats = d1[:, shell].T
    return lesion_feats, shell_feats, cm.sum()


def extract_distant_background(model, ds, sid_to_idx, sid, dev, rng, n_sample=N_DISTANT_PER_SUBJECT):
    """Samples n_sample D1 feature vectors from DISTANT background (outside
    E246's 12-voxel near/far boundary around ANY ET lesion in this subject).
    Returns (n_sample, 32) or None."""
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    brain_mask = img_c[0] != 0
    true_et = tgt_c[ET] > 0.5
    struct = ndimage.generate_binary_structure(3, 1)
    if true_et.any():
        dilated = ndimage.binary_dilation(true_et, structure=struct, iterations=NEAR_DILATION)
        local_shell = dilated & (~true_et) & brain_mask
    else:
        local_shell = np.zeros_like(true_et)
    distant_bg = brain_mask & (~true_et) & (~local_shell)
    idx = np.where(distant_bg)
    if len(idx[0]) == 0:
        return None
    n_avail = len(idx[0])
    sel = rng.choice(n_avail, size=min(n_sample, n_avail), replace=False)
    sel_coords = (idx[0][sel], idx[1][sel], idx[2][sel])

    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    stages = forward_to_dec1_internal(model, img_t)
    d1 = stages['relu2_out'][0].cpu().numpy()  # (32,D,H,W)
    feats = d1[:, sel_coords[0], sel_coords[1], sel_coords[2]].T  # (n_sample, 32)
    return feats
