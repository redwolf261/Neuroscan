"""E250 Stage 2 -- activation PATTERN vs channel IDENTITY, per the
user's exact spec. E249 killed "specific channels are the causal
bottleneck" (target-channel rescue lost to random/dead-channel rescue
on recovery). This tests the alternative: does the JOINT cross-channel
activation pattern (not which channels exist, but how they relate to
each other) carry the missing information.

FOUR CONDITIONS at BN1_out, all passed through the UNCHANGED
ReLU1->conv2->BN2->ReLU2(=D1)->seg_head pathway (only BN1_out itself is
modified; everything downstream is the real, frozen model path):

  1. real            -- unmodified BN1_out (baseline)
  2. channel_shuffled -- each of the 32 channels INDEPENDENTLY permuted
                         across the lesion's own voxels (preserves each
                         channel's own marginal distribution over the
                         lesion, DESTROYS the per-voxel cross-channel
                         relationship -- voxel i's channel-30 value may
                         now be paired with voxel j's channel-4 value)
  3. spatial_shuffled -- the FULL 32-channel vector at each voxel is
                         kept intact, but WHICH voxel gets which vector
                         is permuted (preserves cross-channel structure
                         and marginal per-channel stats, destroys spatial
                         arrangement relative to the lesion geometry --
                         same operation as E249's shuffled_rescue, but
                         applied to ALL 32 channels here, not just 8)
  4. synthetic_matched -- Cholesky-based recoloring (per explicit user
                         decision, Stage 1's pooled DETECTED correlation
                         matrix): G2-A's own per-channel MEAN and
                         VARIANCE preserved exactly, per-voxel SPATIAL
                         LOCATION preserved exactly, but the CROSS-
                         CHANNEL CORRELATION structure is replaced with
                         detected lesions' own pooled correlation matrix.
                         Construction: whiten G2-A's own (n_vox, 32)
                         voxel-channel matrix via its OWN covariance
                         (Cholesky of G2-A's own cov), producing
                         approximately-decorrelated unit-variance data,
                         then recolor via Cholesky of the DETECTED
                         correlation matrix (Stage 1's own estimate),
                         then rescale/re-center to G2-A's own original
                         per-channel mean/std (so marginals are UNCHANGED,
                         only the joint/cross-channel structure moves
                         toward detected's own).

  A COVARIANCE-RANDOMIZED control (per explicit user requirement: "beats
  a covariance-randomized control") is ALSO run: same recoloring
  procedure, but using a RANDOM valid correlation matrix (generated via
  a random orthogonal rotation of the SAME eigenvalue spectrum as the
  detected correlation matrix, so it has comparable "structuredness" but
  is not detected's own specific structure) instead of the real detected
  correlation matrix.

POPULATION: G2-A (primary), G2-B and detected included as reference
populations for condition 1 (real) only, per the user's explicit
comparison request.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e234_layerwise_separability import separability  # noqa: E402
from e245_dec1_internal_decomposition import load_patch, get_masks  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
ET = 0
PATCH = (128, 128, 128)
MIN_VOX = 5
TAU = 0.5
RNG = np.random.default_rng(2501)


def cholesky_recolor(vox_channel, target_corr, eps=1e-6):
    """vox_channel: (n_vox, 32) numpy, this lesion's own BN1 values.
    target_corr: (32,32) correlation matrix to impose.
    Returns a NEW (n_vox, 32) array with G2-A's own per-channel mean/std
    preserved exactly, cross-channel correlation moved toward
    target_corr."""
    mean = vox_channel.mean(axis=0)
    std = vox_channel.std(axis=0) + eps
    z = (vox_channel - mean) / std  # (n_vox, 32), approx unit variance per channel

    own_corr = np.corrcoef(z, rowvar=False)
    own_corr = own_corr + np.eye(32) * 1e-4
    try:
        L_own_inv = np.linalg.inv(np.linalg.cholesky(own_corr))
    except np.linalg.LinAlgError:
        return None
    white = z @ L_own_inv.T  # approx decorrelated, unit variance

    try:
        L_target = np.linalg.cholesky(target_corr)
    except np.linalg.LinAlgError:
        return None
    recolored = white @ L_target.T  # now has target_corr's correlation structure

    # renormalize to exactly unit variance per channel (Cholesky recoloring
    # can drift slightly from unit variance due to the own_corr inversion),
    # THEN rescale/recenter to G2-A's own original mean/std -- marginals
    # preserved by construction, only joint structure changed
    recolored = recolored / (recolored.std(axis=0) + eps)
    result = recolored * std + mean
    return result


def forward_from_bn1(model, cat1, bn1_modified):
    """Replays conv2->BN2->ReLU2(=D1)->seg_head from a (possibly
    modified) bn1_out tensor, using the model's own layer objects."""
    with torch.no_grad():
        relu1_out = F.relu(bn1_modified)
        block2 = model.dec1[1]
        conv2_out = block2.conv(relu1_out)
        bn2_out = block2.bn(conv2_out)
        relu2_out = F.relu(bn2_out)  # = D1
        probs = model.seg_head(relu2_out)
    return {'probs': probs, 'relu2_out': relu2_out}


def get_bn1_and_cat1(model, img_t):
    with torch.no_grad():
        enc1 = model.enc1(img_t); pool1 = model.pool1(enc1)
        enc2 = model.enc2(pool1); pool2 = model.pool2(enc2)
        enc3 = model.enc3(pool2); pool3 = model.pool3(enc3)
        bottleneck = model.bottleneck(pool3)
        upconv3 = model.upconv3(bottleneck)
        cat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = model.dec3(cat3)
        upconv2 = model.upconv2(dec3)
        cat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = model.dec2(cat2)
        upconv1 = model.upconv1(dec2)
        enc1_gated, _ = model.attn_gate1(gate=bottleneck, skip=enc1)
        cat1 = torch.cat([upconv1, enc1_gated], dim=1)
        block1 = model.dec1[0]
        conv1_out = block1.conv(cat1)
        bn1_out = block1.bn(conv1_out)
    return bn1_out


def measure(model, img_t, cm, shell, bn1_modified):
    out = forward_from_bn1(model, None, bn1_modified)
    p = out['probs'][0, ET].cpu().numpy()
    max_p = float(p[cm].max())
    mean_p = float(p[cm].mean())
    recovered = int(max_p > TAU)
    d1_np = out['relu2_out'][0].cpu().numpy()
    sep = separability(d1_np, cm, shell)
    # FP proxy: mean probability OUTSIDE the lesion but inside the shell
    # (local false-positive pressure, cheap to compute here)
    fp_shell = float(p[shell].mean())
    return {'max_prob': max_p, 'mean_prob': mean_p, 'recovered': recovered,
           'sep_D1': sep if sep is not None else '', 'fp_shell': fp_shell}


def random_corr_same_spectrum(target_corr, rng):
    """Generates a random valid correlation matrix with the SAME
    eigenvalue spectrum as target_corr (comparable 'structuredness') via
    a random orthogonal rotation -- the covariance-randomized control."""
    eigvals, eigvecs = np.linalg.eigh(target_corr)
    # random orthogonal matrix (QR of a random Gaussian matrix)
    A = rng.standard_normal((32, 32))
    Q, _ = np.linalg.qr(A)
    new_mat = Q @ np.diag(eigvals) @ Q.T
    # renormalize to a correlation matrix (unit diagonal)
    d = np.sqrt(np.diag(new_mat))
    corr = new_mat / np.outer(d, d)
    return corr + np.eye(32) * 1e-4


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    det_corr = np.load(HERE / 'E250_detected_correlation.npy')
    random_corr = random_corr_same_spectrum(det_corr, RNG)

    phenotypes = {}
    for r in csv.DictReader(open(HERE / 'E243_phenotypes.csv')):
        phenotypes[(r['subject_id'], r['comp_id'])] = r['phenotype']
    e240 = list(csv.DictReader(open(HERE / 'E240_layerwise.csv')))
    seen = set()
    g2a_lesions, g2b_lesions, det_lesions = [], [], []
    for r in e240:
        key = (r['subject_id'], r['comp_id'])
        if key in seen and r['role'] == 'missed':
            continue
        if r['role'] == 'missed':
            seen.add(key)
            phen = phenotypes.get(key)
            if phen == 'G2A_rejected':
                g2a_lesions.append(r)
            elif phen == 'G2B_partial':
                g2b_lesions.append(r)
        elif r['role'] == 'detected':
            det_lesions.append(r)
    # dedupe detected
    seen_det = set(); det_unique = []
    for r in det_lesions:
        key = (r['subject_id'], r['comp_id'])
        if key in seen_det:
            continue
        seen_det.add(key); det_unique.append(r)
    det_lesions = det_unique

    print(f'E250 Stage 2: G2-A={len(g2a_lesions)} G2-B={len(g2b_lesions)} detected={len(det_lesions)}', flush=True)
    if smoke:
        g2a_lesions = g2a_lesions[:8]
        g2b_lesions = g2b_lesions[:6]
        det_lesions = det_lesions[:6]

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    out = HERE / ('E250_smoke.csv' if smoke else 'E250_pattern.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['group', 'subject_id', 'comp_id', 'size', 'condition',
                                       'max_prob', 'mean_prob', 'recovered', 'sep_D1', 'fp_shell'])
    w.writeheader()

    t0 = time.time()
    n_done = 0
    verified = False

    def process_lesion(grp, r, conditions):
        nonlocal n_done, verified
        sid, cid = r['subject_id'], int(r['comp_id'])
        if sid not in sid_to_idx:
            return
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        brain_mask = img_c[0] != 0
        cm, shell = get_masks(tgt_c, cid, brain_mask)
        if cm is None:
            return
        sz = int(cm.sum())
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        bn1_out = get_bn1_and_cat1(model, img_t)

        if not verified:
            with torch.no_grad():
                real_out = model(img_t)
            res_check = measure(model, img_t, cm, shell, bn1_out)
            real_max = float(real_out['probs'][0, ET][torch.from_numpy(cm).to(dev)].max().item())
            assert abs(real_max - res_check['max_prob']) < 1e-4, \
                f'verify failed: {real_max} vs {res_check["max_prob"]}'
            print(f'  [verify] real-condition replay matches model() (diff '
                  f'{abs(real_max - res_check["max_prob"]):.2e})', flush=True)
            verified = True

        bn1_np = bn1_out[0].cpu().numpy()  # (32, D, H, W)
        lesion_idx = np.where(cm)

        for cond in conditions:
            if cond == 'real':
                bn1_mod = bn1_out
            elif cond == 'channel_shuffled':
                mod = bn1_np.copy()
                for c in range(32):
                    vals = mod[c][cm].copy()
                    RNG.shuffle(vals)
                    mod[c][lesion_idx] = vals
                bn1_mod = torch.from_numpy(mod).unsqueeze(0).to(dev)
            elif cond == 'spatial_shuffled':
                mod = bn1_np.copy()
                vox = mod[:, lesion_idx[0], lesion_idx[1], lesion_idx[2]].T.copy()  # (n_vox,32)
                perm = RNG.permutation(vox.shape[0])
                vox_shuf = vox[perm]
                mod[:, lesion_idx[0], lesion_idx[1], lesion_idx[2]] = vox_shuf.T
                bn1_mod = torch.from_numpy(mod).unsqueeze(0).to(dev)
            elif cond in ('synthetic_matched', 'covariance_randomized'):
                vox = bn1_np[:, lesion_idx[0], lesion_idx[1], lesion_idx[2]].T  # (n_vox, 32)
                target = det_corr if cond == 'synthetic_matched' else random_corr
                new_vox = cholesky_recolor(vox, target)
                if new_vox is None:
                    continue
                mod = bn1_np.copy()
                mod[:, lesion_idx[0], lesion_idx[1], lesion_idx[2]] = new_vox.T
                bn1_mod = torch.from_numpy(mod).unsqueeze(0).to(dev)
            else:
                raise ValueError(cond)

            res = measure(model, img_t, cm, shell, bn1_mod)
            w.writerow({'group': grp, 'subject_id': sid, 'comp_id': cid, 'size': sz,
                       'condition': cond, **res})
        n_done += 1
        if n_done % 10 == 0 or smoke:
            print(f'  {n_done} lesions ({time.time()-t0:.0f}s)', flush=True)

    ALL_CONDITIONS = ['real', 'channel_shuffled', 'spatial_shuffled', 'synthetic_matched', 'covariance_randomized']
    for r in g2a_lesions:
        process_lesion('G2A', r, ALL_CONDITIONS)
    fh.flush()
    for r in g2b_lesions:
        process_lesion('G2B', r, ['real'])
    fh.flush()
    for r in det_lesions:
        process_lesion('detected', r, ['real'])
    fh.close()
    print(f'\nE250 Stage 2 complete: {n_done} lesions ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
