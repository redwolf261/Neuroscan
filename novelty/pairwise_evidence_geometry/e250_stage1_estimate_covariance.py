"""E250 Stage 1 -- estimate the pooled detected-lesion cross-channel
correlation matrix at BN1_out (block1, dec1), needed for the synthetic
matched-activation control. Per explicit user decision (this turn's
disambiguation), uses a Cholesky-based recoloring approach: this stage
estimates and saves ONE pooled 32x32 correlation matrix from many
detected lesions' own BN1 voxel-channel data (pooling voxels across
lesions for a stable estimate), verified well-conditioned (invertible)
before use.

Reuses forward_to_dec1_internal (E245) and get_masks/load_patch (E245)
UNCHANGED.
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch, get_masks  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
PATCH = (128, 128, 128)
MAX_VOX_PER_LESION = 200  # cap so a few huge lesions don't dominate the pooled estimate
N_LESIONS_FOR_ESTIMATE = 150  # subsample of detected for speed; still >> 32*33/2 dof needed


def main():
    dev = torch.device('cuda')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    e240 = list(csv.DictReader(open(HERE / 'E240_layerwise.csv')))
    seen = set()
    det_lesions = []
    for r in e240:
        if r['role'] != 'detected':
            continue
        key = (r['subject_id'], r['comp_id'])
        if key in seen:
            continue
        seen.add(key)
        det_lesions.append(r)
    rng = np.random.default_rng(2500)
    det_sample = [det_lesions[i] for i in rng.choice(len(det_lesions), size=min(N_LESIONS_FOR_ESTIMATE, len(det_lesions)), replace=False)]
    print(f'Estimating pooled correlation from {len(det_sample)} detected lesions', flush=True)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    all_vox = []
    t0 = time.time()
    n_done = 0
    for r in det_sample:
        sid, cid = r['subject_id'], int(r['comp_id'])
        if sid not in sid_to_idx:
            continue
        img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
        brain_mask = img_c[0] != 0
        cm, shell = get_masks(tgt_c, cid, brain_mask)
        if cm is None:
            continue
        img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
        stages = forward_to_dec1_internal(model, img_t)
        bn1 = stages['bn1_out'][0].cpu().numpy()  # (32, D, H, W)
        vox = bn1[:, cm].T  # (n_vox, 32)
        if vox.shape[0] > MAX_VOX_PER_LESION:
            idx = rng.choice(vox.shape[0], size=MAX_VOX_PER_LESION, replace=False)
            vox = vox[idx]
        all_vox.append(vox)
        n_done += 1
        if n_done % 20 == 0:
            print(f'  {n_done}/{len(det_sample)} ({time.time()-t0:.0f}s)', flush=True)

    all_vox = np.concatenate(all_vox, axis=0)  # (total_vox, 32)
    print(f'\nTotal pooled voxels: {all_vox.shape[0]}', flush=True)

    corr = np.corrcoef(all_vox, rowvar=False)  # (32, 32)
    eigvals = np.linalg.eigvalsh(corr)
    print(f'Correlation matrix eigenvalues: min={eigvals.min():.4e}  max={eigvals.max():.4f}')
    print(f'Well-conditioned (min eigval > 1e-6): {eigvals.min() > 1e-6}')

    # small ridge regularization for numerical safety in the later Cholesky step
    corr_reg = corr + np.eye(32) * 1e-4
    np.save(HERE / 'E250_detected_correlation.npy', corr_reg)
    print(f'\nSaved E250_detected_correlation.npy ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
