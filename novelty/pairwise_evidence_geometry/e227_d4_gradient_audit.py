"""E227 -- Lesion-conditioned D4 gradient audit. Tests the missing middle
E226 left open: does D4 target geometry (specifically Q_l, E226's cleanest
survivor) translate into an actual optimization signal, or is the
correlation between Q_l and missedness driven by something else?

    E226 established (correlational, size/isolation/distance-controlled):
        D4 target geometry <-> persistent non-learning
    E227 tests the missing middle:
        D4 target geometry -> gradient contribution -> non-learning ?

THREE PRE-REGISTERED OUTCOMES (per this session's own framing, stated
before running):
  (1) low Q_l -> low |G_l^D4| (gradient MAGNITUDE starved) -> plausible
      causal training mechanism, worth pursuing a targeted fix.
  (2) low Q_l -> no meaningful gradient difference -> KILL the D4-gradient
      hypothesis; Q_l correlates with missedness via something else.
  (3) low Q_l -> gradient present, comparable magnitude, but WRONG-
      DIRECTIONED relative to what the main segmentation loss wants for
      that same lesion (measured via cosine similarity, not magnitude) ->
      a distinct "unhelpful signal" mechanism, not ordinary starvation.
      This project's own history (SC-TAM/E25, CC-DiceCE/E103, E225) has
      repeatedly found signal-QUALITY/direction problems where magnitude
      looked fine -- outcome (3) is not a remote possibility, it is this
      project's own most common failure mode for loss-term mechanisms.

METHOD: uses the SAME frozen E131 checkpoint E223/E226 both used (matching
convention -- detected/missed labels are defined ON this exact model, so
gradients must be measured on the SAME model, not a fresh/retrained one).
For each GT ET component l (MIN_VOX=5, same convention throughout this
session), on a representative simulated patch draw that includes it:

  1. Forward pass through the model (gradients enabled).
  2. Compute a LESION-MASKED D4 loss: focal_tversky on the ET channel of
     aux_probs3/t_d4, but with both tensors ZEROED outside the D4 cells
     THIS lesion's own voxels touch (reusing E226's exact cell-membership
     logic) -- isolates the scalar loss contribution attributable to only
     this lesion's own supervision signal, not the whole patch's.
  3. Backprop that scalar; record:
       G_l^head  = ||d(L_D4,masked)/d(aux_probs3)||  at the D4 prediction
                   itself (post-sigmoid, pre-any-further-processing)
       G_l^dec3  = ||d(L_D4,masked)/d(dec3)||         at the shared
                   decoder representation the aux head reads from
  4. SEPARATELY compute the main segmentation loss's gradient at dec1 for
     the SAME lesion (masked to its full-resolution voxels, same idea),
     then take dec3's gradient direction from an equivalent seg-loss-only
     backward pass restricted to this lesion's full-res footprint, for a
     cosine-similarity comparison at a shared measurement point (dec3 is
     upstream of BOTH the aux head and, via upconv2/dec2/dec1, the main
     seg path -- a valid shared point for direction comparison).

Only lesions actually included in the simulated draw are usable (same
"given the lesion is inside a training patch" framing as E224/E226).
ONE representative draw per lesion (not 500 Monte Carlo draws -- gradient
computation is much more expensive than the pure numpy/torch pooling
E226 needed; this is an intentional cost/precision tradeoff, disclosed
here, not hidden) -- the FIRST simulated draw (of 50, seeded
deterministically) that includes the lesion is used.

SCALE: measured cost is ~3s/lesion (two full forward+backward passes
through the whole U-Net), making the full 2999-lesion/1126-subject sweep
~5-6 hours -- too slow for a diagnostic. Scoped down to a STRATIFIED
~300-lesion subsample (60/size-quintile, reusing E226's own already-
computed lesion inventory and size quintile edges, rng seed 42, written to
E227_sample_keys.csv) so small/large lesions stay proportionally
represented rather than risking an unrepresentative subject-order subset.
Only lesions in this sample set get the expensive gradient computation;
all others are skipped after the cheap component-labeling step (which is
needed regardless, to find comp_id numbering matching E226/E223's own
convention).
"""
import sys, os, csv, time
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
from e223_exposure_audit import simulate_patch_draws, FG_BIAS, verify_fg_bias  # noqa: E402
import importlib.util

spec = importlib.util.spec_from_file_location(
    "t", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["e"]; spec.loader.exec_module(t); sys.argv = _a

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
MIN_VOX = 5
ET = 0
K = 4
N_SIM_FOR_INCLUSION = 50  # smaller than E226's 500 -- only need ONE usable draw per lesion


def focal_tversky_masked(probs_ch, target_ch, cell_mask, alpha=0.5, beta=0.5,
                         gamma=4.0 / 3.0, eps=1e-6):
    """Single-channel FocalTversky (matches train_e130's focal_tversky
    exactly, restricted to ONE region channel), computed ONLY over voxels
    where cell_mask is True -- everything else contributes exactly zero
    to tp/fp/fn, so backprop attributes gradient only to the masked
    region. probs_ch, target_ch, cell_mask: (B,1,d,h,w) or (B,d,h,w)."""
    p = probs_ch * cell_mask
    tgt = target_ch * cell_mask
    dims = tuple(range(1, p.dim()))
    tp = (p * tgt).sum(dims)
    fp = (p * (1 - tgt) * cell_mask).sum(dims)
    fn = ((1 - p) * tgt).sum(dims)
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1 - ti) ** gamma).mean()


def lesion_d4_cells(voxels_patch_local, K, grid):
    """voxels_patch_local: (n,3) int, this lesion's own voxels in
    patch-local coords (already filtered to inside the patch). Returns a
    (grid,grid,grid) boolean D4-cell mask: True for cells this lesion's
    own voxels touch. Same cell-index logic as E226 (voxel // K)."""
    cz = voxels_patch_local[:, 0] // K
    cy = voxels_patch_local[:, 1] // K
    cx = voxels_patch_local[:, 2] // K
    mask = np.zeros((grid, grid, grid), dtype=bool)
    mask[cz, cy, cx] = True
    return mask


def main():
    verify_fg_bias()
    dev = torch.device('cuda')
    smoke = '--smoke' in sys.argv

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    # NOTE: parameters stay trainable (requires_grad=True by default) --
    # we need gradients w.r.t. intermediate ACTIVATIONS (aux_probs3, dec3),
    # not parameters, but the forward pass must build a full autograd
    # graph, which requires no .requires_grad_(False) freeze here (unlike
    # E223/E226, which only needed inference). model.eval() still applied
    # for correct BatchNorm/dropout behaviour (matches production eval
    # convention), autograd works fine in eval mode.

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=t.PATCH, fg_bias=FG_BIAS)
    n_total = 5 if smoke else len(ds)

    sample_keys = None
    if not smoke:
        sample_path = HERE / 'E227_sample_keys.csv'
        sample_keys = set()
        with open(sample_path) as f:
            for row in csv.DictReader(f):
                sample_keys.add((row['subject_id'], row['comp_id']))
        print(f'E227 D4 gradient audit, {n_total} TRAIN subjects (scanned), '
              f'{len(sample_keys)} lesions in STRATIFIED SAMPLE (gradient-computed), '
              f'{N_SIM_FOR_INCLUSION} candidate draws/subject, patch={t.PATCH}', flush=True)
    else:
        print(f'E227 D4 gradient audit SMOKE, {n_total} TRAIN subjects, '
              f'{N_SIM_FOR_INCLUSION} candidate draws/subject, patch={t.PATCH}', flush=True)

    out = HERE / ('E227_smoke.csv' if smoke else 'E227_gradient.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=[
        'subject_id', 'comp_id', 'size', 'Q_l',
        'G_head_norm', 'G_dec3_aux_norm', 'G_dec3_seg_norm',
        'cos_dec3_aux_vs_seg', 'detected'])
    w.writeheader()

    sample_subjects = None
    if sample_keys is not None:
        sample_subjects = set(sid for sid, _ in sample_keys)

    t0 = time.time()
    pd, ph, pw = t.PATCH
    grid = pd // K
    n_measured = 0
    for ii in range(n_total):
        # PERFORMANCE: skip subjects with NO lesion in the stratified sample
        # entirely, before paying for _load_subject/simulate_patch_draws/
        # ndi.label -- only 229/1126 subjects have a sampled lesion, so
        # this avoids ~80% of the per-subject overhead for no benefit.
        if sample_subjects is not None:
            sid_probe = os.path.basename(ds.subject_dirs[ii])
            if sid_probe not in sample_subjects:
                continue
        starts, centres, image, target, sid = simulate_patch_draws(
            ds, ii, t.PATCH, N_SIM_FOR_INCLUSION, rng_seed_base=30_000 + ii * 100_000)
        Y = target > 0.5
        et_lbl, et_n = ndimage.label(Y[ET])
        starts_arr = np.array(starts)

        for g in range(1, et_n + 1):
            cm = et_lbl == g
            sz = int(cm.sum())
            if sz < MIN_VOX:
                continue
            if sample_keys is not None and (sid, str(g)) not in sample_keys:
                continue  # not in the stratified sample -- skip the expensive gradient work
            lz, ly, lx = np.where(cm)
            l_lo = np.array([lz.min(), ly.min(), lx.min()])
            l_hi = np.array([lz.max(), ly.max(), lx.max()])
            voxels_native = np.stack([lz, ly, lx], axis=1)

            # find the FIRST draw that includes this lesion
            p_lo = starts_arr; p_hi = starts_arr + np.array([pd, ph, pw]) - 1
            bbox_ok = np.where(np.all(p_lo <= l_hi, axis=1) & np.all(p_hi >= l_lo, axis=1))[0]
            chosen = None
            for k in bbox_ok:
                start = starts_arr[k]
                local = voxels_native - start[None, :]
                inside = np.all((local >= 0) & (local < np.array([pd, ph, pw])), axis=1)
                if inside.any():
                    chosen = (k, local[inside])
                    break
            if chosen is None:
                continue  # lesion never actually included in any candidate draw
            k, local_voxels = chosen
            start = starts_arr[k]

            # --- crop full patch (image + target) at this draw's position ---
            z0, y0, x0 = start
            img_c = image[:, z0:z0+pd, y0:y0+ph, x0:x0+pw]
            tgt_c = target[:, z0:z0+pd, y0:y0+ph, x0:x0+pw]
            if img_c.shape[1:] != (pd, ph, pw):
                img_p = np.zeros((img_c.shape[0], pd, ph, pw), dtype=np.float32)
                tgt_p = np.zeros((tgt_c.shape[0], pd, ph, pw), dtype=np.float32)
                img_p[:, :img_c.shape[1], :img_c.shape[2], :img_c.shape[3]] = img_c
                tgt_p[:, :tgt_c.shape[1], :tgt_c.shape[2], :tgt_c.shape[3]] = tgt_c
                img_c, tgt_c = img_p, tgt_p

            img_t = torch.from_numpy(img_c).unsqueeze(0).to(dev)
            tgt_t = torch.from_numpy(tgt_c).unsqueeze(0).to(dev)

            # --- D4 cell mask for this lesion (patch-local) ---
            cell_mask_np = lesion_d4_cells(local_voxels, K, grid)
            cell_mask = torch.from_numpy(cell_mask_np).float().unsqueeze(0).unsqueeze(0).to(dev)  # (1,1,g,g,g)

            # --- full-res voxel mask for this lesion (patch-local), for the seg-side comparison ---
            vox_mask_np = np.zeros((pd, ph, pw), dtype=np.float32)
            vox_mask_np[local_voxels[:, 0], local_voxels[:, 1], local_voxels[:, 2]] = 1.0
            vox_mask = torch.from_numpy(vox_mask_np).unsqueeze(0).unsqueeze(0).to(dev)  # (1,1,D,H,W)

            # Q_l for this specific draw (matches E226's definition, single-draw not MC-averaged)
            with torch.no_grad():
                out_nograd = model(img_t)
                probs_full = out_nograd['probs']
                et_full_ch = (tgt_t[:, ET:ET+1] > 0.5).float()
                t_d4_full = F.avg_pool3d(et_full_ch, kernel_size=K, stride=K)
                cz = local_voxels[:, 0] // K; cy = local_voxels[:, 1] // K; cx = local_voxels[:, 2] // K
                cell_vals = t_d4_full[0, 0, cz, cy, cx].cpu().numpy()
                Q_l = float(cell_vals.mean())
                pred_et = (probs_full[0, ET] > 0.5)
                detected = int((pred_et.cpu().numpy() & (vox_mask_np > 0)).sum() / sz >= 0.5)

            # === Pass 1: D4 aux-loss gradient, masked to this lesion's cells ===
            model.zero_grad(set_to_none=True)
            out1 = model(img_t)
            aux_probs3 = out1['aux_probs3']  # (1,3,g,g,g)
            dec3_aux = out1['dec3']
            dec3_aux.retain_grad()
            aux_probs3.retain_grad()

            t_d4 = F.avg_pool3d(tgt_t, kernel_size=K, stride=K)  # (1,3,g,g,g)
            L_d4_masked = focal_tversky_masked(
                aux_probs3[:, ET:ET+1], t_d4[:, ET:ET+1], cell_mask)
            if not torch.isfinite(L_d4_masked):
                continue
            L_d4_masked.backward()
            G_head = float(aux_probs3.grad[:, ET:ET+1].norm().item()) if aux_probs3.grad is not None else 0.0
            G_dec3_aux = float(dec3_aux.grad.norm().item()) if dec3_aux.grad is not None else 0.0

            # === Pass 2: main seg-loss gradient at dec3, masked to this lesion's full-res voxels ===
            model.zero_grad(set_to_none=True)
            out2 = model(img_t)
            probs2 = out2['probs']
            dec3_seg = out2['dec3']
            dec3_seg.retain_grad()

            L_seg_masked = focal_tversky_masked(probs2[:, ET:ET+1], tgt_t[:, ET:ET+1], vox_mask)
            if not torch.isfinite(L_seg_masked):
                continue
            L_seg_masked.backward()
            G_dec3_seg = float(dec3_seg.grad.norm().item()) if dec3_seg.grad is not None else 0.0

            cos_sim = float('nan')
            if dec3_aux.grad is not None and dec3_seg.grad is not None:
                ga = dec3_aux.grad.reshape(1, -1)
                gs = dec3_seg.grad.reshape(1, -1)
                if ga.norm() > 0 and gs.norm() > 0:
                    cos_sim = float(F.cosine_similarity(ga, gs).item())

            w.writerow({'subject_id': sid, 'comp_id': g, 'size': sz, 'Q_l': Q_l,
                       'G_head_norm': G_head, 'G_dec3_aux_norm': G_dec3_aux,
                       'G_dec3_seg_norm': G_dec3_seg, 'cos_dec3_aux_vs_seg': cos_sim,
                       'detected': detected})
            n_measured += 1
        fh.flush()
        if (ii + 1) % 25 == 0 or smoke:
            print(f'  {ii+1}/{n_total} subj, {n_measured} lesions measured '
                  f'({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'wrote {out.name}: {n_measured} lesions ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
