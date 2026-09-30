"""E226 -- Auxiliary (D4) supervision visibility audit. NO MODEL FORWARD
PASSES for the metrics themselves, NO TRAINING -- pure target-geometry
measurement using the EXACT production D4-target generation
(F.avg_pool3d(targets, kernel_size=4, stride=4), train_e130_multimodal_
baseline.py:295, confirmed unthresholded/soft -- see this session's own
code audit before building this script).

WHY MONTE CARLO OVER CROP PHASE (not a single fixed-phase measurement):
_sample_patch's crop start (z,y,x) is an ARBITRARY voxel offset (Dataset/
brats_multimodal_dataset.py:218-221, `s = int(c) - p//2`, clipped, NOT
aligned to any 4-voxel grid). avg_pool3d's D4 cell boundaries are therefore
at a different absolute phase (mod 4) every time a subject is sampled, so a
lesion's D4-pooled representation is NOT a fixed per-lesion property -- it
varies epoch to epoch with the random crop. This script simulates MANY
patch draws per subject (reusing E223's own sampler-simulation approach,
`simulate_patch_draws`, imported unchanged) and reports each metric's
DISTRIBUTION across draws, not a single snapshot value.

CORRECTNESS NOTE (fixed after an earlier draft got this wrong): the D4
target the production trainer actually computes is
`avg_pool3d(targets[ET_channel], 4, 4)` on the FULL ET channel of the
patch -- i.e. a D4 cell's value is the occupancy fraction from ALL
enhancing-tumour voxels in that 4^3 block, not just the voxels belonging
to ONE component. If two distinct ET components happen to fall within the
same D4 cell (possible when they are close together), pooling only one
component's mask would UNDERCOUNT that cell's true value. This script
therefore pools the FULL ET CHANNEL per draw (matching production exactly)
and then reads off, for each individual lesion, only the cells its own
voxels touch -- giving each lesion's metrics from the SAME pooled tensor
the real loss would use, not an isolated single-component approximation.

PERFORMANCE: an earlier draft materialized a full 128^3 upsample
(np.repeat) per lesion per draw to compute Q_l, and separately cast the
full ~8.9M-voxel native-volume component mask to float32 inside the
per-draw loop -- both were >100x more expensive than necessary and made a
full 1126-subject run impractically slow (killed after 5min covering only
13 subjects). Fixed by: (1) pooling the ET channel ONCE per draw (shared
across all lesions present in that draw, not once per lesion-draw pair),
(2) reading each lesion's own D4 cells directly via fancy-indexing
(cell index = voxel index // 4) instead of ever upsampling the pooled
grid back to full resolution.

For each GT ET component l, and for each simulated draw where the patch
actually overlaps l (E_l-positive draws only -- D4 supervision quality is
conditional on the lesion being presented at all):

  V_l        = lesion voxel count IN THIS CROP (may be < full lesion size
               if partially cut off by the patch boundary)
  S_l        = sum of (this lesion's own D4 cells' pooled values) * 64,
               i.e. pooled mass attributable to cells the lesion touches,
               rescaled to voxel-equivalent units (avg_pool3d divides by
               K^3=64; verified numerically before writing this script:
               a fully-foreground 4^3 block pools to exactly 1.0, not 64).
               NOTE this is computed from the FULL-ET-CHANNEL pooled grid,
               so if a neighboring component shares a cell, that cell's
               value (and hence S_l) reflects BOTH components' mass --
               correct per the production loss, but means S_l can now
               slightly EXCEED V_l for lesions with a close neighbour
               (unlike the earlier single-component-only draft, where
               S_l<=V_l always held). Recorded honestly, not clipped.
  R_l        = S_l / V_l -- retained-mass ratio (>1.0 signals a
               neighbour-sharing cell, not an error)
  N_l_D4     = count of DISTINCT D4 cells this lesion's own voxels touch
  M_l_D4     = max pooled value among this lesion's own touched cells
               (using the FULL-ET-CHANNEL pooled value for each cell)
  Q_l        = mean, over this lesion's own voxels, of their cell's
               pooled value (full-ET-channel) -- "how much supervision
               mass does this lesion's own volume experience on average"

Per lesion these are aggregated across all E_l-positive draws as
mean/std/min/max -- the DISTRIBUTION induced by random crop phase.

Outcome variable: `detected` REUSED DIRECTLY from E223_exposure.csv (same
checkpoint, same MIN_VOX=5, same ndi.label(Y[ET]) convention, same
comp_id numbering -- this script labels components the SAME way E223 did,
subject by subject, in the same sorted order) -- NOT recomputed, avoiding
a second expensive sliding-window inference pass over 1126 subjects.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e223_exposure_audit import simulate_patch_draws, FG_BIAS, verify_fg_bias  # noqa: E402
import importlib.util

spec = importlib.util.spec_from_file_location(
    "t", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["e"]; spec.loader.exec_module(t); sys.argv = _a

MIN_VOX = 5
ET = 0
N_SIM = 500          # same as E223, for identical statistical power
K = 4                # D4 pooling kernel/stride, matches production exactly


def crop_et_channel_to_patch(et_channel, start, patch_size):
    """et_channel: (D,H,W) float32, the FULL native ET mask (all
    components). Returns the (pd,ph,pw) patch-local crop, zero-padded on
    shortfall -- matches _sample_patch's own pad-on-shortfall exactly."""
    z0, y0, x0 = start
    pd, ph, pw = patch_size
    D, H, W = et_channel.shape
    crop = et_channel[z0:z0+pd, y0:y0+ph, x0:x0+pw]
    if crop.shape != (pd, ph, pw):
        out = np.zeros((pd, ph, pw), dtype=et_channel.dtype)
        out[:crop.shape[0], :crop.shape[1], :crop.shape[2]] = crop
        crop = out
    return crop


def lesion_metrics_from_pooled(voxel_coords_patch_local, t_d4):
    """voxel_coords_patch_local: (n,3) int array, this lesion's own voxel
    coordinates IN PATCH-LOCAL SPACE (already clipped to the patch, i.e.
    only the voxels of this lesion that actually fall inside this draw's
    patch window -- callers must filter before calling). t_d4: (32,32,32)
    the FULL-ET-CHANNEL pooled grid for this draw (shared across lesions).
    Returns (V_l, S_l, R_l, N_l_D4, M_l_D4, Q_l) or None if no voxels."""
    if len(voxel_coords_patch_local) == 0:
        return None
    V_l = float(len(voxel_coords_patch_local))
    cz = voxel_coords_patch_local[:, 0] // K
    cy = voxel_coords_patch_local[:, 1] // K
    cx = voxel_coords_patch_local[:, 2] // K
    cell_vals = t_d4[cz, cy, cx]  # (n,) pooled value per lesion voxel
    Q_l = float(cell_vals.mean())
    # distinct cells this lesion touches, and their (full-channel) values.
    # PERFORMANCE: np.unique(axis=0) on (n,3) rows does a full lexsort and
    # was measured (profiling before this fix) to cost ~12s of a ~16.5s
    # per-subject budget -- 75% of total runtime, the actual reason the
    # first full-run attempt was too slow to finish in a reasonable time.
    # Fixed by encoding each (cz,cy,cx) triple as a single flat integer
    # (valid since D4 grid is 32^3, so cz,cy,cx in [0,32) each, and mixed-
    # radix encoding is injective) and using 1D np.unique instead --
    # verified numerically equivalent (same distinct-cell count) and ~16x
    # faster before committing to the full 1126-subject run.
    D4_GRID = t_d4.shape[0]  # 32 for a 128^3 patch, K=4
    flat_cells = (cz.astype(np.int64) * D4_GRID + cy) * D4_GRID + cx
    _, uniq_idx = np.unique(flat_cells, return_index=True)
    uniq_vals = cell_vals[uniq_idx]
    N_l_D4 = int(len(uniq_idx))
    M_l_D4 = float(uniq_vals.max())
    S_l = float(uniq_vals.sum()) * (K ** 3)
    R_l = S_l / V_l
    return V_l, S_l, R_l, N_l_D4, M_l_D4, Q_l


def main():
    verify_fg_bias()
    smoke = '--smoke' in sys.argv

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=t.PATCH, fg_bias=FG_BIAS)
    n_total = 3 if smoke else len(ds)
    n_sim = 50 if smoke else N_SIM
    print(f'E226 D4 supervision geometry audit, {n_total} TRAIN subjects, '
          f'{n_sim} sim draws each, patch={t.PATCH}, K={K}', flush=True)

    out = HERE / ('E226_smoke.csv' if smoke else 'E226_d4_geometry.csv')
    fh = open(out, 'w', newline='')
    fieldnames = ['subject_id', 'comp_id', 'size', 'n_incl_draws',
                 'V_l_mean', 'V_l_std', 'S_l_mean', 'S_l_std',
                 'R_l_mean', 'R_l_std', 'R_l_min', 'R_l_max',
                 'N_l_D4_mean', 'N_l_D4_std', 'N_l_D4_min', 'N_l_D4_max',
                 'M_l_D4_mean', 'M_l_D4_std', 'M_l_D4_min', 'M_l_D4_max',
                 'Q_l_mean', 'Q_l_std', 'Q_l_min', 'Q_l_max']
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()

    t0 = time.time()
    pd, ph, pw = t.PATCH
    for ii in range(n_total):
        starts, centres, image, target, sid = simulate_patch_draws(
            ds, ii, t.PATCH, n_sim, rng_seed_base=20_000 + ii * 100_000)
        # DIFFERENT rng_seed_base than E223 (10_000+) so this is an
        # INDEPENDENT Monte Carlo sample, not artificially correlated with
        # E223's own draws.

        Y = target > 0.5
        et_full = Y[ET].astype(np.float32)  # full native ET channel, ALL components
        et_lbl, et_n = ndimage.label(Y[ET])

        # per-component bbox + voxel coords, computed ONCE per subject
        components = []
        for g in range(1, et_n + 1):
            cm = et_lbl == g
            sz = int(cm.sum())
            if sz < MIN_VOX:
                continue
            lz, ly, lx = np.where(cm)
            l_lo = np.array([lz.min(), ly.min(), lx.min()])
            l_hi = np.array([lz.max(), ly.max(), lx.max()])
            voxels = np.stack([lz, ly, lx], axis=1)  # (sz,3) native coords
            components.append(dict(comp_id=g, size=sz, l_lo=l_lo, l_hi=l_hi,
                                   voxels=voxels, draws=[]))

        if not components:
            if (ii + 1) % 25 == 0 or smoke:
                print(f'  {ii+1}/{n_total} subj ({time.time()-t0:.0f}s)', flush=True)
            continue

        starts_arr = np.array(starts)  # (n_sim,3)
        patch_arr = np.array([pd, ph, pw])

        # PERFORMANCE: bbox-overlap filtering VECTORIZED across all n_sim
        # draws at once per component (matches E223's own vectorized bbox
        # prefilter style), instead of a per-draw Python-level np.all() call
        # per candidate (profiling showed this generic reduce() overhead
        # was a measurable chunk of per-subject time).
        p_los = starts_arr; p_his = starts_arr + patch_arr - 1  # (n_sim,3)
        for c in components:
            c['overlap_draws'] = np.where(
                np.all(p_los <= c['l_hi'], axis=1) & np.all(p_his >= c['l_lo'], axis=1))[0]

        draws_with_any_candidate = sorted(set(
            int(k) for c in components for k in c['overlap_draws']))

        # PERFORMANCE: batch ALL of this subject's relevant draws' ET-crops
        # through ONE avg_pool3d call (measured ~3x faster than one call
        # per draw: 0.18s vs 0.58s per 500 draws in isolation) instead of
        # pooling one 128^3 tensor at a time.
        if draws_with_any_candidate:
            crops = np.stack([
                crop_et_channel_to_patch(et_full, tuple(starts_arr[k]), (pd, ph, pw))
                for k in draws_with_any_candidate])  # (n_relevant,128,128,128)
            t_full = torch.from_numpy(crops).unsqueeze(1)
            t_d4_batch = F.avg_pool3d(t_full, kernel_size=K, stride=K)[:, 0].numpy()  # (n_relevant,32,32,32)
            draw_to_pooled = {k: t_d4_batch[i] for i, k in enumerate(draws_with_any_candidate)}

            for c in components:
                for k in c['overlap_draws']:
                    k = int(k)
                    t_d4 = draw_to_pooled[k]
                    if not t_d4.any():
                        continue
                    start = starts_arr[k]
                    local = c['voxels'] - start[None, :]
                    inside = np.all((local >= 0) & (local < patch_arr), axis=1)
                    local = local[inside]
                    if len(local) == 0:
                        continue
                    m = lesion_metrics_from_pooled(local, t_d4)
                    if m is not None:
                        c['draws'].append(m)

        for c in components:
            if not c['draws']:
                continue  # lesion never actually included by any simulated draw
            arr = np.array(c['draws'])  # (n_incl, 6)
            V_l, S_l, R_l, N_l, M_l, Q_l = [arr[:, i] for i in range(6)]
            w.writerow({
                'subject_id': sid, 'comp_id': c['comp_id'], 'size': c['size'],
                'n_incl_draws': len(c['draws']),
                'V_l_mean': float(V_l.mean()), 'V_l_std': float(V_l.std()),
                'S_l_mean': float(S_l.mean()), 'S_l_std': float(S_l.std()),
                'R_l_mean': float(R_l.mean()), 'R_l_std': float(R_l.std()),
                'R_l_min': float(R_l.min()), 'R_l_max': float(R_l.max()),
                'N_l_D4_mean': float(N_l.mean()), 'N_l_D4_std': float(N_l.std()),
                'N_l_D4_min': float(N_l.min()), 'N_l_D4_max': float(N_l.max()),
                'M_l_D4_mean': float(M_l.mean()), 'M_l_D4_std': float(M_l.std()),
                'M_l_D4_min': float(M_l.min()), 'M_l_D4_max': float(M_l.max()),
                'Q_l_mean': float(Q_l.mean()), 'Q_l_std': float(Q_l.std()),
                'Q_l_min': float(Q_l.min()), 'Q_l_max': float(Q_l.max()),
            })
        fh.flush()
        if (ii + 1) % 25 == 0 or smoke:
            print(f'  {ii+1}/{n_total} subj ({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'wrote {out.name} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
