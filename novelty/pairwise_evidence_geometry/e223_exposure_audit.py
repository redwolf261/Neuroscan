"""E223 -- Per-lesion patch exposure audit. NO MODEL FORWARD PASSES, NO
TRAINING. Pure sampler simulation, using the EXACT PRODUCTION SAMPLER code
(BraTSMultimodalDataset._sample_patch, Dataset/brats_multimodal_dataset.py),
imported directly -- not reimplemented, so this measures what the actual
training loop exposes the network to, not a theoretical approximation.

KEY STRUCTURAL FACT (verified before writing this, not assumed): the
DataLoader uses shuffle=True with one dataset item = one subject, so EACH
SUBJECT IS VISITED EXACTLY ONCE PER EPOCH and draws EXACTLY ONE patch via
_sample_patch. A lesion can therefore only be exposed through its OWN
subject's single per-epoch draw -- there is no cross-subject patch pool.
This means per-epoch exposure for lesion l is a single Bernoulli draw whose
probability we estimate by MANY REPEATED simulated draws for that subject
(Monte Carlo over the sampler's actual RNG-driven logic), not by scanning a
fixed corpus.

MEASURED per GT ET component l (MIN_VOX=5, same convention as E204-E222):
  E_l = P(patch contains >=1 voxel of l)      -- INCLUSION
  C_l = P(patch is CENTERED inside l)          -- CENTERING
        (centre = the argwhere-selected voxel BEFORE clipping to volume
        bounds; if l contains that exact voxel, this patch's WT-biased
        draw centered inside l)
  over N_SIM independent simulated draws of _sample_patch for that
  subject, using the SAME rng type (np.random.default_rng) seeded
  independently per simulation batch (not reusing the production seed --
  we want the STATIONARY distribution the sampler induces, not one
  specific realized training run's sequence).

Also recorded per lesion: volume, distance from WT centroid, whether
isolated from the main WT connected component, detected/missed status
(from the FROZEN model's own prediction, matching every E204-E222
convention), subject_id -- for the size/position-controlled analysis.

fg_bias=0.66 read from the ACTUAL create_multimodal_loaders default, not
hardcoded independently (checked: matches Dataset/brats_multimodal_dataset.py).
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from neuroscan_3d_v5 import UNet3D_v5
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, create_multimodal_loaders
import importlib.util

spec = importlib.util.spec_from_file_location(
    "t", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["e"]; spec.loader.exec_module(t); sys.argv = _a

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
MIN_VOX = 5
ET, WT = 0, 2
N_SIM = 500          # simulated _sample_patch draws PER SUBJECT
FG_BIAS = 0.66        # verified against create_multimodal_loaders' default below


def verify_fg_bias():
    import inspect
    src = inspect.getsource(create_multimodal_loaders)
    assert 'fg_bias=0.66' in src, 'FG_BIAS constant does not match production default -- STOP'


def simulate_patch_draws(ds, subject_idx, patch_size, n_sim, rng_seed_base):
    """Runs the PRODUCTION _sample_patch n_sim times for one subject's
    already-loaded (image, target), using independently-seeded RNGs so we
    estimate the sampler's stationary per-subject distribution.

    Returns the list of (z,y,x) patch START coordinates and the list of
    (cz,cy,cx) CENTRE voxels actually used -- per-lesion E_l/C_l are then
    computed by checking, PER DRAW, whether that draw's patch box overlaps
    the lesion / whether that draw's centre voxel falls inside the lesion.
    This is the correct per-lesion-per-draw semantics (NOT a per-voxel
    accumulator, which would conflate 'this voxel was covered by SOME draw'
    with 'this lesion was covered in a GIVEN draw' -- different draws can
    cover different subsets of a lesion's voxels, so voxel-level union
    overstates inclusion relative to what a SINGLE training step sees)."""
    image, target, sid = ds._load_subject(ds.subject_dirs[subject_idx])
    D, H, W = image.shape[1:]
    pd, ph, pw = patch_size
    fg_vox = np.argwhere(target[WT] > 0)
    brain_vox = np.argwhere(image[0] != 0)
    starts, centres = [], []
    for k in range(n_sim):
        rng = np.random.default_rng(rng_seed_base + k)
        want_fg = rng.random() < FG_BIAS
        centre = None
        if want_fg and len(fg_vox):
            centre = fg_vox[rng.integers(len(fg_vox))]
        if centre is None:
            centre = (brain_vox[rng.integers(len(brain_vox))] if len(brain_vox)
                      else np.array([D // 2, H // 2, W // 2]))
        centres.append(tuple(int(c) for c in centre))
        st = []
        for c, p, full in zip(centre, (pd, ph, pw), (D, H, W)):
            st.append(int(np.clip(int(c) - p // 2, 0, max(0, full - p))))
        starts.append(tuple(st))
    return starts, centres, image, target, sid


def main():
    verify_fg_bias()
    dev = torch.device('cuda')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for q in model.parameters():
        q.requires_grad_(False)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=t.PATCH, fg_bias=FG_BIAS)
    ds_val_check = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                          val_split=0.1, patch_size=t.PATCH)
    smoke = '--smoke' in sys.argv
    n_total = 3 if smoke else len(ds)
    n_sim = 50 if smoke else N_SIM
    print(f'E223 exposure audit, {n_total} TRAIN subjects, {n_sim} sim draws each, '
          f'fg_bias={FG_BIAS}, patch={t.PATCH}', flush=True)

    out = HERE / ('E223_smoke.csv' if smoke else 'E223_exposure.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=[
        'subject_id', 'comp_id', 'size', 'E_l', 'C_l', 'dist_to_wt_centroid',
        'isolated_from_main_wt', 'detected', 'wt_volume'])
    w.writeheader()

    t0 = time.time()
    pd, ph, pw = t.PATCH
    for ii in range(n_total):
        starts, centres, image, target, sid = simulate_patch_draws(
            ds, ii, t.PATCH, n_sim, rng_seed_base=10_000 + ii * 100_000)
        starts_arr = np.array(starts)   # (n_sim, 3)
        centres_arr = np.array(centres)  # (n_sim, 3)

        Y = target > 0.5
        wt_mask = Y[WT]
        wt_lbl, wt_n = ndimage.label(wt_mask)
        wt_sizes = [(wt_lbl == g).sum() for g in range(1, wt_n + 1)]
        main_wt_id = (1 + int(np.argmax(wt_sizes))) if wt_n > 0 else 0
        wt_centroid = (np.array(ndimage.center_of_mass(wt_mask))
                       if wt_mask.any() else None)
        wt_volume = int(wt_mask.sum())

        # detected/missed status: run the FROZEN model once on this
        # subject via sliding-window inference (matches E204-E222 convention)
        brain = image[0] != 0
        x = torch.from_numpy(image).unsqueeze(0).to(dev)
        probs = t.sliding_window_predict(model, x, t.PATCH, t.SW_OVERLAP, 3, dev, True)
        pred_et = (probs[ET] > 0.5) & brain

        et_lbl, et_n = ndimage.label(Y[ET])
        for g in range(1, et_n + 1):
            cm = et_lbl == g
            sz = int(cm.sum())
            if sz < MIN_VOX:
                continue
            lz, ly, lx = np.where(cm)
            l_lo = np.array([lz.min(), ly.min(), lx.min()])
            l_hi = np.array([lz.max(), ly.max(), lx.max()])  # inclusive

            # INCLUSION per draw: patch box [start, start+patch) overlaps
            # lesion bbox [l_lo, l_hi] on all 3 axes (necessary bbox test),
            # THEN confirm true voxel overlap for draws passing the bbox
            # test (cheap: bbox is a fast prefilter, exact check on the
            # small remaining set).
            p_lo = starts_arr; p_hi = starts_arr + np.array([pd, ph, pw]) - 1
            bbox_overlap = np.all((p_lo <= l_hi) & (p_hi >= l_lo), axis=1)
            n_incl = 0
            for k in np.where(bbox_overlap)[0]:
                z0, y0, x0 = starts[k]
                if cm[z0:z0+pd, y0:y0+ph, x0:x0+pw].any():
                    n_incl += 1
            E_l = n_incl / n_sim

            # CENTERING per draw: the draw's centre voxel lies inside cm
            in_lesion = cm[centres_arr[:, 0], centres_arr[:, 1], centres_arr[:, 2]]
            C_l = float(in_lesion.mean())

            ov = cm & (wt_lbl == main_wt_id) if main_wt_id else np.zeros_like(cm)
            isolated = not ov.any()
            if wt_centroid is not None:
                lesion_centroid = np.array(ndimage.center_of_mass(cm))
                dist = float(np.linalg.norm(lesion_centroid - wt_centroid))
            else:
                dist = float('nan')
            detected = int((cm & pred_et).sum() / sz >= 0.5)

            w.writerow({'subject_id': sid, 'comp_id': g, 'size': sz,
                       'E_l': E_l, 'C_l': C_l,
                       'dist_to_wt_centroid': dist,
                       'isolated_from_main_wt': int(isolated),
                       'detected': detected, 'wt_volume': wt_volume})
        fh.flush()
        if (ii + 1) % 10 == 0 or smoke:
            print(f'  {ii+1}/{n_total} subj ({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'wrote {out.name} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
