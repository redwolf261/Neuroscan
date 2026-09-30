"""E238-H6 -- contextual-competition causal test, per the user's exact
spec (pasted_content, 2026-09-29). MEASUREMENT/INTERVENTION only: frozen
E131 checkpoint throughout, no weights touched.

QUESTION: does P(ET | x_local) differ from P(ET | x_local, x_context)?
I.e. does the surrounding spatial context causally change the model's
prediction on a FIXED local lesion patch, independent of the lesion's own
raw evidence.

METHOD -- context swap at the INPUT level (the cleanest, assumption-free
causal test; no new architecture, no probe, no injected direction):
  1. For each G2 lesion (high_contrast_missed, from e235's build_groups),
     take its own 128^3 native-image patch (RECIPIENT).
  2. Cut out a small LOCAL cube (fixed size, centered on the lesion
     centroid) containing the lesion's own true image content --
     UNCHANGED.
  3. Replace everything OUTSIDE that local cube with DONOR context: the
     corresponding region from a DIFFERENT subject's image, one that
     contains a DETECTED ET lesion of similar local evidence strength
     (t1c contrast/size, matched via the same z-scored nearest-neighbor
     convention as E234).
  4. Run the frozen model on this hybrid volume. Measure the prediction
     on the ORIGINAL local lesion voxels (mask position taken from the
     recipient, unchanged).
  5. Compare against baseline (recipient's own unmodified volume) and
     against a PERMUTATION-NULL control (donor context taken from a
     RANDOM subject/location, not evidence-matched) -- per this
     session's established discipline (E231's marginal-bias lesson): a
     "real" effect must beat a null, not just be nonzero.

TWO DONOR-MATCHING CONDITIONS, run as SEPARATE rows (per explicit user
answer to this turn's AskUserQuestion):
  - same_region: donor drawn from detected lesions whose centroid lies in
    a similar normalized anatomical location (same coarse hemisphere +
    similar normalized z/y/x octant of the brain bounding box) AND
    similar t1c/size.
  - loose_match: donor matched on t1c/size ONLY, ignoring location.

H6 SIGNATURE: local-voxel prediction (mean/max ET prob over the FIXED
lesion mask) changes SIGNIFICANTLY when context is swapped in, relative
to a random-context permutation-null -- for EITHER or both donor
conditions. If prediction barely moves regardless of context (same as
random-context null), local evidence alone already determines the
outcome and H6 dies.

LESION CENTROID / ANATOMICAL LOCATION: no prior experiment cached this
(checked E223/E226/E227/E230's own CSV headers -- none have voxel
coordinates), so it is computed FRESH here, once, directly from each
subject's native target volume (normalized centroid position as a
fraction of that subject's own brain bounding box, so it is comparable
across subjects of slightly different crop/extent).
"""
import sys, csv, time, os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from neuroscan_3d_v5 import UNet3D_v5  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402
from e235_linear_probe_layerwise import build_groups  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
ET = 0
PATCH = (128, 128, 128)
LOCAL_CUBE = 24  # fixed local cube half-extent in voxels around centroid (48^3 total)
MIN_VOX = 5
N_DONOR_CANDIDATES_PER_LESION = 3  # average over a few donors, not just one (variance control)
RNG = np.random.default_rng(777)


def load_patch_and_brain_bbox(ds, sid_to_idx, sid):
    image, target, _ = ds._load_subject(ds.subject_dirs[sid_to_idx[sid]])
    D, H, W = image.shape[1:]
    pd_, ph_, pw_ = PATCH
    cd, ch, cw = D // 2, H // 2, W // 2
    sd_, ed_ = max(0, cd - pd_//2), min(D, cd + pd_//2)
    sh_, eh_ = max(0, ch - ph_//2), min(H, ch + ph_//2)
    sw_, ew_ = max(0, cw - pw_//2), min(W, cw + pw_//2)
    img_c = image[:, sd_:ed_, sh_:eh_, sw_:ew_]
    tgt_c = target[:, sd_:ed_, sh_:eh_, sw_:ew_]
    if img_c.shape[1:] != (pd_, ph_, pw_):
        ip = np.zeros((4, pd_, ph_, pw_), dtype=np.float32)
        tp_ = np.zeros((3, pd_, ph_, pw_), dtype=np.float32)
        ip[:, :img_c.shape[1], :img_c.shape[2], :img_c.shape[3]] = img_c
        tp_[:, :tgt_c.shape[1], :tgt_c.shape[2], :tgt_c.shape[3]] = tgt_c
        img_c, tgt_c = ip, tp_
    # brain bbox within the 128^3 patch: nonzero t1c signal
    brain_mask = img_c[0] > img_c[0].mean() * 0.1  # loose foreground mask
    if brain_mask.sum() < 100:
        brain_mask = np.ones_like(brain_mask, dtype=bool)
    coords = np.argwhere(brain_mask)
    bbox_min = coords.min(axis=0); bbox_max = coords.max(axis=0)
    return img_c, tgt_c, bbox_min, bbox_max


def component_mask_and_centroid(tgt_c, comp_id):
    et_lbl, et_n = ndimage.label(tgt_c[ET] > 0.5)
    if comp_id < 1 or comp_id > et_n:
        return None, None
    cm = et_lbl == comp_id
    if cm.sum() < MIN_VOX:
        return None, None
    centroid = np.array(ndimage.center_of_mass(cm))
    return cm, centroid


def normalized_location(centroid, bbox_min, bbox_max):
    span = (bbox_max - bbox_min).astype(float)
    span[span == 0] = 1.0
    return (centroid - bbox_min) / span  # in [0,1]^3, roughly


def local_contrast_t1c(img_c, cm):
    dil = ndimage.binary_dilation(cm, iterations=3)
    shell = dil & (~cm)
    if shell.sum() < 5:
        return 0.0
    return float((img_c[0][cm].mean() - img_c[0][shell].mean()) / (img_c[0][shell].std() + 1e-6))


def build_hybrid(recipient_img, donor_img, centroid, half=LOCAL_CUBE):
    """Returns hybrid image: donor everywhere, EXCEPT a cube of size
    (2*half)^3 centered at `centroid` (in recipient voxel coords), which
    keeps the recipient's own original content."""
    D, H, W = recipient_img.shape[1:]
    cz, cy, cx = [int(round(c)) for c in centroid]
    z0, z1 = max(0, cz-half), min(D, cz+half)
    y0, y1 = max(0, cy-half), min(H, cy+half)
    x0, x1 = max(0, cx-half), min(W, cx+half)
    hybrid = donor_img.copy()
    hybrid[:, z0:z1, y0:y1, x0:x1] = recipient_img[:, z0:z1, y0:y1, x0:x1]
    return hybrid


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    det, high_contrast_missed, matched_missed, t1c_median = build_groups()
    print(f'E238-H6: G2 (high_contrast_missed, n={len(high_contrast_missed)}) recipients; '
          f'donor pool = detected (n={len(det)})', flush=True)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    recipients = high_contrast_missed[:12] if smoke else high_contrast_missed

    # ---- precompute donor pool GEOMETRY ONLY (sid/comp_id/centroid/loc/
    # size/ct1c) -- NOT the full image array. Holding all 1665 detected
    # lesions' 128^3x4-channel float32 patches simultaneously (~8.4MB each,
    # ~14GB total) caused repeated CUDA OOM crashes (see prior run
    # attempts) even though the arrays themselves are plain numpy, not
    # torch tensors -- the CPU memory pressure was starving the CUDA
    # allocator's own pinned-memory/driver overhead. Donor IMAGES are now
    # loaded on demand, only for the handful actually chosen per
    # recipient (pick_donors returns <=N_DONOR_CANDIDATES_PER_LESION), and
    # discarded immediately after use. ----
    print('Precomputing donor pool geometry (no image data held)...', flush=True)
    donor_pool = []
    det_iter = det[:60] if smoke else det
    for r in det_iter:
        sid = r['subject_id']
        if sid not in sid_to_idx:
            continue
        img_c, tgt_c, bbox_min, bbox_max = load_patch_and_brain_bbox(ds, sid_to_idx, sid)
        cm, centroid = component_mask_and_centroid(tgt_c, int(r['comp_id']))
        if cm is None:
            continue
        loc = normalized_location(centroid, bbox_min, bbox_max)
        ct1c = local_contrast_t1c(img_c, cm)
        donor_pool.append({'sid': sid, 'comp_id': int(r['comp_id']),
                           'centroid': centroid, 'loc': loc, 'size': int(cm.sum()), 'ct1c': ct1c})
        del img_c, tgt_c
    print(f'  donor pool: {len(donor_pool)} usable detected lesions (geometry only)', flush=True)

    def pick_donors(recip_loc, recip_ct1c, recip_size, mode, k):
        cands = list(donor_pool)
        if mode == 'same_region':
            dists = [np.linalg.norm(np.array(c['loc']) - np.array(recip_loc)) for c in cands]
            order = np.argsort(dists)[:max(30, k*5)]
            cands = [cands[i] for i in order]
        feat_r = np.array([np.log(recip_size+1), recip_ct1c])
        scored = []
        for c in cands:
            feat_c = np.array([np.log(c['size']+1), c['ct1c']])
            d = np.linalg.norm(feat_c - feat_r)
            scored.append((d, c))
        scored.sort(key=lambda x: x[0])
        return [c for _, c in scored[:k]]

    def pick_random_donors(k):
        idxs = RNG.choice(len(donor_pool), size=min(k, len(donor_pool)), replace=False)
        return [donor_pool[i] for i in idxs]

    out = HERE / ('E238_h6_smoke.csv' if smoke else 'E238_h6_context_swap.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=[
        'subject_id', 'comp_id', 'size', 'condition', 'donor_sid', 'donor_comp',
        'mean_prob', 'max_prob'])
    w.writeheader()

    t0 = time.time()
    n_done = 0
    for r in recipients:
        sid = r['subject_id']
        if sid not in sid_to_idx:
            continue
        img_c, tgt_c, bbox_min, bbox_max = load_patch_and_brain_bbox(ds, sid_to_idx, sid)
        cm, centroid = component_mask_and_centroid(tgt_c, int(r['comp_id']))
        if cm is None:
            continue
        loc = normalized_location(centroid, bbox_min, bbox_max)
        ct1c = local_contrast_t1c(img_c, cm)
        sz = int(cm.sum())
        mask_t = torch.from_numpy(cm).to(dev)

        conditions = {
            'baseline': [{'sid': sid, 'comp_id': r['comp_id'], 'img': img_c}],
            'same_region': pick_donors(loc, ct1c, sz, 'same_region', N_DONOR_CANDIDATES_PER_LESION),
            'loose_match': pick_donors(loc, ct1c, sz, 'loose_match', N_DONOR_CANDIDATES_PER_LESION),
            'random_null': pick_random_donors(N_DONOR_CANDIDATES_PER_LESION),
        }

        for cond, donors in conditions.items():
            for donor in donors:
                if cond == 'baseline':
                    hybrid = img_c
                    donor_sid, donor_comp = '', ''
                else:
                    # donor image loaded ON DEMAND (donor_pool holds
                    # geometry only, see precompute step above) --
                    # discarded right after building the hybrid.
                    donor_img_c, _, _, _ = load_patch_and_brain_bbox(ds, sid_to_idx, donor['sid'])
                    hybrid = build_hybrid(img_c, donor_img_c, centroid)
                    del donor_img_c
                    donor_sid, donor_comp = donor['sid'], donor['comp_id']
                img_t = torch.from_numpy(hybrid).unsqueeze(0).float().to(dev)
                with torch.no_grad():
                    out_m = model(img_t)
                p = out_m['probs'][0, ET]
                mean_p = float(p[mask_t].mean().item())
                max_p = float(p[mask_t].max().item())
                w.writerow({'subject_id': sid, 'comp_id': r['comp_id'], 'size': sz,
                           'condition': cond, 'donor_sid': donor_sid, 'donor_comp': donor_comp,
                           'mean_prob': mean_p, 'max_prob': max_p})
                del img_t, out_m, p
        n_done += 1
        if n_done % 5 == 0 or smoke:
            torch.cuda.empty_cache()
            print(f'  {n_done}/{len(recipients)} recipients ({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'\nE238-H6 complete: {n_done} recipients ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
