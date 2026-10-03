"""E279 -- Negative-Distribution Characterization, per the user's exact
spec. Direct follow-up to E278 (Outcome B: SAM-equivalent negative
sampling genuinely helps, but R1-B's own construction beats it at
every FP budget, especially the tightest). Before any novelty claim,
the user's explicit next question: WHAT specifically differs between
R1-B's local negatives (N_shell, lesion-boundary-anchored dilation)
and SAM-equivalent's local negatives (N_sam_local, random spatial
offset, NOT anatomy-aware)? Is R1-B better because it samples a
DIFFERENT distribution of negatives, or because the recovery
objective itself creates the advantage (same objective is used for
both, per E278, so this experiment characterizes the DISTRIBUTIONS
themselves).

CORE 4 AXES (per explicit user scoping this session, out of the 9
listed in the original spec -- the ones directly computable from
already-extracted pools without new data, and most diagnostic for
"is R1-B better because of a different negative distribution"):
  1. distance from ET (voxels to nearest ET boundary)
  2. anatomical composition (fraction inside ground-truth WT-not-TC,
     TC, or outside all tumor -- E259's own convention)
  3. production probability (frozen w_P's own confidence at each
     negative voxel -- directly answers "hardness" without a separate
     axis, since production's own confidence IS the hardness measure
     this whole investigation has used throughout)
  4. D1 feature-space distance (Euclidean distance from each negative
     voxel's D1 feature vector to its lesion's own ET-positive
     centroid, in the SAME 32-dim space the classifier actually sees)
Deferred (per explicit user scoping): count-per-positive (already
matched by design in E278, not an open measurement), within/cross-
subject (a fixed design choice in E278, not something to measure),
hardness (redundant with production probability), spatial density/
clustering (a secondary refinement of distance, deferred pending this
core result).

POOL RECONSTRUCTION: per explicit user choice, this experiment
REBUILDS R1-B's shell pool and SAM-equivalent's local-offset pool
using the IDENTICAL extraction logic and rng seed as E278 (not a
fresh, separately-sampled set) -- guaranteeing the characterized
distributions are the SAME ones that produced E278's actual result,
with voxel coordinates and subject IDs additionally retained (E278
only kept D1 features, discarding coordinates once extracted).
"""
import sys, csv, time, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e257_common import (load_model, get_w_prod, load_populations, build_splits,  # noqa: E402
                         ROOT, PATCH, MIN_VOX, MAX_VOX_PER_LESION)
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch, get_masks  # noqa: E402
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset  # noqa: E402

ET, TC, WT = 0, 1, 2
STRUCT = ndimage.generate_binary_structure(3, 1)
EPS = 1e-12
SEED = 999
LOCAL_OFFSET_MIN, LOCAL_OFFSET_MAX = 1, 10


def extract_shell_with_coords(model, ds, sid_to_idx, sid, cid, dev):
    """R1-B's own shell negatives (identical mechanism to e245's
    get_masks, SHELL_DILATION=3), WITH coordinates retained."""
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
    shell_coords = np.argwhere(shell)
    shell_feats = d1[:, shell].T
    et_feats = d1[:, cm].T  # lesion's OWN D1 features, for the D1-distance axis
    et_centroid = np.array(ndimage.center_of_mass(cm))
    et_coords = np.argwhere(cm)
    return {'subject_id': sid, 'coords': shell_coords, 'feats': shell_feats,
           'et_feats_mean': et_feats.mean(axis=0), 'et_centroid': et_centroid,
           'et_coords': et_coords, 'tgt_c': tgt_c, 'img_c': img_c}


def extract_sam_local_with_coords(model, ds, sid_to_idx, sid, cid, dev, rng):
    """SAM-equivalent local-offset negatives (IDENTICAL logic to E278's
    extract_et_and_sam_local), WITH coordinates retained."""
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    brain_mask = img_c[0] != 0
    true_et = tgt_c[ET] > 0.5
    et_lbl, et_n = ndimage.label(true_et)
    if cid < 1 or cid > et_n:
        return None
    cm = et_lbl == cid
    if int(cm.sum()) < MIN_VOX:
        return None
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    stages = forward_to_dec1_internal(model, img_t)
    d1 = stages['relu2_out'][0].cpu().numpy()
    D, H, W = cm.shape

    et_coords = np.argwhere(cm)
    local_coords = []
    for (z, y, x) in et_coords:
        for _ in range(10):
            direction = rng.normal(size=3)
            direction /= (np.linalg.norm(direction) + EPS)
            magnitude = rng.uniform(LOCAL_OFFSET_MIN, LOCAL_OFFSET_MAX)
            offset = np.round(direction * magnitude).astype(int)
            nz, ny, nx = z + offset[0], y + offset[1], x + offset[2]
            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W and brain_mask[nz, ny, nx] and not true_et[nz, ny, nx]:
                local_coords.append((nz, ny, nx))
                break
    if not local_coords:
        return None
    local_coords = np.array(local_coords)
    local_feats = d1[:, local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]].T
    et_centroid = np.array(ndimage.center_of_mass(cm))
    return {'subject_id': sid, 'coords': local_coords, 'feats': local_feats,
           'et_centroid': et_centroid, 'et_coords': et_coords, 'tgt_c': tgt_c, 'img_c': img_c}


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    model = load_model(dev)
    w_et, b_et = get_w_prod(model)
    det_lesions, g2a_lesions, g2b_lesions = load_populations()
    splits = build_splits(det_lesions, g2a_lesions)

    if smoke:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']][:30]
    else:
        det_lesions_fit = [r for r in det_lesions if r['subject_id'] in splits['det_train']]

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    w_p = w_et.astype(np.float64)
    b_p = float(b_et)

    def et_distance_map(tgt_c):
        true_et = tgt_c[ET] > 0.5
        if not true_et.any():
            return None
        return ndimage.distance_transform_edt(~true_et)

    def nearest_et_voxel_distance(coords, dist_map):
        if dist_map is None or len(coords) == 0:
            return np.full(len(coords), np.nan)
        return dist_map[coords[:, 0], coords[:, 1], coords[:, 2]]

    def anatomical_category(coords, tgt_c):
        if len(coords) == 0:
            return np.array([], dtype=object)
        true_tc = tgt_c[TC] > 0.5
        true_wt = tgt_c[WT] > 0.5
        cats = []
        for (z, y, x) in coords:
            if true_tc[z, y, x]:
                cats.append('TC')
            elif true_wt[z, y, x]:
                cats.append('WT_not_TC')
            else:
                cats.append('outside_tumor')
        return np.array(cats, dtype=object)

    def production_prob(feats):
        if len(feats) == 0:
            return np.array([])
        logits = feats @ w_p + b_p
        return 1.0 / (1.0 + np.exp(-logits))

    def d1_distance_to_centroid(feats, et_coords_feats_mean):
        if len(feats) == 0:
            return np.array([])
        return np.linalg.norm(feats - et_coords_feats_mean[None, :], axis=1)

    print(f'Processing {len(det_lesions_fit)} training lesions '
         f'(R1-B shell + SAM-equivalent local, with coordinates)...', flush=True)

    rows = []
    n_done = 0
    for r in det_lesions_fit:
        sid, cid = r['subject_id'], int(r['comp_id'])
        shell_result = extract_shell_with_coords(model, ds, sid_to_idx, sid, cid, dev)
        rng_sam = np.random.default_rng(SEED)  # SAM extraction uses its own rng stream per E278
        sam_result = extract_sam_local_with_coords(model, ds, sid_to_idx, sid, cid, dev, rng_sam)
        if shell_result is None or sam_result is None:
            continue

        tgt_c = shell_result['tgt_c']
        dist_map = et_distance_map(tgt_c)
        et_feats_mean = shell_result['et_feats_mean']  # this lesion's OWN ET-voxel D1 mean

        n_s = min(MAX_VOX_PER_LESION, len(shell_result['coords']))
        if n_s < len(shell_result['coords']):
            sel = rng.choice(len(shell_result['coords']), n_s, replace=False)
            shell_coords_sub = shell_result['coords'][sel]
            shell_feats_sub = shell_result['feats'][sel]
        else:
            shell_coords_sub = shell_result['coords']
            shell_feats_sub = shell_result['feats']

        n_l = min(MAX_VOX_PER_LESION, len(sam_result['coords']))
        if n_l < len(sam_result['coords']):
            sel = rng.choice(len(sam_result['coords']), n_l, replace=False)
            sam_coords_sub = sam_result['coords'][sel]
            sam_feats_sub = sam_result['feats'][sel]
        else:
            sam_coords_sub = sam_result['coords']
            sam_feats_sub = sam_result['feats']

        shell_dist = nearest_et_voxel_distance(shell_coords_sub, dist_map)
        sam_dist = nearest_et_voxel_distance(sam_coords_sub, dist_map)
        shell_cat = anatomical_category(shell_coords_sub, tgt_c)
        sam_cat = anatomical_category(sam_coords_sub, tgt_c)
        shell_prob = production_prob(shell_feats_sub)
        sam_prob = production_prob(sam_feats_sub)
        shell_d1dist = d1_distance_to_centroid(shell_feats_sub, et_feats_mean)
        sam_d1dist = d1_distance_to_centroid(sam_feats_sub, et_feats_mean)

        for i in range(len(shell_coords_sub)):
            rows.append({'source': 'R1B_shell', 'subject_id': sid, 'comp_id': cid,
                        'dist_to_et': float(shell_dist[i]), 'category': shell_cat[i],
                        'prod_prob': float(shell_prob[i]), 'd1_dist_to_et_mean': float(shell_d1dist[i])})
        for i in range(len(sam_coords_sub)):
            rows.append({'source': 'SAM_local', 'subject_id': sid, 'comp_id': cid,
                        'dist_to_et': float(sam_dist[i]), 'category': sam_cat[i],
                        'prod_prob': float(sam_prob[i]), 'd1_dist_to_et_mean': float(sam_d1dist[i])})
        n_done += 1
        if n_done % 50 == 0 or smoke:
            print(f'  {n_done}/{len(det_lesions_fit)} lesions ({time.time()-t0:.0f}s)', flush=True)

    print(f'\nTotal rows: {len(rows)} ({time.time()-t0:.0f}s)', flush=True)

    out = HERE / ('E279_negatives_smoke.csv' if smoke else 'E279_negatives.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['source', 'subject_id', 'comp_id', 'dist_to_et',
                                           'category', 'prod_prob', 'd1_dist_to_et_mean'])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print('\n=== Summary: R1B_shell vs SAM_local ===')
    for source in ['R1B_shell', 'SAM_local']:
        sub = [r for r in rows if r['source'] == source]
        dists = [r['dist_to_et'] for r in sub]
        probs = [r['prod_prob'] for r in sub]
        cats = [r['category'] for r in sub]
        d1dists = [r['d1_dist_to_et_mean'] for r in sub]
        n_wt = sum(1 for c in cats if c == 'WT_not_TC')
        n_tc = sum(1 for c in cats if c == 'TC')
        n_out = sum(1 for c in cats if c == 'outside_tumor')
        print(f'{source}: n={len(sub)}')
        print(f'  dist_to_et: mean={np.mean(dists):.3f} median={np.median(dists):.3f} '
             f'min={np.min(dists):.3f} max={np.max(dists):.3f}')
        print(f'  prod_prob: mean={np.mean(probs):.4f} median={np.median(probs):.4f}')
        print(f'  d1_dist_to_et_mean: mean={np.mean(d1dists):.4f} median={np.median(d1dists):.4f}')
        print(f'  category: WT_not_TC={n_wt/len(sub)*100:.1f}% TC={n_tc/len(sub)*100:.1f}% '
             f'outside_tumor={n_out/len(sub)*100:.1f}%')

    print(f'\nE279 complete ({time.time()-t0:.0f}s). wrote {out.name}', flush=True)


if __name__ == '__main__':
    main()
