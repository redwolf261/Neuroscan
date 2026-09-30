"""E247 -- channel-specific clipping analysis, per the user's exact spec.
G2-A AND G2-B both measured (G2-B as the critical control the user
requires: if informative channels are clipped in G2-A but PRESERVED in
G2-B, that is a feature-selective nonlinear bottleneck, not a generic
ReLU problem).

PER-CHANNEL METRICS (per explicit user decision on this turn's
disambiguation question -- decomposes E234's own separability() formula
to channel level, no new statistic invented):
  z_diff_c(stage) = (lesion_mean_c - shell_mean_c) / shell_std_c
    at a given stage (bn1_out or relu1_out), for channel c. This is
    LITERALLY the per-channel term inside separability()'s own
    sqrt(sum(z_diff_c^2)/n_valid) aggregate -- reused via direct
    reimplementation of that inner computation (not separability()
    itself, since that only returns the aggregate scalar).
  informativeness_c = |z_diff_c(bn1_out)|  (how much channel c alone
    separates lesion from shell, BEFORE ReLU acts on it)
  delta_z_c = z_diff_c(relu1_out) - z_diff_c(bn1_out)  (channel c's own
    separability change across the ReLU step)
  clip_c = fraction of LESION voxels where bn1_out[c] < 0 (channel c's
    own clipping rate, restricted to lesion voxels specifically)

THE QUESTION (per explicit user framing): are the SAME discriminative
channels (high informativeness) systematically MORE clipped in G2-A
than in matched detected -- and does this pattern REVERSE or vanish in
G2-B (the critical control)?

METHOD: per lesion, rank channels 1-32 by informativeness_c (descending).
Compute the correlation between informativeness_c and clip_c ACROSS
CHANNELS WITHIN THAT LESION (Spearman, since we want "are informative
channels disproportionately clipped" as a rank relationship, not
assuming linearity). A POSITIVE correlation means informative channels
are clipped MORE -- the "wrong features suppressed" pattern. Aggregate
this per-lesion correlation across all lesions in each group (G2-A,
matched detected, G2-B) and compare.

ALSO: same correlation test using delta_z_c instead of clip_c (does
informativeness predict the channel's OWN separability loss, not just
its raw clipping rate) -- directly tests "are informative channels the
ones being destroyed" using the same currency as E245/E246's own
Delta_S measurements.
"""
import sys, csv, time, os
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
from e245_dec1_internal_decomposition import forward_to_dec1_internal, load_patch, get_masks  # noqa: E402

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
ET = 0
PATCH = (128, 128, 128)
MIN_VOX = 5


def per_channel_zdiff(tensor_np, lesion_mask, shell_mask):
    """tensor_np: (C, D, H, W). Returns z_diff_c array (C,) -- the SAME
    per-channel quantity computed inside e234's separability(), just
    returned per-channel instead of aggregated. Channels with near-zero
    shell_std (below the SAME relative floor as separability()) are
    marked NaN, matching separability()'s own exclusion convention."""
    C = tensor_np.shape[0]
    flat = tensor_np.reshape(C, -1)
    lesion_flat = lesion_mask.reshape(-1)
    shell_flat = shell_mask.reshape(-1)
    lesion_vals = flat[:, lesion_flat]
    shell_vals = flat[:, shell_flat]
    lesion_mean = lesion_vals.mean(axis=1)
    shell_mean = shell_vals.mean(axis=1)
    shell_std = shell_vals.std(axis=1)
    overall_std = flat.std(axis=1)
    floor = np.maximum(overall_std * 0.01, 1e-6)
    z_diff = (lesion_mean - shell_mean) / shell_std
    z_diff[shell_std < floor] = np.nan
    return z_diff


def measure_lesion_channels(model, ds, sid_to_idx, sid, comp_id, dev):
    if sid not in sid_to_idx:
        return None
    img_c, tgt_c = load_patch(ds, sid_to_idx, sid)
    brain_mask = img_c[0] != 0
    cm, shell = get_masks(tgt_c, comp_id, brain_mask)
    if cm is None:
        return None
    img_t = torch.from_numpy(img_c).unsqueeze(0).float().to(dev)
    stages = forward_to_dec1_internal(model, img_t)
    bn1 = stages['bn1_out'][0].cpu().numpy()   # (32, D, H, W)
    relu1 = stages['relu1_out'][0].cpu().numpy()

    z_bn1 = per_channel_zdiff(bn1, cm, shell)
    z_relu1 = per_channel_zdiff(relu1, cm, shell)
    delta_z = z_relu1 - z_bn1

    C = bn1.shape[0]
    clip_c = np.zeros(C)
    for c in range(C):
        vals = bn1[c][cm]
        clip_c[c] = float((vals < 0).mean()) if len(vals) > 0 else np.nan

    return {'z_bn1': z_bn1, 'z_relu1': z_relu1, 'delta_z': delta_z, 'clip_c': clip_c}


def per_lesion_correlations(result):
    """Given one lesion's per-channel arrays, returns the two Spearman
    correlations the experiment cares about: informativeness vs
    clipping rate, and informativeness vs the channel's own separability
    CHANGE. NaN-safe (excludes channels flagged NaN by the floor)."""
    informativeness = np.abs(result['z_bn1'])
    clip_c = result['clip_c']
    delta_z = result['delta_z']
    valid = ~(np.isnan(informativeness) | np.isnan(clip_c) | np.isnan(delta_z))
    if valid.sum() < 5:
        return None, None
    r_clip, _ = stats.spearmanr(informativeness[valid], clip_c[valid])
    r_delta, _ = stats.spearmanr(informativeness[valid], delta_z[valid])
    return r_clip, r_delta


def main():
    smoke = '--smoke' in sys.argv
    dev = torch.device('cuda')

    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)

    phenotypes = {}
    for r in csv.DictReader(open(HERE / 'E243_phenotypes.csv')):
        phenotypes[(r['subject_id'], r['comp_id'])] = r['phenotype']

    e240 = list(csv.DictReader(open(HERE / 'E240_layerwise.csv')))
    by_pair = {}
    for r in e240:
        pid = int(r['pair_id'])
        by_pair.setdefault(pid, {})[r['role']] = r

    g2a_pairs = []
    for pid, d in by_pair.items():
        if 'missed' not in d or 'detected' not in d:
            continue
        key = (d['missed']['subject_id'], d['missed']['comp_id'])
        if phenotypes.get(key) == 'G2A_rejected':
            g2a_pairs.append((d['missed'], d['detected']))

    g2b_lesions = []
    for pid, d in by_pair.items():
        if 'missed' not in d:
            continue
        key = (d['missed']['subject_id'], d['missed']['comp_id'])
        if phenotypes.get(key) == 'G2B_partial':
            g2b_lesions.append(d['missed'])

    print(f'E247: {len(g2a_pairs)} G2-A pairs, {len(g2b_lesions)} G2-B lesions', flush=True)
    if smoke:
        g2a_pairs = g2a_pairs[:8]
        g2b_lesions = g2b_lesions[:8]

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'train',
                                val_split=0.1, patch_size=PATCH)
    sid_to_idx = {os.path.basename(sd): i for i, sd in enumerate(ds.subject_dirs)}

    out = HERE / ('E247_smoke.csv' if smoke else 'E247_channel_clipping.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['pair_idx', 'group', 'subject_id', 'comp_id',
                                       'r_informativeness_vs_clip', 'r_informativeness_vs_deltaz',
                                       'n_valid_channels'])
    w.writeheader()

    t0 = time.time()
    n_done = 0
    for i, (m, d) in enumerate(g2a_pairs):
        res_m = measure_lesion_channels(model, ds, sid_to_idx, m['subject_id'], int(m['comp_id']), dev)
        res_d = measure_lesion_channels(model, ds, sid_to_idx, d['subject_id'], int(d['comp_id']), dev)
        for grp, res, row_src in [('G2A_missed', res_m, m), ('matched_detected', res_d, d)]:
            if res is None:
                continue
            r_clip, r_delta = per_lesion_correlations(res)
            if r_clip is None:
                continue
            n_valid = int((~np.isnan(res['z_bn1'])).sum())
            w.writerow({'pair_idx': i, 'group': grp, 'subject_id': row_src['subject_id'],
                       'comp_id': row_src['comp_id'], 'r_informativeness_vs_clip': r_clip,
                       'r_informativeness_vs_deltaz': r_delta, 'n_valid_channels': n_valid})
        n_done += 1
        if n_done % 10 == 0 or smoke:
            print(f'  G2-A pairs: {n_done}/{len(g2a_pairs)} ({time.time()-t0:.0f}s)', flush=True)
    fh.flush()

    for i, lesion in enumerate(g2b_lesions):
        res = measure_lesion_channels(model, ds, sid_to_idx, lesion['subject_id'], int(lesion['comp_id']), dev)
        if res is None:
            continue
        r_clip, r_delta = per_lesion_correlations(res)
        if r_clip is None:
            continue
        n_valid = int((~np.isnan(res['z_bn1'])).sum())
        w.writerow({'pair_idx': -1, 'group': 'G2B_partial', 'subject_id': lesion['subject_id'],
                   'comp_id': lesion['comp_id'], 'r_informativeness_vs_clip': r_clip,
                   'r_informativeness_vs_deltaz': r_delta, 'n_valid_channels': n_valid})
        if (i + 1) % 10 == 0 or smoke:
            print(f'  G2-B lesions: {i+1}/{len(g2b_lesions)} ({time.time()-t0:.0f}s)', flush=True)
    fh.close()
    print(f'\nE247 complete: wrote {out.name} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
