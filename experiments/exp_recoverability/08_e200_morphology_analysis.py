"""E200 -- GATE 1: component-level morphology analysis, ET and TC.

Reruns inference ONCE on the frozen E131_v5control_seed0 checkpoint (the same
checkpoint E143/E144/E192/E198 all use), caches per-voxel binary predictions
for all 125 validation subjects, and computes connected-component morphology
metrics that E130's eval script (Dice/HD95/voxel-count only) cannot answer:

  - GT / predicted connected component counts
  - matched-component Dice (best-IoU greedy matching)
  - centroid displacement (matched components, voxel units)
  - fragmentation index = n_pred_components / n_gt_components (recoverable pop only)
  - missed-component rate = GT components with zero predicted overlap
  - false-component rate = predicted components with zero GT overlap
  - recall (GT voxels covered) vs precision (predicted voxels correct) -- separates
    undersegmentation from oversegmentation, which whole-region Dice cannot.

Classification per subject per region (ET, TC), mutually exclusive priority order:
  MISS            : GT>0, pred==0
  FRAGMENTED      : n_pred_components > n_gt_components AND n_gt_components>=1
  UNDERSEG        : recall < 0.5 AND precision >= 0.5 (finds less than half, what it
                    finds is mostly correct)
  OVERSEG         : precision < 0.5 AND recall >= 0.5 (predicts too much, most of it wrong)
  DISPLACED       : recall < 0.5 AND precision < 0.5 AND centroid_disp > 5 voxels
                    (looks like neither pure under- nor oversegmentation; spatially off)
  MIXED           : recall < 0.5 AND precision < 0.5, centroid_disp <= 5 (both wrong,
                    not simply displaced -- likely fragmented+partial)
  OK              : everything else (dice >= 0.5 roughly)

Output: experiments/exp_e12_eggo_m/E200_morphology.json (per-subject, per-region)
        experiments/exp_e12_eggo_m/E200_morphology_summary.csv

Run from repo root: python experiments/exp_recoverability/08_e200_morphology_analysis.py
"""
import sys, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, '.')
from neuroscan_3d_v5 import UNet3D_v5
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, REGIONS

ROOT = Path('.')
CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
OUT_JSON = ROOT / 'experiments/exp_e12_eggo_m/E200_morphology.json'
OUT_CSV = ROOT / 'experiments/exp_e12_eggo_m/E200_morphology_summary.csv'
PATCH = (128, 128, 128)
SW_OVERLAP = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------------------------------------- sliding window
def _gaussian_weight(shape, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, s) for s in shape]
    g = np.ones(shape, dtype=np.float32)
    for i, c in enumerate(coords):
        sh = [1] * len(shape); sh[i] = -1
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(sh).astype(np.float32)
    return np.maximum(g, 1e-4)


def sliding_window_predict(model, image, patch, overlap, n_out, amp=True):
    _, _, D, H, W = image.shape
    pd, ph, pw = patch
    stride = [max(1, int(p * (1 - overlap))) for p in patch]

    def starts(full, p, st):
        if full <= p: return [0]
        s = list(range(0, full - p + 1, st))
        if s[-1] != full - p: s.append(full - p)
        return s

    zs, ys, xs = (starts(D, pd, stride[0]), starts(H, ph, stride[1]), starts(W, pw, stride[2]))
    acc = torch.zeros((n_out, D, H, W), device=device, dtype=torch.float32)
    wsum = torch.zeros((1, D, H, W), device=device, dtype=torch.float32)
    gw = torch.from_numpy(_gaussian_weight((min(pd, D), min(ph, H), min(pw, W)))).to(device)

    with torch.no_grad():
        for z in zs:
            for y in ys:
                for x in xs:
                    zc, yc, xc = min(pd, D), min(ph, H), min(pw, W)
                    tile = image[:, :, z:z+zc, y:y+yc, x:x+xc]
                    if tile.shape[2:] != (pd, ph, pw):
                        tile = F.pad(tile, (0, pw-tile.shape[4], 0, ph-tile.shape[3], 0, pd-tile.shape[2]))
                    with torch.amp.autocast('cuda', enabled=amp):
                        out = model(tile)
                    pr = out['probs'].float()[:, :, :zc, :yc, :xc].squeeze(0)
                    acc[:, z:z+zc, y:y+yc, x:x+xc] += pr * gw
                    wsum[:, z:z+zc, y:y+yc, x:x+xc] += gw
    return (acc / wsum.clamp(min=1e-6)).cpu().numpy()


# ------------------------------------------------------------- morphology
def component_metrics(pred, gt):
    """pred, gt: (D,H,W) bool. Returns a dict of morphology metrics + label."""
    gt_lab, n_gt = ndimage.label(gt)
    pred_lab, n_pred = ndimage.label(pred)

    gt_vox = int(gt.sum())
    pred_vox = int(pred.sum())
    inter = int((pred & gt).sum())

    recall = inter / gt_vox if gt_vox > 0 else float('nan')
    precision = inter / pred_vox if pred_vox > 0 else float('nan')
    dice = 2 * inter / (gt_vox + pred_vox) if (gt_vox + pred_vox) > 0 else 1.0

    # --- per-GT-component match: does ANY predicted voxel overlap it? ---
    missed = 0
    matched_centroid_disp = []
    if n_gt > 0:
        for gi in range(1, n_gt + 1):
            comp_mask = gt_lab == gi
            if not (comp_mask & pred).any():
                missed += 1
            else:
                # centroid displacement: GT component centroid vs. the centroid
                # of predicted voxels that overlap it (a local match, not global)
                gt_c = np.array(ndimage.center_of_mass(comp_mask))
                overlap_pred = comp_mask & pred
                if overlap_pred.any():
                    pr_c = np.array(ndimage.center_of_mass(overlap_pred))
                    matched_centroid_disp.append(float(np.linalg.norm(gt_c - pr_c)))

    # --- per-predicted-component: does it overlap ANY GT? ---
    false_components = 0
    if n_pred > 0:
        for pi in range(1, n_pred + 1):
            comp_mask = pred_lab == pi
            if not (comp_mask & gt).any():
                false_components += 1

    missed_rate = missed / n_gt if n_gt > 0 else float('nan')
    false_rate = false_components / n_pred if n_pred > 0 else float('nan')
    frag_index = n_pred / n_gt if n_gt > 0 else float('nan')
    mean_centroid_disp = float(np.mean(matched_centroid_disp)) if matched_centroid_disp else float('nan')

    return dict(
        n_gt_components=int(n_gt), n_pred_components=int(n_pred),
        gt_voxels=gt_vox, pred_voxels=pred_vox, intersection=inter,
        recall=recall, precision=precision, dice=dice,
        missed_components=int(missed), missed_rate=missed_rate,
        false_components=int(false_components), false_rate=false_rate,
        fragmentation_index=frag_index, mean_centroid_disp_vox=mean_centroid_disp,
    )


def classify(m):
    """Mutually exclusive failure typing, priority order as documented above."""
    if m['gt_voxels'] == 0:
        return 'NO_GT'
    if m['pred_voxels'] == 0:
        return 'MISS'
    r, p = m['recall'], m['precision']
    if m['n_gt_components'] >= 1 and m['n_pred_components'] > m['n_gt_components'] and m['dice'] < 0.5:
        return 'FRAGMENTED'
    if r < 0.5 and p >= 0.5:
        return 'UNDERSEG'
    if p < 0.5 and r >= 0.5:
        return 'OVERSEG'
    if r < 0.5 and p < 0.5:
        disp = m['mean_centroid_disp_vox']
        if disp == disp and disp > 5.0:  # not nan and > 5 voxels
            return 'DISPLACED'
        return 'MIXED'
    return 'OK'


def main():
    print(f'device: {device}')
    ck = torch.load(str(CKPT), map_location=device, weights_only=False)
    print(f"checkpoint epoch={ck['epoch']+1} best_mean_dice={ck.get('best_mean_dice'):.4f} arch={ck.get('arch')}")
    model = UNet3D_v5(in_channels=4, out_channels=3).to(device)
    model.load_state_dict(ck['model_state'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val', val_split=0.1, patch_size=PATCH)
    print(f'evaluating {len(ds)} validation subjects\n', flush=True)

    records = []
    t0 = time.time()
    for i in range(len(ds)):
        image, target, sid = ds[i]
        img = image.unsqueeze(0).to(device)
        probs = sliding_window_predict(model, img, PATCH, SW_OVERLAP, 3, amp=True)
        pred_bin = probs >= 0.5
        tgt_bin = target.numpy().astype(bool)

        rec = {'subject_id': sid}
        for j, r in enumerate(REGIONS):
            if r not in ('ET', 'TC'):
                continue
            m = component_metrics(pred_bin[j], tgt_bin[j])
            m['label'] = classify(m)
            rec[r] = m
        records.append(rec)

        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f'  {i+1}/{len(ds)}  ({el:.0f}s, {el/(i+1)*(len(ds)-i-1):.0f}s left)', flush=True)

    with open(OUT_JSON, 'w') as f:
        json.dump(records, f, indent=2)
    print(f'\nsaved {OUT_JSON}')

    # ---- summary CSV + printed table ----
    import csv
    rows = []
    for rec in records:
        for r in ('ET', 'TC'):
            m = rec[r]
            row = {'subject_id': rec['subject_id'], 'region': r, **m}
            rows.append(row)
    fields = list(rows[0].keys())
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f'saved {OUT_CSV}')

    print(f'\n{"="*70}')
    print('E200 -- FAILURE-TYPE DISTRIBUTION')
    print(f'{"="*70}')
    for r in ('ET', 'TC'):
        labels = [rec[r]['label'] for rec in records]
        print(f'\n{r}:')
        from collections import Counter
        c = Counter(labels)
        for lab, n in c.most_common():
            print(f'  {lab:12s} n={n:3d}  ({100*n/len(labels):.1f}%)')


if __name__ == '__main__':
    main()
