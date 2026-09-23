"""NeuroScan forensic error decomposition -- where does the missing Dice actually go?

For all 125 subjects x 3 regions (ET, TC, WT) on the frozen E131_v5control_seed0
checkpoint, decompose the Dice shortfall into attributable failure mechanisms.

METHOD: each mechanism is scored by COUNTERFACTUAL DICE RECOVERY -- the Dice you
would regain if that failure mode alone were repaired, holding all others fixed.
This makes mechanisms directly comparable in units of the target metric (pp of
Dice) rather than in incommensurable units (voxels, components, mm).

Mechanisms are NOT mutually exclusive by construction (a missed component is also
false-negative volume; a boundary FN is also FN volume). Both the raw attribution
and the disjoint FP/FN x boundary/interior partition are reported. This overlap is
stated explicitly, never hidden.

GT is used for EVALUATION ONLY. This is a forensic audit, not a method.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'exp_e12_eggo_m' / 'e130'))
from neuroscan_3d_v5 import UNet3D_v5
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, REGIONS
import importlib.util

spec = importlib.util.spec_from_file_location(
    "t", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
t = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["e"]; spec.loader.exec_module(t); sys.argv = _a

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
OUT = Path(__file__).resolve().parent
THR = 0.5
BOUNDARY_VOX = 2   # "boundary" = within this many voxels of the GT surface


def dice_np(p, y):
    ps, ts = float(p.sum()), float(y.sum())
    if ts == 0:
        return 1.0 if ps == 0 else 0.0
    return float(2.0 * float((p * y).sum()) / (ps + ts))


def decompose(P, Y):
    """P, Y: bool (D,H,W). Returns dict of mechanism -> counterfactual Dice recovery."""
    Pf, Yf = P.astype(np.float32), Y.astype(np.float32)
    d0 = dice_np(Pf, Yf)
    out = {'dice': d0, 'gt_vox': int(Y.sum()), 'pred_vox': int(P.sum())}
    if Y.sum() == 0:
        out['empty_gt'] = 1
        return out
    out['empty_gt'] = 0

    FP = P & ~Y
    FN = Y & ~P
    out['fp_vox'] = int(FP.sum())
    out['fn_vox'] = int(FN.sum())

    # ---- boundary vs interior split of errors (distance to GT surface) ----
    dt_out = ndimage.distance_transform_edt(~Y)
    dt_in = ndimage.distance_transform_edt(Y)
    near = ((dt_out <= BOUNDARY_VOX) & ~Y) | ((dt_in <= BOUNDARY_VOX) & Y)
    fp_b, fp_i = FP & near, FP & ~near
    fn_b, fn_i = FN & near, FN & ~near
    out['fp_boundary_vox'] = int(fp_b.sum()); out['fp_interior_vox'] = int(fp_i.sum())
    out['fn_boundary_vox'] = int(fn_b.sum()); out['fn_interior_vox'] = int(fn_i.sum())

    def rec(newP):
        return dice_np(newP.astype(np.float32), Yf) - d0

    # ---- counterfactual recoveries, each repairing ONE mechanism ----
    out['rec_fix_all_fp'] = rec(P & Y)
    out['rec_fix_all_fn'] = rec(P | Y)
    out['rec_fix_boundary'] = rec((P & ~fp_b) | fn_b)
    out['rec_fix_interior'] = rec((P & ~fp_i) | fn_i)
    out['rec_fix_fp_boundary'] = rec(P & ~fp_b)
    out['rec_fix_fn_boundary'] = rec(P | fn_b)
    out['rec_fix_fp_interior'] = rec(P & ~fp_i)
    out['rec_fix_fn_interior'] = rec(P | fn_i)

    # ---- connected components ----
    gl, ng = ndimage.label(Y)
    pl, npd = ndimage.label(P)
    out['n_gt_comp'] = int(ng); out['n_pred_comp'] = int(npd)

    missed = np.zeros_like(Y); n_missed = 0
    for g in range(1, ng + 1):
        cm = gl == g
        if not (cm & P).any():
            missed |= cm; n_missed += 1
    out['n_missed_comp'] = n_missed
    out['missed_vox'] = int(missed.sum())
    out['rec_fix_missed_comp'] = rec(P | missed)

    spur = np.zeros_like(P); n_spur = 0
    for q in range(1, npd + 1):
        cm = pl == q
        if not (cm & Y).any():
            spur |= cm; n_spur += 1
    out['n_spurious_comp'] = n_spur
    out['spurious_vox'] = int(spur.sum())
    out['rec_fix_spurious_comp'] = rec(P & ~spur)

    # fragmentation: one GT component covered by >1 predicted component
    frag = 0
    for g in range(1, ng + 1):
        lbl = pl[(gl == g) & P]
        if lbl.size and len(np.unique(lbl)) > 1:
            frag += 1
    # merging: one predicted component spanning >1 GT component
    merge = 0
    for q in range(1, npd + 1):
        lbl = gl[(pl == q) & Y]
        if lbl.size and len(np.unique(lbl)) > 1:
            merge += 1
    out['n_fragmented_gt'] = frag
    out['n_merged_pred'] = merge

    # ---- spatial displacement ----
    disps = []
    for g in range(1, ng + 1):
        cm = gl == g
        ov = cm & P
        if ov.any():
            disps.append(float(np.linalg.norm(
                np.array(ndimage.center_of_mass(cm)) - np.array(ndimage.center_of_mass(ov)))))
    out['mean_centroid_disp'] = float(np.mean(disps)) if disps else float('nan')
    out['global_centroid_disp'] = (
        float(np.linalg.norm(np.array(ndimage.center_of_mass(Y))
                             - np.array(ndimage.center_of_mass(P)))) if P.any() else float('nan'))
    return out


def main():
    n_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    m = UNet3D_v5(4, 3).to(dev).eval()
    m.load_state_dict(ck['model_state'])
    for p in m.parameters():
        p.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    n = len(ds) if n_limit == 0 else min(n_limit, len(ds))
    print(f'forensic decomposition: {n} subjects, regions={REGIONS}', flush=True)

    rows = []
    t0 = time.time()
    for i in range(n):
        img, tgt, sid = ds[i]
        probs = t.sliding_window_predict(m, img.unsqueeze(0).to(dev),
                                         t.PATCH, t.SW_OVERLAP, 3, dev, True)
        Pb = probs >= THR
        Yb = tgt.numpy() > 0.5
        rec = {'subject_id': sid}
        for ri, rn in enumerate(REGIONS):
            for k, v in decompose(Pb[ri], Yb[ri]).items():
                rec[f'{rn}_{k}'] = v

        # ---- nesting violations: ET subset of TC subset of WT ----
        et, tc, wt = Pb[0], Pb[1], Pb[2]
        rec['viol_ET_not_in_TC'] = int((et & ~tc).sum())
        rec['viol_TC_not_in_WT'] = int((tc & ~wt).sum())
        rec['viol_total'] = rec['viol_ET_not_in_TC'] + rec['viol_TC_not_in_WT']
        # counterfactual repairs, both directions (grow outer vs shrink inner)
        d_tc0 = dice_np(tc.astype(np.float32), Yb[1].astype(np.float32))
        d_wt0 = dice_np(wt.astype(np.float32), Yb[2].astype(np.float32))
        d_et0 = dice_np(et.astype(np.float32), Yb[0].astype(np.float32))
        tc_f = tc | et
        wt_f = wt | tc_f
        rec['nest_rec_TC_grow'] = dice_np(tc_f.astype(np.float32), Yb[1].astype(np.float32)) - d_tc0
        rec['nest_rec_WT_grow'] = dice_np(wt_f.astype(np.float32), Yb[2].astype(np.float32)) - d_wt0
        rec['nest_rec_ET_shrink'] = dice_np((et & tc).astype(np.float32), Yb[0].astype(np.float32)) - d_et0
        rec['nest_rec_TC_shrink'] = dice_np((tc & wt).astype(np.float32), Yb[1].astype(np.float32)) - d_tc0
        rows.append(rec)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f'  {i+1}/{n} ({el:.0f}s, ~{el/(i+1)*(n-i-1):.0f}s left)', flush=True)

    keys = sorted({k for r in rows for k in r if k != 'subject_id'})
    with open(OUT / 'FORENSIC_per_subject.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['subject_id'] + keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\nwrote FORENSIC_per_subject.csv ({len(rows)} rows) in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
