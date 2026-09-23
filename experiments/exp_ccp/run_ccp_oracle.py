"""CCP frozen-model oracle -- main runner (WT only).

Runs baseline + 6 oracle conditions x 3 reaction configs on the NeuroScan
125-subject proving-ground cohort. Frozen checkpoint, no training, no GT in
the dynamics.
"""
import sys, json, time, argparse, os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'exp_e12_eggo_m' / 'e130'))

from neuroscan_3d_v5 import UNet3D_v5
from Dataset.brats_multimodal_dataset import BraTSMultimodalDataset, REGIONS
import ccp_core as C

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "e130_train", str(ROOT / 'experiments/exp_e12_eggo_m/e130/train_e130_multimodal_baseline.py'))
_t = importlib.util.module_from_spec(_spec)
_argv = sys.argv; sys.argv = ["eval"]; _spec.loader.exec_module(_t); sys.argv = _argv

CKPT = ROOT / 'experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth'
OUT = Path(__file__).resolve().parent
WT = REGIONS.index('WT')          # == 2
THRESH = 0.5                       # MUST match baseline exactly
ETA = 0.05
LAM = 10.0
TAUS = (0.5, 0.25, 0.125)
DELTA = 0.25
CONFIGS = {"R1": (0.5, 0.3, 0.2), "R2": (0.4, 0.4, 0.2), "R3": (0.5, 0.2, 0.3)}

def dice(pred_bin, tgt_bin):
    ps, ts = float(pred_bin.sum()), float(tgt_bin.sum())
    if ts == 0:
        return 1.0 if ps == 0 else 0.0
    return float(2.0 * float((pred_bin * tgt_bin).sum()) / (ps + ts))

def d_of(q, tgt):
    return dice((q >= THRESH).to(torch.float64), tgt)

def run_ccp(E, alpha, beta, gamma, do_comp=True, do_proof=True, randomize_R=False, rng=None):
    """Returns (q_final, stage_dices_fn_inputs, n_clamped_total).
    Stage fields are returned so caller can score them against GT (eval only)."""
    q = E.clone()
    stages = [q.clone()]
    nclamp = 0

    def potential(qc):
        N = C.neighbourhood_mean(qc)
        A = C.agreement(E, N)
        R = C.reaction_potential(E, N, A, alpha, beta, gamma)
        if randomize_R:
            # preserve the DISTRIBUTION of R, destroy its spatial arrangement
            flat = R.flatten()
            perm = torch.randperm(flat.numel(), generator=rng, device=flat.device)
            R = flat[perm].reshape(R.shape)
        return N, A, R

    # round 1: plain competition
    N, A, R = potential(q)
    if do_comp:
        q, nc = C.competition_step(q, R, ETA); nclamp += nc
    stages.append(q.clone())

    # rounds 2-4: proofreading stages 1..3
    for k, tau in enumerate(TAUS, start=1):
        N, A, R = potential(q)
        if k == 1:   S = E
        elif k == 2: S = E * N
        else:        S = E * N * A
        if do_proof:
            G = C.proofread_potential(R, S, LAM, tau, DELTA)
        else:
            G = R
        if do_comp:
            q, nc = C.competition_step(q, G, ETA); nclamp += nc
        stages.append(q.clone())
    return q, stages, nclamp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', type=str, default='cuda')
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    print(f"checkpoint epoch={ck['epoch']+1} best_mean_dice={ck.get('best_mean_dice'):.4f} arch={ck.get('arch')}", flush=True)
    model = UNet3D_v5(in_channels=4, out_channels=3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters(): p.requires_grad_(False)

    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=_t.PATCH)
    n = len(ds) if a.limit == 0 else min(a.limit, len(ds))
    print(f"subjects: {n} (cohort size {len(ds)})\n", flush=True)

    rows, stage_rows = [], []
    t0 = time.time()
    for i in range(n):
        image, target, sid = ds[i]
        img = image.unsqueeze(0).to(dev)
        probs = _t.sliding_window_predict(model, img, _t.PATCH, _t.SW_OVERLAP, 3, dev, True)

        # --- evidence field (WT only), float64 on CPU for exact conservation ---
        # float64 on GPU: verified bit-identical to CPU float64 (max diff 0.0)
        E = torch.from_numpy(probs[WT]).to(torch.float64).to(dev)
        Y = target[WT].to(torch.float64).to(dev)
        M0 = float(E.sum())
        d_base = d_of(E, Y)

        rec = {'subject_id': sid, 'dice_baseline': d_base,
               'pred_voxels_base': int((E >= THRESH).sum()),
               'gt_voxels': int(Y.sum()),
               'mass_pred': M0, 'mass_gt': float(Y.sum())}
        rec['mass_ratio'] = M0 / max(1e-9, float(Y.sum()))

        # ---- O1: conservation-only smoothing (config-independent) ----
        q = E.clone()
        for _ in range(4):
            q = C.conservative_smoothing_step(q, ETA)
        rec['dice_O1_smooth'] = d_of(q, Y)
        rec['cons_err_O1'] = abs(float(q.sum()) - M0) / max(M0, 1e-9)

        # ---- per-config oracle conditions ----
        for cname, (al, be, ga) in CONFIGS.items():
            # O2 competition only
            q2, st2, nc2 = run_ccp(E, al, be, ga, do_comp=True, do_proof=False)
            rec[f'dice_O2_comp_{cname}'] = d_of(q2, Y)
            rec[f'cons_err_O2_{cname}'] = abs(float(q2.sum()) - M0) / max(M0, 1e-9)
            # O3 proofreading only (no competitive redistribution => field unchanged)
            q3, st3, _ = run_ccp(E, al, be, ga, do_comp=False, do_proof=True)
            rec[f'dice_O3_proof_{cname}'] = d_of(q3, Y)
            # O4 == full CCP: competition + proofreading
            q4, st4, nc4 = run_ccp(E, al, be, ga, do_comp=True, do_proof=True)
            rec[f'dice_O4_ccp_{cname}'] = d_of(q4, Y)
            rec[f'cons_err_O4_{cname}'] = abs(float(q4.sum()) - M0) / max(M0, 1e-9)
            rec[f'mass_ccp_{cname}'] = float(q4.sum())
            rec[f'n_clamped_{cname}'] = nc4
            # O5 randomized-R control
            g = torch.Generator(device=dev); g.manual_seed(1234 + i)
            q5, _, _ = run_ccp(E, al, be, ga, do_comp=True, do_proof=True,
                               randomize_R=True, rng=g)
            rec[f'dice_O5_rand_{cname}'] = d_of(q5, Y)
            # stage-wise recovery curve (GT used for EVALUATION ONLY)
            for k, sf in enumerate(st4):
                stage_rows.append({'subject_id': sid, 'config': cname,
                                   'stage': k, 'dice': d_of(sf, Y)})
        rows.append(rec)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{n}  ({el:.0f}s, ~{el/(i+1)*(n-i-1):.0f}s left)", flush=True)

    import csv
    with open(OUT / 'CCP_per_subject.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    with open(OUT / 'CCP_stages.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['subject_id','config','stage','dice']); w.writeheader()
        for r in stage_rows: w.writerow(r)
    print(f"\nwrote CCP_per_subject.csv ({len(rows)} rows), CCP_stages.csv ({len(stage_rows)} rows)")
    print(f"runtime {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
