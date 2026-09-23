"""GATE 1 -- bias-only sweep in logit space, full cohort.

Question: can the EXISTING readout recover the missed components purely by
moving its decision boundary (logit offset), and what does that cost the bulk?

This is done directly in logit space on all 125 subjects and reports tail vs
bulk separately, which a global probability-threshold sweep cannot resolve.
"""
import sys, csv, json, time
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
# logit offsets: logit 0 == prob 0.5. Positive db makes detection EASIER.
DBS = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0]


def dice(p, y):
    ps, ts = float(p.sum()), float(y.sum())
    if ts == 0: return 1.0 if ps == 0 else 0.0
    return float(2.0 * float((p & y).sum()) / (ps + ts))


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(str(CKPT), map_location=dev, weights_only=False)
    model = UNet3D_v5(4, 3).to(dev).eval()
    model.load_state_dict(ck['model_state'])
    for p in model.parameters():
        p.requires_grad_(False)
    ds = BraTSMultimodalDataset(str(ROOT / 'Dataset' / 'Training'), 'val',
                                val_split=0.1, patch_size=t.PATCH)
    fr = list(csv.DictReader(open(OUT / 'FORENSIC_per_subject.csv')))
    tail_ids = {r['subject_id'] for r in fr
                if float(r['ET_dice']) < 0.5 or float(r['TC_dice']) < 0.5}

    rows = []
    t0 = time.time()
    for i in range(len(ds)):
        img, tgt, sid = ds[i]
        Y = tgt.numpy() > 0.5
        probs = t.sliding_window_predict(model, img.unsqueeze(0).to(dev),
                                         t.PATCH, t.SW_OVERLAP, 3, dev, True)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs / (1 - probs))            # invert sigmoid
        rec = {'subject_id': sid, 'is_tail': int(sid in tail_ids)}
        for db in DBS:
            P = (logits + db) > 0
            for ri, rn in enumerate(REGIONS):
                rec[f'{rn}_db{db}'] = dice(P[ri], Y[ri])
                rec[f'{rn}_fp_db{db}'] = int((P[ri] & ~Y[ri]).sum())
            # recovered missed components (ET/TC)
            for ri, rn in [(0, 'ET'), (1, 'TC')]:
                lbl, nl = ndimage.label(Y[ri])
                base = (logits[ri]) > 0
                rec_n = 0
                for g in range(1, nl + 1):
                    cm = lbl == g
                    if cm.sum() >= 5 and not (cm & base).any() and (cm & P[ri]).any():
                        rec_n += 1
                rec[f'{rn}_recovered_db{db}'] = rec_n
        rows.append(rec)
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(ds)} ({time.time()-t0:.0f}s)', flush=True)

    with open(OUT / 'GATE1_bias_sweep.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)

    # ---- report ----
    tail = np.array([r['is_tail'] == 1 for r in rows])
    print(f'\nn={len(rows)}  tail={tail.sum()}  bulk={(~tail).sum()}')
    print('\n' + '=' * 92)
    print('GATE 1 -- bias-only logit offset (db>0 = easier to detect)')
    print('=' * 92)
    print(f"{'db':>6}{'3-reg':>10}{'delta_pp':>10}{'ET':>9}{'TC':>9}{'WT':>9}"
          f"{'tail3':>9}{'bulk3':>9}{'recET':>7}{'recTC':>7}{'FPx':>8}")
    base3 = None
    fp0 = sum(r['WT_fp_db0.0'] for r in rows)
    for db in DBS:
        per = {rn: np.mean([r[f'{rn}_db{db}'] for r in rows]) for rn in REGIONS}
        m3 = np.mean(list(per.values()))
        if db == 0.0: base3 = m3
        t3 = np.mean([np.mean([r[f'{rn}_db{db}'] for rn in REGIONS])
                      for r, isT in zip(rows, tail) if isT])
        b3 = np.mean([np.mean([r[f'{rn}_db{db}'] for rn in REGIONS])
                      for r, isT in zip(rows, tail) if not isT])
        rE = sum(r[f'ET_recovered_db{db}'] for r in rows)
        rT = sum(r[f'TC_recovered_db{db}'] for r in rows)
        fpx = sum(r[f'WT_fp_db{db}'] for r in rows) / max(fp0, 1)
        print(f"{db:>6.1f}{m3:>10.6f}{100*(m3-base3):>10.3f}{per['ET']:>9.4f}{per['TC']:>9.4f}"
              f"{per['WT']:>9.4f}{t3:>9.4f}{b3:>9.4f}{rE:>7}{rT:>7}{fpx:>8.2f}")
    print('\n  FPx = WT false-positive volume relative to db=0')
    json.dump({'dbs': DBS}, open(OUT / 'GATE1_summary.json', 'w'), indent=2)
    print('wrote GATE1_bias_sweep.csv')


if __name__ == '__main__':
    main()
