"""E225 analysis: 4 questions, measurement only (no E148-E150 framing yet,
per user instruction)."""
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
REGIONS = ('ET', 'TC', 'WT')
TAUS = [round(0.1 * i, 1) for i in range(1, 10)]


def load(run):
    dist = json.load(open(HERE / f'{run}_distributions.json'))
    comp = list(csv.DictReader(open(HERE / f'{run}_components.csv')))
    for r in comp:
        r['size'] = int(r['size'])
        r['detected'] = int(r['detected'])
        r['is_fp'] = int(r['is_fp'])
    return dist, comp


b_dist, b_comp = load('E225_baseline')
c_dist, c_comp = load('E225_cc')


def hist_stats(h):
    h = np.array(h, dtype=np.float64)
    edges = np.linspace(0, 1, len(h) + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    tot = h.sum()
    if tot == 0:
        return dict(n=0, mean=float('nan'), median=float('nan'))
    mean = float((h * centers).sum() / tot)
    cum = np.cumsum(h) / tot
    median = float(centers[np.searchsorted(cum, 0.5)])
    return dict(n=int(tot), mean=mean, median=median)


print('=' * 100)
print('1. PROBABILITY DISTRIBUTIONS (per region: GT-pos, GT-neg, FP-voxel, FN-voxel predicted prob)')
print('=' * 100)
for r in REGIONS:
    print(f'\n  -- {r} --')
    for key, label in [('hist_pos', 'GT-positive  '), ('hist_neg', 'GT-negative  '),
                       ('hist_fp', 'FP voxels    '), ('hist_fn_gtprob', 'FN voxels    ')]:
        bs = hist_stats(b_dist[key][r]); cs = hist_stats(c_dist[key][r])
        print(f'    {label}  baseline: n={bs["n"]:>9} mean={bs["mean"]:.4f} median={bs["median"]:.4f}'
              f'   |   cc: n={cs["n"]:>9} mean={cs["mean"]:.4f} median={cs["median"]:.4f}')

print('\n' + '=' * 100)
print('2. THRESHOLD SWEEP: Dice/FP/FN per region, tau=0.1..0.9')
print('=' * 100)
for r in REGIONS:
    print(f'\n  -- {r} --')
    print(f'  {"tau":>5} {"b_dice":>8} {"c_dice":>8} {"b_fp":>10} {"c_fp":>10} {"b_fn":>10} {"c_fn":>10}')
    for tau in TAUS:
        bs = b_dist['sweep'][r][str(tau)]; cs = c_dist['sweep'][r][str(tau)]
        bdice = 2*bs['tp'] / max(1, 2*bs['tp']+bs['fp']+bs['fn'])
        cdice = 2*cs['tp'] / max(1, 2*cs['tp']+cs['fp']+cs['fn'])
        print(f'  {tau:>5} {bdice:>8.4f} {cdice:>8.4f} {bs["fp"]:>10} {cs["fp"]:>10} {bs["fn"]:>10} {cs["fn"]:>10}')
    # best operating point each
    best_b = max(TAUS, key=lambda t: 2*b_dist['sweep'][r][str(t)]['tp'] / max(1, 2*b_dist['sweep'][r][str(t)]['tp']+b_dist['sweep'][r][str(t)]['fp']+b_dist['sweep'][r][str(t)]['fn']))
    best_c = max(TAUS, key=lambda t: 2*c_dist['sweep'][r][str(t)]['tp'] / max(1, 2*c_dist['sweep'][r][str(t)]['tp']+c_dist['sweep'][r][str(t)]['fp']+c_dist['sweep'][r][str(t)]['fn']))
    bd = 2*b_dist['sweep'][r][str(best_b)]['tp'] / max(1, 2*b_dist['sweep'][r][str(best_b)]['tp']+b_dist['sweep'][r][str(best_b)]['fp']+b_dist['sweep'][r][str(best_b)]['fn'])
    cd = 2*c_dist['sweep'][r][str(best_c)]['tp'] / max(1, 2*c_dist['sweep'][r][str(best_c)]['tp']+c_dist['sweep'][r][str(best_c)]['fp']+c_dist['sweep'][r][str(best_c)]['fn'])
    print(f'  best-tau baseline: tau={best_b} dice={bd:.4f}   |   best-tau cc: tau={best_c} dice={cd:.4f}')
    # does cc dominate baseline across the range, or only at tau=0.5?
    wins = sum(1 for t in TAUS if
              (2*c_dist['sweep'][r][str(t)]['tp'] / max(1, 2*c_dist['sweep'][r][str(t)]['tp']+c_dist['sweep'][r][str(t)]['fp']+c_dist['sweep'][r][str(t)]['fn'])) >
              (2*b_dist['sweep'][r][str(t)]['tp'] / max(1, 2*b_dist['sweep'][r][str(t)]['tp']+b_dist['sweep'][r][str(t)]['fp']+b_dist['sweep'][r][str(t)]['fn'])))
    print(f'  CC beats baseline Dice at {wins}/9 thresholds')

print('\n' + '=' * 100)
print('3. CONNECTED-COMPONENT BEHAVIOUR (tau=0.5)')
print('=' * 100)
for r in REGIONS:
    print(f'\n  -- {r} --')
    for name, comp in [('baseline', b_comp), ('cc', c_comp)]:
        gt = [x for x in comp if x['region'] == r and x['kind'] == 'gt']
        pred_fp = [x for x in comp if x['region'] == r and x['kind'] == 'pred' and x['is_fp'] == 1]
        pred_all = [x for x in comp if x['region'] == r and x['kind'] == 'pred']
        det = [x for x in gt if x['detected'] == 1]
        missed = [x for x in gt if x['detected'] == 0]
        fp_sizes = [x['size'] for x in pred_fp]
        print(f'    {name:>9}: GT components={len(gt)} (detected={len(det)}, missed={len(missed)}, '
              f'detect_rate={100*len(det)/max(1,len(gt)):.1f}%)')
        print(f'               pred components={len(pred_all)}  FP components={len(pred_fp)}  '
              f'FP median size={np.median(fp_sizes) if fp_sizes else float("nan"):.0f}  '
              f'FP mean size={np.mean(fp_sizes) if fp_sizes else float("nan"):.0f}')
        if missed:
            print(f'               missed GT median size={np.median([x["size"] for x in missed]):.0f}  '
                  f'detected GT median size={np.median([x["size"] for x in det]):.0f}')
    # size distribution of FP components -- tiny (noise) vs large (real structure removed)?
    for name, comp in [('baseline', b_comp), ('cc', c_comp)]:
        fp_sizes = sorted(x['size'] for x in comp if x['region'] == r and x['kind'] == 'pred' and x['is_fp'] == 1)
        if fp_sizes:
            n = len(fp_sizes)
            print(f'    {name:>9} FP size percentiles: p10={fp_sizes[int(.1*n)]} p50={fp_sizes[int(.5*n)]} '
                  f'p90={fp_sizes[min(n-1,int(.9*n))]} max={fp_sizes[-1]}')

print('\n' + '=' * 100)
print('4. SUMMARY: ET vs TC vs WT (tau=0.5)')
print('=' * 100)
print(f'  {"region":>6} {"b_dice":>8} {"c_dice":>8} {"d_dice":>8} {"b_fp":>10} {"c_fp":>10} {"fp_%d":>8} {"b_fn":>10} {"c_fn":>10} {"fn_%d":>8}')
for ri, r in enumerate(REGIONS):
    bs = b_dist['sweep'][r]['0.5']; cs = c_dist['sweep'][r]['0.5']
    bdice = 2*bs['tp'] / max(1, 2*bs['tp']+bs['fp']+bs['fn'])
    cdice = 2*cs['tp'] / max(1, 2*cs['tp']+cs['fp']+cs['fn'])
    dfp = 100*(cs['fp']-bs['fp'])/bs['fp']
    dfn = 100*(cs['fn']-bs['fn'])/bs['fn']
    print(f'  {r:>6} {bdice:>8.4f} {cdice:>8.4f} {cdice-bdice:>+8.4f} {bs["fp"]:>10} {cs["fp"]:>10} {dfp:>+7.1f}% '
          f'{bs["fn"]:>10} {cs["fn"]:>10} {dfn:>+7.1f}%')
