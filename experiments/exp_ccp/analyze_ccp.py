"""CCP oracle analysis: aggregate stats, paired tests, mass-allocation, stage curve."""
import csv, json
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'CCP_per_subject.csv')))
st = list(csv.DictReader(open(HERE / 'CCP_stages.csv')))
n = len(rows)
F = lambda k: np.array([float(r[k]) for r in rows])

base = F('dice_baseline')
CONFIGS = ['R1', 'R2', 'R3']
COND = {'O1_smooth': 'Conservation-only smoothing',
        'O2_comp': 'Competition only',
        'O3_proof': 'Proofreading only',
        'O4_ccp': 'Full CCP (comp+proof)',
        'O5_rand': 'Randomized-R control'}

def summarize(d, b):
    dd = d - b
    ci = None
    rng = np.random.default_rng(0)
    bs = np.array([np.mean(rng.choice(dd, len(dd), replace=True)) for _ in range(10000)])
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    try:
        w = stats.wilcoxon(d, b, zero_method='wilcox')
        wp, wstat = float(w.pvalue), float(w.statistic)
    except Exception:
        wp, wstat = float('nan'), float('nan')
    return {'mean_dice': float(d.mean()), 'mean_delta_pp': float(100*dd.mean()),
            'median_delta_pp': float(100*np.median(dd)), 'sd_delta_pp': float(100*dd.std(ddof=1)),
            'ci95_delta_pp': [100*ci[0], 100*ci[1]],
            'wilcoxon_p': wp, 'wilcoxon_stat': wstat,
            'N_improved': int((dd > 0).sum()), 'N_ge_1pp': int((dd >= 0.01).sum()),
            'N_le_-1pp': int((dd <= -0.01).sum()),
            'max_gain_pp': float(100*dd.max()), 'max_loss_pp': float(100*dd.min())}

out = {'n_subjects': n, 'baseline_WT_mean_dice': float(base.mean()), 'conditions': {}}
print(f"n = {n}   BASELINE WT mean Dice = {base.mean():.7f}\n")
print(f"{'Method':<34}{'MeanDice':>10}{'dPP':>8}{'MedPP':>8}{'N+':>5}{'N>=1pp':>8}{'N<=-1pp':>9}{'Wilcox p':>11}")
print(f"{'Baseline':<34}{base.mean():>10.6f}{0.0:>8.3f}{0.0:>8.3f}{'-':>5}{'-':>8}{'-':>9}{'-':>11}")

# O1 is config-independent
s = summarize(F('dice_O1_smooth'), base); out['conditions']['O1_smooth'] = s
print(f"{COND['O1_smooth']:<34}{s['mean_dice']:>10.6f}{s['mean_delta_pp']:>8.3f}{s['median_delta_pp']:>8.3f}"
      f"{s['N_improved']:>5}{s['N_ge_1pp']:>8}{s['N_le_-1pp']:>9}{s['wilcoxon_p']:>11.2e}")

for cond in ['O2_comp', 'O3_proof', 'O4_ccp', 'O5_rand']:
    for c in CONFIGS:
        k = f'dice_{cond}_{c}'
        if k not in rows[0]: continue
        s = summarize(F(k), base); out['conditions'][f'{cond}_{c}'] = s
        print(f"{COND[cond]+' ['+c+']':<34}{s['mean_dice']:>10.6f}{s['mean_delta_pp']:>8.3f}"
              f"{s['median_delta_pp']:>8.3f}{s['N_improved']:>5}{s['N_ge_1pp']:>8}"
              f"{s['N_le_-1pp']:>9}{s['wilcoxon_p']:>11.2e}")

# conservation
ce = []
for c in CONFIGS:
    k = f'cons_err_O4_{c}'
    if k in rows[0]: ce += list(F(k))
ce = np.array(ce + list(F('cons_err_O1')))
out['conservation'] = {'max_rel_err': float(ce.max()), 'mean_rel_err': float(ce.mean())}
print(f"\nCONSERVATION: max rel err {ce.max():.3e}, mean {ce.mean():.3e}")

nc = np.concatenate([F(f'n_clamped_{c}') for c in CONFIGS if f'n_clamped_{c}' in rows[0]])
print(f"CLAMPING: total clamped-voxel events max {nc.max():.0f}, mean {nc.mean():.1f}")
out['clamp'] = {'max': float(nc.max()), 'mean': float(nc.mean())}

# stage-wise recovery
print("\nSTAGE-WISE RECOVERY (mean WT Dice):")
out['stages'] = {}
for c in CONFIGS:
    vals = []
    for k in range(5):
        v = [float(r['dice']) for r in st if r['config'] == c and int(r['stage']) == k]
        vals.append(float(np.mean(v)) if v else float('nan'))
    out['stages'][c] = vals
    print(f"  {c}: D0={vals[0]:.6f} D1={vals[1]:.6f} D2={vals[2]:.6f} D3={vals[3]:.6f} D4={vals[4]:.6f}"
          f"   (D0->D4 = {100*(vals[4]-vals[0]):+.4f}pp)")

# mass allocation
print("\nMASS ALLOCATION:")
rM = F('mass_ratio'); d4 = F('dice_O4_ccp_R1') - base
bins = [(0, 0.9, 'under-allocation r<0.9'), (0.9, 1.1, 'matched 0.9<=r<1.1'), (1.1, 1e9, 'over-allocation r>=1.1')]
out['mass'] = {'spearman_rM_vs_deltaCCP': None, 'bins': {}}
sp = stats.spearmanr(rM, d4)
out['mass']['spearman_rM_vs_deltaCCP'] = [float(sp.statistic), float(sp.pvalue)]
print(f"  Spearman(mass_ratio, dDice_CCP_R1) = {sp.statistic:+.3f} (p={sp.pvalue:.3g})")
for lo, hi, lbl in bins:
    m = (rM >= lo) & (rM < hi)
    if m.sum():
        out['mass']['bins'][lbl] = {'n': int(m.sum()), 'mean_delta_pp': float(100*d4[m].mean()),
                                    'mean_base_dice': float(base[m].mean())}
        print(f"  {lbl:<26} n={int(m.sum()):>4}  mean dCCP {100*d4[m].mean():+.4f}pp  mean base {base[m].mean():.4f}")

# verify mass preserved by CCP
mp = F('mass_pred'); mc = F('mass_ccp_R1')
print(f"\n  mass_CCP vs mass_pred: max rel diff {np.abs((mc-mp)/np.maximum(mp,1e-9)).max():.3e}  (must be ~0)")

# extremes
print("\nEXTREME SUBJECTS (Full CCP R1):")
order = np.argsort(d4)
for lbl, idxs in [('LARGEST LOSSES', order[:3]), ('LARGEST GAINS', order[-3:][::-1])]:
    print(f"  {lbl}:")
    for i in idxs:
        print(f"    {rows[i]['subject_id']}: base {base[i]:.4f} -> CCP {F('dice_O4_ccp_R1')[i]:.4f} "
              f"({100*d4[i]:+.4f}pp) massratio {rM[i]:.3f}")

json.dump(out, open(HERE / 'CCP_summary.json', 'w'), indent=2)
print(f"\nwrote CCP_summary.json")
