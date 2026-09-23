"""E201 -- GATE 2 + GATE 3C: O_i vs morphology, and the recoverability gap G_i = O_i - D_i.

Requires: E143_recoverability.json (O_i, 90 subjects, ET only)
          E200_morphology.json (component metrics, 125 subjects, ET+TC)
          e130/E130_full_eval_E131_v5control_seed0_per_subject.json (D_i, model Dice)

For the 90-subject O_i population (intersected with E200's 125):
  1. O_i vs recall, precision, missed_rate, false_rate, fragmentation_index,
     centroid_disp, MISS-label indicator -- Spearman, raw and partial-on-size.
  2. G_i = O_i - D_i^ET (the recoverability gap). Distribution, and what
     morphology type dominates at high G_i vs low/negative G_i.
  3. Report the G_i-ranked subject list: this IS Gate 5's candidate population.

Run from repo root: python experiments/exp_recoverability/09_e201_oi_vs_morphology.py
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path('.')
O = json.load(open(ROOT / 'experiments/exp_e12_eggo_m/E143_recoverability.json'))['mlp_deep']
morph = {r['subject_id']: r for r in json.load(open(ROOT / 'experiments/exp_e12_eggo_m/E200_morphology.json'))}
ev = {x['subject_id']: x for x in json.load(open(
    ROOT / 'experiments/exp_e12_eggo_m/e130/E130_full_eval_E131_v5control_seed0_per_subject.json'))}

ids = sorted(set(O) & set(morph))
print(f'n subjects with both O_i and morphology: {len(ids)}')

Oi = np.array([O[s] for s in ids])
Di = np.array([ev[s]['dice_ET'] for s in ids])
Gi = Oi - Di

m_et = [morph[s]['ET'] for s in ids]
recall = np.array([m['recall'] if m['recall'] == m['recall'] else np.nan for m in m_et])
precision = np.array([m['precision'] if m['precision'] == m['precision'] else np.nan for m in m_et])
missed_rate = np.array([m['missed_rate'] if m['missed_rate'] == m['missed_rate'] else np.nan for m in m_et])
false_rate = np.array([m['false_rate'] if m['false_rate'] == m['false_rate'] else np.nan for m in m_et])
frag = np.array([m['fragmentation_index'] if m['fragmentation_index'] == m['fragmentation_index'] else np.nan for m in m_et])
disp = np.array([m['mean_centroid_disp_vox'] if m['mean_centroid_disp_vox'] == m['mean_centroid_disp_vox'] else np.nan for m in m_et])
is_miss = np.array([1.0 if morph[s]['ET']['label'] == 'MISS' else 0.0 for s in ids])
labels = [morph[s]['ET']['label'] for s in ids]

def spear(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 5:
        return float('nan'), float('nan'), int(mask.sum())
    r, p = stats.spearmanr(x[mask], y[mask])
    return r, p, int(mask.sum())

print()
print('='*70)
print('GATE 2 -- O_i vs ET morphology (Spearman)')
print('='*70)
for name, arr in [('recall', recall), ('precision', precision), ('missed_rate', missed_rate),
                   ('false_rate', false_rate), ('fragmentation_index', frag),
                   ('centroid_disp', disp), ('is_MISS (point-biserial)', is_miss)]:
    r, p, n = spear(Oi, arr)
    print(f'  O_i vs {name:28s} rho={r:+.3f}  p={p:.4f}  n={n}')

print()
print('  Label distribution at this 90-subject intersection:')
from collections import Counter
for lab, n in Counter(labels).most_common():
    print(f'    {lab:12s} n={n:3d}')

print()
print('='*70)
print('GATE 3C -- Recoverability gap G_i = O_i - D_i')
print('='*70)
print(f'  mean G_i = {Gi.mean():+.4f}   std = {Gi.std():.4f}')
print(f'  G_i > 0.20 (image informative, model underperforms): n = {int((Gi>0.20).sum())}')
print(f'  G_i > 0.10                                          : n = {int((Gi>0.10).sum())}')
print(f'  G_i < -0.10 (model beats the simple observer)       : n = {int((Gi<-0.10).sum())}')
print()
order = np.argsort(-Gi)
print(f'  {"subject":32s} {"O_i":>6s} {"D_i":>6s} {"G_i":>7s} {"label":>12s} {"recall":>7s} {"precision":>7s} {"n_gt_comp":>9s} {"n_pred_comp":>11s}')
for k in order[:25]:
    s = ids[k]
    print(f'  {s:32s} {Oi[k]:6.3f} {Di[k]:6.3f} {Gi[k]:+7.3f} {labels[k]:>12s} '
          f'{recall[k]:7.3f} {precision[k]:7.3f} {m_et[k]["n_gt_components"]:9d} {m_et[k]["n_pred_components"]:11d}')

print()
print('  Bottom 10 (G_i most negative -- model beats simple intensity observer):')
for k in order[-10:]:
    s = ids[k]
    print(f'  {s:32s} {Oi[k]:6.3f} {Di[k]:6.3f} {Gi[k]:+7.3f} {labels[k]:>12s}')

# label breakdown by G_i tercile
print()
print('='*70)
print('Failure-type composition by G_i tercile')
print('='*70)
terc = np.quantile(Gi, [1/3, 2/3])
low = Gi <= terc[0]; mid = (Gi > terc[0]) & (Gi <= terc[1]); high = Gi > terc[1]
for name, mask in [('low G_i (bottom third)', low), ('mid G_i', mid), ('high G_i (top third, THE candidate pop)', high)]:
    c = Counter([labels[k] for k in range(len(ids)) if mask[k]])
    print(f'  {name}: n={mask.sum()}  {dict(c)}')

out = {
    'ids': ids, 'O_i': Oi.tolist(), 'D_i_ET': Di.tolist(), 'G_i': Gi.tolist(),
    'label': labels, 'recall': recall.tolist(), 'precision': precision.tolist(),
    'missed_rate': missed_rate.tolist(), 'false_rate': false_rate.tolist(),
    'fragmentation_index': frag.tolist(), 'centroid_disp': disp.tolist(),
}
json.dump(out, open(ROOT / 'experiments/exp_e12_eggo_m/E201_recoverability_gap.json', 'w'), indent=1)
print('\nsaved E201_recoverability_gap.json')
