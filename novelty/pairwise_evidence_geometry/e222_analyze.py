"""E222 analysis -- PREREGISTERED (written before full results were seen).

Q1/Q2: component-level cross-detection matrix on ET (the region every prior
  experiment this session targeted). For each model's MISSED components,
  what fraction are detected by >=1 of the other models?
Q3: voxel-level disagreement informativeness. Where models disagree (not
  unanimous), what fraction of disagreement voxels are GT-positive?
Q4: naive fixed ensembles, whole-volume Dice, all at z>0 (fixed threshold,
  matching every model's own eval convention -- no per-arm tuning):
    mean       : average of probs, threshold 0.5
    majority   : >=3 of 5 models predict positive
    unanimous  : all 5 predict positive (high-precision arm)
    any        : >=1 model predicts positive (high-recall arm)
    oracle-best: per-subject, the single BEST model (upper bound on
                 selection-based fusion; not achievable without GT)
  Compared against each model's own solo Dice.

DECISION RULE (stated in advance): fusion is worth designing only if Q2
shows >=25% of any model's missed components are caught by >=1 other model
AND Q4's 'any'-recall arm doesn't lose more overall Dice than 'majority'
gains -- i.e. there must be a real complementary-error budget to spend, not
just noise that cancels under any combination rule.
"""
import csv, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
pred_fn = sys.argv[1] if len(sys.argv) > 1 else 'E222_predictions.csv'
comp_fn = sys.argv[2] if len(sys.argv) > 2 else 'E222_components.csv'
preds = list(csv.DictReader(open(HERE / pred_fn)))
comps = list(csv.DictReader(open(HERE / comp_fn)))

names = sorted({k.rsplit('_', 1)[0] for k in comps[0].keys() if k.endswith('_detected')})
print(f'models: {names}\n')

print('=' * 92)
print('Q1/Q2: ET component cross-detection (component-level, MIN_VOX=5, overlap>=0.5)')
print('=' * 92)
det = {n: np.array([int(c[f'{n}_detected']) for c in comps]) for n in names}
n_comp = len(comps)
print(f'total GT ET components: {n_comp}\n')
for n in names:
    print(f'  {n}: detects {det[n].sum()}/{n_comp} ({100*det[n].mean():.1f}%)')

print('\n  Cross-recovery: of model A\'s MISSED components, % detected by >=1 other model')
recovery_rates = {}
for a in names:
    missed_a = det[a] == 0
    if missed_a.sum() == 0:
        continue
    others = np.zeros(n_comp, dtype=bool)
    for b in names:
        if b != a:
            others |= (det[b] == 1)
    recovered = (missed_a & others).sum()
    rate = recovered / missed_a.sum()
    recovery_rates[a] = rate
    print(f'    {a}: {missed_a.sum()} missed, {recovered} recovered by >=1 other = {rate:.1%}')

print('\n  Pairwise: of A misses, % B specifically catches')
for a in names:
    missed_a = det[a] == 0
    if missed_a.sum() == 0:
        continue
    line = f'    {a} misses ({missed_a.sum()}): '
    for b in names:
        if b == a:
            continue
        caught = (missed_a & (det[b] == 1)).sum()
        line += f'{b}={caught}({100*caught/missed_a.sum():.0f}%) '
    print(line)

print('\n' + '=' * 92)
print('Q3: voxel-level disagreement informativeness (ET, brain voxels)')
print('=' * 92)
print('  (computed from TP+FP per model on shared subject/region rows -- see')
print('   note: full voxel disagreement needs raw prediction maps, not just')
print('   aggregate counts; this reports the AGGREGATE proxy: does total FP')
print('   volume correlate with total TP volume across models, i.e. are')
print('   models that catch more true lesion also making more false calls)')
et_rows = [r for r in preds if r['region'] == '0']
for n in names:
    tp = sum(int(r[f'{n}_tp']) for r in et_rows)
    fp = sum(int(r[f'{n}_fp']) for r in et_rows)
    fn = sum(int(r[f'{n}_fn']) for r in et_rows)
    print(f'  {n}: TP={tp} FP={fp} FN={fn}  precision={tp/(tp+fp+1e-9):.3f}  '
          f'recall={tp/(tp+fn+1e-9):.3f}')

print('\n' + '=' * 92)
print('Q4: naive ensemble arms, whole-volume Dice by region (fixed z>0 / count-vote)')
print('=' * 92)
by_subj = defaultdict(dict)
for r in preds:
    by_subj[(r['subject_id'], r['region'])] = r

REGIONS = ['ET', 'TC', 'WT']
print(f"\n  Per-model solo Dice (mean over subjects, from per-subject TP/FP/FN):")
solo = {n: {} for n in names}
for ri, rn in enumerate(REGIONS):
    for n in names:
        rows = [r for r in preds if r['region'] == str(ri)]
        d = []
        for r in rows:
            tp, fp, fn = int(r[f'{n}_tp']), int(r[f'{n}_fp']), int(r[f'{n}_fn'])
            d.append((1.0 if fp == 0 else 0.0) if tp + fn == 0 else 2*tp/(2*tp+fp+fn))
        solo[n][rn] = np.mean(d)
    line = f'  {rn}: ' + '  '.join(f'{n}={solo[n][rn]:.4f}' for n in names)
    print(line)

print(f'\n  NOTE: majority/mean/unanimous/any ensemble arms require RAW per-voxel')
print(f'  prediction maps (not just aggregate TP/FP/FN), which E222 did not cache')
print(f'  (avoided ~5x storage: 125 subjects x 5 models x 3 regions x full volume).')
print(f'  If Q1/Q2 justify it, a targeted re-run on the TAIL subjects only (where')
print(f'  disagreement matters) is cheap; a blind full-cohort voxel cache is not.')

print('\n' + '=' * 92)
print('DECISION')
print('=' * 92)
best_recovery = max(recovery_rates.values()) if recovery_rates else 0
worst_recovery = min(recovery_rates.values()) if recovery_rates else 0
mean_recovery = np.mean(list(recovery_rates.values())) if recovery_rates else 0
print(f'  cross-recovery rates: {recovery_rates}')
print(f'  mean={mean_recovery:.1%}  best={best_recovery:.1%}  worst={worst_recovery:.1%}')
worth_it = mean_recovery >= 0.25
print(f'\n  Complementary-error budget >=25% (mean)?  {"YES" if worth_it else "NO"}')
print(f'  ==> {"Fusion may be worth designing -- proceed to targeted voxel-level Q3/Q4" if worth_it else "KILL -- insufficient complementary error, do not design a fusion rule"}')
