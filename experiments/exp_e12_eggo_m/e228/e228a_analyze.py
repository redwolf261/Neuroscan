"""E228-A analysis -- the causal test itself.

For each condition (baseline/conflict/random) and each group (conflict_
positive/random_control), computes the mean_prob trajectory across epochs
1-5 (PRIMARY endpoint per the user's own spec). Then asks the decisive
question: does CONFLICT SUPPRESSION move the conflict-positive group's
trajectory relative to baseline, MORE than random suppression moves the
random-control group relative to baseline? That differential comparison
is what actually tests the mechanism (per the user's own 3-condition
table): if suppression helps regardless of which lesions it's applied to,
this is not evidence for the conflict-specific hypothesis.
"""
import csv
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDITIONS = ('baseline', 'conflict', 'random')


def load(cond):
    rows = list(csv.DictReader(open(HERE / f'E228A_{cond}' / 'lesion_trajectory.csv')))
    for r in rows:
        r['epoch'] = int(r['epoch']); r['comp_id'] = int(r['comp_id'])
        r['size'] = float(r['size']); r['mean_prob'] = float(r['mean_prob'])
        r['max_prob'] = float(r['max_prob']); r['lesion_dice'] = float(r['lesion_dice'])
        r['detected'] = int(r['detected'])
    return rows


data = {c: load(c) for c in CONDITIONS}


def group_epoch_mean(rows, group, epoch, metric='mean_prob'):
    vals = [r[metric] for r in rows if r['group'] == group and r['epoch'] == epoch]
    return np.mean(vals) if vals else float('nan'), len(vals)


print('=' * 100)
print('MEAN_PROB TRAJECTORY (primary endpoint), per condition x group, epoch 1-5')
print('=' * 100)
for group in ('conflict_positive', 'random_control'):
    print(f'\n  -- {group} --')
    print(f'  {"epoch":>6}  {"baseline":>10}  {"conflict":>10}  {"random":>10}')
    for ep in range(1, 6):
        vals = []
        for cond in CONDITIONS:
            m, n = group_epoch_mean(data[cond], group, ep)
            vals.append(m)
        print(f'  {ep:>6}  {vals[0]:>10.4f}  {vals[1]:>10.4f}  {vals[2]:>10.4f}')

print('\n' + '=' * 100)
print('DELTA FROM BASELINE AT EPOCH 5 (the decisive comparison)')
print('=' * 100)
for group in ('conflict_positive', 'random_control'):
    b5, _ = group_epoch_mean(data['baseline'], group, 5)
    c5, _ = group_epoch_mean(data['conflict'], group, 5)
    r5, _ = group_epoch_mean(data['random'], group, 5)
    print(f'  {group:>20}: baseline={b5:.4f}  conflict={c5:.4f} (delta={c5-b5:+.4f})  '
          f'random={r5:.4f} (delta={r5-b5:+.4f})')

print('\n' + '=' * 100)
print('KEY TEST: does CONFLICT suppression move conflict_positive MORE than')
print('RANDOM suppression moves random_control? (both relative to baseline)')
print('=' * 100)
b5_cp, _ = group_epoch_mean(data['baseline'], 'conflict_positive', 5)
c5_cp, _ = group_epoch_mean(data['conflict'], 'conflict_positive', 5)
delta_conflict_on_cp = c5_cp - b5_cp

b5_rc, _ = group_epoch_mean(data['baseline'], 'random_control', 5)
r5_rc, _ = group_epoch_mean(data['random'], 'random_control', 5)
delta_random_on_rc = r5_rc - b5_rc

print(f'  Delta(conflict-suppression on conflict_positive) = {delta_conflict_on_cp:+.4f}')
print(f'  Delta(random-suppression on random_control)      = {delta_random_on_rc:+.4f}')
if abs(delta_conflict_on_cp) > abs(delta_random_on_rc) * 1.5 and delta_conflict_on_cp > 0:
    print('  ==> Suggests a SPECIFIC conflict effect (conflict-positive moved '
        'more than random-control moved under matched suppression)')
elif abs(delta_conflict_on_cp - delta_random_on_rc) < 0.01:
    print('  ==> Deltas are essentially IDENTICAL -- suggests suppression effect is '
        'GENERIC (any D4 reduction helps similarly), NOT specific to conflict. '
        'This would mean the mechanism is NOT what E227 hypothesized.')
else:
    print('  ==> Ambiguous / mixed -- see per-epoch trajectory and per-lesion detail below.')

print('\n' + '=' * 100)
print('SECONDARY ENDPOINTS: whole-val Dice per condition, epoch 5')
print('=' * 100)
for cond in CONDITIONS:
    rows = list(csv.DictReader(open(HERE / f'E228A_{cond}' / 'epoch_metrics.csv')))
    last = rows[-1]
    print(f'  {cond:>10}: diceET={float(last["val_dice_ET"]):.4f}  '
          f'diceTC={float(last["val_dice_TC"]):.4f}  diceWT={float(last["val_dice_WT"]):.4f}  '
          f'fpET={int(float(last["val_fp_ET"]))}')

print('\n' + '=' * 100)
print('DETECTION STATUS CHANGE (epoch1 -> epoch5) per group per condition')
print('=' * 100)
for group in ('conflict_positive', 'random_control'):
    print(f'\n  -- {group} --')
    for cond in CONDITIONS:
        d1 = [r['detected'] for r in data[cond] if r['group'] == group and r['epoch'] == 1]
        d5 = [r['detected'] for r in data[cond] if r['group'] == group and r['epoch'] == 5]
        print(f'  {cond:>10}: epoch1 detected={sum(d1)}/{len(d1)}  epoch5 detected={sum(d5)}/{len(d5)}')

print('\n' + '=' * 100)
print('PER-LESION DETAIL: conflict_positive group, mean_prob epoch1 vs epoch5, per condition')
print('=' * 100)
lesions_cp = sorted(set((r['subject_id'], r['comp_id']) for r in data['baseline'] if r['group']=='conflict_positive'))
for sid, cid in lesions_cp:
    line = f'  {sid} comp{cid}: '
    for cond in CONDITIONS:
        e1 = next((r['mean_prob'] for r in data[cond] if r['subject_id']==sid and r['comp_id']==cid and r['epoch']==1), None)
        e5 = next((r['mean_prob'] for r in data[cond] if r['subject_id']==sid and r['comp_id']==cid and r['epoch']==5), None)
        line += f'{cond}: {e1:.3f}->{e5:.3f}  '
    print(line)
