"""E236 analysis -- the decision tree, per the user's exact spec.

Experiment A (cross-group): AUC_miss >> 0.5 -> information exists,
  segmentation pathway fails to use it.
Experiment B (within-group): recovers -> utilization/decision problem.
  Does not recover -> representation transformation problem.
"""
import csv
import numpy as np
from scipy import stats

d236 = list(csv.DictReader(open('E236_d2_recoverability.csv')))
for r in d236:
    r['auc'] = float(r['auc']); r['fold'] = int(r['fold'])
    r['n_pos'] = int(r['n_pos']); r['n_neg'] = int(r['n_neg'])

exp_a = [r for r in d236 if r['experiment'] == 'A_cross_group']
exp_b = [r for r in d236 if r['experiment'] == 'B_within_group']

print("=" * 90)
print("EXPERIMENT A: detected-trained D2 probe -> high_contrast_missed AUC")
print("=" * 90)
aucs_a = np.array([r['auc'] for r in exp_a])
print(f"  n_folds={len(aucs_a)}  mean AUC={aucs_a.mean():.4f}  std={aucs_a.std():.4f}")
print(f"  per-fold: {[round(a,4) for a in aucs_a]}")
# one-sample test vs 0.5 (chance)
t_stat, p_a = stats.ttest_1samp(aucs_a, 0.5)
print(f"  one-sample t-test vs 0.5: t={t_stat:.4f}  p={p_a:.4e}")
w_stat, p_a_w = stats.wilcoxon(aucs_a - 0.5)
print(f"  Wilcoxon vs 0.5: p={p_a_w:.4f}")

print("\n" + "=" * 90)
print("EXPERIMENT B: within-group (missed-only) probe recoverability")
print("=" * 90)
aucs_b = np.array([r['auc'] for r in exp_b])
print(f"  n_folds={len(aucs_b)}  mean AUC={aucs_b.mean():.4f}  std={aucs_b.std():.4f}")
print(f"  per-fold: {[round(a,4) for a in aucs_b]}")
t_stat_b, p_b = stats.ttest_1samp(aucs_b, 0.5)
print(f"  one-sample t-test vs 0.5: t={t_stat_b:.4f}  p={p_b:.4e}")
w_stat_b, p_b_w = stats.wilcoxon(aucs_b - 0.5)
print(f"  Wilcoxon vs 0.5: p={p_b_w:.4f}")

print("\n" + "=" * 90)
print("DECISION TREE")
print("=" * 90)
a_strong = aucs_a.mean() > 0.65 and p_a_w < 0.1
b_strong = aucs_b.mean() > 0.65 and p_b_w < 0.1
print(f"Experiment A: AUC={aucs_a.mean():.3f}  {'>> 0.5, INFORMATION PRESENT' if a_strong else 'near 0.5 or unreliable'}")
print(f"Experiment B: AUC={aucs_b.mean():.3f}  {'>> 0.5, RECOVERABLE' if b_strong else 'near 0.5 or unreliable'}")

if a_strong and b_strong:
    print("\n  ==> UTILIZATION / DECISION PROBLEM.")
    print("      D2 representation DOES still identify high-contrast-missed lesions,")
    print("      both from a detected-only-trained probe AND from the lesions' own")
    print("      statistics. The information EXISTS at D2. The segmentation pathway")
    print("      (decoder D1 + seg_head, or the training objective) FAILS TO USE")
    print("      information that a simple linear readout can already extract.")
elif not a_strong and not b_strong:
    print("\n  ==> REPRESENTATION TRANSFORMATION PROBLEM.")
    print("      Neither a detected-trained probe NOR a probe fit to the missed")
    print("      lesions' own statistics can recover lesion identity at D2.")
    print("      The representation genuinely lacks usable lesion information for")
    print("      this population by this stage.")
else:
    print("\n  ==> MIXED: Experiment A and B disagree -- report both numbers, do not")
    print("      force a single conclusion. This itself would be informative (e.g.")
    print("      cross-group generalization fails but within-group structure exists")
    print("      would suggest the INFORMATION IS THERE but encoded differently for")
    print("      missed lesions than for detected ones).")
