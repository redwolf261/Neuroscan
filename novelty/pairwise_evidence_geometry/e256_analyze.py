"""E256 analysis -- oracle-style gate feasibility, per the user's exact
questions. Aggregates E256_feasibility.csv across all 258 held-out
subjects, per spatial category (lesion, local_shell, distant_background).
"""
import csv
import numpy as np
from collections import defaultdict

rows = list(csv.DictReader(open('E256_feasibility.csv')))
for r in rows:
    for k in ['n_voxels', 'n_r1_gt_prod', 'n_r1_gt_prod_correct', 'n_prod_gt_r1',
             'n_prod_gt_r1_correct', 'n_r1_recovers_lesion']:
        r[k] = int(r[k])

by_cat = defaultdict(list)
for r in rows:
    by_cat[r['category']].append(r)

print("="*100)
print("ORACLE FEASIBILITY: where R1 > production, how often is R1 actually correct?")
print("="*100)
for cat in ['lesion', 'local_shell', 'distant_background']:
    rs = by_cat[cat]
    total_vox = sum(r['n_voxels'] for r in rs)
    total_r1_gt = sum(r['n_r1_gt_prod'] for r in rs)
    total_r1_gt_correct = sum(r['n_r1_gt_prod_correct'] for r in rs)
    total_prod_gt = sum(r['n_prod_gt_r1'] for r in rs)
    total_prod_gt_correct = sum(r['n_prod_gt_r1_correct'] for r in rs)
    frac_r1_wins_wrong = 1 - (total_r1_gt_correct / max(1, total_r1_gt))
    print(f"\n  {cat} (total voxels={total_vox:,}):")
    print(f"    R1>prod: {total_r1_gt:,} voxels ({100*total_r1_gt/max(1,total_vox):.2f}% of category)")
    print(f"      of these, R1 prediction correct: {total_r1_gt_correct:,} ({100*total_r1_gt_correct/max(1,total_r1_gt):.2f}%)")
    print(f"      of these, R1 prediction WRONG (i.e. crosses tau=0.5 incorrectly): "
          f"{total_r1_gt - total_r1_gt_correct:,} ({100*frac_r1_wins_wrong:.2f}%)")
    print(f"    prod>R1: {total_prod_gt:,} voxels")
    print(f"      of these, production correct: {total_prod_gt_correct:,} "
          f"({100*total_prod_gt_correct/max(1,total_prod_gt):.2f}%)")

print("\n" + "="*100)
print("WHERE DOES POSSIBLE G2-A/LESION RECOVERY LIVE? (n_r1_recovers_lesion, category=lesion only)")
print("="*100)
lesion_rows = by_cat['lesion']
total_recovers = sum(r['n_r1_recovers_lesion'] for r in lesion_rows)
total_lesion_vox = sum(r['n_voxels'] for r in lesion_rows)
print(f"  Total lesion voxels where p_R1>0.5 (R1 would recover): {total_recovers:,} / {total_lesion_vox:,} "
      f"({100*total_recovers/max(1,total_lesion_vox):.2f}%)")

print("\n" + "="*100)
print("HOW MANY 'FALSE ALARM' VOXELS WOULD A NAIVE 'R1>prod' GATE EXPOSE, BY CATEGORY?")
print("="*100)
print("(this decomposes E255's aggregate FP finding by spatial category)")
for cat in ['lesion', 'local_shell', 'distant_background']:
    rs = by_cat[cat]
    total_r1_gt = sum(r['n_r1_gt_prod'] for r in rs)
    total_r1_gt_correct = sum(r['n_r1_gt_prod_correct'] for r in rs)
    false_alarms = total_r1_gt - total_r1_gt_correct
    print(f"  {cat}: {false_alarms:,} false-alarm voxels (R1 crosses threshold with wrong label, "
         f"among R1>prod voxels)")

print("\n" + "="*100)
print("PER-SUBJECT DISTANT-BACKGROUND FP RATE (checking for a few catastrophic subjects)")
print("="*100)
db_rows = by_cat['distant_background']
by_subj = defaultdict(lambda: [0, 0])  # [n_r1_gt, n_r1_gt_correct]
for r in db_rows:
    by_subj[r['subject_id']][0] += r['n_r1_gt_prod']
    by_subj[r['subject_id']][1] += r['n_r1_gt_prod_correct']
fp_rates = []
for sid, (n_gt, n_correct) in by_subj.items():
    fp = n_gt - n_correct
    fp_rates.append(fp)
fp_rates = np.array(fp_rates)
print(f"  n_subjects={len(fp_rates)}  mean_FP={fp_rates.mean():.1f}  median_FP={np.median(fp_rates):.1f}  "
      f"max_FP={fp_rates.max()}  p90={np.percentile(fp_rates,90):.1f}")

print("\n" + "="*100)
print("VERDICT: is a purely spatial gate feasible, or does the gate need feature-space reasoning?")
print("="*100)
lesion_r1_gt_frac = sum(r['n_r1_gt_prod'] for r in by_cat['lesion']) / max(1, sum(r['n_voxels'] for r in by_cat['lesion']))
distant_r1_gt_frac = sum(r['n_r1_gt_prod'] for r in by_cat['distant_background']) / max(1, sum(r['n_voxels'] for r in by_cat['distant_background']))
print(f"Fraction of LESION voxels where R1>prod: {100*lesion_r1_gt_frac:.2f}%")
print(f"Fraction of DISTANT_BACKGROUND voxels where R1>prod: {100*distant_r1_gt_frac:.2f}%")
if distant_r1_gt_frac > 0.3:
    print("\n  ==> R1 out-ranks production almost EVERYWHERE, including distant background --")
    print("      confirms E255: 'trust R1 wherever R1>prod' is NOT spatially selective on its own.")
    print("      A gate based on RAW RANK (R1 vs prod) is insufficient; a gate needs to use")
    print("      D1 feature-space evidence (or spatial proximity) to distinguish genuine")
    print("      lesion-like regions from generic background where R1 is merely more confident")
    print("      than an already-near-zero production baseline.")
else:
    print("\n  ==> R1 outranking production is reasonably CONCENTRATED near real lesion tissue --")
    print("      a simpler spatial or rank-based gate may be geometrically feasible.")
