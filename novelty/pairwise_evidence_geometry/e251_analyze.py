"""E251 analysis -- readout sufficiency verdict, per the user's exact
framework: the primary endpoint is LESION-LEVEL RECOVERY (not just AUC),
compared R1 (independent, detected-only-trained readout) vs R0
(production seg_head), separately for detected/G2-A/G2-B.

Strongest possible result: R1(G2A) >> R0(G2A) while R1 does not show
the same gap on detected -- would establish D1 contains information the
production readout fails to use. If R1 ALSO fails on G2-A, the readout
hypothesis is wrong and the mechanism search must reopen.
"""
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

rows = list(csv.DictReader(open('E251_readout.csv')))
for r in rows:
    r['r0_max_prob'] = float(r['r0_max_prob'])
    r['r0_recovered'] = int(r['r0_recovered'])
    r['r1_max_prob'] = float(r['r1_max_prob'])
    r['r1_recovered'] = int(r['r1_recovered'])

by_grp = defaultdict(list)
for r in rows:
    by_grp[r['group']].append(r)

print("="*100)
print("LESION-LEVEL RECOVERY RATE: R0 (production) vs R1 (independent, detected-only-trained)")
print("="*100)
for grp in ['detected', 'G2A', 'G2B']:
    rs = by_grp[grp]
    r0_rate = np.mean([r['r0_recovered'] for r in rs])
    r1_rate = np.mean([r['r1_recovered'] for r in rs])
    print(f"\n  {grp} (n={len(rs)}):")
    print(f"    R0 (production seg_head) recovery rate: {r0_rate:.4f}")
    print(f"    R1 (independent D1 readout) recovery rate: {r1_rate:.4f}")
    print(f"    delta (R1 - R0): {r1_rate - r0_rate:+.4f}")

print("\n" + "="*100)
print("PAIRED TEST (same lesions, R1 vs R0 recovery)")
print("="*100)
for grp in ['detected', 'G2A', 'G2B']:
    rs = by_grp[grp]
    r0 = np.array([r['r0_recovered'] for r in rs])
    r1 = np.array([r['r1_recovered'] for r in rs])
    diff = r1 - r0
    if np.all(diff == 0):
        p = 1.0
    else:
        try:
            _, p = stats.wilcoxon(diff)
        except ValueError:
            p = float('nan')
    n_r1_only = int(((r1 == 1) & (r0 == 0)).sum())
    n_r0_only = int(((r0 == 1) & (r1 == 0)).sum())
    print(f"\n  {grp}: R1-only recovered={n_r1_only}  R0-only recovered={n_r0_only}  "
          f"mean_diff={diff.mean():+.4f}  Wilcoxon p={p:.4e}")

print("\n" + "="*100)
print("MAX_PROB DISTRIBUTIONS (continuous comparison)")
print("="*100)
for grp in ['detected', 'G2A', 'G2B']:
    rs = by_grp[grp]
    r0p = np.array([r['r0_max_prob'] for r in rs])
    r1p = np.array([r['r1_max_prob'] for r in rs])
    diff = r1p - r0p
    _, p = stats.wilcoxon(diff) if not np.all(diff == 0) else (0, 1)
    print(f"  {grp}: R0 mean={r0p.mean():.4f}  R1 mean={r1p.mean():.4f}  "
          f"diff={diff.mean():+.4f}  Wilcoxon p={p:.4e}")

print("\n" + "="*100)
print("VERDICT")
print("="*100)
g2a = by_grp['G2A']
det = by_grp['detected']
g2a_r0 = np.mean([r['r0_recovered'] for r in g2a])
g2a_r1 = np.mean([r['r1_recovered'] for r in g2a])
det_r0 = np.mean([r['r0_recovered'] for r in det])
det_r1 = np.mean([r['r1_recovered'] for r in det])

g2a_gap = g2a_r1 - g2a_r0
det_gap = det_r1 - det_r0

r0_arr = np.array([r['r0_recovered'] for r in g2a])
r1_arr = np.array([r['r1_recovered'] for r in g2a])
diff_arr = r1_arr - r0_arr
if np.all(diff_arr == 0):
    p_g2a = 1.0
else:
    try:
        _, p_g2a = stats.wilcoxon(diff_arr)
    except ValueError:
        p_g2a = float('nan')

print(f"G2-A: R1 recovery gain over R0 = {g2a_gap:+.4f} (p={p_g2a:.2e})")
print(f"detected: R1 recovery gain over R0 = {det_gap:+.4f}")
print(f"G2-A-specific gap (G2-A gain minus detected gain): {g2a_gap - det_gap:+.4f}")

r1_recovers_g2a = p_g2a < 0.05 and g2a_gap > 0.05
gap_is_g2a_specific = (g2a_gap - det_gap) > 0.03

if r1_recovers_g2a and gap_is_g2a_specific:
    print("\n  ==> READOUT HYPOTHESIS SURVIVES: an independent readout that NEVER saw a")
    print("      missed lesion during training recovers G2-A lesions from D1 features alone,")
    print("      significantly more than the production seg_head does, and this gain is")
    print("      G2-A-specific (not just a general R1-vs-R0 difference). D1 genuinely")
    print("      contains information the production readout fails to use.")
    print("      NEXT: investigate WHY seg_head suppresses this -- adaptive readout/")
    print("      calibration becomes a defensible design direction.")
else:
    print("\n  ==> READOUT HYPOTHESIS DOES NOT SURVIVE (or only weakly). Per the user's own")
    print("      explicit instruction: do NOT start inventing calibration modules on this")
    print("      basis. The mechanism search must reopen rather than assume a readout fix.")
