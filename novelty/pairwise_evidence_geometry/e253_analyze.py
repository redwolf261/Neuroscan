"""E253 Part B analysis -- the decisive counterfactual interpolation
test, per the user's exact framing: does G2-A recovery rise smoothly as
alpha moves toward R1 WITHOUT comparable degradation on detected lesions?
If yes -> possibility A (wrong learned direction, population-robust
readout geometry is a defensible target). If recovery only appears at
alpha=1, or detected/G2-B degrade just as fast, -> possibility B
(something beyond the linear projection, interpolation can't capture it).
"""
import csv
import numpy as np
from collections import defaultdict

rows = list(csv.DictReader(open('E253_interpolation.csv')))
for r in rows:
    r['alpha'] = float(r['alpha'])
    r['max_prob'] = float(r['max_prob'])
    r['recovered'] = int(r['recovered'])
    r['fp_shell'] = float(r['fp_shell'])

by_grp_alpha = defaultdict(list)
for r in rows:
    by_grp_alpha[(r['group'], r['alpha'])].append(r)

print("="*100)
print("RECOVERY RATE vs ALPHA, by group")
print("="*100)
print(f"{'alpha':>6s} {'G2A_recov':>10s} {'G2A_maxp':>10s} {'det_recov':>10s} {'det_maxp':>10s} "
      f"{'G2B_recov':>10s} {'G2B_maxp':>10s} {'det_fp':>10s}")
print("-"*100)

alphas = sorted(set(r['alpha'] for r in rows))
summary = {}
for alpha in alphas:
    g2a_rows = by_grp_alpha[('G2A', alpha)]
    det_rows = by_grp_alpha[('detected', alpha)]
    g2b_rows = by_grp_alpha[('G2B', alpha)]
    g2a_rec = np.mean([r['recovered'] for r in g2a_rows]) if g2a_rows else float('nan')
    g2a_maxp = np.mean([r['max_prob'] for r in g2a_rows]) if g2a_rows else float('nan')
    det_rec = np.mean([r['recovered'] for r in det_rows]) if det_rows else float('nan')
    det_maxp = np.mean([r['max_prob'] for r in det_rows]) if det_rows else float('nan')
    g2b_rec = np.mean([r['recovered'] for r in g2b_rows]) if g2b_rows else float('nan')
    g2b_maxp = np.mean([r['max_prob'] for r in g2b_rows]) if g2b_rows else float('nan')
    det_fp = np.mean([r['fp_shell'] for r in det_rows]) if det_rows else float('nan')
    summary[alpha] = {'g2a_rec': g2a_rec, 'det_rec': det_rec, 'g2b_rec': g2b_rec, 'det_fp': det_fp}
    print(f"{alpha:6.1f} {g2a_rec:10.4f} {g2a_maxp:10.4f} {det_rec:10.4f} {det_maxp:10.4f} "
          f"{g2b_rec:10.4f} {g2b_maxp:10.4f} {det_fp:10.6f}")

print("\n" + "="*100)
print("SHAPE ANALYSIS: is the rise smooth, or a step only at alpha=1?")
print("="*100)
g2a_curve = np.array([summary[a]['g2a_rec'] for a in alphas])
det_curve = np.array([summary[a]['det_rec'] for a in alphas])
g2b_curve = np.array([summary[a]['g2b_rec'] for a in alphas])
det_fp_curve = np.array([summary[a]['det_fp'] for a in alphas])

# fraction of total G2A gain achieved by alpha=0.5 (halfway)
total_gain = g2a_curve[-1] - g2a_curve[0]
half_idx = alphas.index(0.5) if 0.5 in alphas else len(alphas) // 2
gain_at_half = g2a_curve[half_idx] - g2a_curve[0]
frac_at_half = gain_at_half / total_gain if total_gain > 0 else float('nan')
print(f"Total G2-A recovery gain (alpha=0 to alpha=1): {total_gain:+.4f}")
print(f"Gain achieved by alpha=0.5: {gain_at_half:+.4f} ({frac_at_half*100:.1f}% of total)")

det_total_drop = det_curve[0] - det_curve[-1]
det_drop_at_half = det_curve[0] - det_curve[half_idx]
print(f"\nTotal detected recovery DROP (alpha=0 to alpha=1): {det_total_drop:+.4f}")
print(f"Detected drop by alpha=0.5: {det_drop_at_half:+.4f}")

det_fp_total_rise = det_fp_curve[-1] - det_fp_curve[0]
print(f"\nDetected false-positive (shell) rise, alpha=0 to alpha=1: {det_fp_total_rise:+.6f}")

print("\n" + "="*100)
print("VERDICT")
print("="*100)
smooth_rise = frac_at_half > 0.3  # meaningful progress by the midpoint, not all-or-nothing at alpha=1
minimal_det_cost = det_total_drop < 0.05  # detected recovery stays high throughout
minimal_fp_cost = det_fp_total_rise < 0.05

print(f"G2-A rises smoothly (>30% of gain by alpha=0.5): {'YES' if smooth_rise else 'NO'} ({frac_at_half*100:.1f}%)")
print(f"Detected recovery stays high (drop < 5pp): {'YES' if minimal_det_cost else 'NO'} (drop={det_total_drop*100:.1f}pp)")
print(f"Detected FP stays low (rise < 5pp): {'YES' if minimal_fp_cost else 'NO'} (rise={det_fp_total_rise*100:.1f}pp)")

if smooth_rise and minimal_det_cost and minimal_fp_cost:
    print("\n  ==> POSSIBILITY A CONFIRMED: G2-A recovery rises smoothly toward R1's direction")
    print("      WITHOUT comparable degradation on detected lesions or a false-positive cost.")
    print("      The production head's learned direction is genuinely too specialized to the")
    print("      dominant detected-lesion population -- population-robust readout geometry is")
    print("      a defensible, evidence-backed target for a real intervention design.")
elif not smooth_rise:
    print("\n  ==> Recovery does NOT rise smoothly -- most of the gain requires alpha near 1")
    print("      (full R1). This suggests R1's success may depend on something the simple")
    print("      linear interpolation cannot capture (possibility B) -- a full retrained")
    print("      direction may be needed, not a small nudge from production's own weights.")
else:
    print("\n  ==> G2-A recovery rises but at a REAL cost to detected lesions and/or false")
    print("      positives -- this is a genuine trade-off, not a free lunch. Any readout")
    print("      redesign would need to find a way to gain G2-A without this cost, which a")
    print("      simple linear interpolation of the two existing readouts cannot achieve.")
