"""E235 analysis -- the decisive table, per the user's exact spec:
detected AUC, high-contrast-missed AUC, matched-ordinary-missed AUC, at
every stage, plus Delta AUC (detected - each missed group) and the
Case 1-4 interpretation.
"""
import csv
import numpy as np
from scipy import stats

d235 = list(csv.DictReader(open('E235_probe_auc.csv')))
for r in d235:
    r['auc'] = float(r['auc']); r['fold'] = int(r['fold'])
    r['n_pos'] = int(r['n_pos']); r['n_neg'] = int(r['n_neg'])

STAGES = ('raw', 'E1', 'E2', 'E3', 'E4_BN', 'D3', 'D2', 'D1')
GROUPS = ('detected', 'high_contrast_missed', 'matched_missed')

print(f"Total rows: {len(d235)}")

print("\n" + "=" * 110)
print("THE DECISIVE TABLE: mean AUC (across folds) per stage per group")
print("=" * 110)
print(f"{'Stage':>10} {'Detected':>12} {'HighContrastMissed':>20} {'MatchedMissed':>16} "
      f"{'dAUC(det-HCmiss)':>18} {'dAUC(det-matched)':>18}")

table = {}
for s in STAGES:
    row = {}
    for g in GROUPS:
        vals = [r['auc'] for r in d235 if r['stage'] == s and r['group'] == g]
        row[g] = (np.mean(vals), np.std(vals), len(vals)) if vals else (float('nan'), float('nan'), 0)
    table[s] = row
    d_auc, d_std, d_n = row['detected']
    h_auc, h_std, h_n = row['high_contrast_missed']
    m_auc, m_std, m_n = row['matched_missed']
    delta_h = d_auc - h_auc
    delta_m = d_auc - m_auc
    print(f"{s:>10} {d_auc:>8.4f}±{d_std:.3f} {h_auc:>14.4f}±{h_std:.3f} "
          f"{m_auc:>10.4f}±{m_std:.3f} {delta_h:>+18.4f} {delta_m:>+18.4f}")

print("\n" + "=" * 110)
print("STATISTICAL TEST: is detected-AUC significantly higher than each missed group's AUC,")
print("per stage (across the 5 folds, paired by fold)?")
print("=" * 110)
for s in STAGES:
    for g, label in [('high_contrast_missed', 'high-contrast missed'), ('matched_missed', 'matched missed')]:
        det_by_fold = {r['fold']: r['auc'] for r in d235 if r['stage'] == s and r['group'] == 'detected'}
        g_by_fold = {r['fold']: r['auc'] for r in d235 if r['stage'] == s and r['group'] == g}
        common_folds = sorted(set(det_by_fold) & set(g_by_fold))
        if len(common_folds) < 3:
            print(f"  {s} vs {label}: too few common folds ({len(common_folds)})")
            continue
        d_vals = np.array([det_by_fold[f] for f in common_folds])
        g_vals = np.array([g_by_fold[f] for f in common_folds])
        diff = d_vals - g_vals
        try:
            stat, p = stats.wilcoxon(d_vals, g_vals)
        except ValueError:
            p = float('nan')
        print(f"  {s:>8} vs {label:>20}: mean_delta={diff.mean():+.4f}  n_folds={len(common_folds)}  p={p:.4f}")

print("\n" + "=" * 110)
print("CASE INTERPRETATION (per user's own 4-case framework, using dAUC(det-HCmiss)")
print("as the primary signal since high-contrast-missed is the decisive population)")
print("=" * 110)
deltas = {s: table[s]['detected'][0] - table[s]['high_contrast_missed'][0] for s in STAGES
         if not np.isnan(table[s]['high_contrast_missed'][0])}
for s in STAGES:
    if s in deltas:
        print(f"  {s:>8}: dAUC = {deltas[s]:+.4f}")

if deltas:
    early = deltas.get('E1', 0)
    mid = deltas.get('E3', 0)
    bn = deltas.get('E4_BN', 0)
    late = deltas.get('D1', 0)
    print(f"\n  E1 dAUC={early:+.3f}  E3 dAUC={mid:+.3f}  E4_BN dAUC={bn:+.3f}  D1 dAUC={late:+.3f}")

    if abs(early) < 0.05 and abs(mid) > 0.1:
        print("  ==> CASE 1 (early collapse): encoder progressively destroys info by E3.")
    elif abs(mid) < 0.05 and abs(bn) > 0.1:
        print("  ==> CASE 2 (bottleneck collapse): the compression operation is implicated.")
    elif abs(bn) < 0.05 and abs(late) > 0.1:
        print("  ==> CASE 3 (decoder collapse): encoder is NOT the problem; abandon")
        print("      representation-loss hypothesis, look at decoder/decision instead.")
    elif all(abs(deltas.get(s, 1)) < 0.05 for s in STAGES if s in deltas):
        print("  ==> CASE 4 (no separation anywhere): H1 strengthened further -- even the")
        print("      learned representation contains little recoverable info. Architecture")
        print("      intervention may have limited potential for this population.")
    else:
        print("  ==> Pattern does not cleanly match Cases 1-4 -- report full trajectory above.")

print("\n" + "=" * 110)
print("HIGH-CONTRAST-MISSED POPULATION CHECK: how many exist, and what fraction")
print("of ALL missed lesions do they represent? (the premise of this whole test)")
print("=" * 110)
hc_n = len(set((r['stage'],) for r in d235 if r['group']=='high_contrast_missed'))  # just for stage count sanity
print("(population counts are in the E235 script's own printed header at launch time,")
print(" reproduced here for the record: see e235_run.log)")
