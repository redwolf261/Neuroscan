"""E234 analysis -- the three-outcome falsification test, per the user's
exact spec.

For matched pairs (missed vs detected, EQUALIZED on size/isolation/dist/
t1c-contrast/t2f-contrast via tight matching, verified NOT significantly
different, see e234_build_matches.py's own match-quality report), compares
separability(missed) vs separability(detected) at EVERY stage.

Outcome A (representation-loss): S_raw comparable, S_Ek DROPS for missed
  in the ENCODER -> preserve weak-but-real evidence.
Outcome B (decoder/decision problem): S_E4 comparable, S_Dk DROPS for
  missed in the DECODER -> change decoder evidence conversion, not
  gradient rescue.
Outcome C (calibration under weak evidence): S_D1 comparable throughout,
  but prediction still absent -> decision-boundary calibration mechanism.
"""
import csv
import numpy as np
from scipy import stats

d234 = list(csv.DictReader(open('E234_layerwise.csv')))
STAGES = ('raw', 'E1', 'E2', 'E3', 'E4_BN', 'D3', 'D2', 'D1')

pairs = {}
for r in d234:
    pid = int(r['pair_id'])
    pairs.setdefault(pid, {})[r['role']] = r

complete_pairs = {pid: p for pid, p in pairs.items() if 'missed' in p and 'detected' in p}
print(f"Total rows: {len(d234)}  complete pairs: {len(complete_pairs)}")

print("\n" + "=" * 100)
print("SEPARABILITY AT EACH STAGE: missed vs detected (paired, matched population)")
print("=" * 100)
print(f"{'stage':>8} {'missed_median':>14} {'detected_median':>16} {'ratio(m/d)':>12} {'wilcoxon_p':>12}")

results = {}
for s in STAGES:
    m_vals, d_vals = [], []
    for pid, p in complete_pairs.items():
        mv = p['missed'][f'sep_{s}']
        dv = p['detected'][f'sep_{s}']
        if mv == '' or dv == '':
            continue
        m_vals.append(float(mv)); d_vals.append(float(dv))
    m_vals = np.array(m_vals); d_vals = np.array(d_vals)
    if len(m_vals) < 5:
        print(f"{s:>8}: too few valid pairs ({len(m_vals)})")
        continue
    med_m, med_d = np.median(m_vals), np.median(d_vals)
    ratio = med_m / med_d if med_d != 0 else float('nan')
    try:
        w_stat, p_val = stats.wilcoxon(m_vals, d_vals)
    except ValueError:
        p_val = float('nan')
    results[s] = {'median_missed': med_m, 'median_detected': med_d, 'ratio': ratio,
                 'p': p_val, 'n': len(m_vals)}
    print(f"{s:>8} {med_m:>14.4f} {med_d:>16.4f} {ratio:>12.4f} {p_val:>12.4e}  (n={len(m_vals)})")

print("\n" + "=" * 100)
print("TREND: does the missed/detected ratio DROP as we go deeper into the encoder,")
print("       then further drop (or recover) through the decoder?")
print("=" * 100)
for s in STAGES:
    if s in results:
        r = results[s]
        sig = '***' if r['p'] < 0.001 else ('**' if r['p'] < 0.01 else ('*' if r['p'] < 0.05 else ''))
        bar_len = int(max(0, min(40, r['ratio'] * 20)))
        print(f"  {s:>8}: ratio={r['ratio']:.3f} {sig:>3}  {'#'*bar_len}")

print("\n" + "=" * 100)
print("OUTCOME CLASSIFICATION")
print("=" * 100)
raw_ok = 'raw' in results and results['raw']['p'] > 0.05  # should be ~equal by construction (matched)
print(f"S_raw comparable between missed/detected (should be TRUE by matching design): "
      f"{'YES' if raw_ok else 'NO -- MATCH QUALITY ISSUE, re-check e234_build_matches.py'}")

encoder_stages = ['E1', 'E2', 'E3', 'E4_BN']
decoder_stages = ['D3', 'D2', 'D1']

encoder_drops = [s for s in encoder_stages if s in results and results[s]['p'] < 0.05 and results[s]['ratio'] < 1]
decoder_drops = [s for s in decoder_stages if s in results and results[s]['p'] < 0.05 and results[s]['ratio'] < 1]

print(f"\nEncoder stages with significant DROP for missed (ratio<1, p<0.05): {encoder_drops}")
print(f"Decoder stages with significant DROP for missed (ratio<1, p<0.05): {decoder_drops}")

e4_ok = 'E4_BN' in results and results['E4_BN']['p'] > 0.05
d1_ok = 'D1' in results and results['D1']['p'] > 0.05

print("\n" + "-" * 100)
if not raw_ok:
    print("WARNING: raw separability differs significantly between matched groups --")
    print("the matching may not have fully equalized H1. Interpret results with caution.")
print("-" * 100)

if encoder_drops and not e4_ok:
    print("\n  ==> OUTCOME A: REPRESENTATION-LOSS in the ENCODER.")
    print("      S_raw comparable, but separability drops significantly during encoding.")
    print("      A genuine information-preservation problem -- the encoder is discarding")
    print("      signal that IS present in the input for these matched-difficulty lesions.")
elif e4_ok and decoder_drops:
    print("\n  ==> OUTCOME B: INFORMATION-ACCESS/DECISION problem in the DECODER.")
    print("      Signal survives the encoder (E4/bottleneck comparable) but is lost")
    print("      specifically during decoding. The fix should change how decoder")
    print("      evidence is converted into segmentation -- NOT another encoder-side")
    print("      or gradient-based intervention.")
elif e4_ok and d1_ok and not decoder_drops:
    print("\n  ==> OUTCOME C: signal survives THROUGHOUT (E4 and D1 both comparable)")
    print("      but the lesion is still missed. Representation is not the bottleneck --")
    print("      the interesting mechanism is DECISION-BOUNDARY CALIBRATION under weak")
    print("      evidence, not a representation-preservation or decoder-access problem.")
else:
    print("\n  ==> MIXED/UNCLEAR pattern -- does not cleanly match outcome A, B, or C.")
    print("      Report the full per-stage table above for manual inspection; the")
    print("      trend may be gradual/distributed rather than localized to one region.")
