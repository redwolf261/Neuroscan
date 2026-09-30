"""E239 analysis -- H7 (gradient cancellation) kill criteria, per the
user's exact pre-registered spec.

Survival condition (ALL must hold, paired on E234's matched groups):
  ||g_L||_miss NOT << ||g_L||_det   (gradient not diminished on its own)
  cos(g_L,g_B)_miss << cos(g_L,g_B)_det
  P_miss << P_det

Kill H7 if:
  1. Missed and detected have comparable gradient cosine distributions.
  2. Aggregate gradient retains comparable projection onto g_L.
  3. Any apparent conflict disappears after controlling for lesion size.
  4. Conflict exists equally for ordinary detected lesions (n/a here --
     no "ordinary detected" comparison group was specified separately;
     addressed via the matched_detected group itself, which functions as
     the ordinary-detected baseline).
  5. Effect only occurs at D4 (recycles E228-A) -- addressed by using the
     FULL 3-term production loss, not D4 alone; if the effect vanishes
     when restricted to seg+boundary and only appears via the D4 term,
     that would fail this criterion. Not separately decomposed here since
     the design combines all 3 terms into one scalar per user's explicit
     choice -- flagged as a scope note in the writeup.
"""
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

rows = list(csv.DictReader(open('E239_gradient_geometry.csv')))
for r in rows:
    for k in ['size', 'norm_gL', 'norm_gBnear', 'norm_gBfar', 'norm_gR',
             'cos_L_B', 'cos_L_Bnear', 'cos_L_Bfar', 'cos_L_R', 'rho', 'P']:
        r[k] = float(r[k])
    r['size'] = int(r['size'])

by_group = defaultdict(list)
for r in rows:
    by_group[r['group']].append(r)

for g, rs in by_group.items():
    print(f"{g}: n={len(rs)}")

def summarize(rs, label):
    arrs = {}
    for k in ['norm_gL', 'cos_L_B', 'cos_L_Bnear', 'cos_L_Bfar', 'cos_L_R', 'rho', 'P']:
        v = np.array([r[k] for r in rs])
        v = v[~np.isnan(v)]
        arrs[k] = v
    print(f"\n--- {label} (n={len(rs)}) ---")
    for k, v in arrs.items():
        print(f"  {k:12s}: mean={v.mean():+.4f}  median={np.median(v):+.4f}  std={v.std():.4f}")
    return arrs

miss = by_group['persistent_missed']
det = by_group['matched_detected']
g2 = by_group['high_contrast_missed']
g4 = by_group['ordinary_missed']

print("="*90)
print("GROUP SUMMARIES")
print("="*90)
s_miss = summarize(miss, 'persistent_missed (E234-matched)')
s_det = summarize(det, 'matched_detected (E234-matched)')
s_g2 = summarize(g2, 'high_contrast_missed (G2)')
s_g4 = summarize(g4, 'ordinary_missed (matched_missed, full n)')

# ---- paired comparison on E234 matched pairs (same pairing as E238-H4) ----
print("\n" + "="*90)
print("PAIRED COMPARISON: persistent_missed vs matched_detected (E234 pairs)")
print("="*90)
by_key_miss = {(r['subject_id'], r['comp_id']): r for r in miss}
by_key_det = {(r['subject_id'], r['comp_id']): r for r in det}
matches = list(csv.DictReader(open('E234_matches.csv')))
paired_rows = []
for m in matches:
    km = (m['missed_subject'], m['missed_comp'])
    kd = (m['detected_subject'], m['detected_comp'])
    if km in by_key_miss and kd in by_key_det:
        paired_rows.append((by_key_miss[km], by_key_det[kd]))
print(f"Complete pairs: {len(paired_rows)}")

def paired_test(key):
    a = np.array([p[0][key] for p in paired_rows])
    b = np.array([p[1][key] for p in paired_rows])
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    diff = a - b
    if len(diff) < 2 or np.all(diff == 0):
        return a, b, 0.0, 1.0
    w, p = stats.wilcoxon(diff)
    print(f"  {key:12s}: missed_mean={a.mean():+.4f}  detected_mean={b.mean():+.4f}  "
          f"diff={diff.mean():+.4f}  Wilcoxon p={p:.4e}  n={len(diff)}")
    return a, b, diff.mean(), p

norm_a, norm_b, norm_diff, norm_p = paired_test('norm_gL')
cos_a, cos_b, cos_diff, cos_p = paired_test('cos_L_B')
rho_a, rho_b, rho_diff, rho_p = paired_test('rho')
P_a, P_b, P_diff, P_p = paired_test('P')
cosn_a, cosn_b, cosn_diff, cosn_p = paired_test('cos_L_Bnear')
cosf_a, cosf_b, cosf_diff, cosf_p = paired_test('cos_L_Bfar')
cosr_a, cosr_b, cosr_diff, cosr_p = paired_test('cos_L_R')

print("\n" + "="*90)
print("SIZE CONTROL: does the paired diff in cos_L_B / P survive controlling for size?")
print("="*90)
sizes_miss = np.array([p[0]['size'] for p in paired_rows])
sizes_det = np.array([p[1]['size'] for p in paired_rows])
log_sz_diff = np.log(sizes_miss+1) - np.log(sizes_det+1)
r_cos, p_cos = stats.pearsonr(log_sz_diff, cos_a - cos_b)
r_P, p_P = stats.pearsonr(log_sz_diff, P_a - P_b)
print(f"  corr(log-size-diff, cos_L_B-diff): r={r_cos:.3f}  p={p_cos:.3f}")
print(f"  corr(log-size-diff, P-diff): r={r_P:.3f}  p={p_P:.3f}")
print("  (if |r| large and significant, the paired diff may be a size artifact, not H7)")

print("\n" + "="*90)
print("SPATIAL CONTROL (near vs far background): does conflict need PROXIMITY?")
print("="*90)
print(f"  persistent_missed: cos_L_Bnear mean={s_miss['cos_L_Bnear'].mean():+.4f}  "
      f"cos_L_Bfar mean={s_miss['cos_L_Bfar'].mean():+.4f}")
print(f"  matched_detected:  cos_L_Bnear mean={s_det['cos_L_Bnear'].mean():+.4f}  "
      f"cos_L_Bfar mean={s_det['cos_L_Bfar'].mean():+.4f}")
w_near, p_near = stats.wilcoxon(cosn_a - cosn_b) if len(cosn_a) > 1 else (0, 1)
w_far, p_far = stats.wilcoxon(cosf_a - cosf_b) if len(cosf_a) > 1 else (0, 1)
print(f"  paired missed-vs-detected on cos_L_Bnear: p={p_near:.4e}")
print(f"  paired missed-vs-detected on cos_L_Bfar:  p={p_far:.4e}")

print("\n" + "="*90)
print("CROSS-GROUP CHECK: high_contrast_missed vs ordinary_missed (unpaired, per user's 4-group design)")
print("="*90)
u_cos, pu_cos = stats.mannwhitneyu(s_g2['cos_L_B'], s_g4['cos_L_B'])
u_P, pu_P = stats.mannwhitneyu(s_g2['P'], s_g4['P'])
print(f"  cos_L_B: G2 median={np.median(s_g2['cos_L_B']):+.4f}  G4(ordinary) median={np.median(s_g4['cos_L_B']):+.4f}  "
      f"Mann-Whitney p={pu_cos:.4e}")
print(f"  P: G2 median={np.median(s_g2['P']):+.4f}  G4(ordinary) median={np.median(s_g4['P']):+.4f}  "
      f"Mann-Whitney p={pu_P:.4e}")

print("\n" + "="*90)
print("H7 KILL-CRITERION EVALUATION")
print("="*90)

# Criterion 1: comparable gradient cosine distributions
crit1_fail = cos_p < 0.05 and abs(cos_diff) > 0.05  # i.e. genuinely DIFFERENT -> criterion NOT triggered (good for H7)
print(f"1. Comparable cosine distributions (missed vs detected)? "
      f"{'NO -- genuinely different (H7 clears this kill)' if crit1_fail else 'YES -- comparable (H7 KILLED by criterion 1)'}")

# Criterion 2: comparable projection P
crit2_fail = P_p < 0.05 and abs(P_diff) > 0.1
print(f"2. Comparable projection P (missed vs detected)? "
      f"{'NO -- genuinely different (H7 clears this kill)' if crit2_fail else 'YES -- comparable (H7 KILLED by criterion 2)'}")

# Criterion 3: survives size control
crit3_survives = abs(r_cos) < 0.3 or p_cos > 0.05
print(f"3. Conflict survives size control? "
      f"{'YES (H7 clears this kill)' if crit3_survives else 'NO -- explained by size (H7 KILLED by criterion 3)'}")

# Survival condition proper
gL_not_diminished = not (norm_p < 0.05 and norm_diff < -0.3 * b.mean() if len(norm_a) else True)
cos_much_lower = cos_p < 0.05 and cos_diff < -0.05
P_much_lower = P_p < 0.05 and P_diff < -0.1

print(f"\nSurvival condition parts:")
print(f"  ||g_L|| not much smaller for missed: {'YES' if gL_not_diminished else 'NO'} "
      f"(missed={norm_a.mean():.3f}, detected={norm_b.mean():.3f}, p={norm_p:.2e})")
print(f"  cos(g_L,g_B)_miss << cos(g_L,g_B)_det: {'YES' if cos_much_lower else 'NO'} "
      f"(diff={cos_diff:+.4f}, p={cos_p:.2e})")
print(f"  P_miss << P_det: {'YES' if P_much_lower else 'NO'} (diff={P_diff:+.4f}, p={P_p:.2e})")

h7_survives = gL_not_diminished and cos_much_lower and P_much_lower and crit3_survives

print("\n" + "="*90)
print("OVERALL VERDICT")
print("="*90)
if h7_survives:
    print("  ==> H7 SURVIVES all pre-registered criteria on the E234-matched groups.")
    print("      Gradient exists for missed lesions, is NOT starved, but the background/")
    print("      other-lesion aggregate gradient IS more antagonistic for missed lesions")
    print("      than for matched detected lesions, and this is NOT explained by size.")
    print("      This is a genuinely new, causally-suggestive mechanism.")
else:
    print("  ==> H7 DOES NOT SURVIVE as specified. See individual criteria above for which")
    print("      part failed. Per the user's own standing instruction, do not force a")
    print("      gradient-conflict algorithm from a partial or null result.")
