"""E246 analysis -- two parts, per the user's exact spec.

PART 1: three-way comparison of r0 (clipping fraction) -- G2-A missed vs
matched detected vs G2-B partial, both lesion and shell, plus the
lesion-minus-shell differential (the shell control).

PART 2: mediation test -- does Delta_S_ReLU (= S(relu1)-S(bn1), from
E245's own per-lesion data) or r0 itself correlate with Delta_S_full
(= S(dec1)-S(cat1), from E244's own per-lesion data), across G2-A
lesions specifically (missed only, since that's where the deficit is).
"""
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

# ---- load E246's clipping data ----
clip_rows = list(csv.DictReader(open('E246_clipping.csv')))
for r in clip_rows:
    r['pair_idx'] = int(r['pair_idx'])
    r['r0_lesion'] = float(r['r0_lesion'])
    r['r0_shell'] = float(r['r0_shell'])
    r['r0_diff'] = float(r['r0_diff'])

g2a_missed = [r for r in clip_rows if r['group'] == 'G2A_missed']
matched_det = [r for r in clip_rows if r['group'] == 'matched_detected']
g2b = [r for r in clip_rows if r['group'] == 'G2B_partial']

print("="*100)
print("PART 1: THREE-WAY r0 (CLIPPING FRACTION) COMPARISON")
print("="*100)


def describe(name, rows):
    r0l = np.array([r['r0_lesion'] for r in rows])
    r0s = np.array([r['r0_shell'] for r in rows])
    r0d = np.array([r['r0_diff'] for r in rows])
    print(f"\n  {name} (n={len(rows)}):")
    print(f"    r0_lesion: mean={r0l.mean():.4f}  median={np.median(r0l):.4f}")
    print(f"    r0_shell:  mean={r0s.mean():.4f}  median={np.median(r0s):.4f}")
    print(f"    r0_diff (lesion-shell): mean={r0d.mean():+.4f}  median={np.median(r0d):+.4f}")
    return r0l, r0s, r0d


r0l_a, r0s_a, r0d_a = describe("G2-A missed", g2a_missed)
r0l_d, r0s_d, r0d_d = describe("matched detected", matched_det)
r0l_b, r0s_b, r0d_b = describe("G2-B partial", g2b)

print("\n" + "="*100)
print("PAIRED TEST: G2-A missed vs matched detected (on r0_lesion and r0_diff)")
print("="*100)
by_pair_clip = defaultdict(dict)
for r in clip_rows:
    if r['group'] in ('G2A_missed', 'matched_detected'):
        role = 'missed' if r['group'] == 'G2A_missed' else 'detected'
        by_pair_clip[r['pair_idx']][role] = r
complete_clip = [p for p, d in by_pair_clip.items() if 'missed' in d and 'detected' in d]
print(f"Complete pairs: {len(complete_clip)}")

a_lesion = np.array([by_pair_clip[p]['missed']['r0_lesion'] for p in complete_clip])
d_lesion = np.array([by_pair_clip[p]['detected']['r0_lesion'] for p in complete_clip])
diff_lesion = a_lesion - d_lesion
_, p_lesion = stats.wilcoxon(diff_lesion)
print(f"\nr0_lesion: missed={a_lesion.mean():.4f}  detected={d_lesion.mean():.4f}  "
      f"diff={diff_lesion.mean():+.4f}  Wilcoxon p={p_lesion:.4e}")

a_diff = np.array([by_pair_clip[p]['missed']['r0_diff'] for p in complete_clip])
d_diff = np.array([by_pair_clip[p]['detected']['r0_diff'] for p in complete_clip])
diff_diff = a_diff - d_diff
_, p_diff = stats.wilcoxon(diff_diff)
print(f"r0_diff (lesion-shell): missed={a_diff.mean():+.4f}  detected={d_diff.mean():+.4f}  "
      f"diff={diff_diff.mean():+.4f}  Wilcoxon p={p_diff:.4e}")
print("\n  (r0_diff is the SHELL-CONTROLLED test -- if this is NOT significant even though")
print("   r0_lesion IS, the apparent clipping effect may just reflect G2-A sitting in a")
print("   generally-more-negative local neighborhood, not a lesion-specific phenomenon)")

print("\n" + "="*100)
print("PART 2: MEDIATION TEST -- does clipping predict the separability collapse?")
print("="*100)

# ---- load E245 (dec1 internal stages) for Delta_S_ReLU ----
e245_rows = list(csv.DictReader(open('E245_dec1_internal.csv')))
e245_by_key = {}
for r in e245_rows:
    if r['role'] != 'missed':
        continue
    key = (r['subject_id'], r['comp_id'])
    e245_by_key[key] = {
        'bn1_out': float(r['bn1_out']), 'relu1_out': float(r['relu1_out']),
        'cat1': float(r['cat1']), 'relu2_out': float(r['relu2_out']),
    }

# ---- join with E246's clipping r0 for G2-A missed lesions ----
merged = []
for r in g2a_missed:
    key = (r['subject_id'], r['comp_id'])
    e245_data = e245_by_key.get(key)
    if e245_data is None:
        continue
    delta_S_relu = e245_data['relu1_out'] - e245_data['bn1_out']
    delta_S_full = e245_data['relu2_out'] - e245_data['cat1']  # S(dec1) - S(cat1), = E244's S4-S3
    merged.append({'r0_lesion': r['r0_lesion'], 'r0_diff': r['r0_diff'],
                   'delta_S_relu': delta_S_relu, 'delta_S_full': delta_S_full})

print(f"\nMerged G2-A missed lesions (E246 clipping x E245 stages): n={len(merged)}")

r0l = np.array([m['r0_lesion'] for m in merged])
dS_relu = np.array([m['delta_S_relu'] for m in merged])
dS_full = np.array([m['delta_S_full'] for m in merged])

print("\n--- Delta_S_ReLU vs Delta_S_full (does the ReLU step's own separability")
print("    change predict the FULL cat1->D1 change?) ---")
r_pear, p_pear = stats.pearsonr(dS_relu, dS_full)
r_spear, p_spear = stats.spearmanr(dS_relu, dS_full)
print(f"  Pearson:  r={r_pear:.4f}  p={p_pear:.4e}")
print(f"  Spearman: r={r_spear:.4f}  p={p_spear:.4e}")

print("\n--- r0 (clipping fraction) vs Delta_S_full (does clipping AMOUNT predict")
print("    the full separability collapse?) ---")
r_pear2, p_pear2 = stats.pearsonr(r0l, dS_full)
r_spear2, p_spear2 = stats.spearmanr(r0l, dS_full)
print(f"  Pearson:  r={r_pear2:.4f}  p={p_pear2:.4e}")
print(f"  Spearman: r={r_spear2:.4f}  p={p_spear2:.4e}")

print("\n" + "="*100)
print("VERDICT")
print("="*100)
relu_mediates = p_pear < 0.05 and abs(r_pear) > 0.2
clip_mediates = p_pear2 < 0.05 and abs(r_pear2) > 0.2
shell_controlled_clip_real = p_diff < 0.05

print(f"Delta_S_ReLU correlates with Delta_S_full: {'YES' if relu_mediates else 'NO'} (r={r_pear:.3f}, p={p_pear:.2e})")
print(f"r0 (clipping) correlates with Delta_S_full: {'YES' if clip_mediates else 'NO'} (r={r_pear2:.3f}, p={p_pear2:.2e})")
print(f"Shell-controlled clipping difference (missed vs detected) real: {'YES' if shell_controlled_clip_real else 'NO'} (p={p_diff:.2e})")

if relu_mediates and clip_mediates and shell_controlled_clip_real:
    print("\n  ==> MECHANISM CONFIRMED: negative pre-activation -> ReLU clipping -> loss of")
    print("      discriminative information, surviving the shell control. ReLU clipping")
    print("      specifically explains the cat1->D1 separability collapse for G2-A.")
elif not shell_controlled_clip_real:
    print("\n  ==> CLIPPING DIFFERENCE NOT LESION-SPECIFIC: r0_lesion may differ but r0_diff")
    print("      (shell-controlled) does not survive -- G2-A may simply sit in a generally")
    print("      more-negative local neighborhood, not a lesion-specific clipping phenomenon.")
    print("      The ReLU-clipping mechanism is NOT well supported as specifically about")
    print("      the LESION's own representation.")
elif not (relu_mediates or clip_mediates):
    print("\n  ==> CLIPPING DOES NOT MEDIATE: even if r0/Delta_S_ReLU differ between groups,")
    print("      neither correlates with the FULL cat1->D1 separability collapse across")
    print("      individual G2-A lesions. ReLU clipping co-occurs with the deficit but does")
    print("      not explain its MAGNITUDE. Per the user's own framework: investigate the")
    print("      BN/nonlinear interaction differently rather than forcing the ReLU hypothesis.")
else:
    print("\n  ==> MIXED / PARTIAL support -- report exact pattern, do not force a single verdict.")
