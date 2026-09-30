"""E237 analysis -- five pre-registered kill criteria, per the user's exact
spec, evaluated BEFORE any positive interpretation is reported.

Primary endpoint: binary lesion recovery Y = 1[max_p_in_lesion > 0.5],
paired subject-level comparison (per-subject recovery RATE across that
subject's lesions in the group), intervention vs baseline (alpha=0).

Kill criteria (report ALL five regardless of outcome, do not stop early):
  A. No effect on G2 -- if the 'real' condition's recovery rate at the
     best positive alpha does not exceed baseline (alpha=0) by a
     significant, non-trivial margin on G2 (primary population), the
     phenomenon is NOT causally actionable. FAIL -> kill.
  B. Random-direction control matches real effect -- if 'random_dir'
     produces the same recovery-rate lift as 'real' at matched alpha,
     the effect is generic (any large enough additive push recovers
     lesions), not specific to the D2 probe's learned direction. FAIL -> kill.
  C. Spatial-shuffle control matches real effect -- if 'shuffled' (real
     per-voxel MAGNITUDES, wrong spatial arrangement) matches 'real',
     the effect is about total injected magnitude, not genuine spatial
     evidence localization. FAIL -> kill.
  D. Only extreme alpha works, at high FP cost -- if recovery only
     appears at |alpha|>=2 (grid boundary) while also driving up false
     positives elsewhere (checked via mean_prob outside lesion mask is
     NOT measured in Stage 2 -- flagged as a scope note; using
     within-lesion mean_prob saturation as a weaker proxy), the result
     is not a usable, well-behaved effect.
  E. G3 improves as much as G2 -- if the ordinary (lower-contrast)
     matched-missed population recovers as readily as the high-contrast
     population, the effect is not specific to "evidence exists but is
     unused" (which per E233/E235/E236 should be a G2-specific
     phenomenon) but a generic decision-boundary push applicable to any
     missed lesion regardless of underlying evidence.
"""
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

rows = list(csv.DictReader(open('E237_intervention.csv')))
for r in rows:
    r['alpha'] = float(r['alpha'])
    r['max_prob'] = float(r['max_prob'])
    r['mean_prob'] = float(r['mean_prob'])
    r['recovered'] = int(r['recovered'])
    r['size'] = int(r['size'])

def subject_rates(sub_rows, group, condition, alpha):
    """Per-subject recovery rate for one (group, condition, alpha)."""
    filt = [r for r in sub_rows if r['group'] == group and r['condition'] == condition and r['alpha'] == alpha]
    by_subj = defaultdict(list)
    for r in filt:
        by_subj[r['subject_id']].append(r['recovered'])
    return {sid: np.mean(v) for sid, v in by_subj.items()}

def paired_test(rates_a, rates_b):
    """Paired Wilcoxon over subjects present in both."""
    common = sorted(set(rates_a) & set(rates_b))
    if len(common) < 2:
        return None, None, len(common)
    a = np.array([rates_a[s] for s in common])
    b = np.array([rates_b[s] for s in common])
    diff = a - b
    if np.all(diff == 0):
        return 0.0, 1.0, len(common)
    try:
        w_stat, p = stats.wilcoxon(diff)
    except ValueError:
        p = None
    return diff.mean(), p, len(common)

ALPHAS_POS = [0.25, 0.5, 1.0, 2.0]
ALPHAS_NEG = [-0.5, -1.0, -2.0]
GROUPS = ['G2_high_contrast_missed', 'G3_matched_missed']

print("=" * 100)
print("BASELINE (alpha=0, real condition) recovery rates")
print("=" * 100)
baseline = {}
for g in GROUPS:
    r0 = subject_rates(rows, g, 'real', 0.0)
    baseline[g] = r0
    overall = np.mean(list(r0.values())) if r0 else float('nan')
    print(f"  {g}: n_subjects={len(r0)}  mean recovery rate={overall:.4f}")

print("\n" + "=" * 100)
print("RECOVERY RATE BY CONDITION x ALPHA (mean across subjects)")
print("=" * 100)
summary = {}
for g in GROUPS:
    print(f"\n--- {g} ---")
    for cond in ['real', 'random_dir', 'shuffled']:
        line = f"  {cond:12s}"
        for a in [-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0]:
            rr = subject_rates(rows, g, cond, a)
            m = np.mean(list(rr.values())) if rr else float('nan')
            summary[(g, cond, a)] = (rr, m)
            line += f"  a={a:+.2f}:{m:.3f}"
        print(line)

print("\n" + "=" * 100)
print("KILL CRITERION A: no effect on G2 (primary population)")
print("=" * 100)
g2_base = baseline['G2_high_contrast_missed']
best_a_g2, best_p_g2, best_delta_g2 = None, None, -1
for a in ALPHAS_POS:
    rr, _ = summary[('G2_high_contrast_missed', 'real', a)]
    delta, p, n = paired_test(rr, g2_base)
    print(f"  alpha={a:+.2f}: delta={delta if delta is not None else float('nan'):.4f}  p={p}  n_paired={n}")
    if delta is not None and delta > best_delta_g2:
        best_delta_g2 = delta; best_a_g2 = a; best_p_g2 = p
crit_a_pass = best_delta_g2 is not None and best_delta_g2 > 0.10 and best_p_g2 is not None and best_p_g2 < 0.05
print(f"  BEST: alpha={best_a_g2}, delta={best_delta_g2:.4f}, p={best_p_g2}")
print(f"  Criterion A {'SURVIVES (real effect on G2 detected)' if crit_a_pass else 'FAILS -> KILL'}")

print("\n" + "=" * 100)
print("KILL CRITERION B: random-direction control vs real, at best alpha")
print("=" * 100)
rr_real, _ = summary[('G2_high_contrast_missed', 'real', best_a_g2)] if best_a_g2 else (g2_base, None)
rr_rand, _ = summary[('G2_high_contrast_missed', 'random_dir', best_a_g2)] if best_a_g2 else (g2_base, None)
delta_rb, p_rb, n_rb = paired_test(rr_real, rr_rand)
print(f"  at alpha={best_a_g2}: real - random_dir delta={delta_rb:.4f}  p={p_rb}  n_paired={n_rb}")
crit_b_pass = delta_rb is not None and (p_rb is None or p_rb < 0.05) and delta_rb > 0.05
print(f"  Criterion B {'SURVIVES (real beats random control)' if crit_b_pass else 'FAILS -> random control matches real -> KILL'}")

print("\n" + "=" * 100)
print("KILL CRITERION C: spatial-shuffle control vs real, at best alpha")
print("=" * 100)
rr_shuf, _ = summary[('G2_high_contrast_missed', 'shuffled', best_a_g2)] if best_a_g2 else (g2_base, None)
delta_rc, p_rc, n_rc = paired_test(rr_real, rr_shuf)
print(f"  at alpha={best_a_g2}: real - shuffled delta={delta_rc:.4f}  p={p_rc}  n_paired={n_rc}")
crit_c_pass = delta_rc is not None and (p_rc is None or p_rc < 0.05) and delta_rc > 0.05
print(f"  Criterion C {'SURVIVES (real beats shuffle control)' if crit_c_pass else 'FAILS -> shuffle control matches real -> KILL'}")

print("\n" + "=" * 100)
print("KILL CRITERION D: does the effect appear at a moderate alpha, or only the extreme grid boundary?")
print("=" * 100)
appears_moderate = False
for a in [0.25, 0.5, 1.0]:
    rr, m = summary[('G2_high_contrast_missed', 'real', a)]
    delta, p, n = paired_test(rr, g2_base)
    sig = delta is not None and delta > 0.10 and (p is not None and p < 0.05)
    delta_str = f"{delta:.4f}" if delta is not None else "nan"
    print(f"  alpha={a:+.2f}: mean_rate={m:.3f}  delta_vs_base={delta_str}  p={p}  {'MODERATE-ALPHA EFFECT' if sig else ''}")
    if sig:
        appears_moderate = True
crit_d_pass = appears_moderate
print(f"  Criterion D {'SURVIVES (effect present at moderate alpha, not only alpha=2 boundary)' if crit_d_pass else 'FAILS -> only extreme alpha works -> KILL'}")

print("\n" + "=" * 100)
print("KILL CRITERION E: does G3 (lower-contrast, ordinary missed) recover as readily as G2?")
print("=" * 100)
g3_base = baseline['G3_matched_missed']
if best_a_g2 is not None:
    rr_g3, m_g3 = summary[('G3_matched_missed', 'real', best_a_g2)]
    delta_g3, p_g3, n_g3 = paired_test(rr_g3, g3_base)
    m_g2 = summary[('G2_high_contrast_missed', 'real', best_a_g2)][1]
    print(f"  at alpha={best_a_g2}: G2 delta={best_delta_g2:.4f} (from {np.mean(list(g2_base.values())):.3f} to {m_g2:.3f})")
    print(f"  at alpha={best_a_g2}: G3 delta={delta_g3:.4f} (from {np.mean(list(g3_base.values())):.3f} to {m_g3:.3f})  p={p_g3}  n={n_g3}")
    crit_e_pass = delta_g3 is not None and best_delta_g2 is not None and (best_delta_g2 - delta_g3) > 0.05
    print(f"  Criterion E {'SURVIVES (G2 gains MORE than G3 -- effect is contrast/evidence-specific)' if crit_e_pass else 'FAILS -> G3 improves as much as G2 -> generic decision-boundary push, KILL'}")
else:
    crit_e_pass = False
    print("  Criterion E: N/A, no valid best alpha for G2")

print("\n" + "=" * 100)
print("OVERALL VERDICT")
print("=" * 100)
all_pass = crit_a_pass and crit_b_pass and crit_c_pass and crit_d_pass and crit_e_pass
print(f"  A (real effect on G2):         {'PASS' if crit_a_pass else 'FAIL'}")
print(f"  B (beats random-dir control):  {'PASS' if crit_b_pass else 'FAIL'}")
print(f"  C (beats shuffle control):     {'PASS' if crit_c_pass else 'FAIL'}")
print(f"  D (moderate-alpha, not only extreme): {'PASS' if crit_d_pass else 'FAIL'}")
print(f"  E (G2-specific, not generic push on G3): {'PASS' if crit_e_pass else 'FAIL'}")
if all_pass:
    print("\n  ==> E237 SURVIVES ALL FIVE KILL CRITERIA. This is the first CAUSAL")
    print("      confirmation in the entire investigation: injecting the D2 probe's")
    print("      own learned direction into the D1 pathway causally recovers")
    print("      high-contrast-missed lesions, beyond what a random direction or")
    print("      shuffled-magnitude control achieves, more than it helps ordinary")
    print("      (lower-contrast) missed lesions.")
else:
    print("\n  ==> E237 FAILS at least one kill criterion. The phenomenon does NOT")
    print("      survive as a clean, specific, causally-actionable effect under the")
    print("      pre-registered controls. Report exactly which criterion failed and")
    print("      why, per this session's discipline -- do not manufacture a story.")
