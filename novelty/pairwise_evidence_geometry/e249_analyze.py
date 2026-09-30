"""E249 analysis -- channel rescue, per the user's exact decisive
comparison: NOT "does rescue improve Dice" but "target-channel rescue >
random-channel rescue" (and > dead-channel rescue, > shuffled-rescue),
tested on BOTH recovery/probability AND separability (representation-
level, not just output-level, per this session's E237-informed
discipline), with explicit G2-A vs G2-B specificity check.
"""
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

rows = list(csv.DictReader(open('E249_rescue.csv')))
for r in rows:
    r['max_prob'] = float(r['max_prob'])
    r['mean_prob'] = float(r['mean_prob'])
    r['recovered'] = int(r['recovered'])
    r['sep_D1'] = float(r['sep_D1']) if r['sep_D1'] != '' else None
    r['size'] = int(r['size'])

CONDITIONS = ['baseline', 'target_rescue', 'random_rescue', 'dead_rescue', 'shuffled_rescue']

by_lesion = defaultdict(dict)
for r in rows:
    key = (r['group'], r['subject_id'], r['comp_id'])
    by_lesion[key][r['condition']] = r

lesions_g2a = [k for k in by_lesion if k[0] == 'G2A' and all(c in by_lesion[k] for c in CONDITIONS)]
lesions_g2b = [k for k in by_lesion if k[0] == 'G2B' and all(c in by_lesion[k] for c in CONDITIONS)]
print(f"Complete G2-A lesions: {len(lesions_g2a)}")
print(f"Complete G2-B lesions: {len(lesions_g2b)}")


def summarize_group(lesion_keys, label):
    print(f"\n{'='*100}\n{label} (n={len(lesion_keys)})\n{'='*100}")
    for metric in ['max_prob', 'mean_prob', 'recovered', 'sep_D1']:
        print(f"\n  --- {metric} ---")
        means = {}
        for cond in CONDITIONS:
            vals = np.array([by_lesion[k][cond][metric] for k in lesion_keys
                             if by_lesion[k][cond][metric] is not None])
            means[cond] = vals
            print(f"    {cond:18s}: mean={vals.mean():.4f}  median={np.median(vals):.4f}  n={len(vals)}")
        yield metric, means


def paired_compare(lesion_keys, metric, cond_a, cond_b):
    a = np.array([by_lesion[k][cond_a][metric] for k in lesion_keys
                 if by_lesion[k][cond_a][metric] is not None and by_lesion[k][cond_b][metric] is not None])
    b = np.array([by_lesion[k][cond_b][metric] for k in lesion_keys
                 if by_lesion[k][cond_a][metric] is not None and by_lesion[k][cond_b][metric] is not None])
    diff = a - b
    if len(diff) < 2 or np.all(diff == 0):
        return diff.mean() if len(diff) else float('nan'), 1.0, len(diff)
    _, p = stats.wilcoxon(diff)
    return diff.mean(), p, len(diff)


for metric_gen in summarize_group(lesions_g2a, "G2-A (primary population)"):
    pass

print("\n" + "="*100)
print("DECISIVE COMPARISON (G2-A): target_rescue vs each control")
print("="*100)
for metric in ['max_prob', 'mean_prob', 'recovered', 'sep_D1']:
    print(f"\n  --- {metric} ---")
    for control in ['baseline', 'random_rescue', 'dead_rescue', 'shuffled_rescue']:
        diff, p, n = paired_compare(lesions_g2a, metric, 'target_rescue', control)
        sig = ' ***' if p < 0.001 else (' **' if p < 0.01 else (' *' if p < 0.05 else ''))
        print(f"    target_rescue - {control:16s}: diff={diff:+.4f}  p={p:.4e}  n={n}{sig}")

for metric_gen in summarize_group(lesions_g2b, "G2-B (specificity control)"):
    pass

print("\n" + "="*100)
print("SPECIFICITY CHECK: target_rescue vs random_rescue, G2-A vs G2-B")
print("="*100)
for metric in ['max_prob', 'sep_D1']:
    diff_a, p_a, n_a = paired_compare(lesions_g2a, metric, 'target_rescue', 'random_rescue')
    diff_b, p_b, n_b = paired_compare(lesions_g2b, metric, 'target_rescue', 'random_rescue')
    print(f"\n  {metric}:")
    print(f"    G2-A: diff={diff_a:+.4f}  p={p_a:.4e}  n={n_a}")
    print(f"    G2-B: diff={diff_b:+.4f}  p={p_b:.4e}  n={n_b}")

print("\n" + "="*100)
print("VERDICT")
print("="*100)
diff_maxp, p_maxp, _ = paired_compare(lesions_g2a, 'max_prob', 'target_rescue', 'random_rescue')
diff_sep, p_sep, _ = paired_compare(lesions_g2a, 'sep_D1', 'target_rescue', 'random_rescue')
diff_maxp_dead, p_maxp_dead, _ = paired_compare(lesions_g2a, 'max_prob', 'target_rescue', 'dead_rescue')
diff_maxp_shuf, p_maxp_shuf, _ = paired_compare(lesions_g2a, 'max_prob', 'target_rescue', 'shuffled_rescue')

beats_random = p_maxp < 0.05 and diff_maxp > 0
beats_random_sep = p_sep < 0.05 and diff_sep > 0
beats_dead = p_maxp_dead < 0.05 and diff_maxp_dead > 0
beats_shuffled = p_maxp_shuf < 0.05 and diff_maxp_shuf > 0

print(f"target_rescue > random_rescue (max_prob): {'YES' if beats_random else 'NO'} (p={p_maxp:.2e})")
print(f"target_rescue > random_rescue (separability): {'YES' if beats_random_sep else 'NO'} (p={p_sep:.2e})")
print(f"target_rescue > dead_rescue: {'YES' if beats_dead else 'NO'} (p={p_maxp_dead:.2e})")
print(f"target_rescue > shuffled_rescue: {'YES' if beats_shuffled else 'NO'} (p={p_maxp_shuf:.2e})")

if beats_random and beats_dead and beats_shuffled:
    print("\n  ==> CAUSAL EFFECT CONFIRMED: target-channel rescue beats ALL controls")
    print("      (random, dead, and spatially-shuffled). This is not a generic magnitude")
    print("      effect (per E237's own lesson) -- the SPECIFIC identified channels, in")
    print("      their correct spatial arrangement, causally matter for G2-A recovery.")
    if beats_random_sep:
        print("      AND this holds at the representation level (separability), not just the")
        print("      final probability -- a genuine representational rescue, not a threshold push.")
else:
    print("\n  ==> CAUSAL EFFECT NOT CONFIRMED (or only partially): the target channels do NOT")
    print("      clearly beat all controls. Per this session's E237-informed discipline, do")
    print("      NOT treat this as evidence for a specific-channel causal mechanism -- report")
    print("      exactly which comparisons held and which did not.")
