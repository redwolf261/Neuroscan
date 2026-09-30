"""E238-H6 analysis -- per the pre-registered kill criterion: a context-
swap effect must beat the random-donor permutation-null, not just be
nonzero (E231's own lesson about marginal-bias artifacts applies
directly here).

For each recipient lesion, average mean_prob/max_prob across its donor
replicates within each condition, then paired-compare (same recipient)
condition vs baseline, and condition vs random_null."""
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

rows = list(csv.DictReader(open('E238_h6_context_swap.csv')))
for r in rows:
    r['mean_prob'] = float(r['mean_prob'])
    r['max_prob'] = float(r['max_prob'])
    r['size'] = int(r['size'])

# key: (subject_id, comp_id) -> condition -> list of mean_prob (avg over donors)
by_lesion = defaultdict(lambda: defaultdict(list))
for r in rows:
    key = (r['subject_id'], r['comp_id'])
    by_lesion[key][r['condition']].append(r['mean_prob'])

lesions = sorted(by_lesion.keys())
conditions = ['baseline', 'same_region', 'loose_match', 'random_null']
agg = {c: [] for c in conditions}
valid_lesions = []
for key in lesions:
    d = by_lesion[key]
    if not all(c in d and len(d[c]) > 0 for c in conditions):
        continue
    valid_lesions.append(key)
    for c in conditions:
        agg[c].append(np.mean(d[c]))

for c in conditions:
    agg[c] = np.array(agg[c])

n = len(valid_lesions)
print(f"Valid lesions (all 4 conditions present): {n}")

print("\n" + "="*90)
print("MEAN mean_prob BY CONDITION")
print("="*90)
for c in conditions:
    print(f"  {c:14s}: mean={agg[c].mean():.4f}  std={agg[c].std():.4f}")

def paired(a, b, label):
    diff = a - b
    if np.all(diff == 0):
        return 0.0, 1.0
    w, p = stats.wilcoxon(diff)
    print(f"  {label}: mean_diff={diff.mean():+.4f}  Wilcoxon p={p:.4e}  n_pos={np.sum(diff>0)}/{len(diff)}")
    return diff.mean(), p

print("\n" + "="*90)
print("STEP 1: does ANY context swap move prediction away from baseline?")
print("="*90)
d_sr_base, p_sr_base = paired(agg['same_region'], agg['baseline'], 'same_region - baseline')
d_lm_base, p_lm_base = paired(agg['loose_match'], agg['baseline'], 'loose_match - baseline')
d_rn_base, p_rn_base = paired(agg['random_null'], agg['baseline'], 'random_null - baseline')

print("\n" + "="*90)
print("STEP 2 (THE KILL CRITERION): does same_region/loose_match beat random_null?")
print("="*90)
d_sr_rn, p_sr_rn = paired(agg['same_region'], agg['random_null'], 'same_region - random_null')
d_lm_rn, p_lm_rn = paired(agg['loose_match'], agg['random_null'], 'loose_match - random_null')
d_sr_lm, p_sr_lm = paired(agg['same_region'], agg['loose_match'], 'same_region - loose_match')

print("\n" + "="*90)
print("H6 KILL-CRITERION EVALUATION")
print("="*90)
sr_beats_null = p_sr_rn < 0.05 and abs(d_sr_rn) > 0.02
lm_beats_null = p_lm_rn < 0.05 and abs(d_lm_rn) > 0.02
print(f"  same_region beats random_null: {'YES' if sr_beats_null else 'NO'} (delta={d_sr_rn:+.4f}, p={p_sr_rn:.2e})")
print(f"  loose_match beats random_null: {'YES' if lm_beats_null else 'NO'} (delta={d_lm_rn:+.4f}, p={p_lm_rn:.2e})")

if sr_beats_null or lm_beats_null:
    print("\n  ==> H6 SURVIVES: at least one evidence-matched donor condition produces a")
    print("      context-swap effect on the FIXED local lesion voxels that is significantly")
    print("      LARGER than a random-donor (permutation-null) swap. Context is doing real,")
    print("      specific causal work beyond generic 'any different context perturbs the")
    print("      prediction' -- worth a deeper follow-up (though NOT yet an algorithm).")
else:
    print("\n  ==> H6 DIES: any context-swap effect (if present at all) is indistinguishable")
    print("      from a RANDOM, evidence-mismatched donor swap. The model's prediction on")
    print("      the fixed local lesion voxels is not sensitive to whether the surrounding")
    print("      context is genuinely anatomically/evidence-plausible -- consistent with")
    print("      local evidence (or its absence, per H1/E233) already dominating the")
    print("      decision, not a distinct contextual-competition mechanism.")

print("\n" + "="*90)
print("Same analysis on max_prob (secondary check)")
print("="*90)
agg_max = {c: [] for c in conditions}
for key in valid_lesions:
    d = by_lesion[key]
    for c in conditions:
        agg_max[c].append(np.mean([v for v in d[c]]))  # placeholder, recompute below properly

by_lesion_max = defaultdict(lambda: defaultdict(list))
for r in rows:
    key = (r['subject_id'], r['comp_id'])
    by_lesion_max[key][r['condition']].append(r['max_prob'])
agg_max = {c: [] for c in conditions}
for key in valid_lesions:
    d = by_lesion_max[key]
    for c in conditions:
        agg_max[c].append(np.mean(d[c]))
for c in conditions:
    agg_max[c] = np.array(agg_max[c])
d_sr_rn_m, p_sr_rn_m = paired(agg_max['same_region'], agg_max['random_null'], 'same_region - random_null (max_prob)')
d_lm_rn_m, p_lm_rn_m = paired(agg_max['loose_match'], agg_max['random_null'], 'loose_match - random_null (max_prob)')
