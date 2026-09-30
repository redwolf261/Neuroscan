"""E243 analysis -- stratified D2->D1 replication, per the user's exact
pre-registered decision gate: "E243 survives only if the D2->D1 effect
replicates within at least one predefined subgroup and remains
directionally coherent against matched controls. If it disappears in
both: kill the D1-transition hypothesis. If it survives strongly in one
subgroup: investigate that subgroup's specific transformation."

Repeats E242's EXACT transition-decomposition logic (Delta_S = S(D1) -
S(D2), paired missed-vs-detected), computed SEPARATELY within G2-A
(near-zero/rejected) and G2-B (partial-coverage) phenotypes, using
E240's own separability values (sep_D1, sep_D2) UNCHANGED -- reused, not
recomputed.
"""
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

phenotypes = {}
for r in csv.DictReader(open('E243_phenotypes.csv')):
    phenotypes[(r['subject_id'], int(r['comp_id']))] = r['phenotype']

e240 = list(csv.DictReader(open('E240_layerwise.csv')))
for r in e240:
    r['pair_id'] = int(r['pair_id'])
    r['comp_id'] = int(r['comp_id'])
    for s in ('D2', 'D1'):
        r[f'sep_{s}'] = float(r[f'sep_{s}']) if r[f'sep_{s}'] != '' else None

by_pair = defaultdict(dict)
for r in e240:
    by_pair[r['pair_id']][r['role']] = r

complete = [p for p, d in by_pair.items() if 'missed' in d and 'detected' in d]
print(f"Complete E240 pairs: {len(complete)}")

# attach phenotype to each pair via its MISSED lesion's key
pheno_counts = defaultdict(int)
pair_phenotype = {}
for p in complete:
    m = by_pair[p]['missed']
    key = (m['subject_id'], m['comp_id'])
    phen = phenotypes.get(key, 'not_phenotyped')
    pair_phenotype[p] = phen
    pheno_counts[phen] += 1

print("\nPhenotype distribution among E240's matched pairs:")
for k, v in sorted(pheno_counts.items()):
    print(f"  {k}: {v}")


def run_transition_test(pair_ids, label):
    delta_missed, delta_det = [], []
    for p in pair_ids:
        m, d = by_pair[p]['missed'], by_pair[p]['detected']
        if None in (m['sep_D2'], m['sep_D1'], d['sep_D2'], d['sep_D1']):
            continue
        delta_missed.append(m['sep_D1'] - m['sep_D2'])
        delta_det.append(d['sep_D1'] - d['sep_D2'])
    delta_missed = np.array(delta_missed)
    delta_det = np.array(delta_det)
    n = len(delta_missed)
    if n < 5:
        print(f"\n{label}: n={n} -- TOO FEW for a meaningful test")
        return None
    diff = delta_missed - delta_det
    pooled_std = np.sqrt((delta_missed.var(ddof=1) + delta_det.var(ddof=1)) / 2)
    cohen_d = diff.mean() / pooled_std if pooled_std > 0 else float('nan')
    if np.all(diff == 0):
        p_val = 1.0
    else:
        _, p_val = stats.wilcoxon(diff)
    print(f"\n{label}: n={n}")
    print(f"  Delta_S(D2->D1) missed_mean={delta_missed.mean():.4f}  detected_mean={delta_det.mean():.4f}")
    print(f"  diff={diff.mean():+.4f}  Cohen_d={cohen_d:+.4f}  Wilcoxon_p={p_val:.4e}")
    return {'n': n, 'diff': diff.mean(), 'd': cohen_d, 'p': p_val}


print("\n" + "="*100)
print("REPLICATION: E242's full-population result (reference)")
print("="*100)
full_result = run_transition_test(complete, "ALL missed (E242 replication)")

print("\n" + "="*100)
print("STRATIFIED TEST")
print("="*100)
g2a_pairs = [p for p in complete if pair_phenotype[p] == 'G2A_rejected']
g2b_pairs = [p for p in complete if pair_phenotype[p] == 'G2B_partial']
mid_pairs = [p for p in complete if pair_phenotype[p] == 'mid_excluded']

result_a = run_transition_test(g2a_pairs, "G2-A (near-zero / rejected)")
result_b = run_transition_test(g2b_pairs, "G2-B (partial-coverage)")
result_mid = run_transition_test(mid_pairs, "mid (excluded from pre-registered test, for reference only)")

print("\n" + "="*100)
print("PRE-REGISTERED DECISION GATE")
print("="*100)


def survives(res):
    return res is not None and res['p'] < 0.05 and res['diff'] < 0  # missed loses MORE (same direction as E242)


a_survives = survives(result_a)
b_survives = survives(result_b)
print(f"G2-A replicates (same direction, p<0.05): {'YES' if a_survives else 'NO'}"
      + (f" (p={result_a['p']:.4f}, d={result_a['d']:+.4f})" if result_a else " (insufficient n)"))
print(f"G2-B replicates (same direction, p<0.05): {'YES' if b_survives else 'NO'}"
      + (f" (p={result_b['p']:.4f}, d={result_b['d']:+.4f})" if result_b else " (insufficient n)"))

if not a_survives and not b_survives:
    print("\n  ==> NEITHER subgroup replicates the D2->D1 effect.")
    print("      Per the pre-registered gate: KILL the D1-transition hypothesis.")
    print("      E242's marginal aggregate signal (p=0.033, uncorrected) does not")
    print("      survive stratification -- likely noise or an artifact of pooling")
    print("      heterogeneous failure modes, not a real operation-level target.")
elif a_survives and b_survives:
    print("\n  ==> BOTH subgroups replicate. The D2->D1 effect is not phenotype-specific")
    print("      -- it applies broadly across both rejected and partial-coverage missed")
    print("      lesions. This somewhat strengthens confidence the effect is real (survives")
    print("      stratification) though it does not resolve WHICH phenotype dominates.")
else:
    winner = 'G2-A (near-zero/rejected)' if a_survives else 'G2-B (partial-coverage)'
    print(f"\n  ==> The D2->D1 effect survives ONLY in {winner}.")
    print("      Per the user's own framework: 'D1 is the problem' was too crude --")
    print(f"      the effect is specific to this ONE failure phenotype. This phenotype's")
    print("      own D2->D1 transformation (upconv1 -> attn_gate1 -> cat1 -> dec1) becomes")
    print("      the next legitimate target for operation-level decomposition -- but still")
    print("      NOT yet an architecture change, per the user's own explicit instruction.")
