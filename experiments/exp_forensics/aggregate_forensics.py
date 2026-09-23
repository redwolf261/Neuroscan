"""Aggregate the forensic decomposition into the failure-mechanism attribution table."""
import csv, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'FORENSIC_per_subject.csv')))
n = len(rows)
REGIONS = ['ET', 'TC', 'WT']


def col(k, default=np.nan):
    out = []
    for r in rows:
        v = r.get(k, '')
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(default)
    return np.array(out)


print(f"n = {n} subjects\n")
print("=" * 104)
print("PER-REGION BASELINE AND TOTAL DICE SHORTFALL")
print("=" * 104)
base = {}
for R in REGIONS:
    d = col(f'{R}_dice')
    base[R] = d
    print(f"  {R}: mean Dice {d.mean():.6f}   shortfall from 1.0 = {100*(1-d.mean()):.3f}pp")
mean3 = np.mean([base[R].mean() for R in REGIONS])
print(f"\n  3-region mean Dice = {mean3:.7f}   (authoritative 0.8596971)")
total_short = 100 * (1 - mean3)
print(f"  TOTAL SHORTFALL to be explained = {total_short:.3f}pp (3-region)")

# ---------------- mechanism attribution ----------------
# Each mechanism's counterfactual recovery, averaged over subjects, per region.
MECH = [
    ('Boundary error (FP+FN within 2vox of GT surface)', 'rec_fix_boundary'),
    ('  - boundary FP only',                             'rec_fix_fp_boundary'),
    ('  - boundary FN only',                             'rec_fix_fn_boundary'),
    ('Interior error (FP+FN beyond 2vox)',                'rec_fix_interior'),
    ('  - interior FP only',                             'rec_fix_fp_interior'),
    ('  - interior FN only',                             'rec_fix_fn_interior'),
    ('All false positives',                               'rec_fix_all_fp'),
    ('All false negatives',                               'rec_fix_all_fn'),
    ('Missed components (zero-overlap GT comps)',         'rec_fix_missed_comp'),
    ('Spurious components (zero-overlap pred comps)',     'rec_fix_spurious_comp'),
]

print()
print("=" * 104)
print("MECHANISM ATTRIBUTION -- counterfactual Dice recovery if that mechanism alone were repaired")
print("(NOT mutually exclusive: boundary/interior partition FP+FN; 'all FP'/'all FN' overlap both)")
print("=" * 104)
hdr = f"{'Mechanism':<50}" + "".join(f"{R+' (pp)':>13}" for R in REGIONS) + f"{'3-reg (pp)':>13}" + f"{'%of short':>11}"
print(hdr)
print("-" * 104)
table = {}
for label, key in MECH:
    vals = {}
    for R in REGIONS:
        v = col(f'{R}_{key}')
        vals[R] = 100 * np.nanmean(v)
    three = np.mean([vals[R] for R in REGIONS])
    pct = 100 * three / total_short if total_short > 0 else 0
    table[label.strip()] = {'per_region_pp': vals, 'three_region_pp': three, 'pct_of_shortfall': pct}
    print(f"{label:<50}" + "".join(f"{vals[R]:>13.3f}" for R in REGIONS)
          + f"{three:>13.3f}" + f"{pct:>10.1f}%")

# nesting
print("-" * 104)
nv = col('viol_total')
ng = np.mean([100*np.nanmean(col('nest_rec_TC_grow')), 100*np.nanmean(col('nest_rec_WT_grow'))])
ns = np.mean([100*np.nanmean(col('nest_rec_ET_shrink')), 100*np.nanmean(col('nest_rec_TC_shrink'))])
print(f"{'ET/TC/WT nesting violations (grow-to-fix)':<50}{'':>13}{'':>13}{'':>13}{ng:>13.3f}{100*ng/total_short:>10.1f}%")
print(f"{'ET/TC/WT nesting violations (shrink-to-fix)':<50}{'':>13}{'':>13}{'':>13}{ns:>13.3f}{100*ns/total_short:>10.1f}%")
print(f"    subjects with >=1 violation: {int((nv>0).sum())}/{n}   mean violating voxels: {np.nanmean(nv):.0f}")

# ---------------- subjects affected ----------------
print()
print("=" * 104)
print("SUBJECTS AFFECTED (count with non-trivial contribution, >0.5pp recovery in that region)")
print("=" * 104)
print(f"{'Mechanism':<50}" + "".join(f"{R:>13}" for R in REGIONS))
print("-" * 104)
for label, key in MECH:
    cnts = []
    for R in REGIONS:
        v = col(f'{R}_{key}')
        cnts.append(int(np.nansum(v > 0.005)))
    print(f"{label:<50}" + "".join(f"{c:>13}" for c in cnts))

# ---------------- error-volume split ----------------
print()
print("=" * 104)
print("RAW ERROR VOLUME SPLIT (summed over cohort)")
print("=" * 104)
for R in REGIONS:
    fpb, fpi = col(f'{R}_fp_boundary_vox').sum(), col(f'{R}_fp_interior_vox').sum()
    fnb, fni = col(f'{R}_fn_boundary_vox').sum(), col(f'{R}_fn_interior_vox').sum()
    tot = fpb + fpi + fnb + fni
    print(f"  {R}: total err {tot:,.0f} vox | FP-bnd {100*fpb/tot:5.1f}%  FP-int {100*fpi/tot:5.1f}%"
          f"  FN-bnd {100*fnb/tot:5.1f}%  FN-int {100*fni/tot:5.1f}%"
          f"   => BOUNDARY {100*(fpb+fnb)/tot:.1f}%")

# ---------------- component / displacement stats ----------------
print()
print("=" * 104)
print("COMPONENT AND DISPLACEMENT STATISTICS")
print("=" * 104)
for R in REGIONS:
    print(f"  {R}: GT comps {col(f'{R}_n_gt_comp').mean():6.1f} | pred comps {col(f'{R}_n_pred_comp').mean():6.1f}"
          f" | missed {col(f'{R}_n_missed_comp').mean():5.1f} | spurious {col(f'{R}_n_spurious_comp').mean():5.1f}"
          f" | fragmented {col(f'{R}_n_fragmented_gt').mean():4.1f} | merged {col(f'{R}_n_merged_pred').mean():4.1f}"
          f" | centroid disp {np.nanmean(col(f'{R}_global_centroid_disp')):5.2f} vox")

# ---------------- worst subjects ----------------
print()
print("=" * 104)
print("WORST SUBJECTS BY 3-REGION DICE (top 15) -- and their dominant mechanism")
print("=" * 104)
m3 = np.mean([base[R] for R in REGIONS], axis=0)
order = np.argsort(m3)
print(f"{'subject':<24}{'3reg':>8}{'ET':>8}{'TC':>8}{'WT':>8}   dominant mechanism (by max recovery)")
print("-" * 104)
for i in order[:15]:
    recs = {}
    for label, key in MECH:
        if label.startswith('  '):
            continue
        recs[label] = np.mean([col(f'{R}_{key}')[i] for R in REGIONS])
    dom = max(recs, key=recs.get)
    print(f"{rows[i]['subject_id']:<24}{m3[i]:>8.4f}{base['ET'][i]:>8.4f}{base['TC'][i]:>8.4f}"
          f"{base['WT'][i]:>8.4f}   {dom} ({100*recs[dom]:.2f}pp)")

# concentration: how much of total shortfall lives in the worst subjects?
print()
short_per_subj = (1 - m3)
tot_short_mass = short_per_subj.sum()
for k in [5, 10, 15, 25]:
    frac = short_per_subj[order[:k]].sum() / tot_short_mass
    print(f"  worst {k:>3} subjects hold {100*frac:5.1f}% of the total 3-region Dice shortfall")

json.dump({'n': n, 'mean3': float(mean3), 'total_shortfall_pp': float(total_short),
           'mechanisms': table}, open(HERE / 'FORENSIC_summary.json', 'w'), indent=2)
print("\nwrote FORENSIC_summary.json")
