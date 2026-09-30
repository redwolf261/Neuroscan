"""E233 analysis -- hypothesis 1 (information absent in input) test.

Prediction if hypothesis 1 is TRUE: missed lesions should show LOWER
raw-intensity local contrast/separability than detected lesions, size-
matched -- i.e. even a simple, direct measurement of "how different does
this lesion's own tissue look from its immediate surroundings" should be
weak specifically for lesions the model misses, because the discriminative
signal genuinely isn't there for a classifier (any classifier) to find.

Prediction if hypothesis 1 is FALSE (signal IS present, failure is
downstream -- encoder/decoder/objective/architecture): missed and detected
lesions should show SIMILAR raw-intensity contrast after controlling for
size -- the raw MRI signal discriminates the lesion from background just
as well in both populations, so whatever causes the model to miss some of
them is NOT explainable by the input itself lacking information.
"""
import csv
import numpy as np
from scipy import stats

d233 = list(csv.DictReader(open('E233_intensity.csv')))
e223 = list(csv.DictReader(open('E223_exposure.csv')))
e223_by_key = {(r['subject_id'], r['comp_id']): r for r in e223}

MODALITIES = ('t1c', 't1n', 't2f', 't2w')

joined = []
for r233 in d233:
    key = (r233['subject_id'], r233['comp_id'])
    r223 = e223_by_key.get(key)
    if r223 is None:
        continue
    row = {'size': float(r233['size']), 'detected': int(r223['detected']),
          'isolated': int(r223['isolated_from_main_wt']),
          'dist': float(r223['dist_to_wt_centroid'])}
    for m in MODALITIES:
        row[f'contrast_{m}'] = float(r233[f'contrast_{m}'])
    joined.append(row)

print(f"E233 rows: {len(d233)}  E223 rows: {len(e223)}  joined: {len(joined)}")

size = np.array([r['size'] for r in joined])
det = np.array([r['detected'] for r in joined])
missed = 1 - det
iso = np.array([r['isolated'] for r in joined])
dist = np.array([r['dist'] for r in joined])
log_size = np.log(size + 1)

print(f"\ndetected={det.sum()} ({100*det.mean():.1f}%)  missed={len(joined)-det.sum()}\n")

print("=" * 100)
print("RAW: contrast per modality, detected vs missed")
print("=" * 100)
contrasts = {}
for m in MODALITIES:
    arr = np.array([r[f'contrast_{m}'] for r in joined])
    contrasts[m] = arr
    d, mi = arr[det == 1], arr[det == 0]
    u, p = stats.mannwhitneyu(d, mi)
    direction = "LOWER in missed (supports H1)" if np.median(mi) < np.median(d) else "HIGHER/EQUAL in missed (against H1)"
    print(f"  {m}: detected median={np.median(d):.4f}  missed median={np.median(mi):.4f}  "
          f"p={p:.3e}  -> {direction}")

print("\n" + "=" * 100)
print("SIZE CONFOUND CHECK: does contrast correlate with size?")
print("=" * 100)
for m in MODALITIES:
    r, p = stats.spearmanr(size, contrasts[m])
    print(f"  Spearman(size, contrast_{m}) = {r:+.4f}  p={p:.3e}")

print("\n" + "=" * 100)
print("PARTIAL CORRELATION: missed vs contrast, controlling for size/isolation/distance")
print("=" * 100)
X = np.column_stack([np.ones(len(joined)), log_size, iso, dist])
beta_m, *_ = np.linalg.lstsq(X, missed.astype(float), rcond=None)
resid_m = missed.astype(float) - X @ beta_m

for m in MODALITIES:
    beta_c, *_ = np.linalg.lstsq(X, contrasts[m], rcond=None)
    resid_c = contrasts[m] - X @ beta_c
    r, p = stats.pearsonr(resid_m, resid_c)
    flag = ''
    if p < 0.05:
        flag = ' <-- SIGNIFICANT, supports H1' if r < 0 else ' <-- SIGNIFICANT, but WRONG SIGN for H1'
    print(f"  partial corr(missed, contrast_{m} | log_size, isolated, dist) = {r:+.4f}  p={p:.3e}{flag}")

# also: absolute value of contrast (does the SIGN matter, or just magnitude
# of separability, since a lesion could be hypo- or hyper-intense and
# still be separable either direction)
print("\n" + "=" * 100)
print("ABSOLUTE CONTRAST (magnitude of separability, sign-agnostic)")
print("=" * 100)
for m in MODALITIES:
    abs_c = np.abs(contrasts[m])
    d, mi = abs_c[det == 1], abs_c[det == 0]
    u, p = stats.mannwhitneyu(d, mi)
    beta_c, *_ = np.linalg.lstsq(X, abs_c, rcond=None)
    resid_c = abs_c - X @ beta_c
    r, p_partial = stats.pearsonr(resid_m, resid_c)
    print(f"  |{m}|: detected median={np.median(d):.4f}  missed median={np.median(mi):.4f}  "
          f"raw p={p:.3e}  partial r={r:+.4f} p={p_partial:.3e}")

print("\n" + "=" * 100)
print("SIZE-MATCHED QUINTILE BREAKDOWN (t1c, the primary ET-discriminating modality)")
print("=" * 100)
abs_t1c = np.abs(contrasts['t1c'])
bins = np.percentile(size, [0, 20, 40, 60, 80, 100])
for i in range(5):
    lo, hi = bins[i], bins[i+1]
    sel = (size >= lo) & (size <= hi)
    d_c, m_c = abs_t1c[sel & (det==1)], abs_t1c[sel & (det==0)]
    if len(d_c) < 3 or len(m_c) < 3:
        print(f"  size [{lo:.0f},{hi:.0f}]: too few")
        continue
    p = stats.mannwhitneyu(d_c, m_c)[1]
    print(f"  size [{lo:.0f},{hi:.0f}] (n={sel.sum()}): detected |contrast_t1c|={np.median(d_c):.4f}  "
          f"missed |contrast_t1c|={np.median(m_c):.4f}  p={p:.3f}")

print("\n" + "=" * 100)
print("VERDICT")
print("=" * 100)
sig_count = 0
for m in MODALITIES:
    beta_c, *_ = np.linalg.lstsq(X, np.abs(contrasts[m]), rcond=None)
    resid_c = np.abs(contrasts[m]) - X @ beta_c
    r, p = stats.pearsonr(resid_m, resid_c)
    # FIXED sign convention (caught before trusting the first printed
    # verdict, see E233 memory writeup): resid_m>0 means MORE likely
    # missed; resid_c>0 means MORE separable. If missed lesions have
    # LOWER separability (supports H1), missed and |contrast| are
    # NEGATIVELY correlated -- r<0, not r>0. The original r>0 filter here
    # was backwards and printed the WRONG verdict on the first run.
    if p < 0.05 and r < 0:
        sig_count += 1
print(f"Modalities where |contrast| is significantly associated with missedness "
      f"(size/isolation/dist-controlled): {sig_count}/4")
if sig_count >= 2:
    print("\n  ==> Some support for Hypothesis 1: raw intensity separability IS weaker")
    print("      for missed lesions in multiple modalities, even controlling for size.")
    print("      Input information may be genuinely limited for (some of) these lesions.")
    print("      Does NOT rule out other hypotheses for the REMAINING population --")
    print("      recommend stratifying missed lesions by this signal before further work.")
else:
    print("\n  ==> Hypothesis 1 NOT well-supported. Missed lesions are NOT systematically")
    print("      less separable in raw intensity space than detected ones (size-matched).")
    print("      The discriminative signal appears to BE present in the input for most")
    print("      missed lesions -- the failure is downstream (encoder/decoder/objective/")
    print("      architecture), not an intrinsic input-information limitation.")
    print("      Proceed to test hypothesis 2 (encoder discards signal) next.")
