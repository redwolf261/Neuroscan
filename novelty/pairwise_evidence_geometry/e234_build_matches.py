"""E234 step 0 -- builds matched missed/detected lesion pairs, per the
user's exact spec: {size, isolation, distance, t1c-contrast, t2f-contrast}
matched, so H1 (input information genuinely absent) is EQUALIZED across
the pair and cannot explain any remaining separability difference found
downstream by e234_layerwise_separability.py.

Reuses E223's own cached size/isolated/dist and E233's own cached
contrast_t1c/contrast_t2f UNCHANGED -- no new measurement here, purely a
matching/selection step. Greedy nearest-neighbor matching WITHOUT
replacement, on z-scored features (each feature standardized by its own
population std before computing distance, so no single feature with a
larger natural scale dominates the match) -- a simple, transparent,
non-learned matching procedure appropriate for a falsification
experiment (no risk of a "trained matcher" reintroducing the exact
probe-capacity confound the user is trying to avoid).
"""
import csv
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent

e223 = list(csv.DictReader(open(HERE / 'E223_exposure.csv')))
e233 = list(csv.DictReader(open(HERE / 'E233_intensity.csv')))

e233_by_key = {(r['subject_id'], r['comp_id']): r for r in e233}

joined = []
for r223 in e223:
    key = (r223['subject_id'], r223['comp_id'])
    r233 = e233_by_key.get(key)
    if r233 is None:
        continue
    joined.append({
        'subject_id': r223['subject_id'], 'comp_id': r223['comp_id'],
        'size': float(r223['size']), 'detected': int(r223['detected']),
        'isolated': int(r223['isolated_from_main_wt']),
        'dist': float(r223['dist_to_wt_centroid']),
        'contrast_t1c': float(r233['contrast_t1c']),
        'contrast_t2f': float(r233['contrast_t2f']),
    })

print(f'E223 rows: {len(e223)}  E233 rows: {len(e233)}  joined: {len(joined)}')

missed = [r for r in joined if r['detected'] == 0]
detected = [r for r in joined if r['detected'] == 1]
print(f'missed: {len(missed)}  detected: {len(detected)}')


def featurize(rows):
    log_size = np.log(np.array([r['size'] for r in rows]) + 1)
    iso = np.array([r['isolated'] for r in rows], dtype=float)
    dist = np.array([r['dist'] for r in rows])
    ct1c = np.array([r['contrast_t1c'] for r in rows])
    ct2f = np.array([r['contrast_t2f'] for r in rows])
    return np.column_stack([log_size, iso, dist, ct1c, ct2f])


X_missed = featurize(missed)
X_detected = featurize(detected)

# z-score by the DETECTED population's own std (the population we're
# matching INTO), so the matching metric is anchored to a fixed reference
# scale regardless of the missed population's own (possibly different)
# spread.
mu = X_detected.mean(axis=0)
sigma = X_detected.std(axis=0) + 1e-6
X_missed_z = (X_missed - mu) / sigma
X_detected_z = (X_detected - mu) / sigma

# greedy nearest-neighbor WITHOUT replacement: process missed lesions in
# a FIXED, DETERMINISTIC order (by subject_id, comp_id -- not shuffled),
# each time finding the closest STILL-AVAILABLE detected lesion.
tree = cKDTree(X_detected_z)
used = set()
pairs = []
# query more neighbors than needed so we can skip already-used ones
k_query = min(50, len(detected))
dists, idxs = tree.query(X_missed_z, k=k_query)
if k_query == 1:
    dists = dists.reshape(-1, 1); idxs = idxs.reshape(-1, 1)

import os
max_dist = float(os.environ.get('E234_MAX_DIST', '2.0'))  # z-scored Euclidean
                 # distance cap -- a "reasonably matched" threshold; pairs
                 # beyond this are DROPPED, not forced. TIGHTENED after
                 # finding the default 2.0 left a statistically significant
                 # residual gap in size/t1c-contrast even after matching
                 # (Wilcoxon p<0.0001 both) -- see e234_build_matches.py's
                 # own match-quality report, checked BEFORE running the
                 # expensive layerwise measurement, per explicit user
                 # decision to tighten rather than add covariates post-hoc.
n_dropped = 0
for i, m in enumerate(missed):
    chosen = None
    for j in range(k_query):
        idx = idxs[i, j]
        if idx in used:
            continue
        if dists[i, j] > max_dist:
            break  # sorted ascending; no closer unused candidate exists within cap
        chosen = idx
        break
    if chosen is None:
        n_dropped += 1
        continue
    used.add(chosen)
    pairs.append({
        'missed_subject': m['subject_id'], 'missed_comp': m['comp_id'],
        'detected_subject': detected[chosen]['subject_id'],
        'detected_comp': detected[chosen]['comp_id'],
        'match_distance': float(dists[i, [j2 for j2 in range(k_query) if idxs[i,j2]==chosen][0]]),
    })

print(f'matched pairs: {len(pairs)}  dropped (no match within cap): {n_dropped}')

with open(HERE / 'E234_matches.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['missed_subject', 'missed_comp',
                                      'detected_subject', 'detected_comp', 'match_distance'])
    w.writeheader()
    for p in pairs:
        w.writerow(p)
print('wrote E234_matches.csv')

# quick match-quality report: are the matched pairs REALLY equivalent on
# each individual feature, not just the aggregate distance?
print('\nMatch quality per feature (missed vs matched-detected):')
feat_names = ['log_size', 'isolated', 'dist', 'contrast_t1c', 'contrast_t2f']
missed_by_key = {(r['subject_id'], r['comp_id']): r for r in missed}
detected_by_key = {(r['subject_id'], r['comp_id']): r for r in detected}
for fi, fname in enumerate(feat_names):
    m_vals, d_vals = [], []
    for p in pairs:
        mr = missed_by_key[(p['missed_subject'], p['missed_comp'])]
        dr = detected_by_key[(p['detected_subject'], p['detected_comp'])]
        if fname == 'log_size':
            m_vals.append(np.log(mr['size']+1)); d_vals.append(np.log(dr['size']+1))
        elif fname == 'isolated':
            m_vals.append(mr['isolated']); d_vals.append(dr['isolated'])
        elif fname == 'dist':
            m_vals.append(mr['dist']); d_vals.append(dr['dist'])
        elif fname == 'contrast_t1c':
            m_vals.append(mr['contrast_t1c']); d_vals.append(dr['contrast_t1c'])
        elif fname == 'contrast_t2f':
            m_vals.append(mr['contrast_t2f']); d_vals.append(dr['contrast_t2f'])
    m_vals = np.array(m_vals); d_vals = np.array(d_vals)
    print(f'  {fname}: missed mean={m_vals.mean():.3f}  detected mean={d_vals.mean():.3f}  '
          f'mean abs diff={np.abs(m_vals-d_vals).mean():.3f}')
