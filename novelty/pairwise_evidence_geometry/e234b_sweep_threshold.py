"""Sweep E234's match_dist threshold, reporting pair count AND Wilcoxon
p-values on size/t1c-contrast for the matched population at each
threshold (not just the distance cap) -- same verification discipline as
E234's own original sweep, now aimed at finding the loosest threshold
that still holds statistical equalization, needed to reach ~275 pairs
for H2's properly-powered rerun."""
import csv
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from pathlib import Path

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
missed = [r for r in joined if r['detected'] == 0]
detected = [r for r in joined if r['detected'] == 1]

def featurize(rows):
    log_size = np.log(np.array([r['size'] for r in rows]) + 1)
    iso = np.array([r['isolated'] for r in rows], dtype=float)
    dist = np.array([r['dist'] for r in rows])
    ct1c = np.array([r['contrast_t1c'] for r in rows])
    ct2f = np.array([r['contrast_t2f'] for r in rows])
    return np.column_stack([log_size, iso, dist, ct1c, ct2f])

X_missed = featurize(missed)
X_detected = featurize(detected)
mu = X_detected.mean(axis=0)
sigma = X_detected.std(axis=0) + 1e-6
X_missed_z = (X_missed - mu) / sigma
X_detected_z = (X_detected - mu) / sigma

tree = cKDTree(X_detected_z)
k_query = min(50, len(detected))
dists, idxs = tree.query(X_missed_z, k=k_query)

def match_at(max_dist):
    used = set()
    pairs = []
    for i, m in enumerate(missed):
        chosen = None
        for j in range(k_query):
            idx = idxs[i, j]
            if idx in used:
                continue
            if dists[i, j] > max_dist:
                break
            chosen = idx
            break
        if chosen is None:
            continue
        used.add(chosen)
        pairs.append((m, detected[chosen]))
    return pairs

for md in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1.0]:
    pairs = match_at(md)
    if len(pairs) < 5:
        print(f"max_dist={md}: n={len(pairs)} (too few for stats)")
        continue
    sz_m = np.array([np.log(p[0]['size']+1) for p in pairs])
    sz_d = np.array([np.log(p[1]['size']+1) for p in pairs])
    ct_m = np.array([p[0]['contrast_t1c'] for p in pairs])
    ct_d = np.array([p[1]['contrast_t1c'] for p in pairs])
    _, p_sz = stats.wilcoxon(sz_m - sz_d) if not np.all(sz_m==sz_d) else (0,1)
    _, p_ct = stats.wilcoxon(ct_m - ct_d) if not np.all(ct_m==ct_d) else (0,1)
    print(f"max_dist={md:.2f}: n={len(pairs):4d}  size_p={p_sz:.4f}  t1c_p={p_ct:.4f}  "
          f"{'OK' if p_sz>0.05 and p_ct>0.05 else 'FAILS equalization'}")
