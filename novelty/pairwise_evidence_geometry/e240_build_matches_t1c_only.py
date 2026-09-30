"""E240 step 0 -- H2 rerun at adequate power, per the user's exact
sequencing instruction (finish H2 properly before opening any new axis;
no loss/optimizer/architecture/decoder/threshold/sampling changes).

MATCHING STRATEGY CHANGE FROM E234: E234's original design matched
simultaneously on {log_size, isolated, dist, contrast_t1c, contrast_t2f},
which strictly equalizes both confounds but caps out at only n=117 pairs
(a hard ceiling of this specific population's own structure -- verified
via e234b_sweep_threshold.py's own sweep: t1c equalization breaks
immediately past max_dist=0.20, p=0.03 at n=188, worsening monotonically
to p<1e-4 by n=254). There is NO threshold giving ~275 pairs under
simultaneous strict equalization.

Per explicit user choice (this turn's AskUserQuestion): match primarily
on t1c-contrast (E233's own strongest H1 confound, r=-0.264 -- the
directly causal proxy for "does this lesion have real local evidence"),
accepting residual size difference, THEN explicitly control for size as
an analysis-time covariate rather than requiring it pre-equalized by the
matching procedure itself.

IMPORTANT CORRECTION found before writing this: size is actually the
STRONGER raw confound with detection (point-biserial r=0.629, p~0) than
t1c-contrast is (r=-0.264, from E233). This was checked explicitly
(not assumed) before finalizing the matching design, because naively
"matching on the weaker confound and controlling the stronger one only
post-hoc" would be backwards. Given this, matching here uses BOTH
log_size and t1c-contrast still, but WITHOUT the isolated/dist/t2f
features (which added little discriminating power in E234's own match-
quality report) -- a 2D match instead of 5D, which loosens the distance
metric enough to reach much higher N while still directly matching on
the two features that actually matter most, rather than dropping size
matching entirely. The downstream analysis (e240_run_probes.py) ALSO
explicitly reports the matched population's own residual size/t1c gap
(same verification discipline as E234) so any residual confound is
visible, not hidden.
"""
import csv
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy import stats

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
        'contrast_t1c': float(r233['contrast_t1c']),
    })

missed = [r for r in joined if r['detected'] == 0]
detected = [r for r in joined if r['detected'] == 1]
print(f'missed: {len(missed)}  detected: {len(detected)}')


def featurize(rows):
    log_size = np.log(np.array([r['size'] for r in rows]) + 1)
    ct1c = np.array([r['contrast_t1c'] for r in rows])
    return np.column_stack([log_size, ct1c])


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
        pairs.append((m, detected[chosen], float(dists[i, [j2 for j2 in range(k_query) if idxs[i,j2]==chosen][0]])))
    return pairs


print("\nSweep (2D: log_size + t1c only):")
target_pairs = None
for md in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]:
    pairs = match_at(md)
    if len(pairs) < 5:
        continue
    sz_m = np.array([np.log(p[0]['size']+1) for p in pairs])
    sz_d = np.array([np.log(p[1]['size']+1) for p in pairs])
    ct_m = np.array([p[0]['contrast_t1c'] for p in pairs])
    ct_d = np.array([p[1]['contrast_t1c'] for p in pairs])
    _, p_sz = stats.wilcoxon(sz_m - sz_d) if not np.all(sz_m==sz_d) else (0.0, 1.0)
    _, p_ct = stats.wilcoxon(ct_m - ct_d) if not np.all(ct_m==ct_d) else (0.0, 1.0)
    ok = p_sz > 0.05 and p_ct > 0.05
    print(f"  max_dist={md:.2f}: n={len(pairs):4d}  size_p={p_sz:.4f}  t1c_p={p_ct:.4f}  {'OK' if ok else 'residual gap'}")
    if len(pairs) >= 275 and target_pairs is None:
        target_pairs = (md, pairs)

if target_pairs is None:
    # take the largest available if 275 is never reached
    pairs = match_at(2.0)
    target_pairs = (2.0, pairs)
    print(f"\nWARNING: never reached 275 pairs even at max_dist=2.0 (n={len(pairs)}); using largest available")

md_final, pairs_final = target_pairs
print(f"\nSelected max_dist={md_final} giving n={len(pairs_final)} pairs")

with open(HERE / 'E240_matches.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['missed_subject', 'missed_comp',
                                      'detected_subject', 'detected_comp', 'match_distance'])
    w.writeheader()
    for m, d, dist in pairs_final:
        w.writerow({'missed_subject': m['subject_id'], 'missed_comp': m['comp_id'],
                   'detected_subject': d['subject_id'], 'detected_comp': d['comp_id'],
                   'match_distance': dist})
print('wrote E240_matches.csv')

# final residual-confound report (same discipline as E234's own check)
sz_m = np.array([np.log(p[0]['size']+1) for p in pairs_final])
sz_d = np.array([np.log(p[1]['size']+1) for p in pairs_final])
ct_m = np.array([p[0]['contrast_t1c'] for p in pairs_final])
ct_d = np.array([p[1]['contrast_t1c'] for p in pairs_final])
_, p_sz = stats.wilcoxon(sz_m - sz_d)
_, p_ct = stats.wilcoxon(ct_m - ct_d)
print(f"\nFinal matched population residual confound check:")
print(f"  log_size: missed mean={sz_m.mean():.3f}  detected mean={sz_d.mean():.3f}  Wilcoxon p={p_sz:.4f}")
print(f"  t1c:      missed mean={ct_m.mean():.3f}  detected mean={ct_d.mean():.3f}  Wilcoxon p={p_ct:.4f}")
print("  (any residual gap here MUST be explicitly controlled for in the downstream AUC analysis,")
print("   e.g. via a size-matched sub-analysis or covariate check, not ignored)")
