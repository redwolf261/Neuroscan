"""E230 analysis -- RADDO feasibility, per the user's exact spec.

M0: log(size) + isolation + distance
M1: M0 + D_i           (persistent magnitude, e0+e1)
M2: M0 + D_i + P_i + C_i  (+ velocity/persistence + consistency)

Control: A_i = D_i (deliberately identical formula -- RADDO's claimed
advantage must come from P_i/C_i's INCREMENTAL power over D_i alone, not
from D_i itself, which is already "the simplest alternative").

KILL CRITERIA (all from the user's own spec):
  1. D_i adds no significant information over size/isolation/distance (M0->M1).
  2. P_i/C_i add nothing over D_i alone (M1->M2).
  3. [trajectory test -- e_k -> next-stage recovery -- NOT run here, would
     need per-stage detection status, deferred; noted as a limitation]
  4. A simple residual (A_i, identical to D_i by construction here) performs
     as well as the full M2 model.

e_2 (final-stage error) is NEVER used as a predictor -- only to define
`detected`/`missed`, reused unchanged from E227_gradient.csv (the exact
circularity guard that E229 violated and this experiment was designed to
avoid from the start).
"""
import csv
import numpy as np
from scipy import stats

d230 = list(csv.DictReader(open('E230_raddo.csv')))
e227 = list(csv.DictReader(open('E227_gradient.csv')))
e223 = list(csv.DictReader(open('E223_exposure.csv')))

d230_by_key = {(r['subject_id'], r['comp_id']): r for r in d230}
e223_by_key = {(r['subject_id'], r['comp_id']): r for r in e223}

joined = []
for r227 in e227:
    key = (r227['subject_id'], r227['comp_id'])
    r230 = d230_by_key.get(key)
    r223 = e223_by_key.get(key)
    if r230 is None or r223 is None:
        continue
    joined.append({
        'size': float(r227['size']),
        'detected': int(r227['detected']),
        'isolated': int(r223['isolated_from_main_wt']),
        'dist': float(r223['dist_to_wt_centroid']),
        'e0': float(r230['e0_d4']),
        'e1': float(r230['e1_d2']),
    })

print(f"E227 rows: {len(e227)}  E230 rows: {len(d230)}  joined: {len(joined)}")

size = np.array([r['size'] for r in joined])
det = np.array([r['detected'] for r in joined])
missed = 1 - det
iso = np.array([r['isolated'] for r in joined])
dist = np.array([r['dist'] for r in joined])
e0 = np.array([r['e0'] for r in joined])
e1 = np.array([r['e1'] for r in joined])
log_size = np.log(size + 1)

D_i = np.abs(e0) + np.abs(e1)
P_i = np.abs(e1 - e0)
eps = 1e-6
C_i = 1 - np.abs(e1 - e0) / (e0 + e1 + eps)
A_i = D_i.copy()  # control, identical formula to D_i by design

print(f"\ne0 (D4 error): mean={e0.mean():.4f} std={e0.std():.4f}")
print(f"e1 (D2 error, UNSUPERVISED/exploratory): mean={e1.mean():.4f} std={e1.std():.4f}")
print(f"D_i: mean={D_i.mean():.4f}  P_i: mean={P_i.mean():.4f}  C_i: mean={C_i.mean():.4f}")

# raw group comparison
for name, arr in [('e0', e0), ('e1', e1), ('D_i', D_i), ('P_i', P_i), ('C_i', C_i)]:
    d, m = arr[det==1], arr[det==0]
    u, p = stats.mannwhitneyu(d, m)
    print(f"  {name}: detected median={np.median(d):.4f}  missed median={np.median(m):.4f}  p={p:.3e}")

def fit_r2(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1 - ss_res/ss_tot, ss_res

n = len(missed)
y = missed.astype(float)

X_M0 = np.column_stack([np.ones(n), log_size, iso, dist])
r2_M0, ss_M0 = fit_r2(X_M0, y)

X_M1 = np.column_stack([X_M0, D_i])
r2_M1, ss_M1 = fit_r2(X_M1, y)

X_M2 = np.column_stack([X_M1, P_i, C_i])
r2_M2, ss_M2 = fit_r2(X_M2, y)

X_control = np.column_stack([X_M0, A_i])  # identical to X_M1 here by construction
r2_control, ss_control = fit_r2(X_control, y)

print("\n" + "=" * 90)
print("NESTED MODEL COMPARISON")
print("=" * 90)
print(f"M0 (size+isolated+dist):        R^2={r2_M0:.4f}")
print(f"M1 (M0 + D_i):                  R^2={r2_M1:.4f}  (delta={r2_M1-r2_M0:+.4f})")
print(f"M2 (M1 + P_i + C_i):            R^2={r2_M2:.4f}  (delta={r2_M2-r2_M1:+.4f})")
print(f"Control (M0 + A_i, A_i==D_i):   R^2={r2_control:.4f}  (should equal M1 exactly)")

def f_test(ss_restricted, ss_full, p_restricted, p_full, n):
    F = ((ss_restricted - ss_full) / (p_full - p_restricted)) / (ss_full / (n - p_full))
    p_val = 1 - stats.f.cdf(F, p_full - p_restricted, n - p_full)
    return F, p_val

print("\n--- KILL CRITERION 1: does D_i add info over M0? ---")
F1, p1 = f_test(ss_M0, ss_M1, X_M0.shape[1], X_M1.shape[1], n)
print(f"F={F1:.4f}  p={p1:.4f}  {'SURVIVES' if p1 < 0.05 else 'KILLED'}")

print("\n--- KILL CRITERION 2: do P_i/C_i add info over M1 (D_i alone)? ---")
F2, p2 = f_test(ss_M1, ss_M2, X_M1.shape[1], X_M2.shape[1], n)
print(f"F={F2:.4f}  p={p2:.4f}  {'SURVIVES' if p2 < 0.05 else 'KILLED -- RADDO offers nothing beyond simple D_i'}")

print("\n--- Bootstrap CI on M2 vs M1 incremental R^2 ---")
rng = np.random.default_rng(0)
boots = []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    Xb_M1 = X_M1[idx]; Xb_M2 = X_M2[idx]; yb = y[idx]
    _, ssb_M1 = fit_r2(Xb_M1, yb)
    _, ssb_M2 = fit_r2(Xb_M2, yb)
    ss_tot_b = np.sum((yb - yb.mean())**2)
    r2b_M1 = 1 - ssb_M1/ss_tot_b
    r2b_M2 = 1 - ssb_M2/ss_tot_b
    boots.append(r2b_M2 - r2b_M1)
boots = np.array(boots)
print(f"Delta R^2 (M2-M1) bootstrap: mean={boots.mean():.4f}  "
      f"95% CI [{np.percentile(boots,2.5):.4f}, {np.percentile(boots,97.5):.4f}]")

print("\n" + "=" * 90)
print("VERDICT")
print("=" * 90)
crit1_pass = p1 < 0.05
crit2_pass = p2 < 0.05 and not (np.percentile(boots,2.5) < 0 < np.percentile(boots,97.5))
if not crit1_pass:
    print("  ==> KILL (criterion 1): D_i adds no significant information over")
    print("      size/isolation/distance. RADDO's premise fails at the first gate.")
elif not crit2_pass:
    print("  ==> KILL (criterion 2/4): D_i alone may carry signal, but P_i/C_i")
    print("      (the actual RADDO-specific mechanics -- persistence, velocity,")
    print("      consistency) add NOTHING beyond the simple control A_i=D_i.")
    print("      Per the user's own framing: 'If the sophisticated observer does")
    print("      NOT outperform A_i, there is no reason to build the observer.'")
    print("      RADDO is not justified; a much simpler mechanism (if anything)")
    print("      would be the honest conclusion.")
else:
    print("  ==> SURVIVES the primary feasibility gates. D_i is informative AND")
    print("      P_i/C_i add real incremental power over the simple control.")
    print("      NOTE: trajectory test (kill criterion 3) NOT run here -- would")
    print("      need per-stage recovery tracking, a genuinely bigger diagnostic.")
    print("      Recommend running that before any training commitment.")
