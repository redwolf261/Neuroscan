"""E231 analysis -- RADDO kill criterion 3, per the user's exact spec.

Fit: Delta q ~ log(size) + isolated + dist + e0_signed
Control: Delta q ~ log(size) + isolated + dist + |e0_signed|

If e0_signed (the SIGNED disturbance) explains Delta q beyond what the
UNSIGNED magnitude alone explains, that's evidence of a genuine directional/
mechanistic relationship, not just "big errors precede big changes" (which
||e0|| would already capture via regression-to-the-mean-style effects).

DIRECTION TEST: sign(e0_signed) -> sign(Delta q). If the D4-stage belief
being too LOW (e0<0) predicts the final stage moving UP (Delta q>0, i.e.
"recovering"), that's the mechanistic signature RADDO needs -- not just
"disagreement predicts failure" but "the specific direction of disagreement
predicts the specific direction of subsequent change."
"""
import csv
import numpy as np
from scipy import stats

d231 = list(csv.DictReader(open('E231_trajectory.csv')))
e227 = list(csv.DictReader(open('E227_gradient.csv')))
e223 = list(csv.DictReader(open('E223_exposure.csv')))

d231_by_key = {(r['subject_id'], r['comp_id']): r for r in d231}
e223_by_key = {(r['subject_id'], r['comp_id']): r for r in e223}

joined = []
for r227 in e227:
    key = (r227['subject_id'], r227['comp_id'])
    r231 = d231_by_key.get(key)
    r223 = e223_by_key.get(key)
    if r231 is None or r223 is None:
        continue
    joined.append({
        'size': float(r227['size']),
        'detected': int(r227['detected']),
        'isolated': int(r223['isolated_from_main_wt']),
        'dist': float(r223['dist_to_wt_centroid']),
        'q0': float(r231['q0']),
        'e0': float(r231['e0_signed']),
        'q1': float(r231['q1']),
    })

print(f"E227 rows: {len(e227)}  E231 rows: {len(d231)}  joined: {len(joined)}")

size = np.array([r['size'] for r in joined])
det = np.array([r['detected'] for r in joined])
iso = np.array([r['isolated'] for r in joined])
dist = np.array([r['dist'] for r in joined])
q0 = np.array([r['q0'] for r in joined])
e0 = np.array([r['e0'] for r in joined])
q1 = np.array([r['q1'] for r in joined])
log_size = np.log(size + 1)

delta_q = q1 - q0

print(f"\ndelta_q: mean={delta_q.mean():.4f}  std={delta_q.std():.4f}  "
      f"min={delta_q.min():.4f}  max={delta_q.max():.4f}")
print(f"e0 (signed): mean={e0.mean():.4f}  std={e0.std():.4f}  "
      f"fraction negative={np.mean(e0<0):.3f}")

def fit_r2(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1 - ss_res/ss_tot, ss_res

n = len(delta_q)
y = delta_q

X_base = np.column_stack([np.ones(n), log_size, iso, dist])
r2_base, ss_base = fit_r2(X_base, y)

X_abs = np.column_stack([X_base, np.abs(e0)])  # control: unsigned magnitude
r2_abs, ss_abs = fit_r2(X_abs, y)

X_signed = np.column_stack([X_base, e0])  # test: signed disturbance
r2_signed, ss_signed = fit_r2(X_signed, y)

X_both = np.column_stack([X_base, np.abs(e0), e0])  # both, to see signed's OWN incremental power
r2_both, ss_both = fit_r2(X_both, y)

print("\n" + "=" * 90)
print("MAGNITUDE TEST: does |e0| (control) or e0 (signed) explain Delta q?")
print("=" * 90)
print(f"Base (size+isolated+dist):        R^2={r2_base:.4f}")
print(f"+ |e0| (unsigned control):        R^2={r2_abs:.4f}  (delta={r2_abs-r2_base:+.4f})")
print(f"+ e0 (signed):                    R^2={r2_signed:.4f}  (delta={r2_signed-r2_base:+.4f})")
print(f"+ both |e0| and e0:                R^2={r2_both:.4f}  (delta over |e0| alone={r2_both-r2_abs:+.4f})")

def f_test(ss_r, ss_f, p_r, p_f, n):
    F = ((ss_r - ss_f) / (p_f - p_r)) / (ss_f / (n - p_f))
    return F, 1 - stats.f.cdf(F, p_f - p_r, n - p_f)

print("\n--- Does SIGNED e0 add anything beyond |e0| (the key mechanistic test)? ---")
F, p = f_test(ss_abs, ss_both, X_abs.shape[1], X_both.shape[1], n)
print(f"F={F:.4f}  p={p:.4f}  {'SURVIVES -- direction matters, not just magnitude' if p<0.05 else 'KILLED -- only magnitude matters, sign is uninformative'}")

rng = np.random.default_rng(0)
boots = []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    Xb_abs = X_abs[idx]; Xb_both = X_both[idx]; yb = y[idx]
    _, ssb_abs = fit_r2(Xb_abs, yb)
    _, ssb_both = fit_r2(Xb_both, yb)
    ss_tot_b = np.sum((yb - yb.mean())**2)
    boots.append((1 - ssb_both/ss_tot_b) - (1 - ssb_abs/ss_tot_b))
boots = np.array(boots)
print(f"Bootstrap delta R^2 (signed beyond |e0|): mean={boots.mean():.4f}  "
      f"95% CI [{np.percentile(boots,2.5):.4f}, {np.percentile(boots,97.5):.4f}]")

print("\n" + "=" * 90)
print("DIRECTION TEST: sign(e0) -> sign(Delta q)")
print("=" * 90)
sign_e0 = np.sign(e0)
sign_dq = np.sign(delta_q)
valid_signs = (sign_e0 != 0) & (sign_dq != 0)
agree = (sign_e0[valid_signs] == sign_dq[valid_signs]).mean()
print(f"P(sign(e0) == sign(Delta q)) = {agree:.3f}  (n={valid_signs.sum()}, chance=0.5)")
# note: same-sign here would mean e0>0 (D4 OVER-believes) -> delta_q>0 (final
# INCREASES further) -- i.e. reinforcement, not correction. Opposite-sign
# (e0<0, D4 UNDER-believes -> delta_q>0, final INCREASES/recovers) is the
# "correction/recovery" signature RADDO would want.
opposite = (sign_e0[valid_signs] != sign_dq[valid_signs]).mean()
print(f"P(OPPOSITE signs, i.e. under-belief -> later increase / over-belief -> later decrease) = {opposite:.3f}")

from scipy.stats import binomtest
res = binomtest(int(agree*valid_signs.sum()), valid_signs.sum(), 0.5)
print(f"Binomial test vs chance (p=0.5): p={res.pvalue:.4e}")

# specifically: among lesions where D4 UNDER-believed (e0<0), what fraction
# went on to RECOVER (delta_q>0)?
under = e0 < 0
over = e0 > 0
print(f"\nAmong D4-UNDER-believed lesions (e0<0, n={under.sum()}): "
      f"P(Delta q > 0, i.e. recovered) = {(delta_q[under]>0).mean():.3f}")
print(f"Among D4-OVER-believed lesions (e0>0, n={over.sum()}): "
      f"P(Delta q > 0, i.e. increased further) = {(delta_q[over]>0).mean():.3f}")

print("\n" + "=" * 90)
print("VERDICT")
print("=" * 90)
mag_survives = p < 0.05 and not (np.percentile(boots,2.5) < 0 < np.percentile(boots,97.5))
dir_survives = res.pvalue < 0.05
if mag_survives and dir_survives:
    print("  ==> RADDO's mechanistic premise SURVIVES kill criterion 3.")
    print("      Signed disturbance predicts subsequent change beyond magnitude")
    print("      alone, AND the direction test beats chance. There is a genuine")
    print("      predictive relationship between D4-stage disagreement DIRECTION")
    print("      and what the final stage subsequently does.")
elif mag_survives and not dir_survives:
    print("  ==> PARTIAL: signed e0 adds statistical power over |e0|, but the")
    print("      simple sign-agreement test does not beat chance -- the")
    print("      relationship may be nonlinear/magnitude-dependent rather than a")
    print("      clean directional signal. Interpret cautiously.")
else:
    print("  ==> KILL (criterion 3). e0 predicts final status/magnitude of change")
    print("      but does NOT predict DIRECTION of subsequent recovery beyond")
    print("      simple magnitude. Per the user's own framing: this would mean")
    print("      we'd merely be wrapping a correlational residual in control-")
    print("      theory terminology. Do not build RADDO.")
