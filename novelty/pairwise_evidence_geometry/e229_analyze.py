"""E229 analysis -- does the cross-scale disagreement D_l = Q_l - P_l_fine
predict missedness independently of everything already tested?

Pre-registered kill criterion (same discipline as the ρ*cos feasibility
check): if D_l's incremental contribution (F-test + partial correlation,
bootstrap-CI'd) is not significant after controlling for log(size),
isolation, distance, Q_l, and cos_dec3_aux_vs_seg, KILL the ADRC-transplant
candidate before any architecture work.
"""
import csv
import numpy as np
from scipy import stats

d229 = list(csv.DictReader(open('E229_disagreement.csv')))
e227 = list(csv.DictReader(open('E227_gradient.csv')))
e223 = list(csv.DictReader(open('E223_exposure.csv')))

d229_by_key = {(r['subject_id'], r['comp_id']): r for r in d229}
e223_by_key = {(r['subject_id'], r['comp_id']): r for r in e223}

joined = []
for r227 in e227:
    key = (r227['subject_id'], r227['comp_id'])
    r229 = d229_by_key.get(key)
    r223 = e223_by_key.get(key)
    if r229 is None or r223 is None:
        continue
    joined.append({
        'size': float(r227['size']),
        'Q_l': float(r227['Q_l']),
        'cos': float(r227['cos_dec3_aux_vs_seg']),
        'P_l_fine': float(r229['P_l_fine']),
        'detected': int(r227['detected']),
        'isolated': int(r223['isolated_from_main_wt']),
        'dist': float(r223['dist_to_wt_centroid']),
    })

print(f"E227 rows: {len(e227)}  E229 rows: {len(d229)}  joined: {len(joined)}")

size = np.array([r['size'] for r in joined])
Q_l = np.array([r['Q_l'] for r in joined])
cos = np.array([r['cos'] for r in joined])
P_l = np.array([r['P_l_fine'] for r in joined])
det = np.array([r['detected'] for r in joined])
missed = 1 - det
iso = np.array([r['isolated'] for r in joined])
dist = np.array([r['dist'] for r in joined])
log_size = np.log(size + 1)

valid = np.isfinite(cos) & np.isfinite(Q_l) & np.isfinite(P_l)
print(f"valid: {valid.sum()}")
size, Q_l, cos, P_l, det, missed, iso, dist, log_size = [
    a[valid] for a in (size, Q_l, cos, P_l, det, missed, iso, dist, log_size)]

D_l = Q_l - P_l  # the disturbance/disagreement signal

print(f"\nD_l stats: mean={D_l.mean():.4f} std={D_l.std():.4f} "
      f"min={D_l.min():.4f} max={D_l.max():.4f}")
print(f"D_l for detected: mean={D_l[det==1].mean():.4f}  "
      f"D_l for missed: mean={D_l[det==0].mean():.4f}")
u, p_raw = stats.mannwhitneyu(D_l[det==1], D_l[det==0])
print(f"Raw Mann-Whitney (D_l, detected vs missed): p={p_raw:.4e}")

X_base = np.column_stack([np.ones(len(missed)), log_size, iso, dist, Q_l, cos])
beta_base, *_ = np.linalg.lstsq(X_base, missed.astype(float), rcond=None)
resid_base = missed.astype(float) - X_base @ beta_base
ss_res_base = np.sum(resid_base**2)
ss_tot = np.sum((missed - missed.mean())**2)
r2_base = 1 - ss_res_base/ss_tot

X_ext = np.column_stack([X_base, D_l])
beta_ext, *_ = np.linalg.lstsq(X_ext, missed.astype(float), rcond=None)
resid_ext = missed.astype(float) - X_ext @ beta_ext
ss_res_ext = np.sum(resid_ext**2)
r2_ext = 1 - ss_res_ext/ss_tot

print(f"\nR^2 base (log_size, isolated, dist, Q_l, cos) = {r2_base:.4f}")
print(f"R^2 extended (+ D_l)                            = {r2_ext:.4f}")
print(f"Incremental R^2 from D_l                        = {r2_ext - r2_base:.4f}")

n = len(missed)
p_base_n = X_base.shape[1]
p_ext_n = X_ext.shape[1]
F = ((ss_res_base - ss_res_ext) / (p_ext_n - p_base_n)) / (ss_res_ext / (n - p_ext_n))
p_F = 1 - stats.f.cdf(F, p_ext_n - p_base_n, n - p_ext_n)
print(f"F-test for D_l's incremental contribution: F={F:.4f}  p={p_F:.4f}")

beta_d, *_ = np.linalg.lstsq(X_base, D_l, rcond=None)
resid_d = D_l - X_base @ beta_d
r_partial, p_partial = stats.pearsonr(resid_base, resid_d)
print(f"\npartial corr(missed, D_l | log_size, isolated, dist, Q_l, cos) = "
      f"{r_partial:+.4f}  p={p_partial:.4f}")

rng = np.random.default_rng(0)
boots = []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    Xb = X_base[idx]; mb = missed[idx].astype(float); Db = D_l[idx]
    try:
        beta_m, *_ = np.linalg.lstsq(Xb, mb, rcond=None)
        rm = mb - Xb @ beta_m
        beta_dd, *_ = np.linalg.lstsq(Xb, Db, rcond=None)
        rd = Db - Xb @ beta_dd
        r, _ = stats.pearsonr(rm, rd)
        boots.append(r)
    except Exception:
        continue
boots = np.array(boots)
print(f"bootstrap partial corr: mean={boots.mean():.4f}  "
      f"95% CI [{np.percentile(boots,2.5):.4f}, {np.percentile(boots,97.5):.4f}]")

print("\n" + "=" * 90)
print("VERDICT")
print("=" * 90)
if p_F < 0.05 and p_partial < 0.05 and not (np.percentile(boots,2.5) < 0 < np.percentile(boots,97.5)):
    print("  ==> D_l shows an independent, bootstrap-stable signal. RETAINED.")
    print("      Worth pursuing the ADRC-transplant architecture design.")
else:
    print("  ==> D_l adds no reliable independent signal. KILL this candidate")
    print("      before any architecture work.")
