"""E232 analysis -- the four measurements + decision gate, per the user's
exact spec.

1. Dice: p_cap vs p_0 (whole-volume, already aggregated per subject).
2. FN: does intersection lose small lesions?
3. FP: does it remove unstable false positives?
4. Lesion recovery: are missed lesions represented in high-variance regions?
   -- THE CRITICAL measurement: GT lesion -> A(x), controlling for size/
   isolation (reusing E223's own cached covariates, joined on subject_id/
   comp_id -- NOTE E232 used the VAL split while E223/E226/E227 used the
   TRAIN split's exposure-audit convention on the same underlying subjects
   where they overlap; comp_id numbering is REGENERATED here via this
   script's own ndi.label call on the val subject's full native target,
   so it is NOT guaranteed to match E223's train-side numbering for the
   same subject even if the subject_id happens to coincide -- flagged,
   not silently assumed identical).

DECISION GATE: proceed only if A(x) predicts missed lesions AND p_cap
preserves lesion recall, controlling for size/isolation.
"""
import csv
import numpy as np
from scipy import stats

d232 = list(csv.DictReader(open('E232_ambiguity.csv')))
for r in d232:
    r['size'] = int(r['size']); r['mean_A'] = float(r['mean_A']); r['max_A'] = float(r['max_A'])
    r['detected_p0'] = int(r['detected_p0']); r['detected_pcap'] = int(r['detected_pcap'])
    r['dice_p0'] = float(r['dice_p0']); r['dice_pcap'] = float(r['dice_pcap'])
    r['fn_p0'] = int(r['fn_p0']); r['fn_pcap'] = int(r['fn_pcap'])
    r['fp_p0_total'] = int(r['fp_p0_total']); r['fp_pcap_total'] = int(r['fp_pcap_total'])

print(f"Total lesion rows: {len(d232)}")
subjects = sorted(set(r['subject_id'] for r in d232))
print(f"Subjects: {len(subjects)}")

print("\n" + "=" * 90)
print("1-3. WHOLE-VOLUME: Dice / FN / FP, p_0 vs p_cap (per-subject, deduped)")
print("=" * 90)
seen = set()
dice_p0_list, dice_pcap_list, fn_p0_list, fn_pcap_list, fp_p0_list, fp_pcap_list = [], [], [], [], [], []
for r in d232:
    if r['subject_id'] in seen:
        continue
    seen.add(r['subject_id'])
    dice_p0_list.append(r['dice_p0']); dice_pcap_list.append(r['dice_pcap'])
    fn_p0_list.append(r['fn_p0']); fn_pcap_list.append(r['fn_pcap'])
    fp_p0_list.append(r['fp_p0_total']); fp_pcap_list.append(r['fp_pcap_total'])

dice_p0_arr = np.array(dice_p0_list); dice_pcap_arr = np.array(dice_pcap_list)
fn_p0_arr = np.array(fn_p0_list); fn_pcap_arr = np.array(fn_pcap_list)
fp_p0_arr = np.array(fp_p0_list); fp_pcap_arr = np.array(fp_pcap_list)

print(f"Dice: p_0 mean={dice_p0_arr.mean():.4f}  p_cap mean={dice_pcap_arr.mean():.4f}  "
      f"delta={dice_pcap_arr.mean()-dice_p0_arr.mean():+.4f}")
w_stat, p_dice = stats.wilcoxon(dice_p0_arr, dice_pcap_arr)
print(f"  Wilcoxon signed-rank p={p_dice:.4e}")

print(f"\nFN (voxels): p_0 mean={fn_p0_arr.mean():.1f}  p_cap mean={fn_pcap_arr.mean():.1f}  "
      f"delta={fn_pcap_arr.mean()-fn_p0_arr.mean():+.1f} "
      f"({100*(fn_pcap_arr.mean()-fn_p0_arr.mean())/fn_p0_arr.mean():+.1f}%)")
print(f"FP (voxels): p_0 mean={fp_p0_arr.mean():.1f}  p_cap mean={fp_pcap_arr.mean():.1f}  "
      f"delta={fp_pcap_arr.mean()-fp_p0_arr.mean():+.1f} "
      f"({100*(fp_pcap_arr.mean()-fp_p0_arr.mean())/fp_p0_arr.mean():+.1f}%)")

print("\n" + "=" * 90)
print("4. CRITICAL TEST: does A(x) predict missed lesions? (raw)")
print("=" * 90)
size = np.array([r['size'] for r in d232])
mean_A = np.array([r['mean_A'] for r in d232])
max_A = np.array([r['max_A'] for r in d232])
det = np.array([r['detected_p0'] for r in d232])
missed = 1 - det

for name, arr in [('mean_A', mean_A), ('max_A', max_A)]:
    d, m = arr[det == 1], arr[det == 0]
    u, p = stats.mannwhitneyu(d, m)
    print(f"  {name}: detected median={np.median(d):.6f}  missed median={np.median(m):.6f}  p={p:.3e}")
    direction = "HIGHER in missed" if np.median(m) > np.median(d) else "LOWER in missed"
    print(f"    -> ambiguity is {direction} for missed lesions")

print("\n--- Size confound check ---")
r_size_A, p_size_A = stats.spearmanr(size, mean_A)
print(f"Spearman(size, mean_A) = {r_size_A:.4f}  p={p_size_A:.3e}")

print("\n--- Partial correlation, controlling for size (log) ---")
log_size = np.log(size + 1)
X = np.column_stack([np.ones(len(d232)), log_size])
beta_m, *_ = np.linalg.lstsq(X, missed.astype(float), rcond=None)
resid_m = missed.astype(float) - X @ beta_m
beta_a, *_ = np.linalg.lstsq(X, mean_A, rcond=None)
resid_a = mean_A - X @ beta_a
r_partial, p_partial = stats.pearsonr(resid_m, resid_a)
print(f"partial corr(missed, mean_A | log_size) = {r_partial:+.4f}  p={p_partial:.4e}")

print("\n--- Size-matched quintile breakdown ---")
bins = np.percentile(size, [0, 20, 40, 60, 80, 100])
for i in range(5):
    lo, hi = bins[i], bins[i+1]
    sel = (size >= lo) & (size <= hi)
    d_a, m_a = mean_A[sel & (det==1)], mean_A[sel & (det==0)]
    if len(d_a) < 3 or len(m_a) < 3:
        print(f"  size [{lo:.0f},{hi:.0f}]: too few (det={len(d_a)}, missed={len(m_a)})")
        continue
    p = stats.mannwhitneyu(d_a, m_a)[1]
    print(f"  size [{lo:.0f},{hi:.0f}] (n={sel.sum()}): detected mean_A={np.median(d_a):.6f}  "
          f"missed mean_A={np.median(m_a):.6f}  p={p:.3f}")

print("\n--- Nonzero-consensus-evidence check: among MISSED lesions, do they")
print("    have nonzero p_cap consensus anywhere, i.e. 'ambiguous but not absent'? ---")
missed_rows = [r for r in d232 if r['detected_p0'] == 0]
missed_with_pcap = sum(1 for r in missed_rows if r['detected_pcap'] == 1)
print(f"  Missed lesions (n={len(missed_rows)}): "
      f"{missed_with_pcap} ({100*missed_with_pcap/max(1,len(missed_rows)):.1f}%) "
      f"ALSO detected in p_cap (stable consensus) despite p_0 missing them")

print("\n" + "=" * 90)
print("DECISION GATE")
print("=" * 90)
gate1 = p_partial < 0.05 and r_partial > 0  # A(x) predicts missed, controlling for size, POSITIVE direction
gate2 = dice_pcap_arr.mean() >= dice_p0_arr.mean() - 0.02  # p_cap preserves recall (small tolerance)
print(f"Gate 1 (A(x) predicts missed lesions, size-controlled, positive direction): "
      f"{'PASS' if gate1 else 'FAIL'}  (r={r_partial:+.4f}, p={p_partial:.3e})")
print(f"Gate 2 (p_cap preserves Dice/recall, within 2pp of p_0): "
      f"{'PASS' if gate2 else 'FAIL'}  (p_0={dice_p0_arr.mean():.4f}, p_cap={dice_pcap_arr.mean():.4f})")

if gate1 and gate2:
    print("\n  ==> PASSES the decision gate. The idea has something -- proceed to")
    print("      design an actual algorithm around consensus/ambiguity.")
elif gate1 and not gate2:
    print("\n  ==> PARTIAL: A(x) IS informative about missed lesions, but p_cap")
    print("      loses too much recall to be useful as-is. The ambiguity SIGNAL")
    print("      may still be usable (e.g. as a flag, not as a hard filter), but")
    print("      the naive intersection-consensus prediction itself is not viable.")
else:
    print("\n  ==> FAILS. Ambiguity does not predict missed lesions beyond size, or")
    print("      simply tracks lesion size/noise as the user's own kill criterion")
    print("      anticipated. KILL this candidate.")
