"""E238-H4 analysis -- paired test (missed vs matched-detected, same E234
pair) on forward_loss and d1_grad_norm_lesion, per the pre-registered H4
kill criteria discussed with the user."""
import csv
import numpy as np
from scipy import stats
from collections import defaultdict

rows = list(csv.DictReader(open('E238_h4_objective.csv')))
for r in rows:
    r['pair_idx'] = int(r['pair_idx'])
    r['forward_loss'] = float(r['forward_loss'])
    r['mean_prob'] = float(r['mean_prob'])
    r['max_prob'] = float(r['max_prob'])
    r['d1_grad_norm_lesion'] = float(r['d1_grad_norm_lesion'])
    r['size'] = int(r['size'])

by_pair = defaultdict(dict)
for r in rows:
    by_pair[r['pair_idx']][r['role']] = r

complete_pairs = [p for p, d in by_pair.items() if 'missed' in d and 'detected' in d]
print(f"Complete pairs (both missed and detected valid): {len(complete_pairs)}")

fl_missed = np.array([by_pair[p]['missed']['forward_loss'] for p in complete_pairs])
fl_det = np.array([by_pair[p]['detected']['forward_loss'] for p in complete_pairs])
g_missed = np.array([by_pair[p]['missed']['d1_grad_norm_lesion'] for p in complete_pairs])
g_det = np.array([by_pair[p]['detected']['d1_grad_norm_lesion'] for p in complete_pairs])
sz_missed = np.array([by_pair[p]['missed']['size'] for p in complete_pairs])
sz_det = np.array([by_pair[p]['detected']['size'] for p in complete_pairs])

print("\n" + "="*90)
print("SIZE CHECK (should still be ~matched, confirming E234's equalization held on this subset)")
print("="*90)
w_sz, p_sz = stats.wilcoxon(np.log(sz_missed+1) - np.log(sz_det+1))
print(f"  log-size missed mean={np.log(sz_missed+1).mean():.3f}  detected mean={np.log(sz_det+1).mean():.3f}  Wilcoxon p={p_sz:.4f}")

print("\n" + "="*90)
print("FORWARD LOSS: missed vs matched-detected (paired)")
print("="*90)
diff_fl = fl_missed - fl_det
print(f"  missed mean={fl_missed.mean():.4f}  detected mean={fl_det.mean():.4f}  mean diff={diff_fl.mean():.4f}")
w_fl, p_fl = stats.wilcoxon(diff_fl)
print(f"  Wilcoxon (missed vs detected): stat={w_fl:.2f}  p={p_fl:.2e}")
print(f"  n with missed>detected: {(diff_fl>0).sum()}/{len(diff_fl)}")

print("\n" + "="*90)
print("D1 GRADIENT MAGNITUDE (lesion-restricted): missed vs matched-detected (paired)")
print("="*90)
diff_g = g_missed - g_det
print(f"  missed mean={g_missed.mean():.5f}  detected mean={g_det.mean():.5f}  mean diff={diff_g.mean():.5f}")
w_g, p_g = stats.wilcoxon(diff_g)
print(f"  Wilcoxon (missed vs detected): stat={w_g:.2f}  p={p_g:.2e}")
print(f"  n with missed<detected (suppressed): {(diff_g<0).sum()}/{len(diff_g)}")
ratio = g_missed / (g_det + 1e-12)
print(f"  median grad ratio (missed/detected): {np.median(ratio):.4f}")
print(f"  n with ratio<0.5 (>=2x suppressed): {(ratio<0.5).sum()}/{len(ratio)}")
print(f"  n with ratio<0.1 (>=10x suppressed, E228-A-like collapse): {(ratio<0.1).sum()}/{len(ratio)}")

print("\n" + "="*90)
print("H4 KILL-CRITERION EVALUATION")
print("="*90)
loss_elevated = diff_fl.mean() > 0 and p_fl < 0.05
grad_suppressed = diff_g.mean() < 0 and p_g < 0.05
print(f"  Forward loss elevated for missed (objective sees them as costly): {'YES' if loss_elevated else 'NO'} (p={p_fl:.2e})")
print(f"  D1 gradient suppressed for missed (relative to matched detected): {'YES' if grad_suppressed else 'NO'} (p={p_g:.2e})")

if loss_elevated and grad_suppressed:
    print("\n  ==> H4 GETS A MEASURABLE, NEW SIGNATURE: the objective assigns real cost")
    print("      to these lesions (it has NOT already 'given up' on them), yet the")
    print("      gradient that reaches D1/seg_head is significantly smaller than for a")
    print("      matched detected lesion. This is NOT a repeat of E228-A (that measured")
    print("      the D4 aux head only) -- this is the FIRST evidence of suppression at")
    print("      the MAIN decoder pathway. Worth a live/mechanistic follow-up.")
elif loss_elevated and not grad_suppressed:
    print("\n  ==> H4 (suppression) DIES. The objective sees these lesions as costly AND")
    print("      the gradient reaching D1 is NOT significantly different from matched")
    print("      detected lesions -- there is no suppression signature at this layer.")
    print("      Whatever is failing happens elsewhere (upstream encoder representation,")
    print("      per E235's divergence finding, or a downstream readout/calibration")
    print("      effect), not a gradient-starvation story at D1.")
elif not loss_elevated:
    print("\n  ==> H4 (as 'suppression') DIES DIFFERENTLY: the objective itself does NOT")
    print("      treat missed lesions as more costly than matched detected ones --")
    print("      i.e. the loss is already 'satisfied' or indifferent, not suppressed.")
    print("      This would be a distinct, still-informative finding (worth reporting)")
    print("      but does NOT support an 'objective suppresses the signal' framing.")
