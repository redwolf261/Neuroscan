"""E248 analysis -- source of the protective clipping bias.

DECISIVE TEST: cross-lesion consistency of per-channel protection
(p_pos_c) among DETECTED lesions. Split into two random halves (many
repeats, not one split), correlate each channel's mean p_pos_c across
the two halves. HIGH correlation -> H8-A (fixed channel bias, same
channels protected regardless of which lesion). LOW correlation -> H8-B
(lesion-adaptive, no stable channel identity).

SECONDARY: three-way p_pos_c comparison (detected/G2A/G2B) per channel,
and BN gamma/beta correlation with the observed protection pattern
(explanatory only, not causal, per explicit user caution).
"""
import numpy as np
from scipy import stats

d = np.load('E248_protection.npz')
detected = d['detected']  # (n_det, 32)
g2a = d['G2A']            # (n_g2a, 32)
g2b = d['G2B']            # (n_g2b, 32)
gamma = d['gamma']        # (32,)
beta = d['beta']          # (32,)

print(f"detected: n={detected.shape[0]}  G2A: n={g2a.shape[0]}  G2B: n={g2b.shape[0]}")

print("\n" + "="*100)
print("CROSS-LESION CONSISTENCY TEST (decisive: H8-A fixed-channel-bias vs H8-B adaptive)")
print("="*100)

rng = np.random.default_rng(2480)
N_SPLITS = 200
n_det = detected.shape[0]
pearson_rs, spearman_rs = [], []
for _ in range(N_SPLITS):
    idx = rng.permutation(n_det)
    half1, half2 = idx[:n_det//2], idx[n_det//2:]
    mean1 = detected[half1].mean(axis=0)  # (32,) per-channel mean p_pos across half 1
    mean2 = detected[half2].mean(axis=0)  # per-channel mean p_pos across half 2
    r_p, _ = stats.pearsonr(mean1, mean2)
    r_s, _ = stats.spearmanr(mean1, mean2)
    pearson_rs.append(r_p)
    spearman_rs.append(r_s)

pearson_rs = np.array(pearson_rs)
spearman_rs = np.array(spearman_rs)
print(f"\nAcross {N_SPLITS} random half-splits of {n_det} detected lesions:")
print(f"  Pearson r (split1-mean vs split2-mean, across 32 channels):")
print(f"    mean={pearson_rs.mean():.4f}  median={np.median(pearson_rs):.4f}  "
      f"[{np.percentile(pearson_rs,5):.4f}, {np.percentile(pearson_rs,95):.4f}] (5-95pct)")
print(f"  Spearman r:")
print(f"    mean={spearman_rs.mean():.4f}  median={np.median(spearman_rs):.4f}  "
      f"[{np.percentile(spearman_rs,5):.4f}, {np.percentile(spearman_rs,95):.4f}] (5-95pct)")

# also report the FULL-SAMPLE per-channel mean p_pos, and its own spread
full_mean_p_pos = detected.mean(axis=0)
full_std_p_pos = detected.std(axis=0)
print(f"\nFull-sample (all {n_det} detected) per-channel mean p_pos_c:")
print(f"  range: [{full_mean_p_pos.min():.4f}, {full_mean_p_pos.max():.4f}]")
print(f"  channel-to-channel std of the MEAN: {full_mean_p_pos.std():.4f}")
print(f"  (a wide range + high split-half consistency = clear evidence of fixed bias)")

print("\n" + "="*100)
print("THREE-WAY COMPARISON: per-channel mean p_pos_c, detected vs G2A vs G2B")
print("="*100)
mean_det = detected.mean(axis=0)
mean_g2a = g2a.mean(axis=0)
mean_g2b = g2b.mean(axis=0)
print(f"\n{'ch':4s} {'detected':>10s} {'G2A':>10s} {'G2B':>10s} {'det-G2A':>10s} {'gamma':>8s} {'beta':>8s}")
order = np.argsort(mean_det)[::-1]
for c in order:
    print(f"{c:4d} {mean_det[c]:10.4f} {mean_g2a[c]:10.4f} {mean_g2b[c]:10.4f} "
          f"{mean_det[c]-mean_g2a[c]:+10.4f} {gamma[c]:8.4f} {beta[c]:8.4f}")

# paired test across channels: is detected's protection pattern consistently
# ABOVE g2a's, in a channel-matched sense?
diff_det_g2a = mean_det - mean_g2a
w_stat, p_val = stats.wilcoxon(diff_det_g2a)
print(f"\nWilcoxon (per-channel, detected vs G2A mean p_pos): p={p_val:.4e}  "
      f"n_channels_detected_higher={int((diff_det_g2a>0).sum())}/32")

print("\n" + "="*100)
print("BN1 GAMMA/BETA CORRELATION WITH PROTECTION (EXPLANATORY ONLY, NOT CAUSAL)")
print("="*100)
r_gamma, p_gamma = stats.pearsonr(gamma, mean_det)
r_beta, p_beta = stats.pearsonr(beta, mean_det)
print(f"corr(gamma, detected mean p_pos_c): r={r_gamma:.4f}  p={p_gamma:.4e}")
print(f"corr(beta,  detected mean p_pos_c): r={r_beta:.4f}  p={p_beta:.4e}")
print("(NOTE: correlation with gamma/beta does not establish that these parameters")
print(" CAUSE the protection pattern -- conv1's own weights and the upstream")
print(" activation distribution jointly determine bn1_out; this is descriptive only)")

print("\n" + "="*100)
print("VERDICT")
print("="*100)
consistency_high = pearson_rs.mean() > 0.5
if consistency_high:
    print(f"  ==> H8-A (FIXED CHANNEL BIAS) SUPPORTED: split-half cross-lesion consistency")
    print(f"      is high (mean Pearson r={pearson_rs.mean():.3f} across {N_SPLITS} splits) -- the SAME")
    print(f"      channels are preferentially protected regardless of which detected lesions")
    print(f"      are sampled. The protection is a stable, learned per-channel property of")
    print(f"      conv1/BN1, not something detected lesions' own activation patterns")
    print(f"      independently produce each time.")
else:
    print(f"  ==> H8-B (LESION-ADAPTIVE) SUPPORTED: split-half cross-lesion consistency is")
    print(f"      LOW (mean Pearson r={pearson_rs.mean():.3f}) -- which channels are protected varies")
    print(f"      substantially depending on which lesions are sampled. This argues AGAINST a")
    print(f"      fixed per-channel bias and FOR detected lesions' own activation patterns")
    print(f"      being what produces the protective effect, not a static parameter property.")
