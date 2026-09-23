# E186.1-E189: skip/pooling branch CLOSED. Four experiments, four kills.

Frozen model E131_v5control_seed0 throughout. No training. Dice branch
(0.868620) untouched.

## Chain

E65 -> E79 -> E74 -> E186.1 -> E188 -> E189

## E186.1 -- pool3 spatial permutation: KILLED (direction + K5)

4000 rows, 58 L / 121 S / 221 H. Three arms (pool/skip/both) to avoid the
E62/E64 shared-tensor confound -- necessary: skip carried MORE of the naive
effect than pool.

- P1 (within-2x2x2 permutation) BIT-IDENTICAL to P0 in the pool arm
  (max|delta| = 0.000e+00, 400/400 rows): max-pool is invariant to
  within-window permutation BY CONSTRUCTION. Dead rung in the ladder.
- P2 on L: 83% of units IMPROVED (median +0.967) vs coin-flip for S and H.
  Wrong sign -- hypothesis predicted damage.
- Only 1/58 components crossed z<0 -> z>0. Median L logit -18.222.
  Effect is ~5% of an 18-logit gap.

K1/K2 passed but are satisfiable by EITHER SIGN. Honest verdict: KILL.

## E187 -- SEC++ coherence surplus at pool2: KILLED (K3)

438 rows, 74 L / 145 S / 219 H. delta = 0.1155 measured from E2, not tuned.
C_null derived in exact closed form (3/7)[6r(1)+5r(2)+...+r(6)] over all 7!
permutations; verified against brute force to 1.7e-13 for all 8 winner
positions. Vectorised implementation matches to 4.5e-14.

- dC_L = 0.2323, dC_H = 0.2444 -- hard negatives have MORE coherence surplus
  than missed lesions. Predicted inequality is FALSE.
- AUC(dC) = 0.685 but AUC(p,S) = 0.8548 and AUC(p,S,dC) = 0.8534:
  increment -0.0014. Partial corr(dC, label | p,S) = -0.1159 (p=0.048).
- S_med = 1.0 and C_obs = C_null = 1.2857 in ALL populations: support is
  saturated, so the uniform-support edge case (dC=0) fires dataset-wide.

The 0.685 AUC is p_med leaking through, not coherence.

## E188 -- cross-depth correspondence asymmetry at enc3: PASSED

1584 rows, 58 L / 119 S / 219 H. Four perturbations, skip path only.
Normalised by |dz_intact| to control the floor effect.

| perturbation  | L      | S      | H      | L-S    | p       |
|---------------|--------|--------|--------|--------|---------|
| translation   | -0.105 | +0.482 | +0.158 | -0.587 | 2.0e-15 |
| channel       | +0.015 | +0.192 | +0.469 | -0.177 | 5.7e-07 |
| amplitude     | -0.063 | +0.054 | -0.009 | -0.117 | 2.4e-12 |
| local_spatial | +0.188 | +0.235 | +0.071 | -0.047 | 2.9e-02 |

Orderings DIVERGE:
  S: translation > local_spatial > channel > amplitude
  L: local_spatial > channel > translation ~ amplitude  (translation NEGATIVE)

NEW: translation's sign FLIPS by detectability. E65 established correspondence
dominates; nobody had measured that it inverts for missed lesions.

## E189 -- relief vs recovery: H1 RELIEF. BRANCH KILLED.

394 rows, 57 L / 119 S / 218 H. Three conditions (Intact / Translate / Zero),
identical ROI geometry, skip path only.

DECISIVE (mean component logit, L):
  n(T>Z) = 29/57   n(Z>T) = 28/57   Wilcoxon p = 0.393

A coin flip. Zeroing the ROI reproduces translation's benefit => translation
supplies NO useful displaced information; it removes a suppressive
contribution. H1 by the preregistered criterion.

- The `dz` variant shows dT=+0.400 vs dZ=-0.974 (p=0.014) and LOOKS like H2,
  but zeroing suppresses the SHELL too, so the comp-minus-shell statistic is
  contaminated. On the direct component logit the effect vanishes.
- H controls confirm the mechanism: for hard negatives zeroing hurts far more
  than translating (dT-dZ = +0.787, p=6.9e-20). The skip DOES contribute at
  background sites. For missed lesions removing it costs nothing.
- Crossings: T gave 5/57 (best in the branch; E186.1 gave 1/58), Z gave 1/57.
  But T is not beating Z, so the 5 are not attributable to displaced
  information.

## Terminus

The enc3 skip carries no recoverable information about missed lesions. The
E188 sign reversal is real and previously unmeasured, but its explanation is
suppression relief, reproducible by deletion. No operator follows.

Consistent with E79 (decoder uses skip as ABSOLUTE-COORDINATE lookup,
smoothing recovers 11%) and E74 (translation-sensitivity generalises across
depth, "argues against a skip-specific fix").

## Retracted before running (caught against the project's own kill-list)

- MCB (miss-conditioned boundary loss): E103 killed CC-DiceCE on this model
  with the OPPOSITE sign (+16.5% FN); E33 marked component-wise loss
  reweighting OCCUPIED.
- Skip-arm lead as "unexplored": E64/E65 already ran the split-tensor
  decomposition at enc1 in 2024.
