# E186.1 -- Causal E3 spatial-permutation gate: KILLED (direction + K5)

Frozen model E131_v5control_seed0. 125 val subjects, 108 contributed.
4000 rows. Paired units: L=58, S=121, H=221. No training.

## Design note (deviation from spec, deliberate)

enc3 feeds BOTH pool3->bottleneck AND the decoder skip cat3. Permuting a
single shared tensor hits both -- the confound that invalidated E62/E63 and
was caught only in E64. Each perturbation therefore ran in three arms:
`pool` (permuted->pool3, original->skip) = the actual hypothesis,
`skip` (original->pool3, permuted->skip), `both` (naive).
This mattered: skip carries MORE of the naive effect than pool.

## Integrity checks

- `forward_from_e3` is bit-exact vs `model(x)`: max|sigmoid(z)-probs| = 0.0
- Permutation preserves activation statistics: max|dmean| 9.5e-7, max|dstd| 9.5e-7
  => "the permutation changed the activation level" is closed by construction

## Result (primary arm = pool, primary stat dz = median z_comp - median z_shell)

| pop | n | dz(P0) | dP1 | dP2 | dP3 |
|-----|---|--------|-----|-----|-----|
| L | 58 | 6.339 | 0.000 | **+0.967** | -4.872 |
| S | 121 | 49.005 | 0.000 | -0.068 | -38.715 |
| H | 221 | 0.405 | 0.000 | +0.010 | -0.100 |

## Two findings that decide it

**1. P1 is structurally void.** In the pool arm P1 is BIT-IDENTICAL to P0
(max|delta| = 0.000e+00, 400/400 rows exactly zero; skip arm max 15.19).
perm_local permutes within 2x2x2 blocks = pool3's own windows, so max-pool
is invariant to it BY CONSTRUCTION. The mild rung of the dose-response
ladder cannot move. Design flaw, not a null.

**2. The L effect is real but INVERTED.** 83% of L units IMPROVE under
pool/P2 (median +0.967, mean +1.591) vs 47% (S) and 53% (H) -- coin flips.
So specificity is genuine (p_LS=3.0e-06, p_LH=1.4e-14) but it certifies
that DESTROYING spatial arrangement makes missed lesions look MORE like
lesions to the bottleneck. Opposite of the hypothesis.

**K5 fires:** only **1 of 58** L components crossed from negative to
positive evidence. A ~1-logit median gain on components far below threshold
cannot flip detections. No achievable Dice.

## Verdict

K1 passed (there IS a causal effect). K2 passed (it IS lesion-specific).
But K1/K2 are satisfiable by EITHER SIGN, and P1's structural zero left the
monotonicity check with a dead rung. The preregistered criteria mis-scored
this. Honest verdict: **KILL on direction + K5.**

E186.2 (winner-selection anatomy) NOT run -- it presupposes a
spatial-structure effect this refutes.

## Surviving positive finding (skip arm)

S is damaged enormously by spatial destruction (-11.8 at P2, -18.8 at P3,
p~1e-25) while L barely responds (-1.26). Detected-lesion evidence lives in
the DECODER SKIP's spatial arrangement, not the bottleneck's. For missed
lesions there is little there to destroy -- consistent with E185 (L's
localization already gone at E3, SCR 0.99; absent at bottleneck, 0.31).
