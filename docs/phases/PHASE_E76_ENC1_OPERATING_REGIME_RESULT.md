# Phase E76 — enc1 Operating-Regime Test: PASS

## Origin

Following E75's split result (enc1 magnitude causally confirmed;
bottleneck killed/reversed), the user flagged a real confound: E75's wide
alpha range {0.25, 0.5, 1.0, 2.0} pushed the intact network's own Dice
down substantially at the extremes (0.902 → 0.867 at alpha=2.0),
meaning the causal result there could partly reflect the network
operating far outside its normal regime rather than a clean local
property. E76 tests the sharper, decision-relevant question: does
dS/dalpha|alpha=1 > 0 hold in a tight neighborhood around the network's
actual operating point, where the intact segmentation stays essentially
undisturbed — the regime any real training-time intervention would
actually operate in.

## Method

Identical construction to E75, scoped to enc1 only (per E75's split
result — the bottleneck hypothesis is killed), finer alpha grid:
{0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30}. Direction preservation
re-verified numerically (max diff < 1e-5) before any subject processed.
For each alpha: D_intact(alpha) (no translation) and S(alpha)
(translation-induced Dice drop), same 3-voxel roll as E65/E74/E75.

Pre-declared PASS rule: (a) D_intact stays within 0.02 of its alpha=1.0
value across the WHOLE range (operating-regime stability), AND (b) S(alpha)
is monotonic with a significant, positive local slope at alpha=1
(measured via the paired S(1.10)-S(0.90) comparison).

## Results

### (a) Operating-regime stability: PASS

| alpha | D_intact |
|---|---|
| 0.70 | 0.8921 |
| 0.80 | 0.8976 |
| 0.90 | 0.9007 |
| 1.00 | 0.9023 |
| 1.10 | 0.9020 |
| 1.20 | 0.9008 |
| 1.30 | 0.8990 |

Max deviation from alpha=1.0: **0.0101** — well under the 0.02 threshold,
at every tested alpha. The network stays in its normal operating regime
across the whole ±30% range (unlike E75's alpha=2.0, which dropped 0.035).

### (b) S(alpha): strictly monotonic, clean local slope

| alpha | mean S |
|---|---|
| 0.70 | 0.2772 |
| 0.80 | 0.2925 |
| 0.90 | 0.3057 |
| 1.00 | 0.3174 |
| 1.10 | 0.3268 |
| 1.20 | 0.3356 |
| 1.30 | 0.3436 |

Strictly monotonic across all 7 points (not just the endpoints). Local
slope at alpha=1, S(1.10) − S(0.90) = **+0.0211**, paired t p=7.4e-42,
Wilcoxon p=1.5e-21 — large relative to noise, overwhelmingly significant.

## Verdict: PASS

Both pre-declared conditions hold. The causal chain

    enc1 feature magnitude -> translation sensitivity -> segmentation vulnerability

is confirmed as a genuine LOCAL property of the network's actual
operating point, not an artifact of the wide-range test pushing it out
of distribution. This satisfies the precondition the user set before any
intervention design: "if E76 is clean, then — and only then — design the
enc1 intervention."

## Status

Per the user's own sequencing, this clears the way to design a
representation-level magnitude-control intervention at enc1 (NOT a
spatial gate — E73 already showed gating is a poor way to exploit these
signals). The exact formula is intentionally not yet chosen, per the
user's explicit instruction to establish the operating regime first
before committing to a mechanism.

## Artifacts

- `experiments/exp_e12_eggo_m/e76/run_e76_enc1_operating_regime.py`
- `experiments/exp_e12_eggo_m/e76/E76_enc1_operating_regime_table.json`
- `experiments/exp_e12_eggo_m/e76/E76_enc1_operating_regime_summary.json`
