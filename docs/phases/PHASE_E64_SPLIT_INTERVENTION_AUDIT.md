# Phase E64 — Corrected Split-Intervention Audit (pool-path vs. skip-path)

## Purpose

E62 and E63 were retracted as evidence about `MaxPool3d` (see
[PHASE_E62_E63_INVALIDATED_SKIP_CONNECTION_BUG.md](PHASE_E62_E63_INVALIDATED_SKIP_CONNECTION_BUG.md)):
their shared `forward_from_enc1()` reused a single `enc1` tensor for both `pool1`'s
input and the decoder skip connection, so a within-cell derangement reached both
paths simultaneously — and since `nn.MaxPool3d(2)` is non-overlapping (k=2,s=2),
`pool1`'s output is provably invariant to any such derangement. The entire measured
effect necessarily came through the skip path.

E64 corrects this by constructing **two fully independent tensors** — `E_pool`
(feeds only `pool1`'s input) and `E_skip` (feeds only the decoder's skip
connection into `dec1`) — and running the 2×2 factorial the user specified:
control (orig/orig), pure pool-path intervention (perm/orig), pure skip-path
intervention (orig/perm), and the total/replication arm (perm/perm).

## Method

- **Two checkpoints, run independently**, per explicit request to separate the
  gate-mediated path from a pure ungated skip effect:
  - **v5/E46** (attention-gate checkpoint, same as E62/E63): `E_skip` feeds both
    `attn_gate1`'s own computation (`W_x(E_skip)`) and the final
    `enc1_gated = E_skip * psi` multiplication — a genuinely **gate-mediated**
    skip effect, labeled as such, not claimed as a pure ungated effect.
  - **v3/D4-only** (same checkpoint E58/E59 used): no attention gate; skip is a
    direct `cat1 = torch.cat([upconv1, E_skip], dim=1)` — the clean, gate-free
    control.
- **Differential check** (the check that would have caught E62/E63's bug):
  before the main audit, verify `pool1(real enc1) == pool1(permuted enc1)`
  exactly, verify a pool-only-permuted forward pass produces **zero** change in
  the final output, and verify a skip-only-permuted forward pass produces a
  **non-zero** change — confirming the two paths are genuinely, independently
  intervenable in this implementation before trusting any downstream statistic.
- Same 125-subject validation population, same derangement construction (8
  independent draws/subject), same statistical discipline (paired t-test,
  Wilcoxon, subject-level sign-flip permutation test ≥1000 trials, bootstrap 95%
  CI) as E47/E48/E58/E62/E63.

## Results

| Checkpoint | Δ_pool (mean, p) | Δ_skip (mean, p) | Δ_both | Spearman(size, Δ_skip) |
|---|---|---|---|---|
| v5 (gated) | **0.00000**, p=1.0 | **0.02262**, p=2.0×10⁻²⁸ | 0.02262 | ρ=−0.355, p<0.001 |
| v3 (ungated) | **0.00000**, p=1.0 | **0.02703**, p=4.4×10⁻³³ | 0.02703 | ρ=−0.404, p<0.001 |

- **Δ_pool is exactly `0.00000`** — not merely non-significant, literally
  zero Dice change across all 125 subjects × 8 derangement draws on both
  checkpoints. Confirms to floating-point exactness that the pool1→pool2→pool3
  →bottleneck→decoder-upsampling path is entirely unaffected by within-cell
  subcell rearrangement, on both a gated and an ungated architecture.
- **Δ_skip reproduces E62's original number almost exactly on v5** (0.0226 vs
  E62's reported 0.0226) — confirms E62's entire original effect was always the
  skip path, as predicted.
- **The effect is present, and larger, without the attention gate**: v3's pure
  ungated skip effect (0.0270) exceeds v5's gate-mediated effect (0.0226), and
  its size-dependence is stronger (ρ=−0.404 vs −0.355). This rules out "the
  attention gate specifically is the mechanism" — a plain concatenation skip
  connection shows the same phenomenon, more strongly.
- Size-specificity replicates on both checkpoints: smaller lesions lose
  significantly more Dice from skip-path scrambling, matching E48's original
  signature direction and significance.
- Differential checks passed exactly as predicted on both checkpoints: pool1
  output identical (0.0 diff), pool-only arm produces zero output change,
  skip-only arm produces a large (~1.0 max abs) output change.

## Decision (per pre-declared rule)

Δ_pool ≈ 0 (not significant, in fact exactly zero) AND Δ_skip > 0 (significant) on
**both** checkpoints →

## Verdict: **SKIP-CONNECTION REPRESENTATION EFFECT CONFIRMED — NOT A POOLING EFFECT**

The trained network's decoder is causally sensitive to the raw subcell spatial
arrangement of the highest-resolution encoder feature map (`enc1`, 64³) as it
reaches `dec1` via the skip connection — independent of any gating mechanism
(present on v5, absent on v3, effect real and larger on v3). This is a real,
newly-and-correctly-established finding, size-specific in the same direction as
every prior finding in this chain. It is **not** evidence for `MaxPool3d`
information loss, and does not support PMD, RCD, TSQL, or any other
pooling-operator redesign — those remain discarded, as their evidential
foundation (E62/E63) is gone.

## Implication for future design work

Any future architectural response to this finding should target the
**skip-connection / first-decoder-stage representation**, not the encoder's
downsampling operator. The exact mechanism by which `dec1` (a small
1×1×1-kernel-free conv stack processing the concatenation of `upconv1` and the
gated/ungated `enc1` skip) is sensitive to subcell rearrangement is not yet
characterized — this phase establishes *that* the sensitivity exists and *where*
it lives (skip path, not pool path), not *why*. A follow-up phase, if pursued,
should do a fresh, focused causal/representational analysis of `dec1`'s use of
the skip tensor specifically, before proposing any operator.

## Process note

This phase's differential check (assert pool1's output is bit-identical under
permutation, assert a pool-only intervention produces zero output change) is
exactly the check that, applied to E62/E63's original `forward_from_enc1`, would
have caught the shared-tensor bug immediately. This check pattern — intervene on
one role of a dual-role tensor, assert the other role's downstream output is
unchanged — is now the standard for any future intervention on a
tensor with more than one architectural role.

## Artifacts

- `experiments/exp_e12_eggo_m/e64/run_e64_split_intervention_audit.py`
- `experiments/exp_e12_eggo_m/e64/E64_v5_gated_table.json`, `E64_v5_gated_summary.json`
- `experiments/exp_e12_eggo_m/e64/E64_v3_ungated_table.json`, `E64_v3_ungated_summary.json`
- `experiments/exp_e12_eggo_m/e64/E64_combined_summary.json`
