# E180 Stages 3-4 — Γ measurement (transformation-specific, then aggregate): pre-registration

**Date**: 2026-09-16
**Status**: PRE-REGISTERED. Not run.
**Depends on**: Stage 2 (`E180_2_severity_calibration.json`, `PROGRAM_VERDICT: PROCEED`).

---

## Purpose

Fix $\Gamma_i$, the per-tile representation-induced decision-instability signal, following the
same-operating-point correction: both the degraded baseline and the perturbation must be
measured **from the same degraded representation state** $Z^{deg}$, never from the intact
representation.

## Surviving transform families (from Stage 2)

| Family | $T_s$ (fixed degradation severity) | Stage-2 dY at $T_s$ |
|---|---|---|
| T1_rank | $r=2$ | 0.1555 |
| T4_spectral | $\gamma=3.0$ | 0.1118 |
| T5_smooth | $\sigma=1.5$ | 0.0928 |

T3_mix is DEAD at enc3 (dropped, per Stage 2's pre-registered rule — a 6-neighbor blend is too
weak relative to enc3's 32³ spatial resolution to move the output). T2_permute is carried
forward **only** as the destructive-negative-control reference (dY≈0.996 at its one tested
severity) — never as a scientific family, per Stage 1's explicit instruction.

## Definitions (same-operating-point, per the adopted correction)

For a fixed family with degradation $T_s$:

$$
Z^{deg} = T_s(Z) \quad \text{(applied uniformly to every tile — this is the "current state")}
$$

$$
\boxed{\Gamma_i = d\Big(D(Z^{deg}),\; D(Z^{deg}_{\text{perturb } i})\Big)}
$$

where $Z^{deg}_{\text{perturb }i}$ applies **an additional, second perturbation** at tile $i$
only, on top of the already-degraded volume. This is the actual instability probe: starting
from the degraded operating point, how much does the decision move when tile $i$'s
representation is perturbed further?

### Correction found during implementation — the probe must be decoupled from the family

The first implementation reused *the same transform at the same severity* as the probe
(re-apply $T_s$ within tile $i$). This is broken: `T1_rank` (SVD truncation) is a near-projection
operator, so $T_1(T_1(x)) \approx T_1(x)$ — **idempotent**, producing near-zero Γ that looks like
a null result but is actually a construction artifact (measured: max diff $\approx 1.9\times
10^{-5}$ on a synthetic check). `T4_spectral`'s $\Sigma \to \Sigma^\gamma$ instead **compounds**:
applying $\gamma=3$ twice gives $\Sigma^{\gamma^2}=\Sigma^9$, a severity far outside calibration,
which is why its raw Γ values were implausibly large. `T5_smooth` compounds similarly (two
passes of $\sigma=1.5$ smoothing is not the same as one).

**Adopted fix**: the probe is a single **fixed, family-independent small perturbation**, applied
within tile $i$'s spatial extent on top of whichever $Z^{deg}$ is current:

$$
Z^{deg}_{\text{perturb }i} = Z^{deg} + M_i \odot (\epsilon \cdot v)
$$

where $v$ is a **fixed unit-norm random direction per channel** (same seed across all
subjects/tiles/families, drawn once and reused, analogous to T2's fixed permutation seed — the
direction itself carries no information, only its magnitude matters), $\epsilon$ is a small
scalar calibrated in Stage 2b (below) to sit in a graded regime, and $M_i$ is the Stage-1 tile
mask. This makes the probe: (a) identical in construction regardless of which family produced
$Z^{deg}$, so instability is measured consistently and is never entangled with a family's own
idempotency/compounding algebra; (b) a single severity to calibrate, not one per family.

### Stage 2b — probe calibration (inserted)

Exactly Stage 2's method, applied to the probe itself: measure whole-volume Dice displacement
$d_Y$ between $D(Z^{deg})$ and $D(Z^{deg}_{\text{perturb }i \text{ (all tiles)}})$ across a small
$\epsilon$ grid ($\epsilon$ as a multiplier on $\sigma(Z^{deg})$, scaled to each family's own
activation magnitude so the probe is comparably sized across families), on the same 3 surviving
families and the same GRADED/DEAD/SATURATED decision rule. Output:
`E180_2b_probe_calibration.json`. Fixes the single $\epsilon$ used by every subsequent Γ
measurement.

**First pass** ($\epsilon \in \{0.05, 0.1, 0.2, 0.5, 1.0\}$, 10 subjects): T1_rank and T5_smooth
DEAD at every point (max $d_Y \approx 0.004$ even at $\epsilon=1.0$, i.e. a perturbation as large
as the tensor's own standard deviation); T4_spectral just reached GRADED at the top of the grid
($\epsilon=1.0$, $d_Y=0.025$). Per the "report honestly, do not silently pick a point outside the
tested grid" rule, the grid was **extended** to $\epsilon \in \{0.05, 0.1, 0.2, 0.5, 1.0, 2.0,
4.0, 8.0\}$ rather than forcing DEAD verdicts from an under-ranged first pass — this is recorded
as an amendment, not a silent redo.

## Four displacement metrics, kept separate (Stage 3), not pre-mixed

| Metric | Definition | Source |
|---|---|---|
| $\Gamma$-Dice | $1 - \text{dice\_agree}(\hat Y_{Z^{deg}}, \hat Y_{Z^{deg}_{\text{perturb }i}})$ | reuse `dice_agree` (E165) |
| $\Gamma$-voxel | fraction of foreground voxels (union of the two binary masks) that flip label | new, trivial (XOR count / union count) |
| $\Gamma$-KL | $D_{KL}(p_{\text{perturb }i} \,\|\, p_{Z^{deg}})$, mean over foreground voxels of either mask | new, small (per-voxel Bernoulli KL on the predicted probability map, since `probs` is per-region sigmoid) |
| $\Gamma$-boundary | mean shift in `boundary_distance` (E167) of the **predicted** mask before/after, over voxels near either predicted boundary | reuse `boundary_distance` machinery, applied to predictions not GT |

All are whole-volume (Gaussian-blended sliding-window output), never a purely local statistic,
per the plan's naming discipline.

## Stage 4 — aggregate Γ

Only after Stage 3's four per-family tables exist. Report multiple explicit aggregate variants,
never one silent scalar:

- $\Gamma_i^{\text{Dice}}$, $\Gamma_i^{\text{voxel}}$, $\Gamma_i^{\text{KL}}$,
  $\Gamma_i^{\text{boundary}}$ — each averaged across the 3 surviving families (T1/T4/T5) for
  that metric.
- $\Gamma_i^{\text{family}}$ — the full aggregate, averaged across all (family × metric)
  combinations.

Stage 6 onward primarily uses $\Gamma_i^{\text{family}}$, but Stage 8's nested regression and
any later audit must be able to substitute a single-metric or single-family Γ to check whether
one component is silently driving the whole signal (this is exactly the check the user flagged:
"is the phenomenon prediction instability generally, or is one particular output metric creating
it?").

## What this stage does NOT do

- Does not touch ground truth (all four metrics are prediction-vs-prediction).
- Does not compute Δ yet (Stage 5).
- Does not run on the full 125 subjects yet — Stage 3/4 run on the same small calibration-scale
  sample as Stage 2 (10 subjects) until Stage 5.5's pipeline gate clears.

## Output

- `experiments/exp_e12_eggo_m/e180/E180_gamma_per_subject.json` — keyed by
  `(sid, window_id, family, metric)`, plus the Stage-4 aggregates per `(sid, window_id)`.
