# E180 Stage 5 — Δ counterfactual (corrected, same-operating-point design): pre-registration

**Date**: 2026-09-16
**Status**: PRE-REGISTERED. Not run.
**Depends on**: Stage 3-4 (`E180_gamma_per_subject.json`).

---

## Purpose

Measure the counterfactual benefit of restoring one tile's representation from the same
degraded operating point $Z^{deg}$ that Γ was measured from — the corrected design adopted
after the initial draft's flaw (Δ defined against "restore intact rank" while Γ was measured
from the intact representation, two different operating points).

## Definition

For a fixed family with degradation $T_s$ (from Stage 2: T1_rank r=2, T4_spectral γ=3.0,
T5_smooth σ=1.5):

$$
Z^{deg} = T_s(Z) \quad \text{(applied uniformly, same construction as Stage 3)}
$$

$$
Z^{cf}_i = Z^{deg} + M_i \odot (Z - Z^{deg}) \quad \text{(restore tile } i \text{ to its intact value, everywhere else stays degraded)}
$$

using the Stage 1 tile ledger for $M_i$ (1 inside tile $i$'s enc3-grid extent, 0 elsewhere —
same coordinate mapping as Stage 3's `enc3_tile_bounds`).

$$
\boxed{\Delta_i^{global} = Dice_{\text{volume}}\big(D(Z^{cf}_i),\, Y\big) - Dice_{\text{volume}}\big(D(Z^{deg}),\, Y\big)}
$$

**Ground truth is used only here, only to evaluate** — never to construct Γ, never to select
which tile to restore (same rule as E172).

**Naming discipline**: $\Delta_i^{global}$, not "local Dice improvement" — restoring one tile's
activation affects many output voxels via the Gaussian-blended sliding-window reassembly, so
this is a whole-volume causal effect of a local intervention, not a local metric.

## Why this reuses the same $T_s$, family, and tile ledger as Stage 3

Both quantities must describe the same causal object at the same operating point:
$Z \to Z^{deg}$ (fixed) is the shared baseline; Γ asks "how much does a further perturbation at
tile $i$ move the decision from here," Δ asks "how much does undoing the degradation at tile
$i$ move the decision toward the truth from here." Reusing the exact same $Z^{deg}$ construction
per family is what makes $\rho(\Gamma, \Delta)$ interpretable at all.

## Method

Per subject, per surviving family (T1_rank, T4_spectral, T5_smooth):
1. Compute $Z^{deg}$ once (uniform $T_s$), run sliding-window inference, get `dice_deg` against
   GT.
2. For each tile $i$ in the ledger: construct $Z^{cf}_i$ via the masked splice, run
   sliding-window inference, get `dice_cf_i` against GT, compute $\Delta_i = dice\_cf\_i -
   dice\_deg$.

This exactly mirrors Stage 3's loop structure (one `Z^{deg}` pass, then one pass per tile), so
runtime is comparable — the only change is the per-tile hook does a masked *restoration* toward
the intact value instead of a masked *extra perturbation*.

## Pre-registered checks (not a decision gate yet — Stage 6 is)

- Report the distribution of $\Delta_i$: is it mostly near zero with a heavy tail (a few tiles
  matter a lot), or diffusely spread? This shapes how `Capture@B` should be read in Stage 6.
- Report `dice_deg` vs the checkpoint's known intact Dice (0.8929) as a sanity/context check —
  large degradation cost here would mean $T_s$ pushed the model further out of distribution than
  Stage 2's uniform-application measurement suggested (Stage 2 measured Dice *displacement*
  between predictions, not Dice *against GT* — this is the first time GT enters, so it is also
  the first real check of how costly $T_s$ actually is).

## What this stage does NOT do

- Does not compute Capture@B or ρ(Γ,Δ) yet (Stage 6).
- Does not run competitors (Stage 7).
- Does not scale to 125 subjects until Stage 5.5's pipeline gate passes.

## Output

- `experiments/exp_e12_eggo_m/e180/E180_delta_per_subject.json`, keyed by
  `(sid, window_id, family)`, with `dice_deg`, `dice_cf_i`, `delta_i_global`.
