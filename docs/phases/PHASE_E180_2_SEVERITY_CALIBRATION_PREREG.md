# E180 Stage 2 — Severity calibration: pre-registration

**Date**: 2026-09-16
**Status**: PRE-REGISTERED. Not run.
**Depends on**: Stage 1 (`E180_tile_ledger.json`, transform sanity gates all PASS).

---

## Purpose

Before Γ can be defined, every transform family (T1, T3, T4, T5 — T2 is a control, calibrated
separately and never used to pick the working severity) needs a severity level that produces a
**graded regime**: representation-level change that is measurable in the output, without either
(a) leaving no signal (near-zero output movement at every severity) or (b) catastrophically
destroying every prediction (no distinction between severities). This mirrors E126's precedent
that mild representation perturbation can meaningfully alter downstream behavior while
preserving most of Dice — the working point must sit in that regime, not at either extreme.

This calibration also fixes $T_s$, the **single degradation** used to construct $Z^{deg}$ for
every later Γ/Δ measurement (Stage 5's same-operating-point requirement).

## Method

For each transform family and a small severity grid, on a fixed sample of **10 subjects**
(larger than Stage 1's 5-subject sanity sample, still far short of the full 125 — this is
calibration, not the scientific run):

$$
d_Y = d\big(\hat Y_{\text{intact}}, \hat Y_{\text{perturbed}}\big)
$$

measured as **whole-volume Dice displacement** ($1 - \text{dice\_agree}$, reusing E165's
`dice_agree`) between the intact sliding-window prediction and the prediction with the
transform applied to **every tile uniformly** (not per-tile restoration — that is Stage 5).

### Grids tested

| Transform | Severity grid |
|---|---|
| T1 rank compression | $r \in \{1, 2, 4, 8, 16, 32, 64, 128\}$ (dyadic, E147/E165 grid) |
| T2 permutation (control) | one fixed random permutation, reported for context only |
| T3 local mixing | $\beta \in \{0.05, 0.10, 0.20\}$ |
| T4 spectral reshape | $\gamma \in \{1.5, 2.0, 3.0\}$ and $\gamma \in \{0.5, 0.7\}$ (both concentrating and flattening directions) |
| T5 smoothing | $\sigma \in \{0.5, 1.0, 1.5\}$ |

## Pre-registered decision rule (fixed before running)

For each transform family independently:

| Outcome | Criterion | Action |
|---|---|---|
| **DEAD** | every severity gives $d_Y \leq 0.01$ (near-zero movement) | drop this transform from the Γ family — no signal to measure |
| **SATURATED** | every severity gives $d_Y \geq 0.5$ (catastrophic, prediction unrecognizable) | drop this transform, or restrict to a gentler untested severity if the grid did not reach low enough — report honestly, do not silently pick a point outside the tested grid |
| **GRADED** | at least one severity gives $0.02 \leq d_Y \leq 0.30$, and the family shows monotonic or near-monotonic increase in $d_Y$ with severity | this severity is the candidate working point $T_s$ for that family |

**Program-level rule**: if **zero** transform families reach GRADED, Stage 2 (and therefore
E180) dies here — no controlled degradation regime exists at enc3 for any tested family, and
that is reported as a genuine negative result, not silently patched by expanding the grid
post-hoc.

If at least one family reaches GRADED, its calibrated $T_s$ becomes the fixed degradation used
in Stage 5's $Z \to Z^{deg}$ construction for that family's Γ/Δ test. Families that reach GRADED
independently are each carried forward — Stage 3-4 keep them separate per the transformation-
specific design (no forced single winner at this stage).

## What this stage does NOT do

- Does not compute per-tile Γ yet (that is Stage 3, using the severities fixed here).
- Does not touch ground truth (this is prediction-vs-prediction displacement only, same
  confound guard as E147/E165: never compare against GT here).
- Does not restore/counterfactual anything (Stage 5).

## Output

- `experiments/exp_e12_eggo_m/e180/E180_2_severity_calibration.json`: per transform, per
  severity, $d_Y$ (Dice displacement), plus the GRADED/DEAD/SATURATED verdict and the chosen
  $T_s$ per surviving family.
