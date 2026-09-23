# E183-A — Response geometry audit: pre-registration

**Date**: 2026-09-18
**Status**: PRE-REGISTERED. Not yet run. Scoping only, per `PHASE_E183_RESPONSE_GEOMETRY_SCOPING.md`
and the E182 joint-interpretation closure in `PHASE_E182_ASYMMETRIC_ENERGY_CONTROL_PREREG.md`.

---

## The question, precisely

$$
\boxed{\text{Does the response set } \mathcal{R}(Z_i) = \{D(T_k(Z_i)) - D(Z_i)\}_{k=1}^K
\text{ contain information about } \Delta_i \text{ beyond what } \Gamma_i =
\max_{j,k} d(Y_j, Y_k) \text{ already captures?}}
$$

Not an indiscriminate statistics search. The candidate quantities are derived directly from what
Γ's own definition (a scalar diameter — the single largest pairwise separation) provably cannot
see:

| Γ discards | Candidate geometric quantity |
|---|---|
| Where responses are located (not just how far apart) | Response set centroid / mean displacement magnitude |
| How many independent directions the responses occupy | Effective rank / participation ratio of the response set's covariance |
| Whether perturbations produce coherent (aligned) or scattered motion | Response coherence: mean pairwise cosine similarity between displacement directions |
| Whether different perturbation families agree in direction | Cross-family alignment: cosine similarity between T6's and T7's mean displacement directions, per tile |

Four quantities, each tied to a specific thing Γ discards — not a larger exploratory list.

## Why this requires new (small-scale) compute, stated honestly

Checked before writing this doc: E180/E181/E182's frozen records store only the scalar
`gamma_dice` per tile. The underlying per-transform prediction displacements
$D(T_k(Z_i)) - D(Z_i)$ were computed transiently inside each Γ sliding-window pass but never
persisted. **This is not a pure data-reuse audit** — it requires regenerating those displacements,
this time saving them. Scoped small (10-subject calibration scale, matching every prior
calibration pass in this project) specifically so this stays a cheap mechanistic check, not
another 5000-record campaign, per the explicit instruction to keep this step small.

## Design

### Response vector definition (fixed now)

For tile $i$, transform family $f \in \{T6, T7\}$, at that family's already-locked severities:
the per-transform response is the **whole-volume predicted probability map difference**
$D(T_k(Z_i)) - D(Z_i)$, summarized as a compact per-region vector (ET/TC/WT mean signed
probability displacement within the tile's spatial footprint, matching E180/E181/E182's existing
region-averaging convention) — not the full voxel-grid difference, which would be intractably
large to store and unnecessary for the four candidate quantities above.

### Per-tile response set

For each tile, using the severities already locked (T6: 0.35, 0.50; T7: 0.60, 0.80, 1.10, 1.45),
collect the response vectors $\{r_k\}$, $k = 1 \ldots 6$ (2 from T6 + 4 from T7), each a 3-vector
(ET, TC, WT displacement).

### Four candidate quantities, computed per tile

1. **Centroid magnitude**: $\|\bar r\|$ where $\bar r = \frac{1}{K}\sum_k r_k$.
2. **Effective rank of the response covariance**: entropy-based effective rank (same formula
   used throughout this project, $\exp(H(p))$ on the normalized eigenvalue spectrum of
   $\text{Cov}(\{r_k\})$) — out of a maximum of 3 (only 3 regions).
3. **Response coherence**: mean pairwise cosine similarity $\frac{1}{\binom{K}{2}}\sum_{j<k}
   \cos(r_j, r_k)$.
4. **Cross-family alignment**: cosine similarity between T6's mean response direction
   ($\bar r_{T6} = \frac{1}{2}\sum_{k \in T6} r_k$) and T7's mean response direction
   ($\bar r_{T7} = \frac{1}{4}\sum_{k \in T7} r_k$).

### Δ-pairing (resolved before analysis, not specified in the original design)

The four geometric descriptors are properties of the whole per-tile response set (pooled across
all 6 configs), while Δ is naturally defined per `(tile, transform, severity)`. Resolved:
**$\Delta_i$ = mean of `delta_i_global` across all 6 configs for that tile** — one row per tile,
matching the one response-geometry-descriptor-set per tile. This keeps the analysis at consistent
granularity throughout, at the cost of examining a coarser "mean recoverability" rather than any
single Γ-Δ pair as in E180-E182.

### Analysis

Nested subject-FE regression (reusing `e180_stats.py` verbatim, same discipline as every prior
stage): $\Delta_i \sim \Gamma_i$ (baseline) vs $\Delta_i \sim \Gamma_i + \{$centroid, eff\_rank,
coherence, cross-family alignment$\}$ (extended). Report the incremental $R^2$ of adding the four
geometric quantities, with a within-subject permutation null (same convention as E180 Stage 8 and
E181/E182 Test C), on the small 10-subject sample.

### Pre-registered decision rule

| Outcome | Criterion | Reading |
|---|---|---|
| **Geometry informative** | Incremental $R^2$ over Γ alone is significant (permutation $p < 0.05$) | Response geometry carries information Γ discards — proceed to design a scaled-up geometry audit (125 subjects) before considering any algorithm |
| **Γ sufficient** | Not significant | Γ's scalar diameter already captures what matters for this candidate response-geometry decomposition; investigate other mechanisms, do not force a geometry-based algorithm |

## What this stage does NOT do

- Does not implement "Counterfactual Representation Repair" or any algorithm.
- Does not scale to 125 subjects (contingent on this small-scale result).
- Does not compute an indiscriminate list of geometric statistics beyond the four tied to Γ's
  specific blind spots.
- Does not touch T6/T7's transforms, severities, or Γ/Δ's own definitions — this audits the
  existing phenomenon's data structure, it does not modify the phenomenon.

## Output

- `experiments/exp_e12_eggo_m/e183/E183_A_response_geometry.json`
