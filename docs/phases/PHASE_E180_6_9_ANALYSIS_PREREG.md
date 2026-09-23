# E180 Stages 6-9 — primary test, competitor ladder, incremental information, boundary control

**Date**: 2026-09-17
**Status**: PRE-REGISTERED. Not run. Fixed before seeing any Stage 6-9 statistical result.
**Depends on**: full 125-subject Γ (`E180_gamma_per_subject_full.json`, 2604 records, zero NaN)
and Δ (`E180_delta_per_subject_full.json`, 2604 records, zero NaN), both clean.

---

## The central statistical hazard, stated up front

868 tiles across 125 subjects are **not** 868 independent observations. Tiles within a subject
share 83-94% of their enc3 extent (measured directly, `max_neighbor_enc3_iou` in the tile
ledger) and inherit the same subject-level degradation severity (`dice_deg`). Every stage below
is designed around this constraint. **The pooled tile-level Spearman/R² is never reported as the
decisive statistic** — it is descriptive context only, explicitly labelled as such.

## Two strata, identical specification, no separate tuning

- **A — Full (primary)**: all 125 subjects.
- **B — E167-good (secondary/robustness)**: the predefined 110-subject set
  (`e180_strata.e167_good_subject_ids()`, reused verbatim from E167/E160, never redefined or
  reselected from E180 data).

The identical model specification runs on both. B is never used to rescue a null result on A,
and neither is used to strengthen the other post-hoc.

---

## Stage 6 — primary test: does Γ predict Δ?

### Primary (headline result)

**Subject fixed-effects regression** (subject dummies absorb between-subject difficulty):

$$
\Delta_{si} = \beta_0 + \beta_\Gamma \Gamma_{si} + u_s + \epsilon_{si}
$$

Implemented as within-subject demeaning (equivalent to dummy-variable OLS, already used in the
Stage 5.5 diagnostics): $\Delta_{si} - \bar\Delta_s \sim \Gamma_{si} - \bar\Gamma_s$. Reports
$\beta_\Gamma$, its cluster-robust standard error (clustered by subject), and $p$. This is the
headline number, not the pooled Spearman.

**Question answered**: within the same patient, do tiles with more representation-induced
instability also show more recoverable restoration benefit?

### Secondary (reported, explicitly labelled non-independent / descriptive)

- Pooled tile-level Spearman ρ(Γ,Δ) — labelled **descriptive, non-independent, not evidence**.
- Reduced-granularity robustness (max enc3-center-distance tile pair per subject, the fixed
  criterion locked in Stage 1's amendment).
- Within-subject sign concordance + exact binomial test (per Stage 5.5's lock).

### Per-family reporting

T1_rank, T4_spectral, T5_smooth reported separately throughout — never silently pooled into one
aggregate Γ without also reporting the per-family breakdown (consistent with the T5_smooth-null
finding from Stage 5.5, which must survive into this analysis, not be smoothed over).

---

## Stage 7 — competitor ladder

Ladder, fixed order, each evaluated under the **same subject-aware framework** as Stage 6
(subject fixed-effects regression, not pooled correlation):

$$
\text{Magnitude} \to \text{Perturbation magnitude} \to \text{Uncertainty} \to
\text{VJP sensitivity} \to \Gamma
$$

| Level | Signal | Source |
|---|---|---|
| 1 | $\|Z_i\|$ magnitude | mean absolute enc3 activation per tile (new, cheap, computed from a fresh intact forward pass since `StageFeatures` was not run as part of Γ/Δ's hook-only pipeline) |
| 2 | $\|Z_i - T_s(Z_i)\|$ perturbation magnitude | trivial, computable from the already-stored Γ construction (norm of the uniform degradation's effect on tile $i$'s own activation) |
| 3 | **Uncertainty** $H(p_i)$ | predictive entropy of the region-averaged foreground probability within tile $i$'s spatial extent, from the intact prediction. **Elevated per the novelty audit** — this is the literature-matched TRUST/CertainTTA-style signal any future novelty claim must be shown not to reduce to. |
| 4 | VJP decoder sensitivity $\|\partial \hat Y/\partial Z_i\|$ | single VJP per tile through the enc3→output path, following E172's proven feasibility pattern (~10s/subject, ~4.2GB) |
| 5 | $\Gamma_i$ | Stage 3-4, already computed |

Each competitor's standalone predictive power for Δ is reported with the subject fixed-effects
$\beta$/$p$, not pooled correlation — same standard as Stage 6.

---

## Stage 8 — the decisive gate: incremental information

Nested models, subject-clustered throughout:

$$
M_0 = \text{intercept} + \text{subject FE (basic controls)}
$$
$$
M_1 = M_0 + U \quad (\text{uncertainty})
$$
$$
M_2 = M_1 + S \quad (\text{magnitude, perturbation magnitude, VJP sensitivity})
$$
$$
M_3 = M_2 + \Gamma
$$

**Decisive quantity**:

$$
\boxed{\Delta R^2_\Gamma = R^2(M_3) - R^2(M_2)}
$$

reported **specifically** as $\Gamma$ over $M_1$ (uncertainty alone) as its own named quantity —
$\Delta R^2_{\Gamma \mid U}$ — not just the pooled increment after all competitors, per the Stage
7 addendum. This isolates the exact comparison the novelty audit's narrow claim depends on: if Γ
adds nothing beyond uncertainty specifically, E180's signal is most likely a re-derivation of the
existing entropy-based TTA-selection literature (TRUST, CertainTTA), independent of whatever
$M_2 \to M_3$ shows.

**Permutation null respects subject structure.** Never shuffle 2604 tile rows as IID. The null
shuffles $\Gamma_{si}$ **within subject** (permute tile labels for a fixed subject, preserving
the subject-level marginal), analogous to E169's within-fold permutation convention. 1000
permutations, report $p$ = fraction of null $\Delta R^2$ exceeding observed.

**Pre-registered decision rule**:

| Outcome | Criterion | Verdict |
|---|---|---|
| KILL | $\Delta R^2_{\Gamma\mid U}$ not significant (permutation $p > 0.05$) on stratum A | Γ is redundant with uncertainty; E180 dies as a predictive mechanism regardless of Stage 9 |
| WEAK | $\Delta R^2_{\Gamma\mid U}$ significant but $\Delta R^2_\Gamma$ (over full $M_2$) is not | interesting diagnostic, weak basis for a new computational principle (classification B in the decision tree) |
| SURVIVES | both significant on stratum A, checked against stratum B | proceed to Stage 9 |

---

## Stage 9 — boundary control

Reuses E167's `boundary_distance` + enrichment machinery **verbatim** — no redefinition for
E180. Tests $\Delta \sim \Gamma + \text{boundary distance}$ (subject fixed-effects, same
framework), reporting Γ's incremental contribution after the boundary control, analogous to
Stage 8's structure. If Γ's contribution collapses once boundary proximity is controlled, that is
read as classification C (likely a boundary phenomenon), not explained away.

---

## Decision tree (fixed now, read off mechanically after running, not re-litigated)

| Classification | Criterion |
|---|---|
| **A — Kill** | Γ fails the Stage 6 subject-aware primary association |
| **B — Weak diagnostic** | Γ predicts Δ but Stage 8 shows uncertainty explains it (KILL/WEAK verdict above) |
| **C — Boundary phenomenon** | Γ survives uncertainty (Stage 8 SURVIVES) but not the Stage 9 boundary control |
| **D — Strong mechanism candidate** | Γ survives subject effects + uncertainty + sensitivity + boundary |
| **E — Computational-demand signal** | D, plus Stage 10-11's dose-response/monotonicity (out of scope for this run) |

**No optimization of the analysis to rescue a weak result.** If Γ dies at any gate, it is
reported as dead at that gate — later stages are not run to search for a rescuing framing.
Stages 10-11 (out of scope here) are earned only by classification D or better.

## Output

- `experiments/exp_e12_eggo_m/e180/E180_6_primary_test.json`
- `experiments/exp_e12_eggo_m/e180/E180_7_competitors.json`
- `experiments/exp_e12_eggo_m/e180/E180_8_incremental_r2.json`
- `experiments/exp_e12_eggo_m/e180/E180_9_boundary_control.json`
- Each stratified A (full 125) / B (E167 110), each per-family where applicable.
