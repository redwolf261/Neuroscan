# E181 — Spectral mechanism dissection: hypothesis formulation (no code yet)

**Date**: 2026-09-17
**Status**: HYPOTHESIS FORMULATION ONLY. No transformations implemented, no compute spent.
**Depends on**: E180's frozen result (T4_spectral is the sole surviving family; Stage 10-11
STRONGEST coupling in both strata).

---

## Why this phase exists, and why it must not start with code

E180 established that $\Sigma \to \Sigma^\gamma$ (T4_spectral) produces a real, dose-graded,
counterfactually-coupled Γ→Δ relationship, while T1_rank (pure rank truncation) and T5_smooth
(local smoothing) do not cleanly replicate. But **T4's own construction confounds several
spectral properties at once** — $\Sigma^\gamma$ at fixed total energy simultaneously changes
effective rank, spectral concentration, and the relative weighting of dominant vs. tail
directions. Attributing the phenomenon to "spectral organization" without separating these is
exactly the failure mode this project has hit before (E118's underdetermined-lstsq overfitting,
E153's normalization artifact) — a plausible-sounding mechanism that is actually one of its own
confounded components in disguise.

**The discipline, restated**: formulate competing hypotheses and the minimal experiments that
discriminate between them *before* writing any transformation code. Do not invent a new composite
statistic $\Gamma_{new} = f(\text{rank}, \text{entropy}, \text{energy}, ...)$ — that is the
formula-mining trap that produced nothing durable in E169-E175. The goal is to isolate a
mechanism, not to find a better-fitting scalar.

## Confirmed confound in T4's existing construction

From `run_e180_transform_lab.py:spectral_reshape`: $\Sigma' = \Sigma^\gamma$, rescaled so
$\|\Sigma'\|_2 = \|\Sigma\|_2$ (total energy preserved by construction — H3 is therefore
*partially* controlled already, but never tested as an independent axis on its own). At γ=3 with
energy fixed, raising the spectrum to a power **concentrates energy toward the dominant singular
values** — this mechanically lowers any participation-ratio-style effective-rank measure
(`eff_rank = exp(H(p))`, `part_ratio`) at the same time it increases concentration. H1 (rank) and
H2 (concentration) are **not separated** in the T4 result as measured. This is the central
problem E181 exists to resolve.

## Competing hypotheses

| # | Hypothesis | Claim |
|---|---|---|
| **H1 — Rank (incumbent)** | Γ fundamentally responds to effective-rank limitation | Already heavily investigated (E124-E169b); treated as the default/null explanation to be ruled out, not proven |
| **H2 — Concentration** | Γ responds to the concentration/anisotropy of spectral energy (how unevenly energy is distributed across singular values), independent of rank | Genuinely new relative to the E1-E176 rank-focused search |
| **H3 — Energy** | Γ responds mainly to total representation energy/magnitude, independent of spectral shape | Partially pre-controlled by T4's own rescaling; needs an explicit test as its own axis |
| **H4 — Directional geometry** | Γ responds to how spectral directions ($U$, $V$) are organized relative to the decoder's task-relevant subspace, not to the singular values themselves | The most structurally different hypothesis — asks whether it's the *values* or their *association with directions* that matters |

**H1 is the incumbent, not the target.** The project's own prior investment in rank (E124-E169b,
closed) means a positive H1 result would mean E180's finding is *not* a new phenomenon — it would
be a restatement of the already-explored and already-closed rank story, just discovered via a
different transform family. This must be treated as the null to rule out, not a welcome outcome.

## The central experiment: spectral shape at matched rank

$$
\boxed{\text{Does } R_{\text{eff}}(Z_A) \approx R_{\text{eff}}(Z_B) \text{ with different }
\Sigma_A, \Sigma_B \text{ produce systematically different } \Gamma, \Delta?}
$$

If yes: same rank ≠ same task-relevant spectral organization — a genuine conceptual upgrade over
everything E165-E169b established. If no (Γ, Δ track rank regardless of shape once matched): H1
is confirmed and E180's T4 finding reduces to the already-closed rank story under a new label —
report honestly, do not force a differentiated conclusion.

## The 2×2 decomposition (conceptual, not yet operationalized as code)

$$
\begin{array}{c|cc}
 & \text{same spectral shape} & \text{different spectral shape} \\
\hline
\text{same rank} & A \text{ (no manipulation, control)} & B \\
\text{different rank} & C & D \text{ (closest to T4's actual, uncontrolled behavior)}
\end{array}
$$

Cell B (shape varied, rank matched) is the decisive cell for H1 vs. H2. Cell C (rank varied,
shape matched) is T1_rank's role, already run — serves as the rank-only reference point, not to
be rerun. Cell D is approximately what T4 already did, uncontrolled — establishes that *something*
in that combination works, but does not by itself say which axis.

## Design decisions, locked before any code (resolved 2026-09-17)

1. **"Matched rank" = matched `eff_rank`** (entropy-based, $\exp(H(p))$ over the normalized
   eigenvalue spectrum). This is the exact quantity already computed throughout E169/E180's
   `StageFeatures`, keeping E181 directly comparable to everything already measured, and it is
   continuous/differentiable so a numerical search to match it is tractable. **Not** participation
   ratio (more tail-sensitive, would test a subtly different rank notion) and **not** E165/T1's
   agreement-based $R^*$ (expensive truncation sweep, risks circularity with the closed rank
   story per E165's own confound guard).
2. **H3 (energy) is included in E181's first round**, not deferred. A simple uniform
   singular-value scaling transform ($\Sigma' = c \cdot \Sigma$, $c$ constant, holding normalized
   shape $\Sigma/\|\Sigma\|_2$ exactly fixed) is cheap to build and calibrate with the same
   Stage 2/2b machinery already proven out in E180, and including it now avoids a second
   calibration round later.
3. **H4 (directional geometry) is explicitly deferred to E182**, not attempted in E181. It asks a
   structurally different question (about $U,V$, not $\Sigma$) and is adjacent to T2's already-
   excluded destructive channel-permutation control — designing a non-destructive version needs
   careful thought that is better done once H1-H3 narrow the live hypothesis, rather than risking
   a rushed design that accidentally collapses into T2's already-tested control.
4. **New transformation for cell B** (same rank, different shape): $\Sigma' = \Sigma^\gamma$
   rescaled to hold `eff_rank` fixed (not total energy, unlike T4's existing construction) via a
   numerical search over the rescaling — a genuinely new transform, distinct from T4_spectral.
5. **Probe/severity recalibration is mandatory** for every new transformation (the rank-matched
   shape transform, and the pure-energy-scaling transform) — Stage-2/2b-style graded-regime
   check and fixed-probe calibration, exactly the discipline that caught the idempotency and
   tile-geometry bugs in E180's own Stage 2b/Stage 5.5. Not optional bookkeeping.

## E181's operational experiment set (first round, locked)

| Transform | Tests | Status |
|---|---|---|
| T1_rank (existing) | Cell C: different rank, same shape (rank-only reference) | Already run in E180, reused, not rerun |
| T4_spectral (existing) | Cell D: uncontrolled combination (energy-fixed, rank AND shape both moving) | Already run in E180, reused as context, not the decisive cell |
| **T6 — rank-matched spectral reshape** | Cell B: same `eff_rank` AND same $\|\Sigma\|_2$, different shape — the decisive H1-vs-H2 cell | Designed below, not yet implemented |
| **T7 — pure energy scaling** | H3 in isolation: same normalized shape, same rank (by construction), different total energy | Designed below, not yet implemented |
| Cell A (no manipulation) | Control/baseline, implicit in the existing intact-vs-degraded construction | No new code needed |

## T6 — rank-matched spectral reshape: precise construction

**The trap to avoid**: a naive $\Sigma' = c\Sigma^\gamma$ does not hold `eff_rank` fixed (raising
to a power at any fixed total-energy rescaling still changes the normalized spectral entropy —
this is exactly T4's existing confound). Uniform scaling ($\Sigma'=c\Sigma$) cannot repair the
rank mismatch either, since $p_j' = c\sigma_j/\sum_k c\sigma_k = p_j$ — a positive scalar cancels
in the normalized spectrum, so it never touches entropy. A naive linear interpolation between the
original normalized spectrum $p$ and a target reshaped spectrum $q(\gamma)$ was considered and
**rejected**: for standard entropy, continuous interpolation from $p$ to $q$ generally crosses
the original entropy value only at the trivial point $\lambda=0$, giving no nontrivial solution
in general.

**Adopted construction** — build a same-entropy target spectrum directly, not by interpolating
toward $q(\gamma)$:

$$
p_j = \frac{\sigma_j}{\sum_k \sigma_k} \quad \text{(original normalized spectrum)}
$$

Construct a two-parameter exponential-family target:

$$
q_j(a,b) \propto \exp(-aj - bj^2), \quad j = 1, \ldots, r
$$

Solve numerically for $(a,b)$ such that:

$$
H(q(a,b)) = H(p) \quad \Longrightarrow \quad R_{\text{eff}}(q) = e^{H(q)} = e^{H(p)} = R_{\text{eff}}(p)
\text{ (by construction, exactly)}
$$

subject to **maximizing a shape-distance objective** from $p$ (e.g. $\|q - p\|_1$ or a
divergence measure) so the resulting shape is genuinely different, not a near-copy of $p$ that
merely satisfies the entropy constraint trivially. The shape-distance objective is fixed **before**
running the solver on any tile — a single objective form, applied uniformly, not tuned per-tile
or re-chosen after seeing results.

Reconstruct:

$$
\Sigma'_j = s \cdot q_j, \qquad Z' = U \Sigma' V^\top
$$

where $s$ is chosen so $\|\Sigma'\|_2 = \|\Sigma\|_2$ (energy matched to the original, **not**
to T4's energy convention independently — this keeps H3 from contaminating the T6 result). $U,V$
are unchanged (H4, directional geometry, is out of scope for E181 per the earlier lock).

**Binding constraint on the solver, locked now**: the numerical solver for $(a,b)$ has **no
access to predictions, Dice, Γ, uncertainty, or Δ** — it sees only the tile's own singular-value
spectrum $\sigma$ and the two fixed constraints (entropy match, energy match) plus the fixed
shape-distance objective. It must not be tuned, re-run, or re-parameterized based on any
downstream Γ/Δ outcome. This is a hard requirement, not a preference — violating it would mean
designing the transformation to produce the desired result, invalidating the entire test.

## T7 — pure energy scaling: precise construction

$$
\Sigma' = c \Sigma, \quad c \in \{c_1, c_2, \ldots\} \text{ (a small grid around } c=1 \text{)}
$$

Since $p_j' = c\sigma_j / \sum_k c\sigma_k = p_j$ exactly, $R_{\text{eff}}' = R_{\text{eff}}$ and
the normalized shape is unchanged by construction — no numerical solving needed, this is exact
algebraically. Only $\|\Sigma\|_2$ changes. This is the clean H3 isolation: does representation
magnitude alone move Γ/Δ, independent of rank and shape?

## The resulting causal matrix (target structure after E181)

| Intervention | Rank | Shape | Energy | Direction |
|---|---|---|---|---|
| T1_rank | changes | changes | controlled-ish | same |
| T4_spectral | changes | changes | controlled | same |
| **T6 (rank-matched)** | **same** | **changes** | **same** | same |
| **T7 (energy)** | same | same | **changes** | same |
| H4 / E182 | same-ish | same-ish | controlled | **changes** |

**The critical comparison is T6 vs. T1**: T1 says rank manipulation matters; T6 asks whether
spectral shape matters even when rank is held exactly constant. **T7** asks whether T4's
phenomenon is merely representation magnitude in disguise.

## Sequence, locked

- **E181-A**: implement T6 and T7 (code only, no compute run yet).
- **E181-B**: calibration discipline, extended beyond Stage 2/2b's original checklist —
  identity exactness; severity curve; non-degeneracy; no NaN/Inf; **representation norm**
  (verify T7 actually changes it and T6 does not); **effective rank** (verify T6 actually holds
  it within tolerance and T1/T4 do not); **spectral-shape distance** (verify T6 actually differs
  from the original shape, not a near-trivial solution); no catastrophic prediction collapse.
  **Do not proceed to Γ/Δ measurement until T6's rank-matching is verified within tolerance on a
  calibration sample** — this is the check that prevents accidentally rediscovering T4/T1 under
  a new name.
- **E181-C**: small fresh-subject pilot (mirroring E180's Stage 5.5 discipline).
- **E181-D**: only after calibration passes, the full diagnostic on the locked design.

## E181-C result (recorded 2026-09-17, binding on E181-D)

**PASS — fresh-subject engineering validation; mechanistic discrimination unresolved.**

Locked configuration used: T6 budgets {0.35, 0.50}; T7 levels {0.70, 1.00, 1.30} (1.00 =
identity control). On the same 5 fresh subjects as E180's Stage 5.5 (200 tile×config records,
zero NaN/Inf): T6 showed 100% solver convergence, 100% U,V preservation, rank/energy error at
machine precision (~1e-12 to 1e-16) — the engineering isolation (T6: shape changes, rank/energy/
U/V fixed; T7: energy changes, rank/shape fixed) survives on fresh data.

**Exploratory pooled Spearman (descriptive only, not evidence)**: T6@0.50 ρ=+0.481 (p=0.0017),
T7@1.30 ρ=+0.577 (p=0.0001), both near-zero at their gentler severity. This does **not**
discriminate H2 from H3 — both transforms show a qualitatively similar pattern (signal at
higher severity, absent at lower), and T7's Δ range is substantially larger than T6's at
comparable severity indices, raising a **severity-amplitude confound**: T6 budget=0.50 and T7
c=1.30 are different physical quantities and must not be treated as equivalent "doses" when
compared.

**Explicitly NOT concluded** (per instruction, stated to prevent this from being
mis-read later): not "H2 confirmed," not "H3 confirmed," not "T6 beats T7." The pilot
demonstrated both interventions are viable at scale; discriminating between them requires the
full subject-aware analysis (E181-D), not another pilot-scale exploratory pass.

## E181-D — full 125-subject mechanism test (locked design, addressing the magnitude confound)

**Frozen, not re-tuned from the pilot**: T6 $b \in \{0.35, 0.50\}$; T7 $c \in \{0.70, 1.30\}$.
**$c=1.0$ is excluded from the Γ→Δ correlation cells** (Δ is identically zero there by
construction — a degenerate, undefined correlation, not informative as a data point; it remains
the identity sanity check only).

### The magnitude-confound fix

Do not compare T6 and T7 at raw severity labels as if "0.50 shape distance ≡ 30% energy change."
Every E181-D analysis includes the **actual per-tile perturbation magnitude**
($\|Z^{deg}_i - Z_i\|$, already computed as `activation_rel_diff` in the E181-B/C hooks) as an
explicit covariate, not an assumed-equivalent severity label:

$$
\Delta \sim \Gamma + \text{perturbation magnitude} + \text{transform family} + \text{subject effects}
$$

### Three tests, all subject-aware (E180 Stage 6-9 framework, cluster-robust by subject)

- **Test A** — Does T6 reproduce E180's Γ→Δ relationship (subject-FE regression, per family/
  severity, both strata)? If yes: **H2 survives**.
- **Test B** — Does T7 reproduce it? If yes: **H3 survives too** — informative either way, since
  a positive result here would mean the phenomenon is not exclusively spectral-shape-specific.
- **Test C (decisive)** — Conditional on perturbation magnitude: does transform family (T6 vs
  T7) still predict Δ after controlling for Γ, magnitude, and subject effects? And separately:
  does T6's Γ→Δ relationship survive magnitude control while T7's does not (or vice versa, or
  neither)? This is the comparison that actually discriminates H2 from H3, not a raw-severity
  side-by-side.

### H1 (rank) — still the incumbent, still what T6 directly tests against

T6's entire scientific value rests on $R_{\text{eff}}' = R_{\text{eff}}$ holding exactly (already
verified to ~1e-12). If T6 reproduces the E180 phenomenon under full subject-aware analysis, that
is a direct separation from the rank explanation — arguably E181's single most important result,
independent of how the T6-vs-T7 comparison resolves.

### Locked discipline

No new transform is invented to resolve the T6-vs-T7 ambiguity before running E181-D. No budget
or severity is re-tuned based on the pilot's exploratory numbers. E181-D runs the frozen design
and reports Test A/B/C mechanically, per the same "no rescuing a weak result" discipline used
throughout E180.

## E181-D data collection result (recorded 2026-09-18) and frozen manifest

Complete and clean: 3472 records (125 subjects × 4 configs × ~7 tiles avg), zero NaN/Inf in
`gamma_dice`, `delta_i_global`, `uniform_magnitude`. T6 diagnostics hold at full scale: 95.4%
solver convergence at budget=0.35, 100% at budget=0.50, **100% U,V preservation** at both,
rank/energy relative error at machine precision (~1e-10 to 1e-16). Frozen to
`experiments/exp_e12_eggo_m/e181/FROZEN/MANIFEST.json` (sha256_16-checksummed) before any
inferential analysis — the transforms, severity budgets, tile selection, and inclusion criteria
are not modified after this point regardless of what Test A/B/C show.

## Test A/B/C — pre-specified analysis (fixed BEFORE running, per explicit instruction)

**Experimental unit for inference is the subject, not the tile.** 3472 tile records are not
3472 independent observations (per the tile-overlap finding established in E180). Every test
below uses a subject-aware framework (subject fixed effects, cluster-robust by subject, reusing
`e180/e180_stats.py`'s `subject_fixed_effects_ols`/`within_subject_permutation_null`) — pooled
tile-level statistics are reported only as explicitly-labelled descriptive context, never as the
primary evidence.

**T7 c=1.0 is excluded from all correlation tests** (Δ≡0 by construction there, already excluded
from data collection). **T6 budgets and T7 levels are never compared by their raw labels** — Test
C conditions explicitly on the measured `uniform_magnitude` covariate, since "0.50 shape distance"
and "1.30 energy scale" are not commensurable doses.

- **Test A** — Does T6 (shape-only) reproduce E180's Γ→Δ relationship? Subject-FE regression of
  `delta_i_global ~ gamma_dice` separately for T6@0.35 and T6@0.50, reporting β, cluster-robust
  SE, p, plus the pooled Spearman as labelled-descriptive context (mirroring E180 Stage 6's
  format exactly).
- **Test B** — Same framework, T7@0.70 and T7@1.30.
- **Test C (decisive)** — Magnitude-conditional comparison: subject-FE regression of
  `delta_i_global ~ gamma_dice + uniform_magnitude + transform_family` (transform_family as a
  0/1 indicator, T6 vs T7, pooling both severities within each family), reporting whether
  transform family still predicts Δ after controlling for Γ and actual perturbation magnitude.
  Additionally, report Γ's own coefficient **separately within each family** after magnitude is
  in the model, to see whether T6's Γ→Δ relationship survives magnitude control while T7's does
  not (or vice versa, or neither survives).

### Reading the result (fixed now, not re-interpreted after seeing numbers)

- **T6 survives, T7 does not survive magnitude control** → supports H2 specifically (spectral
  shape carries information beyond rank, energy, *and* generic perturbation magnitude) — the
  single most informative possible outcome.
- **Both T6 and T7 survive** → the phenomenon is broader than spectral shape; do not force H2,
  report honestly that shape is not uniquely implicated.
- **Neither survives Test C** (even if A/B showed raw associations) → the raw Γ→Δ association in
  both families may be substantially explained by generic perturbation magnitude, not by Γ's
  specific construction.

### Explicit epistemic limits (stated before running, carried into the report regardless of
outcome)

The counterfactual restoration (Δ's construction: mask-restore tile i, measure whole-volume Dice
change) is causal evidence about the tested transformation/restoration mechanism specifically.
**The statistical Γ→Δ relationship itself (Test A/B/C) is associational, not causal proof** — Γ
is measured, not independently randomized, so Test A/B/C establish whether instability predicts
restoration benefit, not that instability *causes* it in a fully identified sense. This
distinction is stated in every report of these results, not just here.

### Retained fields (frozen manifest, full precision, never collapsed)

`sid`, `window_id`, `transform`, `severity`, `uniform_magnitude`, `gamma_dice`, `dice_deg`,
`dice_cf_i`, `delta_i_global`, and for T6 records: `requested_budget`, `achieved_shape_distance`,
`converged`, `eff_rank_orig`, `eff_rank_new`, `rel_eff_rank_error`, `energy_orig`, `energy_new`,
`rel_energy_error`, `UV_preserved_pass`, `UV_reconstruction_max_diff`.

## Output

- `experiments/exp_e12_eggo_m/e181/E181_test_A_B.json`
- `experiments/exp_e12_eggo_m/e181/E181_test_C.json`

## Test A/B/C result (recorded 2026-09-18)

**Test A (T6, shape-only)**: SURVIVES the subject-aware primary test at both severities —
budget=0.35: β=+0.353, p=0.0053; budget=0.50: β=+1.221, p=0.0400. n=868 tiles, 125 subjects,
each cell.

**Test B (T7, energy-only)**: mixed — c=0.70: β=+0.430, p=0.0061 (survives); c=1.30: β=+0.009,
p=0.567 (does not survive). Same n.

**Test C (decisive, pooled)**: `Delta ~ Gamma + magnitude + transform_family`, subject-FE,
n=3472/125. $\beta_\Gamma=+0.325$ (p=0.014), $\beta_{\text{magnitude}}=+0.015$ (p=0.140, n.s.),
transform-family indicator significant (subject-permutation $p < 0.001$). $R^2_{\text{within}} =
0.244$.

**Test C (within-family, Γ over magnitude control)**:

| Family | Γ-alone β | Γ-alone p | Γ-over-magnitude perm p | Survives? |
|---|---|---|---|---|
| T6 | +0.824 | 0.0020 | <0.001 | **Yes** |
| T7 | +0.182 | 0.101 (n.s. even without magnitude control) | <0.001 | Yes, but see caveat below |

**Mechanical classification per the pre-registered rule**: `BOTH_SURVIVE` — phenomenon broader
than spectral shape, do not force H2.

### Critical caveat on T7's magnitude control — found after running, reported honestly

T7's `uniform_magnitude` at c=0.70 and c=1.30 is **nearly identical across every subject**
(means 0.2599 vs 0.2599, std≈0.008) — an algebraic consequence of `pure_energy_scaling`'s
relative-magnitude formula $\|cZ-Z\|/\|Z\| = |c-1|$ being exactly symmetric around $c=1$
($|0.7-1| = |1.3-1| = 0.3$). This means **T7's within-family magnitude control has almost no
magnitude variance to condition on** — confirmed by the pooled model's absurd
`magnitude_alone_beta = -385.6` for T7 (vs a sane +0.003 for T6), a degenerate-regression
symptom of near-collinearity, not a real effect. T6's magnitude, by contrast, varies
meaningfully across its two severities (means 0.347 vs 0.513, real spread) — T6's magnitude
control is well-identified; T7's is not.

**Consequence for reading the result**: T6's `SURVIVES_MAGNITUDE_CONTROL=True` is trustworthy.
T7's `SURVIVES_MAGNITUDE_CONTROL=True` rests on a magnitude covariate with essentially no
within-family variance, so the "control" barely constrains anything — T7's result should be read
as **"Γ predicts Δ within T7, largely unconditioned on magnitude in practice"**, not as genuine
evidence that T7's relationship survives a meaningful magnitude test. Also note T7's raw
`gamma_alone_p = 0.101` is not even significant on its own (consistent with Test B's c=1.30
result) — the "survival" at `p<0.001` in the magnitude-conditional model is driven by pooling
both T7 severities together (n=1736), not by a robust per-severity effect.

**Root cause, precisely stated (not merely underpowered — structurally incapable)**: under
$\|cZ - Z\|/\|Z\| = |c-1|$, the two chosen T7 levels are equidistant from identity
($|0.7-1|=|1.3-1|=0.3$), so the conditioning variable Test C needed (actual perturbation
magnitude) could not distinguish the two T7 conditions **by construction**, independent of
sample size. This is an experimental-design limitation in the T7 severity choice, not a
statistical-power failure and not a scientific failure — caught and reported rather than
interpreted through the mechanical `BOTH_SURVIVE` output.

### Final locked interpretation (calibrated correctly, not over- or under-claimed)

**What E181 has established, precisely:**

| Question | Result | Status |
|---|---|---|
| Does T6 shape perturbation reproduce E180's Γ→Δ relationship? | β=.353 p=.0053 (0.35); β=1.221 p=.0400 (0.50) | **PASS** |
| Survives while rank exactly fixed? | Yes (rel. error ~1e-12) | **PASS** |
| Survives while energy exactly fixed? | Yes (rel. error ~1e-16) | **PASS** |
| Survives while U,V exactly fixed? | Yes (100% by construction, verified) | **PASS** |
| Is T6's evidence merely a rank effect? | Directly argues against | **H1 weakened substantially** |
| Does T7 establish energy as the mechanism? | c=0.70 significant, c=1.30 null | **UNRESOLVED** (not rejected) |
| Can T6 vs T7 currently discriminate H2 vs H3? | No — T7 magnitude conditioning is structurally degenerate | **UNRESOLVED** |
| Is Γ→Δ itself causal? | No | **Associational** |

**The strongest part is the experimental isolation, not the p-values:**

$$
\boxed{\text{rank fixed} + \text{energy fixed} + \text{singular directions } U,V \text{ fixed}
\;\Rightarrow\; \text{spectral shape changes still matter}}
$$

**Correctly calibrated claim (adopted wording, do not overstate beyond this):**

> Spectral-shape perturbations produce a reproducible, subject-aware association between
> representation instability and recoverable segmentation benefit, even when effective rank,
> total energy, and singular-vector geometry are held fixed.

**Explicitly NOT established**: "spectral shape is the unique mechanism" (H3 has not been tested
symmetrically — T7's c=0.70 result means energy cannot be said to "do nothing" either; the
correct state is H3 **unresolved**, not rejected). Not causal proof of Γ→Δ (associational
throughout — the counterfactual restoration is causal evidence about the tested
transformation/restoration mechanism; the Γ→Δ statistical relationship itself is not
independently randomized).

## Next phase: E182 — asymmetric energy-control dissection (scoped, not yet implemented)

Purpose: give H3 the same rigor H2 received, by fixing T7's structural magnitude-degeneracy.
**Not** a new giant experiment — one surgical fix to T7's severity design, everything else
(T6, the subject-aware framework, the tile ledger, the probe) stays frozen.

**Design change**: replace the symmetric $c \in \{0.70, 1.30\}$ pair with an asymmetric,
genuinely magnitude-varying set, e.g. $c \in \{0.70, 0.85, 1.15, 1.30\}$ or a one-sided ladder —
exact values to be calibrated (Stage-2/2b-style graded-regime check) before locking, same
discipline as every prior severity choice in this project. The binding requirement is
$\text{uniform\_magnitude}(c_1) \neq \text{uniform\_magnitude}(c_2)$ for every non-identity pair,
verified numerically before any Γ/Δ compute is spent — exactly the kind of check that would have
caught the current design's degeneracy earlier.

**Decisive question E182 answers**: does T6 retain its Γ→Δ relationship after matching/
conditioning on magnitude, while T7 does not? If yes, evidence for spectral organization over
generic representation displacement strengthens substantially. If both survive under a properly
magnitude-varying design, that is also a real result — it would mean Γ measures susceptibility to
representation perturbation broadly, not spectral organization specifically — a different, but
still scientifically valuable, finding.

**What comes after E182, not before**: the question shifts from "can we get significance" to
$\boxed{\text{what computational operation should a segmentation algorithm perform because of
this phenomenon?}}$ — that is E183's scope (principle derivation), not something to pursue by
adding more statistical controls to the current result. E181/E182 remain mechanism discovery;
they do not yet license method design.

## What E181 does NOT do yet

- Does not implement any transformation.
- Does not run any compute.
- Does not touch T1_rank or T5_smooth (both settled per E180's terminal status).
- Does not propose an algorithm (E183+ per the locked roadmap, contingent on E181-E182).

## Locked roadmap (restated, binding)

$$
E181 \text{ (mechanism dissection)} \to E182 \text{ (causal isolation)} \to
E183 \text{ (principle derivation)} \to E184 \text{ (prior-art attack)} \to
E185 \text{ (algorithm)} \to E186 \text{ (cheap gate} \to \text{3-seed training)}
$$

E181's deliverable is a decision about which hypothesis (H1-H4) survives the matched-rank
experiment — not a transformation implementation plan. The next step, once H1-H4 are properly
formulated and the operational questions above are resolved (ideally in discussion, not
unilaterally), is to design the **minimal** set of new transformations needed to populate cells
A/B/C of the 2×2 matrix (C already exists via T1_rank) — and only then write code.
