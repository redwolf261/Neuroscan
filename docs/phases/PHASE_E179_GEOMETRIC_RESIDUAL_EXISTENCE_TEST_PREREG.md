# E179 — Geometric residual existence test (pre-registration)

**Date**: 2026-09-19
**Status**: PRE-REGISTERED. Not yet implemented. Follows the clean E179 prior-art audit
(`PHASE_E179_NUMERICAL_RESIDUAL_PRIOR_ART_AUDIT.md`, 🟢 not found — licenses testing, not a
novelty claim). Purely diagnostic: no training, no new loss, no correction network, no
architecture change, one frozen checkpoint (`E131_v5control_seed0`).

## The quantity under test

$$
\mathcal{R}(x) = \frac{\|\nabla p(x)\|}{|p(x) - \tau| + \epsilon}, \qquad \tau = 0.5 \text{ (the
pipeline's own existing threshold convention, reused verbatim from E167 onward)}
$$

$\|\nabla p(x)\|$: central-difference 3D gradient magnitude of the continuous probability
volume (standard `np.gradient`-style discrete gradient, no smoothing kernel — kept
hyperparameter-free, matching the diagnostic-only spirit of this test). $\epsilon$: small
numerical stabilizer, fixed before running (e.g. $10^{-3}$, chosen to avoid blow-up at
$p=\tau$ exactly, not tuned against any result).

Computed separately per region — **ET, TC, WT** (the pipeline's own fixed `REGIONS` order),
never collapsed into one binary residual. ET is explicitly flagged as the most informative
region given E167's own prior finding (98.6% of ET error within 3 voxels of the GT boundary).

## The trap, stated precisely, and how it's closed

$|p(x)-\tau|$ in the denominator means $\mathcal{R}$ is *mathematically* elevated near the
model's own predicted boundary — finding "high $\mathcal{R}$ → near predicted boundary" would
be a tautology, not a discovery. **Every test below is conditioned on, or directly compared
against, predicted-boundary distance** $d_{\text{pred}}(x) = \text{dist}(x, \partial\hat Y)$,
reusing E167's exact `boundary_distance()` function (interface-to-voxel Euclidean distance
transform) applied to the *predicted* mask instead of GT. This is the load-bearing control of
the entire experiment.

## Test 1 — residual/error relationship

Per subject, per region: compute $p(x)$ (frozen sliding-window inference, unchanged from
E165/E167's own convention), $\mathcal{R}(x)$, and $d_{\text{GT}}(x)$ (E167's `boundary_distance`
applied to GT). Bin voxels by $\mathcal{R}$ decile; report median $d_{\text{GT}}$ and actual
error rate (FP+FN, matching E167's own `err` definition) per bin.

**Population restriction, found necessary and corrected during implementation, before any real
result was computed**: $\mathcal{R}(x)$ computed over the *entire* volume is degenerate — 99.6%
of voxels sit in confident background ($p\approx0$), giving $\mathcal{R}\approx0$ there and
collapsing whole-volume deciles into 1-2 uninformative bins (measured directly on a smoke-test
subject before trusting anything). **Fixed**: Test 1 uses the same predicted-boundary-proximal
candidate population ($d_{\text{pred}}\le 3$) that Tests 2 and 3 already use — consistent across
all three tests, not a post-hoc adjustment chosen to flatter a result.

**Gate 1 (nontriviality)**: a real, monotonic trend — $\mathcal{R}\uparrow \Rightarrow
d_{\text{GT}}\downarrow$ and/or elevated error in high-$\mathcal{R}$ bins. Flat curve → KILL.

## Test 2 — incremental information beyond boundary proximity alone

Three predictors of actual GT error, compared **conditional on $d_{\text{pred}}$** (narrow bins,
matching E167's `STRATA=[1,2,3]` convention):

- $S_1(x) = 1/(|p(x)-\tau|+\epsilon)$ — boundary-proximity-to-own-prediction only
- $S_2(x) = \|\nabla p(x)\|$ — gradient magnitude only
- $S_3(x) = \mathcal{R}(x)$ — the proposed residual

**Gate 2 (incremental information)**: $S_3$ must outperform $S_1$ **within matched
$d_{\text{pred}}$ bins**, not just in a pooled/unconditioned comparison (which would trivially
favor $S_3$ or $S_1$ alike, since both are dominated by the same denominator near the boundary).
If $S_3 \not> S_1$ conditionally → KILL. This is the test that directly closes the trap above.

## Test 3 — recoverability (the causal question)

**Sampling unit**: individual voxels drawn from the predicted-boundary region only
($d_{\text{pred}}(x) \le 3$, matching E167's own boundary-concentration finding and `STRATA`
convention — sampling outside this region would mostly test interior voxels where
$\mathcal{R}$'s denominator is uninformative by construction). Stratify candidates by
$\mathcal{R}$ decile **within each $d_{\text{pred}}$ bin** (0/1/2/3), so the comparison is always
matched-boundary-distance, never confounded by $d_{\text{pred}}$ itself.

**Oracle local correction** (GT used only for measurement, never for candidate selection —
selection is by $\mathcal{R}$ decile alone): for a sampled voxel $i$, set the predicted label to
the GT label within a small fixed-radius ball ($r=2$ voxels, matching E167's own stratum
convention) centered at $i$, nowhere else. $\Delta\text{Dice}_i = \text{Dice}(\text{corrected}) -
\text{Dice}(\text{baseline})$, whole-volume Dice, same convention as every prior stage in this
project.

**Gate 3 (recoverability)**: high-$\mathcal{R}$ decile candidates must show systematically
larger $\Delta\text{Dice}_i$ than low-$\mathcal{R}$ decile candidates, **within the same
$d_{\text{pred}}$ bin**. If the relationship is flat or reversed → KILL. This is the bridge from
"diagnostic correlate" to "candidate computational principle" — the only test in this design
that touches GT for anything beyond final measurement.

## Test 4 — subject-level robustness (mandatory, not optional)

**Every correlation/comparison above is computed separately per subject first**, not pooled
across all voxels from all 125 subjects into one number — pooling would inflate apparent
significance via the same tile/voxel non-independence artifact E180's own subject-fixed-effects
correction was built to fix. Report the distribution of per-subject effect sizes (Spearman ρ for
Tests 1/2, mean $\Delta\text{Dice}$ gap between top/bottom $\mathcal{R}$ deciles for Test 3).

**Gate 4 (robustness)**: the effect must be consistently signed across a strong majority of
subjects (locked threshold: **≥80%** same sign), not merely significant when pooled. A result
that only survives pooling is treated as **KILL**, not as a weaker positive.

## Gate order and kill discipline (locked, per explicit instruction)

Gate 1 → Gate 2 → Gate 3 → Gate 4, evaluated per region (ET/TC/WT separately — a pass in one
region does not rescue a failure in another; each region's verdict is reported on its own
terms). **Any single gate failing in a region kills that region's branch outright** — no
softening, no reframing a flat Gate 2 as "still interesting," no partial-credit narrative. If
all three regions fail at any gate, **E179 is killed completely**: no residual loss, no residual
attention, no PDE block, no further experiment in this direction.

## Result (2026-09-19) — KILLED at Gate 2, decisive on a 10-subject sample

Ran on 10 subjects (not the full 125) — see the scope note below for why stopping here is
honest, not a silent reduction.

**Gate 1 (nontriviality)**: passed cleanly on inspection. After fixing the population-restriction
bug (see above), $\mathcal{R}$ deciles show the expected monotonic trend — higher decile → lower
median $d_{\text{GT}}$, higher error rate (e.g. subject `01041`/ET: decile 1 → median
$d_{\text{GT}}=2.45$, error rate 0.0003; decile 9 → median $d_{\text{GT}}=1.0$, error rate
0.327). Not a flat curve.

**Gate 2 (incremental information beyond boundary proximity) — FAILED, decisively.** Across 10
subjects × 3 regions × up to 4 $d_{\text{pred}}$ bins each (118 total comparisons): $S_3$ (the
proposed residual) beat $S_1$ (plain boundary-proximity, $1/(|p-\tau|+\epsilon)$) in only **26/118
bins (22%)** — the reverse of what Gate 2 requires. Subject-level check (Gate 4's own method,
applied here since the pattern was already clear): mean fraction of a subject's own bins where
$S_3>S_1$ was 0.175 (ET), 0.225 (TC), 0.250 (WT); only 1-2 of 10 subjects per region showed $S_3$
winning a *majority* of their own bins, nowhere close to the pre-registered 80% robustness
threshold and below even a 50% bar.

**This is the literal trap the design was built to test, and it is not avoided.** The gradient
term in $\mathcal{R}$'s numerator does not add information beyond what boundary-proximity to the
model's own prediction already captures — if anything, $S_1$ alone is the stronger single
predictor of error in most matched-distance bins, and $S_2$ (gradient alone) tracks $S_3$ almost
exactly in bins 2-3 (visible directly in the raw numbers — $S_2\approx S_3$ once
$|p-\tau|+\epsilon$ stops varying much within a distance bin), suggesting $\mathcal{R}$'s ratio
construction is not doing meaningful work beyond what either term does alone.

**Per the pre-registered gate order and kill discipline: Gate 2 failing kills the region's
branch outright, no softening.** This holds for all three regions (ET, TC, WT) — none showed
$S_3$ clearing the bar. **E179 is killed** — no residual loss, no residual attention, no PDE
block.

**Scope note, stated explicitly rather than silently reduced**: this verdict is based on 10 of
125 subjects, not the full pre-registered cohort. The pattern (118 comparisons, consistent
direction across all 3 regions, subject-level fractions all well below both the 50% and 80%
bars) is strong enough that running the remaining 115 subjects is very unlikely to reverse the
Gate 2 verdict — but this is a judgment call to stop early, not a completed full-cohort result,
and is flagged as such rather than presented as if the full run had been executed. Test 3
(recoverability) was not run to completion on this sample — moot once Gate 2 fails, per the
locked gate order (Gate 2 → Gate 3), and not worth the additional GPU cost to compute anyway.

## What this does NOT do

No training. No architecture change. No new checkpoint. No correction network — Test 3's
"correction" is a GT-oracle diagnostic probe, not a proposed method, and is explicitly not
reused as one regardless of outcome. GT enters **only** in Test 1 (as the comparison target,
already the norm throughout this project) and Test 3 (as the oracle correction's construction,
explicitly never used to select which voxels are tested). Does not claim novelty — E179's prior-
art audit licensed testing the phenomenon's existence, not any conclusion about it.
