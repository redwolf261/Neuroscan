# E184 (pre-implementation) — Repair operator derivation: math first, no GPU

**Date**: 2026-09-18
**Status**: DERIVATION ONLY. No transforms, no compute, no feasibility experiment yet. This
document is the mathematical design pass that must precede any E184 GPU work — the explicit
discipline requested: derive the smallest possible repair operator and what success/failure
means for it *before* touching the GPU.

---

## Trajectory closure (recorded before deriving anything new)

$$
\text{Rank (E181 H1)} \to \text{insufficient} \quad
\text{Capacity (pre-E180)} \to \text{insufficient} \quad
\text{Spectral shape alone (E181 H2)} \to \text{insufficient as unique mechanism} \quad
\text{Energy alone (E182 H3)} \to \text{insufficient as unique mechanism} \quad
\text{Response geometry (E183-A)} \to \text{no evidence beyond } \Gamma \text{ (robustness check
reversed the primary result: perm } p=0.003 \to 0.586\text{)} \quad
\Gamma \to \boxed{\text{surviving phenomenon}}
$$

**What survives, stated precisely**: $\Gamma_i \to \Delta_i$ (representation-induced instability
predicts counterfactual restoration benefit) survives rank control, energy control, $U,V$
preservation, two structurally distinct perturbation families, asymmetric energy perturbation,
boundary control, and an attempt to find richer information in the response geometry than Γ's
scalar diameter already captures. **This is a measurement of recoverable computation, not yet a
way of exploiting it** — the gap E184 is meant to cross.

## Why the five obvious uses of Γ are rejected (recorded, not re-litigated later)

| Use | Form | Rejected because |
|---|---|---|
| A — Γ→attention | $Z_i' = a(\Gamma_i) Z_i$ | Uncertainty-guided attention/reweighting, occupied |
| B — Γ→extra refinement | high Γ ⇒ more compute | Adaptive/selective refinement, occupied |
| C — Γ→sampling | spend more perturbation budget on high-Γ tiles | Adaptive computation, occupied |
| D — Γ→consistency loss | penalize $D(T_k(Z_i)) - D(Z_i)$ | Perturbation consistency/invariance regularization, occupied |
| E — Γ→auxiliary predictor | learn $Z_i \to \hat\Gamma_i$ | Learned uncertainty/error estimation, occupied |

All five put Γ into an existing pipeline shape. None escape the crowded space (consistent with
the E180 novelty audit's finding that the *application shape*, not just the mechanism, is what's
occupied).

## The conceptual pivot

Not: perturb → measure uncertainty → attention.
Instead: **perturb → observe response → extract invariant → repair representation.**

The question is not "how uncertain is the model" but "does a representation exist, computationally
equivalent to the current one, that is more stable under the same admissible perturbations?" —
representation-conditioned recoverability, not output uncertainty.

---

## Derivation: the smallest possible repair operator

### Step 1 — the naive move, explicitly rejected

$$
\bar Y = \frac{1}{K}\sum_k Y_k, \qquad Y_k = D(T_k(Z))
$$

This is ensembling/consensus prediction over the counterfactual outputs. **Rejected**: it
operates on outputs, not on the representation, and is not what the phenomenon is about — Γ→Δ is
a statement about restoring the *representation*, not averaging *predictions*. Recorded so it is
not silently reinvented later.

### Step 2 — the general repair objective

$$
Z_i^* = \arg\min_{Z'} \Big[ \text{Disp}\big(D(T_1(Z')), \ldots, D(T_K(Z'))\big) +
\lambda \, d\big(D(Z'), D(Z_i)\big) \Big]
$$

Two terms: **instability** (the same dispersion Γ already measures, now as an objective rather
than a diagnostic) and **fidelity** (an anchor preventing the trivial failure mode — a stable but
wrong representation, e.g. $D(Z') = $ all-background, which could be maximally stable and
maximally useless). The fidelity term is not optional; recorded explicitly as the reason minimizing
instability alone is rejected outright, not merely cautioned against.

### Step 3 — reduce to the smallest solvable version: the feasibility question, not the operator itself

Before attempting to solve the general $\arg\min$ (which requires choosing a search space for
$Z'$, a differentiable or gradient-free optimizer, and a stopping rule — all premature), reduce
to the question whose answer determines whether solving it is even worthwhile:

$$
\mathcal{E}_\epsilon(Z) = \{Z' : d(D(Z'), D(Z)) \le \epsilon\} \qquad \text{(the output-equivalence
set at tolerance } \epsilon\text{)}
$$

$$
\boxed{\min_{Z' \in \mathcal{E}_\epsilon(Z)} \Gamma(Z')}
$$

This is the **smallest possible repair operator**: does the minimum-instability point within the
output-preserving neighborhood have meaningfully lower Γ than $Z$ itself? Two possible answers,
both informative, neither requires committing to a full repair algorithm to obtain:

- **Case 1 — repairable**: $\Gamma(Z^*) \ll \Gamma(Z)$ while $D(Z^*) \approx D(Z)$. The network's
  current representation encodes the (approximately) correct computation but in an unnecessarily
  sensitive form. This is the case worth building an algorithm around.
- **Case 2 — intrinsic**: $\Gamma(Z') \approx \Gamma(Z)$ for all $Z' \in \mathcal{E}_\epsilon(Z)$.
  Instability is not a repairable representational defect — it reflects a deeper computational
  constraint of the decoder/task state at that tile. Γ remains a valid *diagnostic* but repair is
  the wrong operation to build on it.

**This is the actual next experiment (not yet run, not yet coded)**: determine which case holds,
on a small feasibility scale, before any repair algorithm is designed in full.

## What "success" and "failure" mean for this smallest operator (fixed now, before any GPU work)

Three arms, per tile, at a small feasibility scale (5-10 subjects, tiles spanning low/mid/high Γ
— not yet specified further, see "what remains before implementation" below):

1. **Original**: $Z$ (reference point, no intervention).
2. **Random movement control**: $Z + \epsilon R$ for a fixed-magnitude random direction $R$
   (matched in scale to whatever movement the counterfactual-informed search uses — the null that
   any representation movement of similar size would look the same).
3. **Counterfactual-informed movement**: $Z^*$, the (approximate) solution to the
   $\min_{Z' \in \mathcal{E}_\epsilon(Z)} \Gamma(Z')$ problem above.

Measure, for each arm: $\Delta\Gamma$ (instability change) and $\Delta \text{Dice}$ (actual
ground-truth Dice change — this is the one place GT enters, exactly as Δ's construction in
E180-E183, evaluate-only, never used to construct the movement).

**Pre-specified readings, fixed before running anything**:

| Pattern | Reading |
|---|---|
| $\Gamma\downarrow$ AND $D(Z^*)\approx D(Z)$ AND $\text{Dice}(Z^*) > \text{Dice}(Z)$, and this is NOT matched by the random-movement control | Crossed from observation to mechanism — repair is a real, exploitable operation |
| $\Gamma\downarrow$ but Dice does not improve | Kill repair as a Dice-improving operation (Γ may still be a valid diagnostic) |
| Dice improves but the random-movement control improves comparably | Not repair-specific — likely ordinary smoothing/denoising from any small representation movement, investigate and probably kill |
| No $Z' \in \mathcal{E}_\epsilon(Z)$ achieves materially lower Γ than $Z$ (Case 2 above) | Abandon repair; reinterpret Γ as intrinsic sensitivity/conditioning, not a repairable defect |

No outcome here licenses skipping straight to a trained repair module — every path above is a
research finding to report, not a step toward committing to the eventual algorithm without
further validation (fidelity-vs-correctness gap, denoising-vs-repair confound, prior-art audit,
and the ≥1pp Dice bar remain separately required, per the standing checklist below).

## Reordering (adopted) — equivalence-class existence comes before all four design questions

The four questions below (search space, optimizer, $\epsilon$, tile selection) are **not
independent** — they were initially listed as parallel open items, which was a mistake. Search
space and optimizer only need to be designed *if* a repairable equivalence class is shown to
exist at all; designing them first risks spending effort making an ill-posed object easy to
search. Adopted order:

$$
\boxed{\text{equivalence-class existence} \to \text{search space} \to \text{optimizer} \to
\epsilon \to \text{tile selection}}
$$

### The sharpened prior question — geometric, not optimization-shaped

$\mathcal{E}_\epsilon(Z)$ was originally defined implicitly in representation space
("representations whose output is close to $D(Z)$"). Made precise: it is a **preimage under the
decoder** of an output-space ball,

$$
\boxed{\mathcal{E}_\epsilon(Z) = D^{-1}\big(B_\epsilon(D(Z))\big)}
$$

**Two existence questions that must not be conflated (this was the actual gap in the first pass)**:

1. **Mere representation non-uniqueness** — does $\exists\, Z' \neq Z$ with $D(Z') \approx D(Z)$?
   For an overparameterized decoder this is almost certainly true and **not interesting on its
   own** — null directions of $D$ exist generically and say nothing about repair.
2. **Stability-changing non-uniqueness (the actual E184 gate)**:
   $$
   \boxed{\exists\, Z_1, Z_2 : d(D(Z_1), D(Z_2)) \le \epsilon \;\text{ and }\;
   |\Gamma(Z_1) - \Gamma(Z_2)| > \delta}
   $$
   for a **predefined, meaningful** $\delta$ (fixed the same way $\epsilon$ must be — independent
   of Dice outcomes, see below). This requires *both* same computational answer *and* different
   counterfactual stability. Only this version gives repair something to exploit.

### Reformulated as a range, not a pairwise existence claim (adopted, cleaner)

$$
\mathcal{G}_\epsilon(Z) = \{\Gamma(Z') : Z' \in \mathcal{E}_\epsilon(Z)\}
\quad\text{(the set of stability values reachable without leaving the output-equivalence class)}
$$

- **Null**: $\text{diam}\,\mathcal{G}_\epsilon(Z) \approx 0$ — the output-equivalence class has
  essentially one stability level. Γ is a property of the output state itself, not an
  independent representational degree of freedom. Representation repair is the wrong
  abstraction — stop, reinterpret Γ as intrinsic.
- **Interesting**: $\text{diam}\,\mathcal{G}_\epsilon(Z) > 0$ — computationally equivalent
  representations exist with different counterfactual stability. The representation carries
  degrees of freedom invisible to the current prediction but visible to counterfactual
  sensitivity.
- **Strong**: $\exists\, Z^* \in \mathcal{E}_\epsilon(Z)$ with $\Gamma(Z^*) < \Gamma(Z)$ by a
  meaningful margin — a direct route to repair, not just heterogeneity.

**Sequencing consequence (adopted)**: we do not need to ask "can we optimize Γ" yet. We first
ask "is Γ variable inside an approximately output-equivalent representation class." Only if
$\text{diam}\,\mathcal{G}_\epsilon(Z) > 0$ does $\min_{Z' \in \mathcal{E}_\epsilon(Z)} \Gamma(Z')$
become worth attempting at all.

### The confound this test must separate out

Representation-displacement magnitude and Γ-change must not be conflated — if perturbations are
generated with unconstrained/arbitrary magnitude, "Γ varies across the equivalence class" could
trivially just restate "larger perturbations move Γ more," which E181/E182 already established
and is not new. Any feasibility probe must control for displacement magnitude, not merely filter
by output-equivalence.

**This equivalence-class question (now the diam-of-$\mathcal G_\epsilon$ version) is the next
paper-only step, before any of the four design questions below are resolved and before any
GPU/optimizer work.** It is cheaper to test than building a search: it asks about existence, not
about how to find or move toward such points efficiently.

## Feasibility probe design (conceptual, locked before implementation — still no GPU)

Four things fixed now:

**A. Locality.** Only the single enc3 tile $Z_i$ is modified. Changes elsewhere would
contaminate the interpretation — matches every prior stage's tile-local intervention discipline.

**B. Small, predefined perturbation family — not an optimizer.** $Z' = Z + \alpha V$ for several
predefined directions $V$ and magnitudes $\alpha$, fixed in advance.

### Construction of $V$ (resolved — the last open mathematical piece)

**Not** the same fixed direction already used throughout E180-E183's Γ-probe construction
(`fixed_direction()` in `run_e180_gamma.py`). Reusing it here would be subtly circular: it would
collapse "does Γ vary across output-equivalent points" into "does re-applying the exact same probe
at different magnitudes move Γ" — largely overlapping with what Stage 2/2b's severity calibration
already characterized, not a new question.

**Also not** directions derived from T6/T7's own transform families (small steps along
$\Sigma^\gamma$ or $cZ$) — considered and rejected for the same reason S5 (search space candidate)
was flagged as risky: if the probe directions only span what is already known to move Γ, finding
that Γ varies along them is a weaker, more expected result, not a genuine test of whether the
equivalence class is Γ-heterogeneous in general.

**Adopted**: $V$ is a **set of several (5-10) independently-drawn fixed random directions**,
seeded separately from the existing Γ-probe seed, spanning enc3's channel space — structurally
identical in *construction* to `fixed_direction()` (a random unit vector over the channel axis)
but **numerically distinct draws**, so the perturbation family used to test the equivalence
class's Γ-heterogeneity is not literally the same object used to measure Γ in the first place.
This tests whether output-equivalent points reached via *different* displacement directions (not
just different magnitudes of one direction) show different Γ — the genuinely open question,
without presupposing which direction matters. Exact count (5 vs 10) and the random seed(s) to be
fixed at implementation time, following the same "fixed before running, never re-drawn to chase a
result" discipline as every other stage.

**C. Output-equivalence filter.** Keep only candidates with $d(D(Z'), D(Z)) \le \epsilon$;
$\epsilon$ fixed independently, per the existing rule.

**D. Stability measurement.** Recompute Γ using the *exact same definition* already used in
E180-E182 on accepted candidates — no new instability metric introduced for this test.

### Decision tree (locked)

```
        Output-equivalent Z'
                │
        Does Γ vary (diam G_eps > 0)?
           /              \
         NO                YES
          │                 │
   KILL REPAIR         Can Γ decrease
   (Γ intrinsic)        meaningfully?
                         /        \
                       NO          YES
                        │           │
                 intrinsic Γ    REPAIRABLE
                    state         candidate
```

Plus an explicit confound branch: Γ decreases, but only because representation displacement
increased, or the (approximately-preserved) output actually changed, or the effect is ordinary
output smoothing — classified as **confound, not success**, same discipline as the later
three-arm test's random-movement control.

### What this stage explicitly does NOT use

**Ground-truth Dice does not enter the existence test at all** — not to select perturbation
directions, not to set $\epsilon$/$\delta$, not to decide which candidates are "accepted." The
existence question must be answerable without looking at whether repair improves Dice; Dice is
strictly a downstream evaluation variable, tested only after the equivalence-class question is
already answered. Locked ordering:

$$
\boxed{\text{discover equivalence structure} \to \text{test stability variation} \to
\text{only then test performance}}
$$

## The four design questions (now explicitly downstream of equivalence-class existence)

1. **Search space for $Z'$** — evaluated once existence is established. Five candidates recorded,
   not yet chosen between: **S1** unconstrained additive ($Z'=Z+\delta$, $\|\delta\|\le r$) —
   too unconstrained on its own. **S2** full channel-mixing ($Z'=AZ$, $A\in\mathbb R^{C\times C}$)
   — too many parameters at bottleneck scale ($256^2$). **S3** affine/channel-wise
   ($Z'=\alpha Z+\beta$ or per-channel) — cheap but risks collapsing into ordinary
   normalization/calibration, not a distinct operation. **S4** low-rank movement
   ($Z'=Z+UV^\top$, small rank) — a constrained tangent-like space. **S5** reuse the same
   transformation family used to define Γ ($Z'=T_\theta(Z)$ for $T_\theta$ in T6/T7's family) —
   conceptually cleanest (repair searches within the same counterfactual equivalence structure
   used to measure instability) but risks circularity: if $T_\theta$ only spans the
   Γ-*measuring* family, the search may be forcing the answer rather than discovering it. Not
   yet resolved between; flagged as needing its own scrutiny before adoption.
2. **Optimizer** — gradient-based vs. derivative-free, contingent on search-space choice. If
   gradient-based: $\Gamma = \max_{k,l} d(Y_k,Y_l)$ is nonsmooth, so optimization would use a
   smooth surrogate (e.g. a log-sum-exp softmax-style relaxation $S_\tau$) purely as the
   optimization target, while all reported/scientific quantities still use the exact,
   pre-registered $\Gamma$ — never the surrogate. This introduces its own hyperparameter $\tau$,
   which is a reason to also consider a low-dimensional derivative-free search (more expensive
   per-evaluation, scientifically cleaner, no surrogate to justify). Explicitly ruled out: a
   learned/neural repair network at this stage — that would convert a feasibility question into
   "can another trained model repair representations," a much harder novelty and attribution
   problem, and is not the smallest possible operator this derivation is trying to isolate.
3. **$\epsilon$ (fidelity tolerance)** — must be fixed **independently of GT Dice and repair
   success**, or the entire test becomes circular (choosing whatever $\epsilon$ makes the result
   look good). Candidate approach: derive $\epsilon$ from the decoder's own natural output
   variation under some benign, already-characterized numerical/reconstruction perturbation
   (e.g. a fixed percentile of AMP-vs-fp32 output variation, or T7's gentlest calibrated energy
   perturbation, both already measured in this project) — not chosen post-hoc. Exact source not
   yet decided.
4. **Tile selection** — must not be Γ-stratified-then-selected-for-highest-Γ (that would be
   circular: select by Γ, then show Γ predicts repair success). Adopted: **stratify by Γ**
   (low/mid/high, fixed quantiles) **plus a random-tile control**, so the feasibility test
   covers the full spectrum rather than cherry-picking the subjects most likely to show an
   effect.

**None of the four are resolved by this document** — they remain open, explicitly downstream of
the equivalence-class question above, which is the actual next step, still on paper, still before
any GPU work.

## Feasibility probe result (2026-09-18) — status corrected, gate NOT yet passed

**First-pass result** (10 subjects, 320 records, 8 directions × 4 α, ε derived from T7's
gentlest calibrated level): mechanically read as "INTERESTING" (mean diam-as-fraction-of-ε
≈0.50 at the well-powered α levels). **This status is corrected before proceeding.**

**The error caught**: the statistic computed was $\text{diam}\{d(D(Z+\alpha V_j), D(Z))\}$ —
the spread of *output displacement from $D(Z)$* across directions within the equivalence
filter — not $\text{diam}\,\mathcal{G}_\epsilon(Z) = \text{diam}\{\Gamma(Z+\alpha V_j)\}$, the
spread of *Γ itself* (Γ being the max-pairwise-distance-across-the-T6/T7-family object that
actually survived E180-E182, not a distance-to-$D(Z)$ statistic). These are different claims:

$$
d(D(Z+\alpha V_1), D(Z)) \neq d(D(Z+\alpha V_2), D(Z)) \quad\text{(shown)}
$$

$$
\Gamma(Z+\alpha V_1) \neq \Gamma(Z+\alpha V_2) \quad\text{(NOT yet shown)}
$$

**Corrected status: `INTERMEDIATE_PASS — output-equivalence neighborhood is directionally
non-isotropic`.** This is real: at matched perturbation magnitude, direction affects
$D(Z+\alpha V)$'s displacement from $D(Z)$ (the decoder's local output map is anisotropic, not
merely magnitude-sensitive) — reliably observed across essentially all 10 subjects at the
well-powered α levels (0.5, 1.0), one direction consistently producing markedly smaller
displacement than the other seven at matched α. **This does not by itself establish the E184
gate.** The gate requires Γ variation, not output-displacement variation.

### The required continuation (small, no new GPU subjects/directions)

Recompute $\Gamma$ for the **already-collected** output-equivalent candidates — no new subjects,
no new directions, no Dice, no GT, no optimizer, no new metric. For each accepted
$Z' = Z + \alpha V_j$ (passing the existing ε filter), compute $\Gamma(Z')$ using the *same*
T6/T7-family construction already used throughout E180-E182 (perturb $Z'$ by the T6/T7 family,
measure max pairwise prediction distance across that family) — this does require additional
GPU inference (one T6/T7-family sweep per accepted candidate), but reuses every existing
component (transforms, severities, Γ definition) verbatim.

**Three possible outcomes, fixed before running**:

- **A** — $\text{diam}\,\mathcal{G}_\epsilon \approx 0$: the directional structure found is only
  in output displacement, not Γ. **Repair branch closes.**
- **B** — $\text{diam}\,\mathcal{G}_\epsilon > 0$ but tracks perturbation magnitude/displacement
  rather than direction independently: likely a trivial local-sensitivity restatement, **still
  not sufficient**.
- **C** — at matched $\alpha$ and controlled output displacement, $\Gamma(Z+\alpha V_a) \neq
  \Gamma(Z+\alpha V_b)$ systematically: $D(Z_a) \approx D(Z_b)$ but $\Gamma(Z_a) \neq
  \Gamma(Z_b)$ — **the actual result sought**. Establishes that current-prediction equivalence
  does not determine counterfactual stability, the precise degree of freedom a repair operator
  could exploit. Only outcome C passes the existence gate and licenses moving to the four design
  questions (search space, optimizer, etc.).

## Γ-continuation result (2026-09-18) — existence gate reading: Outcome C

Ran on the 187 already-identified output-equivalent candidates (10 subjects, 0 new
subjects/directions/GT/optimizer). `E184_gamma_continuation_raw.json`: 187 records, 0 NaN/Inf,
`gamma_Zprime` range [0.0118, 0.1164]. Analysis in `E184_gamma_continuation_analysis.json`.

**diam(Γ) is nonzero at every α, for nearly every subject:**

| α | n_subjects | mean diam(Γ) | mean diam(output-distance) |
|---|---|---|---|
| 0.5 | 10 | 0.0023 | 0.000076 |
| 1.0 | 10 | 0.0029 | 0.000104 |
| 2.0 | 8  | 0.0040 | 0.000077 |
| 4.0 | 2  | 0.0026 | 0.000030 |

**B-vs-C discriminator** — within-group Spearman ρ(output_distance, gamma_Zprime) across 26
(subject, α) groups with ≥3 candidates: mean ρ = **−0.225**, median = **−0.31**, individual ρ
range **−1.0 to +0.8**, only 23% positive. If diam(Γ) were inherited from residual
output-displacement spread among the equivalence-filter survivors (outcome B), a strong,
consistently *positive* correlation would be expected. Instead it is weak, inconsistent in
sign, and more often negative — e.g. subject `01041` at α=1.0 has its largest Γ (0.1113) paired
with a smaller output-distance (8.9e-5) than several lower-Γ candidates (1.1e-4–1.33e-4).

**Reading: Outcome C.** Among candidates satisfying $D(Z_a)\approx D(Z_b)$ (both within ε of
the reference output), $\Gamma(Z_a)\neq\Gamma(Z_b)$ systematically, and this is not explained
by which candidate sits closer to the reference output within that equivalence class.
Current-prediction equivalence does not determine counterfactual stability — the existence gate
is passed at feasibility scale.

**Caveats carried forward, not smoothed over:**
- α=2.0 (8 subjects) and α=4.0 (2 subjects, thin) carry much less weight than α=0.5/1.0 (full
  10 subjects, up to 8 directions), which show the same pattern and should anchor the reading.
- Several correlation groups have small n (3-5); the aggregate (mean ρ, 77% near-zero-or-negative)
  is more trustworthy than any single group's ρ.
- Still a 10-subject, single-tile, single-checkpoint feasibility probe — establishes existence,
  not prevalence or effect size. Not E180/E182's full 125-subject subject-aware regime.
- Strictly the existence question throughout: no GT, no Dice, no repair algorithm anywhere in
  this computation.

Per the locked decision tree, outcome C licenses moving to the four design questions (search
space, optimizer, ε, tile selection) for an actual repair operator — pending the user's
explicit go-ahead, not started here.

## Standing checklist this operator must still clear even if Case 1 holds (not evaluated here)

1. Stable equivalent representations exist and are findable cheaply (this document's question).
2. Repair actually improves ground-truth Dice (this document's success criterion).
3. The improvement is not merely smoothing/denoising (the random-movement control tests this).
4. It is not ordinary consistency regularization in disguise (a later prior-art question).
5. It survives a formal prior-art audit (not yet run — the D/E rejections above were informal).
6. It can plausibly reach the project's standing ≥1pp Dice bar at full scale (not testable at
   feasibility scale, a later question).

Recorded so a future session does not treat Case-1-at-feasibility-scale as sufficient on its own.
