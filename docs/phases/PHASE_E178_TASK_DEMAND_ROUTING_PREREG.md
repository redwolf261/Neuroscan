# E178 — Task-Demand-Weighted Compute Routing: pre-registration

**Date**: 2026-09-19
**Status**: PRE-REGISTERED. Not yet implemented. Follows the clean E177 prior-art audit (🟢, not
found — see that doc). This is the first branch since E169-E176's search closure that the user
has committed real training compute to.

## The principle under test

$$
\boxed{\text{task-required representation complexity} \rightarrow \text{where to spend
inference computation}}
$$

Not: uncertainty → compute. Not: another attention block, loss term, steering direction, or
representation-surgery diagnostic. A genuinely new claim shape relative to everything in the
E180-E187/commutator/geometry branch, which stays frozen and unaffected by this experiment's
outcome either way.

## Formula (locked, simplified from the session's first draft per explicit correction)

$$
D_i = \widehat R_i^* \qquad \text{(NOT the ratio } \widehat R_i^*/(\widehat R_i^* + \lambda
R_{i,\text{eff}} + \epsilon) \text{ — that normalization is deferred, not part of the first
prototype)}
$$

Route the top-$B$ tiles by $D_i$ to receive extra refinement compute. Rationale (user's own):
introducing $R_{i,\text{eff}}$ as a second estimated quantity in the first test conflates two
unknowns; if $\widehat R_i^*$ alone contains useful routing information, that is the result worth
having before adding normalization.

## Critical infrastructure gap found and resolved before any code

**$R^*$ has only ever been measured as one scalar per SUBJECT** (E165's whole-volume truncation
sweep, one agreement threshold against the model's own full-volume undegraded prediction) — never
per-tile. The frozen predictor `rhat_enc1_predictor.pkl` (OOF $R^2=0.63$, Spearman ρ=0.79) was fit
to predict that per-subject scalar from per-subject-**aggregated** E169 features. No per-tile
$R^*$ ground truth exists anywhere in this repo to validate a genuinely per-tile predictor.

**Resolved** (explicit user decision, stated as a carried assumption, not hidden): apply the SAME
frozen ridge weights to each tile's own (non-aggregated) E169 per-tile features, producing
$\widehat R_i^*$ directly. **This is an extrapolation beyond what was validated** — the predictor
was fit and OOF-checked at the subject level; its behavior on the per-tile input distribution is
unverified because no per-tile ground truth exists to check it against. This is a load-bearing
assumption of the whole experiment and must be stated in any resulting writeup, not smoothed over
as if it were a validated per-tile predictor.

## Routing granularity

One 128³ sliding-window tile = one region $i$, matching E165/E169/E180-E187's own tile
convention throughout this project. Not per-voxel.

## Refinement operator $R(\cdot)$ — identical across every arm, locked

$$
\text{dec1}_i^{\text{refined}} = \text{dec1}_i + \text{RefinementBlock}(\text{dec1}_i) \cdot
\mathbb{1}[i \in \text{routed set}]
$$

`RefinementBlock` = two `Conv3DBlock(32,32)` layers (the model's own existing block class,
verbatim — no new architecture family), residual, applied at dec1 (32ch, full resolution) —
the final decoder stage, closest to the output, so effects on Dice are most directly
attributable to routing rather than confounded by further downstream processing. Unrouted tiles
pass through unchanged. **Same operator, same weights, in every arm** — the only thing that
differs between arms is *which* tiles get selected, never the refinement mechanism itself. This
is the single most important experimental control: it isolates the routing *principle* from
network capacity.

## The five arms (locked, corrected from the session's original 4-arm draft)

| Arm | Routing criterion | Signal source |
|---|---|---|
| **A — Baseline** | no refinement anywhere | — |
| **B — Constant-target** | refine the same *fixed* B% of tiles for every subject, chosen by a fixed rule independent of content (e.g. deterministic tile-index order) | none, content-blind |
| **C — Uncertainty** | highest mean predictive entropy | $H_i = -\text{mean}(p\log p + (1-p)\log(1-p))$ over tile $i$'s baseline-forward-pass probability map |
| **D — Size-matched** | highest predicted foreground mass | $\sum(\text{baseline-forward-pass probability map})$ over tile $i$ — genuinely inference-time, no GT, directly tests "did $\hat R^*$ just rediscover tumor location" |
| **E — NeuroScan** | highest $\widehat R_i^*$ | frozen predictor applied per-tile (see gap/resolution above) |
| *(Random, dropped from the 5, kept as an internal sanity check only — see below)* | | |

**Random** is retained as a zero-cost sanity check inside the harness (log its Dice, don't count
it as one of the five headline arms) — if Random ≈ Baseline that's a basic pipeline sanity
check; if Random beats Baseline substantially that would itself be a red flag about the
refinement operator's calibration, investigated before trusting anything else.

## Budget matching — non-negotiable per the user's own framing

For a given budget $B \in \{5\%, 10\%, 20\%\}$ of tiles per subject, **every** arm (B/C/D/E)
refines exactly that same count of tiles for that subject. FLOPs are matched by construction
(identical `RefinementBlock`, identical tile count) — no arm can win by using more compute.

## Leakage control — no test-volume information in predictor fitting (already satisfied,
## verified not just asserted)

The frozen predictor was fit on the 125 E165-measured val subjects with subject-level seeded
5-fold OOF, and is applied unchanged to the training/routing subjects. Routing itself uses only
the tile's own intact forward-pass features (E169's `StageFeatures`, computed from one forward
pass, no GT, no target lesion volume, no Dice, no counterfactual rank measurement at any point
in the routing path) — verified directly against `run_e169_rstar_predictability.py`'s feature
computation before this prereg was written, not assumed.

## Stage 1's refinement operator is UNTRAINED (clarified during implementation, confirmed)

The prereg above specifies the operator's architecture but not its training status. Resolved:
**Stage 1 uses a randomly-initialized, FIXED (never trained) `RefinementBlock`**, identical
weights across every arm. Rationale: Stage 1's question is whether *routing selection* carries
signal, independent of whether the refinement operator itself is any good — training it first
would conflate two variables (does the gain come from routing, or from newly-trained
parameters) before the cheap routing-only question is answered. An untrained residual block
should do close to nothing on its own (small random weights, residual connection ≈ near-
identity), so Stage 1 is closer to a wiring/sanity check than a Dice-improvement claim — a real
Dice claim is Stage 2's question, where the SAME architecture would be trained end-to-end,
contingent on Stage 1 showing routing selection matters at all.

## Staged execution — feasibility gate before the full grid (explicit user decision)

**Stage 1 (this session's target): single seed, single budget (B=10%), all 5 arms + Random
sanity check, full validation cohort.** Check the hard kill criteria below. If killed here, the
3-seed × 3-budget grid is never spent. If it passes, proceed to Stage 2.

**Stage 2 (contingent on Stage 1 passing, not started yet): 3 seeds × {5%, 10%, 20%} budgets,
the full pre-registered gate.**

## Hard kill criteria (locked, brutal, per explicit user instruction)

- **Kill if $E \le B$** (Constant) at matched budget — rank-demand routing has not demonstrated
  value over content-blind allocation. This is the E170b-derived bar: beating only Baseline
  would replicate occupied prior art.
- **Kill if $E \le C$** (Uncertainty) — uncertainty already explains the benefit, no need for
  the rank machinery.
- **Kill if $E > B$ and $E > C$ but the gain collapses under size-matching** (arm D beats or
  matches E) — the predictor is likely just rediscovering tumor location/extent, not measuring
  a genuinely distinct "representational demand" quantity.
- **Pass**: $E > B$, $E > C$, $E > D$, direction consistent — proceeds to Stage 2's 3-seed
  validation. Only after Stage 2 passes does mechanism investigation, formula refinement
  (reintroducing $R_{i,\text{eff}}$ normalization), efficiency curves, or paper-writing begin.

No soft landing, no "interesting partial result" framing if these fail — per explicit
instruction, a failure here means stop and report honestly, not invent a 15-stage explanation.

## FROZEN (2026-09-19) — Stage 2 will not run, branch closed

User's final read, recorded verbatim as the closing statement: "task-required rank is
informative diagnostically but not useful as an inference-time routing signal." NeuroScan
(0.8535) < Constant (0.8561), Δ=−0.0026, and NeuroScan ≈ Random (0.8535 vs 0.8558) — no clean
routing advantage over either control. Stage 2's 3-seed × 3-budget grid will not be run. This
closes the branch, not just this stage — no rescue, no reframing, no E179 inside representation
space. The broader strategic consequence (representation-space interventions as a family are
now considered exhausted after 12 attempts spanning architecture, steering, causal intervention,
Γ-instability, output-preserving surgery, commutator analysis, and this routing test) is
recorded in memory, not re-litigated here.

## Bug caught and fixed before the real run counted

Smoke-testing found the initial `RefinementBlock` init loop zeroed every 1-D parameter,
including BatchNorm's `weight` (γ) — a zero-scale BatchNorm outputs exactly zero regardless of
input, collapsing the ENTIRE block to an exact identity (verified directly: relative diff =
0.0 on a test tensor). All arms were bit-identical to baseline in the first smoke test as a
direct consequence — not a null result, a wiring bug. Fixed: only conv weights/biases are
touched by the custom init; BatchNorm keeps PyTorch's own default (`weight=1, bias=0`). Verified
the fix produces a genuine, substantial perturbation (97.6% relative difference on a test
tensor) before running the real 125-subject pass.

## Stage 1 result (2026-09-19) — KILLED at the first gate: $E \le B$

Ran the full validation cohort (125 subjects), single seed, single budget (10%), untrained
fixed `RefinementBlock`, per the locked design.

**Per-arm mean Dice** (across all 125 subjects, mean of per-subject mean-Dice):

| Arm | Mean Dice | vs Baseline |
|---|---|---|
| Baseline | 0.8595 | — |
| Constant | 0.8561 | −0.0035 |
| Random | 0.8558 | −0.0038 |
| **NeuroScan** | **0.8535** | **−0.0060** |
| Uncertainty | 0.8499 | −0.0096 |
| Size | 0.8473 | −0.0122 |

Every routed arm is worse than Baseline — expected and correctly diagnosed: an untrained random
perturbation to 10% of tiles' dec1 activations should not help, and does not. The real question
is not "did routing help" (it didn't, for any arm, at this stage) but whether *routing choice*
differentiates NeuroScan from the controls at matched damage.

**Paired Wilcoxon signed-rank, NeuroScan vs each arm** (125 paired subjects):

| Comparison | Mean diff | Wilcoxon p | Direction |
|---|---|---|---|
| NeuroScan − Constant | −0.00258 | 0.083 | NeuroScan **worse** |
| NeuroScan − Uncertainty | +0.00359 | 0.004 | NeuroScan better |
| NeuroScan − Size | +0.00620 | 0.0001 | NeuroScan better |
| NeuroScan − Random | −0.00229 | 0.008 | NeuroScan **worse** |
| NeuroScan − Baseline | −0.00605 | <0.0001 | NeuroScan worse (expected) |

**This triggers the first pre-registered hard kill criterion: $E \le B$ (NeuroScan does not
beat the Constant-target arm)** — point estimate is negative (NeuroScan worse), and while the
gap is not itself significant (p=0.083), the pre-registered rule was "$E \le B$", not "E
significantly worse than B" — a non-positive point estimate on the primary comparison is
sufficient to trigger the kill as written, and this was fixed before seeing the result.

Additional context, not a rescue: NeuroScan sits in the *middle* of the damage ranking — less
damaging than Uncertainty and Size, more damaging than Constant and Random. There is no clean
story in either direction ("NeuroScan protects Dice best" is false; "NeuroScan is uniquely
bad" is also false). It is statistically indistinguishable from Random in practical terms
(both mean_diff and direction are small and mixed).

**Per the user's own explicit, pre-committed instruction — no soft landing, no invented
15-stage explanation.** Stage 1 is killed at its first gate. The 3-seed × 3-budget Stage 2 grid
is not run, saving that compute per the design's own purpose.

## What this does NOT establish, and what it does not retract

Does NOT establish that $\widehat R_i^*$ contains zero useful routing information — this test
used an **untrained** refinement operator specifically to isolate routing-selection from
learned-refinement effects (Stage 1's own deliberate design choice), and an untrained operator
applied to the wrong 10% of tiles could easily swamp a real but small routing signal in noise.
Does NOT retract the E177 prior-art audit's clean verdict — the audit licensed testing the idea,
and the idea was tested honestly and failed at the first gate. Does NOT retract E170's premise
verification (deficit exists, correlates with worse Dice, not circular at enc1/enc2) — that
result is about the *deficit*, a different quantity from this experiment's own
$D_i=\widehat R_i^*$ (no deficit, no normalization, per the session's own simplification).

**Open question for the user's own synthesis** (not decided here): whether this result reflects
(a) the per-tile extrapolation of a subject-level-fitted predictor genuinely failing to carry
routing-relevant information at tile granularity (the flagged, unvalidated assumption at the
center of this whole design), (b) the untrained-operator confound being too large relative to
any real routing signal to detect at this scale, or (c) a genuine absence of useful
routing-relevant signal in $\widehat R_i^*$ regardless of operator training. This experiment
cannot distinguish between these three by itself.

## What this does NOT do

Does not modify U-Net architecture beyond the identical, arm-invariant `RefinementBlock`. Does
not use GT/Dice/lesion-volume anywhere in routing (Dice enters only as the final evaluation
metric, same discipline as every prior stage in this project). Does not yet test the full
$D_i$ ratio formula (deferred pending Stage 1's simpler $D_i=\widehat R_i^*$ result). Does not
retrain or refit the frozen $\hat R^*$ predictor. Does not claim novelty — E177's audit licenses
running this experiment, not any conclusion about its outcome.
