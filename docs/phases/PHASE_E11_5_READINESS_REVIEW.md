# Phase E11.5: Implementation Readiness Review

**Status**: ✅ Complete — one real design flaw found and corrected (boundary head supervision), pre-registered predictions and failure criteria set, `baseline_frozen_v2` lineage established

**Date**: 2026-08-04

## Purpose

Treats EGGO-v1 as a pull request submitted by another researcher, before
any code is written. This is the last checkpoint before Phase E12's
pilot training run — the first experiment that validates an
*intervention* rather than a *hypothesis*, per the qualitative shift the
user identified between E11 and E12.

## 1. Gradient audit

| Tensor | Requires grad? | Detached? | Why? |
|---|---|---|---|
| `dec1` (shared trunk) | ✓ | No | Main representation; receives $\mathcal{L}_{seg}$ + $\mathcal{L}_{margin}$ |
| `dec1` **as input to `boundary_head`** | — | **Yes** (`dec1.detach()`) | Prevents `boundary_head`'s own BCE loss from becoming a second, uncontrolled segmentation objective shaping the trunk — see §5 below, the corrected design |
| Boundary score $d_i$ | ✓ (w.r.t. `boundary_head` weights) | — | Optimizes the boundary head itself |
| Boundary score $d_i$, **as used to compute $B_i$ inside $\mathcal{L}_{margin}$** | — | **Yes** | Prevents $\mathcal{L}_{margin}$ from gaming the boundary head's decision function to manipulate its own weight (E9 §3) |
| Uncertainty $\hat U_i$ (from `alpha`, `beta`) | ✓ (w.r.t. evidential head weights, via the existing evidential loss) | — | Separate objective, unchanged |
| Uncertainty $\hat U_i$, **as used in $\mathcal{L}_{margin}$'s weighting** | — | **Yes** | Same reasoning as $B_i$ — prevents gradient-hacking the evidential head into artificially low uncertainty to escape weighting (added in E11, §3) |
| Segmentation logits (`probs`) | ✓ | No | Primary task, unchanged |
| Evidence $\alpha, \beta$ | ✓ | No, for the evidential loss | Separate objective, unchanged from frozen baseline |

**Every detach is now intentional and justified**, and there are **two
separate detach points** on the boundary-related path, not one — a
distinction the original E9 draft did not make clearly (it detached
$B_i$ before use in $\mathcal{L}_{margin}$, but did not also detach
`dec1` before it entered `boundary_head`, which is the flaw found in §5).

## 2. Loss audit — computational graph

```
                         ┌─────────────┐
                    ┌───▶│  seg_head   │──▶ probs ──▶ L_seg ──┐
                    │    └─────────────┘                       │
                    │                                          │
   dec1 ────────────┼───▶┌─────────────────┐                   │
   (shared trunk)    │    │ evidential_head │──▶ α,β ──▶ L_evidential (unchanged)
                    │    └─────────────────┘        │           │
                    │                                │           │
                    │         (detach) ▼             ▼           ▼
                    │      U_hat ──────────┐    [existing losses, sum into L]
                    │                       │
                    └──▶ (detach) ──▶┌──────────────┐            │
                                     │ boundary_head │──▶ d ──▶ L_boundary
                                     └──────────────┘   │    (trains boundary_head
                                                          │     ONLY, blocked from
                                              (detach) ▼  │     dec1 by the detach
                                            B_i ──────────┘     at its input)
                                              │
                                              ▼
                                    ┌──────────────────┐
                         dec1 ─────▶│   L_margin        │──▶ (weighted by
                         (NOT       │  (pairwise hinge,  │    U_hat·B_i,
                         detached)  │   E10's formula)   │    both detached)
                                    └──────────────────┘
                                              │
                                              ▼
                                    gradient flows back
                                    into dec1 directly
                                    (this IS the geometry
                                    -shaping mechanism)

  L_total = L_seg + L_evidential(unchanged) + μ·L_boundary + λ·L_margin
```

**Answering the specified questions directly**:

- **Does $\mathcal{L}_{margin}$ affect the segmentation head?** No —
  $\mathcal{L}_{margin}$ never touches `probs` or `seg_head`'s weights;
  its only path into the network is through `dec1` directly (the
  pairwise distance term uses $z_i$, i.e. `dec1`'s output, not
  anything downstream of `seg_head`).
- **Does it affect the boundary head?** No — both $\hat U_i$ and $B_i$
  are detached before use in $\mathcal{L}_{margin}$'s weight, so no
  gradient from $\mathcal{L}_{margin}$ reaches `boundary_head`'s weights.
- **Does it affect the trunk?** **Yes — this is the intended mechanism.**
  $\mathcal{L}_{margin}$'s gradient reaches `dec1` directly via the
  pairwise embedding-distance term; this is the entire point of the
  algorithm.
- **Does it affect the evidential head?** No, for the same reason as the
  boundary head — $\hat U_i$ is detached.
- **Should each of these happen?** Yes to all — this is exactly the
  intended, minimal-interference design: one new gradient path into the
  trunk (via $\mathcal{L}_{margin}$), zero new gradient paths into any
  existing head, and the two new auxiliary quantities ($\hat U_i$-derived
  weight, $B_i$) act purely as read-only signals, never as attack
  surfaces the margin loss could exploit.

## 3. Sanity predictions (pre-registered before E12)

| Metric | Expected direction | Magnitude (rough) | Confidence |
|---|---|---|---|
| Dice | Slight increase or unchanged | ≤ +1pp | Low — per `PHASE_E6_STRESS_TEST.md`/`PHASE_E7_CAUSALITY_TEST.md`, boundary geometry is real but its causal link to *segmentation quality specifically* (as opposed to calibration) was never directly established, only its temporal precedence over calibration |
| HD95 | Slight decrease (improvement) or unchanged | Small | Low, same reasoning |
| ECE | Improve (decrease) | Small-to-moderate | Medium — this is the outcome most directly motivated by the diagnostic chain (E1.1's ECE finding, E7's boundary-precedes-calibration finding) |
| Latent boundary AUC | Roughly unchanged | — | High — already saturated at 0.9945 by epoch 1 in the unmodified baseline (E7); a margin loss pushing already-near-perfectly-separated classes further apart has little room to move this specific metric |
| Boundary margin (mean pairwise same/opposite-class distance) | Increase | Moderate | Medium — this is the most direct, mechanical prediction, close to a tautology of what $\mathcal{L}_{margin}$ optimizes for |
| Evidence Cohen's d (correct vs. incorrect separation) | Increase | Small-to-moderate | Low-Medium — plausible if calibration improves, but not a direct target of $\mathcal{L}_{margin}$ |
| Training time | +15–30% vs. frozen baseline | — | Medium — based on ABO's own measured overhead category (1.2–2× per `PHASE_E5_ALGORITHM_DESIGN.md` §5.3) and the additional (cheap) boundary head; the 320× negative-cap reduction (E11 §5) should keep this well short of ABO's original *unrefactored* overhead |

**Explicit acknowledgment**: several of these predictions (Dice, HD95,
Evidence Cohen's d) are marked low confidence intentionally — the
diagnostic chain (E1.1–E1.4, E6, E7) established that boundary geometry
*precedes* calibration and is *not fully redundant* with other
predictors, but never established that *actively pushing* the boundary
margin wider *causes* better Dice or better calibration. E12 is where
that causal claim gets its first real test, not before.

## 4. Failure criteria (pre-registered, not decided after seeing results)

EGGO-v1 is considered **not working** if, relative to the frozen
baseline on the same seed/protocol, any of the following hold after the
E12b pilot (20–30 epochs):

1. **Dice drops by more than 0.5 percentage points.**
2. **ECE does not improve** (i.e., is equal to or worse than baseline —
   recall the frozen baseline has no evidential head to compare ECE
   against directly; compare instead against the unmodified-but-with-boundary-head
   `baseline_frozen_v2` control described in §5, run with $\lambda=0$ so
   EGGO is inactive).
3. **Boundary margin (mean pairwise distance between same-batch
   opposite-class anchor embeddings) does not increase** relative to the
   $\lambda=0$ control — if $\mathcal{L}_{margin}$ is active but the
   margin itself doesn't grow, the loss isn't doing its stated job
   regardless of what happens to downstream metrics.
4. **Boundary head collapses** — e.g., $d_i$ becomes constant or
   near-constant across all voxels (checkable via the AUC validation
   from E9 §5/E11 §7 — AUC dropping toward 0.5 would indicate collapse).
5. **$\mathcal{L}_{margin}$ goes to exactly zero within the first few
   epochs and stays there** — would indicate either the margin $\delta_d$
   is set too small (trivially satisfied immediately) or the gate
   ($\hat U_i \cdot B_i$) is degenerate (Failure Mode 1, already flagged)
   and firing on essentially nothing.
6. **Training instability** — NaN/Inf losses, or Dice trajectory that
   diverges/oscillates in a way the frozen baseline's own trajectory
   (smooth and monotonic-ish per `PHASE_E7_CAUSALITY_TEST.md`'s epoch-by-epoch
   table) does not.

**If any of 1–6 occurs, the conclusion is "EGGO-v1 as specified does not
work in its current form"** — not grounds to immediately start
adjusting hyperparameters to rescue it (that risk was explicitly named
in `abo_frozen_lessons_learned`: don't retune reactively without a new,
distinct reason). A genuine failure here should be treated with the same
seriousness as Experiment D's ABO null result — informative, not an
error to patch around.

## 5. Design flaw found and corrected: boundary head supervision (Option A vs. B)

The user's specific caution — "I would not train it with BCE against
segmentation labels directly, because then it simply becomes a second
segmentation head" — identified a **real, previously unaddressed flaw**
in E9's original design. Full correction now in
`PHASE_E9_TRAINABLE_BOUNDARY_HEAD.md` §1 and reflected in
`PHASE_E11_IMPLEMENTATION_SPEC.md` §3; summary:

- **Option A (plain BCE, gradients flow to `dec1`)**: rejected — this
  does make `boundary_head` a second segmentation classifier, adding
  unbudgeted pressure on `dec1` beyond what `seg_head` alone provides.
- **Bare Option B (no supervision, let $\mathcal{L}_{margin}$ alone
  shape it)**: also rejected — without a classification signal, nothing
  keeps $d_i$ interpretable or validatable against E1.3's AUC/R² checks.
- **Resolution — supervised, but gradient-isolated**: `d = boundary_head(dec1.detach())`,
  still trained with BCE (so it stays a real, checkable measurement) but
  with `dec1` detached at the input (so `boundary_head`'s own loss
  cannot reshape the trunk — only `seg_head`'s existing pressure and
  $\mathcal{L}_{margin}$'s direct pull/push can). This is the design
  now specified for E12.

## 6. Frozen baseline lineage: `baseline_frozen_v2`

Per the user's explicit instruction and consistent with
`baseline_frozen_milestone`'s existing rule (re-tag rather than silently
drift the reference point when a real architectural change is needed):

- **`baseline_frozen_v1`** = the existing, untouched
  `neuroscan_3d_fixed.py` / git tag `baseline-frozen` — remains exactly
  as is, permanently, per the original freeze rule. Never edited further.
- **`baseline_frozen_v2`** = `baseline_frozen_v1`'s architecture plus the
  dormant `boundary_head` (initialized, present in the model, but
  contributing zero loss when EGGO is inactive — i.e. $\mu = 0, \lambda = 0$
  reproduces `v1`'s exact behavior, this must be verified empirically as
  part of E12a's smoke test, not assumed). This is the actual reference
  point EGGO-v1 experiments compare against, not `v1` directly, since
  `v1` lacks the boundary head entirely and isn't a fair architectural
  control.
- All EGGO-v1 training runs (E12 onward) use `baseline_frozen_v2` as
  their base architecture. A `$\mu=0, \lambda=0$` run on `v2` is the
  required control for §4's failure criteria (item 2's ECE comparison
  and item 3's margin-growth comparison both need this control, not the
  original `v1`, since `v1` has no boundary head to compare against at
  all).

## 7a. Naming: EGGO-M

Per the readiness review's own reasoning, the margin-only design from E8
is renamed **EGGO-M** ("Margin-only") going forward, not "EGGO" — the
latter name is reserved for whatever full, validated combination of
mechanisms eventually results. This keeps the evolution honest and
traceable in both the repo and any future writeup: EGGO-M (margin only,
this phase) → EGGO-MD (margin + density, only if EGGO-M shows a real
signal and density is revisited) → EGGO (final, if applicable). All
references to "EGGO-v1" in E8–E11 documents refer to what is now named
EGGO-M; not retroactively renamed in those files to preserve the
document history, but this is the name used going forward from E12 on.

## 7b. Additional required diagnostic: boundary-head measurement drift

**Gap identified**: the boundary head reads `dec1.detach()` — by design
(§5), it does not receive gradient from $\mathcal{L}_{margin}$ and so
cannot be *directly* pulled to follow representation changes the margin
loss causes. It only tracks those changes *indirectly*, by being
re-supervised via BCE each step against whatever `dec1` has become. This
is the intended, deliberate mechanism preventing the feedback loop
(Failure Mode 6) — but it creates a real risk that was not previously
called out: if `dec1` moves faster (under $\mathcal{L}_{margin}$'s
pressure) than the boundary head's BCE-driven updates can track, the
head's measurement of $B_i$ could become stale *during* training, not
just between offline diagnostic runs as originally worried about in
Failure Mode 6 — the same underlying risk, now identified as possible
even with the corrected, detached design, just less severe than the
undetached (Option A) version would have been.

**Required additional logging for E12a** (added to the smoke-test scope
below): per-epoch **boundary-head BCE loss** and **boundary-head AUC/accuracy**
against ground truth. If these remain stable (flat or improving) while
$\mathcal{L}_{margin}$ is actively pulling on `dec1`, the measurement
stays trustworthy. If they drift upward (worse BCE) or the AUC degrades
toward 0.5 while margin loss is active, that is direct evidence the
boundary head is lagging behind the representation it's supposed to be
reading — a signal to treat $B_i$ as unreliable for that run, distinct
from and in addition to the already-specified collapse check (§4, item 4,
which only checks for the head going degenerate in an absolute sense,
not specifically for margin-loss-induced drift relative to its own
recent history).

## 7c. E12 structure (per user's two-stage plan, updated with 7b's addition)

**E12a — Smoke test** (2–5 epochs, 1 seed): verify no NaNs, confirm
$\mathcal{L}_{margin}$ decreases over the few epochs run, confirm the
boundary head's AUC roughly matches E1.3's offline-classifier benchmark
(the required check from E9 §5 / E11 §7), confirm runtime is in the
predicted range (§3's training-time prediction), confirm gradients look
sensible (spot-check no exploding/vanishing gradient norms on the new
loss terms specifically, not just the total loss), **and log boundary-head
BCE loss + AUC every epoch to check for measurement drift per §7b**.
Additionally verify `baseline_frozen_v2` with $\mu=0,\lambda=0$
reproduces `baseline_frozen_v1`'s trajectory within numerical tolerance
(§6's requirement, now made an explicit E12a checklist item rather than
left implicit).

**E12b — Pilot** (20–30 epochs, 1 seed): evaluate against §3's
predictions and §4's failure criteria. Only proceed to E13
(hyperparameter sweep) if E12b shows a positive signal per those
pre-registered criteria — not a reactive, post-hoc judgment call.

Neither E12a nor E12b is run as part of this document — this review
establishes what "ready" means; the actual runs are the next, separate
step, pending go-ahead.

## Files

| File | Purpose |
|---|---|
| `PHASE_E9_TRAINABLE_BOUNDARY_HEAD.md` | Corrected per §5 above |
| `PHASE_E11_IMPLEMENTATION_SPEC.md` | Gradient table corrected per §5 above |
| `PHASE_E8_EGGO_V1_SIMPLIFIED.md`, `PHASE_E10_MARGIN_LOSS_SELECTION.md` | Unaffected by this review's correction |

---

**Completed**: 2026-08-04 — EGGO-v1 design is now internally consistent; ready for E12a pending go-ahead
