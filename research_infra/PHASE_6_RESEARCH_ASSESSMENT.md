# Phase 6: Research Assessment Report

**Purpose**: Answer the seven assessment questions using only what was observed in the codebase and what the diagnostic infrastructure (Phases 1-5) can and cannot currently measure. No project is recommended here; this is evidence for that decision, not the decision itself.

**Evidence base**: `PHASE_1_CODEBASE_AUDIT.md` (static read + 2 empirical gradient/pipeline checks), `PHASE_2_RESEARCH_FEASIBILITY.md` (per-metric YES/PARTIALLY/NO tables), `PHASE_3_4_DIAGNOSTIC_FRAMEWORK.md` (working instrumentation, verified bit-identical to baseline when disabled, verified functional when enabled against the real module).

---

## 1. How difficult is Project A (Adaptive Information-Driven MRI Slice Sampling)?

**Difficulty is bimodal, not uniform** — it splits cleanly into two very different cost tiers:

- **Measurement instrumentation**: Low. Slice identity, patient identity, and the one genuinely-trained proxy signal (`Conv2D5Stem`'s fusion attention weight) are all cheap, additive fixes — implemented and verified in this session (`slice_logger.py`, `slice_metrics.csv`).
- **The mechanism itself**: High, and qualitatively different from an instrumentation problem. Phase 1 §6 established — and Phase 3's end-to-end verification independently reconfirmed against the *real* module (12/12 parameter tensors in `scorer`+`score_head` showed zero gradient, an exact match) — that the `AdaptiveSliceSelector`'s ranking network **does not learn from the task loss under the current implementation.** `torch.topk` is not differentiable, and the selected slices are gathered from the raw input rather than from the scorer's own feature path, so no gradient signal reaches the scoring parameters in either training phase (MAE or segmentation).

This means: *"instrument what exists and observe it"* is cheap for Project A, but the central research claim implied by the project's name — that slice selection is **adaptive**, i.e., that it improves via training — is not something the current system does. Making that claim testable requires an architecture change (differentiable relaxation of the selection: Gumbel-softmax top-k, straight-through estimator, or a soft-attention reformulation), which is a different scope of work than instrumentation, and is properly "implementing the research idea" rather than "measuring the existing system."

**Overall**: Low difficulty to observe current (mostly non-adaptive) behavior; Medium-to-High difficulty to make the core "adaptive" premise scientifically testable at all.

---

## 2. How difficult is Project B (Optimization of Multi-Objective Loss Functions)?

**Uniformly low-to-medium, with no mechanism gap.** Every metric in the Phase 2 table (individual losses, gradient norms, gradient flow, gradient cosine similarity, parameter updates, layer-wise gradients) was a pure measurement gap, not a mechanism gap. The underlying system — three active loss terms summed into one scalar, standard AdamW, standard AMP — is exactly the kind of system this class of diagnostic already handles.

Concrete evidence from this session: the full diagnostic stack for Project B (`training_metrics.csv`, `gradient_metrics.csv`, `gradient_metrics_layerwise.csv`, `gradient_metrics_similarity.csv`) was implemented, wired into the real training loop, and verified — in one pass — to produce correct, sensibly-shaped data (e.g., a Dice-vs-FocalTversky cosine similarity of 0.99 and an Evidential-vs-everything similarity near 0 on an untrained model, which is the expected qualitative shape: the two supervised-segmentation losses pointing the same direction, the small independent regularizer pointing elsewhere).

The one real cost is the **AMP `unscale_()` correctness requirement** (not previously present anywhere in the codebase) and the **compute cost of isolated per-loss backward passes** for gradient attribution/similarity (up to 4x backward cost on the batches where it's sampled) — both are known, bounded, already-mitigated-by-sampling-frequency costs, not open research risks.

**Overall**: Low-to-Medium difficulty, fully instrumentable with the existing training mechanism as-is.

---

## 3. What evidence can currently be collected?

| Evidence | Project A | Project B |
|---|---|---|
| Fully available today (post-instrumentation) | Selected-slice identity, patient identity, fusion attention weight distribution, selector/supervision correspondence rate over epochs | Per-batch Dice/FocalTversky/Evidential/Total loss values, total/encoder/decoder/CBAM gradient norms, layer-wise gradient norms, AMP-correct measurements |
| Available with sampling (higher cost) | Slice-selection evolution across training (once logged, this is free — just group-by-epoch on already-collected data) | Per-loss gradient attribution (every 5th batch), gradient cosine similarity/conflict score (every 20th batch) |
| Not available without a mechanism/architecture change | Whether the selector *improves* at picking informative slices (it currently doesn't update via gradient descent at all); true per-slice segmentation loss/uncertainty/difficulty (no per-slice prediction head exists); slice redundancy (no metric exists) | Whether the disabled consistency/pseudo-label losses would help (requires flipping config flags and re-running training, not just observing) |

---

## 4. Which hypothesis appears easier to validate experimentally?

**Project B, based strictly on what the codebase and diagnostics can currently produce.** Every metric needed to study "do these loss terms fight, dominate, or cooperate" is now collectible with the existing training mechanism, unmodified. Project A's instrumentation is equally cheap for the *measurement* half, but its central hypothesis (adaptive, learned slice selection) is not testable at all under the current mechanism — testing it requires first building the thing the research is supposed to be about, which is a materially larger and riskier scope than "instrument and observe."

This is a statement about **experimental tractability given the current codebase**, not about which idea is more interesting or more novel.

---

## 5. What additional instrumentation would still be required?

**For Project A** (beyond what Phases 3-5 already built):
- A differentiable or straight-through selection mechanism, if the research question is "does adaptive selection learn to improve" (architecture change, not instrumentation — see Phase 2 §A options 1/2).
- Alternatively, if reframed as "does a fixed/heuristic/random slice-selection policy comparison" (Phase 2 §A option 3), no further instrumentation is needed beyond what exists — but this is a narrower research claim.
- A per-slice prediction/label mechanism if the research question requires true per-slice loss/uncertainty/difficulty (currently only the fused, single-center-labeled output is supervised).
- A slice-redundancy metric (no existing scaffold — would need to be defined from scratch, e.g. pairwise feature similarity among the 9 selected slices' stem outputs).

**For Project B** (beyond what Phases 3-5 already built):
- If the research question extends to the currently-disabled consistency/pseudo-label/causal losses, those need to be *enabled* (a training-behavior change) before their gradient interactions can be measured — today's instrumentation only observes what's active.
- Parameter-update magnitude (as opposed to gradient norm) would need either a before/after `param.data` diff around `optimizer.step()`, or inspection of AdamW's internal moment estimates (`exp_avg`/`exp_avg_sq`) — neither implemented in this session, both low-difficulty additions to the existing `GradientLogger`.

---

## 6. What are the major technical risks of each project?

**Project A:**
- **Foundational risk (confirmed, not hypothetical)**: the adaptive mechanism as currently written does not learn. Any research built on top of "the model learns to select informative slices" needs to first establish that a fixed mechanism produces this behavior — which is a prerequisite architecture-design task, not a side investigation.
- **Interpretation risk**: "slice index" is in a trilinearly-resampled 64-slice space, not physical acquisition slices (compounded by two separate resampling steps — `Spacingd` then `Resized`). Any claim about "informative anatomical slices" needs to account for this, since the notion of a "slice" the model reasons about is already a synthetic, interpolated construct.
- **Supervision-correspondence risk**: the fixed supervision index (always 32) has no architectural guarantee of being among the selected slices — a risk to any narrative that frames selection as "choosing the most relevant context for the target slice," since the target slice may not even be shown to the model.

**Project B:**
- **Cost risk, not correctness risk**: full instrumentation (loss-specific gradients + similarity) meaningfully slows training if run on every batch; this is already mitigated by configurable sampling frequency, but any analysis drawing conclusions from sampled batches should account for sampling bias if slice/patient/consistency conditions vary systematically with batch position.
- **Scope-boundary risk**: if the research question extends beyond the three currently-active losses (Dice, FocalTversky, Evidential) into the disabled consistency/pseudo-label mechanisms, that crosses from "observe" into "change training behavior and re-run," which is a different kind of validity claim (comparing two training regimes, not diagnosing one).
- **AMP-normalization risk**: any gradient-norm analysis must consistently use the `scaler.unscale_()`-corrected values (as this session's `gradient_logger.py` does); raw, not-unscaled `.grad` reads at other points in a future modified pipeline would silently produce numbers confounded by the AMP scale factor, which drifts over training.

---

## 7. Which project appears to have stronger publication potential, based solely on the observable behavior of the existing NeuroScan implementation?

Based only on what was observed (no external literature search performed as part of this instrumentation task):

- **Project A's strongest, most concrete, already-observed finding is negative**: the adaptive selector does not currently adapt. A publication angle exists here — but it is "diagnosing and fixing a broken differentiability path in slice selection," which is a different (and narrower, more engineering-flavored) contribution than "a novel adaptive sampling method," unless the differentiable-relaxation fix is built and shown to actually change selection behavior in a measurable way (using the now-available `slice_metrics.csv` selection-frequency and correspondence-rate signals as the evaluation tool).
- **Project B's observable behavior is, so far, unremarkable in the specific sense that matters for a paper**: the one similarity measurement collected in this session (Dice vs. FocalTversky ≈ 0.99, Evidential decorrelated from both) is exactly what a well-behaved multi-term loss should look like — no evidence of the kind of "losses fighting" dynamic that would itself be a novel finding. This could change with real training data over many epochs (the single measurement here was on an untrained, randomly-initialized model on synthetic data, purely to verify the instrumentation works) — the actual research question ("does this change over epochs, does it change once consistency/pseudo-label losses are enabled") is unanswered and requires real, extended training runs with the now-available instrumentation to find out one way or the other.

**Neither project's publication potential can be assessed further without running real, extended training with the instrumentation now in place.** What can be said from the codebase alone: Project A already contains a concrete, verified, non-obvious technical finding (the broken gradient path) that is publication-relevant regardless of which direction is chosen, simply because it affects the correctness of any claim the existing codebase's docstrings already make about "adaptive slice selection" contributing "+3% improvement" (a claim that, per this audit, cannot currently be attributed to learned adaptivity, since the mechanism producing the selection never received gradient signal to become adaptive in the first place).

---

## Summary Table

| | Project A | Project B |
|---|---|---|
| Instrumentation-only difficulty | Low | Low-to-Medium |
| Does the core hypothesis require an architecture change to test? | **Yes** | No |
| Concrete finding already produced by this audit | Selector receives zero gradient (verified twice: standalone + real module) | Loss/gradient instrumentation verified correct and non-invasive |
| What's needed before real experiments can start | Decide: fix differentiability (architecture work) vs. narrow the claim to fixed/heuristic comparison | Nothing — instrumentation is ready to run against real training data now |
| Biggest single risk | The premise ("adaptive") may not hold without further engineering | Sampling-frequency/AMP bookkeeping discipline in future extensions |
