# E163 — Vaswani Track 2.0: charter and Phase-1 entry conditions

**Date**: 2026-09-15
**Status**: Charter. No candidates proposed yet. No compute.
**Follows**: E162 (pivot premise not supported)

---

## What died and what did not

**Died**: BraTS 3-region segmentation as the venue. Five audits 🔴 (E: query-based
segmentation; K: modality informativeness; self-knowledge: predictive self-knowledge;
effective rank: WERank/dynamic-rank; causal-quantity-as-training-signal: CausalDisenSeg et al.).
One 🟢 (L) but orthogonal to the headroom.

**Did not die**: the question.

$$\boxed{\textbf{What computation is treated as necessary, but actually isn't?}}$$

E162's finding is a statement about a *search space*, not about the strategy. The strategy was
never "find an unused module."

## The template, stated correctly

Vaswani's move was not "attention is novel" — attention predates the paper. It was:

$$\text{seq2seq} \overset{\text{assumed}}{\longrightarrow} \text{recurrence} \quad\Longrightarrow\quad \textit{does it?}$$

General form:

$$\text{established task} \rightarrow \text{identify mandatory computation } C_2 \rightarrow \text{remove it} \rightarrow \text{reorganize known primitives} \rightarrow \text{competitive or better}$$

$$X \overset{C_1}{\to} H_1 \overset{C_2}{\to} H_2 \overset{C_3}{\to} Y \qquad\Longrightarrow\qquad X \overset{\tilde C_1}{\to} H \overset{\tilde C_3}{\to} Y$$

---

## THE CORRECTION THAT MATTERS MOST

**Stop starting from NeuroScan's failure modes.** That was the hidden trap, and E162 explains
its failure rate structurally:

$$\boxed{\text{On a well-studied task, a measurement's informativeness predicts its prior-art density.}}$$

The failed loop was:

$$\text{NeuroScan failure} \rightarrow \text{generalize to assumption} \rightarrow \text{search} \rightarrow \text{occupied}$$

The correct loop is:

$$\text{task} \rightarrow \text{dominant computational assumption} \rightarrow \text{is it necessary?}$$

— asked **before** attachment to any particular observation.

---

## The thing that must be said before Phase 1

Vaswani's move worked because of a property of the *task*, not of the authors' cleverness:
RNN sequentiality was a **hard constraint on hardware utilization**. Removing it unlocked
parallelism on accelerators that already existed. The reorganization was not merely elegant —
it converted a latent hardware surplus into a training-speed advantage, which is what let them
train larger models and win on quality.

**The implication for candidate selection:** a genuine Vaswani-shaped $C_2$ is usually one
whose removal releases a **resource that is already being wasted** — parallelism, memory
bandwidth, sample efficiency, annotation cost. "This computation is redundant" alone has
historically produced compression papers, not paradigm shifts.

This is a selection criterion, and it is stricter than "is it necessary?". It should be applied
in Phase 1, not discovered in Phase 4.

## The second thing that must be said: the honest base rate

Vaswani-class removals are rare. In deep learning the credible instances over ~15 years number
in the single digits (recurrence → attention; hand-designed features → learned features;
per-task training → in-context learning; and arguably explicit search → diffusion sampling).

This is a course/thesis project on an RTX 5050 with 8 GB. **The realistic target is not "find
the next Transformer."** It is: find a *small, defensible* instance of the template — a
computation genuinely assumed necessary in a *sub-area*, shown unnecessary on a task that fits
the hardware, with a clean prior-art story.

Stating this now so that Phase 1 is not scored against an impossible bar, and so that a modest
survivor is recognized as a success rather than discarded for not being Attention.

---

## Gate structure (as proposed, with entry conditions added)

**Phase 1 — candidate task families.** Identify places where the field says "of course we need
X." No experiments, no architectures, no modules. Each candidate must be written as:

| Field | Requirement |
|---|---|
| Task | established, with a standard benchmark |
| $C_2$ | the computation assumed mandatory, stated precisely |
| Why it exists | the historical/technical constraint that forced it |
| Why it might not be needed | a specific argument, not "maybe it's redundant" |
| **Resource released** | what is currently wasted that removal would free (per the criterion above) |
| Replacement sketch | known primitives, reorganized — one paragraph, no design |

**Phase 2 — immediate prior-art kill.** Exact novelty claim → audit. No coding. Expect most
candidates to die here; that is the gate working.

**Phase 3 — feasibility gate.** Survivors only, checked against the real constraints:
RTX 5050 8 GB, 24 GB RAM, storage, timeframe, dataset access, and whether a ≥1pp-class result
is even measurable on that task.

**Phase 4 — one tiny falsification experiment.** Only then.

---

## Entry conditions before Phase 1 begins

Phase 1 cannot be scoped without facts only the user has. Three questions must be answered
first, because they determine which task families are reachable at all:

1. **Domain constraint.** Must this stay medical imaging (supervisor/course requirement), or is
   any ML task acceptable?
2. **Infrastructure reuse.** Does the 3D/PyTorch/BraTS machinery need to be reused, or is a
   clean start acceptable? (E162 noted this infrastructure is partly sunk.)
3. **Timeframe and deliverable.** How long remains, and is the deliverable a paper, a thesis
   chapter, or a course submission? This sets how many Phase-2 kills are affordable before
   Option B must be taken.

Without these, any candidate list is guesswork. **Do not generate candidates before they are
answered.**

---

## The fallback remains live

If Phase 1 + Phase 2 produce no survivor within the affordable number of iterations, Option B
(the methodology write-up) is the rational endpoint — not a failure, and materially stronger
than a manufactured novelty claim. E162 lists the assets.
