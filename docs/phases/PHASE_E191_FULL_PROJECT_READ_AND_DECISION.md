# E191 — Full-project read and paper decision

**Date**: 2026-09-20
**Status**: DECISION DOCUMENT. No new compute. Based on a full read of the repo, 187 phase
docs, 120 memory files, and verification of artifacts on disk.
**Verdict**: The project's publishable result is **E143/E144**, not the E180–E190 branch.

---

## What the read changed

The E190 sweep concluded "no candidate survives P2" and recommended a methodology report. That
conclusion was **correct for the branch it examined** (latent-intervention → adaptive compute)
but it was scoped to the wrong body of work. Reading the full corpus surfaced a result that
E190's P3 definition never considered, because that definition was built from E165/E167/E133
alone.

$$
\boxed{\textbf{E143/E144 is the strongest result in the project and has never been audited for novelty.}}
$$

Three facts establish this, each verified today against files on disk:

1. **It has no phase document.** `ls docs/phases/ | grep -iE "e143|e144"` → nothing. 187 phase
   docs exist; the strongest statistical finding is not among them. It lives only in a memory
   file and one JSON artifact.
2. **Its prior-art audit was never run.** Its own memory file says so explicitly under
   "MAY NOT": *"that any of this is novel (prior-art audit on recoverability/learnability-
   weighted segmentation NOT run)."* Every other significant branch (E138, E177, E179, E180,
   E190) got an audit. This one did not.
3. **The artifact is real and reproduces exactly.** `E143_recoverability.json`, 90 subjects,
   4 observers. All six pairwise Spearman correlations recomputed today:

   | pair | reported | recomputed | p |
   |---|---|---|---|
   | simple_thresh × mlp_small | 0.786 | **0.786** | 4.2e-20 |
   | simple_thresh × mlp_deep | 0.924 | **0.924** | 1.5e-38 |
   | simple_thresh × mlp_spatial | 0.895 | **0.895** | 1.2e-32 |
   | mlp_small × mlp_deep | 0.806 | **0.806** | 8.9e-22 |
   | mlp_small × mlp_spatial | 0.811 | **0.811** | 3.4e-22 |
   | mlp_deep × mlp_spatial | 0.974 | **0.974** | 1.2e-58 |

   Exact agreement to three decimals on all six.

## The result itself

A **label-blind, cross-fitted, observer-independent** per-subject quantity $O_i$ — how
recoverable the ET label is from raw voxel intensities alone, estimated by observers that never
see subject $i$:

| claim | measurement |
|---|---|
| $O_i$ is a property of the **input**, not of the estimator | 4 deliberately different observers (global threshold → spatial MLP) rank subjects identically, all six pairwise ρ ∈ [0.786, 0.974] |
| $O_i$ **explains out-of-sample error** far beyond standard difficulty | 15 pre-specified covariates: out-of-sample $R^2 = 0.1243$. Add $O_i$: $R^2 = 0.6874$. **ΔR² = +0.5631**, permutation p = 0.0000 (~14σ, 200 shuffles) |
| the model **sits at** that frontier | Regime III (high $O_i$, poor Dice) is **empty, n = 0 of 90** |
| an **independent causal route agrees** | E142: zeroing t1c collapses ET 0.8433 → 0.0015 while WT holds at 0.8927 |

The reason this matters: **12% → 69%** is not a marginal improvement on a difficulty model. It
says most of what looks like "model error" on BraTS ET is not model error at all.

## The prior-art audit, run now (the one that was missing)

Four real searches. The adjacent field is dense but the construction is **not** occupied:

| Adjacent work | What it does | Why it is not this |
|---|---|---|
| EvanySeg, coherence-based evaluators, diffusion-based QC (2409.14874, 2511.09588, 2507.08357) | ground-truth-free **segmentation quality** estimation — predict *this model's* Dice | Predicts the **model's output quality**. $O_i$ is **model-independent** — it measures the input, and is computed by observers that never see the segmentation network at all |
| Uncertainty-based quality prediction (2508.01460), QU-BraTS, Bayesian U-Net variants | predict Dice from **uncertainty maps** of the model | Derived *from the model*. Circular for the question "is the model at the frontier?" — cannot answer it by construction |
| Difficulty estimation for QC (Springer 2025), inter-observer-threshold QC | image-specific difficulty for quality control | Difficulty ≈ annotation disagreement. $O_i$ is intensity-recoverability, and E144 shows it beats 15 standard difficulty covariates by ΔR²=+0.56 |
| Bayes/irreducible-error estimation | theory, mostly classification | No per-subject, cross-fitted, observer-independent segmentation instantiation located |

**Verdict: 🟢 not located.** The distinguishing feature is that $O_i$ is **model-independent and
input-intrinsic** — every located method derives its estimate from the model being evaluated.
That difference is exactly what licenses the frontier claim, which none of them can make.

## Why this is a stronger paper than an algorithm paper

The ≥1pp algorithm route is genuinely closed, and E144 is *why*:

- The only headroom worth ≥1pp is the 15-subject tail (E139: lifting it to 0.40 = **+1.94pp**).
- Those subjects are functionally the t1c ablation (E142, causal).
- **Regime III is empty** — there is no population where information is recoverable but the
  model fails to recover it. An algorithm needs a target population. There isn't one.

So "we could not find an algorithm" and "the model is at the information frontier" are the same
sentence. The second is a result; the first is a complaint. E144 lets it be stated as the first.

This also retroactively explains the kill-list rather than merely listing it: E137, E140, E141,
E147, E150, E178, E179, E188 all failed because each presumed a recoverable-but-unrecovered
population that E144 shows does not exist.

## Honest limits (must appear in the paper)

- **ET only**, n=90 (ET ≥ 200 voxels). Not established for TC or WT.
- **One architecture, one checkpoint, one dataset.** Cross-architecture replication not run.
- **Not an information-theoretic bound.** No Shannon quantity estimated — it is an *empirical
  cross-subject recoverability boundary*. The memory file already flags this; keep it.
- $\rho(O_i, \text{Ridge residual}) = -0.147$, p=0.17, **not significant** — $O_i$'s power is
  largely linearly independent of $Z$ rather than concentrated in its residual. The ΔR²+
  permutation test is the sounder statistic. Report both, as E144 did.
- **Scripts are scratchpad-only** (`e143.py`, `e144.py`). Must be reconstructed into committed,
  seed-fixed, re-runnable form before submission. The JSON survives; the generator does not.

## Decision

**Write the E143/E144 paper.** Framing: *a label-blind, model-independent recoverability
estimator shows a state-of-the-art BraTS model operates at the input-information frontier* —
with the causal modality ablation (E142) as independent corroboration and the kill-list as
consequence, not content.

**Do not** write the E190 methodology report as the primary output. It becomes a discussion
section.

**Do not** start Project 2 Candidate B. It requires LLM RL infrastructure that does not exist
here (the repo is a 3D U-Net on an 8GB RTX 5050) and it is audited-but-unrun — a fresh project,
not a continuation.

**PediMS is not a route.** `find PediMS -type f` → **2 files, one patient**. The memory
describing "~9 patients / 28 scans" overstates what is on disk. Route 1 (new dataset with
headroom) is dead on data availability, not on principle.

## Immediate next steps (in order)

1. Reconstruct `e143.py` / `e144.py` as committed, seeded, re-runnable scripts; confirm the
   JSON regenerates.
2. Write `PHASE_E143_E144_RECOVERABILITY_FRONTIER.md` — the phase doc that was never written.
3. Extend to TC (and optionally WT) — the single highest-value new measurement, since it turns a
   one-region finding into a regional claim. Inference-only, no training, hours not days.
4. Cross-architecture replication on one other model, if the budget allows. This is the
   strongest available robustness evidence.
