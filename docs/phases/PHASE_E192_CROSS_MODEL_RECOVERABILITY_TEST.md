# E192 — Cross-model test: is $O_i$ a ceiling or a model-specific correlate?

**Date**: 2026-09-20
**Status**: COMPLETE. No training, no new inference. Reuses frozen $O_i$ (E143) and six
pre-existing full per-subject evaluations from `e130/`.
**Script**: `scratchpad/e192_crossmodel.py` → `experiments/exp_e12_eggo_m/E192_crossmodel.json`

---

## Why this test

E144 regressed **our checkpoint's** ET error on $O_i$. Two objections were raised before any
paper thesis was allowed to rest on it:

1. The target is $D_i$ (our model's achieved Dice), **not** $D_i^*$ (an attainable ceiling). The
   claim "predicts how well any model could do" was **not** supported by E144 and was withdrawn.
2. Worse: $O_i$ is *itself* an ET Dice score (of a cross-fitted intensity MLP). So E144
   regresses one model's Dice on another model's Dice over the same subjects and labels. A large
   $\Delta R^2$ could be a **shared-difficulty / shared-denominator artifact** rather than an
   information bound.

E192 is the falsification test for objection 2 and a partial test of objection 1.

**Pre-registered reading, fixed before running:**
- If $O_i$ explains only `E131_v5control_seed0` (the model E144 used) → model-specific
  correlate, ceiling interpretation dies.
- If $O_i$ explains **all** models comparably → the quantity is a property of the subject, not
  of any one model, and the ceiling interpretation survives this test.

## Result

Frozen $O_i$ (`mlp_deep`, E143, cross-fitted, never retrained here), same 15-variable $Z$, same
5-fold RidgeCV, 90 subjects present in all six evaluations. 200-shuffle permutation null per
model.

| model | mean ET | $R^2(Z)$ | $R^2(Z{+}O)$ | $\Delta R^2$ | perm p | $\rho(O_i,\text{Dice})$ |
|---|---:|---:|---:|---:|---:|---:|
| E130_baseline_seed0_ep32 | 0.8387 | 0.1470 | 0.6861 | **+0.5391** | 0.0000 | 0.837 |
| E131_v14_seed0 | 0.8359 | 0.1081 | 0.5594 | **+0.4512** | 0.0000 | 0.842 |
| **E131_v5control_seed0** (E144's) | 0.8441 | 0.1243 | 0.6874 | **+0.5631** | 0.0000 | 0.872 |
| E137_v16_seed0 | 0.8329 | 0.1377 | 0.6772 | **+0.5395** | 0.0000 | 0.840 |
| E141_evidence_ep30 | 0.8362 | 0.1021 | 0.5539 | **+0.4517** | 0.0000 | 0.856 |
| E141_shuffled_ep30 | 0.8365 | 0.0967 | 0.5486 | **+0.4519** | 0.0000 | 0.827 |

**E144 reproduces exactly** on its own model ($\Delta R^2 = +0.5631$, matching the stored log to
four decimals) — an independent confirmation that the reconstructed pipeline is faithful.

### Reading

$$
\boxed{\Delta R^2 \in [+0.451, +0.563] \text{ across all six models; } \rho(O_i, \text{Dice}) \in [0.827, 0.872]}
$$

$O_i$ is **not** specific to the model it was validated against. The model E144 used is the
best-explained, but only marginally — it sits at the top of a tight band, not apart from it.
Every model's permutation p = 0.0000.

Notably $O_i$ explains `E141_shuffled_ep30` (a **deliberately corrupted control** model) just as
well ($+0.4519$). Subject-level recoverability governs where even a damaged model fails.

## What this does and does not establish

**Establishes** — objection 2 is substantially answered. A shared-denominator artifact tied to
*our* model's idiosyncrasies would not transfer at this strength to six independently trained
models including an evidence-shuffled control. $O_i$ measures a property of the subject.

**Does not establish** — the ceiling claim in its strong form. Two limits remain, and both must
stay in the paper:

1. **All six models are the same family** — UNet3D variants, same data, same pipeline,
   inter-model $\rho(\text{ET Dice}) = 0.936$–$0.980$. They agree with each other almost as
   strongly as $O_i$ agrees with them. This is evidence that $O_i$ is not
   *checkpoint*-specific; it is **not** evidence that $O_i$ bounds an arbitrary architecture.
   A genuinely different architecture (nnU-Net, a transformer) is the missing control.
2. **$D_i$ remains the target, not $D_i^*$.** $O_i$ predicts *achieved* Dice across models. That
   every model lands near $O_i$ is consistent with a ceiling — and is what a ceiling would
   predict — but does not prove one. The honest phrasing is **empirical cross-subject
   recoverability frontier**, exactly as E144's own memory file already insisted.

## Defensible statement

> A cross-fitted, observer-independent, image-intensity-derived per-subject quantity $O_i$
> raises out-of-sample $R^2$ for ET segmentation error from ~0.10–0.15 to ~0.55–0.69 across six
> independently trained segmentation models ($\Delta R^2 = +0.451$ to $+0.563$, permutation
> p < 0.005 for all six), including a deliberately corrupted control. Standard difficulty
> covariates explain ~12%. Regime III (high $O_i$, poor Dice) is empty (n=0/90).

## Next

1. **TC replication** — now licensed. The question is whether this is ET-specific (tiny,
   contrast-dependent) or a general image-derived notion of segmentation difficulty. That
   distinction shapes the paper.
2. **Cross-architecture** — the strongest remaining control, and the one that would move the
   language from "frontier" toward "ceiling". Requires a non-UNet3D model.
3. Promote `e143.py`/`e144.py`/`e192_crossmodel.py` into committed, seeded scripts.
