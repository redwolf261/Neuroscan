# E193 — TC replication: NEGATIVE. The phenomenon is ET-specific.

**Date**: 2026-09-20
**Status**: COMPLETE. Methodology frozen and transplanted verbatim from E143/E144/E192.
**Outcome**: **Outcome 3** of the three pre-registered possibilities — TC fails while ET is
strong. Pre-registered as "not a failure"; it relocates the question.

---

## Controls held (all five, as specified before running)

1. Same 5-fold cross-fitting — observer never sees the target subject's TC labels.
2. Same $Z$ — **not** redesigned for TC. Only the two definitionally region-indexed entries
   were substituted (log ET size → log TC size; ET fraction → TC fraction). No new variables,
   no reselection.
3. Same 200-shuffle permutation null.
4. Same six models (all have `dice_TC`).
5. ET and TC kept completely separate as inputs. $O^{ET}$ never entered this experiment; the
   ET/TC comparison below is **post hoc only**.

Same seed, same $N=90$, same observers, same architecture, same thresholds.

## Result 1 — observer-independence is already weaker

| | ET (E143) | TC (E193) |
|---|---|---|
| pairwise Spearman range | **0.786 – 0.974** | **0.549 – 0.870** |
| mean recoverability (mlp_deep) | 0.808 | 0.745 |

The four observers agree far less about *which TC subjects are recoverable*. $O^{TC}$ is
substantially more estimator-dependent — i.e. it is closer to "recoverability according to $f$"
than to a property of the image. The H1 property that made ET interesting is already degraded
before the regression is run.

## Result 2 — $\Delta R^2$ collapses, and half the models go negative

| model | mean TC | $R^2(Z)$ | $R^2(Z{+}O)$ | $\Delta R^2$ | perm p | $\rho(O,\text{Dice})$ |
|---|---:|---:|---:|---:|---:|---:|
| E130_baseline_seed0_ep32 | 0.8432 | 0.3430 | 0.4203 | **+0.0773** | 0.0000 | 0.537 |
| E131_v14_seed0 | 0.8412 | 0.2909 | 0.3147 | **+0.0239** | 0.0000 | 0.500 |
| **E131_v5control_seed0** | 0.8472 | 0.2942 | 0.2484 | **−0.0458** | 1.0000 | 0.524 |
| E137_v16_seed0 | 0.8380 | 0.3491 | 0.4484 | **+0.0993** | 0.0000 | 0.461 |
| E141_evidence_ep30 | 0.8429 | 0.2662 | 0.2530 | **−0.0132** | 0.9950 | 0.481 |
| E141_shuffled_ep30 | 0.8430 | 0.2606 | 0.2591 | **−0.0015** | 0.9700 | 0.449 |

$$
\boxed{\text{ET: } \Delta R^2 = +0.451 \ldots +0.563,\ \text{all } p=0.000 \qquad
\text{TC: } \Delta R^2 = -0.046 \ldots +0.099,\ \textbf{3 of 6 negative}}
$$

A **5–10× collapse**, and on three models adding $O^{TC}$ makes out-of-sample prediction
*worse*. Including on `E131_v5control_seed0` — the very model on which the ET result is
strongest. That sign flip is the cleanest possible refutation of a general difficulty law.

Two further observations:

- **$Z$ alone is much stronger for TC** (0.26–0.35 vs ET's 0.10–0.15). Standard difficulty
  covariates already explain TC error reasonably well, leaving little for $O^{TC}$ to add.
- **Regime III is no longer empty for TC**: n=2 of 90 (`00525-001` O=0.892/TC=0.298;
  `01293-000` O=0.959/TC=0.079). Small, but structurally different from ET's clean n=0.

Post hoc only: $\rho(O^{ET}, O^{TC}) = 0.603$ over 81 shared subjects — related but far from
identical quantities.

## What this establishes

**The phenomenon is ET-specific.** There is no general image-derived segmentation-difficulty
law here. Any claim of the form "image intensities predict attainable segmentation quality"
is **refuted as a general statement** by this project's own measurement.

This is a real boundary, found by the pre-registered test that was designed to find it, and it
would have been the single most likely reviewer objection to the ET paper. It is now answered
with data rather than speculation.

## Why ET and not TC — the mechanism is already in the record

This is not mysterious, and the explanation was measured long before E193:

| | ET | TC |
|---|---|---|
| defining evidence | **contrast enhancement**: $\delta = t1c - t1n$ | necrosis + enhancement, a compound region |
| causal modality dependence (E142) | zeroing t1c: **0.8433 → 0.0015** | zeroing t1c: 0.9101 → 0.1010 |
| $\rho$(contrast, Dice) (E136) | **+0.595**, partial +0.565 given size | not the governing variable |

ET is defined by a **single, local, intensity-contrast relationship** that a 4-feature per-voxel
observer can actually evaluate. When $\delta$ is weak or inverted, the label is unrecoverable —
for the MLP *and* for the CNN, which is why $O^{ET}$ transfers across all six models.

TC is a **compound anatomical region**. Necrotic core is defined largely by morphology and
context, not by a per-voxel intensity rule. A context-free observer therefore measures something
that only partly overlaps what limits the segmentation model — hence weak observer agreement,
weak transfer, and negative increments.

$$
\boxed{\text{The recoverability signal exists where the label is defined by a local intensity
relationship, and fails where it is defined by context and morphology.}}
$$

That is a **sharper and more defensible claim** than a general difficulty law, and it is
mechanistically supported by two independent prior measurements (E136 correlational, E142
causal).

## Consequence for the paper

**Narrow the thesis, do not abandon it.** The defensible object is:

> For enhancing tumour — a region defined by a local intensity-contrast relationship — a
> cross-fitted, observer-independent, image-derived quantity $O^{ET}_i$ explains 55–69% of
> out-of-sample segmentation error across six independently trained models, where 15 standard
> difficulty covariates explain 10–15%. The same construction **fails for tumour core**
> ($\Delta R^2$ collapses 5–10×, negative on 3 of 6 models), identifying the boundary of the
> phenomenon: it tracks intensity-defined labels, not segmentation difficulty in general.

E193 moves from "a result with an untested generality claim" to "a result with a measured
boundary and a mechanism." For a paper, the second is stronger.

## Next

The non-UNet architecture control (previously step 3) is now **less urgent and differently
framed**. The open question is no longer "is this general" — E193 answered no. It is whether the
ET result is architecture-independent *within its established scope*. WT is worth one cheap run
for completeness (predicted to behave like TC or worse, being t2f-driven and largely solved at
0.92), but it is no longer decisive either way.

**Artifacts**: `E193_recoverability_TC.json`, `E193_crossmodel_TC.json`, `E193_Z_TC_cache.json`.
Scripts `e193_tc.py`, `e193b_tc_dr2.py` (scratchpad — promote with E143/E144/E192).
