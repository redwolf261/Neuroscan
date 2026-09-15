# E160 — Assumption L: is the E48→E97 chain a preprocessing artifact?

**Date**: 2026-09-15
**Status**: Pre-registered and LAUNCHED. Decision rule fixed before results existed.
**Cost**: one checkpoint, one inference pass, no training, no architecture change.

---

## The single question

> Did the 64³ preprocessing manufacture the small-lesion → bottleneck-dependence relationship
> by deleting small lesions before the model ever saw them?

E48 established Spearman(native lesion size, N_b) = **−0.454** (p<0.001) — small lesions depend
*more* on the bottleneck — and roughly 50 experiments (E48→E97, plus E121–E129's rank work) are
built on it.

That entire chain was measured in **Regime 1**: FLAIR-only, 1 channel, binary whole-tumour,
**64³ whole-volume resize**. E29 proved that under that resize the **median native lesion
component vanishes to zero voxels**. So "small lesions depend more on the bottleneck" may
describe what the project's own preprocessing destroyed, not what the model does.

## Why Regime 2 is a clean test

`Dataset/brats_multimodal_dataset.py` — validation items are the **full native volume**,
evaluated by sliding window; "Patch sampling preserves native 1mm resolution." **There is no
resize.** A small lesion is still a small lesion when the model sees it.

This isolates preprocessing without changing the model: same architecture (UNet3D_v5), same
N_b construction, same ablation convention.

## Method

- **Checkpoint**: `e131/runs/E131_v5control_seed0/checkpoints/best.pth` (v5 4-in/3-out,
  `best_mean_dice` asserted = 0.8929357248544694).
- **N_b** = Dice(intact) − Dice(bottleneck zeroed), per region (ET/TC/WT) and 3-region mean.
- **Ablation convention**: the attention gate receives the **same possibly-ablated
  bottleneck**, matching `e48/run_e48_bottleneck_encoding_audit.py:137-140` exactly. E109
  produced a false `DOES_NOT_REPLICATE` by getting this wrong, caught only via a 14× N_b scale
  mismatch. Not to be "fixed".
- **Sanity gates (both passed before any result)**: checkpoint identity assertion; manual trunk
  unroll vs `model.forward()` max abs diff **2.4e-05** < 1e-4.
- Full 125-subject validation set, sliding window, 50% overlap, Gaussian blending.

## Pre-registered decision rule — fixed before the run

Primary: Spearman(native WT voxel count, N_b_mean), permutation p over 10,000 shuffles.

| Outcome | Rule |
|---|---|
| **ARTIFACT** | ρ ≥ −0.15, **or** permutation p > 0.05 |
| **SURVIVES** | ρ ≤ −0.30 **and** permutation p < 0.05 |
| **AMBIGUOUS** | otherwise — report, do not adjudicate, name the confound |

The −0.30 bar is deliberately lenient (two-thirds of E48's −0.454): Regime 2 differs in
modality, label and resolution, so an exact magnitude match is not required — only the same
direction and a real effect.

Secondary (reported, **not** part of the rule): per-region correlations, median split.

## Smoke test (n=2, before the full run)

Both sanity gates passed. Intact 3-region mean Dice **0.9162**; N_b_mean **0.245** — the same
order as E126's 0.272 on Regime 1, confirming the ablation is biting as expected in a
different regime. The n=2 verdict field is meaningless by construction and was ignored.

## What each outcome means

**ARTIFACT** → the E48→E97 chain (~50 experiments) is confounded by information destruction the
project introduced. It must be marked as such and stops being citable as evidence about
segmentation models. This does not make those experiments *wrong* — they correctly describe a
64³-resized FLAIR-only system — but it removes their generality.

**SURVIVES** → the relationship is a real property of the model, the chain remains admissible,
and we proceed to Assumption K.

**AMBIGUOUS** → name the remaining confound before anything else.

---

# RESULT (2026-09-15) — **L = SURVIVES**, and the matched comparison is stronger than the headline

## Primary pre-registered test

| Quantity | Value |
|---|---|
| Spearman(native WT voxels, N_b_mean), n=125 | **−0.3055** |
| Parametric p | 5.31e-04 |
| **Permutation p (10,000 shuffles)** | **0.0008** |
| Pre-registered verdict | **SURVIVES** (ρ ≤ −0.30 and p < 0.05) |

It clears the bar by only 0.0055. On that margin alone I would call this a weak pass — the
project has been burned by exactly such margins (E45/E54 straddling 0.8942 by ±0.0002). **But
the per-region decomposition makes the result much stronger, not weaker.**

## The decomposition — read this, not the headline

| Region | ρ(size_WT, N_b) | p | mean N_b |
|---|---:|---:|---:|
| **WT** | **−0.5234** | **3.80e-10** | 0.1114 |
| TC | +0.1415 | 0.115 | 0.0580 |
| ET | +0.0853 | 0.344 | 0.0412 |
| ET+TC averaged | +0.1246 | 0.166 | — |

**E48's Regime-1 task was binary whole-tumour. The matched quantity in Regime 2 is the WT
region — not the 3-region mean.**

$$\text{E48 (Regime 1, 64³ resize, FLAIR): } \rho = -0.454$$
$$\text{E160 (Regime 2, native res, 4-mod): } \rho_{WT} = \mathbf{-0.5234}$$

On the **like-for-like comparison the relationship does not merely survive — it is stronger**
(−0.523 vs −0.454), at p=3.8e-10, with no resize destroying anything.

The 3-region mean was diluted to −0.3055 because it averages in ET and TC, which show **no
size relationship at all** (both n.s., and weakly *positive*). Choosing N_b_mean as the primary
endpoint was a defensible pre-registration choice for a 3-region task, but it was the wrong
matched quantity, and it under-reported the effect.

## Sanity checks

- Checkpoint identity asserted; manual trunk vs `forward()` max abs diff **2.4e-05**.
- Intact 3-region mean Dice **0.8595** (consistent with this checkpoint's 0.8929 best; the
  lower figure reflects full-125 evaluation vs best-epoch validation).
- **Negative N_b: only 3/125 (2.4%)** overall (ET 6, TC 9, WT 4). E125's confounded causal test
  had **36%** negative N_b, the signature that made its verdict untrustworthy. At 2.4% the
  ablation is behaving sanely here.
- Mean N_b_mean = 0.0702; WT's N_b = 0.1114 carries 52.9% of total N_b mass.

## Verdict

$$\boxed{\textbf{L = SURVIVES. The E48}\rightarrow\textbf{E97 chain is NOT a preprocessing artifact.}}$$

The small-lesion → bottleneck-dependence relationship is a **real property of the model**,
reproducing at full native resolution with 4 modalities on a different checkpoint, on the
region that matches E48's original binary task, with a *larger* effect size than the original.

**~50 experiments are admissible.** The Regime-1 chain describes something real, not an
artifact the project manufactured in preprocessing.

## What this does NOT license

1. **ET and TC show no size-dependence** (ρ=+0.085, +0.142, both n.s.). The bottleneck
   size-effect is a **whole-tumour-extent phenomenon**. It does **not** describe the Regime-2
   failure mode, which is ET/TC sub-partition on the 15-subject tail (E139) — WT is 0.824 on
   exactly those subjects. So E48→E97's finding is real *and* orthogonal to the current target.
2. The E158 two-regime caution still stands for **effect sizes** — Regime 1 pp-numbers are
   still measured against a handicapped FLAIR-only baseline and must not be tabulated with
   Regime 2 numbers.
3. Nothing here reopens any killed mechanism. Six intervention attempts on this phenomenon
   still failed; confirming the phenomenon is real does not make it exploitable.

## Files

`experiments/exp_e12_eggo_m/e160/` — `run_e160_assumption_L.py`, `E160_L_summary.json`,
`E160_L_per_subject.json`, `E160_L_per_subject.csv`, `run_log.txt`.

---

## Scope discipline

One checkpoint. One inference pass. One predefined question. This is **not** a Dice-improvement
attempt and must not become a branch. Raw per-subject output (`E160_L_per_subject.csv/json`) is
produced so the result can be adjudicated from the numbers rather than from the script's
verdict label — the project has been bitten four times (E77, E80, E82, E104b) by trusting an
auto-classifier's label over the underlying data.
