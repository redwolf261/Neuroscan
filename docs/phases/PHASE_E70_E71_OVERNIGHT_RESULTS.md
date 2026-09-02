# Overnight results (2026-09-02): A96 evaluation, CAS 3-seed confirmation, CDCG prediction-1

All results below came from the unattended overnight pipeline
(`experiments/exp_e12_eggo_m/e70/overnight_pipeline.sh`), which ran every stage as
a pre-registered test with no human intervention. Reported exactly as measured.

## 1. MM_A96 (multimodal, 96³): confirmed strong, resolution's own contribution is not

| Comparison | Mean Δ | Bootstrap 95% CI | Verdict |
|---|---|---|---|
| MM_A96 vs FLAIR-only baseline | **+2.19 pp** | [+1.06, +3.70] EXCLUDES 0 | Solid |
| MM_A96 vs MM (resolution alone, modality already present) | +0.38 pp | [−0.00, +0.79] includes 0 | **Not confirmed** |

MM_A96 (pooled 0.9194, per-subject 0.9060) is the best single-seed number of the
whole project. But the CI for "does 96³ add anything over 64³, given multimodal
input is already present" includes zero — the resolution increase's own marginal
contribution is not statistically distinguishable from noise on this one seed. The
total system's strength comes overwhelmingly from the modality expansion, same
conclusion as MM's own comparison.

## 2. CAS 3-seed confirmation: DOES NOT REPLICATE

| Seed | MM vs FLAIR | **CAS vs MM (the actual claim)** | CAS vs FLAIR (system) |
|---|---|---|---|
| 0 | +1.81 pp, CI excludes 0 | +0.30 pp, CI **excludes 0** | +2.11 pp |
| 1 | +1.45 pp, CI excludes 0 | +0.42 pp, CI excludes 0 (barely; t_p=0.068) | +1.87 pp |
| 2 | +1.98 pp, CI excludes 0 | **−0.01 pp, CI includes 0**, p=0.98 | +1.98 pp |
| **3-seed mean** | **+1.75 pp** (std 0.27) | **+0.24 pp (std 0.22)** | **+1.99 pp** (std 0.12) |

**CAS's own contribution is not confirmed.** Mean +0.24pp across 3 seeds with a
standard deviation almost as large as the mean; seed 2 shows a null result
indistinguishable from zero (p=0.98). This is the third time this exact pattern
has appeared in this project (after CCABA and E45/E54): a real-looking single-seed
effect that a proper 3-seed test shows is not reproducible.

**The modality gain (MM vs FLAIR) is now confirmed on 3 seeds**, cleanly, with
tight variance (std 0.27pp) — this is the project's one solid, reproducible,
statistically robust number. It is not novel, but it is real.

**Combined with the already-falsified gate mechanism** (E70 Section on gate
diagnostics: CAS's gate does magnitude-conditioned smoothing, not correspondence
correction, unanimous across 125 subjects), CAS should now be considered fully
closed: wrong mechanism AND unreplicated effect. Recommend not pursuing it further
as a component of any paper's contribution.

## 3. CDCG prediction 1 (E71): PASSES, clearly

The cheapest, first, most falsifiable test of the CDCG design — can an auxiliary
head predict the network's own bottleneck-ablation Dice sensitivity on truly
held-out subjects — came back strongly positive:

| Quantity | Value |
|---|---|
| Spearman(predicted, measured), held-out n=125 | **+0.701** |
| Permutation p | **< 0.001** |
| Partial Spearman, controlling for native lesion size | **+0.904** (p = 4.1×10⁻⁴⁷) |

This is a real, strong, statistically overwhelming result, trained on only 200
labeled subjects and tested on a genuinely disjoint 125-subject validation set —
not the same population used for training the auxiliary head. The partial
correlation (0.904, *stronger* than the raw correlation) shows this is not
explained by lesion size alone — the network's bottleneck representation carries
information about its own causal sensitivity beyond what size alone predicts.

**This is the first pre-registered test all night — across CAS's confirmation, the
A96 resolution check, and this — that passed cleanly on its own terms.** Per the
E71 design doc's own decision rule, this clears the load-bearing gate: proceed to
prediction 2 (does the signal behave sensibly relative to E48's known
size-dependence finding) in a properly separate, still-to-be-designed phase — the
gating mechanism (Section 3.3 of the design doc) has still not been built or
tested, and should not be assumed to work just because prediction 1 passed.

## Overall picture for the paper

| Candidate | Status |
|---|---|
| Multimodal input (MM) | ✅ Confirmed, 3 seeds, +1.75pp — real but not novel |
| A96 resolution, alone | ❌ Not confirmed (CI includes 0) |
| CAS (correspondence-correction claim) | ❌ Falsified mechanism (prior session) |
| CAS (Dice effect) | ❌ Not confirmed, 3 seeds (this session) |
| **CDCG prediction 1** (self-prediction of causal sensitivity) | ✅ **Passed cleanly** |

The one candidate still standing with genuine, freshly-confirmed evidence behind
it is CDCG — specifically, only the narrow claim that a network can learn to
predict its own causal ablation sensitivity from its internal representation.
The full CDCG mechanism (using that prediction to gate inference-time behavior,
Section 3.3 of the design) has not yet been built or tested, and per the design's
own pre-registered rule, should not be assumed to improve Dice just because
prediction 1 passed.

## Artifacts

- `experiments/exp_e12_eggo_m/e70/E70_A96_evaluation.json`
- `experiments/exp_e12_eggo_m/e70/E70_evaluation.json` (now includes all 3 seeds)
- `experiments/exp_e12_eggo_m/e71/E71_prediction1_summary.json`
- Full run logs: `experiments/exp_e12_eggo_m/e70/runs/{MM,MM_CAS}_seed{0,1,2}/`,
  `MM_A96_seed0/`
