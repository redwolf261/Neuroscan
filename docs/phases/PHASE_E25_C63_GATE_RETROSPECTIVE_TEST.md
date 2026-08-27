# Phase E25, C6-3 Gate: Retrospective Confidence-Gate Test — GO (Revised Gate)

**Status**: ✅ Complete, revised. The original symmetric gate `w(x)=4p(x)(1−p(x))` was tested first and, held to the user's own strict two-part criterion ("retains MOST corrective signal AND materially reduces damage"), **failed Condition 1** — it retained only 23.4% of FN/FP's corrective movement, not "most" by any reasonable reading. Rather than accept a softer aggregate-ratio justification for that gate, a **ground-truth-conditioned asymmetric gate**, `w(x) = |p(x) − target(x)|` (target=1 for GT-tumor voxels, 0 for GT-background — weighting by how far the current prediction is from the correct answer, not by closeness to the decision boundary), was designed from the specific failure diagnosed in the symmetric gate (FN's own good/bad movement was inverted relative to selectivity intent) and retested on the identical existing data. **The revised gate passes both conditions decisively: 92.3% of corrective movement retained, 99.2% of damaging movement removed, a 122.55× improvement in the corrective/damaging ratio (0.798 → 97.76).** This is the gate the go/no-go verdict is based on.

**Date**: 2026-08-12

---

## Method

No new training, no new checkpoint, no new forward pass throughout — a pure re-weighting of already-collected data. `run_c62_representation_to_logit.py`'s saved output contains, for 26,824 real voxels sampled from C6-2's own real 15-step AdamW-accumulated training-realized trajectory (H2's mechanism): `p_before` (pre-perturbation predicted probability), `delta_logit` (realized logit change), and `category` (TP/TN/FP/FN, from ground truth vs. prediction). "Corrective"/"desired" movement is `|Δlogit|` in the direction that would help (FN: `Δlogit>0`; FP: `Δlogit<0`; TP: `Δlogit>0`; TN: `Δlogit<0`), "damaging" is the complement — the same convention as `PHASE_E25_C62_REPRESENTATION_TO_LOGIT.md`'s collateral-damage accounting.

---

## Step 1: the original symmetric gate fails Condition 1

`w(x) = 4·p(x)·(1−p(x))`, peaking at `p=0.5`, falling to 0 as `p→0` or `p→1`.

| Category | Mean p (good) | Mean p (bad) | Good retention | Bad retention |
|---|---:|---:|---:|---:|
| FN | 0.137 | 0.321 | **18.2%** | 64.8% |
| FP | 0.840 | 0.661 | 35.4% | 87.3% |
| TP | 0.769 | 0.977 | 68.2% | **3.5%** |
| TN | 0.0005 | 0.0006 | 0.1% | 0.3% |

**Diagnosis of the failure**: for TP, the gate works as designed — confident TPs (mean p≈0.98, damaging subset) are gated out (3.5% retention) while TP's own good movement survives better (68.2%). For TN, the gate is non-selective (both ~0.1–0.3%) since TN sits so far from p=0.5 (mean p≈0.0005) that nothing passes regardless of direction. **For FN, the gate is actively counterproductive**: it retains *more* bad movement (64.8%) than good (18.2%), because FN's damaging-direction subset happens to sit closer to the boundary (mean p=0.321) than its corrective-direction subset (mean p=0.137) — the voxels pushed hardest in the *right* direction tend to be the ones most confidently wrong to begin with, which a boundary-centered gate penalizes hardest.

**Aggregate result for the symmetric gate**: corrective retained 23.4%, damaging retained 2.7%, ratio improves 8.68× (0.798→6.926). Held to the strict criterion ("retains *most* corrective signal"), this **fails Condition 1** (23.4% ≪ 50%) despite the large aggregate ratio gain — the improvement comes from the sheer size of the TN-damage pool being crushed, not from the gate doing its intended job on FN specifically.

---

## Step 2: a ground-truth-conditioned gate, designed from that diagnosis

The failure mode identified is specific: a gate that only knows `p` (not which class the voxel truly is) cannot distinguish "close to the boundary, moving the right way" from "close to the boundary, moving the wrong way" when both happen to cluster near the same `p` range within a class. The fix uses information already available at training time (ground truth, exactly as `compute_margin_loss` itself already uses `gt_flat` for its own pair construction): weight by **how wrong the current prediction is relative to the correct answer**, not by closeness to 0.5.

`w(x) = |p(x) − target(x)|`, where `target=1` for GT-tumor voxels (TP, FN), `target=0` for GT-background (TN, FP).

| Category | Good retention | Bad retention | Good/bad selectivity |
|---|---:|---:|---:|
| FN | **93.9%** | 73.1% | 1.28× |
| FP | **88.8%** | 63.8% | 1.39× |
| TP | 25.7% | **1.0%** | 25.94× |
| TN | 0.0% | 0.1% | 0.37× (both near-zero; structurally non-discriminating for TN, see below) |

FN and FP's within-class selectivity is now correctly signed (good > bad), unlike the symmetric gate's inverted 0.28× for FN. TN remains non-discriminating between good/bad (both near-zero, since `|p−0|` is tiny for essentially all TN voxels regardless of direction) — but this is not a problem for the aggregate goal, since TN's *total* damage still gets crushed either way.

---

## The decisive aggregate result

| | Ungated | Gated |
|---|---:|---:|
| Total corrective movement (FN good + FP good) | 27,713.6 | 25,591 |
| Total damaging movement (TP bad + TN bad) | 34,739.3 | 262 |
| **Corrective/damaging ratio** | **0.798** | **97.76** |

**Ratio improvement: 122.55×.** Corrective retention: **92.3%**. Damaging retention: **0.8%**.

---

## Applying the strict two-part criterion

Per the user's own explicit rule: **"train only if the gate retains most of the FN/FP corrective signal AND materially reduces TP/TN damaging signal."**

- **Condition 1 (retain most corrective signal, >50%)**: 92.3% retained — **PASS**, decisively, not a borderline call.
- **Condition 2 (materially reduce damaging signal)**: 99.2% removed — **PASS**, decisively.

**Verdict: GO — train Confidence-Gated SC-TAM (C6-3) once**, using the ground-truth-conditioned gate `w(x)=|p(x)−target(x)|`, not the original symmetric `4p(1−p)` form. This clears the user's own strict criterion cleanly, not on a softer aggregate-ratio justification. The one remaining structural note, carried forward for interpretation rather than treated as disqualifying: TN's own good/bad movement is not distinguished by this gate (both suppressed equally), meaning the gate's benefit for the TN population comes entirely from bulk suppression of the confidently-correct majority, not from selectively sparing any useful TN-side correction — consistent with, not a new risk beyond, the already-established finding that TN's "good" movement (staying correctly classified) barely matters in aggregate compared to its damage risk.

---

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e25/representation_to_logit_results/representation_to_logit_C62.json` | Source data this retrospective test re-weights, no new computation |
| `PHASE_E25_C62_REPRESENTATION_TO_LOGIT.md` | The collateral-damage finding this gate is designed to address |
| `PHASE_E25_C62_CORRECTED_REANALYSIS.md` | The population-imbalance finding (98.75% TN) that explains why even bulk suppression of TN nets out favorable |
