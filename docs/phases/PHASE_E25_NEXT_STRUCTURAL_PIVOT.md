# Phase E25: Structural Pivot — Deep Multi-Scale Supervision for Small-Lesion Detection

**Status**: 🔵 Specification complete, implementation not yet started. This document exists to lock the hypothesis, mechanism, and decision gates *before* any code is written, per the explicit instruction not to manufacture another experiment inside the exhausted C6 diagnostic loop.

**Date**: 2026-08-12

---

## 1. Current locked state

| Condition | Best validation Dice | Δ vs. A |
|---|---:|---:|
| A — locked baseline | **0.9063** | — |
| C6-2 — signed SC-TAM | 0.9030 | −0.33pp |
| C6-3 — gated SC-TAM | 0.9026 | −0.37pp |
| **Required** | **0.9183** | **+1.20pp** |

## 2. Exhausted C6 conclusions (not re-derived here)

SC-TAM's local representation-space mechanism can be made to produce correctly-directed movement on error voxels, and that directionality can be made highly selective (C6-3's retrospectively-validated gate: 92.3% corrective retention, 99.2% damage removal, 122.55× ratio improvement) — and neither the base mechanism nor its most promising refinement produces a task-level Dice improvement. Both land at 0.902–0.903, indistinguishable from each other and slightly below the untouched baseline. This closes the family of interventions that act on `dec1`'s representation geometry via an auxiliary margin loss. No further variant of this family is pursued.

---

## 3. Failure model (built from existing 125-subject validation data only — no new inference, no new training)

Source: `experiments/exp_e12_eggo_m/e25/spatial_error_analysis_results/spatial_error_analysis_C62_vs_A.json`, condition A's own per-subject records (`dice_a`, `cc_a`, `gt_lesion_volume`) — re-analyzed with a different question than C6-2.9 originally asked. C6-2.9 asked "where does A vs. C6-2 differ" and found no dominant pattern. This failure model asks a different question: **where does A itself, the better of the two locked results, still fail** — independent of any C6 comparison.

### A. What is already working

A's mean per-subject Dice is 0.887 (median 0.917), and 80/125 subjects (64%) score ≥0.90. For the majority of subjects with a small number of reasonably-sized lesions, the baseline segments cleanly. Slice-position and boundary-distance decompositions (C6-2.9) show no depth- or boundary-specific weakness in A's own error pattern beyond the expected concentration at lesion edges (~11% error rate at the boundary layer, <0.1% in the interior) — ordinary, not anomalous, for this task.

### B. What constitutes the remaining loss — quantified, not assumed

| Quantity | Value |
|---|---:|
| corr(mean per-subject lesion-component size, fraction of that subject's components missed) | **−0.601, p<0.0001** |
| Missed-component rate, subjects below median component size | **43.1%** |
| Missed-component rate, subjects above median component size | **8.2%** |
| Subjects with ≥1 fully missed lesion component | **55/125 (44.0%)** |
| Mean Dice, subjects with ≥1 missed component | 0.849 |
| Mean Dice, subjects with full component detection | 0.917 |
| Share of A's total Σ(1−Dice) shortfall attributable to missed-component subjects | **58.9%** |

This is a strong, previously-unweighted signal. C6-2.9's own connected-component analysis already contained these numbers (`n_missed_components`, `gt_lesion_volume`, `cc_a['n_gt']`) but the phase's framing (A vs. C6-2 comparison) never asked "is A's *own* error concentrated in small-lesion detection" — this failure model asks and answers that question directly from the same, already-collected data.

**Ruled out, based on the same existing data, not assumed**: boundary-distance error rate showed no anomalous concentration in A specifically (C6-2.9, Result 2 — A's own per-bin error rates are unremarkable, the comparison there was only ever about the *A-vs-C6-2 delta*, not A's absolute rate). Slice-depth showed nothing (Result 1). Fragmentation (lesion *count*, independent of size) showed only a weak, non-significant correlation with Dice (r=−0.169, p=0.059) — it is lesion **size**, not lesion **count**, that is the strong, significant signal.

### C. Highest-leverage failure mode, evidence-driven

**Small-lesion whole-component detection failure.** Not a boundary-placement problem (existing detected lesions are segmented reasonably — mean component Dice 0.608 across all detected+missed components pooled, but this number is dragged down by the missed components themselves, which score 0 by definition; the failure is binary detection, not boundary precision on lesions that are found at all). Not a calibration or FP problem (this failure model did not find FP-driven degradation to be the dominant signal — FP components are comparable between A and C6-2 per C6-2.9's Result 4). The evidence points specifically at: **small ground-truth lesion components are disproportionately missed entirely, and this single failure mode accounts for more than half of A's total shortfall from a perfect score.**

---

## 4. Candidate structural mechanism: deep (multi-scale) supervision

Add two auxiliary segmentation heads, one reading `dec3` (128 channels, D/4 resolution) and one reading `dec2` (64 channels, D/2 resolution) — both currently unsupervised intermediate decoder stages that only receive gradient indirectly, filtered entirely through `dec1`'s own downstream loss. Each auxiliary head is a `Conv3d(C,1,kernel_size=1)` + `Sigmoid`, structurally identical in form to the existing `seg_head`, supervised against the ground-truth mask downsampled (via `F.avg_pool3d` or `F.interpolate`, decided at implementation time with a unit test comparing both) to that head's own resolution.

```text
L = L_seg(dec1) + mu*L_boundary + lambda_ds3*L_seg(dec3, GT@D/4) + lambda_ds2*L_seg(dec2, GT@D/2)
```

(SC-TAM's margin term is dropped entirely — not combined with this mechanism, per the ablation-compatibility requirement and to keep this test's causal attribution clean.)

## 5. Why this is genuinely structurally different from SC-TAM

SC-TAM (and its C6-2/C6-3 variants) operates entirely on `dec1`'s existing representation, after the network has already committed to whatever it will produce at full resolution — it can only reshape *where in embedding space* an already-computed voxel representation sits, never *whether the network detects the lesion's existence at all* at an earlier, coarser stage where a small lesion might still be spatially resolvable before being diluted by further upsampling/pooling interactions. This candidate changes **which layers receive direct supervision gradient**, not the *geometry* of any single layer's output — a different point in the network entirely, and a different kind of intervention (multi-resolution architectural supervision vs. representation-space metric learning). It also does not touch `dec1`'s embedding space at all, so it is structurally orthogonal to — not a variant of — everything tested in the C6 family.

## 6. Causal hypothesis (falsifiable)

**Observed failure → intervention → expected measurable consequence → Dice improvement:**

Small lesions are missed because the network's only direct segmentation-loss gradient arrives at full resolution (`dec1`), after 3 rounds of downsample/upsample have already diluted a small lesion's signal without any earlier corrective pressure. Adding direct supervision at `dec3` (D/4) and `dec2` (D/2) forces the network to represent small structures explicitly at coarser scales, where they occupy a proportionally larger fraction of the receptive field and are less prone to being lost. If this hypothesis is correct: the missed-component rate for small lesions specifically should drop, while large-lesion performance (already good) should be roughly unaffected — and validation Dice should improve, with the improvement concentrated in exactly the subject population this failure model identified (the 44% with ≥1 missed component under A).

**Falsifiable prediction, stated explicitly**: if deep supervision is added and trained, but the missed-component rate for small-lesion subjects does not improve relative to A, the hypothesis is wrong regardless of what happens to aggregate Dice — this is the Gate 1 mechanistic check (Section 8 below), checked before any Dice-based conclusion is drawn.

---

## 7. Experimental design

### Hypothesis
Deep supervision at `dec3`/`dec2` reduces the small-lesion missed-component rate relative to A, and this reduction produces a validation Dice improvement.

### Mechanism
Two new auxiliary heads (`aux_head3` on `dec3`, `aux_head2` on `dec2`), each a 1×1×1 `Conv3d`+`Sigmoid`, each supervised by the same `FocalTverskyLoss` already used for the primary head, against a resolution-matched downsampled ground truth. Auxiliary losses are added to the total objective with their own coefficients (`lambda_ds3`, `lambda_ds2`), calibrated (not guessed) before the real run — see Section 9.

### Expected failure signature
If the mechanism is not working: missed-component rate for small-lesion subjects stays at or near A's own 43.1% baseline; auxiliary losses fail to decrease over training (indicating the intermediate representations genuinely cannot support this level of supervision, a real, informative negative result); or Dice regresses relative to A (auxiliary supervision destabilizes the shared trunk).

### Expected success signature
Missed-component rate for small-lesion subjects (below-median component size) drops measurably from A's 43.1%, *before* or alongside any Dice movement — this is the required mechanistic evidence checked at Gate 1, independent of the final Dice number.

### Primary endpoint
Validation Dice (same locked protocol as A/C6-2/C6-3: full 125-subject validation set, per-volume Dice averaged across subjects, matching H4's methodology exactly).

### Secondary endpoints (hypothesis-relevant only)
Missed-component rate, split by lesion-size tercile (matching this failure model's own size-based split); component-level Dice for detected small lesions (distinguishing "detected but poorly segmented" from "fully missed"); large-lesion Dice (must not regress — a control against the mechanism helping small lesions at large lesions' expense).

### Controls
1. **A** — existing, locked, not retrained.
2. **New method (deep supervision only, no SC-TAM)** — the candidate.
3. **Ablation**: identical architecture/training run with `lambda_ds3=lambda_ds2=0` (auxiliary heads present in the architecture, contributing zero loss) — isolates whether any effect comes from the *auxiliary supervision itself* or merely from the additional head parameters/capacity being present. This is the minimum ablation needed to establish causality, per Section 6's own requirement not to run variants before the mechanism is confirmed.

C6-2 and C6-3 remain available as existing reference points (not rerun) but are not part of this experiment's own controlled comparison, since they used a different mechanism entirely (SC-TAM) — comparing this candidate against them is a secondary, contextual comparison only, not the primary ablation.

---

## 8. Decision gates (applied strictly, per the explicit instruction)

- **Gate 1 (mechanistic plausibility)**: does the small-lesion missed-component rate improve relative to A? If not — **STOP**. Do not proceed to interpreting Dice, do not train ablation variants beyond what's needed to confirm the negative result.
- **Gate 2 (task-level movement)**: if Gate 1 passes but validation Dice is statistically/practically indistinguishable from A (within the ~0.3–0.4pp noise band C6-2/C6-3 both fell into) — **STOP**. Do not spend a further phase explaining a small mechanistic-but-not-task-level effect, mirroring the exact C6-3 outcome this pivot exists to avoid repeating.
- **Gate 3 (positive signal)**: if Dice moves materially toward 0.9183 — run the ablation control (already specified above) to confirm the auxiliary *supervision* (not just added capacity) is the cause, and stop there — no further variant sweep.
- **Gate 4 (target proximity)**: if Dice ≥ 0.9183 — freeze the result, move to multi-seed confirmation (only now, not before), do not continue modifying the method.

---

## 9. Implementation plan (not yet executed)

1. Add `aux_head3`/`aux_head2` to `neuroscan_3d_v2.py` (a new subclass or direct extension, following the project's established `UNet3D_v2`-extends-`UNet3D` convention — requires explicit discussion of whether this needs its own `baseline_frozen_v3`-style lineage tag, matching the precedent set when the boundary head was added in E9/E11).
2. Calibrate `lambda_ds3`/`lambda_ds2` from measurement (same discipline as every other constant in this project — e.g. matching auxiliary-loss magnitude to `L_seg`'s own magnitude at initialization, not guessed).
3. Unit tests: auxiliary heads produce correctly-shaped output at their respective resolutions; downsampled-GT construction is verified against a hand-computed reference; gradient reaches `dec2`/`dec3`'s own parameters (not just `dec1`'s, confirming the mechanism actually changes what receives direct supervision).
4. Gate-5-style smoke test (finite losses, sane active behavior, checkpointing) before any real training, matching every prior phase's own discipline.
5. Single locked seed (seed=0, matching A/C6-2/C6-3) for the candidate and its ablation control — no multi-seed search per Section 9's own multi-seed discipline.

---

## 10. Expected outcomes

Stated before running anything, per the falsifiability requirement: missed-component rate for small-lesion subjects should drop from 43.1% toward something closer to large-lesion subjects' own 8.2% rate if the mechanism works as hypothesized; Dice improvement, if it occurs, should be disproportionately concentrated in the 55 subjects A itself already identified as having ≥1 missed component (a directly checkable, falsifiable sub-prediction, not just an aggregate number).

## 11. Actual results

*Not yet run. This document is the pre-registered specification; results will be appended here once the implementation (Section 9) and experiment (Section 7) are executed, not before.*

## 12. Final decision

*Pending Section 11.*

---

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e25/spatial_error_analysis_results/spatial_error_analysis_C62_vs_A.json` | Source data for the failure model (Section 3) — no new inference run |
| `PHASE_E25_C62_SPATIAL_ERROR_ANALYSIS.md` | The original analysis this failure model re-derives a different conclusion from (A-vs-C6-2 comparison found no pattern; A's own absolute failure composition, examined here for the first time, does) |
| `PHASE_E25_C63_GATE_RETROSPECTIVE_TEST.md`, C6-3 training results (this session) | The closed C6 family this pivot moves away from |
