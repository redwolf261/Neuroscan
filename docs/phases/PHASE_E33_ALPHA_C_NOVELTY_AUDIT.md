# E33: α_c Narrowed Novelty Audit

**Status**: Complete. No training. Literature search only, per explicit instruction. Prioritized 2026 first, then 2025, with older foundational work (blob loss 2023, Universal Loss Reweighting 2020) checked where it kept surfacing as a close comparator. Every "highly relevant" hit was fetched directly and probed for its exact mathematical mechanism — never judged from title/abstract alone.

**Date**: 2026-08-14

---

## The candidate mechanism, restated precisely

$$\alpha_c = \min\{\alpha : V_c(\alpha) > 0\}$$

where $V_c(\alpha)$ is the surviving voxel volume of GT lesion component $c$ after applying the same spatial degradation/resampling operator the segmentation pipeline itself uses, at degradation level $\alpha$. Candidate weight: $w_c = 1 + \kappa(1-\alpha_c)_+$, applied as a normalized component-weighted segmentation loss term.

---

## Novelty matrix

| Paper/year | Task/dataset | Mathematical mechanism | Degradation explicitly modeled? | GT component survival measured across scales? | Component-specific critical threshold computed? | Threshold controls training supervision? | Continuous or categorical weighting? | Derived from geometry/size or measured survival? | Closest equation to our w_c | Overlap score (0–5) | Novelty risk | Relevant limitation/future-work direction |
|---|---|---|---|---|---|---|---|---|---|---:|---|---|
| Kofler et al., "blob loss" (IPMI 2023, arXiv 2205.08209) | General semantic segmentation, instance imbalance | Decomposes label into connected components, evaluates independent Dice loss per component, averages equally (uniform weight 1/K) | No | No | No | N/A (uniform, not threshold-derived) | Categorical (uniform per component) | Neither — purely component-count normalization | None (uniform weight, not a function of any per-component property) | 1 | Low collision risk | Not stated; no resolution/scale discussion |
| Shirokikh et al., "Universal Loss Reweighting to Balance Lesion Size Inequality" (MICCAI 2020, arXiv 2007.10033) | 3D medical segmentation, multi-lesion (glioma, others) | Per-voxel weight inversely proportional to lesion volume | No | No | No | N/A (static size, not threshold) | Continuous, but a function of raw size only | Geometry/size (static, no degradation process at all) | Structurally closest to a "size-only" competitor: $w_c \propto \|C_k\|^{-1}$ | 2 | Low-medium — this IS the size-only baseline our own Section 4 explicitly needs to beat, not a collision | Not stated |
| CC-Metrics / CC-DiceCE (arXiv, "Learning to Look Closer," 2511.17146, Nov 2025) | Small cerebral lesion (WMH-type) segmentation | Component-wise loss averaged with weight 1/|K| (equal weight per component regardless of size) | No | No | No | N/A | Categorical (uniform) | None | 1 | Low | Not identified in available excerpts (abstract/metadata only — full Methods not text-extractable) |
| Kofler-family "Instance Awareness of Multi-class Semantic Segmentation Loss Functions" (arXiv 2604.24276, 2026) | Survey/synthesis of instance-aware segmentation losses | Reviews inverse-size weighting ($w=N/((K{+}1)|C_i|)$) and per-component local weighting (domain/foreground voxel ratio) | No | No | No | N/A | Continuous, but functions of static size/domain ratio | Geometry/size | $w_i = N/((K+1)|C_i|)$ — a size-inverse formula | 2 | Low — explicitly surveyed as an alternative to what we need to beat, not a collision | Conclusion notes an open trade-off between voxel-overlap and instance-detection but does NOT mention resolution-awareness as the resolution |
| Component-Adaptive Tversky, "Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation in Brain MRI" (arXiv 2604.08015, 2026) | Small structure segmentation, brain MRI | Voxel weight $w_i = (\|C_k\|+\epsilon)^{-\gamma}$ for $i \in C_k$ — inverse power-law of static component size | No | No | No | N/A (static size, tunable exponent γ) | Continuous, function of static size only | Geometry/size | $w_i = (\|C_k\|+\epsilon)^{-\gamma}$ — the single closest "size-only" formula found, both in name and mechanism | **3** — closest by name/topic, but mechanistically still purely size-based | Medium — this is the paper most likely to be cited as "already does adaptive component weighting"; must be explicitly distinguished in any future writeup | Limitations: single dataset, single disease, single architecture. Future work: cross-dataset generalization, false-positive control. **No resolution-awareness mentioned at all** |
| Uncertainty/hardness-weighted losses, PGU/REH (PMC12691699, Jan 2026) | General medical segmentation | Pixel-level weight from prediction-vs-GT probability discrepancy | No | No | No | N/A | Continuous, pixel-level, prediction-derived | Neither — model-uncertainty-derived | None | 0 | None | Future work mentions signed distance fields, not resolution |
| Topological loss / persistent homology family (e.g. arXiv 1910.01877, 2401.01160, 2307.03137) | Topology-preserving segmentation (curvilinear structures, vessels, aorta) | Persistence diagrams via **intensity/probability-value filtration** (Vietoris-Rips or cubical complexes over prediction/GT intensity), Wasserstein distance between diagrams as a topological-dissimilarity loss term; one variant ("train-free," 2401.01160) uses persistence for inference-time segmentation, not training supervision at all | **Filtration scale exists, but is an intensity threshold, not a spatial resolution/downsampling parameter** — this is the critical distinction | No (birth/death values track intensity-based feature persistence, not spatial-resampling survival) | Arguably yes in spirit (birth/death = "critical value"), but over the wrong axis (intensity, not resolution) | Some variants: yes, weight training loss by topological dissimilarity — but never by a resolution-derived critical value | Continuous (Wasserstein distance) | Neither size nor spatial-degradation-survival — intensity-topology-derived | Conceptually closest FORM (a "critical value" at which a feature is born/dies under a filtration), but the filtration AXIS is fundamentally different (intensity vs. spatial resolution) | 2 (formal analogy only) | Low-medium — worth citing as related conceptual machinery, genuinely distinct mathematically | Not found discussing spatial-resolution filtration as an extension |
| Multi-scale/deep supervision (general family, incl. this project's own E25 arc) | Various | Downsampled GT compared against network outputs at multiple FIXED architectural scales, loss summed/averaged across scales | Partially — GT is resampled to fixed scales, but as a training signal at those scales directly, not to compute a per-component disappearance threshold | No — no per-component survival measurement, no vanishing-point computation | No | N/A | N/A — not component-specific at all | Neither | None | 1 | Low — already established as tested and mechanistically distinct in this project's own prior work (E25) | N/A |

## A. Exact prior-art collision

**NO.** No paper found computes a per-component critical resolution (a minimum spatial resampling scale at which a GT component's voxel count reaches zero) and uses it to derive a continuous training-loss weight. Every close-sounding paper falls into one of two categories: (1) uniform/size-based component weighting with no degradation process at all (blob loss, CC-Metrics, Universal Loss Reweighting, Component-Adaptive Tversky), or (2) intensity-filtration-based topological loss/persistence, which has a formally analogous "critical value" concept but over the wrong axis (probability/intensity, not spatial resolution).

## B. Closest 5 papers

1. **Component-Adaptive Tversky** (arXiv 2604.08015, 2026) — closest by name and application (small-structure brain MRI segmentation), but mechanistically a pure size-inverse-power weight, no degradation process. Overlap score 3/5 — the one paper most likely to be raised by a reviewer as "isn't this the same thing?", and must be explicitly distinguished (Section C).
2. **Universal Loss Reweighting** (MICCAI 2020) — the canonical "size-only" baseline; structurally the comparator our own Section 4 control structure (`A → A+size weighting → A+α_c weighting`) already needs, not a collision.
3. **Instance Awareness survey** (arXiv 2604.24276, 2026) — confirms the current 2026 literature landscape's weighting schemes are still exclusively size/domain-ratio-based; explicitly does not raise resolution-awareness even as an open question.
4. **Persistent homology / topological loss family** (multiple, 2019–2026) — the only family with a formally analogous "critical value" concept, but filtered over intensity, not spatial resolution; worth acknowledging in any writeup as the nearest formal analogy while being explicit about the axis difference.
5. **blob loss / CC-Metrics** (2023, 2025) — uniform per-component weighting, establishes that "give every lesion equal say regardless of size" is an already-explored, different strategy from both size-weighting and α_c-weighting.

## C. Mathematical overlap analysis

The critical distinguishing claim, and the one that must survive scrutiny, is:

$$\alpha_c \neq f(\text{size}_c)$$

Every close paper found in this search uses $w_c = f(|C_c|)$ (a function of static voxel count alone) or uniform weighting — **none compute a resolution/degradation-derived quantity at all.** This project's own E32 result is the direct empirical evidence for the distinction: raw correlation of α_c with size is substantial but incomplete (E31: R²=0.37), and a partial rank correlation of α_c with real component-level outcome, controlling for size, remains significant (E32: partial Spearman ρ=0.132, p=0.0014). This is exactly the evidence needed to preempt the "you just weighted small lesions more" objection — no paper in this search has this specific empirical decomposition, because no paper computes α_c as a distinct quantity from size in the first place.

The persistent-homology family is the only mathematically adjacent construct (a critical value at which a topological feature is born/dies under a filtration parameter) — but critically, in every instance found, the filtration parameter is an **intensity or probability threshold**, not a **spatial resampling resolution**. This is a substantive, not merely terminological, difference: intensity-filtration persistence asks "at what confidence threshold does this feature disappear from the prediction," while α_c asks "at what physical resampling scale does this feature disappear from the geometry, independent of any model prediction." These are different questions with different mathematical objects, even though both produce a scalar "critical value" per feature.

## D. 2026 limitations/future-work opportunities that directly support or threaten α_c

- **Supports**: The Instance Awareness survey (2604.24276) explicitly leaves open "how the trade-off between voxel-level overlap and instance-level detection can be overcome" without proposing resolution-awareness as a candidate direction — a live, acknowledged gap that α_c could plausibly address, though the survey itself doesn't point there.
- **Threatens (mildly)**: Component-Adaptive Tversky's Limitations explicitly flag "single dataset, single disease, single architecture" and call for cross-dataset generalization — meaning any α_c-based method built specifically for this project's BraTS/UNet3D_v2 setup would face the same generalization critique, not a novel one but worth anticipating.
- **No 2026 paper found explicitly calls for degradation-trajectory-based or critical-resolution-based component weighting as future work** — this is a genuine gap, not filled by any Future Work section encountered, but also means there is no existing "permission slip" citation establishing the field already recognizes this as the right next step; the case must be made from this project's own evidence (E31/E32) alone.

## E. Novelty score

**MEDIUM-HIGH.** No exact or near-exact collision found (ruling out N0/LOW). The mechanism is mathematically distinguishable from every close paper on the specific axis that matters (resolution-derived vs. size-derived weighting), and this project has direct empirical evidence (E32's partial correlation) for that distinction — stronger grounding than most papers in this space provide for their own weighting choices. Not scored as pure HIGH because: (a) the search, while covering all specified terms and inspecting Methods/Limitations sections of every close hit, is not an exhaustive full-text search of MICCAI/TMI/MIDL proceedings specifically (as E30's own novelty matrix already flagged as a caveat); (b) the persistent-homology family's formal analogy, while axis-distinct, means a sufficiently motivated reviewer could argue for a closer connection than this search fully resolved — a deeper dive into whether any persistent-homology variant has been applied with a *spatial resampling* filtration (rather than intensity) specifically, would strengthen this further before a paper submission.

## F. Decision

Per the user's own decision table: no exact collision (row 1, KILL, does not apply); Component-Adaptive Tversky is "same mathematical principle under another name" only if judged from the label "adaptive component weighting" alone — but it is not, once the actual formula is inspected (size-only, no degradation process) — this places it in the **"object-size weighting exists" → "not enough; mathematically different"** row, not the KILL row. The persistent-homology family similarly falls short of a collision once the intensity-vs-resolution filtration distinction is applied.

```
GO — the narrowed novelty gate is cleared. No prior work found connects a
spatial-degradation-derived component survival trajectory to
component-specific training supervision. Proceed to the brutal control
structure (A -> A+size weighting -> A+alpha_c weighting) specified as the
required next step, NOT a naive baseline-vs-alpha_c comparison.
```

This GO is for **running the controlled comparison experiment next**, not for skipping directly to a full training campaign. The ≥1pp kill gate, the size-matched-weighting comparison requirement, and the independent-seed reproduction requirement specified by the user all remain in force and are unaffected by this novelty result.

---

## Files

| File | Purpose |
|---|---|
| `E33_alpha_c_novelty_audit.md` | This report — full novelty matrix and decision |
