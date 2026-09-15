# E161 — Assumption K prior-art audit: **OCCUPIED (🔴)**

**Date**: 2026-09-15
**Status**: Audit complete. No compute. K closed as a novelty candidate.
**Follows**: E159 (enumeration), E160 (L = SURVIVES)

---

## The question, as tightly as it can be posed

Not "has anyone done modality attention" — that is too broad and would kill K for the wrong
reason. The specific distinction under test was:

$$\text{modality absent} \;\neq\; \text{modality present but informationally useless}$$

and the architectural assumption:

$$f(x_1,x_2,x_3,x_4) \quad\text{vs}\quad f(x_1,x_2,x_3,x_4;\,I_1,I_2,I_3,I_4)$$

where $I_m$ is **subject-specific informativeness**, not missingness.

**Verdict: the distinction is explicitly named, explicitly targeted, and explicitly solved in
published work.** K is not a gap.

---

## The hits, in order of how directly they collide

### 1. CoReFuse-Med — *"When Fusion Fails: Corruption-Aware Rebalanced Fusion"* (arXiv 2609.10261)

**This is a direct collision with K's core claim.** Its stated contribution is, verbatim:

> "We identify a distinct and largely unexplored failure mode: fusion degradation caused by
> **modality-quality mismatch even when modalities are spatially aligned**."

That is K's premise — present-but-degraded, not missing — claimed as the paper's own novelty.
Worse for us, its diagnostic finding mirrors ours:

> "degraded modalities can exhibit substantial predictive impact despite comparatively weak
> gradient contributions" — i.e. they **interfere** rather than being ignored.

It is label-blind at inference, it is 3D MRI segmentation, and **it evaluates on BraTS.**

### 2. Evidence Fusion with Contextual Discounting (arXiv 2206.11739, MICCAI 2022)

Learns, per modality and **per class**, a discount rate quantifying that source's reliability,
then fuses by Dempster's rule. Evaluated on **BraTS 2021, 1,251 patients — the same dataset and
the same cohort size as this project.**

This is the formal, principled version of $f(x; I_1..I_4)$: $I_m$ is a learned reliability
coefficient per modality per class. The mathematical object K proposed already exists, with a
Dempster–Shafer treatment.

### 3. Sub-Region-Aware Modality Fusion (arXiv 2601.15734)

Learns attention weights $\alpha_{m,r}$ — per modality $m$, per tumour **sub-region** $r$
(NCR/ED/ET) — with region-specific parameters and softmax normalisation, motivated by exactly
the observation E142 rediscovered causally:

> "the necrotic core appears dark in T1c but bright in FLAIR... **enhancing tumor is bright in
> T1c**."

Our E142 finding (t1c owns ET, t2f owns WT) is treated here as **established background
knowledge that motivates the method**, not as a finding.

### 4. UAF-AIMM (Front. Neurosci. 2026) — *specifically the missing/weak t1c case*

Targets brain tumour segmentation **with missing contrast-enhanced T1**, using voxel-wise
predictive variance and learnable gating to fuse by estimated reliability, plus single-image
**test-time adaptation for per-case calibration**. Reports that improvement is "most pronounced
in the contrast-dependent enhancing tumor (ET) subregion."

That is our target region, our target modality, and per-case adaptation.

### 5. Supporting / adjacent

- OmniFuse — general fusion for **low-quality** medical data, explicitly framed around "the
  absence of informative modalities and **imbalanced clinically useful information across
  modalities**."
- Conformal Fusion Under Missing Modalities (2608.07183) — dynamic reliability weighting that
  "naturally handles cases where a view is **uninformative (low evidence)**."
- Modality-Specific Enhancement and Complementary Fusion (2512.09801); "Centering the Value of
  Every Modality" (ECCV 2024).

---

## Assessment against each axis I pre-registered

In E159 I wrote that three things might survive: the medical-segmentation instantiation, the
label-blind identifiability, and the present-vs-absent distinction. All three fail:

| Proposed distinguishing axis | Status |
|---|---|
| Present-but-uninformative ≠ absent | 🔴 CoReFuse-Med claims exactly this as its novelty |
| Per-subject informativeness $I_m$ | 🔴 contextual discounting; conformal reliability weighting |
| Per-region modality specialisation | 🔴 Sub-Region-Aware fusion ($\alpha_{m,r}$) |
| Label-blind at inference | 🔴 CoReFuse-Med is label-blind; UAF-AIMM adapts per case |
| The t1c/ET case specifically | 🔴 UAF-AIMM targets exactly this |
| BraTS, n=1251 | 🔴 contextual discounting uses the same dataset and cohort |

I predicted 🟡 partial hit, like E138. **The actual result is worse: 🔴 on every axis.**

---

## What still belongs to us (and what it is worth)

E142's causal modality ablation is a **clean measurement**: zeroing t1c collapses ET
0.8433 → 0.0015 (−56.15pp) while WT holds at 0.8927; zeroing t2f collapses WT 0.9262 → 0.6455.
Near-independent pathways, established by intervention rather than by attention weights.

That is a good diagnostic and it is **corroborating evidence for a known phenomenon**, not a
discovery. Sub-Region-Aware fusion states the same specialisation as motivating background.

---

## Verdict

$$\boxed{\textbf{K = 🔴 OCCUPIED. Do not design against it.}}$$

Adding K to the killed-by-prior-art list alongside E (query-based segmentation /
MaskFormer–Mask2Former lineage).

---

## What this means for the assumption programme

Three enumerated assumptions have now been audited and all three are closed:

| | Assumption | Verdict |
|---|---|---|
| E | Dense voxelwise prediction is necessary | 🔴 MaskFormer/Mask2Former/SAM |
| K | Modalities are equally informative per subject | 🔴 this audit |
| L | Regime-1 findings transfer | 🟢 **SURVIVES** (E160) — but the phenomenon is orthogonal to the current target |

**The pattern is worth naming.** Every candidate generated from *our own failure diagnosis*
has turned out to be a known phenomenon with a published method attached. That is not bad luck
four times over — it is what it looks like when a problem is well studied and the remaining
headroom is in a 15-subject tail.

Before auditing I (nested ET⊆TC⊆WT), it is worth stating plainly: hierarchical/nested
segmentation has substantial prior art too, and on the evidence of E, K, and E138 the base rate
for "our idea is already published" in this area is very high. I is likely 🔴 as well.

The honest options are now:
1. Audit I, expecting 🔴.
2. Accept that the contribution is **not** a novel algorithm, and write up what the project
   actually has: a rigorous causal-diagnostic methodology, ~20 pre-registered kills with
   caught bugs, the E56 measurement correction, and E160's cross-regime confirmation.
3. Change the problem — a different dataset, task, or failure mode where the literature is
   thinner. This is a scope decision, not a research one, and it belongs to the user.

No GPU work is licensed by this audit.
