# E165–E167 — The closure evidence chain

**Date**: 2026-09-15
**Status**: Complete. This document terminates the architecture-novelty search.
**Scope**: E165 (per-stage rank), E166 (task-equivalent transformation audit), E167
(boundary-residual decomposition), plus the two route audits and the external corroboration.

---

## The scientific claim

Not "we could not improve Dice." The claim is:

$$\boxed{\text{The remaining Dice residual was experimentally decomposed into } \begin{cases}\text{information-limited error}\\ \text{boundary-localised error}\end{cases}}$$

with the two routes to the pre-registered ≥1pp target **independently closed by measurement**.

### The arithmetic that defines the two routes

Cohort 3-region mean **0.8595**; the bar is 0.8695. Verified from
`e160/E160_L_per_subject.json` (n=125, native resolution):

| Route | Target | Requirement |
|---|---|---|
| **(a)** | the 15-subject catastrophic tail (ET 0.227 / TC 0.186, but **WT 0.824**) | lift ET&TC to ~0.31 on all 15 |
| **(b)** | the other 110 (ET 0.8995 / TC 0.9380 / WT 0.9239) | **+2pp each** = eliminate **20% of the ET residual, 32% of the TC residual** |

There is no third route: +2pp on the 110 yields +1.14pp overall; nothing else reaches the bar.

---

## The chain

| # | Question | Result | Verdict |
|---|---|---|---|
| **E165** | Is representational capacity uniformly demanded across the U-Net? | Monotone 35× gradient: enc1 R\*/C **0.613** (R\*=19.6/32) → bottleneck **0.018** (R\*=4.5/256); Wilcoxon p=2.0e-22 | **RANK_LIMITED** (pre-registered) |
| **E166** | Is "replace a block by matching task consequence, not features" novel? | Occupied: DCP (NeurIPS 2018), BERT-of-Theseus (EMNLP 2020), *Local to Global* (ACL 2026), ReplaceMe, Structural Scalpel | **🔴 OCCUPIED — verdict #6** |
| **Route (a)** | Is the tail's residual caused by missing input information? | E142: zeroing t1c collapses ET **0.8433 → 0.0015** (−56.15pp) while WT holds at 0.8927 | **Information-absent → CLOSED** |
| **E167** | Is there enough correctable *interior* error on the 110 for +2pp? | ET: **93.9%** of weighted error within ≤2 voxels, **98.6%** within ≤3; interior **1.4%** | **Boundary-localised → CLOSED** |
| **Ceiling** | Is the boundary residual plausibly at the annotation limit? | Menze inter-rater ET **median 77%** (rater-vs-rater 74–85%); our model **0.900** | **Strong external support** |
| **Plateau** | Is saturation specific to our implementation? | *nnU-Net Revisited* (MICCAI 2024): "scores on BraTS21 are **saturated**"; winning ET 0.8245 (2018) → 0.8203 (2020) | **External corroboration** |

---

## E165 — per-stage task-relevant rank (n=125)

| Stage | C | R* mean | R*/C mean | ≤4 | >half |
|---|---:|---:|---:|---:|---:|
| **enc1** | 32 | 19.62 | **0.613** | 1% | **31%** |
| enc2 | 64 | 10.50 | 0.164 | 53% | 3% |
| enc3 | 128 | 6.78 | 0.053 | 67% | 0% |
| **bottleneck** | 256 | 4.50 | **0.018** | 90% | 0% |
| dec1 | 32 | 4.93 | 0.154 | 82% | 1% |

**Independent replication of E147**: bottleneck R\* mean 4.50 vs E147's 4.95; R\*/C 0.0176 vs
0.0194; 90% at ≤4 vs 84%. Different subject set (125 vs 88), different script, same answer.

**This corrects a claim made earlier in the session.** The assertion that low task-relevant rank
is *universal* (and therefore uninteresting) is **refuted by our own measurement**. E147's "the
network runs at ~1.9% of capacity" is a property of **the bottleneck specifically**, not of the
network.

**Confounds that must be stated with any use of this result:**
1. **Budget, not demand.** enc1 has 32 channels, the bottleneck 256. Absolute ranks are 19.6 vs
   4.5 — only ~4× apart — while the *fractions* are 35× apart. The gradient may substantially
   reflect channel-budget allocation across depth rather than task demand.
2. **Resolution.** enc1 is 128³, the bottleneck 8³. Channel-PCA at those two resolutions is not
   the same measurement.
3. **Dice-neutral.** This is a diagnostic. E147's separate finding — allocation sign runs
   backwards, no capacity scarcity — is untouched at the bottleneck.

Sanity: checkpoint identity asserted; dormant hook an **exact** no-op; rank=C truncation max
diff **0.000e+00**.

## E166 — task-equivalent transformation: OCCUPIED (verdict #6)

The proposal was to replace block $F$ with cheaper $G$ fitted so $D(G(Z)) \approx D(F(Z))$
(task consequence, output space) rather than $G(Z) \approx F(Z)$ (features). **That distinction
is the standard motivating paragraph of a pruning subfield**, not a contribution:

- **Discrimination-aware Channel Pruning** (NeurIPS 2018, 1810.11809): *"optimizes the
  reconstruction error but ignores the discriminative power of channels."* Its TPAMI extension
  **measures** the reconstruction-vs-task-loss divergence.
- ***From Local to Global: Revisiting Structured Pruning Paradigms*** (ACL 2026, 2510.18030):
  *"the dominant local paradigm is task-agnostic: by optimizing layer-wise reconstruction rather
  than task objectives…"* — named as an established taxonomy being **revisited**.
- **BERT-of-Theseus** (EMNLP 2020, 2002.02925): replaces *individual modules inside the network*
  with cheaper successors trained on **task loss only**, explicitly refusing feature matching.

It also fails independently on arithmetic: an approximation constraint on the existing output
targets **+0.00pp** by construction, and the pre-registered bar is +1pp.

**Unclaimed remnant, never tested**: in skip-connected networks, feature-space error may be a
*systematically* bad proxy for task error, because skips route features around the block so much
of $F$'s output never reaches $D$. The audit could not find this stated anywhere. It is a
**diagnostic hypothesis**, not an accuracy method, and it remains untested.

## E167 — boundary-residual decomposition (n=110)

| Region | Dice | Err ≤2vox (weighted) | GT ≤2vox | **Enrichment** | Interior (d>3) |
|---|---:|---:|---:|---:|---:|
| **ET** | 0.8995 | **0.939** | 0.875 | **1.13** | **1.4%** |
| TC | 0.9380 | 0.811 | 0.490 | **2.11** | 2.8% |
| WT | 0.9239 | 0.740 | 0.435 | **1.96** | 15.0% |

Pre-registered rule: CLOSED if ET err_frac_within2 ≥ 0.80. **Result 0.939 → CLOSED.**

**The geometry control is load-bearing and was added deliberately.** A thin structure is
boundary-adjacent by geometry alone, so the raw fraction alone is uninterpretable — the smoke
test showed a subject whose GT was **99.9%** boundary-adjacent, where an error fraction of 94.9%
means enrichment *below 1*. Reporting
$\text{enrichment} = \text{err frac} / \text{GT frac}$ prevents closing the branch on an
artifact — the failure mode that produced E153's normalisation artifact and four auto-classifier
mislabels (E77, E80, E82, E104b).

The control **strengthens** the reading for TC and WT: their GT is only 44–49% boundary-adjacent
yet 74–81% of their error is — enrichment ≈ **2×**, i.e. genuine concentration at the interface,
not geometry. ET's milder 1.13 is expected, since enhancing tumour is a thin shell.

---

## What is and is not established

**Established**: the residual is **boundary-concentrated**.

$$\text{residual} \rightarrow \text{boundary-concentrated} \qquad \textbf{not} \qquad \text{residual} \rightarrow \text{incorrect labels}$$

**Not established**: that those labels are wrong. A genuinely better boundary model would also
produce boundary-concentrated error. Distinguishing the two requires **rater-level evidence**
(repeat or multi-rater annotation), which we do not have. Notably, BraTS 2024 *ran* a
dual-annotation experiment on its test set and published the method but **not the agreement
numbers** — the single datapoint that would settle this.

What makes the closure stand is the **conjunction**: boundary-localised residual + our 0.900
already exceeding the published ET inter-rater median of 0.77 + a field-wide plateau since 2018.
No single leg carries it.

---

## Closure statement (the defensible wording)

> **Under the evaluated input, annotation, and architectural regime, the study found no
> experimentally supported route to the pre-registered ≥1 percentage-point improvement. The
> residual error was concentrated at the ground-truth boundary, while the principal
> information-removal experiment demonstrated that the dominant enhancing-tumour signal was
> already absent from the input for the failing cohort. Together with published inter-rater
> agreement and the observed BraTS performance plateau, these findings motivated closure of the
> architecture-novelty search.**

Explicitly **not** claimed: "no further improvement is possible." That is scientifically
indefensible and is not what was measured.

---

## What E165–E167 are for

They are **the mechanistic investigation that justified terminating the architecture search** —
not a final algorithm. None of them is a contribution on its own; together they are the reason
the search stopped, and the evidence that it stopped for a measured cause rather than exhaustion.

The next question is therefore different in kind:

$$\boxed{\text{If architecture cannot supply the missing } +1\text{pp, what can?}}$$

Candidates outside the closed space (none audited, all require the E163 entry conditions —
domain constraint, infrastructure reuse, timeframe/deliverable — to be answered first): the
annotation/label regime itself; the evaluation metric; the input acquisition; or a different
task family where the bar is meaningful and prior-art density is lower.

## Artifacts

`experiments/exp_e12_eggo_m/e165/` — `run_e165_per_stage_rank.py`, `E165_summary.json`,
`E165_per_subject.{json,csv}`
`experiments/exp_e12_eggo_m/e167/` — `run_e167_boundary_residual.py`, `E167_summary.json`,
`E167_per_subject.{json,csv}`
