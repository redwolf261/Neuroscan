# E158 — Terminal State: E1–E157 Inventory Closed

**Date**: 2026-09-15
**Status**: Boundary document. No compute. Closes the existing-result search.
**Purpose**: Establish a hard line between the exhausted inventory and the assumption-hunting
phase, so no future session unconsciously resurrects a closed branch.

---

## The finding

$$\boxed{\text{The existing inventory is exhausted as a source of novelty claims.}}$$

### Qualification added 2026-09-15 after a full read of all 143 phase docs + 105 memory files

The inventory is exhausted **as measured**. But the full-corpus read surfaced a structural fact
that neither E156 nor E157 had stated, and it materially changes what "exhausted" means:

**Two incompatible measurement regimes are being treated as one project.**

| | Regime 1 (E1–E97, most of the arc) | Regime 2 (E128–E153) |
|---|---|---|
| Input | **FLAIR only** (1 channel) | 4 modalities |
| Label | **binary whole-tumor** | 3 regions (ET/TC/WT) |
| Resolution | 64³ whole-volume resize | 128³ patches |
| Baseline | per-subject **0.8842** | mean Dice **~0.89** across 3 regions |
| Failure mode | small lesions vanish in resize | **ET/TC catastrophic tail** |

The FLAIR-only, binary, 64³ setup was **a self-imposed simplification, not a scientific
baseline** — master report §1: "the other three... are not used, to keep the model and problem
simple." E70's own doc says the consequence outright, pre-registered before the numbers
existed: *"The single-modality restriction was a self-imposed project constraint, not a
scientific baseline."*

**Consequences, none of which were in E156/E158's first draft:**

1. **E70 MM's +1.75pp is not a "known technique that worked."** It is the measured cost of a
   handicap the project chose. That is a *stronger* disqualification than "not novel."
2. **Every pre-E70 pp claim is against a handicapped baseline.** E45's +1.44pp and E54's
   +1.39pp are both vs FLAIR-only 0.8842 (verified: 0.8986 − 0.0144 = 0.8842). Neither has
   ever been tested on a proper 4-modality baseline. Their failure to replicate is real, but
   their *effect sizes* are not comparable to Regime 2 numbers and must never be tabulated
   alongside them.
3. **The E48→E97 chain's target phenomenon may be regime-specific.** "Small lesions depend
   more on the bottleneck" was established entirely in Regime 1, where E29 showed the *median*
   lesion component vanishes to **zero voxels** under the 64³ resize. A large part of that
   chain may have been diagnosing a **preprocessing artifact** the project created, not a
   property of segmentation networks. E27 flagged this as "the single most important audit
   finding" — small lesions may lose information *before training starts* — and the project
   pivoted to resolution work but never retired the Regime-1 conclusions.
4. **Regime 2 has a completely different failure mode** (E133: ET/TC catastrophic tail, WT
   fine, 5/8 ET failures predict *exactly zero voxels*) which the Regime-1 small-lesion
   framing does not describe.

**This does not reopen any closed branch.** Every kill stands on its own evidence. But
"exhausted" should read: *exhausted within two separate regimes, with the older and larger
regime resting on a self-imposed handicap that was later removed.*

This is **not** "NeuroScan failed." It is:

> We searched the full inventory of positive and mechanistic results and found no defensible
> novel principle. Every candidate is either a known technique, an oracle artifact, a
> non-replicating effect, or a phenomenon whose principle is already published.

The value of this document is negative-space value: it prevents the project from repeatedly
renaming occupied ideas, which is the specific failure mode that consumed E44–E157.

---

## Closed ledger

| Track | Status | Reason | Evidence |
|---|---|---|---|
| **E70 MM** (4-modality input) | 🔵 | +1.75pp, 3-seed confirmed (std 0.27) — the project's one robust number, but multimodal input is textbook | E156 |
| **E15** (latent geometry / centroid push) | 🔴 | +3.93pp is **oracle** — GT-selected centroid, rank-1 decoder ($\nabla_z D = D(1-D)w$); mechanism collapsed under E151 | E156 |
| **E25 D4-only** (deep supervision) | 🔵 | +0.33pp, single seed, **none of 6 pairwise comparisons significant** (p>0.07); DS is standard published technique | E156 |
| **E45 D4+D8** | 🔴 | Seed 0 gave **+1.44pp per-subject, doubly significant** (p=0.041 / 0.00075) — a genuine comfortable >1pp. Did **not replicate**: seed 1 +0.36pp at p=0.588. 3-seed mean 0.8939 vs 0.8942 target | E156 |
| **E54/A96** (resolution) | 🔴 | Resolution's own contribution +0.38pp, CI **[−0.00, +0.79]** includes 0; 3-seed 0.8944 vs 0.8942. Early "+1pp" was modality gain misattributed | E156 |
| **E109/E71** (bottleneck self-knowledge) | 🔵 | Real phenomenon (ρ=+0.633 held-out, partial +0.791), but principle published as "predictive self-knowledge" (arXiv 2608.14894), model-space, superset of our interventions | E157 |
| **E126** (rank → $N_b$) | 🟢 *diagnostic* | Genuine causal finding, dose-response confirmed, safety-barred. **Not itself a contribution** | E126 |
| **E147** (demand → rank) | 🟡 | Generator unrecoverable (never committed); and its own evidence is **inverse** — kills allocation | E154 |
| **Adaptive computation** | ⚪ | **Not yet attempted** | — |

### Ledger correction

E45 is **🔴, not 🔵**. It is not merely "known + unstable" — it was subjected to the project's
mandatory 3-seed protocol and **failed against a pre-declared, non-adjusted threshold**. That
is a stronger, cleaner negative than "unstable," and it should be recorded as a completed kill
so no future session re-runs it hoping for a better draw.

---

## Three structural lessons (the real output of E1–E157)

These are worth more than any single kill, because they constrain the next phase.

### 1. Robustness and novelty are anti-correlated across this project's positives

The two results with real statistical backing (MM, deep supervision) are textbook techniques.
The two with interesting mechanisms (E15's geometry, the DS gradient autopsy) are oracle-driven
or explain sub-1pp effects. This is not coincidence: **effects large enough to survive 3 seeds
in this setup have been large enough to have been found already.**

### 2. This network has no scarce resource to reallocate

Established by **three independent routes**:

- **E71** (intervention): bottleneck-dependence and skip-dependence are not substitutable —
  ρ(d_i, suppression-tolerance) = **−0.290**, p=0.001, **sign backwards** from what any
  allocation design requires. They are correlated fragility indicators.
- **E147** (correlation): demand↔capacity runs **inverse** (partial ρ=−0.496); **74/88
  subjects served by rank ≤4 of 256**.
- **E72/E73** (implementation): loss-weighting and spatial gating on the same signal, both
  inert.

$$\boxed{\text{Any design of the form ``predict demand} \rightarrow \text{reallocate capacity'' is closed by direct evidence.}}$$

This is the single most important constraint on the next phase. The over-provisioning is real
(rank ≤4 of 256) — but over-provisioning is **not** scarcity, and the two must not be conflated
again.

### 2b. Single-seed >1pp results in this setup have not survived replication — four times

This is the pattern that most often misleads recall, because the seed-0 numbers are real and
were correctly measured:

| Mechanism | Seed 0 | After 3 seeds |
|---|---|---|
| **E45 D4+D8** | **+1.44 pp per-subject, doubly significant** | seed 1 +0.36 pp (p=0.588); 3-seed 0.8939 |
| **E54 A96** | +1.39 pp per-subject, doubly significant | 3-seed 0.8944, margin < 1/10 between-seed std |
| **E49 CCABA** | +0.51 pp | +0.31 pp, CI crosses zero |
| **E70 CAS** | +0.30 pp, CI excludes 0 | +0.24 pp (std 0.22), seed 2 p=0.98 |

E56's re-score genuinely found that **both E45 and E54 clear the corrected +1pp per-subject
target on their single seed.** Anyone recalling "deep supervision gave a comfortable 1%" is
recalling a real, correctly-measured result. The 3-seed confirmation is what removed it, and
the disqualifier is **seed 1's collapse to p=0.588**, not the 0.0003 threshold miss (which the
project's own doc calls a coin flip).

**Consequence for the next phase:** a single-seed >1pp result here carries very little
information. The 3-seed policy (from E49) exists because of exactly this, and it must not be
relaxed for a promising-looking new mechanism.

### 3. The load-bearing novelty axis must be re-checked, not inherited

E71's novelty rested on model-space vs input-space intervention. Correct in 2026-09-02 against
the CFKD family; **dead** against arXiv 2608.14894. A novelty axis has a shelf life. Any future
claim must be re-audited immediately before use, not cited from an earlier session's search.

---

## Methodological assets that survive (and should be reused)

Not novelty claims, but real infrastructure the next phase inherits:

- **E56**: pooled vs per-subject Dice differ systematically by 1.7–1.9pp. Per-subject is the
  primary endpoint. Never mix.
- **3-seed policy** (from E49's variance finding): no GO/KILL on one seed.
- **Pre-registration with falsifiers**: repeatedly caught real errors (E118's lstsq
  overfitting, E109's gate-conditioning bug found via a 14× scale mismatch, E122's per-stage
  regression bug, E125's confounded causal test).
- **E126's intervention machinery**: manual trunk unroll, blend, ablation, with bit-exactness
  sanity checks. Intact and reusable.
- **E48's ablation convention**: the gate receives the *possibly-ablated* bottleneck. Check
  `e48/run_e48_bottleneck_encoding_audit.py:137-140` directly; do not assume.

---

## The boundary

$$\boxed{\text{E1–E157 inventory CLOSED} \quad\big|\quad \text{E159+ assumption-hunting phase}}$$

**Rules carried across the boundary:**

1. Do not resurrect a closed branch without new external evidence (a retraction, a failed
   replication elsewhere, or a genuinely new measurement — not a new framing of an old one).
2. Do not propose architectures in the next phase until an assumption list exists and has been
   literature-audited.
3. "Predict demand → allocate capacity" is closed. Any proposal that reduces to it is
   pre-rejected by lesson 2 above.
4. Generic dynamic computation, dynamic depth, dynamic resolution, MoE, and input-dependent
   routing are all occupied. "Compute more for hard cases" is not a contribution.

---

## What opens next

The next phase's question, carried forward unchanged:

> **What computation do current segmentation architectures assume they must perform for every
> image, that our evidence suggests they actually don't need to perform?**

with the mandatory sharpening:

> **and what does segmentation require at a voxel that existing dynamic-computation methods
> have not formalized correctly?**

The deliverable of the next phase is **an assumption table, not a module**. Format fixed in
advance:

| Assumption | Why segmentation uses it | Why it might be unnecessary | Our evidence bearing on it | Prior art | Verdict |
|---|---|---|---|---|---|

No GPU work until a row survives the prior-art column.
