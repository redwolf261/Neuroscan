# E200 — Dataset headroom search: is a ≥1pp gain reachable on a different cohort?

**Date**: 2026-09-20
**Status**: SEARCH RESULT. No compute. Decision input only.
**Question**: E196/E198 closed BraTS-GLI by arithmetic. That closure is *dataset-specific*. Is
there an obtainable cohort with genuine headroom?

---

## Criteria (fixed before searching)

1. Published SOTA **well below** the annotation ceiling — real room, not 0.900-vs-0.77
2. Enough subjects for the project's 3-seed policy
3. Fits 8 GB (RTX 5050)
4. **Obtainable** — the binding constraint; `PediMS` on disk is 2 files / 1 patient

## Results

| cohort | SOTA Dice | headroom | verdict |
|---|---|---|---|
| **BraTS-GLI 2023** (current) | ET 0.82–0.90 | **none** — model 0.900 vs inter-rater 0.77 | 🔴 closed by E196/E198 |
| **BraTS-Africa** (SSA) | winning team **ET 0.90**, NETC 0.91, SNFH 0.93 | **none** — same saturation | 🔴 rejected |
| **BraTS-MEN-RT** (meningioma RT) | best solution **DSC 0.8214** | modest | 🟡 marginal |
| **BraTS 2024 post-treatment glioma** | scores not located; adds RC (resection cavity); described as harder — "radiation inflammation, and gliosis, combined with the already naturally ill-defined tumor borders" | unquantified | 🟡 unverified |
| **BraTS-METS** (brain metastases) | **winner lesion-wise Dice 0.65 ± 0.25**; 2nd place **0.596** | **very large** | 🟢 **live** |

## BraTS-METS is the candidate

$$
\boxed{\text{Winning Dice } 0.65 \text{ vs BraTS-GLI's } 0.90 \;\Rightarrow\; \sim25\text{pp of unexploited headroom}}
$$

**Data**: 2024/2025 edition released **1,475 cases (1,296 training)** on Synapse; the 2023
edition released 402 studies / 3,076 lesions publicly. Same mpMRI format as BraTS-GLI — the
existing preprocessing and loader stack should apply with modest changes.

**Why it is not saturated** — and this is the part that matters for method design. The challenge
analysis states the failure mode explicitly: *"common errors among the leading teams included
false negatives for small lesions... Dice scores and lesion detection rates of all algorithms
diminishing with decreasing tumor size, particularly for tumors smaller than 100 mm³."*

This is a **detection** problem (finding small lesions), not a **sub-partition** problem
(dividing a found tumour by contrast). It is therefore *not* closed by E142 — which is a
statement about contrast-based sub-partitioning — nor by E167's boundary-localisation, since
whole-lesion misses are not boundary error.

Lesion-wise scoring compounds this: false negatives score 0 Dice and a fixed HD95 penalty of
374, so small-lesion detection dominates the metric. A method that improves detection recall on
sub-100 mm³ lesions would move the headline number substantially.

## Honest assessment of cost

This is **not** "swap the dataset." It is a new project phase:

| step | cost |
|---|---|
| obtain Synapse access, download ~1,300 cases | days (registration + transfer) |
| preprocessing, caching, loader adaptation | days |
| train baseline, establish 3-seed variance | ~1–2 weeks on an 8 GB GPU |
| failure analysis, novelty audit, candidate search | weeks |

No guarantee the search succeeds — but unlike BraTS-GLI, it would not be *arithmetically*
foreclosed before it starts.

## What transfers from this project

Not the conclusions — the instruments:

- **$O_i$ recoverability** — directly applicable, and BraTS-METS is a genuine external
  validation cohort for it. This alone would answer the "one dataset" limitation in
  `REPORT_RECOVERABILITY_FRONTIER.md`.
- **The headroom gate (E196)** — apply *first* on the new cohort, before any method search.
- **The oracle-arithmetic gate (E195)** — cheaper than a prior-art audit; run it first.
- **The leakage discipline (E197/E198)** — the in-subject/cross-fitted contrast. Caught three
  artifacts here.

## Recommendation

Two coherent options; the middle path is the better one.

1. **Finish now.** Publish the recoverability report. Complete, audited, ET-only.
2. **Extend to BraTS-METS** — a genuinely different and *unsaturated* target. Adds external
   validation to the current result **and** reopens the ≥1pp question on a cohort where it is
   not arithmetically closed.

**Suggested sequencing**: run $O_i$ on BraTS-METS *first*. It is inference-only, needs no
segmentation training, directly strengthens the existing paper, and its result tells you whether
the deeper search is worth funding — the same "cheap gate before expensive commitment"
discipline that produced E195 and E196.

## Not claimed here

No prior-art audit was run for any BraTS-METS method. No claim is made that a ≥1pp gain is
*achievable* there — only that it is not foreclosed by the arithmetic that closed BraTS-GLI.

---

## FULL-TEXT FOLLOW-UP (2026-09-20) — arXiv 2504.12527 read directly

The BraTS-METS 2025 Lighthouse analysis paper was fetched and read (pages 1-20). Two
corrections and one blocker, all material:

### Correction 1 — that paper reports NO team Dice scores

It is a **dataset/protocol description**, not a results paper. Section III "Results" describes
only cohort composition. So the headroom figure does **not** come from it. The
0.65 / 0.596 numbers are from **BraTS-METS 2023** (winner lesion-wise Dice 0.65 ± 0.25; second
place 0.596), which remains the best available evidence but is **two editions old**. Current
SOTA on the 2025 edition is **not established by anything read here**, and could be higher.

*This is exactly the E194 discipline: the number is cited from a real source, but that source is
older and narrower than the enthusiasm around it implied.*

### Correction 2 — the label scheme differs from BraTS-GLI

| BraTS-METS 2025 | definition |
|---|---|
| Label 1 | NETC — non-enhancing tumour core |
| Label 2 | SNFH — surrounding non-enhancing FLAIR hyperintensity |
| Label 3 | ET — enhancing tumour |
| **Label 4** | **RC — resection cavity** (post-treatment only; "majority of cases do not have this") |

Evaluation regions: ET = Label 2; TC = Label 2 + 3; WT = Label 1 + 2 + 3. **Note the paper's
Table 1 label indices are inconsistent with its own Section-3 text** (Table 1 calls Label 2 "ET"
while the text calls Label 3 "ET"). The actual channel ordering must be verified against the
downloaded NIfTI files before any code is written — do not trust either statement alone.

### Correction 3 — the cohort is not homogeneous

| dataset | train | val | test | stage | space |
|---|---:|---:|---:|---|---|
| Duke | 37 | 15 | 30 | pre | SRI24 |
| NCI | 35 | 0 | 1 | pre | SRI24 |
| Missouri | 22 | 25 | 35 | pre | SRI24 |
| WashU | 39 | 2 | 12 | pre | SRI24 |
| Yale | 195 | 0 | 12 | pre | SRI24 |
| UCSF | 322 | 0 | 0 | pre + post | **native** |
| NW | 0 | 46 | 0 | pre | SRI24 |
| UCSD | **646** | 91 | 213 | pre + post | **native** |
| **total** | **1296** | **179** | **303** | | |

732 post-treatment, 1046 pre-treatment. **1,272 of 1,778 cases (UCSF + UCSD) are in NATIVE
SPACE, not atlas-registered** — unlike BraTS-GLI, which is uniformly SRI24. UCSD (646 training
cases, half the training set) is longitudinal follow-up data, a different distribution again.

**Consequence**: the existing preprocessing stack does **not** transfer unchanged. Mixed-space
data is a real engineering cost, and the native/atlas split is a confound that must be handled
in any analysis — including an $O_i$ replication, since $O_i$ is intensity-based and
registration affects interpolation.

### The blocker

Data access requires **Synapse registration + challenge team registration + a data-use
agreement**, all of which need the user to authenticate. Synapse ID: **syn64153130**
(BraTS-Lighthouse 2025). I cannot complete this step; nothing is downloadable without it.

### Revised recommendation

The route is still live — 2023 evidence of large headroom stands, and the failure mode
(small-lesion detection, sub-100 mm³) is genuinely outside everything this project closed. But
it is **less turnkey than the first pass suggested**: unknown current SOTA, mixed coordinate
spaces, a 4th label, and an ambiguous label-index spec.

Before committing weeks: (1) the user registers on Synapse and obtains the data; (2) verify
label ordering against actual files; (3) run $O_i$ on the atlas-registered subset only, as the
cleanest first measurement and a genuine external-validation cohort for the existing paper.
