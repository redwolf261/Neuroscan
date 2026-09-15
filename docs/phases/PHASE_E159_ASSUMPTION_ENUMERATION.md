# E159 — Assumption Enumeration (the Vaswani search)

**Date**: 2026-09-15
**Status**: Enumeration only. No architecture, no GPU, no experiment.
**Follows**: E158 (inventory closed, two-regime qualification)
**Numbering note**: filed as E159 because `PHASE_E158_TERMINAL_STATE_INVENTORY_CLOSED.md`
already occupies E158. The prior-art audit of survivors becomes E160.

---

## Target

$$\boxed{\text{Find a computation medical segmentation treats as necessary, but isn't.}}$$

Not inefficient. Not improvable. **Unnecessary** — an intermediate requirement that something
else can discharge more directly.

---

## Two hard constraints from our own history

Before enumerating, two filters that kill most candidates on sight:

**Filter 1 — no allocation designs.** Closed by three independent routes (E71 sign −0.290
backwards; E147 inverse ρ=−0.496 with 74/88 subjects served by rank ≤4 of 256; E72/E73 inert).
This network has no scarce resource to reallocate. Any assumption whose removal reduces to
"spend compute where it's needed" is pre-rejected.

**Filter 2 — the assumption must be indicted by our evidence.** This is the criterion the
first-pass enumeration failed. A taxonomy of things U-Nets do is not a research program; the
project has already killed ~20 mechanisms chosen that way. The question is not "what does the
architecture assume?" but **"what does our failure mode prove is being assumed wrongly?"**

---

## Part 1 — Generic architectural assumptions, and why they don't qualify

| | Assumption | Verdict | Occupying prior art |
|---|---|---|---|
| A | Every voxel gets computation | 🔴 | sparse/dynamic computation |
| B | Uniform depth per location | 🔴 | early-exit, adaptive depth |
| C | Uniform representational capacity | 🔴 | dynamic channels/width — **plus Filter 1** |
| D | Compression precedes reconstruction | 🔴 | resolution/upsampling literature, saturated |
| **E** | **Every voxel must be explicitly predicted** | 🔴 | **see below — occupied, decisively** |
| F | Representation built independent of task | 🔴 | task-conditioned representation learning |
| G | Decoder inverts encoder | 🔴 | implicit neural representations, direct prediction |
| H | Hierarchy needed for global context | 🔴 | attention, SSMs, global convolution |
| I | One representation serves ET/TC/WT | 🟡 | nested/hierarchical segmentation — needs audit |
| J | One forward-pass commitment | 🔴 | iterative/recurrent refinement, cascades |

### Assumption E is occupied — this needs stating plainly

E was the first-pass lead candidate ("does segmentation need dense voxelwise prediction?").
**It is the dominant segmentation paradigm since ~2021**, not an open question:

- **MaxDeepLab / MaskFormer / Mask2Former** replaced dense per-pixel classification with a
  small set of learned **object queries**; the dense mask is *derived* from N structured
  decisions. This is precisely the "structured decisions → derived mask" shape.
- **DETR** established the query paradigm for detection; segmentation inherited it directly.
- **SAM** does promptable mask decoding from sparse prompts.
- 3D medical variants of query-based segmentation exist.

The analogy to "do we need sequential hidden states per token?" is apt — but that question was
already asked and answered in segmentation by the query-based family. Claiming it now would be
claiming Mask2Former.

**Verdict: 🔴. Do not pursue.** Stated here explicitly so it is not re-proposed.

---

## Part 2 — What our evidence actually indicts

The generic list above is untethered from NeuroScan. Our Regime-2 evidence points somewhere
much narrower and, unusually, **not at the architecture at all.**

### The established causal chain

| Finding | Evidence |
|---|---|
| Failure is a **15-subject tail**, not the bulk | ET/TC bottom deciles are the *same* 15 subjects, Jaccard 0.733; ET median 0.913 but mean 0.819 (E139) |
| The tumour **is found** on those subjects | WT = 0.824 on the same 15; failure is **sub-partition**, not detection (E139) |
| Failures are **total false negatives** | 5/8 ET failures predict **exactly zero voxels**; max prob over whole volume = 0.0000 (E133) |
| **t1c causally owns ET** | zeroing t1c: ET 0.8433 → **0.0015** (−56.15pp); zeroing t2f: WT 0.9262 → 0.6455 (E142) |
| Modality pathways are **near-independent** | removing t1c leaves WT at 0.8927; removing t2f leaves ET at 0.8054 (E142) |
| The evidence is **absent, not mis-read** | E140: per-voxel boundary shift as a per-subject *oracle* gains only +0.0154 ET, 7/15 move exactly 0.000 |
| Three intervention families all failed | E137 conditional readout (−0.89pp), E140 recalibration, E141 supervision (−0.37pp) |

### The assumption this actually indicts

Not capacity, not depth, not density. It is:

$$\boxed{\textbf{K — every subject's modalities are assumed equally informative}}$$

Formally, the architecture computes

$$z = f(x_{t1c}, x_{t1n}, x_{t2f}, x_{t2w})$$

with a **fixed** functional dependence on each channel, identical for every subject. But E142
shows the dependence is **extreme and specialised** (ET is ~entirely a function of t1c), and
E136/E138 show **contrast varies enormously across subjects** — some subjects' t1c carries
almost no enhancement signal at all.

So the network is committed, by construction, to reading ET out of a channel that for ~12% of
subjects *does not contain the answer*. It then predicts exactly zero, with total confidence.

**This is a real assumption, it is specific to multi-modal medical imaging, and it is
established by causal intervention rather than by architectural taxonomy.**

### Why this is NOT a Filter-1 allocation design

Critically, K's removal is *not* "detect low-contrast subjects and give them more compute."
That is allocation, and it is dead — E141 tried exactly it (evidence-conditioned supervision,
−0.37pp, shuffled control gap +0.26pp below threshold).

The honest reading of E140 + E142 together is stronger and more uncomfortable:

> For those subjects the information is **not present in the input**. No readout, no
> recalibration, no reweighting, and no allocation of capacity can recover it.

Which raises the only question in this space that is not already closed:

$$\boxed{\text{Is the missing information recoverable from the \emph{other} modalities, or is it absent from the study?}}$$

E142 gives a partial, encouraging answer: removing t1c leaves **TC at 0.1010** — collapsed —
but WT at 0.8927. So t2f/t1n/t2w *do* carry tumour-extent information on these subjects. The
untested question is whether they carry enough *sub-partition* information to substitute when
t1c is uninformative.

**Status: 🟢 the most interesting open question on the board, and it is a question about
information, not architecture.**

---

## Part 3 — Honest assessment of K's novelty prospects

Stated before any audit, so the audit cannot be read as confirmation:

**Against it**: missing-modality and modality-dropout segmentation is a **large, active
field** (mmFormer, ShaSpec, RFNet, "No Modality Left Behind", and the DMAF-Net family E71's
own search already surfaced). Those methods train for *absent* channels. Our case is a channel
that is *present but uninformative*, which is adjacent — possibly distinguishable, possibly
not. E138 already scored a **partial hit** on the closely-related FiLM-for-contrast idea
(arXiv 2511.16498).

**For it**: the specific framing — *a present-but-uninformative modality behaves like an absent
one, identifiable label-blind before inference* — is narrower than the missing-modality
literature and is backed by an unusual causal measurement (E142's clean −56.15pp
specialisation).

**Most likely outcome**: 🟡 partial hit, like E138. That should be expected, not treated as
failure.

---

## Part 4 — The candidate that survives on different grounds

One non-architectural assumption deserves recording, because the full-corpus read exposed it
and nobody has tested it:

$$\boxed{\textbf{L — findings established in Regime 1 transfer to Regime 2}}$$

The entire E48→E97 chain (~50 experiments) established "small lesions depend more on the
bottleneck" **exclusively** on FLAIR-only, binary, 64³ — a setup where E29 proved the
**median lesion component vanishes to zero voxels** in preprocessing, and which the master
report states was chosen "to keep the model and problem simple."

That chain may have diagnosed **a preprocessing artifact the project created.** Regime 2's
actual failure mode (ET/TC sub-partition on found tumours) is a different phenomenon entirely.

This is not a novelty candidate. It is a **validity question about ~50 experiments**, it costs
one inference run to answer, and it should be settled before any Regime-1 finding is cited in
a write-up.

---

## Recommended order for E160

1. **Test L first** (cheapest, highest information): does the bottleneck/small-lesion
   size-dependence reproduce at 128³ with 4 modalities and 3 regions? One inference run on an
   existing checkpoint. Settles the status of ~50 experiments either way.
2. **Audit K** against the missing-modality / modality-dropout literature, with the
   present-but-uninformative framing stated precisely. Expect a partial hit.
3. **Audit I** (nested ET⊆TC⊆WT representation) only if K returns 🔴 outright.

No GPU beyond step 1's single inference pass. No architecture until an audit returns 🟢 or a
defensible 🟡.
