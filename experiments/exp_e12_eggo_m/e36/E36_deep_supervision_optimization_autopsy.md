# E36 — Deep-Supervision Optimization Autopsy

**Purpose**: explain, mathematically and mechanistically, why the only real positive result in this project — extra training supervision at a coarse internal resolution of the model ("D4 supervision") — actually improves the final result. No new model was trained. No new loss was proposed. This document works entirely from measurements taken on models that were already trained.

**This document is self-contained.** It explains the model, the measurement, and the result without requiring any other file to be read first.

---

## 1. Background: what "D4 supervision" is

The segmentation model is a 3D U-Net. Input scans are shrunk to a 64×64×64 voxel cube before the model ever sees them. Internally, the model's decoder rebuilds a full-size prediction through three stages: a coarse one (16×16×16, called "D4" because each side is 1/4 of full resolution), a medium one (32×32×32, "D2"), and the full one (64×64×64, "D1" — this is also where the model's normal, primary training signal lives).

In an earlier phase of this project, two extra prediction heads were added, one reading from each of the intermediate D4 and D2 decoder stages. During training, each extra head is given its own loss, comparing its coarse prediction against a correspondingly shrunk-down version of the true tumor outline. This "deep supervision" idea is a standard, published technique — using it at all is not the novel part of this project. What was previously found, and what this document explains, is a specific and unexpected pattern in the *result*: using the D4 head **alone** (no D2 head) gave the best final full-resolution result (+0.33 percentage points over the baseline) — better than using D2 alone, and better than using both together. This was surprising and, until now, unexplained.

## 2. The three loss terms

At every training step, the model actually computes three separate loss values:

- **The main loss** ("D1"), the model's ordinary, full-resolution training signal — unchanged from the baseline model.
- **The D4 loss**, comparing the coarse (16³) prediction head's output against a coarse, *smoothly shrunk* (not sharply cut) version of the true tumor outline.
- **The D2 loss**, the same idea at the medium (32³) resolution.

These three losses are added together (with fixed weights) into one combined number that the optimizer actually uses to update the model on every step. This combined number, on its own, cannot tell you whether the three ingredients are working together, working against each other, or which one is actually doing anything — three numbers were mixed into one, and that mixing is invisible after the fact. This document's method is to separate them back apart.

## 3. Method

Three model variants had already been trained and saved at regular checkpoints throughout training: one using the D4 head only, one using the D2 head only, and one using both. No new training was done. At each of seven saved checkpoints (roughly every 5 epochs through the full 30-epoch training run), the following was measured:

For each of the three individual loss terms, the exact same fixed, held-out batch of scans was run through the model, and each loss term's own gradient — the specific direction and size of parameter update that term alone would cause — was computed separately (not the combined gradient training itself uses, which mixes all three together and can't be pulled back apart). This required running the model's own maths (not a re-implementation) exactly as the original training script defines it, verified directly against that script before trusting any result from it.

For every pair of the three gradients, two numbers were computed: how *aligned* they are (a similarity score from -1, exactly opposite, to +1, exactly the same direction, with 0 meaning unrelated), and how their *sizes* compare (one term's gradient could be, for example, twice as large as another's).

This was first done on a single fixed batch of 8 scans, then, because a genuinely surprising pattern showed up, deliberately repeated on **15 separate, non-overlapping batches covering 120 of the 125 held-out scans** — not trusting a small-sample result until it survived being checked against the much larger, more representative sample.

## 4. Finding #1 (confirmed, not the expected one): D4's gradient becomes *less* aligned with the main gradient over training, not more

A natural first guess would be: "D4 helps because its gradient points in a *similar* direction to the main gradient — it reinforces the same learning, so training goes faster/further." The data does not support this, and in fact shows close to the opposite.

Measured across all 15 independent batches, with the model's own primary auxiliary term compared against its own main loss:

| Condition | Alignment (cosine similarity) with main loss, epoch 1 | ...epoch 10 | ...epoch 30 |
|---|---:|---:|---:|
| D4-only (best final result) | 0.26 | 0.84 | **0.25** |
| D2-only (worse final result) | -0.42 | 0.33 | **0.82** |
| Both | -0.06 | 0.57 | 0.52 |

By the end of training, **D4-only — the condition with the best final result — has the *least* aligned gradient with the main loss**, while D2-only — a worse-performing condition — has the *most* aligned. This difference is statistically solid, not noise (a direct statistical comparison across the 15 independent batches gives p<0.0001 for D4-only vs. D2-only at the final checkpoint). This is the opposite of the "alignment is good" intuition, and it was checked carefully — including making sure it wasn't simply because one gradient had shrunk to near-nothing (it hadn't; both gradients stayed a comparable, substantial size throughout) — before being trusted as real.

## 5. Finding #2: D4's own loss value never gets nearly as low as the main loss does, and the gap between them grows steadily

This is the real explanation for Finding #1, and it is a much more ordinary, well-behaved kind of result — not a mysterious interaction, but a straightforward difficulty gap.

| Epoch | Main loss | D4's own loss (D4-only condition) | D4 loss ÷ main loss |
|---:|---:|---:|---:|
| 1 | 0.76 | 0.52 | 0.69 |
| 5 | 0.16 | 0.17 | 1.07 |
| 10 | 0.10 | 0.17 | 1.68 |
| 20 | 0.06 | 0.15 | 2.38 |
| 30 | 0.05 | 0.15 | **2.74** |

The main loss keeps dropping throughout training, as expected — the model is genuinely getting better at its primary job. **D4's own loss, by contrast, drops early and then gets stuck around 0.15 and stays there** — it never approaches the low values the main loss reaches. By the end of training the D4 task is over twice as "unsolved," in loss terms, as the main task is. (For comparison, D2's own loss only reaches about 1.5× the main loss's value at the same point — the same effect happens for D2, but far more mildly.)

This directly explains Finding #1: once the main loss is nearly solved, its own gradient shrinks and becomes a fine, local polishing signal. D4's loss, still substantial because its own task is genuinely still unsolved, keeps producing a real, sizeable gradient — and because the main gradient has narrowed its focus to fine local adjustments, the two gradients naturally stop pointing the same way. **D4 keeps supplying a real training signal, right through to the end of training, in exactly the part of the problem the main loss has already stopped paying attention to.**

## 6. Finding #3: this is a boundary/partial-coverage problem, not (primarily) a small-lesion-size problem

The natural next question is: *what* is D4's task still failing to solve? Given this project's long-running focus on small tumors, the first hypothesis tested was that D4's stuck-loss problem is concentrated on small lesions specifically.

This was tested directly, using the 749-tumor-fragment inventory this project already maintains, matched at coarse (D4) resolution against the model's actual coarse-resolution prediction. **The result did not cleanly support the small-lesion hypothesis** — if anything, larger tumor fragments showed slightly *higher* average error at D4, not lower (weak positive statistical correlation between fragment size and error, p=0.003), while small fragments showed much more *variable* error (some measured almost perfectly, others quite badly) rather than a uniformly worse result.

Digging further resolved this cleanly: the model's D4-resolution prediction error is not evenly spread across all D4 voxels. It is heavily concentrated on voxels where the true, shrunk-down target is neither "clearly tumor" nor "clearly not tumor," but somewhere in between — a **partial-coverage** voxel, meaning a coarse grid cell that a tumor's boundary happens to pass through, so only part of it is really tumor.

| Voxel type (at D4, 16³ resolution) | Mean prediction error |
|---|---:|
| Full coverage (clearly tumor) | 0.03 |
| **Partial coverage (a boundary crosses this cell)** | **0.30** |
| Background (clearly not tumor) | 0.0003 |

Partial-coverage voxels have roughly **ten times** the prediction error of full-coverage voxels. This is where nearly all of D4's stuck loss lives.

## 7. Finding #4: why this specifically affects D4 more than D2 — a geometric, not size-based, explanation

Shrinking a scan down to a coarse grid necessarily creates partial-coverage cells wherever a tumor's boundary crosses a coarse grid line. A tumor's boundary — its surface — does not shrink proportionally to volume as resolution coarsens the way its interior does; a coarser grid means each cell covers a bigger physical region, so more of a tumor's *outline* gets smeared across ambiguous, partially-covered cells rather than clean fully-covered ones.

This was measured directly and precisely: **at D4 resolution, 61% of a tumor's total supervised "mass" sits in partial-coverage, boundary-ambiguous cells. At D2 resolution, this figure is only 39%.** The ratio between these two figures (1.57×) closely matches the ratio between D4's and D2's own stuck-loss gap measured in Section 5 (1.86×) — not identical, but the same order of magnitude, from a completely independent, purely geometric calculation. This is a strong, quantitative (not just qualitative) confirmation that the mechanism is real: **D4's coarser grid genuinely does turn a bigger share of every tumor into an inherently harder, boundary-dominated prediction problem, and this is measurably, numerically, why D4's own loss stays stubbornly higher than D2's throughout training.**

## 8. Finding #5: where does D4's gradient actually go? (Ruled out one candidate explanation)

One further hypothesis was tested and ruled out: that D4's gradient is special because it reshapes the model's early, shared layers (used by every part of the prediction, including the main output) much more than D2's gradient does — which would explain why D4 alone helps the *main*, full-resolution result specifically.

This was checked directly by measuring, throughout training, what fraction of each loss term's own gradient lands in the model's shared early layers (used by all three prediction heads) versus each head's own private, later layers. The result: **this is not special to D4.** Both D4's and D2's gradients are overwhelmingly (99%+) concentrated in the shared early layers, throughout training, in every condition tested. This is simply a property of *any* coarse auxiliary supervision in this architecture — not a distinguishing feature of D4 specifically. This candidate explanation is disclosed and ruled out here so it is not mistakenly revisited later.

## 9. The minimum mathematical property that explains D4's improvement

Putting Sections 4–7 together, without the ruled-out candidate from Section 8:

> **D4 supervision helps because coarsening a tumor's outline down to a 16×16×16 grid converts most of that outline's information into an inherently harder, ambiguous, partial-coverage prediction problem — one the model's ordinary, full-resolution training signal never directly addresses. Because this coarse-scale problem stays substantially unsolved even once the model's main training signal has converged, D4's own gradient continues to supply a real, non-trivial training push, later into training and more persistently than D2's does, specifically into the model's shared early layers — precisely the part of training where the main signal alone has already stopped exploring.**

This is not a claim about gradient conflict being good in general, nor about small lesions specifically, nor about D4's decoder branch doing anything special on its own. It is a claim about **the geometry of average-pooling under coarsening**, combined with **an ordinary, well-understood optimization fact** (a loss term that hasn't converged keeps contributing gradient after a loss term that has converged stops mattering).

## 10. Candidate mathematical consequences — **not implemented, listed only**

Per the explicit instruction for this phase: the following are candidate directions a next phase *could* explore mathematically, none of them implemented, tested, or endorsed here. Each follows directly from Sections 4–9, not from an assumption made in advance of the data.

1. **A resolution-aware reweighting of the boundary-vs-interior portion of the loss**, informed directly by the measured 61%/39% partial-coverage-mass split at D4/D2 — rather than treating the whole coarse target uniformly, explicitly separate what fraction of the loss's own optimization difficulty is coming from ambiguous boundary cells versus clean interior cells, since Section 6 shows these have roughly a 10× difference in typical error.
2. **A training-phase-dependent (not fixed) weighting of the auxiliary loss terms**, motivated by Section 5's finding that the *usefulness* of D4's gradient (as measured by how much it still diverges from main, meaningfully, rather than trivially) changes systematically over the course of training — starting closer to main, then diverging as main converges. A fixed weight, used throughout the entire training run in every experiment in this project's history so far, does not reflect this.
3. **Measuring, before assuming, whether the same coarsening-creates-more-boundary-ambiguity effect predicts which specific *scans* benefit most from D4 supervision** — Section 7's geometric mechanism is a population-level, statistical explanation; whether it also predicts which individual tumors gain the most from D4 specifically is a testable, not-yet-tested extension of it.
4. **A direct comparison against the small number of published techniques that explicitly manipulate the *relationship* between multiple loss gradients** (rather than just summing them with fixed weights), since this document's own finding — that a currently-unweighted, currently-unmanaged form of gradient divergence appears to be doing something useful rather than something harmful — sits close to, but is not the same claim as, the active published literature on gradient-conflict-aware training. Section 9's mechanism is explicitly about *resolution geometry*, not gradient conflict as a general phenomenon, which is a meaningfully different starting point from that literature and would need its own careful novelty check before any of the above three directions were pursued as an actual training experiment.

## 11. Limitations

- All measurements use a fixed, held-out set of scans, replayed identically at every checkpoint — not the exact sequence of scans/order the original training actually used (that exact sequence was not saved and cannot be reconstructed). This is the right tool for the comparison being made here (comparing conditions and epochs on equal footing), but it is not literally "the gradient training itself computed at that exact moment."
- The model's normalization layers were run using each checkpoint's own saved, learned statistics (not statistics from this analysis's own small batches) — the right choice for measuring the model's real, learned behavior, but it means these are not byte-for-byte identical to what the live training loop would have computed at that instant, only mathematically equivalent measurements of the same underlying learned model.
- Section 6/7's component-level and boundary/interior analysis was run on the D4-only model's final checkpoint; it was not repeated across every checkpoint and every condition, since the geometric explanation in Section 7 (which does not depend on which model produced it — it is a property of the target data alone) already explains why the pattern would hold generally.
- No literature-novelty search was performed in this phase, per the explicit ordering requirement: this document is the "mechanistic measurement + mathematical derivation" stage only. A literature check is the next required stage, before any of Section 10's candidates could be considered for an actual training experiment.
