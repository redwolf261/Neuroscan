# Phase E66 — Optimization Topology Audit (block-wise gradient interaction)

## Purpose

Returns the project to its original stated axis (architecture-independent multi-
objective optimization) after the E62–E65 causal-diagnostic branch (retained as
completed side-branch evidence) and a self-corrected multimodal detour (CMRR,
killed on novelty grounds; the follow-up multimodal audit itself killed as scope
creep before any training). Tests the observational premise behind a proposed
"Gradient-Conditioned Objective Routing" mechanism: **does gradient conflict
between this project's live loss terms vary meaningfully — and structurally, not
just in magnitude — by network region?** No new training, no architecture change,
no new optimizer; pure observational diagnostic on existing saved checkpoints.

## Corrections to the original proposal, found during setup

1. **Objective set.** The proposal's L1–L4 (seg/evidential/consistency/pseudo)
   does not match what actually trains in the live EGGO-M pipeline (same lineage
   as every E43–E65 checkpoint). Consistency and pseudo-label losses exist only
   in the older, unrelated, frozen `final_model.py`/Phase-1-3 pipeline. The real,
   live composition is `seg_loss = 0.5·FocalTversky + 0.5·EvidentialBeta`, plus a
   `mu`-weighted boundary BCE and a dormant `lambda_margin`-weighted margin term
   (margin was tested extensively in the EGGO-M arc and found null on Dice at
   E13 — not revived here).
2. **Boundary loss excluded.** `boundary_head` reads `dec1.detach()` (a
   deliberate, documented v2 design choice preventing it from becoming
   separable on its own account). This makes boundary loss's gradient
   architecturally **zero** in encoder/bottleneck/decoder by construction — not
   a discovered phenomenon. Including it would contaminate the outcome decision
   with a known architectural artifact.
3. **`heads` block excluded from the outcome decision.** `seg_head` and
   `evidential_head` are disjoint parameter subsets (FocalTversky's gradient
   only ever touches `seg_head`; Evidential's only ever touches
   `evidential_head`), so their flattened block vectors are **geometrically
   forced to cos=0** at every checkpoint, regardless of any real relationship —
   pure linear algebra, not structure. Reported for transparency, excluded from
   `between_block_range`/`sign_flips`/the KILL-GO logic.

The only pair measured is therefore **FocalTversky vs. EvidentialBeta**, the
only two losses that both flow through the full shared trunk undetached.

## Method

- 7 existing saved checkpoints from E25's `DeepSup_D4only_seed0` run (epochs 1,
  5, 10, 15, 20, 25, 30 — all present on disk, zero new training compute).
- At each checkpoint: one shared forward pass on 4 fixed batches (8 subjects
  each, same subjects reused at every checkpoint so cross-checkpoint
  comparisons aren't confounded by batch variation), then two separate backward
  passes (one per loss) to get each loss's own gradient at that checkpoint's
  frozen weights. No optimizer step; weights never modified.
- 4 parameter blocks (encoder / bottleneck / decoder / heads), matching the
  proposal's own E/B/D/H notation.
- Per (checkpoint, block): `‖g_i‖` per loss, `cos(g_focal, g_evidential)`.

## Results

| Block | Mean cos (7-checkpoint avg) | Std | Range |
|---|---|---|---|
| Encoder | **+0.845** | 0.240 | [0.260, 0.977] |
| Bottleneck | **+0.813** | 0.281 | [0.129, 0.963] |
| Decoder | **+0.463** | 0.139 | [0.214, 0.664] |
| Heads (excluded from outcome) | +0.000 | 0.000 | [0.000, 0.000] (geometrically forced) |

- **FocalTversky and EvidentialBeta are cooperative everywhere, at every
  checkpoint** — cosine is positive in encoder, bottleneck, and decoder across
  all 7 checkpoints, with no instance of conflict (negative cosine) anywhere.
- **Real between-block variation exists** (range = 0.382): encoder and
  bottleneck are strongly cooperative (~0.81–0.85 mean), decoder is
  moderately cooperative (~0.46 mean) — a real, non-trivial magnitude
  difference across the network.
- **No sign flip.** No loss pair changes from cooperative to conflicting (or
  vice versa) across encoder/bottleneck/decoder, at any checkpoint.
- Cosines rise sharply from epoch 1 (0.13–0.26, near-random initialization) to
  epoch 5+ (0.35–0.98), then stay high and fairly stable through epoch 30 —
  consistent with the two losses' gradients aligning as the shared trunk
  specializes.

## Pre-declared outcome evaluation

Per the user's own three-way decision tree:

- **A (homogeneous)**: rejected — between-block range (0.382) exceeds the
  pre-declared meaningful-variation threshold (0.15).
- **B (differs by block, but magnitude-only, no sign flip)**: **matched.**
  The variation is a smooth magnitude gradient (encoder > bottleneck > decoder),
  never a change in cooperative/conflicting sign.
- **C (stable, meaningful block-specific ROLE structure — a sign flip)**: not
  observed.

## Verdict: **KILL the proposed block-specific-role novelty (Outcome B)**

FocalTversky and EvidentialBeta do not exhibit the phenomenon the proposal
needed to motivate "Gradient-Conditioned Objective Routing" as a genuinely new
mechanism: there is no region of the network where these two objectives
meaningfully conflict, and no region where their relationship flips sign. What
exists is a real but ordinary magnitude gradient across network depth — exactly
the kind of structure existing global gradient-surgery/weighting methods
(GradNorm, PCGrad, MGDA, CAGrad-style approaches, all designed around
per-parameter or per-layer scaling) are already built to exploit. Per the
user's own pre-declared rule, this does not clear the bar for a new
architecture-independent optimization principle on this evidence.

## Limitations

- Only 2 of the pipeline's loss terms could be meaningfully compared (boundary
  and margin are architecturally/practically excluded, per the corrections
  above) — this audit cannot rule out conflict structure between a different
  pair of objectives in a pipeline where more than two losses actually share
  gradient paths through the same trunk regions.
- 4 coarse blocks (not per-layer) — a finer-grained audit could in principle
  reveal structure this coarse partition averages away, though the consistent,
  large, stable magnitude gradient found here makes a hidden sign-flip within
  a block less likely (not impossible).
- Gradients measured on 4 fixed training-set batches per checkpoint, not the
  full dataset — sufficient for a first-pass diagnostic but not a
  high-precision estimate.

## Implication

Per the project's own constraint #17 ("no forced novelty" — a rigorous
negative result is preferable to a pseudo-novel algorithm), this closes the
"exploit block-specific gradient conflict between FocalTversky and Evidential"
direction as evidence for the intended multi-objective optimization
contribution. If the multi-objective optimization axis is pursued further, the
next step (not started here) would need either a different, currently-inactive
pair of objectives with a real prospect of conflict (the margin mechanism,
previously killed on Dice grounds but not on gradient-interaction grounds), or
a reconsideration of whether this project's own loss composition is rich
enough to support the intended research question at all.

## Artifacts

- `experiments/exp_e12_eggo_m/e66/run_e66_gradient_topology_audit.py`
- `experiments/exp_e12_eggo_m/e66/E66_gradient_topology_table.json` (28 checkpoint×batch records)
- `experiments/exp_e12_eggo_m/e66/E66_summary.json`
