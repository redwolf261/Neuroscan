# Phase E72 — Fragility-Weighted Loss (FWL): Design & Pilot Protocol

## Origin

Per the user's explicit request after CAS (E70) and CDCG's gating mechanism
(E71) were both killed with diagnosed reasons: "try one more mechanistic
idea from what we've learned tonight." This is that idea, arrived at via the
Observe→Hypothesize→Attempt→Test→Adapt directive rather than another
architecture guess.

## Observe: what tonight established

1. Multimodal input helps substantially and robustly (+1.75pp, 3 seeds) — confirmed.
2. The network's bottleneck representation contains a real, causally-grounded,
   recoverable signal predicting its own bottleneck-ablation sensitivity d_i
   (prediction 1: ρ=0.87 held-out; E71 scaled re-test: ρ=0.866, partial
   ρ=0.855 controlling for lesion size — not a size proxy).
3. Acting on that signal via pathway ROUTING/GATING is dead (E71 gate-sweep:
   Spearman(d_i, suppression-tolerance slope) = −0.290, wrong direction,
   p=0.001) — bottleneck- and skip-dependence are correlated markers of
   overall subject difficulty, not substitutable resources to trade off.
4. Acting on it via EXTRA COMPUTE (route hard subjects to more refinement)
   is occupied by the literature (MAGICORE, test-time-compute-scaling).
5. CAS's real (not intended) mechanism was magnitude-gated smoothing —
   suppressing correction where activations are already high-magnitude.

## Hypothesize

Two verbs remain untried and are not ruled out by fact 3 (routing) or fact 4
(compute): using the causal-fragility signal at the LOSS level (training-time
reweighting) or at the DEPLOYMENT level (flagging, not modeled here).

**Candidate: Fragility-Weighted Loss (FWL).** Use ĥat{d}_i (E71's frozen,
validated aux head prediction) as a per-subject weight multiplying the
segmentation loss during training — concentrating gradient signal on
subjects the network's own causal self-model identifies as fragile. No
architecture change (fact 4 doesn't apply — nothing is routed or gated), no
added inference cost (not occupied by compute-scaling literature).

**Literature check (narrow, not exhaustive)**: generic loss-based /
uncertainty-based curriculum/difficulty reweighting is a heavily occupied
category (confirmed via search: fairness-curriculum segmentation 2026,
adaptive point-weighting, balanced reweighting via historical loss). But
those methods are explicitly criticized in the literature itself for
conflating "difficulty" with loss magnitude / optimization noise — a
weakness this signal doesn't share, since d_i is a measured causal
intervention outcome, not a loss heuristic. Follow-up searches for
"causal-ablation-derived difficulty used as a training-time reweighting
signal" surfaced nothing matching this specific combination (Ablation Based
Counterfactuals is a different mechanism — component-level ablation for
training-sample attribution, not per-subject inference-time causal fragility
used as a loss weight). Reported as "not found," not proof of novelty.

## Attempt, staged

### Stage 1 — weighting-scheme sanity check (no training)
`check_fwl_weighting_sanity.py`: computes ĥat{d}_i for 250 sampled training
subjects using the frozen E70 MM-seed0 checkpoint + frozen E71 scaled aux
head, converts to loss weights under 3 candidate schemes, checks for
degeneracy and whether the weighting reduces to a disguised size proxy.

**Result: PASS, all 3 schemes.**
- Max single-subject weight fraction: 0.0064–0.0100 (want ≤0.05)
- Effective sample size fraction: 0.869–0.964 (want ≥0.50)
- Spearman(weight, native_size) = −0.190, clearly weaker than E48's own
  size-only reference of −0.454 — the weighting tracks causal fragility,
  not a re-derivation of "upweight small lesions."

Selected scheme for the pilot: `softmax(ĥat{d}_i / T=1.0)`, normalized to
mean 1.0 — best ESS fraction (0.964) among passing schemes.

### Stage 2 — short pilot training run (this phase's current step)
`train_e72_fwl_pilot.py`: identical recipe to E70's MM condition (UNet3D_v3,
4-channel, AdamW, CosineAnnealingLR, D4-only deep supervision,
λ_ds3=0.9927, FocalTversky+EvidentialBeta) except (a) per-subject loss
weighting from the frozen, precomputed ĥat{d}_i table, (b) shortened to 12
epochs (not the full 30) for a first, cheap look — matching this project's
own "cheapest, first, most falsifiable test" convention.

**Pre-declared falsifiable prediction**: if FWL does what it claims, the
per-subject Dice delta (FWL − matched MM-seed0 baseline subject) should
correlate POSITIVELY with ĥat{d}_i — the mechanism should specifically help
the subjects it was built to help, not produce a diffuse or unrelated shift.

## Decision rule (pre-declared, before seeing pilot results)

- Dice improves AND Spearman(delta, ĥat{d}_i) > 0, permutation p<0.05:
  mechanism-consistent signal — proceed to the full 3-seed protocol before
  any claim.
- Dice improves but delta does NOT correlate with ĥat{d}_i: report as an
  unexplained shift, NOT as FWL "working as designed" — same discipline
  that caught CAS's real mechanism being different from its intended one.
- Dice does not improve: pilot null, KILL before spending the full 3-seed
  budget, per the no-rescue discipline.

## Results: KILLED after one pre-registered re-test

### v1 — softmax(d̂_i / T=1.0)

Matched-everything comparison (same seed, same 12 epochs, same data split,
freshly re-run baseline `MM_baseline_pilot_seed0` since E70's own MM_seed0
checkpoint only preserved its final epoch-26 state, not a matched-epoch
snapshot):

| Run | per-subject Dice (ep 12) |
|---|---|
| MM baseline (unweighted) | 0.8934 |
| FWL v1 (softmax T=1.0) | 0.8859 |

**Mean Dice: −0.75pp, statistically significant** (paired t p=0.0001,
Wilcoxon p<0.0001). Only 25% of subjects improved.

**Mechanism-consistency check** (pre-declared): Spearman(ĥat{d}_i,
per-subject delta) = **+0.2096** (parametric p=0.019, permutation
p=0.019) — **significant and correctly signed**. FWL v1 does
disproportionately help causally-fragile subjects, exactly as designed —
but the cost imposed on easy subjects (downweighted toward ~0.4x) outweighs
that benefit at this strength. A genuinely distinguishable outcome from a
premise failure: the mechanism works directionally, the net trade is bad.

### v2 — linear floor=0.7 (single pre-registered re-test, gentler weighting)

Per the user's explicit choice to try one gentler weighting rather than stop
after v1. Weight range compressed from v1's ~0.4x–2.5x to ~0.83x–1.19x,
reusing the identical (frozen, already-computed) ĥat{d}_i table — isolating
weighting-function shape as the only variable under test.

| Run | per-subject Dice (ep 12) |
|---|---|
| MM baseline (unweighted, same run as above) | 0.8934 |
| FWL v2 (linear floor=0.7) | 0.8924 |

**Mean Dice: −0.10pp, NOT significant** (paired t p=0.53; Wilcoxon p=0.013,
a small but not decisive distributional signal).

**Mechanism-consistency check**: Spearman(ĥat{d}_i, delta) = **+0.0675,
NOT significant** (p=0.45). At this gentler strength the mechanism no
longer preferentially helps fragile subjects at all.

### Combined reading and verdict

| Weighting | Mean Δ Dice | Mechanism check | Verdict |
|---|---|---|---|
| v1 (aggressive) | −0.75pp (sig.) | ρ=+0.21, p=0.019 (consistent) | targets correctly, net cost too high |
| v2 (gentle) | −0.10pp (n.s.) | ρ=+0.07, p=0.45 (not consistent) | cost shrinks, but so does the mechanism |

Across the two tested points, cost and mechanism-targeting shrink together,
not independently — there is no visible regime where FWL both concentrates
benefit on causally-fragile subjects AND avoids a net cost to easy ones.
This is consistent with, and further evidence for, E71's own gate-sweep
finding (fact 4): bottleneck- and skip-pathway dependence are correlated
markers of *general* subject difficulty, not a separable axis a global
per-subject scalar can exploit for free. Applying the causal-fragility
signal via two structurally different mechanisms — pathway routing (E71,
killed) and loss-level reweighting (E72, killed here) — has now hit the
same underlying wall for a related diagnosed reason.

**Decision, per the project's own no-rescue / one-re-test discipline: FWL is
KILLED.** No further weighting-scheme variants will be attempted.

## What survives

- Prediction 1 (E71) remains real and validated: this network's bottleneck
  representation carries genuine, causally-grounded, recoverable
  self-knowledge of its own ablation fragility (ρ=0.87-0.866 held-out,
  independent of lesion size).
- New from E72: that signal, even though real, does not currently have a
  working DOWNSTREAM USE across the two mechanistic verbs tried
  (routing, loss-reweighting) — both fail for a related reason (difficulty
  is a shared, non-partitionable property of a subject, not a resource to
  reallocate). This narrows, rather than closes, what a future use of the
  signal would need to look like: it would have to act on something other
  than a per-subject scalar trade-off between "more" and "less" attention.

## Artifacts

- `experiments/exp_e12_eggo_m/e72/check_fwl_weighting_sanity.py` (stage 1, PASS)
- `experiments/exp_e12_eggo_m/e72/train_e72_fwl_pilot.py` (v1)
- `experiments/exp_e12_eggo_m/e72/train_e72_fwl_pilot_v2.py` (v2, gentler)
- `experiments/exp_e12_eggo_m/e72/train_e72_mm_baseline_pilot.py` (matched baseline)
- `experiments/exp_e12_eggo_m/e72/check_fwl_mechanism_consistency.py` (v1 check)
- `experiments/exp_e12_eggo_m/e72/check_fwlv2_mechanism_consistency.py` (v2 check)
- `runs/FWL_seed0/`, `runs/FWLv2_seed0/`, `runs/MM_baseline_pilot_seed0/`
  (checkpoints, per-epoch metrics, per-subject result.json each)
