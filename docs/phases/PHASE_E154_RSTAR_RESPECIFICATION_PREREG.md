# E154 — Pre-registration: $R^*$ Re-specification and Checkpoint-Invariance (Gate B′)

**Date**: 2026-09-15
**Status**: PRE-REGISTERED. Not yet run.
**Runs before**: E155 (Gate C). E155 is blocked on this.

---

## Why this experiment exists

E147 reported a per-subject minimum retained-rank requirement `Rstar_self` and gave it
provisional GREEN. Two objections were raised against building Gate C on top of it:

1. The generating script was missing, so `Rstar_self` had an unstated threshold, unstated
   epsilon, and unstated rank-grid semantics — three free parameters. With three free
   parameters a `1..32` spread is not evidence of heterogeneous demand.
2. E129 established that per-stage `N_k` magnitudes are **run-dependent** across equally-good
   checkpoints. If `Rstar_self` inherits that run-dependence, it is a property of one
   optimization trajectory, not of the subject, and every downstream gate is measuring noise.

Objection 1 is now **resolved by recovery** (below). Objection 2 is **still open** and is what
this experiment tests.

---

## Part 0 — Recovery result (already completed, 2026-09-15)

`Rstar_self` has been fully reverse-engineered from the stored artifact. It is **not**
under-determined.

**Recovered definition.** For each subject, `self_ag` is a 9-point agreement curve. `Rstar_self`
is the first grid point at which agreement reaches **0.90**, on the dyadic grid

```
[1, 2, 4, 8, 16, 32, 64, 128, 256]
```

**Verification** (`experiments/exp_e12_eggo_m/E147_repdemand.json`, n=88):

| Threshold | Exact matches |
|---:|---:|
| 0.80 | 58/88 |
| 0.85 | 66/88 |
| **0.90** | **88/88** |
| 0.92 | 81/88 |
| 0.95 | 60/88 |

88/88 at exactly 0.90 and nowhere else. Independently, recomputing E147's headline statistics
from the recovered rule reproduces the doc to 4 decimals:

- `rho(ap, R*) = -0.6385` (doc: `-0.638`)
- `rho(baseDice, R*) = -0.4369` (doc: `-0.437`)

**Threshold sensitivity** (limitation 4 of the E147 doc, now auditable):

| Perturbation | Subjects whose `R*` changes | Spearman vs 0.90 |
|---|---:|---:|
| 0.88 | 8/88 | 0.949 |
| 0.89 | 4/88 | — |
| 0.91 | 3/88 | — |
| 0.92 | 7/88 | 0.906 |

`R*` is robust to ±0.02 in the threshold. The exact value 0.90 is not load-bearing.

**Consequence for the ledger.** E147 limitations 2 and 4 are **partially resolved**. The
*thresholding rule* applied to the stored curve is recovered, and threshold sensitivity is now
auditable. What is recovered is the **last step** of the pipeline.

### What is STILL missing — do not overstate this

A repo-wide search (`self_ag`, `Rstar`, across all `*.py`, excluding venvs) returns **zero
hits**, and there is **no `e146`/`e147`/`e148` experiment directory** — only the bare JSON
artifacts at `exp_e12_eggo_m/` top level. The generating script does not exist on disk.

Therefore the following remain **unknown**:

1. **What `self_ag` measures.** "Agreement" between what and what? The most likely reading is
   agreement between the model's prediction under a rank-`k` truncation and its own full-rank
   prediction (`_self`), versus against ground truth (`_gt`) — the `self`/`gt` suffix pairing
   supports this. **But this is inference, not recovery.**
2. **Where the truncation is applied** (which layer/tensor), and **how** (SVD on what matrix,
   over which axes).
3. **Whether `_ag` is Dice, IoU, voxel agreement, or AP-based.**
4. `Rstar_gt` has **no stored curve** (`gt_ag` is absent from the JSON), so its rule is
   recovered only by analogy and must not carry weight.

**This blocks the stated protocol.** "Recompute `R*` on three checkpoints" cannot be executed
until 1–3 are pinned down, because the recovered threshold rule operates on a curve we cannot
currently regenerate.

### Required Part 0b — reconstruct the curve generator BEFORE Part 1

Before B′ can run, one of the following must be done, in preference order:

- **(i) EXHAUSTED 2026-09-15 — the generator does not exist.** Checked: no `self_ag`/`Rstar`
  hit in any `*.py` repo-wide; no `e146`/`e147`/`e148` directory; `git log --all
  --diff-filter=D` shows no deleted E14x/E15x script; `git log --all --name-only` matching
  `e14[0-9]|e15[0-9]` returns **only** `docs/phases/PHASE_E147_...md` — the doc was committed,
  the code never was. E147/E148/E151/E152/E153 exist solely as loose JSON at
  `exp_e12_eggo_m/` top level with no owning directory. Route (ii) is therefore forced.
- **(ii)** If it is unrecoverable, **re-specify the curve from scratch and say so plainly.**
  The re-specification should be anchored to E126's validated quantity rather than to a guessed
  reconstruction: define `R*_i` as the smallest `k` on the dyadic grid such that reconstructing
  the pool3 window with `k` components leaves bottleneck necessity `N_b` within `delta` of
  intact. That has **one** stated parameter (`delta`) instead of three unstated ones, and
  inherits E126's causal validation.

If (ii) is taken, the resulting `R*` is a **new variable**, not E147's. It must be reported as
such, and E147's `rho = -0.638` / `-0.496` must be **recomputed against it**, not inherited.
Carrying E147's correlations over to a re-specified `R*` without recomputation would be exactly
the E153 failure mode — trusting a number whose definition changed underneath it.

### Two facts the recovery surfaces that were NOT previously in the ledger

**(a) The grid is dyadic and the observed maximum is 32 — the 6th of 9 points.** Every subject
crosses 0.90 by grid point 6; none are censored. So `R*` is a **6-valued ordinal variable**, not
a continuous rank in `1..256`. The apparent `1 -> 32` range is five dyadic steps. Any Gate C
correlation bar must be interpreted against a 6-level ordinal outcome, and rank-based statistics
(Spearman) are mandatory — Pearson on `R*` would be measuring the arbitrary dyadic spacing.

**(b) 31/88 agreement curves are non-monotonic.** Median dip 0.0018, p90 0.0159, **max 0.179**.
The first-crossing rule and a stable-crossing rule (smallest `k` such that all later points stay
above threshold) disagree for only 1/88 subjects, so this does not materially change `R*` —
but it means the agreement curve is not a clean saturating function, and the one large dip
(0.179) should be inspected rather than assumed benign.

---

## Part 1 — The actual experiment (Gate B′)

### Question

Is `R*` a property of the **subject**, or of the **checkpoint**?

### Pre-registered hypothesis

$R^*_i$ computed on independently-seeded, equally-good checkpoints of the same architecture
will agree across seeds.

### Protocol

1. Recompute `R*` from scratch using the recovered rule (dyadic grid, threshold 0.90) on
   **three independently-seeded v5 checkpoints**, over the same 88 subjects.
2. Report pairwise Spearman `rho(R*_seedA, R*_seedB)` for all three pairs.
3. Report the exact-agreement rate (fraction of subjects landing on the identical grid point).

### Checkpoint availability — VERIFIED ON DISK

"v5" is the architecture string `HybridMiniSwin2D5_CSRF`. This was confirmed by matching
`e128/runs/Control_v5amp_seed0` (`best_val_dice = 0.9112618714570999`) against E129's
`v5_amp_control.best_val_dice = 0.9112618714570999` — exact match.

**Constraint discovered during the disk audit.** There are two distinct model families in this
repo and they must not be mixed:

| Family | `in_channels` | Task | Patch |
|---|---:|---|---|
| single-modality | 1 | `Dataset/Training`, 64³ | 64³ |
| BraTS-2023-GLI 4-mod | 4 | `brats2023gli_4mod_3region` | 128³ |

E147/E148's ET/TC analysis is on the **4-modality 3-region** family. Confirmed 4-modality
checkpoints on disk:

- `e131/runs/E131_v5control_seed0` — `arch='v5'`, seed 0, mean dice 0.8929
- `e130/runs/E130_baseline_seed0` — `arch='v5_4in3out'`, seed 0, mean dice 0.8631
- `e141/runs/E141_evidence_seed0` — `arch='v5_4in3out'`, seed 0, mean dice 0.8922
- `e70/runs/MM_seed{0,1,2}` — `in_channels=4`, seeds 0/1/2, modality order `[T1,T1ce,T2,FLAIR]`

### Checkpoint audit — COMPLETED 2026-09-15

All candidates were compared by **state-dict keys, tensor shapes, and relative weight
distance**, not by parameter count alone.

**Option 1 — `e70/runs/MM_seed{0,1,2}`: REJECTED.** Despite `in_channels=4`, these are *not*
the E147 architecture:

- 114 tensors vs 118 — the **entire `attn_gate1` block is absent** (`W_g`, `W_x`, `W_psi`).
- 10 shape mismatches, all in the heads: `evidential_head` is `(2,32,1,1,1)` vs `(6,32,1,1,1)`,
  `boundary_head.bias` and `aux_head3.0.bias` are `(1,)` vs `(3,)`.

These are **1-output-channel binary (WT-only)** models. E147's analysis is 3-region ET/TC/WT.
Running B′ on them would silently answer a different question. Rejected.

**Option 2 — `E130_baseline_seed0`: REJECTED.** `epoch: 1`, mean dice 0.8631. Pairwise relative
weight distance to the other arms is **3.37 and 4.87** — an order of magnitude beyond the
0.76–0.89 seen among converged checkpoints. This is an undertrained artifact, not an
equally-good checkpoint. Including it would manufacture a spurious B′ failure.

**Verified structurally identical to `E131_v5control_seed0`** (keys equal, zero shape
mismatches, 118 tensors, 5,613,521 params):

| Checkpoint | arch | epoch | mean dice | rel. weight dist. to v5control |
|---|---|---:|---:|---:|
| `e131/runs/E131_v5control_seed0` | v5 | 33 | 0.8929 | — (reference) |
| `e141/runs/E141_evidence_seed0` | v5_4in3out | 31 | 0.8922 | 0.8142 |
| `e141/runs/E141_shuffled_seed0` | v5_4in3out | 19 | 0.8899 | 0.7640 |
| `e135/runs/E135_gamma2_seed0` | v5 | 2 | 0.8499 | 0.8601 — **exclude, epoch 2** |

Divergences of 0.76–0.86 are comparable to E129's 0.664 across equally-good checkpoints, so
these are genuinely different points in weight space.

### The remaining problem, stated honestly

**All of these are `seed=0`.** Inspecting `e141/train_e141_evidence.py` confirms
`set_seed(0)` and an identical `RandomState(42)` validation split across arms. They differ by
**training objective** (`--arm baseline|evidence|shuffled`, which reweights the FocalTversky
ET term by a label-blind evidence score), not by initialization or data order.

This matters for interpretation:

- They **are** valid for "is `R*` stable across *equally-good but differently-trained*
  checkpoints" — which is E129's own notion of run-dependence, and is the question that
  actually threatens the branch.
- They are **not** valid for "is `R*` stable across *random seeds*". Seed-invariance remains
  formally untested.
- E141's evidence/shuffled arms were **KILLED** as interventions (−0.37pp / −0.63pp). That is
  irrelevant here — a failed intervention still yields a legitimate equally-good checkpoint —
  but it must be stated so the B′ result is not later misread as endorsing E141.

### Pre-registered checkpoint set for B′

**Primary (objective-variation, 3 checkpoints, zero training cost):**
`E131_v5control_seed0` · `E141_evidence_seed0` · `E141_shuffled_seed0`

**Secondary (training-time variation, same run):** `E131_v5control_seed0` epochs
**005–035 only**. Epochs 040/045/050 are **excluded**: the run suffers the documented
epoch-39 BatchNorm collapse — TC falls 0.914 → 0.190 and mean dice 0.885 → 0.525. Using them
would inject a degenerate checkpoint. `E141_evidence_seed0` epochs 005–040 are clean
(monotone to 0.8893) and may be used as a second within-run series.

This secondary series is **correlated by construction** and is reported as supporting evidence
only; it cannot carry a B′ pass on its own.

**Option 3 (train fresh seeds)** is deferred. It is the only route to a true seed-invariance
claim, but it costs training, and a B′ failure on the primary set would kill the branch anyway —
so it is wasteful to pay that cost before seeing the free result. If the primary set PASSES,
train two fresh seeds to convert the claim from objective-invariance to seed-invariance
**before** relying on it for Gate D.

**Do not** silently substitute the 1-channel `HybridMiniSwin2D5_CSRF` family (`exp00b`,
`e54`, `e49`, `e50`, `e128/Control_v5amp`). Those are single-modality 64³ binary runs on
`Dataset/Training`; despite "v5" naming they are a different task from E147's
`brats2023gli_4mod_3region` at 128³.

### Subject set — audited 2026-09-15

Three different `n` appear across these artifacts and must not be conflated:

| Artifact | n | Relationship |
|---|---:|---|
| validation split | 125 | `RandomState(42)` shuffle, last 10% held out |
| `E148_probes.json` | 90 | subset of val |
| `E147_repdemand.json` | 88 | **strict subset of E148** (overlap 88/88, nothing in E147 absent from E148) |

The two subjects in E148 but not E147 are `BraTS-GLI-00021-000` and `BraTS-GLI-00731-001`.
The 35 subjects in val but not E148 are **not yet explained** and remain E147 limitation 1.

**Consequence for B′:** the validation split is seeded by `RandomState(42)` *independently of
the run seed*, and the E141 source documents that the subject-ID lists were verified to match
across arms. So all three primary checkpoints see the **identical 125-subject val set**, and
`R*` can be recomputed on the identical 88 subjects. The comparison is like-for-like.

**Pre-registered requirement:** B′ must recompute `R*` on **exactly the 88 E147 subject IDs**,
asserted by ID against the stored artifact — not on "the first 88", not on a fresh subsample.
Any subject that fails to produce a curve must be reported, not silently dropped.

### Pre-registered decision rule

Applied to the **3 primary checkpoints** (3 pairs). Because the primary set varies *training
objective* rather than *random seed*, a PASS licenses the claim "`R*` is invariant across
equally-good checkpoints" — which is exactly E129's notion of run-dependence and the one that
threatens the branch — but **not** "`R*` is seed-invariant". Word the verdict accordingly.

| Outcome | Verdict |
|---|---|
| all pairwise `rho >= 0.7` | **PASS.** `R*` is a checkpoint-invariant subject property. Gate A → GREEN (objective-invariance). Proceed to E155; train fresh seeds before Gate D. |
| any pairwise `rho` in `[0.4, 0.7)` | **PARTIAL.** `R*` is part signal, part trajectory. Record the attenuation ceiling (below) and re-derive E155's bars before running it. |
| all pairwise `rho < 0.4` | **KILL.** `R*` is a trajectory artifact. The adaptive-computation thesis does not survive. Do not run E155. |

**Also report** the exact-agreement rate (fraction landing on the identical dyadic grid point).
With a 6-valued ordinal and a modal class holding 43/88, a high Spearman can coexist with poor
exact agreement; both numbers are needed to read the result honestly.

**Chance baseline, pre-registered.** Because `R*` is 6-valued and heavily concentrated
(43/88 at rank 4), a naive agreement rate is inflated by the marginal distribution alone.
Report **Cohen's quadratic-weighted kappa** alongside Spearman, and report the agreement rate
expected from shuffling one seed's labels, so the reader can see the floor.

### The attenuation ceiling — why this must run first

If cross-seed reliability is `r`, then the maximum correlation any early predictor could
achieve with `R*` is bounded roughly by `sqrt(r)`. Concretely: if `r ≈ 0.5`, the ceiling on
Gate C's C1 is about `0.71`; if `r ≈ 0.25`, the ceiling is `0.5` — **exactly E155's C1 bar**,
meaning Gate C would fail mechanically without saying anything about demand.

E154 therefore does not merely gate E155; it **sets E155's maximum observable effect**. Running
Gate C first would be uninterpretable.

---

## What this experiment cannot tell us

- Nothing about whether demand is predictable early (that is E155).
- Nothing about novelty.
- A B′ pass does **not** upgrade `R*` from "minimum retained-rank requirement" to "minimum
  computation". E147 limitation 3 stands: this is a representational-rank intervention, not a
  FLOP or dynamic-depth measurement.
- E147 limitation 1 also stands: 88 subjects, not the full 125-subject validation set. The
  missing 37 should be explained before any headline claim.

---

## Status

**PRE-REGISTERED, not run.**

Checkpoint question: **RESOLVED** — primary set is `E131_v5control_seed0`,
`E141_evidence_seed0`, `E141_shuffled_seed0` (structurally identical, converged, divergent
weights, identical val split, zero training cost).

Blocking item: **Part 0b**. The `self_ag` curve generator is confirmed non-existent
(route (i) exhausted), so B′ cannot run until `R*` is re-specified via route (ii) — anchored to
E126's `N_b` criterion with a single stated `delta`. That re-specification produces a **new
variable**, and E147's correlations must be recomputed against it rather than inherited.

### Honest summary of where the ledger actually stands

| Claim | Status after this audit |
|---|---|
| The 0.90 dyadic threshold rule | **Recovered**, 88/88, reproduces E147's stats to 4 dp |
| Threshold is not load-bearing | **Verified**, Spearman ≥ 0.91 under ±0.02 |
| `R*` is a 6-valued ordinal, not continuous `1..32` | **New finding**, weakens the heterogeneity read |
| 31/88 agreement curves non-monotonic (max dip 0.179) | **New finding**, curve is not cleanly saturating |
| What `self_ag` actually measures | **UNKNOWN** — generator never committed |
| `R*` is checkpoint-invariant | **UNTESTED** — this experiment, blocked on the above |

Gate A should read **YELLOW**, not GREEN, until Part 0b and Part 1 are complete.
