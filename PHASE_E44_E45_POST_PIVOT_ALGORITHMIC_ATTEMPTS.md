# Phase E44–E45: Post-Pivot Algorithmic Attempts (RCGW, D4+D8)

**Date compiled**: 2026-08-19

**Purpose**: this document is self-contained. It records the two algorithmic-intervention attempts made after the project's explicit strategic pivot (documented in the user's own "NeuroScan / EGGO-M — Complete Project Requirements" message, saved to memory as `project_requirements_and_status_v2`): stop open-ended mechanism-hunting, and instead take validated D4 supervision straight to one algorithmic modification, tested via controlled training, judged against the project's own ≥1.0 percentage-point Dice bar.

**Context inherited from the E1–E43 arc** (see `PHASE_RESEARCH_ARC_MASTER_REPORT.md` and the project-requirements memory for full detail): the only confirmed positive result in the project's history is Deep Supervision, D4-only variant, reaching 0.9096 pooled Dice on the canonical baseline (0.9063), a +0.33 percentage-point gain — short of the required +1.0pp bar. Eight separate mechanistic explanations for *why* D4 helps were tested and killed across E34–E43 (Gaussian context, grid entropy, target-transformation error, gradient residual magnitude, gradient diversity, cross-scale consistency, boundary-localized representation change, among others). The project's own conclusion at that point: D4 supervision works, but its mechanism could not be identified from the available trained models — and further mechanism-hunting was explicitly ruled out as the next step.

---

## 1. Phase E44 — Relative-Convergence-Gap Weighting (RCGW)

### 1.1 Idea

Rather than another diagnostic phase, E44 attempted a genuine algorithmic modification: replace D4's *fixed* auxiliary loss weight (λ_ds3 = 0.9927, unchanged across the entire E25/E39 project history) with a *dynamic*, per-epoch weight computed from the model's own measured training trajectory. The idea was grounded directly in a **surviving** finding from the E1–E43 arc (not a killed one): D4's own auxiliary loss stays substantially larger, relative to its own starting value, than the main loss does relative to *its* own starting value — especially late in training. A fixed weight, the reasoning went, under-weights D4 exactly when D4 still has the most relatively-unconverged signal left to contribute.

### 1.2 Formulation 1 — raw ratio (killed)

```
r_D4(t) = L_D4(t) / L_D4(0)
r_0(t)  = L_0(t)  / L_0(0)
G(t)    = r_D4(t) / (r_0(t) + eps)
lambda(t) = lambda_base * clip(G(t), 0.2, 4.0)
```

Piloted for 15 epochs (seed 0, reduced schedule, compared directly against D4-only's own known epoch-by-epoch reference trajectory from `experiments/exp_e12_eggo_m/e25/deep_sup_runs/DeepSup_D4only_seed0/epoch_metrics.csv`). Result: `G(t)` grew essentially **linearly** across the entire 15-epoch window (1.0 → 2.04, epoch 1 → epoch 15), with no sign of leveling off — the external clip bound (4.0) was being approached *by trend*, not held as a safety margin never reached. This directly caused a real training instability: at epoch 6, validation Dice collapsed to 0.7135 versus the reference's 0.8463 at the same point, exactly as `G(t)` first crossed 1.0. **Killed** — the raw ratio has no restoring force once the auxiliary loss plateaus relative to main, which (per the project's own prior finding) is the normal state for most of training, not a transient.

### 1.3 Formulation 2 — saturating transform (killed)

Root cause diagnosed precisely before attempting a fix (not a blind re-tune): the raw ratio is structurally unbounded. Replaced with a mathematically bounded transform of the *same* underlying signal:

```
g(t) = ln( r_D4(t) / (r_0(t) + eps) )
lambda(t) = lambda_base * exp( K * tanh( g(t) / S ) )
```

with `K = ln(4)` (matching Formulation 1's own intended bounds, 0.25×–4× base, but reached as a smooth asymptote rather than an external clip) and `S = 1.0`. Verified numerically before the pilot: `g(t)=0` (D4 and main progressing at the same relative rate) maps to `lambda(t) = lambda_base` exactly, recovering the already-validated fixed-λ condition as a fixed point.

Piloted for 15 epochs, same protocol. Result: the instability was **fully resolved** — no collapse anywhere, the saturating factor grew smoothly (1.0 → 2.33 by epoch 14), well inside the 4.0 asymptote. But Dice performance was **flat-to-slightly-negative** relative to the plain D4-only reference: mean difference across epochs 2–14 was −0.35 percentage points, with wins and losses roughly canceling (final epoch-14 value 0.8983 vs. reference 0.8974, a statistically meaningless +0.09pp). **Killed** — once the instability that plausibly drove Formulation 1's apparent late-epoch "gains" is removed, no real signal remains. Convergence-gap-based auxiliary weight scheduling, in either form tested, does not add value beyond D4-only's already-tuned fixed weight.

### 1.4 Distinction from prior art (checked, not exhaustively searched)

RCGW's ratio-of-relative-progress formulation sits close to established multi-task-learning literature (GradNorm-family gradient-balancing methods; loss-ratio-at-step-t-vs-step-0 scheduling; homoscedastic-uncertainty weighting — all confirmed as active, populated areas via a targeted 2025–2026 literature check). This was disclosed explicitly in the implementation rather than claimed as an unrelated invention. The result made the novelty question moot — the mechanism was killed on its own empirical merits before any novelty claim would have mattered.

### 1.5 Outcome

**Both formulations killed. The "adaptive auxiliary weighting" family is closed** as a route to the ≥1pp target. No third variant was attempted, per the project's own explicit "no post-hoc threshold movement" and "don't keep tuning a null result indefinitely" rules.

---

## 2. Phase E45 — D4 + D8 combined deep supervision

### 2.1 Idea

Following E44's closure, the next direction was architectural rather than another scheduling trick: extend the *already-validated* mechanism (coarser auxiliary supervision helps, and the coarsest existing stage, D4/16³, outperforms both D2/32³ and the D4+D2 combination) one step further in the same geometric progression. The model's own bottleneck (8³ resolution, 256 channels, already computed in every forward pass via `pool1→pool2→pool3→bottleneck`, verified directly against `neuroscan_3d_fixed.py`'s shared trunk) had never been given its own supervision signal.

### 2.2 Architecture

New file `neuroscan_3d_v4.py`, class `UNet3D_v4`, extends `UNet3D_v3` via subclassing only — v1 (`neuroscan_3d_fixed.py`), v2 (`neuroscan_3d_v2.py`), v3 (`neuroscan_3d_v3.py`) remain frozen and untouched, per the project's established lineage convention. Adds one new head, `aux_head_d8`: a 1×1×1 convolution (256→1 channels) + sigmoid, architecturally identical in form to every existing head in the lineage, reading the bottleneck directly (un-detached, so gradient reaches the bottleneck's own parameters, matching `aux_head3`/`aux_head2`'s own convention). Verified directly: forward pass produces `aux_probs_d8` at exactly (B, 1, 8, 8, 8); total parameter count increases by 257 over v3 (5,602,822 → 5,603,079).

### 2.3 Calibration

`avg_pool3d(masks, kernel_size=8, stride=8)` used as the D8 target, matching the exact convention already used for D4/D2 (mass-conserving, verified directly on a synthetic mask before use). Loss weight `lambda_d8` calibrated with the *same* discipline used for every prior constant in this project's history (`experiments/exp_e12_eggo_m/e25/e25b_calibrate_deep_supervision.py`'s own method, reused): measure `FocalTverskyLoss`'s value at fresh initialization (seed 0, `.train()` mode — never `.eval()` on a fresh model, per an earlier project lesson about BatchNorm's large train/eval discrepancy at initialization) and set λ so the new term's *value* contribution matches the main segmentation loss's own value.

This value-matched calibration (λ_d8 = 1.0010) was **not accepted uncritically**: per an earlier project lesson (a prior calibration bug caused a ~75× gradient-magnitude blowup and a full training collapse, from calibrating by loss value alone without checking gradient magnitude), gradient magnitude was checked explicitly before trusting the value-matched number. It failed the check: the unweighted D8 gradient was 2.18× the main loss's own gradient magnitude — real, and outside a conservative 1.5×/0.67× tolerance band (deliberately set well below the historical collapse threshold, not because 2.18× alone was necessarily unsafe). Recalibrated by gradient magnitude instead: **λ_d8 = 0.4581** (final, used for training). `lambda_ds3 = 0.9927` reused unchanged from every prior D4 condition.

### 2.4 Training protocol

Full 30-epoch run, seed 0, canonical protocol (AdamW, cosine LR, batch size 8, dataset split, preprocessing — all unchanged from every prior condition). A pre-declared instability kill condition (validation Dice < 0.5 after epoch 5 triggers an immediate `RuntimeError`, matching the project's own established practice of investigating instability immediately rather than letting a doomed run finish) was implemented and never triggered.

A 2-epoch smoke test was run first and inspected before committing to the full run (val Dice 0.505 → 0.690, healthy climb, no anomalies, plausible early-training divergence from the D4-only-only reference given the new competing loss term).

### 2.5 Result

Trained cleanly end-to-end: no crashes, no NaN/Inf, no instability, monotonically improving training Dice throughout, stable validation precision/recall/HD95/ECE. Full 30-epoch trajectory:

| Epoch | Val Dice | Epoch | Val Dice | Epoch | Val Dice |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5050 | 11 | 0.8781 | 21 | 0.8857 |
| 2 | 0.6898 | 12 | 0.8914 | 22 | 0.9067 |
| 3 | 0.8065 | 13 | 0.8916 | 23 | 0.9040 |
| 4 | 0.8452 | 14 | 0.8973 | 24 | 0.9082 |
| 5 | 0.8592 | 15 | 0.9005 | 25 | 0.9104 |
| 6 | 0.8794 | 16 | 0.8979 | 26 | 0.9098 |
| 7 | 0.8784 | 17 | 0.8922 | 27 | 0.9097 |
| 8 | 0.8670 | 18 | 0.9035 | 28 | **0.9110** |
| 9 | 0.8835 | 19 | 0.9042 | 29 | 0.9099 |
| 10 | 0.8940 | 20 | 0.9030 | 30 | — |

*(indices here are 1-based epoch-of-training; the raw CSV logs 0-indexed epochs 0–29, identical data)*

**Best validation Dice: 0.9110** (epoch 29, 0-indexed 28).

### 2.6 Verdict against the pre-declared criteria

| Comparison | Value | Delta | Result |
|---|---:|---:|---|
| Canonical baseline (0.9063) | 0.9110 | **+0.47pp** | Required ≥1.0pp — **not met** |
| D4-only established best (0.9096) | 0.9110 | +0.14pp | Secondary check — modest edge only |

The pre-declared success criterion (Dice ≥ 0.9163, i.e. ≥1.0 percentage point over the canonical 0.9063 baseline) was **not met**. This is reported plainly, not reframed at a lower threshold: the run was clean and the result real, but it falls well short of the bar the project set for itself. **No further single-seed tuning of this exact configuration** — per the project's own kill/stop discipline, a 0.47pp gain on one seed does not justify additional compute chasing seed-level noise on this specific setup.

---

## 3. Where this leaves the project

Two genuinely different algorithmic directions were tried in direct response to the strategic pivot, both properly scoped (mathematical definition, mechanistic justification tied to a *surviving* project finding, pre-declared success/kill criteria, smoke-tested before full commitment, calibration checked by gradient magnitude not just loss value): a dynamic auxiliary-weighting schedule (killed twice, two formulations) and an architectural extension adding a new, coarser auxiliary head (trained clean, real but sub-threshold result). Neither reached the required ≥1.0pp bar.

This document, together with `project_requirements_and_status_v2` (canonical requirements/status memory) and `phase_e44_rcgw_killed` / `phase_e45_d4_d8_below_target` (phase-specific memory summaries), constitutes the complete, self-contained record of this stage of the project as of 2026-08-19. The next direction is deliberately not specified here — see the live conversation record for the redirect decision that follows this document.

## 4. Artifacts on disk

- `neuroscan_3d_v4.py` — new architecture (project root, git-untracked as of this writing).
- `experiments/exp_e12_eggo_m/e44/` — `train_e44_rcgw.py`, `train_e44b_rcgw_saturating.py`, both pilot run logs. Trajectory JSONs for both pilots were deleted after each was individually reviewed and closed (per this phase's own cleanup, matching the project's practice of not retaining data from a killed condition indefinitely) — the epoch-by-epoch numbers are preserved in Section 1 of this document and in the pilot log text files.
- `experiments/exp_e12_eggo_m/e45/` — `calibrate_lambda_d8.py`, `E45_lambda_d8_calibration.json` (full calibration record, both value-matched and gradient-matched numbers), `train_e45_d4_d8.py`, `e45_full_run_log.txt` (complete 30-epoch log), `runs/D4_D8_seed0/` (full checkpoint set at epochs 1/5/10/15/20/25/30 plus best.pth, and `epoch_metrics.csv` with the complete per-epoch metric history).
