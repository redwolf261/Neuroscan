# Phase E24, Gate 6: Experimental Specification (locked before training)

**Status**: 🟢 Specification locked, diagnostic refactor complete and regression-verified (1,536/1,536 rows, exact match), pre-launch audit correction applied and verified — **ready for training authorization**. Incorporates seven methodological corrections from the design audit (initial-parameter-equality hash gate; E's random direction norm/seed constraints; per-condition-only objective comparison; E22's `+0.2432` as reference not threshold; the `run_counterfactual` interface with a hard regression gate; separated training-vs-counterfactual experiments; the new H2 realized-displacement measurement) plus **one further correction found during the pre-launch launch-configuration audit**: condition E's hinge threshold was initially (incorrectly) set to reuse B's `delta_d_w=0.2553`; E now has its own, separately-calibrated `delta_d_r=0.2022` (measured directly against E's actual frozen random vector, a real ~21% difference from `delta_d_w` — confirming the two were never interchangeable). Both `delta_d_w` and `delta_d_r` are calibrated, verified, and wired into `run_counterfactual.py`; the regression gate re-ran clean after the fix. The four-level H1→H2→H3→H4 interpretation hierarchy (Section 1) is the organizing structure for the entire gate. **No training has been launched — this status reflects that every identified pre-launch gap is now closed, not that training has started.**

**Date**: 2026-08-09

---

## 1. The interpretation hierarchy (revised per audit — locked before any run)

Not merely "primary vs. secondary" — a full four-level hierarchy, each level a distinct, separately-evaluated question. A result at a lower level does **not** substitute for a result at a higher level, and the report must present them in this order, not in whichever order the numbers happen to look most favorable.

- **H1 — Mechanistic (primary, decisive)**: does the new objective's own local descent direction become more task-useful than baseline's? Evaluated as each condition's **own** objective against its **own** Dice change — see Section 5's corrected classification rule (point 3/4 of the audit) for exactly what "more task-useful" means quantitatively, since E22's `+0.2432` is a historical reference point, not a pass/fail threshold.
- **H2 — Representation (secondary mechanistic confirmation)**: does training under B actually produce activation-space representation movement aligned with `d_useful`, and — the audit's new addition — does the *actual* small parameter update (not just the loss-level gradient) induce a `dec1` displacement aligned with `w_hat`? These are two different claims (loss-level gradient constraint vs. realized parameter-space effect) and both are measured, not conflated. See Section 4's new "actual displacement vs. w_hat" metric.
- **H3 — Segmentation (outcome)**: does training with `L_seg + λ·L_margin^w` improve Dice relative to A and C?
- **H4 — Specificity (causal attribution)**: does B outperform E (the random-projection control) — i.e., is any benefit attributable to alignment with `seg_head` specifically, not merely to collapsing the margin into *some* fixed 1-D subspace?

**Explicit anti-pattern this hierarchy is designed to block**: "B gets +1% Dice → therefore the proposed mechanism works." If H1 fails, any H3 improvement needs a *different* explanation and must be reported as unexplained, not attributed to the hypothesized mechanism. The report is required to state each level's result in order (H1, H2, H3, H4), never leading with H3.

---

## 2. The condition matrix (locked)

| Condition | Objective | Purpose |
|---|---|---|
| **A** | `L_seg + λ·L_margin` (`w_hat=None`, `delta_d=3.6659`) | Original EGGO-M baseline — reruns E12f's exact protocol, new run (not reusing the old E12f checkpoints) so A/B/E share identical seed/init/data-order, see Section 3 |
| **B** | `L_seg + λ·L_margin^w` (`w_hat`=live `seg_head` direction, `delta_d_w=0.2553`) | Proposed algorithm |
| **C** | `L_seg` alone (`λ=0`) | Segmentation-only control — **reused, not rerun**: this is exactly E18's existing `lambda_zero` config (same seed, same architecture, same epochs) per `PHASE_E23` Section 8's own note that this ablation needs no new run if seeds/epochs match. Verified below (Section 3) that they do. |
| **E** | `L_seg + λ·L_margin^random` (`r`= a **fixed-at-init, frozen-for-the-whole-run** random unit vector, `delta_d_r=0.2022` — **its own, separately-calibrated constant, corrected per pre-launch audit**, see note below) | Specificity control — isolates whether any benefit comes from task-alignment with `seg_head` specifically, or merely from collapsing the margin to *some* fixed 1-D subspace |

**Pre-launch audit correction (applied before any run)**: an earlier draft of this spec, and the first version of the parameterized `run_counterfactual` implementation, reused B's `delta_d_w=0.2553` for E as well. This is wrong: `w_hat` and `r` are different unit vectors, and their projected-distance distributions against the same fresh-init geometry are measurably different (`delta_d_w=0.2553` vs. the correctly-calibrated `delta_d_r=0.2022`, a ~21% difference — confirmed directly by running E12e's calibration procedure against the actual frozen `r` condition E uses, not assumed). Reusing `delta_d_w` for E would have confounded "is task-alignment useful" with "is a miscalibrated hinge threshold useful," defeating the H4 specificity comparison's purpose. Fixed via `experiments/exp_e12_eggo_m/e24/e24_calibrate_delta_d_r.py`, which calibrates against the bit-identical `r` (seed `999001`) that `run_counterfactual.py`'s `ObjectiveConfig(mode="random_projection")` constructs — verified identical by that script's own built-in cross-check before trusting the result, not merely assumed from a matching seed number.

**D is not run**, per the explicit reasoning accepted here: D would be baseline reproduced through B's code path with the projection reverted, which is exactly what Gate 5's own baseline-path smoke test (and the standalone unit test `test_baseline_preservation`) already regression-tested at the mechanism level. Re-running a full 30-epoch D would spend compute confirming something already established at the unit/smoke level, not testing a new hypothesis.

### Why E uses `w_hat` frozen-at-init, not resampled per batch

A random vector resampled every batch would define a *different* 1-D subspace each time — not a meaningful control for "does task-alignment matter, or does dimensional collapse alone help," since it would never let the margin loss consistently push along *any* stable axis, confounding "wrong axis" with "constantly moving axis" (a version of E17's moving-target problem, self-inflicted). Freezing `w_hat_random` once at initialization (same seed convention as B's own `w_hat`, which evolves with `seg_head` but starts from the same initial random weights) isolates the comparison to "a stable but task-irrelevant axis" vs. "a stable, task-relevant axis" — the actually decisive comparison Section 5's Candidate-comparison logic in `PHASE_E23` was reaching for.

---

## 3. Reproducibility controls (answering the audit's explicit checklist)

- **Seeds**: A, B, E all use **seed 0**, identical to E12f/E13's own established convention. `set_seed(0)` is called before model construction in every condition (per `EGGOMExperiment.__init__`'s existing, unmodified call to `set_seed(seed)` at line 275) — this fixes `torch.manual_seed`/`np.random.seed`/`random.seed` identically across A/B/E.
- **Initial-parameter-equality gate (added per audit)**: `set_seed` being called identically is necessary but, per the audit's correction, not sufficient to *assert* identical initialization — a subtle divergence in construction order (e.g. an extra random draw consumed by one condition's own setup code before the model is built) could desync the RNG state without any seed mismatch being visible. **Required check, run before any of the three conditions' real training starts**: after constructing each of A/B/E's fresh models, compute a hash (e.g. SHA-256 of the concatenated, flattened `state_dict` tensors) of `θ_A^{(0)}`, `θ_B^{(0)}`, `θ_E^{(0)}` and assert all three are bit-identical. Do the same for each condition's freshly-constructed `AdamW` optimizer's initial state (empty moment buffers, but the `param_groups` metadata — `lr`, `betas`, `eps`, `weight_decay` — must match exactly across all three). **This is a gate, not a formality**: if the hashes disagree, training must not proceed until the divergence is found and fixed, the same "stop and debug" standard E22's own Section 12 already established.
- **Data order**: `create_brats_loaders(..., shuffle=True)` with no explicit `generator=`, falling back to the global `torch` RNG — seeded identically pre-construction in every condition, per the same reasoning `PHASE_E18_FOLLOWUP_PRE_E19_REVIEW.md` already flagged and left as "asserted-by-construction, not independently re-verified" for E18's own conditions. **Same caveat applies here, carried forward honestly, not silently assumed stronger than it was for E18.**
- **Multi-seed**: **Seed 0 only for Gate 6.** E13's other 3 seeds are explicitly out of scope for this gate — per the same reasoning already used throughout this arc (E19–E22 all single-seed), extending to multi-seed is a natural, but not yet resourced, follow-up if Gate 6's single-seed result is positive on the primary (mechanistic) question.
- **λ**: `0.1` for A and B (E12f/E13's calibrated value, unchanged — per the explicit instruction not to change λ as part of this algorithmic redesign). E's λ is also `0.1`, applied to `L_margin^random`, for a true apples-to-apples magnitude comparison against B.
- **Epochs**: **30**, matching E12f/E13/E18's established protocol exactly, same checkpoint schedule (`{1,5,10,15,20,25,30}`, per `train_eggo_m.py`'s existing `checkpoint_every=5` default, confirmed in Section 508 of the file).
- **Optimizer/LR schedule**: unchanged `AdamW` + `CosineAnnealingLR`, same `configs/brats.yaml` hyperparameters, identical across A/B/E (no optimizer-level changes anywhere in this gate, per the explicit constraint carried from E23 onward).
- **`r`'s (E's random direction) freeze point and construction (tightened per audit)**: constructed once, immediately after model initialization (same point B's *own* `w_hat` would first be computed, epoch 1 batch 1), then held fixed for the entire 30-epoch run — a plain tensor, never updated, structurally guaranteed inert the same way `w_hat` itself is (detached, `requires_grad=False`, verified by the same unit-test pattern as `test_detach_correctness`). **Must satisfy the same shape/norm constraints as `w_hat` exactly**: `r ∈ ℝ^32`, `‖r‖_2 = 1` — implemented and verified in `run_counterfactual.py`'s `ObjectiveConfig`. Sign is irrelevant by construction (the distance function takes `|r·(z_i-z_j)|`), so no sign convention needs to be fixed. **Generated from a seed independent of the training RNG**: `torch.Generator().manual_seed(999001)`, not drawn from the same `torch.manual_seed(0)` state model construction and data loading consume, so that adding, removing, or reordering any other random draw in the training pipeline cannot silently change which direction E uses. This value is computed once, recorded in the run's saved config/checkpoint, and never regenerated. **`r`'s own hinge threshold, `delta_d_r=0.2022`, is separately calibrated against this exact `r`** (Section 2's pre-launch audit correction) — not borrowed from `w_hat`'s calibration.
- **`E22`'s reusability**: **confirmed NOT usable unmodified** on B's or E's checkpoints, per direct inspection (Section 2 above) — `DELTA_D_CALIBRATED` and `w_hat=None` are hardcoded at 5 call sites, `SEED_DIR` is a module constant. **Required pre-training-adjacent (but not itself training) work**: parameterize `e22_counterfactual_geometry.py` into a function/CLI accepting `checkpoint_dir`, `delta_d`, and an optional `w_hat`-construction mode, reusing its exact internal logic unchanged (same anchor-fixing, same `torch.manual_seed` discipline, same `compute_useful_direction` formula) — this is a refactor of already-audited code, not new experimental logic, and is scoped as the first concrete action of Gate 6, before condition A's training even starts (see Section 6).

---

## 4. Metrics preserved per checkpoint (per the explicit list, cross-referenced to which existing script produces each)

| Metric | Source script (reused, not reinvented) |
|---|---|
| Dice, Precision, Recall | Existing eval loop + small addition (Precision/Recall not currently logged, per `PHASE_E23` Section 7) |
| `L_seg` | Existing `epoch_metrics.csv` logging |
| Margin loss (`L_margin` or `L_margin^w` depending on condition) | Existing logging |
| Active-hinge % | Existing `active_hinge_pct` logging (already condition-aware, since it reads whichever `dist` the `w_hat` branch computed) |
| `‖g_seg‖`, `‖g_margin‖` | E19's exact per-block machinery (`e19_layerwise_gradient_attribution.py`, parameterized for B's `w_hat`/`delta_d_w` the same way E22 needs to be — same required refactor, Section 3) |
| `cos(g_seg, g_margin)` | E14's exact machinery, same parameterization need |
| Parameter-update norm | E20's exact `ShadowAdam` machinery, same parameterization need |
| Representation geometry (`mean_boundary_margin`) | `analyze_eggo_m_checkpoints_v2.py`'s exact formula, unchanged (operates on raw `dec1`, not `w_hat`-dependent, usable as-is on any condition's checkpoints) |
| E22 counterfactual directions (A/B/C/D/E per E22's own internal direction-lettering, not to be confused with this document's condition A–E) | Parameterized E22 (Section 3), the decisive test |
| `cos(Δz_margin, d_useful)` | Same parameterized E22 output |
| `corr(ΔL_margin, ΔDice)`, **computed per-condition against that condition's own objective** (see Section 5's correction — never cross-compared numerically across A/B/E) | Same parameterized E22 output — **the H1 measurement, the single most important number in this gate** |
| `cos(Δz_actual, ŵ)` — **new, per audit point 7**: the actual parameter-space update's realized activation-space displacement, projected onto `ŵ`, distinct from the already-verified loss-level `∂L_margin^w/∂z ∥ ŵ` constraint | Reuses E21's `apply_delta_and_forward` mechanism, parameterized for B's checkpoints — the H2 measurement |

**Explicit acknowledgment of the flagged risk**: active-hinge % (17.33% vs. 6.86% in the Gate-5 smoke run) is **not** interpreted as a success signal anywhere in this spec — noted here once, deliberately, so it doesn't need re-flagging in every later document. `d_w ≤ ‖z_i-z_j‖_2` always, so B trivially has a different active-pair rate than A by construction; this is a calibration-consistency check (already passed, Gate 5), not an outcome metric.

---

## 5. Success / falsification criteria (defined before any run; revised per audit points 3–4, 7–8)

### Critical methodological correction: each condition's objective is evaluated against itself, never cross-compared directly

`L_margin^A(z_i,z_j) = ‖z_i-z_j‖_2` and `L_margin^B(z_i,z_j) = |ŵ·(z_i-z_j)|` are **different functions of different scale and different meaning** — `d_w ≤ ‖z_i-z_j‖_2` always, by construction. A statement like "B reduced its margin loss more than A reduced its margin loss" is **not scientifically meaningful** and must never appear in any report from this gate. The E22-style test for each condition asks a question **local to that condition's own objective**: does locally descending *that condition's own* `L_margin` correspond to improving Dice, for that condition. Concretely:

- For **A**: `corr(ΔL_margin^A, ΔDice)` — this reproduces E22's own already-published measurement (`+0.2432`), used here as the historical reference point this gate is trying to move away from, not re-derived as a new result.
- For **B**: `corr(ΔL_margin^B, ΔDice)` — B's **own** objective against B's **own** Dice change. This is the number that answers H1, and it is never compared numerically against A's `L_margin^A` value (different units, different scale) — only its **sign, magnitude, and significance pattern** relative to E22's established reference points (`+0.2432` wrong-sign for baseline, `−0.8406` as the pipeline's own proof a strong negative correlation is detectable when one exists) are meaningful.
- For **E**: `corr(ΔL_margin^random, ΔDice)` — E's own objective (same functional form as B's, `|r·(z_i-z_j)|`, different, task-irrelevant `r`) against E's own Dice change, for the H4 specificity comparison.

### H1 (primary, mechanistic) — a predefined classification rule, not a single pass/fail threshold

Per the audit's explicit point: E22's `+0.2432` is a **historical reference**, not a performance bar to clear by a specific margin. The classification below distinguishes qualitatively different outcomes, all of which are informative and must be reported using this rule (not reinterpreted after seeing the number):

| `corr(ΔL_margin^B, ΔDice)` result | Classification |
|---|---|
| Still positive, materially unchanged from `+0.24`-ish | **H1 falsified** — mechanism not fixed |
| Positive but meaningfully smaller (e.g. `+0.24 → +0.05`) | **Partial improvement, still wrong-sign** — H1 not supported, but not identically falsified either; report as such |
| Crosses to negative but small/non-significant (e.g. `+0.24 → −0.05`, p not significant) | **Crosses into the hypothesized direction, weak evidence** — H1 weakly supported, flagged as weak explicitly |
| Negative and significant, comparable in strength to `A_seg`'s own `−0.84` sanity-check correlation | **H1 strongly supported** |
| Collapses toward exactly zero with no clear sign | **Objective becomes locally uninformative** — a distinct outcome from either "supported" or "falsified," reported as its own category, not forced into one of the other two |

**Inconclusive** (a cross-cutting flag, applicable to any row above) if results are mixed across the 6 checkpoints — reported honestly, per the same standard E22 applied to its own Falsification 5.

### H2 (secondary mechanistic confirmation) — two distinct claims, both measured

1. **Loss-level gradient constraint** (already verified, Gate 5): `∂L_margin^w/∂z ∥ ŵ`. Not re-litigated here — this is implementation correctness, not a new experimental question at Gate 6.
2. **Realized parameter-space effect** (new, per audit point 7): after the chain rule through `dec1`'s own parameters, `∇_θ L_margin^w = J_θ^T · (∂L_margin^w/∂z)` does **not** necessarily produce a `ŵ`-aligned activation-space *displacement* once the network's actual parameterization is accounted for — `J_θ^T` can spread that one direction across many parameter directions (the same activation-space-vs-parameter-space distinction `PHASE_E23` Section 6 already drew, now measured empirically rather than only argued mathematically). **Required measurement**: reusing E21's own `apply_delta_and_forward` mechanism (perturb `dec1`'s real parameters by the real, small optimizer update, re-forward-pass, measure the resulting `Δz_actual`), compute `cos(Δz_actual, ŵ)` for B's checkpoints. This is a **different, weaker claim than the loss-level constraint**, and the two must be reported separately, never conflated into one "the gradient is w-aligned" statement.

### H3 (outcome), evaluated and reported only after H1/H2

- B's final (epoch-30) Dice compared against A's and C's, paired by subject where possible, same statistical treatment E22 itself used (subject/volume as the unit of inference, mean±SD, no voxel-level inflation).

### Two separate experiments, not one (added per audit point 6)

Gate 6 contains two mechanically distinct experiments that must not be allowed to contaminate each other's definition:

- **Experiment 1 — Training comparison**: train A/B/C/E for 30 epochs, log Dice/`L_seg`/geometry/etc. across the full run. This produces the checkpoints and the H3/H4 numbers.
- **Experiment 2 — Counterfactual mechanism test**: take the resulting checkpoints (from Experiment 1) and, for each, ask "what happens if I locally descend this method's own geometric objective from this checkpoint" — the E22-style protocol, mechanically **identical** across A/B/E except for which objective (`L_margin^A`/`L_margin^B`/`L_margin^random`) is being locally descended. This produces the H1/H2 numbers.

**Experiment 2's protocol must not vary based on Experiment 1's outcome.** E.g., the choice of which checkpoints/subjects/epsilon values to test in the counterfactual rerun is fixed in this document (Section 3: same 6 checkpoints × 8 subjects as E22's own original run) *before* any training result is known, precisely so that a disappointing or promising training trajectory cannot influence how the mechanistic test is run or reported.

### H4 (specificity), evaluated alongside H3

- E's Dice and E's own `corr(ΔL_margin^random, ΔDice)` compared against B's. If E matches or exceeds B on either measure, task-alignment with `seg_head` specifically is not established as the causal driver of any observed benefit, independent of whether B "works" in isolation — the algorithmic contribution's core claim (`PHASE_E23` Section 0.5/0.6) would need reconsideration.

### Disciplined reporting rule (unchanged from the prior draft, restated)

If H1 is falsified or classified as "locally uninformative" but B's Dice (H3) is nonetheless higher than A's, the report must say so plainly — "the mechanism was not fixed, but Dice improved for reasons this gate did not establish" — never let a Dice number stand in for mechanistic success. The report presents H1 → H2 → H3 → H4 in that order, always.

---

## 6. Exact order of operations for Gate 6 (revised per audit — diagnostic refactor is a hard gate before training)

1. **Refactor the diagnostic machinery** (E22 primarily; E14/E19/E20 as needed for the full metric table) into a reusable interface, conceptually:

   ```python
   run_counterfactual(checkpoint_dir, objective_config, seed, ...)
   ```

   where `objective_config` selects `{baseline Euclidean, task-aligned (w_hat=live seg_head), random-projection (w_hat=frozen random)}` and supplies the matching calibration constant (`delta_d=3.6659` for baseline, `delta_d_w=0.2553` for task-aligned, `delta_d_r=0.2022` for random-projection — **each mode's own, separately-calibrated constant, per the pre-launch audit correction above; the two are NOT interchangeable**) and direction-construction seed — **the underlying measurement logic (anchor fixing, `torch.manual_seed` discipline, `compute_useful_direction` formula, Dice/loss/geometry scoring) stays byte-identical across all three `objective_config` values**, only the distance function and its calibrated threshold vary. This directly implements the audit's proposed interface, not a looser approximation of it.
2. **Hard regression gate**: run the refactored `run_counterfactual` against the **original E12f checkpoints** (baseline `objective_config`) and verify its output reproduces E22's own already-published numbers (`corr(ΔL_margin, ΔDice) = +0.2432, p=6.75e-04, n=192`; the `A_seg` sanity-check correlation `−0.8406`; the per-checkpoint tables from `PHASE_E22_COUNTERFACTUAL_OBJECTIVE_GEOMETRY.md`) to a predefined tolerance (exact match expected for a pure refactor with no logic change — any numeric drift beyond floating-point-level noise, e.g. more than 1e-6 relative difference, is treated as a refactor bug, not accepted as "close enough"). **This gate blocks every step below it. No training starts until this passes.**
3. Launch condition A (fresh 30-epoch run, seed 0, `w_hat=None`) — **not** a reuse of the old E12f checkpoints, so A/B/E are trained under byte-identical infrastructure/timing/environment, removing any "different PyTorch/CUDA version between old and new runs" confound.
4. **Before any of A/B/E's real training epochs run**: construct all three models fresh (same seed, same `EGGOMExperiment.__init__` path), compute and compare the initial-parameter-equality hashes (Section 3's new gate). If they mismatch, stop and diagnose before proceeding — do not let three divergent runs complete and only discover the initialization mismatch afterward.
5. Launch condition B (30 epochs, seed 0, `w_hat`=live `seg_head` direction, `delta_d_w=0.2553`).
6. Launch condition E (30 epochs, seed 0, `r`=frozen random unit vector generated from its own independent, documented seed (`999001`) per Section 3, `delta_d_r=0.2022` — **E's own separately-calibrated constant, not B's `delta_d_w`**).
7. Condition C: confirm E18's existing `lambda_zero` seed-0 run's config matches A/B/E's `configs/brats.yaml` byte-for-byte (epochs, LR schedule, batch size) before reusing its checkpoints — if anything differs, **rerun** C rather than silently accept a mismatched control, per the same rigor `PHASE_E13_CODE_AUDIT.md` applied to catching the LR-schedule mismatch in an earlier phase.
8. Run `run_counterfactual` (step 1's tool, now trusted per step 2's regression gate) on A's, B's, and E's checkpoints, **each against its own objective** per Section 5's correction — this produces the **H1 result**, evaluated and reported before any Dice comparison.
9. Run the new `cos(Δz_actual, ŵ)` measurement (Section 5, H2) on B's checkpoints.
10. Run E19/E14/E20's parameterized equivalents on A/B/E's checkpoints for the full metric table (Section 4).
11. Only then, compute and report H3 (outcome) and H4 (specificity), in that order, per Section 1's hierarchy.

**Nothing beyond step 1 (a code refactor, not a training run) has been started.** Step 2 (the regression gate) is the next concrete action and is itself not yet run. Steps 3 onward require your authorization to spend the actual training compute (3 fresh 30-epoch runs, plus reuse of an existing C if it validates).

---

## 7. What would change this plan

If step 2's regression check (parameterized `run_counterfactual` reproducing E22's own already-published baseline numbers, to the stated tolerance) fails, that blocks every step below it until fixed — the same "stop and debug before full execution" discipline E22's own Section 12 already established as this project's standard. Likewise, if step 4's initial-parameter-equality hash check fails, training does not proceed until the divergence is found and resolved.

---

## Files

| File | Purpose |
|---|---|
| `PHASE_E22_COUNTERFACTUAL_OBJECTIVE_GEOMETRY.md` | The falsification protocol this gate directly extends |
| `PHASE_E23_ALGORITHMIC_REDESIGN.md` | The design and novelty-audited hypothesis this gate tests |
| `PHASE_E24_CALIBRATION_AND_IMPLEMENTATION_SPEC.md` | The implementation this gate trains |
| `experiments/exp_e12_eggo_m/e18_ablation_train.py` (`lambda_zero` config) | Prospective source of condition C, pending the byte-for-byte config check in Section 6 step 5 |

---

**Status recap**: Specification complete, locked, and submitted for audit before any Gate 6 compute is spent. No training launched. The one concrete next action (Section 6, step 1) is a code refactor of already-audited diagnostic scripts, not a new experiment — everything after it awaits explicit authorization.
