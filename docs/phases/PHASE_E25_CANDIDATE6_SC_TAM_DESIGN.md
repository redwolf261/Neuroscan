# Phase E25: Candidate 6 — Signed Class-Conditional Task-Aligned Margin (SC-TAM)

**Status**: 🟡 Design locked, three pre-implementation corrections applied — no code written, no training run. This document is the specification E26+ implements against.

**Date**: 2026-08-10

---

## 0. Provenance and three corrections applied before implementation

Candidate 6 (SC-TAM) is derived directly from E24's own failure-mechanism analysis, specifically its Q3 finding: B's tumor voxels moved correctly along `ŵ` (64.5%, p=0.007) but B's background voxels moved *incorrectly* — toward the tumor side of the decision axis (39.1% correct, significantly below chance, p=0.001). SC-TAM's core hypothesis is that this asymmetric miscalibration, not insufficient movement magnitude, is what prevented B's real geometric improvement (H4: p=0.030 vs A) from translating into Dice improvement (H3: null).

Three corrections were made to the original design before it is implemented, each resolved by direct verification against the actual codebase or literature, not by assumption:

1. **Same-class pairs are out of scope — and this is ALREADY true of the existing pair-sampling loop, not a departure requiring new code.** `d^SC_ij = s_i z_i^Tŵ + s_j z_j^Tŵ` is only mathematically coherent as a separation quantity for foreground-background pairs (where it reduces to a true difference); for a same-class pair, `s_i=s_j`, making it a *sum* of two same-signed terms, not a separation. **Confirmed with the design's own author that SC-TAM's hinge applies to FB pairs only — and separately verified by direct inspection and a standalone runtime check that `compute_margin_loss`'s existing loop (`for pos_local, opp_local in ((tumor_local, bg_local), (bg_local, tumor_local))`) already only ever forms cross-class pairs in both iterations** (`pos_local`/`opp_local` are always the opposite class's index pool by construction — confirmed with a synthetic index check before trusting this). This corrects an error in the original version of this document, which incorrectly claimed the existing loop was "symmetric same-class-inclusive" and would need a scoped departure to restrict it — **no pair-construction code needs to change for SC-TAM's FB-only requirement**; only the distance formula (Section 3) and the class-sign convention need to be added.
2. **`p_i` (predicted foreground probability, needed for error/uncertainty weighting) is confirmed available with no restructuring.** Verified directly against `train_eggo_m.py`: `probs = outputs["probs"]` (line 365) is computed from the same forward pass, before `compute_margin_loss` is called (line 404). `p_i` threads in as a new parameter, the same pattern `evidence_flat`/`boundary_logit_flat` already use — not assumed, checked.
3. **"Ante-hoc interpretability" framing (original Section 9) is dropped, not qualified.** Checked directly against the cited literature (ACM Computing Surveys' "Ante-Hoc Methods for Interpretable Deep Models"; an open-access arXiv self-interpretability survey): both define ante-hoc/self-interpretable methods as requiring the **model's forward pass and decision process itself** to be structurally transparent (prototype networks, concept bottlenecks, architecturally-exposed attention) — explicitly excluding "an auxiliary loss regularizing gradients of an otherwise opaque network." SC-TAM's real, defensible property — `∂L_SC-TAM/∂z_i ∥ ŵ`, a known, independently verifiable gradient direction — is genuine and worth stating, but is not the category the ante-hoc literature describes. **Replaced with a narrower, direct statement** (Section 9 below): SC-TAM's auxiliary term has a verifiable gradient direction, unlike a generic auxiliary loss whose effect on `z` would be opaque. No claim to the ante-hoc/self-interpretable category is made anywhere in this document. This follows the same discipline `PHASE_E23_ALGORITHMIC_REDESIGN.md`'s two-round novelty audit already established for this project — correcting an overclaimed literature category before it propagates, not after.

---

## 0.5. Novelty audit — verified vs. unverified prior-art comparisons

Per the same standard `PHASE_E23_ALGORITHMIC_REDESIGN.md`'s Sections 0.5/0.6 established (every load-bearing citation independently checked, not accepted on the strength of a title/abstract): three neighboring method families were checked before locking SC-TAM's novelty framing.

**Verified, real papers, mechanism NOT independently confirmable from available tools:**

- **FBA-Net** (Chung, Lim, Huang, Marrouche, Hamm; arXiv 2306.15189) — confirmed real (title/authors verified directly). Does general foreground/background contrastive learning for semi-supervised atrium segmentation. **Whether its contrastive coordinate is derived from the classifier's own weight vector, and whether its distance is signed or a standard symmetric contrastive distance, could NOT be verified** — the PDF's equations were not extractable with available tools. This is reported as an open `?`, not inferred from the abstract or assumed to follow the common pattern in the field.
- **De Brabandere et al.'s discriminative loss** (arXiv 1708.02551) — already verified real and directly relevant in this project's own prior work (`train_eggo_m.py`'s own margin loss is explicitly modeled on this paper, per `PHASE_E10_MARGIN_LOSS_SELECTION.md`). Confirmed: intra-class compactness + inter-class margin, in a **learned** embedding space, **not** derived from or aligned to a classifier's own weight direction, and **unsigned** (a symmetric pairwise distance). This comparison is trustworthy since this project's own codebase is built on a direct reading of the paper, not a secondhand description.

**Novelty table** (per the corrected framing — `?` means genuinely unverified, not "probably absent"):

| Method | FG/BG contrast | Signed class orientation | Classifier-derived axis | Explicit analytic gradient direction | Relation to SC-TAM |
| --- | --- | --- | --- | --- | --- |
| De Brabandere discriminative loss | Partial (inter-class margin, not FG/BG-specific) | ✗ (confirmed unsigned) | ✗ (confirmed, learned embedding space) | Not emphasized in the original paper | SC-TAM's parent formula (already the basis of this project's own baseline `L_margin`) |
| FBA-Net | ✓ | ? (unverified) | ? (unverified) | ? (unverified) | Mechanism details not independently confirmed from the accessible primary source |
| E24 condition B (this project's own unsigned task-aligned margin) | Partial | ✗ (the exact property SC-TAM changes) | ✓ (verified, this project's own code) | ✓ (verified, Gate 5's orthogonality check) | SC-TAM's direct predecessor |
| SC-TAM (proposed) | ✓ (FB pairs only, per Correction 1) | ✓ | ✓ (identical `ŵ` construction to E24) | ✓ (Section 3's derivation, to be verified by unit test per the same discipline as `test_margin_loss_w.py`) | — |

**What this does and does not establish.** This audit does not claim SC-TAM is the first method to combine these four properties — it establishes that at least one close neighbor (FBA-Net) could not be ruled out as already combining some of them, and says so plainly rather than asserting an unverified distinction. **The narrower, defensible novelty claim**, replacing any "first ever" framing:

> SC-TAM introduces a class-conditional signed margin regularizer whose separation coordinate is explicitly aligned with the segmentation classifier's own detached weight direction — a specific combination derived directly from a documented failure mode (E24's Q3 finding) in this project's own prior candidate, not asserted as absent from the wider literature by exhaustive search.

**Novel algorithm ≠ novel ingredients.** Even where individual ingredients (FG/BG contrast, classifier-derived directions, signed distances) may exist elsewhere in some combination, SC-TAM's specific mathematical construction and its direct derivation from a measured, documented failure mode (not a hypothesis invented independently of data) is the load-bearing claim — consistent with how `PHASE_E23_ALGORITHMIC_REDESIGN.md`'s own final novelty framing was resolved (Section 0.6 there: the diagnostic-to-objective methodology, not the mechanism in isolation, is the strongest claim). A full 15–25-method novelty matrix (as proposed) is not performed in this document — flagged as a required pre-publication step, not yet done, distinct from the pre-implementation corrections above.

---

## 1. Motivation, directly from E24

The existing task-aligned objective (E24's condition B) uses the unsigned projected distance:

```
d^w_ij = |(z_i - z_j)^T w_hat|
```

This is invariant to the *orientation* of separation along `w_hat` — moving a background voxel toward the tumor side and moving a tumor voxel toward the background side produce identical `d^w_ij`. E24's Q3 measurement showed this is not a hypothetical concern: it is what actually happened. SC-TAM's central change is to make the objective **direction-aware**, not just magnitude-aware, by explicitly encoding which side of `w_hat` each class should move toward.

---

## 2. Definitions

Let `z_i ∈ R^32` be voxel `i`'s `dec1` embedding (unchanged scope from E19–E24: `dec1` only). Let `y_i ∈ {0,1}` be ground truth (0=background, 1=tumor). Let `w_hat ∈ R^32`, `‖w_hat‖=1`, be `seg_head`'s own detached weight direction — **identical construction to E23/E24's `w_hat`** (same `seg_head[0].weight`, same normalization, same detach discipline verified in `PHASE_E24_CALIBRATION_AND_IMPLEMENTATION_SPEC.md` Section 2.3), not a new direction.

Class orientation: `s_i = 2y_i - 1` (`+1` for tumor, `-1` for background).

Signed task coordinate: `q_i = s_i * (z_i^T w_hat)`.

**Verified sign convention** (reused from E24's own direct check, not re-derived): pushing an embedding along `+w_hat` increases `seg_head`'s output probability (since `logit = w·z+b` and sigmoid is monotonic increasing) — confirmed by direct forward-pass test before E24's Q3 analysis was trusted. This is the same verified fact SC-TAM's "tumor should move toward `+w_hat`" claim depends on.

---

## 3. Loss (foreground-background pairs only, per Correction 1)

For a sampled foreground-background pair `(i,j)`, `y_i=1, y_j=0`:

```
d^SC_ij = (z_i - z_j)^T w_hat
L_SC = mean_{(i,j) in P_FB} [ m_ij - d^SC_ij ]_+^2
```

**Locked to the SQUARED hinge** (Section 11), matching the original `L_margin`'s `[2δ_d - d]_+^2` form exactly — this corrects the earlier draft, which used an unsquared/linear hinge. The hinge form is deliberately held identical to B's own formula so that C6-2 vs. B isolates the signed-vs-unsigned change alone, not a simultaneous hinge-form change.

Gradient, for an active pair (`m_ij - d^SC_ij > 0`): `∂L_SC/∂z_i = -2[m_ij-d^SC_ij]·w_hat`, `∂L_SC/∂z_j = +2[m_ij-d^SC_ij]·w_hat` — driving the tumor voxel toward `+w_hat` and the background voxel toward `-w_hat`, exactly the asymmetric-failure fix E24's Q3 motivates, with a magnitude that scales with hinge violation (matching the squared-hinge convention's own scaling behavior, verified by direct derivative below in the unit-test suite, not just asserted).

---

## 4. Error/uncertainty weighting (optional extension, per the user's own C6-2/C6-3 staging)

```
u_i = 1 - |2p_i - 1|              (uncertainty, p_i = probs at voxel i, SAME forward pass, verified available per Correction 2)
e_i = 1[ 1(p_i>=0.5) != y_i ]     (misclassification indicator)
omega_i = 1 + alpha*e_i + beta*u_i
omega_ij = (omega_i + omega_j) / 2, DETACHED from the optimization graph (matching U_hat/B's existing detach discipline in compute_margin_loss)

L_SC-TAM = sum(omega_ij * [m_ij - d^SC_ij]_+) / sum(omega_ij)
```

## 5. Boundary weighting (optional further extension, per the user's own C6-4 staging)

```
omega_i = 1 + alpha*e_i + beta*u_i + gamma*b_i    (b_i: normalized boundary-proximity score from a distance transform on the ground-truth mask)
```

---

## 6. Total training objective

```
L_total = L_seg + lambda_SC * L_SC-TAM
```

`L_seg` unchanged (`FocalTverskyLoss + EvidentialBetaLoss`, matching every prior phase's exact formula — not `L_Dice + L_FocalTversky` as the original draft stated; **this document corrects that to match the actual, verified `train_eggo_m.py` implementation**, not the draft's generic description).

---

## 7. Staged experimental structure (user's own, unchanged)

| Stage | Configuration | Purpose |
|---|---|---|
| C6-0 | Baseline Euclidean margin | = Gate 6 condition A, reused, not rerun |
| C6-1 | Task-aligned unsigned | = Gate 6 condition B, reused, not rerun |
| C6-2 | **Signed task-aligned (SC-TAM core)** | The actual algorithmic innovation — isolates the signed-vs-unsigned change alone |
| C6-3 | SC-TAM + error/uncertainty weighting | Tests whether error-targeting adds value beyond signing alone |
| C6-4 | SC-TAM + error + boundary weighting | Tests whether boundary-targeting adds further value |

C6-0 and C6-1 require **no new training** — Gate 6's existing A and B checkpoints/logs are the reference. Only C6-2, C6-3, C6-4 require new training runs.

---

## 8. Interpretation hierarchy (H1–H4, user's own, adapted from Gate 6's own H1/H2 structure)

- **H1 (directional constraint)**: `cos(∇_{z_i} L_SC-TAM, w_hat) ≈ ±1`, sign determined by class — a loss-level, implementation-verifiable claim, same discipline as Gate 5's orthogonality check.
- **H2 (realized representation movement)**: `cos(Δz_actual, w_hat)` should show the correct class-dependent sign and should exceed E24's random-projection control (E) — direct reuse of H2's own machinery (`run_h2_all_conditions.py`'s `ShadowAdam`/perturb-and-forward mechanism), extended to report separately for tumor and background voxels (not pooled, since the whole point is the class-conditional sign).
- **H3 (error correction)**: `E[|Δz| | e=1] / E[|Δz| | e=0] > 1` — direct reuse of the failure-analysis script's own Q5 measurement (`mean_dz_norm_misclassified` / `mean_dz_norm_correct`), already implemented and validated in `run_failure_analysis.py`.
- **H4 (segmentation improvement)**: `ΔDice ≥ 0.012` vs. the locked A baseline — evaluated **only after** H1–H3, per the same ordering discipline Gate 6 already established (mechanistic tests first, outcome last, no reinterpreting H1–H3 around a favorable or unfavorable H4 result).

---

## 9. What this design's interpretability property actually is (corrected, per Section 0 point 3)

SC-TAM's auxiliary term has a gradient direction that is known and independently verifiable (`∂L_SC-TAM/∂z_i = ∓w_hat` for an active pair, exactly, not approximately) — this is a real, checkable property, distinct from a generic auxiliary loss whose effect on `z` would need to be measured empirically rather than derived. **This is not "ante-hoc interpretability"** in the sense the interpretability literature uses that term (which requires the model's entire forward pass/decision process to be structurally transparent) — the rest of the network (encoder, most of the decoder, `seg_head`'s own internal computation) remains a standard opaque network. No claim to that literature's category is made.

---

## 10. Performance requirement (user's own, unchanged, restated as the hard gate)

Per the project's stated requirement: **`ΔDice ≥ +0.012` (1.2 percentage points) vs. the locked A baseline (best Dice 0.9063)**, i.e. the new method must reach **≥0.9183**, under:

- same dataset, split, architecture, training budget, optimizer, evaluation procedure, seed protocol
- ≥3 independent seeds (Gate 6 itself was single-seed; this is a real, not-yet-met requirement for a strong claim)
- subject-level paired statistical testing with confidence intervals
- comparison against C6-1 (unsigned) and a random-direction control, both required, not optional
- ablations: signed-orientation (C6-2 vs C6-1), error weighting (C6-3 vs C6-2), boundary weighting (C6-4 vs C6-3)
- HD95/ASD boundary metrics, not just Dice
- failure cases reported, not only the best configuration

**This document does not claim SC-TAM meets this requirement.** It is a design specification, not a result.

---

## 11. C6-2 locked parameters (resolved) and items still deferred to C6-3/C6-4

**Resolved for C6-2, confirmed with user, locked before implementation** — the explicit purpose is isolating the signed-orientation change as the ONLY thing that differs from B (E24's unsigned condition), so a positive or negative C6-2 result is attributable to signing alone, not confounded with simultaneous hinge-form/sampling/calibration changes:

1. **Hinge form: SQUARED**, `[m_ij - d^SC_ij]_+^2` — **not** the linear form Section 3's formula literally wrote. This corrects Section 3 below; the linear form was an unresolved open item, now closed in favor of matching the original `L_margin`'s squared hinge exactly, so hinge-form is not a second, uncontrolled variable alongside signing.
2. **Pairs: foreground-background only** — required by SC-TAM's own math (Correction 1, Section 0), not a new decision, restated here as locked.
3. **`m_ij` calibration: required, using the identical E12e/E24 methodology** (fresh-init model, `.train()` mode — never `.eval()`, same 10–30%-active-hinge-at-init target, same percentile-based derivation) — see Section 12 below for the actual calibration run.
4. **No error/uncertainty weighting** (`alpha=beta=0`, Section 4 not used) — deferred to C6-3.
5. **No boundary weighting** (`gamma=0`, Section 5 not used) — deferred to C6-4.
6. **Same λ, same training protocol as Gate 6** (seed 0, `mu=0.1`, `lambda=0.1`, 30 epochs, same `configs/brats.yaml`, same checkpoint schedule) — no protocol changes beyond the loss term itself.

**Deferred to C6-3/C6-4, not decided here:**

1. `alpha`, `beta`, `gamma` (error/uncertainty/boundary weight coefficients, Sections 4–5) — C6-3/C6-4's own ablation purpose is partly to determine whether these help at all before tuning them.
2. Boundary-proximity score `b_i` construction (which distance transform, what normalization) — needed only for C6-4.

**Decision rule, per the user's explicit instruction**: C6-2's own result determines what comes next. If C6-2 fails to fix the H1/H2 mechanism, that is evidence class-conditioned sign alone is insufficient, and C6-3/C6-4 become hypothesis-driven rather than arbitrary feature-stacking. If C6-2 fixes the mechanism (H1/H2/H3 pass) but misses the 1.2-point Dice bar (H4), that is a rational, evidence-based reason to investigate error/boundary weighting. **C6-3/C6-4 are not started until C6-2's own H1–H4 sequence is complete and reported.**

---

## Files

| File | Purpose |
|---|---|
| `PHASE_E24_GATE6_EXPERIMENTAL_SPECIFICATION.md` | The locked A/B/E protocol C6-0/C6-1 reuse without rerunning |
| `experiments/exp_e12_eggo_m/e24/run_h2_all_conditions.py` | H2 machinery this design's own H2 reuses (extended for class-conditional reporting) |
| `experiments/exp_e12_eggo_m/e24/run_failure_analysis.py` | H3's Q5 machinery this design's own H3 reuses directly, unchanged |
| `experiments/exp_e12_eggo_m/train_eggo_m.py` | `compute_margin_loss`, the function SC-TAM's implementation will extend (new FB-only pair construction, new signed distance, new optional weighting) |

---

**Status recap**: Design locked with three corrections applied (same-class-pair scoping, confirmed `p_i` availability, dropped ante-hoc claim). No code written. No training run. Four open items (hinge form, `m_ij` calibration, weighting coefficients, boundary score) remain, scoped to not block C6-2's implementation. Next step (not yet started, not authorized here): implement `compute_margin_loss`'s SC-TAM extension, calibrate `m_ij`, write the unit-test suite (same discipline as E24's `test_margin_loss_w.py`), then smoke-test before any training.
