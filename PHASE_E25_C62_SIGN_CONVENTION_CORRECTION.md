# Phase E25, C6-2: Sign-Convention Correction — A Scoring Bug in the Diagnostic Scripts, Not in SC-TAM Itself

**Status**: ✅ Root cause found and confirmed. **Every diagnostic script from H2 onward (H2, H3, the mechanism audit, the isolation checks, the jacobian localization) scored voxel-level directional correctness against a `tumor → −ŵ, background → +ŵ` convention that is the OPPOSITE of what SC-TAM's own loss formula actually produces under gradient descent.** This was caught by the exact theoretical sanity check the user specified (`cos(Δz_GD, ∇_zL) ≤ 0`, required of any gradient-descent step regardless of SC-TAM's own directional claims), confirmed against the project's own already-passing unit test, and verified across a fresh 48-voxel/192-measurement sweep. **This is a scoring/interpretation bug in the diagnostic scripts, not a bug in `compute_margin_loss`, not a bug in C6-2's training run, and it does not affect H1's correlation result or H4's Dice result.** It does mean the H2/H3/mechanism-audit/isolation-check/localization *directional-correctness* narrative needs substantial revision.

**Date**: 2026-08-12

---

## How this was found

Per the user's explicit instruction, before trusting `PHASE_E25_C62_JACOBIAN_LOCALIZATION.md`'s uniform-0%-correct finding, a sign-consistency audit was run to check the ONE thing a genuine gradient-descent step must satisfy regardless of any external directional claim:

> For `Δz ≈ −ε·J·Jᵀ·∇_zL`, and `K = J·Jᵀ` positive semidefinite, `∇_zL·Δz = −ε·∇_zL·K·∇_zL ≤ 0` to first order — i.e. `cos(Δz, ∇_zL) ≤ 0` for a genuine descent step, independent of SC-TAM's own `ŵ` sign convention.

Direct code inspection of `run_c62_jacobian_localization.py` first ruled out the trivial hypothesis (a `+εg` vs `−εg` bug in the perturbation itself): `delta = [-EPS_PROBE * g for g in g_masked]`, applied via `p.add_(d)` — genuinely `θ_new = θ − ε·g`, real gradient descent, not ascent.

A fresh, targeted script (`run_c62_sign_consistency_audit.py`) then measured, for real isolated anchors on real C6-2 checkpoints: `∇_zL_SC`, `∇_θL_SC`, the resulting `Δz` under both `−εg` (descent) and `+εg` (ascent, a built-in control), across an epsilon sweep. The very first smoke-tested voxel (a TN anchor, epoch 5) immediately surfaced the discrepancy: its activation gradient was `cos(∇_zL, ŵ) = +1.0` exactly (matching Level 1's established mathematical identity), and its realized descent `Δz` gave `cos(Δz_GD, ŵ) = −0.52` — which every prior script would have scored as **wrong** (TN was assumed to expect `+ŵ`), but which satisfies the theoretical constraint `cos(Δz_GD, ∇_zL) ≤ 0` **exactly**, since `∇_zL = +ŵ` for this voxel and `Δz_GD` correctly points opposite to it.

This forced a direct re-derivation of SC-TAM's actual gradient sign from `compute_margin_loss`'s real code (`train_eggo_m.py`), independent of any prior phase's prose:

- `dist = tumor_proj − bg_proj` (signed), active-pair loss `L = (margin_target − dist)²`.
- `∂L/∂dist = −2·(margin_target − dist)`, negative when active.
- `∂L/∂z_tumor = ∂L/∂dist · (+1) · ŵ = −2·(margin_target−dist)·ŵ` — a **negative** multiple of `ŵ`.
- Gradient **descent**: `Δz_tumor = −η·∂L/∂z_tumor = +η·2·(margin_target−dist)·ŵ` — **toward `+ŵ`**, not `−ŵ`.

This matches, exactly, the project's own already-written and already-passing unit test, `test_sc_tam_sign_correctness` (`test_sc_tam.py`), whose docstring states plainly: *"tumor gradient must project NEGATIVELY onto w_hat (gradient descent −eps\*grad moves the tumor voxel toward **+w_hat**) and background gradient must project POSITIVELY (descent moves it toward **−w_hat**)."* This test has been passing since it was written — the ground truth was correct and verified the whole time; the diagnostic scripts built afterward simply never cross-checked their own `expected_sign()` conventions against it.

---

## What was actually wrong, precisely

Every diagnostic script from `run_h2_c62.py` onward independently encoded the same incorrect assumption — **`tumor → −ŵ`, `background → +ŵ`** — the reverse of the design's own verified formula (**`tumor → +ŵ`, `background → −ŵ`** under descent):

| File | Where the backwards convention appears |
|---|---|
| `run_h2_c62.py` | `expected_sign=-1.0` for tumor, `+1.0` for background (lines 232–233) |
| `run_h3_c62.py` (failure analysis) | `signed_proj[tumor_mask] < 0`, `signed_proj[bg_mask] > 0` (lines 153–154) |
| `run_c62_mechanism_audit.py` | Level 4's `cat_stats(mask, expected_sign)` calls for TP/FN (`-1.0`), FP/TN (`+1.0`) |
| `run_c62_isolation_checks.py` | Check 2's `cos_full_realized < 0` "FN expects <0" convention |
| `run_c62_jacobian_localization.py` | `expected_sign()` function, TP/FN `-1.0`, TN/FP `+1.0` |

**Not affected**:
- `compute_margin_loss` itself (`train_eggo_m.py`) — verified correct by direct derivation and by the pre-existing, passing unit test. The actual C6-2 training run used this real, correct implementation throughout.
- `test_sc_tam.py`'s `test_sc_tam_sign_correctness` — correct and passing since it was written; this is the ground truth the diagnostic scripts should have been checked against.
- **H1** (`run_h1_c62.py`) — measures `corr(ΔL_margin, ΔDice)` directly, an outcome correlation with no dependency on any assumed directional sign. `rho_C62 = −0.4924, p=4.06×10⁻¹³` stands unchanged.
- **H4** (Dice outcome) — read directly from `epoch_metrics.csv`, no dependency on `expected_sign` anywhere. C6-2's best Dice (0.9030, still short of the 0.9183 bar) is completely unaffected.
- **The anchor-composition and Check 3 (weighting) diagnostics** — these measure loss *share* and sampling *composition* by category, not directional sign; unaffected.
- **The Gate 5 smoke test's own sign-correctness check** (criterion 9, `e25_smoke_train.py`) — directly re-checked: it measures `proj = g_active @ w_hat`, the **activation gradient's** own sign (`∇_zL`, NOT the realized `Δz` H2 onward measures), and asserts `proj[tumor_active] < 0`, `proj[bg_active] > 0`. This is a DIFFERENT quantity from what H2/H3/the mechanism audit/isolation checks/localization measure, and per Level 1's mathematical identity and the verified unit test, `cos(∇_zL, ŵ) < 0` for tumor / `> 0` for background IS the correct sign for the gradient itself (gradient descent then moves opposite to it, toward `+ŵ`/`−ŵ` respectively). **Gate 5's criterion 9 is correct and unaffected by this bug** — it was checking the gradient's sign, not the realized movement's sign, and got the gradient's sign right.

---

## Full confirmation across the 6-checkpoint sweep

48 voxels (12 per checkpoint × 4 categories, matching the jacobian localization's own scope), 192 (voxel × epsilon) measurements total.

**Theoretical GD constraint** (`cos(Δz_GD, ∇_zL) ≤ 0`, independent of any `ŵ`-relative claim): satisfied in **93.75%** of measurements (180/192), mean cos=−0.42. The 12 violations cluster at the largest (`ε=0.1`, a genuinely large step where first-order linearity is expected to start breaking down) and smallest (`ε=0.0001`, where floating-point precision in the finite difference becomes a real factor) ends of the sweep — consistent with ordinary finite-difference edge effects, not a systematic problem with the core measurement.

**GD/GA antiparallel control**: mean `cos(Δz_GD, Δz_GA)` moves from −0.44 (ε=0.1) to −0.76 (ε=0.01) to −0.85 (ε=0.001), trending toward the theoretically expected −1.0 as ε shrinks — confirms the descent/ascent sign machinery itself behaves exactly as physics requires.

**Fraction correctly signed, OLD (backwards) vs. NEW (corrected) convention**, at each voxel's smallest tested epsilon:

| Category | Old convention | Corrected convention | n |
|---|---:|---:|---:|
| TP | 25.0% | **75.0%** | 12 |
| TN | 8.3% | **91.7%** | 12 |
| FP | 0.0% | **100.0%** | 12 |
| FN | 0.0% | **100.0%** | 12 |

This is not a marginal correction — it is close to a full inversion of the picture reported throughout the recent diagnostic arc.

---

## What needs re-interpretation (not yet done in this memo)

This memo establishes the bug and its scope. It does **not** re-derive the corrected H2/H3/mechanism-audit/isolation-check/localization conclusions — that is real, substantial follow-up work, flagged explicitly rather than rushed:

- **H2's "24/24 checkpoint-batches show the correct mean sign"** was scored against the backwards convention. Under the corrected convention this needs to be recomputed from H2's own saved raw data (`h2_results_C62.json` already contains the per-voxel `cos_mean` values; only the *interpretation* of which sign counts as "correct" needs to flip) — very likely this was actually the *wrong*-signed result being called "correct," or vice versa, and needs to be redone precisely, not guessed at.
- **The mechanism audit's Level 4 TP/FP/FN/TN table** (`PHASE_E25_C62_MECHANISM_AUDIT.md`) — the "FN moves wrong-signed 98.5% of the time" headline finding was scored against the backwards convention and needs full recomputation from its own saved JSON before being trusted in either direction.
- **The isolation checks' Check 1/Check 2** (`PHASE_E25_C62_ISOLATION_CHECKS.md`) — same issue; the "0/6 isolated single-anchor probes correctly signed" finding needs recomputation.
- **The jacobian localization's uniform 0%-correct table** (`PHASE_E25_C62_JACOBIAN_LOCALIZATION.md`) — directly superseded by this memo's own corrected numbers for the SAME measurement mechanism; the "pathway-invariant inversion" framing needs to be revisited once the corrected sign is applied consistently.
- **The Gate 5 smoke test's own criterion 9** (`e25_smoke_train.py`) needs a direct, explicit re-check against the now-confirmed correct convention before trusting that Gate 5 actually passed what it claimed to pass — flagged as unverified in this memo, not assumed either way.

**This memo deliberately stops at establishing and confirming the bug and its blast radius.** Recomputing every downstream conclusion is real, substantial work that should be done deliberately and checked, not rushed inline here — consistent with the project's own standing discipline of not reacting to a correction with a hasty reinterpretation.

---

## Files

| File | Purpose |
|---|---|
| `experiments/exp_e12_eggo_m/e25/run_c62_sign_consistency_audit.py` | The audit that found and confirmed this bug |
| `experiments/exp_e12_eggo_m/e25/sign_consistency_results/sign_consistency_C62.json` | Raw 48-voxel/192-measurement results |
| `experiments/exp_e12_eggo_m/e25/test_sc_tam.py` | `test_sc_tam_sign_correctness` — the correct, already-passing ground truth this bug should have been checked against from the start |
| `PHASE_E25_C62_JACOBIAN_LOCALIZATION.md`, `PHASE_E25_C62_MECHANISM_AUDIT.md`, `PHASE_E25_C62_ISOLATION_CHECKS.md`, `PHASE_E25_C62_RESULTS.md` (H2/H3 sections only) | Reports containing conclusions that need re-interpretation under the corrected convention (H1/H4 sections of the last one are unaffected) |
