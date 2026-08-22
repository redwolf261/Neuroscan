# Phase E52 — Adaptive Size-Reweighting, Gradient-Calibrated Re-Attempt: Design

## Motivation

E34 (Adaptive Size-Reweighting, ASR) tested two component-level loss reweighting schemes — a native-GT-size-based weighting (`[S]`) and a coarse-resize-survival-based weighting (`[R]`, `w_c^alpha = 1 + kappa*(1-alpha_c)_+`) — and both **collapsed training catastrophically** (best Dice 0.7360 and 0.7108 respectively, vs. the 0.9063 baseline). See `PHASE_E34_ADAPTIVE_SIZE_REWEIGHTING_KILLED.md` (written retroactively during this project's documentation audit) for the full account.

**The critical detail**: E34's `lambda_cw` was calibrated by matching **loss value** at fresh initialization (`seg_loss ≈ cw_loss`), not gradient magnitude. E34 itself is the likely origin of this project's subsequent mandatory safeguard (first explicitly applied starting with E45's `lambda_d8`, and every deep-supervision-adjacent weight since) — but that safeguard was never applied retroactively to E34's own failed result. It is not known whether E34 failed because component-reweighting is a bad mechanism, or because its untested calibration produced a gradient blowup analogous to the ones later phases explicitly caught and rejected (E45: 2.18× rejected; E49: 25.4× rejected). This phase answers that question directly, cheaply, before treating E34 as a closed kill.

## Why this specific re-attempt, and why it's a genuinely different lever than E44–E51

Every mechanism tried since the strategic pivot (E44–E51) modified **architecture** — what information reaches the decoder and how. This is the first re-attempt at modifying the **objective** — what the model is punished for getting wrong — since E34 itself, and the only one grounded in this project's own causally-established mechanism (E43/E47/E48: coarse-resolution information loss, not boundary-routing, is the real locus of difficulty; small lesions depend disproportionately on exactly the pathway current architectures under-serve). The `[R]` (alpha_c) condition specifically upweights components whose geometry is most degraded by the 64³ resize — a direct, disclosed re-application of the DTC/`alpha_c` measurement from E29–E33, now aimed as a loss weighting rather than an architectural gate.

## Literature positioning (Gate D, before committing compute)

A 2025–2026 literature scan (see live conversation record) confirmed distance/boundary-weighted loss functions are a populated, active area (Generalized Surface Loss, Celaya et al. 2023/2025; Sub-Differentiable Hausdorff Loss, Dec. 2025; Weighted Normalized Boundary Loss; cbDice) — all tested on BraTS-family data, all weighting by **prediction-vs-GT surface distance at training resolution**. None found ties loss weighting to **resize-induced GT degradation** specifically (the `alpha_c` / coarse-resize-survival signal this project measured directly in E29–E31 and causally confirmed the underlying mechanism for in E48). This is the specific angle re-tested here, not a generic boundary loss — the same novelty argument E34 itself already made in its own design phase (E32/E33's novelty audit), unchanged by this recalibration.

## Method

**Reused unchanged from E34** (already verified, tested, no reason to redo): `component_weighted_loss.py` (the vectorized, differentiable, per-component weighted-Dice term — `test_component_weighted_loss.py` already confirmed its correctness against the training dataloader's own resize convention), `E34_weight_table_alpha_c.json` (the `[R]` condition's precomputed per-component weights, `w_c^alpha = 1 + kappa*(1-alpha_c)_+`, kappa=3, fixed a priori), `precompute_labeled_64_cache`.

**Only the `[R]` (alpha_c) condition is re-attempted** — this is the condition directly tied to the causal chain (E43→E47→E48) that motivates the entire post-pivot arc; `[S]` (raw native size) has a weaker connection to that specific mechanism and is not re-run here, keeping this phase scoped to one condition rather than re-litigating both.

**New**: `calibrate_lambda_cw_v2.py`, following E49's own `calibrate_lambda_frac.py` template exactly (measure `L_seg` and `L_cw` value AND gradient magnitude at fresh init, seed 0, `.train()` mode, real training batch — never `.eval()`, per E12e's own established lesson). Unlike `frac_hat`'s deliberately-light auxiliary role in E49/E50/E51 (target ratio 0.10), this term is meant to genuinely reshape which components the segmentation loss emphasizes — a comparable-weight auxiliary term, matching E45's `lambda_d8` treatment (target gradient ratio ≈1.0, not a light side-signal), since that is what E34's own original design intended (`L_total = L_seg + mu*L_boundary + lambda_cw*L_component_weighted`, no textual indication `L_component_weighted` was meant as a light auxiliary).

**Pre-declared kill condition**: identical to every prior phase — val Dice < 0.5 after epoch 5 triggers immediate stop.

**Pre-declared success criterion**: mean ≥1.0pp over canonical baseline (≥0.9163) across 3 seeds, 95% CI excluding D4-only (0.9096) — same bar as every post-E49 condition.

**Pre-declared diagnostic checkpoint before committing to the full 3-seed run**: a 2-epoch smoke test, inspected for health (no NaN/Inf, no early collapse pattern resembling E34's own `[R]` trajectory — val Dice reaching at least the 0.5–0.6 range other healthy runs reach by epoch 2, not E34-`[R]`'s own 0.057–0.069). If the smoke test reproduces E34's collapse pattern even under gradient calibration, this phase stops immediately at that point — no scaling to a full run chasing a result that's already failed its own cheapest check.

## What this phase would tell us either way

- **If gradient calibration fixes the collapse and produces a real, competitive Dice**: E34's original mechanism was sound, and its 2026-08 failure was a calibration artifact — directly consistent with, and further evidence for, this project's own gradient-calibration safeguard being the correct lesson to have drawn.
- **If gradient calibration fixes the collapse but the result still lands in the familiar +0.3–0.5pp band (or below)**: consistent with every post-pivot mechanism tried so far — closes the objective-level lever as cleanly as E44–E51 closed the architecture-level ones, with a stronger case that this project's ceiling is not calibration-artifact-driven.
- **If the collapse persists even under gradient calibration**: rules out calibration as the cause entirely, confirms E34's original kill was correct for a deeper reason (the per-component weighted-Dice formulation itself, not its lambda), and the smoke-test-first design keeps this cheap to discover.
