# Phase E51 — CCABA + Attention Gate Combined (CCAG): Result (3-Seed)

## Context

Following E49 (CCABA, 3-seed mean +0.31pp, tied with D4-only) and E50 (IECG, 3-seed mean +0.02pp, worse than CCABA), five single-mechanism attempts since the strategic pivot had all failed to clear D4-only with statistical confidence, with the more structurally ambitious mechanism (IECG) performing worse than the simpler one (CCABA). Per the user's explicit direction, this was designated the project's **last single-architecture-family attempt**: combine two mechanisms already independently verified as real and non-conflicting — E49's CCABA (bottleneck amplification, causally calibrated to E48's real ablation data) and E46's attention gate (enc1 skip routing) — rather than invent a sixth new mechanism.

## Architecture

`neuroscan_3d_v8.py`, `UNet3D_v8`, extends v3 directly. Composes CCABA's bottleneck amplification with the attention gate, with the gate's conditioning signal (`g`) computed from `bottleneck_amp` (the CCABA-amplified bottleneck) rather than the raw bottleneck — a deliberate, disclosed composition-order choice (see file docstring for rationale).

**Verified before training** (`test_v8_ablation_safety.py`):
1. Full ablation-safety (alpha=0 AND gate forced to identity) reproduces v3's `probs`/`aux_probs3`/`aux_probs2` bit-for-bit (max abs diff 0.0).
2. CCABA-only ablation-safety (gate forced to identity) reproduces v6's own `probs` bit-for-bit — confirms composition didn't alter CCABA's own math.
3. Gate-only ablation-safety (alpha=0) reproduces v5's own `probs`/`attention_map` bit-for-bit.
4. Param overhead exactly additive: 4,883 = 4,625 (v5's own) + 258 (v6's own).

A 2-epoch smoke test (seed 0) was run and inspected before the full commitment: val Dice 0.40 → 0.76, `ccaba_alpha` stable near its 0.1 init, psi non-degenerate (std 0.15, full [0,1] range) under real training — healthy, comparable to prior smoke tests.

## Training

Full 30-epoch runs, 3 seeds (0, 1, 2) from the start per the mandatory post-E49 policy, identical protocol to every prior condition (AdamW, CosineAnnealingLR, mu=0.1, `lambda_ds3=0.9927` and `lambda_frac=0.0203` reused unchanged from CCABA — no new loss term was introduced by adding the gate, since `attention_map` is diagnostic-only in both E46 and this combined model). All 3 seeds trained cleanly, no instability, no kill condition triggered.

## Result

| Seed | best_val_dice |
|---|---|
| 0 | 0.9101 |
| 1 | 0.9092 |
| 2 | 0.9092 |
| **Mean** | **0.9095** |
| Std (n=3) | 0.0005 (≈0.05pp) |
| 95% CI | [0.9082, 0.9108] |

## Verdict vs. pre-declared criteria

- **Required**: mean ≥1.0pp over canonical baseline (≥0.9163) with a 95% CI excluding D4-only (0.9096). **NOT MET, not close.**
- Mean vs. canonical baseline (0.9063): **+0.32pp**.
- Mean vs. D4-only (0.9096): **−0.01pp** — a statistical dead tie (CI [0.9082, 0.9108] contains 0.9096 comfortably).
- Mean vs. CCABA's own 3-seed mean (0.9094, E49): **+0.01pp**, paired t-test (n=3, seed-matched) t=0.115, p=0.919 — indistinguishable from CCABA alone.
- Notably, CCAG's between-seed std (0.05pp) is much tighter than CCABA's own (0.19pp) or IECG's (0.30pp) — the combined model happens to be a lower-variance estimate this time, but of essentially the same central value.

## Honest interpretation

**The two mechanisms did not compose additively, or even improve on either mechanism alone.** CCAG's mean (0.9095) is statistically indistinguishable from CCABA's own mean (0.9094, E49) and from D4-only (0.9096). The attention gate contributed nothing measurable on top of CCABA, which is consistent with — not contradictory to — two prior findings in this project's own causal-diagnostic chain: E46 found the gate's own isolated effect was real but non-decisive (+0.39pp single-seed, later shown by the general seed-noise pattern to likely be smaller), and E47's causal audit found the gate's effect is diffuse across the volume, not concentrated at the boundary where it would plausibly interact with CCABA's own size-conditioned amplification. A gate whose own causal effect was already shown to be non-localized has no obvious reason to synergize with a size-conditioned bottleneck boost, and the data now confirms it doesn't.

This is the **sixth mechanism attempted** since the strategic pivot (E44 killed, E45 +0.47pp single-seed, E46 +0.39pp single-seed, E49/CCABA +0.31pp 3-seed mean, E50/IECG +0.02pp 3-seed mean, E51/CCAG +0.32pp 3-seed mean) and the first genuine multi-mechanism combination — it did not break the pattern established by the prior five.

## Decision (per the project's own pre-declared stop rule)

Per the user's explicit direction going into this run: if CCAG landed in the same band as E44–E50 (or worse), this closes out single-architecture-family mechanism-hunting on this exact 1,251-subject/FLAIR-only/64³ setup — no seventh mechanism should be attempted. That condition is met. **This is the final single-architecture-family attempt for this project.**

## Recommendation

The project's defensible contribution for a paper should now center on:
1. The causal-diagnostic methodology itself (E43→E47→E48): two genuine causal nulls followed by a real, statistically strong, initially counterintuitive finding (small lesions depend *more* on the bottleneck, not less; ρ=−0.454, p<0.001, n=125) — using real interventions on a trained model, not correlational proxies, which is rare practice in this literature per the 2025–2026 scans conducted for E49/E50.
2. The multi-seed variance-correction finding (E49→E50→E51): three separate, honestly-reported instances of a single-seed headline result overstating a real effect once measured properly, converging on the same conclusion — single-seed Dice claims in this line of work carry an unmeasured ±0.2–0.5pp noise band that has likely inflated prior published claims across the field, not just this project's own early practice.

Both are legitimate, disclosable, rare-for-the-field contributions independent of whether any single architecture crosses the project's own self-imposed +1.0pp bar on this exact dataset/setup.

## Artifacts on disk

- `neuroscan_3d_v8.py` (git-tracked)
- `experiments/exp_e12_eggo_m/e51/test_v8_ablation_safety.py` (git-tracked-eligible, verification script)
- `experiments/exp_e12_eggo_m/e51/train_e51_ccaba_attn_combined.py` (disk only, per `experiments/` gitignore convention)
- `experiments/exp_e12_eggo_m/e51/e51_seed{0,1,2}_run_log.txt` (disk only)
- `experiments/exp_e12_eggo_m/e51/runs/CCAG_seed{0,1,2}/epoch_metrics.csv` (disk only, 30 rows each, includes psi diagnostics)
- `experiments/exp_e12_eggo_m/e51/runs/CCAG_seed{0,1,2}/checkpoints/` (disk only, gitignored)
