# Phase E49 — CCABA Multi-Seed Variance Check

## Motivation

E49's single-seed result (0.9114, +0.51pp) was the best of four post-pivot algorithmic attempts (E44 killed, E45 +0.47pp, E46 +0.39pp, E49 +0.51pp), all landing in the same narrow band despite being mechanistically unrelated. Per `PHASE_E49_CCABA_RESULT.md`'s own recommendation, before attempting a fifth single mechanism, this ran the project's first-ever multi-seed variance check — every prior condition in this project's history (E1 through E49) has been evaluated on n=1 seed only.

## Method

Two additional seeds (1, 2) trained with CCABA's identical protocol (same architecture, hyperparameters, calibration constants, 30 epochs) — only the seed changed (`--seed` flag added to `train_e49_ccaba.py`, defaulting to 0 to preserve every prior script's behavior unchanged). Both ran clean, no instability, no kill condition triggered.

## Result

| Seed | best_val_dice | vs. canonical baseline (0.9063) | vs. D4-only (0.9096) |
|---|---|---|---|
| 0 | 0.9114 | +0.51pp | +0.18pp |
| 1 | 0.9075 | +0.12pp | −0.21pp |
| 2 | 0.9093 | +0.30pp | −0.03pp |

**Mean: 0.9094 (+0.31pp over baseline, −0.02pp vs. D4-only).**
**Std across seeds: 0.0019 (≈0.19pp).**
**95% CI on the mean (t-distribution, df=2): [0.9046, 0.9142], i.e. roughly −0.17pp to +0.79pp.**

## Interpretation

Seed 0's headline +0.51pp result was **favorable seed variance, not a stable effect**. The true CCABA effect, best estimated by the 3-seed mean, is **+0.31pp — smaller than reported, and statistically indistinguishable from D4-only** (mean difference −0.02pp; the 95% CI comfortably contains zero for that comparison too). The between-seed standard deviation (~0.19pp) is comparable in magnitude to the entire effect under investigation.

**This retroactively changes how every prior post-pivot result in this project should be read.** E44, E45, and E46 were each evaluated on a single seed. Given CCABA's measured seed-to-seed spread (0.9075 to 0.9114, a 0.39pp range from seed alone, on an otherwise identical run), none of E45's 0.9110, E46's 0.9102, or E49's own 0.9114 can be treated as precise point estimates — each carries an unmeasured ±0.2–0.4pp seed-noise band that was never previously accounted for. It is plausible (though not established, since only CCABA has been seed-checked) that some or all of the apparent "+0.3–0.5pp band" that four mechanistically distinct ideas landed in is itself largely a noise artifact of comparing single-seed runs, rather than four independently-real, independently-capped effects.

## Implication

1. **No mechanism tested so far (including CCABA) can be confidently claimed to beat D4-only, let alone the +1.0pp bar**, once seed variance is accounted for. The project's prior practice of reporting and comparing single-seed Dice numbers as if they were exact has likely been overstating precision throughout the post-pivot arc (E44–E49).
2. **A real ≥1pp claim, if one is to be made for a paper, requires multi-seed evaluation as standard practice going forward** — not just for a final headline result, but for every candidate comparison, given the demonstrated noise floor.
3. This does not mean CCABA (or any of E45/E46) is definitively "no better than baseline" — the CIs are wide, not centered at zero, and combining the causal-diagnosis grounding with a positive (if uncertain) mean effect is still the most defensible mechanism among the four tried. But claiming a specific, precise Dice number for any of them without multi-seed support is no longer honest practice for this project going forward.

## Recommendation for next steps

- Treat all future candidate evaluations as requiring ≥3 seeds before any GO/KILL decision or Dice-number claim, not as an optional follow-up.
- Given CCABA's mean effect (+0.31pp) is real but modest and D4-only-competitive, and combining mechanisms (CCABA + E46's attention gate, both independently verified and mechanistically non-conflicting) remains untested — that combination, evaluated with multi-seed discipline from the start, is a more promising next step than another single new mechanism.
- This variance result itself is a legitimate, disclosable methodological finding for the paper: it strengthens (not weakens) the project's overall rigor to have caught and corrected this before finalizing any headline claim.

## Artifacts on disk

- `experiments/exp_e12_eggo_m/e49/runs/CCABA_seed0/`, `CCABA_seed1/`, `CCABA_seed2/` (disk only, per `experiments/` gitignore convention)
- `experiments/exp_e12_eggo_m/e49/e49_seed1_run_log.txt`, `e49_seed2_run_log.txt` (disk only)
