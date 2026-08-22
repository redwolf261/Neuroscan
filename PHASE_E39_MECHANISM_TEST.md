# E39C — Mechanism Test: Does Residual Information Predict the Controlled Dice Response?

**Verdict: KILL.** The residual-information quantities (`P4`, `M4`) do not track the controlled experiment's actual outcome. They track `λ`'s own magnitude — a much simpler and less interesting explanation — while the real outcome (Dice) does something `λ` and the residual quantities do not: it peaks at a moderate value and declines on both sides.

No new training was performed. All results below are computed from the six already-trained models from E39A/B (λ = 0, 0.125, 0.25, 0.5, 1.0, 2.0).

---

## 0. Restating the standing verdict from E39A/B

Independent of anything in this document: **none of the six trained conditions reached the project's required +1 percentage-point improvement.** The best result (λ=0.25) reached +0.28 percentage points. This is not reopened or reconsidered here — E39C exists only to ask whether the underlying mechanism is worth pursuing further, not whether E39A/B secretly succeeded.

## 1. What was measured

For every one of the six conditions, the exact same per-epoch gradient measurements collected during real training (not a post-hoc replay) were used: the main loss's gradient size, the D4 auxiliary loss's own gradient size, their angular agreement, and the resulting orthogonal-residual size, all already saved by E39A/B. From these, two summary numbers per condition were computed by summing (not averaging or integrating) across all 30 training epochs: `P4` (the residual sized relative to the main gradient) and `M4` (the same residual, squared, matching a "how much extra update energy did this actually contribute" reading).

## 2. The headline numbers

| Condition | λ | Best Dice | ΔDice (pp) | P4 | M4 |
|---|---:|---:|---:|---:|---:|
| A (baseline) | 0.0 | 0.9081 | — | 0.0 | 0.0 |
| λ=0.125 | 0.125 | 0.9086 | +0.06 | 4.6 | 0.9 |
| **λ=0.25 (best)** | 0.25 | **0.9109** | **+0.28** | 8.4 | 3.1 |
| λ=0.5 | 0.5 | 0.9103 | +0.23 | 18.7 | 15.0 |
| λ=1.0 | 1.0 | 0.9104 | +0.24 | 29.6 | 36.7 |
| λ=2.0 | 2.0 | 0.9086 | +0.06 | 58.4 | 153.1 |

## 3. Tests 1–3: correlation with ΔDice (n=6, descriptive only)

Spearman(P4, ΔDice) = +0.49 (p=0.33); Spearman(M4, ΔDice) is identical. Pearson gives essentially no relationship at all (r=+0.01 for P4, r=−0.23 for M4). With six data points, none of this is a powered statistical test, and it is reported here strictly as descriptive evidence, per the explicit instruction for this phase.

## 4. Test 5: leave-one-condition-out sensitivity

Every leave-one-out correlation stayed positive and in a similar range (+0.10 to +0.70), meaning the modest overall correlation is not being propped up by one unusual condition — but it also never got close to a value that would be convincing at any sample size. This stability is not a point in the mechanism's favor; it just means the (weak) result is not an artifact of one outlier point.

## 5. Test 6: is λ=0.25 mechanistically distinguished from λ=0.5/1.0?

No. λ=0.25 (the best-performing condition) has the **lowest** `P4` and `M4` of the three conditions that actually beat baseline by a comparable margin (λ=0.25, 0.5, and 1.0 all land within 0.23–0.28 percentage points of each other) — its residual-information numbers are 3–10× smaller than λ=0.5 and λ=1.0's, despite producing essentially the same (small) Dice improvement. Whatever separates λ=0.25 from its neighbors, it is not "more residual information."

## 6. Test 7: does the inverted-U in Dice show up in P4/M4?

**No, decisively.** Dice rises from baseline through λ=0.25, then falls back down through λ=2.0 — a real inverted-U, peaking in the middle of the tested range. `P4` and `M4` do neither: both climb **monotonically** with λ across the entire grid, reaching their own maximum at λ=2.0 — the condition with the *worst* Dice improvement of any λ tested above baseline. The two curves have fundamentally different shapes. This is the single clearest piece of evidence in this whole phase, and it does not support the mechanism as a predictor of the controlled outcome.

![figure placeholder — see figures/e39c_mechanism_test.png]

## 7. Phase-decomposed analysis (early / middle / late training)

A pre-declared, equal three-way split of the 30-epoch schedule (epochs 0–9, 10–19, 20–29) was computed for both `P4` and `M4`, specifically to test whether the useful signal is concentrated late in training, as E36's own original finding would suggest.

**The early, middle, and late-phase correlations with ΔDice are numerically identical (ρ=+0.49 in every phase).** This was checked directly rather than reported blindly, because three unrelated numbers coming out exactly equal is itself a signal worth investigating: the cause is that Spearman correlation only depends on rank order, and the six conditions' `P4`/`M4` values keep the **same relative ranking** (monotonically increasing with λ) in every single phase of training, not just in the full-trajectory total. There is no phase where the ranking changes, so there is no phase where the correlation with Dice's real, non-monotonic response could differ either. **The phase decomposition adds no new information here** — not because the analysis failed, but because the thing it was designed to detect (a phase-specific signal) genuinely is not present: `λ`'s effect on `P4`/`M4` is uniform across the whole training run, not concentrated late.

## 8. The denominator confound check

This was treated as a real risk, per the explicit prior instruction not to trust `M4` if it turns out to be driven by `|g0|` collapsing rather than genuine residual growth.

Checked directly: across the six conditions' late-training averages, `|g4⊥|` (the residual's own size) grows **monotonically and by close to two orders of magnitude** as λ increases (0 → 0.81), while `|g0|` (the main gradient's size) is **noisy and non-monotonic**, varying only about 3-fold and without a consistent direction (0.90 → 0.51 → 0.75 → 0.33 → 0.44 → 0.39). `M4`'s cross-condition variation correlates almost perfectly with `|g4⊥|` (rank correlation = +1.00) and only moderately, and in the opposite-of-confound direction expected, with `|g0|` (rank correlation = −0.77, not statistically clean at n=6). **This specific confound — `M4` looking large purely because the denominator has shrunk — is not what is happening here.** The dominant driver of `M4`'s growth is a real, substantial, monotonic increase in the auxiliary gradient's own raw size as λ increases — which is expected and unremarkable (a bigger λ multiplies the D4 loss and therefore its gradient), not evidence of anything mechanistically special.

## 9. Verdict

Per the three-way decision structure specified for this phase:

**KILL.** Not because of the denominator confound (checked and ruled out as the dominant explanation) — but because the residual-information quantities `P4` and `M4`, across this controlled sweep, simply track `λ`'s own magnitude, monotonically, in every phase of training, while the actual outcome that matters (Dice) does something structurally different: it rises, peaks at a moderate weight, and falls. A quantity that increases without limit as λ increases cannot, by construction, explain an outcome that peaks and then declines. The phase-resolved analysis, run specifically to give E36's "late-training persistence" story its best chance, found no phase where this changed. The weak, single-digit descriptive correlation reported in Section 3 is best read as an artifact of both quantities happening to be positive for every λ>0 tested, not as evidence of a real, exploitable relationship between residual information and segmentation quality.

## 10. What this does not kill

E38's own finding — that, across the three *pre-existing*, architecturally-different conditions (D4-only, D2-only, Both), `P4`/`Q_r`'s ordering matched the real Dice ordering — is a separate result, on a separate comparison, and is not contradicted by this phase. What this phase specifically kills is the idea that *within one fixed architecture, sweeping only the auxiliary loss weight `λ`*, more residual information predicts a better outcome. Those are different questions, and only the second one was tested here.

## 11. Limitations

- n=6 throughout; every statistic in this document is descriptive, not a properly powered hypothesis test, exactly as instructed.
- Only a single seed was used for this controlled sweep (matching the scope explicitly agreed before training began) — whether this specific λ=0.25 peak, or the mismatch between P4/M4 and Dice, would replicate at other seeds was not tested here.
- The phase-decomposition finding (identical correlation in every phase) is a real, checked result, not an analysis bug — but it is also a consequence of a fairly small, monotonic-by-construction dataset (6 conditions whose residual size is mechanically driven by λ), and should not be over-read as a general claim that phase never matters for this kind of quantity.
