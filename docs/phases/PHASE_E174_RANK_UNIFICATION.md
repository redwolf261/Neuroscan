# E174 — Rank unification and measurement-robustness audit

**Date**: 2026-09-16
**Status**: Complete. CPU-side only (one re-analysis of stored curves); no training, no new inference.
**Purpose**: make the $R_{\mathrm{eff}} \rightarrow R^* \rightarrow$ deficit chain internally
consistent before any write-up, per the corrected sequence.

---

## Step 1 — E169b FROZEN

Copied to `e169/FROZEN/` with SHA-256 checksums and a manifest, before any re-analysis.
These are the existing result and are **not** to be regenerated or optimised:

| stage | A (raw) | D (both size controls) |
|---|---:|---:|
| enc1 | 0.6341 | 0.3929 |
| enc2 | 0.7341 | 0.4514 |
| **enc3** | **0.7893** | **0.5178** |
| bottleneck | 0.3880 | 0.1117 |
| **dec1** | 0.3568 | **0.0042** |

---

## Step 2 — the definitions, traced from code (not assumed)

| Property | E165 (`run_e165_per_stage_rank.py`) | E169 (`run_e169_rstar_predictability.py`) | E170 (`train_e170_tdm.py`) |
|---|---|---|---|
| Representation | stage output, hooked | stage output, hooked | enc1 output, hooked |
| Operation | **SVD truncation**, mean-centred over voxels | **entropy** $\exp(H(p))$ of the spectrum | entropy $\exp(H(p))$ |
| Quantity | $R^*$ (a *requirement*) | $R_{\mathrm{eff}}$ (an *intrinsic property*) | $R_{\mathrm{eff}}$ |
| Voxel scope | all voxels in the tile | **20,000**-voxel subsample | **8,192**-voxel subsample |
| Spatial scope | **per 128³ tile** (hook fires inside the sliding window) | per tile, averaged over tiles | per training patch |
| Criterion | first dyadic $r$ with agreement ≥ 0.90 | n/a (continuous) | n/a |
| Comparison | vs the subject's **own intact prediction** | n/a | n/a |
| Grid | $\{1,2,4,8,16,32,64,128,256\}$ | continuous | continuous |
| GT used? | **No** | No | No |
| Prediction used? | Yes (as the reference) | No | No |

### Correction to a claim I made in the E171 audit

`PHASE_E171_AUDIT_AND_BLOCKERS.md` states E165's $R^*$ is measured on the "full native volume."
**That is wrong about the SVD scope.** E165's truncation is a forward hook firing inside
`sliding_window()`, so the SVD runs **per 128³ tile**, exactly like E169 and E170. Only the
*agreement Dice* is scored on the reassembled full volume.

**This makes the mismatch much narrower than reported.** All three operate per-tile. The real
residual differences are (a) truncation-vs-entropy and (b) subsample size 20,000 vs 8,192.

### The two quantities are NOT the same thing, and should not be conflated

$$R^*_i(l) = \min_R\Big\{R : D\big(F^{(R)}_l(x_i)\big) \approx D\big(F_l(x_i)\big)\Big\}$$

— a **requirement**, defined by an intervention against the subject's own intact prediction.

$$R_{\mathrm{eff}}(l) = \exp\big(H(p)\big),\quad p = \text{normalised spectrum}$$

— an **intrinsic property** of the undisturbed representation, no intervention, no reference.

They answer different questions and must stay separate. Their near-independence is not a bug:
E170's premise test measured ρ($R_{\mathrm{eff}}$, $R^*$) = **−0.09 at enc1, +0.03 at enc2** —
which is precisely why the deficit $[R^* - R_{\mathrm{eff}}]_+$ is not an algebraic artifact
there. **At enc3 they couple at ρ=+0.70**, so any enc3 deficit claim is partly circular and must
be reported separately.

---

## Step 3 — canonical specification (adopted)

**$R^*$ is the anchor**, because it carries the strongest guard already in place: the reference
is the subject's own undegraded prediction, never ground truth.

Canonical: SVD truncation of the stage activation, mean-centred over voxels, per 128³ tile;
reassemble; binary Dice over ET/TC/WT against the intact prediction; $R^*$ = first value on the
dyadic grid reaching agreement ≥ τ, with **τ = 0.90 as the default and τ reported as a
convention, not a constant of nature**.

$R_{\mathrm{eff}}$ is defined separately as above, and **the subsample size is unified to
20,000** (E169's value, the one the frozen predictor was fitted against). E170's 8,192 is
superseded; any future training code must use 20,000.

---

## Step 4 — measurement robustness audit

**Question**: does the E169b phenomenon survive a change of measurement convention?

τ = 0.95 / 0.99 are **not recoverable** from stored data: E165 breaks the sweep loop at the first
crossing of 0.90, so curves terminate there. τ = 0.80 and 0.85 *are* fully recoverable for all
125 subjects at all five stages. Re-ran E169b's exact protocol (both size controls, `pred.*`
removed from features, same seeded by-subject folds) at each:

| τ | enc1 | enc2 | **enc3** | bottleneck | **dec1** |
|---|---:|---:|---:|---:|---:|
| 0.80 | 0.3486 | 0.3667 | **0.5192** | 0.1759 | **0.0617** |
| 0.85 | 0.3926 | 0.3412 | **0.5019** | 0.0860 | **0.0594** |
| **0.90** | 0.3929 | 0.4514 | **0.5178** | 0.1117 | **0.0042** |

**Validity check**: the τ=0.90 row reproduces the frozen E169b values to 4 decimals
(0.3929 / 0.4514 / 0.5178 / 0.1117 / 0.0042), confirming the audit harness is faithful.

### Verdict

$$\boxed{\text{The phenomenon SURVIVES. enc3} \gg \text{dec1 holds at every recoverable } \tau.}$$

enc3 is stable at **0.50–0.52** across τ. dec1 stays **0.004–0.062**, an order of magnitude
lower. The enc3/dec1 ratio ranges 8× to 123× — the *magnitude* of the gap is
convention-sensitive, but its *existence and direction* are not.

**Honest caveats:**

- enc2 is the least stable stage (0.34 → 0.45 across τ); do not report enc2 as a precise value.
- dec1 at τ=0.80/0.85 is ~0.06, not ~0.004 — so "dec1 is zero" overstates it. The defensible
  statement is "an order of magnitude lower than enc3," not "absent."
- τ ≥ 0.95 remains **untested**. A re-run without the early `break` would cost one E165 pass
  (~80 min) and is the single remaining gap in this audit.

---

## Framing correction (adopted)

Not: *"the contribution is the characterisation."* That presumes the novelty conclusion.

Defensible framing:

> Existing work establishes related forms of sample-wise compression, rank estimation, and
> predictive complexity. These experiments investigate whether the **output-required
> representation dimensionality** exhibits stage-specific, instance-level structure in dense
> medical segmentation that is not explained by lesion extent.

And the finding:

$$\boxed{\text{At enc3, required-rank variation remains substantially predictable beyond two independent size proxies } (R^2 \approx 0.50\text{–}0.52\text{ across } \tau), \text{ whereas at dec1 it largely disappears } (0.004\text{–}0.062).}$$

Whether that constitutes a sufficient research contribution is a judgement to make after —
not during — the measurement work. E173's verdict (🟡 EXTEND) stands unchanged.

## Remaining prerequisite before any write-up

One E165 re-run with the early `break` removed, to test τ = 0.95 / 0.99. Until then the
robustness claim covers τ ∈ [0.80, 0.90] only, and should be stated with that range explicit.
