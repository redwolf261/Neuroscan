# E180 Stages 10-11 — T4_spectral dose-response and Γ-Δ coupling: pre-registration

**Date**: 2026-09-17
**Status**: PRE-REGISTERED. Not run.
**Depends on**: Stage 6-9's complete result (`E180_6_primary_test.json`,
`E180_8_incremental_r2.json`, `E180_9_boundary_control.json`).

---

## Stage 6-9 outcome (binding context, not re-litigated)

| Family | Stage 6 (full 125) | Stage 8 (Γ over uncertainty) | Stage 9 (boundary) | Decision |
|---|---|---|---|---|
| T1_rank | β=+0.58, p=0.112 | ΔR²=0.0011, $p < 1/N_{\text{perm}}$ | survives | **KILLED at primary gate** |
| **T4_spectral** | β=+0.75, p<0.0001 | ΔR²=0.0043, $p < 1/N_{\text{perm}}$ | survives | **ADVANCE** |
| T5_smooth | β=+0.45, p=0.124 (full); p<0.0001 (110) | ΔR²=0.0095, $p<1/N_{\text{perm}}$ | survives | **HOLD / mixed, not advanced** |

**Correction to prior reporting, carried forward**: every Stage 8/9 permutation p-value reported
as "0.0000" is **below the empirical resolution of $N_{\text{perm}}=1000$** — i.e.
$p < 1/1000 = 0.001$, not literally zero. All future reporting states this explicitly
(`p < 0.001` or `p < 1/N_perm`), never `p=0.0000`.

**Effect-size framing, carried forward**: T4_spectral's Stage 8 ΔR²=0.0043 is small in absolute
terms even though statistically robust. Stage 10-11 exists to test whether T4_spectral's
signal has **structure** (dose-response, coupling) — that is the basis for interest, not the
magnitude of the Stage 8 coefficient.

**Only T4_spectral proceeds.** T1_rank is reported killed. T5_smooth is reported held/mixed. No
further stage is run on them without a separate, explicit decision.

## Locked, not re-tuned

Per explicit instruction: transform family (T4_spectral, $\Sigma \to \Sigma^\gamma$), probe
severity/eps, subject set (full 125 + E167 110 strata), and tile-selection criterion
(reduced-granularity: max enc3-center-distance pair) are **not** changed based on the Stage 6-9
result. T4 earned this experiment under the already-locked setup; Stage 10-11 runs on the
identical tile ledger, identical $T_s$ (γ=3.0), identical probe (ε=8.0).

---

## The question Stage 10-11 answers

Not "does Γ predict Δ again" — that is Stage 6-9, already answered for T4_spectral. Stage 10-11
asks whether there is a **dose-response structure**, moving from a predictive correlate toward a
mechanistic chain:

$$
\boxed{\text{spectral restoration} \rightarrow \text{systematic reduction in } \Gamma \rightarrow
\text{systematic recovery of task performance}}
$$

## Design

For each tile $i$ in the tile ledger (same tiles Stage 3-5 already used), construct graded
restoration doses from the same fixed degraded state $Z^{deg}$ used throughout:

$$
Z_i^{(k)} = Z_i^{deg} + \alpha_k \left(Z_i^{intact} - Z_i^{deg}\right), \quad
\alpha_k \in \{0, 0.25, 0.50, 0.75, 1.0\}
$$

$\alpha_0=0$ reproduces Stage 5's $Z^{deg}$ (no restoration, the existing $\Delta_i^{(0)}=0$
baseline); $\alpha_4=1.0$ reproduces Stage 5's full restoration $\Delta_i$ already measured.
The three intermediate points ($\alpha \in \{0.25, 0.50, 0.75\}$) are new.

### Three measured quantities per tile, per dose

1. **Δ dose-response**: $\Delta_i^{(k)} = Dice(D(Z_i^{(k)}), Y) - Dice(D(Z^{deg}), Y)$ — extends
   Stage 5's construction to intermediate restoration amounts. GT used only to score, as always.
2. **Γ response**: $\Gamma_i^{(k)}$ — the same fixed-probe instability measurement (Stage 3-4's
   construction) applied at each restoration dose $Z_i^{(k)}$, i.e. does the tile's own
   instability shrink as it is restored?
3. **Coupling**: the tile-level pairing of $\Delta\Gamma_i = \Gamma_i^{(k)} - \Gamma_i^{(0)}$
   against $\Delta Dice_i = \Delta_i^{(k)}$, tested directly — not Γ predicting Δ at a single
   point, but the *change* in one predicting the *change* in the other across doses.

## Pre-registered tests (all subject-aware, per the Stage 6-9 discipline — no pooled-tile
   statistics used as evidence)

### Test 1 — Monotonicity of Δ

For each subject, does $\Delta_i^{(k)}$ increase (non-strictly) as $\alpha_k$ increases? Report
the fraction of tiles showing monotonic (or near-monotonic, allowing one non-monotonic step for
noise) increase, plus a subject-aware trend test (Page's L test or a subject-FE regression of
$\Delta$ on $\alpha$, cluster-robust).

### Test 2 — Monotonicity of Γ (the response side)

Does $\Gamma_i^{(k)}$ decrease (non-strictly) as $\alpha_k$ increases? Same reporting convention
as Test 1. This tests whether restoration actually *does what it is supposed to do* to the
instability signal itself — not assumed, measured.

### Test 3 — Coupling (the decisive test)

Subject fixed-effects regression of $\Delta Dice_i^{(k)}$ on $\Delta\Gamma_i^{(k)}$ (pooling
across the 4 non-baseline dose steps, tile and dose both nested within subject — cluster by
subject as before), with a within-subject-and-within-dose-step permutation null (shuffle
$\Delta\Gamma$ within subject×dose-step cells, preserving both marginals).

## Pre-registered decision rule (fixed now)

| Outcome | Criterion | Reading |
|---|---|---|
| **Strongest** | Test 1 AND Test 2 both monotonic (subject-aware trend significant, $p<0.05$) AND Test 3 significant with the expected sign ($\Delta\Gamma$ decreasing predicts $\Delta Dice$ increasing) | Graded, dose-dependent coupling — evidence for a mechanistic chain, not merely a correlate |
| **Correlate only** | Test 3 significant but Test 1 or Test 2 is irregular/non-monotonic | Γ predicts Δ but restoring the representation does not behave in a graded way — useful correlate, not yet a mechanism |
| **Null** | Test 3 not significant | Dose-response does not hold; T4_spectral remains a single-point predictor (Stage 6-9's result stands, nothing added) |

No result here overrides or reopens Stage 6-9's verdict for T1_rank or T5_smooth. No result here
is used to retroactively rescue a weak Stage 8 effect size — the effect-size caveat from Stage
8 stands regardless of what Stage 10-11 shows.

## What Stage 10-11 does NOT do

- Does not change T4's transform, severity, probe, or subject set.
- Does not reopen T1_rank or T5_smooth.
- Does not propose or build an algorithm (Stage 12+ remains out of scope, gated behind this
  result and, per the separately-run novelty audit, behind showing differentiation from the
  entropy/uncertainty-based TTA-selection literature specifically).

## Cost

5 restoration doses × 868 tiles × 1 sliding-window pass each for Δ, plus the same for Γ (dose 0
and dose 1.0 are already computed, so 3 new dose levels × 2 measurements × 868 tiles). Comparable
GPU cost to one more Δ-scale run (~2-4 hours) restricted to T4_spectral only (not all 3
families), since T1/T5 are not part of this stage.

## Output

- `experiments/exp_e12_eggo_m/e180/E180_10_dose_response.json`
- `experiments/exp_e12_eggo_m/e180/E180_11_coupling_test.json`
