# E182 — Asymmetric energy-control dissection: pre-registration

**Date**: 2026-09-18
**Status**: PRE-REGISTERED. Calibration not yet run.
**Depends on**: E181's frozen result (`PHASE_E181_MECHANISM_DISSECTION_PREREG.md`'s final
locked interpretation). T6's result stands; only T7's severity design is being fixed.

---

## Purpose

E181 gave H2 (spectral shape) a clean, rigorous test via T6: rank, energy, and $U,V$ held
exactly fixed, and the Γ→Δ relationship survived. T7 (energy-only) was meant to give H3
(energy/magnitude) the same rigor, but its two severities ($c=0.70$, $c=1.30$) turned out to be
**structurally equidistant from identity** ($|c-1|=0.3$ for both), so the magnitude-conditioning
variable Test C needed could not distinguish them — an experimental-design defect in T7's
severity choice, not a finding about H3 itself. **E182's sole purpose is to fix that one defect
and give H3 a fair test.** This is a surgical control experiment, not a new exploration branch.

## Locked, not touched

Model/checkpoint, the 125-subject cohort, tile extraction (E180's tile ledger), Γ's definition
(fixed probe, same-operating-point construction), Δ's definition (masked restoration, whole-
volume Dice vs GT), the subject-aware statistical framework (`e180_stats.py`), T6's transform and
its two severities (0.35, 0.50), all inclusion/exclusion rules (T7 identity excluded from
correlation cells, as before), and the pre-registered outcome-interpretation rules. **Only T7's
severity set changes.**

## The one design change

Replace the symmetric $c \in \{0.70, 1.30\}$ pair with a set whose members have **genuinely
different measured `uniform_magnitude`**, verified numerically before locking — not assumed from
the formula. Candidate: an asymmetric or wider one-sided set, e.g.
$c \in \{0.70, 0.85, 1.15, 1.30\}$, but the **exact values are determined by calibration, not
chosen in advance to produce a desired result**.

## Calibration gate (run BEFORE the full 125-subject collection, mirrors E181-B's discipline)

On the same 10-subject calibration sample used throughout this project:

1. Apply each candidate $c$ to enc3 activations.
2. Measure actual `uniform_magnitude` per candidate.
3. Verify magnitude is **monotonic and non-degenerate** across the candidate set — i.e. no two
   candidates give statistically indistinguishable magnitude (the exact failure mode that
   invalidated the original T7 design).
4. Verify invariants: rank unchanged (`eff_rank` before/after — T7's algebraic guarantee is
   exact in isolation; on GPU immediately after a model forward pass, a small, deterministic,
   severity-independent floor of ~1e-5 relative error was found and characterized — see the
   "rank-invariance threshold" note below), normalized spectral shape unchanged, $U,V$ unchanged,
   no NaN/Inf, no catastrophic prediction collapse, Dice degradation in a reasonable (non-
   catastrophic, non-trivial) range — same checklist as E181-B's T6/T7 calibration table.

### Rank-invariance threshold — investigated and corrected (recorded before the gate re-check)

The original `<1e-6` threshold assumed T7's algebraic rank-invariance guarantee (exact in
isolation, verified at 0.0 error on synthetic and detached tensors) would hold at that precision
inside the live pipeline too. On the first calibration pass it did not: `rel_rank_error_max`
measured 4.3e-5 to 7.7e-5 across the four candidates. Investigated before accepting or dismissing:

- **Traced to the same GPU SVD numerical-sensitivity phenomenon already characterized in
  E181-B** — SVD run immediately after a model forward pass (in an active CUDA/AMP context)
  gives a measurably different result than SVD on the identical tensor values captured and
  decomposed in isolation, for tensors with closely-spaced singular values. Reproduced directly:
  isolated/detached tensor → exactly 0.0 error; the same real post-forward-pass activation →
  8.58e-6 error, matching the calibration script's order of magnitude.
- **Confirmed deterministic and stable**, not GPU run-to-run noise: 5 repeated calls on the
  identical captured activation gave the identical error (8.583106e-06) to full float precision
  every time.
- **Confirmed non-systematic with severity**: `rel_rank_error_max` across the 4 candidates
  (7.68e-5, 5.69e-5, 7.15e-5, 4.34e-5) shows no monotonic relationship with $c$ or $|c-1|$,
  consistent with a fixed numerical floor rather than a severity-dependent effect of the
  transform itself.

**Conclusion**: this is the established GPU SVD precision floor for T7 (not previously
characterized at this severity range in E181-B, where T6's tested budgets happened to sit
cleaner), not a design flaw or a bug. **Threshold corrected to `<1e-4`** — comfortably above the
observed ~8e-5 ceiling, and still five orders of magnitude below any scientifically meaningful
rank change. This correction is made once, before re-checking the gate, and is not revisited
per-candidate to chase a more favorable number.
5. **Freeze the severity set** once these checks pass.
6. Only then run the full 125-subject collection.

### Binding rule — no peeking (stated explicitly, this is the discipline that matters most)

**Do not look at Γ→Δ relationships during calibration.** Calibration determines only whether the
intervention is technically valid (magnitude varies, invariants hold, outputs are sane) — it
must never become an optimization loop searching for statistical significance. If this rule is
violated even informally (e.g. noticing which candidate "looks more interesting"), the
calibration is compromised and must be redone with a rule that does not depend on any outcome
statistic.

## Outcomes (fixed now, read out mechanically after running)

1. **T6 survives, T7 fails under properly identified magnitude control** → strongest evidence
   yet that the phenomenon relates specifically to spectral organization rather than generic
   energy perturbation.
2. **T6 and T7 both survive** → spectral shape is demonstrably sufficient but not uniquely
   responsible; the deeper principle may be representation organization/susceptibility more
   broadly, not spectral shape specifically.
3. **T6 survives, T7 behaves inconsistently** (e.g. significant at some magnitudes, not others,
   without a clean pattern) → still useful; energy may influence the phenomenon under some
   perturbation regimes, but evidence stays asymmetric, not a simple H3 confirmation.
4. **T7 turns out stronger than expected** → resist defending H2 reflexively; report as found.
   The mechanism may be broader than spectral shape.

No outcome is treated as a failure requiring redesign — all four are informative and none
license inventing a fifth transform to rescue a preferred hypothesis.

## What comes after E182

**E182 is intended as the last major mechanism-disambiguation experiment before algorithm
synthesis** — explicitly to avoid an E169→E175-style cycle of indefinitely inventing new
diagnostics around the same phenomenon. If E182 establishes the T6 phenomenon is robust and
sufficiently specific, the next question shifts from "what explains the phenomenon?" to

$$
\boxed{\text{what computational operation should a segmentation algorithm perform because of
this phenomenon?}}
$$

— E183's scope (principle derivation), not a further round of statistical control-seeking.

## Output

- `experiments/exp_e12_eggo_m/e182/E182_calibration.json` (calibration gate result, magnitude/
  invariant table per candidate)
- Full-cohort collection and Test A/B/C analysis, contingent on calibration passing, follow the
  identical structure as E181-D and `run_e181_test_ab.py`/`run_e181_test_c.py`, reused not
  rewritten.

## E182 result and joint interpretation (recorded 2026-09-18)

**Raw Test A/B/C** (n=5208, 125 subjects, both families): T6 unchanged from E181 (β=+0.353
p=0.0053 at 0.35; β=+1.221 p=0.0400 at 0.50). T7's redesigned asymmetric severities individually
weaker than the old degenerate design (none reach p<0.05 alone: 0.18, 0.075, 0.89, 0.27), but the
pooled within-family test — now with a properly identified, non-degenerate magnitude covariate
(T7 magnitude-alone β=+0.032, p=3.5e-05, sane; contrast with E181's degenerate -385.6) — shows
**both T6 (ΔR²=0.096, perm p<0.001) and T7 (ΔR²=0.179, perm p<0.001) survive Γ-over-magnitude
control.**

### Joint interpretation (superseding any earlier framing of "spectral shape is the mechanism")

The E181 framing — chase whether T6 uniquely survives while T7 fails — is **retired**. With T7's
magnitude properly identified, it also survives. The correct reading is not "spectral shape is
special" but:

$$
\boxed{H_{\text{common}}: \text{the local response structure of the representation-to-
segmentation mapping contains information about recoverable task performance}}
$$

T6 and T7 are **two different probes of the same underlying structure**, not competing
explanations where one must win. This is a broader, more defensible hypothesis than the original
H2-vs-H3 framing, and is the correct terminal reading of E180-E182 as a unit:

- **E180**: Γ↑ ⟺ Δ↑ under controlled perturbation (T4_spectral survives every gate).
- **E181**: this is not adequately explained by rank alone — T6 holds rank, energy, and U,V
  exactly fixed and the relationship persists.
- **E182**: this is not uniquely a spectral-shape phenomenon either — T7, once properly
  magnitude-identified, shows the same qualitative survival.

**What E180-E182 jointly establish**: representation instability under controlled, invariant-
preserving perturbation — across at least two structurally distinct perturbation families —
predicts recoverable segmentation benefit, beyond subject effects, magnitude, uncertainty,
sensitivity, and (for T4_spectral specifically) boundary proximity and rank. This is the closed
scientific claim this phase of investigation supports. Associational throughout, not causal
proof of Γ→Δ (Δ's construction is the causal piece; the statistical relationship is not
independently randomized).

### What remains unexplained — the next question, not yet answered

$$
\boxed{\text{Does } \Gamma \text{ (a scalar diameter, } \max_{j,k} d(Y_j, Y_k) \text{) discard
information that a richer response-geometry object would retain?}}
$$

Γ retains only the largest pairwise separation among perturbed responses. It discards: **where**
the responses are located, **how many independent directions** they occupy, whether perturbations
produce **coherent** (aligned) or **scattered** motion, and whether **different perturbation
families** (T6 vs T7) produce similar or dissimilar response directions. These are the specific
candidate pieces of information a response-geometry audit would test — not an indiscriminate
list of every possible statistic, but the pieces Γ's own definition provably cannot see.

### Locked decision tree for the next step

$$
E180 \to E181/E182 \to \text{common response phenomenon} \to
\begin{cases}
\text{Does } \Gamma \text{ lose important geometric information?} \\
\quad \text{YES} \to \text{response geometry} \to \text{derive computational principle} \to \text{algorithm} \\
\quad \text{NO} \to \Gamma \text{ itself} \to \text{investigate other mechanisms}
\end{cases}
$$

**Next step, not yet started**: a small, data-reuse response-geometry audit on the already-frozen
E180/E181/E182 records — not a new experiment, not new GPU compute, not E183's full
implementation. Only if this audit shows response geometry contains genuinely incremental
information over Γ does "Counterfactual Representation Repair" (sketched in
`PHASE_E183_RESPONSE_GEOMETRY_SCOPING.md`) gain a stronger basis for pursuit.
