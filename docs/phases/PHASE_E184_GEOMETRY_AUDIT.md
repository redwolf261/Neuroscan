# E184 probe-response geometry audit (retrospective, no new perturbations)

**Date**: 2026-09-19
**Status**: COMPLETE. Continues after the
[commutator audit](PHASE_E186_DECODER_EQUIVALENCE_DIMENSIONALITY_PREREG.md#retrospective-commutator-audit-2026-09-19--see-separate-docmemory-for-full-detail)
froze $C_{\text{task}}$ as a real-but-modest correlate, not a mechanism. User's explicit
instruction: go one level deeper into E184 itself by characterizing the transformation of the
full 6-probe response *configuration*, not just its scalar diameter (Γ) — no new candidate
family, no new perturbations, no training.

## Design (locked before looking at any correlation)

Reused the same 170 E184 candidates already verified in the commutator audit (9/10 subjects,
`BraTS-GLI-00506-000` excluded for the same reconstruction-closeness reason established there —
identical exclusion, not re-litigated). For each candidate's 6-member T6/T7 probe family, built
the full response geometry before ($Z$) and after ($PZ$) perturbation:

$$
\mu_Z = \tfrac{1}{6}\sum_T D(T(Z)), \qquad G_{ij} = d(D(T_i Z), D(T_j Z)) \text{ (6×6 matrix)}
$$

with $d$ = `prediction_distance` (mean absolute probability difference) — the same continuous
metric used for `C_task`/`output_distance` throughout this branch, deliberately distinct from
Γ's own literal thresholded-dice definition (see bug note below).

**Five descriptors, fixed before any correlation was computed:**
1. Centroid displacement: $\Delta_\mu = d(\mu_{PZ}, \mu_Z)$
2. Frobenius norm of pairwise-matrix change: $\|G_{PZ} - G_Z\|_F$
3. **[Sanity check only, not a real descriptor]** max pairwise-distance change — this is
   $|\Gamma(PZ)-\Gamma(Z)| = |\Delta\Gamma|$ by construction (Γ *is* max pairwise distance across
   this same family), flagged as tautological before running and excluded from the real
   analysis; kept only to validate the pipeline (correlation with $|\Delta\Gamma|$ should be
   exactly 1.0).
4. Mean per-probe response displacement: $\overline{r_T}$, $r_T = d(D(T(PZ)), D(T(Z)))$
5. Std of per-probe response displacement: $\text{std}_T(r_T)$

## Bug caught and fixed before trusting any result

First run produced 0/170 valid records — not a null result, a real bug. Diagnosed: the script's
`pairwise_matrix(...).max()` (built from continuous `prediction_distance`) was being used as if
it were Γ for the reconstruction-closeness check against stored `gamma_Zprime` values, but Γ's
real, literal definition throughout E180-E187 is `1 - dice_agree` on **thresholded** (>0.5)
masks — a genuinely different, much smaller-magnitude quantity. Fixed by adding a separate
`gamma_of()` function matching the commutator audit's dice-based definition exactly, used only
for the closeness check and `delta_gamma`; the continuous-distance `pairwise_matrix`/`centroid`
remain the new diagnostic descriptors, never conflated with Γ itself. Verified the fix
reproduces the commutator audit's own values exactly for the same first candidate
(`gamma_Z=0.013658`, `gamma_PZ=0.012989`, `rel_err=7.65%`) before rerunning the full set.

## Result: sanity check clean, all four real descriptors show a substantially stronger signal than the commutator

**Sanity check**: descriptor 3 vs $|\Delta\Gamma|$: $\rho = 1.000000$, max absolute difference
0.0 — confirms the pipeline is wired correctly.

**Primary within-subject Spearman correlations** (170 candidates, 9 subjects):

| Descriptor | ρ vs $\lvert\Delta\Gamma\rvert$ | p | ρ vs signed $\Delta\Gamma$ | p |
|---|---|---|---|---|
| $\Delta_\mu$ (centroid shift) | 0.486 | <0.0001 | −0.336 | <0.0001 |
| $\|\Delta G\|_F$ | 0.379 | <0.0001 | −0.345 | <0.0001 |
| $\overline{r_T}$ (mean displacement) | 0.505 | <0.0001 | −0.345 | <0.0001 |
| $\text{std}_T(r_T)$ | 0.300 | 0.0001 | −0.262 | 0.0006 |

Substantially stronger than the commutator audit's $\rho\approx0.20$ on the same candidates.

**Controls, all four descriptors:**
- *Output-displacement partial*: association survives, and strengthens for all four
  (partial $\rho$ 0.36–0.41 vs raw 0.30–0.51) — but note $\Delta_\mu$ and $\overline{r_T}$ **do**
  correlate directly with output distance (ρ=0.54–0.58, p<0.0001), unlike the commutator's
  near-zero direct correlation. Reported plainly, not smoothed over: these two descriptors are
  less independent of output magnitude than $C_{\text{task}}$ was, even though the *partial*
  association with $\Delta\Gamma$ survives cleanly.
- *Perturbation-magnitude partial*: same pattern — partial associations survive and strengthen
  (0.30–0.49), with $\Delta_\mu$/$\overline{r_T}$ again showing moderate direct correlation with
  `perturb_norm` (ρ=0.41–0.42) that `frob_delta_G`/`std_rT` mostly don't (ρ=0.13–0.18).
- *Subject-preserving permutation null* (2000 draws, all four descriptors): real ρ sits
  4–6 std outside the null distribution for every descriptor, permutation p ≤ 0.001 in all
  cases.

## What the geometry actually looks like — not a clean A/B/C/D separation

The four descriptors are **highly inter-correlated with each other** (Spearman ρ=0.71–0.98,
`delta_mu` vs `mean_rT` almost redundant at 0.981) — they are not independent geometric
signatures but largely different views of one dominant mode of variation. A joint linear model
using all four (within-subject, standardized) explains $R^2=0.42$ of $|\Delta\Gamma|$'s
within-subject variance — moderate, not close to complete.

Checked the ratio $\|\Delta G\|_F / \Delta_\mu$ as a rough translation-vs-expansion indicator:
median 5.0×, range 1.1×–14.6×, consistently well above zero. This rules out **Case C** (pure
centroid translation, everything moving together with unchanged relative geometry) as the
dominant pattern — there is real relative-geometry reshaping (expansion/contraction/reordering
among the 15 pairwise distances), not just shared drift, consistent with why Γ actually moves in
these candidates. Distinguishing cleanly between Case A (one-probe excursion), Case B (global
expansion), and Case D (reordering) was not attempted beyond this ratio check — the four
descriptors' strong mutual correlation means a sharper decomposition would need genuinely new
descriptors, which was explicitly out of scope for this locked, five-descriptor design.

## Reading (not decided here — reported for the user's own synthesis)

The probe-response geometry — specifically overall displacement/reshaping of the 6-probe
response configuration — is a stronger correlate of $\Delta\Gamma$ than the task-space
commutator was on the identical candidate set, survives the same controls more robustly, and is
not explained by output displacement or perturbation magnitude alone (though two of the four
descriptors are more entangled with those than the commutator was). The geometry is not a clean
single-mechanism story (four descriptors collapse toward one dominant mode, $R^2=0.42$ jointly)
but is a materially stronger diagnostic signal than anything found so far in this line of
investigation. No E187-style negative-control check was run in this stage — that would be a
natural next step if pursued, not decided here.
