# E188-A — Local latent recoverability map (pre-registration)

**Date**: 2026-09-20
**Status**: PRE-REGISTERED. Inference only. No training, no new network, no loss, no optimizer.

## The question

$$
\boxed{\text{Is E15 finding a structured region of better latent states, or merely a favorable
direction?}}
$$

Two sub-questions, both answered by the same run:

$$
R_i = \max_j D(z_i + \delta_j) - D(z_i) \;>\; 0 \text{ systematically?}
\qquad
D(z_i + \delta_{\text{E15}}) > D(z_i + \delta_{\text{random}}) \text{ systematically?}
$$

## Correction to the premise, carried explicitly

The decision tree motivating this experiment was anchored on "E15 = +1.40 pp". **That figure does
not hold at α=14 on the current checkpoint.** Verified numbers, `E172_phase1_summary.json`
(same checkpoint `E131_v5control_seed0`, same `dec1` site, 20 subjects, oracle_GT arm):

| α | mean ΔDice |
|---|---|
| 1 | +0.0039 |
| 2 | +0.0068 |
| 4 | +0.0106 |
| 8 | **+0.0139** ← peak |
| 14 | +0.0035 ← already declining |

The +1.4 pp magnitude is real but occurs at **α=8**, and the dose-response is **non-monotonic**
— it reverses before α=14 and collapses by α=28 (`E172_oracle_smoke.json`: 0.8796 → 0.8936 peak
→ 0.8832 → 0.8257). This differs qualitatively from the original E15 record (monotonic through
α=28 on the retired `E12f`/`UNet3D_v2` checkpoint). Experiment A is calibrated to the current
model's measured scale, not the historical one.

## Setup (frozen, matched to the E172 reference so numbers are comparable)

- Checkpoint `E131_v5control_seed0`, asserted by `best_mean_dice == 0.8929357248544694`.
- Intervention site: **`dec1`** (32ch, immediately before `seg_head`), per E15's own spec.
- **Single centered 128³ crop per subject**, *not* sliding-window — matching
  `e172/smoke_oracle_e15.py` exactly. Changing this would break comparability with the +0.0139
  anchor this experiment is calibrated against.
- 10 subjects (first 10 of the fixed validation split), same ordering as E172.
- Dice scored per region (ET/TC/WT) and meaned, same `dice3` convention as E172.

## Arms

**Native**: $D(z)$, no perturbation.

**E15 direction** — per-voxel, binary fg/bg, matching E172's reproduction verbatim:
$$
\text{fg} = \mathbb{1}[\textstyle\sum_r y_r > 0], \quad
\mu_{\text{opp}}(i) = \begin{cases}\mu_{\text{bg}} & i \in \text{fg}\\ \mu_{\text{fg}} & i \notin \text{fg}\end{cases}, \quad
\delta_{\text{E15}}(i) = \epsilon \cdot \frac{z_i - \mu_{\text{opp}}(i)}{\|z_i - \mu_{\text{opp}}(i)\|}
$$
GT enters **only** to select which centroid is opposite — the same restriction as the original
E15, never to pick a decoder-specific sign.

**Random directions**, $N = 32$ per (subject, ε) — **per-voxel unit random**, the strict null:
$$
\delta_j(i) = \epsilon \cdot \frac{g_{j,i}}{\|g_{j,i}\|}, \quad g_{j,i} \sim \mathcal{N}(0, I_{32})
$$
Identical per-voxel displacement magnitude and identical total tensor norm to $\delta_{\text{E15}}$.
**Only the direction differs** — this isolates "is E15's choice of direction special" from "does
moving every voxel by ε help at all".

**ε grid**: $\{4, 8, 12, 14\}$ — brackets the measured peak (8), includes 14 for direct
continuity with both the original E15 headline and the E172 reproduction.

Cost: 10 subjects × 4 ε × (32 random + 1 E15) + 10 native = 1330 forward passes.

## Recorded per (subject, ε)

native Dice · E15 Dice · random mean / median / max / min · **E15 percentile among the 32
randoms** · headroom $R_i = \max_j D(z+\delta_j) - D(z)$ · perturbation norm (verification that
E15 and random are genuinely matched) · output displacement $\|D(z+\delta) - D(z)\|$.

## Pre-registered outcomes (fixed before running)

| Outcome | Criterion | Verdict |
|---|---|---|
| **1 — E15 not special** | Random directions frequently match or beat E15 (E15 percentile near/below median systematically) | **KILL E15 as a structured latent phenomenon** |
| **2 — E15 beats random** | E15 above the random distribution systematically across subjects | E15 exploits latent geometry → proceed to Experiment B (what geometry) |
| **3 — headroom exists, E15 suboptimal** | $R_i > 0$ substantially, but E15 is not near the max | **Most interesting**: $z$ sits in a neighborhood with real recoverable headroom that E15 does not capture. Research problem becomes "identify the good region without GT" |

Outcomes 1 and 3 are both genuinely informative; 1 closes the branch, 3 redirects it.

**Robustness**: report the per-subject distribution, not a pooled mean — 10 subjects, so the
headline is the table and the sign-consistency across subjects, not a p-value on correlated
voxels. Consistent with this project's standing subject-level discipline (E180's subject-FE
correction, E179's ≥80% same-sign rule).

## Harness validation (run before the real pass, passed exactly)

Smoke test on subject `BraTS-GLI-01041-000` reproduces `E172_oracle_smoke.json`'s per-subject
numbers **to 4 decimals on every point**:

| α | E188-A harness | E172 reference |
|---|---|---|
| 0 (native) | 0.8774 | 0.8774 |
| 4 | 0.8775 (+0.0001) | 0.8775 (+0.0001) |
| 8 | 0.8669 (−0.0105) | 0.8669 (−0.0106) |
| 14 | 0.8463 (−0.0312) | 0.8463 (−0.0312) |

The harness is verified against the reference, not assumed.

**A substantive finding falls out of this check**: on *this* subject the E15 intervention is
**net harmful at every ε ≥ 8**, while the 20-subject oracle mean is positive (+0.0139 at α=8).
The E15 effect on the current checkpoint is therefore **heterogeneous across subjects**, not a
uniform improvement — a fact not visible in any aggregate number in the record, and one that
Experiment A's per-subject table will quantify directly.

## What this does NOT do

No training, no new network, no loss, no attention block, no TTA optimizer, no selector, no
refinement network, no 3-seed run. Experiments B (what makes a direction good) and C (does Γ
predict $R_i$) are **contingent on A** and not pre-registered here. GT is used to construct
$\delta_{\text{E15}}$ (inherited from E15's own design, acknowledged as oracle) and to score
Dice; random directions use no GT at all.
