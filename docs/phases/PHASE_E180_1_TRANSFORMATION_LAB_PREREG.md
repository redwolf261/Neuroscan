# E180 Stage 1 — Transformation laboratory + tile ledger: pre-registration

**Date**: 2026-09-16
**Status**: PRE-REGISTERED. Not run. No training, no architecture change, inference only.
**Depends on**: Stage 0 freeze (`experiments/exp_e12_eggo_m/e180/FROZEN/MANIFEST.json`).

---

## Purpose

E180 tests $H_{180}$: representation-induced decision instability ($\Gamma_i$) predicts
recoverable segmentation benefit from representation restoration ($\Delta_i$), beyond simpler
signals and boundary proximity. This stage builds the two prerequisites every later stage
depends on: (1) a deterministic tile-coordinate ledger, and (2) five controlled
representation-space transformations, hook-wired at **enc3/pool3**.

**Target stage: enc3.** Per E126/E165/E169b's mechanistic background — the encoder carries
size-independent rank demand (E169b: $R^2=0.518$ at enc3 after both size controls) and E126's
causal dose-response chain (enc3 window-blending → bottleneck necessity $N_b$) already lives
here. The bottleneck is known near-saturated (E165: mean $R^*/C = 0.018$, 90% of subjects served
by rank $\leq 4$ of 256) and is a weak site to probe instability at — there is little room left
to perturb.

## What "i" means (fixed here, binding on every later stage)

$$
i = \text{one enc3 activation, corresponding to one } 128^3 \text{ sliding-window tile}
$$

Not a voxel, not a "tumor region." Because sliding-window inference uses overlapping tiles
(`overlap=0.25`, Gaussian-blended reassembly per `_gaussian_weight`), one tile's activation
influences many output voxels, and one output voxel is influenced by several tiles. Every
downstream Γ/Δ quantity is therefore a **whole-volume** effect of perturbing/restoring one
tile's enc3 activation — never a purely local quantity.

### Tile ledger

Built once per subject from the existing `sliding_window()` start-index logic
(`starts()` in `run_e165_per_stage_rank.py:147-153`), recorded as:

```text
subject_id, window_id, z0, y0, x0, z1, y1, x1, enc3_shape
```

`window_id` is the tile's index in raster order over `(zs, ys, xs)`. `enc3_shape` records the
per-tile activation shape at hook time (should be constant, `(1, 128, 32, 32, 32)` for a full
128³ input tile, but recorded per-tile to catch edge-tile padding effects). Stored as
`experiments/exp_e12_eggo_m/e180/E180_tile_ledger.json`, keyed by subject, reused verbatim by
every later stage — never recomputed.

### Tile ledger overlap correction (found during Stage 5.5 diagnostics, applied before scale-up)

The 10-subject Stage 5.5 diagnostic pass surfaced two related geometry problems, both real, not
noise:

1. **Exact-duplicate enc3 masks.** `enc3_tile_bounds` maps input-space coordinates to the 32³
   enc3 grid via integer `//4` (the pool1+pool2 downsample factor). When two input tiles differ
   by less than 4 voxels in an axis — which happens for small subjects near the edge of the
   volume — both map to the **identical** enc3 mask. Confirmed directly: subject
   `BraTS-GLI-01189-000`'s `window_id=0` (`x0=0`) and `window_id=1` (`x0=2`) both produced
   `enc3=(0,0,0)-(32,32,32)`. Every downstream Γ/Δ record for such a pair is a literal duplicate.
2. **Heavy overlap between distinct masks.** Even where masks differ, tiles built at the
   inference `overlap=0.25` share up to ~87.5% of their enc3 extent (a 16-voxel input-space
   offset maps to a 4-voxel enc3 offset out of 32, i.e. 28/32 voxels shared). This produces
   near-identical, heavily autocorrelated Γ/Δ pairs across nominally different tiles, which
   inflated an early within-subject fixed-effects regression to an implausible R²≈0.998 — a
   sign of too little genuine independent variation, not a strong true effect.

**Fix, applied to `build_tile_ledger`**: (a) deduplicate — after computing each tile's enc3
mask, drop any tile whose mask exactly matches an already-kept tile for that subject; (b) use a
**separate, sparser overlap for tile selection** (`TILE_LEDGER_OVERLAP = 0.0`, non-overlapping
128³ tiles) independent of the `overlap=0.25` still used for sliding-window **inference/
reassembly quality** inside every Γ/Δ pass. This does not change how predictions are computed —
only which set of tiles is treated as a distinct observation `i`.

**Consequence**: the tile ledger and all Γ/Δ measurements were regenerated after this fix. Any
number reported before this correction (informal 10-subject Spearman ρ, the FE regression
R²≈0.998) is **discarded**, not carried forward — this is exactly the kind of implementation bug
Stage 5.5 exists to catch before the 125-subject run.

### A deeper, structural (non-bug) limit found after the fix — tile overlap is a data property

Even after deduplication and switching tile *selection* to `overlap=0`, adjacent tiles are often
still close together in enc3 space, because BraTS native volumes have limited slack over the
128³ patch: measured slack (native shape minus 128) is frequently only 0–20 voxels per axis
(some subjects even smaller than the patch, requiring padding), while a full enc3 cell is 4
input voxels wide. `sliding_window_starts` places exactly 2 positions per axis whenever slack is
nonzero, regardless of the `overlap` parameter, so on the tight axes two tile positions can be
as little as 1–2 enc3 cells apart out of 32 — still heavily overlapping, structurally, not from
a code defect.

**Adopted resolution**: this is not fixed further by ledger engineering. Every pair of tiles for
a subject gets a recorded **enc3-mask overlap fraction** (intersection-over-union of their enc3
bounding boxes), stored alongside the ledger. The tile-level sample size should be read as
smaller than the raw tile count in proportion to how overlapping the tiles are — this is
reported honestly rather than treated as resolved.

**Measured directly on the 10-subject diagnostic pass**: adjacent tiles share 83-94% of their
enc3 extent even after deduplication (`max_neighbor_enc3_iou` in the ledger). A full-tile-set
subject-fixed-effects regression (Δ ~ Γ + subject dummies) on this data gives an implausible
within-model R²≈0.997-0.999 — **this statistic is discarded as unreliable**, not merely
"controlled for," because neighboring windows are not independent observations in any sense the
regression's degrees-of-freedom accounting assumes.

### Locked analysis hierarchy for the 125-subject run (fixed now, not revisited post-hoc)

Per explicit instruction: do not keep changing the tile-separation criterion until a result
looks strongest. This hierarchy and the separation criterion are fixed before the 125-subject
run and not altered afterward except for documented bugs.

1. **Primary**: tile-level mixed/fixed-effects analysis accounting for subject (all surviving
   tiles, subject random or fixed effects). This is Stage 8's nested regression, extended with a
   subject-effect term.
2. **Robustness**: separated-tile subsampling — for each subject, the pair of tiles with maximum
   enc3-center L2 distance (the criterion already used in the 10-subject diagnostic; **fixed as
   this specific criterion**, not reselected later). Reported alongside the primary analysis, not
   substituted for it.
3. **Sensitivity**: one alternative separation criterion (e.g. minimum pairwise enc3-box IoU
   instead of maximum center distance), reported as a single fixed check, not a search.

### Formal sign/permutation test — a robustness diagnostic, not the main gate

For each subject $j$ with at least 2 tiles, define $s_j = \text{sign}(\rho_j(\Gamma, \Delta))$
(within-subject Spearman correlation sign, or for the 2-tile reduced design, the sign of
$\Delta(\text{tile}_1) - \Delta(\text{tile}_0)$ vs $\Gamma(\text{tile}_1) - \Gamma(\text{tile}_0)$
agreement). Test the count of positive $s_j$ against $H_0: P(s_j=+)=0.5$ via exact binomial.
**This is reported as a robustness diagnostic alongside the primary analysis, not used to
rescue or override a borderline primary result.** The 10-subject diagnostic pass gave 7-8/10
concordant (exact binomial p≈0.055-0.11 depending on family) — reported as directionally
encouraging, not statistically decisive, and that is the correct honest reading at this n.

### Provisional conclusion, stated at the correct confidence level (adopted wording)

> The exploratory association between representation-induced instability and restoration
> benefit survives removal of exact duplicate tiles and persists under substantially reduced
> tile granularity. However, strong tile-level fit statistics are considered unreliable because
> neighboring inference windows share substantial enc3 spatial support. The phenomenon therefore
> remains provisional pending subject-aware analysis on the full validation cohort.

## Transformations

All operate on the enc3 hook output `x: (1, 128, D, H, W)` for one tile, `D=H=W=32` for a
full interior tile.

| # | Name | Definition | Role |
|---|---|---|---|
| T1 | Rank compression | `lowrank_channels(x, r)` — mean-centered SVD truncation over the channel axis, reused verbatim from `run_e165_per_stage_rank.py:77-92`, dyadic grid `{1,2,4,8,16,32,64,128}` (capped at C=128) | scientific |
| T2 | Channel permutation | `x' = P @ x` (channel-axis permutation, fixed random `P` per severity draw) | **destructive negative control only** — never part of the eventual mechanism, per the user's explicit instruction. Demonstrates arbitrary representation destruction is not what Γ is meant to capture. |
| T3 | Local mixing | `x'_i = (1-beta)*x_i + beta*x_j` for spatially neighboring voxels `j` (fixed small 3D kernel, e.g. 6-neighbor average), `beta in {0, .05, .10, .20}` | scientific |
| T4 | Basis-preserving spectral reshape | SVD `x = U S V^T` (same decomposition as T1), modify `S' = S^gamma` preserving `U`,`V`, `gamma in {1.0, 1.5, 2.0, 3.0}` (gamma>1 concentrates energy toward top singular vectors; gamma<1, e.g. 0.5, flattens it) | scientific, extends T1's SVD scaffolding |
| T5 | Local smoothing | `x' = K_sigma * x` (Gaussian spatial kernel, small fixed `sigma`, e.g. `{0.5, 1.0, 1.5}` voxels), applied per-channel, spatial dims only | scientific |

T1's identity case is `r=C=128` (`lowrank_channels` returns `x` unchanged when `r>=c`). T2's
identity is `P = I`. T3's identity is `beta=0`. T4's identity is `gamma=1.0`. T5's identity is
`sigma=0` (no smoothing) — every transform has a well-defined no-op setting used for sanity
checks below.

## Hook wiring

Single hook on `model.enc3`, mirroring `StageTruncator`
(`run_e165_per_stage_rank.py:95-114`): a mutable `(transform_name, param)` pair, dispatching to
the corresponding function, pass-through when `None`. This lets one registered hook set cover
all five families without re-registering per transform.

## Sanity gates (must pass before any subject loop runs)

Reused directly from E165 (`run_e165_per_stage_rank.py:205-223`) and E126
(`unit_test_blend_windows`, `run_e126_graded_rank_reduction_causal_test.py:127-144`):

1. **Dormant hook exact no-op**: with no transform active, two forward passes on the same tile
   must be bit-identical (`max abs diff == 0.0`).
2. **T1 identity**: `r=128` truncation must be near-identity (`max abs diff < 1e-3`, matching
   E165's tolerance for SVD reconstruction rounding).
3. **T2 identity**: `P=I` permutation must be an exact no-op (`max abs diff == 0.0`).
4. **T3 identity**: `beta=0` must be an exact no-op.
5. **T4 identity**: `gamma=1.0` must be near-identity (`max abs diff < 1e-3`, same SVD
   reconstruction tolerance as T1).
6. **T5 identity**: `sigma=0` (kernel size 1, i.e. no smoothing) must be an exact no-op.
7. **Checkpoint identity**: `best_mean_dice == 0.8929357248544694`, per Stage 0.

Any gate failing stops the stage — no severity calibration or Γ/Δ measurement proceeds on a
broken transform.

## What this stage does NOT do

- Does not choose severities yet (Stage 2).
- Does not compute Γ or Δ yet (Stages 3-5).
- Does not touch ground truth.
- Does not modify architecture, training, or any prior experiment's artifacts.

## Output

- `experiments/exp_e12_eggo_m/e180/E180_tile_ledger.json`
- `experiments/exp_e12_eggo_m/e180/E180_1_sanity_summary.json` (pass/fail per gate, per
  transform, on a small subject sample)
