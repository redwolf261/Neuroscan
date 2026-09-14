# Phase E83 — Correspondence Kernel Characterization: confound substantially, not fully, explains E82's anisotropy

## Origin

The user flagged, before any further interpretation of E82's anisotropy
finding: BraTS native volumes are 240x240x155 at 1mm isotropic spacing
(verified directly from a real subject's NIfTI header). This project's
resize to a 64^3 cube (`scipy.ndimage.zoom`, per-axis zoom factors) is
therefore NOT physically isotropic in the resized tensor space:

| Axis | Native size | Resized-voxel size |
|---|---|---|
| x (tensor dim 0) | 240 | 3.750 mm |
| y (tensor dim 1) | 240 | 3.750 mm |
| z (tensor dim 2) | 155 | 2.422 mm |

A fixed VOXEL offset therefore corresponds to a different PHYSICAL
distance per axis — a 3-voxel z-shift (7.27mm) is a smaller physical
displacement than a 3-voxel x-shift (11.25mm). Verified before running
anything: the axis order matches directly (no transpose anywhere in the
loader), so tensor dim 2 is confirmed to be the native 155-slice
(anisotropic) axis.

## Method

Reused E82(b)'s local donor swap construction (magnitude held fixed,
LOCAL probability-change metric at swap sites — confirmed the only
sensitive metric for this sparse intervention in E82). Three tests:
1. Reproduce E82's voxel-matched axis comparison (magnitude 3 on each
   axis, unequal physical mm).
2. **Physical-distance-matched comparison**: x/y at voxel-offset 2
   (7.50mm) vs z at voxel-offset 3 (7.27mm) — the closest achievable
   integer-voxel match.
3. Reflection symmetry: K(+delta) vs K(-delta) per axis.

## Results (125 held-out subjects)

### (1) Voxel-matched (reproduces E82): real, significant

dim0 (x, voxel=3): 0.0530 vs dim2 (z, voxel=3): 0.0339 — effect size
0.0191, paired t p=1.8e-3.

### (2) Physical-distance-matched: effect shrinks ~48%, loses significance

dim0 (x, voxel=2, 7.50mm): 0.0439 vs dim2 (z, voxel=3, 7.27mm): 0.0339 —
effect size **0.0100** (down from 0.0191), paired t **p=0.142**,
Wilcoxon **p=0.135** — NOT significant at this sample size.

### (3) Reflection asymmetry (unplanned finding)

| Axis | K(+3) | K(-3) | p |
|---|---|---|---|
| dim0 (x) | 0.0530 | 0.0478 | 0.372 (symmetric) |
| dim1 (y) | 0.0402 | 0.0470 | 0.288 (symmetric) |
| dim2 (z) | 0.0339 | 0.0483 | **0.041** (asymmetric) |

Only the z-axis (inferior-superior direction) shows a significant
direction asymmetry — not predicted by either hypothesis, flagged here
rather than pursued further this session.

## Verdict: AMBIGUOUS — the confound explains roughly half the effect, not all of it, and the remainder is underpowered to confirm

Per the pre-declared reading and the actual numbers (not rounded to
either convenient conclusion): the axis-difference effect size drops by
~48% once offsets are matched for physical distance rather than voxel
count, and loses statistical significance in the process. This is
**substantial but not complete** evidence for the confound hypothesis —
large enough to say the voxel-spacing/resampling geometry is doing real
work in E82's original anisotropy finding, but the remaining gap
(0.0100) is not itself confidently distinguishable from zero at this
sample size (p=0.14).

**Correct scientific conclusion: E82's "anisotropic learned
representation" claim cannot be confirmed as stated.** At minimum,
roughly half of the observed axis effect is attributable to the
resampling geometry confound the user identified. Whether a genuine,
smaller residual anisotropy exists cannot be confirmed or ruled out from
this test — it would require either a larger sample or a more targeted
statistical design (e.g. many physical-distance-matched offset pairs
rather than one), neither of which is warranted as the next spend given
how much of the original effect the confound alone accounts for.

## What this means for the research direction

This closes the anisotropy thread as a load-bearing finding. The
"correspondence kernel" object the user proposed characterizing
(K(delta) as a stable, direction-dependent, learned property) does not
have enough surviving evidence after this control to serve as the basis
for an intervention — the honest state is that E82's headline
finding was substantially explained by ordinary preprocessing geometry,
not a novel representational phenomenon. Per this project's own
no-rescue discipline, this line does not warrant a further, more
statistically powerful re-test tonight; it is reported as an honest,
informative negative (or at best inconclusive) control result.

**What remains standing** from the broader E65-E82 chain, independent of
this specific anisotropy question:
- Translation vulnerability at the enc1 skip is real and large (E65,
  reproduced multiple times).
- Direction, not magnitude, carries the vulnerability (E78).
- The vulnerability is absolute-coordinate-relative binding, not local
  noise (E79) or a coordinate-frame artifact when both sides move
  together (E81 correction of E80).
- Local donor substitution damage is real and distance-graded even
  within a single axis (E82's radius 1→2→3 result on tensor dim2 alone,
  which does not involve a cross-axis comparison and is therefore
  unaffected by this phase's confound finding).

The isotropic, distance-graded core of the finding survives; the
cross-axis anisotropy claim specifically does not survive scrutiny
intact.

## Artifacts

- `experiments/exp_e12_eggo_m/e83/run_e83_correspondence_kernel.py`
- `experiments/exp_e12_eggo_m/e83/E83_correspondence_kernel_table.json`
- `experiments/exp_e12_eggo_m/e83/E83_correspondence_kernel_summary.json`
