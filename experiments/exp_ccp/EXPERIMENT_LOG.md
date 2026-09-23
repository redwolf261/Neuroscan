# CCP Frozen-Model Oracle -- Experiment Log

## Provenance
- git commit: 01df9b3ab363a15165ee07f6fcd1d11d742e2d79
- checkpoint: experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth
  (epoch 34, arch v5, best_mean_dice 0.8929 [25-subject fast-val])
- dataset: BraTS 2023 GLI, BraTSMultimodalDataset 'val' split, val_split=0.1,
  RandomState(42) shuffle, last 10% = 125 subjects ("NeuroScan 125-subject
  proving-ground cohort" -- NOT an untouched test set)
- inference: unchanged baseline sliding_window_predict, PATCH=(128,128,128),
  SW_OVERLAP=0.5, Gaussian blending, AMP enabled
- threshold: 0.5 (>= 0.5), identical to baseline; NOT re-optimised
- region: WT only (REGIONS index 2). ET/TC untouched.

## Parameters (all preregistered, none tuned on the cohort)
eta=0.05, lambda=10, tau=(0.5,0.25,0.125), delta=0.25
R1=(0.5,0.3,0.2) R2=(0.4,0.4,0.2) R3=(0.5,0.2,0.3)
4 dynamic rounds: competition, then proofread-1/2/3 each + competition.

## Numerics
- CCP dynamics in float64 (GPU). GPU float64 verified BIT-IDENTICAL to CPU
  float64 (max diff 0.0) on the competition step.
- Conservation: max relative error 6.08e-16, mean 7.06e-17 across all
  conditions/configs/subjects.
- mass_CCP vs mass_pred: max rel diff 3.87e-16.
- Per-voxel outflow clamp: triggered 0 times on real data (max 0, mean 0.0).

## Verified implementation properties (unit tests, see transcript)
- Conservation exact under the clamp path (forced with eta=5.0: 113 clamped
  voxels, rel err still 0.0).
- Flux directionality: mass flows toward the higher-R neighbour, zero toward
  the lower-R neighbour.
- NOT smoothing: q=0.9 (low R) loses mass to q=0.1 (high R) -> 0.86 / 0.14.
  Gaussian smoothing would do the opposite.
- Fast neighbourhood_mean verified equal to the explicit 26-shift reference
  (neighbourhood_mean_ref) to 3.33e-16 including at volume boundaries.

## Documented deviations
1. neighbourhood_mean computed by 3x3x3 sum-pool minus centre rather than 26
   explicit shifts (verified numerically identical; reference retained).
2. Dynamics executed on GPU rather than CPU (verified bit-identical).
Neither alters the mathematical experiment.

## Baseline verification
WT mean Dice reproduced = 0.9119067 (authoritative record: 0.9119067). PASS.

## Runtime
- main oracle run: 565s for 125 subjects (~4.5 s/subject)
- arithmetic ceiling run: 161s for 125 subjects

## Outputs
CCP_per_subject.csv, CCP_stages.csv, CCP_summary.json, CCP_ceiling.csv,
ccp_viz_*.png, ccp_run.log
