# E18 Handoff Notes (paused for the night, 2026-08-08 ~05:30)

## Status: 5/6 configs valid and complete. freeze_bn needs a 3rd attempt.

### Complete and valid (ready for e18_measure_rotation.py):
- `e18_none_seed0` (baseline)
- `e18_freeze_encoder_seed0`
- `e18_freeze_decoder_seed0` (note: active_hinge_pct spiked to 100% early
  in training, epoch ~8, settled back to normal 4-9% range by epoch 23 --
  likely early-training instability, address briefly in the writeup)
- `e18_freeze_seg_head_seed0`
- `e18_lambda_zero_seed0`

### NOT valid, needs a 3rd attempt: freeze_bn
Two attempts so far, both collapsed to val_dice~0 by late training despite
healthy train_dice (~0.92-0.93):
- `e18_freeze_bn_BUGGY_seed0`: original bug, momentum=0 set at model
  construction before ANY real BatchNorm running-stat updates -- running
  stats frozen at PyTorch's random init (mean=0, var=1) forever.
- `e18_freeze_bn_ATTEMPT2_STILL_STALE_seed0`: fixed with a 1-epoch warmup
  before freezing -- NOT enough. val_dice still degrades progressively
  from epoch 2 onward, collapsing to ~0 by epoch 20+ (same pattern,
  delayed). Root cause: 1 epoch of momentum=0.1 updates only partially
  converges running_mean/var; freezing that immature snapshot goes stale
  as the live (train-mode) distribution keeps drifting -- consistent with
  E17's finding that the representation keeps rotating substantially
  through epoch ~5-10 before stabilizing.

### Agreed fix for attempt 3 (not yet implemented):
Warmup THROUGH epoch 10, freeze after (not epoch 1 or 5) -- matches E17's
own finding that the representation stabilizes from epoch 10 onward, so
freezing there captures running stats AFTER most of the early rotation
has already happened, avoiding the staleness problem structurally rather
than just delaying it. Also more scientifically apt: directly tests
whether BN's adaptation drives the EARLY rotation specifically (epochs
1-10), which is what E18 is investigating.

**Implementation**: in `e18_ablation_train.py`'s `train()` method, change
```python
if self.ablation == "freeze_bn" and epoch == 0:
    freeze_bn_running_stats(self.model)
```
to trigger after epoch index 9 (i.e. after the 10th real epoch, 0-indexed
`epoch == 9`) instead of `epoch == 0`. Update `freeze_bn_running_stats`'s
docstring accordingly (currently says "1-epoch warmup", needs updating to
reflect the actual chosen warmup length once implemented).

### Launch command for the corrected attempt (after the code fix above):
```bash
cd experiments/exp_e12_eggo_m
"c:/Users/Rivan/Projects/Neuroscan/venv_gpu/Scripts/python.exe" e18_ablation_train.py \
  --ablation freeze_bn --epochs 30 --seed 0 --num_workers 4 --run_name e18_freeze_bn_seed0
```
Run this ALONE (not via a watcher script) -- the earlier GPU OOM incident
was caused by a `pgrep`-based watcher that silently failed to wait
properly in this Git Bash environment. Either run it in the foreground,
or background it directly and poll the log file's mtime/tail manually
(reliable) rather than `ps aux` (shows stale zombie PID entries) or any
`pgrep`-based completion-detection script (not available in this
environment).

### Once freeze_bn (corrected, 3rd attempt) completes:
1. Run `e18_measure_rotation.py` across all 6 configs (baseline +
   freeze_bn[corrected] + freeze_encoder + freeze_decoder +
   freeze_seg_head + lambda_zero).
2. Write `PHASE_E18_REPRESENTATION_ROTATION_SOURCE.md` per the original
   phase spec (motivation/design/definitions/protocol/tables/figures/
   statistical analysis/interpretation/limitations/conclusion, success
   criteria A-E) -- include an honest, undramatic account of:
   - The freeze_bn bug (attempt 1) and its diagnosis.
   - The GPU OOM incident (my mistake, pgrep-based watcher script) and
     recovery.
   - The freeze_bn staleness issue (attempt 2) and why a longer,
     E17-informed warmup (attempt 3) was chosen.
   - The freeze_decoder active_hinge_pct early-training spike (settled
     by epoch 23, likely not a sustained issue).
3. Save findings to memory:
   - `abo_frozen_lessons_learned.md` + `MEMORY.md` (E18's actual
     scientific findings once complete).
   - `windows_training_env_gotchas.md`: add the `pgrep` unavailability
     lesson (Git Bash on this system doesn't have `pgrep`; a
     process-completion watcher using it will silently no-op and can
     cause parallel-launch GPU contention -- use log mtime polling or a
     `while [ -f <lockfile> ]` pattern instead) and the BatchNorm
     freeze-timing lesson (freezing running stats requires enough warmup
     for them to converge to a stable, representative snapshot --
     1 epoch was insufficient here; freezing before the representation's
     own early-training instability window has passed causes progressive
     eval-mode staleness even with a nonzero warmup).

## Files reference
- `e18_ablation_train.py` -- training script (needs the epoch==9 fix for attempt 3)
- `e18_measure_rotation.py` -- measurement script (already validated against E17's reference checkpoints, no changes needed)
- `e18_run_remaining.sh` / `.log` -- the (now fully consumed) queue script from tonight's run
- `e18_run_all.sh` / `.log` -- the original (partially consumed, superseded) queue script
