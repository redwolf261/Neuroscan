#!/bin/bash
# Overnight unattended pipeline. Runs sequentially, each step gated on the
# previous succeeding. Never requires a decision -- runs pre-registered
# tests exactly as designed and records results faithfully, including
# negative ones. Every step's raw output is logged; a final summary is
# written for morning review.
#
# STAGE 0: wait for MM_A96 (already running) to finish.
# STAGE 1: evaluate MM_A96 with paired statistics against MM and FLAIR baselines.
# STAGE 2: CAS seed 1 (cached loader).
# STAGE 3: CAS seed 2 (cached loader).
# STAGE 4: 3-seed CAS confirmation statistics (only if seeds 1 AND 2 both completed cleanly).
# STAGE 5 (if time remains): CDCG prediction-1 fidelity check (cheap, no full training).
#
# Each stage writes its own timestamped log to /tmp/overnight_*.log and its
# own completion marker file. A stage that fails does NOT block later
# independent stages where avoidable, but CAS-seed-2 requires seed-1 to
# have produced a checkpoint (real dependency), and stage 4 requires both.

set -u  # (deliberately NOT set -e -- one stage's failure must not silently
        #  kill stages that don't depend on it; each stage checks its own
        #  preconditions explicitly instead)

cd "$(dirname "$0")/../../.."   # repo root
PY="./venv_gpu/Scripts/python.exe"
LOGDIR="/tmp"
SUMMARY="$LOGDIR/overnight_summary.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$SUMMARY"; }

log "=== Overnight pipeline started ==="

# ---------------- STAGE 0: wait for MM_A96 ----------------
log "Stage 0: waiting for MM_A96 (already running) to finish..."
while ! grep -q "DONE best_per_subject" "$LOGDIR/e70_a96_s0.log" 2>/dev/null; do
    if grep -qE "Traceback|CUDA out of memory|RuntimeError" "$LOGDIR/e70_a96_s0.log" 2>/dev/null; then
        log "Stage 0: MM_A96 appears to have FAILED. Log tail:"
        tail -20 "$LOGDIR/e70_a96_s0.log" >> "$SUMMARY"
        break
    fi
    sleep 30
done
log "Stage 0: MM_A96 wait loop exited."
tail -5 "$LOGDIR/e70_a96_s0.log" >> "$SUMMARY"

# ---------------- STAGE 1: evaluate MM_A96 ----------------
log "Stage 1: evaluating MM_A96 (paired stats vs MM and FLAIR baselines)..."
if [ -f "experiments/exp_e12_eggo_m/e70/runs/MM_A96_seed0/checkpoints/best.pth" ]; then
    "$PY" experiments/exp_e12_eggo_m/e70/evaluate_a96.py > "$LOGDIR/overnight_stage1_eval_a96.log" 2>&1
    if grep -q "Saved" "$LOGDIR/overnight_stage1_eval_a96.log" 2>/dev/null; then
        log "Stage 1: COMPLETE."
    else
        log "Stage 1: evaluation script may have failed -- check $LOGDIR/overnight_stage1_eval_a96.log"
    fi
else
    log "Stage 1: SKIPPED -- MM_A96 checkpoint not found (training likely failed)."
fi

# ---------------- STAGE 2: CAS seed 1 (cached loader) ----------------
log "Stage 2: launching CAS seed 1 (cached loader)..."
"$PY" experiments/exp_e12_eggo_m/e70/train_e70.py --condition MM --seed 1 --batch_size 4 --num_workers 2 --use_cache \
    > "$LOGDIR/overnight_stage2a_mm_s1.log" 2>&1
if grep -q "DONE best_per_subject" "$LOGDIR/overnight_stage2a_mm_s1.log" 2>/dev/null; then
    log "Stage 2a (MM seed1): COMPLETE."
else
    log "Stage 2a (MM seed1): may have FAILED -- check log."
fi

"$PY" experiments/exp_e12_eggo_m/e70/train_e70.py --condition MM_CAS --seed 1 --batch_size 4 --num_workers 2 --use_cache \
    > "$LOGDIR/overnight_stage2b_cas_s1.log" 2>&1
if grep -q "DONE best_per_subject" "$LOGDIR/overnight_stage2b_cas_s1.log" 2>/dev/null; then
    log "Stage 2b (MM_CAS seed1): COMPLETE."
else
    log "Stage 2b (MM_CAS seed1): may have FAILED -- check log."
fi

# ---------------- STAGE 3: CAS seed 2 (cached loader) ----------------
log "Stage 3: launching CAS seed 2 (cached loader)..."
"$PY" experiments/exp_e12_eggo_m/e70/train_e70.py --condition MM --seed 2 --batch_size 4 --num_workers 2 --use_cache \
    > "$LOGDIR/overnight_stage3a_mm_s2.log" 2>&1
if grep -q "DONE best_per_subject" "$LOGDIR/overnight_stage3a_mm_s2.log" 2>/dev/null; then
    log "Stage 3a (MM seed2): COMPLETE."
else
    log "Stage 3a (MM seed2): may have FAILED -- check log."
fi

"$PY" experiments/exp_e12_eggo_m/e70/train_e70.py --condition MM_CAS --seed 2 --batch_size 4 --num_workers 2 --use_cache \
    > "$LOGDIR/overnight_stage3b_cas_s2.log" 2>&1
if grep -q "DONE best_per_subject" "$LOGDIR/overnight_stage3b_cas_s2.log" 2>/dev/null; then
    log "Stage 3b (MM_CAS seed2): COMPLETE."
else
    log "Stage 3b (MM_CAS seed2): may have FAILED -- check log."
fi

# ---------------- STAGE 4: 3-seed confirmation ----------------
S1_OK=0; S2_OK=0
[ -f "experiments/exp_e12_eggo_m/e70/runs/MM_seed1/checkpoints/best.pth" ] && \
    [ -f "experiments/exp_e12_eggo_m/e70/runs/MM_CAS_seed1/checkpoints/best.pth" ] && S1_OK=1
[ -f "experiments/exp_e12_eggo_m/e70/runs/MM_seed2/checkpoints/best.pth" ] && \
    [ -f "experiments/exp_e12_eggo_m/e70/runs/MM_CAS_seed2/checkpoints/best.pth" ] && S2_OK=1

if [ "$S1_OK" = "1" ] && [ "$S2_OK" = "1" ]; then
    log "Stage 4: both seeds 1 and 2 have checkpoints -- running 3-seed confirmation..."
    "$PY" experiments/exp_e12_eggo_m/e70/evaluate_e70.py --seeds 0 1 2 > "$LOGDIR/overnight_stage4_3seed.log" 2>&1
    if grep -q "Saved E70_evaluation.json" "$LOGDIR/overnight_stage4_3seed.log" 2>/dev/null; then
        log "Stage 4: COMPLETE. 3-seed CAS confirmation done."
    else
        log "Stage 4: evaluation may have failed -- check log."
    fi
else
    log "Stage 4: SKIPPED -- seed1_ok=$S1_OK seed2_ok=$S2_OK (not both complete)."
fi

# ---------------- STAGE 5: CDCG prediction-1 check (if built) ----------------
if [ -f "experiments/exp_e12_eggo_m/e71/check_cdcg_prediction1.py" ]; then
    log "Stage 5: running CDCG prediction-1 fidelity check..."
    "$PY" experiments/exp_e12_eggo_m/e71/check_cdcg_prediction1.py > "$LOGDIR/overnight_stage5_cdcg.log" 2>&1
    log "Stage 5: script exited. Check $LOGDIR/overnight_stage5_cdcg.log for the result."
else
    log "Stage 5: SKIPPED -- CDCG check script not yet written."
fi

log "=== Overnight pipeline finished. Review $SUMMARY and individual stage logs. ==="
