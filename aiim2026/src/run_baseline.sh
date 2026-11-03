#!/bin/bash
# AIIM 2026 baseline reproduction: flat LeWM planning on PushT, d in {25,50,75}, 3 seeds.
# Protocol from Mind the Gap (arXiv 2607.12547) Table 3: h=5, r=5, block=5, CEM 300x30,
# budget 50/100/150 for d=25/50/75, 50 episodes per config.
# Usage: bash run_baseline.sh [device]   (device: cpu | mps, default cpu)
set -u
DEVICE="${1:-cpu}"
PY=/Users/wanglanlan/miniforge3/envs/aiim/bin/python3
SWM=/Users/wanglanlan/course/Quentropy/E2_JEPA/aiim2026/third_party/stable-worldmodel/scripts/plan
LOGDIR=/Users/wanglanlan/course/Quentropy/E2_JEPA/aiim2026/runs
mkdir -p "$LOGDIR"

cd "$SWM"
for seed in 42 43 44; do
  for pair in "25 50" "50 100" "75 150"; do
    set -- $pair; D=$1; B=$2
    TAG="d${D}_seed${seed}"
    echo "=== $TAG (budget $B, device $DEVICE) ==="
    "$PY" eval_wm.py --config-name=pusht.yaml \
      policy=pusht/weights.pt \
      eval.goal_offset_steps=$D \
      eval.eval_budget=$B \
      +plan_config.history_len=3 \
      solver.device=$DEVICE \
      seed=$seed \
      2>&1 | tail -20 | tee "$LOGDIR/flat_${TAG}.log"
    grep "metrics:" -A0 "$LOGDIR/flat_${TAG}.log" | head -2
  done
done
echo "ALL DONE. Aggregate with: python3 src/collect_baseline.py"
