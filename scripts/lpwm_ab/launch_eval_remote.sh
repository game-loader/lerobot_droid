#!/usr/bin/env bash
# Separate simulator-evaluation process; never restart or mutate running A/B training.
set -euo pipefail
ROOT=${LPWM_RUN_ROOT:-/root/lpwm_ab}
STAMP=${TRAIN_STAMP:-20260917-183742}
OUTPUT="$ROOT/evaluations/$STAMP"
mkdir -p "$OUTPUT"
cd "$ROOT/repo"
export PYTHONPATH="$ROOT/repo/src"
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=${EVAL_EGL_DEVICE:-0}
export LIBERO_CONFIG_PATH="$ROOT/libero_config"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup "$ROOT/.venv-eval/bin/python" -u scripts/lpwm_ab/watch_eval.py \
    --root "$ROOT" --stamp "$STAMP" --output "$OUTPUT" \
    --python "$ROOT/.venv-eval/bin/python" --device cuda \
    --episodes-per-task 10 --seed 42 --video \
    --credential-file "$ROOT/.swanlab_api_key" --swanlab-mode online \
    > "$OUTPUT/watcher.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUTPUT/watcher.pid"
printf 'EVAL_WATCHER_STARTED pid=%s root=%s gpu=%s\n' "$PID" "$OUTPUT" "$CUDA_VISIBLE_DEVICES"
