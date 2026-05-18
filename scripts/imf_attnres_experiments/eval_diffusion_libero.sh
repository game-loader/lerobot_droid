#!/usr/bin/env bash
# Evaluate a Diffusion Policy checkpoint on LIBERO spatial (all 10 tasks).
#
# Usage:
#   bash scripts/imf_attnres_experiments/eval_diffusion_libero.sh [step] [n_episodes]
#
# Examples:
#   bash scripts/imf_attnres_experiments/eval_diffusion_libero.sh 100000 50
#   bash scripts/imf_attnres_experiments/eval_diffusion_libero.sh 50000 10
#
set -euo pipefail

cd /data/lerobot-imf-attnres-exp/lerobot-imf-attnres

export EXP_ROOT=/data/lerobot-imf-attnres-exp
export LIBERO_CONFIG_PATH=$PWD/.libero_config
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR=$EXP_ROOT/cache/uv
export HF_HOME=$EXP_ROOT/cache/hf-home
export XDG_CACHE_HOME=$EXP_ROOT/cache/xdg
export TMPDIR=$EXP_ROOT/tmp
export CUDA_VISIBLE_DEVICES=${GPU:-0}

SUITE=spatial
STEPS=${1:-100000}
N_EPISODES=${2:-50}
BATCH_SIZE=${EVAL_BATCH:-10}

RUN_NAME="diffusion-libero-${SUITE}-s${STEPS}-b64"
CKPT_DIR=$EXP_ROOT/outputs/train/$RUN_NAME/checkpoints

# Find the checkpoint
STEP_DIR=$(printf "%06d" $STEPS)
CKPT=$CKPT_DIR/$STEP_DIR/pretrained_model

if [ ! -d "$CKPT" ]; then
    # Try without zero-padding
    CKPT=$CKPT_DIR/$STEPS/pretrained_model
fi

if [ ! -d "$CKPT" ]; then
    echo "ERROR: Checkpoint not found at $CKPT_DIR/$STEP_DIR or $CKPT_DIR/$STEPS"
    echo "Available checkpoints:"
    ls $CKPT_DIR/ 2>/dev/null || echo "  (none)"
    exit 1
fi

OUT_DIR=$EXP_ROOT/outputs/eval/$RUN_NAME/step_${STEP_DIR}_ep${N_EPISODES}

echo "Evaluating: $CKPT"
echo "Suite: libero_spatial (10 tasks)"
echo "Episodes per task: $N_EPISODES"
echo "Output: $OUT_DIR"
echo ""

uv run --extra libero --extra evaluation lerobot-eval \
    --policy.path=$CKPT \
    --policy.device=cuda \
    --policy.use_amp=false \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.control_mode=relative \
    --env.observation_height=256 \
    --env.observation_width=256 \
    --env.camera_name=agentview_image,robot0_eye_in_hand_image \
    --env.init_states=true \
    --env.episode_length=280 \
    --env.max_parallel_tasks=1 \
    --eval.batch_size=$BATCH_SIZE \
    --eval.n_episodes=$N_EPISODES \
    --eval.use_async_envs=true \
    --output_dir=$OUT_DIR \
    --seed=1000

echo ""
echo "Eval complete. Results:"
python -c "
import json
from pathlib import Path
info = json.loads(Path('$OUT_DIR/eval_info.json').read_text())
overall = info.get('overall', {})
print(f\"  Overall success rate: {overall.get('pc_success', 'N/A')}%\")
print(f\"  Episodes: {overall.get('n_episodes', 'N/A')}\")
print(f\"  Eval time: {overall.get('eval_s', 'N/A'):.1f}s\")
if 'per_task' in info:
    print('  Per-task:')
    for task, metrics in info['per_task'].items():
        print(f\"    {task}: {metrics.get('pc_success', 'N/A')}%\")
"
