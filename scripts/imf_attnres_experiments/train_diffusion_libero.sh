#!/usr/bin/env bash
# Train standard Diffusion Policy on LIBERO spatial suite.
# Target: reproduce SmolVLA paper's 78.3% success rate on libero_spatial.
#
# Protocol (from SmolVLA / LeRobot LIBERO benchmark):
#   - Train one model on all 10 tasks of libero_spatial (432 episodes)
#   - Eval: rollout each task separately, report mean success rate
#   - Training budget: 100k steps, batch_size=64
#   - Diffusion policy: resnet18 backbone, horizon=16, n_action_steps=8
#
# Usage:
#   cd /data/lerobot-imf-attnres-exp/lerobot-imf-attnres
#   bash scripts/imf_attnres_experiments/train_diffusion_libero.sh
#
set -euo pipefail

cd /data/lerobot-imf-attnres-exp/lerobot-imf-attnres

# ─── Environment ───────────────────────────────────────────────────────────────
export EXP_ROOT=/data/lerobot-imf-attnres-exp
export LIBERO_CONFIG_PATH=$PWD/.libero_config
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export PYTHONUNBUFFERED=1

export UV_CACHE_DIR=$EXP_ROOT/cache/uv
export HF_HOME=$EXP_ROOT/cache/hf-home
export HF_DATASETS_CACHE=$EXP_ROOT/cache/hf-datasets
export XDG_CACHE_HOME=$EXP_ROOT/cache/xdg
export TMPDIR=$EXP_ROOT/tmp

export WANDB_DIR=$EXP_ROOT/wandb
export WANDB_CACHE_DIR=$EXP_ROOT/cache/wandb
export WANDB_CONFIG_DIR=$EXP_ROOT/cache/wandb-config
export WANDB_SILENT=True

# GPU selection (override with: GPU=1 bash train_diffusion_libero.sh)
export CUDA_VISIBLE_DEVICES=${GPU:-0}

# ─── Config ────────────────────────────────────────────────────────────────────
SUITE=spatial
STEPS=${STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-64}
SAVE_FREQ=${SAVE_FREQ:-10000}
LR=${LR:-1e-4}
WANDB_MODE=${WANDB_MODE:-online}

RUN_NAME="diffusion-libero-${SUITE}-s${STEPS}-b${BATCH_SIZE}"
OUT_DIR=$EXP_ROOT/outputs/train/$RUN_NAME
DATASET_ROOT=$EXP_ROOT/datasets/libero_spatial

# ─── Ensure dataset subset exists ─────────────────────────────────────────────
# Reuse the existing materialized subset (already prepared by train_eval_suite.py)
if [ ! -d "$DATASET_ROOT/meta" ]; then
    echo "Dataset subset not found at $DATASET_ROOT. Materializing..."
    python -c "
import sys
sys.path.insert(0, 'scripts/imf_attnres_experiments')
from train_eval_suite import ensure_subset_dataset
from pathlib import Path
ensure_subset_dataset(
    source_root=Path('/data/lerobot_datasets/HuggingFaceVLA/libero'),
    subset_root=Path('$DATASET_ROOT'),
    episodes_path=Path('$EXP_ROOT/spatial_episodes.txt'),
)
print('Dataset subset ready.')
"
fi

echo "Dataset: $DATASET_ROOT"
echo "Output:  $OUT_DIR"
echo "Steps:   $STEPS, Batch: $BATCH_SIZE, LR: $LR"
echo "GPU:     $CUDA_VISIBLE_DEVICES"
echo ""

# ─── Training ─────────────────────────────────────────────────────────────────
uv run --extra training --extra libero --extra evaluation lerobot-train \
    --dataset.repo_id=local/libero_spatial \
    --dataset.root=$DATASET_ROOT \
    --dataset.use_imagenet_stats=true \
    --policy.type=diffusion \
    --policy.device=cuda \
    --policy.n_obs_steps=2 \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    --policy.vision_backbone=resnet18 \
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
    --policy.resize_shape='[128,128]' \
    --policy.crop_ratio=0.9 \
    --policy.crop_is_random=true \
    --policy.use_group_norm=false \
    --policy.spatial_softmax_num_keypoints=32 \
    --policy.use_separate_rgb_encoder_per_camera=true \
    --policy.down_dims='[256,512,1024]' \
    --policy.kernel_size=5 \
    --policy.n_groups=8 \
    --policy.diffusion_step_embed_dim=128 \
    --policy.use_film_scale_modulation=true \
    --policy.noise_scheduler_type=DDPM \
    --policy.num_train_timesteps=100 \
    --policy.num_inference_steps=10 \
    --policy.beta_schedule=squaredcos_cap_v2 \
    --policy.prediction_type=epsilon \
    --policy.clip_sample=true \
    --policy.clip_sample_range=1.0 \
    --policy.optimizer_lr=$LR \
    --policy.optimizer_betas='[0.95,0.999]' \
    --policy.optimizer_weight_decay=1e-6 \
    --policy.scheduler_name=cosine \
    --policy.scheduler_warmup_steps=500 \
    --policy.push_to_hub=false \
    --steps=$STEPS \
    --batch_size=$BATCH_SIZE \
    --num_workers=4 \
    --prefetch_factor=2 \
    --persistent_workers=true \
    --save_checkpoint=true \
    --save_freq=$SAVE_FREQ \
    --log_freq=100 \
    --eval_freq=0 \
    --output_dir=$OUT_DIR \
    --wandb.enable=true \
    --wandb.project=lerobot-diffusion-libero-baseline \
    --wandb.disable_artifact=true \
    --wandb.mode=$WANDB_MODE \
    --job_name=$RUN_NAME

echo ""
echo "Training complete. Checkpoints at: $OUT_DIR/checkpoints/"
echo ""
echo "To evaluate the best checkpoint, run:"
echo "  bash scripts/imf_attnres_experiments/eval_diffusion_libero.sh <step>"
