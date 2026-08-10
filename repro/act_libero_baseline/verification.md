# Verification

## Baseline target
- Policy: ACT (`--policy.type=act`)
- Benchmark: LIBERO (`--env.type=libero`)
- Dataset: `HuggingFaceVLA/libero`
- Repo commit: `ba27aab7` on `main`

## Environment route
- `uv sync --locked --python 3.12 --extra training --extra evaluation --extra libero`
- Python 3.14 failed because `mujoco==3.7.0` fell back to source build and required `MUJOCO_PATH`.
- `egl-probe==1.0.2` failed with CMake 4.x because its `CMakeLists.txt` declares `cmake_minimum_required(VERSION 2.8.12)`. A temporary wrapper calling `/usr/bin/cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5` fixed the build.
- CUDA runtime initially failed with missing `libcudnn_graph.so*`; resolved by exporting `LD_LIBRARY_PATH` to all `.venv/lib/python3.12/site-packages/nvidia/*/lib` directories.
- LIBERO assets/config prepared under `~/.libero/` using the repo's Dockerfile pattern.

## Verified commands
### Train + eval smoke
```bash
CUDA_LIBS=$(find "$PWD/.venv/lib/python3.12/site-packages/nvidia" -maxdepth 3 -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
MUJOCO_GL=egl HF_HOME=/tmp/hf UV_CACHE_DIR=/tmp/uv-cache uv run lerobot-train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  '--dataset.episodes=[0]' \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --env.type=libero \
  --env.task=libero_spatial \
  '--env.task_ids=[0]' \
  --batch_size=1 \
  --num_workers=0 \
  --steps=1 \
  --eval_freq=1 \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --save_freq=1 \
  --wandb.enable=false \
  --output_dir=outputs/act_libero_smoke_cuda \
  --job_name=act_libero_smoke_cuda
```

### Standalone eval
```bash
CUDA_LIBS=$(find "$PWD/.venv/lib/python3.12/site-packages/nvidia" -maxdepth 3 -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
MUJOCO_GL=egl HF_HOME=/tmp/hf UV_CACHE_DIR=/tmp/uv-cache uv run lerobot-eval \
  --policy.path=outputs/act_libero_smoke_cuda/checkpoints/000001/pretrained_model \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  '--env.task_ids=[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval.use_async_envs=false \
  --env.max_parallel_tasks=1 \
  --output_dir=outputs/act_libero_smoke_cuda_eval
```

## Observed outputs
- Training checkpoint:
  - `outputs/act_libero_smoke_cuda/checkpoints/000001/pretrained_model/`
- Training-loop eval video:
  - `outputs/act_libero_smoke_cuda/eval/videos_step_000001/libero_spatial_0/eval_episode_0.mp4`
- Standalone eval report:
  - `outputs/act_libero_smoke_cuda_eval/eval_info.json`
- Standalone eval video:
  - `outputs/act_libero_smoke_cuda_eval/videos/libero_spatial_0/eval_episode_0.mp4`

## Observed metrics
Standalone eval (`outputs/act_libero_smoke_cuda_eval/eval_info.json`):
- `avg_sum_reward = 0.0`
- `avg_max_reward = 0.0`
- `pc_success = 0.0`
- `n_episodes = 1`
- `eval_s ≈ 8.19s`

## Trust status
- **Verified:** the current `lerobot` repo can train and evaluate ACT on LIBERO with `uv` once the environment issues above are fixed.
- **Not yet reproduced:** any published ACT benchmark number on LIBERO. The current run is only a 1-step / 1-episode smoke validation, so 0% success is expected and not meaningful for comparison.

## Next anchor
Scale the same command from smoke to a real training run (more episodes, more steps, possibly multi-suite eval) if benchmark-quality metrics are desired.
