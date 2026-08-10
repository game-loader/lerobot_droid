# IMF-AttnRes on LIBERO: Training and Simulation Evaluation Notes

Date: 2026-05-16

This note records the practical procedure used to train the `imf-attnres` LeRobot policy on a single LIBERO scene and evaluate it with the LIBERO simulator.

> Security note: do not store W&B API keys in scripts or this document. Export `WANDB_API_KEY` in the shell before launching training.

## Code/worktree

The integration was developed in the LeRobot worktree:

```text
/tmp/lerobot-imf-attnres
```

Policy name:

```text
imf-attnres
```

The model is used through the standard LeRobot policy interface with:

```bash
--policy.type=imf-attnres
```

## Dataset

A local remapped subset was used instead of passing an episode filter to the original dataset. This avoids sampler/global-index mismatch issues when training with a subset.

Dataset path:

```text
/tmp/lerobot-imf-attnres/data/libero_task10
```

Dataset metadata:

```text
repo_id: local/libero_task10
task text: put the bowl on the plate
total_episodes: 49
total_frames: 4563
fps: 10.0
split: train 0:49
```

Main features:

```text
observation.images.image   uint8 image [256, 256, 3]
observation.images.image2  uint8 image [256, 256, 3]
observation.state          float32 [8]
action                     float32 [7]
```

The state has 8 dimensions while action has 7 dimensions because the observation state includes the two gripper joint positions in addition to end-effector pose information, whereas the action is the LIBERO 7-DoF control vector.

## Training

Final training run directory:

```text
/tmp/lerobot-imf-attnres/runs/imf-attnres-libero-task10-subset-20260515-181831
```

Final training output directory:

```text
/tmp/lerobot-imf-attnres/outputs/train/imf-attnres-libero-task10-subset-20260515-181831
```

Final checkpoint:

```text
/tmp/lerobot-imf-attnres/outputs/train/imf-attnres-libero-task10-subset-20260515-181831/checkpoints/020000/pretrained_model
```

Training duration was about 2h22m for 20k steps on CUDA.

### Training launch command

The command was saved as:

```text
/tmp/lerobot-imf-attnres/runs/imf-attnres-libero-task10-subset-20260515-181831/launch_train_task10_subset.sh
```

Equivalent command:

```bash
cd /tmp/lerobot-imf-attnres

export UV_CACHE_DIR=/tmp/uv-cache
export HF_HOME=/tmp/hf-home
export HF_DATASETS_CACHE=/tmp/hf-datasets-cache
# Export WANDB_API_KEY in the shell before launching; do not write the key into scripts/logs.
export WANDB_SILENT=True
export PYTHONUNBUFFERED=1

uv run --extra training lerobot-train \
  --dataset.repo_id=local/libero_task10 \
  --dataset.root=/tmp/lerobot-imf-attnres/data/libero_task10 \
  --dataset.use_imagenet_stats=true \
  --policy.type=imf-attnres \
  --policy.device=cuda \
  --policy.n_obs_steps=2 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.n_layer=4 \
  --policy.n_emb=256 \
  --policy.n_head=1 \
  --policy.n_kv_head=1 \
  --policy.backbone_type=attnres_full \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=null \
  --policy.resize_shape='[128,128]' \
  --policy.spatial_softmax_num_keypoints=32 \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.num_inference_steps=1 \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_weight_decay=1e-6 \
  --policy.push_to_hub=false \
  --steps=20000 \
  --batch_size=32 \
  --num_workers=4 \
  --prefetch_factor=2 \
  --persistent_workers=true \
  --save_checkpoint=true \
  --save_freq=5000 \
  --log_freq=50 \
  --eval_freq=0 \
  --output_dir=/tmp/lerobot-imf-attnres/outputs/train/imf-attnres-libero-task10-subset-20260515-181831 \
  --wandb.enable=true \
  --wandb.project=lerobot-imf-attnres \
  --wandb.disable_artifact=true \
  --wandb.mode=online
```

W&B was used for online metric syncing, with artifact upload disabled:

```bash
--wandb.disable_artifact=true
--policy.push_to_hub=false
```

Observed final log line:

```text
End of training
```

A checkpoint was saved at step 20000.

## LIBERO simulation evaluation

Evaluation used the standard LeRobot `lerobot-eval` entrypoint and the integrated LIBERO environment.

Important environment variables for headless rendering:

```bash
export LIBERO_CONFIG_PATH=/tmp/lerobot-imf-attnres/.libero_config
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export UV_CACHE_DIR=/tmp/uv-cache
```

The local LIBERO config pointed to existing assets and package BDDL/init files:

```yaml
assets: /home/droid/.libero/assets
bddl_files: /tmp/lerobot-imf-attnres/.venv/lib/python3.12/site-packages/libero/libero/bddl_files
init_states: /tmp/lerobot-imf-attnres/.venv/lib/python3.12/site-packages/libero/libero/init_files
```

Task mapping used for evaluation:

```text
LIBERO suite: libero_goal
task_id: 8
task language: put the bowl on the plate
```

The corresponding init-state file contains 50 pre-generated states:

```text
libero_goal/put_the_bowl_on_the_plate.pruned_init
shape: (50, 79)
```

### Init-state behavior

With the default `env.init_states=true`, LIBERO eval is not sampling arbitrary random initial states on every reset. LeRobot loads the pre-generated LIBERO init states and applies them via `set_init_state(...)`.

Relevant logic is in `src/lerobot/envs/libero.py`:

```python
raw_obs = self._env.set_init_state(self._init_states[self.init_state_id % len(self._init_states)])
self.init_state_id += self._reset_stride
```

Therefore:

- `eval.batch_size=1, eval.n_episodes=50` covers `init_state[0] ... init_state[49]` sequentially.
- `eval.batch_size=10, eval.n_episodes=50` also covers the 50 states, but through 10 vectorized environments. Each sub-env advances with stride 10.
- SmolVLA evaluated through the same LeRobot `lerobot-eval` + `env.type=libero` path uses the same environment reset/init-state logic; the policy changes, but the LIBERO reset logic does not.

The first 10 init states showed small tabletop pose perturbations. Across all 50 states, approximate xy ranges were:

```text
bowl:         x range 3.0 cm, y range 2.9 cm
plate:        x range 2.9 cm, y range 2.6 cm
wine bottle:  x range 2.8 cm, y range 3.0 cm
cream cheese: x range 3.9 cm, y range 3.5 cm
```

Rendered comparison image:

```text
/home/droid/project/roboimi/outputs/imf_attnres_eval_preview/init_states_goal8/init_states_first10_grid.png
```

## Eval run: 10 episodes

Run directory:

```text
/tmp/lerobot-imf-attnres/runs/imf-attnres-libero-goal8-eval10-20260516-151322
```

Output directory:

```text
/tmp/lerobot-imf-attnres/outputs/eval/imf-attnres-libero-goal8-eval10-20260516-151322
```

Command:

```bash
cd /tmp/lerobot-imf-attnres

export UV_CACHE_DIR=/tmp/uv-cache
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export LIBERO_CONFIG_PATH=/tmp/lerobot-imf-attnres/.libero_config
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

uv run --extra libero --extra evaluation lerobot-eval \
  --policy.path=/tmp/lerobot-imf-attnres/outputs/train/imf-attnres-libero-task10-subset-20260515-181831/checkpoints/020000/pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=false \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids='[8]' \
  --env.control_mode=relative \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.camera_name=agentview_image,robot0_eye_in_hand_image \
  --env.init_states=true \
  --env.episode_length=300 \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --output_dir=/tmp/lerobot-imf-attnres/outputs/eval/imf-attnres-libero-goal8-eval10-20260516-151322 \
  --seed=1000
```

Result:

```text
successes: 10 / 10
pc_success: 100.0
avg_sum_reward: 1.0
avg_max_reward: 1.0
eval_s: 27.28
eval_ep_s: 2.73
```

Videos:

```text
/tmp/lerobot-imf-attnres/outputs/eval/imf-attnres-libero-goal8-eval10-20260516-151322/videos/libero_goal_8/
```

A preview video was copied to:

```text
/home/droid/project/roboimi/outputs/imf_attnres_eval_preview/imf_attnres_libero_goal8_episode0.mp4
```

A 5x slow-motion version was created at:

```text
/home/droid/project/roboimi/outputs/imf_attnres_eval_preview/imf_attnres_libero_goal8_episode0_slow5x.mp4
```

## Eval run: 50 episodes, parallel

Run directory:

```text
/tmp/lerobot-imf-attnres/runs/imf-attnres-libero-goal8-eval50-parallel-20260516-162036
```

Output directory:

```text
/tmp/lerobot-imf-attnres/outputs/eval/imf-attnres-libero-goal8-eval50-parallel-20260516-162036
```

Command:

```bash
cd /tmp/lerobot-imf-attnres

export UV_CACHE_DIR=/tmp/uv-cache
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export LIBERO_CONFIG_PATH=/tmp/lerobot-imf-attnres/.libero_config
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

uv run --extra libero --extra evaluation lerobot-eval \
  --policy.path=/tmp/lerobot-imf-attnres/outputs/train/imf-attnres-libero-task10-subset-20260515-181831/checkpoints/020000/pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=false \
  --env.type=libero \
  --env.task=libero_goal \
  --env.task_ids='[8]' \
  --env.control_mode=relative \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.camera_name=agentview_image,robot0_eye_in_hand_image \
  --env.init_states=true \
  --env.episode_length=300 \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=10 \
  --eval.n_episodes=50 \
  --eval.use_async_envs=true \
  --output_dir=/tmp/lerobot-imf-attnres/outputs/eval/imf-attnres-libero-goal8-eval50-parallel-20260516-162036 \
  --seed=1000
```

Result:

```text
successes: 49 / 50
pc_success: 98.0
avg_sum_reward: 0.98
avg_max_reward: 0.98
eval_s: 81.90
eval_ep_s: 1.64
failed episode index: 11
```

LeRobot saved the first 10 eval videos by default:

```text
/tmp/lerobot-imf-attnres/outputs/eval/imf-attnres-libero-goal8-eval50-parallel-20260516-162036/videos/libero_goal_8/
```

The failed episode 11 was not among the first 10 saved videos. To inspect it visually, run a targeted eval or script that starts from/records the corresponding init state.

## Useful verification snippets

Read overall eval metrics:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('/tmp/lerobot-imf-attnres/outputs/eval/imf-attnres-libero-goal8-eval50-parallel-20260516-162036/eval_info.json')
info = json.loads(p.read_text())
print(info['overall'])
metrics = info['per_task'][0]['metrics']
print('failures:', [i for i, s in enumerate(metrics['successes']) if not s])
PY
```

Check generated videos:

```bash
find /tmp/lerobot-imf-attnres/outputs/eval/imf-attnres-libero-goal8-eval50-parallel-20260516-162036/videos -type f -name '*.mp4' -print
```

Create a 5x slow-motion video with ffmpeg:

```bash
ffmpeg -y \
  -i input.mp4 \
  -filter:v 'setpts=5.0*PTS' \
  -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  output_slow5x.mp4
```

## LIBERO four-suite main result experiment

A later four-suite experiment aligned to the SmolVLA LIBERO table format is recorded under:

```text
/home/droid/project/lerobot/docs/imf_attnres_libero_main_experiment/
```

Key result table:

```text
Spatial 52.4 | Object 0.0 | Goal 6.4 | Long 4.4 | Average 15.8
```

The durable experiment workspace is:

```text
/data/lerobot-imf-attnres-exp
```
