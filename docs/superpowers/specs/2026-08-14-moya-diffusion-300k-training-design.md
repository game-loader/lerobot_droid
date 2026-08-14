# Moya Diffusion 300k Training Design

## Goal

Train a state-only Diffusion Policy for 300,000 optimizer steps on the local
100-episode randomized charger-grasp LeRobot v3 dataset. Record training and
periodic Newton evaluation metrics in SwanLab, then run a final 100-episode
Newton evaluation from the last checkpoint.

## Dataset

The immutable training source is:

```text
root: /home/droid/project/Moya_newton_sim/.worktrees/feat-fused-batched-env/runs/lerobot/randomized_grasp_100_20260813-230331/dataset
repo_id: moya_newton/randomized_grasp_100
version: v3.0
episodes: 100
frames: 93,000
frequency: 60 Hz
observation.state: float32[39]
action: float32[14]
```

There are no image features. LeRobot infers the 39D state input and 14D action
output from dataset metadata. The existing MIN_MAX processors normalize both
features; constant action dimensions use the processor epsilon path.

## Policy And Optimization

Use a smaller state-only temporal U-Net instead of the 250M-parameter PushT
default:

```text
policy.type: diffusion
down_dims: [128, 256, 512]
parameters: 16,889,102
n_obs_steps: 2
horizon: 64
n_action_steps: 32
num_train_timesteps: 100
num_inference_steps during periodic eval: 20
optimizer: Adam, lr 1e-4
scheduler: cosine, 500 warmup steps
batch_size: 256
steps: 300,000
precision: FP32 with TF32 enabled by LeRobot
```

The configuration was benchmarked on the RTX 5090 with the real dataset.
Batch 256 completed finite forward/backward updates, and the 17M-parameter
policy coexisted with the 16-world Newton environment without running out of
GPU memory.

## Checkpoint And Evaluation Schedule

```text
log_freq: 100
save_freq: 10,000
eval_freq: 10,000
periodic eval episodes: 16
periodic eval batch size: 16
episode horizon: 930
async envs: false
rendering/video: disabled
```

This produces 30 checkpoints and 30 periodic success-rate measurements. The
periodic evaluator uses 20 reverse-diffusion steps to control overhead. After
training, the final checkpoint is evaluated separately on 100 episodes with
100 reverse-diffusion steps.

Newton success requires every approved condition:

```text
true_grasp_ever
and clear_table_ever
and final_lift_height >= 0.015 m
and final_table_contacts == 0
and final_hand_contacts > 0
```

## Reusable Eval Environment Lifecycle

`lerobot-train` creates its evaluation environment once and reuses it at every
evaluation interval. `eval_policy_all()` currently closes every task
environment after one call, which is correct for standalone evaluation but not
for training-time reuse.

Add a default-true `close_envs_after_eval` option to `eval_policy_all()`.
Standalone evaluation keeps the current close behavior. Training passes
`False`, retains the environment across intervals, and closes it once when
training ends. A regression test must cover both default closing and explicit
reuse.

## SwanLab Recording

Use SwanLab 0.9.4's official W&B compatibility bridge before LeRobot creates
its existing `WandBLogger`:

```python
swanlab.sync_wandb(
    mode="online",
    wandb_run=False,
    log_dir=os.environ["SWANLAB_LOG_DIR"],
)
```

The package is injected reproducibly with `uv run --with swanlab==0.9.4`; it
does not modify the repository lockfile or runtime environment. The machine is
already authenticated to SwanLab as `game-loader`. W&B artifact upload is
disabled, and `wandb_run=False` prevents W&B cloud upload.

SwanLab project: `lerobot-moya-diffusion`.

Recorded scalar series include:

- `train/loss`, `train/grad_norm`, and `train/lr`;
- `train/steps`, `samples`, `epochs`, `update_s`, and `dataloading_s`;
- `eval/pc_success`, `eval/avg_sum_reward`, and `eval/eval_s`.

## Run Artifacts And Recovery

The run is stored below the ignored `outputs/train/` tree. Its launch script,
PID, combined log, SwanLab local data, checkpoints, and final evaluation JSON
remain together in one timestamped directory.

The launch command runs with writable Hugging Face and Warp caches under
`/tmp`. A checkpoint is saved before the matching evaluation. If interrupted,
training resumes from the `last` checkpoint using the saved train config; it
does not silently restart from step zero.

## Acceptance Criteria

- the lifecycle regression tests pass;
- a SwanLab smoke run records a finite training scalar without W&B upload;
- the 300,000-step process starts on CUDA and creates its first checkpoint and
  periodic Newton success metric as scheduled;
- the final checkpoint exists;
- the final 100-episode Newton evaluation completes with no videos; and
- final success rate and output paths are reported from persisted artifacts.
