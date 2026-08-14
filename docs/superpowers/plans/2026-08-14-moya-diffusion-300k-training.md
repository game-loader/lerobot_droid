# Moya Diffusion 300k Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make periodic training evaluation reuse the Moya Newton environment safely, then launch the approved 300,000-step state-only Diffusion Policy run with SwanLab logging and CUDA evaluation.

**Architecture:** `eval_policy_all()` keeps standalone evaluation's close-on-return behavior through a default-true option, while `lerobot-train` opts into reuse and performs the final close after training. A timestamped ignored run directory owns a self-contained launcher, logs, PID, SwanLab files, caches, checkpoints, and the later final evaluation output; LeRobot writes into the initially absent `$RUN_DIR/train` child because it rejects pre-existing output directories.

**Tech Stack:** Python 3.12, PyTorch, LeRobot, Gymnasium vector environments, pytest, uv, SwanLab 0.9.4, CUDA, Newton/Warp.

---

### Task 1: Make Eval Environment Closing Configurable

**Files:**
- Modify: `src/lerobot/envs/configs.py:58`
- Modify: `src/lerobot/scripts/lerobot_eval.py:706`
- Modify: `src/lerobot/scripts/lerobot_train.py:533`
- Test: `tests/scripts/test_lerobot_eval_rendering.py`
- Test: `tests/scripts/test_lerobot_train_rendering.py`

- [ ] **Step 1: Write failing close and reuse regression tests**

Add two small fake environments with counted use and `close()` methods, stub `run_one()` to mark the supplied environment as used and return one successful episode, and cover both environments in these tests:

```python
class CloseTrackingEnv:
    def __init__(self) -> None:
        self.close_calls = 0
        self.use_calls = 0

    def mark_used(self) -> None:
        self.use_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _mark_used_and_return_metrics(task_group, task_id, env, **_kwargs):
    env.mark_used()
    return task_group, task_id, {
        "sum_rewards": [1.0],
        "max_rewards": [1.0],
        "successes": [True],
        "video_paths": [],
    }


@pytest.mark.parametrize("max_parallel_tasks", [1, 2])
def test_eval_policy_all_closes_envs_by_default(monkeypatch, max_parallel_tasks: int) -> None:
    envs = {0: CloseTrackingEnv(), 1: CloseTrackingEnv()}
    monkeypatch.setattr(
        lerobot_eval,
        "run_one",
        _mark_used_and_return_metrics,
    )

    lerobot_eval.eval_policy_all(
        {"suite": envs},
        policy=None,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=1,
        max_parallel_tasks=max_parallel_tasks,
    )

    assert [env.use_calls for env in envs.values()] == [1, 1]
    assert [env.close_calls for env in envs.values()] == [1, 1]


@pytest.mark.parametrize("max_parallel_tasks", [1, 2])
def test_eval_policy_all_can_reuse_envs(monkeypatch, max_parallel_tasks: int) -> None:
    envs = {0: CloseTrackingEnv(), 1: CloseTrackingEnv()}
    monkeypatch.setattr(
        lerobot_eval,
        "run_one",
        _mark_used_and_return_metrics,
    )

    kwargs = dict(
        envs={"suite": envs},
        policy=None,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=1,
        max_parallel_tasks=max_parallel_tasks,
        close_envs_after_eval=False,
    )
    lerobot_eval.eval_policy_all(**kwargs)
    lerobot_eval.eval_policy_all(**kwargs)

    assert [env.use_calls for env in envs.values()] == [2, 2]
    assert [env.close_calls for env in envs.values()] == [0, 0]
```

- [ ] **Step 2: Run the tests and confirm the new keyword fails**

Run:

```bash
uv run pytest tests/scripts/test_lerobot_eval_rendering.py -q
```

Expected: the reuse cases fail because `eval_policy_all()` does not accept `close_envs_after_eval`.

- [ ] **Step 3: Add the lifecycle option and guard both close paths**

Add the keyword-only argument:

```python
close_envs_after_eval: bool = True,
```

In both sequential and threaded `finally` blocks, replace unconditional closing with:

```python
if close_envs_after_eval:
    env.close()
```

Only start sequential prefetch after an environment was closed, because its purpose is to avoid memory overlap between distinct task environments:

```python
if close_envs_after_eval and i + 1 < len(tasks):
```

- [ ] **Step 4: Make training opt into reuse**

Add the non-configurable environment capability to the base config and enable it only for Moya:

```python
supports_eval_env_reuse: ClassVar[bool] = False
```

```python
supports_eval_env_reuse: ClassVar[bool] = True
```

Pass the capability-derived argument from the periodic training call:

```python
close_envs_after_eval=not cfg.env.supports_eval_env_reuse,
```

Do not change standalone `lerobot-eval`; its omitted argument preserves closing by default. Add this owner context manager and wrap the training loop with it so cleanup runs on normal completion or exceptions:

```python
@contextmanager
def _close_eval_envs_after_training(eval_env: Any) -> Iterator[None]:
    try:
        yield
    finally:
        if eval_env:
            close_envs(eval_env)
```

Move the actual `make_env()` call from early initialization to immediately before this context manager, after policy, processors, optimizer, dataloader, and `accelerator.prepare()` are ready. Add focused tests proving PushT retains close-after-eval behavior, Moya enables reuse, the capability is absent from `dataclasses.fields()`, and the context manager closes exactly once when its body succeeds or raises.

Keep sequential lazy prefetch outside the per-task `finally` block:

```python
try:
    tg, tid, metrics = task_runner(task_group, task_id, env)
    _accumulate_to(tg, metrics)
    per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
finally:
    if close_envs_after_eval:
        env.close()

if close_envs_after_eval and i + 1 < len(tasks):
    next_env = tasks[i + 1][2]
    if hasattr(next_env, "_ensure"):
        prefetch_thread = threading.Thread(target=next_env._ensure, daemon=True)
        prefetch_thread.start()
```

This ensures an exception skips prefetching the next lazy environment. Test the failure path with two environments and assert the second environment's `_ensure()` is never called.

- [ ] **Step 5: Run focused tests and static checks**

Run:

```bash
uv run pytest tests/scripts/test_lerobot_eval_rendering.py tests/scripts/test_lerobot_train_rendering.py -q
uv run ruff check src/lerobot/envs/configs.py src/lerobot/scripts/lerobot_eval.py src/lerobot/scripts/lerobot_train.py tests/scripts/test_lerobot_eval_rendering.py tests/scripts/test_lerobot_train_rendering.py
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 6: Commit the lifecycle fix**

```bash
git add src/lerobot/envs/configs.py src/lerobot/scripts/lerobot_eval.py src/lerobot/scripts/lerobot_train.py tests/scripts/test_lerobot_eval_rendering.py tests/scripts/test_lerobot_train_rendering.py docs/superpowers/plans/2026-08-14-moya-diffusion-300k-training.md
git commit -m "fix(eval): reuse environments during training"
```

### Task 2: Create A Reproducible 300k Launcher

**Files:**
- Create at runtime: `outputs/train/moya_diffusion_300k_<timestamp>/launch.sh`

- [ ] **Step 1: Create the timestamped run directory**

Run:

```bash
timestamp="$(date +%Y%m%d-%H%M%S)"
run_dir="$PWD/outputs/train/moya_diffusion_300k_$timestamp"
mkdir -p "$run_dir"
ln -sfn "$(basename "$run_dir")" outputs/train/moya_diffusion_300k_latest
printf '%s\n' "$run_dir"
```

This creates the directory below `outputs/train/` and records the current run in `outputs/train/moya_diffusion_300k_latest` for monitoring.

- [ ] **Step 2: Write the launcher with fail-fast shell behavior**

Create an executable `launch.sh` containing:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/droid/project/lerobot_droid
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_ROOT=/home/droid/project/Moya_newton_sim/.worktrees/feat-fused-batched-env/runs/lerobot/randomized_grasp_100_20260813-230331/dataset

export HF_HOME="$RUN_DIR/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WARP_CACHE_PATH="$RUN_DIR/cache/warp"
export SWANLAB_LOG_DIR="$RUN_DIR/swanlab"
export UV_CACHE_DIR="$ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_DATASETS_CACHE" "$WARP_CACHE_PATH" "$SWANLAB_LOG_DIR"
cd "$ROOT"

exec uv run --with swanlab==0.9.4 python -c '
import os
import swanlab

swanlab.sync_wandb(
    mode="online",
    wandb_run=False,
    log_dir=os.environ["SWANLAB_LOG_DIR"],
)

from lerobot.scripts.lerobot_train import main

main()
' \
  --dataset.repo_id=moya_newton/randomized_grasp_100 \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.down_dims='[128,256,512]' \
  --policy.num_inference_steps=20 \
  --env.type=moya_newton \
  --env.device=cuda:0 \
  --env.episode_length=930 \
  --env.headless=true \
  --env.sim_substeps=8 \
  --env.preset=randomized_grasp_v1 \
  --env.success_min_final_lift_height=0.015 \
  --steps=300000 \
  --batch_size=256 \
  --num_workers=4 \
  --log_freq=100 \
  --save_freq=10000 \
  --eval_freq=10000 \
  --eval.n_episodes=16 \
  --eval.batch_size=16 \
  --eval.use_async_envs=false \
  --wandb.enable=true \
  --wandb.project=lerobot-moya-diffusion \
  --wandb.disable_artifact=true \
  --job_name="$(basename "$RUN_DIR")" \
  --output_dir="$RUN_DIR/train"
```

- [ ] **Step 3: Validate shell syntax and resolved CLI configuration**

Run:

```bash
bash -n outputs/train/moya_diffusion_300k_<timestamp>/launch.sh
uv run lerobot-train --help
```

Expected: shell syntax succeeds and the training entry point is available.

### Task 3: Launch And Verify The Long-Running Job

**Files:**
- Create at runtime: `outputs/train/moya_diffusion_300k_<timestamp>/train.log`
- Create at runtime: `outputs/train/moya_diffusion_300k_<timestamp>/train.pid`
- Create at runtime: `outputs/train/moya_diffusion_300k_<timestamp>/launcher.pid`

- [ ] **Step 1: Start the launcher detached**

Run from the repository root with `RUN_DIR` set to the absolute timestamped directory:

```bash
nohup "$RUN_DIR/launch.sh" >"$RUN_DIR/train.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$RUN_DIR/launcher.pid"
printf '%s\n' "$pid" >"$RUN_DIR/train.pid"
```

The launcher uses `exec`, so the ID remains the training process ID.

- [ ] **Step 2: Verify process, CUDA, and resolved configuration**

Check:

```bash
ps -fp "$(cat outputs/train/moya_diffusion_300k_<timestamp>/train.pid)"
nvidia-smi
tail -n 120 outputs/train/moya_diffusion_300k_<timestamp>/train.log
```

Expected evidence in the log: dataset loads 93,000 frames, policy uses CUDA, output directory is the timestamped run's `train` child, steps equal 300,000, batch size equals 256, and training reports a finite loss.

- [ ] **Step 3: Verify SwanLab local state and online run URL**

Inspect `swanlab/` and the training log. Persist the SwanLab URL in `swanlab_url.txt` when it appears.

- [ ] **Step 4: Monitor through the first stable logged interval**

Wait until at least one `train/loss` report after initialization, confirm the PID remains alive, and check GPU memory and utilization. If initialization fails, diagnose the concrete error before restarting; never leave two training processes for the same run.

- [ ] **Step 5: Leave the verified 300k process running**

Report the run directory, PID, SwanLab URL, current step/loss, and the commands used to monitor it. The later completion phase runs the final 100-episode evaluation from the final checkpoint with `policy.num_inference_steps=100` and no videos.
