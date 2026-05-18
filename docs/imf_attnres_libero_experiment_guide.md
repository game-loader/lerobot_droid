# IMF-AttnRes LIBERO 实验启动与评估指南

日期：2026-05-18

本文说明当前 LeRobot worktree 中是否包含 LIBERO 实验虚拟环境，以及如何启动 IMF-AttnRes 的 LIBERO 训练和仿真 eval。

## 1. 正式 worktree 位置

当前正式 LeRobot worktree：

```text
/data/lerobot-imf-attnres-exp/lerobot-imf-attnres
```

对应 git 分支：

```text
feat/imf-attnres-policy
```

确认：

```bash
cd /data/lerobot-imf-attnres-exp/lerobot-imf-attnres
git branch --show-current
git worktree list
```

## 2. 是否有 LIBERO 实验虚拟环境？

有。当前实验使用仓库内的本地虚拟环境：

```text
/data/lerobot-imf-attnres-exp/lerobot-imf-attnres/.venv
```

已验证环境：

```text
Python 3.12.13
lerobot import path: /data/lerobot-imf-attnres-exp/lerobot-imf-attnres/src/lerobot
```

检查命令：

```bash
cd /data/lerobot-imf-attnres-exp/lerobot-imf-attnres
.venv/bin/python - <<'PY'
import sys, torch, lerobot
print(sys.executable)
print(torch.__version__)
print('cuda:', torch.cuda.is_available())
print(lerobot.__file__)
PY
```

说明：

- `.venv` 是本地实验环境，不提交到 git。
- 训练/eval 可直接用 `.venv/bin/python scripts/...` 启动 wrapper。
- 也可以继续用 `uv run --extra ... lerobot-train/lerobot-eval`。
- 如果 `.venv` 缺失，在 worktree 下重建：

```bash
cd /data/lerobot-imf-attnres-exp/lerobot-imf-attnres
uv sync --extra training --extra libero --extra evaluation
```

## 3. 常用环境变量

训练或 eval 前建议：

```bash
cd /data/lerobot-imf-attnres-exp/lerobot-imf-attnres

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

# W&B key 只在 shell 里导出，不写进脚本或文档。
export WANDB_API_KEY=<your-wandb-api-key>
export WANDB_DIR=$EXP_ROOT/wandb
export WANDB_CACHE_DIR=$EXP_ROOT/cache/wandb
export WANDB_CONFIG_DIR=$EXP_ROOT/cache/wandb-config
export WANDB_SILENT=True
```

## 4. LIBERO 数据位置和 suite 映射

源数据：

```text
/data/lerobot_datasets/HuggingFaceVLA/libero
```

wrapper 会生成/复用本地 remapped 子集：

```text
/data/lerobot-imf-attnres-exp/datasets/libero_spatial
/data/lerobot-imf-attnres-exp/datasets/libero_object
/data/lerobot-imf-attnres-exp/datasets/libero_goal
/data/lerobot-imf-attnres-exp/datasets/libero_long
```

| wrapper suite | LIBERO env.task | episode length | episodes |
|---|---|---:|---:|
| `spatial` | `libero_spatial` | 280 | 432 |
| `object` | `libero_object` | 280 | 454 |
| `goal` | `libero_goal` | 300 | 428 |
| `long` | `libero_10` | 520 | 379 |

主要 features：

```text
observation.images.image   [256, 256, 3]
observation.images.image2  [256, 256, 3]
observation.state          [8]
action                     [7]
```

## 5. 启动 IMF-AttnRes LIBERO 训练

实验 wrapper：

```text
scripts/imf_attnres_experiments/train_eval_suite.py
```

它会：

1. 准备 suite 子集；
2. 调用 `lerobot-train` 训练 `policy.type=imf-attnres`；
3. 调用 `lerobot-eval` 评估 checkpoint；
4. 用 rollout10 选择最佳 checkpoint；
5. 对最佳 checkpoint 做 final rollout50。

### 5.1 单进程训练 + 训练后 eval

```bash
cd /data/lerobot-imf-attnres-exp/lerobot-imf-attnres

.venv/bin/python scripts/imf_attnres_experiments/train_eval_suite.py \
  --suite=spatial \
  --gpu=0 \
  --steps=20000 \
  --save-freq=5000 \
  --eval-every=5000 \
  --eval-episodes=10 \
  --final-eval-episodes=50 \
  --eval-batch-size=10 \
  --batch-size=32 \
  --num-workers=4 \
  --policy-n-obs-steps=2 \
  --policy-horizon=16 \
  --policy-n-action-steps=8 \
  --policy-n-emb=384 \
  --policy-n-layer=12 \
  --policy-optimizer-lr=1e-4 \
  --policy-optimizer-weight-decay=1e-6 \
  --policy-optimizer-grad-clip-norm=2.5 \
  --policy-scheduler-type=cosine_decay_with_warmup \
  --policy-scheduler-warmup-steps=1000 \
  --policy-scheduler-decay-steps=20000 \
  --policy-scheduler-decay-lr=2.5e-6 \
  --run-suffix=debug-d384-l12 \
  --wandb-mode=online \
  --overwrite
```

wrapper 内训练命令会设置：

```text
--policy.type=imf-attnres
--wandb.disable_artifact=true
--policy.push_to_hub=false
```

### 5.2 训练中并行 eval checkpoint

推荐两个进程：训练进程 + watch/eval 进程。

训练进程：

```bash
cd /data/lerobot-imf-attnres-exp/lerobot-imf-attnres

.venv/bin/python scripts/imf_attnres_experiments/train_eval_suite.py \
  --suite=object \
  --gpu=0 \
  --steps=100000 \
  --save-freq=10000 \
  --eval-every=10000 \
  --eval-episodes=10 \
  --final-eval-episodes=50 \
  --eval-batch-size=10 \
  --batch-size=24 \
  --num-workers=4 \
  --policy-n-obs-steps=2 \
  --policy-horizon=16 \
  --policy-n-action-steps=8 \
  --policy-n-emb=384 \
  --policy-n-layer=12 \
  --policy-optimizer-lr=1e-4 \
  --policy-optimizer-weight-decay=1e-6 \
  --policy-optimizer-grad-clip-norm=2.5 \
  --policy-scheduler-type=cosine_decay_with_warmup \
  --policy-scheduler-warmup-steps=2000 \
  --policy-scheduler-decay-steps=100000 \
  --policy-scheduler-decay-lr=2.5e-6 \
  --run-suffix=obj-d384-l12-lr1e-4 \
  --wandb-mode=online \
  --train-only \
  --overwrite
```

watch/eval 进程需使用同样的 `suite`、`steps`、`eval-every`、`run-suffix` 和关键超参，并加 `--skip-train --watch-eval`：

```bash
cd /data/lerobot-imf-attnres-exp/lerobot-imf-attnres

.venv/bin/python scripts/imf_attnres_experiments/train_eval_suite.py \
  --suite=object \
  --gpu=0 \
  --steps=100000 \
  --save-freq=10000 \
  --eval-every=10000 \
  --eval-episodes=10 \
  --final-eval-episodes=50 \
  --eval-batch-size=10 \
  --batch-size=24 \
  --num-workers=4 \
  --policy-n-obs-steps=2 \
  --policy-horizon=16 \
  --policy-n-action-steps=8 \
  --policy-n-emb=384 \
  --policy-n-layer=12 \
  --policy-optimizer-lr=1e-4 \
  --policy-optimizer-weight-decay=1e-6 \
  --policy-optimizer-grad-clip-norm=2.5 \
  --policy-scheduler-type=cosine_decay_with_warmup \
  --policy-scheduler-warmup-steps=2000 \
  --policy-scheduler-decay-steps=100000 \
  --policy-scheduler-decay-lr=2.5e-6 \
  --run-suffix=obj-d384-l12-lr1e-4 \
  --wandb-mode=disabled \
  --skip-train \
  --watch-eval \
  --poll-s=90
```

注意：watch/eval 进程可能覆盖 run 目录里的 `train_cmd.json`。排查真实训练命令优先看：

1. `train.log` 开头的 command；
2. 实际 `ps` 进程；
3. checkpoint 内保存的 config；
4. 最后再看 `train_cmd.json`。

## 6. 单独启动 LIBERO eval

已有 checkpoint 时可直接使用 `lerobot-eval`。

示例：评估 spatial suite rollout10：

```bash
cd /data/lerobot-imf-attnres-exp/lerobot-imf-attnres

CKPT=/data/lerobot-imf-attnres-exp/outputs/train/<run-name>/checkpoints/010000/pretrained_model
OUT=/data/lerobot-imf-attnres-exp/outputs/eval/manual-$(date +%Y%m%d-%H%M%S)

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
  --eval.batch_size=10 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=true \
  --output_dir=$OUT \
  --seed=1000
```

四个标准 suite：

```text
libero_spatial  episode_length=280
libero_object   episode_length=280
libero_goal     episode_length=300
libero_10       episode_length=520
```

只评估某个 task_id，例如 `libero_goal` 的 task 8：

```bash
uv run --extra libero --extra evaluation lerobot-eval \
  --policy.path=$CKPT \
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
  --eval.batch_size=10 \
  --eval.n_episodes=50 \
  --eval.use_async_envs=true \
  --output_dir=$OUT \
  --seed=1000
```

## 7. Eval 指标与 init state

LIBERO success rate 来自环境 `check_success()`：

```text
pc_success = mean(episode_success) * 100
```

单个 episode 内只要任意时刻 success 为 true，该 episode 就算成功。

在多 task eval 中：

```text
--eval.n_episodes=10
```

表示每个 task 10 次 rollout。

`--env.init_states=true` 时使用 LIBERO 预生成 init states，不是完全随机初始化：

- `eval.batch_size=1, eval.n_episodes=50` 顺序覆盖 50 个 init states；
- `eval.batch_size=10, eval.n_episodes=50` 也覆盖 50 个 init states，只是并行跑。

## 8. 监控和结果读取

查看训练日志：

```bash
RUN=/data/lerobot-imf-attnres-exp/runs/<run-name>
tail -f $RUN/train.log
```

查看 checkpoint eval：

```bash
cat $RUN/eval_results.json
cat $RUN/best_checkpoint.json
cat $RUN/final_result.json
```

查看 eval overall 指标：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('/data/lerobot-imf-attnres-exp/outputs/eval/<run-name>/step_010000/eval_info.json')
info = json.loads(p.read_text())
print(info['overall'])
PY
```

查看视频：

```bash
find /data/lerobot-imf-attnres-exp/outputs/eval -type f -name '*.mp4' | tail -20
```

查看本机 GPU/进程：

```bash
nvidia-smi
pgrep -af 'lerobot-train|lerobot-eval|train_eval_suite|adaptive_experiment_manager'
```

远端 5880 节点：

```bash
ssh -F /dev/null -o StrictHostKeyChecking=accept-new droid@100.73.14.65 \
  'nvidia-smi; pgrep -af "lerobot-train|lerobot-eval|train_eval_suite" || true'
```

## 9. 注意事项

- 不要把 W&B API key 写进脚本、日志或文档。
- checkpoint 不上传 W&B artifact：保持 `--wandb.disable_artifact=true`。
- 当前 policy 已注册为 `policy.type=imf-attnres`。
- 当前 suite 子集的 stats 是从 aggregate LIBERO stats 复制来的，不是 per-suite 重新计算。
- 做论文/表格实验时记录：branch、commit、run command、checkpoint path、eval seed、eval.n_episodes、eval.batch_size、env.task、env.init_states。
