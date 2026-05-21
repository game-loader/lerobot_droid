# IMF-AttnRes + SmolVLM 消融实验

日期：2026-05-21

## 1. 实验目标

在 LIBERO spatial/object 上对 IMF-AttnRes + SmolVLM2-500M（frozen）进行消融实验，测试：
- **backbone_type**: `attnres_full`, `attnres_diff`, `diff_transformer`
- **action_latent_mode**: `dct`, `identity`
- **semigroup_consistency**: enabled (from step 1000), disabled

共 3×2×2×2 = 24 个实验配置。

## 2. 三台机器分工

| 机器 | GPU | 负责实验 | 训练方式 |
|------|-----|----------|----------|
| 本机 (droid-5090) | 1× RTX 5090 32GB | diff_transformer 2×LR 特殊实验 | 单卡 |
| 5880 (droid@100.73.14.65) | 2× RTX Ada 49GB | attnres_full + attnres_diff (12个) | 双卡 accelerate 顺序执行 |
| L20 (droid@100.119.99.14) | 8× L20 46GB | attnres_diff-object + diff_transformer (12个) | 双卡 accelerate, 4并发 |

## 3. 目录结构

三台机器共享相同的项目路径：

```
/data/lerobot-imf-attnres-exp/
├── lerobot-imf-attnres/              # 项目代码
│   ├── .venv/
│   └── .libero_config/config.yaml    # 各机器路径不同，指向 venv 内的 libero 资源
├── datasets/
│   ├── libero_spatial/               # ~6.2GB, 432 episodes
│   └── libero_object/                # ~8.7GB, 454 episodes
├── outputs/train/                    # 训练输出 (按 job_name 命名)
│   ├── imf-attnres_full-spatial-dct-nosemi/
│   ├── imf-attnres_full-spatial-dct-semi/
│   └── ...
├── runs/
│   ├── ablation-5880/launch_all.sh   # 5880 全部实验脚本
│   ├── ablation-l20/                 # L20 实验脚本
│   │   ├── common.sh                # 共享环境变量和参数
│   │   ├── wave1.sh                 # attnres_diff on object (4并发)
│   │   ├── wave2.sh                 # diff_transformer on spatial (4并发)
│   │   ├── wave3.sh                 # diff_transformer on object (4并发)
│   │   └── launch_all.sh           # 顺序执行 wave1→wave2→wave3
│   └── ablation-diff_trans-2xlr/     # 本机特殊实验
│       └── launch.sh
└── cache/huggingface/                # L20 专用 HF 缓存 (根分区空间不足)
```

## 4. .libero_config/config.yaml

各机器的 config.yaml 指向 venv 内安装的 libero 包资源：

```yaml
bddl_files: /data/lerobot-imf-attnres-exp/lerobot-imf-attnres/.venv/lib/python3.12/site-packages/libero/libero/bddl_files
init_states: /data/lerobot-imf-attnres-exp/lerobot-imf-attnres/.venv/lib/python3.12/site-packages/libero/libero/init_files
datasets: /data/lerobot-imf-attnres-exp/datasets
assets: /data/lerobot-imf-attnres-exp/lerobot-imf-attnres/.venv/lib/python3.12/site-packages/libero/libero/assets
```

本机例外（有独立 LIBERO 安装）：
```yaml
bddl_files: /home/droid/project/LIBERO/libero/libero/bddl_files
init_states: /home/droid/project/LIBERO/libero/libero/init_files
datasets: /data/lerobot-imf-attnres-exp/datasets
assets: /home/droid/project/LIBERO/libero/libero/assets
```

## 5. 共享训练参数

```text
policy.type=imf-attnres
n_layer=16, n_emb=768, n_head=8, n_kv_head=8
use_smolvlm_vl_encoder=true
vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
load_vlm_weights=true, freeze_vlm_encoder=true
vlm_resize_shape=[512,512]
num_inference_steps=2
do_mask_loss_for_padding=false
p_drop_emb=0.05, p_drop_attn=0.05
optimizer_lr=1e-4, grad_clip_norm=5.0
scheduler: cosine_decay_with_warmup, warmup=500, decay_steps=50000
batch_size=4 (nosemi) / 2 (semi, 因 semigroup 额外前向传播导致 OOM)
eval_freq=5000, eval.n_episodes=10, eval.batch_size=2
steps=50000
wandb.project=lerobot-imf-attnres-libero-ablation
```

Semigroup 参数（启用时）：
```text
semigroup_loss_weight=0.05
semigroup_start_step=1000
semigroup_warmup_steps=2000
```

## 6. 环境变量

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export LIBERO_CONFIG_PATH=/data/lerobot-imf-attnres-exp/lerobot-imf-attnres/.libero_config
export PYTHONUNBUFFERED=1
# L20 专用（根分区空间不足）：
export HF_HOME=/data/lerobot-imf-attnres-exp/cache/huggingface
export HF_DATASETS_CACHE=/data/lerobot-imf-attnres-exp/cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 7. 启动方式

**5880（双卡 accelerate，顺序执行 12 个实验）：**
```bash
ssh droid@100.73.14.65
nohup bash /data/lerobot-imf-attnres-exp/runs/ablation-5880/launch_all.sh \
  > /data/lerobot-imf-attnres-exp/runs/ablation-5880/master.log 2>&1 &
```

**L20（4 并发双卡实验，3 波顺序）：**
```bash
ssh droid@100.119.99.14
nohup bash /data/lerobot-imf-attnres-exp/runs/ablation-l20/launch_all.sh \
  > /data/lerobot-imf-attnres-exp/runs/ablation-l20/master.log 2>&1 &
```

**本机（单卡）：**
```bash
nohup bash /data/lerobot-imf-attnres-exp/runs/ablation-diff_trans-2xlr/launch.sh \
  > /data/lerobot-imf-attnres-exp/runs/ablation-diff_trans-2xlr/train.log 2>&1 &
```

## 8. 代码同步

项目不通过 git 远程仓库同步，直接 rsync：
```bash
# 同步代码（排除 .venv/outputs/wandb）
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  --exclude='wandb' --exclude='outputs' \
  /data/lerobot-imf-attnres-exp/lerobot-imf-attnres/ \
  droid@TARGET:/data/lerobot-imf-attnres-exp/lerobot-imf-attnres/

# 同步数据集
rsync -avz /data/lerobot-imf-attnres-exp/datasets/libero_spatial/ \
  droid@TARGET:/data/lerobot-imf-attnres-exp/datasets/libero_spatial/

# 同步脚本
rsync -avz /data/lerobot-imf-attnres-exp/runs/ablation-l20/ \
  droid@100.119.99.14:/data/lerobot-imf-attnres-exp/runs/ablation-l20/
```

## 9. 命名规则

`imf-{backbone}-{suite}-{mode}-{semi}`

- backbone: `attnres_full`, `attnres_diff`, `diff_trans`（diff_transformer 缩写）
- suite: `spatial`, `object`
- mode: `dct`, `identity`
- semi: `semi`, `nosemi`

## 10. 已知问题与解决

| 问题 | 原因 | 解决 |
|------|------|------|
| Semi 实验 OOM | semigroup 需额外 3 次 flow_map 前向传播 | batch_size 从 4 降到 2 |
| L20 根分区满 | HF datasets cache 写入 ~/.cache (在 /) | 设置 HF_HOME 到 /data |
| L20 SmolVLM 下载失败 | 4 进程并发访问 HF Hub | 设置 HF_HUB_OFFLINE=1 |
| 5880 FileExistsError | 输出目录已存在 | 脚本中 rm -rf 旧目录或跳过已完成 |
| L20 端口冲突 | 4 个 accelerate 实验用同一端口 | 每个实验指定不同 --main_process_port |
| 训练速度 L20 较慢 | 4 并发实验竞争 CPU/内存带宽 | ~2.2 step/s vs 单卡 ~3 step/s |

## 11. 训练速度参考

| 机器 | 配置 | 速度 |
|------|------|------|
| 本机 5090 | 单卡, batch_size=4, SmolVLM | ~9 step/s |
| 5880 双卡 | accelerate 2-GPU, batch_size=4 | ~3.1 step/s |
| L20 双卡×4 | 4 并发 accelerate, batch_size=4 | ~2.2 step/s |
