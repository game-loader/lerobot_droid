# IMF-AttnRes 多机实验启动指南

## 机器概览

| 机器 | GPU | 内存 | IP | 用途 |
|------|-----|------|-----|------|
| 本机 (5090) | 1× RTX 5090 32GB | 64GB | localhost | 单卡实验 |
| 5880 | 2× RTX Ada 6000 49GB | 128GB | 100.73.14.65 | 双卡实验 |
| L20 | 8× L20 46GB | 128GB | 100.119.99.14 | 多组双卡并行实验 |

## 前置准备

### 代码同步（排除 .libero_config 避免覆盖远程路径配置）

```bash
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  --exclude='wandb' --exclude='outputs' --exclude='.libero_config' --exclude='.claude' \
  /data/lerobot-imf-attnres-exp/lerobot-imf-attnres/ \
  droid@<IP>:/data/lerobot-imf-attnres-exp/lerobot-imf-attnres/
```

### 远程 .libero_config 配置（每台机器只需设置一次）

```bash
ssh droid@<IP> 'cat > /data/lerobot-imf-attnres-exp/lerobot-imf-attnres/.libero_config/config.yaml << EOF
bddl_files: /data/lerobot-imf-attnres-exp/lerobot-imf-attnres/.venv/lib/python3.12/site-packages/libero/libero/bddl_files
init_states: /data/lerobot-imf-attnres-exp/lerobot-imf-attnres/.venv/lib/python3.12/site-packages/libero/libero/init_files
datasets: /data/lerobot-imf-attnres-exp/datasets
assets: /data/lerobot-imf-attnres-exp/lerobot-imf-attnres/.venv/lib/python3.12/site-packages/libero/libero/assets
EOF'
```

### 依赖安装（远程机器）

```bash
ssh droid@<IP> '/home/droid/.local/bin/uv pip install swanlab num2words \
  --python /data/lerobot-imf-attnres-exp/lerobot-imf-attnres/.venv/bin/python'
```

## 数据集路径

| 数据集 | 本机路径 | 远程路径 |
|--------|----------|----------|
| libero_combined (40 tasks, 1693 eps) | `/data/lerobot_datasets/HuggingFaceVLA/libero` | `/data/lerobot-imf-attnres-exp/datasets/libero_combined` |
| libero_object | `/data/lerobot-imf-attnres-exp/datasets/libero_object` | 同左 |
| libero_spatial | `/data/lerobot-imf-attnres-exp/datasets/libero_spatial` | 同左 |
| libero_goal | `/data/lerobot-imf-attnres-exp/datasets/libero_goal` | 同左 |
| libero_long | `/data/lerobot-imf-attnres-exp/datasets/libero_long` | 同左 |

## 启动脚本位置

```
runs/
├── ablation-local-swanlab/    # 本机单卡实验
├── combined-txenc/            # 本机 combined 数据集 200k 实验
├── 5880-combined-identity/    # 5880 双卡 identity latent mode
├── 5880-imf-txenc/            # 5880 双卡 spatial 单任务
├── l20-combined-dct/          # L20 GPU 0,1 attnres_full + dct
├── l20-combined-vanilla/      # L20 GPU 2,3 vanilla backbone
└── l20-smolvla/               # L20 GPU 4,5 SmolVLA 官方模型
```

## 启动实验

### 本机（单卡）

```bash
nohup /data/lerobot-imf-attnres-exp/runs/<script_dir>/launch.sh \
  > /data/lerobot-imf-attnres-exp/runs/<script_dir>/train.log 2>&1 &
```

### 远程（双卡 accelerate）

```bash
ssh droid@<IP> 'rm -rf /data/lerobot-imf-attnres-exp/outputs/train/<output_dir>; \
  nohup bash /data/lerobot-imf-attnres-exp/runs/<script_dir>/launch.sh \
  > /data/lerobot-imf-attnres-exp/runs/<script_dir>/train.log 2>&1 & echo "PID: $!"'
```

### L20 多组并行（注意 GPU 分配和端口）

- GPU 0,1 端口 29500（默认）
- GPU 2,3 端口 29501
- GPU 4,5 端口 29502
- GPU 6,7 端口 29503

## 关键参数说明

| 参数 | 说明 |
|------|------|
| `TORCH_NCCL_ENABLE_MONITORING=0` | 禁用 NCCL watchdog，防止 eval 时超时 abort |
| `--eval.batch_size=10` | 并行 eval 环境数，增大可加速 eval |
| `--persistent_workers=false` | 避免 worker 长期占用内存 |
| `--env.task=libero_spatial,libero_object,libero_goal,libero_10` | 多 suite eval（逗号分隔） |
| `--policy.vlm_image_forward_batch_size=0` | 不分批过 ViT（L20 46GB 够用） |

注意：`libero_long` 的 env.task 名称是 `libero_10`，不是 `libero_long`。

## 常见问题

### NCCL 超时导致 SIGABRT
多卡训练时 eval 只在 rank 0 运行，rank 1 空等超时。解决：设置 `TORCH_NCCL_ENABLE_MONITORING=0`。

### 孤儿 forkserver 进程占满内存
主进程被 kill 后 dataloader workers 不会自动退出。清理：
```bash
ssh droid@<IP> 'ps aux | grep "forkserver\|lerobot-train\|accelerate" | grep -v grep | awk "{print \$2}" | xargs kill -9'
```

### FileExistsError 输出目录已存在
启动前清理：`rm -rf /data/lerobot-imf-attnres-exp/outputs/train/<output_dir>`

### SmolVLA 加载失败（HF_HUB_OFFLINE）
SmolVLA 需要从 Hub 下载 image processor，不能设 `HF_HUB_OFFLINE=1`。

### .libero_config 路径错误
rsync 同步代码时会覆盖远程的 `.libero_config`。同步时加 `--exclude='.libero_config'`。
