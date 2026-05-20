# LIBERO 训练经验总结

日期：2026-05-19

本文总结在 LIBERO 仿真基准上训练 Diffusion Policy 和 IMF-AttnRes 的实践经验，包括超参数选择、常见问题和性能对比。

## 1. Diffusion Policy Baseline

### 1.1 配置

```text
policy.type=diffusion
模型参数量: 89M
vision_backbone: resnet18 (ImageNet pretrained)
resize_shape: [128,128]
crop_ratio: 0.9 (random crop during training)
use_group_norm: false (必须为 false，否则与预训练权重冲突)
down_dims: [256,512,1024]
horizon: 16
n_action_steps: 8
n_obs_steps: 2
noise_scheduler: DDPM, 100 train timesteps
num_inference_steps: 10
optimizer: Adam lr=1e-4, betas=(0.95,0.999), weight_decay=1e-6
scheduler: cosine, warmup=500
batch_size: 64
```

### 1.2 结果

| Suite | Step 10k | Step 20k | Step 30k | 最佳 |
|-------|----------|----------|----------|------|
| Spatial | 69% | 79% | **84%** | 84% (30k) |
| Object | 79% | 87% | **99%** | 99% (30k) |
| Goal | 8% | 6% | 5% | 8% (10k) |

### 1.3 关键发现

- **Spatial 和 Object 收敛快**：20k 步即可达到论文报告水平（78.3%）。
- **Goal 完全失败（~6%）**：因为 LeRobot 的 diffusion policy 没有 language/task conditioning。Goal 的 10 个任务共享相同视觉场景但目标不同，模型无法区分。OpenVLA 论文中的 DP baseline 使用了 DistilBERT language conditioning 才达到 ~68%。
- **use_group_norm=true + pretrained weights 会报错**：不能在预训练 ResNet 上替换 BatchNorm。
- **`diffusers` 包需要额外安装**：`uv sync --extra diffusion`。

## 2. IMF-AttnRes

### 2.1 推荐配置（v6/v7 验证）

```text
policy.type=imf-attnres
模型参数量: 136M
n_layer: 16
n_emb: 768
n_head: 8
n_kv_head: 8
backbone_type: attnres_full
vision_backbone: resnet18 (ImageNet pretrained)
resize_shape: [128,128]
crop_ratio: 0.9
num_inference_steps: 2 (推荐；1步也可用但2步更稳定)
p_drop_emb: 0.05
p_drop_attn: 0.05
optimizer_grad_clip_norm: 5.0 (必须设置，否则 JVP 导致梯度爆炸)
do_mask_loss_for_padding: false
scheduler_type: cosine_decay_with_warmup
scheduler_warmup_steps: 500
scheduler_decay_steps: 与 total steps 相同
batch_size: 64
num_workers: 24
```

### 2.2 结果

| 版本 | Loss 类型 | Spatial 最佳 | Object 最佳 | 备注 |
|------|----------|-------------|-------------|------|
| v3 (uniform t,r) | MSE | 61% (5k) | - | 无 grad clip 时 8k 步爆炸 |
| v4 (logit-normal) | MSE | 72% (30k) | - | 收敛慢，spike 频繁 |
| v6 (logit-normal) | Pseudo-Huber | **79%** (25k) | - | 稳定，无 spike |
| v7 (logit-normal, 2步) | Pseudo-Huber | 76% (5k) | 85% (20k) | 2步推理 |

### 2.3 Loss Spike 问题与解决

**现象**：训练中 ~4% 的 batch 出现 loss 突然升高（正常 0.15 → spike 0.5-8.0）。

**根因**：IMF compound velocity 中 JVP 计算的 `du_dt` 在个别样本上产生极端值（正常 ~50，spike 时 300-800）。

**诊断数据**（v5 实验，8893 个 log 点）：

| 指标 | SPIKE | NON-SPIKE | 倍数 |
|------|-------|-----------|------|
| delta_du_dt_MAX | 30.2 | 7.3 | 4.1x |
| du_dt_norm_MAX | 86.7 | 51.8 | 1.7x |
| delta(t-r) | 0.117 | 0.117 | 1.0x |
| target_norm | 11.85 | 11.86 | 1.0x |
| t / r | 0.47/0.36 | 0.48/0.36 | 1.0x |

**排除的假设**：
- ~~t≈r 导致数值不稳定~~ → delta(t-r) 在 spike/non-spike 间无差异
- ~~数据 outlier~~ → target_norm 完全相同
- ~~t≈1 高方差~~ → t 集中在 0.47 附近
- ~~AttnRes entropy 过低~~ → entropy 稳定在 2.4-2.5

**解决方案**：
1. **Pseudo-Huber loss**（已验证有效）：对大误差的惩罚从二次变为线性，抑制 spike 对梯度的影响。
2. **Grad clip=5.0**（必须）：防止 spike 时的大梯度破坏参数。
3. 可选：对 `du_dt` 做 per-sample norm clipping（未实验）。

## 3. 训练速度与资源

### 3.1 数据加载优化

| num_workers | data_s (5090) | data_s (5880) |
|-------------|---------------|---------------|
| 4 | 0.77s | - |
| 12 | 0.29s | - |
| 24 | 0.20s | 0.50s |

瓶颈是 parquet 图像解码（CPU bound），不是磁盘 IO。

### 3.2 训练速度

| 机器 | GPU | 模型 | 速度 |
|------|-----|------|------|
| 本机 | RTX 5090 (32GB) | Diffusion 89M | ~6 step/s (24 workers) |
| 本机 | RTX 5090 (32GB) | IMF-AttnRes 136M | ~4 step/s (24 workers) |
| 5880 | RTX 5880 (48GB) | IMF-AttnRes 136M | ~3.5 step/s (24 workers) |

### 3.3 Eval 注意事项

- **`use_async_envs=false` 会触发 LIBERO seed bug**：`'OffScreenRenderEnv' object has no attribute 'env'`。必须用 `use_async_envs=true`。
- **Eval 时内存压力**：24 个 persistent dataloader workers + async env workers 可能导致 OOM。本机 62GB RAM 时 `eval.batch_size` 不超过 2。5880 (128GB RAM) 可以用 10。
- **Eval 耗时**：10 tasks × 10 episodes × 280 steps，约 3-5 分钟（async），15-20 分钟（sync）。
- **训练中 eval 会暂停训练**：eval 占用 GPU，训练进程等待。

## 4. LIBERO Suite 特性

| Suite | Episodes | Episode Length | 任务区分方式 | 需要 Language Conditioning |
|-------|----------|---------------|-------------|--------------------------|
| Spatial | 432 | 280 | 物体空间位置不同 | 否 |
| Object | 454 | 280 | 操作对象不同 | 否 |
| Goal | 428 | 300 | 目标状态不同（场景相同） | **是** |
| Long | 379 | 520 | 长序列多步任务 | 否 |

## 5. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `ImportError: 'diffusers'` | 未安装 diffusion 依赖 | `uv sync --extra diffusion` |
| `BatchNorm + pretrained weights` | use_group_norm=true 与预训练冲突 | 设为 false |
| `OffScreenRenderEnv has no attribute 'env'` | LIBERO sync env seed bug | 用 `use_async_envs=true` |
| Eval OOM | 训练 workers + eval envs 超出 RAM | 减小 `eval.batch_size` |
| W&B init timeout | 网络问题 | 重试或 `wandb.mode=offline` |
| Loss spike (IMF) | JVP du_dt 极端值 | Pseudo-Huber loss + grad_clip=5 |
| Goal suite ~0% | 无 task conditioning | 需加 language embedding |
