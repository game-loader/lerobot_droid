# FastWAM 的 DINOv3 Q-Planning 后训练

独立放在 `RL/fastwam_q`，不改 FastWAM BC，也不接入原来的 PPO / AM-Q actor 更新。
参考 Q-Planning 官方代码的 chunk-TD、动作查询 Transformer、HL-Gauss 头、EMA target、
1:1 demo/online replay，以及本仓库 FastWAM 的 video-KV cache 推理接口。来源见 `NOTICE`。

## 实现

- **视觉**：真实的 DINOv3 ViT-L/16；默认参与训练。224×224，每相机 196 个 patch tokens，
  丢弃 CLS 与 4 个 register tokens；保留独立的 patch/camera embedding 给 Q decoder 使用。
- **文本**：冻结的 `google/t5-v1_1-base` encoder，按任务缓存；注意力正确屏蔽 padding。
- **Q decoder**：18 层、1024 hidden、16 heads、FFN 4096；action chunk 逐步投影为 query，
  self-attention + 图文 cross-attention，CLS 汇聚后 MLP 输出 101 个价值 bins。
- **训练**：FP32 参数/AdamW + BF16 autocast，默认 DINO 与 decoder activation checkpointing。
  Q 的目标网络冻结并保持 eval，优化后做 EMA；T5 不进入优化器。
  Decoder 训练 attention 使用 math SDPA 保持 FP32 中间量；推理不强制后端。
  GB200 上的逐层失稳定位与修复证据见 [DIAGNOSIS_20260930.md](DIAGNOSIS_20260930.md)。
- **TD**：`sum(gamma**k * r[t+k]) + gamma**H * Q_target(s[t+H], recorded_next_chunk)`。
  终止/超时无 bootstrap；末尾不足 H 步的 chunk 保留终止奖励并 padding/mask，不丢掉。
- **在线数据**：每个 microbatch 一半原始示范、一半累计 online rollout；没有 online 时全 demo。
  抽样起点可逐帧滑动（默认 `stride=1`），不是每次必须按 0、32、64 切不重叠块。
- **规划**：一次 FastWAM video prefill，独立噪声采样 N 个候选；只运行 3 步 action flow。
  Q 的图文编码也只算一次。`candidate_batch_size` 限制动作分支/critic 分批评估的显存。
  默认对全部候选 softmax(Q / temperature) 加权，`n_elites>0` 可选官方代码的 top-k 变体。

DINOv3 替换属于本实现的改动，尚无性能复现结论；不沿用论文的成功率作为本实现结果。

## 安装与离线 Q 训练

```bash
uv sync --locked --extra fastwam-q

uv run --no-sync python -m RL.cli.train_fastwam_q \
  --fastwam-checkpoint /path/to/fastwam/pretrained_model \
  --repo-id your-name/demonstrations \
  --root /path/to/lerobot_dataset \
  --output outputs/fastwam_q \
  --steps 12000 --batch-size 8 --accumulation 4
```

`batch-size` 是 microbatch；上例有效 batch=32。训练只读取 FastWAM 的配置和处理器统计，
**不加载 FastWAM 的 Wan/动作模型**。图像使用独立 DINO 预处理；action 使用 BC checkpoint
保存的 normalizer，不能改用 rollout 的均值方差。

默认模型 `facebook/dinov3-vitl16-pretrain-lvd1689m` 需要已获得的 Meta/HF 权重访问权限。
可通过 `--config config.json` 指向已有本地 DINOv3 **Transformers 格式**目录；不会下载或
暗中替换成其他视觉模型：

```json
{
  "dino_model": "/path/to/dinov3-vitl16",
  "freeze_dino": false,
  "gradient_checkpointing": true,
  "camera_keys": ["observation.images.wrist_left", "observation.images.wrist_right"],
  "execution_steps": 24
}
```

H 和 action_dim 默认从 BC config 读取；不是写死只支持 7 维或 32 步。
`workers=0` 默认保守避免视频解码多进程问题，可自行用 `--workers 4`（spawn）提高数据吞吐。
`metrics.jsonl` 记录 loss、Q、梯度范数和 CUDA 峰值。每隔 1000 更新及结束保存 checkpoint。

```bash
# 保留 optimizer、target Q、normalizer；继续执行 200 次 Q 更新
uv run --no-sync python -m RL.cli.train_fastwam_q \
  --resume outputs/fastwam_q/last \
  --repo-id your-name/demonstrations --root /path/to/demos \
  --online-repo-id local/rollout_round1 --online-root /path/to/rollout_round1 \
  --output outputs/fastwam_q_round1 --steps 200 --batch-size 8
```

可重复提供 `--online-repo-id / --online-root` 加入多轮数据。在线数据必须含每步 `next.reward`
及 episode 边界；`next.done`/`next.truncated` 若存在会使用。也可用
`--online-labels labels.json` 提供 `{"0": true, "1": false, ...}`，把奖励放在该 episode
最后一步。纯 demo 无 reward 列时按末尾成功 +1 处理；online 不会默认全成功。

## 用于 FastWAM 部署

```python
from RL.fastwam_q import FastWAMQPlanner

planner, postprocessor = FastWAMQPlanner.from_checkpoints(
    "/path/to/fastwam/pretrained_model", "outputs/fastwam_q/last", device="cuda"
)
planner.reset()  # 每个 episode 开始
# obs: 原始、已带 batch 维度的 LeRobot 图像/state/task 字典；float RGB 在 [0,1]
normalized_action = planner.select_action(obs)
env_action = postprocessor(normalized_action)  # 只在最后执行一次原 BC 后处理
```

默认取 chunk 前 10 步，然后重新规划；可通过 Q config 更改为 24/32 等。
推理只加载 online Q，不加载 target Q/optimizer。候选均在 BC normalized action 空间打分。
如果原 postprocessor 有 gripper toggle，应在发送到环境时执行，不得在 Q 加权前执行。

## Python 在线循环

```python
from RL.fastwam_q import FastWAMQReplay, FastWAMQTrainer
from RL.fastwam_q.checkpoint import load_checkpoint

q, normalizer = load_checkpoint("outputs/fastwam_q/last")
trainer = FastWAMQTrainer(q, normalizer, device="cuda")
replay = FastWAMQReplay(q.config)
# demos 是 LeRobotChunkDataset(..., demonstrations=True, config=q.config)
# collect(planner, iteration) 由你的环境提供，返回 episode 字典；不自动启动机器人。
for metrics in trainer.self_improve(
    planner, demos, replay, collect, iterations=5, updates=200, output="outputs/self_improve"
):
    print(metrics)
```

`collect` 返回的 episode 字典：

```python
{
    "images": images,    # 每个实际执行步骤的当前多相机图像，uint8 或 [0,1] float
    "actions": actions,  # 实际执行的动作，demo 的物理动作坐标系，尚未 normalizer
    "rewards": rewards,  # 每步实际任务奖励，失败 episode 也要保留
    "done": done,        # 可省略，末尾自动作为终止
    "task": "stack cups"
}
```

只存真正执行的前缀，再拼接后续重规划动作；不要把没有执行的 32 步计划后缀当 rollout。
若环境对 gripper 做了额外 toggle，存入 replay 的动作要转换回 demo 约定（例如标准
`sign(-(2*x-1))` 对二值 gripper 的逆为 `(1-env_action)/2`）。成功判定、场景复位和硬件连接
属于环境，不在这里伪造自动化。`self_improve` 会让 planner 使用 trainer 的同一个 online Q。
可恢复保存的 `replay.pt`，逐条 `replay.add_episode(...)`；数据来自自己生成的 checkpoint。

**训练与 rollout 同卡共存时还需要 FastWAM 本身的显存**，下面的 Q-only 数值不包含它。
显存不够可分阶段运行上述 train CLI 和部署进程，或将两者放在不同 GPU。

## 显存：当前代码的实测参考

真实网络结构、**随机权重与随机输入**、RTX 5090、PyTorch 2.11.0+cu128；双相机 224×224、
H=32、18 层 Q decoder、冻结 T5、DINO 全量训练、FP32 权重/AdamW + BF16 autocast。
每个设置执行两次优化，包含 Adam 状态初始化。不是预训练权重上的正式数据训练结果。

| microbatch | activation checkpointing | PyTorch peak allocated | peak reserved |
|---:|:---:|---:|---:|
| 1 | 开 | 11.79 GiB | 11.87 GiB |
| 4 | 开 | 11.87 GiB | 12.12 GiB |
| 8 | 开 | 11.98 GiB | 12.44 GiB |
| 1 | 关 | 11.80 GiB | 12.40 GiB |
| 4 | 关 | 12.99 GiB | 13.56 GiB |
| 8 | 关 | 14.89 GiB | 15.67 GiB |

计数：online Q 607,857,765 参数，其中 DINO 303,129,600、decoder/适配器 304,728,165；
冻结 T5 encoder 109,628,544。online + target + T5 权重约 4.94 GiB，梯度约 2.26 GiB，
Adam moments 约 4.53 GiB。静态训练状态共 **11.73 GiB**，无需假设论文约数“1B”。

单纯 Q 训练建议预留 **16–20 GiB** 工作空间；24 GB 卡更宽裕，32 GB 卡适合继续增大 batch/
相机数。16 GB 卡用 checkpointing、小 microbatch + accumulation，并注意同卡其他进程。
上述为两步实测，不保证长跑峰值；分辨率、镜头数、kernel、外部进程和数据预取都会影响。

```bash
uv run --no-sync python -m RL.fastwam_q.memory_estimate
# 可选短小随机网络显存实测，不下载权重，不启动 FastWAM/机器人
uv run --no-sync python -m RL.fastwam_q.memory_estimate --profile --batches 1 4 8
uv run --no-sync python -m RL.fastwam_q.memory_estimate --profile --batches 8 --no-checkpointing
```

## 魔改 FR3 bundle + 已有 RGB cache

`--fastwam-bundle` 直接读取 `config.yaml` 与 `dataset_stats.json`，不加载 BC/Wan VAE。
动作使用 bundle 的 `action.default.global_mean/global_std`，统一 FP32 z-score、epsilon=1e-8、
clip[-5,5]，统计随 Q checkpoint 保存。`--cache-root` 读取已有
`zlib1_native_rgb_uint8` 缓存的数值数组和 RGB 帧，不重建缓存、不解码视频。
当前缓存入口用于示范数据：保留所有 episode，每个 episode 末尾成功奖励 +1，其余为 0；
随机抽取可重叠的 32 步 chunk，不跨 episode。没有失败 rollout，不能把离线 TD loss 当作机器人成功率。

```bash
uv run --no-sync python -m RL.cli.train_fastwam_q \
  --fastwam-bundle /path/to/fr3_bundle \
  --cache-root /path/to/franka4_lossless \
  --repo-id local/franka4_merged --root /path/to/franka4_merged \
  --config /path/to/q_config.json --output outputs/fr3_q \
  --steps 45000 --save-freq 5000 --batch-size 32 --workers 8 \
  --swanlab-project fastwam-dinov3-q --run-name fr3-fourtask-dinov3-vitl-b32-45k \
  --credential-file /path/to/existing/swanlab.key
```

DINOv3 必须使用已授权的真实预训练权重；官方仓库拒绝访问时不进行随机初始化训练或模型替换。
`prepare_models.py` 为指定的 DINOv3 ViT-L 和 T5 下载固定 revision，失败记录明确的零步阻塞状态。
GB200 启动脚本位于 `scripts/fastwam_q/run_fr3_cached.sh`，与已有其他训练的代码、环境隔离，
仅选择 GPU0；不对共享环境安装或卸载包。SwanLab 使用 online/private，记录 TD loss、Q、
梯度范数、学习率、耗时、显存和 checkpoint step。

### ModelScope 原生 DINOv3 权重

已验证用户指定的 `ciqiangxu/DINOv3` 中
`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` 可作为同型号预训练来源。
完整 SHA-256 为 `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`。
`scripts/fastwam_q/convert_dinov3_native.py` 做严格权重映射、QKV 拆分和 Meta 原版前向对比，
通过后保存本地 Transformers checkpoint；`prepare_models.py` 识别转换凭据后不再访问 gated HF 仓库。

原始 checkpoint 的 RoPE periods 是 BF16 舍入后的持久化 buffer，不能用理想 FP32 等比序列替代。
转换将它们保存在 backbone config 的 `native_rope_periods`；Q 的 DINO encoder 在初始化和
checkpoint 重载时恢复这组频率。使用普通 HF 直接加载转换目录进行独立特征推理时，也须应用此配置，
不能忽略该扩展。转换保留 DINOv3 LICENSE，并在 `conversion.json` 记录来源 revision、hash 和对比误差。

## 5k checkpoint 稳定性对照

`RL.cli.validate_fastwam_q` 和 `scripts/fastwam_q/{plan_stability,run_stability_queue,summarize_stability}.py`
用于独立验证目录，不覆盖或自动恢复45k正式训练。所有处理从相同5k Q/EMA/Adam状态出发，实际batch64、
共享预先保存的随机帧索引及每步随机种子。比较原LR、只降DINO、只降decoder、两者都降、低LR加300步重启warmup、
全critic FP32，以及只冻结DINO参数（保留训练模式的RoPE增强，避免额外混杂）。

每组最多4000更新，覆盖原先6k–8k的不稳定区间。保护在参数更新前检查非有限值、单次梯度>=1e6、
或连续20次梯度>1e3。每步保存分组梯度、裁剪前后范数、裁剪系数、TD误差、HL目标熵/KL、Q/logit统计。
每250步用固定256个按任务平衡的训练样本检查固定5k目标误差和EMA目标漂移；这不是独立验证集或机器人成功率。

第一轮是单seed筛选，不对相关的optimizer step做独立重复推断，也不把4k窗口稳定当作45k稳定。
`summary.json`/`comparison.json`中的gradient screen条件只是数值稳定筛选；选择后续方案还需比较学习进展，
不应将冻结/低LR导致的无学习误判为效果提升。每组独立SwanLab online/private记录，结束或guard触发后保存checkpoint。
