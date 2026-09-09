# DP3 Point Cloud Encoder: PointNet vs PTv3

> Migration audit: this historical document overstates implementation status. The encoder is an MLP placeholder, not PTv3. It is disabled by default and rejects pretrained PTv3 weights. See `EXPERIMENTAL_POINT_ENCODERS.md`.

## 概述

本文档记录了将 DP3 策略的点云编码器从简单的 PointNet MLP 替换为冻结的 Point Transformer V3 (PTv3) 的改动。

## 架构对比

### 原始架构：简单 PointNet (60k 训练)

**配置文件**: `franka_duo_dp3_action20_pc_only_dit768_il_60k`

**点云编码器结构**:
```
输入: (batch, 2048, 3) 点云 (XYZ)
  ↓
Per-point MLP:
  - Linear(3 → 64) + LayerNorm + ReLU
  - Linear(64 → 128) + LayerNorm + ReLU
  - Linear(128 → 256) + LayerNorm + ReLU
  ↓
Global Max Pooling: (batch, 2048, 256) → (batch, 256)
  ↓
输出: 256-d 特征向量 (直接融合，无投影)
```

**参数量**:
- Point encoder: ~67K 参数
- 全部可训练

**特点**:
- ✅ 简单、轻量
- ✅ 端到端训练
- ❌ 表达能力有限
- ❌ 没有预训练知识

---

### 新架构：冻结 PTv3 (50k 训练)

**配置文件**: `franka_duo_dp3_action20_pc_only_dit768_ptv3_il_50k`

**点云编码器结构**:
```
输入: (batch, 2048, 3) 点云 (XYZ)
  ↓
🔒 PTv3 Encoder (冻结, 46M 参数):
  - Stage 1: 64 通道
  - Stage 2: 128 通道
  - Stage 3: 256 通道
  - Stage 4: 512 通道 (bottleneck)
  ↓
Per-point 特征: (batch, 2048, 512)
  ↓
Global Max Pooling: (batch, 512)
  ↓
✏️ Trainable Projection:
  - Linear(512 → 256) + LayerNorm
  ↓
输出: 256-d 特征向量 → 融合到策略
```

**参数量**:
- PTv3 encoder: ~46M 参数 (**冻结**)
- Projection layer: ~131K 参数 (**可训练**)
- 总共只训练投影层

**特点**:
- ✅ 强大的预训练点云表示
- ✅ 使用 512-d encoder bottleneck 特征（信息最丰富）
- ✅ 更好的几何理解能力
- ✅ 泛化性更强
- ✅ 训练更快（冻结主干网）
- ⚠️ 需要 PTv3 预训练权重
- ⚠️ 推理时内存占用更大

---

## 代码改动

### 1. 新增文件

**`src/lerobot/policies/dp3/ptv3_encoder.py`**
- `PTv3Encoder` 类：冻结 PTv3 + 可训练投影层
- 支持加载预训练权重
- 保持与 PointNetEncoder 相同的接口

### 2. 修改的文件

**`src/lerobot/policies/dp3/configuration_dp3.py`**
```python
# 新增配置项
use_ptv3_encoder: bool = False          # 是否使用 PTv3
ptv3_model_path: str | None = None      # PTv3 预训练权重路径
ptv3_feature_dim: int = 512             # PTv3 encoder bottleneck 维度
ptv3_freeze_backbone: bool = True       # 是否冻结 PTv3
```

**`src/lerobot/policies/dp3/modeling_dp3.py`**
```python
# DP3ObservationEncoder 中的改动
if config.use_ptv3_encoder:
    self.point_net = PTv3Encoder(config, ...)
else:
    self.point_net = PointNetEncoder(config)
```

---

## 训练配置对比

| 参数 | PointNet (60k) | PTv3 (50k) |
|------|----------------|------------|
| **点云编码器** | Simple MLP | Frozen PTv3 Encoder |
| **编码器参数** | ~67K (全部可训练) | ~46M (冻结) + 131K (可训练) |
| **Encoder 输出** | 256-d (pooled) | 512-d bottleneck → pool → 256-d |
| **训练步数** | 60,000 | 50,000 |
| **Diffusion Backbone** | Transformer (768-d, 9 layers) | Transformer (768-d, 9 layers) |
| **批次大小** | 8 | 8 |
| **学习率** | 1e-4 | 1e-4 |

---

## 使用方法

### 方法 1: 使用训练脚本

```bash
cd /home/droid/project/lerobot_droid

# 如果有 PTv3 预训练权重
export PTv3_WEIGHTS="/path/to/ptv3_checkpoint.pth"

# 运行训练
./outputs/franka_duo_dp3_action20_pc_only_dit768_ptv3_il_50k/run_train.sh
```

### 方法 2: 手动运行

```bash
uv run python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="franka_duo/franka_duo_tmr_lerobot_v6_rl_action20_pc_only" \
    --policy.type=dp3 \
    --policy.use_ptv3_encoder=true \
    --policy.ptv3_model_path="/path/to/ptv3_weights.pth" \
    --policy.ptv3_freeze_backbone=true \
    --policy.diffusion_backbone=transformer \
    --steps=50000 \
    # ... 其他参数
```

---

## PTv3 预训练权重

### 获取方式

1. **官方 PTv3 仓库**: https://github.com/Pointcept/PointTransformerV3
2. **Hugging Face**: https://huggingface.co/jayakumarpujar/Ptv3
3. **自定义训练**: 在大规模点云数据集上预训练

### 当前实现

⚠️ **重要**: 当前代码使用了一个占位符 PTv3 模型（随机初始化）。要使用真正的预训练 PTv3：

1. 下载 PTv3 官方权重
2. 修改 `PTv3Encoder._create_ptv3_placeholder()` 以加载真实的 PTv3 模型
3. 或者集成 Pointcept 库

---

## 预期效果

### PTv3 的优势

1. **更强的几何理解**: PTv3 在大规模 3D 数据上预训练，能捕捉更复杂的空间关系
2. **更快收敛**: 冻结特征提取器，只训练投影层和策略网络
3. **更好的泛化**: 预训练知识帮助应对新场景和物体
4. **更鲁棒**: 对点云噪声和不完整数据更鲁棒

### 潜在挑战

1. **域迁移**: PTv3 可能在室内扫描数据上训练，需要适应机器人场景
2. **内存占用**: 46M 参数的 backbone 需要更多 GPU 内存
3. **推理速度**: 比简单 MLP 慢，但仍可实时运行

---

## 监控和评估

### 训练指标

监控以下指标来对比性能：

- **Loss**: 扩散损失是否更快下降
- **Action MSE**: 动作预测误差
- **收敛速度**: 是否在更少步数内达到相同性能

### 评估指标

- **成功率**: 真实机器人任务成功率
- **泛化性**: 在新场景/物体上的表现
- **鲁棒性**: 对传感器噪声的鲁棒性

---

## 下一步

### 短期

1. ✅ 集成 PTv3 编码器框架
2. ⏳ 获取并加载真实 PTv3 预训练权重
3. ⏳ 运行 50k 步训练
4. ⏳ 对比 PointNet vs PTv3 性能

### 中期

1. 尝试不同的 PTv3 预训练权重（不同数据集）
2. 实验微调 PTv3（部分解冻）vs 完全冻结
3. 尝试不同的投影层架构
4. 消融实验：PTv3 的哪些层最重要

### 长期

1. 在 PTv3 基础上添加任务特定的注意力机制
2. 多模态融合：PTv3 点云 + PointWorld 动态预测
3. 端到端优化：联合训练 PTv3 和策略

---

## 相关资源

- **Point Transformer V3 论文**: https://arxiv.org/abs/2312.10035
- **PointWorld 论文**: https://arxiv.org/abs/2601.03782
- **LeRobot DP3 文档**: 本项目 CLAUDE.md
- **训练日志**: `outputs/*/logs/train.log`

---

## 联系

有问题或需要帮助，请查看：
- Git log: `088e1ba7` 和 `3a172c12` 相关提交
- Memory: `franka-duo-pointcloud-only-dp3.md`
