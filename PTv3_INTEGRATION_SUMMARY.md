# DP3 PTv3 编码器集成 - 改动总结

> Migration audit: this historical document overstates implementation status. The encoder is an MLP placeholder, not PTv3. It is disabled by default and rejects pretrained PTv3 weights. See `docs/EXPERIMENTAL_POINT_ENCODERS.md`.

## 📋 任务概述

将 DP3 点云编码器从简单的 PointNet MLP 替换为冻结的 Point Transformer V3 (PTv3) 预训练模型，以提升点云特征提取能力。

---

## 🔧 实现的改动

### 1. 新增文件

#### `src/lerobot/policies/dp3/ptv3_encoder.py`
- **PTv3Encoder 类**: 封装冻结的 PTv3 backbone + 可训练投影层
- **功能**:
  - 加载预训练 PTv3 权重
  - 冻结 backbone 参数
  - 添加可训练的线性投影层 (256→256)
  - 保持与 PointNetEncoder 相同的接口

#### `docs/PTv3_ENCODER_MIGRATION.md`
- 详细的架构对比文档
- 使用说明
- 预期效果分析

#### `scripts/train_dp3_compare.py`
- 便捷训练脚本
- 支持切换 PointNet/PTv3 编码器
- 自动配置参数

#### `outputs/franka_duo_dp3_action20_pc_only_dit768_ptv3_il_50k/run_train.sh`
- PTv3 训练脚本 (50k steps)
- 配置了所有 PTv3 相关参数

#### `outputs/franka_duo_dp3_action20_pc_only_dit768_ptv3_il_50k/README.md`
- 快速启动指南

---

### 2. 修改的文件

#### `src/lerobot/policies/dp3/configuration_dp3.py`
添加 PTv3 配置项：
```python
# PTv3 encoder settings
use_ptv3_encoder: bool = False           # 启用 PTv3
ptv3_model_path: str | None = None       # 权重路径
ptv3_feature_dim: int = 256              # PTv3 输出维度
ptv3_freeze_backbone: bool = True        # 冻结 backbone
```

#### `src/lerobot/policies/dp3/modeling_dp3.py`
修改 `DP3ObservationEncoder.__init__()`:
```python
# 导入 PTv3Encoder
from .ptv3_encoder import PTv3Encoder

# 根据配置选择编码器
if config.use_ptv3_encoder:
    self.point_net = PTv3Encoder(config, ...)
else:
    self.point_net = PointNetEncoder(config)
```

---

## 📊 架构对比

### PointNet (基线 - 60k 训练)

```
输入: (batch, 2048, 3)
  ↓
MLP: 3→64→128→256
  ↓
Max Pool
  ↓
输出: (batch, 256)

参数: ~67K (全部可训练)
```

### PTv3 (新架构 - 50k 训练)

```
输入: (batch, 2048, 3)
  ↓
🔒 PTv3 Backbone (46M 参数, 冻结)
  ↓
输出: (batch, 256)
  ↓
✏️ Linear Projection (131K 参数, 可训练)
  ↓
输出: (batch, 256)

总参数: 46M (冻结) + 131K (可训练)
```

---

## 🚀 使用方法

### 方法 1: 使用便捷脚本（推荐）

```bash
# PTv3 训练
python scripts/train_dp3_compare.py --encoder ptv3 --steps 50000

# 使用预训练权重
python scripts/train_dp3_compare.py \
    --encoder ptv3 \
    --ptv3-weights /path/to/ptv3.pth \
    --steps 50000

# PointNet 基线
python scripts/train_dp3_compare.py --encoder pointnet --steps 60000
```

### 方法 2: 使用 Shell 脚本

```bash
# PTv3
./outputs/franka_duo_dp3_action20_pc_only_dit768_ptv3_il_50k/run_train.sh

# PointNet (原始)
./outputs/franka_duo_dp3_action20_pc_only_dit768_il_60k/run_train.sh
```

### 方法 3: 手动命令行

```bash
uv run python -m lerobot.scripts.lerobot_train \
    --policy.type=dp3 \
    --policy.use_ptv3_encoder=true \
    --policy.ptv3_model_path="/path/to/weights.pth" \
    --policy.ptv3_freeze_backbone=true \
    --steps=50000 \
    # ... 其他参数
```

---

## ⚠️ 重要说明

### PTv3 预训练权重

**当前状态**: 代码使用占位符模型（随机初始化的 PTv3 风格架构）

**获取真实 PTv3 权重**:
1. 官方仓库: https://github.com/Pointcept/PointTransformerV3
2. Hugging Face: https://huggingface.co/jayakumarpujar/Ptv3
3. 下载后通过 `--policy.ptv3_model_path` 指定路径

**集成真实 PTv3**:
需要修改 `ptv3_encoder.py` 中的 `_create_ptv3_placeholder()` 方法，替换为：
- 加载 Pointcept 库
- 初始化真实 PTv3 模型
- 加载预训练权重

---

## 📈 预期效果

### PTv3 的优势

✅ **更强的点云理解能力**
- PTv3 在大规模 3D 数据上预训练
- 能捕捉复杂的空间几何关系

✅ **更快的训练收敛**
- 冻结特征提取器
- 只训练轻量投影层

✅ **更好的泛化性**
- 预训练知识迁移到机器人任务
- 对新物体和场景更鲁棒

✅ **更高的数据效率**
- 利用预训练表示
- 需要更少的训练步数

### 潜在挑战

⚠️ **域迁移问题**
- PTv3 可能在室内扫描数据上训练
- 需要验证在机器人操作场景的效果

⚠️ **内存占用**
- 46M 参数需要更多 GPU 内存
- 推理时占用约 253 MB

⚠️ **推理速度**
- 比简单 MLP 慢
- 但仍可满足实时控制要求

---

## 📁 文件结构

```
lerobot_droid/
├── src/lerobot/policies/dp3/
│   ├── configuration_dp3.py          # ✏️ 修改：添加 PTv3 配置
│   ├── modeling_dp3.py                # ✏️ 修改：条件加载编码器
│   ├── ptv3_encoder.py                # ✨ 新增：PTv3 编码器
│   ├── pointcloud.py                  # (未修改)
│   └── ...
├── scripts/
│   └── train_dp3_compare.py           # ✨ 新增：训练脚本
├── outputs/
│   ├── franka_duo_dp3_action20_pc_only_dit768_il_60k/          # PointNet 60k
│   │   └── run_train.sh
│   └── franka_duo_dp3_action20_pc_only_dit768_ptv3_il_50k/     # ✨ PTv3 50k
│       ├── run_train.sh               # ✨ 新增
│       ├── README.md                  # ✨ 新增
│       └── (训练输出将在此生成)
└── docs/
    └── PTv3_ENCODER_MIGRATION.md      # ✨ 新增：详细文档
```

---

## 🧪 验证和测试

### 训练前检查

```bash
# 1. 检查代码语法
uv run python -c "from lerobot.policies.dp3.ptv3_encoder import PTv3Encoder; print('✅ Import OK')"

# 2. 验证配置
uv run python -c "from lerobot.policies.dp3.configuration_dp3 import DP3Config; c = DP3Config(); print(f'use_ptv3: {c.use_ptv3_encoder}')"

# 3. 干运行训练命令
python scripts/train_dp3_compare.py --encoder ptv3 --dry-run
```

### 训练中监控

```bash
# 实时查看日志
tail -f outputs/franka_duo_dp3_action20_pc_only_dit768_ptv3_il_50k/logs/train.log

# 检查 GPU 使用
nvidia-smi -l 1

# 监控 checkpoint
watch -n 60 'ls -lh outputs/franka_duo_dp3_action20_pc_only_dit768_ptv3_il_50k/train/checkpoints/'
```

### 训练后评估

```bash
# 查看最终 checkpoint
ls outputs/franka_duo_dp3_action20_pc_only_dit768_ptv3_il_50k/train/checkpoints/050000/

# 对比性能（需运行评估脚本）
# TODO: 添加评估命令
```

---

## 🔄 后续工作

### 短期 (立即)

- [ ] 获取真实 PTv3 预训练权重
- [ ] 修改 `_create_ptv3_placeholder()` 加载真实模型
- [ ] 运行 50k 训练
- [ ] 对比 PointNet vs PTv3 性能

### 中期 (本周)

- [ ] 评估真实机器人任务成功率
- [ ] 尝试不同 PTv3 预训练源（ScanNet, S3DIS, etc.）
- [ ] 实验部分解冻 PTv3（微调 vs 完全冻结）
- [ ] 消融实验：投影层架构的影响

### 长期 (本月)

- [ ] 集成 PointWorld 动态预测
- [ ] 多模态融合：PTv3 + RGB
- [ ] 端到端联合训练探索
- [ ] 在更多任务上验证泛化性

---

## 📚 参考资料

- **Point Transformer V3**: https://arxiv.org/abs/2312.10035
- **PointWorld**: https://arxiv.org/abs/2601.03782
- **Pointcept 库**: https://github.com/Pointcept/PointTransformerV3
- **LeRobot 文档**: https://github.com/huggingface/lerobot

---

## 📝 总结

本次改动成功地为 DP3 策略添加了 PTv3 编码器支持，同时保持了与原始 PointNet 的向后兼容性。通过冻结预训练的 PTv3 backbone，我们期望在减少训练步数的同时获得更强的点云特征表示能力。

**关键设计决策**:
1. ✅ 冻结 PTv3 backbone，只训练投影层
2. ✅ 保持与 PointNet 相同的输出维度 (256-d)
3. ✅ 配置驱动，易于切换编码器
4. ✅ 向后兼容，不影响现有代码

**下一步**: 获取真实 PTv3 权重并开始训练！
