# DINOv3 Q critic 梯度失稳定位（2026-09-30）

## 已验证的直接因素

在 GB200、当前 NVIDIA PyTorch 环境中，critic decoder 的 **BF16 efficient SDPA +
dropout + 已高度饱和的 attention** 产生显著反向数值误差。18 层 decoder 逐层放大这些误差，
再传入视觉编码器。不能把 DINO 参数最终梯度最大直接解释为 DINO 是起因。

本结论限于这份实现、保存权重及测试样本，不声称所有 BF16 attention 都不稳定，
也没有证明学习率、TD 自举或初始化对长期学习没有影响。

实测环境：`torch==2.13.0a0+8145d630e8.nv26.06`，CUDA **13.3**；未更改该环境。

## 同权重 / 同 batch 静态对照

使用 baseline 的 global step 7029 guard checkpoint；保护触发前尚未应用第 7030 次更新。
重放配对采样计划的 update 2030，实际 batch64、原 action normalizer。
所有静态诊断均不做 optimizer/EMA 更新，关闭 activation checkpointing 以避免 hook 重算干扰。
同一 seed，不同 backend 的 dropout RNG 消耗可能不同，因此另做了同一 efficient backend 的
BF16/FP32 独立 attention 重放。

| 路径 | loss | 裁剪前总梯度 L2 |
|---|---:|---:|
| BF16 / 默认 backend / 原 dropout | 4.6364 | 1,138,865 |
| BF16 / math backend / 原 dropout | 4.5962 | 21.55 |
| FP32 / 默认 backend / 原 dropout | 4.5525 | 25.08 |
| BF16 / 默认 backend / dropout 关闭 | 4.5868 | 12.73 |
| BF16 / math backend / dropout 关闭 | 4.5839 | 13.01 |
| FP32 / 默认 backend / dropout 关闭 | 4.5818 | 9.63 |

默认训练 decoder 的真实 autograd kernel 是 `ScaledDotProductEfficientAttentionBackward0`。
关闭 dropout 会改变默认 backend；不能仅凭 dropout 消融就归因于正则化本身。

## 逐层反向传播

下表为同一 BF16 默认路径，decoder 从输出侧往输入侧传播时的参数梯度 L2。
层号为 1-based；输出头梯度仅约 6.13。

| 层 | 参数梯度 L2 |
|---|---:|
| 18 | 1.86 |
| 17 | 14.20 |
| 16 | 45.86 |
| 15 | 1,619.51 |
| 14 | 4,052.18 |
| 13 | 23,509.02 |
| 1 | 96,988.53 |

参数梯度不是局部 Jacobian；完整日志同时记录激活、模块输入/输出梯度、LayerNorm 方差。
模块输入梯度包含所有 fan-out，不能简单把任意 input/output 比值当作该模块独立放大率。

## 独立 attention 重放：数值误差的定位

捕获 decoder 第 17、18 层 self/cross attention 的 Q/K/V、mask、上游梯度和 CUDA RNG。
相同 efficient backend、相同 dropout RNG、相同数值的 Q/K/V 分别用 BF16 与 FP32 计算。
无需 DINO、TD 目标、Adam 或训练循环，即可重现反向偏差。

- 第 17 层 cross attention：attention scores 最大绝对值约 **601,277**；
  **92.69%** 行的最大 softmax 概率 > `1 - 1e-7`。
- 这些饱和行的 query 梯度：BF16 L2 **0.06685**，FP32 L2 **2.882e-6**。
- 第 18 层 cross attention：BF16/FP32 query 梯度相对误差 **53.50**，方向余弦 **0.0202**，
  但前向输出相对误差仅 **0.00231**。
- 存在 softmax 已 one-hot、FP32 query 梯度为 0，而 BF16 给出明显非零梯度的行。

这是低精度融合反向的舍入/抵消误差证据，不是正常的大 TD 标量目标。
具体底层 kernel 内部哪一次抵消产生误差尚未逐指令验证。

## 放大条件与其他对照

- Q/K 投影没有额外 normalization；第 17 层 cross-attention Q/K 投影最大奇异值估计
  从 5k 的约 **35.21 / 34.26** 增至 guard checkpoint 的 **100.68 / 95.03**。
  已观察到 projection 尺度与 softmax 饱和；未用训练消融证明是哪项设置首先导致尺度增长。
- 18 层 pre-norm residual decoder 的后部激活 RMS 可达千量级；5k 权重也已有大激活。
  数值误差可以跨层累积，并通过 cross attention 传给视觉编码器。
- 冻结 DINO 的 guard 权重，3 个配对 batch：默认 BF16 梯度约 **25,852 / 176,685 /
  238,168**；math BF16 约 **5.83 / 22.33 / 5.54**；FP32 约 **9.66 / 5.76 / 5.12**。
- 5k 权重在同一 update2030 batch：BF16 默认 **9.01**，math **4.01**，FP32 **2.35**。
  说明数值偏差随学习后的权重状态变严重，不是该 batch 的输入尺度突然越界。
- 原 gradient clipping 作用于总梯度；伪梯度占主导后，即便裁剪到 10，方向仍然错误，
  还会压低其他有效梯度。该后果是机制解释，不是独立的 clipping 消融结果。
- 官方 decoder Xavier 初始化在当前改写版中缺失：仍是实现差异，但当前证据不足以将其
  认定为本次百万级梯度的直接原因。

## 最小修复及验证边界

`DecoderLayer.forward` 仅在 training 时用 `sdpa_kernel(SDPBackend.MATH)`。
保留 BF16 的其他运算、DINO 全量微调、dropout、18 层结构、原 LR、TD 目标、Adam/EMA
与 activation checkpointing；eval/inference 不强制 math backend。

短验证从已退化的 baseline guard checkpoint 恢复 Adam/EMA，重放失败更新及后续 199 次更新。
它不是 45k 稳定性或动作排名/机器人成功率验证，也不会自动恢复正式训练。
最终数值以远端 `fixed_optimizer_200/summary.json` 为准。

短验证已完成：global step **7029 → 7229**，200 次真实 batch64 更新。
梯度中位数 **8.12**、最大 **425.02**、裁剪比例 **41%**；未触发保护。
前/后 25 次 loss 中位数 **2.747 → 2.096**。
修复消除了此次已复现的灾难性放大，但 41% 裁剪意味着不能宣称全部优化问题已解决。
另开 `fixed_continuation_5k_9k/` 从原 5k 权重用同一 4000×64 采样计划验证；结果不自动晋升。
该长窗口验证的初始固定面板 MAE **0.02859353**、MSE **0.00399759**，与原七组实验一致。

SwanLab：
- 200 次短验证：<https://swanlab.cn/@game-loader/fastwam-q-stability/runs/eymbcecx>
- 5k→9k 配对验证：<https://swanlab.cn/@game-loader/fastwam-q-stability/runs/7y7r12hb>

修复后 custom suite：**894 passed, 7 skipped**；Ruff 与诊断/后端选择专项测试通过。

## 可复核文件

远端根目录：
`/data/workspace/droid_fastwam_q_20260930/stability_20260930/layer_diagnosis/`

- `baseline_guard/`：六路静态全网络对照及逐层统计。
- `baseline_capture/`：kernel 名称及捕获的 attention tensors。
- `attention_replay_correct_mask.json`：独立 backend/precision/dropout 重放。
- `attention_rows.json`：相同 efficient backend 的逐行精度对照。
- `attention_weights.json`：5k/guard 投影尺度比较。
- `no_dino_guard/`、`initial_5k/`：冻结 DINO / 早期状态复核。
- `fixed_source/`：独立修复源码；原实验冻结源码与正式训练源码保持不变。
- `fixed_optimizer_200/`：短训练日志、SwanLab URL、最终 summary。

工具：`RL.cli.diagnose_fastwam_q`、`RL.fastwam_q.diagnostics` 以及
`scripts/fastwam_q/{replay_attention,analyze_attention_rows,inspect_attention_weights,verify_attention_fix}.py`。

本工作未写入 ARA；未改变共享 Torch/CUDA 环境、视频 cache 或机器人。
