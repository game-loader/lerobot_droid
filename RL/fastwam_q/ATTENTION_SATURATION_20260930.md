# 正式训练重启与 attention one-hot 归因（2026-09-30）

## 正式训练已启动

- 主机：`GB200-Robot-NAS` / `r01dgx02`，物理 GPU1。
- 独立目录：`/data/workspace/droid_fastwam_q_20260930/production_math_20260930_1652/`。
- `source/` 为独立修复源码，`run/` 为正式权重/指标，`logs/train.log` 为启动日志。
- tmux：`q-math-production-45k`。
- SwanLab：<https://swanlab.cn/@game-loader/fastwam-dinov3-q/runs/c556hl9r>。
- 从预训练 DINOv3、冻结的预训练 T5、新初始化的 Q decoder 开始；未续接旧 Q/Adam/EMA。
- batch64、45,000 次更新、每5,000次保存；原 LR 3e-4 / DINO LR 9e-5、dropout0.1、
  全量 DINO 微调、BF16、activation checkpointing、TD/EMA、seed42 均保留。
- 唯一训练计算修复：online decoder 的训练 attention 使用 math SDPA；没有添加 QK norm、
  改初始化或更改奖励。eval/inference 不强制 math。
- 模型源码 SHA256：`a080049b087c2bc0d8fdff003bd4d48b7cf386a39643206513f2816b758acb29`。
  后端与源码 hash 已写入训练配置及 SwanLab config。
- 17:07:38 读取到 step610，loss2.18616、裁剪前 grad_norm2.11698；已记录的62个指标点
  梯度最大12.03575，最近20点梯度中位数2.40492、loss中位数2.07420。
  这里只是初期数值状态，不能当作45k稳定性、Q排序能力或机器人成功率验证。
- 旧正式训练、checkpoint、所有诊断保留；GPU1暂与5k→9k修复验证共享。
  原视频/RGB cache保持只读，未重新缓存视频。
- CLI/launcher 修改后 custom suite：894 passed、7 skipped；Ruff及shell语法检查通过。

## “93%”到底是哪一个 softmax

这是**旧失稳权重 global step7029**、配对计划 update2030 的一个 batch64，
decoder **第17层 cross-attention**（代码 `online.layers[16].cross_attn`）的统计。
不是所有层、所有训练样本的93%；不是DINO内部attention；不是101-bin价值头的softmax。

该层的 attention query 来自 `norm2(动作query隐藏状态) + 时间位置`，再经该层Q投影；
key/value 来自 `[T5任务tokens, 三路DINO图像patch+camera/spatial embeddings]`，经K/V投影。
这里 attention 的Q/query与critic输出的标量Q价值不是同一个东西。

本batch：Q形状 `[64,16,33,64]`，K形状 `[64,16,622,64]`。
33个query是CLS+32个动作位置；622个key是34个批内文本槽位+3×196图像patch。
短文本padding在mask中排除。每行在key轴计算：

`A = softmax(q_att @ k_att.T / sqrt(64) + padding_mask)`。

“近one-hot”的定义是 FP32重算的 `max(A) > 1 - 1e-7`，在attention dropout之前统计。
它表示**单个head的某个query几乎只读取一个key对应的V**，不表示成功率93%、
动作选择概率93%，也不表示整个多head模型只能看到一个patch。

## padding、CLS、任务及token归因

| 第17层 cross-attention统计范围 | 行数 | 近one-hot比例 |
|---|---:|---:|
| 原统计，包含padding query | 33,792 | 92.6935% |
| 排除padding后的CLS+真实动作 | 32,896 | 92.9080% |
| 仅CLS | 1,024 | 92.6758% |
| 仅真实动作query | 31,872 | 92.9154% |
| 仅padding动作query | 896 | 84.8214% |

因此不是padding夸大的假象。四任务有效行饱和率分别为93.25%、91.82%、95.52%、87.67%。
第18层cross-attention有效行也有89.68%饱和；第17/18层self-attention约68.96%/69.28%。

第17层cross-attention有效行的赢家key归属：

- 文本2.985%、头部相机50.000%、左腕34.013%、右腕13.002%。
- 42.1267%的有效行选择同一个头部patch `(row12,col2)`，坐标为resize后14×14网格的0-based索引。
- 第0、3、4、7、12、13、15号head全部只选择头部相机，且约92–94%的有效行
  选择该固定patch；第2号head也只选择头部相机，但更分散。
- 有效动作query的赢家与同sample/head的CLS赢家一致率 **99.7051%**。
- **98.5352%** 的sample/head组合，其CLS和所有有效动作query的赢家完全相同。

说明该层读取context的路由已经高度同质化，不能简单解释为“不同动作各自找到相关物体”。
不过top1一致不能证明整个Q完全不依赖动作；residual、self-attention和V路径仍在。

已从原lossless cache只解压四个任务各一张RGB画面，保存
`analysis/head_patch_12_2_examples.png`。红框位置均是桌面左下方背景，而不是主要操作对象。
这支持固定空间attention sink的解释；DINO的背景patch也可能编码全局语义，
不能仅凭画面就断言该token无信息。文本选择也不是全无意义，例如抽屉任务常选择`drawer`。

## 为什么饱和：输入并没爆，学习后的投影增益很大

旧全网络trace与实际SDPA输入共同显示：

| 第17层cross-attention量 | RMS |
|---|---:|
| pre-norm前query隐藏状态 | 1154.22 |
| LayerNorm后/进入Q投影的输入 | 1.3781 |
| 进入K投影的context | 1.4117 |
| Q投影后，实际attention query | 120.27 |
| K投影后，实际attention key | 66.57 |

LayerNorm确实把输入压回正常尺度，但它位于Q投影**之前**，不限制投影后的范数。
Q/K投影的最大奇异值估计从5k的35.21/34.26增至7029的100.68/95.03。
该层score绝对值最大601,277，第一名与第二名score差的中位数 **786.21**。
对于这样的差值，`exp(-gap)`几乎为零；不是softmax代码逻辑错误或缺少减最大值。

对保存的Q/K做CPU FP32重算仍复现92.69%，证明饱和并非仅由BF16 softmax输出舍入造成。
它首先是已学习的Q/K尺度及方向造成的极端分数差，之后使低精度反向更容易出问题。

### learned camera/spatial embeddings的作用：离线近似线性归因

对**匹配的7029权重**，利用K投影的线性性扣除特定embedding的K贡献，上游query不变。
保存K来自BF16投影，扣除贡献以FP32计算，因此是近似的静态反事实，不是重训消融。

- camera embedding输入RMS1.0351，其K投影贡献RMS57.1578；扣除后视觉K剩余RMS30.5643。
  这些贡献不正交，不能把RMS平方直接当独立能量比例。相机标识对view偏好影响很强。
- spatial embedding输入RMS仅0.03685，K贡献RMS1.1771；但乘上巨大query后会改变patch间排名。
- 只扣除spatial contribution：近one-hot率92.9080%→68.3670%；头部patch(12,2)
  的选择比例42.1267%→**2.2070%**，相机选择比例变化很小。
- camera+spatial一起扣除仍有75.5593%饱和，不能把全部失稳都归因于位置embedding。
- 把保存的每个Q/K向量仅做unit-RMS归一化后，饱和率变为0；该层平均pmax约0.00362。
  这确认尺度的重要性，但**不证明添加QK norm就能训练出更好的critic**。

最受当前证据支持的机制是：学习后的Q/K增益使相机/位置偏置及视觉差异都被放大，
部分head反复读取固定位置，softmax高度饱和；BF16 efficient+dropout反向的数值误差
再通过深层decoder放大。究竟哪个训练设置最先推动投影增益增长，尚未做因果重训消融。

## 对优化及本轮修复意味着什么

softmax的导数为 `diag(p)-p p.T`；完全one-hot时，经概率权重传给Q/K的真实导数近零。
V路径、输出投影及residual仍可以训练，不等于整个网络没有梯度。
旧独立重放中，饱和行的query梯度L2为BF16 **0.06685** / FP32 **2.882e-6**，
是反向数值误差的直接证据，详见 `DIAGNOSIS_20260930.md`。

本轮math backend修复的是该反向数值路径，**不会主动约束Q/K增益或消除one-hot**。
为了可解释对照，没有将离线归一化或embedding扣除混进正式训练。
如果较长训练窗口仍出现路由坍缩/价值排序差，需将结构问题与数值问题分开验证，
不能仅凭小grad_norm认定critic已经学好。

## 可复核文件

生产目录中的：

- `launch_receipt.json`：实际启动、初始化方式、源码hash、SwanLab和step610读取记录。
- `analysis/attention_tokens_7029.json`：有效query/head/task/token、分数差、尺度反事实。
- `analysis/attention_embedding_attribution_7029.json`：camera/spatial近似线性归因。
- `analysis/head_patch_12_2_examples.png`：四任务只读RGB抽查，红框为被集中读取的位置。
- `analysis/attribute_attention_tokens.py`：实际执行的CPU分析代码。

原始捕获在 `stability_20260930/layer_diagnosis/baseline_capture/`，采样计划在
`stability_20260930/plan/`；没有修改旧证据、共享Torch/CUDA环境或机器人。
本工作未写入ARA。
