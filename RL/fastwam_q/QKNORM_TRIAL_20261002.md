# QK normalization 5k对照训练（2026-10-02）

## 已启动

- 主机：`GB200-Robot-NAS` / `r01dgx02`，GPU0。
- 独立目录：`/data/workspace/droid_fastwam_q_20260930/production_qknorm_20261002_1048/`。
- tmux：`q-qknorm-5k`；SwanLab：<https://swanlab.cn/@game-loader/fastwam-dinov3-q/runs/dzpx8sg9>。
- batch64，5,000次更新，5k保存并另存last。
- 从原始预训练DINOv3/T5、新初始化Q decoder/Adam/EMA开始，不续接旧45k权重。
- 18层decoder、全量DINO微调、head LR3e-4/DINO LR9e-5、dropout0.1、BF16、
  math训练attention、activation checkpointing、TD/EMA、seed42及数据处理均保持不变。
- 既有lossless RGB cache只读使用；没有重新缓存视频、修改共享环境或操作机器人。

## 唯一模型计算改动

`FastWAMQConfig.qk_norm=True`，`qk_norm_eps=1e-6`。
decoder的self/cross attention在投影、拆head后做无可学习仿射增益的逐head RMS normalization：

`q_hat = q / sqrt(mean(q²) + eps)`，K同理。

RMS以FP32计算，然后回到原AMP dtype；V不归一化。保留SDPA默认的`1/sqrt(head_dim)`缩放，
head_dim64时score近似为`8*cos(q,k)`，不是默认缩放作用于unit-L2向量造成的过平坦分数。
没有可学习temperature/gain，没有新增参数，没有gate、register、LayerScale或初始化改动。

使用继承MultiheadAttention的critic专用模块，保留原投影参数名字及初始化。
默认`qk_norm=False`，旧checkpoint配置缺少此字段时仍保持旧计算路径。
online及EMA target、后续从新checkpoint加载的推理模型都应用相同的QK normalization。

模型SHA256：`b065b0d90a2351d5c2973e72b3325387c7a789d36210d2a294c7ea7aabf6a97b`。
可训练参数仍为608,072,805；BC统计hash仍为
`2197e4e9b7497c04f6e11ecff161c3225a9073c814e7e5fa5cefc0cb101cacdb`。

## 早期实测与验证边界

11:15:41读取到step320：loss1.68861、HL-KL0.57715、裁剪前grad_norm2.45986；
峰值allocated显存16.633GiB。已记录33点，最大裁剪前梯度12.35866。
这只证明已经开始真实更新且初期未见灾难发散，不是5k稳定性或价值排序改善结论。

完成5k后，`launch.sh`自动运行18层attention探针，输出
`analysis/attention_005000/attention_summary.json`。
原math-only正式run的5k权重已用同一采样计划update2030/2031/2032，三个batch64做完对照，
保存在`analysis/baseline_math_005000/`。train/eval有效行cross饱和率分别21.69%/22.38%，
self分别42.10%/42.19%；不要把旧45k的90.97%当成新5k的直接同step对照。

这192个chunk来自训练示范池，不是独立留出集。标签仍为变化的TD目标；
attention变得不饱和或训练HL-KL降低，都不能单独证明critic能排序好坏候选动作。

## 验证与文件

- 专项8 tests通过；custom suite：896 passed、7 skipped；Ruff/shell语法通过。
- 测试验证了大投影增益下归一化后的score边界、padding、有限反向、相同参数初始化及旧state_dict加载。
- 远端`launch_receipt.json`含真实step320、SwanLab、源码hash及初始化/对照设置。
- `run/training_config.json`、`run/metrics.jsonl`、`run/progress.json`和`logs/train.log`持续保存进度。
- 旧正式训练、权重和所有诊断均保留；本工作未写入ARA。
