# QK normalization 从5k续训至总40k（2026-10-02）

用户要求将已经验证的QK normalization版本训练40k。解释为**总计40,000 optimizer更新**：
从5,000 checkpoint恢复，再训练35,000；没有重新初始化，也不是额外40,000。

- 主机：`GB200-Robot-NAS` / `r01dgx02`，GPU0。
- 新目录：`/data/workspace/droid_fastwam_q_20260930/production_qknorm_40k_20261002_1455/`。
- tmux：`q-qknorm-40k`；`logs/train.log`为持续日志。
- 恢复自：`production_qknorm_20261002_1048/run/step_005000`。
- SwanLab续训run：<https://swanlab.cn/@game-loader/fastwam-dinov3-q/runs/cb22a9b6>。
  旧5k run继续保留，新曲线从global5001开始。
- batch64，每5k保存：global10k、15k、20k、25k、30k、35k、40k，结束另存last。
- online/EMA权重、Adam、scaler、global step、CPU/CUDA RNG及保存的action normalizer均恢复。
  原checkpoint752个Adam参数状态的step全部为5000，LR3e-4 / 9e-5；EMA头和online头不同，
  不是将EMA重置成online。具体只读核验在`resume_audit.json`。
- 模型及学习设置不变：18层、原dropout、全量DINO、BF16/math后端、activation checkpointing、
  TD/EMA，QK norm继续为无可学习增益的逐head RMS norm、eps1e-6；未加gate/register。
- 模型源码SHA256仍为`b065b0d90a2351d5c2973e72b3325387c7a789d36210d2a294c7ea7aabf6a97b`。

15:00读取到global5050：loss1.58106、HL-KL0.48498、裁剪前grad_norm1.46213，
证明已经恢复并完成实际训练更新，目标进度正确显示40,000。

## 续训元数据和采样

CLI补充`initial_step`、`target_steps`、`resume_checkpoint`、`data_seed`，
修正resume时progress以前误把“追加更新数”显示为总目标的行为。
`--steps 35000`仍表示本次追加数，`target_steps=5000+35000=40000`。

此前checkpoint未保存DataLoader worker RNG状态。本次使用seed42+global5000=5042，
开始可复现的新随机replay采样流，避免按原seed重播早期worker采样序列。
模型/dropout RNG仍从checkpoint恢复。这不是逐batch完全等价于未中断训练；
样本仍从同一示范池均匀、有放回抽取相同chunk长度，数据/归一化均未改变。

## 后续自动检查与边界

`launch.sh`在成功完成40k后，自动对step_040000重放与原5k/45k相同的三个batch64，
统计18层self/cross attention、逐head/任务/token概率质量，输出`analysis/attention_040000/`。
这是训练池的只读注意力诊断，不是独立动作排序或机器人成功率评估。

旧5k、45k、所有诊断和原cache保留，未改变共享Python/CUDA环境或机器人，未重建视频cache。
`launch_receipt.json`含实际进度、恢复状态、SwanLab和源码hash。
本轮CLI检查后custom suite896 passed、7 skipped；Ruff与launcher shell语法通过。
本工作没有写入ARA。
