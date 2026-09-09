# SmolVLA 离线 RL

实现入口：`uv run --no-sync python -m RL.cli.train_smolvla_offline`。
**不要**把 SmolVLA checkpoint 交给旧的 `RL.cli.train_offline`（仍是 Diffusion/DP3 路径）。

## 阶段顺序

1. **IQL warm-up**：只更新 Q/V 及其 target，不更新 Actor。
2. **Dynamics warm-up**：只训练预测未来观测表征的 ensemble。
3. **Actor + AM-Q**：固定 Q/V、Dynamics，更新 Actor；定期 AM-Q 比较 candidate 和 behavior。

AM-Q 本身不是另一个要训练的网络：它用已训练的 Dynamics 想象 rollout、用 IQL 的 min-Q
评分，再决定是否更新 behavior。整个离线流程始终冻结 SmolVLM **及 IL 学到的 state projection**。
正式流程在进入 Actor 前检查 held-out Dynamics 验证误差；不合格时停止并保留 warm-up checkpoint，
可以增加 Dynamics 总步数后恢复。`--smoke` 明确跳过这个前置门槛以验证软件路径，但仍执行 AM-Q 接纳门控。

## 共享观测 token + 小型 Transformer

```text
三路 RGB + 20D state + 固定任务文本
              ↓ 原 checkpoint 的 processors
冻结 SigLIP / connector / text embedding + 冻结 state projection
              ↓ prefix embeddings [B,241,960]
           冻结 SmolVLM text layers（同一个实例）
              ├── 原生完整 K/V [B,16,241,640] → Actor
              └── 最后层 condition tokens [B,241,960]
                        ├── Q1/Q2/V 的小型 query Transformer
                        └── Dynamics ensemble + action tokens
                                     ↓ 预测下一份 prefix embeddings
                                     └── 回到同一个冻结 SmolVLM text layers
```

- 当前 50k IL checkpoint：241 个条件位置为 192 个三相机图像 token、48 个文本槽位和
  1 个 state token；padding mask 排除文本 padding。没有把三张 RGB 横向拼成一张图。
- current/behavior **共享同一 SmolVLM 和 state projection 实例**，仅动作专家及动作投影有独立参数。
- 原生 prefix / 最后层 tokens / K/V 都由同一次冻结编码产生，`no_grad + detach`；
  不新增或微调视觉编码器，不再池化/展平为 20,480 维向量。
- 新增 value query token 在**独立小网络**内，不插入原 SmolVLM，不改变已训 Actor 的条件。

### Q / V 结构

```text
Q1/Q2: condition tokens → LayerNorm + Linear(960→128) + 位置编码 ─┐
       32×20D executed action → 每步一个 token + 动作顺序编码 ──┤ memory
       2 个可学习 value queries → 2 层小型 Transformer decoder ──┘
                                → mean(query tokens) → Linear → scalar Q

V:     同样的 2 个 value queries + 2 层 Transformer，但 memory 只有观测，禁止输入 action
```

每层含 query self-attention、对 memory 的 cross-attention、FFN；默认 hidden=128、
4 heads、FFN=256、dropout=0。只更新 query，不重新编码整段 observation memory。
Q1/Q2/V 参数独立，Q 优化器与 V 优化器的参数不重叠；target Q1/Q2 是冻结副本并做 Polyak 更新。
小型 token projection 属于价值头的可训练参数，**输入的 SmolVLM 表征始终冻结且一致**。
动作只纳入有效执行步和活动维度；动作 padding 不进入 attention softmax，V 不会泄露数据集动作。
IQL 的 expectile / TD / min-Q advantage 数学目标不变。

默认尺寸的实测参数量（不含 Actor / 冻结 SmolVLM；target 单列）：

| 模块 | 原 MLP / K/V 版 | token-attention 版 |
| --- | ---: | ---: |
| Q1 | 5,481,217 | 561,025 |
| Q2 | 5,481,217 | 561,025 |
| V | 5,309,185 | 553,985 |
| **Q/V 可训练合计** | **16,271,619** | **1,676,035（减少 89.7%）** |
| Dynamics，3 heads | 578,694 | 782,979 |
| **Q/V + Dynamics** | **16,850,313** | **2,459,014（减少 85.4%）** |

新版额外冻结 target Q 合计 1,122,050 参数。Dynamics 本身比原来的极小 tokenwise MLP 略大，
但现在能通过注意力聚合观测与动作，整体辅助网络仍显著缩小。不会把冻结 backbone 重复计入。

### AM-Q 的 token dynamics

AM-Q 是评估/接纳算法，不是额外一个 Q 网络；其 Dynamics 改成 3 个独立 attention heads。
每个 head 默认 hidden=64、2 层、4 heads、FFN=128；future-slot queries 预测下一时刻的
prefix residual，另有 reward / done 两个 query。只更新视觉/state slots，语言、image special
tokens 和 padding 的 **输入 prefix** 保持逐位不变；重新编码后的文本 hidden states 可随图像改变。

训练目标是下一份 **VLM 输入 prefix**，不是各层独立的 K/V。AM-Q rollout 将预测 prefix 送回
同一个冻结 SmolVLM，重新得到 next Actor K/V 和 next Critic tokens，避免两套预测表征互不一致，
也不会反复复用起始图像。它仍不是生成 RGB，且无需再次运行 SigLIP。

prefix 回归采用当前 prefix 每个 token 的 RMS（至少 1）归一化；visual/state 两组误差等权，
避免 192 个图像 token 淹没 1 个 state token。terminal next-feature targets 不计入回归。
验证 loss = 归一化 prefix MSE + reward MSE + done BCE；disagreement 使用同样尺度的
ensemble prefix 方差。**这些量与旧 K/V 版不同，旧验证数值和阈值校准不能直接沿用**。
默认 `--max-validation-loss 1.0` / `--max-disagreement 0.1` 只是待验证的起点。

这仍是近似 latent dynamics：通过同一个 VLM 保证计算表征一致，**不保证预测 prefix 来自真实
可达观测**，也不保证原始 state-projection/视觉流形约束。必须检查 held-out 误差、长 rollout
误差和 ensemble disagreement。参数少不等于实机成功率高；AM-Q 每个未来状态需再跑冻结
text layers，不能仅从参数下降宣称整条训练路径按比例加速。

## 流匹配策略梯度

保留 SmolVLA 的速度场和 `t=1 → 0` 约定。离线 Actor 训练使用常数 diffusion coefficient 的
reverse-SDE Euler–Maruyama 转移：

```text
score = -(x + (1-t)*v) / t
mean  = x + dt * (v - g²*score/2)
std   = g * sqrt(-dt)
x_next ~ Normal(mean, std² I)
```

所有训练转移（包括最后一步）方差严格为正。`log_prob` 是真实 Gaussian log-density，
采样与重算使用相同方差；不是 pseudo-logprob，也没有“采样 sigma 与概率 sigma 不同”的默认替换。
PPO 在旧策略采样的同一条 trace 上计算新旧概率比；每个 flow 子步骤共享 IQL 给出的环境优势。

当前 IL 预测 horizon 为 **64**，执行 **32** 步，实际 action 为 **20D**，模型 padding 宽度为 32。
保留完整生成 trace，但 PPO 的 event reduction 只计执行窗口中的有效步和实际变化的动作维度；
不把 padding 通道计入。该执行窗口条件概率目标沿用仓库的 masked-denoising PPO 约定，
不声称它是将未执行潜变量积分消去之后的精确最终动作边缘密度。

普通 SmolVLA 推理仍是原来的确定性 Euler 转移（初始噪声可随机），源码推理路径未修改。
测试验证了固定初始噪声时适配器与原 IL sampler 输出一致。
RL 随机采样与确定性部署之间仍需真实 rollout 验证，不能只用训练 loss 宣称策略变好。

## 数据与奖励

当前数据的 24 个 episode 已由用户确认均为真实成功。运行时使用显式选项：

```text
--confirmed-manifest-rewards
```

它读取 `franka_duo_extras/derived_manifest.json` 中的 `manifest_reward`，验证 schema、episode
数量和逐段长度，再按转换后的本地 episode 顺序映射；源 episode 编号有缺号，不能直接作为本地索引。
不会把 `saved` 自动当成功，也不会修改源数据集。

其他数据可提供 `--labels labels.json`（二选一）：

```json
{"episodes": [{"episode_index": 0, "success": true}, {"episode_index": 1, "success": false}]}
```

标签必须恰好覆盖所有本地 episodes。本版本是**单任务** runner，验证 `meta/tasks.parquet`
与 `--task` 一致，不以旧 episode metadata 中的文本为准。

默认不重叠地切出执行窗口。非终止窗口的 next observation 是执行窗口之后的真实观测；
最后窗口按有效长度 padding，成功奖励放在窗口最后一帧并折扣，`discount=gamma**valid_length`。
终止窗口不 bootstrap、不训练 next-prefix 回归（末尾已没有下一段真实观测）；失败也视为 episode
结束，而不是跨 reset bootstrap。当前数据共 321 个 decision windows；默认 seed 1000 按 episode
划分 244 个训练、77 个验证窗口，避免同一 episode 同时进入 Dynamics 训练和验证。

全成功演示可以训练这条路径，但失败/恢复覆盖有限。AM-Q 估值不能替代真实任务成功率测试。

## 运行

使用现有 CUDA 环境，无需重新同步/裁剪依赖。若缺少依赖，可保留已有包安装：

```bash
uv sync --locked --inexact --extra training --extra smolvla-server --extra test
```

先仅训练 IQL（示例总步数，不是已经验证最优的超参数）：

```bash
IL=outputs/train/franka_duo_smolvla_vlm_only_512_h64_a32_50k/checkpoints/050000/pretrained_model
uv run --no-sync python -m RL.cli.train_smolvla_offline \
  --checkpoint "$IL" \
  --dataset-root datasets/franka_duo_lerobot_rgb20d_v1 \
  --confirmed-manifest-rewards \
  --output-dir outputs/rl100/smolvla_token_iql_10k \
  --hidden 128 --critic-layers 2 --critic-heads 4 --value-tokens 2 \
  --dynamics-hidden 64 --dynamics-layers 2 --dynamics-heads 4 \
  --device cuda --offline --iql-steps 10000 --stop-after iql
```

继续 Dynamics、Actor（总步数是累计目标，IQL 不会重复训练）：

```bash
uv run --no-sync python -m RL.cli.train_smolvla_offline \
  --checkpoint "$IL" \
  --resume outputs/rl100/smolvla_token_iql_10k/checkpoints/iql_010000 \
  --dataset-root datasets/franka_duo_lerobot_rgb20d_v1 \
  --confirmed-manifest-rewards \
  --output-dir outputs/rl100/smolvla_token_offline_10k \
  --device cuda --offline \
  --iql-steps 10000 --dynamics-steps 10000 --actor-steps 10000 \
  --actor-lr 1e-7 --amq-interval 50 --amq-horizon 5
```

也可第一次就不指定 `--stop-after`，按顺序自动跑三个阶段。可用 `--stop-after dynamics` 在
训练 Actor 前人工检查验证结果。动力学验证不达标时应检查覆盖/模型或延长训练，不应只放宽门槛。

默认 batch=8、IQL expectile=0.7、target tau=0.005、Q/V lr=3e-4、Dynamics lr=3e-4、
3 个 Dynamics heads、10 个 flow steps、flow noise=0.1、PPO clip=0.1。
可用 `--hidden` / `--critic-layers` / `--critic-heads` / `--value-tokens` 调整价值头；
用 `--dynamics-hidden` / `--dynamics-layers` / `--dynamics-heads` 调整动力学。宽度须能整除 heads。
`run.json` 和启动日志记录 token shape、结构参数与实际参数量。
Actor 默认 lr=1e-7：32×20 的联合 log-ratio 很敏感，这只是保守起点，需监控 KL/clip fraction。

## Checkpoint 与部署

每个阶段结束及 `--save-every` 间隔保存到新目录，原子发布并校验文件 hash：

```text
checkpoints/actor_005000/
  pretrained_model/   # 最后一次 AM-Q 接纳的 behavior，适合后续评估；不是未接纳的 candidate
  training_state.pt   # candidate 动作参数、IQL/Dynamics、优化器、阶段计数和 Torch RNG
  run.json
  manifest.json
```

没有任何更新被接纳时，`pretrained_model` 仍保留 IL 权重。这是预期行为。
恢复时同时恢复 behavior 和 candidate，不会把尚未接纳的候选策略冒充部署模型。
恢复要求原 IL、任务、标签、轨迹 Parquet/metadata、held-out split、模型和 flow 超参数兼容，
输出到新目录；batch 遍历会重新建立，因此不承诺中断点后 minibatch 顺序逐位一致。
视频文件未包含在该轨迹 fingerprint 中，恢复期间也必须保持视频不变。

新格式为 `smolvla_offline_rl_v2_tokens`，**明确拒绝旧 `smolvla_offline_rl_v1` 的 MLP/KV
训练状态**，不能把它们当成可继续训练的新头。原 50k IL checkpoint 仍兼容；从它初始化 Actor，
重新训练新的 Q/V 和 Dynamics，再进入 Actor 阶段。v2 恢复同时检查 Transformer 层数/宽度/
heads/value-token 数和 token layout，不会静默部分加载。

WS 服务可加载接纳后的 `pretrained_model`；新模型完整预测输出是 **64×20**，实际执行前 32 步。
本 runner 不连接硬件、不替换当前在线 WS 模型、不提供真实机器人 rollout 成功率评估。

## 验证

```bash
uv run --no-sync python -m pytest \
  tests/rl100/test_smolvla_offline.py \
  tests/rl100/test_iql.py tests/rl100/test_ppo.py tests/rl100/test_dynamics.py -q

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
SMOLVLA_TEST_CHECKPOINT="$IL" \
SMOLVLA_TEST_DATASET=datasets/franka_duo_lerobot_rgb20d_v1 \
uv run --no-sync python -m pytest tests/rl100/test_smolvla_offline_cuda.py -q
```

软件 smoke（不是有意义的 RL 训练，不应部署其结果）：

```bash
uv run --no-sync python -m RL.cli.train_smolvla_offline \
  --checkpoint "$IL" --dataset-root datasets/franka_duo_lerobot_rgb20d_v1 \
  --confirmed-manifest-rewards --output-dir outputs/rl100/smolvla_smoke_new \
  --device cuda --offline --smoke
```
