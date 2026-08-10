# IMF-AttnRes + SmolVLM vs. SmolVLA：除 action expert/head 之外的差异排查

日期：2026-05-21
仓库：`/data/lerobot-imf-attnres-exp/lerobot-imf-attnres`

## Scope

本文主体比较 **SmolVLA** 与当前 **IMF-AttnRes + `use_smolvlm_vl_encoder=True`** 在 action expert/action head 之外的差异：

- 包含：config、policy pre/post processor、VLM/image/text/state conditioning、训练/eval protocol、normalization、chunking、dataset/env 对齐。
- 额外包含：当前仓库里仍存在的 **ResNet IMF 训练脚本**，因为如果低 success run 来自这些脚本，它本来就不是 SmolVLM/SmolVLA-style VLA。
- 尽量不比较：SmolVLA `lm_expert` 与 IMF `IMFTransformer1D/AttnRes` 的内部结构。
- 但会列出 **head 边界接口** 相关差异，例如输入 condition token、action chunk 长度、loss/time sampling，因为这些虽然连接 head，但会直接影响 LIBERO eval success。

> 已修复项：`VISUAL` normalization。当前 IMF SmolVLM path 已在 config / processor 里强制 `VISUAL=IDENTITY`，因此本文不再把它作为未修复根因，只保留和其它路径的对比。

---

## 0. 最可能影响 LIBERO success 的差异速览

优先级从高到低：

1. **先确认实际训练脚本是否真的走 SmolVLM/VLA path**：`scripts/imf_attnres_experiments/train_eval_suite.py` 仍是 ResNet + no language/tokenizer path；只有类似 `/data/lerobot-imf-attnres-exp/runs/ablation-local-swanlab/launch.sh` 这种显式 `use_smolvlm_vl_encoder=true` 的实验才是 SmolVLM path。
2. **预训练初始化不等价**：official SmolVLA 结果通常来自 `lerobot/smolvla_base` / `lerobot/smolvla_libero` 的整策略 checkpoint；IMF 只加载 generic SmolVLM VLM encoder，`state_projection`、IMF conditioning/action 侧都是新训。
3. **时间窗口 / action chunk 长度不同**：SmolVLA 是 `1 obs -> 50 actions`；IMF 默认是 `2 obs -> horizon 16 -> execute 8 actions`，当前 SmolVLM 实验也多是 `1 obs -> 16/8`。
4. **STATE/ACTION normalization 不同**：SmolVLA 用 `MEAN_STD`；IMF 默认仍是 `MIN_MAX`。
5. **action padding loss 处理不同**：SmolVLA 对 `action_is_pad` 做 loss mask；当前 IMF SmolVLM 实验脚本显式 `do_mask_loss_for_padding=false`，尾部被 clamp 的未来动作会进 loss。
6. **语言/视觉 prefix 进入 VLM text layers 的方式不同**：SmolVLA 总是经由 SmolVLM text layers + 自定义 mask/KV cache；IMF 默认 `vlm_text_encoder_mode="embedding"` 时跳过 text layers且不做 SmolVLA 的 embedding scaling，当前实验虽设了 `transformer`，但仍不是 SmolVLA 的 prefix/suffix 联合机制。
7. **prefix/mask/position_id 不同**：SmolVLA 是 prefix/suffix 2D mask + KV cache；IMF 默认是固定长度 condition tokens flatten 后喂给 IMF head。
8. **camera/key/empty camera protocol 不同**：官方 `smolvla_libero` eval 常用 `camera1/camera2 + empty_camera`；IMF 训练通常是 `image/image2` 两路。
9. **optimizer/scheduler/batch size/grad clip 差异**：SmolVLA 默认 AdamW；IMF 默认 Adam，且当前部分实验脚本 batch 很小、甚至 `grad_clip_norm=0`。
10. **loss/time sampling/inference steps 不同**：SmolVLA 是标准 flow matching MSE + 10 denoise steps；IMF 是 MeanFlow/JVP/pseudo-Huber/DCT/2 steps 等。

---

## 1. Config 层差异

### 1.0 先区分：当前代码里有两类 IMF 实验路径

排查 success gap 时，首先要确认跑的是哪条 IMF path：

**A. 旧/主 sweep 脚本：ResNet IMF，不是 SmolVLM/VLA**

`scripts/imf_attnres_experiments/train_eval_suite.py` 里训练命令虽然是 `policy.type=imf-attnres`，但没有打开 `use_smolvlm_vl_encoder`，并显式配置了 ResNet：

```bash
--policy.type=imf-attnres
--policy.vision_backbone=resnet18
--policy.pretrained_backbone_weights=null
--policy.resize_shape=[128,128]
--policy.num_inference_steps=1
```

位置：

- 脚本参数默认：`scripts/imf_attnres_experiments/train_eval_suite.py:311-324`
- 训练命令：`scripts/imf_attnres_experiments/train_eval_suite.py:395-423`
- IMF 默认 `use_smolvlm_vl_encoder=False`：`src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:41`
- IMF processor 只有 SmolVLM path 才 tokenize language：`src/lerobot/policies/imf_attnres/processor_imf_attnres.py:37-47`

这类实验和 SmolVLA 的差异不只是 action head，而是 **没有 SmolVLM 视觉语言 encoder、没有 language conditioning、图像输入是 128 ResNet 表征**。

**B. 当前 ablation launch：SmolVLM IMF path**

`/data/lerobot-imf-attnres-exp/runs/ablation-local-swanlab/launch.sh` 显式：

```bash
--policy.use_smolvlm_vl_encoder=true
--policy.load_vlm_weights=true
--policy.freeze_vlm_encoder=true
--policy.vlm_text_encoder_mode=transformer
```

位置：`/data/lerobot-imf-attnres-exp/runs/ablation-local-swanlab/launch.sh:20-27`

本文后续重点比较的是 **B：IMF-AttnRes + SmolVLM**；但如果你看的低 success run 来自 A，那么首要原因就是它本来不是 SmolVLA-style VLA。

### 1.1 Observation/action temporal shape

**SmolVLA**

- `n_obs_steps=1`
- `chunk_size=50`
- `n_action_steps=50`
- dataset delta：
  - `observation_delta_indices=[0]`
  - `action_delta_indices=range(chunk_size)`
- 位置：
  - `src/lerobot/policies/smolvla/configuration_smolvla.py:28-30`
  - `src/lerobot/policies/smolvla/configuration_smolvla.py:149-155`

**IMF-AttnRes**

- `n_obs_steps=2`
- `horizon=16`
- `n_action_steps=8`
- dataset delta：
  - obs: `range(1-n_obs_steps, 1)`，默认 `[-1, 0]`
  - action: `range(1-n_obs_steps, 1-n_obs_steps+horizon)`，默认 `[-1, ..., 14]`
- 位置：
  - `src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:14-16`
  - `src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:354-360`

**影响**

- IMF 的训练样本与 SmolVLA 官方并不是同一个行为克隆问题：它多看一帧历史，但预测较短 horizon，并且执行 chunk 更短。
- eval 时 SmolVLA 一次 rollout 50 个 action，IMF 通常每 8 步重新规划。LIBERO 上这会改变动作平滑性、闭环频率、误差累积模式。
- 若想做 “只换 action head” 对比，应至少做一组：
  - `n_obs_steps=1`
  - `horizon/chunk_size=50`
  - `n_action_steps=50`
  - `num_inference_steps` 对齐 SmolVLA 的 10

### 1.2 End-of-episode sampling

**SmolVLA**

- 没有 policy-level `drop_n_last_frames`。

**IMF-AttnRes**

- 若未设置，默认：

```python
drop_n_last_frames = horizon - n_action_steps - n_obs_steps + 1
```

- 默认值：`16 - 8 - 2 + 1 = 7`
- 位置：`src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:25`, `:153-154`
- 训练 sampler 使用它：`src/lerobot/scripts/lerobot_train.py:447-456`

**影响**

- IMF 会丢掉每个 episode 末尾一段训练 frame，主要避免 **实际执行的 action slice** 越界。
- 但 IMF loss 默认对完整 `horizon` 计算；若 `do_mask_loss_for_padding=false`，完整 horizon 中被 dataset clamp 的尾部 action 仍会参与 loss。
- dataset 边界 clamp/padding 逻辑在 `src/lerobot/datasets/dataset_reader.py:178-195`；IMF inference 只取 `start=n_obs_steps-1` 到 `start+n_action_steps`，见 `src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:1283-1285`。
- 这对 IMF 是合理的，但和 SmolVLA 的训练分布不同，尤其 LIBERO 成功常发生在 episode 末段，末段动作可能被少采样。

### 1.3 Normalization mapping

**SmolVLA**

```python
VISUAL = IDENTITY
STATE  = MEAN_STD
ACTION = MEAN_STD
```

位置：`src/lerobot/policies/smolvla/configuration_smolvla.py:32-38`

**IMF-AttnRes**

默认：

```python
VISUAL = MEAN_STD
STATE  = MIN_MAX
ACTION = MIN_MAX
```

位置：`src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:18-24`

当前已修复：当 `use_smolvlm_vl_encoder=True` 时，IMF 强制 `VISUAL=IDENTITY`：

- config 修正：`src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:149-152`
- processor 修正：`src/lerobot/policies/imf_attnres/processor_imf_attnres.py:48-59`

但 **STATE/ACTION 仍不同**：

- SmolVLA：mean/std normalized action/state
- IMF：min/max normalized action/state

**影响**

- LIBERO env action space 是 `[-1, 1]`，但 dataset action 的 min/max 不一定覆盖完整 env bound。
- IMF 使用 `MIN_MAX` 可能让模型输出经 unnormalization 后更受 dataset min/max 限制；SmolVLA 的 `MEAN_STD` 允许更自然地产生超过训练均值若干 std 的动作。
- Flow/MeanFlow 的 target scale 也会不同，直接影响 loss magnitude、gradient 和 gripper 维度权重。

### 1.4 预训练 checkpoint / 参数初始化语义不同

**SmolVLA**

- 从 scratch `--policy.type=smolvla` 时，`load_vlm_weights` 的默认值是 `False`，但官方文档/CI 的强结果路径通常不是纯 scratch：
  - finetune 文档从 `--policy.path=lerobot/smolvla_base` 出发。
  - LIBERO CI smoke train 也是 `--policy.path=lerobot/smolvla_base --policy.load_vlm_weights=true`。
  - released eval 直接用 `--policy.path=lerobot/smolvla_libero`。
- 这意味着 official checkpoint 不只是 generic SmolVLM，而是整套 SmolVLA 权重：VLM/text/vision（取决于训练设置）、`state_proj`、action projection、expert 等都可能经过大规模预训练/finetune。
- 位置：
  - config 默认：`src/lerobot/policies/smolvla/configuration_smolvla.py:84-86`
  - pretrained load 逻辑：`src/lerobot/policies/factory.py:533-537`, `src/lerobot/policies/pretrained.py:95-133`
  - docs/CI recipe：`docs/source/smolvla.mdx:56-65`, `.github/workflows/benchmark_tests.yml:183-195`

**IMF-AttnRes + SmolVLM**

- `load_vlm_weights=True` 只加载 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 到 `IMFAttnResSmolVLMVLEncoder`。
- `state_projection` 是 IMF 里新建的 `Linear(8或state+env, hidden)`，不是 SmolVLA `state_proj: 32 -> hidden` 的 pretrained 权重。
- IMF action/conditioning 侧无法直接加载 `lerobot/smolvla_base` 的 expert/action projection 权重。
- 位置：
  - `src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:41-45`
  - `src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:191-199`
  - `src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:230-233`

**影响**

- 如果比较对象是 official `smolvla_libero`，那差异远大于 “只换 action expert”：前面的 state projection、VLM 是否经过 SmolVLA 训练、processor stats/checkpoint protocol 都不同。
- 要验证 “只换 action head” 的假设，需要明确 baseline 是：
  1. `policy.type=smolvla` 从 generic SmolVLM scratch 训练；还是
  2. `policy.path=lerobot/smolvla_base` finetune；还是
  3. released `lerobot/smolvla_libero`。

### 1.5 VLM/vision/text 是否冻结不同

**SmolVLA**

- 默认 config：`freeze_vision_encoder=True`, `train_expert_only=True`, `train_state_proj=True`。
- 当 `train_expert_only=True` 时，会冻结整个 `self.vlm`；当 `train_expert_only=False` 时，VLM 大部分参数可训练，仅显式冻结 `lm_head`、norm、部分 text layer。
- LIBERO CI smoke train 显式设置：
  - `--policy.freeze_vision_encoder=false`
  - `--policy.train_expert_only=false`
- 位置：
  - config：`src/lerobot/policies/smolvla/configuration_smolvla.py:68-71`
  - requires_grad：`src/lerobot/policies/smolvla/smolvlm_with_expert.py:150-180`
  - CI：`.github/workflows/benchmark_tests.yml:184-189`

**IMF-AttnRes + SmolVLM**

- 默认/current 多数 SmolVLM 实验是：
  - `freeze_vlm_encoder=True`
  - `load_vlm_weights=True`
- 代码会把 `self.vlm.parameters()` 全部 `requires_grad=False` 并 `eval()`。
- 但 `state_projection` 不属于 `self.vlm`，仍可训练；在 transformer mode 且 `vlm_state_in_text_layers=True` 时，梯度可穿过 frozen text layers 回到 `state_projection`，但不会更新 VLM 权重。
- 位置：
  - config：`src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:41-45`
  - freeze：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:235-239`
  - train/eval mode：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:275-279`

**影响**

- 若 official SmolVLA 在 LIBERO 上 fine-tune 了 VLM/vision/text，而 IMF 固定 generic SmolVLM，language/visual grounding 能力会明显不同。
- 即使都冻结 VLM，SmolVLA 的 `state_proj` 和 expert glue 可能来自 pretrained SmolVLA；IMF 的 state projection 和 condition adapter 是新初始化。

---

## 2. Processor 层差异

### 2.1 Tokenizer pipeline 是否总是启用

**SmolVLA**

processor 总是：

1. rename
2. add batch dim
3. newline task
4. tokenize
5. device
6. normalize

位置：`src/lerobot/policies/smolvla/processor_smolvla.py:69-85`

**IMF-AttnRes**

只有 `config.use_smolvlm_vl_encoder=True` 时才：

- `NewLineTaskProcessorStep`
- `TokenizerProcessorStep`

位置：`src/lerobot/policies/imf_attnres/processor_imf_attnres.py:33-47`

**影响**

- IMF ResNet path 完全没有 language/task conditioning。
- IMF SmolVLM path 有 task tokens，但如果某次训练忘记打开 `use_smolvlm_vl_encoder=true`，Goal/Object 等 suite 会明显吃亏。

### 2.2 Tokenizer 参数基本一致，但 IMF 多了显式 truncation

**SmolVLA**

- `tokenizer_max_length=48`
- `pad_language_to="longest"`
- `padding_side="right"`
- 没显式传 `truncation`，使用 tokenizer processor 默认值。
- 位置：
  - `src/lerobot/policies/smolvla/configuration_smolvla.py:59-60`, `:93`
  - `src/lerobot/policies/smolvla/processor_smolvla.py:73-78`

**IMF-AttnRes**

- `vlm_tokenizer_max_length=48`
- `vlm_pad_language_to="longest"`
- `vlm_tokenizer_padding_side="right"`
- `vlm_tokenizer_truncation=True`
- 位置：
  - `src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:46-50`
  - `src/lerobot/policies/imf_attnres/processor_imf_attnres.py:40-46`

**影响**

- 默认基本对齐。
- 但 IMF 后续还会在 model 内部再次 pad/truncate 到固定 48，见下一节。

### 2.3 Language token 长度处理不同

**SmolVLA**

- 直接使用 tokenizer 输出。
- batch 内是 `padding="longest"`，所以 language sequence length 随 batch 变化。
- 位置：`src/lerobot/policies/smolvla/modeling_smolvla.py:693-700`

**IMF-AttnRes**

- 在 VLM encoder 里再次 `_pad_or_truncate_language()` 到固定 `vlm_tokenizer_max_length=48`。
- 位置：
  - `src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:303-334`
  - `src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:379-386`

**影响**

- IMF 的 condition token 数固定，便于 transformer head。
- 但 eval batch size 1 时也会有大量 masked/padded text slots；SmolVLA prefix length 更短、更动态。

---

## 3. Image/camera 处理差异

### 3.1 图像 resize path

**SmolVLA**

- 每个 camera：
  - 取当前/最后一帧
  - `resize_with_pad(..., 512, 512)`
  - `[0,1] -> [-1,1]`
- 位置：
  - `src/lerobot/policies/smolvla/configuration_smolvla.py:44-45`
  - `src/lerobot/policies/smolvla/modeling_smolvla.py:415-435`

**IMF SmolVLM path**

- stack 成 `(B, S, N, C, H, W)`
- flatten 成 `(B*S, N, C, H, W)`
- `resize_with_pad(..., 512,512)`
- `[0,1] -> [-1,1]`
- 位置：
  - `src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:336-344`
  - `src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:1213-1228`

**影响**

- 几何 resize 本身对齐。
- 但 IMF 如果 `n_obs_steps=2`，会对两帧分别编码并把 token 翻倍；SmolVLA 只用当前帧。

### 3.1.1 如果没有打开 SmolVLM path，IMF 图像处理完全不同

**IMF ResNet path**

- 默认 `vision_backbone="resnet18"`，`pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1"`，也支持 `resize_shape/crop/spatial_softmax`。
- `scripts/imf_attnres_experiments/train_eval_suite.py` 显式使用：

```bash
--policy.vision_backbone=resnet18
--policy.pretrained_backbone_weights=null
--policy.resize_shape=[128,128]
--policy.spatial_softmax_num_keypoints=32
```

- 位置：
  - config：`src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:27-37`
  - ResNet encoder：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:76-143`
  - ResNet train script：`scripts/imf_attnres_experiments/train_eval_suite.py:409-415`

**影响**

- 这条 path 没有 SigLIP/SmolVLM connector，也没有 language token；图像尺度通常是 normalized 后的 128 ResNet input，而不是 512 padded `[-1,1]` SmolVLM input。
- 如果低 success run 来自 `train_eval_suite.py`，不能解释为“只换 action head 失败”，因为视觉/语言 encoder 已完全不同。

### 3.2 Camera missing / empty camera support 不同

**SmolVLA**

- 支持 `empty_cameras`。
- 缺失 camera 时构造 `-1` dummy image，并 mask 为 0。
- 位置：
  - config 添加 empty camera feature：`src/lerobot/policies/smolvla/configuration_smolvla.py:123-130`
  - image prepare：`src/lerobot/policies/smolvla/modeling_smolvla.py:421-455`

**IMF-AttnRes**

- 会把 config 中所有 image feature stack 起来。
- 没有 SmolVLA 式 missing camera mask/dummy image。
- 要求所有 image feature shape 一致。
- 位置：
  - `src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:625-633`
  - `src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:1197-1206`
  - `src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:348-352`

**影响**

- 若和官方 `smolvla_libero` 对比，注意官方 eval 常用：

```bash
--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}
--policy.empty_cameras=1
```

见 `.github/workflows/benchmark_tests.yml:120-130`。

- 当前 IMF 训练/eval 通常直接使用 LIBERO 默认 `observation.images.image` / `image2` 两路 camera。
- 这会导致和官方 SmolVLA checkpoint 不是完全相同的 camera-key / camera-count protocol。

### 3.3 LIBERO env processor 共享，但 camera naming protocol 可能不同

LIBERO env processor 会：

- flip image 180°
- 把 raw robot state 转成 8D state `[eef_pos, axis_angle, gripper_qpos]`

位置：`src/lerobot/processor/env_processor.py:53-82`

官方 SmolVLA eval 示例在 CI 中会把 env camera 映射到 `camera1/camera2`，并额外 empty camera：

- `.github/workflows/benchmark_tests.yml:120-130`

IMF scripts 多数使用：

- `--env.camera_name=agentview_image,robot0_eye_in_hand_image`
- 默认 env mapping 到 `observation.images.image` / `image2`
- 例：`/data/lerobot-imf-attnres-exp/runs/ablation-local-swanlab/launch.sh:47-58`

**影响**

- 如果比较的是 official `lerobot/smolvla_libero`，它可能使用 3-camera config；IMF 是 2-camera config。
- 如果自己从 scratch 训练 SmolVLA，也要明确 camera keys 和 empty camera 是否和 IMF 对齐。

---

## 4. VLM / prefix / conditioning 差异

### 4.1 VLM 的使用角色不同

**SmolVLA**

- 构建 `SmolVLMWithExpertModel`。
- image/language/state 作为 prefix，action/noise 作为 suffix。
- prefix 与 suffix 在同一个 VLM+expert forward 里通过 mask 交互。
- 位置：
  - `src/lerobot/policies/smolvla/modeling_smolvla.py:567-606`
  - `src/lerobot/policies/smolvla/modeling_smolvla.py:787-804`
  - `src/lerobot/policies/smolvla/modeling_smolvla.py:830-843`

**IMF-AttnRes + SmolVLM**

- 只使用 SmolVLM vision/connector/text embedding/text layers 作为 condition encoder。
- 默认/current `vlm_conditioning_mode="flat_tokens"` 下，输出 condition tokens 后 flatten 给 IMF head。
- 当前代码也支持 `vlm_conditioning_mode="layerwise"`，会把每层 SmolVLM prefix hidden states 暴露给 IMF head；但它仍不等价于 SmolVLA 的 action suffix KV-cache / expert coupling。
- 位置：
  - `src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:129-170`
  - flat path：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:1277-1285`
  - layerwise path：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:1248-1275`
  - head layerwise coupling：`src/lerobot/policies/imf_attnres/imf_transformer1d.py:449-570`

**影响**

- 这不只是 action head 内部不同；前面 condition 进入动作生成器的方式也完全不同。
- SmolVLA 的 action suffix 可以 layer-wise attend/cross-attend 到 prefix KV；IMF 默认 flat path 只能拿到已经编码好的 condition tokens。IMF layerwise path 更接近，但仍没有复用 SmolVLA 的 KV cache / expert implementation。

### 4.2 Text transformer 使用方式不同

**SmolVLA**

- prefix 会经过 SmolVLM text layers。
- 默认 `num_vlm_layers=16`。
- 位置：
  - `src/lerobot/policies/smolvla/configuration_smolvla.py:95-97`
  - `src/lerobot/policies/smolvla/smolvlm_with_expert.py:100-103`
  - `src/lerobot/policies/smolvla/modeling_smolvla.py:797-804`

**IMF-AttnRes**

- 默认 `vlm_text_encoder_mode="embedding"`：只使用 token embeddings，不跑 text transformer layers。
- 可选 `vlm_text_encoder_mode="transformer"`：保留前 `vlm_text_num_layers=16` 层。
- 位置：
  - `src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:54-62`
  - `src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:204-237`
  - `src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:356-405`

**影响**

- 如果 IMF 实验没有显式设置 `--policy.vlm_text_encoder_mode=transformer`，那 language grounding 很可能比 SmolVLA 弱很多。
- 当前某些实验脚本已设置 transformer，例如 `ablation-local-swanlab/launch.sh:24-27`。

### 4.3 Image/language embedding scaling

**SmolVLA**

- image connector output 乘 `sqrt(hidden_dim)`。
- language embedding 也乘 `sqrt(hidden_dim)`。
- 位置：
  - image：`src/lerobot/policies/smolvla/modeling_smolvla.py:665-670`
  - language：`src/lerobot/policies/smolvla/modeling_smolvla.py:693-700`

**IMF-AttnRes**

- `transformer` mode 会在 `_encode_vlm_prefix()` 中通过 `_scale_vlm_token_embeddings()` 对 image/language token 做 `sqrt(hidden_dim)` scaling。
- `embedding` mode 走 `token_parts = [image_tokens, language_tokens]`，不会 scale。
- 位置：
  - scale 函数：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:388-390`
  - scale 调用：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:408-411`
  - embedding-mode return：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:424-429`

**影响**

- 因此如果实验仍是默认 `embedding` mode，语言/视觉 token 幅值也和 SmolVLA 不同。
- 当前 `ablation-local-swanlab/launch.sh` 设置了 `vlm_text_encoder_mode=transformer`，这条实验的 scaling 基本对齐；真正剩余 mismatch 是：`transformer` mode 虽经过 text layers，但 mask/position 行为仍不同于 SmolVLA 自定义 2D prefix-LM mask。

### 4.4 State token 处理不同

**SmolVLA**

- state 先 pad 到 `max_state_dim=32`
- `state_proj: 32 -> hidden_size`
- 只用当前 state
- 位置：
  - `src/lerobot/policies/smolvla/configuration_smolvla.py:40-42`
  - `src/lerobot/policies/smolvla/modeling_smolvla.py:484-488`
  - `src/lerobot/policies/smolvla/modeling_smolvla.py:583-585`
  - `src/lerobot/policies/smolvla/modeling_smolvla.py:704-715`

**IMF-AttnRes**

- 使用实际 `robot_state_feature.shape[0]`，LIBERO 是 8。
- 如有 env_state，则 concat 后投影。
- 每个 observation step 一个 state token。
- transformer mode 且 `vlm_state_in_text_layers=True` 时，state token 也进入 text layers。
- 位置：
  - `src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:230-233`
  - `src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:482-495`

**影响**

- SmolVLA 是 1 个 state token；IMF 默认 `n_obs_steps=2` 时是 2 个 state tokens，并且 image/language tokens 也按历史重复。
- state token 是否进入 text layers 会改变梯度路径和 condition token 语义。
- SmolVLA 的 state projection 输入是 padded 32D，适配其 pretrained policy 习惯；IMF 是真实 8D/拼 env_state 后的 projection，不能复用 SmolVLA `state_proj` 权重。

### 4.5 Prefix order 与 history flatten

**SmolVLA prefix order**

```text
camera1 image tokens
camera2 image tokens
(optional image special tokens)
language tokens
state token
```

位置：`src/lerobot/policies/smolvla/modeling_smolvla.py:637-729`

**IMF condition order**

每个 observation step：

```text
all camera image tokens
fixed 48 language tokens
state token
```

然后：

```python
(B*S, T, D) -> (B, S*T, D)
```

位置：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:1277-1285`

**影响**

- IMF 默认 2-step history 会重复 language tokens 两次。
- condition sequence length 约等于 `n_obs_steps * (num_images*tokens_per_image + 48 + 1)`，远长于 SmolVLA 单帧 prefix。
- 这改变了 condition attention 的负担，也可能稀释 action head 对关键语言 token 的关注。

### 4.5.1 IMF inference 队列还会维护语言/图像/状态历史

**SmolVLA**

- `reset()` 只维护 action queue。
- `select_action()` 每次 action queue 空时，直接用当前 batch 的 image/state/language 生成 chunk。
- 位置：`src/lerobot/policies/smolvla/modeling_smolvla.py:251-255`, `:343-350`

**IMF-AttnRes**

- `reset()` 维护 state/image/env_state/action queue；SmolVLM path 还维护 `OBS_LANGUAGE_TOKENS` 和 `OBS_LANGUAGE_ATTENTION_MASK` queue。
- `select_action()` 先 `populate_queues()`，action queue 空时通过 queued history 调 `predict_action_chunk()`。
- direct `predict_action_chunk()` 会把 current state/image/env repeat 成 history；language 若是 `(B,L)` 则在 `_prepare_smolvlm_language()` 中 repeat 到 `B*n_obs_steps`。
- 位置：
  - queue 初始化：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:615-627`
  - direct/queued predict：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:639-655`
  - select_action：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:657-669`
  - language repeat/flatten：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:1165-1212`

**影响**

- 对标准 `lerobot-eval` 来说，processor 会提供语言 token，通常不会缺 key；但 IMF 的 eval path 语义仍是“带历史队列的 VLA”，和 SmolVLA 当前帧 prefix 不同。
- 若使用 async/RTC/policy server 或直接调 `predict_action_chunk()`，要确认 batch 中语言 token/mask 与 state/image 历史同步，否则会出现条件缺失或重复当前语言 token 的行为。

### 4.6 Attention mask / position ids / KV cache 完全不同

**SmolVLA**

- 使用 `make_att_2d_masks()` 构造自定义 2D block mask。
- prefix inference 时先填 KV cache，再 action suffix denoise。
- position ids 用 `torch.cumsum(pad_masks)-1`，会跳过 padding。
- 位置：
  - mask 构造：`src/lerobot/policies/smolvla/modeling_smolvla.py:792-804`
  - inference KV cache：`src/lerobot/policies/smolvla/modeling_smolvla.py:830-843`
  - position ids：`src/lerobot/policies/smolvla/modeling_smolvla.py:796`, `:834`

**IMF-AttnRes**

- embedding mode：主要靠 masked language embedding 置零，无 SmolVLA 2D block mask。
- transformer mode：传标准 attention mask 给 HF text model。
- position ids 是简单 `torch.arange(prefix_len)`，不基于 pad mask cumsum。
- 没有 SmolVLA-style prefix KV cache 暴露给 action suffix。
- 位置：
  - zero masked language：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:379-386`
  - transformer mask/position：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:431-442`

**影响**

- 这会明显改变 image/text/state token 混合方式。
- 如果语言 padding 很多，SmolVLA position ids 与 IMF arange 行为也不同。

### 4.7 LIBERO 原始观测分辨率可能不同

**SmolVLA official / CI**

- LIBERO env config 默认 `observation_height=360`, `observation_width=360`。
- SmolVLA CI eval/train 命令没有覆盖 height/width，因此走默认 360。
- 位置：
  - env 默认：`src/lerobot/envs/configs.py:332-333`
  - CI eval/train：`.github/workflows/benchmark_tests.yml:120-130`, `:183-205`

**当前 IMF launch**

- 显式使用：

```bash
--env.observation_height=256
--env.observation_width=256
```

- 位置：`/data/lerobot-imf-attnres-exp/runs/ablation-local-swanlab/launch.sh:50-51`

**影响**

- 两边最终都会 resize/pad 到 512 给 SmolVLM，但原始 render 分辨率不同会改变细节、aliasing 和 crop/pad 前的信息量。
- 如果和 official `smolvla_libero` 对比，应把 env resolution 一起记录/对齐。

---

## 5. Action 边界接口差异（不展开 head 内部）

### 5.1 Action dim padding

**SmolVLA**

- action pad 到 `max_action_dim=32`。
- 模型输出后切回真实 action dim。
- 位置：
  - config：`src/lerobot/policies/smolvla/configuration_smolvla.py:40-42`
  - prepare action：`src/lerobot/policies/smolvla/modeling_smolvla.py:490-493`
  - unpad output：`src/lerobot/policies/smolvla/modeling_smolvla.py:297-300`

**IMF-AttnRes**

- action head 输入/输出就是真实 `action_feature.shape[0]`，LIBERO 是 7。
- 位置：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:710-713`, `:1260-1285`

**影响**

- SmolVLA 的 pretrained/action expert 习惯 32-D padded action space。
- IMF 是 exact 7-D action space，训练 target 统计和网络输入维度都不同。

### 5.2 Action queue / eval execution cadence

**SmolVLA**

- queue 只存 action。
- 一次生成 `n_action_steps=50`，逐步 pop。
- 位置：`src/lerobot/policies/smolvla/modeling_smolvla.py:251-255`, `:343-350`

**IMF-AttnRes**

- queue 存 obs history、language tokens/masks、action。
- 一次生成 `n_action_steps=8`。
- 位置：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:612-623`, `:654-666`

**影响**

- IMF 更频繁重规划，但 action chunk 更短；可能更闭环，也可能更不平滑。
- LIBERO 成功率对 gripper close/open 的时序很敏感，chunk cadence 不同会产生大差异。

---

## 6. Loss / time sampling / inference steps 差异（head-adjacent）

> 这部分和 action head 训练目标有关，不是“前处理/后处理”，但如果目标是解释 success rate 差异，必须列出。

### 6.1 SmolVLA flow matching

- noise: normal
- time: `Beta(1.5, 1.0)` 后缩放到 `[0.001,1]`
- `x_t = t*noise + (1-t)*actions`
- target velocity `u_t = noise - actions`
- loss: MSE
- inference: `num_steps=10`, Euler from noise to action
- 位置：
  - `src/lerobot/policies/smolvla/modeling_smolvla.py:621-635`
  - `src/lerobot/policies/smolvla/modeling_smolvla.py:774-810`
  - `src/lerobot/policies/smolvla/modeling_smolvla.py:812-881`

### 6.2 IMF MeanFlow / JVP objective

- time: logit-normal sampling of `(t,r)`
- data_proportion controls `r=t` 的比例
- 使用 JVP 估计 `du/dt`
- target 是 `e - x`
- loss 默认 pseudo-Huber
- 默认 action latent 是 DCT，并带 high-frequency loss weights
- optional semigroup consistency
- inference 默认 `num_inference_steps=1`，实验常设 2
- 位置：
  - config：`src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:91-124`
  - loss：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:1294-1323`
  - inference：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:1260-1285`

**影响**

- 即使前面的 VLM conditioning 完全一致，训练目标也不是 SmolVLA 的 flow matching 目标。
- DCT latent + pseudo-Huber 可能改善平滑/稳定，但和 SmolVLA 官方 success 不是直接可比。

### 6.3 `action_is_pad` loss mask 差异

**SmolVLA**

- `forward()` 中如果 batch 有 `action_is_pad`，会把 episode 越界的 action loss 置 0，并用 valid count 归一化。
- 位置：`src/lerobot/policies/smolvla/modeling_smolvla.py:380`, `:387-411`

**IMF-AttnRes**

- config 默认 `do_mask_loss_for_padding=True`，代码也支持相同思想的 mask。
- 但当前 SmolVLM 实验脚本显式：

```bash
--policy.do_mask_loss_for_padding=false
```

- 位置：
  - config：`src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:101-103`
  - loss mask 代码：`src/lerobot/policies/imf_attnres/modeling_imf_attnres.py:1318-1323`
  - 当前脚本：`/data/lerobot-imf-attnres-exp/runs/ablation-local-swanlab/launch.sh:34-35`

**为什么重要**

- LeRobot dataset 在越过 episode 边界时会 clamp 到边界帧并生成 `action_is_pad`：
  - `src/lerobot/datasets/dataset_reader.py:178-195`
- IMF 的 `drop_n_last_frames = horizon - n_action_steps - n_obs_steps + 1` 主要保证 **将要执行的 `n_action_steps`** 尽量有效；它不保证整个 `horizon` 的所有训练 target 都不 padding。
- 如果 `do_mask_loss_for_padding=false`，模型会学习一部分 “clamped tail action”，这和 SmolVLA 的训练目标不一致，尤其 `horizon=16, n_action_steps=8` 时后半段更容易受影响。
- 另外，如果 `action_latent_mode="dct"`，IMF 的 loss 是在 DCT latent 上算，再用原始时间步的 `action_is_pad` mask 去乘 latent loss。DCT 系数不是一一对应原始时间步，所以这和 SmolVLA “原始 action timestep loss mask” 也不严格等价。

**建议**

- 对齐 SmolVLA 时，把 IMF 改回：

```text
do_mask_loss_for_padding=true
```

- 或者把 `drop_n_last_frames` 设到足以保证完整 horizon 不越界，但这会进一步改变训练 frame 分布；更接近 SmolVLA 的做法是保留 mask，并在 “只换 head” 对齐实验里先设 `action_latent_mode=identity`，避免 DCT latent mask 语义差异。

---

## 7. Optimizer / scheduler / training protocol 差异

### 7.1 Optimizer type

**SmolVLA**

- `AdamWConfig`
- defaults:
  - lr `1e-4`
  - betas `(0.9,0.95)`
  - weight_decay `1e-10`
  - grad_clip_norm `10`
- 位置：
  - `src/lerobot/policies/smolvla/configuration_smolvla.py:73-82`
  - `src/lerobot/policies/smolvla/configuration_smolvla.py:132-147`

**IMF-AttnRes**

- `AdamConfig`
- defaults:
  - lr `1e-4`
  - betas `(0.95,0.999)`
  - weight_decay `1e-6`
  - grad_clip_norm `10`
  - scheduler 默认 `"none"`
- 位置：
  - `src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:127-135`
  - `src/lerobot/policies/imf_attnres/configuration_imf_attnres.py:279-301`

**影响**

- Adam vs AdamW、betas、weight decay 均不同。
- IMF 的 JVP objective 对 gradient spike 更敏感，之前经验里 grad clip 很重要。

### 7.2 当前实验脚本里的 grad clip / batch size

例如：

`/data/lerobot-imf-attnres-exp/runs/ablation-local-swanlab/launch.sh`

- `--policy.n_obs_steps=1`
- `--policy.horizon=16`
- `--policy.n_action_steps=8`
- `--policy.num_inference_steps=2`
- `--batch_size=8`
- 当前显示 `--policy.optimizer_grad_clip_norm=0`
- 位置：`launch.sh:17-45`, `:59-61`

**影响**

- `grad_clip_norm=0` 表示训练代码不会真正 clip 到有限值，只计算 infinite norm；对 IMF/JVP 可能风险很高。
- batch size 8 远小于很多 SmolVLA recipe 的 32/64；VLM conditioning + from-scratch IMF 更容易不稳定。

### 7.3 当前实验脚本和 SmolVLA 对齐程度

当前 `ablation-local-swanlab/launch.sh` 已经对齐/半对齐了一些点：

- `n_obs_steps=1`
- `use_smolvlm_vl_encoder=true`
- `vlm_text_encoder_mode=transformer`
- `load_vlm_weights=true`
- `freeze_vlm_encoder=true`

但仍未对齐：

- `horizon=16` vs SmolVLA `chunk_size=50`
- `n_action_steps=8` vs SmolVLA `50`
- `num_inference_steps=2` vs SmolVLA `num_steps=10`
- `action_latent_mode=dct` vs SmolVLA raw action flow matching
- `STATE/ACTION=MIN_MAX` vs SmolVLA `MEAN_STD`
- `optimizer_betas=(0.95,0.999)` / Adam vs SmolVLA `(0.9,0.95)` / AdamW
- `optimizer_grad_clip_norm=0` vs SmolVLA `10`
- `do_mask_loss_for_padding=false` vs SmolVLA mask padding loss
- `observation_height/width=256` vs SmolVLA CI/env 默认 `360`

位置：`/data/lerobot-imf-attnres-exp/runs/ablation-local-swanlab/launch.sh:17-45`, `:50-51`

### 7.4 SmolVLA 官方/文档训练 recipe 通常从 pretrained path 出发

SmolVLA 文档：

```bash
--policy.path=lerobot/smolvla_base
--batch_size=64
--steps=20000
```

位置：`docs/source/smolvla.mdx:56-65`

LIBERO PEFT 文档：

```bash
--policy.path=lerobot/smolvla_base
--dataset.repo_id=HuggingFaceVLA/libero
--steps=100000
--batch_size=32
--peft.method_type=LORA
```

位置：`docs/source/peft_training.mdx:17-32`

LIBERO example from scratch:

```bash
--policy.type=smolvla
--policy.load_vlm_weights=true
--steps=100000
--batch_size=4
```

位置：`docs/source/libero.mdx:131-147`

**影响**

- 如果你对比的是 official `smolvla_libero` success，那么它可能不等价于 “从 scratch 训练一个 SmolVLM encoder + 新 action head”。
- IMF 当前实验通常不是 `policy.path=lerobot/smolvla_base` 的可加载 action expert 初始化；因此 “除了 head 外的差异” 之外，还存在初始化/finetune protocol 差异。

### 7.5 `dataset.use_imagenet_stats`

IMF scripts 常设或曾常设：

```bash
--dataset.use_imagenet_stats=true
```

位置：

- `scripts/imf_attnres_experiments/train_eval_suite.py:395-400`
- `/data/lerobot-imf-attnres-exp/runs/ablation-local-swanlab/launch.sh:11-15`

SmolVLA CI smoke train 显式：

```bash
--dataset.use_imagenet_stats=false
```

位置：`.github/workflows/benchmark_tests.yml:183-192`

**影响**

- 对 ResNet IMF path，这会直接影响图像 normalization。
- VISUAL 已改为 IDENTITY 后，对 SmolVLM path 的 image tensor 不应再生效。
- 但如果 postprocessor/preprocessor 保存 stats 或某些路径没有正确使用 SmolVLM visual identity，仍可能混淆。
- 建议 SmolVLM IMF 实验也显式设 `--dataset.use_imagenet_stats=false`，减少歧义。

---

## 8. Eval protocol 差异

### 8.1 Official SmolVLA LIBERO checkpoint eval camera protocol

CI smoke eval：

```bash
--policy.path=lerobot/smolvla_libero
--env.type=libero
--env.task=libero_spatial
--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}
--policy.empty_cameras=1
```

位置：`.github/workflows/benchmark_tests.yml:120-130`

### 8.2 当前 IMF eval protocol

典型 IMF script：

```bash
--env.type=libero
--env.task=libero_object
--env.camera_name=agentview_image,robot0_eye_in_hand_image
--env.observation_height=256
--env.observation_width=256
--eval.batch_size=2
--eval.n_episodes=10
--eval.use_async_envs=true
```

位置：`/data/lerobot-imf-attnres-exp/runs/ablation-local-swanlab/launch.sh:47-58`

**影响**

- Official SmolVLA checkpoint 与 IMF checkpoint 的 eval camera key/count 可能不同。
- eval episodes 数不同：训练中常 `10 per task`，最终有些是 `50 per task`；官方报告通常要确认 protocol。
- `use_async_envs` 不同本身不应改变策略，但 seed/reset bug 与并行 init state 可能造成 protocol 差异。

---

## 9. 建议的对齐实验矩阵

为了判断 success gap 是否来自 action head 以外的细节，建议按以下顺序做 ablation：

### A. 只对齐最关键的前后处理

```text
use_smolvlm_vl_encoder=true
VISUAL=IDENTITY  # 已修
STATE=MEAN_STD
ACTION=MEAN_STD
dataset.use_imagenet_stats=false
vlm_text_encoder_mode=transformer
vlm_state_in_text_layers=true
do_mask_loss_for_padding=true
```

目的：排除 normalization 和 language/VLM path mismatch。

如果当前 low-success run 仍来自 `scripts/imf_attnres_experiments/train_eval_suite.py`，应先新建一条明确的 SmolVLM launch；否则比较的是 ResNet IMF vs SmolVLA，而不是 SmolVLM IMF vs SmolVLA。

### B. 对齐 temporal chunk

```text
n_obs_steps=1
horizon=50
n_action_steps=50
num_inference_steps=10
action_latent_mode=identity
```

目的：把行为克隆问题改到更接近 SmolVLA。

### C. 对齐 camera protocol

若和 official `smolvla_libero` 比：

```text
camera keys: camera1/camera2
empty_cameras=1 或为 IMF 实现等价 masked empty camera
rename_map 与 env.camera_name_mapping 对齐
```

目的：排除 camera count/key/order 造成的差异。

### D. 对齐训练 protocol

```text
optimizer=AdamW-like
betas=(0.9,0.95)
weight_decay=1e-10 或做网格
grad_clip_norm=5/10，不要 0
batch_size 尽量 >=32，或用 gradient accumulation
scheduler warmup/decay 对齐
```

目的：排除优化器和 batch size 导致的收敛差异。

### E. 对齐初始化 / 冻结策略

```text
# 明确比较对象：
# 1) 从 generic SmolVLM scratch 训 SmolVLA vs IMF
# 2) 从 lerobot/smolvla_base finetune SmolVLA vs IMF
# 3) released lerobot/smolvla_libero vs IMF

freeze_vlm_encoder / freeze_vision_encoder / train_expert_only 明确记录
state projection 是否 pretrained 明确记录
是否使用 PEFT / LoRA 明确记录
```

目的：避免把 pretrained SmolVLA policy 的收益误认为 action head 架构收益。

---

## 10. 结论

在修复 VISUAL 之后，剩余最值得优先排查的不是单一 bug，而是 **protocol mismatch**：

1. 先确认 run 类型：如果是 `train_eval_suite.py`，那是 ResNet/no-language IMF，和 SmolVLA 差异远超 action head。
2. SmolVLA 的官方路径是 `1 obs + 50 action chunk + MEAN_STD state/action + VLM text layers + prefix/suffix mask/KV cache`。
3. 当前 SmolVLM IMF path 多数实验是 `1/2 obs + horizon 16 + execute 8 + MIN_MAX state/action + fixed 48 language tokens + flattened/layerwise condition tokens + no SmolVLA KV cache`。
4. 当前 SmolVLM IMF 实验还存在 `do_mask_loss_for_padding=false`、`grad_clip_norm=0`、VLM冻结/初始化不等价、原始 env resolution 不同等训练 protocol 差异。
5. 这些差异都在 action head 之外或 head 接口边界上，足以让 LIBERO success rate 出现大幅差距。

如果目标是验证 “只换 action head 是否有效”，建议先做一组严格对齐 experiment：**pretrained 初始化/冻结策略、STATE/ACTION normalization、padding loss mask、n_obs/horizon/n_action_steps、num_inference_steps、VLM text mode、camera protocol、env resolution、optimizer/batch/grad clip** 全部对齐后，再比较 action head 本身。
