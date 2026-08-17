# RL-100 Diffusion RL Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated `RL/` package that applies RL-100-style offline IQL/diffusion PPO and online vector GAE/diffusion PPO to the existing state-conditioned LeRobot Diffusion Policy and Moya Newton environment, while preserving optional 2D image observations behind an explicit encoder interface.

**Architecture:** Keep LeRobot's policy, processor, dataset, and environment implementations as the system of record. Add adapters that construct decision-level transitions and stochastic DDIM traces, then implement state-based critics, state dynamics, offline/online trainers, and standard LeRobot-compatible checkpoints. Every algorithm consumes a shared dictionary observation contract; image tensors are preserved, while image-conditioned critics fail explicitly until an image feature encoder is registered.

**Tech Stack:** Python 3.12, PyTorch, diffusers DDIM schedule parameters, LeRobot v3 datasets/processors, Gymnasium vector environments, Moya Newton CUDA, pytest, uv.

---

## File Structure

```text
RL/
  __init__.py                         # package exports and version
  types.py                            # observation, decision batch, denoising trace contracts
  config.py                           # validated JSON-serializable RL configs
  checkpointing.py                    # LeRobot bundle plus RL state save/resume
  adapters/
    __init__.py
    checkpoint.py                     # policy/processors/normalization/action mask
    lerobot_v3.py                     # v3 episodes to decision transitions and labels
    moya_newton.py                    # fused CUDA env and SAME_STEP terminal state
  policy/
    __init__.py
    observation_encoder.py            # complete state encoder and image extension interface
    ddim.py                           # stochastic DDIM step and per-event log probability
    diffusion_adapter.py              # LeRobot policy trace sampling/replay
  algorithms/
    __init__.py
    iql.py                            # double-Q IQL and expectile value update
    ppo.py                            # masked denoising clipped objective
    gae.py                            # per-world variable-discount GAE
    dynamics.py                       # state ensemble, validation, OPE promotion gate
  trainers/
    __init__.py
    offline.py                        # IQL, offline PPO, optional dynamics/OPE loop
    online.py                         # vector rollout buffer, GAE, online PPO loop
  cli/
    __init__.py
    inspect_dataset.py
    train_offline.py
    train_online.py
  README.md
  MIGRATION.md
  NOTICE
tests/rl100/
  test_types_and_config.py
  test_lerobot_v3_adapter.py
  test_checkpoint_adapter.py
  test_observation_encoder.py
  test_ddim_trace.py
  test_iql.py
  test_ppo.py
  test_gae.py
  test_dynamics.py
  test_checkpointing.py
  test_moya_adapter.py
  test_offline_trainer.py
  test_online_trainer.py
```

## Task 1: Package Contracts And Configuration

**Files:**
- Create: `RL/__init__.py`
- Create: `RL/types.py`
- Create: `RL/config.py`
- Create: `RL/adapters/__init__.py`
- Create: `RL/policy/__init__.py`
- Create: `RL/algorithms/__init__.py`
- Create: `RL/trainers/__init__.py`
- Create: `RL/cli/__init__.py`
- Test: `tests/rl100/test_types_and_config.py`

- [ ] **Step 1: Write failing shape and config tests**

```python
import pytest
import torch

from RL.config import RLConfig, TraceConfig
from RL.types import DecisionBatch, ObservationBatch


def test_decision_batch_validates_shapes_and_preserves_images():
    obs = ObservationBatch(
        features={
            "observation.state": torch.zeros(2, 2, 39),
            "observation.images.front": torch.zeros(2, 2, 3, 16, 16),
        }
    )
    batch = DecisionBatch(
        observation=obs,
        next_observation=obs,
        action=torch.zeros(2, 32, 14),
        action_valid=torch.ones(2, 32, dtype=torch.bool),
        reward=torch.zeros(2, 1),
        done=torch.zeros(2, 1, dtype=torch.bool),
        discount=torch.full((2, 1), 0.99**32),
    )
    batch.validate(state_dim=39, action_dim=14, chunk_size=32, n_obs_steps=2)
    assert "observation.images.front" in batch.observation.features


def test_trace_config_rejects_zero_variance():
    with pytest.raises(ValueError, match="sigma_min"):
        TraceConfig(sigma_min=0.0)


def test_config_json_round_trip(tmp_path):
    config = RLConfig()
    path = tmp_path / "rl_config.json"
    config.save_json(path)
    assert RLConfig.load_json(path) == config
```

- [ ] **Step 2: Run the tests and verify missing-module failure**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_types_and_config.py -q`

Expected: collection fails because `RL.types` and `RL.config` do not exist.

- [ ] **Step 3: Implement immutable contracts and validated dataclass configs**

Use these public signatures:

```python
@dataclass(frozen=True)
class ObservationBatch:
    features: dict[str, torch.Tensor]

    def batch_size(self) -> int:
        sizes = {tensor.shape[0] for tensor in self.features.values()}
        if len(sizes) != 1:
            raise ValueError(f"observation batch sizes disagree: {sorted(sizes)}")
        return sizes.pop()

    def to(self, device: torch.device | str) -> "ObservationBatch":
        return ObservationBatch({key: value.to(device) for key, value in self.features.items()})

    def index_select(self, indices: torch.Tensor) -> "ObservationBatch":
        return ObservationBatch({key: value.index_select(0, indices) for key, value in self.features.items()})


@dataclass(frozen=True)
class DecisionBatch:
    observation: ObservationBatch
    next_observation: ObservationBatch
    action: torch.Tensor
    action_valid: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    discount: torch.Tensor

    def validate(self, *, state_dim: int, action_dim: int, chunk_size: int, n_obs_steps: int) -> None:
        state = self.observation.features["observation.state"]
        expected_state = (self.action.shape[0], n_obs_steps, state_dim)
        if state.shape != expected_state:
            raise ValueError(f"observation.state must have shape {expected_state}, got {tuple(state.shape)}")
        if self.action.shape[1:] != (chunk_size, action_dim):
            raise ValueError(f"action must end in {(chunk_size, action_dim)}, got {tuple(self.action.shape)}")
        if self.action_valid.shape != self.action.shape[:2]:
            raise ValueError("action_valid must match action batch and chunk dimensions")

    def to(self, device: torch.device | str) -> "DecisionBatch":
        return dataclasses.replace(
            self,
            observation=self.observation.to(device),
            next_observation=self.next_observation.to(device),
            action=self.action.to(device),
            action_valid=self.action_valid.to(device),
            reward=self.reward.to(device),
            done=self.done.to(device),
            discount=self.discount.to(device),
        )


@dataclass(frozen=True)
class DenoisingTrace:
    latents: torch.Tensor
    next_latents: torch.Tensor
    timesteps: torch.Tensor
    old_log_prob: torch.Tensor
    final_actions: torch.Tensor


@dataclass(frozen=True)
class TraceConfig:
    num_inference_steps: int = 10
    eta: float = 1.0
    sigma_min: float = 0.0067
    sigma_max: float = 0.1


@dataclass(frozen=True)
class RLConfig:
    trace: TraceConfig = field(default_factory=TraceConfig)
    state_key: str = "observation.state"
    n_obs_steps: int = 2
    state_dim: int = 39
    action_dim: int = 14
    chunk_size: int = 32
    gamma: float = 0.99
```

All validators must raise `ValueError` with the field name and actual value. JSON serialization must be deterministic and reject unknown fields.

- [ ] **Step 4: Run contract tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_types_and_config.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the reviewed task**

```bash
git add RL/__init__.py RL/types.py RL/config.py RL/adapters RL/policy RL/algorithms RL/trainers RL/cli tests/rl100/test_types_and_config.py
git commit -m "feat(rl): add RL-100 migration contracts"
```

## Task 2: LeRobot V3 Sparse-Reward Adapter

**Files:**
- Create: `RL/adapters/lerobot_v3.py`
- Create: `RL/cli/inspect_dataset.py`
- Test: `tests/rl100/test_lerobot_v3_adapter.py`

- [ ] **Step 1: Write failing reward and chunk-boundary tests**

```python
def test_terminal_success_uses_fifteen_millimeters():
    accepted = episode_metadata(final_lift_height_m=0.015)
    rejected = episode_metadata(final_lift_height_m=0.014999)
    assert terminal_success(accepted, min_final_lift_height_m=0.015)
    assert not terminal_success(rejected, min_final_lift_height_m=0.015)


def test_build_decisions_adds_only_terminal_sparse_reward():
    episode = fake_episode(length=35, state_dim=39, action_dim=14)
    decisions = build_episode_decisions(
        episode,
        success=True,
        n_obs_steps=2,
        chunk_size=32,
        gamma=0.99,
    )
    assert len(decisions) == 2
    assert decisions[0].reward.item() == 0.0
    assert decisions[1].reward.item() == 1.0
    assert decisions[0].action_valid.sum().item() == 32
    assert decisions[1].action_valid.sum().item() == 3
    assert decisions[1].done.item()


def test_image_keys_survive_decision_construction():
    episode = fake_episode(length=3, include_image=True)
    decision = build_episode_decisions(episode, success=False, n_obs_steps=2, chunk_size=2, gamma=0.99)[0]
    assert "observation.images.front" in decision.observation.features
```

- [ ] **Step 2: Run the adapter tests and verify failure**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_lerobot_v3_adapter.py -q`

Expected: import failure for `RL.adapters.lerobot_v3`.

- [ ] **Step 3: Implement summary validation and decision construction**

Public API:

```python
def terminal_success(metadata: Mapping[str, Any], *, min_final_lift_height_m: float = 0.015) -> bool:
    required = {
        "true_grasp_ever",
        "clear_table_ever",
        "final_lift_height_m",
        "final_table_contacts",
        "final_hand_contacts",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"episode metadata is missing {sorted(missing)}")
    return bool(
        metadata["true_grasp_ever"]
        and metadata["clear_table_ever"]
        and float(metadata["final_lift_height_m"]) >= min_final_lift_height_m
        and int(metadata["final_table_contacts"]) == 0
        and int(metadata["final_hand_contacts"]) > 0
    )

def load_episode_labels(summary_path: Path, *, expected_episode_count: int) -> dict[int, bool]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    records = payload.get("episodes", [])
    labels = {
        int(record["episode_index"]): terminal_success(record)
        for record in records
    }
    if len(records) != len(labels):
        raise ValueError("collection summary contains duplicate episode_index values")
    if sorted(labels) != list(range(expected_episode_count)):
        raise ValueError("collection summary episode indices do not match the dataset")
    return labels

def build_episode_decisions(
    episode: Mapping[str, torch.Tensor],
    *,
    success: bool,
    n_obs_steps: int,
    chunk_size: int,
    gamma: float,
) -> list[DecisionBatch]:
    states = episode["observation.state"]
    actions = episode["action"]
    decisions = []
    for start in range(0, states.shape[0], chunk_size):
        stop = min(start + chunk_size, states.shape[0])
        valid_steps = stop - start
        history_indices = torch.arange(start - n_obs_steps + 1, start + 1).clamp_min(0)
        next_index = min(stop, states.shape[0] - 1)
        next_history_indices = torch.arange(next_index - n_obs_steps + 1, next_index + 1).clamp_min(0)
        chunk = actions[start:stop]
        padded = torch.cat([chunk, chunk[-1:].expand(chunk_size - valid_steps, -1)], dim=0)
        valid = torch.arange(chunk_size) < valid_steps
        terminal = stop == states.shape[0]
        decisions.append(
            DecisionBatch(
                observation=ObservationBatch({"observation.state": states[history_indices].unsqueeze(0)}),
                next_observation=ObservationBatch({"observation.state": states[next_history_indices].unsqueeze(0)}),
                action=padded.unsqueeze(0),
                action_valid=valid.unsqueeze(0),
                reward=torch.tensor([[float(success and terminal)]], dtype=torch.float32),
                done=torch.tensor([[terminal]], dtype=torch.bool),
                discount=torch.tensor([[gamma**valid_steps]], dtype=torch.float32),
            )
        )
    return decisions
```

`LeRobotV3DecisionDataset.from_root()` applies this function to every episode returned by `LeRobotDataset`, concatenates the decision records, and exposes their label/chunk statistics through an `inspection_summary()` method.

Use episode indices from LeRobot metadata, not row adjacency alone. Left-pad initial observation history with the first observation and right-pad final action chunks with the last valid action while setting `action_valid=False`. Summary mismatches, duplicate episode labels, missing acceptance fields, non-finite tensors, and non-monotonic frame indices must fail.

- [ ] **Step 4: Implement the inspection CLI**

The CLI prints JSON containing dataset path, episodes, frames, decisions, positive labels, negative labels, state/action shapes, image keys, and partial-chunk count. It exits nonzero if observed shapes differ from config.

- [ ] **Step 5: Run unit tests and real-data inspection**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_lerobot_v3_adapter.py -q
UV_CACHE_DIR=.uv-cache uv run python -m RL.cli.inspect_dataset \
  --dataset-root /home/droid/project/Moya_newton_sim/.worktrees/feat-fused-batched-env/runs/lerobot/randomized_grasp_100_20260813-230331/dataset \
  --repo-id moya_newton/randomized_grasp_100 \
  --summary /home/droid/project/Moya_newton_sim/.worktrees/feat-fused-batched-env/runs/lerobot/randomized_grasp_100_20260813-230331/collection_summary.json
```

Expected JSON fields: `episodes=100`, `frames=93000`, `positive_labels=94`, `negative_labels=6`, `state_shape=[39]`, `action_shape=[14]`.

- [ ] **Step 6: Commit the reviewed task**

```bash
git add RL/adapters/lerobot_v3.py RL/cli/inspect_dataset.py tests/rl100/test_lerobot_v3_adapter.py
git commit -m "feat(rl): adapt LeRobot v3 sparse rewards"
```

## Task 3: Checkpoint Processors And Observation Encoders

**Files:**
- Create: `RL/adapters/checkpoint.py`
- Create: `RL/policy/observation_encoder.py`
- Test: `tests/rl100/test_checkpoint_adapter.py`
- Test: `tests/rl100/test_observation_encoder.py`

- [ ] **Step 1: Write failing processor and encoder tests**

```python
def test_checkpoint_adapter_normalization_round_trip(real_checkpoint):
    adapter = CheckpointAdapter.load(real_checkpoint, device="cpu")
    raw = torch.tensor([[0.01] * 14])
    normalized = adapter.normalize_action(raw)
    restored = adapter.unnormalize_action(normalized)
    torch.testing.assert_close(restored, raw)


def test_active_action_mask_uses_saved_ranges(real_checkpoint):
    adapter = CheckpointAdapter.load(real_checkpoint, device="cpu")
    assert adapter.active_action_indices.tolist() == [0, 1, 2, 12, 13]


def test_state_encoder_outputs_fixed_feature_size():
    encoder = StateFeatureEncoder(state_dim=39, n_obs_steps=2, hidden_dims=(128, 128), output_dim=128)
    output = encoder(ObservationBatch({"observation.state": torch.zeros(4, 2, 39)}))
    assert output.shape == (4, 128)


def test_images_require_registered_feature_encoder():
    encoder = StateFeatureEncoder(state_dim=39, n_obs_steps=2, hidden_dims=(64,), output_dim=32)
    observation = ObservationBatch({
        "observation.state": torch.zeros(1, 2, 39),
        "observation.images.front": torch.zeros(1, 2, 3, 16, 16),
    })
    with pytest.raises(ImageEncoderRequiredError, match="observation.images.front"):
        encoder(observation)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_checkpoint_adapter.py tests/rl100/test_observation_encoder.py -q`

Expected: missing adapter/encoder modules.

- [ ] **Step 3: Implement complete checkpoint loading and processor access**

`CheckpointAdapter.load()` must call `PreTrainedConfig.from_pretrained`, `DiffusionPolicy.from_pretrained`, and `make_pre_post_processors`. It finds the loaded `NormalizerProcessorStep` and `UnnormalizerProcessorStep`, then normalizes batched histories/actions without the inference pipeline's add-batch step. The action mask is `(action.max - action.min) > tolerance`; it must contain at least one active dimension.

Public API:

```python
@dataclass
class CheckpointAdapter:
    policy: DiffusionPolicy
    preprocessor: PolicyProcessorPipeline
    postprocessor: PolicyProcessorPipeline
    active_action_mask: torch.Tensor
```

The methods exercised by the tests are `load`, `normalize_observation`, `normalize_action`, `unnormalize_action`, and `processor_fingerprint`; each delegates tensor transforms to the loaded normalization processor rather than reimplementing checkpoint statistics.

- [ ] **Step 4: Implement state encoder and image extension contract**

```python
class ObservationFeatureEncoder(nn.Module, ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, observation: ObservationBatch) -> torch.Tensor:
        raise NotImplementedError


class StateFeatureEncoder(ObservationFeatureEncoder):
    """MLP over flattened state history; rejects unhandled image keys."""
```

Allow a future `image_encoder: ObservationFeatureEncoder | None` to be composed with the state encoder. Do not implement an RGB backbone in this task.

- [ ] **Step 5: Run tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_checkpoint_adapter.py tests/rl100/test_observation_encoder.py -q`

Expected: all tests pass, including real checkpoint processor loading.

- [ ] **Step 6: Commit the reviewed task**

```bash
git add RL/adapters/checkpoint.py RL/policy/observation_encoder.py tests/rl100/test_checkpoint_adapter.py tests/rl100/test_observation_encoder.py
git commit -m "feat(rl): load LeRobot diffusion checkpoints"
```

## Task 4: Stochastic DDIM Trace And PPO Objective

**Files:**
- Create: `RL/policy/ddim.py`
- Create: `RL/policy/diffusion_adapter.py`
- Create: `RL/algorithms/ppo.py`
- Test: `tests/rl100/test_ddim_trace.py`
- Test: `tests/rl100/test_ppo.py`

- [ ] **Step 1: Write failing DDIM numerical tests**

```python
def test_stochastic_ddim_replay_has_unit_ratio(tiny_diffusion_adapter):
    observation = ObservationBatch({"observation.state": torch.zeros(2, 2, 39)})
    trace = tiny_diffusion_adapter.sample_trace(observation, generator=torch.Generator().manual_seed(7))
    replayed = tiny_diffusion_adapter.recompute_log_prob(observation, trace)
    torch.testing.assert_close(replayed, trace.old_log_prob, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(replayed).all()


def test_masked_log_probability_excludes_padding_and_constant_dimensions():
    log_prob = torch.ones(3, 2, 4, 5)
    step_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    dim_mask = torch.tensor([1, 0, 1, 0, 0], dtype=torch.bool)
    reduced = reduce_event_log_prob(log_prob, step_mask=step_mask, action_dim_mask=dim_mask)
    assert reduced[:, 0].tolist() == [4.0, 4.0, 4.0]
    assert reduced[:, 1].tolist() == [6.0, 6.0, 6.0]


def test_clipped_loss_backpropagates_to_unet(tiny_diffusion_adapter):
    trace, observation = trace_fixture(tiny_diffusion_adapter)
    new_log_prob = tiny_diffusion_adapter.recompute_log_prob(observation, trace)
    loss, metrics = denoising_ppo_loss(
        new_log_prob,
        trace.old_log_prob,
        torch.ones(2),
        step_mask=torch.ones(2, 32, dtype=torch.bool),
        action_dim_mask=torch.ones(14, dtype=torch.bool),
        clip_ratio=0.2,
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in tiny_diffusion_adapter.policy.diffusion.unet.parameters())
    assert metrics["ratio_mean"] == pytest.approx(1.0, rel=1e-5)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_ddim_trace.py tests/rl100/test_ppo.py -q`

Expected: missing DDIM/adapter/PPO modules.

- [ ] **Step 3: Implement a current-diffusers stochastic DDIM transition**

Use explicit `(timestep, previous_timestep)` schedule entries rather than assuming fixed integer spacing. Support checkpoint prediction types `epsilon` and `sample`, checkpoint clipping, and the configured positive sigma bounds. Return per-event log probability with shape `[batch, horizon, action_dim]`:

```python
@dataclass(frozen=True)
class DDIMStepOutput:
    previous_sample: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    log_prob: torch.Tensor


def stochastic_ddim_step(
    *, scheduler: DDIMScheduler, model_output: torch.Tensor,
    timestep: int, previous_timestep: int | None, sample: torch.Tensor,
    eta: float, sigma_min: float, sigma_max: float,
    previous_sample: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> DDIMStepOutput:
    alpha_t = scheduler.alphas_cumprod[timestep].to(sample)
    alpha_previous = (
        scheduler.final_alpha_cumprod.to(sample)
        if previous_timestep is None
        else scheduler.alphas_cumprod[previous_timestep].to(sample)
    )
    beta_t = 1.0 - alpha_t
    if scheduler.config.prediction_type == "epsilon":
        predicted_clean = (sample - beta_t.sqrt() * model_output) / alpha_t.sqrt()
        predicted_noise = model_output
    elif scheduler.config.prediction_type == "sample":
        predicted_clean = model_output
        predicted_noise = (sample - alpha_t.sqrt() * predicted_clean) / beta_t.sqrt()
    else:
        raise ValueError(f"unsupported prediction_type={scheduler.config.prediction_type!r}")
    if scheduler.config.clip_sample:
        predicted_clean = predicted_clean.clamp(
            -scheduler.config.clip_sample_range,
            scheduler.config.clip_sample_range,
        )
    variance = ((1.0 - alpha_previous) / beta_t) * (1.0 - alpha_t / alpha_previous)
    std = (eta * variance.clamp_min(0).sqrt()).clamp(min=sigma_min, max=sigma_max)
    direction = (1.0 - alpha_previous - std.square()).clamp_min(0).sqrt() * predicted_noise
    mean = alpha_previous.sqrt() * predicted_clean + direction
    if previous_sample is None:
        noise = torch.randn(sample.shape, dtype=sample.dtype, device=sample.device, generator=generator)
        previous_sample = mean + std * noise
    log_prob = torch.distributions.Normal(mean, std).log_prob(previous_sample.detach())
    return DDIMStepOutput(previous_sample, mean, std, log_prob)
```

The implementation body computes `alpha_t`, `alpha_previous`, predicted clean action, clipped clean action, DDIM variance, bounded standard deviation, transition mean, sampled or supplied previous sample, and `Normal(mean, std).log_prob(previous_sample.detach())`, then returns all four tensors in `DDIMStepOutput`.

Detach only the sampled/stored `previous_sample` in the log-probability residual; preserve gradients through the current mean. Reject non-finite output immediately.

- [ ] **Step 4: Implement LeRobot diffusion trace sampling and replay**

`DiffusionRLAdapter` receives `CheckpointAdapter` and `TraceConfig`. It creates a private `DDIMScheduler` from the policy beta/prediction/clipping config, prepares state plus optional checkpoint image conditioning, and records the full trace. Executable actions are indices `[n_obs_steps - 1 : n_obs_steps - 1 + n_action_steps]`.

- [ ] **Step 5: Implement masked PPO math**

```python
def reduce_event_log_prob(log_prob, *, step_mask, action_dim_mask) -> torch.Tensor:
    mask = step_mask.unsqueeze(0).unsqueeze(-1) & action_dim_mask.view(1, 1, 1, -1)
    return (log_prob * mask).sum(dim=(-1, -2))

def denoising_ppo_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    *, step_mask: torch.Tensor,
    action_dim_mask: torch.Tensor,
    clip_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    old_reduced = reduce_event_log_prob(old_log_prob, step_mask=step_mask, action_dim_mask=action_dim_mask)
    new_reduced = reduce_event_log_prob(new_log_prob, step_mask=step_mask, action_dim_mask=action_dim_mask)
    ratio = torch.exp(new_reduced - old_reduced)
    expanded_advantage = advantage.reshape(1, -1).expand_as(ratio)
    unclipped = ratio * expanded_advantage
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * expanded_advantage
    loss = -torch.minimum(unclipped, clipped).mean()
    metrics = {
        "ratio_mean": float(ratio.detach().mean()),
        "clip_fraction": float(((ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)).float().mean()),
        "approx_kl": float((old_reduced.detach() - new_reduced.detach()).mean()),
    }
    return loss, metrics
```

Sum event dimensions, compute one ratio per denoising sub-policy and sample, clip ratios, average across samples and denoising steps, and report finite ratio/clip/KL statistics.

- [ ] **Step 6: Run numerical tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_ddim_trace.py tests/rl100/test_ppo.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit the reviewed task**

```bash
git add RL/policy/ddim.py RL/policy/diffusion_adapter.py RL/algorithms/ppo.py tests/rl100/test_ddim_trace.py tests/rl100/test_ppo.py
git commit -m "feat(rl): add stochastic diffusion PPO traces"
```

## Task 5: IQL, GAE, State Dynamics, And OPE

**Files:**
- Create: `RL/algorithms/iql.py`
- Create: `RL/algorithms/gae.py`
- Create: `RL/algorithms/dynamics.py`
- Test: `tests/rl100/test_iql.py`
- Test: `tests/rl100/test_gae.py`
- Test: `tests/rl100/test_dynamics.py`

- [ ] **Step 1: Write failing algorithm tests**

```python
def test_iql_update_and_advantage_are_finite(decision_batch):
    encoder = StateFeatureEncoder(state_dim=39, n_obs_steps=2, hidden_dims=(64,), output_dim=32)
    iql = IQL(
        feature_encoder=encoder,
        action_dim=14,
        chunk_size=4,
        active_action_mask=torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.bool),
        hidden_dims=(64, 64),
        expectile=0.7,
        tau=0.005,
        q_lr=3e-4,
        v_lr=3e-4,
    )
    metrics = iql.update(decision_batch)
    advantage = iql.advantage(decision_batch.observation, decision_batch.action, decision_batch.action_valid)
    assert advantage.shape == (decision_batch.action.shape[0], 1)
    assert all(math.isfinite(value) for value in metrics.values())


def test_vector_gae_does_not_cross_worlds():
    reward = torch.tensor([[[1.0], [0.0]], [[0.0], [2.0]]])
    done = torch.tensor([[[True], [False]], [[False], [True]]])
    result = compute_vector_gae(
        reward=reward,
        value=torch.zeros_like(reward),
        next_value=torch.zeros_like(reward),
        done=done,
        discount=torch.full_like(reward, 0.9),
        gae_lambda=0.95,
    )
    assert result.advantage[0, 0].item() == 1.0
    assert result.advantage[0, 1].item() > 0.0


def test_promotion_gate_requires_valid_dynamics_and_margin():
    gate = PolicyPromotionGate(relative_margin=0.05, max_validation_loss=0.1)
    assert not gate.decide(candidate_return=1.1, behavior_return=1.0, critic_return=0.0, dynamics_validation_loss=0.2).promote
    assert gate.decide(candidate_return=1.1, behavior_return=1.0, critic_return=0.0, dynamics_validation_loss=0.05).promote
```

- [ ] **Step 2: Run tests and verify failure**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_iql.py tests/rl100/test_gae.py tests/rl100/test_dynamics.py -q`

Expected: missing algorithm modules.

- [ ] **Step 3: Implement action packing and double-Q IQL**

Pack only active action dimensions, zero padded action steps, and append the valid-step mask to the Q input. Implement two Q MLPs, target Q copies, a value MLP, expectile regression, TD target `reward + discount * (1 - done) * V(next_state)`, Polyak updates, independent optimizers, gradient clipping, and normalized detached `Q - V` advantages.

- [ ] **Step 4: Implement per-world variable-discount GAE**

```python
@dataclass(frozen=True)
class GAEResult:
    advantage: torch.Tensor
    returns: torch.Tensor


def compute_vector_gae(*, reward, value, next_value, done, discount, gae_lambda) -> GAEResult:
    delta = reward + discount * (~done).to(reward.dtype) * next_value - value
    advantage = torch.zeros_like(delta)
    accumulator = torch.zeros_like(delta[0])
    for index in range(delta.shape[0] - 1, -1, -1):
        alive = (~done[index]).to(delta.dtype)
        accumulator = delta[index] + discount[index] * gae_lambda * alive * accumulator
        advantage[index] = accumulator
    return GAEResult(advantage=advantage, returns=advantage + value)
```

Iterate time in reverse while maintaining one accumulator per environment. Validate `[time, env, 1]` shapes.

- [ ] **Step 5: Implement state dynamics and OPE gate**

The ensemble consumes encoded state plus packed action and predicts next encoded state delta, reward, and termination logit. Train each member on a deterministic bootstrap index set, expose held-out validation loss and disagreement, and refuse OPE when validation is above threshold. `PolicyPromotionGate` applies the configured relative margin and returns a structured decision with every compared value.

- [ ] **Step 6: Run algorithm tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_iql.py tests/rl100/test_gae.py tests/rl100/test_dynamics.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit the reviewed task**

```bash
git add RL/algorithms/iql.py RL/algorithms/gae.py RL/algorithms/dynamics.py tests/rl100/test_iql.py tests/rl100/test_gae.py tests/rl100/test_dynamics.py
git commit -m "feat(rl): add state IQL GAE and dynamics"
```

## Task 6: Checkpointing And Offline Trainer

**Files:**
- Create: `RL/checkpointing.py`
- Create: `RL/trainers/offline.py`
- Create: `RL/cli/train_offline.py`
- Test: `tests/rl100/test_checkpointing.py`
- Test: `tests/rl100/test_offline_trainer.py`

- [ ] **Step 1: Write failing checkpoint and offline-step tests**

```python
def test_checkpoint_round_trip_preserves_standard_bundle(tmp_path, tiny_offline_components):
    saved = save_rl_checkpoint(tmp_path, **tiny_offline_components)
    assert (saved / "pretrained_model" / "model.safetensors").is_file()
    assert (saved / "rl_state.pt").is_file()
    restored = load_rl_checkpoint(saved, device="cpu")
    assert restored.provenance.processor_fingerprint == tiny_offline_components["processor_fingerprint"]


def test_offline_smoke_step_updates_iql_and_policy(tiny_offline_trainer, decision_batch):
    before = clone_parameters(tiny_offline_trainer.policy)
    metrics = tiny_offline_trainer.train_step(decision_batch)
    assert metrics["iql/q_loss"] >= 0.0
    assert math.isfinite(metrics["actor/loss"])
    assert parameters_changed(before, tiny_offline_trainer.policy)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_checkpointing.py tests/rl100/test_offline_trainer.py -q`

Expected: missing checkpoint/trainer modules.

- [ ] **Step 3: Implement atomic checkpoint persistence and validation**

Copy the source standard bundle into a temporary output directory, overwrite policy weights with `save_pretrained`, retain processor files, write RL state/config/provenance/metrics, fsync files, and atomically rename. Provenance includes base model SHA-256, processor fingerprint, dataset/summary paths, feature keys, dimensions, chunk size, action mask, and source RL-100 commit. Resume checks every immutable field before loading state dicts.

- [ ] **Step 4: Implement offline training orchestration**

`OfflineTrainer` owns current/old diffusion adapters, IQL, optional dynamics, optimizers, metric JSONL, and checkpoints. One actor update samples a no-grad old trace, evaluates normalized IQL advantage on its executable chunk, replays each denoising step with the current U-Net, accumulates divided loss gradients, clips once, steps once, and refreshes the old policy only according to its update/promotion schedule.

- [ ] **Step 5: Implement offline CLI and smoke mode**

Required flags: checkpoint, dataset root, repo ID, summary, output directory, device, IQL steps, actor steps, inference steps, seed, and `--smoke`. Smoke resolves to one IQL step, one actor step, batch size two, and two denoising steps; it never enables a long run accidentally.

- [ ] **Step 6: Run tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_checkpointing.py tests/rl100/test_offline_trainer.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit the reviewed task**

```bash
git add RL/checkpointing.py RL/trainers/offline.py RL/cli/train_offline.py tests/rl100/test_checkpointing.py tests/rl100/test_offline_trainer.py
git commit -m "feat(rl): add offline diffusion RL trainer"
```

## Task 7: Moya Adapter And Online Trainer

**Files:**
- Create: `RL/adapters/moya_newton.py`
- Create: `RL/trainers/online.py`
- Create: `RL/cli/train_online.py`
- Test: `tests/rl100/test_moya_adapter.py`
- Test: `tests/rl100/test_online_trainer.py`

- [ ] **Step 1: Write failing terminal-state and online-update tests**

```python
def test_terminal_next_state_uses_final_obs():
    reset_state = np.full((2, 39), 9.0, dtype=np.float32)
    final_state = np.full((2, 39), 3.0, dtype=np.float32)
    selected = select_transition_next_state(reset_state, {"final_obs": final_state}, np.array([True, False]))
    assert np.all(selected[0] == 3.0)
    assert np.all(selected[1] == 9.0)


def test_online_buffer_and_update_are_finite(tiny_online_trainer, fake_vector_env):
    batch = tiny_online_trainer.collect(fake_vector_env, decisions=2)
    assert batch.time_steps == 2
    metrics = tiny_online_trainer.update(batch)
    assert math.isfinite(metrics["actor/loss"])
    assert math.isfinite(metrics["value/loss"])
```

- [ ] **Step 2: Run tests and verify failure**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_moya_adapter.py tests/rl100/test_online_trainer.py -q`

Expected: missing Moya/online modules.

- [ ] **Step 3: Implement Moya environment creation and terminal transition extraction**

Create the environment only through `MoyaNewtonEnvConfig` and `make_env`, preserving contract validation and 15-millimeter success. `select_transition_next_state()` handles array and collated info forms, validates final shape `(num_envs, 39)`, and substitutes terminal rows only. No rendering or video dependency is permitted.

- [ ] **Step 4: Implement vector rollout storage**

Store decision-major, env-minor observations, traces, old log probabilities, raw executed actions, valid-step masks, accumulated reward, selected next state, done, discount, and success. If an environment terminates partway through a chunk, stop adding reward/action validity for it while other worlds continue. Reset policy observation histories only for done worlds.

- [ ] **Step 5: Implement online GAE/PPO/value update**

Compute value and next value on state features, call vector GAE, normalize valid advantages, replay stored traces in minibatches, apply the same masked denoising PPO objective, update the value network with clipped gradients, then sync the rollout policy. Log interaction steps separately from optimization steps.

- [ ] **Step 6: Implement online CLI and smoke mode**

Required flags: input RL/base checkpoint, output directory, policy device, sim device, num envs, rollout decisions, PPO epochs, inference steps, seed, and `--smoke`. Smoke fixes 16 worlds, two denoising steps, one decision, one PPO epoch, headless mode, and no video.

- [ ] **Step 7: Run tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_moya_adapter.py tests/rl100/test_online_trainer.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit the reviewed task**

```bash
git add RL/adapters/moya_newton.py RL/trainers/online.py RL/cli/train_online.py tests/rl100/test_moya_adapter.py tests/rl100/test_online_trainer.py
git commit -m "feat(rl): add Newton online diffusion RL"
```

## Task 8: Documentation, Attribution, And Real Smoke Tests

**Files:**
- Create: `RL/README.md`
- Create: `RL/MIGRATION.md`
- Create: `RL/NOTICE`
- Modify: only implementation files required by smoke defects

- [ ] **Step 1: Document exact commands and limitations**

`README.md` must provide environment prerequisites, dataset inspection, offline smoke/train, online smoke/train, resumption, output layout, and standard LeRobot evaluation commands. It states that state RL is complete, actor image tensors are preserved, and an image critic encoder is not implemented.

- [ ] **Step 2: Document source-to-target migration changes**

`MIGRATION.md` maps RL-100 `critic.py`, `uni_ppo.py`, diffusion log-prob patch, online buffers, dynamics, and workspace stages to target modules. For each mapping, explain removal of point-cloud assumptions, LeRobot processor usage, decision chunks, sparse terminal reward, constant-action masking, stochastic DDIM adaptation, SAME_STEP handling, Python 3.12/uv, and standard checkpoint output. State exclusions: DP3/PointNet, flow, distillation, old runners, real-robot drivers, and automatic image critic features.

- [ ] **Step 3: Add Apache attribution**

`NOTICE` identifies RL-100, its source commit, Apache-2.0 license, the rewritten/adapted modules, and inherited diffusers/LeRobot attribution. Preserve any copied source header in the corresponding file.

- [ ] **Step 4: Run the focused CPU suite**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100 -q
UV_CACHE_DIR=.uv-cache uv run pytest tests/policies/test_diffusion_state_only.py tests/envs/test_moya_newton.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run real checkpoint and dataset smoke**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m RL.cli.train_offline \
  --checkpoint outputs/train/moya_diffusion_300k_20260815-093537/train/checkpoints/080000/pretrained_model \
  --dataset-root /home/droid/project/Moya_newton_sim/.worktrees/feat-fused-batched-env/runs/lerobot/randomized_grasp_100_20260813-230331/dataset \
  --repo-id moya_newton/randomized_grasp_100 \
  --summary /home/droid/project/Moya_newton_sim/.worktrees/feat-fused-batched-env/runs/lerobot/randomized_grasp_100_20260813-230331/collection_summary.json \
  --output-dir outputs/rl100/offline_smoke \
  --device cuda \
  --smoke
```

Expected: label split 94/6, finite IQL and actor metrics, and reloadable `pretrained_model` output.

- [ ] **Step 6: Run CUDA Newton online smoke**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m RL.cli.train_online \
  --checkpoint outputs/rl100/offline_smoke \
  --output-dir outputs/rl100/online_smoke \
  --device cuda \
  --sim-device cuda:0 \
  --num-envs 16 \
  --smoke
```

Expected: Moya contract passes, 16 CUDA worlds collect one decision, online buffer and GAE shapes validate, one actor/value update is finite, no video is created, and the output reloads.

- [ ] **Step 7: Inspect outputs and repository diff**

Run:

```bash
git diff --check
git status --short
find outputs/rl100/offline_smoke outputs/rl100/online_smoke -maxdepth 3 -type f -printf '%p %s\n' | sort
```

Expected: no whitespace errors; only planned source/docs/tests plus pre-existing user changes are present; smoke outputs contain configs, provenance, metrics, RL state, and standard model bundles.

- [ ] **Step 8: Commit documentation and any reviewed smoke fixes**

```bash
git add RL/README.md RL/MIGRATION.md RL/NOTICE RL tests/rl100
git commit -m "docs(rl): document RL-100 state migration"
```

## Task 9: Independent Review And Final Verification

**Files:**
- Review all `RL/` and `tests/rl100/` files
- Modify only files required to address confirmed findings

- [ ] **Step 1: Request specification-compliance review**

Give a fresh review agent the design spec, this plan, source commit, and final diff. It must verify every required behavior, especially sparse reward placement, image fail-fast semantics, stochastic variance, likelihood masks, SAME_STEP state, and checkpoint compatibility.

- [ ] **Step 2: Request code-quality review**

Give a second fresh review agent the implementation and tests without the first review's conclusion. It must report correctness issues, device/shape bugs, numerical risks, unsafe online behavior, and missing tests, ordered by severity with file/line references.

- [ ] **Step 3: Fix confirmed findings test-first**

For every accepted finding, add or tighten a regression test, verify it fails, apply the smallest correction, and rerun the focused suite. Reject suggestions that conflict with the approved state/image boundary or add unrelated abstractions.

- [ ] **Step 4: Run final verification**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100 -q
UV_CACHE_DIR=.uv-cache uv run pytest tests/policies/test_diffusion_state_only.py tests/envs/test_moya_newton.py -q
UV_CACHE_DIR=.uv-cache uv run python -m RL.cli.inspect_dataset \
  --dataset-root /home/droid/project/Moya_newton_sim/.worktrees/feat-fused-batched-env/runs/lerobot/randomized_grasp_100_20260813-230331/dataset \
  --repo-id moya_newton/randomized_grasp_100 \
  --summary /home/droid/project/Moya_newton_sim/.worktrees/feat-fused-batched-env/runs/lerobot/randomized_grasp_100_20260813-230331/collection_summary.json
git diff --check
git status --short --branch
```

Expected: tests pass, dataset inspection remains 94/6, diff check is clean, and unrelated `AGENTS.md`/prior plan changes remain untouched.

- [ ] **Step 5: Commit final reviewed corrections**

```bash
git add RL tests/rl100
git commit -m "fix(rl): address RL migration review"
```
