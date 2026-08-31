# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Online vector PPO orchestration for a stochastic LeRobot Diffusion Policy.

The trainer deliberately works at the policy decision boundary.  A policy
decision contains an action chunk, while the environment still receives one
action at a time.  Rollouts are stored decision-major and environment-minor
(``[T, N, ...]``), which keeps vector GAE independent per world and makes a
partial final chunk explicit through ``action_valid``.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from RL.algorithms.gae import compute_vector_gae
from RL.algorithms.ppo import (
    denoising_ppo_loss,
    denoising_ppo_reduced_metrics,
    reduce_event_log_prob,
)
from RL.checkpointing import RLCounters, RLProvenance, save_rl_checkpoint
from RL.config import RLConfig
from RL.policy.diffusion_adapter import DiffusionRLAdapter
from RL.policy.observation_encoder import (
    ObservationFeatureEncoder,
    StateFeatureEncoder,
    is_image_feature,
)
from RL.types import DenoisingTrace, ObservationBatch

_STATE_ALIASES = ("observation.state", "agent_pos", "state", "environment_state")


def _finite_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"metric {key!r} is non-finite: {value!r}")
        result[str(key)] = converted
    return result


def _info_metrics(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f"info/{prefix}/{name}": float(value) for name, value in metrics.items()}


def _validate_optimizer_parameters(
    optimizer: torch.optim.Optimizer, module: nn.Module, *, name: str, device: torch.device
) -> None:
    module_parameters = list(module.parameters())
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    module_ids = [id(parameter) for parameter in module_parameters]
    if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != set(module_ids):
        raise ValueError(f"{name} optimizer parameters must exactly match the module parameters")
    if any(parameter.device != device for parameter in optimizer_parameters):
        raise ValueError(f"{name} optimizer parameters must be on {device}")


def _as_bool_vector(value: Any, size: int, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.bool_)
    if array.ndim == 0:
        array = np.full(size, bool(array), dtype=np.bool_)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {array.shape}")
    return array


def _as_tensor(value: Any, *, device: torch.device | str | None = None) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _env_action_bounds(
    env: Any, action: np.ndarray, *, num_envs: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return broadcastable finite action bounds for a vector environment."""

    space = getattr(env, "single_action_space", None)
    if space is None:
        space = getattr(env, "action_space", None)
    if space is None:
        return None
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    shape = getattr(space, "shape", None)
    if low is None or high is None or shape is None:
        return None
    low_array = np.asarray(low, dtype=np.float32)
    high_array = np.asarray(high, dtype=np.float32)
    if action.shape == (num_envs, *tuple(shape)):
        low_array = low_array.reshape(tuple(shape))
        high_array = high_array.reshape(tuple(shape))
        low_array = low_array[None, ...]
        high_array = high_array[None, ...]
    elif action.shape == tuple(shape) and num_envs == 1:
        low_array = low_array.reshape(action.shape)
        high_array = high_array.reshape(action.shape)
    elif low_array.shape == action.shape and high_array.shape == action.shape:
        pass
    else:
        raise ValueError(
            f"environment action must match action-space shape {tuple(shape)} for {num_envs} worlds, "
            f"got {action.shape}"
        )
    if not np.all(np.isfinite(low_array)) or not np.all(np.isfinite(high_array)):
        return None
    return low_array, high_array


def _validate_env_action(env: Any, action: np.ndarray, *, num_envs: int) -> None:
    """Fail before stepping when a policy action violates the environment contract."""

    bounds = _env_action_bounds(env, action, num_envs=num_envs)
    if bounds is None:
        return
    low_array, high_array = bounds
    if np.any(action < low_array) or np.any(action > high_array):
        raise ValueError("policy action is outside the environment action space")


def _clip_env_action(env: Any, action: np.ndarray, *, num_envs: int) -> np.ndarray:
    """Project a finite policy action onto a finite environment Box boundary."""

    bounds = _env_action_bounds(env, action, num_envs=num_envs)
    if bounds is None:
        return action
    low_array, high_array = bounds
    return np.clip(action, low_array, high_array).astype(np.float32, copy=False)


def _raw_feature_mapping(raw: Any, *, num_envs: int) -> dict[str, Tensor]:
    """Convert common Gym/Moya observation forms into named tensors.

    Moya returns ``{"agent_pos": ndarray}``; LeRobot processors use
    ``observation.state``.  Image keys are retained verbatim so an explicitly
    supplied image encoder can be used later.
    """

    if isinstance(raw, ObservationBatch):
        return dict(raw.features)
    if isinstance(raw, Mapping):
        # Some vector wrappers put the actual observation under ``observation``.
        if "observation" in raw and isinstance(raw["observation"], Mapping):
            nested = _raw_feature_mapping(raw["observation"], num_envs=num_envs)
            extras = {key: value for key, value in raw.items() if key != "observation"}
            if extras:
                nested.update(_raw_feature_mapping(extras, num_envs=num_envs))
            return nested
        state_key = next((key for key in _STATE_ALIASES if key in raw), None)
        result: dict[str, Tensor] = {}
        if state_key is not None:
            result["observation.state"] = _as_tensor(raw[state_key])
        for key, value in raw.items():
            if key == state_key or key in _STATE_ALIASES:
                continue
            if isinstance(key, str):
                tensor_value = _as_tensor(value)
                if key.startswith("observation.") or is_image_feature(key, tensor_value):
                    result[key] = tensor_value
        if not result:
            raise ValueError(
                "observation mapping must contain one of "
                f"{_STATE_ALIASES!r} or an image feature"
            )
    else:
        result = {"observation.state": _as_tensor(raw)}

    for key, value in list(result.items()):
        if value.ndim == 0:
            raise ValueError(f"observation feature {key!r} must not be scalar")
        if value.shape[0] != num_envs:
            # A single real robot commonly exposes unbatched observations
            # (state=[D], image=[C,H,W], point cloud=[P,3]).  Treat these as
            # a one-world vector environment while preserving strict checks
            # for genuinely batched environments.
            if num_envs == 1:
                value = value.unsqueeze(0)
                result[key] = value
            else:
                raise ValueError(
                    f"observation feature {key!r} must have leading dimension {num_envs}, "
                    f"got {tuple(value.shape)}"
                )
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise ValueError(f"observation feature {key!r} contains non-finite values")
    if "observation.state" not in result:
        raise ValueError("online state path requires an observation.state feature")
    return result


def _final_observation_rows(raw: Any, *, num_envs: int) -> list[dict[str, Tensor] | None]:
    """Normalize Gymnasium ``final_obs`` payloads into per-world feature rows.

    Vector environments commonly expose terminal observations as an object
    array of mappings, while some adapters return a dense batched state array.
    This helper deliberately has no simulator-specific assumptions (in
    particular, no fixed 39D state width), so DP3 real-robot observations are
    handled exactly like state-only observations.
    """

    if isinstance(raw, Mapping):
        mapped = _raw_feature_mapping(raw, num_envs=num_envs)
        return [
            {key: value[index].clone() for key, value in mapped.items()}
            for index in range(num_envs)
        ]

    array = np.asarray(raw)
    # Dense numeric arrays are interpreted as a single state feature.  This
    # covers the common ``final_obs: [N, state_dim]`` contract.
    if array.dtype != object and array.ndim >= 1 and array.shape[0] == num_envs:
        mapped = _raw_feature_mapping(array, num_envs=num_envs)
        return [
            {key: value[index].clone() for key, value in mapped.items()}
            for index in range(num_envs)
        ]

    object_rows = np.asarray(raw, dtype=object).reshape(-1)
    if object_rows.shape != (num_envs,):
        raise ValueError(
            f"final observation payload must have one row per environment, got {object_rows.shape}"
        )
    rows: list[dict[str, Tensor] | None] = []
    for _index, item in enumerate(object_rows.tolist()):
        if item is None:
            rows.append(None)
            continue
        mapped = _raw_feature_mapping(item, num_envs=1)
        rows.append({key: value[0].clone() for key, value in mapped.items()})
    return rows


def _select_transition_frames(
    next_frames: Mapping[str, Tensor],
    info: Mapping[str, Any],
    done: np.ndarray,
    *,
    num_envs: int,
) -> dict[str, Tensor]:
    """Replace SAME_STEP reset frames with terminal ``final_obs`` rows.

    Real-robot adapters usually return the terminal observation directly and
    therefore do not provide ``final_obs``.  If a vector wrapper does expose
    SAME_STEP payloads, this generic path keeps state, point cloud, and image
    features aligned without importing a simulator adapter.
    """

    final_key = next(
        (key for key in ("final_obs", "final_observation") if key in info), None
    )
    if final_key is None:
        return {key: value.clone() for key, value in next_frames.items()}

    for mask_key in (
        "_final_obs",
        "_final_observation",
        "final_obs_mask",
        "final_observation_mask",
    ):
        if mask_key in info:
            mask = _as_bool_vector(info[mask_key], num_envs, name=f"info[{mask_key}]")
            if not np.array_equal(mask, done):
                raise ValueError(f"info[{mask_key}] mask does not match done")

    rows = _final_observation_rows(info[final_key], num_envs=num_envs)
    selected = {key: value.clone() for key, value in next_frames.items()}
    for world in np.flatnonzero(done):
        row = rows[int(world)]
        if row is None:
            raise ValueError(f"done world {int(world)} is missing final observation")
        for key, base in next_frames.items():
            if key not in row:
                raise ValueError(
                    f"done world {int(world)} final observation is missing feature {key!r}"
                )
            value = row[key]
            expected_shape = tuple(base.shape[1:])
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"final observation feature {key!r} shape {tuple(value.shape)} "
                    f"does not match {expected_shape}"
                )
            selected[key][int(world)] = value.to(
                device=base.device, dtype=base.dtype
            )
    return selected


def _reset_history(features: Mapping[str, Tensor], n_obs_steps: int) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for key, value in features.items():
        # A caller may provide an already stacked history. State history is
        # unambiguous; images need an additional channel axis, so a single
        # RGB frame [N,3,H,W] is never confused with [N,T,C,H,W].
        is_state_history = key == "observation.state" and value.ndim == 3
        is_image_history = is_image_feature(key, value) and value.ndim >= 5
        if (is_state_history or is_image_history) and value.shape[1] == n_obs_steps:
            result[key] = value.clone()
        else:
            result[key] = value.unsqueeze(1).expand(-1, n_obs_steps, *value.shape[1:]).clone()
    return result


def _append_history(history: Mapping[str, Tensor], features: Mapping[str, Tensor]) -> dict[str, Tensor]:
    keys = set(history) | set(features)
    result: dict[str, Tensor] = {}
    for key in sorted(keys):
        if key not in history or key not in features:
            raise ValueError(f"observation feature set changed during rollout: missing {key!r}")
        value = features[key]
        previous = history[key]
        if previous.ndim != value.ndim + 1 or previous.shape[0] != value.shape[0]:
            raise ValueError(
                f"observation history shape mismatch for {key!r}: "
                f"history={tuple(previous.shape)} frame={tuple(value.shape)}"
            )
        if previous.shape[2:] != value.shape[1:]:
            raise ValueError(
                f"observation feature shape changed for {key!r}: "
                f"history={tuple(previous.shape)} frame={tuple(value.shape)}"
            )
        result[key] = torch.cat((previous[:, 1:], value.unsqueeze(1)), dim=1)
    return result


def _history_batch(features: Mapping[str, Tensor], *, device: torch.device | str) -> ObservationBatch:
    prepared: dict[str, Tensor] = {}
    for key, value in features.items():
        tensor = value.to(device)
        if key == "observation.state":
            tensor = tensor.float()
        prepared[key] = tensor
    return ObservationBatch(prepared)


def _move_features(features: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in features.items()}


def _sequence_observation(items: Sequence[ObservationBatch]) -> ObservationBatch:
    if not items:
        raise ValueError("observation sequence must be nonempty")
    keys = tuple(items[0].features)
    if any(tuple(item.features) != keys for item in items[1:]):
        raise ValueError("observation feature keys disagree across rollout decisions")
    return ObservationBatch(
        {key: torch.stack([item.features[key] for item in items], dim=0) for key in keys}
    )


def _flatten_sequence(observation: ObservationBatch) -> ObservationBatch:
    result: dict[str, Tensor] = {}
    for key, value in observation.features.items():
        if value.ndim < 2:
            raise ValueError(f"sequence feature {key!r} must have [time,world,...] axes")
        result[key] = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
    return ObservationBatch(result)


def _trace_select(trace: DenoisingTrace, indices: Tensor) -> DenoisingTrace:
    return DenoisingTrace(
        latents=trace.latents.index_select(1, indices),
        next_latents=trace.next_latents.index_select(1, indices),
        timesteps=trace.timesteps,
        old_log_prob=trace.old_log_prob.index_select(1, indices),
        final_actions=trace.final_actions.index_select(0, indices),
    )


class StateValueNetwork(nn.Module):
    """Small value head over an explicit observation feature encoder."""

    def __init__(
        self,
        encoder: ObservationFeatureEncoder,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if not isinstance(encoder, ObservationFeatureEncoder):
            raise ValueError("encoder must implement ObservationFeatureEncoder")
        layers: list[nn.Module] = []
        current = encoder.output_dim
        for index, hidden in enumerate(tuple(hidden_dims)):
            if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden <= 0:
                raise ValueError(f"hidden_dims[{index}] must be positive, got {hidden!r}")
            layers.extend((nn.Linear(current, hidden), nn.ReLU()))
            current = hidden
        layers.append(nn.Linear(current, 1))
        self.encoder = encoder
        self.head = nn.Sequential(*layers)

    def forward(self, observation: ObservationBatch) -> Tensor:
        features = self.encoder(observation)
        value = self.head(features)
        if value.ndim != 2 or value.shape[-1] != 1:
            raise ValueError(f"value network must return [batch,1], got {tuple(value.shape)}")
        if not torch.isfinite(value).all().item():
            raise ValueError("value network output contains non-finite values")
        return value


@dataclasses.dataclass(frozen=True)
class OnlineRolloutBatch:
    """Decision-major vector rollout storage.

    ``observation`` and ``next_observation`` contain feature tensors with shape
    ``[time, world, ...]``.  Scalar fields use ``[time, world, 1]`` and action
    fields use ``[time, world, chunk, ...]``.  ``traces`` holds one detached
    stochastic DDIM trace per decision; each trace has a world batch axis.
    """

    observation: ObservationBatch
    next_observation: ObservationBatch
    traces: tuple[DenoisingTrace, ...]
    action: Tensor
    action_valid: Tensor
    reward: Tensor
    done: Tensor
    discount: Tensor
    executed_steps: Tensor
    success: Tensor
    values: Tensor | None = None
    next_values: Tensor | None = None
    terminated: Tensor | None = None
    truncated: Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ObservationBatch) or not isinstance(
            self.next_observation, ObservationBatch
        ):
            raise ValueError("observation and next_observation must be ObservationBatch instances")
        if not self.traces:
            raise ValueError("traces must be nonempty")
        if self.action.ndim != 4 or self.action.shape[0] != len(self.traces):
            raise ValueError(f"action must have shape [time,world,chunk,action], got {tuple(self.action.shape)}")
        time, worlds, chunk, _action_dim = self.action.shape
        if time == 0 or worlds == 0 or chunk == 0:
            raise ValueError("rollout dimensions must be nonzero")
        if self.action_valid.shape != (time, worlds, chunk) or self.action_valid.dtype != torch.bool:
            raise ValueError(
                f"action_valid must have shape {(time, worlds, chunk)} and bool dtype, "
                f"got {tuple(self.action_valid.shape)} {self.action_valid.dtype}"
            )
        for name, value in (
            ("reward", self.reward),
            ("done", self.done),
            ("discount", self.discount),
            ("executed_steps", self.executed_steps),
            ("success", self.success),
        ):
            if value.shape not in {(time, worlds), (time, worlds, 1)}:
                raise ValueError(f"{name} must have shape [time,world] or [time,world,1], got {tuple(value.shape)}")
            if name in {"done", "success"} and value.dtype != torch.bool:
                raise ValueError(f"{name} must have bool dtype")
            if name not in {"done", "success"} and not value.is_floating_point() and name != "executed_steps":
                raise ValueError(f"{name} must have a floating dtype")
            if name == "executed_steps" and value.dtype not in (torch.int32, torch.int64):
                raise ValueError("executed_steps must have integer dtype")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} contains non-finite values")
        # Rewards are finite transition scalars.  RL-100's default Moya
        # contract is sparse terminal binary reward, but a real-robot
        # environment may provide dense shaping through the same Gym API.
        discount_values = self.discount.float()
        if torch.any((discount_values < 0.0) | (discount_values > 1.0)).item():
            raise ValueError("discount must lie in [0, 1]")
        valid_steps = self.action_valid.to(torch.int64).sum(dim=-1)
        if torch.any(valid_steps <= 0).item() or torch.any(valid_steps > chunk).item():
            raise ValueError("action_valid must contain at least one valid step per transition")
        invalid_seen = (~self.action_valid).to(torch.int64).cumsum(dim=-1) > 0
        if (invalid_seen & self.action_valid).any().item():
            raise ValueError("action_valid must be a contiguous valid prefix")
        if not torch.equal(self.executed_steps.reshape(time, worlds), valid_steps):
            raise ValueError("executed_steps must equal the action_valid prefix length")
        for name, value in (("values", self.values), ("next_values", self.next_values)):
            if value is not None and value.shape not in {(time, worlds), (time, worlds, 1)}:
                raise ValueError(f"{name} has an invalid shape {tuple(value.shape)}")
        if self.observation.features[next(iter(self.observation.features))].shape[:2] != (time, worlds):
            raise ValueError("observation features must have [time,world,...] axes")
        if self.next_observation.features[next(iter(self.next_observation.features))].shape[:2] != (time, worlds):
            raise ValueError("next_observation features must have [time,world,...] axes")
        for index, trace in enumerate(self.traces):
            if trace.final_actions.shape[0] != worlds:
                raise ValueError(f"trace {index} world count disagrees with rollout")

    @property
    def time_steps(self) -> int:
        return int(self.action.shape[0])

    @property
    def num_envs(self) -> int:
        return int(self.action.shape[1])

    @property
    def world_count(self) -> int:
        return self.num_envs

    @property
    def old_log_prob(self) -> tuple[Tensor, ...]:
        return tuple(trace.old_log_prob for trace in self.traces)

    @property
    def trace(self) -> tuple[DenoisingTrace, ...]:
        return self.traces

    def to(self, device: torch.device | str) -> OnlineRolloutBatch:
        def move_observation(value: ObservationBatch) -> ObservationBatch:
            return value.to(device)

        return dataclasses.replace(
            self,
            observation=move_observation(self.observation),
            next_observation=move_observation(self.next_observation),
            traces=tuple(
                DenoisingTrace(
                    latents=trace.latents.to(device),
                    next_latents=trace.next_latents.to(device),
                    timesteps=trace.timesteps.to(device),
                    old_log_prob=trace.old_log_prob.to(device),
                    final_actions=trace.final_actions.to(device),
                )
                for trace in self.traces
            ),
            action=self.action.to(device),
            action_valid=self.action_valid.to(device),
            reward=self.reward.to(device),
            done=self.done.to(device),
            discount=self.discount.to(device),
            executed_steps=self.executed_steps.to(device),
            success=self.success.to(device),
            values=None if self.values is None else self.values.to(device),
            next_values=None if self.next_values is None else self.next_values.to(device),
            terminated=None if self.terminated is None else self.terminated.to(device),
            truncated=None if self.truncated is None else self.truncated.to(device),
        )


class OnlineTrainer:
    """Collect vector rollouts and optimize a diffusion actor with PPO/GAE."""

    def __init__(
        self,
        *,
        current_policy: DiffusionRLAdapter,
        old_policy: DiffusionRLAdapter | None = None,
        actor_optimizer: torch.optim.Optimizer | None = None,
        value_encoder: ObservationFeatureEncoder | None = None,
        value_network: nn.Module | None = None,
        value_optimizer: torch.optim.Optimizer | None = None,
        metrics_path: str | Path | None = None,
        actor_clip_ratio: float = 0.2,
        value_clip_ratio: float = 0.2,
        gradient_clip_norm: float = 1.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_epochs: int = 1,
        minibatch_size: int | None = None,
        seed: int = 0,
        logger: Callable[[Mapping[str, float]], Any] | None = None,
        debug: bool = False,
        image_encoder: ObservationFeatureEncoder | None = None,
        value_hidden_dims: Sequence[int] = (128, 128),
        reward_mode: str = "terminal_success",
    ) -> None:
        if not isinstance(current_policy, DiffusionRLAdapter):
            raise ValueError("current_policy must be a DiffusionRLAdapter")
        if old_policy is None:
            # Loading a second adapter from the same local source keeps the
            # rollout policy independent from the trainable actor.
            old_policy = DiffusionRLAdapter(
                copy.deepcopy(current_policy.checkpoint), current_policy.trace_config
            )
        if not isinstance(old_policy, DiffusionRLAdapter):
            raise ValueError("old_policy must be a DiffusionRLAdapter")
        if current_policy.policy is old_policy.policy:
            raise ValueError("current and old policies must be independent modules")
        current_policy.assert_transition_compatible(old_policy)
        if not torch.equal(current_policy.checkpoint.active_action_mask, old_policy.checkpoint.active_action_mask):
            raise ValueError("current and old policies must use the same active action mask")
        for name, value in (("actor_clip_ratio", actor_clip_ratio), ("value_clip_ratio", value_clip_ratio)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if not 0 <= float(value) < 1:
                raise ValueError(f"{name} must lie in [0,1)")
        if isinstance(gradient_clip_norm, bool) or not isinstance(gradient_clip_norm, (int, float)) or not math.isfinite(float(gradient_clip_norm)) or float(gradient_clip_norm) <= 0:
            raise ValueError("gradient_clip_norm must be positive and finite")
        for name, value in (("gamma", gamma), ("gae_lambda", gae_lambda)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must lie in [0,1]")
        if isinstance(ppo_epochs, bool) or not isinstance(ppo_epochs, int) or ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be positive")
        if minibatch_size is not None and (isinstance(minibatch_size, bool) or not isinstance(minibatch_size, int) or minibatch_size <= 0):
            raise ValueError("minibatch_size must be positive when supplied")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if reward_mode not in {"terminal_success", "environment"}:
            raise ValueError(
                "reward_mode must be 'terminal_success' or 'environment', "
                f"got {reward_mode!r}"
            )

        self.current_policy = current_policy
        self.old_policy = old_policy
        self.actor_optimizer = actor_optimizer or torch.optim.Adam(
            current_policy.policy.parameters(), lr=1e-6
        )
        if not isinstance(self.actor_optimizer, torch.optim.Optimizer):
            raise ValueError("actor_optimizer must be a torch optimizer")
        _validate_optimizer_parameters(
            self.actor_optimizer,
            current_policy.policy,
            name="actor",
            device=self.device,
        )
        self.actor_clip_ratio = float(actor_clip_ratio)
        self.value_clip_ratio = float(value_clip_ratio)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.seed = seed
        self.reward_mode = reward_mode
        self.logger = logger
        if not isinstance(debug, bool):
            raise ValueError("debug must be a bool")
        self.debug = debug
        self._generator = old_policy.make_generator(seed)
        self._runtime_contract: dict[str, str | int | float | bool | None] = {}
        self._replay_verified = False
        self._replay_info = {
            "old_replay_abs_delta_max": 0.0,
            "old_replay_abs_delta_mean": 0.0,
        }
        self._transition_info = current_policy.denoising_step_diagnostics()

        if value_network is not None and value_encoder is not None:
            raise ValueError("provide either value_network or value_encoder, not both")
        if value_network is None:
            if value_encoder is None:
                config = current_policy.policy.config
                state_feature = config.robot_state_feature
                if state_feature is None:
                    raise ValueError("policy must define a state feature for the default value critic")
                value_encoder = StateFeatureEncoder(
                    state_dim=state_feature.shape[0],
                    n_obs_steps=config.n_obs_steps,
                    hidden_dims=(128, 128),
                    output_dim=128,
                    image_encoder=image_encoder,
                    ignore_extra_features=True,
                )
            value_network = StateValueNetwork(value_encoder, hidden_dims=value_hidden_dims)
        if not isinstance(value_network, nn.Module):
            raise ValueError("value_network must be a torch.nn.Module")
        self.value_network = value_network
        if value_optimizer is None:
            self.value_network.to(self.device)
            self.value_optimizer = torch.optim.Adam(
                self.value_network.parameters(), lr=3e-4
            )
        else:
            self.value_optimizer = value_optimizer
        if not isinstance(self.value_optimizer, torch.optim.Optimizer):
            raise ValueError("value_optimizer must be a torch optimizer")
        _validate_optimizer_parameters(
            self.value_optimizer,
            self.value_network,
            name="value",
            device=self.device,
        )
        self.metrics_path = Path(metrics_path) if metrics_path is not None else None
        if self.metrics_path is not None:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._metrics_snapshot: list[dict[str, float]] = []
        self.counters = RLCounters()
        self.old_policy.policy.eval()
        for parameter in self.old_policy.policy.parameters():
            parameter.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.current_policy.policy.parameters()).device

    @property
    def value_device(self) -> torch.device:
        return next(self.value_network.parameters()).device

    def sampler_state(self) -> dict[str, Tensor]:
        """Return the explicit rollout generator state for exact resumption."""

        return {"generator": self._generator.get_state().detach().cpu().clone()}

    def set_runtime_contract(self, contract: Mapping[str, Any]) -> None:
        if not isinstance(contract, Mapping):
            raise ValueError("online runtime contract must be a mapping")
        normalized: dict[str, str | int | float | bool | None] = {}
        for key, value in contract.items():
            if not isinstance(key, str) or not key:
                raise ValueError("online runtime contract keys must be nonempty strings")
            if value is None or isinstance(value, (str, int, bool)) or (
                isinstance(value, float) and math.isfinite(value)
            ):
                normalized[key] = value
            else:
                raise ValueError(
                    f"online runtime contract value {key!r} must be a finite scalar"
                )
        self._runtime_contract = normalized

    def training_contract(self) -> dict[str, str | int | float | bool | None]:
        return {
            "actor_clip_ratio": self.actor_clip_ratio,
            "value_clip_ratio": self.value_clip_ratio,
            "gradient_clip_norm": self.gradient_clip_norm,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ppo_epochs": self.ppo_epochs,
            "minibatch_size": self.minibatch_size,
            "seed": self.seed,
            "reward_mode": self.reward_mode,
            **self._runtime_contract,
        }

    def restore_sampler_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or "generator" not in state:
            raise ValueError("online sampler state must contain a generator state")
        generator_state = state["generator"]
        if not isinstance(generator_state, Tensor) or generator_state.dtype != torch.uint8:
            raise ValueError("online sampler generator state must be a uint8 tensor")
        # PyTorch's CUDA generator implementation accepts a strided CPU
        # ByteTensor for set_state, even though samples are generated on CUDA.
        self._generator.set_state(generator_state.detach().cpu().contiguous())

    def _value(self, observation: ObservationBatch) -> Tensor:
        normalized = self.current_policy.checkpoint.normalize_observation(
            observation.to(self.value_device),
            # Real-robot camera streams commonly arrive as uint8.  Convert
            # them to [0, 1] before applying the checkpoint visual statistics,
            # matching the actor/DP3 AM-Q preprocessing path.
            convert_visual_uint8=True,
        )
        value = self.value_network(normalized)
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        if value.ndim != 2 or value.shape[-1] != 1:
            raise ValueError(f"value network must return [batch,1], got {tuple(value.shape)}")
        if not torch.isfinite(value).all().item():
            raise ValueError("value network output contains non-finite values")
        return value

    def _write_metrics(self, metrics: Mapping[str, float]) -> dict[str, float]:
        finite = _finite_metrics(metrics)
        if self.metrics_path is not None:
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        finite,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        self._metrics_snapshot.append(dict(finite))
        self.counters = dataclasses.replace(self.counters, metrics_rows=self.counters.metrics_rows + 1)
        if self.logger is not None:
            self.logger(finite)
        return finite

    def record_metrics(self, metrics: Mapping[str, float]) -> dict[str, float]:
        return self._write_metrics(metrics)

    @staticmethod
    def _split_step(result: Any, *, num_envs: int) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError("vector environment step must return (obs,reward,terminated,truncated,info)")
        observation, reward, terminated, truncated, info = result
        if not isinstance(info, Mapping):
            raise ValueError("vector environment info must be a mapping")
        reward_array = np.asarray(reward, dtype=np.float32).reshape(num_envs)
        terminated_array = _as_bool_vector(terminated, num_envs, name="terminated")
        truncated_array = _as_bool_vector(truncated, num_envs, name="truncated")
        if not np.isfinite(reward_array).all():
            raise ValueError("environment reward contains non-finite values")
        return observation, reward_array, terminated_array, truncated_array, info

    @staticmethod
    def _split_reset(result: Any) -> tuple[Any, Mapping[str, Any]]:
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
            if not isinstance(info, Mapping):
                raise ValueError("vector environment reset info must be a mapping")
            return observation, info
        return result, {}

    @staticmethod
    def _terminal_success(
        info: Mapping[str, Any], done: np.ndarray, reward: np.ndarray
    ) -> np.ndarray:
        """Extract terminal success when available, with reward fallback.

        Moya supplies ``final_info[*]["is_success"]``.  A real-robot adapter
        may instead expose ``is_success``/``success`` directly, or only a
        positive terminal reward.  All forms are normalized to one boolean
        per environment and nonterminal rows are forced false.
        """

        num_envs = int(done.shape[0])
        result: np.ndarray = np.zeros(num_envs, dtype=np.bool_)
        found: np.ndarray = np.zeros(num_envs, dtype=np.bool_)

        raw_final = info.get("final_info")
        if isinstance(raw_final, Mapping):
            for key in ("is_success", "success"):
                if key in raw_final:
                    values = _as_bool_vector(raw_final[key], num_envs, name=f"final_info[{key}]")
                    result[:] = values
                    found[:] = True
                    break
        elif raw_final is not None:
            array = np.asarray(raw_final, dtype=object).reshape(num_envs)
            for index, item in enumerate(array.tolist()):
                if isinstance(item, Mapping):
                    for key in ("is_success", "success"):
                        if key in item:
                            value = item[key]
                            if not isinstance(value, (bool, np.bool_)):
                                raise ValueError(
                                    f"final_info[{index}][{key}] must be bool, got {value!r}"
                                )
                            result[index] = bool(value)
                            found[index] = True
                            break

        for key in ("is_success", "success"):
            if key not in info:
                continue
            values = _as_bool_vector(info[key], num_envs, name=f"info[{key}]")
            result[~found] = values[~found]
            found |= ~found
            break

        # Positive terminal reward is the least-surprising fallback for a
        # generic environment that does not provide a separate success flag.
        result[~found] = reward[~found] > 0
        result[~done] = False
        return result

    def collect(
        self,
        env: Any,
        *,
        decisions: int,
        seed: int | Sequence[int] | None = None,
    ) -> OnlineRolloutBatch:
        """Collect ``decisions`` policy chunks from a vector environment."""

        if env is None or not hasattr(env, "step") or not hasattr(env, "reset"):
            raise ValueError("env must provide reset() and step()")
        if isinstance(decisions, bool) or not isinstance(decisions, int) or decisions <= 0:
            raise ValueError("decisions must be positive")
        num_envs = getattr(env, "num_envs", None)
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("env.num_envs must be a positive integer")
        reset_observation, _reset_info = self._split_reset(env.reset(seed=seed))
        current_frames = _move_features(
            _raw_feature_mapping(reset_observation, num_envs=num_envs), self.device
        )
        policy_history = _reset_history(current_frames, self.current_policy.policy.config.n_obs_steps)

        observations: list[ObservationBatch] = []
        next_observations: list[ObservationBatch] = []
        traces: list[DenoisingTrace] = []
        actions: list[Tensor] = []
        valid_actions: list[Tensor] = []
        rewards: list[Tensor] = []
        dones: list[Tensor] = []
        discounts: list[Tensor] = []
        executed_steps: list[Tensor] = []
        successes: list[Tensor] = []
        values: list[Tensor] = []
        next_values: list[Tensor] = []
        terminated_rows: list[Tensor] = []
        truncated_rows: list[Tensor] = []

        # The policy owns its processor/device; all rollout history is moved
        # there before sampling and copied back only for environment stepping.
        for _decision in range(decisions):
            observation_batch = _history_batch(policy_history, device=self.device)
            with torch.no_grad():
                trace = self.old_policy.sample_trace(observation_batch, generator=self._generator)
                if not self._replay_verified:
                    self._replay_info = self.old_policy.verify_trace_replay(observation_batch, trace)
                    self._replay_verified = True
                raw_chunk = self.old_policy.executable_actions(trace).detach()
                value = self._value(observation_batch).detach()
            chunk_size = raw_chunk.shape[1]
            action_dim = raw_chunk.shape[2]
            action_row = torch.zeros((num_envs, chunk_size, action_dim), dtype=torch.float32, device=self.device)
            valid_row = torch.zeros((num_envs, chunk_size), dtype=torch.bool, device=self.device)
            reward_row = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
            success_row = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            step_row = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
            terminal_history: dict[int, dict[str, Tensor]] = {}
            working_history = {key: history_value.clone() for key, history_value in policy_history.items()}
            working_frames = {key: frame_value.clone() for key, frame_value in current_frames.items()}
            term_row: np.ndarray = np.zeros(num_envs, dtype=np.bool_)
            trunc_row: np.ndarray = np.zeros(num_envs, dtype=np.bool_)

            for action_index in range(chunk_size):
                action_np = raw_chunk[:, action_index, :].detach().to("cpu").numpy().astype(np.float32, copy=False)
                previously_done = term_row | trunc_row
                if previously_done.any():
                    action_np[previously_done] = 0.0
                # Diffusion sampling can produce tiny out-of-Box excursions
                # after unnormalization. Project only the command sent to the
                # environment; PPO likelihoods remain those of the latent DDIM
                # transition stored in ``trace``.
                action_np = _clip_env_action(env, action_np, num_envs=num_envs)
                _validate_env_action(env, action_np, num_envs=num_envs)
                action_row[:, action_index] = torch.as_tensor(
                    action_np, dtype=torch.float32, device=self.device
                )
                next_raw, _env_reward, terminated, truncated, info = self._split_step(
                    env.step(action_np), num_envs=num_envs
                )
                done_np = terminated | truncated
                newly_done = done_np & ~term_row & ~trunc_row
                active_np = ~(term_row | trunc_row)
                valid_row[:, action_index] = torch.as_tensor(active_np, dtype=torch.bool, device=self.device)
                step_row += torch.as_tensor(active_np, dtype=torch.int64, device=self.device)
                term_row |= terminated
                trunc_row |= truncated

                reset_frames = _move_features(
                    _raw_feature_mapping(next_raw, num_envs=num_envs), self.device
                )
                # SAME_STEP vector environments may provide terminal
                # ``final_obs`` payloads.  Use the generic adapter for all
                # modalities; real-robot environments normally return their
                # terminal observation directly and omit this payload.
                next_frames = _select_transition_frames(
                    reset_frames, info, done_np, num_envs=num_envs
                )

                # Build one terminal-aware history for the stored transition,
                # while continuing the next policy history from SAME_STEP's
                # reset observation returned by the environment.
                terminal_appended = _append_history(working_history, next_frames)
                for world in range(num_envs):
                    if newly_done[world]:
                        # This is the terminal state used by the transition;
                        # SAME_STEP's reset observation is not substituted.
                        terminal_history[world] = {
                            key: terminal_appended[key][world].clone()
                            for key in terminal_appended
                        }
                        final_success = bool(
                            self._terminal_success(info, done_np, _env_reward)[world]
                        )
                        success_row[world] = final_success
                        if self.reward_mode == "terminal_success":
                            reward_row[world] = float(final_success)

                if self.reward_mode == "environment":
                    reward_row += torch.as_tensor(
                        _env_reward * active_np,
                        dtype=torch.float32,
                        device=self.device,
                    )
                working_history = _append_history(working_history, reset_frames)
                working_frames = reset_frames

                # Histories for terminal worlds are reset to the returned
                # SAME_STEP observation for the next policy decision.  Alive
                # worlds retain their temporal history.
                for world in np.flatnonzero(newly_done):
                    reset_features = {
                        key: value[world : world + 1].clone() for key, value in reset_frames.items()
                    }
                    reset_history = _reset_history(reset_features, self.current_policy.policy.config.n_obs_steps)
                    for key in working_history:
                        working_history[key][world : world + 1] = reset_history[key]

                if bool((term_row | trunc_row).all()):
                    # All worlds have a terminal transition; remaining chunk
                    # actions are padding and need not be sent to the env.
                    break

            # ``working_history`` is the policy history for the next decision;
            # terminal transition histories are kept separately in next_row.
            next_row_features: dict[str, Tensor] = {
                key: value.clone() for key, value in working_history.items()
            }
            for world, history in terminal_history.items():
                for key, history_value in history.items():
                    next_row_features[key][world] = history_value
            next_row = _history_batch(next_row_features, device=self.device)

            observations.append(observation_batch)
            next_observations.append(next_row)
            traces.append(trace)
            actions.append(action_row)
            valid_actions.append(valid_row)
            rewards.append(reward_row.unsqueeze(-1))
            dones.append(torch.as_tensor(term_row | trunc_row, dtype=torch.bool, device=self.device).unsqueeze(-1))
            discounts.append(self.gamma ** step_row.float().unsqueeze(-1))
            executed_steps.append(step_row)
            successes.append(success_row.unsqueeze(-1))
            values.append(value)
            next_values.append(self._value(next_row).detach())
            terminated_rows.append(torch.as_tensor(term_row, dtype=torch.bool, device=self.device).unsqueeze(-1))
            truncated_rows.append(torch.as_tensor(trunc_row, dtype=torch.bool, device=self.device).unsqueeze(-1))
            policy_history = working_history
            current_frames = working_frames

        rollout = OnlineRolloutBatch(
            observation=_sequence_observation(observations),
            next_observation=_sequence_observation(next_observations),
            traces=tuple(traces),
            action=torch.stack(actions),
            action_valid=torch.stack(valid_actions),
            reward=torch.stack(rewards),
            done=torch.stack(dones),
            discount=torch.stack(discounts),
            executed_steps=torch.stack(executed_steps),
            success=torch.stack(successes),
            values=torch.stack(values),
            next_values=torch.stack(next_values),
            terminated=torch.stack(terminated_rows),
            truncated=torch.stack(truncated_rows),
        )
        self.counters = dataclasses.replace(
            self.counters,
            environment_steps=self.counters.environment_steps + int(rollout.executed_steps.sum().item()),
            decisions_seen=self.counters.decisions_seen + rollout.time_steps * rollout.num_envs,
        )
        return rollout

    def _advantages(self, rollout: OnlineRolloutBatch) -> tuple[Tensor, Tensor]:
        reward = rollout.reward
        done = rollout.done
        discount = rollout.discount
        if reward.ndim == 2:
            reward = reward.unsqueeze(-1)
        if done.ndim == 2:
            done = done.unsqueeze(-1)
        if discount.ndim == 2:
            discount = discount.unsqueeze(-1)
        if rollout.values is None:
            raise ValueError("online rollout is missing value estimates")
        value = rollout.values if rollout.values.ndim == 3 else rollout.values.unsqueeze(-1)
        if rollout.next_values is None:
            raise ValueError("online rollout is missing next value estimates")
        next_value = rollout.next_values if rollout.next_values.ndim == 3 else rollout.next_values.unsqueeze(-1)
        result = compute_vector_gae(
            reward=reward.float(),
            value=value.float(),
            next_value=next_value.float(),
            done=done,
            discount=discount.float(),
            gae_lambda=self.gae_lambda,
        )
        advantage = result.advantage
        valid = torch.isfinite(advantage)
        if not valid.all().item():
            raise ValueError("online advantage contains non-finite values")
        centered = advantage - advantage.mean()
        scale = centered.square().mean().sqrt().clamp_min(1e-6)
        return centered / scale, result.returns

    def update(self, rollout: OnlineRolloutBatch) -> dict[str, float]:
        """Apply one PPO update as an all-or-nothing transaction."""

        if not isinstance(rollout, OnlineRolloutBatch):
            raise ValueError("rollout must be an OnlineRolloutBatch")
        actor_parameters = [parameter.detach().clone() for parameter in self.current_policy.policy.parameters()]
        old_parameters = [parameter.detach().clone() for parameter in self.old_policy.policy.parameters()]
        value_parameters = [parameter.detach().clone() for parameter in self.value_network.parameters()]
        actor_optimizer_state = copy.deepcopy(self.actor_optimizer.state_dict())
        value_optimizer_state = copy.deepcopy(self.value_optimizer.state_dict())
        counters = self.counters
        metrics_snapshot_length = len(self._metrics_snapshot)
        metrics_file_existed = self.metrics_path is not None and self.metrics_path.exists()
        metrics_file_size = (
            self.metrics_path.stat().st_size
            if self.metrics_path is not None and self.metrics_path.exists()
            else None
        )
        try:
            return self._update_impl(rollout)
        except BaseException:
            with torch.no_grad():
                for parameter, snapshot in zip(
                    self.current_policy.policy.parameters(), actor_parameters, strict=True
                ):
                    parameter.copy_(snapshot)
                for parameter, snapshot in zip(
                    self.old_policy.policy.parameters(), old_parameters, strict=True
                ):
                    parameter.copy_(snapshot)
                for parameter, snapshot in zip(
                    self.value_network.parameters(), value_parameters, strict=True
                ):
                    parameter.copy_(snapshot)
            self.actor_optimizer.load_state_dict(actor_optimizer_state)
            self.value_optimizer.load_state_dict(value_optimizer_state)
            self.counters = counters
            del self._metrics_snapshot[metrics_snapshot_length:]
            if metrics_file_size is not None and self.metrics_path is not None:
                with self.metrics_path.open("r+b") as handle:
                    handle.truncate(metrics_file_size)
            elif (
                self.metrics_path is not None
                and not metrics_file_existed
                and self.metrics_path.exists()
            ):
                self.metrics_path.unlink()
            raise

    def _update_impl(self, rollout: OnlineRolloutBatch) -> dict[str, float]:
        if not isinstance(rollout, OnlineRolloutBatch):
            raise ValueError("rollout must be an OnlineRolloutBatch")
        advantage, returns = self._advantages(rollout)
        time, worlds = rollout.time_steps, rollout.num_envs
        batch_size = time * worlds
        minibatch = self.minibatch_size or batch_size
        flat_obs = _flatten_sequence(rollout.observation).to(self.device)
        flat_advantage = advantage.reshape(batch_size).to(self.device)
        flat_returns = returns.reshape(batch_size).to(self.value_device)
        flat_valid = rollout.action_valid.reshape(batch_size, rollout.action.shape[2]).to(self.device)
        flat_old_values = (
            rollout.values.reshape(batch_size, 1).to(self.value_device)
            if rollout.values is not None
            else self._value(flat_obs).detach().to(self.value_device)
        )
        flat_traces = self._flatten_traces(rollout.traces)
        permutation_generator = torch.Generator(device="cpu").manual_seed(self.seed + self.counters.global_updates)

        actor_losses: list[float] = []
        value_losses: list[float] = []
        diagnostic_chunks: dict[int, list[tuple[Tensor, Tensor, Tensor]]] = {}
        for _epoch in range(self.ppo_epochs):
            permutation = torch.randperm(batch_size, generator=permutation_generator)
            for start in range(0, batch_size, minibatch):
                indices = permutation[start : start + minibatch].to(self.device)
                obs_mb = flat_obs.index_select(indices)
                valid_mb = flat_valid.index_select(0, indices)
                adv_mb = flat_advantage.index_select(0, indices)
                trace_mb = _trace_select(flat_traces, indices)
                value_obs = flat_obs.index_select(indices).to(self.value_device)
                old_value_mb = flat_old_values.index_select(0, indices)
                returns_mb = flat_returns.index_select(0, indices)

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.value_optimizer.zero_grad(set_to_none=True)
                losses: list[Tensor] = []
                try:
                    for step_index, new_log_prob in enumerate(self.current_policy.iter_recomputed_log_prob(obs_mb, trace_mb)):
                        old_log_prob = self.current_policy.executable_log_prob(trace_mb.old_log_prob[step_index : step_index + 1])
                        new_executable = self.current_policy.executable_log_prob(new_log_prob)
                        loss, _metrics = denoising_ppo_loss(
                            new_executable,
                            old_log_prob,
                            adv_mb,
                            step_mask=valid_mb,
                            action_dim_mask=self.current_policy.checkpoint.active_action_mask,
                            clip_ratio=self.actor_clip_ratio,
                        )
                        (loss / len(self.current_policy.timesteps)).backward()
                        losses.append(loss.detach())
                        new_joint_log_prob = reduce_event_log_prob(
                            new_executable.detach(),
                            step_mask=valid_mb,
                            action_dim_mask=self.current_policy.checkpoint.active_action_mask,
                        )
                        old_joint_log_prob = reduce_event_log_prob(
                            old_log_prob.detach(),
                            step_mask=valid_mb,
                            action_dim_mask=self.current_policy.checkpoint.active_action_mask,
                        )
                        diagnostic_chunks.setdefault(step_index, []).append(
                            (
                                new_joint_log_prob.cpu(),
                                old_joint_log_prob.cpu(),
                                valid_mb.detach().cpu(),
                            )
                        )
                    gradient_norm = torch.nn.utils.clip_grad_norm_(self.current_policy.policy.parameters(), self.gradient_clip_norm)
                    if not torch.isfinite(gradient_norm).item():
                        raise ValueError("actor gradients must be finite")

                    value_prediction = self._value(value_obs)
                    clipped_prediction = old_value_mb + (value_prediction - old_value_mb).clamp(
                        -self.value_clip_ratio, self.value_clip_ratio
                    )
                    value_loss = torch.maximum(
                        (value_prediction - returns_mb).square(),
                        (clipped_prediction - returns_mb).square(),
                    ).mean()
                    if not torch.isfinite(value_loss).item():
                        raise ValueError("value loss must be finite")
                    value_loss.backward()
                    value_gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.value_network.parameters(), self.gradient_clip_norm
                    )
                    if not torch.isfinite(value_gradient_norm).item():
                        raise ValueError("value gradients must be finite")
                except Exception:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    self.value_optimizer.zero_grad(set_to_none=True)
                    raise
                self.actor_optimizer.step()
                self.value_optimizer.step()
                if not all(torch.isfinite(parameter).all().item() for parameter in self.current_policy.policy.parameters()):
                    raise ValueError("actor optimizer produced non-finite parameters")
                if not all(torch.isfinite(parameter).all().item() for parameter in self.value_network.parameters()):
                    raise ValueError("value optimizer produced non-finite parameters")
                actor_losses.append(float(torch.stack(losses).mean().item()))
                self.counters = dataclasses.replace(self.counters, actor_updates=self.counters.actor_updates + 1)
                value_losses.append(float(value_loss.detach().item()))

        self.counters = dataclasses.replace(self.counters, global_updates=self.counters.global_updates + 1)
        # A rollout/update cycle is the natural on-policy synchronization
        # point. This keeps the next rollout's behavior policy current.
        self.sync_old_policy()
        per_step_metrics: dict[int, dict[str, float]] = {}
        per_step_new: list[Tensor] = []
        per_step_old: list[Tensor] = []
        per_step_masks: list[Tensor] = []
        for step_index in range(len(self.current_policy.timesteps)):
            chunks = diagnostic_chunks.get(step_index)
            if not chunks:
                raise ValueError(f"missing PPO diagnostics for denoising step {step_index}")
            new_step = torch.cat([item[0] for item in chunks], dim=1)
            old_step = torch.cat([item[1] for item in chunks], dim=1)
            mask_step = torch.cat([item[2] for item in chunks], dim=0)
            summary = denoising_ppo_reduced_metrics(
                new_step,
                old_step,
                step_mask=mask_step,
                action_dim_mask=self.current_policy.checkpoint.active_action_mask.detach().cpu(),
                clip_ratio=self.actor_clip_ratio,
            )
            per_step_metrics[step_index] = summary
            per_step_new.append(new_step)
            per_step_old.append(old_step)
            per_step_masks.append(mask_step)
        if any(not torch.equal(per_step_masks[0], mask) for mask in per_step_masks[1:]):
            raise ValueError("PPO diagnostic masks disagree across denoising steps")
        step_count = len(per_step_new)
        overall_new = torch.cat(per_step_new, dim=0).reshape(1, -1)
        overall_old = torch.cat(per_step_old, dim=0).reshape(1, -1)
        overall_mask = per_step_masks[0].repeat((step_count, 1))
        aggregate = denoising_ppo_reduced_metrics(
            overall_new,
            overall_old,
            step_mask=overall_mask,
            action_dim_mask=self.current_policy.checkpoint.active_action_mask.detach().cpu(),
            clip_ratio=self.actor_clip_ratio,
        )
        terminal_count = rollout.done.float().sum()
        success_count = rollout.success.float().sum()
        # ``success`` is populated only on terminal transitions. Normalize by
        # completed episodes rather than by decision rows; a world can finish
        # early and be autoreset several times within one collection window.
        episode_success_rate = success_count / terminal_count.clamp_min(1.0)
        chunk_success_rate = success_count / float(rollout.success.numel())
        metrics = {
            "actor/loss": sum(actor_losses) / len(actor_losses),
            "actor/ratio_mean": aggregate["ratio_mean"],
            "actor/clip_fraction": aggregate["clip_fraction"],
            "actor/approx_kl": aggregate["approx_kl"],
            "value/loss": sum(value_losses) / len(value_losses),
            "value/mean": float(flat_old_values.mean().item()),
            "gae/advantage_mean": float(advantage.mean().item()),
            "gae/return_mean": float(returns.mean().item()),
            "rollout/environment_steps": float(rollout.executed_steps.sum().item()),
            "rollout/reward_mean": float(rollout.reward.float().mean().item()),
            "rollout/success_rate": float(episode_success_rate.item()),
            "rollout/chunk_success_rate": float(chunk_success_rate.item()),
            "rollout/success_count": float(success_count.item()),
            "rollout/terminal_count": float(terminal_count.item()),
            "info/actor/ppo_epochs": float(self.ppo_epochs),
            "info/actor/old_policy_sync_age_updates": float(
                self.counters.actor_updates - self.counters.last_old_policy_sync_actor_update
            ),
            "info/actor/old_policy_sync_count": float(self.counters.old_policy_syncs),
        }
        metrics.update(_info_metrics("actor", aggregate))
        metrics.update(_info_metrics("actor", self._replay_info))
        if self.debug:
            for step_index, summary in per_step_metrics.items():
                metrics.update(_info_metrics(f"actor/denoise_{step_index:02d}", summary))
            for diagnostic in self._transition_info:
                step_index = int(diagnostic["step"])
                metrics.update(_info_metrics(f"actor/denoise_{step_index:02d}", diagnostic))
        return self._write_metrics(metrics)

    @staticmethod
    def _flatten_traces(traces: Sequence[DenoisingTrace]) -> DenoisingTrace:
        if not traces:
            raise ValueError("traces must be nonempty")
        first = traces[0]
        if any(trace.timesteps.shape != first.timesteps.shape or not torch.equal(trace.timesteps, first.timesteps) for trace in traces[1:]):
            raise ValueError("rollout traces must share one denoising schedule")
        return DenoisingTrace(
            latents=torch.cat([trace.latents for trace in traces], dim=1),
            next_latents=torch.cat([trace.next_latents for trace in traces], dim=1),
            timesteps=first.timesteps,
            old_log_prob=torch.cat([trace.old_log_prob for trace in traces], dim=1),
            final_actions=torch.cat([trace.final_actions for trace in traces], dim=0),
        )

    def sync_old_policy(self) -> None:
        self.old_policy.policy.load_state_dict(self.current_policy.policy.state_dict(), strict=True)
        self.counters = dataclasses.replace(
            self.counters,
            old_policy_syncs=self.counters.old_policy_syncs + 1,
            last_old_policy_sync_actor_update=self.counters.actor_updates,
        )

    def save_checkpoint(
        self,
        destination: str | Path,
        *,
        provenance: RLProvenance,
        rl_config: RLConfig,
    ) -> Path:
        temporary_metrics: Path | None = None
        metrics_path = self.metrics_path
        if metrics_path is None and self._metrics_snapshot:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="rl-online-metrics-",
                suffix=".jsonl",
                delete=False,
            ) as handle:
                temporary_metrics = Path(handle.name)
                for row in self._metrics_snapshot:
                    handle.write(
                        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            metrics_path = temporary_metrics
        try:
            return save_rl_checkpoint(
                destination,
                current_policy=self.current_policy.checkpoint,
                old_policy=self.old_policy.policy,
                actor_optimizer=self.actor_optimizer,
                optimizers={"value": self.value_optimizer},
                counters=self.counters,
                provenance=provenance,
                rl_config=rl_config,
                metrics_path=metrics_path,
                trainer_state={
                    "value_network": dict(self.value_network.state_dict()),
                    "online_contract": self.training_contract(),
                },
                sampler_state=self.sampler_state(),
            )
        finally:
            if temporary_metrics is not None:
                temporary_metrics.unlink(missing_ok=True)
