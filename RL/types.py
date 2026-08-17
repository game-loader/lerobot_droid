# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validated tensor contracts shared by the RL migration."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from torch import Tensor

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_INDEX_DTYPES = {torch.int32, torch.int64}


def _nonfinite_value_summary(value: Tensor) -> str:
    finite = torch.isfinite(value)
    invalid_values = value.detach().flatten()[~finite.flatten()]
    preview = invalid_values[:3].cpu().tolist()
    return f"{preview!r} ({invalid_values.numel()} non-finite value(s))"


def _validate_tensor(name: str, value: Tensor, *, ndim: int | None = None) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(
            f"{name} must be a torch.Tensor, got {type(value).__name__}: actual={value!r}"
        )
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {tuple(value.shape)}")
    if value.numel() == 0:
        raise ValueError(f"{name} must be nonempty, got shape {tuple(value.shape)}")
    if not torch.isfinite(value).all().item():
        raise ValueError(
            f"{name} must contain only finite values: actual={_nonfinite_value_summary(value)}"
        )


def _validate_floating_tensor(name: str, value: Tensor, *, ndim: int) -> None:
    _validate_tensor(name, value, ndim=ndim)
    if not value.is_floating_point():
        raise ValueError(f"{name} must have a floating-point dtype, got {value.dtype}")


@dataclass(frozen=True)
class ObservationBatch:
    """A batch of named observation tensors sharing their leading dimension."""

    features: dict[str, Tensor]

    def __post_init__(self) -> None:
        if not isinstance(self.features, dict):
            raise ValueError(
                f"features must be a dict, got {type(self.features).__name__}: "
                f"actual={self.features!r}"
            )
        if not self.features:
            raise ValueError(f"features must be nonempty: actual={self.features!r}")

        batch_sizes: set[int] = set()
        for key, tensor in self.features.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"feature keys must be nonempty strings, got {key!r}")
            _validate_tensor(key, tensor)
            if tensor.ndim == 0:
                raise ValueError(f"{key} must have a batch dimension, got shape {tuple(tensor.shape)}")
            batch_sizes.add(tensor.shape[0])

        if len(batch_sizes) != 1:
            raise ValueError(f"features batch sizes disagree: actual={sorted(batch_sizes)}")
        batch_size = next(iter(batch_sizes))
        if batch_size == 0:
            raise ValueError("features batch size must be nonzero: actual=0")

    def batch_size(self) -> int:
        return next(iter(self.features.values())).shape[0]

    def to(self, device: torch.device | str) -> ObservationBatch:
        return ObservationBatch({key: value.to(device) for key, value in self.features.items()})

    def index_select(self, indices: Tensor) -> ObservationBatch:
        _validate_tensor("indices", indices, ndim=1)
        if indices.dtype not in _INDEX_DTYPES:
            raise ValueError(f"indices must have dtype torch.int32 or torch.int64, got {indices.dtype}")
        return ObservationBatch({key: value.index_select(0, indices) for key, value in self.features.items()})


@dataclass(frozen=True)
class DecisionBatch:
    """One batch of decision-level transitions with fixed-size action chunks."""

    observation: ObservationBatch
    next_observation: ObservationBatch
    action: Tensor
    action_valid: Tensor
    reward: Tensor
    done: Tensor
    discount: Tensor

    def __post_init__(self) -> None:
        self._validate_intrinsic()

    def _validate_intrinsic(self) -> None:
        if not isinstance(self.observation, ObservationBatch):
            raise ValueError(
                "observation must be an ObservationBatch, "
                f"got {type(self.observation).__name__}: actual={self.observation!r}"
            )
        if not isinstance(self.next_observation, ObservationBatch):
            raise ValueError(
                "next_observation must be an ObservationBatch, "
                f"got {type(self.next_observation).__name__}: actual={self.next_observation!r}"
            )

        _validate_floating_tensor("action", self.action, ndim=3)
        batch_size, chunk_size, action_dim = self.action.shape
        if batch_size == 0 or chunk_size == 0 or action_dim == 0:
            raise ValueError(f"action dimensions must be nonzero, got shape {tuple(self.action.shape)}")
        if self.observation.batch_size() != batch_size:
            raise ValueError(
                "observation batch size must match action batch size, "
                f"got {self.observation.batch_size()} and {batch_size}"
            )
        if self.next_observation.batch_size() != batch_size:
            raise ValueError(
                "next_observation batch size must match action batch size, "
                f"got {self.next_observation.batch_size()} and {batch_size}"
            )

        _validate_tensor("action_valid", self.action_valid, ndim=2)
        if self.action_valid.dtype != torch.bool:
            raise ValueError(f"action_valid must have dtype torch.bool, got {self.action_valid.dtype}")
        if self.action_valid.shape != (batch_size, chunk_size):
            raise ValueError(
                "action_valid must have shape "
                f"{(batch_size, chunk_size)}, got {tuple(self.action_valid.shape)}"
            )

        _validate_floating_tensor("reward", self.reward, ndim=2)
        if self.reward.shape != (batch_size, 1):
            raise ValueError(f"reward must have shape {(batch_size, 1)}, got {tuple(self.reward.shape)}")

        _validate_tensor("done", self.done, ndim=2)
        if self.done.dtype != torch.bool:
            raise ValueError(f"done must have dtype torch.bool, got {self.done.dtype}")
        if self.done.shape != (batch_size, 1):
            raise ValueError(f"done must have shape {(batch_size, 1)}, got {tuple(self.done.shape)}")

        _validate_floating_tensor("discount", self.discount, ndim=2)
        if self.discount.shape != (batch_size, 1):
            raise ValueError(f"discount must have shape {(batch_size, 1)}, got {tuple(self.discount.shape)}")

    def validate(self, *, state_dim: int, action_dim: int, chunk_size: int, n_obs_steps: int) -> None:
        self._validate_intrinsic()
        for name, value in {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "chunk_size": chunk_size,
            "n_obs_steps": n_obs_steps,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        batch_size = self.action.shape[0]
        expected_state_shape = (batch_size, n_obs_steps, state_dim)
        for observation_name, observation in (
            ("observation", self.observation),
            ("next_observation", self.next_observation),
        ):
            try:
                state = observation.features["observation.state"]
            except KeyError as exc:
                raise ValueError(f"{observation_name}.state is required: actual=<missing>") from exc
            if state.shape != expected_state_shape:
                raise ValueError(
                    f"{observation_name}.state must have shape {expected_state_shape}, "
                    f"got {tuple(state.shape)}"
                )

        expected_action_shape = (batch_size, chunk_size, action_dim)
        if self.action.shape != expected_action_shape:
            raise ValueError(
                f"action must have shape {expected_action_shape}, got {tuple(self.action.shape)}"
            )
        expected_action_valid_shape = (batch_size, chunk_size)
        if self.action_valid.shape != expected_action_valid_shape:
            raise ValueError(
                "action_valid must have shape "
                f"{expected_action_valid_shape}, got {tuple(self.action_valid.shape)}"
            )

    def to(self, device: torch.device | str) -> DecisionBatch:
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
    """Stored stochastic denoising transitions for policy-ratio recomputation."""

    latents: Tensor
    next_latents: Tensor
    timesteps: Tensor
    old_log_prob: Tensor
    final_actions: Tensor

    def __post_init__(self) -> None:
        _validate_floating_tensor("latents", self.latents, ndim=4)
        trace_shape = tuple(self.latents.shape)
        if any(size == 0 for size in trace_shape):
            raise ValueError(f"latents dimensions must be nonzero, got shape {trace_shape}")

        _validate_floating_tensor("next_latents", self.next_latents, ndim=4)
        if self.next_latents.shape != self.latents.shape:
            raise ValueError(
                f"next_latents must have shape {trace_shape}, got {tuple(self.next_latents.shape)}"
            )

        _validate_tensor("timesteps", self.timesteps, ndim=1)
        if self.timesteps.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"timesteps must have an integer dtype, got {self.timesteps.dtype}")
        if self.timesteps.shape != (trace_shape[0],):
            raise ValueError(
                f"timesteps must have shape {(trace_shape[0],)}, got {tuple(self.timesteps.shape)}"
            )

        _validate_floating_tensor("old_log_prob", self.old_log_prob, ndim=4)
        if self.old_log_prob.shape != self.latents.shape:
            raise ValueError(
                f"old_log_prob must have shape {trace_shape}, got {tuple(self.old_log_prob.shape)}"
            )

        expected_final_shape = trace_shape[1:]
        _validate_floating_tensor("final_actions", self.final_actions, ndim=3)
        if self.final_actions.shape != expected_final_shape:
            raise ValueError(
                f"final_actions must have shape {expected_final_shape}, got {tuple(self.final_actions.shape)}"
            )
