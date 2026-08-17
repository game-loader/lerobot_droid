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

"""Generalized advantage estimation over time-major world batches."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GAEResult:
    """Advantages and value targets with shape ``[time, world, 1]``."""

    advantage: Tensor
    returns: Tensor


def _validate_float_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != 3 or value.shape[-1] != 1:
        raise ValueError(f"{name} must have shape [time,world,1], got {tuple(value.shape)}")
    if any(size == 0 for size in value.shape):
        raise ValueError(f"{name} dimensions must be nonzero, got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise ValueError(f"{name} must have a floating-point dtype, got {value.dtype}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def compute_vector_gae(
    *,
    reward: Tensor,
    value: Tensor,
    next_value: Tensor,
    done: Tensor,
    discount: Tensor,
    gae_lambda: float,
) -> GAEResult:
    """Compute per-world GAE with terminal masking for bootstrap and recursion."""

    _validate_float_tensor("value", value)
    expected_shape = value.shape
    expected_device = value.device
    expected_dtype = value.dtype
    for name, tensor in (
        ("next_value", next_value),
        ("reward", reward),
        ("discount", discount),
    ):
        _validate_float_tensor(name, tensor)
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} must match value shape {tuple(expected_shape)}, got {tuple(tensor.shape)}"
            )
        if tensor.device != expected_device:
            raise ValueError(f"{name} must be on device {expected_device}, got {tensor.device}")
        if tensor.dtype != expected_dtype:
            raise ValueError(f"{name} must have dtype {expected_dtype}, got {tensor.dtype}")

    if not isinstance(done, Tensor):
        raise ValueError(f"done must be a torch.Tensor, got {type(done).__name__}")
    if done.dtype != torch.bool:
        raise ValueError(f"done must have dtype torch.bool, got {done.dtype}")
    if done.shape != expected_shape:
        raise ValueError(f"done must match values shape {tuple(expected_shape)}, got {tuple(done.shape)}")
    if done.device != expected_device:
        raise ValueError(f"done must be on device {expected_device}, got {done.device}")
    if torch.any((discount < 0) | (discount > 1)).item():
        raise ValueError("discount must lie in [0, 1]")

    if isinstance(gae_lambda, bool) or not isinstance(gae_lambda, (int, float)):
        raise ValueError(f"gae_lambda must be a finite number in [0, 1], got {gae_lambda!r}")
    gae_lambda = float(gae_lambda)
    if not math.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be a finite number in [0, 1], got {gae_lambda!r}")

    not_done = ~done
    deltas = reward + discount * not_done * next_value - value
    next_advantage = torch.zeros_like(deltas[0])
    reverse_advantages: list[Tensor] = []
    for time_index in range(value.shape[0] - 1, -1, -1):
        next_advantage = (
            deltas[time_index] + discount[time_index] * gae_lambda * not_done[time_index] * next_advantage
        )
        reverse_advantages.append(next_advantage)

    advantage = torch.stack(reverse_advantages[::-1])
    returns = advantage + value
    if not torch.isfinite(advantage).all().item() or not torch.isfinite(returns).all().item():
        raise ValueError("GAE result contains non-finite values")
    return GAEResult(advantage=advantage, returns=returns)


def compute_gae(
    values: Tensor,
    next_values: Tensor,
    reward: Tensor,
    discount: Tensor,
    done: Tensor,
    *,
    gae_lambda: float,
) -> GAEResult:
    """Backward-compatible alias using the original plural argument names."""

    return compute_vector_gae(
        reward=reward,
        value=values,
        next_value=next_values,
        done=done,
        discount=discount,
        gae_lambda=gae_lambda,
    )
