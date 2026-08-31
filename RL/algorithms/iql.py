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

"""State-conditioned double-Q implicit Q-learning for action chunks."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from RL.policy.observation_encoder import ObservationFeatureEncoder
from RL.types import DecisionBatch, ObservationBatch


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return converted


class ActionPacker(nn.Module):
    """Pack active action dimensions and append the valid-step mask."""

    def __init__(
        self, *, action_dim: int, chunk_size: int, active_action_mask: Tensor
    ) -> None:
        super().__init__()
        _positive_int("action_dim", action_dim)
        _positive_int("chunk_size", chunk_size)
        if not isinstance(active_action_mask, Tensor) or active_action_mask.dtype != torch.bool:
            raise ValueError("active_action_mask must be a boolean torch.Tensor")
        if active_action_mask.shape != (action_dim,):
            raise ValueError(
                f"active_action_mask must have shape {(action_dim,)}, "
                f"got {tuple(active_action_mask.shape)}"
            )
        if not active_action_mask.any().item():
            raise ValueError("active_action_mask must select at least one dimension")
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.register_buffer("active_action_mask", active_action_mask.detach().clone())

    @property
    def active_indices(self) -> Tensor:
        return torch.nonzero(self.active_action_mask, as_tuple=False).flatten()

    @property
    def output_dim(self) -> int:
        return self.chunk_size * int(self.active_action_mask.sum().item()) + self.chunk_size

    def forward(self, action: Tensor, action_valid: Tensor) -> Tensor:
        if not isinstance(action, Tensor) or not action.is_floating_point():
            raise ValueError("action must be a floating-point torch.Tensor")
        if action.ndim != 3 or action.shape[1:] != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"action must have shape [batch,{self.chunk_size},{self.action_dim}], "
                f"got {tuple(action.shape)}"
            )
        if not torch.isfinite(action).all().item():
            raise ValueError("action must contain only finite values")
        if not isinstance(action_valid, Tensor) or action_valid.dtype != torch.bool:
            raise ValueError("action_valid must be a boolean torch.Tensor")
        if action_valid.shape != action.shape[:2]:
            raise ValueError(
                f"action_valid must have shape {tuple(action.shape[:2])}, "
                f"got {tuple(action_valid.shape)}"
            )
        if not action_valid.any(dim=1).all().item():
            raise ValueError("action_valid must select at least one action step per batch item")
        invalid_seen = (~action_valid).to(torch.int64).cumsum(dim=1) > 0
        if (invalid_seen & action_valid).any().item():
            raise ValueError("action_valid must be a contiguous True prefix followed by False padding")

        active = action.float().index_select(-1, self.active_indices.to(action.device))
        active = torch.where(action_valid.to(action.device).unsqueeze(-1), active, 0.0)
        packed = torch.cat(
            (active.flatten(start_dim=1), action_valid.to(action.device).float()), dim=-1
        )
        if not torch.isfinite(packed).all().item():
            raise ValueError("packed action contains non-finite values")
        return packed


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for index, hidden_dim in enumerate(hidden_dims):
            _positive_int(f"hidden_dims[{index}]", hidden_dim)
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


def expectile_loss(
    residual: Tensor, *, expectile: float, reduction: str = "mean"
) -> Tensor:
    if not isinstance(residual, Tensor) or not residual.is_floating_point():
        raise ValueError("residual must be a floating-point torch.Tensor")
    if not torch.isfinite(residual).all().item():
        raise ValueError("residual must contain only finite values")
    expectile = _finite_float("expectile", expectile)
    if not 0 < expectile < 1:
        raise ValueError(f"expectile must be in (0, 1), got {expectile}")
    weights = torch.where(residual > 0, expectile, 1.0 - expectile)
    loss = weights * residual.square()
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"unsupported reduction={reduction!r}")


def compute_td_target(
    *, reward: Tensor, discount: Tensor, done: Tensor, next_value: Tensor
) -> Tensor:
    expected_shape = reward.shape
    for name, value in (("reward", reward), ("discount", discount), ("next_value", next_value)):
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating-point torch.Tensor")
        if value.ndim != 2 or value.shape[-1] != 1 or value.shape != expected_shape:
            raise ValueError(
                f"{name} must have matching shape [batch,1], got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")
    if not isinstance(done, Tensor) or done.dtype != torch.bool or done.shape != expected_shape:
        raise ValueError(f"done must be boolean with shape {tuple(expected_shape)}")
    if torch.any((discount < 0) | (discount > 1)).item():
        raise ValueError("discount must lie in [0, 1]")
    target = reward.float() + discount.float() * (~done).float() * next_value.float()
    if not torch.isfinite(target).all().item():
        raise ValueError("TD target contains non-finite values")
    return target


def _float_observation(observation: ObservationBatch, device: torch.device) -> ObservationBatch:
    return ObservationBatch(
        {
            key: value.to(device=device, dtype=torch.float32)
            if value.is_floating_point()
            else value.to(device)
            for key, value in observation.features.items()
        }
    )


def _polyak_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_parameter.lerp_(source_parameter, tau)
        for target_buffer, source_buffer in zip(
            target.buffers(), source.buffers(), strict=True
        ):
            target_buffer.copy_(source_buffer)


def _require_finite_loss(name: str, loss: Tensor) -> None:
    if not isinstance(loss, Tensor) or loss.ndim != 0 or not loss.is_floating_point():
        raise ValueError(f"{name} must be a scalar floating-point torch.Tensor")
    if not torch.isfinite(loss).item():
        raise ValueError(f"{name} must be finite before backward and optimizer step")


def _clip_finite_gradients(
    name: str, parameters: Sequence[nn.Parameter], max_norm: float
) -> None:
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    if not torch.isfinite(total_norm).item():
        raise ValueError(f"{name} gradients must be finite before optimizer step")


class IQL(nn.Module):
    """Double-Q IQL over prepared normalized decision batches."""

    def __init__(
        self,
        *,
        feature_encoder: ObservationFeatureEncoder,
        action_dim: int,
        chunk_size: int,
        active_action_mask: Tensor,
        hidden_dims: Sequence[int],
        expectile: float,
        tau: float,
        q_lr: float,
        v_lr: float,
        gradient_clip_norm: float = 10.0,
    ) -> None:
        super().__init__()
        if not isinstance(feature_encoder, ObservationFeatureEncoder):
            raise ValueError("feature_encoder must implement ObservationFeatureEncoder")
        self.expectile = _finite_float("expectile", expectile)
        if not 0 < self.expectile < 1:
            raise ValueError(f"expectile must be in (0, 1), got {self.expectile}")
        self.tau = _finite_float("tau", tau)
        if not 0 < self.tau <= 1:
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        q_lr = _finite_float("q_lr", q_lr)
        v_lr = _finite_float("v_lr", v_lr)
        if q_lr <= 0 or v_lr <= 0:
            raise ValueError("q_lr and v_lr must be positive")
        self.gradient_clip_norm = _finite_float("gradient_clip_norm", gradient_clip_norm)
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")

        self.feature_encoder = feature_encoder
        self.target_encoder = copy.deepcopy(feature_encoder)
        self.action_packer = ActionPacker(
            action_dim=action_dim,
            chunk_size=chunk_size,
            active_action_mask=active_action_mask,
        )
        input_dim = feature_encoder.output_dim + self.action_packer.output_dim
        self.q1 = _MLP(input_dim, hidden_dims)
        self.q2 = _MLP(input_dim, hidden_dims)
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)
        self.value = _MLP(feature_encoder.output_dim, hidden_dims)
        for module in (self.target_encoder, self.target_q1, self.target_q2):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        q_parameters = [
            *self.feature_encoder.parameters(),
            *self.q1.parameters(),
            *self.q2.parameters(),
        ]
        v_parameters = list(self.value.parameters())
        if {id(parameter) for parameter in q_parameters} & {
            id(parameter) for parameter in v_parameters
        }:
            raise ValueError("q_optimizer and v_optimizer parameters must be disjoint")
        self.q_optimizer = torch.optim.Adam(q_parameters, lr=q_lr)
        self.v_optimizer = torch.optim.Adam(v_parameters, lr=v_lr)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def train(self, mode: bool = True) -> IQL:
        super().train(mode)
        self.target_encoder.eval()
        self.target_q1.eval()
        self.target_q2.eval()
        return self

    def _q_input(self, features: Tensor, packed_action: Tensor) -> Tensor:
        return torch.cat((features, packed_action), dim=-1)

    def update(self, batch: DecisionBatch) -> dict[str, float]:
        if not isinstance(batch, DecisionBatch):
            raise ValueError(f"batch must be a DecisionBatch, got {type(batch).__name__}")
        batch = batch.to(self.device)
        observation = _float_observation(batch.observation, self.device)
        next_observation = _float_observation(batch.next_observation, self.device)
        packed_action = self.action_packer(batch.action, batch.action_valid)

        with torch.no_grad():
            target_features = self.target_encoder(observation)
            target_q_input = self._q_input(target_features, packed_action)
            q_bar = torch.minimum(
                self.target_q1(target_q_input), self.target_q2(target_q_input)
            )
        value_prediction = self.value(target_features.detach())
        value_residual = q_bar - value_prediction
        value_loss = expectile_loss(value_residual, expectile=self.expectile)

        with torch.no_grad():
            next_features = self.target_encoder(next_observation)
            next_value = self.value(next_features)
            td_target = compute_td_target(
                reward=batch.reward,
                discount=batch.discount,
                done=batch.done,
                next_value=next_value,
            )
        online_features = self.feature_encoder(observation)
        online_q_input = self._q_input(online_features, packed_action)
        q1_prediction = self.q1(online_q_input)
        q2_prediction = self.q2(online_q_input)
        q_loss = functional.mse_loss(q1_prediction, td_target) + functional.mse_loss(
            q2_prediction, td_target
        )
        q_parameters = [
            *self.feature_encoder.parameters(),
            *self.q1.parameters(),
            *self.q2.parameters(),
        ]
        value_parameters = list(self.value.parameters())
        _require_finite_loss("value_loss", value_loss)
        _require_finite_loss("q_loss", q_loss)

        self.v_optimizer.zero_grad(set_to_none=True)
        self.q_optimizer.zero_grad(set_to_none=True)
        try:
            value_loss.backward()
            q_loss.backward()
            _clip_finite_gradients(
                "value_loss", value_parameters, self.gradient_clip_norm
            )
            _clip_finite_gradients("q_loss", q_parameters, self.gradient_clip_norm)
        except Exception:
            self.v_optimizer.zero_grad(set_to_none=True)
            self.q_optimizer.zero_grad(set_to_none=True)
            raise

        self.v_optimizer.step()
        self.q_optimizer.step()

        _polyak_update(self.target_encoder, self.feature_encoder, self.tau)
        _polyak_update(self.target_q1, self.q1, self.tau)
        _polyak_update(self.target_q2, self.q2, self.tau)
        metrics = {
            "q_loss": float(q_loss.detach().item()),
            "v_loss": float(value_loss.detach().item()),
            "q_mean": float(torch.minimum(q1_prediction, q2_prediction).detach().mean().item()),
            "v_mean": float(value_prediction.detach().mean().item()),
            "td_target_mean": float(td_target.detach().mean().item()),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"IQL metrics contain non-finite values: {metrics}")
        return metrics

    @torch.no_grad()
    def q_value(
        self,
        observation: ObservationBatch,
        action: Tensor,
        action_valid: Tensor,
    ) -> Tensor:
        """Evaluate the conservative (minimum) Q estimate for an action chunk.

        This is intentionally the raw ``min(Q1, Q2)`` value.  It is the
        critic used by AM-Q model rollouts and must not be replaced by the
        normalized IQL actor advantage ``Q - V``.
        """

        prepared = _float_observation(observation, self.device)
        packed_action = self.action_packer(
            action.to(self.device), action_valid.to(self.device)
        )
        features = self.feature_encoder(prepared)
        q_input = self._q_input(features, packed_action)
        result = torch.minimum(self.q1(q_input), self.q2(q_input))
        if result.ndim != 2 or result.shape[-1] != 1 or not torch.isfinite(result).all().item():
            raise ValueError("IQL q_value must be finite with shape [batch,1]")
        return result

    @torch.no_grad()
    def min_q(
        self,
        observation: ObservationBatch,
        action: Tensor,
        action_valid: Tensor,
    ) -> Tensor:
        """Compatibility name used by the RL-100 AM-Q evaluator."""

        return self.q_value(observation, action, action_valid)

    @torch.no_grad()
    def min_q_features(
        self,
        features: Tensor,
        action: Tensor,
        action_valid: Tensor,
    ) -> Tensor:
        """Evaluate min-Q from a precomputed observation feature latent.

        RL-100's multimodal AM-Q rollout evolves encoded DP3 features rather
        than raw point clouds/RGB.  This method keeps the IQL critic reusable
        in that latent space while ``min_q`` continues to accept raw
        :class:`ObservationBatch` inputs for ordinary training.
        """

        if not isinstance(features, Tensor) or features.ndim != 2:
            raise ValueError(
                f"features must have shape [batch,feature_dim], got {getattr(features, 'shape', None)}"
            )
        if features.shape[-1] != self.feature_encoder.output_dim:
            raise ValueError(
                "feature latent width disagrees with encoder: "
                f"expected {self.feature_encoder.output_dim}, got {features.shape[-1]}"
            )
        if not features.is_floating_point() or not torch.isfinite(features).all().item():
            raise ValueError("features must be finite floating-point values")
        packed_action = self.action_packer(
            action.to(self.device), action_valid.to(self.device)
        )
        q_input = self._q_input(features.to(self.device), packed_action)
        result = torch.minimum(self.q1(q_input), self.q2(q_input))
        if result.ndim != 2 or result.shape[-1] != 1 or not torch.isfinite(result).all().item():
            raise ValueError("IQL min_q_features must be finite with shape [batch,1]")
        return result

    @torch.no_grad()
    def advantage(
        self,
        observation: ObservationBatch,
        action: Tensor,
        action_valid: Tensor,
        *,
        normalize: bool = True,
        eps: float = 1e-6,
    ) -> Tensor:
        if not isinstance(normalize, bool):
            raise ValueError(f"normalize must be a bool, got {normalize!r}")
        eps = _finite_float("eps", eps)
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        prepared = _float_observation(observation, self.device)
        packed_action = self.action_packer(
            action.to(self.device), action_valid.to(self.device)
        )
        features = self.feature_encoder(prepared)
        q_input = self._q_input(features, packed_action)
        raw = torch.minimum(self.q1(q_input), self.q2(q_input)) - self.value(features)
        if normalize and raw.shape[0] > 1:
            centered = raw - raw.mean()
            variance = centered.square().mean()
            result = raw if variance.item() <= eps else centered / torch.sqrt(variance + eps)
        else:
            result = raw
        if not torch.isfinite(result).all().item():
            raise ValueError("IQL advantage contains non-finite values")
        return result.detach()
