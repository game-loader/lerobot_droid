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

"""State-history dynamics ensemble and offline policy-promotion gate."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from RL.algorithms.iql import ActionPacker, _finite_float, _float_observation, _positive_int
from RL.policy.observation_encoder import ObservationFeatureEncoder
from RL.types import DecisionBatch, ObservationBatch


@dataclass(frozen=True)
class DynamicsPrediction:
    state_delta: Tensor
    reward_logit: Tensor
    done_logit: Tensor

    def __post_init__(self) -> None:
        if self.state_delta.ndim != 4:
            raise ValueError(
                f"state_delta must have shape [ensemble,batch,obs,state], got {tuple(self.state_delta.shape)}"
            )
        expected_prefix = self.state_delta.shape[:2]
        for name, value in (
            ("reward_logit", self.reward_logit),
            ("done_logit", self.done_logit),
        ):
            if value.shape != (*expected_prefix, 1):
                raise ValueError(f"{name} must have shape {(*expected_prefix, 1)}, got {tuple(value.shape)}")
        if not all(
            torch.isfinite(value).all().item()
            for value in (self.state_delta, self.reward_logit, self.done_logit)
        ):
            raise ValueError("dynamics prediction contains non-finite values")

    @property
    def reward_probability(self) -> Tensor:
        return self.reward_logit.sigmoid()

    @property
    def done_probability(self) -> Tensor:
        return self.done_logit.sigmoid()


@dataclass(frozen=True)
class PromotionDecision:
    candidate_return: float
    behavior_return: float
    critic_return: float
    reference_return: float
    required_return: float
    relative_margin: float
    dynamics_validation_loss: float
    max_validation_loss: float
    rollout_disagreement: float | None
    max_rollout_disagreement: float | None
    promote: bool
    reason: str


class _DynamicsMember(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> None:
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


class StateDynamicsEnsemble(nn.Module):
    """Frozen state features plus learned next-state-history, reward, and done heads."""

    def __init__(
        self,
        *,
        feature_encoder: ObservationFeatureEncoder,
        state_dim: int,
        n_obs_steps: int,
        action_dim: int,
        chunk_size: int,
        active_action_mask: Tensor,
        hidden_dims: Sequence[int],
        ensemble_size: int = 5,
        learning_rate: float = 1e-3,
        bootstrap_seed: int = 0,
        state_loss_weight: float = 1.0,
        reward_loss_weight: float = 1.0,
        done_loss_weight: float = 1.0,
        gradient_clip_norm: float = 10.0,
        state_key: str = "observation.state",
    ) -> None:
        super().__init__()
        if not isinstance(feature_encoder, ObservationFeatureEncoder):
            raise ValueError("feature_encoder must implement ObservationFeatureEncoder")
        for name, value in (
            ("state_dim", state_dim),
            ("n_obs_steps", n_obs_steps),
            ("ensemble_size", ensemble_size),
        ):
            _positive_int(name, value)
        if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
            raise ValueError(f"bootstrap_seed must be an integer, got {bootstrap_seed!r}")
        if not isinstance(state_key, str) or not state_key:
            raise ValueError(f"state_key must be a nonempty string, got {state_key!r}")
        learning_rate = _finite_float("learning_rate", learning_rate)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.state_loss_weight = _finite_float("state_loss_weight", state_loss_weight)
        self.reward_loss_weight = _finite_float("reward_loss_weight", reward_loss_weight)
        self.done_loss_weight = _finite_float("done_loss_weight", done_loss_weight)
        if min(self.state_loss_weight, self.reward_loss_weight, self.done_loss_weight) < 0:
            raise ValueError("dynamics loss weights must be nonnegative")
        if self.state_loss_weight + self.reward_loss_weight + self.done_loss_weight <= 0:
            raise ValueError("at least one dynamics loss weight must be positive")
        self.gradient_clip_norm = _finite_float("gradient_clip_norm", gradient_clip_norm)
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")

        self.state_dim = state_dim
        self.n_obs_steps = n_obs_steps
        self.ensemble_size = ensemble_size
        self.bootstrap_seed = bootstrap_seed
        self.state_key = state_key
        self.feature_encoder = copy.deepcopy(feature_encoder)
        self.feature_encoder.eval()
        for parameter in self.feature_encoder.parameters():
            parameter.requires_grad_(False)
        self.action_packer = ActionPacker(
            action_dim=action_dim,
            chunk_size=chunk_size,
            active_action_mask=active_action_mask,
        )
        input_dim = feature_encoder.output_dim + self.action_packer.output_dim
        output_dim = n_obs_steps * state_dim + 2
        self.members = nn.ModuleList(
            _DynamicsMember(input_dim, hidden_dims, output_dim) for _ in range(ensemble_size)
        )
        self.optimizers = [torch.optim.Adam(member.parameters(), lr=learning_rate) for member in self.members]

    @property
    def device(self) -> torch.device:
        return next(self.members.parameters()).device

    @torch.no_grad()
    def sync_feature_encoder(self, source: ObservationFeatureEncoder) -> None:
        """Refresh the frozen dynamics feature copy from a trained encoder.

        The dynamics heads remain independently optimized, but their feature
        projection can follow IQL's representation after critic warm-up. This
        avoids silently freezing a random pre-IQL encoder snapshot when the
        ensemble is constructed before the offline loop starts.
        """

        if not isinstance(source, ObservationFeatureEncoder):
            raise ValueError("source must be an ObservationFeatureEncoder")
        if source.output_dim != self.feature_encoder.output_dim:
            raise ValueError("source and dynamics feature encoders must have matching output_dim")
        self.feature_encoder.load_state_dict(source.state_dict(), strict=True)
        self.feature_encoder.eval()

    def train(self, mode: bool = True) -> StateDynamicsEnsemble:
        super().train(mode)
        self.feature_encoder.eval()
        return self

    def _state(self, observation: ObservationBatch) -> Tensor:
        try:
            state = observation.features[self.state_key]
        except KeyError as exc:
            raise ValueError(f"observation is missing state_key={self.state_key!r}") from exc
        expected_shape = (observation.batch_size(), self.n_obs_steps, self.state_dim)
        if state.shape != expected_shape or not state.is_floating_point():
            raise ValueError(
                f"{self.state_key} must be floating with shape {expected_shape}, "
                f"got {tuple(state.shape)} {state.dtype}"
            )
        if not torch.isfinite(state).all().item():
            raise ValueError(f"{self.state_key} must contain only finite values")
        return state.float()

    def _model_input(self, observation: ObservationBatch, action: Tensor, action_valid: Tensor) -> Tensor:
        prepared = _float_observation(observation, self.device)
        self._state(prepared)
        with torch.no_grad():
            features = self.feature_encoder(prepared)
        packed = self.action_packer(action.to(self.device), action_valid.to(self.device))
        return torch.cat((features, packed), dim=-1)

    def _decode(self, outputs: Tensor) -> DynamicsPrediction:
        state_values = self.n_obs_steps * self.state_dim
        return DynamicsPrediction(
            state_delta=outputs[..., :state_values].reshape(
                outputs.shape[0], outputs.shape[1], self.n_obs_steps, self.state_dim
            ),
            reward_logit=outputs[..., state_values : state_values + 1],
            done_logit=outputs[..., state_values + 1 : state_values + 2],
        )

    @torch.no_grad()
    def predict(
        self, observation: ObservationBatch, action: Tensor, action_valid: Tensor
    ) -> DynamicsPrediction:
        model_input = self._model_input(observation, action, action_valid)
        outputs = torch.stack([member(model_input) for member in self.members])
        return self._decode(outputs)

    def next_state_history(self, observation: ObservationBatch, prediction: DynamicsPrediction) -> Tensor:
        state = self._state(_float_observation(observation, prediction.state_delta.device))
        expected = (
            self.ensemble_size,
            observation.batch_size(),
            self.n_obs_steps,
            self.state_dim,
        )
        if prediction.state_delta.shape != expected:
            raise ValueError(
                f"prediction state_delta must have shape {expected}, "
                f"got {tuple(prediction.state_delta.shape)}"
            )
        return state.unsqueeze(0) + prediction.state_delta

    def bootstrap_indices(
        self, batch_size: int, *, member_index: int, device: torch.device | str = "cpu"
    ) -> Tensor:
        _positive_int("batch_size", batch_size)
        if (
            isinstance(member_index, bool)
            or not isinstance(member_index, int)
            or not 0 <= member_index < self.ensemble_size
        ):
            raise ValueError(f"member_index must be in [0, {self.ensemble_size}), got {member_index!r}")
        generator = torch.Generator().manual_seed(self.bootstrap_seed + member_index)
        return torch.randint(batch_size, (batch_size,), generator=generator, dtype=torch.long).to(device)

    def _targets(self, batch: DecisionBatch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = batch.to(self.device)
        observation = _float_observation(batch.observation, self.device)
        next_observation = _float_observation(batch.next_observation, self.device)
        model_input = self._model_input(observation, batch.action, batch.action_valid)
        state_delta = self._state(next_observation) - self._state(observation)
        reward = batch.reward.float()
        done = batch.done.float()
        if not ((reward == 0) | (reward == 1)).all().item():
            raise ValueError("dynamics sparse reward targets must be binary 0 or 1")
        return model_input, state_delta, reward, done

    def _member_losses(
        self,
        output: Tensor,
        state_delta: Tensor,
        reward: Tensor,
        done: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state_values = self.n_obs_steps * self.state_dim
        predicted_state = output[..., :state_values].reshape_as(state_delta)
        predicted_reward = output[..., state_values : state_values + 1]
        predicted_done = output[..., state_values + 1 : state_values + 2]
        # The terminal next observation in the LeRobot decision adapter is a
        # repeated final frame.  It is a bookkeeping value, not a learned
        # physical transition, so terminal samples do not contribute to the
        # state-delta regression. Reward and done heads still train on them.
        nonterminal = (~done.bool()).view(-1, 1, 1)
        state_squared_error = (predicted_state - state_delta).square()
        state_loss = torch.where(nonterminal, state_squared_error, 0.0).sum()
        state_loss = state_loss / nonterminal.expand_as(state_squared_error).sum().clamp_min(1)
        reward_loss = functional.binary_cross_entropy_with_logits(predicted_reward, reward)
        done_loss = functional.binary_cross_entropy_with_logits(predicted_done, done)
        total = (
            self.state_loss_weight * state_loss
            + self.reward_loss_weight * reward_loss
            + self.done_loss_weight * done_loss
        )
        return total, state_loss, reward_loss, done_loss

    @staticmethod
    def _validate_member_losses(member_index: int, losses: tuple[Tensor, Tensor, Tensor, Tensor]) -> None:
        total, state_loss, reward_loss, done_loss = losses
        for component, loss in (
            ("state_loss", state_loss),
            ("reward_loss", reward_loss),
            ("done_loss", done_loss),
            ("total_loss", total),
        ):
            if loss.numel() != 1 or not torch.isfinite(loss.detach()).item():
                raise ValueError(f"dynamics member {member_index} {component} must be a finite scalar")

    def update(self, batch: DecisionBatch) -> dict[str, float]:
        if not isinstance(batch, DecisionBatch):
            raise ValueError(f"batch must be a DecisionBatch, got {type(batch).__name__}")
        model_input, state_delta, reward, done = self._targets(batch)
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        member_losses: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        try:
            for member_index, member in enumerate(self.members):
                indices = self.bootstrap_indices(
                    model_input.shape[0], member_index=member_index, device=self.device
                )
                output = member(model_input.index_select(0, indices))
                losses = self._member_losses(
                    output,
                    state_delta.index_select(0, indices),
                    reward.index_select(0, indices),
                    done.index_select(0, indices),
                )
                self._validate_member_losses(member_index, losses)
                member_losses.append(losses)

            for losses in member_losses:
                losses[0].backward()
            for member_index, member in enumerate(self.members):
                gradient_norm = torch.nn.utils.clip_grad_norm_(member.parameters(), self.gradient_clip_norm)
                if not torch.isfinite(gradient_norm).item():
                    raise ValueError(
                        f"dynamics member {member_index} gradient norm must be finite before optimizer step"
                    )
        except Exception:
            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)
            raise

        totals: list[Tensor] = []
        states: list[Tensor] = []
        rewards: list[Tensor] = []
        dones: list[Tensor] = []
        for optimizer, losses in zip(self.optimizers, member_losses, strict=True):
            optimizer.step()
            total, state_loss, reward_loss, done_loss = losses
            totals.append(total.detach())
            states.append(state_loss.detach())
            rewards.append(reward_loss.detach())
            dones.append(done_loss.detach())
        metrics = {
            "loss": float(torch.stack(totals).mean().item()),
            "state_loss": float(torch.stack(states).mean().item()),
            "reward_loss": float(torch.stack(rewards).mean().item()),
            "done_loss": float(torch.stack(dones).mean().item()),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"dynamics metrics contain non-finite values: {metrics}")
        return metrics

    @torch.no_grad()
    def validation_loss(self, batch: DecisionBatch) -> float:
        model_input, state_delta, reward, done = self._targets(batch)
        losses = [
            self._member_losses(member(model_input), state_delta, reward, done)[0] for member in self.members
        ]
        result = float(torch.stack(losses).mean().item())
        if not math.isfinite(result):
            raise ValueError("dynamics validation loss is non-finite")
        return result

    @torch.no_grad()
    def disagreement(self, observation: ObservationBatch, action: Tensor, action_valid: Tensor) -> Tensor:
        prediction = self.predict(observation, action, action_valid)
        next_state = self.next_state_history(observation, prediction)
        disagreement = next_state.var(dim=0, unbiased=False).mean(dim=(-1, -2), keepdim=False)
        result = disagreement.unsqueeze(-1)
        if not torch.isfinite(result).all().item():
            raise ValueError("dynamics disagreement contains non-finite values")
        return result


class DP3FeatureDynamicsEnsemble(nn.Module):
    """RL-100 style ensemble dynamics in DP3 latent feature space.

    Unlike :class:`StateDynamicsEnsemble`, this model predicts the next
    encoded observation window (or a delta in that space).  The encoder is a
    frozen deep copy of the IQL/DP3 feature encoder and therefore consumes the
    point cloud, both wrist RGB streams, and proprioceptive state.  Raw visual
    tensors are never generated by the dynamics model; callers carry the
    initial modalities for actor conditioning while rolling this latent state.
    """

    latent_dynamics = True

    def __init__(
        self,
        *,
        feature_encoder: ObservationFeatureEncoder,
        n_obs_steps: int,
        action_dim: int,
        chunk_size: int,
        active_action_mask: Tensor,
        hidden_dims: Sequence[int],
        ensemble_size: int = 5,
        learning_rate: float = 1e-3,
        bootstrap_seed: int = 0,
        predict_delta: bool = True,
        prediction_mode: str = "last",
        reward_loss_weight: float = 1.0,
        done_loss_weight: float = 1.0,
        gradient_clip_norm: float = 10.0,
    ) -> None:
        super().__init__()
        if not isinstance(feature_encoder, ObservationFeatureEncoder):
            raise ValueError("feature_encoder must implement ObservationFeatureEncoder")
        _positive_int("n_obs_steps", n_obs_steps)
        _positive_int("ensemble_size", ensemble_size)
        if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
            raise ValueError("bootstrap_seed must be an integer")
        learning_rate = _finite_float("learning_rate", learning_rate)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.reward_loss_weight = _finite_float("reward_loss_weight", reward_loss_weight)
        self.done_loss_weight = _finite_float("done_loss_weight", done_loss_weight)
        if self.reward_loss_weight < 0 or self.done_loss_weight < 0:
            raise ValueError("dynamics loss weights must be nonnegative")
        if self.reward_loss_weight + self.done_loss_weight <= 0:
            raise ValueError("at least one dynamics loss weight must be positive")
        self.gradient_clip_norm = _finite_float("gradient_clip_norm", gradient_clip_norm)
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if not isinstance(predict_delta, bool):
            raise ValueError("predict_delta must be a bool")
        if prediction_mode not in {"last", "full"}:
            raise ValueError("prediction_mode must be 'last' or 'full'")

        self.n_obs_steps = n_obs_steps
        self.ensemble_size = ensemble_size
        self.bootstrap_seed = bootstrap_seed
        self.predict_delta = predict_delta
        self.prediction_mode = prediction_mode
        self.feature_encoder = copy.deepcopy(feature_encoder)
        self.feature_encoder.eval()
        for parameter in self.feature_encoder.parameters():
            parameter.requires_grad_(False)
        if feature_encoder.output_dim % n_obs_steps != 0:
            raise ValueError(
                "feature_encoder.output_dim must be divisible by n_obs_steps, "
                f"got {feature_encoder.output_dim} and {n_obs_steps}"
            )
        self.action_packer = ActionPacker(
            action_dim=action_dim,
            chunk_size=chunk_size,
            active_action_mask=active_action_mask,
        )
        self._feature_input_dim = self.latent_dim if prediction_mode == "last" else feature_encoder.output_dim
        self._feature_output_dim = (
            self.latent_dim if prediction_mode == "last" else feature_encoder.output_dim
        )
        output_dim = self._feature_output_dim + 2
        input_dim = self._feature_input_dim + self.action_packer.output_dim
        self.members = nn.ModuleList(
            _DynamicsMember(input_dim, hidden_dims, output_dim) for _ in range(ensemble_size)
        )
        self.optimizers = [torch.optim.Adam(member.parameters(), lr=learning_rate) for member in self.members]

    @property
    def device(self) -> torch.device:
        return next(self.members.parameters()).device

    @property
    def latent_dim(self) -> int:
        return int(self.feature_encoder.output_dim // self.n_obs_steps)

    @torch.no_grad()
    def sync_feature_encoder(self, source: ObservationFeatureEncoder) -> None:
        if not isinstance(source, ObservationFeatureEncoder):
            raise ValueError("source must implement ObservationFeatureEncoder")
        if source.output_dim != self.feature_encoder.output_dim:
            raise ValueError("source and dynamics encoders must have matching output_dim")
        self.feature_encoder.load_state_dict(source.state_dict(), strict=True)
        self.feature_encoder.eval()

    def train(self, mode: bool = True) -> DP3FeatureDynamicsEnsemble:
        super().train(mode)
        self.feature_encoder.eval()
        return self

    def encode_observation(self, observation: ObservationBatch) -> Tensor:
        prepared = _float_observation(observation, self.device)
        with torch.no_grad():
            features = self.feature_encoder(prepared)
        if features.ndim != 2 or features.shape[-1] != self.feature_encoder.output_dim:
            raise ValueError(f"feature encoder must emit [batch, output_dim], got {tuple(features.shape)}")
        return features

    def _model_input(self, observation: ObservationBatch, action: Tensor, action_valid: Tensor) -> Tensor:
        features = self.encode_observation(observation)
        features = self._select_model_features(features)
        packed = self.action_packer(action.to(self.device), action_valid.to(self.device))
        return torch.cat((features, packed), dim=-1)

    def _select_model_features(self, features: Tensor) -> Tensor:
        if features.ndim == 3:
            if features.shape[1:] != (self.n_obs_steps, self.latent_dim):
                raise ValueError("latent history has an unexpected shape")
            return (
                features[:, -1] if self.prediction_mode == "last" else features.reshape(features.shape[0], -1)
            )
        if features.ndim != 2:
            raise ValueError("features must have shape [batch,output_dim] or [batch,obs,latent]")
        if features.shape[-1] != self.feature_encoder.output_dim:
            raise ValueError(
                f"features width must be {self.feature_encoder.output_dim}, got {features.shape[-1]}"
            )
        if self.prediction_mode == "last":
            return features.reshape(features.shape[0], self.n_obs_steps, self.latent_dim)[:, -1]
        return features

    def _decode(self, outputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        feature_values = self._feature_output_dim
        feature_delta = outputs[..., :feature_values]
        reward_logit = outputs[..., feature_values : feature_values + 1]
        done_logit = outputs[..., feature_values + 1 : feature_values + 2]
        return (
            feature_delta.reshape(
                outputs.shape[0],
                outputs.shape[1],
                1 if self.prediction_mode == "last" else self.n_obs_steps,
                self.latent_dim,
            ),
            reward_logit,
            done_logit,
        )

    @torch.no_grad()
    def predict(
        self, observation: ObservationBatch, action: Tensor, action_valid: Tensor
    ) -> LatentDynamicsPrediction:
        return self.predict_latent(self.encode_observation(observation), action, action_valid)

    @torch.no_grad()
    def predict_latent(
        self, features: Tensor, action: Tensor, action_valid: Tensor
    ) -> LatentDynamicsPrediction:
        if not isinstance(features, Tensor) or features.ndim != 2:
            raise ValueError("features must have shape [batch,output_dim]")
        model_features = self._select_model_features(features.to(self.device))
        packed = self.action_packer(action.to(self.device), action_valid.to(self.device))
        model_input = torch.cat((model_features, packed), dim=-1)
        outputs = torch.stack([member(model_input) for member in self.members])
        feature_delta, reward_logit, done_logit = self._decode(outputs)
        return LatentDynamicsPrediction(feature_delta, reward_logit, done_logit)

    def _targets(self, batch: DecisionBatch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = batch.to(self.device)
        current = self.encode_observation(batch.observation)
        following = self.encode_observation(batch.next_observation)
        current_history = current.reshape(current.shape[0], self.n_obs_steps, self.latent_dim)
        following_history = following.reshape(following.shape[0], self.n_obs_steps, self.latent_dim)
        model_features = self._select_model_features(current_history)
        target_history = following_history - current_history if self.predict_delta else following_history
        target = (
            target_history[:, -1]
            if self.prediction_mode == "last"
            else target_history.reshape(target_history.shape[0], -1)
        )
        model_input = torch.cat(
            (model_features, self.action_packer(batch.action, batch.action_valid)), dim=-1
        )
        return model_input, target, batch.reward.float(), batch.done.float()

    def _member_losses(
        self,
        output: Tensor,
        target: Tensor,
        reward: Tensor,
        done: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        feature_values = self._feature_output_dim
        predicted = output[..., :feature_values]
        reward_logit = output[..., feature_values : feature_values + 1]
        done_logit = output[..., feature_values + 1 : feature_values + 2]
        feature_loss = functional.mse_loss(predicted, target)
        reward_loss = functional.binary_cross_entropy_with_logits(reward_logit, reward)
        done_loss = functional.binary_cross_entropy_with_logits(done_logit, done)
        total = feature_loss + self.reward_loss_weight * reward_loss + self.done_loss_weight * done_loss
        return total, feature_loss, reward_loss, done_loss

    def update(self, batch: DecisionBatch) -> dict[str, float]:
        model_input, target, reward, done = self._targets(batch)
        losses: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        try:
            for index, member in enumerate(self.members):
                indices = self.bootstrap_indices(model_input.shape[0], member_index=index, device=self.device)
                losses.append(
                    self._member_losses(
                        member(model_input.index_select(0, indices)),
                        target.index_select(0, indices),
                        reward.index_select(0, indices),
                        done.index_select(0, indices),
                    )
                )
                losses[-1][0].backward()
                norm = torch.nn.utils.clip_grad_norm_(member.parameters(), self.gradient_clip_norm)
                if not torch.isfinite(norm).item():
                    raise ValueError("dynamics gradients must be finite")
            for optimizer in self.optimizers:
                optimizer.step()
        except Exception:
            for optimizer in self.optimizers:
                optimizer.zero_grad(set_to_none=True)
            raise
        metrics = {
            "loss": float(torch.stack([item[0].detach() for item in losses]).mean().item()),
            "feature_loss": float(torch.stack([item[1].detach() for item in losses]).mean().item()),
            "reward_loss": float(torch.stack([item[2].detach() for item in losses]).mean().item()),
            "done_loss": float(torch.stack([item[3].detach() for item in losses]).mean().item()),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("dynamics metrics contain non-finite values")
        return metrics

    @torch.no_grad()
    def validation_loss(self, batch: DecisionBatch) -> float:
        model_input, target, reward, done = self._targets(batch)
        values = [
            self._member_losses(member(model_input), target, reward, done)[0] for member in self.members
        ]
        result = float(torch.stack(values).mean().item())
        if not math.isfinite(result):
            raise ValueError("dynamics validation loss is non-finite")
        return result

    @torch.no_grad()
    def bootstrap_indices(
        self, batch_size: int, *, member_index: int, device: torch.device | str = "cpu"
    ) -> Tensor:
        _positive_int("batch_size", batch_size)
        if (
            isinstance(member_index, bool)
            or not isinstance(member_index, int)
            or not 0 <= member_index < self.ensemble_size
        ):
            raise ValueError("member_index out of range")
        generator = torch.Generator().manual_seed(self.bootstrap_seed + member_index)
        return torch.randint(batch_size, (batch_size,), generator=generator, dtype=torch.long).to(device)

    def next_feature_history(
        self, features: Tensor | ObservationBatch, prediction: LatentDynamicsPrediction
    ) -> Tensor:
        if isinstance(features, ObservationBatch):
            features = self.encode_observation(features)
        if (
            isinstance(features, Tensor)
            and features.ndim == 2
            and features.shape[-1] == self.feature_encoder.output_dim
        ):
            features = features.reshape(features.shape[0], self.n_obs_steps, self.latent_dim)
        expected = (self.ensemble_size, features.shape[0], self.n_obs_steps, self.latent_dim)
        if features.ndim != 3 or features.shape[-1] != self.latent_dim:
            raise ValueError("features must have shape [batch,n_obs_steps,latent_dim]")
        if self.prediction_mode == "last":
            expected_last = (self.ensemble_size, features.shape[0], 1, self.latent_dim)
            if prediction.feature_delta.shape != expected_last:
                raise ValueError(
                    f"prediction feature_delta must have shape {expected_last}, "
                    f"got {tuple(prediction.feature_delta.shape)}"
                )
            next_last = (
                features[:, -1:].unsqueeze(0) + prediction.feature_delta
                if self.predict_delta
                else prediction.feature_delta
            )
            shifted = features.unsqueeze(0).expand(self.ensemble_size, -1, -1, -1).clone()
            shifted[:, :, :-1] = features[:, 1:].unsqueeze(0)
            shifted[:, :, -1:] = next_last
            return shifted
        if prediction.feature_delta.shape != expected:
            raise ValueError(
                f"prediction feature_delta must have shape {expected}, got {tuple(prediction.feature_delta.shape)}"
            )
        base = features.unsqueeze(0)
        return base + prediction.feature_delta if self.predict_delta else prediction.feature_delta

    # Alias used by AM-Q's state-dynamics protocol.
    next_state_history = next_feature_history


@dataclass(frozen=True)
class LatentDynamicsPrediction:
    feature_delta: Tensor
    reward_logit: Tensor
    done_logit: Tensor

    def __post_init__(self) -> None:
        if self.feature_delta.ndim != 4:
            raise ValueError("feature_delta must have shape [ensemble,batch,obs,latent]")
        prefix = self.feature_delta.shape[:2]
        for name, value in (("reward_logit", self.reward_logit), ("done_logit", self.done_logit)):
            if value.shape != (*prefix, 1):
                raise ValueError(f"{name} must have shape {(*prefix, 1)}")
        if not all(
            torch.isfinite(value).all().item()
            for value in (self.feature_delta, self.reward_logit, self.done_logit)
        ):
            raise ValueError("latent dynamics prediction contains non-finite values")

    @property
    def reward_probability(self) -> Tensor:
        return self.reward_logit.sigmoid()

    @property
    def done_probability(self) -> Tensor:
        return self.done_logit.sigmoid()


class PolicyPromotionGate:
    def __init__(
        self,
        *,
        relative_margin: float,
        max_validation_loss: float,
        max_rollout_disagreement: float | None = None,
        use_critic_reference: bool = True,
        inclusive_margin: bool = False,
        epsilon: float = 1e-6,
    ) -> None:
        self.relative_margin = _finite_float("relative_margin", relative_margin)
        self.max_validation_loss = _finite_float("max_validation_loss", max_validation_loss)
        if max_rollout_disagreement is not None:
            max_rollout_disagreement = _finite_float("max_rollout_disagreement", max_rollout_disagreement)
            if max_rollout_disagreement < 0:
                raise ValueError("max_rollout_disagreement must be nonnegative")
        self.max_rollout_disagreement = max_rollout_disagreement
        if not isinstance(use_critic_reference, bool):
            raise ValueError("use_critic_reference must be a bool")
        self.use_critic_reference = use_critic_reference
        if not isinstance(inclusive_margin, bool):
            raise ValueError("inclusive_margin must be a bool")
        self.inclusive_margin = inclusive_margin
        self.epsilon = _finite_float("epsilon", epsilon)
        if self.relative_margin < 0 or self.max_validation_loss < 0 or self.epsilon <= 0:
            raise ValueError("relative_margin/max_validation_loss must be nonnegative and epsilon positive")

    @property
    def delta_threshold_floor(self) -> float:
        """Smallest absolute margin used by the legacy epsilon-safe mode."""

        return self.epsilon

    def decide(
        self,
        *,
        candidate_return: float,
        behavior_return: float,
        critic_return: float,
        dynamics_validation_loss: float,
        rollout_disagreement: float | None = None,
    ) -> PromotionDecision:
        values = {
            "candidate_return": _finite_float("candidate_return", candidate_return),
            "behavior_return": _finite_float("behavior_return", behavior_return),
            "critic_return": _finite_float("critic_return", critic_return),
            "dynamics_validation_loss": _finite_float("dynamics_validation_loss", dynamics_validation_loss),
        }
        if rollout_disagreement is not None:
            rollout_disagreement = _finite_float("rollout_disagreement", rollout_disagreement)
            if rollout_disagreement < 0:
                raise ValueError("rollout_disagreement must be nonnegative")
        reference = (
            max(values["behavior_return"], values["critic_return"])
            if self.use_critic_reference
            else values["behavior_return"]
        )
        margin_scale = (
            abs(reference)
            if self.inclusive_margin and not self.use_critic_reference
            else max(abs(reference), self.epsilon)
        )
        required = reference + self.relative_margin * margin_scale
        if values["dynamics_validation_loss"] < 0:
            raise ValueError("dynamics_validation_loss must be nonnegative")
        if values["dynamics_validation_loss"] > self.max_validation_loss:
            promote = False
            reason = "dynamics_validation_loss"
        elif self.max_rollout_disagreement is not None and rollout_disagreement is None:
            promote = False
            reason = "missing_rollout_disagreement"
        elif (
            self.max_rollout_disagreement is not None
            and rollout_disagreement is not None
            and rollout_disagreement > self.max_rollout_disagreement
        ):
            promote = False
            reason = "rollout_disagreement"
        elif (
            values["candidate_return"] < required
            if self.inclusive_margin
            else values["candidate_return"] <= required
        ):
            promote = False
            reason = "insufficient_return"
        else:
            promote = True
            reason = "promote"
        return PromotionDecision(
            candidate_return=values["candidate_return"],
            behavior_return=values["behavior_return"],
            critic_return=values["critic_return"],
            reference_return=reference,
            required_return=required,
            relative_margin=self.relative_margin,
            dynamics_validation_loss=values["dynamics_validation_loss"],
            max_validation_loss=self.max_validation_loss,
            rollout_disagreement=rollout_disagreement,
            max_rollout_disagreement=self.max_rollout_disagreement,
            promote=promote,
            reason=reason,
        )
