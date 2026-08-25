# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""State-only AM-Q evaluation for offline diffusion-policy promotion.

AM-Q is an offline policy-selection gate, not the actor advantage. A learned
state dynamics ensemble imagines the result of policy action chunks and the
IQL critic scores the same candidate and behavior policies on paired initial
states. The score is the batch expectation of the per-trajectory sum of Q
values. Image-conditioned rollout is intentionally left as an extension
point: imagined observations currently contain the normalized state history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from RL.algorithms.dynamics import StateDynamicsEnsemble
from RL.algorithms.iql import IQL, _finite_float, _positive_int
from RL.policy.diffusion_adapter import DiffusionRLAdapter
from RL.policy.observation_encoder import is_image_feature
from RL.types import ObservationBatch


@dataclass(frozen=True)
class AMQResult:
    """Scalar AM-Q rollout result and model-health diagnostics."""

    amq_return: float
    mean_predicted_reward: float
    mean_predicted_done: float
    max_disagreement: float
    rollout_horizon: int

    def as_metrics(self, prefix: str = "amq") -> dict[str, float]:
        return {
            f"{prefix}/return": self.amq_return,
            f"{prefix}/predicted_reward": self.mean_predicted_reward,
            f"{prefix}/predicted_done": self.mean_predicted_done,
            f"{prefix}/max_disagreement": self.max_disagreement,
            f"{prefix}/rollout_horizon": float(self.rollout_horizon),
        }


def _state_observation(observation: ObservationBatch, *, state_key: str) -> ObservationBatch:
    if not isinstance(observation, ObservationBatch):
        raise ValueError(
            f"observation must be an ObservationBatch, got {type(observation).__name__}"
        )
    if state_key not in observation.features:
        raise ValueError(f"observation is missing state_key={state_key!r}")
    image_keys = [
        key
        for key, value in observation.features.items()
        if key != state_key and is_image_feature(key, value)
    ]
    if image_keys:
        raise ValueError(
            "state-only AM-Q rollout cannot imagine image observations; "
            f"register an image encoder first (keys={image_keys})"
        )
    return ObservationBatch({state_key: observation.features[state_key]})


class AMQEvaluator:
    """Evaluate candidate and behavior policies on paired imagined rollouts."""

    def __init__(
        self,
        *,
        dynamics: StateDynamicsEnsemble,
        iql: IQL,
        candidate_policy: DiffusionRLAdapter,
        behavior_policy: DiffusionRLAdapter,
        gamma: float = 0.99,
        rollout_horizon: int = 20,
        state_key: str = "observation.state",
        discounted: bool = False,
    ) -> None:
        if not isinstance(dynamics, StateDynamicsEnsemble):
            raise ValueError("dynamics must be a StateDynamicsEnsemble")
        if not isinstance(iql, IQL):
            raise ValueError("iql must be an IQL instance")
        for name, policy in (
            ("candidate_policy", candidate_policy),
            ("behavior_policy", behavior_policy),
        ):
            if not isinstance(policy, DiffusionRLAdapter):
                raise ValueError(f"{name} must be a DiffusionRLAdapter")
            image_features = tuple(policy.policy.config.image_features)
            if image_features:
                raise ValueError(
                    "state-only AM-Q does not support image-conditioned policies; "
                    f"{name} declares image features {image_features!r}"
                )
        candidate_policy.assert_transition_compatible(behavior_policy)
        if not torch.equal(
            candidate_policy.checkpoint.active_action_mask,
            behavior_policy.checkpoint.active_action_mask,
        ):
            raise ValueError("candidate and behavior policies must share the active action mask")
        candidate_state = candidate_policy.policy.config.robot_state_feature
        behavior_state = behavior_policy.policy.config.robot_state_feature
        if (
            candidate_state is None
            or behavior_state is None
            or tuple(candidate_state.shape) != tuple(behavior_state.shape)
        ):
            raise ValueError("candidate and behavior policies must share the state feature shape")
        if candidate_policy.policy.config.n_action_steps != dynamics.action_packer.chunk_size:
            raise ValueError("policy n_action_steps must match dynamics chunk_size")
        if behavior_policy.policy.config.n_action_steps != dynamics.action_packer.chunk_size:
            raise ValueError("behavior policy n_action_steps must match dynamics chunk_size")
        for name, policy in (
            ("candidate_policy", candidate_policy),
            ("behavior_policy", behavior_policy),
        ):
            state_feature = policy.policy.config.robot_state_feature
            action_feature = policy.policy.config.action_feature
            if state_feature is None or tuple(state_feature.shape) != (dynamics.state_dim,):
                raise ValueError(
                    f"{name} state feature shape is incompatible with dynamics"
                )
            if action_feature is None or tuple(action_feature.shape) != (
                dynamics.action_packer.action_dim,
            ):
                raise ValueError(
                    f"{name} action feature shape is incompatible with dynamics"
                )
        gamma = _finite_float("gamma", gamma)
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must lie in [0, 1]")
        _positive_int("rollout_horizon", rollout_horizon)
        if not isinstance(state_key, str) or not state_key:
            raise ValueError("state_key must be a nonempty string")
        if not isinstance(discounted, bool):
            raise ValueError("discounted must be a bool")
        self.dynamics = dynamics
        self.iql = iql
        self.candidate_policy = candidate_policy
        self.behavior_policy = behavior_policy
        self.gamma = gamma
        self.rollout_horizon = rollout_horizon
        self.state_key = state_key
        self.discounted = discounted

    @property
    def device(self) -> torch.device:
        return self.dynamics.device

    def _normalized_initial(self, observation: ObservationBatch, *, normalized: bool) -> ObservationBatch:
        state = _state_observation(observation, state_key=self.state_key)
        if normalized:
            return state.to(self.device)
        return self.behavior_policy.checkpoint.normalize_observation(state).to(self.device)

    @torch.no_grad()
    def evaluate(
        self,
        policy: DiffusionRLAdapter,
        observation: ObservationBatch,
        *,
        normalized: bool = False,
        seed: int = 0,
    ) -> AMQResult:
        """Run a deterministic-seed, state-only model rollout and score min-Q."""

        if policy not in (self.candidate_policy, self.behavior_policy):
            raise ValueError("policy must be the configured candidate or behavior adapter")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        state_history = self._normalized_initial(observation, normalized=normalized)
        batch_size = state_history.batch_size()
        action_dim = policy.policy.config.action_feature.shape[0]
        chunk_size = policy.policy.config.n_action_steps
        action_valid = torch.ones(
            batch_size, chunk_size, dtype=torch.bool, device=self.device
        )
        generator = policy.make_generator(seed)
        alive = torch.ones(batch_size, device=self.device)
        terminated = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        discount = 1.0
        total_q = torch.zeros(batch_size, device=self.device)
        reward_sum = torch.zeros((), device=self.device)
        done_sum = torch.zeros((), device=self.device)
        active_count = torch.zeros((), device=self.device)
        max_disagreement = torch.zeros(batch_size, device=self.device)

        for _step in range(self.rollout_horizon):
            active = alive > 0
            if not active.any().item():
                break
            # The dynamics state is normalized; policy sampling expects raw
            # observations and performs checkpoint normalization internally.
            raw_state = policy.checkpoint.unnormalize_observation(state_history)
            trace = policy.sample_trace(raw_state, generator=generator)
            action = trace.final_actions[:, policy.execution_slice, :]
            if action.shape != (batch_size, chunk_size, action_dim):
                raise ValueError(f"policy executable action has unexpected shape {tuple(action.shape)}")
            prediction = self.dynamics.predict(state_history, action, action_valid)
            q_value = self.iql.min_q(state_history, action, action_valid).squeeze(-1)
            # RL-100's AM-Q is the mean Q over the modeled rollout, without a
            # Bellman discount. Keep an opt-in discounted mode for experiments
            # that want the environment's gamma semantics instead.
            step_weight = discount if self.discounted else 1.0
            total_q = total_q + step_weight * alive * q_value
            reward_probability = prediction.reward_probability.mean(dim=0).squeeze(-1)
            done_probability = prediction.done_probability.mean(dim=0).squeeze(-1)
            active_float = active.to(reward_probability.dtype)
            reward_sum = reward_sum + (reward_probability * active_float).sum()
            done_sum = done_sum + (done_probability * active_float).sum()
            active_count = active_count + active_float.sum()
            next_state_ensemble = self.dynamics.next_state_history(
                state_history, prediction
            )
            disagreement = next_state_ensemble.var(
                dim=0, unbiased=False
            ).mean(dim=(-1, -2))
            max_disagreement = torch.maximum(
                max_disagreement, torch.where(active, disagreement, 0.0)
            )
            predicted_next_state = next_state_ensemble.mean(dim=0)
            current_state = state_history.features[self.state_key]
            terminated = terminated | (done_probability >= 0.5)
            active_rows = (alive > 0).view(-1, 1, 1) & ~terminated.view(-1, 1, 1)
            state_history = ObservationBatch(
                {
                    self.state_key: torch.where(
                        active_rows, predicted_next_state, current_state
                    )
                }
            )
            # A hard model terminal ends the imagined trajectory.  Do not let
            # the residual sigmoid continuation weight leak Q terms from later
            # steps after a terminal prediction; soft nonterminal probabilities
            # still provide the usual uncertainty-weighted continuation.
            hard_done = done_probability >= 0.5
            alive = alive * (~hard_done).to(alive.dtype)
            alive = alive * (1.0 - done_probability).clamp(0.0, 1.0)
            discount *= self.gamma ** chunk_size

        result = AMQResult(
            # AM-Q is E[sum_t Q(s_t, a_t)] over the initial-state batch.  Do
            # not divide by the number of alive steps; that would make the
            # score policy-dependent when one rollout terminates early.
            amq_return=float(total_q.mean().item()),
            mean_predicted_reward=float(
                (reward_sum / active_count.clamp_min(1.0)).item()
            ),
            mean_predicted_done=float(
                (done_sum / active_count.clamp_min(1.0)).item()
            ),
            max_disagreement=float(max_disagreement.max().item()),
            rollout_horizon=self.rollout_horizon,
        )
        if not all(
            math.isfinite(value)
            for value in (
                result.amq_return,
                result.mean_predicted_reward,
                result.mean_predicted_done,
                result.max_disagreement,
            )
        ):
            raise ValueError("AM-Q rollout returned non-finite diagnostics")
        return result

    @torch.no_grad()
    def paired(
        self,
        observation: ObservationBatch,
        *,
        normalized: bool = False,
        seed: int = 0,
    ) -> tuple[AMQResult, AMQResult]:
        """Evaluate candidate and behavior with the same initial states and seed."""

        candidate = self.evaluate(
            self.candidate_policy, observation, normalized=normalized, seed=seed
        )
        behavior = self.evaluate(
            self.behavior_policy, observation, normalized=normalized, seed=seed
        )
        return candidate, behavior
